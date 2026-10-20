"""
moe_speedtest.py - speed and memory test for a small MoE language model.

Settings you can change:
  E  = number of experts            (more E -> more total parameters)
  g  = granularity                  (each expert is cut into g thin pieces)
  k  = experts' worth of work per token (the router picks k pieces)
Expert size (d_expert) stays FIXED, so work per token is the same for every E.

Data is random tokens. The loss number means nothing here.
This script only measures tokens/sec and memory.
"""
import json
import time
from typing import cast

import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- CONFIGURATION ----------------

E = 4
g = 16
k = 4

D_MODEL = 256
N_LAYERS = 20
N_HEADS = 32
D_EXPERT = 512

SHARED_HIDDEN = 0
VOCAB = 32000
SEQ = 512
BATCH = 10

WARMUP = 3
STEPS = 30
AUX_COEF = 0.01

DTYPE = "bf16"       # "bf16" or "fp32"
dev = "cuda" if torch.cuda.is_available() else "cpu"
# OUTPUT_FILE = "E_4_g_1_k_1.json"     # Example: "results.jsonl"

# ---------------------------------------------------------------- layers
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.w


def rope_cache(seq, head_dim, device, base=10000.0):
    pos = torch.arange(seq, device=device).float()
    dim = torch.arange(0, head_dim, 2, device=device).float()

    freq = 1.0 / (base ** (dim / head_dim))
    angles = pos[:, None] * freq[None, :]

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]

    even = x[..., 0::2]
    odd = x[..., 1::2]

    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos

    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.nh, self.hd = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        #Batch_size, seq_length, dimension
        B, S, D = x.shape

        qkv = self.qkv(x)

        #Dimensions --> B: 0, S: 1, 3: 2, self.nh: 3, self.hd: 4
        qkv = qkv.reshape(B, S, 3, self.nh, self.hd)  
        #changes order of dimension for attention to process each head independently
        qkv = qkv.permute(2, 0, 3, 1, 4) 

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        y = y.transpose(1, 2)
        y = y.reshape(B, S, D)

        # 6. Project the result
        y = self.o(y)
        return y



class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """E experts, each cut into g pieces -> N = E*g pieces. Router picks k pieces."""

    def __init__(self, d, d_expert, E, g, k, shared_hidden=0):
        super().__init__()

        #total dimensions be divisible by g
        assert d_expert % g == 0, "d_expert must be divisible by g"

        #N: No. of experts, topk: selected among N, h: dimension of each expert
        self.N, self.topk, self.h = E * g, k, d_expert // g


        assert self.topk <= self.N, "topk shouldn't go beyond N"
        self.router = nn.Linear(d, self.N, bias=False)

        #Initialise weights of gate, up and down
        self.wg = nn.Parameter(torch.randn(self.N, d, self.h) * 0.02)
        self.wu = nn.Parameter(torch.randn(self.N, d, self.h) * 0.02)
        self.wd = nn.Parameter(torch.randn(self.N, self.h, d) * 0.02)
        self.shared = SwiGLU(d, shared_hidden) if shared_hidden > 0 else None

        #how many tokens were routed to each expert
        self.last_counts: list[int] = []

    def forward(self, x):
        B, S, D = x.shape
        xt = x.reshape(-1, D)
        T = xt.shape[0] #Tokens

        # Scores of Expert per Token
        probs = self.router(xt).float().softmax(-1)  
        w, idx = probs.topk(self.topk, dim=-1)           # [prob, Expert index]
        w = w / w.sum(-1, keepdim=True)                  #Normalize to 1

        flat_e = idx.reshape(-1)                #Tokens Routed to Expert N
        
        #Token Count for each Expert 
        cnt = torch.bincount(flat_e, minlength=self.N)

        # Switch-style load-balancing loss
        # fraction of token assignments going to each expert
        f = cnt.float() / (T * self.topk)  

        # aux penalizes uneven routing    
        aux = self.N * (f.detach() * probs.mean(0)).sum()

        #(Saw in Unsloth) Token Batching per Expert
        order = flat_e.argsort()
        tok = torch.arange(T, device=x.device)[:, None].expand(T, self.topk).reshape(-1)[order]
        wt = w.reshape(-1)[order]
        counts = cnt.tolist()
        self.last_counts = counts

        xs = xt[tok]        #Tokens in order acc to experts
        outs, start = [], 0     #Processed, tok seq start index for expert 

        #i: expert id, c: token count
        for i, c in enumerate(counts):
            if c == 0:
                continue
            xi = xs[start:start + c]
            hi = F.silu(xi @ self.wg[i]) * (xi @ self.wu[i])
            outs.append(hi @ self.wd[i])
            start += c
        y = torch.cat(outs, dim=0)

        out = torch.zeros_like(xt)      #make empty buckets for storing
        #Place processed ones acc to token index
        out.index_add_(0, tok, (y * wt[:, None]).to(out.dtype)) 
        out = out.view(B, S, D)
        if self.shared is not None:
            out = out + self.shared(x).to(out.dtype)
        return out, aux


class Block(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.n1, self.n2 = RMSNorm(a.d_model), RMSNorm(a.d_model)
        self.attn = Attention(a.d_model, a.n_heads)
        self.moe = MoE(a.d_model, a.d_expert, a.E, a.g, a.k, a.shared_hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        m, aux = self.moe(self.n2(x))
        return x + m, aux


class LM(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.emb = nn.Embedding(a.vocab, a.d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(a) for _ in range(a.n_layers))
        self.norm = RMSNorm(a.d_model)
        self.head_dim = a.d_model // a.n_heads

    def forward(self, idx, targets):
        B, S = idx.shape
        cos, sin = rope_cache(S, self.head_dim, idx.device)
        x = self.emb(idx)
        aux_total = 0.0
        for blk in self.blocks:
            x, aux = blk(x, cos, sin)
            aux_total = aux_total + aux
        logits = F.linear(self.norm(x), self.emb.weight)     # tied embeddings
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return loss, aux_total / len(self.blocks)


# ---------------------------------------------------------------- main
from types import SimpleNamespace
def main():
    a = SimpleNamespace(
        E=E,
        g=g,
        k=k,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_expert=D_EXPERT,
        shared_hidden=SHARED_HIDDEN,
        vocab=VOCAB,
        seq=SEQ,
        batch=BATCH,
        warmup=WARMUP,
        steps=STEPS,
        aux_coef=AUX_COEF,
        dtype=DTYPE,
        # out=OUTPUT_FILE,
    )

    torch.manual_seed(0)
    dev = torch.device("cuda")
    if (a.d_expert // a.g) % 32 != 0:
        print(f"WARNING: piece size {a.d_expert // a.g} is not a multiple of 32")

    model = LM(a).to(dev)
    total = sum(p_.numel() for p_ in model.parameters())
    piece = 3 * a.d_model * (a.d_expert // a.g)
    active = total - a.n_layers * (a.E * a.g - a.k) * piece
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)}, HIP {torch.version.hip})"
                              if dev.type == "cuda" else ""))
    print(f"E={a.E} g={a.g} k={a.k} pieces={a.E * a.g} picked/token={a.k} "
          f"piece_size={a.d_expert // a.g}")
    print(f"total params: {total / 1e6:.1f}M | active params/token: {active / 1e6:.1f}M")

    use_amp = a.dtype == "bf16"
    toks_per_step = a.batch * a.seq

    def step():
        x = torch.randint(0, a.vocab, (a.batch, a.seq), device=dev)
        y = torch.randint(0, a.vocab, (a.batch, a.seq), device=dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, aux = model(x, y)
            total_loss = loss + a.aux_coef * aux
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss.item()

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(a.warmup):
        step()
    sync()
    t0 = time.time()
    for _ in range(a.steps):
        loss = step()
    sync()
    dt = time.time() - t0

    tps = a.steps * toks_per_step / dt
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")

    # router health (last step, averaged over layers)
    blocks = cast(list[Block], list(model.blocks))
    dormant = [sum(1 for c in blk.moe.last_counts if c == 0) for blk in blocks]
    maxshare = [max(blk.moe.last_counts) / sum(blk.moe.last_counts) for blk in blocks]

    print(f"\nspeed: {tps:,.0f} tokens/sec  ({dt / a.steps:.2f} s/step, {toks_per_step} tokens/step)")
    print(f"peak memory: {peak:.2f} GB")
    print(f"router (last step): avg dormant pieces/layer = {sum(dormant) / len(dormant):.1f}, "
          f"avg biggest-piece share = {sum(maxshare) / len(maxshare):.3f} "
          f"(even split would be {1 / (a.E * a.g):.3f})")
    for n in (100e6, 300e6, 500e6, 1e9):
        print(f"  time for {n / 1e6:,.0f}M tokens: {n / tps / 3600:.1f} hours")

"""
    if a.out:
        rec = dict(E=a.E, g=a.g, k=a.k, d_model=a.d_model, n_layers=a.n_layers, d_expert=a.d_expert,
                   seq=a.seq, batch=a.batch, dtype=a.dtype, total_M=round(total / 1e6, 1),
                   active_M=round(active / 1e6, 1), tok_per_s=round(tps), peak_GB=round(peak, 2), 
                   projected_hours={f"{n/1e6:.0f}M": round(n/tps/3600, 1) for n in (100e6, 300e6, 500e6, 1e9)}
                   )
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
"""

if __name__ == "__main__":
    main()