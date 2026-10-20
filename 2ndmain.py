"""
FORMULA
NN_act = V*d + L*(4*d^2 + 3*d*d_expert*k/g
                + 3*d*shared_hidden + d*E*g + 2*d) + d

F      = 6*N_act + 6*L*seq*d
T_step = a + L*(alpha + beta*E*g) + delta*max(0, L-L0)^2
        + c*L*heads*batch*seq^2 + batch*seq*F / R
Hours  = total_tokens / (batch*seq) * T_step * 1.15 / 3600

USAGE
python moe_train_time.py              

"""

import json
import statistics
import subprocess
import sys

import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import nnls
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================= CONFIG: the TARGET model =======================

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

TOTAL_TOKENS = 1e9        # how many tokens you want to train on

WARMUP = 3                # measurement settings (used in every calibration test)
STEPS = 30               # lower this (for example 40) to calibrate faster
AUX_COEF = 0.01

DTYPE = "bf16"            # "bf16" or "fp32"
SEED = 0


# ======================= calibration settings ===========================

#Tests different model depths to measure how training time scales with the number of layers.
CALIB_LAYERS = (1, 2, 4, 8, 12, 16, 24)   
BUFFER = 1.15             # margin for data loading, checkpoints, evaluation, noise
L0 = 16                   # depth above which the extra "delta" term can switch on


# ---------------------------------------------------------------- formula
def active_params(t):
    d = t["d_model"]
    per_layer = (4 * d * d + 3 * d * t["d_expert"] * t["k"] / t["g"]
                + 3 * d * t["shared_hidden"] + d * t["E"] * t["g"] + 2 * d)      
    return t["vocab"] * d + t["n_layers"] * per_layer + d


def total_params(t):
    d = t["d_model"]
    per_layer = (4 * d * d + 3 * d * t["d_expert"] * t["E"]
                + 3 * d * t["shared_hidden"] + d * t["E"] * t["g"] + 2 * d)
    return t["vocab"] * d + t["n_layers"] * per_layer + d


def features(t):
    L, d, S, B = t["n_layers"], t["d_model"], t["seq"], t["batch"]
    # Approximate training FLOPs per token, not measured FLOPs.
    flops_tok = 6 * active_params(t) + 6 * L * S * d
    return [1.0, L, L * t["E"] * t["g"],
            L * t["n_heads"] * B * S * S, B * S * flops_tok,
            max(0, L - L0) ** 2]

def predict_step(t, c):
    return float(np.dot(features(t), c))


# ---------------------------------------------------------------- model
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
        assert d % n_heads == 0, "d_model must be divisible by n_heads"
        self.nh, self.hd = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k_, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k_ = apply_rope(k_, cos, sin)
        y = F.scaled_dot_product_attention(q, k_, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, S, D)
        return self.o(y)


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

    def __init__(self, d, d_expert, E_, g_, k_, shared_hidden=0):
        super().__init__()
        assert d_expert % g_ == 0, "d_expert must be divisible by g"
        self.N, self.topk, self.g = E_ * g_, k_, d_expert // g_
        assert self.topk <= self.N, "topk shouldn't go beyond N"
        self.router = nn.Linear(d, self.N, bias=False)
        self.wg = nn.Parameter(torch.randn(self.N, d, self.g) * 0.02)
        self.wu = nn.Parameter(torch.randn(self.N, d, self.g) * 0.02)
        self.wd = nn.Parameter(torch.randn(self.N, self.g, d) * 0.02)
        self.shared = SwiGLU(d, shared_hidden) if shared_hidden > 0 else None

    def forward(self, x):
        B, S, D = x.shape
        xt = x.reshape(-1, D)
        T = xt.shape[0]

        probs = self.router(xt).float().softmax(-1)
        w, idx = probs.topk(self.topk, dim=-1)
        w = w / w.sum(-1, keepdim=True)

        flat_e = idx.reshape(-1)
        cnt = torch.bincount(flat_e, minlength=self.N)

        f = cnt.float() / (T * self.topk)
        aux = self.N * (f.detach() * probs.mean(0)).sum()

        order = flat_e.argsort()
        tok = torch.arange(T, device=x.device)[:, None].expand(T, self.topk).reshape(-1)[order]
        wt = w.reshape(-1)[order]
        counts = cnt.tolist()

        xs = xt[tok]
        outs, start = [], 0
        for i, c in enumerate(counts):
            if c == 0:
                continue
            xi = xs[start:start + c]
            hi = F.silu(xi @ self.wg[i]) * (xi @ self.wu[i])
            outs.append(hi @ self.wd[i])
            start += c
        y = torch.cat(outs, dim=0)

        out = torch.zeros_like(xt)
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
        logits = F.linear(self.norm(x), self.emb.weight)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return loss, aux_total / len(self.blocks)


# ---------------------------------------------------------------- one benchmark run (worker)
def run_one(cfg):
    """Train for a few steps with the given settings; return median step time and peak memory."""
    a = SimpleNamespace(**cfg)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = LM(a).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
        use_amp = a.dtype == "bf16"

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
        times = []
        for _ in range(a.steps):
            t0 = time.perf_counter()
            step()
            sync()
            times.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")
        return dict(ok=True, step_median_s=statistics.median(times),
                    step_mean_s=sum(times) / len(times), peak_GB=peak)
    except torch.cuda.OutOfMemoryError:
        return dict(ok=False, reason="out of memory")


def run_in_subprocess(cfg):
    proc = subprocess.run([sys.executable, __file__, "--worker"],
                        input=json.dumps(cfg), capture_output=True, text=True)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return dict(ok=False, reason="crashed: " + proc.stderr[-300:])


# ---------------------------------------------------------------- calibration
def target_cfg():
    return dict(E=E, g=g, k=k, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                d_expert=D_EXPERT, shared_hidden=SHARED_HIDDEN, vocab=VOCAB, seq=SEQ,
                batch=BATCH, warmup=WARMUP, steps=STEPS, aux_coef=AUX_COEF,
                dtype=DTYPE, seed=SEED)


def plan_runs(t):
    runs = []

    def add(**changes):
        runs.append({**t, **changes})

    for L in sorted({l for l in CALIB_LAYERS if l < t["n_layers"]} | {t["n_layers"]}):
        add(n_layers=L)
    for g_ in (1, 2, 3, 4, 6, 8, 12, 16):
        if g_ != t["g"] and t["E"] * g_ >= t["k"] and t["d_expert"] % g_ == 0:
            add(g=g_, n_layers=1)
    for E_ in (2, 3, 4, 5):
        if E_ != t["E"] and E_ * t["g"] >= t["k"]:
            add(E=E_, n_layers=1)
    for b in (2, 4, 8, 12, 16, 20):
        if b != t["batch"]:
            for L_ in sorted({1, min(4, t["n_layers"])}):
                add(batch=b, n_layers=L_)
    for s in (32, 64, 128, 256, 512, 1024):
        if s != t["seq"]:
            add(seq=s, n_layers=1)
    for h in (t["n_heads"] * 2, t["n_heads"] * 4, t["n_heads"] // 2):
        if h >= 1 and h != t["n_heads"] and t["d_model"] % h == 0 and t["d_model"] // h >= 32:
            add(n_heads=h, n_layers=1)
    if t["seq"] >= 128:
        add(batch=max(1, t["batch"] // 2))
    return runs


def shape_key(c):
    return (c["E"], c["g"], c["k"], c["n_heads"], c["d_model"], c["d_expert"])


def too_big(cfg, failed):
    """True if a test with the same model shape, and smaller or equal depth, batch and seq, already failed."""
    s = shape_key(cfg)
    return any(s == fs and cfg["n_layers"] >= fl and cfg["batch"] >= fb and cfg["seq"] >= fq
               for fs, fl, fb, fq in failed)


def calibrate(t):
    plan = plan_runs(t)
    val_cfg = plan.pop() if t["seq"] >= 128 else None
    print(f"Calibrating: {len(plan)} short tests "
        f"({WARMUP} warmup + {STEPS} steps each)\n")
    
    rows, times, failed = [], [], []
    for i, cfg in enumerate(plan, 1):
        label = (f"[{i}/{len(plan)}] L={cfg['n_layers']} E={cfg['E']} g={cfg['g']} k={cfg['k']} "
                 f"H={cfg['n_heads']} B={cfg['batch']} S={cfg['seq']}")
        if too_big(cfg, failed):
            print(label, "-> skipped (a smaller test with the same model already failed)")
            continue
        res = run_in_subprocess(cfg)
        if not res.get("ok"):
            print(label, f"-> FAILED ({res.get('reason')})")
            failed.append((shape_key(cfg), cfg["n_layers"], cfg["batch"], cfg["seq"]))
            continue
        print(label, f"-> {res['step_median_s']:.3f} s/step, {res['peak_GB']:.2f} GB")
        rows.append(features(cfg))
        times.append(res["step_median_s"])

    if len(times) < 6:
        sys.exit("Too few tests worked. Try a smaller model or batch.")

    A = np.array(rows) / np.array(times)[:, None]
    scale = np.abs(A).max(axis=0)
    scale[scale == 0] = 1.0              # delta column is all zeros when no test is deeper than L0
    sol, _ = nnls(A / scale, np.ones(len(times)))
    c = sol / scale

    errs = A @ c - 1
    print(f"\nSettings: dtype={t['dtype']}, torch={torch.__version__}, "
        f"aotriton={os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '0')}")
    print(f"Fit error (rms): {np.sqrt(np.mean(errs ** 2)) * 100:.1f}%")
    for name, v in zip(("a", "alpha", "beta", "c", "1/R", "delta"), c):
        print(f"  {name:6s} = {v:.4g}")
    if c[4] > 0:
        print(f"  R = {1 / c[4] / 1e12:.3f} TFLOP/s")

    if val_cfg is not None and too_big(val_cfg, failed):
        print("Held-out check skipped: a smaller test with the same model already failed.")
        val_cfg = None

    if val_cfg is not None:
        val = run_in_subprocess(val_cfg)
        if val.get("ok"):
            pred = predict_step(val_cfg, c)
            real = float(val["step_median_s"])
            error = (pred / real - 1) * 100
            print(
                f"Held-out check: predicted {pred:.3f}s, "
                f"measured {real:.3f}s, error {error:+.1f}%"
            )
        else:
            print(f"Held-out validation failed: {val.get('reason')}")
    return c


# ---------------------------------------------------------------- estimate
def estimate(t, c):
    step = predict_step(t, c)
    toks = t["batch"] * t["seq"]
    print("\n" + "=" * 50)
    print(f"Active params: {active_params(t) / 1e6:.1f}M | "
        f"Total params: {total_params(t) / 1e6:.1f}M")
    print(f"Step time: {step:.3f} s | {toks / step:,.0f} tokens/sec")
    for n in (100e6, 300e6, 500e6, TOTAL_TOKENS):
        h = n / toks * step * BUFFER / 3600
        print(f"{n / 1e6:,.0f}M tokens: {h:.1f} hours (rough range {h * 0.85:.1f}-{h * 1.15:.1f})")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        cfg = json.load(sys.stdin)
        print("RESULT " + json.dumps(run_one(cfg)))
    else:
        t = target_cfg()
        c = calibrate(t)
        estimate(t, c)


