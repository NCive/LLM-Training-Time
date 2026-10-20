# LLM Training Time: Granular MoE Speedtest & Training Time Predictor

A lightweight, high-performance benchmarking and empirical calibration suite for **Granular Mixture-of-Experts (MoE)** Transformer language models implemented in PyTorch.

This project enables you to:
1. **Analytically predict** end-to-end training wall-clock time across token milestones (e.g., 100M to 1B tokens) using Non-Negative Least Squares (NNLS) calibration from quick micro-benchmarks without training full models ([`2ndmain.py`](file:///d:/Python%20Projects/LLM%20Training%20Time/2ndmain.py)).
2. **Directly benchmark** token throughput (tokens/sec), peak GPU memory (VRAM), and expert routing dynamics on custom MoE architectures ([`main.py`](file:///d:/Python%20Projects/LLM%20Training%20Time/main.py)).

---

## The Training Time Scaling Formula

At the core of this project is a physically grounded empirical cost model that predicts the execution time of a single training step ($T_{\text{step}}$, in seconds) and extrapolates full training durations:

### 1. Step Latency Equation

$$\mathbf{T_{\text{step}} = a + L \cdot (\alpha + \beta \cdot E \cdot g) + \delta \cdot \max(0, L - L_0)^2 + c \cdot L \cdot H \cdot B \cdot S^2 + \frac{B \cdot S \cdot F}{R}}$$

$$\mathbf{\text{Training Hours} = \frac{\text{Total Tokens}}{B \cdot S} \times T_{\text{step}} \times \frac{\text{BUFFER}}{3600}}$$

### 2. What Each Part Means

| Term | Component | Detailed Explanation & Mechanics |
|---|---|---|
| **$a$** | **Fixed Baseline Step Cost** | A constant base cost incurred on every training step regardless of model dimensions or batch size. It accounts for Python runtime dispatch, CUDA launch overheads, optimizer baseline overhead, and host-device synchronization latency. |
| **$L \cdot \alpha$** | **Linear Layer Cost** | Proportional cost incurred for each Transformer layer ($L$). Captures sequential intra-block operations including [RMSNorm](file:///d:/Python%20Projects/LLM%20Training%20Time/main.py#L49-L59), residual stream additions, activation caches, and standard inter-layer data movement. |
| **$L \cdot \beta \cdot E \cdot g$** | **Granular Expert Dispatch Cost** | An additional per-layer cost scaling with total expert pieces ($N = E \cdot g$). Because the dispatch pipeline evaluates active expert slices sequentially in a loop, higher granularity ($g$) or expert count ($E$) increases kernel launch overhead and token gathering/scattering cycles. |
| **$\delta \cdot \max(0, L - L_0)^2$** | **Deep-Model Non-Linear Penalty** | A penalty term activated exclusively for deep architectures exceeding threshold $L_0$ (default $L_0 = 16$). Below this depth, the term is identically zero. Above $L_0$, it captures quadratic latency degradation stemming from cumulative memory pressure, L2 cache evictions, and memory bandwidth contention. |
| **$c \cdot L \cdot H \cdot B \cdot S^2$** | **Attention Quadratic Cost** | Self-attention compute and memory bandwidth cost across $L$ layers and $H$ attention heads. Grows quadratically ($S^2$) with sequence length because every token attends to all preceding tokens in the causal context window. |
| **$\frac{B \cdot S \cdot F}{R}$** | **Arithmetic FLOPs / GPU Speed** | Pure compute execution time: total floating point operations per step ($B \cdot S \cdot F$) divided by the effective hardware throughput $R$ (in FLOP/s). Represents the ideal arithmetic runtime on GPU tensor cores. |
| **$a, \alpha, \beta, c, \delta, R$** | **Fitted Coefficients** | Hardware- and environment-specific constants derived empirically via Non-Negative Least Squares ([`scipy.optimize.nnls`](file:///d:/Python%20Projects/LLM%20Training%20Time/2ndmain.py#L28)) from fast micro-benchmarks. |
| **$\text{BUFFER}$** | **Real-World Overhead Factor** | Production multiplier (default `1.15`, or $+15\%$) accounting for unmeasured real-world factors: data loading I/O, checkpoint saving, periodic validation loss evaluation, and GPU thermal throttling. |

### 3. Parameter & FLOPs Calculations

- **Active Parameters ($N_{\text{act}}$)**: Parameters evaluated per token:
  $$N_{\text{act}} = V \cdot d + L \cdot \left(4d^2 + 3d \cdot d_{\text{expert}} \cdot \frac{k}{g} + 3d \cdot \text{shared\_hidden} + d \cdot E \cdot g + 2d\right) + d$$

- **Total Parameters ($N_{\text{total}}$)**: All allocated weights in memory:
  $$N_{\text{total}} = V \cdot d + L \cdot \left(4d^2 + 3d \cdot d_{\text{expert}} \cdot E + 3d \cdot \text{shared\_hidden} + d \cdot E \cdot g + 2d\right) + d$$

- **Approximate Training FLOPs per Token ($F$)**:
  $$F = 6 \cdot N_{\text{act}} + 6 \cdot L \cdot S \cdot d$$
  *(Includes standard forward + backward multiplier of $\approx 6 \times \text{params}$ plus self-attention projection FLOPs).*

---

## Table of Contents

- [The Training Time Scaling Formula](#the-training-time-scaling-formula)
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Mechanics](#architecture--mechanics)
  - [Granular MoE ($E$, $g$, $k$)](#granular-moe-e-g-k)
  - [Optimized Token Routing & Batching](#optimized-token-routing--batching)
- [Repository Structure](#repository-structure)
- [Requirements & Installation](#requirements--installation)
- [Usage Guide](#usage-guide)
  - [1. Direct Benchmark (`main.py`)](#1-direct-benchmark-mainpy)
  - [2. Micro-Benchmark Calibration & Prediction (`2ndmain.py`)](#2-micro-benchmark-calibration--prediction-2ndmainpy)
- [Configuration Reference](#configuration-reference)
- [Interpreting the Metrics](#interpreting-the-metrics)

---

## Overview

Mixture-of-Experts (MoE) architectures allow models to scale parameter count while maintaining a constant per-token compute budget. However, measuring and predicting training performance is non-trivial due to router overhead, dynamic memory allocations, kernel launches, and load imbalance.

This repository provides two complementary tools:
- **Parametric Calibration & Estimator ([`2ndmain.py`](file:///d:/Python%20Projects/LLM%20Training%20Time/2ndmain.py))**: Executes an automated suite of short, isolated micro-tests over varied layer counts, expert granularities, batch sizes, and sequence lengths. It fits the timing formula via constrained regression to project step latency and multi-hour training durations.
- **Direct Speedtest ([`main.py`](file:///d:/Python%20Projects/LLM%20Training%20Time/main.py))**: Runs synthetic training steps on your target MoE model to observe actual hardware throughput, peak memory allocation, and routing health (e.g., dormant experts, routing skew).

---

## Key Features

- **Granular Expert Decomposition**: Supports fine-grained expert splitting ($g$ sub-experts per expert) with top-$k$ sub-expert routing for finer parameter specialization.
- **Modern Transformer Primitives**:
  - Pre-layer [RMSNorm](file:///d:/Python%20Projects/LLM%20Training%20Time/main.py#L49-L59) with float32 stabilization.
  - Rotary Position Embeddings ([RoPE](file:///d:/Python%20Projects/LLM%20Training%20Time/main.py#L61-L82)).
  - Fast causal self-attention via PyTorch `F.scaled_dot_product_attention`.
  - [SwiGLU](file:///d:/Python%20Projects/LLM%20Training%20Time/main.py#L122-L131) activation functions in expert feed-forward layers.
  - Tied input and output embedding projections.
- **Unsloth-Inspired Grouped Token Dispatch**: Expert dispatching batches tokens assigned to the same expert contiguously using `argsort` and `index_add_`, avoiding memory-inefficient looped masked tensors.
- **Switch-Transformer Load Balancing Loss**: Penalizes routing collapse to ensure balanced expert utilization across tokens.
- **Robust Out-Of-Memory (OOM) Protection**: Micro-benchmarks run in isolated subprocesses (`--worker` mode) to catch CUDA OOM errors gracefully without terminating the calibration sweep.
- **Hardware Agnostic**: Built for NVIDIA CUDA and AMD ROCm / HIP environments (includes flags for AOTriton acceleration).

---

## Architecture & Mechanics

### Granular MoE ($E$, $g$, $k$)

Instead of conventional coarse experts, each expert block is parameterized by:
- $E$: Number of base experts.
- $g$: Granularity factor. Each base expert is partitioned into $g$ smaller slices of hidden dimension $h = d_{\text{expert}} / g$.
- $N = E \times g$: Total available expert slices in the layer.
- $k$: Total number of slices selected by the router per token.

Each token selects its top-$k$ slices from the total $N$ pool, normalizing routing weights via softmax. This allows a richer combinatorial selection of sub-experts while keeping the per-token computational workload constant.

### Optimized Token Routing & Batching

Rather than running full tensor masks across experts, the dispatch pipeline groups tokens:
1. Tokens obtain routing probabilities via a linear projection `self.router(xt)`.
2. The router extracts top-$k$ indices and normalizes weights.
3. Token indices are sorted by expert ID (`flat_e.argsort()`), reordering tokens to process each expert's work in contiguous slices.
4. Expert activations are computed in a loop only over active experts ($c > 0$).
5. Results are accumulated back to token order via `out.index_add_`.

---

## Repository Structure

```text
.
├── main.py        # Direct MoE speed and peak memory benchmark (moe_speedtest.py)
├── 2ndmain.py     # Analytical calibration, NNLS regression & time predictor (moe_train_time.py)
├── .gitignore     # Git ignore rules for Python, PyTorch, checkpoints & logs
├── LICENSE        # Apache License 2.0
└── README.md      # Project documentation
```

---

## Requirements & Installation

### Prerequisites
- Python 3.9+
- [PyTorch](https://pytorch.org/) 2.0+ (CUDA or ROCm enabled)
- NumPy
- SciPy

### Installation

Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone <repo-url>
cd "LLM Training Time"

# Install dependencies (example with pip)
pip install torch numpy scipy
```

*Note for AMD ROCm users:* Both scripts set `os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"` to leverage AOTriton flash-attention kernels where supported.

---

## Usage Guide

### 1. Direct Benchmark (`main.py`)

Run synthetic forward and backward passes with the AdamW optimizer to evaluate raw hardware throughput and VRAM consumption:

```bash
python main.py
```

#### Example Output:
```text
device: cuda (NVIDIA GeForce RTX 4090)
E=4 g=16 k=4 pieces=64 picked/token=4 piece_size=32
total params: 24.3M | active params/token: 11.2M

speed: 124,500 tokens/sec  (0.04 s/step, 5120 tokens/step)
peak memory: 1.84 GB
router (last step): avg dormant pieces/layer = 2.1, avg biggest-piece share = 0.042 (even split would be 0.016)
  time for 100M tokens: 0.2 hours
  time for 300M tokens: 0.7 hours
  time for 500M tokens: 1.1 hours
  time for 1,000M tokens: 2.2 hours
```

---

### 2. Micro-Benchmark Calibration & Prediction (`2ndmain.py`)

Execute an automated suite of fast micro-benchmarks to fit the timing equation and estimate full training duration:

```bash
python 2ndmain.py
```

#### What Happens During Calibration:
1. **Sweep Planning**: Generates configurations across reduced depths, batch sizes, sequence lengths, and granularities.
2. **Subprocess Isolation**: Tests each configuration in a standalone subprocess (`python 2ndmain.py --worker`).
3. **OOM Handling**: If a test runs out of memory, it skips heavier configurations that share the same shape to avoid hanging.
4. **NNLS Optimization**: Fits coefficients $[a, \alpha, \beta, c, 1/R, \delta]$.
5. **Validation**: Compares predicted step time against an unseen held-out configuration.
6. **Projection**: Outputs estimated training hours for 100M, 300M, 500M, and custom token targets (with margin boundaries).

---

## Configuration Reference

Both scripts contain a configuration section at the top of the file:

| Parameter | Default | Description |
|---|---|---|
| `E` | `4` | Number of base experts. |
| `g` | `16` | Granularity (slices per base expert). $d_{\text{expert}}$ must be divisible by $g$. |
| `k` | `4` | Number of expert slices picked per token by the router. |
| `D_MODEL` | `256` | Hidden dimension of the model ($d$). Must be divisible by `N_HEADS`. |
| `N_LAYERS` | `20` | Total number of Transformer blocks ($L$). |
| `N_HEADS` | `32` | Number of attention heads. |
| `D_EXPERT` | `512` | Total hidden capacity per base expert ($d_{\text{expert}}$). |
| `SHARED_HIDDEN` | `0` | Hidden dimension for an optional shared dense expert ($0$ disables). |
| `VOCAB` | `32000` | Vocabulary size ($V$). |
| `SEQ` | `512` | Sequence length in tokens ($S$). |
| `BATCH` | `10` | Micro-batch size ($B$). |
| `TOTAL_TOKENS` | `1e9` | Target training tokens for duration estimation (`2ndmain.py`). |
| `WARMUP` | `3` | Warmup steps to initialize CUDA graphs and JIT caches. |
| `STEPS` | `30` | Measurement steps per benchmark. |
| `AUX_COEF` | `0.01` | Multiplier for the auxiliary load-balancing routing loss. |
| `DTYPE` | `"bf16"` | Precision format (`"bf16"` with autocast or `"fp32"`). |
| `BUFFER` | `1.15` | Overhead multiplier for I/O, checkpoints, and evaluation (`2ndmain.py`). |

---

## Interpreting the Metrics

- **Tokens/sec**: Total tokens processed per second ($\frac{\text{batch} \times \text{seq}}{\text{step\_time}}$).
- **Dormant Pieces**: The average number of sub-experts receiving $0$ tokens in a step. High dormant count indicates underutilized expert capacity or router collapse.
- **Biggest-Piece Share**: Fraction of routed tokens received by the most popular sub-expert. Compare this to the uniform distribution baseline ($1 / (E \times g)$).
- **$R$ (TFLOP/s)**: Effective compute throughput extracted by NNLS. Values close to hardware limits indicate efficient GPU tensor core utilization.
- **Fit Error (RMS)**: The residual error of the analytical model on the calibration dataset (typically under 10%).

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](file:///d:/Python%20Projects/LLM%20Training%20Time/LICENSE) file for the full license text.
