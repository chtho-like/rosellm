# FlashAttention CUDA C++ ladder on RTX 4070 (device 1)

This report tracks a CUDA C++ implementation of **causal SDPA forward**
(FlashAttention-style online softmax) on:

- GPU: `NVIDIA GeForce RTX 4070` (SM89)
- device: `1`

Baseline comparisons come from PyTorch SDPA backends and `flash-attn`.

## 0. How to reproduce

Build kernels:

```bash
cd /data/projects/rosellm/notebooks/cuda/flashattn_cpp
make -j ARCH=sm_89
```

Run sweep + plots:

```bash
python bench_and_plot.py \
  --device 1 --causal 1 --check \
  --batch 1 --heads 32 \
  --seq-lens 256,512,1024,2048,4096 \
  --head-dims 64,128
```

Outputs:

- `out/latest/results.csv`
- `out/latest/plots/time_focus_d*.png`
- `out/latest/plots/tflops_focus_d*.png`
- `out/latest/autotune_table.json` (best kernel per shape from the ladder)

## 1. What “FlashAttention” means here (from first principles)

We compute:

- `Q, K, V` of shape `(B, H, S, D)`
- causal mask (autoregressive): query position `i` can only attend to keys
  `j <= i`

Math (per head):

1. `scores = (Q @ K^T) * scale`
2. `P = softmax(scores)` (masked for causal)
3. `O = P @ V`

The key performance problem is that `scores` is `S×S`.

FlashAttention avoids materializing `S×S` by doing:

- block-by-block `QK^T`
- **online softmax** (keep running max/sum)
- fused accumulation of `P @ V`

## 2. Ladder steps in this directory

- `flashattn_0_warp_naive`
  - warp-per-row
  - global-memory K/V
  - per-key online softmax update
- `flashattn_1_smem_tiled`
  - stage K/V tiles into shared memory
  - block online softmax (one `alpha` per K tile)
- `flashattn_2_causal_cutoff`
  - skip future K tiles entirely for causal workloads
- `flashattn_3_mma_bm128_bn32_d64`
  - first Tensor Core (MMA) kernel for `D=64`
  - QK^T and PV use `mma.sync` (FP16 inputs, FP32 accumulate)
- `flashattn_4_mma_bm64_bn32_d128`
  - Tensor Core (MMA) kernel for `D=128` (baseline tiling)
- `flashattn_5_mma_bm64_bn64_d128_optin`
  - experiment: increase `BN` to `64` for `D=128` via dynamic shared memory
- `flashattn_6_mma_bm64_bn32_d64`
  - retile `D=64` MMA kernel for better occupancy on small `S`
- `flashattn_7_mma_bm128_bn64_d64_kvshare`
  - experiment: increase `BN` to `64` for `D=64` and reuse one buffer for K^T/V
- `flashattn_8_mma_bm128_bn64_d64_exp2`
  - exp2-based softmax/alpha/beta experiment (sometimes faster, sometimes not)
- `flashattn_9_mma_bm128_bn64_d64_8warps`
  - increase warps per block for BN=64 D=64 (lower regs/thread; did not win here)
- `flashattn_10_mma_bm128_bn64_d64_8warps_exp2`
  - step 9 + exp2 softmax/alpha/beta
- `flashattn_11_mma_bm128_bn64_d64_preload_v_optin`
  - experiment: preload V into a dedicated SMEM buffer (opt-in dynamic SMEM)
  - can reduce redundant global loads, but may hurt occupancy
- `flashattn_12_mma_bm128_bn64_d64_kvshare_v_cp_async`
  - cp.async prefetch for V while doing softmax (overlap copy with compute)
  - small but measurable win vs step 7/8 on large S here
- `flashattn_13_mma_bm128_bn64_d64_kvshare_v_cp_async_ws1`
  - warp-specialized V copy experiment (1 copy warp)
  - slower here: extra sync / copy-warp bottleneck
- `flashattn_14_dispatch_autotune_d64`
  - dispatch + lightweight autotune for D=64
  - picks between step 6 (small BM/BN) and step 12 (larger BM/BN + cp.async)

## 3. Results

All raw numbers are in `out/latest/results.csv` and plots are under
`out/latest/plots/`.

Below is a compact summary comparing:

- `flash_attn` (Dao-AILab `flash-attn` package)
- `torch_sdpa_auto` (PyTorch auto-selected backend)
- `best_cpp` (best C++ kernel per shape from the ladder above)

### 3.1 Sweep (D=64, causal=True)

| S | flash_attn (ms) | torch_sdpa_auto (ms) | best_cpp (ms) | best_cpp / flash_attn |
|---:|---:|---:|---:|---:|
| 256  | 0.024261 | 0.025186 | 0.044090 (flashattn_14_dispatch_autotune_d64) | 1.82x |
| 512  | 0.056081 | 0.054900 | 0.127221 (flashattn_14_dispatch_autotune_d64) | 2.27x |
| 1024 | 0.157039 | 0.155473 | 0.364474 (flashattn_14_dispatch_autotune_d64) | 2.32x |
| 2048 | 0.513871 | 0.507681 | 1.219690 (flashattn_14_dispatch_autotune_d64) | 2.37x |
| 4096 | 1.851003 | 1.845195 | 4.391990 (flashattn_14_dispatch_autotune_d64) | 2.37x |

Correctness (checked on `S=256, D=64`):

- `flashattn_14_dispatch_autotune_d64` max_abs_err ≈ `3.5e-05` vs CPU reference

### 3.2 Sweep (D=128, causal=True)

| S | flash_attn (ms) | torch_sdpa_auto (ms) | best_cpp (ms) | best_cpp / flash_attn |
|---:|---:|---:|---:|---:|
| 256  | 0.029055 | 0.029270 | 0.088458 (flashattn_5_mma_bm64_bn64_d128_optin) | 3.04x |
| 512  | 0.078276 | 0.079668 | 0.276891 (flashattn_5_mma_bm64_bn64_d128_optin) | 3.54x |
| 1024 | 0.252589 | 0.256298 | 0.942345 (flashattn_5_mma_bm64_bn64_d128_optin) | 3.73x |
| 2048 | 0.906693 | 0.921057 | 3.452890 (flashattn_5_mma_bm64_bn64_d128_optin) | 3.81x |
| 4096 | 3.398630 | 3.448727 | 13.146700 (flashattn_5_mma_bm64_bn64_d128_optin) | 3.87x |

Notes:

- For D=64, the new dispatch/autotune step (step 14) reliably picks the best
  kernel per S (BM=64/BN=32 for S=256, BM=128/BN=64 + cp.async for larger S).
- We are still ~`2.3x` behind `flash-attn` on large S. Closing this gap likely
  needs deeper pipelining (double-buffered K/V with less synchronization),
  more aggressive register-level tiling, and a larger autotuned kernel zoo.
- For D=128, the gap is larger (≈`3–4x`). This likely needs a different tiling
  strategy (and probably more than one kernel family) plus autotuning.
