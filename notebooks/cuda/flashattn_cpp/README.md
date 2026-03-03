# cuda/flashattn_cpp

Goal: implement **FlashAttention SDPA forward** using a pure **CUDA C++**
stack, as a step-by-step optimization ladder (one kernel per `.cu` file),
and benchmark on **device 1** with plots + tables.

This directory mirrors the style used in `gemm_cuda/` and `gemm_tensor/`:

- each optimization step is its own `flashattn_*.cu` file
- each file builds to a standalone binary via `flashattn_main.cuh`
- a Python script runs all kernels + baselines, and plots with matplotlib

## Quick start

Build:

```bash
cd /data/projects/rosellm/notebooks/cuda/flashattn_cpp
make -j ARCH=sm_89
```

Run one kernel:

```bash
./flashattn_2_causal_cutoff --device 1 --check
```

Run a full sweep + plots (includes `flash-attn` + PyTorch SDPA backends):

```bash
python bench_and_plot.py --device 1 --check
```

Outputs:

- `./out/latest/results.csv`
- `./out/latest/results.json`
- `./out/latest/plots/*.png`

## Kernel ladder (easy → hard)

- `flashattn_0_warp_naive.cu`: warp-per-row, global-memory K/V, per-key online softmax
- `flashattn_1_smem_tiled.cu`: shared-memory K/V tiling + block online softmax
- `flashattn_2_causal_cutoff.cu`: causal cutoff (skip future K-tiles)
- `flashattn_3_mma_bm128_bn32_d64.cu`: first Tensor Core (MMA) kernel for D=64
- `flashattn_4_mma_bm64_bn32_d128.cu`: Tensor Core (MMA) kernel for D=128
- `flashattn_5_mma_bm64_bn64_d128_optin.cu`: BN=64 experiment for D=128 (dynamic SMEM)
- `flashattn_6_mma_bm64_bn32_d64.cu`: retile D=64 kernel for better occupancy
- `flashattn_7_mma_bm128_bn64_d64_kvshare.cu`: BN=64 experiment for D=64 (K/V SMEM reuse)
- `flashattn_8_mma_bm128_bn64_d64_exp2.cu`: exp2-based softmax update experiment
- `flashattn_9_mma_bm128_bn64_d64_8warps.cu`: BM=128 BN=64 D=64 with 8 warps (less reg pressure)
- `flashattn_10_mma_bm128_bn64_d64_8warps_exp2.cu`: step 9 + exp2-based softmax/alpha/beta
- `flashattn_11_mma_bm128_bn64_d64_preload_v_optin.cu`: experiment: preload V into dedicated SMEM (needs opt-in; may hurt occupancy)
- `flashattn_12_mma_bm128_bn64_d64_kvshare_v_cp_async.cu`: cp.async prefetch for V while doing softmax
- `flashattn_13_mma_bm128_bn64_d64_kvshare_v_cp_async_ws1.cu`: warp-specialized V cp.async (1 copy warp; CUTLASS-style experiment)
- `flashattn_14_dispatch_autotune_d64.cu`: dispatch + lightweight autotune across D=64 kernels (cand6 vs cand12)

## Report

See `REPORT.md` for environment, commands, raw tables, plots, and analysis.
