# HGEMM (FP16, Tensor Cores) Optimization Ladder on RTX 4070

This report documents a step-by-step attempt to optimize a custom **FP16
HGEMM** kernel (`C = A @ B`, **row-major**) using **NVIDIA Tensor Cores**
on the **NVIDIA GeForce RTX 4070** (SM89).

All kernel sources live in `notebooks/cuda/gemm_tensor/`.

The goal of this ladder is to explore common industry techniques (tiling,
shared memory staging, `ldmatrix`, `mma.sync`, pipelining) and measure the
resulting performance against **cuBLAS**.

Important note (honesty):

- In this exploration, the best custom kernels are still **below cuBLAS**
  for apples-to-apples comparisons.
- The closest we got is ~**0.918x** cuBLAS in FP32-accumulation mode at
  `M=N=K=4096`.

## 0. What is HGEMM (from first principles)

We want to compute a matrix multiplication:

- `A` is an `M x K` matrix
- `B` is a `K x N` matrix
- `C` is an `M x N` matrix

For each output element:

- `C[i, j] = sum_{kk=0..K-1} A[i, kk] * B[kk, j]`

HGEMM usually means:

- inputs are **FP16** (`half`)
- output is **FP16**
- “compute” may be:
  - **FP16 accumulation** (lower precision, often faster), or
  - **FP32 accumulation** (higher precision, often the default in ML).

### What does “TFLOP/s” mean here?

GEMM does about:

- FLOPs ≈ `2 * M * N * K`
  - one multiply + one add per `kk`

If a kernel takes `t` seconds:

- `TFLOP/s = (2*M*N*K) / t / 1e12`

This report benchmarks **square** shapes: `M=N=K=size`.

## 1. Tensor Cores: the mental model (beginner-friendly)

Tensor Cores accelerate matrix multiply using **fixed-shape** matrix
multiply-accumulate instructions (MMA).

For FP16 GEMM on SM80+ GPUs (including SM89), a common instruction shape is:

- `m16n8k16`

Interpretation:

- it computes a `16 x 8` output tile
- it reduces over `K=16`

At a high level, a warp does:

1. load an `A` fragment and a `B` fragment from **shared memory** using
   `ldmatrix`
2. run `mma.sync` to update accumulators
3. repeat for all `K` tiles

Key performance problem:

- Tensor Cores are extremely fast, so you must feed them efficiently.
- That usually means:
  - coalesced global memory loads
  - shared-memory staging
  - avoiding shared-memory bank conflicts (padding/swizzling)
  - enough work per threadblock (good tiling)

## 2. Fairness rules used here (what we compare)

1. **Row-major GEMM**: `C(MxN) = A(MxK) * B(KxN)`, all in row-major.
2. **Same shapes** for cuBLAS and custom kernels.
3. **Same timing method**:
   - warmup iterations
   - CUDA events on the same stream
   - enough iterations to reach a minimum benchmark time
4. **Tensor Cores enabled in cuBLAS**:
   - `cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)`
   - `cublasGemmEx(..., algo=CUBLAS_GEMM_DEFAULT_TENSOR_OP)`

## 3. Hardware + software environment

- GPU: `NVIDIA GeForce RTX 4070`
- Compute capability: `8.9` (SM89)
- Driver: `580.82.07`
- CUDA toolkit: `nvcc 12.8` (V12.8.93)

You can confirm locally with:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvcc --version
```

Profiling note:

- Nsight Compute (`ncu`) is installed, but this machine does not allow access
  to GPU performance counters for non-privileged users (`ERR_NVGPUCTRPERM`).
- Because of that, the analysis here relies on:
  - end-to-end timing (CUDA events), and
  - compiler resource usage (`ptxas -v`: registers/shared memory).

## 4. Build and run

From `notebooks/cuda/gemm_tensor/`:

Build all kernels:

```bash
make -j ARCH=sm_89
```

Run one kernel (example):

```bash
./hgemm_9_mma_m16n8k16_f32acc_b64n128_bk32 \
  --device 0 --sizes 1024,2048,4096 --accum f32 --no-check
```

Generate JSON + plots for a subset of kernels:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install matplotlib

# FP32-accumulation kernels (names contain 'f32acc')
python3 bench_and_plot.py \
  --device 0 \
  --sizes 256,512,1024,2048,4096,6144 \
  --warmup 5 --min-ms 200 --max-iters 200 \
  --accum f32 --kernels f32acc --no-check

# FP16-accumulation kernels (names do NOT contain 'f32acc')
python3 bench_and_plot.py \
  --device 0 \
  --sizes 256,512,1024,2048,4096,6144 \
  --warmup 5 --min-ms 200 --max-iters 200 \
  --accum f16 --kernels f16acc --no-check
```

This report’s numbers were generated into:

- FP32-acc ladder: `results/run_20260215_013238/`
- FP16-acc ladder: `results/run_20260214_092606/`

Each run directory contains:

- `*.json`: per-size timing + TFLOP/s for cuBLAS and custom kernel
- `*.log`: raw stdout/stderr
- `plots/*_tflops.png`: TFLOP/s curves
- `plots/*_speedup.png`: speedup curves (custom / cuBLAS)
- `summary.md`: quick per-kernel “best speedup” summary

## 5. The optimization ladder (one file per step)

This repo uses a “ladder” style: each kernel variant is a separate `.cu`
file with a clear, focused change.

Two branches exist:

- **FP16-accumulation**: uses `mma.sync ... f16.f16.f16.f16`
- **FP32-accumulation**: uses `mma.sync ... f32.f16.f16.f32`

### 5.1 FP16-accumulation ladder (`--accum f16`, `--kernels f16acc`)

- Step 0: `hgemm_0_mma_m16n8k16_simple.cu`
  - Minimal `ldmatrix + mma.sync` baseline (BK=16).
- Step 1: `hgemm_1_mma_m16n8k16_smem_swizzle.cu`
  - Add shared-memory padding and a simple A swizzle (bank-conflict reduction).
- Step 2: `hgemm_2_mma_m16n8k16_bk32.cu`
  - Increase BK from 16 → 32 (fewer stages, fewer barriers).
- Step 3: `hgemm_3_mma_m16n8k16_bk32_cp_async_db.cu`
  - Try `cp.async` + double-buffer shared memory (hurts occupancy here).
- Step 5: `hgemm_5_mma_m16n8k16_bk64.cu`
  - Increase BK to 64 (hurts occupancy/register pressure here).
- Step 6: `hgemm_6_mma_m16n8k16_bk32_simplify.cu`
  - Refactor/simplify Step 2; does not improve.
- Step 7: `hgemm_7_mma_m16n8k16_bk32_cp_async_db_simplify.cu`
  - Refactor Step 3; still slower.

Extra experiments in this FP16-acc branch:

- Step 14: `hgemm_14_mma_m16n8k16_b64n128_bk32.cu`
  - Smaller block tile (better for small sizes, worse for large sizes).
- Step 15: `hgemm_15_mma_m16n8k16_bk16_cp_async_db.cu`
  - BK=16 + `cp.async` double-buffering (overhead dominates here).
- Step 16: `hgemm_16_mma_m16n8k16_b256n128_bk32.cu`
  - Increase BM to 256 (more shared memory + more warps; slower here).
- Step 17: `hgemm_17_mma_m16n8k16_bk32_nopad.cu`
  - Remove shared-memory padding (bank conflicts dominate; much slower).

### 5.2 FP32-accumulation ladder (`--accum f32`, `--kernels f32acc`)

- Step 4: `hgemm_4_mma_m16n8k16_f32acc_bk32.cu`
  - First FP32-acc kernel; larger register pressure.
- Step 8: `hgemm_8_mma_m16n8k16_f32acc_b128n64_bk32.cu`
  - Change block shape to (128,64) to reduce registers.
- Step 9: `hgemm_9_mma_m16n8k16_f32acc_b64n128_bk32.cu`
  - Change block shape to (64,128); best large-size kernel in this branch.
- Step 10: `hgemm_10_mma_m16n8k16_f32acc_b64n128_bk64.cu`
  - BK=64 (shared memory + registers increase too much; slower).
- Step 11: `hgemm_11_mma_m16n8k16_f32acc_b64n128_bk32_nopad.cu`
  - Remove padding (bank conflicts dominate; slower).
- Step 12: `hgemm_12_mma_m16n8k16_f32acc_b64n128_bk32_cp_async_b.cu`
  - `cp.async` double-buffering for B only (occupancy drop; slower).
- Step 13: `hgemm_13_mma_m16n8k16_f32acc_b128n128_bk32.cu`
  - Larger block tile (128,128); high registers => 1 block/SM; slower.
- Step 18: `hgemm_18_mma_m16n8k16_f32acc_b64n128_bk32_bswizzle.cu`
  - Add a simple B shared-memory swizzle (helps small sizes slightly).
- Step 19: `hgemm_19_mma_m16n8k16_f32acc_b64n128_bk32_bpad16.cu`
  - Try a different B padding (worse here).
- Step 20: `hgemm_20_mma_m16n8k16_f32acc_b64n128_bk32_cp_async_db.cu`
  - cp.async double-buffering for both A and B (improves mid sizes).
- Step 21: `hgemm_21_mma_m16n8k16_f32acc_b128n128_bk32_lb2.cu`
  - Step 13 + `__launch_bounds__(512, 2)` to reduce registers (did not help).
- Step 22: `hgemm_22_mma_m16n8k16_f32acc_b64n128_bk32_cp_async_ws2.cu`
  - Warp-specialized copy/compute experiment (2 dedicated copy warps).
- Step 23: `hgemm_23_dispatch_autotune_f32acc.cu`
  - Dispatcher + lightweight auto-tuner (best-of envelope in one binary).
- Step 24: `hgemm_24_dispatch_autotune_zoo_f32acc.cu`
  - Bigger dispatcher zoo + auto-tuner (adds Step 18/19/21-style candidates).

## 6. Results

### 6.1 FP32-accumulation results (apples-to-apples vs cuBLAS `--accum f32`)

Best kernel per size (from `results/run_20260215_013238/`):

| size | best kernel | custom TFLOP/s | cuBLAS TFLOP/s | speedup |
|---:|---|---:|---:|---:|
| 256 | hgemm_20_mma_m16n8k16_f32acc_b64n128_bk32_cp_async_db | 3.908 | 4.510 | 0.867 |
| 512 | hgemm_24_dispatch_autotune_zoo_f32acc | 17.165 | 21.978 | 0.781 |
| 1024 | hgemm_20_mma_m16n8k16_f32acc_b64n128_bk32_cp_async_db | 32.996 | 35.986 | 0.917 |
| 2048 | hgemm_20_mma_m16n8k16_f32acc_b64n128_bk32_cp_async_db | 36.867 | 41.063 | 0.898 |
| 4096 | hgemm_9_mma_m16n8k16_f32acc_b64n128_bk32 | 40.518 | 44.150 | 0.918 |
| 6144 | hgemm_24_dispatch_autotune_zoo_f32acc | 38.880 | 44.186 | 0.880 |

Key plots (the full set is under `results/run_20260215_013238/plots/`):

- Step 24 speedup: `results/run_20260215_013238/plots/hgemm_24_dispatch_autotune_zoo_f32acc_speedup.png`
- Best-of speedup: `results/run_20260215_013238/plots/best_of_speedup.png`

### 6.2 FP16-accumulation results (apples-to-apples vs cuBLAS `--accum f16`)

Best kernel per size (from `results/run_20260214_092606/`):

| size | best kernel | custom TFLOP/s | cuBLAS TFLOP/s | speedup |
|---:|---|---:|---:|---:|
| 256 | hgemm_14_mma_m16n8k16_b64n128_bk32 | 4.125 | 4.875 | 0.846 |
| 512 | hgemm_14_mma_m16n8k16_b64n128_bk32 | 18.234 | 32.108 | 0.568 |
| 1024 | hgemm_14_mma_m16n8k16_b64n128_bk32 | 43.570 | 59.095 | 0.737 |
| 2048 | hgemm_2_mma_m16n8k16_bk32 | 60.261 | 76.389 | 0.789 |
| 4096 | hgemm_2_mma_m16n8k16_bk32 | 67.837 | 82.836 | 0.819 |
| 6144 | hgemm_2_mma_m16n8k16_bk32 | 66.890 | 84.327 | 0.793 |

Key plots (the full set is under `results/run_20260214_092606/plots/`):

- Step 2 TFLOP/s: `results/run_20260214_092606/plots/hgemm_2_mma_m16n8k16_bk32_tflops.png`
- Step 2 speedup: `results/run_20260214_092606/plots/hgemm_2_mma_m16n8k16_bk32_speedup.png`

## 7. Why some “optimizations” got slower (resource + occupancy intuition)

On GPUs, “fancy” features can easily backfire if they:

- increase **registers per thread** too much
- increase **shared memory per block** too much
- reduce the number of active blocks/warps per SM (lower occupancy)
- add synchronization overhead without enough compute to hide it

Here are a few key kernels and their compiler-reported resources
(`nvcc -Xptxas=-v`):

| kernel | registers/thread | shared memory/block |
|---|---:|---:|
| hgemm_2_mma_m16n8k16_bk32 | 76 | 18944 B |
| hgemm_3_mma_m16n8k16_bk32_cp_async_db | 94 | 37888 B |
| hgemm_9_mma_m16n8k16_f32acc_b64n128_bk32 | 64 | 13824 B |
| hgemm_12_mma_m16n8k16_f32acc_b64n128_bk32_cp_async_b | 64 | 22528 B |
| hgemm_13_mma_m16n8k16_f32acc_b128n128_bk32 | 80 | 18944 B |
| hgemm_18_mma_m16n8k16_f32acc_b64n128_bk32_bswizzle | 80 | 13824 B |
| hgemm_20_mma_m16n8k16_f32acc_b64n128_bk32_cp_async_db | 70 | 27648 B |
| hgemm_21_mma_m16n8k16_f32acc_b128n128_bk32_lb2 | 60 | 18944 B |

Interpretation examples:

- Step 3 (cp.async + double buffer) doubles shared memory, so fewer blocks fit.
  Even if each block overlaps copy with compute, the occupancy loss can dominate.
- Step 13 (bigger block tile) increases registers per thread, causing only
  **1 block/SM** for that kernel on this GPU.
- Step 18 adds extra address arithmetic (swizzle), which increases registers.

## 8. What we would try next (if we continue the ladder)

Beating cuBLAS is hard because cuBLAS is already heavily tuned.

If we continue, the most promising directions are:

1. **More realistic pipelining** without blowing up shared memory:
   - We tried a simple warp-specialized copy/compute variant (Step 22), but it
     did not win on this GPU for these square shapes.
   - A stronger version would likely need better scheduling (more stages,
     better copy/compute overlap, and more careful warp role assignment).
2. **More advanced shared-memory layouts** for operand B:
   - swizzles/layouts designed specifically for `ldmatrix.trans`
3. **Epilogue and store optimizations**:
   - vectorized stores, reduce conversion overhead (for FP32-acc kernels)
4. **Try cublasLt for baseline sanity**:
   - verify whether `cublasGemmEx` default is already optimal for this GPU

## 9. References (good background reading)

- https://salykova.github.io/gemm-gpu
- PTX ISA docs (for `ldmatrix` / `mma.sync`)
- CUTLASS examples (for modern Tensor Core GEMM pipelines)
