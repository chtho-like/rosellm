# SGEMM (FP32, CUDA Cores only) Optimization Ladder on RTX 4070

This report documents a step-by-step attempt to optimize a custom FP32
SGEMM kernel (`C = A @ B`, row-major) on the **NVIDIA GeForce RTX 4070**
(SM89), **without using Tensor Cores** (no WMMA/MMA/TF32).

The goal is to get as close as possible to (and ideally exceed) cuBLAS
SGEMM performance on this exact GPU, while keeping each optimization step
in its own `.cu` file with readable diffs.

All kernel sources live in `notebooks/cuda/gemm_cuda/`.

## 0. What is SGEMM (from first principles)

We want to compute:

- `A` is an `M x K` matrix (FP32)
- `B` is a `K x N` matrix (FP32)
- `C` is an `M x N` matrix (FP32)

For every output element:

- `C[i, j] = sum_{kk=0..K-1} A[i, kk] * B[kk, j]`

This is *a lot* of multiply-add operations:

- FLOPs (floating point operations) ≈ `2 * M * N * K`
  - `1` multiply + `1` add per `kk`

SGEMM is the single most important kernel behind linear layers in deep
learning, and a classic high-performance computing benchmark.

## 1. Constraints and fairness rules used here

1. **No Tensor Cores**:
   - cuBLAS is configured to use true FP32 compute (`CUBLAS_COMPUTE_32F`),
     with default math mode (`CUBLAS_DEFAULT_MATH`).
   - We do **not** enable TF32 (`--tf32` is not used).

2. **Same math definition**:
   - We use fused multiply-add (`fmaf`) as the basic primitive.

3. **Same measurement methodology** for cuBLAS and custom kernels:
   - Warmup iterations
   - CUDA event timing on the same stream
   - Enough iterations to reach a minimum benchmark time

4. **One kernel per binary**:
   - Each `sgemm_*.cu` compiles to a standalone executable via
     `sgemm_main.cuh`.

## 2. Hardware + software environment

- GPU: `NVIDIA GeForce RTX 4070` (Compute Capability `8.9`, SM89)
- Driver: `580.82.07`
- CUDA toolkit: `nvcc 12.8` (as installed on this machine)

You can confirm locally with:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvcc --version
```

Profiling note:

- Nsight Compute (`ncu`) is installed, but this machine currently does not
  allow access to GPU performance counters for non-privileged users
  (`ERR_NVGPUCTRPERM`).
- Because of that, this report focuses on:
  - end-to-end kernel timing (CUDA events), and
  - compiler-reported register/shared-memory usage (`ptxas -v`).

## 3. Benchmark harness and what “TFLOP/s” means

The harness lives in `sgemm_bench.cuh` and does:

- Allocate device buffers
- Fill inputs with a deterministic pseudo-random pattern
- Run cuBLAS (`cublasGemmEx`) and the custom kernel
- Time both with CUDA events

TFLOP/s is computed as:

- `TFLOP/s = (2*M*N*K) / (time_seconds) / 1e12`

All benchmarks in this report use **square** shapes (`M=N=K=size`).

## 4. How to build and run

From `notebooks/cuda/gemm_cuda/`:

Build everything:

```bash
make -j ARCH=sm_89
```

Run one kernel (example):

```bash
./sgemm_5_double_buffered --device 0 --sizes 1024,2048,4096
```

Generate results (JSON + plots) for all kernels:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install matplotlib

python3 bench_and_plot.py \
  --device 0 \
  --sizes 256,512,1024,1536,2048,2560,3072,3584,4096,6144 \
  --warmup 5 --min-ms 200 --max-iters 200 --no-check
```

This report’s results were generated into:

- `results/run_20260215_012528/`

That folder contains:

- One `*.json` per kernel (numbers)
- One `*.log` per kernel (raw stdout/stderr)
- `plots/*.png` (TFLOP/s curves + speedup curves)
- `summary.md` (quick summary)
- `failures.json` (kernels that failed to run)

## 5. Optimization steps (one file per step)

This ladder is intentionally incremental. The key idea is always:

- **Reuse** data (shared memory tiling)
- **Increase compute per thread** (register tiling)
- **Make loads/stores efficient** (vectorized float4, avoid bank conflicts)
- **Pipeline** work (double buffering)

### Step 0: Naive

- File: `sgemm_0_naive.cu`
- Idea: 1 thread computes 1 `C[i,j]`, reads `A`/`B` from global memory

Expected: correct but very slow (no reuse).

### Step 1: Shared-memory tiling

- File: `sgemm_1_tiled_smem.cu`
- Idea: stage a `16x16` tile of A/B into shared memory

Expected: still simple, but better reuse inside a block.

### Step 2: Thread (register) tiling

- File: `sgemm_2_thread_tile_8x8.cu`
- Idea: each thread computes an `8x8` output tile in registers

This increases arithmetic intensity per thread a lot.

### Step 3: Vectorized float4 global loads/stores

- File: `sgemm_3_thread_tile_8x8_vec4.cu`
- Idea: use `float4` (128-bit) loads/stores when `N%4==0 && K%4==0`

This reduces memory transaction overhead when staging tiles.

### Step 4: Shared-memory layout to reduce bank conflicts

- File: `sgemm_4_smem_pad_bcf.cu`
- Idea:
  - store A as `[BK][BM + PAD]` (transposed in shared memory)
  - pad to break conflict-heavy strides

This reduces shared-memory bank conflicts in the inner loop.

### Step 5: Double-buffered shared memory (best for large sizes here)

- File: `sgemm_5_double_buffered.cu`
- Idea: ping-pong two shared-memory buffers to prefetch the next tile

This is the strongest kernel we found for large sizes (≥ 2560) in this
exploration.

### Step 6: cp.async multi-stage (exploration; slower here)

- File: `sgemm_6_cp_async_multistage.cu`
- Idea: use `cp.async` (SM80+) to overlap B tile copies with compute

On this GPU and for this SIMT FP32 kernel, it did **not** beat the
double-buffered version in Step 5.

### Step 7–11: Parameter experiments (mostly did not beat Step 5)

These files explore alternative tilings and mappings:

- `sgemm_7_db_b128n64_tm8tn4.cu` (smaller BN, fewer registers/thread)
- `sgemm_8_db_b128n128_tm8tn4.cu` (512 threads/block; slower)
- `sgemm_9_double_buffered_bk16.cu` (BK=16; similar/slower)
- `sgemm_10_db_tm16tn4.cu` (different block geometry; slower)
- `sgemm_11_db_b256n128_tm16tn8.cu` (very register-heavy; slower)

### Step 12: Larger tile 256x128 (beats cuBLAS at size=2048)

- File: `sgemm_12_db_b256n128_tm8tn8.cu`
- Idea: compute a larger threadblock tile (256x128) with 512 threads

This kernel slightly **beats cuBLAS at `M=N=K=2048`** in our run.

### Step 13: 256x128 with BK=16 (runs with a register cap)

- File: `sgemm_13_db_b256n128_bk16.cu`
- Status: **runs** (after adding `--maxrregcount=128` for this target in the
  `Makefile`)

Why this was needed:

- The kernel uses a **512-thread block**.
- Without a register cap, it compiled to **141 registers/thread** (too many).
  - `141 regs/thread * 512 threads = 72192 regs/block`, which cannot fit in
    one SM’s register file on this GPU.
- With `--maxrregcount=128`, it compiles to **128 regs/thread** and launches.

Result:

- Step 13 is the first 256x128 BK=16 variant in this ladder that reliably
  reaches ~**1.05x** cuBLAS at `M=N=K=2048` on RTX 4070.
- Later, Step 20 improves this family slightly and becomes the current best.

### Step 14: 256x128, BK=16, PAD=8 (single-buffer; slower)

- File: `sgemm_14_sb_b256n128_bk16_pad8.cu`
- Idea: keep BK=16 and bring PAD back, but give up double-buffering so
  static shared memory stays under 48 KiB.

Result (honest):

- This variant is **significantly slower** than Step 13 on the sizes we care
  about, so it is mainly a “why this tradeoff is bad” datapoint.

### Step 15: 256x128, BK=32 (single-buffer; slower)

- File: `sgemm_15_sb_b256n128_bk32.cu`
- Idea: increase BK from 16 → 32 to reduce K-tiles (fewer barriers / loop
  overhead), while keeping shared memory at ~48 KiB per block.

Notes:

- This kernel uses a 512-thread block and needs a register cap to launch on
  RTX 4070 (`--maxrregcount=128` is set for this target in the `Makefile`).
- After fixing a shared-memory indexing bug (B tile indexing must be local, not
  global), it runs correctly.

Result:

- Even with fewer K-tiles, it is **slower** than the best BK=16 kernels in this
  ladder (Step 13 / Step 5), likely due to lack of double buffering and higher
  register/shared-memory pressure.

### Step 16: 256x128, BK=32, vectorized A loads (no win)

- File: `sgemm_16_sb_b256n128_bk32_vec4a.cu`
- Idea: try a more “instruction-dense” inner loop by vectorizing some A-side
  loads and explicitly doing the `Fma4` work on the vector lanes.

Result (honest):

- This did **not** improve performance on RTX 4070 in our runs, likely because
  the kernel remains limited by deeper scheduling/occupancy factors rather than
  a few scalar instructions in the load path.

### Step 17: 256x128, BK=32 + double buffering via opt-in dynamic shared memory

- File: `sgemm_17_db_b256n128_bk32_optin.cu`
- Idea:
  - Step 13 already uses ~48 KiB shared memory with BK=16 double buffering.
  - For BK=32 double buffering we need ~96 KiB shared memory per block.
  - Opt-in to the larger shared-memory limit (~99 KiB) and use dynamic shared
    memory (`extern __shared__`) to allocate the 2-stage buffers.

Technique demonstrated:

- Host-side opt-in:
  - `cudaDevAttrMaxSharedMemoryPerBlockOptin`
  - `cudaFuncAttributeMaxDynamicSharedMemorySize`
  - `cudaFuncAttributePreferredSharedMemoryCarveout`

Result (honest):

- This kernel runs correctly, but it is **not faster** than Step 13 on this GPU.
- Under the 512-thread + 128-reg/thread constraint, it ends up with register
  spilling (seen via `ptxas -v`), which hurts throughput.

### Step 18: BK=32 opt-in + staged prefetch/stores (reduce reg pressure)

- File: `sgemm_18_db_b256n128_bk32_optin_staged_prefetch.cu`
- Idea: reduce peak live registers by:
  - splitting next-tile prefetch into two groups, and
  - storing those groups into the “next” shared-memory stage during the compute
    loop.

Result (honest):

- This reduces spilling vs Step 17, but it still spills and does **not** win in
  wall-clock performance (extra instructions + less overlap).

### Step 19: Register-capped variant of the “aggressive” TM=16 kernel

- File: `sgemm_19_db_b256n128_tm16tn8_rc128.cu`
- Idea: compile the Step 11-style kernel with `--maxrregcount=128` to trade
  registers for occupancy.

Result (honest):

- This is mostly a **warning example**: capping a very register-heavy kernel can
  cause *massive* local-memory spilling, which is usually worse than low
  occupancy.

### Step 20: BK=16 double buffering + PAD=8 via opt-in dynamic shared memory

- File: `sgemm_20_db_b256n128_bk16_pad8_optin.cu`
- Idea:
  - Keep the Step 13 compute mapping (256x128 block tile, BK=16).
  - Add padding (`PAD=8`) to reduce bank conflicts in shared memory.
  - Use opt-in dynamic shared memory so we can exceed the default 48 KiB limit
    while still keeping 2-stage double buffering.

Result:

- This is now the **best** variant of the 256x128 BK=16 family in this ladder,
  and slightly improves large-size performance (e.g. `4096^3`) while keeping
  correctness.

### Step 21: Dispatcher + lightweight auto-tuner (kernel family selection)

- File: `sgemm_21_dispatch_autotune.cu`
- Idea:
  - Keep a small set of kernel families in one binary (like a real library).
  - For each shape, benchmark the candidates once and cache the best choice.

Implementation notes:

- The candidate zoo here is intentionally small:
  - Step 7-like kernel (good for small sizes)
  - Step 5-like kernel (often good for mid sizes)
  - Step 20-like kernel (often good for large sizes)
- The large (512-thread) candidate is register-capped *per-kernel* using
  `__launch_bounds__(..., 1)`.
  - This avoids using a global `--maxrregcount` on the whole translation unit,
    which can accidentally introduce spilling in other candidates and break the
    tuner’s decision.

Result:

- Step 21 does not introduce a new math trick; it packages multiple kernels and
  reliably reaches the “best-of per size” envelope automatically.

### Step 23: Smaller CTA tile (BM=128, BN=128) + double-buffering

- File: `sgemm_23_db_b128n128_bk16_pad8.cu`
- Idea:
  - Try a smaller block tile to allow more concurrent blocks/warps per SM.
  - Keep the same “classic” ingredients: shared-memory tiling, A transposed in
    SMEM, padding, and 2-stage double buffering.

Result (honest):

- On this GPU and these square shapes, this does **not** beat the best kernels
  above; the smaller tile has lower arithmetic intensity and loses at large
  sizes.

### Step 24: cp.async pipelining for B (SM80+)

- File: `sgemm_24_db_b256n128_bk16_pad8_optin_cp_async_b.cu`
- Idea:
  - Keep Step 20’s mapping and shared-memory layout.
  - Use `cp.async` for operand **B** (global->shared) so the copy can overlap
    with the FMA-heavy compute loop.

Result (honest):

- This demonstrates a real pipelining technique, but it does not consistently
  win here (the extra instructions and synchronization still matter).

### Step 25: Bigger dispatcher zoo (more kernel families + auto-tune)

- File: `sgemm_25_dispatch_autotune_zoo.cu`
- Idea:
  - Start from Step 21 (dispatch + per-shape auto-tune).
  - Expand the candidate set with more advanced techniques:
    - Step 22-style **warp-level broadcast** (reduce redundant SMEM loads).
    - Step 18-style **BK=32 staged prefetch** (deeper software pipeline with
      lower peak register pressure).

Implementation notes:

- This translation unit is compiled with `--maxrregcount=128` so the larger
  (512-thread) candidates always launch on RTX 4070.

Result:

- Step 25 improves the best-of envelope a bit on smaller sizes (e.g. 256/512),
  and keeps the Step 21-style “best kernel per size” behavior in one binary.

## 6. Results (numbers)

The full per-kernel JSON outputs are under:

- `results/run_20260215_012528/*.json`

### 6.1 Best kernel per size (across this ladder)

| size (M=N=K) | best kernel | best custom TFLOP/s | cuBLAS TFLOP/s | speedup |
|---:|---|---:|---:|---:|
| 256 | sgemm_25_dispatch_autotune_zoo | 1.795 | 4.046 | 0.444 |
| 512 | sgemm_25_dispatch_autotune_zoo | 7.510 | 8.047 | 0.933 |
| 1024 | sgemm_21_dispatch_autotune | 12.069 | 12.829 | 0.941 |
| 1536 | sgemm_21_dispatch_autotune | 12.273 | 13.369 | 0.918 |
| 2048 | sgemm_25_dispatch_autotune_zoo | 14.598 | 13.875 | 1.052 |
| 2560 | sgemm_5_double_buffered | 14.552 | 15.043 | 0.967 |
| 3072 | sgemm_5_double_buffered | 14.548 | 15.455 | 0.941 |
| 3584 | sgemm_25_dispatch_autotune_zoo | 14.997 | 15.182 | 0.988 |
| 4096 | sgemm_21_dispatch_autotune | 14.709 | 16.424 | 0.896 |
| 6144 | sgemm_21_dispatch_autotune | 15.287 | 16.055 | 0.952 |

Takeaway:

- We exceed cuBLAS at **size=2048** with Step 25 (~**1.05x**).
- For larger sizes, our best kernels are still behind cuBLAS in this run
  (e.g. ~0.90x at size=4096).

## 7. Results (plots)

All plots are already generated:

- `results/run_20260215_012528/plots/`

Each kernel has two plots:

- `*_tflops.png`: custom TFLOP/s vs cuBLAS TFLOP/s
- `*_speedup.png`: speedup vs cuBLAS

Examples to look at first:

- Best-of envelope: `results/run_20260215_012528/plots/best_of_speedup.png`
- Step 5 (best large-size in this ladder): `results/run_20260215_012528/plots/sgemm_5_double_buffered_tflops.png`
- Step 25 (bigger dispatch zoo): `results/run_20260215_012528/plots/sgemm_25_dispatch_autotune_zoo_speedup.png`

## 8. Why beating cuBLAS is hard (honest conclusion)

Even without Tensor Cores, cuBLAS FP32 SGEMM uses extremely tuned SIMT
kernels (often CUTLASS-derived), with carefully chosen tile shapes,
instruction scheduling, and architecture-specific tuning.

This ladder implements the “standard” hand-optimized ingredients:

- shared-memory tiling
- register tiling
- vectorized loads/stores
- shared-memory bank-conflict mitigation
- double buffering / pipelining

We get close (and beat cuBLAS for one size), but consistently beating
cuBLAS across all large square sizes on Ada without Tensor Cores would
likely require:

- more warp-level specialization (warp tiles + warp scheduling),
- more careful instruction scheduling / software pipelining,
- and/or auto-tuning across multiple kernel families (dispatch).

Step 25 is the start of that “production-style” approach, but consistently
beating cuBLAS across *most* large square sizes would likely require:

- more kernel families (different BM/BN/BK regimes),
- deeper pipelining with fewer synchronization points,
- and tile-parameter auto-tuning (not just choosing among a fixed small zoo).
