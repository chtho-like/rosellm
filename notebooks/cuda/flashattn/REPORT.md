# FlashAttention on RTX 4070 (SM89): step-by-step kernel optimization

This report builds a step-by-step optimization ladder for **causal scaled
dot-product attention (SDPA) forward**, a common operation in LLM inference, on
the **local RTX 4070 (SM89)**. A unified benchmark compares:

- The production-grade FlashAttention implementation from the Dao-AILab
  repository, distributed as `flash-attn`
- PyTorch `scaled_dot_product_attention` with automatic backend selection
  (`torch_sdpa_auto`)
- Explicit PyTorch backends, to align and understand different implementation paths:
  - `torch_sdpa_flash_pytorch`: PyTorch built-in FlashAttention backend
  - `torch_sdpa_efficient_cutlass`: memory-efficient attention (CUTLASS)
  - `torch_sdpa_cudnn`: cuDNN SDPA backend
- A naive `matmul + softmax + matmul` baseline that materializes the `S×S`
  attention matrix
- Triton implementations `triton_v0` through `triton_v9` in this directory

The number of optimization iterations is intentionally unbounded: additional
`triton_flashattn_*` versions can be added. This report first establishes the
critical path from correctness to **outperforming `torch_sdpa_auto`**.

## 0. Reproducible environment

Hardware:

- GPU: NVIDIA GeForce RTX 4070
- Compute Capability: 8.9 (SM89)
- VRAM: ~11.6 GiB

Software:

- Driver: 580.82.07
- CUDA Toolkit (`nvcc`): 12.8
- PyTorch: 2.9.0+cu128
- Triton: 3.5.0

## 1. Problem definition from first principles

We implement **SDPA forward**.

Given Q, K, and V:

- Shape: `(B, H, S, D)`
  - `B`: batch size
  - `H`: number of heads
  - `S`: sequence length
  - `D`: vector dimension of each head (`head_dim`)
- Causal autoregressive mask: query `i` may attend only to keys and values with
  positions `j <= i`

For one attention head:

1. `scores = (Q @ K^T) * scale`
2. `P = softmax(scores)` (the upper triangle is `-inf` under causal masking)
3. `O = P @ V`

The primary difficulty is memory traffic:

- `scores` and `P` are `S×S`; storing them explicitly is slow and consumes
  substantial VRAM bandwidth.
- FlashAttention uses **tiling plus online softmax** to avoid materializing the
  `S×S` matrices.

## 2. Benchmark methodology

All implementations use `bench_flashattn.py`:

```bash
cd /data/projects/rosellm/notebooks/cuda/flashattn
python bench_flashattn.py \
  --causal 1 --dtype fp16 \
  --batch 1 --heads 32 \
  --seq-lens 256,512,1024,2048,4096 \
  --head-dims 64,128 \
  --warmup 25 --iters 100 --check
```

Timing method:

- Warm up first so kernel JIT compilation and caches stabilize.
- Use CUDA events to measure kernel execution only, excluding Python overhead.

Output files:

- `out/latest/results.csv`: raw table containing `avg_ms` and `tflops` for each backend
- `out/latest/results.csv` also contains:
  - `config`: configuration selected by Triton autotuning or dispatch, useful
    for reviewing and fixing configurations
  - `error`: failure reason when a shape/configuration exceeds shared-memory
    or another resource limit
- `out/latest/plots/*`: automatically generated time and TFLOPs curves

## 3. Optimization ladder: one file per step

Every version implements `flash_attn(q, k, v, causal=True)` and maintains
numerical stability through FP32 max and sum accumulation.

Key changes from v0 through v9:

- `triton_flashattn_0_baseline.py` (`triton_v0_baseline`)
  - Minimal working implementation: online softmax plus tiling
  - Under causal attention, it still traverses every K block and masks future
    tokens to `-inf`, wasting roughly half of the arithmetic
- `triton_flashattn_1_codegen_hints.py` (`triton_v1_codegen_hints`)
  - Adds code-generation hints, such as alignment information for loop indices
- `triton_flashattn_2_autotune.py` (`triton_v2_autotune`)
  - Autotunes v0 over `BLOCK_M`, `BLOCK_N`, warps, and stages
- `triton_flashattn_3_fixed_best.py` (`triton_v3_fixed_best`)
  - Fixes a manually selected configuration, superseded by later versions
- `triton_flashattn_4_causal_cutoff.py` (`triton_v4_causal_cutoff`)
  - **Key optimization**: for causal attention, traverses only through
    `end_n = min(S, start_m + BLOCK_M)`
  - Directly removes many invalid K-block dot products, bringing executed work
    closer to the theoretical useful work
- `triton_flashattn_5_even_fastpath.py` (`triton_v5_even_fastpath`)
  - Uses a mask-free fast path for evenly divisible shapes such as
    `S%BLOCK==0` and `D%16==0`
- `triton_flashattn_6_autotune_cutoff.py` (`triton_v6_autotune_cutoff`)
  - Autotunes the v4+v5 implementation to find the best local configuration
- `triton_flashattn_7_fixed_best_cutoff.py` (`triton_v7_fixed_best_cutoff`)
  - Fixes the best v6 configuration measured locally; it is strong for some
    shapes but not necessarily the most robust
- `triton_flashattn_8_dispatch_fixed_best.py` (`triton_v8_dispatch_fixed_best`)
  - **Dispatches by `head_dim` and `seq_len`**, converting the best v6
    configurations into a directly usable set and avoiding the problem where
    v7 is optimal for D=64 but very slow for D=128
- `triton_flashattn_9_autotune_table.py` (`triton_v9_autotune_table`)
  - Dispatches through a shape-to-configuration table generated by offline
    autotuning, approximating automatic tile selection
  - The default table is `triton_flashattn_v9_table_sm89.json`; generate or
    update it with `tune_triton_flashattn_table.py`

## 4. Key results for the most important shape

One common inference shape is:

- `B=1, H=32, S=2048, D=64`
- dtype: fp16
- causal: True

Command:

```bash
python bench_flashattn.py \
  --causal 1 --dtype fp16 \
  --batch 1 --heads 32 \
  --seq-lens 2048 --head-dims 64 \
  --warmup 50 --iters 200
```

Current measurements, which vary slightly with driver, temperature, and clock
state, include the `flash-attn==2.7.3` comparison:

- `flash_attn`: ~0.512 ms
- `torch_sdpa_auto`: ~0.510 ms
- `triton_v9_autotune_table`: ~0.481 ms (faster)

Corresponding TFLOPs, calculated from theoretical causal FLOPs:

- `flash_attn`: ~33.6 TFLOPs
- `torch_sdpa_auto`: ~33.7 TFLOPs
- `triton_v9_autotune_table`: ~35.7 TFLOPs

On this local GPU, v9 **outperforms** `torch_sdpa_auto` and also **outperforms**
the production `flash-attn` implementation for this shape.

Complete sweep plots:

- `out/latest/plots/time_focus_c1_d64.png`
- `out/latest/plots/tflops_focus_c1_d64.png`

### 4.1 Sweep (S=256/512/1024/2048/4096, D=64, causal=True)

Command:

```bash
python bench_flashattn.py \
  --causal 1 --dtype fp16 \
  --batch 1 --heads 32 \
  --seq-lens 256,512,1024,2048,4096 \
  --head-dims 64,128 \
  --warmup 25 --iters 100 --check
```

The three primary baselines and the final implementation, where the best Triton
version is usually v9:

| S | flash_attn (ms) | torch_sdpa_auto (ms) | best triton (ms) | best vs flash_attn |
|---:|---:|---:|---:|---:|
| 256  | 0.0240 | 0.0238 | 0.0185 | +30.2% |
| 512  | 0.0561 | 0.0552 | 0.0449 | +24.9% |
| 1024 | 0.1566 | 0.1549 | 0.1395 | +12.2% |
| 2048 | 0.5121 | 0.5105 | 0.4810 |  +6.5% |
| 4096 | 1.8494 | 1.8431 | 1.7825 |  +3.8% |

Notes:

- `+X%` means the best Triton implementation is faster, computed as
  `flash_attn_time / best_triton_time - 1`.
- The `S=256` correctness check gives `max_abs_err ~ 4.88e-4`, which is normal
  for FP16.

The complete raw table, including every optimization version and the failure
reason for each shape, is available in:

- `out/latest/results.csv`
- `out/latest/results.json`

### 4.2 Sweep (S=256/512/1024/2048/4096, D=128, causal=True)

These values also come from `out/latest/results.csv`; the best Triton version is
usually v9, v6, or v8:

| S | flash_attn (ms) | best triton (ms) | best vs flash_attn |
|---:|---:|---:|---:|
| 256  | 0.0290 | 0.0308 |  -5.8% |
| 512  | 0.0780 | 0.0888 | -12.2% |
| 1024 | 0.2522 | 0.2897 | -13.0% |
| 2048 | 0.9080 | 0.9811 |  -7.4% |
| 4096 | 3.3950 | 3.5566 |  -4.5% |

Interpretation:

- At `D=128`, the FlashAttention CUDA kernel remains stronger; the Triton
  implementation currently trails by 4% to 13%.
- The gap narrows as `S` grows, indicating that the main weakness is in the
  **small-scale, high-register-pressure** regime.
- v9 dispatches through an offline-autotuned table. For small-S, D=128 shapes
  such as `S=256/512`, it tends to choose smaller tiles such as `BLOCK_M=32`,
  which substantially narrows the gap.
- Some `triton_v1/v3/v4/v5` configurations exceed the **shared-memory limit**
  at `D=128`. `bench_flashattn.py` now records these failures in the `error`
  column and continues the complete sweep instead of aborting the benchmark.

### 4.3 Contribution of every optimization step at S=2048

Within the same sweep, results for `S=2048, D=64, causal=True` are as follows,
in milliseconds:

| backend | avg_ms | relative to flash_attn |
|---|---:|---:|
| `flash_attn` | 0.5121 | baseline |
| `torch_sdpa_auto` | 0.5105 | +0.31% |
| `triton_v0_baseline` | 0.9390 | -45.47% |
| `triton_v1_codegen_hints` | 1.0868 | -52.88% |
| `triton_v2_autotune` | 0.8717 | -41.25% |
| `triton_v3_fixed_best` | 1.0880 | -52.94% |
| `triton_v4_causal_cutoff` | 0.5903 | -13.26% |
| `triton_v5_even_fastpath` | 0.5846 | -12.40% |
| `triton_v6_autotune_cutoff` | 0.4825 | +6.13% |
| `triton_v7_fixed_best_cutoff` | 0.4844 | +5.71% |
| `triton_v8_dispatch_fixed_best` | 0.4857 | +5.43% |
| `triton_v9_autotune_table` | 0.4810 | +6.46% |

First-principles interpretation:

- v0/v1/v2/v3 **resemble FlashAttention but execute many invalid
  multiplications under causal attention**.
  - They still traverse every K block and erase future tokens with an `-inf` mask.
  - They therefore pay nearly the non-causal arithmetic cost while only half
    of the work is useful under the causal mask.
- v4 removes the largest source of causal waste through the `end_n` cutoff.
  - This is often one of the most important explanations for poor causal
    inference performance.
- v5 removes boundary masks for evenly divisible shapes, yielding a small but
  consistent gain.
- v6/v7/v8/v9 tune tiles, warps, and stages on top of v4+v5, ultimately
  outperforming `flash_attn` and `torch_sdpa_auto` for the target shape.

This explains the ladder structure:

- Several optimizations are structural rather than incremental; v4 deletes a
  large block of invalid computation.
- Fine-grained tuning of `BLOCK_M`, `BLOCK_N`, warps, and stages matters only
  after those structural inefficiencies have been removed in v6 and later.

## 5. cuBLAS baseline: why fusion is faster

The naive `torch_matmul_softmax` implementation performs:

1) `Q @ K^T` using cuBLAS GEMM, producing `S×S`
2) `softmax(scores)`, reading and writing `S×S` again
3) `P @ V` using cuBLAS GEMM

This writes the `S×S` matrix back to VRAM, creating substantial bandwidth
and cache pressure.

FlashAttention-style algorithms, including these Triton kernels, use online
softmax and fuse softmax with `P @ V` into one streaming pass. Avoiding the
materialized `S×S` matrix improves end-to-end performance.

## 6. Nsight Compute (`ncu`) analysis of v9

Run `ncu` through `sudo` to access hardware counters:

```bash
sudo ncu --set full --target-processes all \
  --kernel-name-base demangled -k regex:_flashattn_fwd_kernel \
  --launch-skip 10 --launch-count 1 \
  -o out/latest/ncu_v9 \
  /data/projects/anaconda3/bin/python profile_flashattn.py \
    --backend v9 --init empty --warmup 10
```

Generated report:

- `out/latest/ncu_v9.ncu-rep`

Example command for printing key report sections:

```bash
ncu --import out/latest/ncu_v9.ncu-rep --page details --print-summary per-kernel
```

Selected measurements from one local sample:

- Duration: ~480 us for one `S=2048, D=64` forward pass
- Compute (SM) Throughput: ~40.8%
- Memory Throughput: ~30.8%
- Registers / thread: 120
- Dynamic shared memory / block: ~16.4 KiB

Interpretation:

- This kernel has a mixed compute and memory-access bottleneck; neither resource
  can realistically remain at 100% utilization.
- High `regs/thread` indicates a deliberate exchange of registers for less
  memory traffic or recomputation. This is reasonable but reduces occupancy.

### 6.1 `ncu` comparison with FlashAttention for the same shape

Use `profile_flashattn.py` to profile the `flash_attn` forward kernel as well:

```bash
sudo ncu -f --set full --target-processes all \
  --kernel-name-base demangled -k regex:flash_fwd \
  --launch-count 1 \
  -o out/latest/ncu_flash_attn \
  /data/projects/anaconda3/bin/python profile_flashattn.py \
    --backend flash_attn --init empty --warmup 0
```

Key `ncu` metrics for the same shape:

| kernel | Duration (us) | Compute Throughput | Memory Throughput | Regs/thread | Dyn smem/block |
|---|---:|---:|---:|---:|---:|
| `flash_fwd_kernel` (flash-attn) | 513.76 | 39.29% | 25.15% | 255 | 49.15 KiB |
| `_flashattn_fwd_kernel` (v9) | 479.71 | 40.84% | 30.82% | 120 | 16.38 KiB |

Interpretation:

- v9 is faster at approximately 480 us versus 514 us.
- v9 consumes fewer registers and less dynamic shared memory, giving the
  scheduler more flexibility.
- v9 reaches higher reported memory and compute throughput percentages,
  indicating better overall hardware utilization.

## 7. Recommended next optimization steps

To continue pursuing maximum performance:

1) Continue optimizing the `D=128` kernel with lower register pressure and
   better tiles; it still trails `flash-attn`.
2) Refine the K/V prefetch pipeline with more aggressive `num_stages` settings
   and better-matched tiles.
3) Remove masks more aggressively through specialized causal diagonal and tail blocks.
4) Compare `ncu` metrics systematically against `flash-attn` and `torch_sdpa_auto`.

## 8. FlashAttention (`flash-attn`) comparison status

Compiling `flash-attn` from source is expensive. Use the following command to
avoid compiling unused `sm90`, `sm100`, and `sm120` targets:

```bash
FLASH_ATTN_CUDA_ARCHS=80 NVCC_THREADS=8 \
  python -m pip install --no-cache-dir flash-attn==2.7.3
```

After installation, rerun the benchmark to add a `flash_attn` row to the tables
and plots.

Completed: across the `B=1, H=32, D=64, causal=True` sweep with
`S=256/512/1024/2048/4096`, the best Triton implementation, currently usually
`triton_v9_autotune_table`, is faster than `flash_attn`; see Section 4.1.

Remaining work: `D=128` still trails `flash_attn` by approximately 4% to 13%;
see Section 4.2.
