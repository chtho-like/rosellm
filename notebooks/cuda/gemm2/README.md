# cuda/gemm2

This is a more "interrogation-oriented" progression for implementing and optimizing CUDA GEMM (`C = A @ B`) from scratch. Its goal is to cover every level an interviewer might probe, from the most basic implementation through WMMA, cp.async, multistage pipelines, MMA PTX, SMEM swizzling, and warp-shuffle collective stores.

If your resume says that you are "proficient in GEMM optimization," an interviewer will very likely drill down through these layers. This codebase is organized around that sequence of follow-up questions.

The more complete "interrogation checklist," ranging from absolute basics to advanced internals, is in `cuda/gemm2/INTERVIEW.md`.

## Requirements

- `nvcc` (CUDA Toolkit)
- An NVIDIA GPU (you can compile without a GPU, but you cannot run the programs)

## Choosing `-arch`

`nvcc` requires a target GPU architecture (Compute Capability).

Common examples:

- `-arch=sm_80`: A100 / A30 (Ampere)
- `-arch=sm_89`: RTX 4090 / RTX 4070 / L20 (Ada)
- `-arch=sm_90`: H100 / H200 (Hopper)

If you are unsure, compiling with `sm_80` will support most recent GPUs, although performance may not be optimal.

## Quick start

Compile and run any one file from this directory (each `.cu` file is a standalone runnable demo):

```bash
cd cuda/gemm2
nvcc -O3 -std=c++17 -lineinfo -arch=sm_80 gemm_2_thread_tile_8x8_float4.cu \
  -o gemm_2
./gemm_2 1024 1024 1024 0
```

## File progression (easiest to hardest)

- `gemm_0_naive.cu`: 1 thread = 1 output; establishes the indexing and layout mental model
- `gemm_1_tiled_smem.cu`: shared-memory tiling (block tile + K tile)
- `gemm_2_thread_tile_8x8_float4.cu`: per-thread register tiling (8x8) + vectorized float4 LD/ST
- `gemm_3_smem_padding_bcf.cu`: SMEM padding + transposed A-tile layout to reduce bank conflicts
- `gemm_4_double_buffered.cu`: SMEM double buffering (reduces synchronization and establishes a pipeline structure)
- `gemm_5_cp_async_multistage.cu`: cp.async (SM80+) + 3-stage pipeline (asynchronous data movement)
- `gemm_6_splitk_workspace.cu`: Split-K (workspace + reduction, avoiding atomics)
- `gemm_7_epilogue_fusion.cu`: epilogue fusion (GEMM + bias + ReLU)
- `gemm_8_wmma_fp16_tensorcore.cu`: WMMA FP16 (Tensor Cores), with 4 warps/block computing a 32x32 tile
- `gemm_9_wmma_tf32_stages.cu`: WMMA TF32 (SM80+) + cp.async + 3-stage pipeline + block swizzle
- `gemm_10_mma_m16n8k16_smem_swizzle.cu`: MMA PTX m16n8k16 + ldmatrix + SMEM swizzle + collective store

## LeetCUDA optimization coverage checklist (item-by-item mapping)

The following items come from LeetCUDA's GEMM/HGEMM/SGEMM optimization dimensions. Every item maps to a corresponding implementation in this directory:

| LeetCUDA optimization | Implementation in this directory |
|---|---|
| CUDA Cores | `gemm_0` ~ `gemm_7` |
| Loop over K | Nearly every file (starting with `gemm_0`) |
| Tile Block (BMxBK) | `gemm_1`, `gemm_3` ~ `gemm_5`, `gemm_9`, `gemm_10` |
| Tile Threads (T 8x8) | `gemm_2` ~ `gemm_5`, `gemm_7` |
| Pack LDST (128 bits) | `gemm_2` ~ `gemm_5`, `gemm_7` (float4), `gemm_10` (int4/8 half values) |
| SMEM Padding | `gemm_3` ~ `gemm_5`, `gemm_9`, `gemm_10` |
| Double Buffers | `gemm_4` (SMEM double buffering) |
| Copy Async (cp.async) | `gemm_5`, `gemm_9` |
| Multi Stages (2~4) | `gemm_5` (3 stages), `gemm_9` (3 stages) |
| Register Double Buffers | `gemm_4` (prefetch the next tile from gmem to registers) |
| Block Swizzle | `gemm_9` (implemented by factorizing grid.z) |
| Warp Swizzle | `gemm_9`, `gemm_10` (changes the traversal order of tiles within a warp) |
| WMMA (m16n16k16 / TF32 m16n16k8) | `gemm_8` (FP16), `gemm_9` (TF32) |
| Tile MMAs | `gemm_9` (2x4 WMMA tiles per warp), `gemm_10` (4x4 MMA tiles per warp) |
| Tile Warps | `gemm_8` (4 warps/block), `gemm_9` (8 warps/block), `gemm_10` (8 warps/block) |
| MMA (m16n8k16) | `gemm_10` |
| SMEM Swizzle / Permuted | `gemm_10` (`SwizzleACol`) |
| Collective Store (Shfl) | `gemm_10` (warp-shuffle aggregation into 128-bit stores) |
| Layout NN | `gemm_0` ~ `gemm_7`, `gemm_9`, `gemm_10` |
| Layout TN | `gemm_8` (B is KxN column-major, equivalent to TN storage semantics) |
| SGEMM FP32 / TF32 | `gemm_0` ~ `gemm_7` (FP32 CUDA cores), `gemm_9` (TF32 Tensor Cores) |

## Important note (dimension constraints)

Some of the more advanced kernels require dimensions to be multiples of particular tile sizes so that the implementation can focus on the optimization structure:

- `gemm_8_wmma_fp16_tensorcore.cu`: `M,N` must be multiples of 32, and `K` must be a multiple of 16
- `gemm_9_wmma_tf32_stages.cu`: `M,N` must be multiples of 128, and `K` must be a multiple of 8 (`N,K` must also satisfy the requirements for float4 copies)
- `gemm_10_mma_m16n8k16_smem_swizzle.cu`: `M,N` must be multiples of 128, and `K` must be a multiple of 16

Production implementations can remove these restrictions with more elaborate prologue/epilogue and boundary handling. In a whiteboard coding interview, however, the usual approach is to implement the main path correctly first and then discuss a boundary-safe version.
