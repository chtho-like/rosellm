# cuda/gemm2

这是一套更“拷打向”的 CUDA GEMM 手撕与优化阶梯（`C = A @ B`），目标是把面试里能追问到的层次从最浅一路铺到 WMMA、cp.async、多 stage、MMA PTX、SMEM swizzle、warp shuffle collective store。

如果你简历里写了“精通 GEMM 优化”，面试官大概率会沿着这些层层追问；这套代码的结构就是按“追问路径”组织的。

更完整的“拷打清单”（从零基础到极深）在 `cuda/gemm2/INTERVIEW.md`。

## 你需要什么

- `nvcc`（CUDA Toolkit）
- 一块 NVIDIA GPU（没有 GPU 也能编译，但无法运行）

## 如何选择 `-arch`

`nvcc` 需要指定目标 GPU 架构（Compute Capability）。

常见例子：

- `-arch=sm_80`: A100 / A30（Ampere）
- `-arch=sm_89`: RTX 4090 / RTX 4070 / L20（Ada）
- `-arch=sm_90`: H100 / H200（Hopper）

如果你不确定，先用 `sm_80` 编译能覆盖大多数新卡，但性能可能不是最优。

## Quick start

在本目录下编译并运行一个文件即可（每个 `.cu` 都是独立可运行的 demo）：

```bash
cd cuda/gemm2
nvcc -O3 -std=c++17 -lineinfo -arch=sm_80 gemm_2_thread_tile_8x8_float4.cu \
  -o gemm_2
./gemm_2 1024 1024 1024 0
```

## 文件梯度（从易到难）

- `gemm_0_naive.cu`: 1 thread = 1 output，建立索引/布局心智模型
- `gemm_1_tiled_smem.cu`: 共享内存分块（Block tile + K tile）
- `gemm_2_thread_tile_8x8_float4.cu`: 线程寄存器分块（8x8）+ float4 向量化 LD/ST
- `gemm_3_smem_padding_bcf.cu`: SMEM padding + A tile 转置布局，降低 bank conflicts
- `gemm_4_double_buffered.cu`: SMEM 双缓冲（减少同步，做出 pipeline 形状）
- `gemm_5_cp_async_multistage.cu`: cp.async（SM80+）+ 3-stage pipeline（异步搬运）
- `gemm_6_splitk_workspace.cu`: Split-K（workspace + reduce，避免 atomics）
- `gemm_7_epilogue_fusion.cu`: Epilogue fusion（GEMM + bias + ReLU）
- `gemm_8_wmma_fp16_tensorcore.cu`: WMMA FP16（Tensor Cores），4 warps/block 做 32x32
- `gemm_9_wmma_tf32_stages.cu`: WMMA TF32（SM80+）+ cp.async + 3-stage + block swizzle
- `gemm_10_mma_m16n8k16_smem_swizzle.cu`: MMA PTX m16n8k16 + ldmatrix + SMEM swizzle + collective store

## LeetCUDA 优化点覆盖清单（逐条对齐）

下面这些点来自 LeetCUDA 的 GEMM/HGEMM/SGEMM 优化维度；每一项都在本目录里有对应落点：

| LeetCUDA 优化点 | 在本目录的落点 |
|---|---|
| CUDA Cores | `gemm_0` ~ `gemm_7` |
| Loop over K | 几乎所有文件（从 `gemm_0` 开始） |
| Tile Block (BMxBK) | `gemm_1`, `gemm_3` ~ `gemm_5`, `gemm_9`, `gemm_10` |
| Tile Threads (T 8x8) | `gemm_2` ~ `gemm_5`, `gemm_7` |
| Pack LDST (128 bits) | `gemm_2` ~ `gemm_5`, `gemm_7`（float4），`gemm_10`（int4/8 half） |
| SMEM Padding | `gemm_3` ~ `gemm_5`, `gemm_9`, `gemm_10` |
| Double Buffers | `gemm_4`（SMEM 双缓冲） |
| Copy Async (cp.async) | `gemm_5`, `gemm_9` |
| Multi Stages (2~4) | `gemm_5`（3 stages）, `gemm_9`（3 stages） |
| Register Double Buffers | `gemm_4`（gmem->regs 预取下一 tile） |
| Block Swizzle | `gemm_9`（grid.z 因式分解实现） |
| Warp Swizzle | `gemm_9`, `gemm_10`（改变 warp 内 tile 的遍历顺序） |
| WMMA (m16n16k16 / TF32 m16n16k8) | `gemm_8`（FP16）, `gemm_9`（TF32） |
| Tile MMAs | `gemm_9`（warp 内 2x4 WMMA tiles）, `gemm_10`（warp 内 4x4 MMA tiles） |
| Tile Warps | `gemm_8`（4 warps/block）, `gemm_9`（8 warps/block）, `gemm_10`（8 warps/block） |
| MMA (m16n8k16) | `gemm_10` |
| SMEM Swizzle / Permuted | `gemm_10`（`SwizzleACol`） |
| Collective Store (Shfl) | `gemm_10`（warp shfl 聚合成 128-bit store） |
| Layout NN | `gemm_0` ~ `gemm_7`, `gemm_9`, `gemm_10` |
| Layout TN | `gemm_8`（B 为 KxN 的 col-major，等价于 TN 的存储语义） |
| SGEMM FP32 / TF32 | `gemm_0` ~ `gemm_7`（FP32 CUDA cores）, `gemm_9`（TF32 Tensor Cores） |

## 重要提示（维度限制）

一些更“硬核”的 kernel 为了把重点放在优化结构上，会要求维度是某些 tile 的倍数：

- `gemm_8_wmma_fp16_tensorcore.cu`: `M,N` 是 32 的倍数，`K` 是 16 的倍数
- `gemm_9_wmma_tf32_stages.cu`: `M,N` 是 128 的倍数，`K` 是 8 的倍数（并且 `N,K` 还需要满足 float4 copy 的条件）
- `gemm_10_mma_m16n8k16_smem_swizzle.cu`: `M,N` 是 128 的倍数，`K` 是 16 的倍数

这些限制在真实生产里可以通过更复杂的 “prologue/epilogue + 边界处理” 消除；但面试手撕时通常先把主干写对，再讨论边界版本。

