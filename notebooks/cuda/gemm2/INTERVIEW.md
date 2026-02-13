# CUDA GEMM 面试“拷打”全景路线图（从零到极深）

你可以把 CUDA GEMM 面试看成一棵会不断向下“追问”的树：

1) 先问你是否理解 GEMM 的数学与内存布局  
2) 再问你能不能把它映射到 GPU 的线程/内存层次  
3) 再一路追到共享内存、寄存器、流水线、cp.async、多 stage  
4) 最后追到 Tensor Cores：WMMA → MMA PTX → ldmatrix → SMEM swizzle → collective store  

本目录的代码是按这条追问路径组织的：`cuda/gemm2/gemm_*.cu`。

读法建议：

- 纯新手：按 Level 0 → 20 顺序读，每个 Level 只抓“核心概念 + 一两个坑”
- 面试准备：把每个 Level 的“你必须能讲清楚/写出来”当作 checklist
- 真正手撕：按每个 Level 的 Hands-on 指向的 `.cu` 文件练

## Level 0：你真的懂 GEMM 是什么吗？

你必须能从零解释清楚：

- GEMM 是什么：`C = A @ B`，其中 `A` 是 `M×K`，`B` 是 `K×N`，`C` 是 `M×N`
- 单个输出元素：`C[i,j] = sum_{kk=0..K-1} A[i,kk] * B[kk,j]`
- Row-major（行主序）索引公式：
  - `A[i,kk]` 在内存里是 `A[i*K + kk]`
  - `B[kk,j]` 在内存里是 `B[kk*N + j]`
  - `C[i,j]` 在内存里是 `C[i*N + j]`

面试官常见追问：

- “如果 B 是 col-major 呢？”（你要能写出索引）
- “如果是 `C = alpha*A@B + beta*C` 呢？”（你要能解释为什么这是 GEMM 的真实形态）

Hands-on：

- 读/改 `gemm_0_naive.cu`：把索引写到你完全不需要思考为止

## Level 1：GPU 上怎么映射线程？

最经典的第一版：

- 1 thread 负责 1 个 `C[i,j]`
- `threadIdx.x` 对应列 `j`
- `threadIdx.y` 对应行 `i`

你要能讲清楚：

- block 和 grid 的含义
- 为什么需要边界判断（`i >= M` 或 `j >= N`）
- 为什么这种写法很慢（下一关就要追问你）

Hands-on：

- `gemm_0_naive.cu`

## Level 2：为什么 naive 慢？（内存层次与复用）

你必须从第一性原理解释：

- 计算量：每个 `C[i,j]` 做 `K` 次乘加（FMA）
- 内存量：naive 会对 `A` 和 `B` 做大量重复读取
- 在 GPU 上，瓶颈经常是“内存带宽”，不是算力

面试官常见追问：

- “global memory coalescing 是什么？”
- “同一个 warp 的线程访问全局内存，什么模式最理想？”
- “A 和 B 哪个更容易被复用？复用的方向是什么？”

核心回答：

- **A 的一行**会被 C 的很多列复用；**B 的一列**会被 C 的很多行复用  
  所以你要把 A 和 B 的子块搬到更快的存储（shared memory）里，多次复用

Hands-on：

- 继续看 `gemm_0_naive.cu` 的内层 `kk` 循环，想清楚“同一个 A/B 元素会被多少次使用”

## Level 3：共享内存分块（Block tile + K tile）

这就是面试里最经典的“第一层优化”：

- 每个 block 负责计算 C 的一个小块（例如 `16×16`）
- K 维度分块：每次只加载 `K_tile` 这段到 shared memory
- 计算时反复使用 shared memory 中的 A_tile、B_tile

你必须能讲清楚：

- 为什么要 `__syncthreads()`（保证 tile 被整个 block 写完再读）
- 为什么 K 要循环分块（shared memory 放不下整个 K）

Hands-on：

- `gemm_1_tiled_smem.cu`

## Level 4：线程寄存器分块（Thread tile / Register blocking）

面试官开始“加压力”的常见套路：

> “你这个每个 thread 只算一个输出，太稀疏了。能不能让每个 thread 多算一些？”

核心点：

- 让每个 thread 计算一个 `TM×TN` 小块（存在寄存器里）
- 这样一来：
  - 同样的 shared memory 读可以贡献更多计算
  - 每个 thread 的算术强度更高

Hands-on：

- `gemm_2_thread_tile_8x8_float4.cu`：每 thread 计算 8×8

## Level 5：向量化 LD/ST（Pack 128-bit）

继续追问：

> “你能不能减少 load/store 指令数量？”

核心点：

- 使用 `float4`（FP32）或 `int4`（8 个 FP16）进行 128-bit 读写
- 前提条件：
  - 地址对齐（通常要求维度是 4 或 8 的倍数）
  - 访问模式连续

Hands-on：

- `gemm_2_thread_tile_8x8_float4.cu`（float4）
- `gemm_10_mma_m16n8k16_smem_swizzle.cu`（int4/8 half）

常见坑：

- 没有对齐就强行 `reinterpret_cast<float4*>`（可能变慢甚至出错）
- 边界块（tail block）怎么处理（生产代码要有“边界版”）

## Level 6：shared memory bank conflicts（你会被追问到崩）

如果你写了 shared memory，面试官很可能会问：

> “你知道 bank conflict 吗？你的代码会不会冲突？”

第一性原理解释（你要能用自己的话讲）：

- shared memory 被分成 32 个 bank
- 一个 warp 同时访问 shared memory 时：
  - 如果每个线程落在不同 bank：并行、快
  - 如果多个线程落在同一个 bank 的不同地址：冲突，硬件会拆成多次事务
  - 如果所有线程读同一个地址：broadcast，通常不算冲突

常见缓解手段（LeetCUDA 里强调的）：

- **padding**：让一行的长度不再是“坏的倍数”，打散 bank 映射
- **layout transform**：例如把 A_tile 变成 `[BK][BM]`（在 SMEM 中转置）
- **swizzle/permutation**：对索引做 xor/zigzag，让 ldmatrix/ld.shared 更冲突少

Hands-on：

- `gemm_3_smem_padding_bcf.cu`（padding + A_tile 在 SMEM 中按 `[BK][BM]` 存）

## Level 7：双缓冲（Double buffering）

追问升级：

> “你每次 tile 都 `load -> syncthreads -> compute -> syncthreads`，能不能减少同步？”

核心思路：

- 用两个 shared memory buffer 轮换
- tile t 计算时，预取 tile t+1 的数据（先到寄存器）
- 这样每个 tile 主循环通常只需要 1 次 `__syncthreads()`

Hands-on：

- `gemm_4_double_buffered.cu`

## Level 8：cp.async + 多 stage（SM80+）

这通常是“高级 GPU 岗”最常见的深挖点之一：

> “你知道 cp.async 吗？能不能把 gmem->smem 和 compute overlap？”

必须能讲清楚：

- cp.async 是“异步拷贝队列”：把 global memory 的数据搬到 shared memory
- `commit_group`：提交一组拷贝
- `wait_group`：等待队列里某些组完成（典型是保持 pipeline 深度）
- 为什么需要多 stage（2~4 很常见）：让更多拷贝在飞，隐藏更长的内存延迟

Hands-on：

- `gemm_5_cp_async_multistage.cu`（3 stages）
- `gemm_9_wmma_tf32_stages.cu`（Tensor Core 版本的 cp.async + stage + drain tail）

更深的追问（你至少要知道关键词）：

- “cp.async 的 zero-fill/valid bytes 怎么做？”
- “cp.async.bulk / TMA 是什么？（Hopper/SM90）”
- “异步拷贝与 `__syncthreads()` 的关系是什么？为什么经常一起出现？”

## Level 9：Occupancy 与资源权衡（别只会背 kernel）

你写的 kernel 很可能会被问：

- 这个 block 用了多少 shared memory？
- 每个线程大概多少寄存器？
- occupancy 怎么估？为什么 occupancy 高不一定更快？

你必须能讲清楚“权衡”：

- tile 大：算术强度高，但 shared/register 多，occupancy 可能下降
- tile 小：occupancy 高，但复用差，容易内存瓶颈

Hands-on：

- 对比 `gemm_2`、`gemm_3`、`gemm_4`：改 tile 大小观察寄存器/SMEM 增长趋势（真实 profiling 需要 GPU）

## Level 10：Split-K（原子 vs workspace）

追问：

> “如果 K 特别大，但 M/N 不大，怎么提高并行度？”

两种典型答案：

- 面试友好：Split-K + atomicAdd（简单但会有 contention、非确定性）
- 生产友好：Split-K + workspace + reduce（确定性更强，常更快）

Hands-on：

- `gemm_6_splitk_workspace.cu`

你要能讲 trade-off：

- workspace 需要额外显存
- reduce kernel 有额外开销
- 但避免了大量 atomics

## Level 11：Epilogue fusion（面试官最爱问“工程化”的点）

追问：

> “GEMM 后面通常还接什么？能不能 fuse？”

常见 epilogue：

- bias / add
- activation（ReLU/GELU/SwiGLU 的一部分）
- scaling（例如量化/反量化的 scale）
- `alpha/beta`（BLAS 语义）

Hands-on：

- `gemm_7_epilogue_fusion.cu`（GEMM + bias + ReLU）

## Level 12：Tensor Cores 入门（你不能只会背 TFLOPS）

你必须能讲清楚：

- Tensor Core 是什么：专门做矩阵乘加的硬件单元（warp 级别）
- 支持的数据类型：FP16/BF16/TF32/更低位（视架构而定）
- “TF32 是什么”：用 FP32 输入但内部截断 mantissa（更快但精度下降）

Hands-on：

- `gemm_8_wmma_fp16_tensorcore.cu`（WMMA FP16）

## Level 13：WMMA API（能写出来就很加分）

你要能写并解释：

- fragment 类型：matrix_a / matrix_b / accumulator
- `load_matrix_sync`：从内存装载到 fragment
- `mma_sync`：执行 Tensor Core GEMM
- `store_matrix_sync`：把 accumulator 写回内存

常见坑：

- layout（row_major vs col_major）写错
- leading dimension（stride）传错
- 维度不是 WMMA tile 的整数倍

Hands-on：

- `gemm_8_wmma_fp16_tensorcore.cu`

## Level 14：TF32 Tensor Core SGEMM（更贴近 LLM 推理/训练）

追问：

> “FP32 GEMM 你为什么不用 TF32 Tensor Core？”

你要能讲：

- TF32 的精度 vs 性能
- 对 LLM 来说通常精度够不够（取决于场景）

Hands-on：

- `gemm_9_wmma_tf32_stages.cu`

## Level 15：Tensor Core 的层次化分块（Block / Warp / WMMA）

这是很多候选人卡住的地方：他们能写一个 warp 的 WMMA，但说不清怎么扩到 128×128 这种 block tile。

你要能讲清楚：

- block tile（128×128）由多少 warps 组成
- 每个 warp 负责哪一块输出
- 一个 warp 内为什么还要做多次 WMMA（tile MMAs）

Hands-on：

- `gemm_9_wmma_tf32_stages.cu`（8 warps/block，warp 内 2×4 WMMA tiles）

## Level 16：多 stage pipeline 的“drain tail”（很多人写不对）

追问：

> “你用了 3-stage，那最后剩下的 stage 怎么处理？”

关键点：

- pipeline 主循环通常只计算一部分 tiles
- 最后要“排空”（drain）剩余 stage，否则少算 K 的最后几块

Hands-on：

- `gemm_9_wmma_tf32_stages.cu`（最后有 drain tail）

## Level 17：MMA PTX（这已经非常深）

面试官如果真的要“拷打到骨头”，可能会问：

> “WMMA 只是 API。你会不会 mma.sync？ldmatrix 你知道吗？”

你要能讲清楚：

- `ldmatrix`：把 shared memory 的矩阵块按 Tensor Core 需要的方式装到寄存器
- `mma.sync`：真正的 Tensor Core 指令（warp 级）
- “寄存器里的 fragment 到底每个 lane 拿了哪些元素？”（非常深，但至少要知道有官方 mapping 文档）

Hands-on：

- `gemm_10_mma_m16n8k16_smem_swizzle.cu`

## Level 18：SMEM swizzle / permuted layout（为什么需要）

追问：

> “你用 ldmatrix 读 SMEM，会不会 bank conflict？怎么解决？”

核心点：

- ldmatrix 的访问模式很特殊，容易触发 bank conflicts
- 常见解法：对 SMEM 的存储布局做 swizzle（xor / zigzag / permute）
- CUTLASS/CuTe 里会把这一步做成可组合的 layout 变换

Hands-on：

- `gemm_10_mma_m16n8k16_smem_swizzle.cu`（`SwizzleACol`）

## Level 19：Collective store（shfl 打包 128-bit store）

追问：

> “你每个 lane scatter store 太多了，能不能更像 CUTLASS 那样写 epilogue？”

核心点：

- 每个 lane 手里只有一小部分输出
- 如果让每个 lane 都去 store，会产生很多小 store（带宽不友好）
- 用 `__shfl_sync` 把 4 个 lane 的 half2 聚合成 128-bit，再一次写回

更深的点（知道关键词即可）：

- Hopper/SM90 有 `stmatrix`，可以更高效地从 regs 写回 SMEM

Hands-on：

- `gemm_10_mma_m16n8k16_smem_swizzle.cu`（warp shuffle + `uint4` store）

## Level 20：再往下还能多深？（极限深挖方向）

到这里，已经超过大多数面试“手撕”要求了，但如果你写了“精通”，面试官依然可能沿着这些方向继续问：

- Hopper WGMMA（warp-group MMA）：为什么需要 warp-group？调度/同步怎么做？
- TMA（Tensor Memory Accelerator）：为什么比 cp.async 更强？如何做多维搬运？
- Cluster launch：为什么要 cluster？和共享 L2/SMEM 有什么关系？
- L2 persistence / cache policy：怎么让 B 的某些块更留在 L2？
- Persistent GEMM：让一个 CTA 处理多个 tiles，减少 launch/scheduling 开销
- 更复杂的 epilogue：bias + activation + scaling + quantize/dequantize 全融合
- 数值与确定性：Split-K 的可复现性、累加顺序、Kahan、FP32 accumulate
- 实际系统：与通信（AllReduce）重叠、与多流并行、与框架（cuBLASLt）接口

如果你能把 Level 0~19 的代码和原理讲清楚，再能对 Level 20 的关键词给出合理解释与取舍，基本就属于“写在简历上也不怕被拷打”的程度。

