# CUDA GEMM Interview "Interrogation" Roadmap (From Zero to Advanced Internals)

You can think of a CUDA GEMM interview as a tree of progressively deeper follow-up questions:

1) First, the interviewer asks whether you understand the mathematics and memory layouts of GEMM
2) Next, they ask whether you can map GEMM onto the GPU's thread and memory hierarchy
3) They continue down through shared memory, registers, pipelining, cp.async, and multistage execution
4) Finally, they reach Tensor Cores: WMMA → MMA PTX → ldmatrix → SMEM swizzle → collective store

The code in this directory follows that sequence of questions: `cuda/gemm2/gemm_*.cu`.

Suggested reading paths:

- Absolute beginners: read Levels 0 → 20 in order, focusing on only the core concept and one or two pitfalls at each level
- Interview preparation: treat each level's "What you must be able to explain/implement" items as a checklist
- Whiteboard coding practice: implement the `.cu` file listed under Hands-on for each level

## Level 0: Do you actually understand GEMM?

You must be able to explain the following from first principles:

- What GEMM is: `C = A @ B`, where `A` is `M×K`, `B` is `K×N`, and `C` is `M×N`
- One output element: `C[i,j] = sum_{kk=0..K-1} A[i,kk] * B[kk,j]`
- Row-major indexing formulas:
  - `A[i,kk]` is stored at `A[i*K + kk]`
  - `B[kk,j]` is stored at `B[kk*N + j]`
  - `C[i,j]` is stored at `C[i*N + j]`

Common interviewer follow-ups:

- "What if B is column-major?" (You must be able to write the indexing expression.)
- "What about `C = alpha*A@B + beta*C`?" (You must explain why this is GEMM's practical form.)

Hands-on:

- Read and modify `gemm_0_naive.cu` until you can write the indexing without having to think about it

## Level 1: How do you map threads on a GPU?

The canonical first implementation is:

- 1 thread computes 1 element `C[i,j]`
- `threadIdx.x` maps to column `j`
- `threadIdx.y` maps to row `i`

You must be able to explain:

- What blocks and grids represent
- Why boundary checks are required (`i >= M` or `j >= N`)
- Why this implementation is slow (which leads directly to the next question)

Hands-on:

- `gemm_0_naive.cu`

## Level 2: Why is the naive implementation slow? (Memory hierarchy and reuse)

You must explain from first principles:

- Computation: each `C[i,j]` performs `K` multiply-add operations (FMAs)
- Memory traffic: the naive implementation repeatedly reloads the same `A` and `B` values
- On a GPU, memory bandwidth is often the bottleneck rather than arithmetic throughput

Common interviewer follow-ups:

- "What is global-memory coalescing?"
- "What is the ideal global-memory access pattern for threads in the same warp?"
- "Which operand, A or B, is easier to reuse, and along which dimension is it reused?"

Core answer:

- **One row of A** is reused across many columns of C, and **one column of B** is reused across many rows of C
  Therefore, you should move subtiles of A and B into faster storage (shared memory) and reuse them repeatedly

Hands-on:

- Continue examining the inner `kk` loop in `gemm_0_naive.cu`, and determine how many times each A/B element is reused

## Level 3: Shared-memory tiling (Block tile + K tile)

This is the canonical first optimization in an interview:

- Each block computes a small tile of C (for example, `16×16`)
- Partition the K dimension: load only the current `K_tile` segment into shared memory
- Repeatedly reuse the A and B tiles in shared memory during computation

You must be able to explain:

- Why `__syncthreads()` is necessary (the entire block must finish writing the tile before any thread reads it)
- Why K must be processed in tiles (shared memory cannot hold the entire K dimension)

Hands-on:

- `gemm_1_tiled_smem.cu`

## Level 4: Per-thread register tiling (Thread tile / Register blocking)

A common way for the interviewer to increase the pressure is:

> "Computing only one output per thread is too sparse. Can each thread compute several outputs?"

Core idea:

- Have each thread compute a `TM×TN` subtile held in registers
- This provides two benefits:
  - The same shared-memory loads contribute to more arithmetic
  - Each thread has higher arithmetic intensity

Hands-on:

- `gemm_2_thread_tile_8x8_float4.cu`: each thread computes an 8×8 tile

## Level 5: Vectorized LD/ST (Pack 128-bit)

The next follow-up is:

> "Can you reduce the number of load/store instructions?"

Core idea:

- Use `float4` for FP32 or `int4` for eight FP16 values to perform 128-bit loads and stores
- Preconditions:
  - Proper address alignment (dimensions generally must be multiples of 4 or 8)
  - Contiguous access patterns

Hands-on:

- `gemm_2_thread_tile_8x8_float4.cu` (float4)
- `gemm_10_mma_m16n8k16_smem_swizzle.cu` (int4/8 half values)

Common pitfalls:

- Forcing `reinterpret_cast<float4*>` on an unaligned address (this may reduce performance or even produce an error)
- Handling boundary tiles (tail blocks require a boundary-safe path in production code)

## Level 6: Shared-memory bank conflicts (Expect aggressive follow-ups)

If you use shared memory, the interviewer will very likely ask:

> "Do you know what a bank conflict is? Does your code have any?"

First-principles explanation that you must be able to give in your own words:

- Shared memory is divided into 32 banks
- When a warp accesses shared memory concurrently:
  - If every thread addresses a different bank, accesses proceed in parallel and are fast
  - If multiple threads address different locations in the same bank, a conflict occurs and the hardware splits the request into multiple transactions
  - If all threads read the same address, the access is broadcast and normally does not count as a conflict

Common mitigation techniques emphasized in LeetCUDA:

- **Padding**: change each row's length so it is no longer an unfavorable multiple, dispersing the bank mapping
- **Layout transform**: for example, store A_tile as `[BK][BM]` (transposed in SMEM)
- **Swizzle/permutation**: apply XOR or zigzag index transformations to reduce conflicts in ldmatrix/ld.shared access patterns

Hands-on:

- `gemm_3_smem_padding_bcf.cu` (padding + A_tile stored as `[BK][BM]` in SMEM)

## Level 7: Double buffering

The follow-up becomes:

> "Every tile performs `load -> syncthreads -> compute -> syncthreads`. Can you reduce synchronization?"

Core approach:

- Alternate between two shared-memory buffers
- While computing tile t, prefetch tile t+1 into registers
- This usually reduces the main tile loop to one `__syncthreads()` per tile

Hands-on:

- `gemm_4_double_buffered.cu`

## Level 8: cp.async + multistage pipeline (SM80+)

This is one of the most common deep-dive topics for advanced GPU roles:

> "Do you understand cp.async? Can you overlap gmem->smem transfers with computation?"

You must be able to explain:

- cp.async provides an asynchronous copy queue that moves data from global memory into shared memory
- `commit_group` submits a group of copies
- `wait_group` waits until a specified number of queued groups remain outstanding, which is how pipeline depth is typically maintained
- Why multiple stages are necessary (2–4 are common): more in-flight copies can hide longer memory latency

Hands-on:

- `gemm_5_cp_async_multistage.cu` (3 stages)
- `gemm_9_wmma_tf32_stages.cu` (Tensor Core version with cp.async, staging, and a drain tail)

Deeper follow-ups (you should at least recognize the terminology):

- "How do cp.async zero-fill and valid bytes work?"
- "What are cp.async.bulk and TMA?" (Hopper/SM90)
- "What is the relationship between asynchronous copies and `__syncthreads()`? Why are they often used together?"

## Level 9: Occupancy and resource trade-offs (Do not merely memorize a kernel)

Likely questions about your kernel include:

- How much shared memory does each block use?
- Approximately how many registers does each thread use?
- How do you estimate occupancy, and why is higher occupancy not always faster?

You must be able to explain the trade-offs:

- Larger tiles: higher arithmetic intensity, but more shared memory and registers, potentially reducing occupancy
- Smaller tiles: higher occupancy, but less reuse and a greater risk of becoming memory-bound

Hands-on:

- Compare `gemm_2`, `gemm_3`, and `gemm_4`: change tile sizes and observe the trend in register/SMEM use (real profiling requires a GPU)

## Level 10: Split-K (Atomics vs workspace)

Follow-up:

> "If K is very large but M and N are small, how do you increase parallelism?"

Two standard answers:

- Interview-friendly: Split-K + atomicAdd (simple, but introduces contention and nondeterminism)
- Production-friendly: Split-K + workspace + reduction (more deterministic and often faster)

Hands-on:

- `gemm_6_splitk_workspace.cu`

You must be able to explain the trade-off:

- A workspace requires additional device memory
- A reduction kernel introduces additional overhead
- It avoids a large number of atomic operations

## Level 11: Epilogue fusion (A favorite engineering-focused interview topic)

Follow-up:

> "What operations commonly follow GEMM? Can you fuse them?"

Common epilogue operations:

- bias / add
- activation (ReLU/GELU/part of SwiGLU)
- scaling (for example, quantization/dequantization scales)
- `alpha/beta` (BLAS semantics)

Hands-on:

- `gemm_7_epilogue_fusion.cu` (GEMM + bias + ReLU)

## Level 12: Tensor Core fundamentals (You must understand more than the TFLOPS number)

You must be able to explain:

- What a Tensor Core is: a hardware unit specialized for warp-level matrix multiply-accumulate operations
- Supported data types: FP16/BF16/TF32/lower-precision formats, depending on the architecture
- What TF32 is: FP32 inputs with a truncated internal mantissa, providing higher performance at reduced precision

Hands-on:

- `gemm_8_wmma_fp16_tensorcore.cu` (WMMA FP16)

## Level 13: WMMA API (Being able to implement this is a strong signal)

You must be able to write and explain:

- Fragment types: matrix_a / matrix_b / accumulator
- `load_matrix_sync`: load from memory into a fragment
- `mma_sync`: execute the Tensor Core GEMM operation
- `store_matrix_sync`: write the accumulator back to memory

Common pitfalls:

- Using the wrong layout (row_major vs col_major)
- Passing the wrong leading dimension (stride)
- Dimensions that are not integer multiples of the WMMA tile

Hands-on:

- `gemm_8_wmma_fp16_tensorcore.cu`

## Level 14: TF32 Tensor Core SGEMM (More relevant to LLM inference/training)

Follow-up:

> "Why are you not using TF32 Tensor Cores for FP32 GEMM?"

You must be able to explain:

- The precision/performance trade-off of TF32
- Whether its precision is sufficient for LLMs (the answer depends on the use case)

Hands-on:

- `gemm_9_wmma_tf32_stages.cu`

## Level 15: Hierarchical Tensor Core tiling (Block / Warp / WMMA)

This is where many candidates struggle: they can implement WMMA for one warp but cannot explain how it scales to a block tile such as 128×128.

You must be able to explain:

- How many warps compose a 128×128 block tile
- Which output region each warp computes
- Why one warp still executes multiple WMMA operations (tile MMAs)

Hands-on:

- `gemm_9_wmma_tf32_stages.cu` (8 warps/block, with 2×4 WMMA tiles per warp)

## Level 16: Draining the tail of a multistage pipeline (A frequent implementation error)

Follow-up:

> "You use a 3-stage pipeline. How do you handle the remaining stages at the end?"

Key points:

- The pipeline's main loop usually computes only a subset of the tiles
- At the end, you must drain the remaining stages, or the last few K tiles will not be computed

Hands-on:

- `gemm_9_wmma_tf32_stages.cu` (contains a drain tail at the end)

## Level 17: MMA PTX (This is already very advanced)

If the interviewer truly wants to probe the lowest levels, they may ask:

> "WMMA is only an API. Can you use mma.sync? Do you understand ldmatrix?"

You must be able to explain:

- `ldmatrix`: loads a shared-memory matrix tile into registers in the layout required by Tensor Cores
- `mma.sync`: the actual warp-level Tensor Core instruction
- "Exactly which fragment elements does each lane hold in registers?" (This is highly advanced, but you should at least know that the official mapping documentation exists.)

Hands-on:

- `gemm_10_mma_m16n8k16_smem_swizzle.cu`

## Level 18: SMEM swizzle / permuted layout (Why it is necessary)

Follow-up:

> "Will reading SMEM with ldmatrix cause bank conflicts? How do you resolve them?"

Core points:

- ldmatrix has a specialized access pattern that can readily trigger bank conflicts
- A common solution is to swizzle the SMEM storage layout with XOR, zigzag, or another permutation
- CUTLASS/CuTe expresses this operation as a composable layout transformation

Hands-on:

- `gemm_10_mma_m16n8k16_smem_swizzle.cu` (`SwizzleACol`)

## Level 19: Collective store (shfl packing for 128-bit stores)

Follow-up:

> "Each lane issues many scattered stores. Can you implement an epilogue more like CUTLASS?"

Core points:

- Each lane holds only a small subset of the output
- If every lane stores independently, the kernel issues many small stores, which is inefficient for memory bandwidth
- Use `__shfl_sync` to aggregate half2 values from four lanes into one 128-bit store

Deeper point (recognizing the term is sufficient):

- Hopper/SM90 provides `stmatrix`, which can write data from registers back to SMEM more efficiently

Hands-on:

- `gemm_10_mma_m16n8k16_smem_swizzle.cu` (warp shuffle + `uint4` store)

## Level 20: How much deeper can this go? (Advanced follow-up directions)

At this point, you are already beyond the whiteboard-coding expectations of most interviews. However, if your resume says "expert," the interviewer may continue in these directions:

- Hopper WGMMA (warp-group MMA): Why are warp groups necessary? How are scheduling and synchronization handled?
- TMA (Tensor Memory Accelerator): Why is it more capable than cp.async? How does it perform multidimensional transfers?
- Cluster launch: Why use clusters? How do they relate to shared L2/SMEM resources?
- L2 persistence / cache policy: How can selected tiles of B remain resident in L2?
- Persistent GEMM: Have one CTA process multiple tiles to reduce launch/scheduling overhead
- More elaborate epilogues: fully fuse bias + activation + scaling + quantization/dequantization
- Numerical behavior and determinism: Split-K reproducibility, accumulation order, Kahan summation, and FP32 accumulation
- Real systems: overlap with communication (AllReduce), concurrency across multiple streams, and framework integration (cuBLASLt)

If you can explain the code and principles in Levels 0–19, and can discuss the terminology and trade-offs in Level 20 coherently, you can safely claim this expertise on your resume and withstand detailed technical questioning.
