# FlashAttention on RTX 4070 (SM89): step-by-step kernel optimization

本报告的目标是：在 **本机 RTX 4070（SM89）** 上，针对 LLM 推理最常见的
**causal scaled dot-product attention（SDPA）forward**，实现一个逐步优化阶梯，
并用统一 benchmark 对比：

- 业界 FlashAttention（`flash-attn` 包，Dao-AILab 仓库的实现）
- PyTorch `scaled_dot_product_attention`（`torch_sdpa_auto`，自动选择后端）
- PyTorch 明确指定后端（用于对齐/理解不同实现路线）：
  - `torch_sdpa_flash_pytorch`：PyTorch 内置 FlashAttention 后端
  - `torch_sdpa_efficient_cutlass`：mem-efficient attention（CUTLASS）
  - `torch_sdpa_cudnn`：cuDNN SDPA 后端
- 朴素基线：`matmul + softmax + matmul`（会 materialize `S×S` attention 矩阵）
- 本目录的 Triton 版本 `triton_v0` ~ `triton_v9`

注意：本目录的“优化次数不设上限”意味着：你可以继续在 `triton_flashattn_*`
上追加版本；本报告先把 **从 correctness 到超越 torch_sdpa_auto** 的关键路径跑通。

## 0. 环境信息（可复现）

硬件：

- GPU: NVIDIA GeForce RTX 4070
- Compute Capability: 8.9 (SM89)
- 显存: ~11.6 GiB

软件：

- Driver: 580.82.07
- CUDA Toolkit (`nvcc`): 12.8
- PyTorch: 2.9.0+cu128
- Triton: 3.5.0

## 1. 问题定义（从零开始）

我们做的是 **SDPA forward**：

给定 Q/K/V：

- 形状：`(B, H, S, D)`
  - `B`: batch size
  - `H`: head 数
  - `S`: sequence length（序列长度）
  - `D`: head_dim（每个 head 的向量维度）
- causal（自回归 mask）：第 `i` 个 query 只能看 `j <= i` 的 key/value

数学形式（单个 head）：

1. `scores = (Q @ K^T) * scale`
2. `P = softmax(scores)`（causal 时上三角为 `-inf`）
3. `O = P @ V`

难点在于：

- `scores` / `P` 是 `S×S`，如果直接存下来，会非常慢且非常吃显存带宽
- FlashAttention 的核心思想是：**分块 + online softmax**，避免 materialize
  `S×S`

## 2. Benchmark 方法（尽量公平）

统一用 `bench_flashattn.py`：

```bash
cd /data/projects/rosellm/notebooks/cuda/flashattn
python bench_flashattn.py \
  --causal 1 --dtype fp16 \
  --batch 1 --heads 32 \
  --seq-lens 256,512,1024,2048,4096 \
  --head-dims 64,128 \
  --warmup 25 --iters 100 --check
```

计时方式：

- 先 warmup（让 kernel JIT / cache 稳定）
- 用 CUDA events 只计 kernel 执行时间（不含 Python 端开销）

输出文件：

- `out/latest/results.csv`：原始表格（含每个 backend 的 `avg_ms` / `tflops`）
- `out/latest/results.csv` 也包含：
  - `config`：Triton autotune / dispatch 选到的配置（方便复盘和固化）
  - `error`：某些 shape/config 触发 shared memory 等资源超限时的失败原因
- `out/latest/plots/*`：自动生成的曲线图（time / TFLOPs）

## 3. 优化阶梯（每一步一个文件）

所有版本都实现同一件事：`flash_attn(q, k, v, causal=True)`，并保持数值稳定
（fp32 的 max/sum 累加）。

从 v0 到 v9 的关键变化：

- `triton_flashattn_0_baseline.py` (`triton_v0_baseline`)
  - 最小可用：online softmax + 分块
  - 但在 causal 情况下：仍然会遍历所有 K-block，然后用 mask 把未来 token
    置为 `-inf`（浪费一半算力）
- `triton_flashattn_1_codegen_hints.py` (`triton_v1_codegen_hints`)
  - 尝试加一些 codegen hint（例如让 loop index 对齐）
- `triton_flashattn_2_autotune.py` (`triton_v2_autotune`)
  - 对 v0 做 autotune（探索 `BLOCK_M/BLOCK_N/warps/stages`）
- `triton_flashattn_3_fixed_best.py` (`triton_v3_fixed_best`)
  - 固化一个手选 config（后续会被更强版本取代）
- `triton_flashattn_4_causal_cutoff.py` (`triton_v4_causal_cutoff`)
  - **关键优化**：causal 时只遍历 `end_n = min(S, start_m + BLOCK_M)`
  - 直接减少大量无效 K-block dot（让“做了多少有效工作”更接近理论）
- `triton_flashattn_5_even_fastpath.py` (`triton_v5_even_fastpath`)
  - 对常见整除尺寸（例如 `S%BLOCK==0`、`D%16==0`）走无 mask fastpath
- `triton_flashattn_6_autotune_cutoff.py` (`triton_v6_autotune_cutoff`)
  - 在 v4+v5 的基础上 autotune，找本机最优 config
- `triton_flashattn_7_fixed_best_cutoff.py` (`triton_v7_fixed_best_cutoff`)
  - 固化 v6 在本机上测出来的最优 config（对部分 shape 很强，但不一定最稳）
- `triton_flashattn_8_dispatch_fixed_best.py` (`triton_v8_dispatch_fixed_best`)
  - **按 head_dim/seq_len 做简单 dispatch**，把 v6 的最优 config 固化成一套
    “可直接用”的版本（避免 v7 只对 D=64 最优、D=128 很慢的问题）
- `triton_flashattn_9_autotune_table.py` (`triton_v9_autotune_table`)
  - 用离线 autotune 生成的 shape→config 表做 dispatch（更接近“自动挑 tile”）
  - 配置表默认是 `triton_flashattn_v9_table_sm89.json`，可用
    `tune_triton_flashattn_table.py` 生成/更新

## 4. 关键结果（先看最重要的形状）

最常见推理形状之一：

- `B=1, H=32, S=2048, D=64`
- dtype: fp16
- causal: True

用命令：

```bash
python bench_flashattn.py \
  --causal 1 --dtype fp16 \
  --batch 1 --heads 32 \
  --seq-lens 2048 --head-dims 64 \
  --warmup 50 --iters 200
```

当前测到（会随驱动/温度/时钟略有波动），并且已经纳入 FlashAttention
对比（`flash-attn==2.7.3`）：

- `flash_attn`: ~0.512 ms
- `torch_sdpa_auto`: ~0.510 ms
- `triton_v9_autotune_table`: ~0.481 ms（更快）

对应的 TFLOPs（以 causal 理论 FLOPs 计）：

- `flash_attn`: ~33.6 TFLOPs
- `torch_sdpa_auto`: ~33.7 TFLOPs
- `triton_v9_autotune_table`: ~35.7 TFLOPs

这意味着：在本机这一张卡上，我们的 v9 **已经超过** `torch_sdpa_auto`。
并且在这个形状上，也 **超过** 业界 `flash-attn`。

完整 sweep 曲线请看：

- `out/latest/plots/time_focus_c1_d64.png`
- `out/latest/plots/tflops_focus_c1_d64.png`

### 4.1 sweep（S=256/512/1024/2048/4096, D=64, causal=True）

命令：

```bash
python bench_flashattn.py \
  --causal 1 --dtype fp16 \
  --batch 1 --heads 32 \
  --seq-lens 256,512,1024,2048,4096 \
  --head-dims 64,128 \
  --warmup 25 --iters 100 --check
```

关注的 3 个基线 + 我们的最终版（best triton，通常是 v9）：

| S | flash_attn (ms) | torch_sdpa_auto (ms) | best triton (ms) | best vs flash_attn |
|---:|---:|---:|---:|---:|
| 256  | 0.0240 | 0.0238 | 0.0185 | +30.2% |
| 512  | 0.0561 | 0.0552 | 0.0449 | +24.9% |
| 1024 | 0.1566 | 0.1549 | 0.1395 | +12.2% |
| 2048 | 0.5121 | 0.5105 | 0.4810 |  +6.5% |
| 4096 | 1.8494 | 1.8431 | 1.7825 |  +3.8% |

说明：

- “+X%” 表示 best triton 更快：`flash_attn_time / best_triton_time - 1`。
- 对 `S=256` 做了 correctness check：`max_abs_err ~ 4.88e-4`（fp16 量级正常）。

完整 raw 表（包含每个优化版本、以及每个 shape 的失败原因）在：

- `out/latest/results.csv`
- `out/latest/results.json`

### 4.2 sweep（S=256/512/1024/2048/4096, D=128, causal=True）

同样来自 `out/latest/results.csv`（best triton 通常是 v9/v6/v8）：

| S | flash_attn (ms) | best triton (ms) | best vs flash_attn |
|---:|---:|---:|---:|
| 256  | 0.0290 | 0.0308 |  -5.8% |
| 512  | 0.0780 | 0.0888 | -12.2% |
| 1024 | 0.2522 | 0.2897 | -13.0% |
| 2048 | 0.9080 | 0.9811 |  -7.4% |
| 4096 | 3.3950 | 3.5566 |  -4.5% |

说明（直觉版）：

- `D=128` 下，FlashAttention 的 CUDA kernel 仍然更强（目前我们落后 4%~13%）。
- 随着 `S` 增大，我们的差距在缩小，说明我们主要是在 **小规模/高寄存器压力**
  的 regime 里还没卷赢。
- v9 用离线 autotune 的 table 做 dispatch，在 `S=256/512` 这类“小 S + D=128”
  的形状上会更偏向更小的 tile（例如 `BLOCK_M=32`），所以差距明显缩小。
- 另外，`triton_v1/v3/v4/v5` 在 `D=128` 的某些 config 上会触发
  **shared memory 超限**；现在 `bench_flashattn.py` 会把这些失败记录在
  `error` 列并继续跑完整 sweep，避免整个 benchmark 直接崩掉。

### 4.3 每一步优化到底带来了什么（用 S=2048 这行看最清楚）

同一轮 sweep 里，`S=2048, D=64, causal=True` 的结果如下（单位 ms）：

| backend | avg_ms | 相对 flash_attn |
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

如何读这张表（从第一性原理出发）：

- v0/v1/v2/v3：**看起来像 FlashAttention，但在 causal 时做了大量“无效乘法”**
  - 因为它们仍然会遍历所有 K-block，然后用 `-inf` mask 把未来 token 抹掉
  - 结果是：你在算力上付出了接近 non-causal 的代价，但有效工作量只有 causal 的一半
- v4：第一次把 causal 的“大头浪费”砍掉（通过 `end_n` cutoff）
  - 这一步通常是 causal 推理里最关键的 “为什么你慢” 的原因之一
- v5：把“边界 mask”在整除形状上消掉，属于小但稳定的提升
- v6/v7/v8/v9：在 v4+v5 的基础上调参（tile/warp/stage），把性能推到超过
  `flash_attn` / `torch_sdpa_auto`

这也解释了为什么本目录的优化是“阶梯式”的：

- 很多优化不是“微调”，而是直接把一大块无效计算删掉（v4）
- 真正进入“拼刺刀”的阶段之后，才是 `BLOCK_M/BLOCK_N/warps/stages` 的细调（v6+）

## 5. 对比 cuBLAS 基线（为什么 fused 会更快）

朴素实现（`torch_matmul_softmax`）会做：

1) `Q @ K^T`（cuBLAS GEMM，输出 `S×S`）
2) `softmax(scores)`（再读写一遍 `S×S`）
3) `P @ V`（cuBLAS GEMM）

这会把 `S×S` 矩阵写回显存，带来巨大的带宽和缓存压力。

FlashAttention 类算法（本目录 triton kernel）用在线 softmax，把 softmax 和
`P @ V` 融合到同一个 streaming 过程中，避免 materialize `S×S`，所以端到端更快。

## 6. Nsight Compute（ncu）剖析（v9）

为了拿到硬件计数器，需要用 `sudo` 跑 ncu：

```bash
sudo ncu --set full --target-processes all \
  --kernel-name-base demangled -k regex:_flashattn_fwd_kernel \
  --launch-skip 10 --launch-count 1 \
  -o out/latest/ncu_v9 \
  /data/projects/anaconda3/bin/python profile_flashattn.py \
    --backend v9 --init empty --warmup 10
```

生成报告：

- `out/latest/ncu_v9.ncu-rep`

用命令打印关键 section（示例）：

```bash
ncu --import out/latest/ncu_v9.ncu-rep --page details --print-summary per-kernel
```

本机一次采样中（节选）：

- Duration: ~480 us（对应 `S=2048, D=64` 的一次 forward）
- Compute (SM) Throughput: ~40.8%
- Memory Throughput: ~30.8%
- Registers / thread: 120
- Dynamic shared memory / block: ~16.4 KiB

解读（直觉版）：

- 这类 kernel 通常是“算 + 访存”混合瓶颈，不可能 100% 吃满某一项
- `regs/thread` 偏高，说明我们用寄存器换了访存/重算（合理，但会影响 occupancy）

### 6.1 与 FlashAttention kernel 的 ncu 对照（同一形状）

同样用 `profile_flashattn.py`，把 `flash_attn` 的 forward kernel 也采一遍：

```bash
sudo ncu -f --set full --target-processes all \
  --kernel-name-base demangled -k regex:flash_fwd \
  --launch-count 1 \
  -o out/latest/ncu_flash_attn \
  /data/projects/anaconda3/bin/python profile_flashattn.py \
    --backend flash_attn --init empty --warmup 0
```

同一形状下的关键指标（ncu 输出）：

| kernel | Duration (us) | Compute Throughput | Memory Throughput | Regs/thread | Dyn smem/block |
|---|---:|---:|---:|---:|---:|
| `flash_fwd_kernel` (flash-attn) | 513.76 | 39.29% | 25.15% | 255 | 49.15 KiB |
| `_flashattn_fwd_kernel` (v9) | 479.71 | 40.84% | 30.82% | 120 | 16.38 KiB |

直觉解读：

- v9 时间更短（~480us vs ~514us）
- v9 的寄存器和动态共享内存占用更低，给了调度器更多空间
- v9 的 Memory/Compute 吞吐百分比都更高，说明整体更接近硬件上限

## 7. 下一步还能怎么继续卷（建议路线）

如果继续追求极限，可以按这个顺序：

1) 继续优化 `D=128` 的 kernel（更低寄存器压力/更好的 tile；目前仍落后 flash-attn）
2) 更细的 K/V 预取 pipeline（更 aggressive 的 `num_stages` + 更合理的 tile）
3) 更激进的 mask 消除（causal 下对角块/尾块的特化）
4) 更系统的 ncu 指标对比（对齐 `flash-attn` / `torch_sdpa_auto`）

## 8. FlashAttention（`flash-attn`）对比状态

`flash-attn` 的源码编译很重。为了避免编译 `sm90/sm100/sm120` 的无用目标，
建议用：

```bash
FLASH_ATTN_CUDA_ARCHS=80 NVCC_THREADS=8 \
  python -m pip install --no-cache-dir flash-attn==2.7.3
```

安装完成后，重新跑 benchmark，就会在表格/图里出现 `flash_attn` 这一行。

已完成：在 `B=1, H=32, D=64, causal=True` 且 `S=256/512/1024/2048/4096` 的
sweep 上，best triton（目前通常是 `triton_v9_autotune_table`）均快于
`flash_attn`（见 4.1）。

仍需继续：`D=128` 仍落后 `flash_attn`（约 4%~13%，见 4.2）。
