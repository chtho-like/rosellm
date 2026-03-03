# cuda/flashattn

目标：在当前机器（RTX 4070 / SM89）上，把 **scaled dot-product
attention（SDPA）forward** 做成一个“逐步优化阶梯”，并用统一的 benchmark
脚本和图表，把每一步和业界 FlashAttention（`flash-attn`）做对比。

说明：

- 本目录以 **Triton kernel** 为主（更容易快速试错 + 自动调参 + 生成 Tensor
  Core 代码）。
- benchmark 同时包含 PyTorch SDPA 的明确后端：
  - `torch_sdpa_flash_pytorch`: PyTorch 内置 Flash-Attention 后端
  - `torch_sdpa_efficient_cutlass`: PyTorch mem-efficient attention（CUTLASS）
  - `torch_sdpa_cudnn`: cuDNN SDPA 后端
- 每个优化版本对应一个独立文件：`triton_flashattn_*.py`。
- 统一用 `bench_flashattn.py` 进行 benchmark、落盘结果、画图。

## 你需要什么

- 一块 NVIDIA GPU（本机是 RTX 4070 / SM89）
- `python` + `torch` + `triton`（本机已经有）
- 可选：安装业界 `flash-attn` 作为对比基线

## Quick start

在本目录下直接跑 benchmark：

```bash
cd /data/projects/rosellm/notebooks/cuda/flashattn
python bench_flashattn.py \
  --causal 1 --dtype fp16 \
  --batch 1 --heads 32 \
  --seq-lens 256,512,1024,2048,4096 \
  --head-dims 64,128 --check
```

如果你希望把业界 `flash-attn`（Dao-AILab/flash-attention）也纳入对比，
推荐用下面的方式安装（避免编译无用架构）：

```bash
FLASH_ATTN_CUDA_ARCHS=80 NVCC_THREADS=8 \
  python -m pip install --no-cache-dir flash-attn==2.7.3
```

输出：

- `./out/latest/results.json`
- `./out/latest/results.csv`
- `./out/latest/plots/*.png`

如果你希望把 Triton 的 shape→tile 配置“尽量交给 autotune”，可以先跑一次
离线调参，生成 v9 的配置表：

```bash
python tune_triton_flashattn_table.py \
  --device 0 --causal 1 --dtype fp16 \
  --batch 1 --heads 32 \
  --seq-lens 256,512,1024,2048,4096 \
  --head-dims 64,128
```

它会生成 `triton_flashattn_v9_table_sm89.json`（默认按 SM 号命名）。你也可以
用环境变量覆盖路径：

```bash
FLASHATTN_TRITON_AUTOTUNE_TABLE=/path/to/table.json \
  python bench_flashattn.py ...
```

## 文件梯度（从易到难）

- `triton_flashattn_0_baseline.py`: 最小可用 FlashAttention（online softmax）
- `triton_flashattn_1_codegen_hints.py`: 增加 `tl.multiple_of` / 对齐等 codegen
  hint
- `triton_flashattn_2_autotune.py`: 增加 autotune（探索 block/warp/stage）
- `triton_flashattn_3_fixed_best.py`: 固化在本机上测出来的最优配置
- `triton_flashattn_4_causal_cutoff.py`: causal 场景下减少无效 K-block 计算
- `triton_flashattn_5_even_fastpath.py`: 针对整除尺寸的无 mask fastpath
- `triton_flashattn_6_autotune_cutoff.py`: 在 cutoff+fastpath 上继续 autotune
- `triton_flashattn_7_fixed_best_cutoff.py`: 固化本机最强配置（当前最佳）
- `triton_flashattn_8_dispatch_fixed_best.py`: 按 head_dim/seq_len dispatch 的
  固化最优配置（更适合作为“直接用”的版本）
- `triton_flashattn_9_autotune_table.py`: 使用离线 autotune 生成的
  shape→config 表进行 dispatch（更接近“自动挑 tile”，且避免运行时 autotune）

## 报告

详细的环境、命令、结果表格、图、以及逐步分析写在 `REPORT.md`。
