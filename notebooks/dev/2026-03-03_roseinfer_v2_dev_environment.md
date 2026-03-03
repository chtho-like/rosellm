# RoseInfer v2：开发环境与工具选择（Python / CUDA / 可选 kernel 库）

> 目标：在 H100（CUDA 12.8）上获得“开发速度 + 可复现 + 高性能依赖可插拔”的综合最优。

---

## 1) Python 环境工具选择：`uv`（推荐）

选择 `uv` 的理由：
- 安装快、解析快（开发效率直接提升）
- 兼容 `pip`/`virtualenv` 生态（工程摩擦小）
- 适合同时管理纯 Python 依赖与“可选 CUDA 扩展库”的安装流程（扩展库仍用其官方安装方式）

原则：
- **Python 依赖用 `uv` 管**（可复现、快速）
- **CUDA 扩展库（FlashMLA/DeepEP/DeepGEMM/flash-attn/flashinfer）按“可选插件”安装**，不强行写死在 `pyproject.toml` 的必选依赖里

---

## 2) 推荐的环境组织方式

### 2.1 venv（单仓库）

在仓库根目录：
- `.venv/`：开发虚拟环境
- `DG_JIT_CACHE_DIR` 等 JIT cache：放到可控位置，避免污染 HOME

示例（概念性）：

```bash
cd /workspace/rosellm
uv venv .venv
source .venv/bin/activate

# 安装 rosellm + 基础 dev 工具（来自 pyproject 的可选依赖）
uv pip install -e '.[dev,test]'
```

### 2.2 额外推理/benchmark 依赖（建议以“可选组”或 requirements 文件维护）

RoseInfer v2 推理与 benchmark 常用依赖（建议安装）：
- `safetensors`, `accelerate`（加载/设备映射的基础工具）
- `tokenizers` / `transformers`（仅 tokenizer 与通用工具；**不使用 remote model code**）
- `openai`, `httpx`, `orjson`（OpenAI API 与高性能 JSON）
- `numpy`, `matplotlib`（benchmark 作图）

> 这些依赖建议以 `uv pip install ...` 的方式集中安装，并在 notebooks/dev 记录实际版本（写入 BackendReport/RunReport）。

---

## 3) 可选 CUDA kernel 库（性能关键，按需安装）

### 3.1 FlashAttention（flash-attn）
- H100 上是“可靠默认后端”的基础件
- 注意与 torch/CUDA 版本匹配

### 3.2 FlashMLA（DeepSeek MLA 专用）
- 高性能 decode/prefill（尤其 decode）
- 依赖 CUDA/arch；通常需要编译
- 在我们框架中作为 `attention_mla` 的可选 backend：可用则选，不可用自动降级

### 3.3 DeepGEMM（FP8 block-wise GEMM）
- H100 上的 FP8 dense/grouped GEMM 关键后端
- 建议配置 JIT cache：
  - `DG_JIT_CACHE_DIR=/workspace/.cache/deep_gemm`（示例）

### 3.4 DeepEP（MoE all-to-all）
- decode/prefill 分别偏好 `low_latency` / `high_throughput`
- 单机 2×H100 场景建议优先 `low_latency`（若可用）

### 3.5 FlashInfer（补充）
- MLA attention / all2all / MoE kernels 的候选后端
- 作为增强与 fallback（不作为唯一依赖）

---

## 4) 版本记录与可复现

建议每次 benchmark 自动记录（落盘到 `run_report.json`）：
- `torch.__version__`, `torch.version.cuda`, `nccl` 版本
- GPU 型号、SM 能力
- `flash-attn`, `flashinfer`, `deep_gemm`, `deepep`, `flashmla` 的版本或 commit（若可用）
- 选择到的后端组合（BackendReport）

这比“写死版本”更重要：因为 kernel 库往往是源代码编译/本地 wheel，版本信息必须可追踪。

