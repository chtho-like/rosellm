# DeepSeek 官方 infra 后端接入计划（FlashMLA / DeepEP / DeepGEMM + FlashAttention/FlashInfer）

> 目标：把 **FlashMLA / DeepEP / DeepGEMM** 当作 2×H100（SM90）上 DeepSeek-V2-Lite-FP8 的“默认高性能路径”，同时保留 FlashAttention/FlashInfer 等后端作为补充与 fallback。  
> 这里强调的是**接入形态/接口/选择逻辑**，不是具体实现代码。

---

## 1) FlashMLA（MLA attention 专用）

### 我们需要从 FlashMLA 得到什么
- DeepSeek-style MLA 的高性能 prefill/decode kernels（尤其 decode T=1）。
- 对 KV layout、block_size、head_dim 等有硬约束时，能被 BackendReport 明确暴露并自动降级。

### 约束与现实（从 vLLM 的集成可推断）
- FlashMLA 明显与 CUTLASS 生态绑定：vLLM 的 FlashMLA cmake 集成显式加入 cutlass include（见 `vllm/cmake/external_projects/flashmla.cmake`）。
- 对 CUDA 版本/架构有要求：Hopper(SM90) 才能用；且编译侧会筛 arch（同文件）。
- 典型运行时约束（需要我们在 engine 配置阶段就统一收敛）：
  - KV cache block size：业界常见强制 64（vLLM 在平台逻辑里也会对 FlashMLA 做 block_size 收敛）。
  - MLA dims 约束（head_dim/rope dims 等）。

### 接入形态（我们这边）
- `backends/attention_mla/flashmla.py` 提供：
  - `prefill(...)` / `decode(...)`
  - `is_available()`（是否编译安装成功）
  - `is_compatible(model_cfg, kv_layout, block_size, dtype, ...)`
  - `describe()`（FlashMLA 版本/commit、CUDA 版本、支持的 block_size/head_dim 等）
- 默认优先级：在 H100 上 **FlashAttn-MLA → FlashMLA → FlashInfer-MLA**（FlashMLA 若不可用不阻塞系统）。

---

## 2) DeepEP（MoE all-to-all 通信后端）

### 我们需要从 DeepEP 得到什么
- 专为 MoE token dispatch/combine 优化的 all-to-all（比 NCCL naive 更低延迟/更高吞吐）。
- 两种工作模式：
  - `low_latency`：适配 decode，小 token batch；更可能 CUDA graph 兼容（vLLM DBO 依赖它）
  - `high_throughput`：适配 prefill，大 token batch

### 业界经验（从 vLLM 文档抽取的关键点）
- vLLM 的 EP 部署指南明确列了 backend 的适用场景与取舍（见 `vllm/docs/serving/expert_parallel_deployment.md`）：
  - `deepep_low_latency`：decode，CUDA graph 支持，masked layout
  - `deepep_high_throughput`：prefill，contiguous layout
  - 并提示 DeepEP kernels 在 mixed workloads 可能表现一般（需要调度/分离策略配合）

### 接入形态（我们这边）
- 把 all2all 抽象成 `comm/all2all.py` 的统一接口：
  - `dispatch(hidden_states, topk_ids, topk_weights, ...) -> dispatched_states, permute_meta`
  - `combine(expert_outputs, permute_meta, ...) -> combined_states`
  - `supports_cuda_graph: bool`
  - `preferred_layout: enum {masked, contiguous}`（影响 MoE runner 的输入布局）
- `backends/moe/dispatcher_deepep.py` 只负责把这个接口映射到 DeepEP 的具体调用与 buffer 管理。

### 默认策略（2×H100）
我们默认追求“decode 低延迟 + 可 CUDA graph”，因此建议：
- **默认 `deepep_low_latency`**（除非用户明确指定或纯 prefill 压测）。
- 如果要在一个 engine 里同时极致 prefill 与 decode：需要更复杂的策略（例如 PD disaggregation 或 “分两套 worker”），不建议在 v0 强行做；但接口上保留未来扩展空间。

---

## 3) DeepGEMM（FP8 block-wise GEMM / grouped GEMM / MQA logits）

### 我们需要从 DeepGEMM 得到什么
DeepSeek FP8 block-wise（128×128 scale_inv）权重下的：
- Dense FP8 GEMM（线性层：QKV/O/MLP 等）
- MoE grouped GEMM（expert MLP）
-（可选）MQA/MLA 相关 logits kernel（取决于 DeepGEMM 暴露的 API）

### 约束与现实（从 vLLM 的 wrapper 可推断）
- DeepGEMM 有明确的 dims/对齐约束，否则需要 fallback（vLLM 在 `vllm/utils/deep_gemm.py` 中检查 N/K 的倍数约束）。
- DeepGEMM 是 JIT/扩展库形态，需要：
  - 可控的 cache 目录（vLLM 使用 `DG_JIT_CACHE_DIR`）
  - 建议预编译（SGLang 文档也建议先 compile deep_gemm）
- DeepGEMM 的 scale 格式可能支持更激进的压缩（例如 vLLM 提到 UE8M0 打包 scale 的策略）。

### 接入形态（我们这边）
- `linear_fp8/` 和 `moe/runner_*` 都以 DeepGEMM 为“默认实现”：
  - `linear_fp8/deepgemm_impl.py`
  - `moe/runner_deepgemm.py`
- 统一的 `Fp8BlockwiseWeight`/`QuantScaleFormat` 抽象：
  - 不把 deepgemm 的 scale packing 细节泄漏到模型代码
  - loader 负责把 HF 的 `weight_fp8 + scale_inv` 转换成 backend 需要的 layout

### fallback 策略（必须明确且可解释）
当 DeepGEMM 不可用或 dims 不满足时：
- `CUTLASS block-fp8`（第二选择，性能也很强）
- `Triton`（可开发/可 autotune，但上限不一定）
- `torch dequant`（只作为 correctness/debug）

所有 fallback 必须写入 `BackendReport`（原因链：不可用/不兼容/维度不满足/编译失败等）。

---

## 4) FlashAttention（FlashAttn-MLA）与 FlashInfer（补充后端）

### FlashAttention（推荐作为“可靠默认”）
- H100 上 FlashAttention（FA3）成熟，工程风险低。
- 作为 MLA attention 的默认路径很合理：先保证“稳定高性能”，再用 FlashMLA 拉极限。

### FlashInfer（作为可选强化）
- FlashInfer 可能提供：
  - MLA attention kernels（部分平台/shape 约束）
  - all2allv 通信（某些 NVLink / MNNVL 环境更有优势）
  - MoE kernels（与 TRT-LLM/CUTLASS 组合）
- 接入方式同样走 registry：可用则加入候选，不可用不影响主路径。

---

## 5) 与后续自研 rose* 的关系（为什么现在这样抽象）

我们现在把关键点拆成 3 类可替换的后端：
- `rosemla`：替换 `attention_mla/*`
- `roseep`：替换 `comm/all2all.py` + `moe/dispatcher_*`
- `rosegemm`：替换 `linear_fp8/*` + `moe/runner_*`

只要上层 `StepPlan/LocalBatch`（engine/executor IR）不变，我们就能：
- 先用 FlashMLA/DeepEP/DeepGEMM 把性能做到业界水平；
- 再在 2×H100 的特化前提下，用 rose* 做更激进的 intranode/NVLink-only 专用实现，冲击更高上限。

