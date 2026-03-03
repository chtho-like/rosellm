# RoseInfer v2（重构）详细设计：DeepSeek-V2-Lite-Chat-FP8（2×H100，DPA+EP）

> 日期：2026-03-03  
> 目标：用一个“小号但形态接近 V3 的 DeepSeek-V2-Lite FP8”作为训练场，把 **MLA + MoE（EP）+ FP8 block-wise** 的完整高性能推理链路在 2×H100 上跑到接近（并具备追平/超越空间）vLLM/SGLang。  
> 最高优先级：**性能**。第二优先级：**清晰、易懂、优雅且不过度**。  
> 硬约束：**不使用 HF remote code**；模型/权重/后端都由 rosellm 自定义实现；但要支持官方 deepseek infra（FlashMLA / DeepEP / DeepGEMM）作为高性能后端。

---

## 0. 术语与关键结论（对齐业界）

### 0.1 术语

- **DPA / DP-Attention**：只在 attention/KV cache 维度做 data-parallel（每个 DP worker 维护自己那份 KV cache），用来避免 MLA 模型在 TP 下 KV cache 复制导致的显存浪费和吞吐下降。SGLang 把它叫 DPA（见 `sglang/docs/advanced_features/dp_dpa_smg_guide.md`）。
- **EP / Expert Parallel**：MoE experts 分 shard 到多卡，通过 all-to-all 做 token dispatch/combine（DeepEP / FlashInfer / NCCL 等）。
- **TP(Attention)**：张量并行切 head/hidden。对 MLA（KV head 少）通常会造成 KV cache 复制，收益有限甚至负收益。

### 0.2 两卡 DeepSeek（MLA+MoE）最佳实践结论

在仅 2×H100 且 EP=2 的前提下，**最佳 parallel 形态是：DPA=2 + EP=2，Attention TP=1**。  
这是 vLLM 的 DP+EP（并围绕它做 DBO）和 SGLang 的 DPA+EP（并围绕它做 EP 模块化/多后端）的共同“精华”。

---

## 1. 设计目标（按优先级排序）

1. **高性能默认路径**（无需“低配 baseline 才能跑通”）
   - 默认启用：DPA=2、EP=2、paged KV cache、MLA attention backend、FP8 block-wise GEMM（DeepGEMM 优先）、MoE dispatch/combine（DeepEP 优先）。
   - “能跑”不是目标，“跑得快”才是目标；fallback 只用于 debug/可用性。
2. **清晰的 vLLM-v1 风格结构**
   - engine/scheduler 与 model_executor/worker 清晰解耦
   - 模型定义、权重加载、算子后端各自独立
3. **插件化后端，且“实际选择”可解释/可复现**
   - attention / moe-a2a / moe-runner / linear-fp8 / kv-cache / cudagraph / overlap 都是可插拔
   - 每次运行产出统一 `BackendReport`（选择了什么、为什么、版本号、降级原因）
4. **为后续自研 kernel（rosemla/roseep/rosegemm）预留清晰接口**
   - 先把系统 IR/接口做对，未来替换实现不动上层逻辑

---

## 2. 总体架构（精炼版 vLLM-v1 + 吸收 SGLang DPA/EP 框架）

### 2.1 分层与职责

建议把 `rosellm/rosellm/roseinfer/` 完全重构为以下层次：

1) `serve/`（OpenAI 协议）
- FastAPI 只负责协议与流式输出；不直接接触模型/后端

2) `engine/`（系统大脑）
- `LLMEngine`：唯一入口，负责生命周期、warmup、metrics、profiling
- `Scheduler`：continuous batching + chunked prefill；输出“每一步要执行的 step plan”
- `SequenceManager`：请求状态机（prefill/decoding/finished）、stop 条件、logprob（可选）

3) `executor/`（分布式执行）
- **rank0（GPU0）= API Server + Scheduler + Driver + Worker0（执行模型）**
- **rank1..（GPU1..）= Worker（执行模型）**
- 通信协议：优先使用 `torch.distributed` collectives（两卡稳定可靠）；后续可扩展更复杂 RPC，但不需要一开始引入

4) `models/`（自研模型）
- `models/deepseek_v2/`：DeepSeek-V2 系列（Lite 与大号共享同一套 primitives）

5) `backends/`（性能关键）
- attention_mla / kv_cache / moe（dispatch + runner + overlap）/ linear_fp8 / comm(all2all)

6) `weights/`（权重加载）
- safetensors index/shard loader、FP8 block-wise scale 格式、per-rank expert 选择

7) `profiling/` + `benchmarks/`
- 统一插桩（NVTX）、nsys/torch.profiler 集成、结果 JSON 与绘图

### 2.2 建议的目录树（初稿）

```text
rosellm/rosellm/roseinfer/
  serve/
    openai/
      server.py
      schemas.py
      launch.py               # torchrun entry: rank0(http+worker0), others worker loop
  engine/
    llm_engine.py
    scheduler.py              # continuous batching + chunked prefill
    sequences.py              # Sequence / Request state machine
    sampling.py
    reports.py                # BackendReport / RunReport
  executor/
    parallel_context.py       # process groups + rank info
    driver.py                 # rank0 orchestration
    worker.py                 # execute step
    transport.py              # dist collectives wrappers (broadcast/gather)
  models/
    registry.py
    deepseek_v2/
      config.py               # parse config.json (no transformers remote)
      model.py
      layers_attention_mla.py
      layers_moe.py
      layers_norm.py
      chat.py
      weights.py              # tensor spec + mapping keys
  backends/
    attention_mla/
      flashattn_mla.py        # fallback, H100 first-choice if flashmla absent
      flashmla.py             # optional, high-perf decode/prefill kernels
      flashinfer_mla.py       # optional
    kv_cache/
      paged.py                # block manager + block tables
      layouts.py              # KV layouts: bf16/fp8 (for MLA)
    moe/
      api.py                  # FusedMoE interface (dispatch/runner/combine)
      dispatcher_deepep.py
      dispatcher_flashinfer.py
      dispatcher_nccl.py
      runner_deepgemm.py
      runner_cutlass.py
      runner_triton.py
      overlap_dbo.py           # optional: overlap comm/compute
    linear_fp8/
      fp8_blockwise.py        # W8A8 blockwise FP8 linear interface
      deepgemm_impl.py
      cutlass_impl.py
      torch_fallback.py
    comm/
      all2all.py              # DeepEP/FlashInfer/NCCL abstraction
  weights/
    hf_download.py            # cache-only (hf_hub_download)
    safetensors_index.py
    fp8_blockwise.py
    loader.py                 # per-rank load + tied weights
  profiling/
    nvtx.py
    nsys.py
    torch_profiler.py
  benchmarks/
    serving_offline.py
    serving_online.py
    compare_vllm.py
```

> 注：不追求文件数最少，而追求“改一个地方不牵连其他地方”。但也避免像大项目一样把一件事拆成十几个无意义小文件。

---

## 3. 并行设计（2×H100：DPA=2 + EP=2）

### 3.1 组（groups）与语义（明确命名，避免混乱）

在两卡场景，我们采用 **重叠 groups**（业界已验证可行）：

- `attn_dp_group`：world（size=2）
  - 语义：attention/KV cache 按 rank 分摊请求与 KV（每 rank 维护自己的 KV，不复制）
- `ep_group`：world（size=2）
  - 语义：MoE all-to-all dispatch/combine

注意：这里不是正交 dp×ep（那需要 4 卡），而是 **DPA 与 EP 共享同一组 rank**。

### 3.2 运行时约束（避免死锁，保证吞吐）

- 每个 step，所有 rank 必须调用相同数量/顺序的 MoE all-to-all（由模型结构保证）。
- 某个 rank 若没有 token，也必须进入 MoE（输入 empty），这样 collective 仍能匹配。
- Scheduler 要做 **request-level / token-level 的负载均衡**，尽量避免某 rank 长期空转（否则其它 rank 会被 all-to-all 同步点拖慢）。  
  - 重要约束：**DPA 下每个 sequence 的 KV cache 必须“粘在”一个 rank 上**（不做跨 rank KV 迁移）。因此“均衡”主要通过 *新请求路由*、*chunked prefill*、以及每 rank 本地队列的 batch shaping 实现，而不是在 rank 间随意搬运已有序列。

### 3.3 rank0/其它 rank 的职责

- **rank0（GPU0）**：HTTP + Scheduler + Driver + Worker0（执行模型）  
- **rank1（GPU1）**：Worker1（执行模型，等待 rank0 broadcast 的 step plan）

Driver/Worker 不做复杂 RPC，靠 dist collectives：

1. rank0 **broadcast 一个很小的 `StepControl`**（step_id、profiling 开关、是否启用 overlap/graph 等“全局一致”的控制字段）
2. 各 rank **在本地** 从自己的队列/sequence 状态构建 `LocalBatch`（不跨 rank 传大张量；KV cache 也不迁移）
3. 各 rank 执行 `execute_step(local_batch)`（MoE all-to-all 在模型内部发生）
4. 各 rank gather `StepResult`（只传最小信息，如 next_token_ids），rank0 更新序列并流式输出

> 这样设计的好处：**MoE 本身就需要 collectives**，我们额外同步只做“最薄一层”，系统更稳、行为更可预测，且便于 profiling。

---

## 4. 核心数据结构（尽量少、尽量“张量化”）

### 4.1 StepControl / LocalBatch（分布式执行 IR）

设计原则：worker 侧尽量不碰 Python list/dict（可选 metadata 例外），避免开销；关键字段用 tensor。

`StepControl`（rank0 广播，必须很小）：
- `step_id: int`
- `global_mode: enum {MIXED, PREFILL_ONLY, DECODE_ONLY}`（用于全局一致的策略选择；每 rank 仍可在本地选择 empty batch）
- `profiling: {off|torch|nsys|...}`（统一起停，避免多进程时间轴错位）
- `overlap_policy`（可选：是否启用 DBO/TBO；若启用，microbatch slicing 必须全局一致）
- `backend_overrides`（可选：强制某次 step 用特定后端，用于对比/调试）

`LocalBatch`（每 rank 只消费自己的那份）：
- `input_ids`（prefill: [B, T], decode: [B, 1]）
- `positions / position_ids`
- `kv_cache_metadata`（paged KV：block_tables、slot_mapping、context_lens 等）
- `seq_ids`（把输出 token 对应回 sequence manager）
- `sampling_params`（温度、top_p 等；可选下推到 worker 端采样以减少回传）

`StepResult`（gather 回 rank0）：
- `seq_ids`
- `next_token_ids`
- `finish_flags`（eos/stop）
- （可选）`logprob` / `top_logprobs`（OpenAI 可选字段）

### 4.2 KV Cache（paged，DPA 天然友好）

每个 rank 有自己的 `BlockManager` + `PagedKVCache`：
- block size：默认 **64**（强烈建议对齐 FlashMLA/vLLM 的常见约束；vLLM 对 FlashMLA 会强制 block_size=64）
- KV dtype：优先支持 `bf16` 与 `fp8_e4m3`（在 H100 上 decode 容量/带宽很关键）
- KV layout：为 MLA 定制（DeepSeek 的 K/V 不是标准 MHA/GQA）

关键点：DPA 模式下，KV 完全不需要跨 rank 迁移；只有 MoE 中间激活需要 all-to-all。

---

## 5. DeepSeek-V2-Lite（FP8 block-wise）模型实现要点（自研，不 remote code）

### 5.1 共享“大号 V3 技术”的原因

DeepSeek-V2 Lite 虽小，但关键形态与 V3 接近：
- MLA（latent attention / MQA 类）
- MoE（top-k routing）
- 权重 FP8 block-wise（128×128 scale_inv）

因此它非常适合作为“练兵模型”：把 DPA+EP、FP8 kernels、MoE overlap、KV layout 这些大号必备技术先在小模型上跑顺并做极致优化。

### 5.2 权重格式（FP8 block-wise）

权重为 `float8_e4m3fn` + `*_scale_inv`（float32），块大小通常 128×128；边缘块可能 padding（scale_inv 的 block grid 以 ceil-div 计算）。

设计建议：
- `weights/fp8_blockwise.py` 定义统一的 `BlockwiseFp8Weight` 表示（含 weight_fp8 + scale_inv + block_shape + original_shape）
- `linear_fp8/` 的实现只依赖这个结构，不依赖 HF key 命名

### 5.3 FP8 GEMM（默认 DeepGEMM，fallback CUTLASS/torch）

默认：DeepGEMM（H100/Hopper 支持）：
- DeepGEMM 在 vLLM 侧有明确的 dims 要求（例如 N%64、K%128 等），需要在 loader/pack 阶段就保证满足或降级（见 vLLM `vllm/utils/deep_gemm.py`）。
- 注意：FlashMLA/DeepGEMM 都明显与 CUTLASS 生态耦合（FlashMLA cmake 明确包含 cutlass include；见 `vllm/cmake/external_projects/flashmla.cmake`）。

fallback：
- CUTLASS block-fp8 kernel（需要我们自己封装/或用已有实现）
- torch fallback（dequant 或慢路径，仅 debug）

### 5.4 MoE（分成“通信”和“计算”两个可替换维度）

借鉴 SGLang 的模块化 EP：

MoE = Router + Dispatcher(A2A) + Runner(GEMM) + Combine

- Router：本地计算 topk（尽量用 fused kernel 或至少 vectorized）
- Dispatcher：DeepEP（优先） / FlashInfer / NCCL
- Runner：DeepGEMM（优先） / CUTLASS / Triton

并预留 overlap：
- **DBO/TBO**：把 local batch 拆成两个 microbatches，在 dispatcher/runner/combine 之间插 yield points，用线程/stream 交错执行（参考 vLLM DBO 与 SGLang TBO/SBO 思路）。

---

## 6. 后端选择策略（自动优先高性能，且可解释）

### 6.1 BackendRegistry（统一入口）

每类后端都实现：
- `is_available()`
- `is_compatible(model_cfg, engine_cfg, device_caps)`
- `estimate_cost()`（可选：启发式）
- `describe()`（版本、依赖、限制）

Registry 负责：
1) 按“优先级列表”尝试（对齐 vLLM 的做法）  
2) 选出第一个兼容且可用的  
3) 生成 `BackendReport`（包含降级原因链）

### 6.2 默认优先级（H100 / DeepSeek MLA + MoE + FP8）

- `attention_mla`：
  1. FlashAttn-MLA（H100 上可靠，作为默认）
  2. FlashMLA（若可用且 dims/块大小兼容，优先 decode）
  3. FlashInfer-MLA（可选）

- `moe_a2a`：
  1. DeepEP low_latency（decode 为主时；CUDA graph 友好）
  2. DeepEP high_throughput（prefill token 很大时）
  3. FlashInfer all2all
  4. NCCL all_to_all（兜底）

- `moe_runner`：
  1. DeepGEMM（H100 默认）
  2. CUTLASS grouped GEMM
  3. Triton grouped GEMM
  4. torch

- `linear_fp8`：
  1. DeepGEMM（满足 dims 约束时）
  2. CUTLASS W8A8 block-fp8
  3. torch（dequant）

> 关键：所有“自动选择”都必须落盘记录，方便 benchmark 对齐 vLLM 并定位差距。

---

## 7. CUDA Graph / 编译（高性能但不破坏可理解性）

### 7.1 Decode CUDA Graph（优先）

decode 的 shape 相对稳定（T=1），非常适合 CUDA graph；但 MoE all-to-all backend 必须 CG compatible：
- DeepEP high_throughput 通常不 CG compatible（vLLM 也会禁用）
- DeepEP low_latency 更友好（vLLM DBO 依赖它）

策略：
- 如果 `moe_a2a` 选择了 CG 不兼容后端，则自动关闭 decode graph，并在 `BackendReport` 写明原因。
- 支持 `--enforce-cudagraph`（仅用于实验/对比）。

### 7.2 Prefill/Extend 的图优化

prefill token 数变化大，常规 full graph 难；SGLang 做了 PCG（piecewise cuda graph），但它对 DP attention / MoE A2A 等会自动禁用（见 `sglang/docs/advanced_features/piecewise_cuda_graph.md` 的兼容性列表）。

对我们而言（2×H100，先把链路跑快）：
- v0 先不强推 prefill PCG（容易复杂且 debug 成本高）
- 但结构上要把“split ops（MoE dispatch）”做成显式节点，未来接 PCG 很自然

---

## 8. 调度（Scheduler）要点：让 DPA+EP 不被“prefill 卡 decode”拖垮

### 8.1 关键矛盾

DPA 模式下各 rank 维护独立 KV、可独立做 attention；但 EP 的 MoE 会在层内引入同步点。  
若某 rank 长时间跑大 prefill（长 prompt），另一个 rank 的 decode 会在 MoE 同步点被拖住 → decode latency 爆炸。

### 8.2 解决路线（优先 vLLM 思路：chunked prefill）

默认启用：
- **chunked prefill**：把长 prefill 拆成小 chunk（例如每 step 最多 N tokens 的 prefill），让 decode 更频繁地被调度到 GPU 上。
- **token-load balance**：每 step 在 rank 间按“总 tokens”平衡分配 local batch（类似 SGLang “total_tokens” load balance 的思想）。

不默认启用（但保留扩展点）：
- PD disaggregation（prefill/decode 分离实例）：两卡 EP=2 场景收益不一定大，且系统复杂度上升；后续再评估。

---

## 9. 观测性与工程化（让性能迭代更快）

### 9.1 BackendReport（每次运行必出）

统一输出：
- 选到的 attention backend / moe-a2a backend / moe-runner / linear-fp8 / kv dtype / block_size / cudagraph / overlap 开关
- 关键依赖版本：torch/cuda/nccl/flashattn/flashinfer/deepep/deepgemm/flashmla
- 自动降级原因链（例如：DeepGEMM dims 不满足 → 退 CUTLASS）

### 9.2 Profiling（nsys + torch.profiler）

默认插 NVTX：
- step boundary
- attention prefill/decode
- MoE dispatch/runner/combine
- KV cache ops（alloc/append）

然后：
- nsys：抓 timeline + NVLink/NVSwitch/NCCL 活动
- torch.profiler：抓 kernel-level 的热点分布与 graph breaks

---

## 10. 未来：rosemla / roseep / rosegemm（从 2×H100 场景“自研极致”）

你提出的路线非常对：先用官方 infra 快速到业界水平，再针对 2×H100 的“特化场景”从零做更激进的优化。

### 10.1 roseep（intra-node / NVLink only）

目标：把 “两卡 all-to-all（dispatch/combine）”做成更极致的 intranode path：
- 利用 NVLink P2P + 固定两端 rank（world_size=2）的特殊性
- 只支持两卡/单机，换取更少的泛化代码与更高性能
- 在 dispatch/combine 中尽可能：
  - 减少中间 buffer 次数与格式转换
  - 优化 token permutation（GPU kernel 做 prefix-sum + scatter/gather）
  - 支持 CUDA graph（关键）

### 10.2 rosegemm（DeepSeek block-fp8 的极致 GEMM）

方向：
- 基于 CUTLASS 3.x + TMA + warp-specialization（Hopper）
- 做“为 DeepSeek 常见 N/K/M 形状”的 kernel 特化与 autotune（小 batch decode 与大 batch prefill 分别优化）
- 把 scale 布局（UE8M0/packed scales）与 GEMM 融合，减少带宽

### 10.3 rosemla（MLA attention 的极致 decode/prefill）

方向：
- 借鉴 FlashMLA（其源码内含 CUTLASS）但做更“2×H100 + 目标 head_dim + block_size”特化
- 重点盯 decode（T=1）吞吐与延迟：
  - KV 访问模式（block table）与 cache locality
  - 可能的 FP8 KV layout 与 fuse（qkv proj + logits + softmax*V）

> 这三件事都应该在 v2 架构下作为可插拔 backend 实现，不改变 engine/scheduler 的 IR。

---

## 11. 里程碑（建议顺序：先快后更快）

> 这里不是“baseline→优化”，而是“高性能默认路径先上线”，然后再往更极致推进。

### M0：架构落地（不追求所有优化都做完，但接口必须对）
- v2 目录结构落地（engine/executor/models/backends/weights）
- DPA=2 + EP=2 的 step 协议跑通（dist broadcast/gather）
- DeepSeek-V2-Lite-Chat-FP8：权重加载（只加载本 rank experts）+ 正确性验证

### M1：高性能默认路径
- DeepGEMM linear + MoE runner 接入（满足 dims 时默认用）
- DeepEP all2all 接入（默认 low_latency；prefill 场景可切 high_throughput）
- MLA attention：FlashAttn-MLA 接入（默认）
- paged KV cache（block_size=64）

### M2：逼近/超越 vLLM 的关键增益
- overlap（DBO/TBO）+ decode CUDA graph（条件满足时默认开）
- FlashMLA（若能稳定编译/运行）作为 decode 优先后端
- 更细的 scheduler：更好的 token-balance + chunked prefill 策略

### M3：自研 rose*（在 2×H100 上“定制化打满”）
- roseep（intranode 专用）
- rosegemm/rosemla（特化 kernel + autotune）

---

## 12. 风险清单（提前设计规避）

1) **DP-Attention + EP 同步点导致 decode latency 被 prefill 拖垮**  
→ 默认 chunked prefill + token-balance；必要时引入 prefill delayer/更激进调度。

2) **DeepEP/DeepGEMM/FlashMLA 的编译与环境依赖**  
→ 作为 optional deps，但在 H100 环境默认启用；提供明确的 “依赖不可用→fallback→报告”。

3) **CUDA graph 兼容性**  
→ 用 `BackendReport` 明确记录为什么开/关；只在确定兼容时启用。

4) **FP8 block-wise 的 dims/对齐要求导致频繁 fallback**  
→ loader/pack 阶段尽量满足要求（padding/对齐）；否则明确降级策略。

---

## 13. 结语：为什么这套能“比业界更好”

- 取其精华：
  - vLLM：paged KV + 后端优先级选择 + DBO 思想
  - SGLang：DPA 的并行语义、EP 模块化、后端可插拔、overlap 的结构化表达
  - TRT-LLM：mapping/plan 明确、kernel 边界清晰
- 去其糟粕：
  - 不引入大而全的配置面与复杂部署组件（两卡场景不需要）
  - 不把 DP（复制模型）与 DPA（attention DP）混在一起，内部命名明确
  - 不依赖 remote code；模型/权重/后端都由我们控制，便于长期维护与极致优化
