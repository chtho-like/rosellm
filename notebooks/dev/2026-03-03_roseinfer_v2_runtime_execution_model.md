# RoseInfer v2：运行时执行模型（rank0 同时做 HTTP + Worker0）

> 本文回答一个常见误解：`torchrun` 的 rank0 不是“只有 driver 不跑模型”。在我们采用的业界主流设计里，**rank0 既是控制面（HTTP/Scheduler/Driver）也是数据面（GPU0 上的 Worker0）**。

---

## 1) 进程拓扑（2×H100，DPA=2 + EP=2）

用 `torchrun --nproc_per_node=2` 启动两个进程：

- **rank0（CUDA:0）**
  - OpenAI HTTP Server（FastAPI/uvicorn）
  - `LLMEngine`（请求队列、调度器、metrics）
  - `Driver`（跨 rank 协调 step，同步 broadcast/gather）
  - **Worker0**（执行模型 forward；参与 MoE all-to-all）
- **rank1（CUDA:1）**
  - **Worker1**（执行模型 forward；参与 MoE all-to-all）

核心原因：
- 不浪费 GPU0：DPA 的目的就是两张卡都执行 attention/dense。
- EP all-to-all 必须所有 ranks 同步参与：若 rank0 不 forward，rank1 会在 all-to-all 卡死。

---

## 2) 控制面与数据面的并发模型（rank0 内部）

### 2.1 推荐模式：Engine 后台循环 + HTTP 只做入队/出队

rank0 启动时创建一个“后台 engine loop”，持续做：
1. 从请求队列取 active sequences（prefill / decode）
2. 生成一个很小的 `StepControl`（step_id、profiling 开关、overlap/graph 等全局一致的控制字段）
3. `executor.step(step_control)`：
   - rank0 broadcast `StepControl`（rank0 自己也会收到）
   - **各 rank 在本地构建自己的 `LocalBatch`**（sequence/KV 均为本地所有，不跨 rank 传大张量）
   - rank0 执行 Worker0 的 local batch
   - rank1 执行 Worker1 的 local batch
   - gather `StepResult` 回 rank0
4. rank0 更新 sequence 状态，把 token 增量推送到对应的 streaming channel

HTTP handler 的工作变成：
- 把新请求（messages、采样参数等）封装成 `Request` 入队
- 立刻返回一个 streaming iterator（从该请求的输出 channel 读 token）

这样可保证：
- **continuous batching**：并发请求越多，越容易聚合成大 batch
- HTTP 不直接做 heavy compute：不会被单个请求长时间阻塞

### 2.2 具体实现选型（不绑定，但建议）

两种等价实现方式：

**A) asyncio-only（推荐）**
- engine loop 用 `asyncio.create_task` 跑后台协程
- 每个请求一个 `asyncio.Queue` 作为 token channel
- 适配 OpenAI SSE streaming 很自然

**B) 线程 + 队列**
- engine loop 在单独线程
- 请求/输出用 thread-safe queue
- HTTP 侧把 queue 包装成 async generator

优先选 A，除非遇到第三方库强制线程模型。

---

## 3) 跨 rank step 协议（broadcast/gather）

### 3.1 必须满足的同步条件

- 每个 step，所有 ranks 必须执行完全一致的：
  - broadcast（接收同一个 `StepControl`）
  - MoE 内部的 all-to-all（dispatch/combine 次数/顺序一致）
  - gather（回传 step_result）
- 某 rank 若当步没有本地序列，也必须参与 step：
  - `LocalBatch` 可以是 empty（B=0），但 forward 仍要跑到 MoE 的 collective（输入 empty tensor）。

### 3.2 结果回传要“最小化”

为了减少 rank 间通信：
- 默认只回传 `next_token_ids` + `seq_ids` + finish flags  
（不回传 full logits）

如需 logprobs：
- 让 worker 端算 top-k 并只回传 top-k（或只回传 logprob of sampled token）

---

## 4) DPA 的“粘性序列”约束（避免 KV 迁移）

在 DPA 下，每个 rank 维护自己的 paged KV cache，因此：
- 一个 sequence 从 admission 起就被分配到某个 `attn_dp_rank`
- 后续所有 prefill/decode 都在该 rank 上执行（不在 ranks 间迁移）

负载均衡手段：
- admission routing：新请求分配到更空闲的 rank
- chunked prefill：避免长 prompt 把某 rank 长时间占满
- per-rank batch shaping：每个 rank 在自己的队列里挑选合适的序列组成 microbatch

---

## 5) Profiling（nsys / torch.profiler）如何对齐多进程

### 5.1 统一开关

- rank0 发起 profile start/stop（写入 `StepControl` 控制字段）
- 所有 ranks 在同一步骤开始/结束 profile（避免时间轴错位）

### 5.2 产物组织

建议按如下结构落盘：

```text
notebooks/dev/artifacts/<exp_name>/
  run_report.json
  nsys/
    rank0.qdrep
    rank1.qdrep
  torch_profiler/
    rank0.json
    rank1.json
```

并在 `run_report.json` 里记录：
- backend 选择结果（BackendReport）
- 依赖版本（torch/cuda/nccl/flashattn/deepep/deepgemm/flashmla）
- profile 文件路径
