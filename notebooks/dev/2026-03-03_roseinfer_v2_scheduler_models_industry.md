# 业界 “分布式调度” 长什么样？（以及我们为什么先做集中式 Scheduler）

> 这篇笔记澄清一个容易混淆的点：业界确实存在“分布式调度”，但它通常发生在 **更粗粒度的层级（多引擎/多副本/多阶段的路由与编排）**；而在一个 **EP 同步很强** 的单实例内部，调度往往依然是“逻辑集中”的（只是执行是分布式的）。

---

## 1) 先定义：你说的 “调度” 可能指 3 件不同的事

1. **Engine 内调度（microbatch / continuous batching）**  
   - 决定这个 step 跑哪些 sequence、prefill chunk 多长、decode 批多大等。
2. **多副本/多引擎的请求路由（DP routing / load balancing）**  
   - 决定一个新请求送到哪个引擎（哪个 GPU 组），往往要考虑队列长度、KV cache/prefix cache 命中等。
3. **多阶段编排（Prefill/Decode 分离、或 pipeline stages）**  
   - 把同一请求在不同“阶段实例”之间流转（需要 KV transfer 或 stage-to-stage 通信）。

业界所谓“分布式调度”，更多是 (2)/(3)；(1) 在单实例里通常还是集中式（为了确定性与低开销）。

---

## 2) 业界有哪些“分布式调度”形态（摘取 vLLM / SGLang 的精华）

### 2.1 多引擎 DP：每个 DP rank/副本独立调度（分布式的 engine-level scheduling）

典型做法：
- 多个 **独立的 core engine 进程**（每个有自己的 KV cache 与本地 scheduler）
- 一个前端（或外部）负载均衡器把请求分发到不同 engine

vLLM 在 DP 部署文档中明确描述了：
- 每个 DP rank 是独立的 core engine 进程（前端通过 socket/RPC 连接）
- online 部署可做 **内部 LB** 或 **外部 LB**（把每个 DP rank 当作独立服务端点）  
  参考：`vllm/docs/serving/data_parallel_deployment.md:1`

这就是非常典型的“分布式调度”：调度分散在每个引擎里，路由在更上一层。

### 2.2 DPA（DP-Attention）：每个 DP worker 可跑不同 forward mode，但要在 MoE 处同步

对 DeepSeek/MLA 模型，SGLang 把 “attention 走 DP” 明确写成：
- 每个 DP worker **可以独立处理不同类型 batch**（prefill/decode/idle）
- 但在 MoE 前后要同步（因为 MoE all-to-all）  
  参考：`sglang/docs/basic_usage/deepseek_v3.md:120`

这是一种“分布式（各 worker）+ 同步点（MoE）”的调度风格：  
**局部自由度** 与 **全局同步约束** 并存。

### 2.3 Prefill/Decode 分离（PD disaggregation）：调度分布在两类实例 + router 编排

这是更强的分布式编排：
- prefill 实例有自己的 scheduler
- decode 实例有自己的 scheduler
- router 把请求拆到 prefill，再把 KV 传到 decode  
  参考：`sglang/docs/advanced_features/pd_disaggregation.md:1`

---

## 3) 为什么单实例（2×H100，DPA=2+EP=2）我们先做“集中式 Scheduler + 分布式执行”

在 DPA+EP 下，MoE 层的 all-to-all 带来强同步约束：
- 每个 step 所有 rank 都必须进入相同的 collective 序列，否则死锁
- overlap/微批/graph capture 等优化要求“全局一致”的决策（例如 microbatch slicing）

因此在 **单实例** 内，最稳、最易懂、也最不容易引入隐性开销的方式是：
- rank0 作为 **唯一的逻辑 Scheduler**（掌握全局状态、做 chunked prefill 与 token-balance）
- 通过一个很小的 `StepControl` 做跨 rank 同步
- 每个 rank 的 worker 执行本 rank 的 local batch（KV 粘在本 rank，不迁移）

这本质上是：
- **调度集中**（便于保证一致性/优化策略）
- **执行分布式**（充分利用两卡；MoE all-to-all 正常工作）

> 后续如果要做到“像大规模系统那样的分布式调度”，我们会把它放到 **engine 之上的路由层**（类似 SMG / 外部 LB），而不是把单实例内部搞成多个互相抢占/协调的 scheduler。

---

## 4) 我们的演进路线（从单实例到分布式调度）

1. **v2 单实例（本阶段）**：集中式 Scheduler（rank0）+ worker0/1 执行（DPA+EP）
2. **多实例 DP（将来）**：起多个 roseinfer v2 实例（每实例 2×H100），每实例内部仍集中调度；外部加一个 router（可 Rust/可 Python）做 cache-aware routing
3. **PD 分离（将来）**：把 prefill/decode 拆成两个实例池，router 编排 + KV transfer（仅当 SLA/吞吐需要）

