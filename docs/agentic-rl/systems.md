# Systems and Infrastructure

Agentic RL is often generation-bound and environment-bound rather than purely
optimizer-bound. A fast trainer is useless when rollout GPUs wait on browsers,
sandboxes, users, reward models, or straggling episodes. The systems objective
is to maximize **valid, appropriately on-policy, reward-bearing training tokens
per unit time and cost** without corrupting the statistical objective.

## 1. Components

| Component | Responsibility | Stateful data |
|---|---|---|
| Task registry/curriculum | choose task, seed, group, and priority | task attempts and difficulty |
| Rollout controller | create/cancel episodes and enforce budgets | trajectory lifecycle |
| Policy inference server | sample exact token IDs/log-probabilities | KV cache and policy version |
| Agent loop | render history, parse actions, manage memory | transcript/scaffold state |
| Environment worker | reset, step, snapshot, validate | world state |
| Tool/sandbox pool | execute code, APIs, GUI, retrieval | process/tool state |
| Reward/verifier service | compute versioned feedback | evaluator models/rubrics |
| Trajectory store | immutable event and artifact lineage | tokens, observations, rewards |
| Advantage/data worker | group, filter, estimate, and pack | returns and tensors |
| Actor trainer | compute objective and update policy | weights, optimizer, scheduler |
| Critic trainer | learn values when configured | critic checkpoint |
| Reference server/model | compute KL log-probabilities | frozen reference |
| Weight publisher | atomically deliver new policies | version manifest |
| Evaluation plane | hidden fixed and interactive suites | protected tasks/results |
| Observability/control | metrics, traces, incidents, stop/rollback | run state |

Keep control-plane state (versions, leases, lineage, permissions) separate from
high-volume data-plane tensors and artifacts.

## 2. Three placement patterns

### Fully colocated

Actor training and rollout inference share the same accelerators at different
times. Workers switch modes by freeing/reallocating optimizer states, KV cache,
and inference buffers.

Advantages:

- high utilization on a small cluster;
- no dedicated duplicate actor weights;
- direct/local weight transition.

Costs:

- mode-switch latency and memory fragmentation;
- generation and training cannot fully overlap;
- failures affect both phases;
- training and inference prefer different parallel layouts.

### Disaggregated

Dedicated inference workers collect experience while dedicated trainers update.

Advantages:

- stages overlap;
- each pool uses a suitable engine/topology;
- independent scaling for rollout, reward, environment, and training.

Costs:

- duplicate weights and reference/critic memory;
- checkpoint conversion and network broadcast;
- behavior-policy lag;
- more queueing, failure, and lineage complexity.

### Hybrid

Some actors/references/critics are colocated and some rollout capacity is
disaggregated. HybridFlow/veRL represents placements/dataflows flexibly
([Sheng et al.](https://arxiv.org/abs/2409.19256)). The right layout depends on
model size, rollout/optimization ratio, environment latency, and interconnect.

## 3. The asynchronous rollout architecture

An efficient long-horizon collector separates model inference from agent and
environment execution:

```text
many async AgentLoops
    | batched generate(token_ids, sampling, policy_version)
    v
inference gateway / load balancer
    |
    +--> policy replica / DP group 0
    +--> policy replica / DP group 1
    +--> ...

AgentLoop action --> parser/authorizer --> environment/tool RPC
       ^                                      |
       +-------------- observation -----------+
```

While one episode waits for a tool, other episodes supply generation work.
Continuous batching merges ready requests with different histories. The
inference engine must return exact tokens; the agent loop can decode them for
parsing and display.

veRL documents this client/server pattern and warns that text round-tripping can
change tokenization
([agentic RL documentation](https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst)).

## 4. Throughput model

For one policy update, approximate wall time as

\[
T_{\text{sync}}
=T_{\text{rollout}}+T_{\text{reward}}+T_{\text{prep}}+T_{\text{train}}
+T_{\text{sync}}+T_{\text{eval}}.
\]

With pipelining, steady-state throughput is bounded by the slowest stage plus
dependencies and bubbles:

\[
\text{throughput}\lesssim
\min(C_{\text{rollout}},C_{\text{env}},C_{\text{reward}},C_{\text{train}}).
\]

But raw trajectories/s is misleading. Define effective yield:

\[
Y=
\frac{\text{valid action tokens admitted to updates}}
{\text{accelerator-seconds} + \lambda\,\text{environment/tool cost}}.
\]

Track losses from:

- prompt/prefix recomputation;
- padding and packing;
- invalid actions and infrastructure failures;
- all-equal groups/dynamic-sampling rejection;
- overlong/truncated episodes;
- policy-lag rejection;
- evaluator errors;
- safety quarantine.

A system can double generation tokens/s while reducing learning yield if those
tokens are too stale or uninformative.

## 5. Generation performance

### Prefill and decode

Prefill processes the full prompt in parallel and is compute-heavy; decode
produces one token per sequence and is usually memory/KV-cache sensitive. Agent
workloads repeatedly alternate them as tool observations extend context.

Optimize with:

- continuous batching;
- chunked prefill to avoid blocking decodes;
- prefix/KV-cache reuse for common task/group prompts;
- paged KV allocation;
- prefill/decode disaggregation;
- speculative decoding only when behavior probabilities and exact sampled tokens
  remain correct for training;
- structured decoding with efficient grammar masks;
- request routing by prefix locality and length.

Prefix caching changes performance, not the probability distribution, when
implemented exactly. Approximate/speculative methods require proof that accepted
token samples and log-probabilities match the declared behavior policy.

### Group rollout

GRPO/RLOO groups share a prompt and often an initial environment snapshot.
Compute the common prefix once and fork KV/state with copy-on-write where safe.
Environment branches must receive independent seeds or controlled shared
randomness according to the estimator.

## 6. Variable-length scheduling and stragglers

Agent episode latency has a heavy tail: different token lengths, numbers of tool
calls, environment delays, and retries. A synchronous batch that waits for every
trajectory wastes resources.

Options:

- dynamic admission of new episodes;
- per-turn/action token budgets;
- partial-rollout continuation across iterations;
- length-aware batching and routing;
- speculative task duplication with first-valid completion (careful: changes
  attempt accounting);
- defined partial-group policy;
- cancellation with explicit truncation semantics;
- separate queues for long-context or slow-tool tasks.

Completion-time filtering creates bias if hard/long failures miss update
deadlines. Compare launched and admitted distributions by task, reward, length,
and latency.

## 7. Partial rollout continuation

For an unfinished trajectory, preserve environment snapshot, agent state, token
history, policy segments, and remaining budget. At the next collection window,
resume with a current or declared policy.

If earlier turns came from policy \(v\) and later turns from \(v+1\), the
trajectory is multi-policy. Options:

1. train only newly generated tokens using prior history as fixed context;
2. apply correct per-segment behavior ratios;
3. discard mixed trajectories;
4. use an explicitly off-policy estimator.

Never label the whole trajectory with the newest version. Kimi k1.5 publicly
described partial rollout continuation and training only newly generated
segments in its iterative system
([Kimi k1.5, Section 3.3](https://arxiv.org/abs/2501.12599)).

## 8. Policy versioning and weight publication

Each policy release needs:

```json
{
  "policy_version": 184,
  "training_step": 9520,
  "checkpoint_hash": "sha256:...",
  "model_config_hash": "sha256:...",
  "tokenizer_hash": "sha256:...",
  "chat_template_hash": "sha256:...",
  "sampler_contract": 6,
  "created_at": "...",
  "ready": true
}
```

Publication protocol:

1. checkpoint trainer shards and metadata;
2. convert to inference layout deterministically;
3. compute hashes and run a small logit parity suite;
4. transfer/stream all shards;
5. load into a shadow slot;
6. verify complete model and tokenizer/template;
7. atomically switch new requests to the version;
8. allow old requests to drain with their original version;
9. acknowledge replica versions to the controller.

Avoid in-place partial weight updates. A trajectory sampled across an
unidentified mixture of tensors has no valid behavior policy.

## 9. Train/inference numerical mismatch

Even identical checkpoint files can yield different logits because of:

- different model implementations;
- attention/kernels and reduction order;
- tensor/expert parallel partitioning;
- precision, quantization, and fused operations;
- rotary scaling or position IDs;
- MoE router tie-breaking/capacity;
- logits processors and constrained masks;
- tokenizer/chat-template/version differences.

Run parity tests on fixed prefixes:

1. compare input token IDs and positions;
2. compare selected-layer activations where possible;
3. compare logits/top-k and sampled-token log-probabilities;
4. compare greedy output;
5. compare behavior old-logprob stored at rollout to trainer recomputation.

Set tolerances by precision and measure their effect on policy ratios. “Greedy
text matches” can hide log-probability mismatch large enough to destabilize PPO.

## 10. Distributed actor training

Large actors combine parallel dimensions:

- **data parallelism:** different trajectory batches;
- **tensor parallelism:** shard matrix operations;
- **pipeline parallelism:** shard layers;
- **sequence/context parallelism:** shard long sequences;
- **expert parallelism:** distribute MoE experts;
- **FSDP/ZeRO:** shard parameters, gradients, and optimizer states.

Agentic RL adds variable-length packed samples and potentially several models
(actor, critic, reference, reward). Ensure:

- segment-isolated attention;
- globally correct token/sample denominators;
- group members are available where advantage is computed;
- MoE load balancing does not correlate with padding/packing artifacts;
- gradient clipping uses the global sharded norm;
- checkpointing captures data/curriculum and async cursors.

## 11. Model placement and memory budget

For parameter count \(P\), rough training memory includes:

- weights: \(P b_w\);
- gradients: \(P b_g\);
- optimizer states/master weights: often several \(P\) bytes;
- activations: function of batch, sequence, layers, hidden width, checkpointing;
- temporary/fused-kernel workspaces;
- communication buffers.

PPO may add a critic of similar size and frozen reference/reward models. GRPO
removes the critic but increases grouped generation. Decide whether to:

- shard/freeze/offload reference;
- precompute reference log-probabilities;
- share backbone for reward/value heads (with coupling risks);
- colocate models and time-share memory;
- dedicate separate GPU pools;
- use LoRA/adapters for the actor.

Account for rollout KV cache separately. Long context can make KV memory larger
than active weights per request.

## 12. Environment service design

Environment workers need lease-based lifecycle:

1. allocate immutable task snapshot;
2. issue environment/episode ID and capability token;
3. accept ordered action with idempotency key;
4. persist transition before acknowledging;
5. renew lease while active;
6. terminate and collect artifacts;
7. scrub secrets/state before reuse.

Use action sequence numbers to reject duplicates/out-of-order delivery. A retry
after ambiguous timeout must query action status before re-execution. This is
critical for email, payments, file deletion, or any irreversible tool.

Pool warm sandboxes to reduce startup cost, but prove cleanup/isolation between
tasks. Reused containers can leak state across episodes.

## 13. Backpressure and admission control

Unbounded queues create stale rollouts and memory failure. Each boundary needs a
policy:

- maximum queued tasks/requests/tokens/bytes;
- priority and fairness by task family;
- deadline and cancellation semantics;
- whether producer blocks, spills, or resamples;
- lag/age admission at trainer;
- disk/object-store quotas and retention;
- circuit breakers for failed tools/reward services.

The curriculum should react to capacity intentionally. Do not let “fastest
environment wins” become the implicit training distribution.

## 14. Fault tolerance and exactly-once meaning

Exactly-once execution is generally impossible across arbitrary external side
effects. Aim for:

- at-least-once event delivery with deduplication;
- idempotent actions where possible;
- durable event log before acknowledgment;
- immutable trajectory finalization;
- deterministic identifiers from run/task/attempt/turn;
- leases for orphan detection;
- checkpointed queue and curriculum cursors;
- explicit partial/infrastructure-invalid status.

Failure injection should kill inference replicas, trainers, environment workers,
reward services, storage, and network links. Verify no action is silently
duplicated and no corrupted trajectory enters training.

## 15. Observability

Use a trace ID spanning curriculum, inference, agent, tool, environment, reward,
packing, and training. One trace should answer:

- which policy and sampler generated every token;
- how long each prefill/decode/tool transition took;
- what action parser/authorizer decided;
- what state transition and side effects occurred;
- which reward component came from which evaluator;
- how the advantage was computed;
- which minibatch/update consumed the token;
- which checkpoint resulted.

### Core metrics

**Rollout:** prompt/output tokens, TTFT, inter-token latency, cache hit, batch
size, generation utilization, group completion.

**Environment:** reset/step/validation latency, error type, sandbox allocation,
CPU/memory/disk/network usage.

**Data:** launched/completed/valid/admitted tokens, filter reasons, policy lag,
replay success.

**Training:** tokens/s, model FLOP utilization, communication, memory, loss,
ratio/KL/clip/entropy, gradient, checkpoint time.

**End to end:** verified successes per hour/dollar, valid learning tokens per
accelerator-second, safety incidents, evaluation regressions.

Avoid storing unrestricted chain-of-thought or sensitive observations in general
metrics/logs. Use access-controlled artifacts and a defined retention policy.

## 16. Security architecture

RL actively searches behavior space; assume the policy will find accidental
capabilities and reward loopholes.

- isolate policy plane from control plane;
- never mount cluster credentials in tool sandboxes;
- use short-lived least-privilege capability tokens scoped to one action/task;
- deny network by default, proxy allowlisted destinations, log egress;
- separate training environment, validator, and hidden evaluation networks;
- treat tool/web output as untrusted prompt content;
- cap forks, processes, files, output, compute, and time;
- prohibit access to reward model prompts, expected answers, and data registries;
- hard-stop unauthorized actions outside the learned reward;
- retain immutable security audit logs and a kill switch.

Safety controls must not depend on the same policy being optimized.

## 17. Cost accounting

Separate:

- rollout inference accelerator-hours;
- actor/critic/reference/reward training/inference;
- environment CPU/GPU/memory/storage/network;
- paid tools/APIs/search;
- human annotation/adjudication;
- failed/rejected generation;
- evaluation and hyperparameter search;
- checkpoint/object storage and transfer.

Report marginal final-run cost and total development cost separately. A vendor
figure for a final pretraining run rarely includes data acquisition, failed
runs, post-training, evaluation, serving, or staff.

## 18. Capacity-planning procedure

1. Measure one episode distribution: input/output tokens, turns, tool time,
   reward time, success, accepted yield.
2. Benchmark inference prefill/decode by context and concurrency.
3. Benchmark environment and sandbox capacity independently.
4. Benchmark trainer tokens/s and update ratio.
5. Choose policy-update cadence and maximum lag.
6. Solve approximate stage capacities; provision the bottleneck plus failure
   headroom.
7. Run a closed-loop load test because batching/queueing interactions are
   nonlinear.
8. Verify statistical distribution and safety under load, not just throughput.

## 19. Framework reading map

- [veRL / HybridFlow](https://github.com/verl-project/verl): flexible RL
  dataflows, FSDP/Megatron, vLLM/SGLang, multi-turn agent loop.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF): Ray, DeepSpeed, vLLM, hybrid
  engine, PPO/critic-free methods.
- [AReaL](https://github.com/inclusionAI/AReaL): large-scale asynchronous RL and
  staleness-aware execution.
- [Agent Lightning](https://github.com/microsoft/agent-lightning): decoupled
  agent execution/training and credit decomposition.
- [verl-agent](https://github.com/langfengQ/verl-agent): long-horizon multi-turn
  environments and GiGPO.
- [Mooncake](https://github.com/kvcache-ai/Mooncake): disaggregated serving and
  distributed KV-cache infrastructure relevant to long-context rollouts.

Pin a commit when studying code. The moving default branch is evidence only for
its current state, not for a paper's historical experiment.

## References

1. Guangming Sheng et al.,
   [“HybridFlow: A Flexible and Efficient RLHF Framework”](https://arxiv.org/abs/2409.19256),
   2024/2025.
2. Shangchun Fu et al.,
   [“AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning”](https://arxiv.org/abs/2505.24298),
   2025.
3. Kimi Team et al.,
   [“Kimi k1.5: Scaling Reinforcement Learning with LLMs”](https://arxiv.org/abs/2501.12599),
   2025.
4. Qin Zhu et al.,
   [“Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving”](https://arxiv.org/abs/2407.00079),
   2024.
5. Guibin Zhang et al.,
   [“The Landscape of Agentic Reinforcement Learning for LLMs”](https://arxiv.org/abs/2509.02547),
   TMLR, 2026.
