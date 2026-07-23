# Frontier-Lab Case Studies: Evidence Matrix

**Verified through:** 2026-07-19.

These studies reconstruct publicly disclosed reinforcement-learning (RL)
training pipelines across the major lineages:

- generation-by-generation studies of [DeepSeek](deepseek.md),
  [Zhipu AI / General Language Model (GLM)](glm.md), and
  [Moonshot AI / Kimi](kimi.md);
- the preference, constitutional, reasoning, tool, and embodied-agent lineage
  across [OpenAI, Anthropic, and Google DeepMind](openai-anthropic-google.md);
- the open-weight and technical-report lineage across
  [Alibaba Qwen, Meta, and Mistral](qwen-meta-mistral.md); and
- industrial and reproducible components from
  [ByteDance Seed, NVIDIA, Microsoft, xAI, and the open community](open-industry-and-community.md).

The source base is limited to official papers, model and system cards,
repositories, release notes, and first-party engineering posts. A chapter may
explain a mechanism from first principles, but it attributes that mechanism to
a laboratory only when a primary source does.

For a cross-vendor quantitative view, start with the
[model training token ledger](../../model-training-token-ledger.md). For dated
DeepSeek checkpoint, specialist, and API events, use the
[DeepSeek release timeline](../../deepseek-release-timeline.md).

The word “actual” has a strict meaning here: **actually disclosed by a primary
source**. A technically plausible pipeline is not evidence that a company used
it. Public weights do not reveal data lineage, internal ablations, production
routing, system prompts, online updates, or unreleased safety layers.

## Evidence labels

- **[D] Disclosed:** stated directly in a primary source.
- **[C] Confirmed artifact:** visible in released weights, configuration, or
  source code.
- **[R] Reproduced:** rerun by this repository with recorded artifacts.
- **[I] Inference:** conclusion from disclosed facts and stated assumptions.
- **[U] Unknown:** absent, contradictory, ambiguous, or not verifiable.

Vendor-reported benchmark results are always identified as such; they are not
independent reproductions.

## Choose a reading route

| Goal | Start here | Then read | What you should be able to reconstruct |
|---|---|---|---|
| Learn the complete operational object | [Curriculum](../index.md) | [training pipeline](../training-pipeline.md), [systems](../systems.md), then this matrix | Every artifact from task creation through rollout, reward, update, evaluation, and deployment |
| Understand critic-free reasoning RL | [DeepSeekMath and R1](deepseek.md) | [Qwen, Group Sequence Policy Optimization (GSPO), and Soft Adaptive Policy Optimization (SAPO)](qwen-meta-mistral.md), then [open recipes](open-industry-and-community.md) | Group construction, advantage normalization, token aggregation, clipping, entropy, and data curriculum |
| Understand long-horizon actor-critic RL | [GLM-5 and GLM-5.2](glm.md) | [Single-Rollout Asynchronous Optimization (SAO) derivation and implementation](../derivations-and-code.md#20-single-rollout-asynchronous-optimization-derived), then [industrial asynchronous systems](open-industry-and-community.md) | Value targets, skipped observations, policy lag, importance correction, and streaming rollout scheduling |
| Understand preference and constitutional alignment | [OpenAI, Anthropic, and Google DeepMind](openai-anthropic-google.md) | [reward and evaluation vocabulary](../../glossary.md#reward-and-evaluation-vocabulary) | Comparison collection, reward-model fitting, policy optimization, AI feedback, rule rewards, and their failure modes |
| Understand data and environment operations | [Data and environments](../data-and-environments.md) | [DeepSeek](deepseek.md), [GLM](glm.md), [Kimi](kimi.md), and [open industry](open-industry-and-community.md) | Task provenance, reset semantics, executable verification, hidden splits, anti-hacking, and trajectory schemas |
| Build a small reproduction | [Derivations and code](../derivations-and-code.md) | [source-level lab](../source-lab.md), then one case chapter | A pinned, tested mechanism reproduction whose claims stay inside its evidence boundary |

## Latest public generation at the cutoff

| Lab | Latest public generation | Disclosed architecture | Context window | Agentic post-training disclosure | Critical unknown |
|---|---|---|---:|---|---|
| DeepSeek | V4 preview release (2026-04-24) | Flash 284B total / 13B active; Pro 1.6T / 49B mixture of experts (MoE) | 1M tokens | >10 Group Relative Policy Optimization (GRPO) specialists merged by full-vocabulary multi-teacher on-policy distillation; million-token fault-tolerant rollouts and the DeepSeek Elastic Compute (DSec) sandbox platform described | accelerator count, duration, cost, exact task/reward volumes |
| Zhipu AI | GLM-5.2 (2026-06-16) | 753B hosted checkpoint; SAO paper rounds to 750B total / 40B active; earlier family convention 744B / 40B | 1M tokens | SAO is explicitly reported as deployed; critic Proximal Policy Optimization (PPO) for variable compacted sub-traces; token-level advantages/loss; >10 domain-specialist models merged with parallel on-policy distillation; online anti-hack guard | GLM-specific SAO/PPO hyperparameters and stage scope, task volume, hardware and cost |
| Moonshot AI | K3 (2026-07-17) | 2.8T total; 896 experts, 16 selected per token; Kimi Delta Attention (KDA) + Attention Residuals (AttnRes) + LatentMoE | 1M tokens | product demonstrations disclose long agent runs, but K3 training/RL report was still pending | active parameters, data/tokens, RL algorithm, environments, hardware and cost |

K3 full weights were announced for 2026-07-27, after this verification cutoff.
No later claims are inferred from the announcement.

## Broader lineage coverage

“Most detailed public anchor” is deliberately different from “newest product.”
The newest release often exposes less training detail than an earlier paper.

| Lineage | Most detailed public anchor for learning | What the anchor exposes | Later-generation boundary |
|---|---|---|---|
| OpenAI | InstructGPT; WebGPT for browser actions | prompt/comparison collection, reward model, PPO, Kullback-Leibler (KL) control, pretraining auxiliary loss; browser action schema, behavioral cloning (BC), PPO experiment, and best-of-$N$ | o-series, Codex, deep research, computer use, and GPT-5.x disclose capability and mechanism direction, not an equivalent full recipe |
| Anthropic | Helpful and Harmless Assistant; Constitutional AI | iterative human preference collection; critique/revision Supervised Fine-Tuning (SFT); AI preference labels; PPO and KL; red-team flywheel | Claude 3.7 and later reveal reasoning-RL behavior and safety evidence, but not exact data, optimizer, reward weights, or compute |
| Google DeepMind | Playhouse; Sparrow; Reinforcement Learning from AI Feedback (RLAIF); Gemini 2.5 | embodied temporal-progress reward; search/rules/Advantage Actor-Critic (A2C); AI-label REINFORCE; heterogeneous verifier/generative/human-critic rewards and complex environments | Gemini 3.x publishes newer capability/model-card evidence without a replacement source-level RL recipe |
| Alibaba Qwen | Qwen2.5-Math; Qwen3; GSPO; SAPO | math data and verifier factory, staged reasoning/general RL, sequence-ratio and smooth-ratio objectives, coding environments | Qwen3-Coder/AgentWorld and later models add environment and agent scale; exact newest production mixtures remain partial |
| Meta | Llama 2 and Llama 3 reports | classic actor-critic Reinforcement Learning from Human Feedback (RLHF), iterative preference data, reward transforms, synthetic-data flywheel, tool annotations, and rejection sampling | Llama 4 and Muse releases disclose later capabilities and high-level online/multi-agent training, not comparable run manifests |
| Mistral | Magistral | modified GRPO, compositional reward, data filtering, exact small-model settings, asynchronous generation/verification, and negative results | Devstral and later unified products disclose real agent trajectories/capability but not an equally exact optimizer recipe |
| ByteDance Seed | Seed1.5-Thinking; Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO); ReTool; HybridFlow/veRL | value-based long-CoT RL, verifier and cold-start operations, streaming rollouts, open critic-free recipe, tool-interpreter RL, distributed dataflow | Seed2.0 capability disclosures do not establish that every earlier recipe or hyperparameter was retained |
| NVIDIA | Llama-Nemotron, AceReason, ProRL, Cascade 1/2, Nemotron 3 | architecture transformation, distillation, domain curricula, prolonged/cascaded/unified RL, multi-environment asynchrony, Multi-Domain On-Policy Distillation (MOPD), and failure accounting | each generation changes base model and stage graph; do not splice settings into one fictional “Nemotron recipe” |
| Microsoft | rStar2-Agent; Agent Lightning | exact math/Python action data and GRPO with Resample-on-Correct (GRPO-RoC) stages; runtime/learner separation and hierarchical trajectory credit | Agent Lightning is a framework and rStar2 is a recipe; neither proves the private training of every Microsoft model |
| xAI | Grok 4/4.1 Fast/4.5 first-party releases | scale direction, broad verifiable tasks, simulated tool environments, model graders, asynchronous hours-long software rollouts | architecture, data mixture, optimizer, reward coefficients, exact tasks, and full compute for the latest agent training remain unknown |
| Open community | Open-R1, OpenThoughts, AReaL, PRIME, Skywork-OR1, Search-R1, GiGPO, OpenRLHF, slime | inspectable data/configs, distillation, asynchronous correction, process reward, entropy control, retrieval agents, step credit, and programmable training stacks | components come from different checkpoints, revisions, tasks, and assumptions; combining them is a new experiment, not an existing reproduced frontier run |

## Longitudinal pattern

The DeepSeek, GLM, and Kimi lineages show a broadly similar capability factory,
with important implementation differences:

```text
curated pretraining
  -> domain and long-context mid-training
  -> broad SFT / cold start
  -> specialist reasoning, code, search, tool, and preference training
  -> executable or generated agent environments
  -> online RL with rules, tests, learned/generative judges, and anti-hacking
  -> distillation/consolidation into one general checkpoint
  -> deployment-compatible low precision and long-context serving
```

This diagram is a comparative abstraction. The exact sequence differs:

- DeepSeek V3.2 consolidated specialists with mixed GRPO; V4 replaces the final
  consolidation with multi-teacher On-Policy Distillation (OPD).
- GLM-5 uses sequential reasoning → agentic → general RL plus cross-stage
  distillation; GLM-5.2 moves irregular compacted long-horizon traces from
  group-relative optimization to critic PPO.
- Moonshot k1.5 and K2 use a value-model-free sequence log-ratio objective;
  K2.5 adds token-ratio clipping, multimodal RL, and Parallel-Agent
  Reinforcement Learning (PARL) for a learned multi-agent orchestrator.
- K3's training sequence was not public at the cutoff, so K2.5 is the latest
  deeply documented Moonshot post-training recipe.

## Disclosure comparison

| Item | DeepSeek | GLM | Kimi |
|---|---|---|---|
| Major open weights | Extensive | Extensive for recent open series | Extensive for K2/K2.5 and research models; K3 pending at cutoff |
| Architecture reports/configs | Detailed for many generations | Detailed since GLM-4.5/GLM-5 | Detailed for Moonlight/K2/K2.5; K3 high-level only |
| Pretraining tokens | Often disclosed | Disclosed for major reports | Disclosed for Moonlight/K2/K2.5 continuation |
| Exact source mixture | Usually unknown | Usually unknown | Usually unknown |
| Optimizer/schedule | Detailed in flagship reports | Detailed in flagship reports | Detailed for Moonlight/K2; partial elsewhere |
| Agent task/environment counts | V3.2 discloses 85,267 tasks and >1,800 environments | GLM-5 discloses >10K Software Engineering (SWE) environments and thousands of terminal tasks; totals incomplete | K2 discloses >3K real and >20K synthetic tools; trajectory totals are broad |
| RL hyperparameters | Very detailed for revised R1; partial later | Detailed for GLM-5 reasoning RL; agent RL partial; 5.2 PPO partial | k1.5 objective detailed; many exact coefficients/counts absent |
| Training hardware/cost | V2/V3/R1 unusually detailed; V4 absent | GLM-130B detailed; later flagship totals absent | topology details for K2; total accelerator count/time/cost absent |
| Complete training code/data | No | No; Slime is a strong public framework, not a frozen full recipe | No |
| Production orchestration | Partially described | Partially described | Partially described |

“Open weights” should not be rewritten as “fully open source and reproducible.”
None of the three releases a complete raw corpus, all reward/prompts, full
training orchestration, optimizer state, and exact production stack for the
latest flagship model.

## Generation milestones

| Capability milestone | DeepSeek | GLM | Kimi |
|---|---|---|---|
| Early instruction/preference alignment | DeepSeek Large Language Model (LLM): SFT + Direct Preference Optimization (DPO) | ChatGLM: SFT + RLHF at high level | early Kimi recipe undisclosed |
| Critic-free verifiable RL | DeepSeekMath introduces GRPO | GLM-Z1 / later expert RL; details mature in 4.5/5 | k1.5 value-model-free mirror-descent-like objective |
| Long reasoning production pipeline | R1/R1-Zero | GLM-Z1 and GLM-5 reasoning RL | k1.5 |
| Explicit production agentic RL | V3.1/V3.2 | GLM-4.5 | K2 |
| Multimodal agent RL | specialized branches; flagship disclosures vary | GLM-4.5V/4.6V | K2.5 |
| Trainable multi-agent orchestration | not comparably disclosed in reviewed flagship report | sub-agent workflows supported; exact orchestration training varies | K2.5 PARL/Agent Swarm |
| Million-token flagship | V4 | GLM-5.2 | K3 |
| Latest consolidation | multi-teacher OPD | parallel OPD + critic PPO for compact traces | K3 unknown; K2.5 unified RL/PARL documented |

## Optimization and consolidation mechanisms at a glance

The names below do not denote interchangeable losses. They change the sampling
unit, baseline, trust mechanism, credit unit, or even whether the operation is
reinforcement learning at all.

| Mechanism | Experience and baseline | Update control | Operational reason to use it | Principal implementation hazard | Case evidence |
|---|---|---|---|---|---|
| Classic RLHF PPO | current-policy responses; learned reward and value models; usually a fixed reference policy | token likelihood-ratio clipping, KL shaping/penalty, and value clipping | optimize subjective preferences while controlling drift | reward/value calibration, policy/reference token mismatch, stale rollouts, and large memory footprint | InstructGPT and Anthropic Helpful and Harmless / Constitutional AI in [OpenAI/Anthropic/Google](openai-anthropic-google.md) |
| GRPO | $G$ responses for one prompt; group-relative return replaces a learned critic | PPO-like token ratio, with implementation-dependent normalization and KL | verifiable reasoning where many samples can be scored cheaply | uniform-reward groups, response-length bias, group-variance bias, and $G$-fold rollout cost | DeepSeekMath/R1 in [DeepSeek](deepseek.md) and Qwen-family use in [Qwen/Meta/Mistral](qwen-meta-mistral.md) |
| DAPO | critic-free response groups plus dynamic resampling of informative groups | asymmetric low/high clipping, token-global aggregation, and overlong shaping | keep learning after many prompts become uniformly easy or hard | hidden resampling cost, changed prompt distribution, length incentives, and denominator changes | ByteDance recipe in [open industry](open-industry-and-community.md) |
| GSPO / GSPO-token | response groups; scalar or customized token advantages | one length-normalized sequence ratio; GSPO-token uses a stop-gradient carrier to preserve chosen token credit | align the trust unit with sequence-level reward while retaining an optional token-credit path | geometric-mean ratio is not the raw sequence-density ratio; wrong `detach` placement silently changes gradients | [Qwen lineage](qwen-meta-mistral.md) and [derivation](../derivations-and-code.md#17-derive-gspo-and-the-gspo-token-gradient-carrier) |
| SAPO | normally group-relative advantages; no value critic in the published Qwen experiments | sigmoid surrogate with separate temperatures for positive and negative advantage | replace a hard clip boundary with a smooth gradient gate | confusing the gate $4p(1-p)$ with the full log-policy gradient coefficient $4p(1-p)\,r\,\hat A$ | [Qwen lineage](qwen-meta-mistral.md) and [derivation](../derivations-and-code.md#18-derive-sapos-smooth-trust-gate) |
| Kimi k1.5/K2 objective | $K$ old-policy responses and group-centered outcomes; no token/state value model | squared sequence log-ratio toward an analytically reward-tilted target | avoid premature local credit judgments in self-correcting long reasoning | a whole-sequence objective still needs exact length, truncation, old-policy, and minibatch semantics | [Kimi](kimi.md#45-value-model-free-policy-optimization-d) |
| SAO | one completed trajectory at a time; learned critic; skip-observation Generalized Advantage Estimation (GAE) | direct rollout-to-current-policy ratio with strict two-sided masking | train on highly variable, long agent rollouts without waiting to form a group | masked data loss, critic/policy clock mismatch, corrupted observation masks, and unbounded policy lag | GLM-5.2 evidence in [GLM](glm.md) and [source lab](../source-lab.md#10-sao-as-a-source-level-experiment) |
| Decoupled asynchronous PPO | queued behavior-policy samples, a recent proximal snapshot, and the current learner | outer behavior-to-proximal correction plus inner proximal PPO ratio | overlap rollout and training while distinguishing two sources of policy drift | version metadata loss, high-variance importance weights, and non-replayable environment state | AReaL in [open industry](open-industry-and-community.md) |
| PARL | orchestrator trajectories; subagents frozen and excluded from the orchestrator gradient | task reward plus annealed parallelism/finish shaping | learn whether, what, and how to delegate under a wall-time objective | serial collapse, meaningless spawning, context leakage, and confusing total work with critical-path time | K2.5 Agent Swarm in [Kimi](kimi.md#116-parl-agent-swarm-d) |
| OPD / MOPD | states sampled by the student; teacher token distributions on those states | supervised distribution matching, not a policy-gradient advantage | consolidate specialist capabilities without training only on teacher-state data | teacher/domain routing, tokenizer equality, state coverage, and capability interference | DeepSeek V4, GLM-5, and NVIDIA in the corresponding case chapters |

Read the executable PPO, GRPO, SAO, GSPO, GSPO-token, and SAPO reference
implementations in the [source-level lab](../source-lab.md). A forward value
matching a paper equation is not enough: tests also check gradients, masks,
normalization denominators, and invariances.

## The production artifact chain

A real training operation is easier to understand as a chain of versioned
artifacts than as one optimizer name. At every arrow, record the producer,
schema, version, unit, filters, and rejected count.

| Stage | Required artifact | Minimum fields for an auditable run | Typical silent failure |
|---|---|---|---|
| 1. Task acquisition | immutable raw-task record | source/license, acquisition time, content hash, language/domain, sensitive-data flags | benchmark or evaluation artifacts leak into training |
| 2. Task normalization | canonical task contract | initial-state constructor, permitted tools, success predicate, budgets, termination causes | a formatter changes the problem or makes the answer visible |
| 3. Split construction | provenance-aware split manifest | semantic cluster, repository/problem family, timestamp, generator lineage, split hash | near-duplicates or generated siblings cross train/test boundaries |
| 4. Environment build | pinned executable image | image digest, dependencies, network/credential policy, reset/replay method, hidden tests | the reward depends on mutable packages, web pages, or clock state |
| 5. Cold start | demonstration/SFT trajectory | observation/action roles, tokenizer and template, policy-action mask, author/teacher, validation result | tool observations are trained as if the policy emitted them |
| 6. Rollout request | sampling envelope | policy/checkpoint hash, temperature, top-$p$/top-$k$, seed, maximum tokens/actions/time, scaffold version | “same model” evaluations use different hidden inference budgets |
| 7. Event trace | append-only trajectory | exact token identifiers (IDs) and log-probabilities, action parse, observation bytes/hash, timestamps, policy version per segment | retokenization or transcript reconstruction changes the sampled action |
| 8. Reward vector | unscalarized grader outputs | grader/verifier version, raw component values, confidence, failure code, hidden-test access | one scalar hides which proxy was exploited or unavailable |
| 9. Admission/filtering | attempt census | accepted, rejected, truncated, timed out, duplicate, invalid, uniformly rewarded, and resampled counts | the learner sees a different task distribution than the reported prompt set |
| 10. Credit assignment | return/advantage tensor | reward-to-action mapping, bootstrap boundaries, $\gamma$, $\lambda$, group baseline, normalization axes | delayed terminal reward is copied across observations or padding tokens |
| 11. Policy update | immutable optimizer batch | behavior/reference/proximal/current log-probabilities, masks, ratios, clips, KL, entropy, denominators | an implementation with the same algorithm name optimizes a different estimator |
| 12. Weight publication | checkpoint lineage event | parent hash, optimizer state, data window, update index, numerical health, rollout-consumer acknowledgement | rollout workers silently mix incompatible policy versions |
| 13. Evaluation | frozen protocol bundle | task and environment versions, scaffold, sampling budget, trials/seeds, cost, confidence interval | a model gain is actually a router, tool, judge, or test-time-compute gain |
| 14. Deployment gate | signed release decision | safety/quality thresholds, regression owners, authority limits, canary, rollback hash, incident telemetry | offline success is mistaken for permission to take irreversible actions |

The chain also explains why no reviewed frontier release is fully reproducible:
the public sources disclose different subsets of these artifacts. An equation,
weights, or a model card can be valuable without being a complete run manifest.

## Reading the model tables correctly

### Parameter accounting

Always distinguish:

- total versus active parameters per token;
- main model versus Multi-Token Prediction (MTP) modules;
- inclusion/exclusion of embeddings/output head;
- stored checkpoint tensors versus report convention.

Examples:

- DeepSeek-V3 is reported as 671B/37B, while hosted files can total 685B when
  MTP modules are included.
- GLM-4.5 report counts can exclude embeddings/output and differ from hosting
  totals.
- K3 discloses 2.8T total and selected experts but not active parameters; simple
  `total × selected / experts` arithmetic would omit dense/shared components.

### Token accounting

“Trained on 15.5T tokens” means processed token exposure, not 15.5T unique
tokens. Continued pretraining, repeated high-quality data, synthetic rewrites,
and cooldown phases may reuse underlying documents. Do not sum stage totals
unless the report defines them as additive.

### Cost accounting

DeepSeek V3's approximately \$5.576M figure uses a disclosed \$2/H800-hour
rental-equivalent and covers listed runs. It does not establish full company
development cost. The absence of a Kimi/GLM cost figure does not justify
estimating it from model size without hardware, utilization, restarts, and
training-token details.

### Benchmark accounting

Long-reasoning and agent results can use:

- 64K–100K reasoning-token caps;
- hundreds or thousands of tool calls;
- multiple parallel rollouts and aggregation;
- different context-overflow policies;
- model-specific scaffolds and sandboxes;
- vendor or learned judges.

Case-study tables retain these settings near the score. Cross-vendor numerical
rankings are avoided when protocols are not aligned.

## What the public evidence supports about “real operations”

Across the reviewed cases, the sources collectively support frontier
post-training operations that can involve:

1. task and environment mining/generation;
2. tool/schema/user/agent synthesis;
3. high-concurrency sandbox execution;
4. exact token/log-probability transport from rollout to trainer;
5. executable rewards where possible and learned/generative judges elsewhere;
6. online anti-hacking and hidden validation;
7. specialist models/teachers and on-policy distillation;
8. asynchronous collection, versioned weight sync, packing, and fault recovery;
9. repeated SFT/distillation/RL/curriculum loops rather than one final RL job;
10. deployment-compatible low precision, sparse attention, and long context.

The sources do **not** establish the complete live Application Programming
Interface (API) workflow, real user-data use beyond explicit disclosures,
continuous online learning, hidden safety rules, or model routing. Those remain
**[U]**.

## Primary-source policy

Each vendor chapter cites the exact official paper, model card, repository, or
release post next to claims. Secondary media and unauthenticated benchmark
tables are excluded from the factual pipeline. See the repository-wide
[research standard](../../research-method.md) for contradiction and freshness
rules.
