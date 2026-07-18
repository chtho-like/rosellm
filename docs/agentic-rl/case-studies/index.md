# Frontier-Lab Case Studies: Evidence Matrix

**Verified through:** 2026-07-19.

These studies reconstruct public training pipelines for
[DeepSeek](deepseek.md), [Zhipu AI / GLM](glm.md), and
[Moonshot AI / Kimi](kimi.md). They use official papers, model cards,
repositories, release notes, and first-party engineering posts.

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

## Latest public generation at the cutoff

| Lab | Latest public generation | Disclosed architecture | Context | Agentic post-training disclosure | Critical unknown |
|---|---|---|---:|---|---|
| DeepSeek | V4 Preview (2026-04-24) | Flash 284B/13B; Pro 1.6T/49B MoE | 1M | >10 GRPO specialists merged by full-vocabulary multi-teacher on-policy distillation; million-token fault-tolerant rollouts and DSec sandbox described | GPU count, duration, cost, exact task/reward volumes |
| Zhipu AI | GLM-5.2 (2026-06-16) | 753B hosted checkpoint; SAO paper rounds to 750B/40B active; earlier family convention 744B/40B | 1M | SAO is explicitly reported as deployed; critic PPO for variable compacted sub-traces; token-level advantages/loss; >10 experts merged with parallel on-policy distillation; online anti-hack guard | GLM-specific SAO/PPO hyperparameters and stage scope, task volume, hardware and cost |
| Moonshot AI | K3 (2026-07-17) | 2.8T; 896 experts, 16 selected; KDA + AttnRes + LatentMoE | 1M | product demonstrations disclose long agent runs, but K3 training/RL report was still pending | active params, data/tokens, RL algorithm, environments, hardware and cost |

K3 full weights were announced for 2026-07-27, after this verification cutoff.
No later claims are inferred from the announcement.

## Longitudinal pattern

All three lineages show a broadly similar capability factory, with important
implementation differences:

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
  consolidation with multi-teacher on-policy distillation.
- GLM-5 uses sequential reasoning → agentic → general RL plus cross-stage
  distillation; GLM-5.2 moves irregular compacted long-horizon traces from
  group-relative optimization to critic PPO.
- Moonshot k1.5 and K2 use a value-model-free sequence log-ratio objective;
  K2.5 adds token-ratio clipping, multimodal RL, and PARL for a learned
  multi-agent orchestrator.
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
| Agent task/environment counts | V3.2 discloses 85,267 tasks and >1,800 environments | GLM-5 discloses >10K SWE environments and thousands of terminal tasks; totals incomplete | K2 discloses >3K real and >20K synthetic tools; trajectory totals are broad |
| RL hyperparameters | Very detailed for revised R1; partial later | Detailed for GLM-5 reasoning RL; agent RL partial; 5.2 PPO partial | k1.5 objective detailed; many exact coefficients/counts absent |
| Training hardware/cost | V2/V3/R1 unusually detailed; V4 absent | GLM-130B detailed; later flagship totals absent | topology details for K2; total GPU/time/cost absent |
| Complete training code/data | No | No; Slime is a strong public framework, not a frozen full recipe | No |
| Production orchestration | Partially described | Partially described | Partially described |

“Open weights” should not be rewritten as “fully open source and reproducible.”
None of the three releases a complete raw corpus, all reward/prompts, full
training orchestration, optimizer state, and exact production stack for the
latest flagship model.

## Generation milestones

| Capability milestone | DeepSeek | GLM | Kimi |
|---|---|---|---|
| Early instruction/preference alignment | DeepSeek LLM: SFT + DPO | ChatGLM: SFT + RLHF at high level | early Kimi recipe undisclosed |
| Critic-free verifiable RL | DeepSeekMath introduces GRPO | GLM-Z1 / later expert RL; details mature in 4.5/5 | k1.5 value-model-free mirror-descent-like objective |
| Long reasoning production pipeline | R1/R1-Zero | GLM-Z1 and GLM-5 reasoning RL | k1.5 |
| Explicit production agentic RL | V3.1/V3.2 | GLM-4.5 | K2 |
| Multimodal agent RL | specialized branches; flagship disclosures vary | GLM-4.5V/4.6V | K2.5 |
| Trainable multi-agent orchestration | not comparably disclosed in reviewed flagship report | sub-agent workflows supported; exact orchestration training varies | K2.5 PARL/Agent Swarm |
| Million-token flagship | V4 | GLM-5.2 | K3 |
| Latest consolidation | multi-teacher OPD | parallel OPD + critic PPO for compact traces | K3 unknown; K2.5 unified RL/PARL documented |

## Reading the model tables correctly

### Parameter accounting

Always distinguish:

- total versus active parameters per token;
- main model versus MTP/prediction modules;
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

DeepSeek V3's approximately $5.576M figure uses a disclosed $2/H800-hour
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

The sources strongly support that frontier post-training is an integrated
operation involving:

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

The sources do **not** establish the complete live API workflow, real user-data
use beyond explicit disclosures, continuous online learning, hidden safety
rules, or model routing. Those remain **[U]**.

## Primary-source policy

Each vendor chapter cites the exact official paper, model card, repository, or
release post next to claims. Secondary media and unauthenticated benchmark
tables are excluded from the factual pipeline. See the repository-wide
[research standard](../../research-method.md) for contradiction and freshness
rules.
