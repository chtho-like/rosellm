# Agentic Reinforcement Learning: Zero-to-Researcher Curriculum

**Agentic Reinforcement Learning (Agentic RL)** trains a language-model policy through repeated
interaction with a stateful environment. The defining difficulty is not merely
that a response contains several reasoning tokens. The policy takes actions,
receives new observations, changes the world, and must assign delayed outcomes
back to earlier decisions.

The clean formal boundary is a useful starting point:

- ordinary one-response **Large Language Model (LLM) Reinforcement Learning
  (RL)** can often be approximated as a contextual bandit or a degenerate
  one-step **Markov Decision Process (MDP)**;
- agentic RL is normally a finite-horizon **Partially Observable Markov Decision
  Process (POMDP)**: the complete environment state is hidden, so the policy
  acts from observations and history over more than one consequential step.

This distinction follows the formalization in Zhang et al.,
[*The Landscape of Agentic Reinforcement Learning for LLMs*](https://arxiv.org/abs/2509.02547)
(*Transactions on Machine Learning Research*, TMLR 2026, Sections 2–3). It is a
modeling boundary, not a naming rule: a reasoning trace can itself contain
actions, and a single Application Programming Interface (API) response can
drive multiple hidden environment transitions. Always identify the actual
policy and environment boundary.

## The whole object in one equation

Model an episode as a POMDP

\[
\mathcal{M} = \langle \mathcal{S}, \mathcal{O}, \mathcal{A}, P, O, R,
\rho_0, \gamma, T \rangle.
\]

At time \(t\), the environment has latent state \(s_t\). The agent receives an
observation \(o_t \sim O(\cdot\mid s_t)\), constructs a history or learned memory
\(h_t\), and samples an action

\[
a_t \sim \pi_\theta(\cdot\mid h_t), \qquad
s_{t+1} \sim P(\cdot\mid s_t,a_t).
\]

The trajectory is

\[
\tau=(s_0,o_0,h_0,a_0,r_0,\ldots,s_T,o_T),
\]

and the basic objective is

\[
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}
\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].
\]

For an autoregressive LLM, each semantic action \(a_t\) is itself a token
sequence. Agentic RL therefore has at least three clocks:

1. **token time** inside an LLM generation;
2. **decision time** between environment interactions; and
3. **episode time** from task initialization to termination.

Many implementation bugs are really clock-mismatch bugs: an episode-level
reward is copied to every token, a tool observation is accidentally trained as
if the policy generated it, or an importance ratio spans tokens sampled by
different policy versions.

## Knowledge hierarchy

The hierarchy below is both a curriculum and a debugging map. A failure at a
higher level is often caused by an unchecked assumption lower in the tree.

### Layer A — Prerequisites

1. **Probability and statistics**
   - conditional distributions, expectation, variance, covariance;
   - log-likelihood, entropy, cross-entropy, Kullback–Leibler (KL) divergence;
   - Monte Carlo estimation, control variates, importance sampling;
   - confidence intervals, hypothesis tests, and multiple comparisons.
2. **Optimization**
   - stochastic gradients, momentum, AdamW, schedules, clipping;
   - constrained and trust-region optimization;
   - numerical stability, mixed precision, and distributed reductions.
3. **Deep learning and transformers**
   - autoregressive factorization and teacher forcing;
   - attention, residual blocks, normalization, mixture-of-experts (MoE)
     routing;
   - tokenization, chat templates, masking, key-value (KV) caches, and sampling.
4. **Software and systems**
   - PyTorch autograd and distributed training;
   - asynchronous programming, queues, remote procedure calls (RPCs), retries,
     and idempotency;
   - containers, sandboxes, observability, and reproducible data pipelines.

### Layer B — Classical sequential decision making

1. Markov chains, MDPs, POMDPs, belief state, and history state.
2. Return, value \(V^\pi\), action value \(Q^\pi\), and advantage \(A^\pi\).
3. Bellman expectation and optimality equations.
4. Dynamic programming, Monte Carlo, temporal difference learning, and
   eligibility traces.
5. On-policy versus off-policy learning; exploration versus exploitation.
6. Policy gradients, actor–critic methods, generalized advantage estimation,
   trust regions, and Proximal Policy Optimization (PPO).
7. Offline RL, imitation learning, inverse RL, hierarchical RL, model-based RL,
   multi-agent RL, and constrained/safe RL.

The canonical foundation is Sutton and Barto,
[*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html)
(2nd ed., 2018). PPO originates in Schulman et al.,
[*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347)
(2017).

### Layer C — LLM post-training before agents

1. **Supervised Fine-Tuning (SFT):** demonstrations, chat schemas, loss masks,
   rejection sampling, and distillation.
2. **Preference learning:** comparison collection, Bradley–Terry reward models,
   annotator noise, calibration, and uncertainty.
3. **Reinforcement Learning from Human Feedback (RLHF):**
   policy/value/reference/reward models, KL regularization, PPO, and online
   sampling. Ouyang et al.,
   [*Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155)
   (2022) is the standard end-to-end reference.
4. **Offline preference optimization:** Direct Preference Optimization (DPO)
   and related objectives, including what their fixed-dataset assumptions
   exclude.
5. **RL with verifiable rewards (RLVR):** executable or rule-based correctness
   for math, code, formal proof, games, and structured outputs.
6. **Reasoning RL:** long sampled solutions, group-relative baselines,
   process/outcome rewards, entropy dynamics, and distillation. DeepSeekMath
   introduced Group Relative Policy Optimization (GRPO) in its published recipe
   ([Shao et al., 2024, Section 3](https://arxiv.org/abs/2402.03300)); DeepSeek-R1
   demonstrated a large-scale reasoning-RL pipeline
   ([Guo et al., 2025](https://arxiv.org/abs/2501.12948)).

### Layer D — The agent interface

1. **Observation design**
   - user messages, tool results, screen pixels, files, database state;
   - partial observability, truncation, summarization, and stale observations;
   - untrusted content and prompt-injection boundaries.
2. **Action design**
   - free text, structured calls, code, Graphical User Interface (GUI) actions,
     physical controls;
   - grammar constraints, parameter validation, authorization, and abstention;
   - macro-actions versus primitive actions.
3. **State and memory**
   - transcript-as-state, belief state, scratchpads, episodic/semantic memory;
   - retrieval, compression, write policies, forgetting, and privacy.
4. **Transition dynamics**
   - deterministic simulators, stochastic users, live web/services, and robots;
   - latency, timeouts, side effects, hidden state, and non-stationarity.
5. **Termination**
   - success, failure, budget exhaustion, unsafe action, deadlock, and timeout;
   - who may declare completion and how it is independently verified.

### Layer E — Agent capabilities learned by RL

1. reasoning and inference;
2. planning, replanning, and subgoal selection;
3. tool selection, argument construction, and result integration;
4. information search and evidence synthesis;
5. working, episodic, and long-term memory management;
6. reflection, error recovery, and uncertainty-aware verification;
7. perception and grounding across text, image, audio, video, and action;
8. long-horizon execution and budget allocation;
9. communication with users and other agents;
10. self-improvement through curriculum, self-play, and environment generation.

These capabilities are not independent modules. For example, tool choice alters
future observations, memory alters the effective state, and planning changes
the distribution of credit-assignment distances.

### Layer F — Tasks and environments

1. math, code execution, unit tests, and formal theorem proving;
2. search, browsing, research, retrieval, and question answering;
3. operating systems, terminals, software engineering, and cybersecurity;
4. APIs, databases, enterprise workflows, and customer-support simulations;
5. GUI, mobile, desktop, and web navigation;
6. games, embodied control, robotics, and vision-language-action tasks;
7. science, experimentation, and laboratory automation;
8. multi-agent cooperation, competition, negotiation, and markets.

For each environment learn: reset semantics, observation/action schema, hidden
state, stochasticity, reward, validator, horizon, cost model, concurrency,
sandbox, versioning, train/test split, contamination risk, and failure policy.

### Layer G — Experience and data engine

1. task sourcing and rights/provenance;
2. expert demonstrations and behavioral cloning;
3. synthetic task and trajectory generation;
4. teacher distillation and best-of-N/rejection sampling;
5. on-policy rollout and policy-version tracking;
6. replay, off-policy data, importance weighting, and staleness;
7. failure mining, adversarial generation, and automatic curricula;
8. difficulty estimation and dynamic sampling;
9. deduplication and contamination barriers;
10. trajectory schema, token preservation, compression, and lineage;
11. human correction, preference, process, and outcome labels;
12. quality control, inter-annotator agreement, and audit sampling.

### Layer H — Reward and feedback

1. binary exact-match and executable validators;
2. graded task progress and environment-native scores;
3. process rewards at token, span, turn, or subgoal level;
4. human preferences and learned reward models;
5. LLM judges with calibration and adversarial controls;
6. safety, policy compliance, permission, and reversibility constraints;
7. cost, latency, token, tool-call, and resource penalties;
8. novelty, diversity, exploration, and information-gain bonuses;
9. multi-objective scalarization, lexicographic constraints, and Pareto tradeoffs;
10. uncertainty, ensembles, reward hacking, and causal reward validation.

### Layer I — Optimization and credit assignment

1. sequence-level REINFORCE and control variates;
2. actor–critic, Generalized Advantage Estimation (GAE), PPO, and value-model
   training;
3. leave-one-out and group-relative baselines: REINFORCE Leave-One-Out (RLOO),
   GRPO, and variants;
4. KL regularization, reference policies, clipping, and trust regions;
5. token-, turn-, segment-, subgoal-, and trajectory-level advantages;
6. sparse delayed reward, eligibility, return decomposition, and value targets;
7. off-policy correction and truncated importance sampling;
8. replay and asynchronous-policy lag;
9. hierarchical policies and option-level credit;
10. multi-agent centralized training/decentralized execution;
11. constrained optimization for safety and budgets;
12. entropy control, mode collapse, length bias, and gradient starvation.

### Layer J — Training systems

1. policy, reference, critic, reward, verifier, and judge placement;
2. colocated versus disaggregated generation/training;
3. synchronous, partially asynchronous, and fully asynchronous loops;
4. rollout inference with vLLM/SGLang-like engines;
5. exact sampled-token retention and tokenizer/template consistency;
6. weight broadcast, checkpoint conversion, and policy versioning;
7. data/tensor/pipeline/context/expert/sequence parallelism;
8. variable-length packing, masks, loss normalization, and load balance;
9. environment RPC, backpressure, straggler mitigation, and fault recovery;
10. deterministic debugging, trace storage, metrics, and cost accounting;
11. sandbox isolation, secrets, network policy, and side-effect control.

The requirement to retain exact sampled tokens is not cosmetic. Text can be a
non-invertible representation of a token stream; decoding and re-encoding may
change boundaries and therefore log-probabilities. The veRL agentic-RL
documentation explicitly uses a token-based generation API for this reason
([veRL, “Agentic RL Training”](https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst)).

### Layer K — Evaluation and science

1. static capability, interactive success, and end-to-end utility;
2. pass@1/pass@k, success-weighted cost, latency, and action count;
3. hidden tasks, dynamic environments, leakage, and benchmark contamination;
4. stochastic trials, paired seeds, confidence intervals, and power;
5. judge agreement, human validation, and failure taxonomies;
6. robustness to tool errors, prompt injection, state perturbation, and drift;
7. generalization across tasks, tools, horizons, languages, and environments;
8. ablations that separate data, inference compute, reward, and optimization;
9. safety cases, red teams, incident analysis, and deployment gates;
10. reproducibility, artifact lineage, and honest negative results.

### Layer L — Frontier-lab reconstruction

For every model family, reconstruct the same stages:

1. base-model architecture and tokenizer;
2. pretraining data and optimization;
3. context/domain mid-training;
4. SFT and cold-start data;
5. reward/verifier/judge construction;
6. reasoning and agentic RL;
7. distillation and model-family transfer;
8. evaluation protocol;
9. deployment-relevant disclosures; and
10. unknowns and common unsupported claims.

Use the shared [evidence matrix](case-studies/index.md) before comparing scores
or recipes. The case-study set includes:

- generation-by-generation reconstructions of
  [DeepSeek](case-studies/deepseek.md),
  [Zhipu AI's General Language Model (GLM)](case-studies/glm.md), and
  [Moonshot Kimi](case-studies/kimi.md);
- preference-to-agent lineages for
  [OpenAI, Anthropic, and Google DeepMind](case-studies/openai-anthropic-google.md);
- open-weight and report-backed lineages for
  [Alibaba Qwen, Meta, and Mistral](case-studies/qwen-meta-mistral.md); and
- algorithm, system, and reproduction studies for
  [ByteDance Seed, NVIDIA, Microsoft, xAI, and the open community](case-studies/open-industry-and-community.md).

Read each lineage in time order. For every generation, write a five-column
ledger: **disclosed fact**, **confirmed artifact**, **reproduction result**,
**bounded inference**, and **unknown**. Then map every supported training stage
onto the task, environment, trajectory, reward, optimizer, evaluation, and
deployment artifacts in the evidence matrix. This prevents a newer benchmark
table from being mistaken for a newly disclosed training recipe.

## Recommended learning sequence

| Phase | Read | Build | Exit criterion |
|---|---|---|---|
| 1 | [Terminology](terminology.md) and classical RL prerequisites | tabular bandit and MDP | Explain why a multi-turn tool task is not a contextual bandit. |
| 2 | [Mathematics](mathematical-foundations.md) | REINFORCE on a toy sequence task | Derive the estimator and measure its variance. |
| 3 | [Algorithms](algorithms.md) | PPO, RLOO, and GRPO on identical samples | Explain every normalization, mask, and ratio. |
| 4 | [Data and environments](data-and-environments.md) | deterministic tool sandbox | Reset/replay a trajectory bit-for-bit and prove split isolation. |
| 5 | [Training pipeline](training-pipeline.md) | multi-turn rollout collector | Preserve exact tokens, tool boundaries, rewards, and policy-version identifiers. |
| 6 | [Systems](systems.md) | asynchronous rollout/trainer prototype | Quantify Graphics Processing Unit (GPU) idle time, queue delay, and policy staleness. |
| 7 | [Evaluation and safety](evaluation-and-safety.md) | hidden stochastic test suite | Report uncertainty, cost, and a failure taxonomy. |
| 8 | [Case studies](case-studies/index.md) | reconstruct one model generation and reproduce one disclosed mechanism at small scale | Separate disclosed, confirmed-artifact, reproduced, inferred, and unknown claims; preserve the exact benchmark and inference protocol. |
| 9 | [Source lab](source-lab.md) | extend `roserlhf` | Pass numerical, invariance, and end-to-end tests. |

## What “source-level understanding” means

You have reached source-level understanding only if you can trace one sampled
token through all of these representations:

1. environment observation bytes;
2. chat template and tokenizer identifiers (IDs);
3. packed rollout tensors and attention masks;
4. inference policy version and sampled log-probability;
5. tool/action parser and environment transition;
6. reward components and termination record;
7. return/advantage estimator;
8. importance ratio and clipped objective;
9. per-token loss mask and normalization denominator;
10. distributed gradient reduction;
11. optimizer update and new checkpoint;
12. weight synchronization back to rollout workers.

If any step is described only as “the framework handles it,” that step remains
part of the curriculum.

## Non-negotiable distinctions

- **Reasoning RL is not automatically agentic RL.** A long chain of thought may
  still be a single environment action.
- **Tool-use SFT is not tool-use RL.** Demonstrations teach imitation; RL needs
  policy-dependent experience and reward-linked updates.
- **A verifier is not necessarily a reward model.** A compiler or exact checker
  has different error modes from a learned preference predictor.
- **Outcome reward does not reveal causal credit.** Copying a terminal score to
  every token is an estimator choice, not an explanation of which action helped.
- **Public training reports are not production runbooks.** Undisclosed data,
  system prompts, routing, tools, safety layers, and online adaptation remain
  unknown unless a primary source says otherwise.
- **Benchmark improvement is not general agency.** It may reflect data overlap,
  more inference samples, environment-specific reward shaping, or evaluator
  sensitivity.

## Core references

1. Richard S. Sutton and Andrew G. Barto,
   [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html),
   2nd ed., 2018.
2. Ronald J. Williams,
   [“Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning”](https://doi.org/10.1007/BF00992696),
   *Machine Learning* 8, 1992.
3. John Schulman et al.,
   [“Proximal Policy Optimization Algorithms”](https://arxiv.org/abs/1707.06347),
   2017.
4. Long Ouyang et al.,
   [“Training language models to follow instructions with human feedback”](https://arxiv.org/abs/2203.02155),
   2022.
5. Zhihong Shao et al.,
   [“DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models”](https://arxiv.org/abs/2402.03300),
   2024.
6. Daya Guo et al.,
   [“DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning”](https://arxiv.org/abs/2501.12948),
   2025.
7. Guibin Zhang et al.,
   [“The Landscape of Agentic Reinforcement Learning for LLMs”](https://arxiv.org/abs/2509.02547),
   TMLR, 2026.

The [annotated bibliography](bibliography.md) expands this list by prerequisite,
algorithm, capability, environment, system, evaluation, and model family.
