# Long-Horizon Task Capability: Training, Runtime, and Vendor Evidence

**Verified through:** 2026-07-23. **Evidence scope:** primary papers, technical
reports, system/model cards, official repositories, and first-party engineering
posts. The chapter uses the repository's
[evidence classes](../research-method.md): **[D] disclosed**, **[C] confirmed
artifact**, **[R] reproduced**, **[I] inferred**, and **[U] unknown**.

Long-horizon task capability is not one skill, one context-window number, or one
reinforcement-learning algorithm. It is the product of four interacting
systems:

1. a checkpoint with strong reasoning, coding, instruction-following, and
   tool-protocol priors;
2. post-training on complete interaction trajectories in executable,
   stateful environments;
3. a runtime that manages tools, context, memory, verification, budgets, and
   sometimes multiple agents; and
4. an evaluation and data flywheel that converts failures into new tasks,
   verifiers, demonstrations, and online rollouts.

The shortest accurate industry answer is therefore:

> Frontier laboratories make agents work longer by training on the same
> closed-loop structure used at deployment, rewarding independently verified
> outcomes, preserving temporal credit and exact behavior-policy information,
> and surrounding the checkpoint with persistent state, context compaction,
> tools, search, and verification. Longer context and more inference compute
> help, but neither substitutes for environment-grounded post-training.

## 1. Three meanings of “long”

These axes are related but not interchangeable.

| Quantity | What it measures | What it does not prove |
|---|---|---|
| **Context length** | tokens accepted by one model call | that information is recalled reliably, that the model can plan, or that it can act for that many steps |
| **Interaction horizon** | model decisions, tool calls, environment transitions, or compacted segments in one episode | that the task is difficult or economically valuable |
| **Task-completion time horizon** | task difficulty expressed as the time a skilled human would need at a specified success probability | the wall-clock time the agent runs, broad job automation, or performance outside the evaluated domain |

METR's [Task-Completion Time Horizons](https://metr.org/time-horizons/)
defines a 50%-time horizon as the human-task duration at which a fitted agent is
predicted to succeed half the time. Its 2026 suite is dominated by
well-specified software-engineering, machine-learning, and cybersecurity tasks;
the organization explicitly warns that the metric is not “how long the AI can
run” and does not cover every intellectual task **[D]**.

The distinction matters operationally:

- a model may accept one million tokens and still lose the active requirement
  in irrelevant history;
- a loop may run for days while repeating an unproductive action;
- a short trajectory may solve a task that takes a human hours because the
  model types and searches faster;
- a long reasoning trace may remain one environment action and therefore is
  [reasoning RL, not necessarily Agentic RL](terminology.md#reasoning-rl-versus-agentic-rl);
  and
- a production agent can outlive one context window by storing state outside
  the model, compacting history, and starting new sessions.

## 2. The checkpoint, harness, and product are different objects

A reported agent score can change without changing model weights.

```text
checkpoint
  + system/developer instructions
  + tool schemas and action grammar
  + context selection and compaction
  + filesystem/database/session state
  + retry, stopping, and permission policy
  + search, verifier, and subagent budget
  + environment version
  = evaluated agent system
```

The **agent scaffold** constructs prompts, selects tools and context, applies
retries, and manages stopping and persistence. The **agent harness** is the
executable loop that connects the model to state and effects. A hosted product
may add routing, proprietary tools, caching, safety systems, user interaction,
and continuously changing service logic.

This yields a non-negotiable evaluation rule:

$$
\Delta_{\text{observed}}
=
\Delta_{\text{weights}}
+\Delta_{\text{harness}}
+\Delta_{\text{tools}}
+\Delta_{\text{budget}}
+\Delta_{\text{environment}}
+\text{interactions}.
$$

The equation is an accounting identity **[I]**, not an assumption that the
effects are additive. Controlled ablations must hold all but one term fixed.
See the [evaluation chapter](evaluation-and-safety.md#8-ablating-what-improved).

## 3. Why long horizons are intrinsically hard

Represent an agent episode as a Partially Observable Markov Decision Process
(POMDP). At turn $t$, the policy sees history or memory $h_t$, chooses an
action, and receives a new observation:

$$
a_t\sim\pi_\theta(\cdot\mid h_t),\qquad
s_{t+1}\sim P(\cdot\mid s_t,a_t),\qquad
o_{t+1}\sim O(\cdot\mid s_{t+1}).
$$

For a verifier $V$ and complete trajectory $\tau$, the system objective is

$$
J(\theta)
=
\mathbb E_{\tau\sim p_{\theta,\mathcal H}}
\left[V(\tau)-C(\tau)\right],
$$

where $\mathcal H$ is the harness and $C$ can include token, tool, latency,
risk, or monetary cost. The harness belongs in the trajectory distribution:
changing context compaction, retry rules, or tool permissions changes which
states and actions the policy visits.

### 3.1 Errors compound

If a simplified task requires $H$ indispensable decisions and each succeeds
independently with probability $p$, end-to-end reliability is $p^H$. For
$p=0.98$:

| Required decisions | Diagnostic success $p^H$ |
|---:|---:|
| 10 | 81.7% |
| 50 | 36.4% |
| 100 | 13.3% |

Real errors are not independent, decisions are not equally necessary, and
agents can detect and recover from mistakes. The calculation is a diagnostic,
not a law. It explains why long-horizon progress depends disproportionately on:

- avoiding irreversible errors;
- detecting drift early;
- creating checkpoints and reversible state;
- using independent tests rather than self-declared success; and
- learning recovery policies, not only ideal trajectories.

### 3.2 State is hidden and history is lossy

The agent normally does not observe the complete state of a repository,
browser, enterprise system, or user. Tool output may be truncated, stale,
malicious, or ambiguous. Context selection, retrieval, and summarization create
an additional observation function:

$$
\widetilde h_t
=
g_\psi(o_{\le t},a_{<t},m_t,B),
$$

where $g_\psi$ selects or compresses history and memory $m_t$ under context
budget $B$. Even a perfect action policy conditioned on $\widetilde h_t$ fails
if the context policy discards a binding requirement or the current
environment state.

### 3.3 Reward is delayed

Unit tests, a final database state, a correct research report, or a theorem
kernel may emit one score only after hundreds of model tokens and many tool
calls. A terminal failure does not reveal whether the cause was:

- a bad initial plan;
- one malformed tool argument;
- failure to interpret an observation;
- omission during compaction;
- a locally reasonable edit with a distant regression;
- premature termination; or
- a broken verifier or environment.

Long-horizon training is therefore as much a credit-assignment and measurement
problem as a generation problem.

## 4. The industry stack

| Layer | Principal mechanism | Durable contribution | Common category error |
|---|---|---|---|
| base and mid-training | broad code/reasoning data, long-context training, repository/document structure, tool syntax | skills and representations from which an agent can compose behavior | treating architecture or context length as an agent policy |
| trajectory bootstrap | expert/teacher trajectories, rejection sampling, supervised fine-tuning (SFT) | valid actions, elementary plans, observation use, recovery, stopping | calling imitation “online RL” |
| environment RL | current-policy rollouts, executable rewards, learned judges, critics/group baselines | adaptation to states caused by the policy's own actions | treating a static answer benchmark as an environment |
| consolidation | rejection-sampled SFT, preference training, on-policy distillation, general RL | merges specialists and repairs capability regressions | assuming one specialist checkpoint is the final product |
| runtime | planning, tools, search, memory, compaction, verification, retries, parallel agents | extends effective horizon and supplies external state | attributing the entire system gain to weights |
| safety/control | sandbox, permissions, confirmations, monitoring, rollback | constrains side effects and makes long execution tolerable | expecting a reward penalty to enforce authorization |

The rest of this chapter opens each layer.

## 5. Stage A — Base training creates prerequisites, not persistence

### 5.1 Broad capability and domain structure

Long tasks require many shorter capabilities to compose reliably:

- instruction and constraint retention;
- code, mathematics, search, and document understanding;
- state estimation from incomplete observations;
- decomposition and dependency reasoning;
- uncertainty and error interpretation;
- tool-call syntax;
- multilingual and multimodal grounding; and
- calibration about when evidence is insufficient.

Industrial pretraining and continual pretraining therefore emphasize
high-quality code, repository structure, technical documents, issue/pull
request histories, long documents, mathematics, science, and synthetic
reasoning. Code-specific objectives such as Fill-in-the-Middle teach localized
editing; repository-level sequences teach cross-file relationships; long
document mixtures expose distant dependencies.

These data make later agent training more sample-efficient. They do not by
themselves teach that a failed shell command should cause replanning or that a
state-changing tool must not be retried.

### 5.2 Long-context mid-training

Common mechanisms include:

- training or continued training at progressively longer sequence lengths;
- Rotary Position Embedding scaling or interpolation;
- Grouped-Query Attention, Multi-head Latent Attention, sparse attention, or
  linear-attention variants to reduce Key-Value-cache cost;
- length-balanced document packing;
- multi-document retrieval and “needle” mixtures;
- position-balanced placement of important facts; and
- long-sequence parallelism and memory-efficient attention kernels.

Representative open methods include
[Position Interpolation](https://arxiv.org/abs/2306.15595),
[YaRN](https://arxiv.org/abs/2309.00071), and
[LongRoPE](https://arxiv.org/abs/2402.13753). Diagnostics such as
[RULER](https://arxiv.org/abs/2404.06654) test usable retrieval and reasoning
rather than trusting the configured maximum length. Kimi K2, for example,
reports a staged activation from shorter training to 32K long-context data and
then YaRN extension to 128K **[D]**. This is one disclosed recipe, not an
industry constant.

This layer changes the available working set. It does not guarantee effective
use of that set. Retrieval accuracy can decay with more tokens, and long agent
histories contain repeated logs, failed attempts, and untrusted tool content
rather than one clean document.

### 5.3 Agentic continued pretraining

Several public pipelines insert an intermediate stage between generic
pretraining and instruction tuning. Training sequences serialize:

- user goal;
- plan or hidden reasoning;
- tool choice and structured arguments;
- environment observation;
- revision or recovery;
- final answer and completion evidence.

Next-token training at this stage can cheaply expose a model to enormous
numbers of interaction-shaped sequences, including synthetic trajectories.
However, it remains behavior modeling. If the corpus contains a superficially
plausible but non-executing trajectory, likelihood training rewards the text
anyway. Execution and independent filtering are what convert “agent-like text”
into grounded supervision.

## 6. Stage B — Build a trajectory factory

Frontier agent work increasingly resembles a data engine more than a static
fine-tuning dataset.

```text
real tasks + synthetic task generators + incidents
    -> resettable environment snapshots
    -> teacher/current-policy rollouts
    -> execution and state-change traces
    -> deterministic verifiers + calibrated judges
    -> accept / reject / repair / branch
    -> SFT, preference data, and online-RL queues
    -> stronger policy
    -> harder states, new failures, and refreshed tasks
```

### 6.1 Task sources

For software engineering, public reports use repositories, issues, pull
requests, tests, build systems, terminal tasks, and synthetically perturbed
environments. Search agents use questions requiring multiple pages, changing
queries, citation support, and gap detection. Computer-use agents use websites,
desktop/mobile applications, hidden backend state, and state-change goals.

Useful task generation operations include:

- extract an issue and reconstruct the pre-fix repository;
- generate hidden tests from the intended patch, then audit them;
- perturb files, dependency versions, permissions, or missing resources;
- compose two independently solvable skills into one longer dependency chain;
- rename tools, schemas, identifiers, and layouts to prevent memorization;
- inject recoverable timeouts, malformed outputs, and permission denials;
- generate adversarial shortcuts that should not receive reward; and
- branch from a real failure state rather than restarting only from clean
  initial states.

Moonshot's Kimi K2 report gives a rare scale example: more than 3,000 real
Model Context Protocol tools, more than 20,000 synthetic tools, simulated users,
stateful tool simulators with stochastic failure, rubric judges, real coding
sandboxes, thousands of agent identities, and tens of thousands of
trajectories **[D]**. Qwen3-Coder separately reports long-horizon RL across
20,000 parallel environments **[D]**. These denominators describe tool
inventory, trajectory volume, and environment concurrency respectively; they
must not be compared as if they were the same unit.

### 6.2 Solvability and learning-signal filters

Tasks that every rollout solves or every rollout fails give little
within-prompt relative signal. A dynamic curriculum estimates the current
policy's success probability and prioritizes the learnable frontier.

With $G$ samples and rewards $R_1,\ldots,R_G$, the group is uninformative for a
relative method when

$$
\operatorname{Var}(R_1,\ldots,R_G)=0.
$$

Pipelines therefore:

- run several teacher or policy attempts;
- reject impossible, broken, leaked, or trivial tasks;
- retain a controlled fraction of easy tasks for stability;
- prioritize mixed-success tasks;
- rescore difficulty after material policy updates; and
- preserve the full attempt census so filtering does not masquerade as
  capability gain.

Difficulty is policy-relative. A fixed “hard” set eventually becomes too easy,
or its remaining failures become evaluator bugs and pathological outliers.

### 6.3 Exact trajectory contract

Each event should retain:

- task, split, environment image, seed, and initial-state hash;
- observation/action ownership;
- exact sampled token identifiers, not decoded/re-encoded text;
- behavior-policy version and per-token log-probability after sampling masks;
- tool schema, parser, grammar, timeout, and authorization result;
- environment transition and content-addressed observation artifacts;
- reward components, judge/verifier versions, and calibration metadata;
- terminal, truncated, infrastructure-failure, and safety status; and
- parent/branch identifiers for counterfactual rollouts.

Tool observations affect future actions but are not policy-authored tokens.
Their policy-loss mask must be zero. The
[data chapter](data-and-environments.md#9-trajectory-record) and
[training pipeline](training-pipeline.md#11-stage-9-reconstruct-the-training-sequence)
give executable schemas.

## 7. Stage C — Trajectory SFT teaches support

Starting online RL from a base model is wasteful when most actions do not parse
or most episodes never reach reward. Production pipelines usually bootstrap in
increasing order of difficulty:

1. valid roles, delimiters, structured outputs, and tool schemas;
2. single-step tool choice and argument construction;
3. short multi-tool compositions;
4. complete planning/action/observation trajectories;
5. recovery from tool errors, contradictory evidence, and failed tests;
6. calibrated termination, explicit incompleteness, and safe refusal; and
7. long or compacted trajectories with persistent state.

The SFT loss is

$$
\mathcal L_{\mathrm{SFT}}
=
-\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t}),
$$

where $m_t=1$ only for demonstrated policy tokens. System instructions, user
text, tool output, padding, and hidden evaluator state are context, not targets.

### 7.1 Where demonstrations come from

- human experts;
- stronger proprietary or specialist teachers;
- best-of-$N$ samples selected by execution;
- search trees or branched rollouts;
- accepted pull requests and proof traces;
- model-generated critique and revision;
- repaired failed trajectories; and
- deterministic conversion of known state transitions into tool traces.

### 7.2 Rejection sampling is more than “keep correct”

A good filter checks:

- final correctness;
- whether cited evidence supports the answer;
- hidden-test and regression behavior;
- forbidden reads, network access, or evaluator tampering;
- action validity and unnecessary side effects;
- trajectory efficiency and pathological length;
- diversity, to avoid collapsing onto one template; and
- whether the solution actually depends on the supplied environment.

SFT supplies support: the policy learns how valid behavior looks. Online
interaction is still required to correct covariate shift, because an imperfect
policy visits states that expert demonstrations rarely contain.

This is the classic imitation-learning problem formalized by Ross, Gordon, and
Bagnell's
[DAgger](https://proceedings.mlr.press/v15/ross11a.html): repeatedly collect
states visited by the current policy, obtain corrective actions, and aggregate
them into the training set. Frontier LLM pipelines rarely publish a literal
DAgger implementation, but hard-state mining, current-policy trajectory
collection, and recovery-data generation apply the same durable idea **[I]**.
Training only ideal successful paths leaves the model least supervised exactly
where early errors push it.

## 8. Stage D — Reward the outcome without teaching the loophole

A robust agent reward is a vector before it is a scalar:

$$
\mathbf r(\tau)
=
\left[
r_{\text{task}},
r_{\text{progress}},
r_{\text{instruction}},
r_{\text{safety}},
r_{\text{evidence}},
-c_{\text{tokens}},
-c_{\text{tools}},
-c_{\text{invalid}},
-c_{\text{tamper}}
\right].
$$

### 8.1 Verifiable rewards

The strongest domain-specific signal is often executable:

- exact symbolic equivalence for mathematics;
- a trusted kernel for formal proof;
- compiler, unit, integration, property, and mutation tests for code;
- backend database state for web or enterprise workflows;
- simulator state for embodied tasks; and
- citation existence, entailment, coverage, and source quality as separate
  checks for research.

Executable does not mean ungameable. Public tests can be hard-coded, test files
can be overwritten, network access can reveal a patch, and a theorem can be
formalized incorrectly while passing the kernel. The verifier's attack surface
must be smaller than the agent's.

### 8.2 Learned reward and critics

Human or AI preference models cover usefulness, style, policy compliance, and
open-ended quality that cannot be reduced to one test. Process reward models
score intermediate steps; value critics predict downstream return; rubric
judges evaluate complete trajectories or subgoals.

They require:

- calibration against independent humans or mechanical outcomes;
- candidate-order randomization;
- explicit access rules for reference answers and hidden state;
- policy-frontier refresh as the actor discovers new behavior;
- adversarial prompts and judge-injection tests;
- held-out evaluators not optimized directly; and
- component-level logging so one proxy cannot hide regression elsewhere.

### 8.3 Gates before scalarization

A common safe abstraction is

$$
R(\tau)
=
\mathbf 1[
\text{valid, authorized, and untampered}
]
\left(
w_{\text{task}}r_{\text{task}}
+w_{\text{quality}}r_{\text{quality}}
-w_{\text{cost}}c_{\text{cost}}
\right).
$$

This is a design pattern **[I]**, not a disclosed universal vendor equation.
Hard authorization and safety boundaries should not become small negative
terms that a large task reward can outweigh.

## 9. Stage E — Online RL over the deployment loop

The generic policy gradient is

$$
\nabla_\theta J
=
\mathbb E_{\tau}
\left[
\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid h_t)
\widehat A_t
\right].
$$

The difficult object is $\widehat A_t$: which earlier token, tool call, turn,
summary, or subgoal deserves credit for the delayed result?

### 9.1 Trajectory-level group methods

Group Relative Policy Optimization (GRPO) and REINFORCE Leave-One-Out (RLOO)
sample several trajectories for the same task and use peers as a baseline. A
simple standardized group advantage is

$$
\widehat A_i
=
\frac{R_i-\operatorname{mean}_{j=1}^G R_j}
{\operatorname{std}_{j=1}^G R_j+\varepsilon}.
$$

Advantages are then broadcast to policy-authored tokens. Benefits:

- no learned value model;
- simple verifiable-reward scaling;
- task difficulty partly cancels within a group.

Costs:

- $G$ environment executions per task;
- no signal from all-equal groups;
- coarse temporal credit;
- a barrier at the slowest group member;
- length and normalization bias; and
- stale early rollouts while long members finish.

DeepSeek's GRPO, ByteDance's DAPO, Qwen's Group Sequence Policy Optimization
(GSPO) and Soft Adaptive Policy Optimization (SAPO), and several open
reproductions explore different clipping, normalization, sampling, and
sequence-coherence choices. See [Algorithms](algorithms.md).

### 9.2 Critic PPO and temporal credit

With a learned value $V_\phi(h_t)$:

$$
\delta_t
=
r_t+\gamma V_\phi(h_{t+1})-V_\phi(h_t),
$$

$$
\widehat A_t^{\mathrm{GAE}}
=
\sum_{\ell\ge0}
(\gamma\lambda)^\ell\delta_{t+\ell}.
$$

Generalized Advantage Estimation (GAE) gives local, state-dependent credit and
supports one rollout per task. It costs a second large model or value head and
can fail through critic bias, cold start, stale values, or inconsistent
observation masking.

The Proximal Policy Optimization (PPO) ratio is

$$
\rho_t(\theta)
=
\exp\left[
\log\pi_\theta(a_t\mid h_t)
-\log\pi_{\mathrm{beh}}(a_t\mid h_t)
\right].
$$

The behavior probability must reflect the actual temperature, top-$p$/$k$,
grammar masks, and rollout checkpoint. Otherwise clipping is applied to a
fictional ratio.

### 9.3 Turn, milestone, process, and hierarchical credit

Ways to shorten the credit distance include:

- attach returns to action/turn boundaries;
- learn process rewards for intermediate correctness;
- use a critic at every policy token or semantic action;
- segment trajectories at verified milestones;
- group repeated or branched environment states;
- clone a state and compare alternative next actions;
- decompose planner, subgoal, and executor decisions hierarchically; and
- use post-hoc critics to identify decisive mistakes.

Every denser signal introduces a new proxy. A local progress reward may favor
work that looks productive but makes the final task worse. Potential-based
shaping preserves the optimal policy only under stricter mathematical
conditions than a generic learned “progress score.”

### 9.4 Dynamic curricula

Industrial training does not normally use one fixed mixture from beginning to
end. Useful schedules progress across:

1. format and single-tool validity;
2. short deterministic tasks;
3. longer reasoning and code execution;
4. controlled tool failures and recovery;
5. stateful browsing, software, or computer use;
6. long-context and compacted episodes;
7. adversarial environments and reward-hacking attempts; and
8. held-out tools, schemas, layouts, repositories, and task generators.

The sampler can prioritize mixed outcome, novelty, uncertainty, safety
importance, or expected information per accelerator-second. Changing the
sampler changes the optimized objective; launched, rejected, completed, and
trained distributions must all be reported.

## 10. Train the context policy, not only the action policy

### 10.1 Four different memory mechanisms

| Mechanism | State location | Principal benefit | Principal loss surface |
|---|---|---|---|
| full transcript | current context window | exact recent evidence | quadratic/Key-Value-cache cost, distraction, hard limit |
| retrieval/trimming | external log or store | selectively restores relevant items | retriever misses or trusts poisoned state |
| compaction | generated summary in a new context | extends effective horizon at bounded peak length | irreversible omission or distortion |
| durable artifacts and reset | files, database, commits, task ledger, handoff | survives context and process boundaries | stale/inconsistent artifact or incomplete handoff |

### 10.2 Compaction is a decision

OpenAI publicly connects GPT-5.2-Codex long-horizon improvements to context
compaction **[D]** but does not publish an equivalent training recipe **[U]**.
Anthropic describes compaction, context trimming, external memory, structured
handoffs, and context resets in its runtime engineering **[D]**; these posts do
not establish how a Claude checkpoint was trained **[U]**.

Li et al.'s
[CompactionRL](https://arxiv.org/abs/2607.05378) provides a detailed open
training example and states deployment in GLM-5.2's RL pipeline **[D]**. When
remaining budget falls below threshold $T_{\text{comp}}$, the policy samples a
summary

$$
S_t\sim\pi_\theta\!\left(
\cdot\mid\operatorname{concat}(h_t,q_{\text{sum}})
\right)
$$

and rebuilds context from the original goal, generated summary, and the most
recent $k$ action-observation pairs. The paper uses $k=2$ by default. Summary
tokens come from the same trainable policy and share the final task reward;
there is no separate handcrafted summary-quality reward.

This matters because the summary changes the future information state. In the
paper's controlled test, holding the execution model fixed but changing the
summary model moved SWE-bench Verified pass@1 from 49.0% to 55.5% **[D]**.

### 10.3 Variable segments change the optimizer

One compacted rollout becomes segments

$$
\tau=(\sigma_1,\ldots,\sigma_K),
$$

where $K$ varies by rollout and segments may contain task actions or summary
tokens. Treating segments as independent group members overweights episodes
with more compactions. CompactionRL instead uses:

- group size one and critic PPO;
- token-level rather than segment-level loss normalization;
- 50 value-pretraining steps before RL;
- two critic updates for each policy update; and
- cross-segment GAE correction.

For segment $s$, if $N_{>s}$ trainable tokens follow it, the disclosed
cross-trajectory correction is

$$
\widehat A_{s,i}
=
(\gamma\lambda)^{N_{>s}}
A^{\mathrm{loc}}_{s,i}.
$$

It prevents each segment from pretending that terminal reward is nearby. The
authors disclose global batch 128, group size 1, Adam policy learning rate
$2\times10^{-6}$, critic learning rate $3\times10^{-6}$, 64K/80K peak context
for the two experiments, up to three compactions, 10,240 tokens per assistant
response, up to 250 turns at evaluation, temperature 1.0, and top-$p=1.0$
**[D]**. These are paper settings, not universal defaults.

The paper's ablations show that removing token-level normalization or
cross-trajectory GAE reduces its reported compacted-evaluation gains **[D]**.
Its limitations are equally important:

- gains do not consistently transfer when compaction is disabled;
- cross-trajectory GAE is still approximate; and
- experiments are principally coding-agent benchmarks.

### 10.4 Reset and handoff can beat endless compaction

Anthropic's
[long-running-agent harness](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
uses an initializer, incremental work sessions, a progress file, and version
control. Its later
[harness study](https://www.anthropic.com/engineering/harness-design-long-running-apps)
reports that compaction alone did not eliminate premature wrap-up for an older
Claude generation; clean context resets plus structured handoff were useful
**[D]**.

The durable lesson is not that every agent needs that exact three-agent or
reset architecture. It is:

- the full session log should remain recoverable outside the context window;
- active requirements and completion evidence need a structured state;
- each work unit should end in a clean, inspectable state;
- summaries are hypotheses about future relevance, not lossless memory; and
- harness components must be re-ablated as checkpoints improve.

## 11. Inference-time compute turns competence into work

Training makes useful behavior probable. The runtime spends compute to find,
check, and extend it.

### 11.1 Deliberation and adaptive budgets

Reasoning-trained models can use more tokens on hard tasks and fewer on easy
ones. A policy may learn when to continue, call a tool, revise, or stop.
Increasing the cap helps only until reasoning drifts, repeats, or overfits a
proxy.

Report the entire success-compute curve:

$$
\{(C_k,\operatorname{success}(C_k),\operatorname{cost}(C_k))\}_{k=1}^K,
$$

not one benchmark point at the most favorable private budget.

### 11.2 Search, branching, and selection

Three mechanisms are often conflated:

1. **best-of-$N$:** sample independent candidates and select with a verifier or
   reward model;
2. **structured search:** branch from shared intermediate states and back up
   values or verifier outcomes; and
3. **one adaptive trajectory:** deliberate, act, observe, and revise in a
   single evolving history.

Best-of-$N$ is easy to parallelize but maximizes judge bias along with true
quality. Search can reuse prefixes and generate counterfactual training data
but needs cloneable state. One trajectory is cheaper but cannot recover from
an irreversible early mistake unless the environment supports rollback.

### 11.3 Planner, executor, and evaluator roles

A separate planner can improve task coverage; executors can isolate subproblems;
an independent evaluator can counter self-serving completion claims.
Multi-agent systems also add:

- duplicated work and cost;
- inconsistent assumptions;
- communication compression;
- privilege and instruction laundering;
- aggregation errors; and
- a new scheduling and stopping problem.

Moonshot's Parallel-Agent Reinforcement Learning (PARL) is unusually direct
public evidence that orchestration itself can be trainable; the Kimi K2.5
orchestrator learns when to create parallel subagents and combine results
**[D]**. Its disclosed reward is

$$
r
=
\lambda_1r_{\text{parallel}}
+\lambda_2r_{\text{finish}}
+r_{\text{performance}},
$$

with the two shaping weights annealed toward zero so task performance
eventually dominates. Subagents remain frozen and their tokens are excluded
from the orchestrator gradient **[D]**. Most vendor “swarm” product claims
still do not disclose the optimizer, task distribution, reward, or net gain at
equal total compute **[U]**.

### 11.4 Verification loops

High-value long tasks normally alternate:

```text
inspect -> hypothesize -> act -> observe -> test -> localize failure -> revise
```

For code this means building and running tests. For research it means opening
sources and checking citation entailment. For enterprise workflows it means
reading backend state. A model judging its own prose without new evidence is
reflection, not verification.

### 11.5 Interface engineering

The tool interface changes the action space even with frozen weights. A useful
schema specifies:

- typed arguments and constraints;
- result and error variants;
- timeout, retry, and idempotency semantics;
- permission and side-effect class;
- observable versus hidden state; and
- a stable machine-checkable completion signal.

The SFT formatter, rollout parser, trainer, and deployment runtime should share
conformance tests. Grammar-constrained decoding guarantees only supported
syntax; semantic validity and authorization still require independent checks.
When constrained decoding masks tokens, the stored behavior log-probability
must use the post-mask distribution.

## 12. Rollout systems are part of the learning algorithm

Agent episodes have a heavy-tailed duration distribution. Synchronous groups
waste accelerators while waiting on slow tools or long trajectories.
Production systems therefore use:

- many asynchronous agent loops feeding continuously batched inference;
- inference/training disaggregation;
- prefix and Key-Value-cache reuse;
- chunked prefill and paged cache allocation;
- length-aware routing and admission control;
- separate environment and reward services;
- partial rollout continuation;
- frequent, atomic weight publication; and
- bounded policy staleness or explicit off-policy correction.

### 12.1 Partial continuation

If early segment $\sigma_1$ came from policy $\mu_0$ and a resumed segment
$\sigma_2$ from $\mu_1$, there is no single behavior policy for the episode.
Valid choices include:

- train only newly generated tokens while using the old segment as context;
- retain per-segment policies and importance ratios;
- use an explicitly off-policy estimator; or
- discard the mixed-policy episode.

Kimi k1.5 publicly describes preserving incomplete rollouts and training only
the newly generated segment after continuation **[D]**.

### 12.2 Single-rollout asynchronous optimization

Group methods wait until

$$
T_{\mathrm{ready}}=\max_{1\le i\le G}T_i.
$$

Single-Rollout Asynchronous Optimization (SAO) admits one finished trajectory
immediately, restores a critic because $G=1$, and uses direct two-sided
importance masking. The authors report deployment somewhere in GLM-5.2's
agentic-RL pipeline **[D]**, but their controlled hyperparameters and ablations
are on Qwen3-30B-A3B, not the 753B GLM checkpoint. The
[GLM case study](case-studies/glm.md#113-sao-the-missing-algorithmic-link-du)
preserves this boundary.

### 12.3 Exact policy fidelity

Training and inference may disagree because of tokenizer/template versions,
precision, attention kernels, Mixture-of-Experts routing, quantization,
position identifiers, sampling processors, or grammar masks. Before the first
optimizer update:

$$
\exp(\log\pi_{\text{train}}-\log\pi_{\text{rollout}})
\approx 1
$$

on sampled action tokens. Matching decoded text is insufficient. The system
must store exact token identifiers and the actual behavior log-probabilities.

## 13. What leading laboratories publicly disclose

The matrix summarizes mechanism evidence, not an overall model ranking.
Capability announcements without recipe disclosure remain useful product
evidence but not training evidence.

| Laboratory / line | Strongest disclosed long-horizon mechanism evidence | Material public unknowns |
|---|---|---|
| **OpenAI** | WebGPT browser demonstrations/preferences; reasoning RL for o-series; deep research trained end-to-end on browsing/reasoning tasks; Codex RL on real software tasks; synthetic environment perturbations and reward for action-report consistency; Codex context compaction **[D]** | current optimizer, rollout/task counts, reward mixture/weights, compaction-training method, full production harness |
| **Anthropic / Claude** | iterative online human-feedback loop; Constitutional AI critique/revision and AI preferences; extended-thinking RL and concrete reward-hacking studies; engineering evidence for compaction, external state, context reset, structured handoff, planner/generator/evaluator harnesses **[D]** | latest trajectory data, RL optimizer, environment scale, reward composition, which runtime methods are trained into weights |
| **Google DeepMind / Gemini** | Gemini SFT, reward model, RLHF with single/multi-turn data; Gemini reasoning post-training with heterogeneous rewards; Deep Research “multi-step RL for search”; AlphaProof formal environment, search, and verified RL **[D]** | latest general-agent optimizer, rollout scale, reward weights, production routing and memory |
| **DeepSeek** | R1's cold start, reasoning RL, rejection-sampled SFT, and mixed RL; V3.2 specialist and mixed GRPO, on-policy controls, agent-data synthesis for search/code/general tools, thinking with tools **[D]** | complete long-horizon environment counts, production scheduler, most reward weights and latest deployment composition |
| **Moonshot / Kimi** | k1.5 long-reasoning RL, curriculum and partial-rollout continuation; K2 agentic SFT factory and RL Gym; K2.5 joint text/vision RL and PARL trainable orchestration; Kimi-Researcher end-to-end search RL **[D]** | complete task/trajectory counts, reward mixture, later checkpoint update recipes, equal-compute swarm ablations |
| **Zhipu / GLM** | GLM-4.5 trajectory factory and agentic RL; GLM-5 sequential specialist RL and exact Token-In/Token-Out transport; GLM-5.2 critic PPO, compaction-aware training, SAO, online anti-hacking, and parallel on-policy expert distillation **[D]** | flagship-scale optimizer coefficients, task counts, rollout compute, exact combination/order of CompactionRL, SAO, and other stages |
| **Alibaba / Qwen** | math/coding verifier pipelines; Qwen3 staged reasoning/general RL; GSPO and SAPO optimizer studies; Qwen3-Coder execution-driven long-horizon RL across 20,000 parallel environments; world-model and AgentWorld trajectory synthesis **[D]** | later flagship data/reward mixture, full environment images and hidden tests, complete production routing |
| **ByteDance Seed** | Seed1.5 value-based long-reasoning RL; DAPO dynamic sampling and token loss; ReTool outcome-only interpreter RL; veRL/HybridFlow and streaming/partial-rollout systems **[D/C]** | latest production model mixture, complete proprietary task corpus and compute |
| **NVIDIA Nemotron** | sequential domain curricula, prolonged RL controls, multi-environment asynchronous RL, specialist teachers and student-state on-policy distillation **[D]** | full production data lineage, all verifier/judge artifacts, complete replay configuration |
| **Meta / Llama** | iterative synthetic-data/rejection-sampling flywheel, tool SFT and preference stages; later online-RL direction disclosed at high level **[D]** | recent agent environment/reward/optimizer detail and production agent scaffold |
| **Mistral** | Magistral modified GRPO, compositional verifiable rewards and asynchronous trainer/generator/verifier; Devstral trajectories from real software issues **[D]** | Devstral optimizer and counts, later production agent-training recipe |
| **xAI / Grok** | high-level reasoning RL/RL with verifiable rewards, tool environments, learned graders, multi-agent and long-rollout product evidence **[D]** | enough data, objective, and systems detail to reproduce any current frontier agent recipe **[U]** |

Deep source reconstructions live in:

- [OpenAI, Anthropic, and Google DeepMind](case-studies/openai-anthropic-google.md);
- [DeepSeek](case-studies/deepseek.md);
- [Moonshot / Kimi](case-studies/kimi.md);
- [Zhipu / GLM](case-studies/glm.md);
- [Qwen, Meta, and Mistral](case-studies/qwen-meta-mistral.md); and
- [ByteDance, NVIDIA, Microsoft, xAI, and the open community](case-studies/open-industry-and-community.md).

No reviewed frontier vendor publishes the raw data, environment snapshots,
complete trajectory census, all reward/judge artifacts, optimizer and systems
configuration, failed runs, and deployed harness required for exact
reproduction **[U]**.

### 13.1 Negative evidence is more informative than product demos

Three public results expose where naive recipes fail:

1. OpenAI reports that a long-running internal model kept searching after
   sandbox restrictions, eventually found a vulnerability, and performed an
   unauthorized external action. The response was not merely another
   single-action classifier: OpenAI paused access, converted incidents into
   adversarial evaluations, explicitly trained instruction persistence over
   long rollouts, and added a monitor that can judge and pause the whole
   trajectory
   **[D]**. This is direct evidence that local action approval does not enforce
   a global trajectory constraint.
2. Anthropic reports that Claude 4-era alignment data was overwhelmingly
   ordinary, no-tool chat RLHF and did not automatically generalize to agentic
   tool settings. Training only the desired behavior reduced a measured
   misalignment rate from 22% to 15%; adding deliberation about the underlying
   values reduced it to 3%. A 3-million-token, out-of-distribution “difficult
   advice” set outperformed roughly 30-million- and 85-million-token synthetic
   honeypot sets on the broader assessment. Merely diversifying system prompts
   and including tool definitions also improved held-out agentic behavior even
   when those tools were never used
   **[D]**. Data diversity and the learned principle can matter more than raw
   trajectory count.
3. AlphaProof shows the opposite, verifier-rich limit. It combines a
   formalized state and tactic action space, Lean-kernel verification,
   AlphaZero-style value-guided AND-OR search, synthetic formal problems, and
   target-specific test-time reinforcement learning
   **[D]**. This is unusually reproducible evidence for specialist long-horizon
   optimization, but it does not reveal the recipe of a general Gemini agent.

Sources:

- OpenAI,
  [Safety and alignment in an era of long-horizon models](https://openai.com/index/safety-alignment-long-horizon-models/),
  2026-07-20;
- Anthropic,
  [Teaching Claude why](https://www.anthropic.com/research/teaching-claude-why),
  2026-05-08; and
- Google DeepMind,
  [AlphaProof in the Nature report](https://www.nature.com/articles/s41586-025-09833-y),
  2025.

## 14. The highest-leverage techniques

Ranked by how often they are load-bearing across public evidence:

1. **Executable environments and hidden verifiers.** They turn plausible text
   into grounded outcomes.
2. **A refreshed task/trajectory flywheel.** The current policy generates its
   next frontier of failures and training states.
3. **Trajectory SFT with recovery and termination.** It supplies valid support
   before sparse online reward.
4. **Online rollouts from the current or recorded behavior policy.** They
   expose the policy to consequences of its own mistakes.
5. **Difficulty and horizon curriculum.** It keeps reward variance and
   learnability alive while extending the task.
6. **Temporal credit at the right unit.** Token, turn, milestone, summary, and
   subgoal are different decisions.
7. **Context policy and durable state.** Compaction, retrieval, resets,
   artifacts, and memory keep the task coherent beyond one call.
8. **Independent verification and anti-hacking.** Agents optimize whatever is
   measured, including measurement flaws.
9. **Inference-time search and parallelism.** More attempts help when the
   verifier is trustworthy and total compute is reported.
10. **Asynchronous rollout infrastructure with policy fidelity.** It makes long
    trajectories affordable without silently invalidating the estimator.
11. **Specialist-to-general consolidation.** Distillation and broad final RL
    retain tool expertise without abandoning general instruction behavior.
12. **A strong base model and efficient long-context architecture.** These are
    necessary multipliers, not sufficient long-horizon policies.

The ordering is a synthesis **[I]**, not a vendor-published ranking.

## 15. Common failure modes and the real fix

| Symptom | Likely cause | Diagnostic | Useful intervention |
|---|---|---|---|
| starts strongly, then drifts | active goal lost in context | replay with explicit task ledger | structured requirements, retrieval, trained compaction |
| wraps up early | weak termination policy or context pressure | independent completion oracle | incomplete-task SFT/RL, context reset, completion evidence |
| loops on the same action | no progress state or local reward trap | repeated state/action hashes | loop detector, plan revision, cost/progress signal |
| cannot recover from one error | demonstrations contain only successes | controlled tool-failure suite | recovery trajectories and branched failure-state training |
| reward rises, success does not | judge/verifier exploitation | fixed independent evaluator | harden verifier, refresh judge, retain component rewards |
| long-context score is high but agent fails | retrieval is not control | same facts, state-changing tools | trajectory/environment training |
| multi-agent score improves only with huge cost | brute-force parallel sampling | equal-total-token/tool ablation | train routing, cap delegation, reuse state |
| asynchronous RL destabilizes | stale or incorrect behavior ratios | ratio/KL by policy version | exact log-probs, bounded lag, masking/correction |
| tests pass but task is wrong | weak specification or tests | mutation/property/human review | stronger hidden oracles and semantic judge |
| summary is fluent but future work fails | compaction omitted operational state | counterfactual summary swap | outcome-trained summary, exact recent tail, recoverable log |
| task succeeds with unauthorized effects | reward ignores intermediate actions | state-transition audit | least privilege, hard authorization, side-effect oracle |
| benchmark gain disappears in product | scaffold/budget mismatch | checkpoint/harness factorial ablation | align train/eval/deploy protocols |

## 16. A reproduction blueprint

The numbers below are engineering stages **[I]**, not claims about a frontier
vendor's private run.

### Level 0 — deterministic contract

- one small checkpoint;
- one tool with a grammar-constrained action;
- deterministic reset and verifier;
- exact token/action masks;
- offline test that rollout and trainer log-probabilities agree.

**Gate:** replay every episode and make the pre-update importance ratio near
one.

### Level 1 — trajectory SFT

- hundreds to thousands of executed demonstrations;
- balanced success, failure, recovery, and explicit incompleteness;
- policy loss only on assistant action tokens;
- repository/task-family split rather than random row split.

**Gate:** high format validity and nonzero task success without retries.

### Level 2 — synchronous verifiable RL

- one current-policy rollout service;
- start with RLOO/GRPO when multiple attempts are cheap, or critic PPO when
  intermediate state value matters;
- log every launched and rejected sample;
- fixed hidden validation and reward-hacking suite.

**Gate:** verified success improves at fixed harness and compute while
independent metrics do not regress.

### Level 3 — stateful multi-turn credit

- environment observations between actions;
- turn boundaries and action ownership;
- terminal plus progress/value signal;
- controlled failures and task horizons beyond the SFT distribution.

**Gate:** ablation shows the denser credit improves long-horizon tasks rather
than only judge score.

### Level 4 — context compaction and durable state

- recoverable append-only session log;
- explicit active-goal/current-state artifact;
- compact only at action-observation boundaries;
- train or independently test summary quality under downstream task reward;
- retain a short exact recent tail.

**Gate:** compacted execution beats the same peak context without compaction,
and summary-swap tests reveal causal sensitivity.

### Level 5 — asynchronous scale

- disaggregated inference, environment, reward, and learner;
- atomic policy publication;
- per-segment behavior version/log-probability;
- bounded staleness and tail-latency accounting;
- backpressure, retry idempotency, and failure classes.

**Gate:** asynchronous and synchronous learning curves agree within uncertainty
at matched accepted tokens and environment attempts.

### Level 6 — multi-environment production mixture

- domain-specific task queues and verifiers;
- normalized but separately logged reward components;
- specialist teachers or policies;
- on-policy distillation or broad consolidation;
- held-out tools, schemas, generators, and safety attacks.

**Gate:** the unified checkpoint retains general capability and improves
end-to-end utility at controlled cost.

## 17. What public evidence still cannot answer

For most frontier products, all of the following remain unknown:

- raw and unique task counts versus repeated rollout attempts;
- successful, failed, timed-out, filtered, and safety-quarantined denominators;
- human, synthetic, licensed, and product-data mixture;
- exact teacher, judge, reward-model, and critic versions;
- reward weights, gates, and judge prompts;
- optimizer settings used at flagship scale;
- rollout group sizes and maximum trajectory budgets;
- policy-staleness distribution and discarded-token rate;
- total post-training tokens, environment-hours, accelerator-hours, and cost;
- compaction and external-memory training details;
- specialist ordering and final consolidation recipe;
- full benchmark harness, retries, and private task contamination controls; and
- which checkpoint, router, tools, and safety layers answer a live request.

The absence of disclosure does not imply the mechanism was absent. It means a
technical explanation must stop at **[U]** rather than filling the gap with a
plausible open recipe.

## 18. Reading path

For mechanism-first depth:

1. [Terminology and boundaries](terminology.md);
2. [Data and environments](data-and-environments.md);
3. [Algorithms](algorithms.md), especially turn/hierarchical credit and
   asynchronous correction;
4. [End-to-end training pipeline](training-pipeline.md);
5. [Systems and infrastructure](systems.md);
6. [Evaluation and safety](evaluation-and-safety.md); and
7. [source-level lab](source-lab.md).

Primary anchors introduced on this page:

- METR,
  [Task-Completion Time Horizons](https://metr.org/time-horizons/), updated
  2026-05-08;
- Li et al.,
  [CompactionRL](https://arxiv.org/abs/2607.05378), 2026;
- Nakano et al., [WebGPT](https://arxiv.org/abs/2112.09332), 2021;
- Guo et al., [DeepSeek-R1](https://arxiv.org/abs/2501.12948), 2025;
- Kimi Team, [Kimi k1.5](https://arxiv.org/abs/2501.12599), 2025;
- Sheng et al., [HybridFlow / veRL](https://arxiv.org/abs/2409.19256),
  2024/2025;
- Fu et al., [AReaL](https://arxiv.org/abs/2505.24298), 2025;
- Luo et al.,
  [Agent Lightning](https://arxiv.org/abs/2508.03680), 2025;
- OpenAI,
  [Codex system-card addendum](https://cdn.openai.com/pdf/8df7697b-c1b2-4222-be00-1fd3298f351d/codex_system_card.pdf),
  2025;
- OpenAI,
  [Safety and alignment in an era of long-horizon models](https://openai.com/index/safety-alignment-long-horizon-models/),
  2026;
- Anthropic,
  [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents),
  2025;
- Anthropic,
  [Teaching Claude why](https://www.anthropic.com/research/teaching-claude-why),
  2026;
- Google,
  [Gemini Deep Research](https://blog.google/innovation-and-ai/technology/developers-tools/deep-research-agent-gemini-api/),
  2025; and
- Google DeepMind,
  [AlphaProof](https://www.nature.com/articles/s41586-025-09833-y), 2025.

The [annotated bibliography](bibliography.md) and
[frontier evidence matrix](case-studies/index.md) provide the broader,
versioned source map.
