# Historical Development of Agentic Reinforcement Learning

Agentic RL did not appear in one paper. It is the convergence of sequential
decision making, policy gradients, language-model post-training, tool-using
agents, verifiable reasoning, and distributed generation systems. This chapter
tracks those strands separately so that later terminology is not projected
backward onto earlier work.

## 1. Before language models: the sequential-decision foundation

### 1950s–1980s: dynamic programming, control, and temporal difference ideas

Bellman's dynamic programming formalized optimal sequential decisions through
recursive value functions. The essential insight is that a decision is valuable
because of both its immediate reward and the future state distribution it
induces. This is exactly what one-response preference optimization abstracts
away and agentic RL restores.

By the late 1980s, temporal-difference methods combined bootstrapping with
experience. Sutton's 1988 paper introduced a family of TD methods and connected
them to prediction over temporally extended trajectories
([Sutton, 1988](https://doi.org/10.1007/BF00115009)).

### 1992: likelihood-ratio policy gradients

Williams' REINFORCE estimator showed that a stochastic policy can be optimized
without differentiating through environment dynamics:

\[
\nabla_\theta J(\theta)
=\mathbb{E}\left[G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].
\]

This identity remains the core of modern LLM RL. PPO, RLOO, GRPO, and many
agent-training objectives primarily differ in their advantage estimator,
regularization, sampling structure, and update constraints—not in the basic
score-function identity
([Williams, 1992](https://doi.org/10.1007/BF00992696)).

### 1990s–2017: actor–critic, trust regions, and PPO

Actor–critic algorithms learned a value baseline to reduce policy-gradient
variance. Generalized Advantage Estimation later exposed a tunable bias–variance
tradeoff for advantage estimates
([Schulman et al., 2015](https://arxiv.org/abs/1506.02438)). Trust Region Policy
Optimization constrained destructive policy movement
([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)); PPO replaced the
harder constrained step with a clipped surrogate that was easier to scale
([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)).

Deep RL simultaneously demonstrated learning from high-dimensional
observations. DQN learned Atari control from pixels
([Mnih et al., 2015](https://doi.org/10.1038/nature14236)), and AlphaGo combined
supervised learning, reinforcement learning, value prediction, and tree search
([Silver et al., 2016](https://doi.org/10.1038/nature16961)). These systems were
not LLM agents, but they established the modern pattern of demonstrations,
self-generated experience, learned value/reward signals, search, and policy
improvement.

## 2. Language modeling meets human feedback

### 2017: preferences as a scalable reward interface

Christiano et al. trained a reward predictor from human comparisons and then
optimized a policy against it
([*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741)).
The paper used control tasks rather than LLMs, but the data loop—sample behavior,
ask humans to compare clips, fit reward, optimize policy—became the conceptual
template for RLHF.

### 2017–2020: Transformers and language-model preference optimization

The Transformer made large autoregressive sequence policies practical
([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)). Ziegler et al. applied
preference learning and PPO-style fine-tuning to language models
([*Fine-Tuning Language Models from Human Preferences*](https://arxiv.org/abs/1909.08593),
2019). Stiennon et al. then documented a full summarize–compare–reward-model–RL
pipeline at larger scale
([*Learning to summarize from human feedback*](https://arxiv.org/abs/2009.01325),
2020).

These systems were mostly one-prompt/one-response optimization. They supplied
the reward-model and policy-optimization machinery later reused for agents, but
their environment could often be approximated as a contextual bandit.

### 2021–2022: web interaction and instruction following

WebGPT is an important bridge. The model navigated a text browser, issued search
queries, followed links, collected references, and produced an answer; human
demonstrations and comparisons trained the behavior and reward model
([Nakano et al., 2021](https://arxiv.org/abs/2112.09332)). It made environment
interaction, evidence collection, and long action sequences central to an LLM
post-training pipeline.

InstructGPT documented the now-canonical three-stage recipe: supervised
fine-tuning on demonstrations, reward modeling on ranked outputs, and PPO
against the learned reward with a KL-related constraint
([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155), Figure 2 and Section 3).
The recipe was not a general agent-training system, but it standardized the
components from which many later systems were built.

Constitutional AI showed how written principles and model-generated critique,
revision, and preference labels could reduce direct human labeling in part of
the alignment loop
([Bai et al., 2022](https://arxiv.org/abs/2212.08073)). This established AI
feedback as a practical, though imperfect, source of scalable supervision.

## 3. The agent scaffold era

### 2022: reasoning interleaved with actions

ReAct represented model behavior as interleaved reasoning traces and actions,
then used environment observations to continue the trajectory
([Yao et al., 2022](https://arxiv.org/abs/2210.03629)). The original contribution
was primarily prompting/fine-tuning and trajectory design rather than online
agent RL. Its lasting impact was the explicit thought–action–observation loop
that many agent environments expose.

### 2023: tool use, reflection, memory, and embodied open-endedness

- Toolformer self-labeled API calls and fine-tuned a model to insert them
  ([Schick et al., 2023](https://arxiv.org/abs/2302.04761)). It is tool-use
  training, but not online RL.
- Reflexion stored verbal feedback in episodic memory to improve later attempts
  without weight updates
  ([Shinn et al., 2023](https://arxiv.org/abs/2303.11366)). “Verbal reinforcement
  learning” here should not be confused with gradient-based policy optimization.
- Voyager used an automatic curriculum, executable feedback, iterative prompting,
  and a growing skill library in Minecraft
  ([Wang et al., 2023](https://arxiv.org/abs/2305.16291)). It demonstrated
  open-ended agent improvement while keeping the base model frozen.

This era matters because it separated **agent capability produced by a scaffold**
from **capability stored in model weights**. Agentic RL later tries to internalize
some of those successful behaviors through policy updates, but rigorous
experiments must still ablate the scaffold.

## 4. Verifiable reasoning changes the economics of RL

### 2023–2024: preference optimization and process supervision

DPO expressed a KL-regularized preference objective as a simple classification
loss on preference pairs, avoiding online rollout and an explicit reward model
([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). It made
preference-based post-training easier but did not solve interactive credit
assignment.

*Let's Verify Step by Step* released process-supervision methodology and PRM800K,
showing advantages of labeling intermediate mathematical reasoning steps for
reliable selection
([Lightman et al., 2023](https://arxiv.org/abs/2305.20050)). Process labels
foreshadow turn- and step-level rewards for agents, while also introducing the
risk of optimizing plausible-looking intermediate behavior.

### 2024: GRPO and large-scale math RL

DeepSeekMath introduced Group Relative Policy Optimization (GRPO). Instead of a
learned critic, it sampled a group of outputs for the same question and
normalized their rewards to form relative advantages
([Shao et al., 2024](https://arxiv.org/abs/2402.03300), Section 3). The technique
lowered critic memory/compute requirements and became a foundation for many
reasoning-RL recipes.

At the same time, interactive evaluation and training resources expanded across
software engineering, web navigation, tool use, and embodied environments.
SWE-bench provided repository-level issue resolution tasks
([Jimenez et al., 2024](https://arxiv.org/abs/2310.06770)); AgentBench evaluated
LLMs as agents across multiple environments
([Liu et al., 2023](https://arxiv.org/abs/2308.03688)); and WebArena supplied
realistic reproducible web tasks
([Zhou et al., 2023](https://arxiv.org/abs/2307.13854)).

## 5. Reasoning RL becomes a model-development stage

### January 2025: DeepSeek-R1

DeepSeek-R1-Zero applied large-scale RL directly to a base model without a
preliminary SFT cold start in the disclosed experiment. The report describes
emergent longer reasoning and self-reflection, alongside readability and
language-mixing problems. DeepSeek-R1 added a small cold-start dataset, a
reasoning-oriented RL stage, rejection-sampled SFT data mixed with non-reasoning
tasks, and a final RL stage combining verifiable and preference signals
([Guo et al., 2025](https://arxiv.org/abs/2501.12948), Sections 2.2–2.3).

The key historical point is not that RL “invented reasoning.” It is that a
frontier developer publicly documented online policy optimization with largely
verifiable rewards as a major capability-training stage and released distilled
models that transferred the behavior to smaller dense checkpoints.

### January 2025: Kimi k1.5

Moonshot AI's Kimi k1.5 report described long-context RL, improved policy
optimization, and a multimodal model trained on text and vision reasoning tasks
([Team et al., 2025](https://arxiv.org/abs/2501.12599)). It emphasized the
engineering around long rollouts—sampling, length curriculum, reward design,
and data diversity—rather than treating the optimizer name as the complete
recipe.

### 2025: algorithm and data-engine refinement

The open research community rapidly tested which parts of R1-style training
were essential:

- DAPO documented decoupled clipping, dynamic sampling, token-level policy
  gradient loss, and overlong-reward shaping in a large-scale open recipe
  ([Yu et al., 2025](https://arxiv.org/abs/2503.14476)).
- *Understanding R1-Zero-Like Training* analyzed biases created by GRPO reward
  normalization and length normalization and proposed Dr. GRPO
  ([Liu et al., 2025](https://arxiv.org/abs/2503.20783)).
- Search-R1 trained models to interleave reasoning with retrieval using outcome
  rewards
  ([Jin et al., 2025](https://arxiv.org/abs/2503.09516)).
- ReTool trained strategic code-interpreter use through RL
  ([Feng et al., 2025](https://arxiv.org/abs/2504.11536)).

The field's center of gravity moved from a single final answer to trajectories
that include retrieval, code execution, tests, and external observations.

## 6. Agentic RL becomes an explicit systems category

### 2025: decoupled and asynchronous frameworks

Generating interactive trajectories is slower and more variable than generating
fixed-length responses. Tool latency creates GPU bubbles; environment workers
fail independently; long episodes create stragglers; and trainers consume data
at a different rate from collectors.

HybridFlow/veRL formalized flexible placement and dataflow for RL post-training
([Sheng et al., 2024/2025](https://arxiv.org/abs/2409.19256)). Its agent loop
later separated asynchronous inference servers from environment clients and
retained exact generated tokens. AReaL explored fully asynchronous RL for
language reasoning
([Fu et al., 2025](https://arxiv.org/abs/2505.24298)). Agent Lightning decoupled
arbitrary agent execution from training and introduced hierarchical trajectory
decomposition
([Luo et al., 2025](https://arxiv.org/abs/2508.03680)).

These systems made policy staleness, token consistency, environment RPC,
backpressure, and trajectory tracing first-class research concerns.

### 2025–2026: formal consolidation

Zhang et al. distinguished preference-based LLM fine-tuning from Agentic RL by
modeling the former as a degenerate one-step MDP and the latter as a POMDP. Their
survey organized the field by capabilities—reasoning, planning, tool use,
memory, reflection/self-improvement, perception, and long-horizon interaction—
and by task domains
([Zhang et al., 2026](https://arxiv.org/abs/2509.02547)).

This formalization is valuable because it prevents “agentic” from becoming a
marketing synonym for any strong model. An actual agentic-RL claim should expose
the environment, transitions, horizon, feedback, trajectory collection, and
credit assignment.

## 7. The current frontier

As of this repository's current research window, the central problems are not
settled:

1. **Long-horizon credit:** terminal success remains a noisy explanation of
   which early action mattered.
2. **Environment scale:** realistic, diverse, resettable, safe environments are
   more expensive to build than static prompt datasets.
3. **Reward validity:** learned judges and process rewards are exploitable;
   verifiers cover only tasks with checkable outcomes.
4. **On-policy throughput:** interactive rollouts dominate wall time and produce
   severe length and latency variance.
5. **Policy lag:** asynchronous generation improves hardware utilization while
   increasing distribution mismatch.
6. **Generalization:** narrow environment success often fails to transfer to new
   tools, schemas, horizons, or hidden state distributions.
7. **Safety under optimization:** agents can discover unintended actions that
   satisfy a proxy reward, so sandbox and authorization design are part of the
   learning system, not deployment decoration.
8. **Scientific attribution:** base-model strength, SFT data, scaffold changes,
   inference compute, verifier quality, and RL updates are commonly confounded.
9. **Incomplete disclosure:** frontier labs rarely publish full data mixtures,
   reward rubrics, cluster topology, ablation history, or production agent loops.

The next advance will likely be a co-design of model, data engine, environment,
algorithm, distributed runtime, and evaluator—not an optimizer acronym in
isolation.

## A compact lineage

```text
Dynamic programming / MDPs
  -> temporal-difference learning and policy gradients
  -> deep RL, actor-critic, trust regions, PPO, self-play
  -> human preferences as learned reward
  -> language-model RLHF and instruction following
  -> interactive browser/tool agents and agent scaffolds
  -> verifiable reasoning rewards and group-relative optimization
  -> long reasoning + tool-integrated trajectories
  -> asynchronous, environment-rich, long-horizon Agentic RL
```

This lineage is conceptual, not a claim that every later system uses every
earlier technique.

## References

The links in the chronology point to the primary papers. See the
[annotated bibliography](bibliography.md) for grouping by learning objective
and for framework repositories.
