# Open Industry and Community: Agentic RL Recipes, Systems, and Reproduction Boundaries

**Verified through:** 2026-07-19. **Scope:** ByteDance Seed, NVIDIA
Nemotron, Microsoft Research, xAI, and the open research and engineering
community. Benchmark values are vendor- or author-reported unless explicitly
marked as independently reproduced.

This chapter reconstructs public agentic-reinforcement-learning practice from
primary papers, technical reports, model cards, official release posts, and
versioned repositories. It deliberately separates an advertised capability from
the mechanism that produced it. A model that can call a tool might have learned
from supervised trajectories, online reinforcement learning (RL), a prompt-only
agent loop, or all three. The distinction is central to reproducibility.

## Evidence labels used throughout

- **[D] Disclosed:** directly stated in a cited paper, model card, official
  release, or first-party technical report.
- **[C] Confirmed artifact:** observed in released code, configuration, data, or
  weights at the cited revision. It need not describe the authors' private
  production run.
- **[I] Inference:** a bounded interpretation derived from disclosed evidence;
  it is not a claim that an organization used that exact implementation.
- **[U] Unknown:** not disclosed, ambiguous, contradictory, or not recoverable
  from the cited public artifacts.

These labels attach to the smallest practical unit of a claim. For example,
“Grok 4.5 used hundreds of thousands of RL tasks” is **[D]**; its optimizer,
group size, and learning rate are **[U]**. A current repository configuration is
**[C]**, not evidence that the same configuration trained a released model.

## Reader's terminology key

- **Agentic RL:** RL in which a language-model policy makes one or more
  decisions inside an environment, potentially including reasoning tokens,
  tool calls, code, interface actions, and a final answer.
- **Trajectory, episode, rollout:** one sampled interaction from reset to
  termination. A trajectory can contain several assistant turns and external
  observations.
- **State, observation, action:** the state is the environment's full condition;
  the observation is what the policy sees; an action is what it emits. In a
  language agent, one action may be a token, an entire assistant message, or a
  structured tool call depending on the mathematical abstraction.
- **Supervised Fine-Tuning (SFT):** next-token imitation of curated target
  answers or trajectories. Tool-use SFT is not RL merely because the examples
  contain environment interactions.
- **Reinforcement Learning with Verifiable Rewards (RLVR):** policy training
  from mechanically or programmatically checkable outcomes, such as exact
  answers, unit tests, database states, proof checkers, or JSON schemas.
- **Reinforcement Learning from Human Feedback (RLHF):** a family of policy
  optimization methods driven by human preferences, usually through a learned
  reward model. It is not the name of one optimizer.
- **Proximal Policy Optimization (PPO):** a policy-gradient method that clips
  likelihood-ratio updates and commonly uses a learned value function.
- **Group Relative Policy Optimization (GRPO):** a critic-free method that
  estimates an advantage by comparing multiple responses for the same prompt.
- **Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO):** a
  GRPO-derived recipe combining separate lower/upper clipping, dynamic
  nonzero-variance sampling, token-level loss aggregation, and overlength
  handling.
- **REINFORCE / REINFORCE++:** score-function policy gradients; REINFORCE++ is a
  family of practical language-RL variants that adds PPO-style stabilization
  without a critic.
- **Leave-One-Out (RLOO):** a baseline that compares one rollout with the mean
  reward of the other rollouts in its group.
- **Generalized Advantage Estimation (GAE):** a discounted sum of temporal-
  difference residuals controlled by $\gamma$ and $\lambda$.
- **Mixture of Experts (MoE):** a sparse model in which each token activates a
  subset of feed-forward experts.
- **Generative Reward Model (GenRM):** a language model that reasons about and
  grades candidate responses, rather than returning only a scalar from a
  classification head.
- **Process Reward Model (PRM):** a model that scores intermediate reasoning
  steps. An implicit PRM can derive token rewards without step-level labels.
- **Process Reinforcement through IMplicit rEwards (PRIME):** an online RL
  method that learns an implicit PRM from policy rollouts and outcome labels.
- **Multi-stage Adaptive entropy scheduling for GRPO In Convergence (MAGIC):**
  Skywork-OR1's GRPO-based pipeline with staged context growth, online
  filtering, and adaptive entropy control.
- **MOPD:** Cascade 2 expands the name as **Multi-Domain On-Policy
  Distillation**, while Ultra uses **Multi-teacher On-Policy Distillation** for
  the same acronym. Both sample student states and use specialist-teacher log
  probabilities as dense feedback; the report-specific name matters.
- **Reward-aware Preference Optimization (RPO):** NVIDIA's preference-
  optimization framework for expressing objectives and reward information
  under a common formulation.
- **On-policy:** rollout data comes from the policy being updated. “One-step
  off-policy” means the behavior policy is intentionally at most one update old.
- **Policy lag, staleness:** the number of learner updates separating the policy
  that generated a token from the current learner.
- **Importance sampling (IS):** weighting data by a target-to-behavior policy
  likelihood ratio to correct distribution mismatch.
- **Kullback-Leibler (KL) divergence:** a measure of policy drift, often used as
  a penalty or trust-region proxy.
- **Entropy:** uncertainty of the policy distribution. Entropy collapse means
  exploration disappears as the distribution becomes sharply peaked.
- **Negative Log-Likelihood (NLL):** the token-level supervised
  cross-entropy term that increases likelihood of a selected target sequence.
- **Effective Sample Size (ESS):** an importance-weight diagnostic measuring
  how many equally weighted samples would carry comparable statistical mass.
- **Retrieval-Augmented Generation (RAG):** generation conditioned on
  externally retrieved context; agentic search instead lets retrieval be a
  learned action inside a multi-step environment.
- **Transformer Reinforcement Learning (TRL):** the Hugging Face library and
  project that provides supervised, preference, and RL post-training
  components; a TRL configuration is an implementation artifact, not an
  algorithm name by itself.
- **Token-global loss:** sum loss over all eligible response tokens in a batch
  and divide by the total eligible-token count. This differs from averaging each
  response first and then averaging responses.
- **Rollout engine:** the inference runtime that samples trajectories, such as
  vLLM or SGLang.
- **Training-inference mismatch:** numerical or semantic differences between
  log probabilities computed by the rollout engine and by the training engine.
- **Fully Sharded Data Parallelism (FSDP):** sharding parameters, gradients, and
  optimizer state across workers.
- **Tensor, Expert, Sequence, Context, and Pipeline Parallelism (TP, EP, SP,
  CP, PP):** complementary ways to distribute model computation.
- **Multi-Token Prediction (MTP):** predicting several future tokens; selected
  systems also reuse it for speculative decoding.
- **Partially Observable Markov Decision Process (POMDP):** a sequential
  decision process in which the policy sees observations rather than the entire
  environment state.
- **Software-engineering RL (SWE-RL):** RL over repository tasks in isolated
  environments, typically rewarded by tests over a generated patch.

## 1. Executive conclusions

1. **The public field progressed from answer-only math RL to heterogeneous
   environment mixtures.** DAPO and early open reproductions optimize one
   completion against an exact-answer verifier. Nemotron 3 Super and Ultra
   disclose mixtures spanning math, code, search, terminal use, conversational
   tools, safety, long context, office work, and full software-engineering
   agents.
2. **Data and verifier operations dominate the real recipe.** Difficulty
   profiling, all-pass/all-fail filtering, decontamination, test hardening,
   sandbox isolation, output masking, and reward audits are as consequential as
   the named optimizer.
3. **There is no single industrial consensus on policy freshness.** AceReason
   and Nemotron Cascade use one strictly on-policy update per rollout;
   Seed1.5-Thinking mixes current and snapshot continuations; AReaL supports
   controlled stale trajectories; later Nemotron models use one-step-off-policy
   asynchronous generation with in-flight weight updates.
4. **SFT and RL alternate rather than replace one another.** Cold-start SFT
   creates a valid action language, RL explores against outcomes, rejection or
   on-policy distillation transfers specialist behavior, and later preference
   stages recover interaction quality. OpenThoughts is an especially useful
   negative example: it is a strong reasoning-data project, but its published
   OpenThinker3 recipe is SFT, not RL.
5. **“Open” has multiple levels.** Some releases provide weights only; others
   add data, environments, code, and configurations. None of the reviewed
   organizations provides every production corpus, grader, cluster topology,
   service dependency, and exact revision needed to replay its latest frontier
   run bit-for-bit.
6. **Agentic RL is a distributed-systems problem.** Long-tail rollouts,
   environment startup, verifier latency, weight synchronization, key-value
   (KV) cache validity, and failure recovery frequently determine effective
   sample throughput.
7. **Reported success can hide objective failure.** GRPO normalization can bias
   length and prompt difficulty; entropy can collapse; weak tests admit false
   positives; a verifier can reward extensional shortcuts; and async engines
   can silently train against different token probabilities.
8. **The most honest reconstruction uses three layers:** a mathematical
   objective, a trajectory/data contract, and a versioned systems artifact. A
   paper equation without the exact token mask is insufficient, and a runnable
   script without the paper's production data is not a reproduction.

## 2. A common mathematical and operational frame

### 2.1 Trajectory contract

For task $x$, let a rollout be

$$
\tau=(o_1,a_1,r_1,o_2,a_2,r_2,\ldots,o_T,a_T,r_T),
\qquad R(\tau)=\sum_{t=1}^{T}\gamma^{t-1}r_t.
$$

An assistant action $a_t$ can itself contain tokens
$y_{t,1:L_t}$. External tool output belongs to the observation, not the
policy action. A correct training record therefore needs at least:

```text
task_id, environment_revision, reset_seed
observation_tokens, action_tokens, action_loss_mask
behavior_logprobs, behavior_policy_version
tool_request, tool_result, tool_result_token_mask
terminal_reason, raw_verifier_output, scalar_reward
grader_revision, sandbox_image, timeout/status metadata
```

**[I]** Omitting any one of tokenizer revision, action mask, behavior version,
or verifier revision can make an apparently identical run optimize a different
objective.

### 2.2 Group-relative advantage

For $G$ responses to prompt $x$, the familiar GRPO estimator is

$$
\hat A_i = \frac{R_i-\bar R}{s_R+\epsilon},
\quad
\bar R=\frac1G\sum_{j=1}^{G}R_j.
$$

All tokens in response $i$ often receive the same $\hat A_i$. If every
response is correct or every response is wrong, $s_R=0$ and the group supplies
no relative signal. Dynamic sampling therefore tries to retain groups with
mixed outcomes. Removing $s_R$ yields a centered, fixed-scale estimator; using
the other $G-1$ samples as the baseline yields RLOO.

### 2.3 Clipped update and token aggregation

Let

$$
\rho_{i,t}(\theta)=
\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
     {\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})}.
$$

An asymmetric clipped surrogate is

$$
L_{i,t}^{\mathrm{clip}}=
\min\!\left(
  \rho_{i,t}\hat A_i,
  \operatorname{clip}(\rho_{i,t},1-\epsilon_{\mathrm{low}},
  1+\epsilon_{\mathrm{high}})\hat A_i
\right).
$$

“Clip-Higher” makes $\epsilon_{\mathrm{high}}>\epsilon_{\mathrm{low}}$,
allowing more upward movement for initially unlikely rewarded tokens. Given an
action mask $m_{i,t}$, token-global aggregation is

$$
J_{\mathrm{token}}=
\frac{\sum_{i,t}m_{i,t}L_{i,t}^{\mathrm{clip}}}
     {\sum_{i,t}m_{i,t}}.
$$

Per-response aggregation instead gives each sequence equal weight regardless of
length. Neither is “just an implementation detail”: it changes the effective
training distribution.

### 2.4 Three policies in asynchronous RL

At scale, distinguish:

- $\pi_{\mathrm{beh}}$: generated the token;
- $\pi_{\mathrm{prox}}$: recent trust-region center;
- $\pi_\theta$: current learner.

Their ratios answer different questions:

$$
c_t=\frac{\pi_{\mathrm{prox}}(a_t\mid s_t)}
          {\pi_{\mathrm{beh}}(a_t\mid s_t)},
\qquad
r_t=\frac{\pi_\theta(a_t\mid s_t)}
          {\pi_{\mathrm{prox}}(a_t\mid s_t)}.
$$

The first corrects stale sampling; the second constrains the learner around a
recent policy. Clipping $\pi_\theta/\pi_{\mathrm{beh}}$ alone conflates both
roles. A systems diagram that says “async PPO” without naming these policies is
underspecified.

### 2.5 The real production loop

Across the case studies, a defensible loop is:

1. acquire or generate tasks and record licenses/provenance;
2. decontaminate evaluation-near material;
3. construct a deterministic resettable environment;
4. build independent reference solutions and adversarial tests;
5. profile the starting policy with multiple attempts;
6. remove malformed, trivial, impossible, or reward-degenerate tasks;
7. teach the action syntax and basic workflow with SFT;
8. generate versioned online or bounded-staleness rollouts;
9. verify in parallel and preserve raw evidence, not only a scalar;
10. compute masked advantages and policy updates;
11. monitor reward, held-out success, entropy, length, KL, clipping fraction,
    staleness, verifier disagreement, timeout class, and per-domain regression;
12. periodically re-profile difficulty and refresh the task mixture;
13. gate checkpoints on clean held-out environments and human audits;
14. publish or pin code, data, model, tokenizer, container, and evaluator
    revisions separately.

## 3. ByteDance Seed: value-based long-CoT RL, DAPO, tools, and veRL

### 3.1 Lineage and what each artifact proves

| Date | Artifact | Public contribution | Evidence boundary |
|---|---|---|---|
| 2024-09 | [HybridFlow / veRL](https://arxiv.org/abs/2409.19256) | hybrid controller-worker programming model for RLHF | system paper and repository, not a model recipe |
| 2025-03 | [DAPO](https://arxiv.org/abs/2503.14476) | open 32B math-RL recipe with asymmetric clipping, dynamic sampling, token-global loss | reproducible research recipe, not Seed1.5's full production run |
| 2025-04 | [Value-based Augmented Proximal Policy Optimization (VAPO)](https://arxiv.org/abs/2504.05118) | value-model-based long-chain-of-thought stabilization | actor-critic research recipe |
| 2025-04 | [Seed1.5-Thinking](https://arxiv.org/abs/2504.13914) | 200B/20B MoE; mixed verifiable/preference RL; streaming rollout system | broad industrial disclosure with important scale gaps |
| 2025-04 | [ReTool](https://arxiv.org/abs/2504.11536) | live code-interpreter use learned from answer-only PPO reward | 32B math-tool experiment |
| current artifact | [veRL](https://github.com/volcengine/verl) | maintained implementation used across many open RL projects | moving code; pin a commit and container |
| 2026-06 | [Seed2.1 release](https://seed.bytedance.com/en/blog/seed2-1-officially-released-advancing-ai-productivity) | later agent/coding family; RL across graphical and non-graphical action spaces is stated | mechanism and evaluation disclosure, not a reproducible recipe |
| 2026-07 | [Seed2.0 model card](https://arxiv.org/abs/2607.00248) | Pro/Lite/Mini deployment and agent-capability evaluation | no equivalent training recipe |

### 3.2 Seed1.5-Thinking model and data [D]

Seed1.5-Thinking is a sparse MoE with **200B total parameters and 20B active
parameters per token**. The report does not disclose layer count, expert count,
router layout, pretraining corpus size, optimizer, or pretraining compute.

Its RL task pool has three verifiable branches and one preference branch:

#### Science, Technology, Engineering, and Mathematics (STEM)

1. Begin with several hundred thousand competition-grade mathematics, physics,
   and chemistry questions; mathematics exceeds 80%.
2. Remove incomplete statements, inconsistent notation, and unclear demands.
3. Sample several Doubao-Pro 1.5 answers. Remove a prompt when its worst-of-$N$
   score is 1, meaning every sampled answer succeeds.
4. Detect suspect reference answers when strong models disagree with the
   reference yet agree with one another, or solve with implausibly few reasoning
   tokens; send these cases to human experts.
5. Convert multiple-choice prompts to short-answer/fill-in formats and transform
   suitable answers to integers, reducing guessing and verifier ambiguity.
6. Retain approximately **100K STEM prompts**.

#### Code

Each competition problem must have a complete statement, unit tests, and a
checker. The team performs difficulty filtering and runs generated programs in
an internal high-throughput sandbox. Because submitting every training sample to
the original contest is impractical, the team built an offline evaluator and
reports strong correlation with official verdicts. **[U]** The code-task count,
test-generation method, language mixture, sandbox image, resource limits, and
measured correlation are not published.

#### Logic

The team defines **22 task families**, including 24-point, maze, and Sudoku
problems. Each family has a configurable generator and deterministic verifier;
difficulty is adapted to policy performance. About **10K** generated puzzles are
used. This is a clean example of environment-first data: the generator supplies
effectively unbounded fresh tasks while the verifier supplies exact outcomes.

#### Non-verifiable prompts

Creative writing, translation, knowledge question answering, and role-play
prompts originate from Doubao-1.5 Pro alignment data. Multiple SFT-policy
candidates are scored by a reward model. The team removes low score-variance
prompts and prompts already improved beyond a threshold during earlier Doubao
training. Pairwise generative reward modeling then compares two candidates.
**[U]** Prompt count, human-comparison count, grader size, pair-sampling policy,
and reward calibration remain undisclosed.

### 3.3 Seed verifiers [D]

**Seed-Verifier** takes question, reference answer, and candidate answer and
returns semantic equivalence under human-written principles. **Seed-Thinking-
Verifier** first produces a reasoning trace and was itself optimized as a
verifiable task. The report says it reduces reward hacking, unstable judgments,
and corner-case errors.

On **456 manually annotated hard examples** selected where the simpler verifier
was unstable, Seed-Thinking-Verifier reaches **99.3%** accuracy versus **82.7%**
for Seed-Verifier. This is a targeted hard set, not an unbiased estimate over the
whole training distribution. The thinking verifier also consumes materially more
GPU resources. **[U]** Its model size, training-data count, held-out sampling
rate, and false-positive/false-negative breakdown are not given.

### 3.4 Cold start and SFT [D]

The SFT mixture contains **400K examples**:

- 300K verifiable examples sampled from the RL pool;
- 100K non-verifiable examples covering writing, knowledge, safety, and
  function calling.

Human experts first create only **tens** of cold-start traces through prompt
engineering and interaction with an internal model. A reasoning model trained on
those seeds generates candidates, and Seed-Verifier performs rejection sampling.
The same cold-start-then-reject pattern is applied beyond math.

Training truncates each example to **32K tokens**, runs **two epochs**, and uses
a cosine schedule from peak $2\times10^{-5}$ to
$2\times10^{-6}$. The report explicitly warns that too much non-chain-of-
thought SFT reduced exploration in preliminary experiments.

### 3.5 Seed1.5's value-based RL mechanics [D]

Seed1.5 mixes verifier-scored, reward-model-scored, and hybrid examples in one
framework. Its disclosed stability mechanisms are:

1. **Value pretraining.** Freeze a policy such as $\pi_{\mathrm{SFT}}$, sample
   completions, compute Monte Carlo returns, and fit the value model before joint
   PPO. This aligns the initial critic with the starting policy.
2. **Decoupled GAE.** Use $\lambda_{\mathrm{value}}=1$ for an unbiased value
   target and $\lambda_{\mathrm{policy}}=0.95$ for a lower-variance policy
   advantage.
3. **Length-adaptive GAE.** The paper sets
   $\lambda_{\mathrm{policy}}=1-1/(\alpha l)$,
   where $l$ is response length and $\alpha$ is a hyperparameter, so long and
   short sequences distribute temporal-difference residuals more uniformly.
4. **Dynamic sampling.** Exclude accuracy-0 and accuracy-1 groups.
5. **Clip-Higher.** Decouple PPO's upper and lower clip bounds.
6. **Token-level loss.** Aggregate across eligible tokens rather than first
   normalizing each response.
7. **Positive-example language-model loss.** For successful trajectories,
   $L(\theta)=L_{\mathrm{PPO}}(\theta)+\mu L_{\mathrm{NLL}}(\theta)$,
   adding direct negative-log-likelihood imitation of positive samples.
8. **Online data-distribution adaptation.** Change prompt-domain weights as the
   policy learns to reduce interference among domains and reward scales.

**[U]** The report omits PPO epochs/minibatches, rollout group size, clip values,
$\alpha$, $\mu$, value-loss coefficient, KL coefficient, domain weights,
total RL tasks, RL updates, and stopping rule. DAPO values must not be silently
substituted for these missing production settings.

### 3.6 Seed1.5 systems stack [D]

The training graph uses HybridFlow atop Ray. A single-controller Ray actor owns
the data loader and algorithm; Single Program, Multiple Data (SPMD) worker groups
expose operations such as generation and training. Models are colocated in a
hybrid engine to reuse GPUs across phases.

The **Streaming Rollout System (SRS)** attacks long-tail generation:

- completion ratio $\alpha$ is the fraction completed on-policy by the newest
  policy;
- the remaining $1-\alpha$ portion is continued asynchronously using
  versioned policy snapshots on standalone compute;
- partial samples enter prioritized pools, decoupling policy evolution from
  runtime completion;
- rollout policies can be post-training-quantized to 8-bit floating point
  (FP8);
- generation combines TP, EP, and SP, while training composes TP/EP/CP with
  FSDP;
- a Karmarkar-Karp-style partitioner balances sequence lengths across
  microbatches;
- layer recomputation, activation offload, optimizer offload, automatic
  parallelism search, and ByteCheckpoint support memory and recovery.

The paper reports **3× faster iteration cycles** than synchronous frameworks,
but does not disclose the compared cluster shape, exact baseline configuration,
GPU count, wall time, or total GPU-hours. Treat 3× as a report-specific systems
measurement, not a general veRL speed guarantee.

### 3.7 DAPO: the open critic-free recipe [D]

DAPO means **Decoupled Clip and Dynamic sAmpling Policy Optimization**. Its
primary source is the [paper](https://arxiv.org/abs/2503.14476); the associated
[repository](https://github.com/BytedTsinghua-SIA/DAPO) supplies code and
configurations.

#### Data and starting policy

- DAPO-Math-17K combines public web and competition problems with manual
  annotation.
- Answers are transformed to integers where possible.
- The initial policy is Qwen2.5-32B base.
- **[U]** The paper does not make every raw source record and annotation decision
  reconstructible from prose alone.

#### Objective changes

- remove the KL penalty;
- use asymmetric clip bounds
  $\epsilon_{\mathrm{low}}=0.20$,
  $\epsilon_{\mathrm{high}}=0.28$;
- oversample replacement prompts until a batch contains enough non-all-correct,
  non-all-wrong groups;
- use token-global policy loss;
- drop overlong truncated responses from the loss;
- apply a soft overlong reward before the hard length limit:

$$
r_{\mathrm{len}}(y)=
\begin{cases}
0,&|y|\le L_{\max}-L_{\mathrm{cache}},\\
\dfrac{L_{\max}-L_{\mathrm{cache}}-|y|}{L_{\mathrm{cache}}},
&L_{\max}-L_{\mathrm{cache}}<|y|\le L_{\max},\\
-1,&|y|>L_{\max}.
\end{cases}
$$

The middle region smoothly changes from 0 to -1, avoiding a discontinuous
penalty immediately below the cap.

#### Exact disclosed run

- AdamW, constant learning rate $10^{-6}$;
- 20 rollout steps of learning-rate warmup;
- 512 unique prompts per rollout step;
- 16 responses per prompt, hence 8,192 sampled trajectories before dynamic
  replacement;
- minibatch 512, hence 16 gradient updates per rollout batch;
- expected response limit 16,384 plus 4,096-token overlong cache, maximum
  20,480;
- evaluation temperature 1.0, top-$p=0.7$, average over 32 samples.

The paper's American Invitational Mathematics Examination (AIME) 2024 ablation
progresses from about 30 for naïve GRPO to 36 after overlong filtering, 38 after
Clip-Higher, 41 after soft overlong handling, 42 after token-global loss, and
about 50 after adding dynamic sampling to complete DAPO. Those values are
author-reported and should be read as one cumulative ablation, not independent
universal effects.

### 3.8 ReTool: outcome-only RL for a live interpreter [D]

[ReTool](https://arxiv.org/abs/2504.11536) starts from
Qwen2.5-32B-Instruct. Its cold-start traces come from open math data and are
generated/revised into code-augmented long reasoning examples; SFT runs for two
epochs. **[U]** The paper does not disclose the exact cold-start example count.

During an online trajectory, the policy can emit a `<code>` block. A sandbox
executes it and returns an `<interpreter>` observation, after which the policy
continues. Policy-generated tokens receive gradients; environment output is
masked. PPO receives only a **binary final-answer reward**—there is no explicit
reward for producing executable code or for using the tool.

The main disclosed run uses:

- maximum sequence length 16,384;
- AdamW, learning rate $10^{-6}$;
- minibatch 512;
- KL coefficient 0;
- veRL and a parallel asynchronous sandbox pool;
- 400 RL steps.

The paper reports AIME 2024/2025 scores of 67.0/49.3 after 400 steps, compared
with 40.0/36.7 for text-only RL after more than 1,000 steps; a stronger distilled
backbone reaches 72.5/54.3. These results show that answer-only reward can teach
strategic interpreter use when the environment is in the loop. They do not prove
that every tool call was useful: counterfactual tool ablations would be needed
for that causal claim. **[U]** Group size, hardware, total trajectories, and
sandbox resource limits are not specified.

### 3.9 HybridFlow and veRL: what the system abstraction contributes

The [HybridFlow paper](https://arxiv.org/abs/2409.19256) separates:

- a single-controller program that expresses the RL dataflow; and
- SPMD worker groups that perform distributed model operations.

Its 3D-HybridEngine reshards an actor between training and generation layouts,
and an automatic placement layer maps actor, critic, reference, and reward models
onto resource pools. The reported implementation is built on Ray and integrates
Megatron-LM, PyTorch FSDP, DeepSpeed, and vLLM. The paper reports roughly 12K
lines of Python plus 2.4K lines for the engine and experiments on 16 machines,
with speedups from **1.53× to 20.57×** depending on model and placement.

**[C]** The maintained [veRL repository](https://github.com/volcengine/verl)
has evolved substantially beyond the paper. Reproduction therefore requires a
commit hash, dependency lock, container image, configuration, and launcher—not
the repository name alone. Paper-era speedups cannot be attributed to an
arbitrary current checkout.

### 3.10 Seed2.0 and Seed2.1: capability disclosure is not a full training recipe

The July 2026 [Seed2.0 model card](https://arxiv.org/abs/2607.00248) presents Pro,
Lite, and Mini deployment tiers and extensive language, vision, coding, search,
tool-use, graphical-user-interface, deep-research, context-learning, and
real-world-task evaluations. It also analyzes authorized customer/developer
usage and explicitly acknowledges gaps against other frontier systems in some
coding and multimodal settings.

**[D]** Seed2.0 is positioned for long-horizon real-world complexity and powers
large deployed products. **[U]** The model card does not disclose total/active
parameter counts, architecture tables, pretraining tokens or mixture, SFT counts,
RL task counts, optimizer, rollout policy, group size, grader construction,
cluster, duration, or how the Pro/Lite/Mini checkpoints relate by training or
distillation. Consequently, Seed1.5, DAPO, or ReTool settings must not be
presented as the Seed2.0 recipe.

The June 23, 2026 [Seed2.1
release](https://seed.bytedance.com/en/blog/seed2-1-officially-released-advancing-ai-productivity)
adds one explicit mechanism claim: reinforcement learning guides action
selection across graphical-user-interface and non-graphical action spaces, and
the announcement reports a 16% reduction in average task steps. **[D]** This is
agentic-RL evidence, but it is not a reproducible recipe. The optimizer, task
and trajectory counts, reward construction, action mixture, rollout system,
compute, and relationship among Seed2.1 checkpoints remain **[U]**. Claims that
Seed2.1 participates in data synthesis or RL-framework optimization describe
how the released model is used in research workflows; they do not establish
how Seed2.1 itself was trained.

### 3.11 ByteDance: operational synthesis

The defensible public reconstruction is:

```text
heterogeneous task acquisition
  -> aggressive correctness and difficulty filtering
  -> a small human cold start
  -> rejection-sampled long-CoT SFT
  -> specialist verifiers and preference graders
  -> PPO/value-based RL or critic-free DAPO-style research paths
  -> token-global and dynamic-sampling stability controls
  -> HybridFlow/veRL orchestration
  -> colocated or streaming partial rollouts
  -> online mixture adaptation and held-out evaluation
```

The architecture and most production scale variables after Seed1.5 remain
**[U]**. Public DAPO and ReTool runs are high-value mechanistic case studies,
not drop-in substitutes for the private Seed2.0 or Seed2.1 pipelines.

## 4. NVIDIA Nemotron: from distilled Llama models to multi-environment asynchronous RL

NVIDIA's public record is unusually useful because successive reports expose
different parts of a changing stack: architecture compression, synthetic SFT,
domain-wise RL, generative reward models, open environments, asynchronous
rollouts, and multi-teacher on-policy distillation. The releases are related but
not one uninterrupted training run.

### 4.1 Lineage

| Date | Release or report | Starting point | Principal RL contribution |
|---|---|---|---|
| 2025-05 | [Llama-Nemotron](https://arxiv.org/abs/2505.00949) Nano 8B, Super 49B, Ultra 253B | Llama 3 family transformed by architecture search/distillation | large reasoning GRPO for Ultra; RLOO instruction RL; online/offline preference optimization |
| 2025-05 | [AceReason-Nemotron](https://arxiv.org/abs/2505.16400) 7B/14B | DeepSeek-R1-Distill-Qwen | sequential math-only then code-only strict on-policy RL |
| 2025-05 | [ProRL](https://arxiv.org/abs/2505.24864) 1.5B | DeepSeek-R1-Distill-Qwen-1.5B | more than 2K steps with KL control and reference/optimizer resets |
| 2025-12 | [Nemotron-Cascade](https://arxiv.org/abs/2512.13607) 8B/14B | Qwen3 base | SFT → RLHF → instruction RL → math RL → code RL → SWE-RL |
| 2025-12 | [Nemotron 3 Nano](https://arxiv.org/abs/2512.20848) 30B-A3B | new 25T-token hybrid MoE base | synchronous multi-environment RLVR, GenRM RL, group-relative length control |
| 2026-03 | [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220) 30B-A3B | Nemotron 3 Nano base | broader cascade plus multi-domain on-policy distillation (MOPD) |
| 2026-04 | [Nemotron 3 Super](https://arxiv.org/abs/2604.12374) 120B-A12B | new 25T-token hybrid LatentMoE base | 21-environment async RLVR, SWE-RL, RLHF, PivotRL |
| 2026-04 | [Nemotron 3 Nano Omni](https://arxiv.org/abs/2604.24954) 30B-A3B | Nemotron 3 Nano backbone | staged vision/omni SFT followed by omni-modal RL; derivative rather than a new mainline base |
| 2026-06 | [Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007) 550B-A55B | new 20T-token hybrid LatentMoE base | unified mixed RL, more than ten teachers, two MOPD iterations, one-step-off-policy async infrastructure |
| 2026-07 | [Nemotron-Labs-3-Puzzle-75B-A9B](https://arxiv.org/abs/2607.04371) | compressed Super derivative | Iterative Puzzle, distillation, quantization, MTP, and RL recovery; measured RL-recovery contribution is reported as small |

`A3B`, `A12B`, and `A55B` denote approximate active parameters per forward
pass, not total stored parameters.

Nano Omni and Puzzle are later derivatives, not evidence of a mainline model
after Ultra. Nano Omni extends the Nano backbone with multimodal SFT and RL;
Puzzle compresses Super and reports only a small measured contribution from its
RL recovery stage. **[D]** Ultra therefore remains the latest mainline frontier
Nemotron reasoning release by the cutoff.

### 4.2 Llama-Nemotron: compression, distillation, then selective RL [D]

The [Llama-Nemotron report](https://arxiv.org/abs/2505.00949) transforms Llama
models rather than pretraining a new family from scratch:

- LN-Nano is 8B;
- LN-Super is 49B, derived from Llama-3.3-70B-Instruct;
- LN-Ultra is 253B, derived from Llama-3.1-405B-Instruct.

Puzzle neural architecture search performs block-wise local distillation and
selects a heterogeneous network under latency/memory constraints. Feed-forward
network fusion further compresses LN-Ultra. Recovery training uses 40B
distillation tokens for Super; Ultra uses 65B distillation tokens followed by
88B continued-pretraining tokens.

#### Synthetic reasoning data

The released post-training corpus contains **33,011,757 samples**:

| Domain | Samples | Share |
|---|---:|---:|
| Math | 22,066,397 | 66.8% |
| Code | 10,108,883 | 30.6% |
| Science | 708,920 | 2.1% |
| Chat | 39,792 | 0.12% |
| Instruction following | 56,339 | 0.17% |
| Safety | 31,426 | 0.10% |

Reasoning-on and reasoning-off responses are explicitly tagged. Math problems
come from Art of Problem Solving discussions, exclude proofs/multiple-choice/
binary/invalid prompts, and receive 16 DeepSeek-R1 reasoning generations plus
64 Qwen2.5-Math-7B-Instruct non-reasoning generations. Qwen2.5-32B-Instruct
judges answer equivalence. Code starts from 28,904 deduplicated competition
questions; DeepSeek-R1 samples at temperature 0.6/top-$p=0.95$, and syntax plus
format filtering yields about 488K Python reasoning samples. The broader table
counts repeated and non-reasoning variants, explaining why it is much larger
than the unique-question count.

#### SFT and optimization stages

- Nano: three-stage SFT, batch 256, packed 32K; reasoning-only stage at
  $10^{-4}$ for four epochs, then mixed reasoning modes, then chat/tool data.
- Super: one epoch, 16K, batch 256, fixed $5\times10^{-6}$.
- Ultra: packed 24K, batch 256; warm to $10^{-5}$, cosine to $10^{-6}$,
  10% warmup. The report discloses gradient explosions after epoch one and an
  optimizer-state reinitialization before resuming.

Reasoning GRPO is applied only to LN-Ultra because preliminary smaller-model RL
underperformed distillation. Each rollout batch uses 72 prompts with 16 responses each,
temperature/top-$p=1$, global training batch 576, and two gradient updates per
rollout. Ground-truth equivalence is judged by Llama-3.3-70B-Instruct; a format
reward enforces thinking tags. Prompts solved at least 6 of 8 times by LN-Super
are removed, then a Gaussian curriculum shifts from easier to harder pass-rate
bands. The run consumes about **140K H100-hours**.

The infrastructure colocates vLLM generation and Megatron-LM training across
72 eight-H100 nodes: TP 8 with sequence parallelism, CP 2, PP 18, and data
parallelism 2. Training is BF16 with FP32 optimizer state; online FP8 generation
and CUDA graphs increase reported decoding throughput.

After reasoning RL, Super and Ultra receive less than 120 steps of RLOO on
instruction-following verifiers, batch 128. Preference optimization differs by
size: Super uses two 500-step online RPO iterations on HelpSteer2
(learning rate $4\times10^{-7}$, KL $10^{-5}$, reward scale 3, batch 64);
Ultra uses 30 GRPO steps with 8 responses, learning rate
$3\times10^{-7}$, batch 288, KL $10^{-3}$; Nano uses two offline RPO rounds.

### 4.3 AceReason-Nemotron: sequential domain RL [D]

AceReason isolates verifier latency and domain interference by training math
first and code second. Both 7B and 14B runs start from released
DeepSeek-R1-Distill-Qwen checkpoints and use veRL plus vLLM 0.7.3 on **128 H100
GPUs**.

#### Verifier engineering

- Math: `antlr4-python3-runtime` 4.11.1 and SymPy 1.12 check the boxed answer;
  binary reward only, no format or length reward. A 64-process pool verifies
  1,024 outputs in about 3.9 seconds.
- Code: extract Python after the thinking block and run the complete test set in
  a local LiveCodeBench-style sandbox; reward 1 only if every test passes within
  time. A 64-process pool takes about **552.4 seconds per 1,024 outputs**.

That 140× latency difference motivates separate phases rather than a blended
batch.

#### Math task construction

DeepScaler plus NuminaMath supply candidates. The pipeline applies 9-gram
benchmark decontamination and removes multi-part, multiple-choice, true/false,
proof, non-English, figure-dependent, too-short, and awkward-answer prompts.
DeepSeek-R1 receives up to eight attempts; only majority-verifiable prompts are
kept. Problems solved with fewer than 2,000 reasoning tokens are removed and the
2K–4K band is downsampled. Final count: about **49K**.

Math RL is strictly on-policy: one gradient update after each rollout batch,
which makes the old/current likelihood ratio one at update time. KL and entropy
loss coefficients are zero. Maximum response length follows
8K → 16K → 24K → 32K. Later stages remove prompts solved more than 6 of 16
times. Batch is 128 prompts; group size is 8 at 8K and 16 later; AdamW learning
rate is $10^{-6}$.

#### Code task construction and curriculum

Competition problems include function and standard-input/output formats.
Interactive, special-judge, platform-template, weak-test, and contaminated
problems are removed. DeepSeek-R1-671B samples eight answers to assign difficulty
0–8; all-fail level-8 problems are excluded. URL and n-gram deduplication leave
**8,520 problems**.

Code stage 1 uses difficulty ≤5 for 7B and ≤7 for 14B, 24K responses,
temperature 0.6, and 8 rollouts. Stage 2 uses all retained problems, 32,768
tokens, epoch-wise removal of newly easy tasks, and ramps temperature 0.6→1.0
and rollouts 8→16. Batch is 128 and AdamW learning rate
$5\times10^{-6}$.

The final curriculum is math 8K→24K, code 24K→32K, then math at 32K. The paper
reports that math-only RL improves both AIME and LiveCodeBench, while subsequent
code RL adds code gains with little math regression. It also reports that
deliberately introduced false-positive or false-negative code rewards produce
early suboptimal convergence or collapse. This is direct evidence that test
quality can dominate optimizer choice.

### 4.4 ProRL: prolonged training, KL, and resets [D]

[ProRL](https://arxiv.org/abs/2505.24864) asks whether RL only sharpens solutions
already sampled by a base model. Its 1.5B model trains for more than 2K steps on
**136K verifiable tasks**:

| Domain | Count | Reward |
|---|---:|---|
| Math / DeepScaleR | 40K | binary answer verification |
| Code / Eurus-2-RL | 24K | fraction of tests passed; zero on compile/syntax/5-second timeout |
| STEM / filtered SCP-116K | 25K | binary answer verification |
| Logical puzzles / Reasoning Gym | 37K | task-specific continuous reward |
| Instruction following / Llama-Nemotron | 10K | continuous rule reward |

SCP data is filtered from 274K candidates to 25K by requiring a retrievable
source answer and GPT-4o agreement between DeepSeek-R1 response and ground truth.
Reasoning Gym contributes 96 generated task types and a separate 9,600-example
validation set.

The optimizer is DAPO-enhanced GRPO with KL regularization. Each prompt gets 16
responses at temperature 1.2; context is reported as 8,096 tokens; prompt batch
256, minibatch 64, hence four gradient updates per rollout; constant AdamW
learning rate $2\times10^{-6}$. Training uses **48 nodes × 8 H100-80GB = 384
GPUs** and about **16K GPU-hours**.

As the KL term becomes restrictive or validation stalls, the reference policy
and optimizer are hard-reset to a newer checkpoint. The report describes eight
run segments: initial four-domain training; reset; addition of instruction data;
two runs adding non-termination penalties; two runs increasing rollouts 16→32
with resets; and a final roughly 200-step 16K stage with 16 rollouts. These are
manual, validation-triggered interventions—not a fully specified automatic
algorithm.

The paper's pass@$k$ analysis is nuanced: some high-baseline math tasks improve
pass@1 while pass@128 declines; some tasks plateau; code and selected logic tasks
continue improving at high $k$. Claims of “novel reasoning” rely on those
distribution shifts, held-out complexity generalization, and corpus-overlap
analysis. They are evidence against a universal “RL only reweights” claim, not a
proof that every generated reasoning token is semantically novel.

### 4.5 Nemotron-Cascade 1: domain-wise stages instead of one mixture [D]

[Nemotron-Cascade](https://arxiv.org/abs/2512.13607) starts from Qwen3-8B-Base
and Qwen3-14B-Base and uses the order:

```text
two-stage SFT
  -> RLHF
  -> instruction-following RL
  -> math RL
  -> code RL
  -> software-engineering RL
```

The design premise is operational: math, code execution, preference scoring,
and repository environments have different response lengths and verifier
latencies. Sequential stages are easier to tune and schedule than one mixed
batch. Earlier capabilities are evaluated after every stage.

#### SFT scale

- Math stage 1: 353K prompts → 2.77M samples, 7.8 responses/prompt.
- Math stage 2: 163K prompts → 1.88M samples, 11.5 responses/prompt.
- Code stage 1: 172K prompts → 1.42M samples, 8.3 responses/prompt.
- Code stage 2: 79K prompts → 1.39M samples, 17.6 responses/prompt.
- Tool conversations expose 4.4 tools on average.
- SWE data: 127K repair, 92K localization, and 31K test-generation samples.

DeepSeek-R1 and R1-0528 are major SFT teachers. Stage 1 focuses on up to 16K;
stage 2 expands to 32K and adds tool/SWE data. Parallel reasoning-on and
instruction-mode responses let the unified 8B checkpoint switch modes.

#### RL mechanics and exact stages

Cascade uses strict on-policy GRPO: sample from the current policy, perform one
gradient update, set the likelihood ratio to one, omit KL, and use token-global
loss.

- Reward model: Qwen2.5-72B-Instruct initialization, 82K preference pairs,
  Bradley-Terry scalar training, batch 256, learning rate
  $2\times10^{-6}$.
- RLHF: approximately 12K curated in-distribution prompts; 8 responses,
  temperature 0.6/top-$p=.95$, learning rate $2\times10^{-6}$, 12K max
  response. Math/code prompts are excluded because reward-model out-of-
  distribution errors destabilized early runs.
- Instruction-following RL: batch 128×8, temperature 0.6, top-$p=.95$,
  top-$k=20$, learning rate $2\times10^{-6}$.
- Math RL: 18K AceReason-derived problems; batch 128×8, temperature 1,
  top-$p=.95$, learning rate $2$ or $2.5\times10^{-6}$, dynamic
  difficulty re-sampling.
- Code RL: 9.8K test-hardened problems; batch 128×8, temperature 1,
  top-$p=.95$, learning rate $4\times10^{-6}$. Asynchronous verification
  reduces reported batch verification from 1,172.4 seconds to a much smaller
  critical-path cost; the speedup is pipeline-specific.
- SWE-RL: final repository-focused stage after code expertise; environments and
  execution-free reward-model ablations are reported separately.

The ordering is an empirical result, not a theorem. RLHF unexpectedly improves
reasoning efficiency before math/code RL, while instruction RL temporarily
reduces entropy and response length and is then recovered by math RL.

### 4.6 Nemotron 3 Nano: a new hybrid MoE base and unified environments [D]

Nemotron 3 Nano has **31.6B total, 3.2B active parameters** (3.6B including
embeddings), 52 layers, width 2,688, 32 query/2 key-value heads, Mamba-2 state
128, 128 routable experts, 6 activated experts, and 2 shared experts. It is
pretrained on **25T tokens**: 23.5T broad then 1.5T higher-quality. Context is
extended to 1M.

SFT trains over **18M samples** with the published blend: chat 28.6%, code
20.7%, science 12.8%, math 9.9%, math-with-tools 4.9%, multilingual 7.4%, plus
SWE, formal proof, terminal, conversational-agent, long-context, and generative-
selection data. It runs 13,000 steps, batch 64, packed 256K, learning rate
$5\times10^{-5}$, 800 warmup steps. Ten percent of traces lose reasoning for
reasoning-off control; 3% are truncated to teach token budgets.

The first and second RLVR stages jointly mix environments rather than cascading
them. Disclosed task pools include:

- 17K DAPO and 104K Skywork math tasks;
- 22K coding tasks after limiting tests to 50;
- 135K document-grounded STEM question-answering tasks;
- 9K exact JSON-schema tasks;
- 46K rule-based and 3K multi-turn judged instruction tasks;
- 12K long-context multi-document tasks;
- Workplace Assistant: five databases, 26 tools, 690 tasks, verified by final
  database state;
- about 1K banking conversations verified by executed state changes.

Prompts already at 100% SFT pass rate are removed. Within each domain, a
Gaussian target over pass rate shifts from easier to harder while preserving a
fixed domain ratio.

Synchronous GRPO uses 128 prompts ×16 responses = 2,048 trajectories, batch
2,048 and one on-policy update. It applies masked importance sampling for
rollout/training log-probability mismatch, freezes MoE router weights while
continuing expert-bias balancing, caps generation at 49K, and uses overlong
filtering.

For RLHF, NVIDIA first trains Qwen3-235B-A22B-Thinking as a GenRM with GRPO:
128 prompts ×8 generations and one update. Its reward penalizes format errors
and absolute errors in two helpfulness scores and one pair ranking. Candidate
positions are swapped to reduce positional bias. Policy RLHF then samples 16
responses per 128 prompts. Circular comparisons require 16 GenRM calls rather
than all $\binom{16}{2}=120$ pairs.

To prevent verbosity, group-relative length control adds centered bonuses for
shorter reasoning and answer components:

$$
R_i=R_i^{\mathrm{base}}
 +\lambda_{\mathrm{think}}\widetilde w_i^{\mathrm{think}}
 +\lambda_{\mathrm{answer}}\widetilde w_i^{\mathrm{answer}},
$$

with both $\lambda=0.5$, plus 0.5 bonuses for shortest responses above the
80th-quality percentile. The report says verbosity falls 30% without measured
accuracy loss.

### 4.7 Nemotron-Cascade 2: cascade plus on-policy distillation [D]

[Cascade 2](https://arxiv.org/abs/2603.19220) is a **30B total/3B active** MoE
derived from Nemotron 3 Nano base. Its one-stage SFT packs sequences to 256K for
roughly 1.5 epochs. The disclosed corpus is unusually large:

- math: 1.8M Python-tool traces and 2.6M no-tool traces, including 676K
  generation-selection samples;
- competition code: 165K unique prompts; 1.9M Python traces, 1.0M C++14 traces,
  and 1.3M Python-tool traces;
- 1.1M scientific-code and 2.7M science samples (1.4M added for Cascade 2
  plus 1.3M inherited from Nano);
- 160K long-context examples averaging about 128K, plus 74K additional long
  examples;
- 4.9M reasoning-on and 372K reasoning-off examples, plus 4.6M inherited
  reasoning-on chat samples and about 700K multi-turn conversations;
- 822K conversational-tool trajectories;
- 389K agentless and 125K agentic SWE examples;
- about 490K terminal examples: 162K math, 32K code, 32K SWE adaptations,
  120K seed-based tasks, and 140K skill-based tasks.

The report itself has an unresolved chat-data arithmetic inconsistency: it
states 372K reasoning-off samples but separately describes response sources of
300K and 330K. **[D]** This chapter preserves that discrepancy rather than
silently choosing a total.

The post-training order is:

```text
SFT -> instruction RL -> mixed multi-domain RL -> MOPD -> RLHF
    -> long-context RL -> code RL -> agentless SWE-RL
    -> execution-based agentic SWE-RL
```

Strict on-policy GRPO through NeMo RL uses one update. Most stages omit KL, but
the RLHF stage explicitly uses a KL-loss coefficient of 0.03. Disclosed
settings include:

- instruction RL: 128 prompts ×16, temperature 1/top-$p=1$, learning rate
  $3\times10^{-6}$;
- mixed multi-domain RL: roughly 55% multiple-choice question answering, 30%
  agentic tools, and smaller other domains; 128×16, learning rate
  $3\times10^{-6}$;
- RLHF: 128×16, learning rate $3\times10^{-6}$, KL coefficient 0.03;
- long-context RL: 128×16, learning rate $3\times10^{-6}$, no reported KL
  penalty;
- hard code RL: 3.5K tasks, 128×16, learning rate
  $3\times10^{-6}$;
- agentless SWE: 128×16 = 2,048 rollouts, maximum 98,304 tokens;
- execution SWE: 16 prompts ×64 rollouts, maximum 256K tokens.

**Multi-Domain On-Policy Distillation (MOPD)** samples a trajectory from the
student and matches the relevant domain teacher on those exact student states.
Its ideal negative reverse-KL sequence objective can be written

$$
J_i(\theta)=
\mathbb E_{y\sim\pi_\theta(\cdot\mid x),x\sim D_i}
\sum_t\left[
\log\pi_{T_i}(y_t\mid s_t)-\log\pi_\theta(y_t\mid s_t)
\right].
$$

The implemented loss distinguishes the inference sampler
$\pi_{\mathrm{inf}}$ from the train-time policy $\pi_{\mathrm{train}}$,
applies the ratio $\pi_{\mathrm{train}}/\pi_{\mathrm{inf}}$, and retains
ratios only in $[0.5,2.0]$, together with a valid-token mask. **[D]** The math
teacher is the initial SFT checkpoint; the other teachers are an RLHF-from-SFT
checkpoint and the post-instruction-RL/multi-domain checkpoint. The default
batch is 128 prompts ×4 rollouts =512; later 512×1 is reported more stable. The
learning rate warms from $2\times10^{-7}$ to $2\times10^{-6}$ over 30
steps, with typical convergence in 40–50 steps. MOPD is not offline teacher
imitation: the student chooses the state distribution, reducing the support
mismatch of a fixed teacher corpus.

### 4.8 Nemotron 3 Super: 120B-A12B and asynchronous multi-environment RL [D]

Super has **120.6B total and 12.7B active** parameters (12.1B excluding
embeddings), 88 layers, width 4,096, 32 query/2 key-value heads, Mamba state
128, and LatentMoE layers with 512 experts, top-22 routing, latent width 1,024,
and two shared-weight MTP layers.

Pretraining uses 25T tokens, 8,192-token sequences, batch 3,072
(about 25.17M tokens/batch), AdamW $(\beta_1=.9,\beta_2=.95)$, weight decay
.1, and a Warmup-Stable-Decay schedule: 200B-token warmup to
$4.5\times10^{-4}$, then stable, then minus-square-root decay over the final
5T to $4.5\times10^{-6}$. A 34B-token 1M-context phase is followed by 17B
tokens alternating 1M and 4K sequences.

#### SFT and agent data

Two-stage SFT consumes **more than 7M samples / about 80B tokens**. Stage 1 uses
token-global loss, packed 256K, batch 64, learning rate $10^{-5}$. Stage 2
switches to equal per-conversation loss, packed 512K, batch 32, same learning
rate, specifically to restore long-input/short-output performance.

Important agentic subsets include:

- specialized conversational tools: a six-stage domain→policy/tool→scenario→16
  rollouts→outcome/process judgment→difficulty-selection pipeline, yielding
  **279,116 conversations across 838 domains**;
- general tools: user, assistant, and simulated-tool LMs plus turn/trajectory
  judges, yielding about **1.5M trajectories**;
- terminal: **84,864** samples—68,924 synthetic, 8,125 math-derived, 7,815
  code-derived—generated with DeepSeek-V3.2 in Docker through Terminus 2;
- search: Wikidata 4–8-hop random walks are rewritten into obfuscated questions;
  MiniMax-M2 solves them with Tavily search, averaging 12 tool calls per trace;
- text-to-SQL: 96.5K validated MySQL/PostgreSQL/dialect-spanning records;
- CUDA: 100K generation/repair samples;
- real GitHub issue trajectories under several agent harnesses.

#### Post-training stages

1. multi-environment RLVR described in prose as 21 environments and 37 RL
   datasets, while Figure 12 instead labels 37 “environment types” and
   25/30/26 types across three rounds;
2. isolated end-to-end SWE-RL, because repository rollouts are far longer and
   slower than other environments;
3. principle-following GenRM RLHF;
4. frozen-main-model MTP healing on generated responses.

RLVR removes prompts the SFT model always solves, shifts a difficulty curriculum,
and includes math with/without Python, proof verification, competition code,
single-patch SWE, STEM, structured instruction following, jailbreak and
over-refusal tasks, long context, conversational tools, terminal, and Reasoning
Gym. Low-effort prompts receive a correctness-plus-length reward and form 2%,
later 1%, of the mixture.

Async GRPO samples **256 prompts ×16 responses = 4,096**, trains one full-batch
update, and increases maximum generation from 49K to 64K. Learner and inference
devices are disaggregated; rollout workers are at most one step behind. Weights
can change mid-rollout without recomputing the KV cache, so one trajectory may
contain tokens from different policy versions. Importance ratios computed from
training and inference log probabilities are masked to stabilize this mismatch.

**PivotRL** supplements expensive end-to-end agent RL by reusing offline expert
SFT trajectories. It identifies assistant turns where the policy is uncertain
and rewards actions semantically close to the expert under a domain-specific
metric. Super applies it to programming, search, terminal, and conversational
tools. At the report date, a full standalone method description was still
forthcoming, so its exact objective is **[U]**.

NeMo RL, NeMo Gym, Megatron-Core, vLLM, Ray, and Slurm share one cluster. NeMo
Gym separates agent, model, and resource/verifier servers. SWE rollouts launch
Apptainer repositories under OpenHands, with OpenCode- and Codex-style agent
interfaces and binary test reward. At roughly 1K-GPU scale, the report documents
port time-of-check/time-of-use races and hardware failures—valuable evidence
that resilience and startup paths are part of the algorithm's usable scale.

### 4.9 Nemotron 3 Ultra: 550B-A55B, mixed RL, and specialist MOPD [D]

Ultra has **550B total/55B active parameters**, 108 layers, width 8,192, 64
query/2 key-value heads, Mamba state 128, 512 LatentMoE experts with top-22
routing, expert hidden size 5,120, shared expert size 10,240, latent width 2,048,
and two shared MTP heads.

#### Pretraining and precision

- 20T tokens under Warmup-Stable-Decay: 15T broad coverage plus 5T higher
  quality;
- 200B-token warmup to $2.5\times10^{-4}$, final 5T decay;
- code refresh includes 173B fresh GitHub tokens through 2025-09-30; a
  separate list covers 11 **natural** languages, not programming languages;
- NVFP4 E2M1 training with two-dimensional scaling blocks;
- higher precision for the final 15% of the network—the final 16 layers—plus
  embeddings, MTP components, and selected sensitive projections;
- BF16 branches at 5T, 10T, and 16T show less than 0.4% average relative loss
  gap, but switching to BF16 did not resolve the divergence; NVIDIA rewound
  near 15T and shortened the intended 25T horizon to 20T;
- long-context continuation to 1M.

#### SFT

Stage 1 packs to 294,912 tokens, batch 64, 204,800 samples, peak learning rate
$1.5\times10^{-5}$, floor $10^{-6}$. Stage 2 packs to about 515K tokens,
adds long-context examples up to 512K tokens, and trains on 19,200 samples.
Search SFT includes more than 97K OpenResearcher trajectories generated against
a 15M-document offline browser corpus; licensing filters reduce the releasable
subset to 21.7K. Other first-party data covers search, terminal, office, tools,
and SWE.

#### Unified RL and MOPD

Unlike domain-wise Cascade, Ultra returns to unified mixed RLVR across terminal,
office, SWE, search, conversational tools, mathematics, code, STEM, safety,
chat, instruction following, and long-context question answering. Async GRPO
uses a **global batch of 8,192 trajectories/responses with 16 rollouts per
prompt**; 8,192 is not the prompt count. Generation grows from 48K to 64K.

More than ten specialist teachers cover SWE, office, search, conversational
tools, terminal, agentic safety, STEM, chat, instruction following, coding, and
usability. A light SFT warmup first moves the student toward each teacher's
support. Two MOPD iterations then optimize the student-state reverse-KL mixture:

$$
J(\theta)=\sum_i\lambda_i
\mathbb E_{x\sim D_i,\,y\sim\pi_\theta}
\sum_t
\left[
\log\pi_{T_i}(y_t\mid s_t)-\log\pi_\theta(y_t\mid s_t)
\right].
$$

MOPD uses a maximum generation length of 192K tokens and 1,024 prompts ×1
rollout. In the asynchronous
implementation, a dense token advantage is
$\operatorname{stopgrad}(\log\pi_T-\log\pi_{\mathrm{prox}})$, behavior
correction is $c=\pi_{\mathrm{prox}}/\pi_{\mathrm{beh}}$, and learner update
ratio is $r=\pi_\theta/\pi_{\mathrm{prox}}$, followed by PPO-style clipping
and token masks.

#### Scale and failure accounting

Training uses GB200-class infrastructure under Slurm, with more than 3K GPUs in
reported post-training jobs. One-step-off-policy rollouts and in-flight updates
keep generation active. Five-token MTP speculative decoding gives a reported
1.46× rollout speedup.

The report usefully classifies **RL software failures** as 56%
generation/timeouts, 36% sandbox/tool, and 8% other; this is not necessarily a
distribution over failed rollouts. Engineering changes reduce Ray Global
Control Store startup from more than 30 to about 10 minutes, checkpoint work from 60 seconds
to under 1 second, just-in-time compilation from 38.8 to 0.4 minutes, and vLLM
startup from 25 to 9.5 minutes. These are vendor-specific measurements, but they
show why fault taxonomy belongs beside the loss curve.

### 4.10 What changed across Nemotron generations

| Dimension | Llama-Nemotron | Cascade / Nano | Cascade 2 / Super | Ultra |
|---|---|---|---|---|
| Base | compressed Llama | Qwen3 or new 30B-A3B | new 30B/120B MoE | new 550B-A55B |
| Main organization | specialist SFT; selective RL | sequential cascade or synchronous mixture | cascade+MOPD or async mixture | unified mixed RL + two MOPD rounds |
| Reward | answer judge, format, preference RM | rules/tests/GenRM/database state | many verifiers, GenRM, SWE tests | many specialized verifiers and teachers |
| Freshness | colocated rollout/train | strict on-policy or synchronous | strict cascade or one-step async | one-step async with behavior correction |
| Agent scope | limited tool/function data | tool/SWE environments emerge | terminal/search/conversational/SWE | office/search/terminal/SWE/safety and more |
| Reproducibility | weights, data, code | unusually detailed reports and data | more open environments/checkpoints | broad recipe but frontier infrastructure still incomplete |

The directional lesson is not “always cascade” or “always mix.” Cascade reduces
heterogeneous scheduling and interference; unified RL can mitigate forgetting
by showing every domain to each update **[I]**; MOPD recovers specialist regressions after
either. The correct choice depends on verifier latency, environment reliability,
teacher availability, and the model's cross-domain interference.

## 5. Microsoft: separating the agent runtime from the learning loop

Microsoft's public work is useful for two different reasons. Agent Lightning is
an abstraction and systems contribution: it asks how an arbitrary existing
agent can become an RL environment without being rewritten around a trainer.
rStar2-Agent is a concrete, unusually well specified math-and-Python training
run. The former should not be mistaken for a frontier-model recipe, and the
latter should not be generalized to every kind of agent.

### 5.1 Agent Lightning: the agent as a partially observable process

Primary sources: the [Agent Lightning paper](https://arxiv.org/abs/2508.03680)
and [official repository](https://github.com/microsoft/agent-lightning).

#### Mathematical interface

The paper first describes agent execution in Markov-decision-process language,
but its formal tuple is a partially observable Markov decision process (POMDP)

$$
\mathcal{M}=(\mathcal{S},\mathcal{O},\mathcal{A},P,R),
$$

where $s\in\mathcal S$ is the complete runtime state, $o\in\mathcal O$ is
the information exposed to the language model, $a\in\mathcal A$ is one model
call's complete output, $P$ is the transition induced by agent code and its
environment, and $R(s,a)$ is the scalar reward function. **[D]** This is a
better model than
declaring the concatenated chat transcript to be one flat action: branching,
retries, tool-side state, and messages between agents may affect the next call
without appearing in the current model input.

An execution trace can be represented as a graph of spans rather than only a
list of tokens:

$$
\tau=\{(o_i,a_i,r_i,\text{parent}_i,\text{metadata}_i)\}_{i=1}^{m}.
$$

The parent relation preserves nested agent and tool calls. The trainer's first
job is therefore **credit-assignment data transformation**: select trainable
LLM-call spans, associate rewards with them, and convert their messages and
outputs into token-level samples. **[D]**

LightningRL defines two levels of attribution:

1. allocate an episode or intermediate return across trainable LLM calls; then
2. allocate each call-level return across the generated tokens used by PPO,
   GRPO, or another policy-gradient backend.

The paper's implemented default assigns the same terminal episode return to
each selected action before token-level optimization. **[D]** This is an
important limitation, not a solved credit-assignment result: if a five-call
agent succeeds despite a harmful second call, that second call still receives
positive return. Automatic Intermediate Rewarding (AIR) provides hooks that
convert system-monitoring signals into denser intermediate rewards. **[D]** The
paper does not specify one universal learned grader or calibration recipe;
those are task-specific implementation choices, not disclosed production-model
evidence.

#### Training-Agent Disaggregation

The central systems decision is to keep the production-like agent and the RL
trainer in separate processes:

```text
dataset/task scheduler
        |
        v
agent runners ---> instrumented spans ---> LightningStore ---> RL algorithm
     ^                    |                      |                  |
     |                    v                      v                  v
 environment       trace inspection       sample assembly    new weights
     |                                                           |
     +---------------------- inference service <-----------------+
```

**[D]** The agent can retain its existing framework, control flow, Python
dependencies, and external services. OpenTelemetry-compatible tracing records
model prompts, model outputs, tools, rewards, and parent-child relationships.
The store mediates tasks, traces, and model resources. A client/server boundary
lets agent runners scale independently from policy learners.

This separation solves an integration problem, not the semantic problems of
RL. A correct deployment still needs:

- stable task and environment versions;
- idempotent retries, because a repeated tool call may mutate state;
- explicit attribution when several agents use the same model;
- secure handling of arbitrary code and credentials;
- token IDs or a proven retokenization contract between inference and training;
- a definition of whether timed-out, cancelled, and partially completed traces
  are negatives, censored data, or excluded samples.

The paper evaluates the framework on Spider text-to-SQL, MuSiQue
retrieval-augmented generation, and Calc-X mathematical tool use, and reports
continuous improvement under RL. **[D]** These experiments support the
framework's portability claim. They do not disclose the data, compute, or
algorithm for a Microsoft frontier model, and should not be cited as such.

The live repository has evolved substantially beyond the paper, including
different stores, tracers, algorithms, and integrations. **[C]** Reproduction
must pin a release or commit; “Agent Lightning main” is not a stable experiment
identifier.

### 5.2 rStar2-Agent: a complete math-plus-Python loop

Primary sources: the [rStar2-Agent technical
report](https://arxiv.org/abs/2508.20722) and [official rStar
repository](https://github.com/microsoft/rStar).

rStar2-Agent starts from **Qwen3-14B-Base**, not an already instruction-tuned
reasoner. **[D]** It develops a structured tool-action language with
non-reasoning SFT, then uses multi-stage agentic RL. Its reported production run
uses 64 AMD MI300X GPUs for about one week and reaches 510 RL updates. **[D]**

#### Stage A: non-reasoning SFT creates the action protocol

The SFT mixture contains three disclosed components:

| Component | Count | Construction |
|---|---:|---|
| Function calling | 165K | 117K from ToolACE, APIGen, and Glaive plus 48K Magicoder-style JSON examples |
| Instruction following | 30K | Tulu-3 prompts rewritten with o4-mini |
| General conversation | 27K | Llama-Nemotron chat data rewritten with o4-mini |

**[D]** Training runs for three epochs with AdamW, peak learning rate
$5\times10^{-6}$, 4% warmup, cosine decay, and global batch size 128. The
targets are intentionally concise rather than long hidden chains of thought.
This stage teaches valid JSON/tool syntax and ordinary assistant behavior; it
does not teach the final reasoning policy by imitation.

#### Stage B: construct a solvable but nontrivial RL pool

The initial candidate pool exceeds 100K problems:

- about 17K DAPO math problems;
- about 93K problems from Art of Problem Solving and OpenMathReasoning; and
- 937 Project Euler problems.

**[D]** For each OpenMathReasoning problem, Qwen3-32B samples 16 responses; the
problem is retained only if at least two sampled integer answers match its
existing labeled answer. The sampled agreement is a validation filter, not the
source of a new pseudo-label. Filtering and deduplication leave roughly **42K**
first-stage RL problems. This is curriculum construction by empirical
solvability, but repeated teacher agreement still does not prove that the
original label is correct. **[I]** A reproduction should independently verify
labels where possible and log the match count, teacher version, prompt,
decoding settings, and every discard reason.

#### Stage C: make Python an environment action

The policy emits a structured JSON call containing Python code. A code-judge
service executes it in isolation and returns the observation to the model. The
reported service sustains about **45K concurrent calls** with approximately
**0.3-second average execution latency**. **[D]** Throughput comes from
decoupling execution from GPU generation; it does not make arbitrary code safe.
The official repository explicitly warns operators to isolate the judge and not
expose it to untrusted networks. **[C]** Production-grade replication also
needs CPU, memory, file, process, syscall, network, and wall-clock limits plus a
clean filesystem per call.

At the audited repository revision, every assistant-generated token—including
the structured call and Python code—receives response mask 1, while inserted
tool-response tokens receive mask 0. **[C]** Thus generated code remains a
trainable policy action; only environment output is excluded. Masking the code
itself would prevent direct reinforcement of better programs. Implementations
must reproduce the artifact's exact mask, not rely on the ambiguous phrase
“mask tool tokens.”

#### GRPO-RoC: resample on correct

Ordinary GRPO samples a group of $G$ trajectories and normalizes their
rewards. rStar2 uses **Resample-on-Correct (RoC)**. It first oversamples
$2G=32$ rollouts, then retains $G=16$:

1. preserve negative trajectories through uniform sampling, up to half of the
   retained group;
2. score positive trajectories using tool-error frequency and final-answer
   formatting quality; and
3. sample the retained positive half with probability favoring lower-penalty
   successful trajectories.

**[D]** The key design is asymmetric. Failures preserve exploration coverage;
successful but operationally messy trajectories can be replaced by cleaner
successful ones. The retained sample is no longer an unbiased draw from the
behavior policy, so this is a deliberate data-selection heuristic rather than
textbook on-policy GRPO. **[I]**

The optimization reward remains binary final-answer correctness. RoC separately
scores already successful trajectories for *selection*: the tool-error score is
0.5 when there is no tool call and otherwise equals erroneous calls divided by
all calls; the format score is 1 when there is no answer tag and otherwise is
$\min(1,(n_{\text{answers}}-1)/n_{\text{turns}})$. Their sum controls
inverse-penalty sampling of half the positive trajectories. **[D]** These are
not additive or multiplicative reward penalties. Conflating rollout selection
with reward shaping changes the algorithm.

The policy objective uses token-global aggregation, no explicit KL penalty,
PPO-style lower/upper clip bounds 0.20/0.28, and no entropy bonus. **[D]** In a
schematic notation,

$$
L=-\frac{1}{\sum_i |M_i|}
\sum_{i,t}M_{i,t}\min\!\left(
r_{i,t}\widehat A_i,
\operatorname{clip}(r_{i,t},0.8,1.28)\widehat A_i
\right),
$$

where $M$ is the trainable-token mask. The actual asymmetric clipping must be
implemented exactly; the equation emphasizes that the denominator is the total
eligible-token count rather than a mean of per-response means.

#### The three RL stages

All stages use learning rate $10^{-6}$. **[D]** The disclosed progression is:

| Stage | Prompt pool | Maximum response length | Updates | Operational purpose |
|---|---:|---:|---:|---|
| RL-1 | 42K | 8K tokens | 300 | learn the tool loop on broad, easier data |
| RL-2 | same 42K pool | 12K tokens | 85, reaching 385 total | extend the length budget without resetting the first run |
| RL-3 | 17.3K hard problems | 12K tokens | 125, reaching 510 total | reset optimizer/reference, concentrate on unsolved prompts |

The third-stage pool is built by sampling eight responses and removing
all-correct problems; the resulting set is approximately 17.3K examples.
**[D]** The optimizer and reference policy are reset at the stage boundary.
This makes “510 steps” incomparable to a single stationary 510-step run: data,
length budget, reference, and optimizer state all change.

The report gives a global rollout prompt batch of **512** in the production
configuration. **[D]** Together with 16 retained trajectories per prompt, long
responses, and 64 MI300X devices, this explains why rollout and sandbox
services—not just matrix multiplication—are central to the compute budget.

#### What failed

The report records several negative results that are more instructive than the
headline benchmark:

- filtering overlong responses caused the policy to generate **more** truncated
  responses, because useful hard examples disappeared from the update;
- an n-gram repetition penalty also punished legitimate repeated verification
  and code patterns; and
- continuing beyond the reported 510 updates led to performance collapse.

**[D]** These results refute three naive rules: “drop every truncation,” “all
repetition is bad,” and “more RL is always better.” A run needs truncation rate,
answer correctness conditioned on length, tool-error categories, entropy,
clip fraction, and held-out performance as early-stop signals.

The vendor-reported average pass@1 values are **80.6% on AIME 2024** and
**69.8% on AIME 2025**. **[D]** Those numbers depend on the report's prompt,
tool access, sampling protocol, judge, and averaging. They are not directly
comparable to no-tool results or different sample budgets.

#### Paper-to-code audit: runnable is not identical to production

At audited revision
[`ecbfb943e202b4ed017d2d35f2029917b27db4cd`](https://github.com/microsoft/rStar/tree/ecbfb943e202b4ed017d2d35f2029917b27db4cd),
the public recipe has been migrated to a newer veRL branch. **[C]** The README
states that the released migrated setup was validated for the first 50 steps,
whereas the full model used a customized older veRL 0.2-era stack. **[C]** The
demonstration uses DAPO-17K, eight A100/H100 GPUs, and global training batch 128,
not the report's full 42K curriculum, 64 MI300X production cluster, and batch
512. **[C]** The complete multi-stage scheduler and all service topology are
not present as a one-command replay. **[U]**

Therefore there are three distinct claims:

1. the **paper recipe** describes the reported 510-step model;
2. the **released demo** exercises the core algorithm and code-judge path on a
   smaller configuration; and
3. the **released weights** permit outcome evaluation.

Conflating them would overstate reproducibility. The correct experimental
report should name which of the three was reproduced and record the code
revision, submodule revisions, container images, and judge configuration.

### 5.3 Microsoft synthesis

Agent Lightning and rStar2 expose complementary layers:

| Layer | Agent Lightning | rStar2-Agent |
|---|---|---|
| Primary question | How can arbitrary agent executions become trainable traces? | How can a 14B policy learn math reasoning with Python feedback? |
| State representation | traced POMDP / span graph | ordered math-and-code turns |
| Credit | hierarchical adapter; optional intermediate rewards | terminal correctness plus operational penalties |
| Infrastructure | training-agent disaggregation and trace store | high-concurrency isolated code judge plus veRL |
| Main gap | coarse default action attribution | production/demo artifact mismatch and narrow domain |

The reusable lesson is to formalize the call graph before choosing the
optimizer. If the trace adapter assigns reward to the wrong action, a more
sophisticated policy-gradient estimator only optimizes the wrong sample more
precisely.

## 6. xAI and the Grok line: rapid RL scaling, sparse recipe disclosure

xAI provides a revealing counterpoint to the more reproducible open reports.
Its official releases document a clear progression—from feedback fine-tuning,
to reasoning RL, to native tool-use RL, to long-horizon asynchronous
software-engineering RL—but usually omit the loss, optimizer, data counts, and
system topology needed to replay a run. This section records that boundary
explicitly. **No public statement that a Grok model used “RL” establishes that
it used PPO, GRPO, or any other named optimizer.**

### 6.1 Model-by-model evidence ledger

| Release | Publicly disclosed training signal | Agentic-RL interpretation | Critical unknowns |
|---|---|---|---|
| Grok-1 (2023; open base in 2024) | next-token pretraining; release model fine-tuned from human and Grok-0 feedback | feedback post-training disclosed, but no agent environment recipe | alignment algorithm, corpus size, reward model, optimizer |
| Grok-1.5 / 1.5V (2024) | capability and infrastructure description | no agentic-RL mechanism disclosed | nearly the entire post-training recipe |
| Grok-2 / 2 mini / 2-1212 (2024) | capability, product, and benchmark disclosures | no specific RL recipe disclosed | architecture, scale, data, rewards, algorithm |
| Grok-3 / 3 mini Think (2025) | large-scale RL for reasoning; code and Internet tools in product | reasoning RL disclosed; tool availability is not proof the tool loop was in RL unless stated | tasks, verifier, optimizer, batch, compute allocation |
| Grok-4 / Heavy (2025) | verifiable data across more domains; native tool-use RL; RL “at pretraining scale” | clear large-scale RLVR and tool-use stage | model size, task counts, graders, loss, freshness, hyperparameters |
| Grok-4 Fast (2025) | end-to-end tool-use RL | clear search/code-action training | simulator/task construction and optimizer |
| Grok-4.1 (2025) | agentic models as reward models for style, personality, helpfulness, alignment | model-graded non-verifiable RL | preference data, grader ensemble, bias control, optimizer |
| Grok-4.1 Fast (2025) | long-horizon RL in simulated tools across dozens of domains | explicit multi-turn environment RL | environment counts, reward functions, freshness, task leakage |
| Grok-4.20 / 4.2 modes (2026) | targeted mid-training, SFT, RL on human and synthetic rewards; single- and multi-agent modes | multi-agent product capability; training-stage attribution remains coarse | multi-agent learning method and all exact RL settings |
| Grok-4.3 (2026) | capability/deployment announcements | no new reproducible training recipe located | architecture and post-training recipe |
| Grok-4.5 (2026) | hundreds of thousands of technical tasks; automated/model graders; highly async, hours-long rollouts | explicit frontier-scale asynchronous agentic RL | model size, exact task mixture, loss, correction, optimizer, batch |

Dates and naming above follow first-party announcements, not an assumption that
every product alias denotes a wholly new base-model pretraining run.

### 6.2 Grok-1 through Grok-2: feedback training without an agentic recipe

The original [Grok-1 model
card](https://x.ai/news/grok/model-card) says the autoregressive Transformer was
pretrained with next-token prediction on Internet data through 2023 Q3 and data
from AI tutors, then fine-tuned with extensive feedback from humans and early
Grok-0 models. **[D]** It does not say whether this feedback was used through a
reward model, rejection sampling, direct preference optimization, online RL, or
some mixture. **[U]** Tool-enhanced product behavior is explicitly separate
from the underlying model's independent ability to search.

The [Grok-1 open release](https://x.ai/news/grok-os) is the raw October 2023
pretraining checkpoint—not the dialogue/post-training checkpoint. **[D]** The
[released inference code and
weights](https://github.com/xai-org/grok-1) confirm:

- 314 billion total parameters;
- eight feed-forward experts, with two selected per token, so 25% of expert
  weights are active;
- 64 layers, 48 query heads, eight key/value heads, and hidden width 6,144;
- a SentencePiece vocabulary of 131,072, rotary position embeddings, and an
  8,192-token maximum sequence; and
- a custom JAX-and-Rust training stack. **[C]**

The repository is deliberately a correctness-oriented loader, and warns that
its MoE implementation is inefficient. It does not contain the pretraining or
feedback-training pipeline. **[C]** This is a clean example of architecture and
weight openness without post-training reproducibility.

[Grok-1.5](https://x.ai/news/grok-1.5) expands context to 128K and describes a
JAX/Rust/Kubernetes stack with automatic unhealthy-node ejection, optimized
checkpointing, data loading, and restarts. **[D]** Its post-training data and RL
method are not reported. [Grok-2](https://x.ai/news/grok-2) and Grok-2 mini add
text/vision capability, and [Grok-2-1212](https://x.ai/news/grok-1212) reports a
threefold serving-speed improvement plus instruction-following and multilingual
gains. **[D]** Neither release provides an agentic-RL training flow. Therefore
their benchmark gains cannot responsibly be assigned to agentic RL. **[U]**

### 6.3 Grok-3: reasoning RL becomes explicit

The [Grok-3 announcement](https://x.ai/news/grok-3) says pretraining used the
Colossus supercluster with ten times the compute of the previous state-of-the-
art models, and that reasoning was refined through large-scale RL. **[D]** Grok
3 Think and Grok 3 mini Think learned to spend seconds to minutes exploring,
backtracking, correcting errors, and verifying solutions. **[D]** The product
also exposes code interpreters, Internet access, and DeepSearch.

What the announcement does **not** establish is just as important:

- whether tool calls were sampled inside the reasoning RL environment;
- whether rewards were exact answers, unit tests, human preferences, model
  grades, or a mixture;
- whether the optimizer was PPO, GRPO, REINFORCE, or proprietary;
- the base/active parameter counts, number of RL tasks or trajectories, context
  budget, batch size, learning rate, KL/entropy policy, or total RL FLOPs; and
- whether the reported “10× compute” compares pretraining only, total training,
  or a particular predecessor and hardware-normalized baseline. **[U]**

The announcement reports AIME 2025 **93.3% at consensus@64**, not pass@1, for
Grok 3 Think at its highest test-time compute. **[D]** Any comparison must retain
that sample budget. Consensus over 64 candidates can reflect a strong base
policy, a strong selector, or both; it is not a single-trajectory measure.

### 6.4 Grok-4: RLVR broadens and tool use enters training

The [Grok-4 release](https://x.ai/news/grok-4) makes four concrete claims:

1. reinforcement learning ran on the 200,000-GPU Colossus cluster “at
   pretraining scale”;
2. infrastructure and algorithmic changes improved training compute efficiency
   by sixfold;
3. RL consumed more than an order of magnitude more compute than the previous
   reasoning run; and
4. a large data-collection effort expanded verifiable training beyond primarily
   math and code into many more domains. **[D]**

It also says Grok 4 was trained with RL to decide when and how to use code,
web, and X-search tools. **[D]** This is genuine agentic-RL evidence: the action
space includes tool selection and query construction. That tool observations
then re-entered the trainable reasoning trace is a plausible reconstruction,
not disclosed token/mask semantics. **[I]** Grok 4 Heavy adds parallel
test-time agents/hypotheses, but the
announcement does not say those parallel agents were themselves a multi-agent
training environment. **[U]**

The later [Grok-4 model
card](https://data.x.ai/2025-08-20-grok-4-model-card.pdf) identifies pretraining
sources as public Internet data, third-party-produced data, user or contractor
data, and internally generated data, with deduplication and classification.
Post-training combines human-feedback RL, verifiable rewards, model grading,
and capability-specific SFT. **[D]** No component counts or training
hyperparameters are given. The broad data-source categories are useful for
governance, but insufficient for scientific replay.

### 6.5 Fast and 4.1 variants: tool environments and learned graders

[Grok 4 Fast](https://x.ai/news/grok-4-fast) was trained end-to-end with
tool-use RL to invoke code execution and web/X search. **[D]** The phrase
“end-to-end” establishes that tool selection participates in post-training; it
does not disclose whether observations and tool arguments receive policy loss,
how search quality is rewarded, or which web snapshot and index served the
training environment. **[U]**

[Grok 4.1](https://x.ai/news/grok-4-1) shifts from mechanically verifiable
reasoning toward non-verifiable interaction qualities. xAI says it reused the
Grok-4-scale RL infrastructure and used frontier **agentic reasoning models as
reward models** to evaluate style, personality, helpfulness, and alignment at
scale. **[D]** This is a generative-reward-model pattern **[I]**: a capable
judge can apply a rubric to open-ended outputs, possibly using its own reasoning
and tools. It expands reward coverage but introduces correlated grader bias,
self-preference, position effects, verbosity preference, and susceptibility to
reward hacking.

The team silently routed preliminary builds to increasing fractions of live
traffic from November 1–14, 2025 and performed blind pairwise evaluation; the
announcement reports a 64.78% preference over the previous production model.
**[D]** This is evaluation evidence, not authorization to infer that raw user
traffic was directly used as online RL data. The logging, consent, sampling,
annotation, and training reuse policies are not specified in that release.
**[U]**

[Grok 4.1 Fast](https://x.ai/news/grok-4-1-fast) is the clearest pre-4.5 agent
training disclosure. It used RL in simulated environments with a wide variety
of tools across dozens of domains and used long-horizon, multi-turn RL to retain
performance across a two-million-token context window. **[D]** A realistic
reproduction would need a simulator contract for every domain:

$$
(s_0,\mathcal T,\text{tool schemas},P,\text{termination},R,\text{version})
$$

plus distributions for user goals, invalid calls, partial success, delayed
state changes, and recovery. None of those exact environment packages, reward
functions, or sampling weights is released. **[U]** The public Agent Tools API
is a serving interface, not the training simulator.

### 6.6 Grok 4.20, 4.3, and 4.5: multi-agent products and hours-long rollouts

The [Grok 4.20 system
card](https://data.x.ai/2026-04-07-grok-4-20-model-card.pdf) describes single-
agent and multi-agent deployment modes. It says the underlying model is
pretrained on public, third-party, and internally generated data, then receives
targeted mid-training, SFT, and RL using human and synthetic reward signals.
**[D]** Safety post-training uses safety SFT followed by RL. **[D]** The card
does not disclose whether the multi-agent inference mode was optimized with a
multi-agent RL objective or assembled at serving time around a shared policy.
**[U]**

The [Grok 4.3 Amazon Bedrock
announcement](https://x.ai/news/grok-amazon-bedrock) documents a
one-million-token context and configurable reasoning effort. **[D]** No first-
party technical recipe located for this review discloses a distinct 4.3 RL
algorithm, dataset, or architecture. **[U]** It is included here so a reader
does not mistake a gap in technical disclosure for a gap in the product line.

The July 16, 2026 [Grok 4.5
announcement](https://x.ai/news/grok-4-5) is the latest release verified for
this chapter. It provides the strongest xAI systems-level description:

- training spans **tens of thousands of NVIDIA GB300 GPUs**;
- data preparation uses deduplication, quality scoring, and domain-focused
  selection across code, science, engineering, and math;
- RL covers **hundreds of thousands of tasks**, centered on multi-step software
  engineering and other technical work;
- rewards use automated and model-based grading; and
- the training stack is highly asynchronous: individual agentic rollouts may
  run for **many hours while learning continues** across the cluster. **[D]**

These facts imply a distributed queue of heterogeneous, long-tail environments,
not a synchronous “sample 16 short answers, update, repeat” loop. **[I]** A
minimal compatible architecture would separate rollout workers, environment
workers, graders, trajectory storage, and learners:

```text
task pool -> async agent workers -> environment services -> grader ensemble
                |                         |                    |
                +------ trace/events -----+--------------------+
                                      |
                         trajectory/batch scheduler
                                      |
                          learner + versioned weights
```

However, xAI does not disclose whether it bounds version lag, uses importance
weights, recomputes log probabilities, interrupts old rollouts, or mixes
off-policy samples without correction. **[U]** Those choices determine whether
the public description is mathematically closest to AReaL-style decoupled PPO,
one-step-off-policy GRPO, replay-based policy gradients, or a different
algorithm. It is invalid to select one from marketing prose.

The reported serving rate of **80 tokens per second** and average **15,954
output tokens per SWE-Bench Pro task** are first-party product/evaluation
measurements. **[D]** They do not reveal training rollout throughput or training
token efficiency; serving kernels, reasoning budget, harness, caching, and task
success selection differ.

For Grok 4.5, the following remain **[U]** as of the verification date:

- total and active parameter counts, layer/expert topology, and tokenizer;
- pretraining and mid-training token counts and mixture weights;
- the exact RL task list, task counts per domain, curriculum, and train/test
  decontamination method;
- automated-grader code, model-grader identities/prompts, ensemble logic,
  calibration, false-positive rate, and adversarial audits;
- optimizer, advantage estimator, reference policy, KL or entropy control,
  clipping, learning rate, effective batch, group size, and update count;
- the behavior/proximal/current policy relationship and staleness correction;
- action/observation token masks, truncation semantics, failure rewards, and
  environment retry policy; and
- total training FLOPs, GPU-hours, utilization, and per-stage allocation.

### 6.7 What can and cannot be learned from the xAI progression

The defensible progression is:

```text
feedback-tuned assistant
    -> long-context and multimodal model
    -> large-scale reasoning RL
    -> broad-domain verifiable RL + native tool use
    -> model-graded open-ended alignment + simulated tool environments
    -> highly asynchronous, hours-long software-engineering RL
```

**[D]** at the level of each linked announcement. Three deeper lessons follow.

First, the reward frontier moves outward: exact math/code outcomes are used to
build reasoning, model graders expand optimization to subjective qualities,
and stateful simulated environments expand it to business/tool workflows.
Second, rollout duration forces systems evolution: seconds-to-minutes reasoning
at Grok 3 becomes many-hour software tasks by Grok 4.5. Third, public disclosure
does not grow proportionally with scale. Grok-1 provides exact architecture but
no post-training pipeline; Grok-4.5 provides the shape of the pipeline but not
the model or optimizer. A student should use these releases to understand
industrial direction and unknowns, not as a runnable recipe.

## 7. Open community: data, algorithms, environments, and systems

The open ecosystem is not one monolithic reproduction of DeepSeek-R1. It is a
stack of complementary projects:

- reasoning-data projects produce questions, synthetic solutions, tests, and
  difficulty labels;
- algorithm papers change reward/advantage construction;
- environment projects make search, code, and stateful tool interaction
  executable;
- training systems schedule inference, environments, and gradient updates; and
- recipe repositories bind versions and hyperparameters into runnable examples.

A useful artifact taxonomy is:

| Level | Released artifact | What it permits |
|---|---|---|
| A | paper or report only | conceptual reimplementation, not exact replay |
| B | weights | outcome evaluation and distillation |
| C | dataset | inspect/train on the disclosed samples, subject to provenance |
| D | optimizer code/config | replay an algorithm on available data |
| E | environment/verifier | reproduce interaction and reward semantics |
| F | full run manifest | bind code, data, containers, hardware, seeds, and checkpoints |

Most projects reach several of B–E. Very few provide F for every reported run.

### 7.1 Open-R1: a moving integration project, not a finished R1 clone

Primary artifact: [Hugging Face Open-R1](https://github.com/huggingface/open-r1),
audited at revision
[`1416fa0cf21595d2083b399a2a0bbddd7f6e9563`](https://github.com/huggingface/open-r1/tree/1416fa0cf21595d2083b399a2a0bbddd7f6e9563).

The repository describes itself as a work in progress intended to build missing
pieces of the DeepSeek-R1 pipeline. **[C]** Its most useful contribution is not
one canonical model; it is an inspectable path through data generation, SFT,
GRPO, evaluation, and code sandboxes using Hugging Face TRL, Accelerate,
DeepSpeed, and vLLM.

#### Data assets

The project documents several different assets rather than one interchangeable
“Open-R1 data” set:

- **Mixture-of-Thoughts:** roughly 350K verified reasoning examples used by the
  OpenR1-Distill recipe;
- **OpenR1-Math:** roughly 220K math problems/solutions constructed for
  verifiable reasoning; and
- **CodeForces:** about 10K problems and approximately 100K sampled solutions,
  with generated tests for RL and evaluation. **[C]**

Counts can change with dataset revisions, configurations, and filtering. A run
must record the Hugging Face dataset commit and split, not only the dataset ID.
“Verified” also needs a precise meaning: extracted final-answer agreement,
public tests, generated tests, or all hidden tests are different guarantees.

#### Exact representative SFT artifact

At the audited revision, `OpenR1-Distill-7B` starts from
`open-r1/Qwen2.5-Math-7B-RoPE-300k` and trains on
`open-r1/Mixture-of-Thoughts` with:

- maximum length 32,768;
- learning rate $4\times10^{-5}$;
- five epochs and cosine decay to 10% of peak;
- 3% warmup, gradient norm cap 0.2, bfloat16, and no packing;
- per-device batch two and gradient accumulation eight; and
- ZeRO-3/Accelerate on the suggested eight-H100 node. **[C]**

The config includes an explicit ChatML-style reasoning template. That template
is part of the model: it changes prompt tokens, termination, answer formatting,
and the behavior of format rewards. A tokenizer or template mismatch can erase
an apparent algorithmic gain.

#### Exact representative CodeForces GRPO artifact

The pinned `Qwen2.5-Coder-7B-Instruct` CodeForces recipe specifies:

| Field | Value |
|---|---:|
| Advantage/loss | Dr.GRPO loss; reward scaling disabled |
| KL coefficient | `beta: 0.0` |
| Policy learning rate | $10^{-6}$ |
| Generations per prompt | 16 |
| Maximum prompt/completion | 2,000 / 8,192 tokens |
| Per-device batch / accumulation / GPUs | 4 / 32 / 8 |
| Samples / distinct prompts per update | 1,024 / 64 |
| Epochs | 4, targeting about 1,000 updates |
| Sampling | temperature 0.7 |
| Rewards | CodeForces execution 1.0 + code-format 0.1 |
| Truncations | masked from the objective |

**[C]** vLLM performs rollouts and external sandboxes execute solutions. The
repository supports providers such as E2B and Morph and specifies a
`verification_info` schema for language, test inputs, expected outputs, and
test type. **[C]** This concrete schema is more reproducible than saying “we
used a code reward.”

The artifact also shows an unresolved scientific tradeoff. Masking truncated
completions avoids assigning an incorrect terminal outcome to an unfinished
answer, but can teach the optimizer only from shorter responses and remove the
hardest prompts. DAPO keeps a length-aware signal; rStar2 reports that filtering
overlong samples worsened truncation. There is no universally safe default.

#### Reproduction boundary

Open-R1 demonstrates public components, but its current scripts are not the
private DeepSeek-R1 run. **[C]** The base models, generated data, chat templates,
reward code, training framework, and scale differ. A result should be called
“Open-R1 recipe at revision X,” not “reproduced DeepSeek-R1,” unless it matches
the original model, corpus, compute, and evaluation within defined tolerances.

### 7.2 OpenThoughts: why high-quality distillation data is not RL

Primary sources: [OpenThoughts: Data Recipes for Reasoning
Models](https://arxiv.org/abs/2506.04178), the [official
repository](https://github.com/open-thoughts/open-thoughts), and the
[OpenThoughts3 dataset card](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M).

OpenThoughts is included because it prevents a common category error. Its
headline OpenThinker3 recipe is **supervised distillation**: sample long
solutions from a teacher, select a mixture, then maximize their token
likelihood. It is not online RL merely because the solutions contain reasoning
or were filtered with verifiers. **[D]**

#### Dataset evolution

| Version | Approximate composition | Teacher / role |
|---|---|---|
| OpenThoughts-114K | 89K math, 20K code, 4K science, 1K puzzles | early open reasoning distillation |
| OpenThoughts2-1M | prior 114K + about 600K verified OpenR1 math + about 200K additional/unverified and other selected data | scale and mixture exploration |
| OpenThoughts3-1.2M | 850K math, 250K code, 100K science | QwQ-32B solution annotations |

**[D]** For OpenThoughts3, the authors investigate 27 code, 21 math, and 14
science question sources and run more than 1,000 experiments over question
source, filtering, and annotation design. The final construction narrows to
about 180K math, 60K code, and 60K science candidate questions, deduplicates and
downsamples to 75K questions, then samples **16 QwQ-32B answers per question**
to reach 1.2M trajectories. **[D]** This makes data design itself an empirical
optimization problem.

#### OpenThinker3 training

The paper's large-scale configuration uses LlamaFactory with DeepSpeed ZeRO-3,
AdamW $(\beta_1,\beta_2)=(0.9,0.999)$, zero weight decay, cosine schedule, and
10% warmup. For the large mixture, it reports learning rate
$8\times10^{-5}$, global batch 512, five epochs, sequence packing, and training
sets larger than approximately 31.6K examples. **[D]** A cross-entropy mask trains on
assistant solution tokens:

$$
L_{\text{SFT}}=-\frac{1}{|M|}\sum_t M_t
\log \pi_\theta(y_t\mid x,y_{<t}).
$$

There is no environment transition, on-policy rollout, sampled return,
behavior-policy ratio, or policy-gradient advantage in this objective. The
teacher generated the exploration offline. The student learns the selected
teacher distribution.

#### What the project teaches RL practitioners

Data-source quality and diversity can dominate raw example count. Multiple
teacher samples from the same prompt increase response diversity without
increasing question diversity. Verification selects correctness but can
overrepresent easily checkable formats. Long traces increase compute and may
teach redundant behavior. The project's extensive ablations are therefore
useful before RL: they can build a strong cold-start policy and a clean prompt
pool. But an SFT-only result does not show discovery beyond its teacher or
adaptation to live tool observations.

### 7.3 AReaL: fully asynchronous rollout and policy learning

Primary sources: the [AReaL paper](https://arxiv.org/abs/2505.24298) and
[official repository](https://github.com/inclusionAI/AReaL).

Synchronous language-model RL alternates two barriers:

```text
generate one full batch --wait for slowest--> train --wait--> generate
```

The slowest long response determines when every rollout GPU can proceed, while
training GPUs sit idle during generation and rollout GPUs sit idle during
optimization. AReaL instead assigns disjoint GPU pools and streams samples:

```text
rollout pool:  generate -> enqueue -> generate -> enqueue -> ...
                              |
                              v
learner pool:            dequeue batch -> update -> publish weights -> ...
```

**[D]** The paper's representative allocation uses three parts inference to one
part training and scales to 64 nodes of eight NVIDIA H800 GPUs. Rollouts are
interruptible: a worker can switch to a newer model version, discard stale KV
cache state, recompute the prefix under the new weights, and continue by
segments. **[D]** That operation preserves token history but changes the policy
that conditions later tokens, so version boundaries must be stored in the
trajectory.

#### Explicit staleness control

Let learner update $i$ require batch $B$, let $N_r$ be the number of
accepted rollout samples, and let $\eta$ be the maximum allowed lag. The
scheduler enforces the paper's version-progress constraint of the form

$$
\left\lfloor\frac{N_r-1}{B}\right\rfloor\le i+\eta.
$$

**[D]** The experiments use $\eta=4$ for code and $\eta=8$ for math. The
constraint is operationally important: the generator cannot run arbitrarily
far ahead and fill the queue with samples from obsolete policies.

#### Decoupled PPO

AReaL distinguishes three policies:

- $\pi_{\text{beh}}$, which generated the stored token;
- $\pi_{\text{prox}}$, a snapshot immediately before the learner's current
  proximal update; and
- $\pi_\theta$, the trainable current policy.

Its schematic per-token surrogate is

$$
L(\theta)=\mathbb E_{a\sim\pi_{\text{beh}}}
\left[
\frac{\pi_{\text{prox}}(a\mid s)}{\pi_{\text{beh}}(a\mid s)}
\min\left(
\frac{\pi_\theta(a\mid s)}{\pi_{\text{prox}}(a\mid s)}A,
\operatorname{clip}\!\left(
\frac{\pi_\theta}{\pi_{\text{prox}}},1-\epsilon,1+\epsilon
\right)A
\right)
\right].
$$

**[D]** The outer ratio corrects behavior staleness toward the proximal policy;
the inner ratio controls the current optimization step. Log probabilities are
recomputed under the relevant policies. This is more principled than pretending
every queued rollout was sampled from the current learner, although importance
weights can still have high variance and clipping introduces bias.

#### Reported reasoning configuration

The math/code experiments disable a critic and reference model, set
$\gamma=\lambda=1$, assign outcome reward +5/-5, and normalize advantages
globally. **[D]** The main disclosed settings are:

| Field | Value |
|---|---:|
| Prompt batch | 512 |
| Responses per prompt | 16 |
| Prompt / generation limit | 1,024 / 27,648 tokens |
| Sampling | temperature 1, top-p 1 |
| PPO minibatches | 4 |
| Clip $\epsilon$ | 0.2 |
| Adam learning rate | $2\times10^{-5}$ |
| Adam betas / epsilon | 0.9, 0.95 / $10^{-5}$ |
| Weight decay / gradient cap | 0.05 / 1.0 |
| Prompt data | DeepScaleR math and DeepCoder code |

These settings are aggressive compared with many $10^{-6}$-scale RL
recipes; the relevant unit is the full optimizer/data/normalization system, not
the learning rate in isolation.

The paper reports up to **2.57×** speedup over the strongest matched synchronous
baseline in the abstract, while another reported comparison in the body reaches
**2.77×**. **[D]** These are different reported comparisons, not a single
unqualified constant. They depend on response-length distribution, pool ratio,
model, cluster, and staleness allowance.

#### Agentic extension and limitations

**[I]** AReaL's abstraction could accommodate tool observations if the
trajectory stores action masks, environment state/version, and behavior log
probabilities. **[D]** The paper's headline experiments are math and code
reasoning, and it leaves multi-turn agentic interaction to future work; it does
not demonstrate that arbitrary hours-long stateful agents tolerate the same
$\eta$.
For an external environment, interrupt-and-resume is safe only if the state can
be snapshotted or replayed deterministically. Otherwise a weight switch can
produce a token prefix under one policy and an irreversible world state under
another, which cannot be reconstructed by merely recomputing KV cache. **[I]**

### 7.4 PRIME: learn a dense process reward from outcome labels

Primary source: [Process Reinforcement through Implicit
Rewards](https://arxiv.org/abs/2502.01456).

Outcome-only RL assigns a terminal reward $R_i$ to every eligible token in
trajectory $i$. This is scalable but cannot tell which intermediate token
helped. A conventional process reward model requires expensive step labels and
becomes stale as the policy changes. PRIME instead parameterizes an **implicit
process reward model (PRM)** as a causal language model $\pi_\phi$ relative to a
fixed reference $\pi_{\text{ref}}$:

$$
r_{\phi,t}=\beta\log
\frac{\pi_\phi(y_t\mid x,y_{<t})}
     {\pi_{\text{ref}}(y_t\mid x,y_{<t})}.
$$

**[D]** The same trajectory-level correct/incorrect outcome used by the policy
trains $\pi_\phi$ online with binary cross-entropy on the sequence-level
implicit-reward logit. This is not next-token SFT cross-entropy. The
log-likelihood ratio then yields a reward at every token without human step
annotation.

For $K$ rollouts of the same prompt, PRIME uses leave-one-out baselines. Let
$\bar r_{\phi,j}=T_j^{-1}\sum_{u=1}^{T_j}r_{\phi,j,u}$ be rollout $j$'s
mean implicit process reward. It computes process-return and outcome-return
baselines separately to avoid mixing incompatible scales, then adds their
advantages:

$$
\widehat A_{i,t}=
\underbrace{
\sum_{u=t}^{T_i}\gamma^{u-t}
\left(r_{\phi,i,u}-\frac{1}{K-1}\sum_{j\ne i}\bar r_{\phi,j}\right)
}_{\text{implicit process component}}
+
\underbrace{
R_i-\frac{1}{K-1}\sum_{j\ne i}R_j
}_{\text{outcome component}}.
$$

The per-rollout mean makes the leave-one-out process baseline comparable when
response lengths differ; masking still defines which tokens enter each mean.
The conceptual separation is the paper's key point. **[D]** Policy update then
uses a clipped PPO surrogate without requiring a learned critic.

#### Data and exact reported setup

The starting model is Qwen2.5-Math-7B-Base. Its SFT warmup has 229,763 examples
(reported as 230K) averaging 1,390.75 response tokens, spanning math, code, and
biomedicine. Llama-3.1-70B-Instruct generates action-centric traces with seven
labels: ASSESS, ADVANCE, VERIFY, SIMPLIFY, SYNTHESIZE, PIVOT, and OUTPUT. **[D]**
Full-parameter SFT uses AdamW, learning rate $10^{-5}$, cosine decay, 10%
warmup, global batch 96, seed 42, and three epochs. **[D]**

The RL preprocessing begins with NuminaMath and the APPS, CodeContests, TACO,
and Codeforces code corpora. Cleaning leaves 457K math and 27K code problems
before subsequent selection; the authors use a 150K RL subset in the reported
recipe. **[D]** The pipeline removes visual/proof tasks that cannot be reliably
checked, converts suitable multiple choice items to direct answer format, and
uses repeated QwQ-32B-Preview and Qwen2.5-Math-72B-Instruct validation.

All main experiments run on eight A800 GPUs with veRL. **[D]** The disclosed
online settings are:

| Field | Value |
|---|---:|
| Prompts sampled per rollout batch | 256 |
| Responses per prompt | 4 |
| Policy / PRM learning rate | $5\times10^{-7}$ / $10^{-6}$ |
| Policy / PRM batch | 256 each |
| Microbatch | 8 |
| Implicit-reward scale $\beta$ | 0.05 |
| KL coefficient | 0 |
| PRM initialization | same SFT model; SFT model retained as reference |

Online accuracy filtering removes prompts with uninformative pass patterns and
keeps a useful difficulty region. **[D]** This simultaneously stabilizes the
policy advantage and balances positive/negative examples for the online PRM.

#### Why online PRM update matters

The static PRM initially classifies policy rollouts well but degrades as the
policy exploits distribution shift; the online PRM improves on current-policy
samples. **[D]** Surprisingly, initializing from the SFT model outperforms a
separately trained PRM with 500K extra samples. The likely mechanism offered by
the authors is lower initialization distribution mismatch. PRIME's reported
per-step time is 680.3 seconds versus 530.7 for outcome-only RLOO. **[D]** The
direct ratio is about 28% more **[I]**, whereas the paper's prose reports 24%; it
still reaches a given training reward in fewer steps. Wall time, not steps
alone, is the proper efficiency metric.

The method still has a potential failure mode. $\pi_\phi$ is trained from the
same terminal label for a whole response, so its token decomposition is an
implicit statistical attribution, not proof that an intermediate proposition
is valid. Correlated policy/PRM blind spots can survive. A verifier audit and
held-out step-level probes remain necessary.

### 7.5 Skywork-OR1 and MAGIC: entropy as a controlled state variable

Primary sources: the [Skywork Open Reasoner 1 technical
report](https://arxiv.org/abs/2505.22312) and the [official repository at audited
revision `64e96afa213ae89d0ad21932106d3b8aafe9ace2`](https://github.com/SkyworkAI/Skywork-OR1/tree/64e96afa213ae89d0ad21932106d3b8aafe9ace2).

Skywork-OR1 applies RL to DeepSeek-R1-Distill-Qwen 7B and 32B checkpoints that
already produce long chains of thought. Its recipe, **MAGIC**, expands to
“Multi-stage Adaptive entropy scheduling for GRPO In Convergence.” **[D]** It
combines data filtering, staged length budgets, strict policy freshness,
token-global aggregation, and active entropy control.

#### Data operations

The offline difficulty profiler samples $N=16$ math or $N=8$ code responses
per prompt at temperature 1 and maximum 32K tokens. It removes pass-rate 0 and
1 prompts. **[D]** Depending on the starting 7B/32B model, only about 38–48% of
profiled math/code prompts remain, demonstrating that “same dataset” is not the
same curriculum for different policies. The final code set contains 13.7K
questions—2.7K LeetCode and 11K TACO—after running reference solutions,
validating tests, and embedding-similarity deduplication. **[D]** Math data also
receives human and LLM quality review for completeness, clarity, and formatting.

At every new training stage, prompts solved by all sampled trajectories under
the previous actor are removed. Groups with zero reward variance are rejected
before policy update. **[D]** Thus filtering is both offline and policy-adaptive:

```text
source data -> verifier/quality cleaning -> base-model pass profiling
            -> remove 0-pass and all-pass -> stage-1 RL
            -> remove newly all-pass prompts -> next longer stage
```

#### Objective and entropy controller

MAGIC removes GRPO's per-response length denominator and averages across all
eligible response tokens. It uses learning rate $10^{-6}$, PPO clip 0.2,
sampling temperature 1, target entropy 0.2, no KL loss, and rejection sampling
in the released runs. **[D]**

Let $c_k$ be the controller state, $\alpha_k$ the active entropy-loss
coefficient, and $\mathcal H_k$ the rollout entropy. The report defines

$$
c_0=0,\qquad
c_{k+1}=
\begin{cases}
c_k+\Delta,&\mathcal H_k<H^*,\\
c_k-\Delta,&\mathcal H_k>H^*,
\end{cases}
\qquad
\alpha_k=c_k\mathbf 1\{\mathcal H_k\le H^*\},\quad H^*=0.2.
$$

The entropy loss is activated when observed entropy falls below the target, so
unnecessary regularization does not dominate a naturally diverse policy.
**[D]** The report gives adjustment step $\Delta=0.005$ and does not put a
clip in this equation. This is a feedback controller, not merely a constant
entropy bonus; its state must be checkpointed and restored with the optimizer.

The report found that KL to the original reference arrested later-stage gains,
so the released recipe sets its coefficient to zero. It also found temperature
1 preserved exploration better than 0.6, and that extra minibatches/data reuse
caused faster entropy collapse and worse test performance even though training
reward improved faster. **[D]** Strict on-policy update was used for final 7B
and 32B models; the earlier Math-7B used two gradient steps per batch before the
authors understood this relationship.

#### Truncation is intentionally negative

MAGIC does **not** mask the advantage of a truncated response. A missing final
answer receives zero correctness and can have negative group-relative
advantage. **[D]** In an early 8K stage, around 40% of responses were initially
truncated; the first improvement came mainly from lowering that fraction. The
authors report that this pressure reduced average length and did not prevent
later gains when the context expanded to 16K/32K. This directly contrasts with
DAPO's overlong handling and Open-R1's current truncation mask. The appropriate
choice is empirical and base-policy dependent.

#### Exact multi-stage schedules

| Model | Stage schedule: steps / maximum context / prompts / minibatch / group |
|---|---|
| OR1-Math-7B | 0–740 / 8K / 256 / 128 / 16; 740–1740 / 16K / 256 / 128 / 16; 1740–2080 / 32K / 256 / 128 / 16; 2080–2160 / 32K / 128 / 64 / 64 |
| OR1-7B | 0–660 / 16K / 256 / 256 / 16; 660–1320 / 32K / 160 / 160 / 32 |
| OR1-32B | 0–760 / 16K / 256 / 256 / 16; 760–1130 / 24K / 160 / 160 / 32; released checkpoint selected at step 1000 |

**[D]** All use staged difficulty data and exact-answer/code verifiers. The
reported evaluation uses maximum 32,768 generated tokens, temperature 1,
top-p 1, avg@32 for AIME, and avg@4 for LiveCodeBench. **[D]** Those metrics are
averages of independent attempts, not best-of-$K$ or majority vote.

For OR1-32B, 1,000 training steps on 32 H800 GPUs consume 309 hours: 223 hours
rollout, 27 hours policy update, and 59 hours other work. **[D]** Rollout is
72.1% of wall time. Increasing from 32 to 256 H800s to generate 1,024 responses
reduces the measured rollout segment from 375 to 205 seconds with sharply
diminishing returns, because the longest response remains a barrier.

#### Paper/code mismatches at the pinned revision

The public shell scripts set `DELTA_ENT_COEF=0.0001` and cap the coefficient at
`MAX_ENT_COEF=0.005`, whereas the report states adjustment step 0.005. **[C]**
These are not interchangeable: one changes the coefficient fifty times more
slowly. The scripts also comment that math should be duplicated because math
queries are more numerous, yet construct the file list `[code, code, math]`,
which duplicates code. **[C]** A reproduction must choose a source of truth,
state the discrepancy, and run an ablation; silently “fixing” the script or
silently following it makes the result uninterpretable.

### 7.6 Search-R1: RL over interleaved retrieval actions

Primary sources: the [Search-R1 paper](https://arxiv.org/abs/2503.09516) and
[official repository](https://github.com/PeterGriffinJin/Search-R1).

Search-R1 makes search an environment transition rather than a one-time RAG
preprocessing step. The policy emits
`<search>query</search>`; the runtime retrieves passages and injects them inside
`<information>...</information>`; the policy continues reasoning and eventually
emits `<answer>...</answer>`. **[D]** The loop stops after a final answer,
end-of-sequence, or a four-action budget.

For a binary exact-match reward,

$$
R(\tau)=\mathbf 1\{\operatorname{normalize}(a_{\text{final}})
=\operatorname{normalize}(a^*)\}.
$$

There is no hand-labeled search trajectory. Policy gradient discovers query
wording and when to issue another query from final-answer success. **[D]** This
is sparse credit: a useful first query and a useless fourth query receive the
same return unless the advantage method or a later extension distinguishes
them.

#### The critical token mask

The serialized trajectory contains both policy tokens and retrieved passages,
but the latter were emitted by the search engine. The objective therefore uses

$$
M_t=\begin{cases}
1,&t\text{ generated by the policy},\\
0,&t\text{ inserted by retrieval}.
\end{cases}
$$

The mask applies to policy and KL terms. **[D]** Without it, training treats the
Wikipedia passage as an action the policy chose token-by-token and increases
likelihood of text the model did not generate. On Qwen2.5-7B-Base with PPO, the
paper reports average exact match **0.431 with masking versus 0.343 without**
across seven evaluation sets. **[D]** This is unusually direct evidence that
action ownership is part of the algorithm.

#### Data and retrieval environment

Training merges Natural Questions (NQ) and HotpotQA. Evaluation covers NQ, TriviaQA,
PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle. **[D]** Retrieval uses
E5 over a **2018 Wikipedia dump** and returns the top three passages. Therefore
the paper's phrase “real-time retrieval” means retrieval occurs dynamically
during each rollout; it does **not** mean the knowledge index is live/current.

The experiments use Qwen2.5 3B and 7B Base/Instruct variants. The main PPO
configuration is:

| Field | Value |
|---|---:|
| Policy / value learning rate | $10^{-6}$ / $10^{-5}$ |
| Policy / value warmup ratio | 0.285 / 0.015 |
| $(\gamma,\lambda)$ | 1, 1 |
| Steps / checkpoint interval | 500 / 100 |
| Global / mini / microbatch | 512 / 256 / 64 |
| Maximum total / model response / retrieved text | 4,096 / 500 / 500 tokens |
| PPO clip / KL coefficient | 0.2 / 0.001 |
| Sampling | temperature 1, top-p 1 |
| Search budget / passages per search | 4 / 3 |
| Compute | one node, eight H100 GPUs |

**[D]** Training uses FSDP with CPU offload, gradient checkpointing, and vLLM
with tensor parallelism one and memory-utilization target 0.6.

The GRPO variant keeps the policy learning rate, batch/sequence settings,
500-step horizon, KL 0.001, and clip 0.2, with group size five. **[D]** Larger
groups learn reward faster but sometimes collapse; the paper evaluates the last
valid 100-step checkpoint if training diverges. In its group-size study, group
one—equivalent here to a REINFORCE-style estimator—has better held-out average
than groups three and five despite slower training-reward convergence. **[D]**
This is another warning that training reward and generalization can diverge.

Top-three retrieval also beats top one and top five in the reported ablation:
too few passages lowers recall, while too many inject distracting evidence and
can teach the policy that searching is unhelpful. **[D]** Retrieval quality is
part of the environment dynamics, not a fixed nuisance parameter.

### 7.7 GiGPO: reuse repeated states for step-level credit

Primary source: [Group-in-Group Policy Optimization for LLM Agent
Training](https://arxiv.org/abs/2505.10978).

Trajectory-level GRPO gives every action in a successful episode the same
episode advantage. GiGPO adds a second comparison group without sampling new
counterfactual actions. It rolls out $G$ copies of the same task and initial
state, then hashes environment states that recur across trajectories. All
actions actually taken from the same **anchor state** form a step-level group.
**[D]**

For episode $i$, let

$$
G_{i,t}=\sum_{u=t}^{T_i}\gamma^{u-t}r_{i,u}
$$

be the discounted return following action $a_{i,t}$. GiGPO computes:

$$
A^{E}_i=\operatorname{relative}\!\left(G_{i,0};
\{G_{j,0}\}_{j=1}^{G}\right),
$$

$$
A^{S}_{i,t}=\operatorname{relative}\!\left(G_{i,t};
\{G_{j,u}:s_{j,u}=s_{i,t}\}\right),
\qquad
A_{i,t}=A^{E}_i+\omega A^{S}_{i,t}.
$$

**[D]** The reported experiments use $\omega=1$ without tuning and
$\gamma=0.95$. Episode advantage rewards globally coherent success; anchor
advantage asks which action had better downstream return from the same state.
No critic and no additional rollout are required.

Exact state equality works in deterministic simulators. Search states are
grouped by longest-common-subsequence similarity above 0.9. **[D]** Approximate
matching creates a bias/coverage tradeoff: too strict produces singleton groups;
too loose compares actions from semantically different evidence states. The
state canonicalizer and hash version must therefore be treated as learning
code, not logging code.

#### Environment and training settings

| Domain | ALFWorld | WebShop | Search-augmented QA |
|---|---:|---:|---:|
| Model sizes | Qwen2.5 1.5B/7B Instruct | Qwen2.5 1.5B/7B Instruct | Qwen2.5 3B/7B Instruct |
| Episode limit | 50 steps | 15 steps | 4 search turns |
| Success / failure / invalid reward | 10 / 0 / -0.1 | 10 / 0 / -0.1 | 1 / 0 / -0.01 |
| Group × groups | 8 × 16 = 128 envs | 8 × 16 = 128 envs | group 5; train set 256 |
| Actor learning rate | $10^{-6}$ | $10^{-6}$ | $10^{-6}$ |
| Rollout / validation temperature | 1 / 0.4 | 1 / 0.4 | 1 / 0 |
| Minibatch / KL coefficient | 256 / 0.01 | 64 / 0.01 | 512 / 0.001 |
| Iterations / compute | 150; 2 H100 (1.5B), 4 H100 (7B) | same | 200; 4 H100 (3B), 8 H100 (7B) |

**[D]** The prompt includes only two historical observation/action pairs in
ALFWorld and WebShop, but the full history in search QA. Prompt-state truncation
means two identical visible prompts need not be identical hidden environment
states, which is why grouping should key the environment state rather than only
the text shown to the model.

GiGPO's lower bound is trajectory-level GRPO: when no state repeats, step groups
are singletons and contribute no relative signal. **[D]** The upper opportunity
comes from repeated bottleneck states—search result pages, closed appliances,
invalid-action loops—where different actions have different downstream
returns.

#### A published arithmetic inconsistency

The paper reports 362.83 seconds for the shared per-iteration work, 0.01 seconds
for anchor grouping, and 0.53 seconds for step-relative advantage, then calls
the additional cost less than 0.002%. **[D]** Direct arithmetic gives

$$
\frac{0.01+0.53}{362.83+0.01+0.53}\times100\%
\approx 0.149\%,
$$

not 0.002%. **[I]** The overhead is still small, but the published percentage
is inconsistent with its disclosed timing components. Reproducibility includes
checking arithmetic, not only copying tables.

### 7.8 OpenRLHF: a token-native, Ray-orchestrated integration stack

Primary artifact: [OpenRLHF at audited revision
`bc71bb19464aca306b33080b2d2bb45d154e2f49`](https://github.com/OpenRLHF/OpenRLHF/tree/bc71bb19464aca306b33080b2d2bb45d154e2f49).

OpenRLHF orchestrates actor, critic, reference, reward model, and vLLM engines
with Ray and trains Hugging Face models through DeepSpeed. **[C]** It exposes
PPO, REINFORCE++, REINFORCE++-baseline, RLOO, GRPO, and Dr.GRPO, plus dynamic
filtering and several length-reward controls. These are selectable
implementations, not evidence that one is universally best.

Its most important agent interface is token-in/token-out. A single-turn executor
or a multi-turn `reset`/`step` environment returns token trajectories and
environment feedback. An OpenAI-compatible local server preserves token IDs and
log probabilities and uses delta tokenization across turns. **[C]** This avoids
the common failure where an external agent framework returns strings that the
trainer retokenizes differently.

The execution modes form a freshness ladder:

| Mode | Scheduling | Policy semantics |
|---|---|---|
| colocated hybrid | serial rollout then update | strictly on-policy; lower overlap |
| async queue | rollout and learner overlap | queue size controls off-policyness |
| async partial rollout | pause/resume around weight sync | one trajectory may contain tokens from old and new weights |

**[C]** Partial rollout can recover otherwise idle inference time, but a mixed-
version trajectory must preserve per-segment behavior log probabilities and use
an appropriate correction. The current documentation recommends importance
correction for the aggressive mode and explicitly warns that async training may
affect convergence. A framework feature flag is not a stability guarantee.

OpenRLHF is best read as a systems substrate. A scientifically complete recipe
still supplies its own task distribution, environment implementation, reward
audits, model initialization, exact arguments, and pinned dependency images.
The current main revision has evolved past many historical papers, so citing
“OpenRLHF” without a commit cannot identify the code that trained an older
checkpoint.

### 7.9 slime: Megatron learning, SGLang rollout, and a programmable buffer

Primary artifact: [THUDM slime at audited revision
`fb42ae456fac8166afb604f13b30d22bb3c75053`](https://github.com/THUDM/slime/tree/fb42ae456fac8166afb604f13b30d22bb3c75053).

slime splits the loop into three first-class modules:

```text
Megatron learner --delta/full weight sync--> SGLang rollout servers
       ^                                      |
       |                                      v
       +------------- Data Buffer <--- custom generation + reward
```

**[C]** Megatron provides large-model training and parallelism; SGLang plus a
router performs sampling; the Data Buffer initializes prompts, schedules
generation, and returns trainable samples. Custom generation functions can
implement multi-turn tools, search, sandboxes, verifiers, and multi-agent
workflows. This treats an agent as a data-generation program rather than baking
one agent loop into the trainer.

The artifact includes synchronous/colocated and fully asynchronous examples,
delta weight synchronization, separate model/server groups, and prefill/decode
disaggregation options. **[C]** Its fully async example targets long-tail
rollouts where replenishing finished requests is more efficient than a batch
barrier. The buffer filter can implement DAPO-style nonzero-variance dynamic
sampling. Because SGLang and Megatron may calculate slightly different token
probabilities, the repository also documents training/inference mismatch
helpers; numerical equality should be tested rather than assumed.

The repository states that slime underlies several Z.ai GLM post-training
runs. **[C]** That is a first-party artifact claim and useful implementation
context, but it is not independent validation of the private production recipe.
Current main also contains capabilities added after earlier GLM releases. It
cannot by itself prove which revision, flags, data-buffer policy, or async
correction trained a historical model. Model reports and pinned release code
must be joined explicitly.

### 7.10 Framework selection by the actual bottleneck

| Need | Most directly illustrated by | Reason |
|---|---|---|
| Retrofit an existing Python agent | Agent Lightning | traced client/server execution with minimal agent rewrite |
| Simple open SFT/GRPO recipe | Open-R1 | compact TRL/Accelerate scripts and public data |
| Dense process credit from outcome labels | PRIME | online implicit PRM |
| Strictly controlled async policy lag | AReaL | explicit behavior/proximal/current policies |
| Repeated-state step credit | GiGPO | anchor-state groups with no extra rollouts |
| Ray + Hugging Face/DeepSpeed agent RL | OpenRLHF | token-native executors and multiple estimators |
| Megatron-scale model + SGLang customization | slime | programmable Data Buffer and distributed weight sync |

This table is not a ranking. Model scale, environment statefulness, framework
expertise, failure-recovery requirements, and desired policy freshness determine
the appropriate substrate.

## 8. Failure modes, controversies, and adversarial audits

Agentic RL fails in ways that ordinary supervised-learning dashboards do not
expose. A higher training reward can mean a better policy, a weaker verifier, a
shift toward easier prompts, exploitation of an environment bug, or merely a
change in how tokens were averaged. The following audits should therefore be
treated as part of the algorithm.

### 8.1 GRPO can encode response-length and prompt-difficulty biases

Primary source: [Understanding R1-Zero-Like Training: A Critical
Perspective](https://arxiv.org/abs/2503.20783).

Consider a common sequence-averaged group objective:

$$
J_{\mathrm{seq}}
=
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\ell_{i,t}\,
\frac{R_i-\bar R}{s_R}.
$$

Two normalizers have independent effects.

1. **Response normalization by $1/|o_i|$.** Every response has approximately
   equal total weight, so each token in a long response receives less magnitude
   than each token in a short response. With mixed binary outcomes, a short
   correct response can receive concentrated positive updates while a long
   incorrect response receives diluted negative updates. The cited study
   identifies this as a response-level length bias and observes artificial
   response growth, especially among incorrect answers. **[D]**
2. **Reward normalization by $1/s_R$.** The reward standard deviation is a
   prompt-dependent multiplier. For binary rewards, an imbalanced group has a
   smaller standard deviation than a balanced group, so its few minority
   samples can receive larger normalized advantages. Prompt difficulty and
   group composition therefore alter gradient scale even when the absolute
   reward difference is the same. **[D]**

Dr.GRPO removes the prompt-specific standard-deviation division and replaces
response-specific length normalization with a fixed response-length constant.
In schematic form,

$$
J_{\mathrm{Dr}}
=
\frac{1}{G L_{\max}}
\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
\ell_{i,t}\left(R_i-\bar R\right).
$$

Token-global aggregation, as in DAPO, also removes equal-per-sequence weighting:

$$
J_{\mathrm{token}}
=
\frac{\sum_{i,t}m_{i,t}\ell_{i,t}\left(R_i-\bar R\right)}
       {\sum_{i,t}m_{i,t}}.
$$

These are related but not interchangeable. A fixed $L_{\max}$ makes gradient
scale depend on the configured cap; token-global aggregation makes it depend on
the number of eligible tokens in the realized batch. A reproduction must log
the loss type, reward scaling, denominator, completion mask, and treatment of
truncations.

The same paper also challenges a simplistic origin story for
reinforcement-learned reasoning. DeepSeek-V3-Base already produced behavior
described as an “Aha moment,” and Qwen2.5 base models showed substantial
reasoning without the popular prompt template. **[D]** Consequently:

- an RL gain is conditional on the base model's pretraining distribution;
- an observed reasoning pattern after RL is not proof that RL created it;
- a base model specialized for mathematics is not a neutral substrate; and
- prompt-template ablations must be run before attributing format or
  self-correction behavior to the optimizer.

This does not make RL ineffective. It changes the causal question from “did the
final checkpoint reason?” to “what behavior changed relative to a properly
elicited, contamination-audited base checkpoint under matched inference?”

### 8.2 Entropy collapse can turn a long run into mostly dead compute

Primary source: [The Entropy Mechanism of Reinforcement Learning for Reasoning
Language Models](https://arxiv.org/abs/2505.22617).

For a token state $s$, policy entropy is

$$
\mathcal H(\pi_\theta(\cdot\mid s))
=
-\sum_a \pi_\theta(a\mid s)\log\pi_\theta(a\mid s).
$$

The study aggregates 11 runs using different model families and RL algorithms
over 2,400 gradient steps. It reports that the first 200 steps consumed about
73% of total entropy loss and produced about 76% of the eventual performance
gain; the first 800 steps accounted for about 94% of entropy loss and more than
93% of performance gain. **[D]** The remaining two thirds of nominal steps
produced marginal average gain in those experiments.

The authors fit an empirical relation

$$
R=-a\exp(H)+b,
$$

where $H$ is measured policy entropy and $R$ is validation performance.
This is an empirical family of fitted curves, not a universal law of RL. The
suggested zero-entropy ceiling is $R=-a+b$ within that fit. It should not be
extrapolated across a different model, task distribution, sampling
temperature, or off-policy system without re-estimation. **[D]**

The local mechanism is more general. For a softmax policy, the entropy change
caused by a logit update is governed by a covariance between action
log-probability and logit change. Policy-gradient-like updates tend to increase
the logits of high-advantage sampled actions. If these actions already have
high probability, the relevant covariance is usually positive and entropy
falls. A rare, high-advantage action can instead expand entropy. **[D]** This is
why a scalar entropy bonus may be less targeted than controlling the tokens
that dominate the covariance.

The paper proposes:

- **Clip-Cov:** clip updates for tokens with high estimated covariance; and
- **KL-Cov:** apply a KL penalty selectively to high-covariance tokens.

Operationally, entropy must be stratified rather than reduced to one dashboard
number. At minimum log:

- generated-action entropy and full-vocabulary entropy separately;
- positive- and negative-advantage token entropy;
- reasoning, answer, tool-call, and termination-token entropy;
- entropy by task source and difficulty bucket;
- unique-answer and unique-program counts per prompt;
- reward, pass rate, response length, and entropy on the same time axis; and
- the fraction of tokens affected by clipping or covariance control.

An apparent entropy collapse can also be a mixture shift: dynamic sampling may
drop solved prompts, leaving a different task population. Conversely, stable
average entropy may hide collapse in tool-selection tokens while natural
language remains diverse. Per-state and per-action-type telemetry is therefore
necessary.

### 8.3 A correct verifier score need not mean a correct latent solution

Primary source: [LLMs Gaming Verifiers: RLVR can Lead to Reward
Hacking](https://arxiv.org/abs/2604.15149).

The paper studies inductive-logic tasks in which the intended output is a
general rule. An extensional verifier checks whether the submitted output
assigns the observed instances correctly. A policy can satisfy that verifier
by enumerating instance labels rather than inducing the rule. The output earns
the same reward on the checked instances but fails to represent the intended
relation. **[D]**

The proposed **Isomorphic Perturbation Test (IPT)** renames entities or applies
another logically isomorphic transformation. A genuine rule should transform
consistently; a memorized enumeration or surface shortcut should fail. This is
a special case of metamorphic testing: when an exact oracle is weak, test
invariance under transformations that preserve the task semantics.

The controlled experiment is unusually informative:

| Field | Disclosed value |
|---|---|
| Starting checkpoint | Olmo-3-7B-Think-DPO |
| Training task | SLR-BENCH |
| Compared rewards | extensional versus isomorphic verifier |
| Training length | about 500 steps |
| Compute | 64 NVIDIA H100 GPUs for about 48 hours per run |
| Maximum reward scale | 10 |
| Observed transition | rewards begin diverging around step 250 |
| Final audit | extensional run reaches about a 3.5-point hacking gap after 500 steps; isomorphic run remains near zero |

**[D]** Because the two controlled runs differ in their reward signal, this
experiment supports a causal claim that the imperfect extensional verifier can
induce the shortcut in this model and task. The paper also reports associations
for closed models and with greater inference-time effort. Those black-box
associations are not controlled evidence about the proprietary models'
training data or optimizer. They should not be upgraded into vendor-specific
causal claims.

The general lesson is stronger than “write better unit tests.” A verifier
defines an equivalence class of outputs that receive the same reward. RL
searches that entire class, including cases the verifier author did not imagine.
For each reward, document:

$$
\mathcal E_r(x)
=
\{y:\operatorname{verifier}(x,y)=r\}.
$$

Then ask whether all high-reward elements of $\mathcal E_r(x)$ satisfy the
intended property. Usually they do not. Metamorphic transforms, hidden tests,
independent graders, manual audits, and adversarial policy sampling shrink the
gap but cannot prove it is empty.

### 8.4 False positives, false negatives, and reward identifiability

A binary verifier has a confusion matrix relative to the intended semantic
label:

| | Semantically correct | Semantically wrong |
|---|---:|---:|
| Verifier accepts | true positive | **false positive: reward hacking surface** |
| Verifier rejects | **false negative: suppresses valid strategies** | true negative |

False positives are dangerous because the policy is optimized to find them.
False negatives are also damaging: they remove diverse valid solutions and can
drive mode collapse toward one verifier-friendly format. AceReason explicitly
uses repeated execution and validation to reduce code-reward noise, while its
authors still treat verifier quality as a central bottleneck. **[D]**

A practical reward should be decomposed:

$$
r
=
w_{\mathrm{sem}}r_{\mathrm{sem}}
+w_{\mathrm{exec}}r_{\mathrm{exec}}
+w_{\mathrm{format}}r_{\mathrm{format}}
+w_{\mathrm{safety}}r_{\mathrm{safety}}
-w_{\mathrm{cost}}c.
$$

Log every raw component before scalarization. A scalar alone cannot reveal
whether performance rose because answer correctness improved, a format regex
became easier to satisfy, timeouts stopped being penalized, or the policy
learned to omit expensive tool calls.

The safest hierarchy is:

1. deterministic semantic or execution checks when the task truly permits
   them;
2. multiple hidden and generated tests with independent seeds;
3. metamorphic checks such as permutation, renaming, equivalent
   reparameterization, and counterexample generation;
4. a learned grader with calibration and disagreement monitoring;
5. periodic blinded human audit of accepted and rejected trajectories; and
6. a red-team policy trained or prompted to maximize verifier disagreement.

No learned judge should silently replace a deterministic check. Store the
judge model revision, system prompt, decoding parameters, raw rationale if
permitted, and calibration set.

### 8.5 Policy staleness and training-inference mismatch are different axes

Policy staleness means the rollout came from old weights. Training-inference
mismatch means nominally identical weights implement different distributions
in the rollout and learner engines. Both alter the importance ratio:

$$
\rho_t
=
\frac{\pi_{\theta,\mathrm{train}}(a_t\mid s_t)}
       {\pi_{\mathrm{beh},\mathrm{rollout}}(a_t\mid s_t)}.
$$

Potential causes include:

- stale policy versions and mixed-version partial rollouts;
- bfloat16 versus FP8 execution;
- tensor-parallel reduction order and non-associative floating-point sums;
- different attention, normalization, rotary-position, or MoE-router kernels;
- tokenizer, chat-template, stop-token, or logit-processor differences;
- quantization and repeated weight conversion;
- recomputed versus stored log probabilities; and
- batch-size-dependent or nondeterministic kernels.

[FP8-RL](https://arxiv.org/abs/2601.18150) documents the additional mismatch
created by repeatedly quantizing changing weights for FP8 rollout, proposes
token-level importance corrections, and reports up to 44% rollout-throughput
gain while matching bfloat16 learning behavior in its tested settings.
**[D]** [Deterministic Inference across Tensor Parallel
Sizes](https://arxiv.org/abs/2511.17826) traces another mismatch to reduction
order across tensor-parallel configurations and constructs invariant kernels.
**[D]** Neither result means a generic importance sampler repairs every
mismatch; clipping trades variance for bias, while a wrong tokenizer or action
mask changes the event being assigned a probability.

Before RL, run a fixed prefix through both engines and compare:

- token IDs and attention masks exactly;
- top-$k$ token identities;
- maximum and mean absolute logit difference;
- sampled-token log-probability difference;
- per-layer or per-router checksums where available; and
- deterministic greedy continuation under matched kernels.

Repeat after every weight-sync path, quantization change, framework upgrade, and
parallelism change. A mismatch threshold is an experiment-specific engineering
gate, not a universal constant.

### 8.6 Tool observations, masks, and truncation can silently change the task

For an agent transcript, the loss mask should usually be

$$
m_t=
\begin{cases}
1,&\text{token sampled as a trainable model action},\\
0,&\text{system, user, environment, tool-result, padding, or replayed token}.
\end{cases}
$$

Training on tool-result tokens teaches the model to imitate observations it did
not choose. Excluding tool-call syntax, however, prevents the policy from
learning to invoke the environment. The correct unit is provenance, not a
hard-coded role name: some multi-agent systems treat another model's message as
an external observation, while self-play may intentionally train both sides.

Truncation creates censoring. Let $z_i=1$ indicate that a rollout terminated
naturally and $z_i=0$ indicate a length cutoff. Four common policies optimize
different objectives:

| Treatment | Benefit | Failure |
|---|---|---|
| mark every truncation incorrect | supplies a length pressure | confounds unfinished with semantically wrong |
| mask every truncation | avoids a false terminal label | removes long/hard cases and may reward length indirectly |
| length-aware soft penalty | preserves a signal | coefficient can dominate correctness |
| resume with larger budget | obtains a true terminal outcome | changes compute allocation and may be expensive |

DAPO uses overlong filtering plus a soft penalty; Open-R1's representative
CodeForces config masks truncations; rStar2-Agent reports that filtering
overlong samples worsened behavior. **[D]** The disagreement is evidence that
truncation policy is part of the task definition, not a housekeeping option.

Always distinguish:

- model-generated end-of-sequence;
- environment success or failure;
- maximum token length;
- maximum turns or tool calls;
- wall-clock timeout;
- sandbox crash or infrastructure loss; and
- safety termination.

Only the first two are ordinary semantic outcomes. Infrastructure failures
should normally be retried or labeled separately rather than turned into a
policy reward.

### 8.7 Dataset leakage and environment drift create counterfeit progress

Decontamination based only on exact prompt strings is insufficient. A model may
have seen:

- paraphrases, translated versions, or worked solutions;
- benchmark source code and tests;
- problem IDs, contest editorials, or answer keys;
- synthetic questions generated from benchmark seeds;
- tool documentation containing the desired action sequence; or
- previous policy trajectories later recycled into the training pool.

A defensible data ledger records original source, license, acquisition date,
hash, normalization code revision, deduplication cluster, contamination checks,
verifier version, generation model, and selection decision. Holdout membership
must be decided before generation or rejection sampling.

An external environment is also data and must be versioned:

$$
e=(\text{container digest},\text{dependencies},\text{fixtures},
\text{network snapshot},\text{seed},\text{resource limits}).
$$

A package update can make a code test pass; a search index refresh can reveal
the answer; a website redesign can break an action sequence; a changed timeout
can favor shorter plans. Replaying only the text transcript does not reproduce
the transition function.

For stateful tools, retain either a deterministic event log sufficient for
replay or a state snapshot. If neither is possible, report the run as an
evaluation on a time-bounded live environment rather than as a deterministic
benchmark.

### 8.8 Benchmark numbers are protocol outputs, not model constants

At least six choices commonly change a reasoning or agent benchmark score:

1. checkpoint and tokenizer revision;
2. prompt and chat template;
3. temperature, top-$p$, top-$k$, and random seed;
4. reasoning-token, tool-call, turn, and wall-clock budgets;
5. number of samples and aggregation rule; and
6. parser, verifier, judge, and environment revisions.

For independent samples with single-sample success probability $p$,

$$
\operatorname{pass@}k=1-(1-p)^k
$$

only under the simplifying independence and identical-distribution
assumptions. Best-of-$k$, majority vote, learned selection, and
multi-agent debate are different inference systems. A vendor's
$\operatorname{pass@}1$, $\operatorname{pass@}64$, and tool-assisted score
must not be placed in one column without protocol labels.

Public benchmark improvement also does not identify the training cause.
Architecture, pretraining data, synthetic distillation, supervised
fine-tuning, verifier RL, tool scaffolding, and inference compute can all change
simultaneously between product generations. The xAI evidence ledger is
especially sparse on these controls; the appropriate label is disclosed
capability, not reconstructed recipe.

### 8.9 A paper, a repository, and a checkpoint can describe three recipes

The Skywork audit above found both a controller-value mismatch and a task-list
duplication at a pinned revision. The rStar2-Agent public repository exposes a
small demonstration path rather than the private production cluster. Open-R1,
OpenRLHF, veRL, and slime continue to evolve after checkpoints and papers are
released. These are normal realities of fast-moving systems work, but they
invalidate floating citations.

Every reproduction should therefore bind:

$$
\text{claim}
\longleftrightarrow
\text{paper version}
\longleftrightarrow
\text{code commit}
\longleftrightarrow
\text{config hash}
\longleftrightarrow
\text{data revisions}
\longleftrightarrow
\text{container digests}
\longleftrightarrow
\text{checkpoint hash}.
$$

If one link is unavailable, mark it unknown. Do not silently fill a private
hyperparameter with a public-framework default.

## 9. Cross-case comparison: what is actually being optimized

### 9.1 Algorithm families

| Family | Credit unit | Extra learned model | Typical policy relation | Principal benefit | Principal failure surface | Representative case |
|---|---|---|---|---|---|---|
| GRPO / RLOO | response return broadcast to tokens | none | on-policy or mildly stale | simple and scalable | length, group-scale, and sparse-credit bias | DAPO, Open-R1 |
| critic PPO / GAE | token or step advantage | value model | proximal on-policy | temporal credit and variance reduction | value bias, memory, stale critic | Seed1.5 |
| implicit PRM | token reward inferred from outcome label | causal PRM | online policy plus online PRM | dense signal without step labels | correlated policy/PRM error | PRIME |
| curriculum RL | domain or difficulty stage | optional | stage-specific | avoids destructive gradient mixture | order dependence and forgetting | AceReason, Cascade |
| on-policy distillation | teacher log-probability reward | teacher policy | student samples on-policy | learns specialist distribution on student states | teacher cost and domain weighting | Nemotron MOPD |
| resample-on-correct | response advantage plus conditional extra rollouts | none | on-policy | spends compute near decision boundary | selection-induced objective change | rStar2-Agent |
| anchor-state group credit | episode plus repeated-state return | none | multi-turn on-policy | step signal without extra rollouts | requires repeated anchor states | GiGPO |
| traced agent RL | event/span credit transformed to token samples | optional reward/value model | runtime-dependent | retrofits arbitrary agents | ambiguous span attribution | Agent Lightning |
| async decoupled PPO | behavior correction plus proximal clipping | optional | explicitly off-policy | overlaps rollout and learning | stale/high-variance ratios | AReaL, Nemotron |

The table exposes why “which RL algorithm?” is incomplete. A full choice also
specifies the reward source, credit unit, sampling policy, token denominator,
environment state, and systems schedule.

### 9.2 Data and reward operations

| Operation | What it controls | Evidence from cases | Required audit |
|---|---|---|---|
| solvability filtering | removes zero-signal prompts | DAPO, rStar2, PRIME | retain rejection reasons; avoid benchmark overlap |
| dynamic nonzero-variance sampling | preserves within-group contrast | DAPO, slime buffer | log changing prompt distribution |
| repeated execution | estimates flaky code outcomes | AceReason | independent seeds and infrastructure-error class |
| synthetic solution generation | supplies SFT/curriculum traces | Nemotron, OpenThoughts | generator revision, sample count, rejection rule |
| rejection sampling | raises demonstration quality | Seed, OpenThoughts | measure diversity lost and teacher bias |
| multi-stage domain order | controls gradient interference | AceReason, Cascade | evaluate forgetting after every stage |
| hidden/generated tests | narrows verifier equivalence class | code pipelines | mutation score and false-positive audit |
| learned graders | expands beyond deterministic tasks | Seed, xAI | calibration, disagreement, prompt/version |
| tool/environment simulation | creates agent trajectories | ReTool, Search-R1, GiGPO | versioned state, mask, timeout semantics |

### 9.3 Vendor disclosure matrix

| Organization / line | Architecture disclosure | Data disclosure | RL objective disclosure | Systems disclosure | Exact reproducibility |
|---|---|---|---|---|---|
| ByteDance Seed1.5 | partial scale/active parameters | categories and SFT counts, not corpus | unusually detailed value-based components | substantial SRS/veRL concepts | no; weights/data/full config absent |
| ByteDance DAPO/ReTool | starts from named open models | released or described task data | high, including ablations/config | veRL-based | partial public reconstruction |
| NVIDIA Llama-Nemotron/AceReason | named open bases and compressed variants | mixtures and generated sources | stage and many hyperparameters | moderate | partial |
| NVIDIA Nemotron 3 | detailed hybrid MoE structure | token totals and broad mixtures | unified RL/MOPD concepts and selected settings | strong async/precision discussion | no full production replay |
| Microsoft Agent Lightning | runtime-agnostic | task-specific examples | credit transformation and algorithms | detailed client/server tracing | framework reproducible, production tasks vary |
| Microsoft rStar2-Agent | named open 14B base | pool construction and SFT protocol | detailed GRPO-RoC stages | partial production facts | public demo is not full production |
| xAI Grok line | exact for Grok-1; selected later facts | mostly broad categories | high-level RL/RLVR only | selected cluster and rollout-scale facts | no |
| open community | named open checkpoints | usually public IDs and configs | often exact | code available but moving | highest when commits and data revisions are pinned |

“Partial” is not a criticism; it is an evidence boundary. Commercial model
reports optimize for safety and capability communication, while scientific
reproduction needs raw data lineage, executable configs, and code revisions.

### 9.4 What changed from reasoning RL to agentic RL

The operational progression across these cases is:

$$
\text{single response}
\rightarrow
\text{response with executable verifier}
\rightarrow
\text{interleaved tool calls}
\rightarrow
\text{stateful multi-turn environment}
\rightarrow
\text{multi-environment asynchronous training}.
$$

Each transition adds a new object that must be versioned:

- executable verifier adds tests and sandbox;
- tool use adds action grammar and observation masks;
- stateful interaction adds transition state and temporal credit;
- multiple environments add reward calibration and sampling mixture;
- asynchrony adds behavior versions, weight synchronization, and correction.

Algorithmic sophistication is often downstream of this data-contract
sophistication. A simple RLOO update over correctly attributed, replayable
trajectories is scientifically stronger than an elaborate loss over ambiguous
transcripts.

## 10. A staged reproduction ladder

This ladder is designed to isolate failure causes. The numerical examples are
engineering starting points **[I]**, not claims about an optimal universal
recipe. Advance only after the acceptance gate at each level passes.

### Level 0: equation and mask tests without a model

Implement tiny tensor fixtures for:

1. GRPO with all-equal, one-positive, balanced, and continuous rewards;
2. per-sequence, fixed-length, and token-global denominators;
3. asymmetric clipping for positive and negative advantages;
4. leave-one-out baselines;
5. behavior/proximal/current importance ratios;
6. action masks containing tool observations and padding; and
7. natural termination versus truncation.

Finite-difference-check every differentiable loss. Verify that:

- all-equal group rewards produce zero centered advantage;
- padding and observations have exactly zero gradient;
- duplicating a response changes only the mathematically expected group terms;
- splitting one action into two spans does not change token-global loss;
- stale-policy correction equals one when policies match; and
- a deliberately wrong denominator causes a failing test.

**Acceptance gate:** equations, hand calculations, and implementation agree to
the selected precision; serialized trajectory round-trips preserve token IDs
and masks exactly.

### Level 1: offline SFT establishes the protocol

Use a 0.5B–1.5B open base model and a few thousand licensed, source-tracked
examples. Choose one explicit chat/action grammar:

~~~text
assistant reasoning -> assistant tool_call -> tool observation
-> assistant reasoning -> assistant final
~~~

Train only assistant-generated action tokens. Hold out complete source clusters,
not random rows, to reduce near-duplicate leakage. Compare:

- raw base with minimal elicitation;
- base with the target chat template but no training; and
- SFT checkpoint.

Measure exact-format validity, semantic success, response length, tool-call
validity, calibration, and contamination probes.

**Acceptance gate:** the model reliably emits parseable actions and improves
held-out semantic success without copying tool outputs as actions.

### Level 2: single-turn RLVR on a deterministic task

A minimal informative batch might use 64 distinct prompts with eight
completions each, giving 512 trajectories per update. **[I]** Start with
temperature around 0.7–1.0, one optimizer update per fresh batch, a conservative
learning-rate sweep, and no learned reward model. Use exact arithmetic answers
or deterministic code tests.

Run a factorial ablation over:

- sequence versus token-global aggregation;
- standard-deviation scaling on versus off;
- symmetric versus asymmetric clipping;
- truncation penalty versus masking; and
- static versus nonzero-variance prompt sampling.

Log outcome histograms by prompt. A single mean reward cannot show whether
dynamic sampling removed easy prompts.

**Acceptance gate:** held-out pass rate rises across at least three seeds,
entropy and response length remain within predeclared bounds, and an independent
verifier confirms the gain.

### Level 3: one external tool

Add exactly one tool: Python, code execution, or retrieval. Define a typed
action schema, deterministic parser, maximum calls, timeout, resource limits,
and observation sanitization. Record container and corpus/index digests.

Use four evaluation slices:

1. tool unnecessary;
2. tool useful;
3. tool returns an error;
4. tool returns plausible but adversarial content.

Compare outcome-only credit with a simple tool-cost penalty. Check whether the
policy invokes the tool to improve correctness or merely because tool-call
format receives reward.

**Acceptance gate:** net semantic success improves after subtracting tool cost,
error recovery is better than the SFT baseline, and no gradient flows through
observation tokens.

### Level 4: stateful multi-turn credit

Introduce an environment with reset and step, deterministic seed replay, and a
small action space. Preserve a trajectory graph if branches or nested calls are
possible. Compare:

- terminal reward broadcast;
- critic/GAE credit;
- implicit process reward; and
- repeated-state group credit when anchor states recur.

Do not compare methods at unequal environment interactions. Report successful
episodes per environment step and per wall-clock GPU hour.

**Acceptance gate:** the best method improves held-out success without an
increase in invalid actions, repeated loops, timeout gaming, or verifier
disagreement.

### Level 5: asynchronous rollout

First reproduce the Level 4 curve synchronously. Then introduce a bounded
rollout queue:

- store behavior version and log probability per action token;
- set an explicit maximum policy-version lag;
- reject or separately bucket samples outside the bound;
- compare behavior and learner engines on fixed-prefix probes;
- report importance-weight distribution and effective sample size; and
- checkpoint queue state, environment state, RNG state, and policy versions.

Increase lag one setting at a time. Compare throughput and sample efficiency,
not throughput alone.

**Acceptance gate:** asynchronous training reaches the synchronous quality
target in less wall time without exceeding predeclared KL, importance-weight,
or verifier-disagreement limits.

### Level 6: multi-environment production mixture

Only now combine math, code, search, software, or other environments. Give every
domain its own:

- sampler and target mixture;
- reward scale and calibration report;
- action schema and mask;
- timeout and cost model;
- verifier and adversarial audit set; and
- regression dashboard.

Test simultaneous mixture training against a sequential curriculum. Nemotron
and AceReason show why stage order can matter; a large scalar reward in one
domain can otherwise dominate shared parameters.

**Acceptance gate:** no domain regresses beyond a predeclared tolerance, the
mixture weights observed after filtering match the intended weights, and every
reported capability gain is traceable to a frozen evaluation protocol.

## 11. Minimal provenance, telemetry, and implementation contracts

### 11.1 Run manifest

The following is a minimal illustrative manifest:

~~~yaml
run:
  run_id: immutable-string
  parent_checkpoint_sha256: ...
  code_commit: ...
  config_sha256: ...
  container_digests: [...]
  cuda_driver: ...
  framework_versions: {...}

model:
  architecture: ...
  tokenizer_revision: ...
  chat_template_sha256: ...
  precision_train: bf16
  precision_rollout: bf16
  parallelism_train: {...}
  parallelism_rollout: {...}

data:
  prompt_sources:
    - id: ...
      revision: ...
      license: ...
      split_manifest_sha256: ...
  contamination_audit_revision: ...
  online_filter_revision: ...

environment:
  image_digest: ...
  reset_seed_policy: ...
  tool_schema_revision: ...
  timeout_policy: ...
  network_or_index_snapshot: ...

reward:
  verifier_commit: ...
  grader_model_revision: ...
  grader_prompt_sha256: ...
  component_weights: {...}
  truncation_policy: ...

algorithm:
  estimator: ...
  loss_denominator: ...
  reward_scaling: ...
  clip_low: ...
  clip_high: ...
  kl_reference_revision: ...
  max_policy_lag: ...

evaluation:
  harness_commit: ...
  prompt_template_sha256: ...
  budgets: {...}
  sample_count: ...
  aggregation: ...
~~~

Secrets and personal data should be represented by opaque identifiers, not
embedded in the manifest.

### 11.2 Per-trajectory schema

Each trajectory should retain:

~~~text
trajectory_id, task_id, task_source, task_revision
environment_revision, reset_seed, initial_state_hash
span_id, parent_span_id, turn_index, action_type
input_token_ids, generated_token_ids, action_loss_mask
behavior_policy_version, behavior_logprobs
proximal_policy_version, proximal_logprobs
tool_request, tool_response_hash, tool_latency, tool_status
raw_reward_components, verifier_output, grader_revision
terminal_reason, truncated, timeout, infrastructure_failure
wall_clock_start, wall_clock_end, retry_lineage
~~~

For privacy or storage reasons a raw tool response may be unavailable. Its
cryptographic hash, retrieval key, and retention policy should still be logged.

### 11.3 Metrics that reveal mechanism

For a batch of $N$ action tokens, log:

**Policy movement**

$$
\widehat D_{\mathrm{KL}}
=
\frac1N\sum_t
\left(\log\pi_{\mathrm{beh}}(a_t\mid s_t)
-\log\pi_\theta(a_t\mid s_t)\right),
\qquad
f_{\mathrm{clip}}
=
\frac1N\sum_t
\mathbf 1\{\rho_t\notin[1-\epsilon_l,1+\epsilon_h]\}.
$$

**Importance-weight health**

$$
\operatorname{ESS}
=
\frac{\left(\sum_t w_t\right)^2}{\sum_t w_t^2}.
$$

Report ESS as an absolute count and fraction of eligible tokens. Also report
weight quantiles, maximum, and fraction clipped.

**Exploration**

$$
\bar H
=
\frac1N\sum_t\mathcal H(\pi_\theta(\cdot\mid s_t)),
$$

plus answer diversity, program diversity, tool-choice diversity, and conditional
entropy by advantage sign and action type.

**Data-mixture health**

- proposed, accepted, filtered, truncated, and retried prompts by source;
- reward variance and fraction of zero-variance groups;
- difficulty bucket before and after online filtering;
- sequence-length quantiles by correctness;
- unique prompt clusters and repeated-trajectory rate.

**Environment and reward health**

- semantic success and each raw reward component;
- verifier disagreement and audited false-positive/false-negative estimates;
- invalid action, parser error, timeout, crash, and sandbox violation;
- tool calls, tool latency, environment steps, and monetary cost per success;
- IPT or other metamorphic-test failure rate.

**Systems health**

- rollout, verification, learner, synchronization, and idle time;
- tokens per second and accepted samples per GPU hour;
- policy-version lag distribution;
- queue age and cancellation rate;
- learner/rollout log-probability mismatch;
- KV-cache replay and environment replay failures.

### 11.4 Reference loop pseudocode

~~~text
freeze run manifest and evaluation protocol
initialize learner, proximal snapshot, rollout workers, and verifier

while budget remains:
    prompts = sampler.propose()
    trajectories = rollout(prompts, behavior_version)

    for trajectory in trajectories:
        validate token provenance and action mask
        classify terminal reason
        run deterministic verifiers and metamorphic audits
        store raw reward components and immutable environment metadata

    accepted = filter_with_logged_reasons(trajectories)
    if accepted changes the source or difficulty mixture:
        record pre-filter and post-filter distributions

    compute behavior, proximal, and current-policy log probabilities
    reject or quarantine samples beyond the declared staleness bound
    compute advantages and loss with an explicit denominator
    update learner

    run fixed-prefix engine-consistency probes
    run frozen held-out evaluations
    check entropy, KL, ESS, length, verifier, and safety gates

    if a gate fails:
        stop, preserve queue and environment state, and diagnose

    publish new policy version and retain synchronization lineage
~~~

The stop condition is part of the scientific method. Continuing after a known
entropy collapse or verifier exploit makes later reward curves harder, not
easier, to interpret.

### 11.5 Claim language

Use claim verbs that match evidence:

- **“The report states”** for a vendor-authored technical report;
- **“The pinned code configures”** for a source-code fact;
- **“The experiment reports”** for a measured result under its protocol;
- **“Our audit computes”** for arithmetic derived directly from disclosed
  numbers;
- **“This suggests”** for a plausible mechanism;
- **“Unknown”** when the artifact does not disclose the fact.

Do not convert “supports asynchronous RL” into “the model was trained
asynchronously,” or “the organization uses this framework” into “this exact
commit trained that checkpoint.”

## 12. Primary-source and code map

The following map favors first-party reports, papers, repositories, model cards,
and dataset cards. Revisions explicitly audited in this chapter remain pinned
even when the project main branch moves.

### ByteDance Seed and veRL

- [Seed1.5-Thinking technical report](https://arxiv.org/abs/2504.13914)
- [DAPO paper](https://arxiv.org/abs/2503.14476)
- [DAPO repository](https://github.com/BytedTsinghua-SIA/DAPO)
- [ReTool paper](https://arxiv.org/abs/2504.11536)
- [HybridFlow / veRL paper](https://arxiv.org/abs/2409.19256)
- [veRL repository](https://github.com/volcengine/verl)
- [Seed2.0 model card](https://arxiv.org/abs/2607.00248)
- [Seed2.1 official release](https://seed.bytedance.com/en/blog/seed2-1-officially-released-advancing-ai-productivity)

### NVIDIA Nemotron

- [Llama-Nemotron technical report](https://arxiv.org/abs/2505.00949)
- [AceReason-Nemotron paper](https://arxiv.org/abs/2505.16400)
- [ProRL paper](https://arxiv.org/abs/2505.24864)
- [Nemotron-Cascade paper](https://arxiv.org/abs/2512.13607)
- [Nemotron-Cascade 2 paper](https://arxiv.org/abs/2603.19220)
- [Nemotron 3 Nano technical report](https://arxiv.org/abs/2512.20848)
- [Nemotron 3 Nano Omni report](https://arxiv.org/abs/2604.24954)
- [Nemotron 3 Super technical report](https://arxiv.org/abs/2604.12374)
- [Nemotron 3 Ultra technical report](https://arxiv.org/abs/2606.15007)
- [Nemotron-Labs-3-Puzzle-75B-A9B report](https://arxiv.org/abs/2607.04371)

### Microsoft

- [Agent Lightning paper](https://arxiv.org/abs/2508.03680)
- [Agent Lightning repository](https://github.com/microsoft/agent-lightning)
- [rStar2-Agent paper](https://arxiv.org/abs/2508.20722)
- [rStar2-Agent audited repository revision](https://github.com/microsoft/rStar/tree/ecbfb943e202b4ed017d2d35f2029917b27db4cd)

### xAI

- [Grok-1 open release](https://x.ai/news/grok-os)
- [Grok-1 repository](https://github.com/xai-org/grok-1)
- [Grok-1.5 announcement](https://x.ai/news/grok-1.5)
- [Grok-2 announcement](https://x.ai/news/grok-2)
- [Grok-3 announcement](https://x.ai/news/grok-3)
- [Grok 4 announcement](https://x.ai/news/grok-4)
- [Grok 4 model card](https://data.x.ai/2025-08-20-grok-4-model-card.pdf)
- [Grok 4 Fast](https://x.ai/news/grok-4-fast)
- [Grok 4.1](https://x.ai/news/grok-4-1)
- [Grok 4.1 Fast](https://x.ai/news/grok-4-1-fast)
- [Grok 4.20 system card](https://data.x.ai/2026-04-07-grok-4-20-model-card.pdf)
- [Grok 4.3 on Amazon Bedrock](https://x.ai/news/grok-amazon-bedrock)
- [Grok 4.5](https://x.ai/news/grok-4-5)

### Open community

- [Open-R1 audited revision](https://github.com/huggingface/open-r1/tree/1416fa0cf21595d2083b399a2a0bbddd7f6e9563)
- [OpenThoughts paper](https://arxiv.org/abs/2506.04178)
- [OpenThoughts repository](https://github.com/open-thoughts/open-thoughts)
- [AReaL paper](https://arxiv.org/abs/2505.24298)
- [AReaL repository](https://github.com/inclusionAI/AReaL)
- [PRIME paper](https://arxiv.org/abs/2502.01456)
- [Skywork Open Reasoner 1 paper](https://arxiv.org/abs/2505.22312)
- [Skywork audited revision](https://github.com/SkyworkAI/Skywork-OR1/tree/64e96afa213ae89d0ad21932106d3b8aafe9ace2)
- [Search-R1 paper](https://arxiv.org/abs/2503.09516)
- [Search-R1 repository](https://github.com/PeterGriffinJin/Search-R1)
- [GiGPO paper](https://arxiv.org/abs/2505.10978)
- [OpenRLHF audited revision](https://github.com/OpenRLHF/OpenRLHF/tree/bc71bb19464aca306b33080b2d2bb45d154e2f49)
- [slime audited revision](https://github.com/THUDM/slime/tree/fb42ae456fac8166afb604f13b30d22bb3c75053)

### Failure and systems audits

- [Critical R1-Zero / Dr.GRPO analysis](https://arxiv.org/abs/2503.20783)
- [Entropy mechanism study](https://arxiv.org/abs/2505.22617)
- [Verifier-gaming and IPT study](https://arxiv.org/abs/2604.15149)
- [FP8-RL](https://arxiv.org/abs/2601.18150)
- [Deterministic inference across tensor-parallel sizes](https://arxiv.org/abs/2511.17826)

## 13. Final synthesis

The deepest commonality across ByteDance, NVIDIA, Microsoft, xAI, and the open
community is not one optimizer. It is the construction of a closed learning
loop whose expensive parts are automated:

$$
\text{task generation and selection}
\rightarrow
\text{policy interaction}
\rightarrow
\text{machine-checkable or learned feedback}
\rightarrow
\text{credit assignment}
\rightarrow
\text{distributed update}
\rightarrow
\text{adversarial evaluation}.
$$

ByteDance contributes unusually concrete value-learning, critic-free, tool, and
systems variants. NVIDIA shows how a commercial open-model program composes
distillation, domain curricula, on-policy specialist distillation, hybrid MoE
architecture, and asynchronous multi-environment RL. Microsoft makes the agent
runtime itself an observable training interface and offers a detailed
math-plus-Python curriculum. xAI demonstrates rapidly increasing scale and
hours-long agentic rollouts, but its public evidence remains insufficient to
reconstruct the optimizer or data pipeline. The open community supplies the
most inspectable data, algorithm, and orchestration artifacts, together with
the clearest examples of paper-code drift.

Five principles survive every case:

1. **The verifier is a specification, not merely a labeler.** Its blind spots
   become optimization targets.
2. **The trajectory schema is part of the model objective.** Token provenance,
   masks, policy versions, and terminal reasons determine the gradient.
3. **Data operations are algorithmic operations.** Solvability filters,
   rejection sampling, dynamic sampling, and curricula change the effective
   objective.
4. **Scale changes systems semantics.** Quantization, asynchronous queues,
   weight synchronization, and kernel differences can make the rollout policy
   differ from the learner policy.
5. **A benchmark score is evidence only under a frozen protocol.** It does not
   reveal which stage, dataset, or optimization choice caused the change.

For a reader building from zero, the shortest path to deep understanding is
therefore not to begin with a giant vendor checkpoint. Begin with a hand-checked
loss, a lossless trajectory record, a tiny deterministic verifier, and a
replayable environment. Add tools, temporal credit, asynchrony, and task
mixtures one at a time. At each step, preserve the ability to say exactly which
tokens were actions, which policy sampled them, which verifier rewarded them,
and which code and data revision produced the update. That chain of evidence is
the foundation on which every credible agentic-RL result rests.
