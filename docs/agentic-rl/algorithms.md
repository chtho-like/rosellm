# Algorithms for LLM and Agentic RL

An algorithm name is not a training recipe. The update depends on sampling,
reward, credit unit, baseline, importance ratio, clipping, KL, loss reduction,
data reuse, and policy lag. This chapter decomposes those choices and shows when
the major algorithm families are appropriate.

## 1. A common policy-gradient template

Most online methods in this chapter can be written schematically as

\[
\widehat{\nabla J}(\theta)=
\frac{1}{Z}\sum_{i,t,j}
m_{i,t,j}\,w_{i,t,j}(\theta)\,
\widehat A_{i,t,j}\,
\nabla_\theta\log\pi_\theta(x_{i,t,j}\mid h_{i,t},x_{i,t,<j})
-\nabla_\theta\mathcal R(\theta).
\]

- \(i\): trajectory/sample
- \(t\): environment turn
- \(j\): policy-generated token
- \(m\): action-token mask
- \(w\): importance/clipping/truncation term
- \(\widehat A\): advantage or centered return
- \(Z\): token/turn/sequence/task normalization
- \(\mathcal R\): KL, entropy, constraint, or auxiliary regularizer

To compare two methods, fill in these six blanks rather than comparing their
names.

## 2. Behavioral cloning and SFT

SFT minimizes negative log-likelihood of demonstrated policy tokens:

\[
L_{\text{SFT}}=-\frac{1}{Z}\sum_{i,t,j}
m_{i,t,j}\log\pi_\theta(x^*_{i,t,j}\mid h^*_{i,t},x^*_{i,t,<j}).
\]

Strengths:

- stable, simple, and data-efficient when demonstrations are high quality;
- teaches syntax and provides support before online exploration;
- can distill expensive search/teacher trajectories.

Limits:

- trains on expert-state distribution, not states induced by learner mistakes;
- weights every demonstrated token rather than task outcome;
- cannot exceed demonstration support through reward-driven exploration;
- success filtering can hide unsuccessful attempts and selection cost.

SFT remains part of almost every practical Agentic RL pipeline, but it is not RL.

## 3. REINFORCE

For sampled trajectory \(\tau_i\),

\[
\widehat g=rac{1}{B}\sum_i\sum_t
(G_{i,t}-b(h_{i,t}))\nabla\log\pi_\theta(a_{i,t}\mid h_{i,t}).
\]

Use REINFORCE when:

- rollouts are genuinely on-policy;
- reward is cheap enough to sample many trajectories;
- a simple unbiased estimator is valuable for debugging;
- no reliable critic exists.

Its primary problem is variance. A terminal binary reward on a forty-turn
episode supplies weak evidence to early decisions. Baselines, reward-to-go,
curricula, process rewards, and branching rollouts address different parts of
that problem.

### REINFORCE with a batch baseline

Subtract a prompt/task-conditioned batch mean. If each task has one sample,
mixing unrelated task difficulty into a global mean produces noisy credit. If a
sample contributes to its own mean, the finite-batch estimator is scaled/coupled;
a leave-one-out baseline has cleaner conditional independence.

### REINFORCE++ and related stabilizations

Modern “REINFORCE++” recipes combine REINFORCE-style critic-free advantages
with PPO-derived engineering such as clipping, KL control, and normalization.
The label is used differently across implementations. Inspect the actual loss,
especially whether ratios are clipped, how rewards are normalized, and whether
old-policy samples are reused. See Hu,
[*REINFORCE++*](https://arxiv.org/abs/2501.03262) (2025) and the implementation
used by the selected framework.

## 4. RLOO

For \(G\) independent outputs on the same task,

\[
\widehat A_i=R_i-\frac{1}{G-1}\sum_{k\ne i}R_k.
\]

RLOO removes the critic and uses other samples as a control variate. It is
attractive when:

- tasks can be evaluated with a scalar outcome;
- multiple rollouts per task are affordable;
- within-task comparisons remove large difficulty variance;
- long-horizon state values are too expensive or unreliable.

It still assigns coarse trajectory-level credit unless rewards/returns are
defined per turn. With \(G=2\), advantages are exact opposites and highly noisy.
As \(G\) grows, rollout cost rises. Ahmadian et al. found simple REINFORCE-style
methods competitive for RLHF when carefully tuned
([*Back to Basics*](https://arxiv.org/abs/2402.14740), 2024).

## 5. Actor–critic and PPO

PPO normally combines:

1. on/near-policy rollouts under \(\pi_{\text{old}}\);
2. a critic \(V_\phi(h_t)\);
3. GAE advantages;
4. clipped importance-ratio surrogate;
5. value loss, entropy, and often reference KL;
6. several minibatch epochs over one rollout buffer.

The policy objective is

\[
L^{\text{CLIP}}=
\mathbb E\left[
\min(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)
\right].
\]

### When PPO is valuable

- intermediate states are meaningful and a critic can learn them;
- reward arrives across turns and GAE reduces variance;
- rollouts are expensive enough that multiple epochs help;
- additional critic memory/compute is affordable;
- careful trust-region-like control matters.

### Why PPO is hard for LLM agents

- a critic comparable in size to the actor is expensive;
- value prediction over long, diverse language histories is difficult;
- multiple epochs increase off-policy mismatch;
- token-level ratios do not directly constrain semantic-action change;
- variable lengths and sparse reward make loss/value normalization delicate;
- asynchronous collectors introduce additional behavior-policy lag.

### PPO checklist

- Are values predicted per token, turn boundary, or final sequence?
- Are time-limit truncations bootstrapped?
- Over which population are advantages normalized?
- Is value clipping used, and relative to which old value?
- How many optimizer epochs/minibatches reuse each sample?
- Is ratio clipping symmetric? Is there a target-KL early stop?
- Is reference KL a reward, loss, or both?
- What is the global loss denominator?

## 6. GRPO

DeepSeekMath introduced Group Relative Policy Optimization to eliminate the
learned critic. For each question/task, sample \(G\) outputs, compute rewards,
and estimate relative advantages, commonly

\[
\hat A_i=
\frac{R_i-\operatorname{mean}(R_{1:G})}
{\operatorname{std}(R_{1:G})+\varepsilon}.
\]

A PPO-like clipped surrogate then compares current and old policy probabilities.
Read the exact objective in DeepSeekMath
([Shao et al., 2024, Section 3](https://arxiv.org/abs/2402.03300)); later
implementations called “GRPO” differ in KL and normalization.

### Advantages

- no critic model or value-loss tuning;
- within-task comparison reduces between-task difficulty variance;
- verifiable scalar rewards are easy to integrate;
- group sampling naturally supplies exploration and pass-rate statistics.

### Limitations

- group generation multiplies rollout cost;
- groups with uniform reward contribute no relative signal;
- group standard deviation is noisy and can distort prompt weighting;
- terminal reward is usually copied to all response tokens;
- token/sequence length normalization can bias toward or against long outputs;
- multi-turn trajectories need an additional credit-decomposition rule.

### GRPO is not inherently agentic

GRPO can optimize one generated math response or a multi-turn tool trajectory.
The environment and credit unit determine whether the application is agentic.

## 7. Dr. GRPO

*Understanding R1-Zero-Like Training* identifies two important sources of bias
in common GRPO implementations:

1. dividing by group reward standard deviation changes the weight of prompts;
2. averaging each response's token loss by its length changes response/token
   weighting.

Dr. GRPO removes these normalization choices in the analyzed objective, using a
constant normalization instead
([Liu et al., 2025](https://arxiv.org/abs/2503.20783)). The lesson is broader
than one variant: an apparently harmless denominator defines which prompts and
tokens dominate the update.

Use controlled experiments. Removing normalization can increase raw gradient
scale and require learning-rate/clipping changes; an ablation that changes both
does not isolate estimator bias.

## 8. DAPO

DAPO reports an open large-scale reasoning-RL recipe built around four changes
([Yu et al., 2025](https://arxiv.org/abs/2503.14476)):

1. **clip-higher:** separate lower/upper ratio bounds to preserve exploration;
2. **dynamic sampling:** discard groups with all-correct or all-incorrect rewards
   and resample until a useful batch is formed;
3. **token-level policy-gradient loss:** aggregate over valid tokens rather than
   first averaging each sample equally;
4. **overlong reward shaping:** avoid noisy hard truncation penalties near the
   maximum length.

Operational implications:

- dynamic sampling needs an attempt census and can be expensive as the policy
  saturates or tasks are impossible;
- token-level aggregation changes the influence of response length;
- asymmetric clipping affects positive and negative advantage exploration;
- soft overlong penalties must not become the main learned proxy.

DAPO is a recipe, not simply a replacement equation.

## 9. GSPO and sequence-level ratios

Token-level PPO clipping treats each token ratio separately even though reward
often belongs to the complete response. Group Sequence Policy Optimization
defines an importance ratio from sequence likelihood and applies sequence-level
clipping/optimization
([Zheng et al., *GSPO*](https://arxiv.org/abs/2507.18071), 2025).

If sequence score uses a geometric mean,

\[
\rho_i^{\text{seq}}=
\exp\left(\frac{1}{L_i}\sum_j
[\log\pi_\theta(x_{i,j})-\log\pi_{\text{old}}(x_{i,j})]
\right),
\]

the length average avoids exponentially small/large raw products but defines a
particular length normalization. Sequence clipping aligns the trust signal with
sequence reward; it supplies less localized control over individual token
changes. For multi-turn agents, decide whether the “sequence” is a turn or a
complete trajectory.

## 10. ReMax

ReMax uses a deterministic greedy rollout as a baseline for the stochastic
sample, avoiding a critic
([Li et al., 2023](https://arxiv.org/abs/2310.10505)). Conceptually,

\[
\hat A=R(y_{\text{sample}})-R(y_{\text{greedy}}).
\]

It can be effective when an extra baseline rollout is cheaper than critic
training and reward is sequence-level. In stochastic agent environments, both
rollouts should share a controlled initial state/seed when interpreting the
difference; otherwise environment randomness contaminates the baseline.

## 11. Process-reward and value-free methods

### PRIME

PRIME learns an implicit process reward from policy rollouts and outcome labels,
then uses dense rewards during policy optimization
([Cui et al., 2025](https://arxiv.org/abs/2502.01456)). It addresses sparse
credit without requiring manually labeled reasoning steps. Its learned process
signal can still be wrong or exploitable; validate it against interventions and
held-out outcomes.

### Explicit process reward models

Train a model on human/AI/verifier labels for intermediate steps. The policy may
optimize the sum of step rewards or use them to estimate advantages. Specify:

- label target and information visible to labeler;
- whether scores are calibrated across positions/tasks;
- how step count changes total reward;
- whether outcome reward is retained;
- adversarial robustness and policy/judge relationship.

### Potential-based shaping

Adding \(\gamma\Phi(s')-\Phi(s)\) preserves optimal policies under standard
conditions. Learned “progress” scores generally do not have this guarantee.

## 12. Turn-level and hierarchical agent methods

Trajectory-level GRPO can treat a twenty-turn episode as one response, but this
does not solve temporal credit. Agent-specific methods add structure.

### Turn-level advantages

Compute returns from each action boundary and apply the turn advantage to its
generated tokens. This uses environment reward timing or a value/process model.
It reduces credit distance but can reward locally useful actions that harm the
final task unless downstream return remains included.

### Group-in-group baselines

GiGPO groups at both trajectory and step levels to exploit repeated/branched
interaction structure
([Feng et al., 2025](https://arxiv.org/abs/2505.10978)). Its value depends on
meaningful state/action grouping; superficial text similarity is not causal
equivalence.

### Hierarchical decomposition

Agent Lightning models arbitrary agent execution as transitions and decomposes
credit across higher-level and lower-level decisions
([Luo et al., 2025](https://arxiv.org/abs/2508.03680)). Hierarchical methods are
useful when planner/subgoal/executor boundaries have operational meaning. They
can misassign credit if the decomposition is imposed by a brittle parser.

### Branching counterfactuals

Clone an environment state, try multiple candidate actions or continuations,
and compare downstream outcomes. This supplies state-conditional relative
credit, but state cloning and extra rollouts are expensive. Shared randomness
can reduce variance when transitions are stochastic.

## 13. Off-policy and asynchronous correction

Agent rollouts are slow, so collectors frequently lag behind the trainer.
Possible controls:

1. **bounded staleness:** reject trajectories older than \(K\) versions;
2. **ratio/KL gate:** reject or downweight samples whose behavior/target
   divergence exceeds a threshold;
3. **truncated importance weights:** trade bias for variance control;
4. **V-trace:** clipped off-policy actor–critic targets
   ([Espeholt et al., 2018](https://arxiv.org/abs/1802.01561));
5. **single-update consumption:** avoid repeated reuse of already stale samples;
6. **frequent lightweight weight broadcast:** reduce lag at system cost;
7. **fully off-policy methods/replay:** require a different algorithmic analysis
   than calling the data “near on-policy.”

AReaL studies asynchronous language-model RL and staleness-aware training
([Fu et al., 2025](https://arxiv.org/abs/2505.24298)). Measure lag both in
versions and actual divergence; ten tiny updates can be closer than one large
update.

## 14. Offline preference objectives

Offline preference optimization is useful for bootstrap and alignment but is not
an online agent algorithm by itself.

### DPO

For preferred \(y_w\), rejected \(y_l\), reference \(\pi_{\text{ref}}\),

\[
L_{\text{DPO}}=-\mathbb E\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right).
\]

It avoids reward-model and online-rollout infrastructure. For agent trajectories,
the pair can be complete trajectories or segments, but fixed preferences do not
automatically cover states induced by the updated policy.

### When to use offline preferences

- safety/style/subjective outcomes lack robust scalar reward;
- online environment access is expensive or risky;
- human corrections are available;
- a conservative bootstrap is needed before exploration.

Refresh data online or evaluate distribution shift if the trained policy moves
beyond the comparison dataset.

## 15. Model-based, search, and self-play extensions

### Search plus policy learning

Tree search, best-of-N, beam search, and verifier-guided expansion can produce
stronger trajectories. Options:

- distill selected trajectories with SFT;
- use search outcomes as policy-gradient samples with correct behavior
  probabilities (often difficult);
- train value/process models from the search tree;
- improve a policy iteratively as in expert iteration.

Do not report search-time and single-sample policies under the same compute
budget without separation.

### Learned world models

A model of transitions can generate cheap experience or plan imagined futures.
Model bias is especially dangerous when an LLM agent exploits simulated tool or
user behavior that does not match the real environment. Alternate model
learning with real-environment validation.

### Self-play

Use opponent/user/task-generator policies to create curricula. Population-based
self-play can prevent overfitting to one opponent. Anchor progress to external
tasks so agents do not learn private protocols that score well only against one
another.

## 16. Algorithm selection matrix

| Situation | Useful starting point | Why | Main warning |
|---|---|---|---|
| High-quality demonstrations, weak initial policy | SFT | stable support acquisition | covariate shift |
| Scalar verified outcome, cheap multiple samples | RLOO/GRPO | no critic, within-task baseline | coarse credit and group cost |
| Meaningful intermediate states/rewards | PPO + turn critic/GAE | temporal credit and sample reuse | critic cost/error |
| Fully on-policy simple baseline | REINFORCE + valid baseline | transparent estimator | high variance |
| Concern about GRPO normalization bias | Dr. GRPO-style ablation | exposes denominator effects | retune scale fairly |
| Need open reasoning-RL recipe | DAPO-style recipe | exploration/dynamic sampling details | selection and length effects |
| Long stale asynchronous rollouts | bounded lag + off-policy correction | improves utilization | bias/variance and version integrity |
| Sparse long-horizon reward | value/process/branching/hierarchy | denser credit | learned proxy exploitation |
| Subjective or safety preference | reward model + PPO or DPO | human/AI preference signal | judge/reward overoptimization |
| Expensive unsafe environment | SFT/offline data + sandboxed conservative RL | limits exploration risk | weak coverage of novel states |

Start with the simplest estimator that can represent the credit structure, then
add complexity only after a controlled diagnostic demonstrates the need.

## 17. Fair algorithm comparison

Hold constant or account for:

- initial policy and tokenizer/template;
- unique task distribution and environment versions;
- total launched and accepted rollout tokens;
- number of environment executions and verifier/judge calls;
- accelerator-hours and wall-clock budget;
- sampling configuration and inference-time samples;
- reward function and all shaping;
- context/action/episode limits;
- loss normalization and global batch;
- evaluation scaffold, tools, seeds, and budgets;
- hyperparameter search budget.

If GRPO samples eight outputs per prompt and PPO samples one, equal optimizer
steps do not mean equal data or compute. If dynamic sampling discards easy/hard
groups, report all attempted samples.

## 18. Ablation order

When a run fails, change one axis at a time:

1. verify masks, log-probabilities, ratios, rewards, and group/turn alignment;
2. reproduce the update on a frozen offline batch;
3. compare no-update evaluation to rule out environment drift;
4. inspect curriculum difficulty and trainable-group fraction;
5. vary baseline/advantage estimator;
6. vary loss normalization;
7. vary clipping/KL/learning rate;
8. add process/value credit only after trajectory-level baseline is understood;
9. introduce asynchrony only after synchronous parity.

Algorithmic novelty cannot rescue a corrupted trajectory contract.

## References

1. Ronald J. Williams,
   [“Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning”](https://doi.org/10.1007/BF00992696),
   1992.
2. John Schulman et al.,
   [“Proximal Policy Optimization Algorithms”](https://arxiv.org/abs/1707.06347),
   2017.
3. Arash Ahmadian et al.,
   [“Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs”](https://arxiv.org/abs/2402.14740),
   2024.
4. Zhihong Shao et al.,
   [“DeepSeekMath”](https://arxiv.org/abs/2402.03300), 2024.
5. Qiying Yu et al.,
   [“DAPO”](https://arxiv.org/abs/2503.14476), 2025.
6. Zichen Liu et al.,
   [“Understanding R1-Zero-Like Training”](https://arxiv.org/abs/2503.20783),
   2025.
7. Sergey Levine et al.,
   [“Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems”](https://arxiv.org/abs/2005.01643),
   2020.
