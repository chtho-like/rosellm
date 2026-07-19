# Algorithms for Large Language Models and Agentic Reinforcement Learning

This chapter studies **Large Language Model (LLM) Agentic Reinforcement Learning
(Agentic RL)**. An algorithm name is not a training recipe. The update depends on sampling,
reward, credit unit, baseline, importance ratio, clipping, KL, loss reduction,
data reuse, and policy lag. This chapter decomposes those choices and shows when
the major algorithm families are appropriate.

## 1. A common policy-gradient template

Most online methods in this chapter can be written schematically as

$$
\widehat{\nabla J}(\theta)=
\frac{1}{Z}\sum_{i,t,j}
m_{i,t,j}\,w_{i,t,j}(\theta)\,
\widehat A_{i,t,j}\,
\nabla_\theta\log\pi_\theta(x_{i,t,j}\mid h_{i,t},x_{i,t,<j})
-\nabla_\theta\mathcal R(\theta).
$$

- $i$: trajectory/sample
- $t$: environment turn
- $j$: policy-generated token
- $m$: action-token mask
- $w$: importance/clipping/truncation term
- $\widehat A$: advantage or centered return
- $Z$: token/turn/sequence/task normalization
- $\mathcal R$: KL, entropy, constraint, or auxiliary regularizer

To compare two methods, fill in these six blanks rather than comparing their
names.

## 2. Behavioral cloning and Supervised Fine-Tuning

**Supervised Fine-Tuning (SFT)** is behavioral cloning for model outputs. It
minimizes negative log-likelihood of demonstrated policy tokens:

$$
L_{\text{SFT}}=-\frac{1}{Z}\sum_{i,t,j}
m_{i,t,j}\log\pi_\theta(x^*_{i,t,j}\mid h^*_{i,t},x^*_{i,t,<j}).
$$

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

For sampled trajectory $\tau_i$,

$$
\widehat g=\frac{1}{B}\sum_i\sum_t
(G_{i,t}-b(h_{i,t}))\nabla\log\pi_\theta(a_{i,t}\mid h_{i,t}).
$$

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

## 4. REINFORCE Leave-One-Out

**REINFORCE Leave-One-Out (RLOO)** uses other samples for the same task as a
baseline for the current sample.

For $G$ independent outputs on the same task,

$$
\widehat A_i=R_i-\frac{1}{G-1}\sum_{k\ne i}R_k.
$$

RLOO removes the critic and uses other samples as a control variate. It is
attractive when:

- tasks can be evaluated with a scalar outcome;
- multiple rollouts per task are affordable;
- within-task comparisons remove large difficulty variance;
- long-horizon state values are too expensive or unreliable.

It still assigns coarse trajectory-level credit unless rewards/returns are
defined per turn. With $G=2$, advantages are exact opposites and highly noisy.
As $G$ grows, rollout cost rises. Ahmadian et al. found simple REINFORCE-style
methods competitive for RLHF when carefully tuned
([*Back to Basics*](https://arxiv.org/abs/2402.14740), 2024).

## 5. Actor–critic and Proximal Policy Optimization

**Proximal Policy Optimization (PPO)** normally combines:

1. on/near-policy rollouts under $\pi_{\text{old}}$;
2. a critic $V_\phi(h_t)$;
3. GAE advantages;
4. clipped importance-ratio surrogate;
5. value loss, entropy, and often reference KL;
6. several minibatch epochs over one rollout buffer.

The policy objective is

$$
L^{\text{CLIP}}=
\mathbb E\left[
\min(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)
\right].
$$

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

## 6. Group Relative Policy Optimization

DeepSeekMath introduced **Group Relative Policy Optimization (GRPO)** to eliminate the
learned critic. For each question/task, sample $G$ outputs, compute rewards,
and estimate relative advantages, commonly

$$
\hat A_i=
\frac{R_i-\operatorname{mean}(R_{1:G})}
{\operatorname{std}(R_{1:G})+\varepsilon}.
$$

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

## 8. Decoupled Clip and Dynamic sAmpling Policy Optimization

**Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)** reports an
open large-scale reasoning-RL recipe built around four changes
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

## 9. Group Sequence Policy Optimization and sequence-level ratios

Token-level PPO clipping treats each token ratio separately even though reward
often belongs to the complete response. **Group Sequence Policy Optimization
(GSPO)** defines an importance ratio from sequence likelihood and applies sequence-level
clipping/optimization
([Zheng et al., *GSPO*](https://arxiv.org/abs/2507.18071), 2025).

If sequence score uses a geometric mean,

$$
\rho_i^{\text{seq}}=
\exp\left(\frac{1}{L_i}\sum_j
[\log\pi_\theta(x_{i,j})-\log\pi_{\text{old}}(x_{i,j})]
\right),
$$

the length average avoids exponentially small/large raw products but defines a
particular length normalization. Sequence clipping aligns the trust signal with
sequence reward; it supplies less localized control over individual token
changes. For multi-turn agents, decide whether the “sequence” is a turn or a
complete trajectory.

For $G$ responses to one query, the published objective is

$$
J_{\mathrm{GSPO}}(\theta)
=\frac1G\sum_{i=1}^{G}
\min\left(
\rho_i^{\mathrm{seq}}\widehat A_i,
\operatorname{clip}(\rho_i^{\mathrm{seq}},1-\epsilon_l,1+\epsilon_h)
\widehat A_i
\right).
$$

The average is over responses, not over all response tokens. Consequently,
each response has equal outer weight. Differentiating an unclipped response
shows why its action tokens are weighted equally:

$$
\nabla_\theta \rho_i^{\mathrm{seq}}
=\rho_i^{\mathrm{seq}}\frac1{L_i}
\sum_{t=1}^{L_i}\nabla_\theta
\log\pi_\theta(y_{i,t}\mid x,y_{i,<t}).
$$

This is a different estimator from multiplying every token by its own token
ratio. It also means one outlier token can move the sequence ratio outside the
trust band and suppress the complete sequence.

### GSPO-token for multi-turn credit

The paper defines a token-customizable variant for settings such as multi-turn
RL. Let `sg` denote stop-gradient and

$$
\rho_{i,t}^{\mathrm{GSPO-token}}
=\operatorname{sg}(\rho_i^{\mathrm{seq}})
\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
{\operatorname{sg}[\pi_\theta(y_{i,t}\mid x,y_{i,<t})]}.
$$

Numerically, the fraction is one, so every token sees the same sequence ratio.
Its derivative flows only through token $t$. If all token advantages equal

$$
\widehat A_{i,t}=\widehat A_i,
$$

GSPO and GSPO-token have the same objective value and theoretical gradient. If
turn/token advantages differ, GSPO-token preserves those differences while
keeping a sequence-level clipping condition. The repository tests both the
value and gradient equivalence rather than assuming the stop-gradient trick is
cosmetic.

Do not transfer the paper's tiny GSPO clip bounds mechanically. Its controlled
Qwen3-30B-A3B run used left/right bounds $3\times10^{-4}$ and
$4\times10^{-4}$, while its GRPO baseline used $0.2/0.27$, precisely
because sequence and token ratios have different scales.

## 10. Soft Adaptive Policy Optimization

**Soft Adaptive Policy Optimization (SAPO)** replaces the binary hard-clipping
gate with a differentiable token gate
([Gao et al., 2025](https://arxiv.org/abs/2511.20347)). For token ratio

$$
r_{i,t}=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}}),
$$

choose a temperature by advantage sign,

$$
\tau_{i,t}=\begin{cases}
\tau_{\mathrm{pos}},&\widehat A_{i,t}>0,\\
\tau_{\mathrm{neg}},&\widehat A_{i,t}\le0,
\end{cases}
$$

and define

$$
f_{i,t}(r)=\frac{4}{\tau_{i,t}}
\sigma[\tau_{i,t}(r-1)].
$$

The sequence-normalized objective is

$$
J_{\mathrm{SAPO}}(\theta)
=\frac1G\sum_{i=1}^{G}\frac1{L_i}
\sum_{t=1}^{L_i}f_{i,t}(r_{i,t})\widehat A_{i,t}.
$$

This formula applies the smooth surrogate to the ratio itself; it is not
`ratio × advantage × log-probability`. With

$$
p_{i,t}=\sigma[\tau_{i,t}(r_{i,t}-1)],
$$

differentiation gives

$$
\nabla_\theta J_{\mathrm{SAPO}}
=\frac1G\sum_i\frac1{L_i}\sum_t
\underbrace{4p_{i,t}(1-p_{i,t})}_{\text{smooth gradient gate}}
r_{i,t}\widehat A_{i,t}
\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t}).
$$

At $r=1$, the gate is exactly one for any positive temperature, so the
on-policy gradient matches the unclipped ratio objective. Farther away, it
decays continuously instead of switching to zero at a hard boundary. The paper
uses $\tau_{\mathrm{neg}}>\tau_{\mathrm{pos}}$: a negative sampled-token
update raises logits for a very large set of unsampled vocabulary items, so its
off-policy gradient is damped faster.

SAPO is token-adaptive, yet the paper derives a GSPO-like smooth sequence gate
when steps are near on-policy and within-sequence log-ratio dispersion is low.
That connection is conditional, not identity. A heterogeneous sequence retains
useful near-on-policy tokens that hard sequence clipping might discard. The
paper reports SAPO use for Qwen3-VL; it is not evidence that every later Qwen
text model used the same objective.

## 11. ReMax

ReMax uses a deterministic greedy rollout as a baseline for the stochastic
sample, avoiding a critic
([Li et al., 2023](https://arxiv.org/abs/2310.10505)). Conceptually,

$$
\hat A=R(y_{\text{sample}})-R(y_{\text{greedy}}).
$$

It can be effective when an extra baseline rollout is cheaper than critic
training and reward is sequence-level. In stochastic agent environments, both
rollouts should share a controlled initial state/seed when interpreting the
difference; otherwise environment randomness contaminates the baseline.

## 12. Process-reward and value-free methods

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

Adding $\gamma\Phi(s')-\Phi(s)$ preserves optimal policies under standard
conditions. Learned “progress” scores generally do not have this guarantee.

## 13. Turn-level and hierarchical agent methods

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

## 14. Off-policy and asynchronous correction

Agent rollouts are slow, so collectors frequently lag behind the trainer.
Possible controls:

1. **bounded staleness:** reject trajectories older than $K$ versions;
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

## 15. Single-Rollout Asynchronous Optimization

**Single-Rollout Asynchronous Optimization (SAO)** addresses a structural
mismatch between long asynchronous agent episodes and prompt-group methods
([Hou et al., 2026](https://arxiv.org/abs/2607.07508)). It combines:

1. one rollout per prompt, consumed as soon as it finishes;
2. a learned critic because group size one has no group-relative baseline;
3. two critic updates per actor update in the paper experiments;
4. frozen critic-attention parameters with trainable Mixture-of-Experts
   projections;
5. skip-observation token-level Generalized Advantage Estimation; and
6. **Direct Double-Sided Importance Sampling (DIS)** from stored rollout
   log-probabilities to current-policy log-probabilities.

DIS keeps a token only when

$$
1-\epsilon_l<
\exp(\log\pi_\theta-\log\pi_{\text{rollout}})
<1+\epsilon_h.
$$

Outside the interval its policy contribution is zero. Unlike PPO clipping,
this masks both tails for either advantage sign. SAO is attractive when:

- trajectory lengths have a severe long tail;
- each environment state naturally yields one attempt;
- the learner and rollout service run concurrently;
- a critic can learn state-dependent baselines; and
- exact behavior tokens and log-probabilities are preserved.

Its costs are a second large model, critic cold start, additional value updates,
off-policy bias, and discarded tail tokens. The authors state that SAO was
deployed in GLM-5.2's agentic-RL pipeline, but publish controlled hyperparameters
only for Qwen3-30B-A3B. See the
[full derivation](derivations-and-code.md#20-single-rollout-asynchronous-optimization-derived)
and [GLM evidence analysis](case-studies/glm.md#113-sao-the-missing-algorithmic-link-du).

## 16. Offline preference objectives

Offline preference optimization is useful for bootstrap and alignment but is not
an online agent algorithm by itself.

### DPO

For preferred $y_w$, rejected $y_l$, reference $\pi_{\text{ref}}$,

$$
L_{\text{DPO}}=-\mathbb E\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right).
$$

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

## 17. Model-based, search, and self-play extensions

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

## 18. Algorithm selection matrix

| Situation | Useful starting point | Why | Main warning |
|---|---|---|---|
| High-quality demonstrations, weak initial policy | SFT | stable support acquisition | covariate shift |
| Scalar verified outcome, cheap multiple samples | RLOO/GRPO | no critic, within-task baseline | coarse credit and group cost |
| Meaningful intermediate states/rewards | PPO + turn critic/GAE | temporal credit and sample reuse | critic cost/error |
| Fully on-policy simple baseline | REINFORCE + valid baseline | transparent estimator | high variance |
| Concern about GRPO normalization bias | Dr. GRPO-style ablation | exposes denominator effects | retune scale fairly |
| Need open reasoning-RL recipe | DAPO-style recipe | exploration/dynamic sampling details | selection and length effects |
| Sequence reward and unstable token ratios | GSPO | aligns clipping and optimization with response reward | one outlier can suppress a full sequence |
| Need smooth token-adaptive off-policy attenuation | SAPO | continuous trust gate; keeps useful tokens | temperatures and ratio dispersion require measurement |
| Long stale asynchronous rollouts | bounded lag + off-policy correction | improves utilization | bias/variance and version integrity |
| Long-tail agent rollouts, one sample per changing state | SAO-style actor–critic | no prompt-group barrier; state baseline | critic cost and DIS bias |
| Sparse long-horizon reward | value/process/branching/hierarchy | denser credit | learned proxy exploitation |
| Subjective or safety preference | reward model + PPO or DPO | human/AI preference signal | judge/reward overoptimization |
| Expensive unsafe environment | SFT/offline data + sandboxed conservative RL | limits exploration risk | weak coverage of novel states |

Start with the simplest estimator that can represent the credit structure, then
add complexity only after a controlled diagnostic demonstrates the need.

## 19. Fair algorithm comparison

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

## 20. Ablation order

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
8. Zhenyu Hou, Yujiang Li, Jie Tang, and Yuxiao Dong,
   [“Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning”](https://arxiv.org/abs/2607.07508),
   2026.
9. Chujie Zheng et al.,
   [“Group Sequence Policy Optimization”](https://arxiv.org/abs/2507.18071),
   2025.
10. Chang Gao et al.,
    [“Soft Adaptive Policy Optimization”](https://arxiv.org/abs/2511.20347),
    2025.
