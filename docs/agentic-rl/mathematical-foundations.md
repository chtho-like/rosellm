# Mathematical Foundations

This chapter derives the objects that an Agentic RL implementation manipulates.
The goal is to make every tensor in a training batch traceable to a probability,
estimator, or explicit approximation.

## 1. Notation and the two nested policies

An LLM agent has a semantic policy over environment actions and an
autoregressive policy over tokens.

- \(s_t\): latent environment state at decision step \(t\)
- \(o_t\): observation emitted from \(s_t\)
- \(h_t\): policy-visible information state built from observations, past
  actions, and memory
- \(a_t\): semantic action at decision step \(t\)
- \(x_{t,j}\): generated token \(j\) within action \(a_t\)
- \(L_t\): number of policy-generated tokens in \(a_t\)
- \(r_t\): reward associated with a transition or action
- \(T\): number of environment decisions before termination
- \(\theta\): trainable policy parameters

If the environment consumes a generated token sequence directly, then

\[
a_t=(x_{t,1},\ldots,x_{t,L_t})
\]

and

\[
\log \pi_\theta(a_t\mid h_t)
=\sum_{j=1}^{L_t}
\log\pi_\theta(x_{t,j}\mid h_t,x_{t,<j}).
\]

If a deterministic parser \(g\) converts text to a structured action,
\(a_t=g(x_{t,1:L_t})\), several token sequences can map to the same action.
The environment-level probability is then

\[
\Pr_\theta(a_t\mid h_t)
=\sum_{x:g(x)=a_t}\pi_\theta(x\mid h_t).
\]

Production trainers almost never compute this sum. They optimize the sampled
token sequence. This is a valid likelihood-ratio estimator for the behavior
actually sampled, but it means formatting tokens can receive credit even when
the environment treats several formats as equivalent.

## 2. From next-token likelihood to a policy

Pretraining models a token sequence \(x_{1:n}\) as

\[
p_\theta(x_{1:n})=\prod_{i=1}^{n}p_\theta(x_i\mid x_{<i}).
\]

Maximum likelihood minimizes

\[
\mathcal{L}_{\text{NLL}}(\theta)
=-\sum_i\log p_\theta(x_i\mid x_{<i}).
\]

During agent execution, the same conditional distribution becomes a stochastic
policy. The difference is causal data generation:

- in teacher forcing, the next prefix comes from a fixed dataset;
- in online RL, sampled tokens determine the semantic action, which changes the
  next observation and therefore the future training distribution.

Cross-entropy and policy gradient can operate on the same logits, but they
estimate gradients of different objectives under different data distributions.

## 3. MDPs, POMDPs, histories, and belief states

An MDP satisfies

\[
p(s_{t+1},r_t\mid s_0,a_0,\ldots,s_t,a_t)
=p(s_{t+1},r_t\mid s_t,a_t).
\]

An agent rarely observes \(s_t\) completely. In a POMDP it sees
\(o_t\sim O(\cdot\mid s_t)\). A theoretically sufficient information state is
the belief distribution

\[
b_t(s)=p(s_t=s\mid o_{0:t},a_{0:t-1}).
\]

The Bayesian update is proportional to

\[
b_{t+1}(s')\propto
O(o_{t+1}\mid s')
\sum_s P(s'\mid s,a_t)b_t(s).
\]

LLM agents generally do not maintain this distribution explicitly. They use a
serialized history, summary, retrieval result, recurrent state, or learned
memory \(h_t=f(o_{0:t},a_{0:t-1})\). The policy is therefore
\(\pi_\theta(a_t\mid h_t)\), and the quality of \(f\) is part of the agent.

This matters experimentally. If training supplies hidden state or evaluator
annotations that inference cannot observe, the policy is trained on a different
information structure from the deployed agent.

## 4. Return, value, action value, and advantage

For discount \(\gamma\in[0,1]\), define reward-to-go

\[
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
\]

Under policy \(\pi\),

\[
V^\pi(h_t)=\mathbb{E}_\pi[G_t\mid h_t],
\]

\[
Q^\pi(h_t,a_t)=\mathbb{E}_\pi[G_t\mid h_t,a_t],
\]

and

\[
A^\pi(h_t,a_t)=Q^\pi(h_t,a_t)-V^\pi(h_t).
\]

Using history \(h_t\) rather than latent state \(s_t\) is deliberate: the critic
must be clear about whether it is allowed privileged environment state. A
centralized training critic may use extra state, but that changes the estimator
and must not leak into the actor at inference.

For an MDP, the Bellman expectation equations are

\[
V^\pi(s)=\mathbb{E}_{a\sim\pi,s'\sim P}
[R(s,a,s')+\gamma V^\pi(s')],
\]

\[
Q^\pi(s,a)=\mathbb{E}_{s'\sim P}
[R(s,a,s')+\gamma\mathbb{E}_{a'\sim\pi}Q^\pi(s',a')].
\]

These equations explain why a useful critic can propagate sparse terminal
success backward. They do not guarantee that a learned critic will generalize
under a changing language policy.

## 5. Trajectory probability

For a fully observed MDP with initial-state distribution \(\rho_0\),

\[
p_\theta(\tau)=
\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid h_t)
P(s_{t+1}\mid s_t,a_t).
\]

Only the policy term depends on \(\theta\) when the environment dynamics are
fixed with respect to the policy parameters. Therefore

\[
\nabla_\theta\log p_\theta(\tau)
=\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid h_t).
\]

This is the crucial reason model-free policy gradients do not need a
differentiable browser, compiler, simulator, user, or robot.

## 6. Deriving REINFORCE

Let \(R(\tau)\) denote total return and

\[
J(\theta)=\mathbb{E}_{\tau\sim p_\theta}[R(\tau)]
=\sum_\tau p_\theta(\tau)R(\tau).
\]

Differentiate:

\[
\begin{aligned}
\nabla_\theta J
&=\sum_\tau \nabla_\theta p_\theta(\tau)R(\tau)\\
&=\sum_\tau p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau)R(\tau)\\
&=\mathbb{E}_{\tau\sim p_\theta}
\left[R(\tau)\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid h_t)\right].
\end{aligned}
\]

Rewards before action \(a_t\) cannot be caused by that action. Replacing total
return with reward-to-go preserves expectation and reduces variance:

\[
\nabla_\theta J
=\mathbb{E}\left[
\sum_t G_t\nabla_\theta\log\pi_\theta(a_t\mid h_t)
\right].
\]

Expanding the semantic action into generated tokens gives

\[
\nabla_\theta J
=\mathbb{E}\left[
\sum_t\sum_{j=1}^{L_t}
G_t\nabla_\theta
\log\pi_\theta(x_{t,j}\mid h_t,x_{t,<j})
\right].
\]

A terminal outcome therefore supplies the same Monte Carlo return to all policy
tokens in a turn unless a finer estimator or reward decomposition is added.

## 7. Why a baseline does not bias the gradient

Subtract any baseline \(b(h_t)\) that does not depend on the sampled action:

\[
\mathbb{E}_{a\sim\pi_\theta}
[b(h)\nabla_\theta\log\pi_\theta(a\mid h)]
=b(h)\sum_a\pi_\theta(a\mid h)
\nabla_\theta\log\pi_\theta(a\mid h).
\]

Since \(\pi\nabla\log\pi=\nabla\pi\),

\[
b(h)\nabla_\theta\sum_a\pi_\theta(a\mid h)
=b(h)\nabla_\theta 1=0.
\]

Thus

\[
(G_t-b(h_t))\nabla\log\pi_\theta(a_t\mid h_t)
\]

has the same expectation and can have much lower variance. A learned critic
approximates \(V^\pi(h_t)\). A leave-one-out or group mean uses other samples
for the same prompt/task. Baseline validity depends on the estimator details:
including the current action's reward in a sample mean, normalizing by a
sample-dependent standard deviation, or coupling trajectories can introduce
finite-sample bias even when the intuitive “center the rewards” story sounds
correct.

## 8. Actor–critic and temporal-difference residuals

For a learned value \(V_\phi\), the one-step TD residual is

\[
\delta_t=r_t+\gamma V_\phi(h_{t+1})-V_\phi(h_t).
\]

If \(V_\phi=V^\pi\), \(\delta_t\) is an unbiased sample of the one-step
advantage conditioned on \((h_t,a_t)\). In practice the critic is approximate
and trained on a non-stationary distribution.

Generalized Advantage Estimation is

\[
\hat A_t^{\text{GAE}(\gamma,\lambda)}
=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.
\]

- \(\lambda=0\): one-step TD, lower variance and more bootstrapping bias.
- \(\lambda=1\): Monte Carlo return minus value, less bootstrapping bias and
  higher variance for finite episodes.

Timeout handling is mathematical, not bookkeeping. A true terminal state has
no future value. A rollout truncated only because a collector reached a length
budget may require bootstrapping from \(V(h_T)\). Marking both as `done=True`
changes targets.

## 9. Importance sampling and policy versions

Suppose data came from behavior policy \(\mu\) but the target is \(\pi_\theta\).
For a single action,

\[
\mathbb{E}_{a\sim\mu}
\left[\frac{\pi_\theta(a\mid h)}{\mu(a\mid h)}f(a)\right]
=\mathbb{E}_{a\sim\pi_\theta}[f(a)]
\]

when support is sufficient. The ratio is

\[
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid h_t)}{mu(a_t\mid h_t)}.
\]

For a token sequence, a trajectory/action ratio is a product of token ratios:

\[
\rho_t
=\prod_j
\frac{\pi_\theta(x_{t,j}\mid h_t,x_{t,<j})}
{\mu(x_{t,j}\mid h_t,x_{t,<j})}.
\]

Equivalently, log-ratios add. Products over long sequences can have enormous
variance. Practical LLM algorithms therefore use token-level clipping,
sequence-level clipping, truncated ratios, small policy lag, or objectives that
accept some bias.

An old-policy log-probability stored with each generated token is meaningful
only if it was computed under the exact token prefix, chat template, model
weights, sampling mask, and vocabulary used for generation.

## 10. PPO's clipped surrogate

Let behavior policy be \(\pi_{\theta_{\text{old}}}\) and define

\[
r_t(\theta)=
\frac{\pi_\theta(a_t\mid h_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid h_t)}.
\]

PPO maximizes

\[
L^{\text{CLIP}}(\theta)=
\mathbb{E}_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
\]

The sign of \(\hat A_t\) matters:

- for positive advantage, an increase above \(1+\epsilon\) stops improving the
  surrogate;
- for negative advantage, a decrease below \(1-\epsilon\) stops improving it.

Clipping is not a hard guarantee on KL divergence. A trainer should still log
approximate KL, clip fraction, ratio distribution, entropy, and gradient norms.

LLM PPO often adds a value loss and entropy term:

\[
L=L_{\text{policy}}-c_vL_{\text{value}}+c_H\mathcal{H}(\pi_\theta),
\]

with sign convention depending on whether code minimizes loss or maximizes an
objective. A single sign error can make the “entropy bonus” reduce entropy.

## 11. KL regularization and the reference policy

To preserve capabilities and limit reward overoptimization, optimize

\[
J_{\text{KL}}(\theta)=
\mathbb{E}_{x,y\sim\pi_\theta}
\left[r(x,y)-\beta
\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
\right].
\]

The sampled log-ratio

\[
k(x,y)=\log\pi_\theta(y\mid x)-\log\pi_{\text{ref}}(y\mid x)
\]

is an unbiased Monte Carlo estimate of forward
\(D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\text{ref}})\) when \(y\) is sampled from
\(\pi_\theta\). If samples come from an older behavior policy, this
interpretation needs correction.

For tokens,

\[
k(x,y)=\sum_j
\left[
\log\pi_\theta(y_j\mid x,y_{<j})-
\log\pi_{\text{ref}}(y_j\mid x,y_{<j})
\right].
\]

Applying a per-token KL penalty makes longer responses accumulate more penalty.
Dividing by length changes the optimized objective. Neither choice is neutral;
state the normalization explicitly.

The KL-regularized optimum for a fixed context and reward has form

\[
\pi^*(y\mid x)=
\frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right),
\]

which is the relationship used to derive DPO-style preference objectives. DPO
is valuable background, but a fixed pairwise dataset does not directly solve
multi-turn on-policy credit assignment.

## 12. Group and leave-one-out baselines

For a task \(q\), sample \(G\) trajectories with returns \(R_1,\ldots,R_G\).
A leave-one-out baseline is

\[
b_i^{\text{LOO}}=\frac{1}{G-1}\sum_{k\ne i}R_k,
\qquad
\hat A_i=R_i-b_i^{\text{LOO}}.
\]

Because \(b_i^{\text{LOO}}\) excludes action/trajectory \(i\), it is independent
of that sample conditional on the task when rollouts are independent. RLOO uses
this structure to avoid a learned critic.

A commonly shown GRPO normalization is

\[
\hat A_i=
\frac{R_i-\overline R}{\operatorname{std}(R_1,\ldots,R_G)+\varepsilon}.
\]

This makes updates invariant to affine scaling within a group and emphasizes
relative performance. It also creates important edge cases:

- if all rewards are equal, the group supplies no learning signal;
- small groups produce noisy mean and standard deviation estimates;
- prompt difficulty affects reward variance and therefore gradient scale;
- standard-deviation normalization can overweight low-variance groups;
- including \(R_i\) in \(\overline R\) couples the sample with its baseline;
- outcome advantage copied across all tokens interacts with response-length
  normalization.

These are estimator properties, not minor implementation details. DeepSeekMath
introduced GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300));
Dr. GRPO analyzes several normalization biases
([Liu et al., 2025](https://arxiv.org/abs/2503.20783)).

## 13. Loss normalization and length bias

Suppose token loss for response \(i\) is

\[
\ell_i=-\sum_{j=1}^{L_i}m_{i,j}\,s_{i,j},
\]

where \(m\) marks policy-generated trainable tokens and \(s\) is a surrogate
term. At least three batch reductions are common:

### Per-token normalization

\[
L_{\text{token}}=
\frac{\sum_i\sum_j m_{i,j}s_{i,j}}
{\sum_i\sum_j m_{i,j}}.
\]

Every token has equal weight; long trajectories influence more terms.

### Per-sequence normalization

\[
L_{\text{seq}}=
\frac{1}{B}\sum_i
\frac{\sum_jm_{i,j}s_{i,j}}{\sum_jm_{i,j}}.
\]

Every sequence has equal total weight; each token in a long response receives
less weight.

### Per-turn or per-trajectory normalization

Multi-turn systems can average within turns, then trajectories, then tasks. This
changes the relative influence of long episodes, verbose actions, and tasks with
many decisions.

The denominator is part of the objective. When distributed workers have
different token counts, averaging already-normalized local losses is generally
not equivalent to globally summing numerators and denominators.

## 14. Multi-turn credit assignment

For a terminal-only task reward \(R_T\), Monte Carlo credit uses

\[
\hat A_t=R_T-b(h_t)
\]

for every turn. It is unbiased with a valid baseline but high variance and does
not distinguish an essential early tool call from irrelevant later text.

Possible refinements include:

### Turn-level environment rewards

Use verified progress \(r_t\) and reward-to-go. Dense rewards reduce temporal
distance but can alter the optimum. Potential-based shaping

\[
F(s_t,s_{t+1})=\gamma\Phi(s_{t+1})-\Phi(s_t)
\]

preserves optimal policies under standard MDP conditions, whereas arbitrary
“looks like progress” rewards do not.

### Learned values

Estimate \(V(h_t)\) or \(Q(h_t,a_t)\). This can use all downstream reward but is
susceptible to approximation error and distribution shift.

### Process rewards

Score turns, subgoals, or reasoning spans with rules, humans, or a learned model.
The supervision becomes denser, but the policy can optimize evaluator artifacts.

### Counterfactual and branching estimates

From a saved state before action \(a_t\), sample alternate continuations to
estimate its marginal effect. This is closer to causal credit but multiplies
environment and rollout cost and requires reproducible state cloning.

### Hierarchical credit

Treat a plan/subgoal as an option with its own termination and return. A manager
chooses subgoals; a worker executes primitive actions. The hierarchy can shorten
credit paths, provided the option boundaries are meaningful and trainable.

## 15. Entropy and exploration

For a categorical token distribution,

\[
\mathcal H(\pi(\cdot\mid h))
=-\sum_x\pi(x\mid h)\log\pi(x\mid h).
\]

Entropy is not identical to useful behavioral diversity:

- high token entropy may vary punctuation without changing the semantic action;
- low token entropy may still support diverse long-horizon plans through early
  branch choices;
- sampling temperature alters rollout distribution but not by itself the
  objective's entropy regularization;
- tool schemas and constrained decoding reduce syntactic entropy while possibly
  improving semantic exploration.

Log entropy per token, semantic action diversity, unique successful strategies,
and group reward variance separately.

## 16. Multi-objective and constrained RL

Agent utility normally includes success, safety, cost, latency, and user
preference. A scalar reward might be

\[
r_t=w_s r_t^{\text{success}}
-w_c c_t^{\text{cost}}
-w_v c_t^{\text{violation}}.
\]

The weights encode policy decisions and can hide unacceptable tradeoffs. For a
hard expected constraint

\[
\mathbb E_\pi[C(\tau)]\le d,
\]

form a Lagrangian

\[
\mathcal L(\theta,\lambda)
=J_R(\theta)-\lambda(J_C(\theta)-d),
\qquad \lambda\ge0.
\]

Alternating policy ascent and multiplier ascent/descent (depending on sign
convention) can enforce an expectation constraint, but it does not guarantee
zero catastrophic violations. Authorization checks, sandboxing, and action
filters remain necessary outside the learned policy.

Lexicographic objectives—first satisfy safety, then maximize task utility—can be
more appropriate than allowing enough task reward to compensate for a severe
violation.

## 17. Estimator diagnostics

Track quantities that reveal the estimator, not just final reward:

- mean, standard deviation, and histogram of raw reward components;
- return and advantage by task, turn, length, and success/failure;
- group reward variance and fraction of all-equal groups;
- policy/behavior token log-ratio distribution;
- approximate KL to old and reference policies;
- clip fraction for positive and negative advantages;
- entropy by position and semantic decision;
- value loss, explained variance, and calibration by horizon;
- gradient norm before/after clipping and by parameter group;
- effective sample size for importance weights,
  \((\sum_i w_i)^2/\sum_iw_i^2\);
- response/episode length and correlation with reward/advantage;
- policy lag measured in optimizer steps and divergence, not wall time alone.

A rising reward with collapsing group variance may mean the curriculum became
too easy. A stable loss with exploding ratios may mean clipping hides unusable
off-policy data. A successful average with a heavy tail of safety violations is
not a successful policy.

## 18. Tensor contract for an implementation

A minimal padded batch can use shapes

| Tensor | Shape | Meaning |
|---|---:|---|
| `input_ids` | `[B, L]` | prompt, observations, and sampled policy tokens |
| `attention_mask` | `[B, L]` | valid non-padding positions |
| `action_mask` | `[B, L-1]` | next-token positions generated by the policy |
| `old_logprobs` | `[B, L-1]` | behavior log-probability of sampled next token |
| `ref_logprobs` | `[B, L-1]` | reference-policy log-probability, if used |
| `turn_ids` | `[B, L-1]` | environment decision associated with each token |
| `rewards` | `[B, T]` or events | reward components at decision boundaries |
| `advantages` | `[B, T]` or `[B, L-1]` | credit estimate aligned to turns/tokens |
| `policy_version` | `[B]` or per segment | checkpoint that sampled the action |

For causal logits `logits[:, :-1]`, the target is `input_ids[:, 1:]`. The action
mask must align with the target token, not the input position. Tool observations,
system prompts, user messages, padding, and tokens generated by other policies
must have zero policy-gradient mask.

For packed variable-length trajectories, replace padding with segment IDs,
cu-seqlens, or block-diagonal attention metadata. The same logical contract must
remain testable.

## 19. Invariants worth testing

1. A constant reward across all independent actions yields zero expected
   baseline-centered policy gradient.
2. Adding a constant to every reward in an exactly centered group does not
   change normalized advantages.
3. Masked observation tokens contribute exactly zero policy loss and gradient.
4. Re-tokenizing decoded rollout text is never used to reconstruct behavior
   log-probabilities.
5. With \(\theta=\theta_{\text{old}}\), every valid importance ratio is one
   within numerical tolerance.
6. Padding or repacking the same examples leaves the globally normalized loss
   unchanged.
7. A true terminal state does not bootstrap; a collector truncation follows the
   configured bootstrap rule.
8. Distributed loss reduction matches a single-process concatenated batch.
9. Old log-probabilities are immutable and tagged with the behavior policy.
10. Reward, action, and observation events reconstruct the original environment
    trajectory in order.

## 20. Common mathematical category errors

- Calling a sample baseline a “value function” when it does not condition on
  state/history.
- Calling clipped PPO exactly on-policy after multiple epochs over the same
  rollout without acknowledging increasing mismatch.
- Treating a KL estimator from behavior-policy samples as the target policy's
  exact forward KL.
- Dividing token loss by sequence length and claiming the objective is unchanged.
- Copying terminal reward to every token and calling the resulting signal
  token-level process supervision.
- Using privileged evaluator state in a critic without documenting centralized
  training.
- Interpreting a standard-deviation-normalized group advantage as an absolute
  measure of task utility.
- Assuming lower estimator variance implies lower bias or better final policy.

## References

1. Ronald J. Williams,
   [“Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning”](https://doi.org/10.1007/BF00992696),
   1992.
2. Richard S. Sutton and Andrew G. Barto,
   [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html),
   2nd ed., 2018.
3. John Schulman et al.,
   [“High-Dimensional Continuous Control Using Generalized Advantage Estimation”](https://arxiv.org/abs/1506.02438),
   2015.
4. John Schulman et al.,
   [“Proximal Policy Optimization Algorithms”](https://arxiv.org/abs/1707.06347),
   2017.
5. Zhihong Shao et al.,
   [“DeepSeekMath”](https://arxiv.org/abs/2402.03300), 2024.
6. Zichen Liu et al.,
   [“Understanding R1-Zero-Like Training: A Critical Perspective”](https://arxiv.org/abs/2503.20783),
   2025.
