# Derivations and Code: From Trajectory Probability to Agent Updates

This chapter does not assume the policy-gradient formula. It derives each step,
maps the result to autoregressive tokens, and then shows the smallest correct
code pattern. The purpose is to make it impossible to hide a changed objective
behind words such as “normalize,” “clip,” or “distill.”

The policy is a **Large Language Model (LLM)** trained with **Reinforcement
Learning (RL)** from interactive trajectories.

## 1. Start with the Partially Observable Markov Decision Process distribution

A **Partially Observable Markov Decision Process (POMDP)** models sequential
decisions when the environment has a latent state but exposes only observations.
An LLM agent usually sees a transcript, tool result, or screenshot rather than
the complete browser, operating system, or user state.

Let an episode contain latent states \(s_t\), observations \(o_t\), policy-visible
histories \(h_t\), semantic actions \(a_t\), rewards \(r_t\), and transitions
\(P\). For simplicity, assume the observation is emitted from the state and the
environment does not depend differentiably on policy parameters.

\[
p_\theta(\tau)=
\rho_0(s_0)O(o_0\mid s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid h_t)
P(s_{t+1}\mid s_t,a_t)
O(o_{t+1}\mid s_{t+1}).
\]

Take logs:

\[
\begin{aligned}
\log p_\theta(\tau)
&=\log\rho_0(s_0)+\log O(o_0\mid s_0)\\
&\quad+\sum_t\left[
\log\pi_\theta(a_t\mid h_t)
+\log P(s_{t+1}\mid s_t,a_t)
+\log O(o_{t+1}\mid s_{t+1})
\right].
\end{aligned}
\]

Only the policy term contains \(\theta\), so

\[
\nabla_\theta\log p_\theta(\tau)
=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid h_t).
\]

No gradient through the browser, compiler, human, game, or database is required.
Their behavior changes which trajectories are sampled, but the score-function
identity differentiates the probability of selecting the actions.

## 2. Derive the likelihood-ratio gradient

Let total discounted return be

\[
R(\tau)=\sum_{t=0}^{T-1}\gamma^tr_t.
\]

The objective is

\[
J(\theta)=\int p_\theta(\tau)R(\tau)d\tau.
\]

Assuming differentiation and integration can be exchanged:

\[
\begin{aligned}
\nabla_\theta J
&=\int \nabla_\theta p_\theta(\tau)R(\tau)d\tau\\
&=\int p_\theta(\tau)
\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}
R(\tau)d\tau\\
&=\mathbb E_{\tau\sim p_\theta}
[R(\tau)\nabla_\theta\log p_\theta(\tau)]\\
&=\mathbb E\left[
R(\tau)\sum_t\nabla_\theta\log\pi_\theta(a_t\mid h_t)
\right].
\end{aligned}
\]

The identity \(\nabla p=p\nabla\log p\) is the “log-derivative trick.” It is
mathematics, not a heuristic.

Monte Carlo code for one trajectory is conceptually:

```python
loss = 0.0
for logprob_of_action in trajectory.logprobs:
    loss = loss - total_return * logprob_of_action
loss.backward()
```

The code minimizes negative expected return. It has high variance because every
action receives every reward.

## 3. Why reward-to-go is valid

Split return around action \(a_t\):

\[
R(\tau)=R_{<t}+\gamma^tG_t,
\]

where \(R_{<t}\) contains rewards strictly before \(a_t\). Conditioned on
history \(h_t\), those past rewards do not depend on the sampled \(a_t\):

\[
\begin{aligned}
\mathbb E[R_{<t}\nabla\log\pi(a_t\mid h_t)\mid h_t]
&=R_{<t}\sum_a\pi(a\mid h_t)\nabla\log\pi(a\mid h_t)\\
&=R_{<t}\nabla\sum_a\pi(a\mid h_t)\\
&=0.
\end{aligned}
\]

Therefore past rewards can be removed:

\[
\nabla J=
\mathbb E\left[\sum_t\gamma^tG_t
\nabla\log\pi(a_t\mid h_t)\right].
\]

Many implementations absorb \(\gamma^t\) into the state-visitation
distribution or return convention. State the convention before comparing code.

## 4. Expand one semantic action into LLM tokens

Suppose turn \(t\) action is token sequence

\[
a_t=(x_{t,1},\ldots,x_{t,L_t}).
\]

Autoregressive factorization gives

\[
\log\pi_\theta(a_t\mid h_t)=
\sum_{j=1}^{L_t}
\log\pi_\theta(x_{t,j}\mid h_t,x_{t,<j}).
\]

Substitute into the gradient:

\[
\nabla J=
\mathbb E\left[
\sum_t\sum_j
\hat A_t
\nabla\log\pi_\theta(x_{t,j}\mid h_t,x_{t,<j})
\right].
\]

The same turn advantage is often broadcast to every generated token. The loss
still distinguishes tokens because their log-probability gradients differ. It
does **not** tell us that every token had equal causal effect.

Code alignment:

```python
logits = model(input_ids).logits              # [B, L, V]
targets = input_ids[:, 1:]                    # [B, L-1]
logits = logits[:, :-1]                       # predicts targets
logprobs = logits.log_softmax(-1)
selected = logprobs.gather(-1, targets[..., None]).squeeze(-1)
token_advantages = turn_advantages.gather(1, turn_ids)
loss = -(selected * token_advantages * action_mask).sum() / action_mask.sum()
```

Tool observations must have `action_mask=0`; otherwise the gradient treats
environment text as an action chosen by the policy.

## 5. Baselines: prove zero expected gradient

For any \(b(h_t)\) independent of sampled action:

\[
\mathbb E[b(h_t)\nabla\log\pi(a_t\mid h_t)\mid h_t]=0.
\]

Therefore replace return with

\[
\hat A_t=G_t-b(h_t).
\]

The variance-minimizing scalar baseline under a simple conditional formulation
is related to a gradient-norm-weighted return expectation, while \(V^\pi(h_t)\)
is the familiar practical approximation. “Subtract the mean” is not universally
optimal; it is a control variate choice.

### Leave-one-out baseline

For \(G\) independent rollouts of one task:

\[
b_i=\frac{1}{G-1}\sum_{k\ne i}R_k,
\qquad A_i=R_i-b_i.
\]

Conditional on the task, \(b_i\) excludes rollout \(i\). For \(G=3\) and
rewards `[0, 1, 2]`:

```text
sample 0: 0 - (1+2)/2 = -1.5
sample 1: 1 - (0+2)/2 =  0.0
sample 2: 2 - (0+1)/2 = +1.5
```

```python
other_mean = (rewards.sum(-1, keepdim=True) - rewards) / (group_size - 1)
advantages = rewards - other_mean
```

## 6. Derive temporal-difference residual and GAE

If \(V=V^\pi\), then

\[
\begin{aligned}
\mathbb E[r_t+\gamma V(h_{t+1})-V(h_t)\mid h_t,a_t]
&=Q^\pi(h_t,a_t)-V^\pi(h_t)\\
&=A^\pi(h_t,a_t).
\end{aligned}
\]

Define TD residual

\[
\delta_t=r_t+\gamma V(h_{t+1})-V(h_t).
\]

An \(n\)-step advantage is

\[
\hat A_t^{(n)}=
\sum_{l=0}^{n-1}\gamma^lr_{t+l}
+\gamma^nV(h_{t+n})-V(h_t).
\]

It can be written as a sum of TD residuals:

\[
\hat A_t^{(n)}=
\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}.
\]

GAE forms an exponentially weighted mixture of \(n\)-step estimators:

\[
\hat A_t^{\text{GAE}}=(1-\lambda)
\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)}.
\]

Substitute the TD-sum and exchange summations:

\[
\hat A_t^{\text{GAE}}=
\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}.
\]

Finite episode code walks backward:

```python
next_advantage = 0.0
next_value = bootstrap_value
for t in reversed(range(T)):
    nonterminal = 1.0 - terminated[t]
    delta = reward[t] + gamma * nonterminal * next_value - value[t]
    advantage[t] = delta + gamma * lam * nonterminal * next_advantage
    next_value = value[t]
    next_advantage = advantage[t]
```

True termination sets `nonterminal=0`. A collector truncation generally supplies
a bootstrap value instead. Padding is skipped, not declared terminal.

## 7. Importance sampling: where old log-probabilities enter

Samples come from behavior \(\mu\), but we want an expectation under target
\(\pi\):

\[
\mathbb E_{a\sim\pi}[f(a)]
=\sum_a\mu(a)\frac{\pi(a)}{\mu(a)}f(a)
=\mathbb E_{a\sim\mu}[\rho(a)f(a)].
\]

For one token:

\[
\rho_j=\exp[
\log\pi_\theta(x_j\mid x_{<j})-
\log\mu(x_j\mid x_{<j})].
\]

For a complete token sequence:

\[
\rho_{1:L}=\prod_j\rho_j
=\exp\left(\sum_j\log\rho_j\right).
\]

Long products have extreme variance. Token clipping, sequence geometric means,
truncated weights, and bounded policy lag change bias/variance differently.

```python
log_ratio = current_logprobs - old_logprobs
ratio = torch.exp(log_ratio)
```

If `old_logprobs` came from unmasked logits but sampling used top-p, the
denominator is wrong. If MoE routing differs between rollout and learner, the
numerator is not the probability of the same computation. DeepSeek V3.2's Keep
Routing/Keep Sampling Mask and GLM-5's token-in/token-out mismatch gates target
these concrete failures.

## 8. Proximal Policy Optimization clipping piece by piece

**Proximal Policy Optimization (PPO)** constrains how much a sampled action's
probability may profitably move during one batch of updates. “Proximal” means
the new policy is encouraged to remain near the rollout policy.

Define unclipped \(u(\rho,A)=\rho A\) and clipped
\(c(\rho,A)=\operatorname{clip}(\rho,1-\epsilon_l,1+\epsilon_h)A\). PPO uses

\[
L(\rho,A)=\min(u,c).
\]

### Positive advantage \(A>0\)

\[
L(\rho,A)=
\begin{cases}
\rho A,&\rho\le1+\epsilon_h,\\
(1+\epsilon_h)A,&\rho>1+\epsilon_h.
\end{cases}
\]

Decreasing probability remains penalized; excessive increase stops receiving
surrogate improvement.

### Negative advantage \(A<0\)

\[
L(\rho,A)=
\begin{cases}
(1-\epsilon_l)A,&\rho<1-\epsilon_l,\\
\rho A,&\rho\ge1-\epsilon_l.
\end{cases}
\]

Excessive probability decrease stops improving; increasing a bad action remains
penalized.

```python
unclipped = ratio * advantage
clipped = ratio.clamp(1 - eps_low, 1 + eps_high) * advantage
surrogate = torch.minimum(unclipped, clipped)
loss = -masked_mean(surrogate, action_mask)
```

PPO clipping does not impose a hard KL bound. Log ratio distributions, clip
fractions by advantage sign, and KL separately.

## 9. KL-regularized reward and its optimal policy

For fixed context \(x\), optimize over distributions \(\pi\):

\[
\max_\pi
\sum_y\pi(y\mid x)r(x,y)
-\beta\sum_y\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
\]

subject to \(\sum_y\pi(y\mid x)=1\). Add Lagrange multiplier \(\lambda\):

\[
\mathcal L=\sum_y\pi_y r_y
-\beta\sum_y\pi_y\log\frac{\pi_y}{q_y}
+\lambda(\sum_y\pi_y-1).
\]

Differentiate with respect to \(\pi_y\):

\[
r_y-\beta\left(\log\frac{\pi_y}{q_y}+1\right)+\lambda=0.
\]

Rearrange:

\[
\pi_y=q_y\exp(r_y/\beta)\exp((\lambda-\beta)/\beta).
\]

Normalization produces

\[
\boxed{
\pi^*(y\mid x)=
\frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}
}
\]

and therefore

\[
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}
{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x).
\]

This identity connects preference rewards and policy log-ratios.

## 10. Derive DPO from Bradley–Terry preferences

Assume probability that response \(y_w\) is preferred over \(y_l\):

\[
p(y_w\succ y_l\mid x)=
\sigma(r(x,y_w)-r(x,y_l)).
\]

Insert the KL-optimal implicit reward. The context-only \(\log Z(x)\) cancels:

\[
\begin{aligned}
r(x,y_w)-r(x,y_l)
&=\beta\log\frac{\pi_\theta(y_w\mid x)}
{\pi_{\text{ref}}(y_w\mid x)}\\
&\quad-\beta\log\frac{\pi_\theta(y_l\mid x)}
{\pi_{\text{ref}}(y_l\mid x)}.
\end{aligned}
\]

Negative log-likelihood yields

\[
L_{\text{DPO}}=-\mathbb E\log\sigma\left[
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right].
\]

Minimal code:

```python
chosen_reward = beta * (chosen_logp - ref_chosen_logp)
rejected_reward = beta * (rejected_logp - ref_rejected_logp)
loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()
```

DPO depends on a static comparison distribution. It does not by itself collect
new states induced by an interactive agent policy.

## 11. Group Relative Policy Optimization: what is and is not removed

**Group Relative Policy Optimization (GRPO)** replaces a learned value-model
baseline with comparisons among several responses sampled for the same prompt.

For one prompt/task with \(G\) rollouts:

\[
\bar R=\frac1G\sum_iR_i,
\qquad
s_R=\sqrt{\frac1G\sum_i(R_i-\bar R)^2},
\qquad
A_i=\frac{R_i-\bar R}{s_R+\epsilon}.
\]

GRPO removes a learned critic, not:

- multiple rollout generation;
- reward/verifier infrastructure;
- old/reference policies;
- clipping/KL decisions;
- length/token normalization;
- multi-turn credit decisions.

For binary reward and all failures/successes, \(s_R=0\) and the group has no
relative signal. Dynamic sampling resamples such groups, changing which prompts
enter the update and consuming unreported generation unless attempts are logged.

## 12. Why standard-deviation normalization changes prompt weights

Consider two prompts with centered reward deviations of equal absolute size but
different group spread:

```text
prompt A deviations: [-0.5, +0.5], std=0.5 -> advantages [-1, +1]
prompt B deviations: [-0.1, +0.1], std=0.1 -> advantages [-1, +1]
```

The raw evidence for B is five times smaller, but standardized magnitude is
identical. Conversely, a wide reward-scale task is downweighted. This can be
desirable scale invariance or unwanted prompt weighting; it is not algebraically
neutral.

When the current sample participates in \(\bar R\) and \(s_R\), its advantage is
coupled to its own reward. Dr. GRPO removes analyzed reward-std and per-response
length normalizations to target an unbiased constant-normalized objective
([Liu et al., 2025](https://arxiv.org/abs/2503.20783)).

## 13. Token versus sequence normalization

Let sequence \(i\) have \(L_i\) valid action tokens and common advantage \(A_i\).

Per-token batch mean:

\[
L_{\text{token}}=-
\frac{\sum_i\sum_{j=1}^{L_i}s_{i,j}}
{\sum_iL_i}.
\]

Per-sequence mean:

\[
L_{\text{seq}}=-
\frac1B\sum_i\frac1{L_i}
\sum_{j=1}^{L_i}s_{i,j}.
\]

If two sequences have identical per-token surrogate but lengths 10 and 100:

- token mean gives the long sequence ten times as many terms;
- sequence mean gives each sequence equal total weight.

Neither is universally correct. DAPO explicitly uses token-level aggregation;
Dr. GRPO analyzes per-response length normalization. Distributed code must sum
global numerator/denominator; averaging local normalized losses changes weights
when ranks contain different token counts.

## 14. On-policy distillation

Let student generate trajectory \(y\sim\pi_\theta\). A teacher supplies its full
distribution at the student-visited prefixes. Reverse-KL OPD minimizes

\[
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_E)=
\sum_v\pi_\theta(v\mid h)
\left[\log\pi_\theta(v\mid h)-\log\pi_E(v\mid h)\right].
\]

Exact code at one token position:

```python
student_logp = student_logits.log_softmax(-1)
teacher_logp = teacher_logits.log_softmax(-1)
student_p = student_logp.exp()
reverse_kl = (student_p * (student_logp - teacher_logp)).sum(-1)
loss = masked_mean(reverse_kl, action_mask)
```

This is not the sampled-token approximation

```python
student_selected_logp - teacher_selected_logp
```

which uses only the sampled action. Full vocabulary costs memory/bandwidth;
DeepSeek V4 describes caching teacher hidden states, loading one teacher head at
a time, and computing exact KL with a specialized kernel.

Why “on-policy”: student chooses the histories/prefixes. Why “distillation”:
teacher probabilities provide the target. It reduces offline exposure mismatch
but remains sensitive to reverse-KL mode seeking, teacher disagreement, teacher
selection, and cost.

## 15. Multi-teacher OPD

For teachers \(E_1,\ldots,E_M\):

\[
L(\theta)=\sum_{i=1}^{M}w_i(h)
D_{\mathrm{KL}}(\pi_\theta(\cdot\mid h)
\Vert\pi_{E_i}(\cdot\mid h)).
\]

Questions hidden by the compact formula:

- Are weights fixed, task-routed, confidence-weighted, or learned?
- What happens when teachers disagree?
- Is every teacher defined over the same tokenizer/vocabulary?
- Does student sample with deployment quantization/tools?
- Are teacher logits computed on every token or only policy spans?
- Does one teacher see privileged evidence?

The exact answers define the training pipeline.

## 16. Partial rollout with multiple behavior policies

Suppose turns \(0\ldots k\) were sampled by \(\mu_0\) and later turns by
\(\mu_1\). Trajectory probability is

\[
p(\tau)=\rho_0P
\left[\prod_{t=0}^{k}\mu_0(a_t\mid h_t)P_t\right]
\left[\prod_{t=k+1}^{T-1}\mu_1(a_t\mid h_t)P_t\right].
\]

There is no single behavior checkpoint for the whole episode. Correct records
attach policy version/log-probabilities to each segment.

Options:

1. train only new-policy segment; earlier turns are masked context;
2. use per-segment importance ratios;
3. apply a fully off-policy estimator;
4. discard mixed episodes.

Kimi k1.5 publicly describes continuation with only newly generated segment
trained. Restarting every incomplete trajectory would waste compute and select
against naturally long solutions.

## 17. Compaction-aware sub-traces and critic PPO

Long-running agents may compact old transcript into summaries and emit multiple
trainable sub-traces. Let episode \(i\) produce \(K_i\) segments of length
\(L_{i,k}\). If \(K_i\) varies, a fixed group-per-prompt advantage is awkward:

- what is a “group member”: episode, segment, or compacted state?
- episodes with many segments create more token losses;
- segment rewards share downstream return and are statistically dependent;
- summaries change the information state.

GLM-5.2 discloses moving such traces to critic PPO, individual-rollout learning,
token-level advantages, retention of all sub-traces, and token-level loss. A
critic can condition on each compacted information state:

\[
A_{i,k,j}\approx Q(h_{i,k,j},x_{i,k,j})-V(h_{i,k,j}).
\]

The disclosure does not specify critic architecture, GAE, or coefficients; the
mathematical motivation should not be mistaken for a reproducible recipe.

## 18. Single-Rollout Asynchronous Optimization, derived

Hou et al.'s
[Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)
introduces **Single-Rollout Asynchronous Optimization (SAO)**. It is especially
important because the authors explicitly report deploying it in the GLM-5.2
agentic-RL pipeline. The controlled paper experiments, however, use
Qwen3-30B-A3B; this derivation describes the published method rather than an
undisclosed GLM-5.2 configuration.

### 18.1 Why asynchronous group sampling increases lag

Let a prompt have \(G\) rollouts with durations \(T_1,\ldots,T_G\). A grouped
method cannot construct its group-relative advantage until

\[
T_{\mathrm{ready}}=\max_{1\le i\le G}T_i.
\]

If rollout \(i\) finishes at \(T_i\), its avoidable wait is
\(T_{\mathrm{ready}}-T_i\). The total within-group waiting work is

\[
W=\sum_{i=1}^{G}(T_{\mathrm{ready}}-T_i).
\]

This is not merely wasted wall time. If the learner updates while the group
waits, early trajectories become more off-policy. SAO uses one rollout per
prompt and admits it when ready, removing \(W\) and the group barrier. It also
removes the same-prompt group baseline, so a critic must control variance.

### 18.2 Direct Double-Sided Importance Sampling

**Direct Double-Sided Importance Sampling (DIS)** compares the current policy
directly with the rollout engine's stored probability:

\[
r_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
       {\pi_{\mathrm{rollout}}(a_t\mid s_t)}
=\exp[\ell_t(\theta)-\ell_t^{\mathrm{rollout}}].
\]

There is no intermediate “old policy” checkpoint. This saves inference and
checkpoint history, but it is valid only if rollout logs describe the actual
sampling distribution. Top-\(p\) masking, temperature, vocabulary mapping,
Mixture-of-Experts routing, and numerical backend differences must still be
handled.

SAO's calibration is

\[
f(r_t;\epsilon_l,\epsilon_h)=
\begin{cases}
r_t,&1-\epsilon_l<r_t<1+\epsilon_h,\\
0,&\text{otherwise}.
\end{cases}
\]

Compare the gradients qualitatively:

| Condition | PPO with \(A>0\) | PPO with \(A<0\) | SAO DIS |
|---|---:|---:|---:|
| \(r<1-\epsilon_l\) | active | saturated | zero |
| inside interval | active | active | active |
| \(r>1+\epsilon_h\) | saturated | active | zero |

Therefore “double-sided clipping” is potentially misleading: it **rejects**
both tails instead of replacing the ratio by a boundary value.

The paper prints the maximization objective

\[
L(\theta)=\widehat{\mathbb E}_t\left[
f(r_t(\theta))\widehat A_t
\log\pi_\theta(a_t\mid s_t)
\right].
\]

There is a source-level ambiguity. If \(f\) remains attached to the computation
graph inside the accepted interval, then for
\(g(\theta)=r(\theta)A\log\pi_\theta\),

\[
\nabla g
=A r\,[1+\log\pi_\theta(a\mid s)]
\nabla\log\pi_\theta(a\mid s),
\]

which is not the usual importance-weighted policy gradient \(Ar\nabla\log\pi\).
If the ratio is stop-gradient, it *is* a sampled policy-gradient weight. The
paper's equation does not show a stop-gradient operator. A faithful
implementation must inspect released code or ask the authors; it must not hide
the choice.

Minimal code with the conventional detached-weight interpretation is:

```python
def sao_policy_loss(current_logp, rollout_logp, advantage,
                    action_mask, eps_low, eps_high):
    log_ratio = current_logp - rollout_logp
    ratio = log_ratio.exp()
    trusted = (ratio > 1.0 - eps_low) & (ratio < 1.0 + eps_high)
    weight = (ratio * trusted).detach()
    numerator = -(weight * advantage.detach() * current_logp * action_mask).sum()
    denominator = action_mask.sum().clamp_min(1)
    return numerator / denominator, trusted
```

The strict inequalities match the paper. Unit tests should include values
exactly at both boundaries because `>=` versus `>` changes their gradients.

### 18.3 The critic and two-timescale updates

SAO trains an actor \(\pi_\theta\) and critic \(V_\phi\). In the paper's
controlled experiments, each actor update is preceded or accompanied by
\(K=2\) critic updates:

\[
\phi\leftarrow\operatorname{Opt}_V(\phi)^K,
\qquad
\theta\leftarrow\operatorname{Opt}_\pi(\theta).
\]

The intention is a two-timescale process: the baseline should track the moving
policy faster than the actor changes. More critic steps do not guarantee this;
learning rate, replay distribution, bootstrap target, and representation drift
also matter. The paper freezes the critic's attention parameters and updates
its Mixture-of-Experts projections after observing unstable full-attention
gradients. That is an empirical regularizer, not a theorem that attention never
needs value adaptation.

Explained variance diagnoses baseline fit:

\[
\operatorname{EV}=1-
\frac{\operatorname{Var}(R-V_\phi(s))}
     {\operatorname{Var}(R)}.
\]

\(\operatorname{EV}=1\) is perfect prediction, \(0\) is no better in variance
than a constant mean, and a negative value is worse. When
\(\operatorname{Var}(R)\approx0\), the statistic is ill-conditioned and must be
guarded.

### 18.4 Skip-observation Generalized Advantage Estimation

**Generalized Advantage Estimation (GAE)** normally recurses through adjacent
decision states. An agent transcript interleaves model-generated actions and
environment-generated observations:

\[
\tau=[a_0,o_0,a_1,o_1,\ldots].
\]

Assigning policy/value loss to observation tokens is conceptually wrong: the
policy did not sample them. Let \(a_{i,N}\) be the final token of action \(i\)
and \(a_{i+1,0}\) the first token of the next action. SAO bridges the external
observation span:

\[
\begin{aligned}
\delta_i
&=r_i+\gamma V_\phi(a_{i+1,0})-V_\phi(a_{i,N}),\\
\widehat A(a_{i,N})
&=\delta_i+\gamma\lambda\widehat A(a_{i+1,0}).
\end{aligned}
\]

This skips *loss positions*, not information. The prefix of
\(a_{i+1,0}\) contains \(o_i\), so the next value can condition on the tool
result. For a terminal transition, replace the next value and recursive
advantage with zero. For a timeout truncation, bootstrap only when the
environment semantics say the episode could validly continue.

```python
def skip_observation_gae(action_end_values, next_action_values, rewards,
                         terminals, gamma, lam):
    # One entry per semantic model action, not per environment token.
    next_advantage = 0.0
    advantages = [0.0] * len(rewards)
    for i in reversed(range(len(rewards))):
        continuation = 1.0 - float(terminals[i])
        delta = (rewards[i]
                 + gamma * continuation * next_action_values[i]
                 - action_end_values[i])
        next_advantage = delta + gamma * lam * continuation * next_advantage
        advantages[i] = next_advantage
    return advantages
```

The paper also tests step-average and last-token step values in its appendix;
both underperform token-wise value training in that experiment. This does not
prove token-wise values dominate in every scaffold.

### 18.5 Evidence boundary for GLM-5.2

The paper's Qwen3-30B-A3B experiment discloses group size one, batch 128, 128K
maximum length, actor learning rate \(10^{-6}\), critic learning rate
\(5\times10^{-6}\), and \(K=2\). Reasoning uses
\((\epsilon_l,\epsilon_h)=(0.3,5.0)\); coding uses \((0.8,3.0)\).
These unusually wide *ratio* intervals must not be confused with conventional
PPO clip values.

The only direct GLM-5.2 statement is that SAO was deployed in its agentic-RL
pipeline. The GLM-5.2-specific stages, data, rewards, coefficients, compute,
and gains are not published. The correct lesson is the algorithm and its
systems motivation—not a fabricated 753B recipe.

## 19. Potential-based reward shaping

Add shaping

\[
F(s_t,s_{t+1})=\gamma\Phi(s_{t+1})-\Phi(s_t).
\]

Shaped discounted return telescopes:

\[
\begin{aligned}
G'_0
&=\sum_{t=0}^{T-1}\gamma^t[r_t+\gamma\Phi(s_{t+1})-\Phi(s_t)]\\
&=G_0-\Phi(s_0)+\gamma^T\Phi(s_T).
\end{aligned}
\]

For fixed initial state and appropriate terminal potential, action rankings are
preserved. Arbitrary process scores do not have this guarantee. A judge reward
for “looks like progress” can change the optimum to perform judge-visible
rituals.

## 20. Constrained objectives

Suppose task return \(J_R(\theta)\) and expected safety/cost constraint

\[
J_C(\theta)=\mathbb E[C(\tau)]\le d.
\]

Lagrangian:

\[
\mathcal L(\theta,\lambda)=
J_R(\theta)-\lambda(J_C(\theta)-d),\qquad\lambda\ge0.
\]

Policy update ascends \(\theta\) on reward minus constraint advantage. Multiplier
update increases \(\lambda\) when observed cost exceeds budget:

```python
policy_loss = -(reward_surrogate - lambda_cost * cost_surrogate)
lambda_cost = max(0.0, lambda_cost + dual_lr * (mean_cost - budget))
```

This controls an expectation, not worst-case catastrophe. Tool authorization,
sandbox, and irreversible-action confirmation remain hard external constraints.

## 21. A complete one-batch tensor calculation

Consider two trajectories, two action tokens each:

```text
old logp       [[-1.00, -0.50], [-0.20, -0.70]]
current logp   [[-0.90, -0.60], [-0.10, -1.00]]
advantages     [[+1.00, +1.00], [-0.50, -0.50]]
mask           [[1, 1], [1, 0]]
clip epsilon   0.20
```

Log-ratios at valid positions: `[+0.10, -0.10, +0.10]`.

Ratios:

\[
[e^{.1},e^{-.1},e^{.1}]
\approx[1.10517,0.90484,1.10517].
\]

All lie inside `[0.8, 1.2]`, so no clipping. Surrogates:

\[
[1.10517,0.90484,-0.55259].
\]

Mean objective over three valid tokens:

\[
\frac{1.10517+0.90484-0.55259}{3}
\approx0.48581.
\]

Minimized loss is \(-0.48581\). The fourth token is masked and contributes
neither numerator nor denominator.

Use this style of hand calculation before trusting a fused/distributed kernel.

## 21. Numerical and statistical checks

For any implementation:

1. `current == old` implies ratio one before update.
2. masked logit gradient is zero.
3. padding/repacking does not change globally normalized loss.
4. \(\lambda=1\) GAE equals Monte Carlo return minus value at true terminals.
5. uniform group reward yields no centered relative signal.
6. DPO loss is \(\log 2\) when chosen/rejected implicit margins are equal.
7. exact reverse KL is non-negative up to numerical tolerance and zero for equal
   distributions.
8. adding a constant to group rewards leaves centered advantages unchanged.
9. multiplying rewards changes non-standardized but not standardized group
   advantages; this is an objective property to document.
10. local-rank normalized losses match global concatenated loss only when
    denominators are aggregated correctly.

## References

1. Ronald J. Williams,
   [“Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning”](https://doi.org/10.1007/BF00992696),
   1992.
2. John Schulman et al.,
   [“High-Dimensional Continuous Control Using Generalized Advantage Estimation”](https://arxiv.org/abs/1506.02438),
   2015.
3. John Schulman et al.,
   [“Proximal Policy Optimization Algorithms”](https://arxiv.org/abs/1707.06347),
   2017.
4. Rafael Rafailov et al.,
   [“Direct Preference Optimization”](https://arxiv.org/abs/2305.18290), 2023.
5. Zhihong Shao et al.,
   [“DeepSeekMath”](https://arxiv.org/abs/2402.03300), 2024.
6. Zichen Liu et al.,
   [“Understanding R1-Zero-Like Training”](https://arxiv.org/abs/2503.20783),
   2025.
