# Source-Level Agentic RL Lab

This lab turns the equations into code small enough to audit. It intentionally
starts below an LLM framework: one semantic action is one token, the environment
has three turns, and an exact terminal verifier supplies reward. Once every
tensor is understood, the same contracts extend to transformer tokens, tool
observations, packed trajectories, and distributed rollout.

## 1. Repository map

| File | Purpose |
|---|---|
| `rosellm/roserlhf/advantages.py` | returns, GAE, leave-one-out/group advantages, turn-to-token mapping |
| `rosellm/roserlhf/losses.py` | token log-probabilities, masked reductions, clipped policy loss, sequence ratios |
| `rosellm/roserlhf/trajectory.py` | exact sampled-token and policy-version trajectory contract |
| `examples/agentic_rl_toy.py` | complete collect → verify → advantage → update loop |
| `tests/test_roserlhf_advantages.py` | terminal/truncation, GAE, group, masking invariants |
| `tests/test_roserlhf_losses.py` | target alignment, PPO clipping, padding, gradient-mask invariants |
| `tests/test_roserlhf_trajectory.py` | event ordering, exact log-probability, mixed-policy audit |

Run the focused suite:

```bash
pytest -q \
  tests/test_roserlhf_advantages.py \
  tests/test_roserlhf_losses.py \
  tests/test_roserlhf_trajectory.py
```

Run the toy training loop:

```bash
python examples/agentic_rl_toy.py --steps 150 --group-size 32 --seed 7
```

The default run should learn all four three-action tasks. The point is not the
score; it is being able to predict each tensor and failure.

## 2. The toy environment

There are four instructions/tasks, each mapping to a verified three-action path:

```python
TARGET_PATHS = tensor([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
])
```

The policy stores logits with shape

\[
[N_{\text{tasks}},T,N_{\text{actions}}]=[4,3,2].
\]

For each task, the collector samples \(G\) complete trajectories. The exact
verifier returns one only if all three actions match. This gives a sparse,
delayed terminal reward. The policy receives no step-level hint.

This is a finite-horizon MDP rather than a contextual bandit because actions are
ordered environment decisions and reward depends on the resulting path. The
simple policy does not need history because task + turn is sufficient for this
particular deterministic environment. Replacing it with a recurrent/Transformer
policy does not change the rollout or estimator contract.

## 3. Collect under a behavior snapshot

`collect_rollouts` runs under `torch.no_grad()`:

1. repeat each task ID \(G\) times;
2. compute behavior logits/probabilities;
3. sample each turn;
4. store sampled actions;
5. store their behavior log-probabilities;
6. execute the verifier;
7. return an immutable logical batch.

For \(B\) tasks, shapes are:

| Tensor | Shape |
|---|---:|
| `task_ids` | `[B * G]` |
| `actions` | `[B * G, T]` |
| `old_logprobs` | `[B * G, T]` |
| `rewards` | `[B * G]` |

The behavior log-probabilities are detached evidence. Recomputing them later
from text would be wrong for a real LLM because decoding/re-tokenization, masks,
quantization, routing, or changed weights can alter the distribution.

## 4. Group-relative credit

Reshape reward to `[B, G]` and compute

\[
A_{b,i}=\frac{R_{b,i}-\overline R_b}
{\operatorname{std}(R_{b,1:G})+\epsilon}.
\]

`group_standardized_advantages` uses population standard deviation and returns
exactly zero when a group is uniform. That no-signal behavior is an explicit
test. A production curriculum should log how often all samples fail or succeed.

The trajectory advantage has shape `[B * G]`. Because the toy has only terminal
reward and no critic/process reward, it is copied to all three generated action
positions:

\[
A_{i,0}=A_{i,1}=A_{i,2}=A_i.
\]

This is trajectory-level credit broadcast to tokens, not evidence that every
turn was equally causal.

## 5. Recompute current log-probabilities

After collection, call the trainable policy again on the exact task/action
sequence. Initially current weights equal behavior weights, so

\[
\log\pi_\theta(a)-\log\pi_{\text{old}}(a)=0,
\qquad \rho=1.
\]

The unit tests make ratio-one an invariant. In a real disaggregated system, a
failure indicates mismatched weights, tokenizer/template, positions, attention,
MoE routing, quantization, constrained sampling, or alignment.

For an autoregressive transformer, compute:

```python
logits = model(input_ids).logits
current_logprobs = gather_token_logprobs(
    logits[:, :-1],
    input_ids[:, 1:],
)
```

The shift is visible deliberately. An action mask aligned to target tokens must
also have length `sequence_length - 1`.

## 6. Clipped policy objective

The implementation computes

\[
\rho_{i,j}=
\exp(\log\pi_\theta(x_{i,j})-
\log\pi_{\text{old}}(x_{i,j})),
\]

\[
s_{i,j}=\min\left(
\rho_{i,j}A_{i,j},
\operatorname{clip}(\rho_{i,j},1-\epsilon_l,1+\epsilon_h)A_{i,j}
\right),
\]

and minimizes the negative globally token-normalized mean:

\[
L=-\frac{\sum_{i,j}m_{i,j}s_{i,j}}
{\sum_{i,j}m_{i,j}}.
\]

The test `test_clipping_uses_advantage_sign` verifies the asymmetric effect of
the `min`: for positive advantage, a ratio above the upper bound stops adding
gain; for negative advantage, the more negative unclipped term remains.

Diagnostics:

- non-negative approximate KL \(\rho-1-\log\rho\);
- clip fraction;
- mean ratio;
- objective and minimized loss.

The safety clamp on log-ratio prevents floating overflow only. It is not a
substitute for rejecting stale/corrupt samples.

## 7. Action masks

Real input contains system, user, assistant, and tool/environment spans:

```text
system      m=0
user        m=0
assistant   m=1  <- policy-generated action tokens
tool result m=0  <- environment observation
assistant   m=1
padding     m=0
```

`test_masked_observation_has_zero_gradient` assigns an enormous advantage to a
masked position and proves its gradient remains exactly zero.

Mask mistakes train qualitatively wrong behavior:

- tool result target → model learns to imitate the environment;
- user target → model learns to produce both sides of dialogue;
- hidden reasoning unintentionally masked → no RL signal reaches it;
- padding included → length/batch-dependent gradient noise;
- shifted mask → every token receives its neighbor's credit.

## 8. Terminal versus truncation

`discounted_returns` and GAE separate:

- `terminated=True`: true absorbing success/failure; future value is zero;
- `terminated=False` with `bootstrap_value`: collection stopped but the process
  could continue;
- `valid_mask=False`: padding; not a state transition at all.

Example with \(\gamma=.5\), rewards `[1, 2]`, no true terminal, bootstrap 10:

\[
G_1=2+.5(10)=7,
\qquad G_0=1+.5(7)=4.5.
\]

Marking the time limit terminal would instead produce `[2, 2]` and bias the
critic/actor against states near the collector limit.

## 9. GAE source trace

For each step in reverse:

\[
\delta_t=r_t+\gamma(1-d_t)V_{t+1}-V_t,
\]

\[
\hat A_t=\delta_t+gamma\lambda(1-d_t)\hat A_{t+1}.
\]

The code preserves `next_value` and `next_advantage` while crossing padding.
With \(\lambda=1\) and a true final terminal, tests prove

\[
\hat A_t=G_t-V_t,
\qquad \text{value target}=G_t.
\]

Before optimizing a large critic, reproduce this equality on a hand-created
batch and test time-limit behavior.

## 10. Leave-one-out versus standardized group baselines

For RLOO:

\[
A_i=R_i-\frac{1}{G-1}\sum_{k\ne i}R_k.
\]

The baseline excludes the current trajectory. For standardized GRPO-like
advantages, current reward participates in the mean/std. These estimators have
different finite-sample coupling and prompt weighting. Both are implemented so
the same rollout batch can be compared without changing data.

Exercise: log gradient variance over 1,000 independently sampled groups for:

1. raw REINFORCE;
2. global batch mean;
3. leave-one-out by task;
4. group mean only;
5. group mean/std.

Hold samples and loss normalization fixed.

## 11. Sequence ratios

`sequence_log_ratio` exposes two choices:

\[
\Delta_{\text{sum}}=\sum_j
[\log\pi_\theta(x_j)-\log\pi_{\text{old}}(x_j)],
\]

or

\[
\Delta_{\text{mean}}=\frac1L\Delta_{\text{sum}}.
\]

Exponentiating the sum gives the exact sampled sequence probability ratio but
can explode/vanish with length. Exponentiating the mean gives a geometric-mean
token ratio and changes length weighting. The API forces this choice to be
named; there is no neutral default in research interpretation.

## 12. Exact trajectory records

`PolicyAction` requires:

- contiguous turn;
- exact token IDs;
- one behavior log-probability per token;
- behavior policy version;
- decoded text and optional parsed semantic action.

`TransitionRecord` keeps observation artifact reference, separate reward
components, terminal/truncated flags, and optional state hash. `Trajectory`
enforces one transition per action, ordered turns, and no events after finish.

Partial continuation can produce policy versions `[v1, v2]`; the record exposes
this rather than pretending the complete episode came from v2.

This small class is not a distributed storage format. It defines invariants a
Parquet/Arrow/event-log implementation must preserve.

## 13. Progressive exercises

### Level 0 — Hand calculation

- Enumerate every trajectory probability for one task.
- Compute expected terminal reward.
- Derive the exact policy gradient by summation.
- Compare it with Monte Carlo estimates.

### Level 1 — Baselines

- Implement raw REINFORCE and RLOO.
- Measure variance and confirm baseline-centered expected gradient.
- Show a group of all failures has no relative signal.

### Level 2 — Multi-turn rewards

- Add a verified reward after each correct prefix.
- Compare trajectory reward, reward-to-go, and GAE.
- Add potential-based shaping and verify the optimal path is unchanged.

### Level 3 — Partial observability

- Hide task ID; reveal a noisy clue in the first observation.
- Replace table policy with GRU/Transformer memory.
- Compare full history, summary, and learned recurrent state.

### Level 4 — Tokenized semantic actions

- Encode each action as multiple tokens with multiple equivalent spellings.
- Add a parser and invalid format.
- Demonstrate that token-level credit changes formatting probability even when
  semantic actions are equivalent.

### Level 5 — Tool observations and masks

- Insert environment-generated result tokens between policy spans.
- Pack multiple trajectories with block-diagonal attention.
- Prove tool/padding/cross-example gradients are zero.

### Level 6 — Actor–critic

- Learn turn values.
- Plot calibration and explained variance by remaining horizon.
- Compare PPO/GAE with critic-free estimators at equal rollout compute.

### Level 7 — Asynchrony

- Separate collector/trainer processes.
- Tag policy versions and simulate delayed trajectories.
- Plot divergence/ratio/gradient quality versus policy lag.
- Add bounded-lag rejection and a V-trace variant.

### Level 8 — Real small language model

- Replace policy table with a tiny causal transformer.
- Use exact token-in/token-out rollouts.
- Add a deterministic calculator/search/code tool.
- Record task/env/tokenizer/policy/reward hashes.
- Reproduce synchronous single-GPU learning before adding distributed engines.

## 14. Debugging order

1. Freeze one trajectory and print every token, role, mask, turn, reward.
2. Verify behavior/current ratio is one before the first update.
3. Hand-compute return/advantage and compare.
4. Run loss forward twice with padding/repacking; result must match.
5. Inspect gradients at masked and selected logits.
6. Overfit a tiny deterministic task set.
7. Replay collection offline and match the online update.
8. Establish single/multi-GPU global-denominator parity.
9. Only then add asynchronous rollout and policy lag.

## 15. Reading production source

After the local lab, trace the same concepts in pinned revisions:

- [veRL](https://github.com/verl-project/verl): agent loop, token-in/token-out,
  FSDP/Megatron, vLLM/SGLang, dataflow placement.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF): Ray actors, hybrid engine,
  PPO and critic-free recipes.
- [Slime](https://github.com/THUDM/slime): exact trajectory masks, SGLang
  rollout, PPO utilities, asynchronous rollout, OPD.
- [AReaL](https://github.com/inclusionAI/AReaL): policy-lag-aware asynchronous
  language RL.
- [Agent Lightning](https://github.com/microsoft/agent-lightning): decoupled
  arbitrary-agent tracing and hierarchical credit.
- [verl-agent](https://github.com/langfengQ/verl-agent): long-horizon
  environments and group-in-group optimization.

For each framework, locate the concrete implementation of:

1. token/action mask;
2. old log-probability origin;
3. advantage estimator;
4. policy ratio and clipping;
5. KL location and estimator;
6. loss denominator;
7. policy version and weight sync;
8. terminal/truncation semantics;
9. environment failure filtering;
10. checkpoint/resume state.

If documentation and source differ, the pinned source governs that revision and
the discrepancy should be recorded.

## References

1. Ronald J. Williams,
   [“Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning”](https://doi.org/10.1007/BF00992696),
   1992.
2. John Schulman et al.,
   [“Proximal Policy Optimization Algorithms”](https://arxiv.org/abs/1707.06347),
   2017.
3. Zhihong Shao et al.,
   [“DeepSeekMath”](https://arxiv.org/abs/2402.03300), 2024.
4. Guangming Sheng et al.,
   [“HybridFlow”](https://arxiv.org/abs/2409.19256), 2024/2025.
