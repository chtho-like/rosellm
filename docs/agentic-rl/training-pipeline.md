# End-to-End Training Pipeline

This chapter follows one Agentic RL run from task specification to a promoted
checkpoint. It describes the operational pipeline that an optimizer equation
leaves out: data lineage, exact token handling, environment execution, reward,
advantage construction, distributed updates, evaluation, and rollback.

## 1. The pipeline at a glance

```text
task sources + environment snapshots
              |
              v
      task/version registry ---------> hidden evaluation registry
              |
              v
 demonstrations / teacher rollouts --> tool-format + behavior SFT
              |                              |
              |                              v
              +------------------------ initial policy
                                             |
                         +-------------------+-------------------+
                         |                                       |
                         v                                       |
                rollout inference servers                        |
                         | exact tokens + logprobs                |
                         v                                       |
 agent loops <--> versioned environments/tools                    |
                         | event trajectories                     |
                         v                                       |
             validators / reward models / judges                  |
                         | reward components                      |
                         v                                       |
          filtering + returns + advantages + packing              |
                         | training tensors                       |
                         v                                       |
      policy update (+ critic/reference/reward as configured)     |
                         | new checkpoint                         |
                         +-------------------+-------------------+
                                             |
                                   fixed + interactive eval
                                             |
                           promote / continue / rollback / stop
```

The arrows are versioned data products. If a model score cannot be traced back
through each arrow, the experiment is not reproducible.

## 2. Stage 0 — Write the learning contract

Before selecting PPO or GRPO, fix the experiment's meaning.

```yaml
objective:
  target_distribution: agent_tasks_v4
  primary_metric: verified_success_rate
  constraints:
    unsafe_action_rate: 0
    p95_episode_cost_usd: 0.20
    p95_turns: 30
  regression_floors:
    base_knowledge_eval: 0.98_of_initial
    tool_schema_holdout: 0.75
budgets:
  rollout_tokens: 2.0e9
  optimizer_tokens: 2.0e9
  environment_hours: 50000
  accelerator_hours: 12000
stop_rules:
  - safety_violation
  - no_validation_gain_for_5_evals
  - kl_above_limit_for_3_updates
```

Specify target task mixture, success, costs, hard constraints, regression floors,
compute/data budgets, and stop rules. Otherwise dynamic curricula and reward
weights silently redefine success during the run.

## 3. Stage 1 — Freeze interfaces and artifacts

Assign immutable identifiers to:

- base/SFT policy checkpoint;
- tokenizer, vocabulary, and chat template;
- tool schemas and structured-output grammar;
- task corpus and split manifest;
- environment image, snapshot, and validator;
- reward-model/judge checkpoint and rubric;
- training code revision and resolved configuration;
- dependency/container image and hardware topology.

Run conformance tests between the SFT data formatter, rollout server, agent
parser, and trainer. The same assistant/tool boundary must produce identical
token IDs and loss masks everywhere.

## 4. Stage 2 — Build an initial policy with support

Pure RL from a base model is scientifically interesting but operationally
expensive when most trajectories are invalid. A typical bootstrap is:

1. **format SFT:** valid chat roles, action delimiters, and JSON/tool schemas;
2. **single-step SFT:** when and how to call each tool;
3. **trajectory SFT:** planning, observation use, recovery, and termination;
4. **negative/recovery coverage:** examples from malformed calls, tool failures,
   ambiguous instructions, permission denial, and unsafe requests;
5. **optional distillation/rejection sampling:** execute teacher samples and
   retain independently verified trajectories.

Train only policy-authored target spans. If a tool result becomes a target token,
the model is rewarded for predicting the environment rather than choosing the
next action.

Evaluate the SFT policy before RL. Record validity, success, entropy, trajectory
length, tool-use distribution, and safety. This checkpoint is both baseline and
often the KL reference.

## 5. Stage 3 — Configure rollout sampling

The behavior distribution is defined by more than model weights:

- temperature, top-p/top-k/min-p;
- repetition and presence penalties;
- maximum tokens per action and episode;
- constrained-decoding grammar and token masks;
- stop sequences and end-of-turn rules;
- number of trajectories per task/group;
- environment seed and user-simulator policy;
- tool retry, timeout, and error policy;
- context truncation, summarization, and memory retrieval.

Record the resolved sampler with every trajectory. For policy-gradient training,
store log-probabilities under the **actual behavior distribution after masks and
sampling transformations**, or recompute them using an exactly equivalent
frozen behavior policy.

Temperature changes the behavior logits. If base logits are \(z\), sampling
uses \(z/T\); the behavior log-probability is derived from the scaled and masked
distribution, not the unscaled training logits.

## 6. Stage 4 — Select and launch tasks

A curriculum sampler emits `(task_id, environment_version, seed, group_id)`.
For group-relative algorithms, all \(G\) trajectories in a group normally share
the task but use independent policy/environment sampling seeds.

The dispatcher should:

1. reject evaluation IDs through an enforced denylist;
2. enforce desired task weights and per-domain quotas;
3. reserve group membership before dispatch;
4. attach the behavior policy version;
5. account for launched, completed, failed, timed-out, and filtered attempts;
6. retry infrastructure failures without converting them into task failures;
7. avoid retrying a state-changing action unless idempotency is known.

If only completed fast episodes reach the trainer, asynchronous collection
creates latency-based selection bias. Preserve the attempt census and define
what happens to stragglers at update boundaries.

## 7. Stage 5 — Execute the agent loop

Conceptual pseudocode:

```python
obs = env.reset(task, env_seed)
state = agent.initialize(obs)

for turn in range(max_turns):
    prompt = agent.render(state)
    sample = policy.generate_tokens(prompt, sampler, policy_version)
    action = agent.parse(sample.token_ids)

    log_policy_event(
        prompt_ids=prompt.token_ids,
        token_ids=sample.token_ids,
        old_logprobs=sample.logprobs,
        policy_version=policy_version,
        parser_version=agent.parser_version,
    )

    if not action.valid:
        transition = invalid_action_transition(action.error)
    elif not authorizer.allows(action):
        transition = denied_action_transition(action)
    else:
        transition = env.step(action)

    log_transition(transition)
    state = agent.update(state, action, transition.observation)

    if transition.terminated or transition.truncated:
        break
```

The policy server should return token IDs and log-probabilities directly. Text
is retained for analysis and parsing, not decoded then re-tokenized for
training. This exact-token constraint is emphasized in veRL's
[agentic RL design](https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst).

## 8. Stage 6 — Validate and compute reward

Compute reward after the complete event log is immutable. Keep components
separate:

```text
R = success
  + progress_shaping
  + process_quality
  + user_preference
  - tool_cost
  - latency_cost
  - invalid_action_cost
  - safety_violation_cost
  - KL_regularization
```

For every component store:

- name and version;
- raw score before scaling/clipping;
- weight and transformation;
- turn/span/trajectory scope;
- evidence used by the evaluator;
- evaluator model/validator version;
- error and uncertainty metadata.

Hard safety or permission violations should usually terminate/quarantine the
episode and trigger an incident path, not merely subtract a reward that task
success can outweigh.

### Outcome-verifiable tasks

Run the trusted validator in a separate context. Do not expose hidden tests,
expected outputs, or evaluator traces to the policy. Distinguish validator
failure from agent failure.

### Learned reward or judge

Batch scoring may reduce cost, but preserve the exact rubric and input view.
Calibrate score/ranking against humans or verifiers on held-out data. Freeze the
judge during a policy comparison unless the experiment studies co-evolution.

### KL reward

If KL is included as a per-token shaped reward, compute reference log-probability
on exactly the sampled tokens and prefixes. Make clear whether the KL enters the
reward/return, the policy loss, or both; double application is a common bug.

## 9. Stage 7 — Accept, filter, and quarantine

Separate three classes:

1. **statistically valid training episode:** environment and logging succeeded;
   task success may be zero;
2. **infrastructure-invalid episode:** corrupt snapshot, missing tokens, server
   crash, reward failure, duplicate event, irrecoverable version mismatch;
3. **safety incident:** unauthorized or unexpected behavior requiring review.

Do not drop ordinary failed trajectories merely because reward is low. That
changes the behavior distribution and eliminates negative advantages. Dynamic
sampling may drop all-equal groups by design, but it must log the acceptance
rule and denominator.

Validate:

- token/log-probability lengths and finite values;
- policy, tokenizer, template, parser, environment, and reward versions;
- event ordering and action-to-observation pairing;
- group completeness or declared partial-group policy;
- termination versus truncation;
- reward ranges and component availability;
- duplicate trajectory/action delivery;
- evaluator leakage and sandbox incidents.

## 10. Stage 8 — Construct returns and advantages

Choose the credit unit explicitly.

### Actor–critic / PPO

1. map reward events to decision steps;
2. obtain value predictions for each history/turn;
3. bootstrap only for configured nonterminal truncations;
4. compute TD residuals and GAE;
5. optionally normalize advantages over a defined population;
6. broadcast each turn advantage to its policy-token span, unless token-level
   values/rewards are modeled.

### RLOO / GRPO-like

1. group trajectories by task instance;
2. compute scalar or turn-aligned return for each sample;
3. form leave-one-out or group-standardized advantages;
4. handle zero-variance groups according to the declared algorithm;
5. record group statistics before filtering/normalization;
6. map trajectory/turn advantages to policy tokens.

### Multi-turn decomposition

If reward is attached at each turn, compute reward-to-go per turn. If a process
model assigns span rewards, document whether they are additive rewards, direct
advantages, value targets, or sample weights; these choices are not equivalent.

## 11. Stage 9 — Reconstruct the training sequence

For each trajectory, create a token stream with roles/segments:

```text
[system][user][assistant action 0][tool observation 0]
[assistant action 1][tool observation 1]...[assistant final]
```

Build:

- causal `input_ids` and position metadata;
- attention structure that prevents cross-example leakage when packed;
- `action_mask` selecting only tokens sampled by the trained policy;
- old behavior log-probabilities aligned to next-token targets;
- reference log-probabilities if required;
- turn IDs and advantage per action token;
- value targets and masks for the critic;
- sample/task/group weights;
- policy and environment metadata outside model tensors.

The last input token predicts the next token, so logits at position \(i\) align
with target `input_ids[i+1]`. Unit-test this shift with a hand-built sequence.

Context overflow is not solved by silently dropping early observations. Define
whether examples are rejected, truncated, summarized, memory-compressed, or
split—and whether old log-probabilities remain valid under the transformed
prefix.

## 12. Stage 10 — Compute current and auxiliary log-probabilities

Run the current actor on the exact stored sequence. Gather log-softmax at each
sampled target token:

```python
shifted_logits = logits[:, :-1]
targets = input_ids[:, 1:]
current_logprobs = log_softmax(shifted_logits, -1).gather(
    dim=-1, index=targets[..., None]
).squeeze(-1)
```

Apply `action_mask` only after the target shift is aligned. For tensor-parallel
vocabularies, use a numerically equivalent distributed log-softmax/gather.

Compute reference log-probabilities with a frozen checkpoint or reuse immutable
precomputed values only when prefixes and reference version are identical.
Compute critic values at the declared boundary (token, turn-end, or separate
state token).

## 13. Stage 11 — Form the policy objective

For a PPO/GRPO-style token surrogate:

```python
log_ratio = current_logprobs - old_logprobs
ratio = exp(log_ratio)
unclipped = ratio * advantages
clipped = clamp(ratio, 1 - clip_low, 1 + clip_high) * advantages
policy_terms = minimum(unclipped, clipped)
loss = -global_masked_reduce(policy_terms, action_mask)
```

Production details that change the objective:

- symmetric versus asymmetric clipping;
- token- versus sequence-level ratio;
- token, sequence, turn, or task normalization;
- multiple optimizer epochs and minibatch order;
- maximum log-ratio or numerical clamp;
- KL to old versus reference policy;
- entropy bonus and its mask;
- per-task and importance weights;
- overlong or invalid samples;
- gradient accumulation and distributed denominator.

Log the fraction clipped separately for positive and negative advantages and
the ratio distribution by policy lag.

## 14. Stage 12 — Optimize

A typical update:

1. zero or accumulate gradients according to a declared global batch;
2. forward actor (and critic if present) under mixed precision;
3. compute policy, value, entropy, KL, and auxiliary losses;
4. globally normalize using true valid-token/sample denominators;
5. backpropagate with activation checkpointing as configured;
6. unscale gradients if loss scaling is used;
7. compute and all-reduce the intended global gradient norm;
8. clip once under the chosen sharding semantics;
9. optimizer step and schedule step;
10. update adaptive KL controller/target models if configured;
11. increment immutable policy version;
12. checkpoint model, optimizer, scheduler, RNG, curriculum, and data cursors.

For multiple epochs over a rollout batch, samples become increasingly off-policy
relative to the current actor. Monitor divergence by epoch; “PPO” does not make
arbitrary reuse statistically safe.

## 15. Stage 13 — Synchronize rollout weights

Two common designs:

### Stop-and-synchronize

Pause collectors, finish an update, convert/shard weights, load every inference
worker, then resume with a new version. This minimizes version ambiguity but
creates idle time.

### Asynchronous versioned refresh

Collectors continue under older checkpoints while new weights are published.
Each trajectory retains its behavior version. The trainer admits samples only
within a lag/divergence policy and uses the corresponding old log-probabilities.

A robust publication protocol writes a complete checkpoint to a temporary
version, verifies hashes, atomically marks it ready, and requires workers to
acknowledge the exact version. Never let a server sample while some tensors have
new weights and others old weights.

## 16. Stage 14 — Evaluate during training

Use at least four suites:

1. **training-distribution validation:** detects optimization progress;
2. **held-out task/state generalization:** detects overfitting;
3. **base capability regressions:** knowledge, instruction following, language,
   and non-agent tasks;
4. **safety/adversarial evaluation:** permission, prompt injection, data
   exfiltration, reward tampering, and denial/abstention.

Freeze task IDs, environment versions, budgets, sampler, scaffold, tools, and
judges across checkpoint comparisons. Run repeated seeds for stochastic
environments and use confidence intervals. Keep evaluation workers and artifacts
isolated from training.

## 17. Stage 15 — Decide: promote, continue, rollback, or stop

Promotion is a gate, not “latest checkpoint wins.” Require:

- statistically supported primary-metric improvement;
- all hard constraints and regression floors;
- acceptable cost/latency/length distribution;
- no unresolved safety incidents;
- reward–evaluation agreement on independent metrics;
- reproducible checkpoint and data lineage;
- review of qualitative failures, not only averages.

Rollback to a known checkpoint and curriculum/reward version when reward
hacking, collapse, infrastructure corruption, or safety regression occurs.
Preserve the failed run; it is diagnostic data.

## 18. Scaling gates

Increase scale only after the previous gate passes.

### Gate A: single process, tiny model, deterministic environment

- exact trajectory replay;
- hand-calculated returns and advantages;
- finite-difference or autograd gradient checks;
- overfit a tiny set and observe expected direction.

### Gate B: multi-process CPU environment + one training GPU

- RPC retry/idempotency tests;
- queue backpressure and cancellation;
- no lost/duplicate events;
- identical loss to offline replay.

### Gate C: multi-GPU synchronous

- single-GPU parity on the same global batch;
- correct global denominators and gradient norms;
- checkpoint resume equality within tolerance.

### Gate D: inference/training disaggregation

- exact policy-version/log-probability audit;
- weight publication atomicity;
- tokenizer/template/parser parity;
- throughput and GPU-bubble trace.

### Gate E: asynchronous and multi-node

- lag-stratified ratio/KL metrics;
- straggler and selection-bias accounting;
- worker/node failure injection;
- bounded replay and recovery;
- end-to-end cost and safety load test.

## 19. Minimal synchronous loop

```python
policy = load_policy(sft_checkpoint)
reference = load_frozen_reference(sft_checkpoint)
policy_version = 0

while not stop_condition():
    tasks = curriculum.sample(batch_tasks)
    trajectories = collect_groups(
        policy=policy,
        policy_version=policy_version,
        tasks=tasks,
        group_size=group_size,
    )

    valid, incidents = validate_trajectory_records(trajectories)
    quarantine(incidents)
    rewards = reward_pipeline(valid)
    advantages = estimator(valid, rewards)
    batch = pack_exact_tokens(valid, advantages)

    for epoch in range(update_epochs):
        for minibatch in batch.iter_minibatches():
            current = policy.logprobs(minibatch)
            reference_lp = reference.logprobs(minibatch)
            loss, metrics = policy_objective(
                current=current,
                old=minibatch.old_logprobs,
                reference=reference_lp,
                advantages=minibatch.advantages,
                action_mask=minibatch.action_mask,
            )
            optimizer_step(loss)

    policy_version += 1
    checkpoint(policy, optimizer, policy_version, data_cursors=True)
    evaluate_and_gate(policy, policy_version)
```

This loop omits performance machinery but not the logical data dependencies. An
asynchronous design must preserve equivalent provenance while allowing stages
to overlap.

## 20. Run dashboard

Monitor by task, policy version, and rollout lag:

### Capability

- verified success and partial progress;
- pass@1/pass@k under fixed sample budgets;
- tool/action validity, recovery, and termination accuracy;
- held-out and regression metrics.

### Learning dynamics

- raw reward components, returns, group variance, advantages;
- current/old ratios, clip fractions, approximate KL;
- reference KL, entropy, response/episode length;
- critic explained variance and value calibration;
- gradient/parameter norms and optimizer statistics.

### Systems

- generation and training tokens/s;
- policy GPU model-forward utilization;
- environment latency distribution and failure types;
- queue depth, backpressure, stragglers, accepted-token yield;
- weight-sync time, policy lag, checkpoint/restart time;
- accelerator, environment, tool/API, and annotation cost.

### Safety and integrity

- authorization denials and attempted violations;
- prompt-injection success;
- validator/judge errors and suspicious reward outliers;
- evaluation ID access attempts;
- non-replayable state and artifact/hash failures.

## 21. Common pipeline failures

| Symptom | Likely causes |
|---|---|
| Reward rises, verified success flat | judge/reward exploitation, curriculum shift, leakage |
| KL instantly large | wrong reference/template, shifted token alignment, stale behavior logprobs |
| Ratio is not one before first update | sampling/training distribution mismatch or reconstruction bug |
| Model emits tool observations | SFT/RL mask trained environment spans |
| Longer responses dominate | token-sum objective, reward-length correlation, truncation policy |
| Multi-GPU differs from one GPU | local denominator averaging, packing leakage, gradient clipping semantics |
| Async training unstable | policy lag, completion-time selection bias, mixed versions, duplicate samples |
| All groups give zero advantage | curriculum too easy/hard, verifier broken, grouping wrong |
| Tool syntax improves but task success falls | proxy reward overweighted or parser/scaffold changed |
| Resume diverges immediately | RNG, optimizer, scheduler, curriculum, queue, or environment cursor missing |

## References

1. Long Ouyang et al.,
   [“Training language models to follow instructions with human feedback”](https://arxiv.org/abs/2203.02155),
   2022.
2. Zhihong Shao et al.,
   [“DeepSeekMath”](https://arxiv.org/abs/2402.03300), 2024.
3. Daya Guo et al.,
   [“DeepSeek-R1”](https://arxiv.org/abs/2501.12948), 2025.
4. Guangming Sheng et al.,
   [“HybridFlow: A Flexible and Efficient RLHF Framework”](https://arxiv.org/abs/2409.19256),
   2024/2025.
5. Shangchun Fu et al.,
   [“AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning”](https://arxiv.org/abs/2505.24298),
   2025.
6. veRL,
   [“Agentic RL Training”](https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst),
   documentation, accessed through the linked moving branch; pin a revision for
   an experiment.
