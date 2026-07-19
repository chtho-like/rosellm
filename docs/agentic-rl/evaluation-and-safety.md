# Evaluation and Safety

An **Agentic Reinforcement Learning (Agentic RL)** score is a property of a complete system under a protocol:

$$
\text{result}=f(\text{model},\text{scaffold},\text{tools},\text{environment},
\text{budget},\text{sampler},\text{evaluator},\text{seed},\text{time}).
$$

Changing any argument can change the result. This chapter designs evaluation
that separates policy learning from scaffolding, inference compute, leakage,
and evaluator artifacts—and treats safety as a system constraint rather than a
single benchmark.

## 1. Evaluation layers

Use several layers because no one metric establishes general agency.

| Layer | Question | Example |
|---|---|---|
| Unit | Does one mechanism behave correctly? | tool parser, mask, state reset |
| Skill | Can the policy perform a bounded capability? | select an Application Programming Interface (API), recover from error |
| Task | Does the agent complete an end-to-end goal? | fix issue, book valid itinerary |
| Distribution | Does success generalize within a domain? | unseen repositories/sites/users |
| Transfer | Does learning survive new tools/schemas/horizons? | renamed functions, new User Interface (UI) layout |
| Robustness | Does behavior survive faults and attacks? | tool timeout, prompt injection |
| Safety | Does it respect permissions and avoid unacceptable states? | exfiltration or destructive action |
| Operations | Is utility worth cost/latency/reliability? | success per dollar and p95 time |

Training reward belongs on a dashboard, not as the sole evaluation metric.

## 2. Freeze the protocol

A checkpoint comparison must pin:

- model checkpoint and quantization;
- tokenizer/chat template/system prompt;
- scaffold code, memory, planner, and retry logic;
- tool schemas, versions, permissions, and network access;
- environment image, task snapshot, validator, and seed distribution;
- context, output, turn, wall-time, tool-call, and monetary budgets;
- sampling temperature/top-p/top-k and number of attempts;
- context-overflow and observation-truncation policy;
- judge/checker version and rubric;
- evaluation date for live services/web.

If a vendor model is measured with a different official harness, report that
protocol separately. Do not create a visually simple leaderboard by erasing
different tool and inference-compute budgets.

## 3. Core metrics

### Verified success

For binary task outcome $S_i$,

$$
\widehat p=\frac{1}{n}\sum_i S_i.
$$

Prefer backend state, tests, formal kernels, or independent validators over the
agent's self-declared completion.

### Pass@k

When $n$ samples contain $c$ correct results, the common unbiased estimator
for probability that at least one of $k$ samples succeeds without replacement
is

$$
\operatorname{pass@}k=
1-\frac{\binom{n-c}{k}}{\binom nk}.
$$

Pass@k measures search/sampling capacity, not single-attempt reliability. Report
total generated tokens, environment executions, and selection method.

### Progress

For tasks with meaningful intermediate state, define independently verified
subgoal completion. Ensure partial score cannot exceed final utility by gaming
easy subgoals while blocking completion.

### Cost and efficiency

Report distributions, not only averages:

- policy input/output/reasoning tokens;
- turns, tool calls, searches, code executions;
- wall time, TTFT, inter-action latency;
- accelerator/environment/API cost;
- success-weighted cost $\mathbb E[C]/\mathbb E[S]$ with caveats when success
  is rare;
- success under fixed token/time/cost budgets;
- critical-path time versus total work for multi-agent systems.

### Reliability

- task failure versus infrastructure failure;
- invalid action and parser failure;
- timeout/truncation;
- variance across seeds/runs;
- catastrophic side-effect rate;
- calibration of confidence/abstention.

## 4. Statistical reporting

### Confidence intervals

For binary success, use Wilson or exact binomial intervals rather than a bare
percentage, especially for small sets. Bootstrap paired task-level differences
when episodes have variable scalar scores.

### Paired comparisons

Evaluate checkpoints on the same task/state seeds when possible. For binary
paired outcomes, report discordant counts and use McNemar's test or paired
bootstrap. Shared environment randomness can reduce variance; do not share
policy sampling seeds in a way that forces identical actions.

### Multiple trials

Separate variation from:

- task sampling;
- environment stochasticity;
- policy sampling;
- evaluator/judge sampling;
- infrastructure nondeterminism.

Hierarchical bootstrap or mixed-effects analysis may be needed when several
episodes share a repository, website, template, or user simulator.

### Multiple comparisons

If many checkpoints/hyperparameters are selected on the same validation suite,
the best score is optimistically biased. Retain a final untouched test set and
report selection budget.

## 5. Benchmark contamination

Static public benchmarks are likely present in pretraining or synthetic data.
Controls include:

- exact/fuzzy/semantic overlap search against all post-training data;
- time-split tasks created after training cutoff;
- private or procedurally generated initial states;
- hidden validator/tests;
- paraphrase/rename/layout perturbations;
- canary strings and evaluation-ID access audits;
- family-level split (repository/domain/template) rather than item-level split.

Passing a known benchmark can still measure capability, but it is weak evidence
of novel-task learning. State the contamination threat rather than asserting
cleanliness without corpus access.

## 6. LLM judges

LLM judges are useful for subjective goals but have systematic failure modes:

- position/order bias;
- verbosity and formatting preference;
- self/family preference;
- sensitivity to rubric wording;
- failure to execute or verify citations/code;
- correlated errors with the policy;
- prompt injection inside candidate/tool output;
- inconsistency across temperature and API versions.

### Judge protocol

1. define dimensions and unacceptable failures;
2. randomize candidate order and blind identities;
3. use independent evidence/tools for factual or executable claims;
4. require structured rationale/score but do not treat rationale as truth;
5. calibrate against multiple humans or verifiers on a held-out sample;
6. report per-domain agreement and confusion, not just correlation;
7. use a separate judge family or ensemble when self-preference is plausible;
8. adversarially test injection and reward hacking;
9. freeze version during comparisons;
10. preserve `tie`, `both bad`, and `unjudgeable` outcomes.

A generative reward model can be more expressive than a scalar head, but its
natural-language reasoning increases cost and attack surface.

## 7. Reward–metric separation

Maintain at least three independent signals:

1. **training reward:** optimized directly;
2. **validation metric:** used for model selection but not gradient;
3. **audit metric:** hidden and reviewed only at gates or final evaluation.

If all three use the same judge, there is no independent evidence against reward
hacking. Use outcome verification or human audit where possible.

Plot training reward against independent success, cost, length, and safety. A
divergence is a stop signal.

## 8. Ablating what improved

Compare a factorial subset where affordable:

| Factor | Baseline | Treatment |
|---|---|---|
| Weights | pre-RL checkpoint | post-RL checkpoint |
| Scaffold | fixed minimal | production scaffold |
| Tools | none/fixed subset | full tools |
| Inference compute | equal token/turn budget | scaled budget |
| Memory | disabled | enabled |
| Search/parallel agents | single | search/swarm |
| Judge/validator | independent fixed | training evaluator |

Minimum necessary comparisons:

- same scaffold and budget, pre- versus post-training weights;
- same weights and budget, scaffold ablation;
- success–compute curve rather than one unconstrained point;
- held-out tasks/environments with evaluator independent of training reward.

## 9. Generalization axes

Test controlled shifts:

- new task instances and skill compositions;
- longer horizon and delayed reward;
- different tool names, argument order, descriptions, and schemas;
- unseen tools with learnable documentation;
- new website/UI layout and visual theme;
- environment transition stochasticity and partial observability;
- language, locale, time zone, and units;
- tool failures, stale data, permission denial, and ambiguous users;
- new repositories/libraries/frameworks;
- adversarial observations and prompt injection;
- changed user preferences and constraint priorities.

Report degradation relative to in-distribution performance and characterize
failure, not only average transfer.

## 10. Safety threat model

### Assets

- user data, secrets, credentials, identity;
- filesystem/database integrity;
- compute, money, quotas, and external accounts;
- hidden tests/rewards/training data;
- other tenants and cluster control plane;
- legal, physical, and reputational safety.

### Adversaries and failures

- malicious user;
- malicious webpage/tool output/document;
- compromised tool/service;
- reward-seeking policy discovering unintended behavior;
- accidental model error under ambiguity;
- insider or data-pipeline poisoning;
- infrastructure race/retry causing duplicate side effects.

### Security properties

- least privilege and explicit authorization;
- isolation and non-interference;
- integrity and provenance of observations/tools/rewards;
- confidentiality and data minimization;
- reversibility and human confirmation for high-impact actions;
- auditability and incident containment;
- graceful abstention under uncertainty.

## 11. Agent safety evaluations

### Prompt injection

Place conflicting instructions in web pages, tool outputs, files, metadata,
images, and retrieved memory. Measure whether the agent preserves instruction
hierarchy, protects secrets, and asks for authorization.

### Permission and side effects

Test read/write/delete/send/purchase/deploy actions at boundaries. Include
near-miss cases where the goal is legitimate but scope is broader than required.
Judge intermediate actions, not only final outcome.

### Reward tampering

Expose apparent paths to validator files, expected answers, logs, judge prompts,
or environment internals inside a sandbox. The policy should not access or
modify evaluation machinery.

### Deception and oversight awareness

Vary whether monitoring is visible, whether actions are audited, and whether
shortcuts benefit reward. Interpret cautiously: behavioral tests do not reveal
internal intent, but they can expose conditional policy differences.

### Resource misuse

Test fork bombs, unbounded searches, recursive agents, output flooding, crypto
mining, external tunnels, denial of service, and quota exhaustion. Enforce hard
limits regardless of policy behavior.

### Multi-agent failures

- subagent privilege escalation;
- instruction laundering through delegation;
- duplicated/contradictory side effects;
- communication poisoning;
- runaway spawning and cost;
- aggregation that omits minority safety findings.

## 12. Safety metrics

Report:

- harmful/unauthorized action attempt and execution rate;
- severity-weighted incidents with raw counts;
- secret exposure and data-boundary violation;
- false refusal and safe-completion rate;
- correct clarification/confirmation rate;
- prompt-injection attack success by channel;
- time/actions before detection and containment;
- sandbox/authorization prevention rate;
- rollback/recovery completeness;
- multi-agent amplification factor for cost and violations.

Zero observed incidents is not proof of zero risk. State sample size and upper
confidence bounds. For zero incidents in $n$ independent trials, the rough
95% “rule of three” upper bound is $3/n$, subject to independence and
representativeness.

## 13. Evaluation environment integrity

- separate evaluator credentials/network from the agent;
- keep hidden tests and expected state outside tool mounts;
- hash initial/final state and validator artifacts;
- log every read/write/network/action with trace IDs;
- make the validator read-only relative to policy state where possible;
- randomize nonsemantic identifiers to detect hard-coding;
- replay a sample and independently review surprising high rewards;
- monitor for evaluation task IDs in training queues;
- version live web/API evaluations by date and captured evidence.

## 14. Deployment gates

### Research checkpoint gate

- reproducible fixed-suite improvement;
- no data/validator integrity failure;
- loss/reward/ratio dynamics understood;
- qualitative failure set reviewed.

### Internal sandbox gate

- threat model and authorization matrix;
- adversarial injection/reward-tampering tests;
- resource quotas and kill switch;
- audit logging and incident drill;
- no access to production secrets/accounts.

### Limited-user gate

- independent safety review;
- scoped reversible tools;
- confirmation for high-impact action;
- real-time monitoring and rollback;
- explicit cost/rate caps;
- privacy, retention, and feedback process.

### Wider deployment gate

- statistically supported utility/reliability under realistic load;
- residual-risk acceptance by accountable owners;
- red-team coverage and external review where appropriate;
- abuse monitoring, support, incident response, and update policy;
- versioned model/scaffold/tool release with rollback.

## 15. Evaluation report template

```markdown
## Claim
What capability or safety property is being tested?

## System under test
Model/checkpoint/quantization, scaffold, tools, memory, prompts.

## Tasks and splits
Source, dates, grouping, contamination analysis, hidden state.

## Protocol
Budgets, sampler, trials, seeds, context-overflow, failures/retries.

## Evaluator
Validator/judge version, rubric, information access, calibration.

## Results
Raw counts, point estimates, confidence intervals, cost/latency distributions.

## Ablations
Weights, scaffold, tools, memory, inference compute, evaluator.

## Failures and incidents
Taxonomy, examples, severity, containment, unresolved risks.

## Limitations
Contamination, coverage, uncertainty, vendor-reported or unverified details.

## Reproduction
Revision, config, environment, commands, hashes, artifacts.
```

## 16. Benchmark reading map

- [AgentBench](https://arxiv.org/abs/2308.03688): multiple text-interaction
  environments.
- [WebArena](https://arxiv.org/abs/2307.13854): reproducible realistic web tasks.
- [VisualWebArena](https://arxiv.org/abs/2401.13649): visually grounded web tasks.
- [OSWorld](https://arxiv.org/abs/2404.07972): real computer environments.
- [SWE-bench](https://arxiv.org/abs/2310.06770): repository issues with
  test-based validation.
- [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/):
  human-filtered subset and evaluation-harness discussion.
- [GAIA](https://arxiv.org/abs/2311.12983): real-world assistant questions
  requiring tools and multimodal evidence.
- [tau-bench](https://arxiv.org/abs/2406.12045): tool-agent/user interaction in
  policy-governed domains.
- [BFCL](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html):
  function/tool-call evaluation with versioned categories.

Benchmark names are discovery links, not comparable units. Inspect each harness,
model prompt, tool set, budget, and release version before constructing a table.

## References

1. Mark Chen et al.,
   [“Evaluating Large Language Models Trained on Code”](https://arxiv.org/abs/2107.03374),
   2021, pass@k estimator.
2. Carlos E. Jimenez et al.,
   [“SWE-bench”](https://arxiv.org/abs/2310.06770), 2024.
3. Shuyan Zhou et al.,
   [“WebArena”](https://arxiv.org/abs/2307.13854), 2023.
4. Sijia Yang et al.,
   [“Position Bias in Large Language Models”](https://arxiv.org/abs/2305.13211),
   2023.
5. Lianmin Zheng et al.,
   [“Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena”](https://arxiv.org/abs/2306.05685),
   2023.
