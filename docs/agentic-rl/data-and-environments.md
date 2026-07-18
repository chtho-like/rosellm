# Data and Environments

In **Agentic Reinforcement Learning (Agentic RL)**, the dataset is no longer only a table of prompts and answers. It
is a distribution over initial world states plus a versioned program that turns
actions into future observations. The environment is therefore part of the
training data, reward function, security boundary, and scientific claim.

## 1. Start from a task contract

Before collecting a trajectory, define one immutable task record:

```yaml
task_id: repo_fix/8f2c...
task_family: software_engineering
spec_version: 3
initial_state_uri: snapshot://sha256/...
instruction: "Fix the pagination regression and preserve public behavior."
allowed_tools: [read_file, search, apply_patch, run_tests]
tool_schema_version: 7
max_turns: 40
max_generated_tokens: 24000
wall_time_seconds: 900
network_policy: deny
success_validator: pytest://tests/test_pagination.py
hidden_validator: pytest://hidden/test_regressions.py
cost_budget: 2.50
split: train
provenance: internal_bug_archive_v2
license: internal_research
```

This record answers what the model is trying to do, what it may observe/change,
how success is checked, and which budget is comparable across policies. If any
field changes, increment a version; otherwise a measured improvement can come
from an unnoticed environment change.

## 2. The data layers

Treat the complete corpus as linked layers rather than one shuffled file.

| Layer | Unit | Purpose | Primary leakage risk |
|---|---|---|---|
| Base tasks | initial state + instruction | define the training distribution | evaluation task duplication |
| Demonstrations | task + expert trajectory | bootstrap valid behavior and formats | teacher answers copied into tests |
| Preferences | two or more trajectory segments + label | train preference/reward models | labeler sees hidden test state |
| Process labels | turn/span + judgment | dense credit or verifier training | hindsight/privileged information |
| Online rollouts | exact policy trajectory | estimate policy-gradient objective | missing policy/version metadata |
| Replay | historical trajectories | improve data efficiency | uncontrolled policy staleness |
| Evaluation episodes | hidden tasks and seeds | measure generalization | accidental training ingestion |
| Incidents/red-team | adversarial trajectory + root cause | safety improvement | overfitting to public attacks |

Every derived record should point back to task, environment, policy, sampler,
and parent trajectory IDs.

## 3. Task acquisition

### Human-authored tasks

Domain experts can write tasks with realistic ambiguity and constraints. The
operation needs:

1. an authoring rubric with positive and negative examples;
2. a structured validator or separate expert solution where possible;
3. independent review before the item enters training;
4. provenance, license, author, timestamp, and revision history;
5. difficulty and skill tags assigned without reading model outcomes first;
6. a conflict process for underspecified or impossible tasks.

Do not pay authors solely per accepted item without auditing duplicates and
template variation; that incentive can generate superficially different but
behaviorally identical tasks.

### Naturally occurring tasks

Repositories, support logs, workflows, games, and scientific records offer
realistic distributions. Collection must separate data availability from the
right to train, redistribute, or expose it. Remove secrets and personal data;
retain a non-sensitive manifest so lineage remains auditable.

For software tasks, freeze the repository revision, dependencies, test harness,
and issue text. A floating package registry or external API makes later reset
and replay unreliable.

### Programmatic and synthetic tasks

Generators can vary entities, constraints, state layouts, tool schemas, and
difficulty. A robust generator has:

- a sampled latent specification;
- a renderer that produces the agent-visible instruction/state;
- an independent solver or validator;
- property tests that reject ambiguous instances;
- holdout generator templates and parameter ranges;
- near-duplicate and semantic-equivalence checks.

An LLM task generator without an independent validator can amplify its own
mistakes. Use rejection, human audit, execution, or formal checking rather than
treating model confidence as correctness.

### Self-play and adversarial generation

One policy proposes tasks, attacks, or user behavior; another attempts them.
Self-play can track the learner's capability boundary, but it can also collapse
into a private convention. Periodically anchor the generated distribution to
human-authored tasks and independent validators.

## 4. Split construction and contamination control

Create splits at the highest shared causal unit, not after rendering individual
prompts.

- Same repository, website template, customer account, theorem family, or
  generator seed lineage should not cross splits unless the experiment
  explicitly measures within-family generalization.
- Hash exact normalized content, but also detect fuzzy text, AST/code, graph,
  and state similarity.
- Search base-model pretraining disclosures and public benchmark availability;
  a hidden evaluation harness cannot remove knowledge already learned during
  pretraining.
- Keep validator code and expected outputs inaccessible to policy context.
- Store evaluation IDs in a denylist enforced by every training-data writer.
- Test the final assembled prompt, not only raw source documents, for overlap.

Report contamination checks as evidence with thresholds and false-positive
audits. “Deduplicated” without method, scope, and granularity is not informative.

## 5. Environment contract

A training environment should expose a narrow interface:

```python
class Environment:
    def reset(self, task, seed) -> Observation: ...
    def step(self, action) -> Transition: ...
    def snapshot(self) -> SnapshotRef: ...
    def restore(self, snapshot) -> None: ...
    def validate(self) -> Evaluation: ...
    def close(self) -> None: ...
```

`Transition` should distinguish:

- observation;
- reward components, each with source/version;
- `terminated`: the task reached a true absorbing success/failure condition;
- `truncated`: a collector budget stopped an otherwise continuing episode;
- side-effect summary;
- environment state hash;
- timing and resource usage;
- structured error category.

Gymnasium's terminated/truncated distinction is important because value
bootstrapping differs. An RPC timeout is not necessarily an MDP terminal state.

## 6. Observation design

An observation must contain only information legitimately available to the
deployed policy.

### Text and structured results

- serialize with explicit roles and tool-call IDs;
- cap large results by a documented truncation/summarization rule;
- preserve raw artifacts outside the prompt for audit;
- mark untrusted tool content so it cannot masquerade as system instruction;
- return typed errors instead of free-form stack traces when secrets may leak.

### Pixels, audio, and multimodal state

- record resolution, sampling rate, compression, viewport, coordinate system,
  timestamp, and preprocessing;
- prevent hidden evaluator overlays or inaccessible metadata from entering the
  model input;
- make observation latency and frame skipping part of the environment version.

### Memory and retrieval

Memory reads are observations; memory writes are actions with future
consequences. Log the retrieval query, index version, candidates, scores,
selected chunks, and visibility rules. Otherwise performance changes can be
caused by a silently rebuilt index rather than policy learning.

## 7. Action design

### Text

Plain text may communicate with a user or terminate with an answer. Define how
the environment distinguishes “thinking,” user-visible text, and an action. Do
not rely on undocumented hidden delimiters.

### Structured tool calls

A tool schema includes name, description, typed arguments, constraints,
permission class, idempotency, timeout, and result schema. Validate syntax before
execution and semantics/authorization before side effects.

When constrained decoding is used, record the grammar version and which token
probabilities were masked. Old log-probabilities must correspond to the actual
masked sampling distribution.

### Code and shell actions

Execute inside an isolated, resource-limited sandbox with:

- immutable base image and copy-on-write task state;
- unprivileged user and dropped capabilities;
- CPU, memory, process, disk, GPU, and wall-time quotas;
- deny-by-default network and mounted secrets;
- audited filesystem scopes;
- output size limits and process-tree cleanup;
- a separate trusted validator context.

### GUI and physical actions

Coordinates without the originating screen geometry are uninterpretable. Log
viewport, scale, focus, window state, input event, and resulting screenshot or
state delta. For irreversible physical actions, train in simulation and retain
an external safety controller; policy reward is not sufficient authorization.

## 8. Determinism, state cloning, and replay

Perfect determinism is not always possible, but sources of nondeterminism must
be named:

- task seed and simulator RNG;
- policy sampling RNG;
- tool/service randomness;
- clock, locale, and time zone;
- network content and API version;
- database concurrency;
- GPU nondeterminism;
- retry ordering and asynchronous races.

A replay test should restore a snapshot, apply the recorded semantic actions,
and compare state hashes and observations under an explicit tolerance. If replay
fails, mark the trajectory non-replayable and do not use branching
counterfactual credit as if the state were identical.

## 9. Trajectory record

Preserve events, not only concatenated text. A compact logical schema is:

```json
{
  "trajectory_id": "traj_...",
  "task_id": "task_...",
  "environment": {"name": "repo_env", "version": "3", "seed": 4182},
  "policy": {"checkpoint": "sha256:...", "tokenizer": "sha256:..."},
  "sampler": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 2048},
  "events": [
    {"kind": "observation", "turn": 0, "artifact": "sha256:..."},
    {
      "kind": "policy_action",
      "turn": 0,
      "token_ids": [101, 202, 303],
      "old_logprobs": [-0.2, -1.1, -0.4],
      "text_artifact": "sha256:...",
      "parsed_action": {"tool": "search", "args": {"query": "..."}}
    },
    {"kind": "tool_result", "turn": 0, "artifact": "sha256:..."},
    {"kind": "reward", "turn": 0, "name": "cost", "value": -0.01}
  ],
  "termination": {"terminated": true, "truncated": false, "reason": "success"},
  "evaluation": {"success": 1, "validator_version": "9"}
}
```

Large observations belong in content-addressed artifacts. The trajectory keeps
immutable references and hashes. Personally identifying or secret content needs
access control and retention/deletion policy; content addressing does not
override privacy obligations.

## 10. Bootstrap data stages

### Stage A: format and tool SFT

Train valid role boundaries, structured calls, and basic tool semantics from
high-precision demonstrations. Mask system/user/tool-observation tokens. Test
the exact chat template and parser used in rollout.

### Stage B: behavior cloning on expert trajectories

Include recovery and clarification trajectories, not only successful shortest
paths. Otherwise the initial policy has no support for states reached after its
own mistakes.

### Stage C: rejection sampling or distillation

Sample a stronger teacher or current policy, execute trajectories, retain those
passing independent validators, deduplicate, and SFT. Record teacher, sampling
budget, pass rate, filter reasons, and whether the student sees teacher-only
state.

### Stage D: online RL

Collect fresh policy-dependent trajectories, compute versioned reward, estimate
advantages, update, evaluate, and refresh weights. Preserve failures: they are
needed for relative baselines, critics, reward analysis, and future curricula.

### Stage E: hard-case and recovery curriculum

Mine low-success tasks, long episodes, recurring invalid actions, tool errors,
and safety near misses. Generate controlled variants and reserve some for hidden
evaluation before training on the rest.

## 11. Human data operations

Human feedback is a measurement process. Design it like one.

### Annotator selection and training

1. define required domain/language expertise;
2. obtain informed consent and protect sensitive material;
3. train with gold and ambiguous examples;
4. qualify on held-out calibration items;
5. monitor drift without exposing all gold answers;
6. compensate for task time, including difficult abstentions and review.

### Label interface

Show only information the rubric permits. Randomize candidate order. Preserve
the raw independent labels before adjudication. Offer `tie`, `both bad`,
`unjudgeable`, and safety escalation rather than forcing arbitrary preference.

### Quality measurement

Track agreement by task type and annotator, gold accuracy, response time,
position bias, verbosity preference, and adjudication rate. A single global
agreement number can hide a broken domain.

### Pair selection

Random pairs waste labels when both trajectories are obviously different;
uncertainty or disagreement sampling is more informative. But active sampling
changes the label distribution, so importance weighting or a clear target
distribution is needed when reporting reward-model accuracy.

## 12. AI feedback and learned judges

AI feedback scales critique, ranking, decomposition, and process labels. Record:

- judge model/checkpoint/API date;
- full rubric and prompt template;
- candidate order randomization;
- sampling configuration;
- access to reference answer or hidden state;
- calibration against independent humans/verifiers;
- self-judging relationship to the trained policy;
- prompt-injection and output-parser defenses.

Use ensembles or heterogeneous checks for high-impact labels. Correlated models
can agree and still be wrong; agreement is not correctness.

## 13. Verifier construction

A verifier should be sound with respect to the task specification and hard to
game through the action channel.

- **Math:** canonicalize representations; test equivalence with domain and
  singularity constraints; avoid unsafe arbitrary expression evaluation.
- **Code:** hidden tests, resource limits, nondeterminism controls, mutation and
  property tests; do not expose expected output through logs.
- **Formal proof:** use a trusted kernel and pin library versions.
- **Search/research:** verify citation entailment, source quality, coverage, and
  final answer separately.
- **GUI/workflow:** inspect backend state rather than screenshots or model claims.
- **Safety:** verify both final state and prohibited intermediate actions.

Run adversarial tests against the verifier before using it as reward. The model
will search for validator weaknesses more aggressively as training succeeds.

## 14. Curriculum and sampling

Sampling determines the objective actually optimized. Let \(p_{\text{target}}(q)\)
be the desired task distribution and \(p_{\text{train}}(q)\) the rollout
distribution. Oversampling hard tasks changes the objective unless corrected.

Useful signals include:

- empirical success probability;
- group reward variance (neither all fail nor all succeed);
- uncertainty of value/reward models;
- novelty and underrepresented skills;
- episode cost and expected information per accelerator-second;
- regression and safety priority.

Dynamic sampling that keeps only groups with mixed outcomes improves gradient
density but conditions the training set on policy samples. Log the original
attempt distribution, discarded groups, and acceptance probability so results
can be interpreted.

Curricula should include:

1. syntax and single-tool tasks;
2. short deterministic compositions;
3. recovery from controlled tool failures;
4. longer partial-observation tasks;
5. stochastic users/services;
6. adversarial and safety-constrained tasks;
7. out-of-distribution tools, schemas, and goals.

## 15. Data quality dashboard

At minimum, measure by task family and policy version:

- unique tasks, templates, initial states, and trajectories;
- exact/fuzzy/semantic duplicate rates;
- success, invalid-action, timeout, and environment-error rates;
- reward-component distributions and correlations;
- trajectory turns, policy tokens, observation tokens, wall time, and cost;
- all-equal group fraction and effective trainable-token yield;
- replay success and state-hash mismatch;
- annotator/judge agreement and calibration;
- sandbox/security violations;
- train/evaluation similarity;
- data acceptance/rejection reasons at every filter.

Never report only the retained dataset. Selection rates reveal distribution
shift and how much generation compute was discarded.

## 16. Failure modes

| Failure | Observable symptom | Control |
|---|---|---|
| Reward leakage | agent prints or reads expected answer | isolate validator and audit observations |
| State leakage | hidden metadata appears in prompt | strict observation schema and taint tracking |
| Reset drift | same task produces different starting state | immutable snapshots and replay test |
| Tool mismatch | training calls parse but deployment calls fail | shared schema/parser conformance tests |
| Survivor bias | dataset contains only successful trajectories | retain attempt census and failures |
| Curriculum collapse | all sampled tasks are too easy/hard | track mixed-outcome rate and target mixture |
| Judge exploitation | score rises but human/verified success falls | adversarial calibration and holdout judge |
| Length proxy | longer trajectories receive higher reward | length-stratified analysis and causal ablation |
| Environment overfit | success vanishes after schema/theme changes | held-out tools, layouts, and generators |
| Unsafe exploration | policy finds prohibited side effect | deny-by-default sandbox and hard constraints |

## 17. Release checklist for a dataset/environment

- [ ] Task contract and environment interface are documented.
- [ ] Provenance, license, privacy, and retention decisions are recorded.
- [ ] Train/validation/test grouping and contamination checks are reproducible.
- [ ] Reset, snapshot, replay, and termination semantics have tests.
- [ ] Tool schemas, parser, grammar, and permission rules are versioned.
- [ ] Validator has adversarial and false-positive/false-negative tests.
- [ ] Trajectories retain exact tokens, policy versions, and reward components.
- [ ] Human/AI label rubrics and calibration results are available.
- [ ] Data filters report both accepted and rejected counts.
- [ ] Evaluation state and validators are inaccessible to training workers.
- [ ] Sandbox and incident-response procedures are exercised.

## References

1. Long Ouyang et al.,
   [“Training language models to follow instructions with human feedback”](https://arxiv.org/abs/2203.02155),
   2022, Sections 3 and 4 for demonstration/comparison operations.
2. Reiichiro Nakano et al.,
   [“WebGPT: Browser-assisted question-answering with human feedback”](https://arxiv.org/abs/2112.09332),
   2021, Sections 2–3 for browser environment and human data.
3. Michael Bowling et al.,
   [“Gymnasium: A Standard Interface for Reinforcement Learning Environments”](https://farama.org/Announcing-The-Farama-Foundation),
   and the [terminated/truncated API](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
4. Guibin Zhang et al.,
   [“The Landscape of Agentic Reinforcement Learning for LLMs”](https://arxiv.org/abs/2509.02547),
   TMLR, 2026, environment/resource sections.
5. Tianbao Xie et al.,
   [“OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments”](https://arxiv.org/abs/2404.07972),
   2024.
6. Carlos E. Jimenez et al.,
   [“SWE-bench: Can Language Models Resolve Real-World GitHub Issues?”](https://arxiv.org/abs/2310.06770),
   2024.
