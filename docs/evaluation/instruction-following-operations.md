# Evaluating and Operating Instruction-Following Systems

**Verified through:** 2026-07-23.

This chapter treats instruction following as a versioned contract executed by a
model-plus-system, not as a style preference. It defines a reference
architecture, evaluation suite, metrics, release gates, telemetry, and incident
loop for chat assistants, tool-using agents, coding systems, and structured
workflows.

Read the [method map](instruction-following-methods.md) first. Use the
[vendor evidence audit](instruction-following-vendors.md) when deciding whether
a claimed model behavior is disclosed training, a product control, a research
prototype, or unknown.

## 1. Reference architecture

### Stage 0 — Define the instruction contract

Record the following before model selection:

| Field | Example |
|---|---|
| task | update two documentation pages and publish them |
| required outcomes | specified text added; strict documentation build passes |
| forbidden outcomes | no source-code edits; no `research/` files staged |
| order | inspect → edit → validate → review diff → publish |
| scope | exact repository, branch, files, and external systems |
| authority | repository policy outranks user prose; user chooses business intent |
| applicability | publication rules apply only after checks pass |
| completion evidence | test result, commit ID, remote ID, deployed HTTP response |
| ambiguity policy | ask only when a risky choice cannot be safely inferred |
| failure policy | report the failed stage; never narrate unverified completion |
| reversibility | preview first; require confirmation before irreversible effects |
| expiry | contract closes when the accepted outcome and report are delivered |

Separate:

- **linguistic authority:** which message role wins a conflict;
- **domain authority:** who may authorize a payment, deletion, deployment, or
  policy exception;
- **evidence authority:** which source proves task completion; and
- **execution capability:** what the runtime credentials actually permit.

A system message cannot grant a business permission the user or operator does
not have.

### Stage 1 — Parse atomic instructions

Normalize the request into a typed registry:

```json
{
  "contract_version": "if-2026-07-23-001",
  "instructions": [
    {
      "id": "I-01",
      "kind": "required_outcome",
      "text": "Run the strict documentation build.",
      "authority": "repository_policy",
      "scope": ["current repository"],
      "active_from": "task_start",
      "expires": "task_end",
      "criticality": "high",
      "verifier": "docs_render_exit_and_semantic_review"
    }
  ],
  "conflicts": [],
  "ambiguities": [],
  "approvals": []
}
```

Extract:

- action, object, target, and completion condition;
- positive and negative constraints;
- order and dependencies;
- counts, bounds, units, language, style, and format;
- scope and non-interference requirements;
- applicable role, policy, and trust level;
- effective turn, supersession, and expiry;
- required tools, evidence, approvals, and checks; and
- consequence if the rule is violated.

The parser may be an LLM, rules, or both. Verify critical extracted rules
against the original message; a clean registry built from a misread request is
still wrong.

### Stage 2 — Resolve conflicts and feasibility

Construct a precedence and dependency graph.

```text
instruction source priority
  + applicability and domain authority
  + explicit supersession
  + temporal validity
  -> active contract
```

Detect:

- direct contradiction;
- infeasible counts or schemas;
- missing object, version, locale, or authorization;
- a lower-trust instruction embedded in untrusted content;
- mutually exclusive completion conditions;
- an irreversible action without approval;
- a requested check that the environment cannot run; and
- ambiguous wording whose interpretations produce materially different
  outcomes.

Possible outcomes are proceed, choose the unambiguous higher-priority rule, ask
a targeted question, offer feasible alternatives, refuse, or escalate. Do not
silently discard an inconvenient rule.

### Stage 3 — Route the task

Route by required capability and risk:

- literal transformation;
- structured extraction;
- complex natural-language generation;
- long-context or multi-turn revision;
- code or document editing;
- tool/API workflow;
- policy-governed transaction;
- multilingual or multimodal operation; or
- high-impact decision requiring human authority.

Select:

- exact checkpoint and provider;
- reasoning or non-reasoning mode;
- prompt and chat template version;
- grammar or schema decoder;
- available tools and credentials;
- task ledger and memory policy;
- validator set;
- retry and cost budget; and
- human review threshold.

Do not route solely by average model score. A cheap model plus deterministic
schema enforcement may be ideal for extraction, while a complex policy task
needs stronger semantic reasoning and an independent review.

### Stage 4 — Build a constraint-covered plan

Every required instruction should map to a plan step or final-output element.
Every prohibition should map to a guard.

| Contract item | Plan binding | Verification |
|---|---|---|
| inspect before editing | read-only reconnaissance step | trace shows inspection before first write |
| modify only two files | edit allowlist | final diff path set |
| preserve accepted section | patch precondition | semantic diff and regression check |
| run tests | validation step | immutable command/result record |
| do not publish on failure | conditional branch | no external mutation after failed gate |

Reject a plan when it has:

- an uncovered critical requirement;
- an action with no contract justification;
- a forbidden target;
- an invalid order;
- missing rollback or confirmation;
- an unverifiable completion claim; or
- a tool whose schema or authority is unavailable.

### Stage 5 — Execute with least privilege

At each step:

1. rehydrate the active contract and relevant state;
2. expose only the tools and resources needed now;
3. validate arguments, targets, and authorization;
4. record the exact call before or atomically with execution;
5. preserve result, error, time, and environment version;
6. update the task ledger;
7. check whether the next step remains authorized; and
8. stop on a critical violation or ambiguous state.

Use idempotency keys, transactions, previews, and dry runs where the domain
supports them. Never ask a language model to enforce a permission that the
runtime can enforce deterministically.

### Stage 6 — Verify independently

Run three different classes of oracle:

1. **task oracle:** is the substantive outcome correct?
2. **instruction oracle:** did every active rule pass?
3. **action/state oracle:** were only authorized transitions made, and does the
   external state match the claim?

Instruction checks include:

- required content and omissions;
- prohibited content or action;
- order;
- count and length;
- schema and grammar;
- language, style, and audience;
- scope and non-interference;
- multi-turn retention and revision consistency;
- hierarchy and prompt-injection handling;
- correct tool, arguments, result use, and recovery;
- completion evidence; and
- appropriate clarification, refusal, or escalation.

Use deterministic checks for formal properties, executable checks for code and
state, and semantic judges or humans only for properties that cannot be reduced
to rules. Keep at least one audit signal outside the generator's optimization
loop.

### Stage 7 — Repair without regression

Return localized failures:

```json
{
  "failed": [
    {
      "instruction_id": "I-07",
      "type": "forbidden_scope",
      "evidence": "diff contains src/runtime.py",
      "required_repair": "remove out-of-scope change and re-run all checks"
    }
  ]
}
```

Prefer a minimal repair. Then rerun the complete suite, not only the failed
rule: a revision that fixes length may drop required content; repairing a tool
argument may invalidate an earlier approval.

Set bounded retries. Repeated failure can indicate an impossible contract,
weak model, bad schema, broken tool, or validator disagreement. Escalate rather
than looping indefinitely.

### Stage 8 — Decide the outcome

| Condition | Outcome |
|---|---|
| task and every critical instruction pass | accept |
| noncritical requirements remain and the contract allows partial work | partial result with named gaps |
| a material ambiguity remains | ask a targeted clarification |
| a formal failure is repairable within budget | repair and revalidate |
| instruction is lower-priority, malicious, unsafe, or unauthorized | correctly decline that instruction |
| requested outcome is impossible or unavailable | report limitation and alternatives |
| high-impact check or approval is unresolved | human review |
| an external action cannot be verified | report unverified failure, never success |

One critical prohibition should not be averaged away by many easy passes.

### Stage 9 — Render an auditable result

Report:

- completed outcome;
- scope actually changed;
- requirements intentionally not executed and why;
- checks and their result identifiers;
- model, tool, policy, and artifact versions where material;
- unresolved ambiguity or partial failure;
- external-state confirmation; and
- next approval or human decision.

Avoid internal chain-of-thought and private prompt text. The trace should expose
decisions and evidence needed for audit, not hidden reasoning.

### Stage 10 — Trace and learn

Persist, under appropriate privacy controls:

- contract and instruction IDs;
- active, superseded, and expired versions;
- parser output and conflict resolution;
- model, prompt, chat template, reasoning mode, and decoder;
- context construction and compaction version;
- plan-to-constraint mapping;
- tool schemas, calls, results, and state observations;
- candidate outputs and validator results;
- repair attempts and final outcome;
- latency, tokens, tool calls, retries, and cost;
- user correction and downstream impact; and
- retention, access, and redaction metadata.

Record failures and blocked actions, not only accepted answers. A success-only
trace cannot diagnose selection bias.

## 2. Threat model

| Threat | Failure path | Required test or control |
|---|---|---|
| ambiguous request | model chooses an unintended interpretation | ambiguity pairs and clarification gate |
| conflicting constraints | model silently prioritizes a familiar format | satisfiability and conflict tests |
| scope creep | helpfulness reward expands the task | negative constraints and target allowlist |
| forgotten early rule | recency or compaction removes instruction | position-balanced multi-turn tests |
| stale revision | old draft overrides accepted edit | versioned state and no-regression oracle |
| prompt injection | tool or document text becomes a command | typed trust channels and hierarchy suite |
| over-refusal | safety or hierarchy training blocks benign tasks | benign-conflict and ordinary capability slices |
| invalid structured output | free decoding violates syntax | schema/grammar decoding plus validation |
| schema-valid wrong output | syntax passes but semantics fail | task and field-level semantic oracles |
| wrong tool | tool syntax succeeds for an irrelevant API | relevance and no-tool cases |
| phantom tool/state | model narrates a call or completion | immutable execution trace and state readback |
| retry side effect | failed operation is repeated non-idempotently | idempotency and transaction guard |
| judge bias | evaluator rewards length or its own family | deterministic checks, order randomization, human audit |
| reward hacking | model exploits validator edge cases | adversarial/metamorphic validator tests |
| benchmark contamination | public tasks enter training | hidden transformations and live incident suites |
| model alias drift | provider silently changes behavior | immutable IDs, canary and rollback |
| multilingual drift | translated content loses the rule | per-language constraints and locale validators |
| multimodal injection | image/document includes malicious text | content/instruction separation and visual injection tests |

## 3. Evaluation dataset design

### 3.1 Sample the deployment distribution

Start from:

- accepted production tasks;
- explicit user corrections;
- support tickets and failed workflows;
- incident reports;
- policy changes;
- near misses caught by validators or humans;
- model/provider upgrades; and
- synthetic transformations of critical rules.

Stratify by task, domain, language, length, turn count, model route, tool use,
risk, and outcome. A benchmark dominated by simple formatting will not predict
a stateful agent.

### 3.2 Cover orthogonal instruction families

A first serious private suite should include:

| Slice | Representative cases |
|---|---|
| atomic positive | include exact required fields |
| negative | no extra prose, no external contact, no unrelated edit |
| order | approval before action; validation before publication |
| count/length | exact sentences, items, characters, or bounded output |
| format | JSON Schema, table, diff, citation, tool call |
| content | required concepts, exclusions, audience level |
| complex composition | interacting conditions, exceptions, nested scopes |
| ambiguity | entity, date, locale, or intent requiring clarification |
| multi-turn retention | early constraints active after many turns |
| revision | partial supersession without regression |
| hierarchy | system/developer/user/tool conflict |
| injection | malicious content in web, document, image, or tool result |
| tool choice | relevant, irrelevant, parallel, sequential, and no-tool |
| tool recovery | errors, missing arguments, stale state, partial success |
| authorization | forbidden resources and irreversible actions |
| multilingual | same logical constraint across languages and scripts |
| multimodal | text rules bound to regions, frames, speakers, or pages |
| failure reporting | missing files, unavailable tools, failed tests |
| stability | repeated identical or paraphrased trials |

### 3.3 Construct matched perturbations

For each high-risk base case, vary one factor:

- move a rule from early to middle to late context;
- paraphrase without changing logic;
- add an irrelevant distractor;
- negate one clause;
- swap instruction priority;
- insert an indirect injection;
- change language or script;
- make one tool irrelevant;
- introduce a recoverable tool failure;
- change a target identifier; or
- supersede one earlier field.

Metamorphic pairs test whether behavior changes only when the contract changes.

### 3.4 Keep distinct splits

Maintain:

- training;
- prompt/harness development;
- model selection;
- release gate;
- canary;
- incident regression; and
- periodic hidden refresh.

Deduplicate by meaning and transformation lineage, not only exact text. Do not
let the same template family populate training and the only hidden test.

## 4. Metrics

### 4.1 Constraint and prompt success

For task $j$ with constraints $\mathcal C_j$, define
$z_{ji}\in\{0,1\}$ as the verdict for constraint $i$.

Constraint-level accuracy is

$$
A_{\mathrm{constraint}}
=\frac{\sum_j\sum_{i\in\mathcal C_j}z_{ji}}
{\sum_j\lvert\mathcal C_j\rvert}.
$$

All-constraints prompt accuracy is

$$
A_{\mathrm{all}}
=\frac{1}{N}\sum_{j=1}^{N}
\prod_{i\in\mathcal C_j}z_{ji}.
$$

Report both. A response satisfying nine of ten rules contributes 90% to the
first and zero to the second.

### 4.2 Severity-weighted failure

Assign weights by consequence, not evaluation convenience:

$$
R_{\mathrm{instruction}}
=\frac{\sum_j\sum_i w_{ji}(1-z_{ji})}
{\sum_j\sum_i w_{ji}}.
$$

Report critical failure rate separately. Wrong language and an unauthorized
payment should not be interchangeable points.

### 4.3 Task and non-interference

Track:

- substantive task success;
- required-content recall;
- forbidden-content/action rate;
- extraneous action or scope-creep rate;
- unrelated-diff rate;
- exact count/length pass;
- format and schema validity;
- semantic field correctness; and
- unnecessary-refusal rate.

### 4.4 Multi-turn and revision

Measure:

- active-instruction retention by turn distance;
- first violation turn;
- regression after a correction;
- obsolete-instruction resurrection;
- successful local amendment rate;
- state reconstruction after compaction;
- cross-session memory precision and inappropriate-memory use; and
- consistency across repeated conversation branches.

Plot retention versus turns or intervening tokens. One aggregate conceals the
failure horizon.

### 4.5 Hierarchy and injection

Report:

- correct priority resolution;
- prompt-injection attack success rate;
- hidden-instruction extraction rate;
- benign lower-priority usefulness;
- over-refusal on non-conflicting tasks;
- policy violation severity; and
- generalization to unseen attack forms and channels.

Ordinary instruction following and hierarchy must both pass. A model that
obeys every user request is insecure; a model that ignores users is unusable.

### 4.6 Tools and agents

Measure:

- tool relevance and no-tool accuracy;
- tool name and argument correctness;
- schema validity;
- parallel/sequential call ordering;
- execution success;
- faithful use of tool results;
- recovery after controlled error;
- unauthorized tool or resource use;
- fabricated call/result/completion rate;
- final environment-state correctness; and
- claims of success without confirmation.

### 4.7 Repair, stability, and efficiency

Track:

- first-attempt success;
- success after one localized repair;
- regressions introduced by repair;
- attempts to success and retry exhaustion;
- pass variance over repeated trials;
- p50/p95/p99 latency;
- input, output, and reasoning tokens;
- tool and validator calls;
- cost per accepted task;
- cost per prevented critical failure; and
- human-review load.

Do not compare a one-shot model with a hidden best-of-16 or unlimited-retry
system without reporting the budget.

### 4.8 Ambiguity and selective action

Measure the decision to ask separately from question quality and final task
success:

- ambiguity-detection precision and recall;
- ask-when-needed and proceed-when-clear rates;
- silent-guess and unnecessary-question rates;
- whether the question targets the decision-changing variable;
- information gained per user turn;
- task success after the clarification answer;
- repeated or compound-question burden; and
- unsafe action before clarification.

Use matched ambiguous, unambiguous, and disambiguated cases. A system that asks
on every prompt avoids guessing but is not useful; one that always acts can
look efficient while silently choosing the wrong intent. Set the ask threshold
by reversibility, consequence, and the cost of one more turn.

## 5. Oracle and judge protocol

### Deterministic oracles

Use parsers, regular expressions, JSON Schema, static analysis, compilers,
tests, diffs, event logs, and state queries. Version the checker and preserve
its raw verdict.

Test the checker with:

- known pass and fail examples;
- Unicode and whitespace variants;
- semantically equivalent outputs;
- empty and truncated candidates;
- adversarial strings;
- large/deep schemas;
- invalid but reward-seeking outputs; and
- metamorphic transformations.

### Semantic LLM judges

Provide:

- original contract and authority labels;
- candidate output and relevant state;
- one atomic criterion at a time;
- explicit pass/fail/unknown definitions;
- required evidence spans;
- randomized candidate order;
- refusal and API-error handling; and
- no access to vendor identity where possible.

Calibrate on human-labeled cases. Report agreement by slice and retain
`unknown` rather than forcing a judgment.

### Human review

Reviewers need:

- the contract and active instruction registry;
- source artifacts and tool trace;
- a rule-level checklist;
- authority to resolve ambiguity;
- independent double review for critical cases; and
- an adjudication path.

Sample automated passes as well as failures. A judge that approves everything
can otherwise look cheap and stable.

## 6. Experiment and ablation matrix

Change one factor at a time:

1. ordinary versus instruction-augmented from-scratch or continued pretraining,
   followed by the same post-training;
2. base versus instruction-tuned checkpoint;
3. ordinary SFT versus complex/negative/hierarchy data;
4. pretraining/SFT mixture weights;
5. preference or RL stage on/off;
6. rule-based rewards on/off;
7. reasoning mode and effort;
8. raw prompt versus extracted contract;
9. no examples versus fixed, retrieved, or automatically selected
   demonstrations, including order;
10. hand-written versus automatically optimized prompt or pipeline;
11. free decoding versus schema/grammar decoding;
12. one sample versus matched best-of-$N$;
13. self-critique versus independent verifier;
14. no task ledger versus versioned state;
15. full versus compacted context;
16. always-act versus risk-calibrated ask/proceed routing;
17. broad versus least-privilege tool exposure;
18. retry/repair policy;
19. model or route; and
20. harness or serializer version.

Use paired tasks, identical budgets, multiple seeds or trials, confidence
intervals, and failure-slice breakdowns. Attribute a gain to a component only
when a matched ablation supports it.

## 7. Release gates

### Example gate structure

Do not copy universal thresholds. Set them from domain risk.

| Gate | Example policy |
|---|---|
| critical instruction violation | zero in hidden critical set |
| unauthorized external action | zero |
| claim of completion without state proof | zero |
| all-constraints pass | no regression and above product target |
| negative/scope slice | stricter target than general style |
| hierarchy/injection | no regression; named attack ceiling |
| tool relevance and arguments | target by tool-risk class |
| multi-turn retention | target at required production horizon |
| repair regression | below explicit ceiling |
| over-refusal | below domain-specific ceiling |
| latency/cost | within production service objective |
| judge audit | required human agreement floor |

Require sign-off from product, model/evaluation, security, and domain owners
where their boundaries are implicated.

### Canary and rollback

Deploy by immutable model, prompt, harness, and policy versions. Canary on a
small traffic slice; compare first-pass compliance, critical failures, scope
creep, refusals, repair load, cost, and latency. Preserve rapid rollback for
model aliases, prompts, tool schemas, and compaction changes.

## 8. Incident response

When an instruction-following failure causes or nearly causes material impact:

1. contain the affected route, tool, permission, or model alias;
2. preserve prompts, instruction registry, context, calls, results, and state;
3. classify the first failed boundary;
4. assess whether any external action occurred;
5. notify the responsible owner;
6. create a minimal regression and metamorphic variants;
7. fix the narrowest responsible layer;
8. rerun the full critical suite;
9. canary and monitor the repair; and
10. document residual uncertainty and prevention.

Root-cause labels include:

- ambiguous or contradictory contract;
- missing or mislabeled instruction;
- model interpretation;
- SFT/preference/RL regression;
- context truncation or compaction;
- hierarchy or injection failure;
- tool schema/serializer mismatch;
- validator false pass;
- retry or transaction error;
- authorization design;
- model/provider drift; and
- human review failure.

Do not “fix the prompt” when the real fault is an overbroad credential or a
missing state check.

## 9. Training and product feedback loop

For each failure:

1. retain an immutable incident test;
2. create privacy-safe variants;
3. decide whether the fix belongs in data, reward, model, prompt, decoder,
   harness, policy, or authorization;
4. add positive and near-miss negative examples;
5. verify labels and validators;
6. train or configure the narrow fix;
7. test generalization and over-refusal;
8. run regression across unrelated capabilities;
9. measure production canary behavior; and
10. keep unresolved cases as explicit unknowns.

Avoid training directly on every user preference. Some corrections are
idiosyncratic, contradictory, malicious, or product-specific.

## 10. Deployment patterns by risk

### Low-stakes drafting

- clear prompt contract;
- capable instruction-tuned model;
- light format validation;
- user-visible editability; and
- sampled failure review.

### Structured extraction

- typed schema;
- constrained decoding;
- nullable/unknown fields;
- semantic field validation;
- source-span binding; and
- rejection or human review for invalid records.

### Internal knowledge or workflow assistant

- versioned user and organizational instructions;
- instruction/content trust separation;
- retrieval and access controls;
- tool relevance/no-tool checks;
- task ledger across turns;
- rule-level validators; and
- confirmation for material actions.

### Coding or document agent

- repository/document policy discovery;
- explicit allowed-path set;
- plan-to-constraint mapping;
- patch rather than broad rewrite;
- diff, compiler, test, and semantic checks;
- no claim of success without results; and
- explicit publish/commit scope.

### High-impact transactional agent

- formal policy and domain authority;
- least-privilege, short-lived credentials;
- deterministic preconditions and transaction limits;
- independent verification;
- explicit human approval;
- idempotent execution and state readback;
- immutable audit log; and
- rapid route disablement.

## 11. Clean acceptance checklist

### Contract

- [ ] Task, scope, completion, authority, and expiry are explicit.
- [ ] Positive, negative, order, and conditional rules are atomic.
- [ ] Conflicts, ambiguity, and infeasibility have a defined policy.
- [ ] Critical instructions have independent verifiers.

### System

- [ ] Exact model, mode, prompt, template, decoder, and tools are versioned.
- [ ] Trusted instructions are separate from untrusted content.
- [ ] Context compaction preserves active state and supersession.
- [ ] Tool permissions and targets are least privilege.
- [ ] External actions are idempotent or transactionally protected.

### Evaluation

- [ ] Task, constraint, and state oracles are separate.
- [ ] Per-constraint and all-constraints results are both reported.
- [ ] Negative, scope, hierarchy, multi-turn, tool, and failure slices exist.
- [ ] Repeated trials and uncertainty are reported.
- [ ] Judge agreement and validator adversarial tests pass.
- [ ] Cost, latency, repair, and human-review burden are measured.

### Operations

- [ ] Critical release gates and rollback owners are named.
- [ ] Canary telemetry detects scope, hierarchy, tool, and state failures.
- [ ] Incident traces can reproduce the contract and environment.
- [ ] Production corrections feed a private regression suite.
- [ ] Publication or completion is confirmed independently of submission.
