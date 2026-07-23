# Evaluating and Operating Factual LLM Systems

**Verified through:** 2026-07-23.

This chapter is a production runbook. It assumes the failure taxonomy and
method families in the
[hallucination mitigation map](hallucination.md), then defines the contracts,
metrics, traces, and release gates needed to make a factuality claim auditable.

The objective is not “zero hallucinations” as an unmeasured slogan. It is a
versioned risk target for a named distribution:

- which claims the system may make;
- which evidence is authoritative;
- which error severities are acceptable;
- when the system asks, abstains, or escalates;
- what answer coverage, latency, and cost are required; and
- how residual errors are detected and corrected.

## 1. Reference architecture

### Stage 0 — Define the truth and authority contract

Before model selection, record:

| Field | Example decision |
|---|---|
| task | answer employee policy questions, not give legal advice |
| claim types | policy rule, eligibility, effective date, calculation, recommendation |
| authority | signed current policy document outranks wiki and email |
| freshness | effective version at the requested date; ingestion delay below one hour |
| evidence requirement | every policy rule and date needs a source span |
| allowed inference | arithmetic from cited inputs is allowed and labeled |
| conflict rule | show unresolved conflict; do not silently choose |
| answer outcomes | answer, partial answer, clarify, abstain, escalate |
| error severity | wrong eligibility is critical; missing optional background is minor |
| human boundary | benefits specialist approves high-impact or disputed cases |

Without this contract, “grounded” can mean faithful to a document that the
organization does not consider authoritative.

### Stage 1 — Normalize and route the request

Extract or clarify:

- user intent and requested output;
- entity and identifiers;
- time or effective date;
- jurisdiction, organization, product, or repository;
- units and precision;
- factual versus creative mode;
- freshness and external-evidence need;
- security classification and access scope; and
- consequence if the answer is wrong.

Route among parametric answer, supplied-document analysis, internal retrieval,
web research, structured API/database, executable calculation, or human
review. A false-premise and ambiguity detector should run before an answer is
committed.

### Stage 2 — Acquire evidence

For retrieval:

1. enforce access control;
2. apply entity, date, locale, and version filters;
3. use sparse and dense recall;
4. fuse and rerank;
5. expand the necessary parent/neighbor context;
6. deduplicate shared origins;
7. retain provenance and exact content;
8. flag extraction or parser errors; and
9. snapshot volatile external sources when policy permits.

For tools:

1. choose an authoritative typed interface;
2. validate arguments and permissions;
3. execute once or idempotently;
4. preserve result, status, time, and version;
5. normalize units and missing values; and
6. verify the resulting external state when an action occurred.

### Stage 3 — Score evidence sufficiency

Evaluate each requested subclaim, not only the query as a whole.

```text
requested claim
  -> direct supporting evidence?
  -> correct entity, time, unit, jurisdiction, and version?
  -> authoritative and permitted source?
  -> conflicts or missing dependencies?
  -> enough evidence to answer at the allowed risk?
```

Possible transitions are:

- generate from the evidence packet;
- reformulate or decompose the query;
- retrieve another source;
- use a database, calculator, or specialist tool;
- ask the user;
- return a partial answer with explicit gaps;
- abstain; or
- escalate.

Do not use the generator's willingness to answer as the sufficiency test.

### Stage 4 — Construct a bounded evidence packet

The packet should include:

- stable source and chunk identifiers;
- title, owner, URL or record key;
- effective, publication, and retrieval dates;
- authority class;
- exact supporting text or structured value;
- surrounding context needed for qualifiers;
- permitted claim types;
- conflicts and missing fields; and
- untrusted-content markers.

Keep system instructions outside untrusted retrieved content. Context
compression must preserve negation, tables, units, exceptions, dates, and
source boundaries.

### Stage 5 — Generate an auditable draft

Ask for:

- atomic material claims;
- source IDs adjacent to supported claims;
- explicit separation of fact, calculation, inference, and recommendation;
- nullable fields rather than invented values;
- named conflicts and limitations;
- no claim about a tool or external state without its event ID; and
- an output schema compatible with deterministic validation.

A useful internal draft representation is:

```json
{
  "claims": [
    {
      "text": "The policy became effective on 2026-04-01.",
      "kind": "retrieved_fact",
      "materiality": "high",
      "evidence_ids": ["policy-v7-page-2-sentence-4"],
      "effective_date": "2026-04-01",
      "confidence_source": "evidence_and_verifier"
    }
  ],
  "gaps": [],
  "conflicts": [],
  "recommended_outcome": "answer"
}
```

This structure is a trace boundary. It does not make the claim true.

### Stage 6 — Verify independently

Run claim-type-specific checks:

1. source and span exist;
2. cited span entails the claim;
3. every material external claim has evidence;
4. authority, date, entity, jurisdiction, and units match;
5. calculations reproduce from cited inputs;
6. tool claims bind to real events;
7. required caveats and conflicts are present;
8. output schema and business invariants pass; and
9. an independent or human audit is triggered for high-risk cases.

Use at least one audit signal that was not optimized directly during training or
prompt tuning. A generator and verifier from the same family with the same
context may share errors.

### Stage 7 — Apply a decision policy

Do not collapse all verifier scores into a vague confidence number.

| Condition | Outcome |
|---|---|
| all material claims supported and policy passes | answer |
| evidence supports only part of the request | partial answer plus named gaps |
| entity, time, or scope is ambiguous | ask a targeted clarification |
| sources conflict but conflict is decision-relevant | present the conflict or escalate |
| evidence absent, stale, unauthorized, or below threshold | abstain |
| high-impact decision or unresolved critical check | qualified human review |
| tool/action result cannot be verified | report failure; never narrate success |

Error thresholds should be stricter for material claims. A long response with
twenty correct background facts and one wrong dosage, price, deadline, or
permission is not “95% good.”

### Stage 8 — Render provenance honestly

Display:

- citations adjacent to claims;
- source owner, date, and version;
- a short evidence preview;
- the difference between quotation, synthesis, inference, and recommendation;
- partial-search and inaccessible-source notices;
- conflicts and abstentions without euphemism; and
- a way to report a specific claim-level error.

Do not show a probability unless it was calibrated for the current version,
domain, and outcome definition.

### Stage 9 — Trace, monitor, and learn

Persist enough to reproduce a failure without storing prohibited user data:

- request and normalized task class;
- model, prompt, policy, router, retriever, corpus, index, and tool versions;
- queries, retrieved IDs and ranks, source hashes, and sufficiency decisions;
- model sampling settings and complete candidate IDs;
- atomic claims, evidence links, verifier outputs, and final gate;
- answer/ask/abstain/escalate outcome;
- latency and cost by stage;
- user correction and downstream impact; and
- retention, redaction, and access metadata.

Log rejected evidence and failed checks as well as successful paths. Otherwise
the retained trace hides selection bias.

## 2. Threat model

A factual system should be tested against failures in every component.

| Component | Accidental failure | Adversarial failure |
|---|---|---|
| request | ambiguity, typo, missing attachment | false premise or instruction conflict |
| corpus | stale or contradictory record | poisoned document or malicious metadata |
| parser/OCR | lost table cell or negation | layout crafted to hide instructions |
| retriever | semantic near-match, wrong entity | keyword stuffing or retrieval manipulation |
| reranker | popular source outranks authority | prompt injection in candidate text |
| context builder | truncates exception or citation boundary | untrusted text becomes system-like instruction |
| generator | unsupported synthesis or omission | follows injected instructions |
| tool layer | timeout, wrong unit, stale cache | unauthorized call or forged result |
| citation layer | wrong span or incomplete support | citation laundering |
| verifier | judge bias, number/negation error | prompt injection or reward hacking |
| UI | hides source date or conflict | trust badge implies endorsement |
| feedback loop | corrected cases are not learned | attackers steer labels or evaluation |

Prompt-injection protection and factuality interact. A poisoned source can be
relevant and factually phrased while instructing the model to ignore its
evidence policy. Relevance filtering alone is insufficient.

## 3. Evaluation dataset design

### 3.1 Build from the deployment distribution

Stratify by:

- task and claim type;
- domain, language, locale, and jurisdiction;
- entity popularity and long-tail rarity;
- current versus historical date;
- answer length and number of atomic claims;
- number and kind of required sources;
- single-hop versus multi-hop synthesis;
- structured, prose, table, image, audio, video, or code evidence;
- user expertise and ambiguity;
- risk severity; and
- expected outcome: answer, partial, ask, abstain, or escalate.

Preserve a frozen representative set, a recent-traffic set, a targeted
regression set, and a hidden adversarial set. Do not tune prompts or thresholds
against the final audit set.

### 3.2 Include cases that expose guessing

Every suite should contain:

- unanswerable questions;
- nonexistent entities and events;
- false premises;
- insufficient but topically relevant context;
- stale versus current versions;
- conflicting authoritative sources;
- same-name entities;
- negation and exceptions;
- swapped units or populations;
- plausible fabricated citations;
- evidence located near the middle or end of long context;
- poisoned retrieval documents;
- tool timeout, empty result, partial result, and malformed result;
- repeated runs of the same request; and
- questions for which clarification is the only safe response.

If every benchmark question has one short known answer, it trains and measures
guessing rather than real-world selective reliability.

### 3.3 Use benchmark suites for components, not as deployment proof

| Benchmark / method | Surface | Useful role | Important limitation |
|---|---|---|---|
| [TruthfulQA](https://arxiv.org/abs/2109.07958) | misconceptions and imitative falsehoods | tests truthfulness under adversarially selected questions | narrow and public; not current web or grounding |
| [SimpleQA](https://openai.com/index/introducing-simpleqa/) | short-form parametric fact recall | separates correct, incorrect, and not-attempted answers | label noise and short-answer scope |
| [SimpleQA Verified](https://deepmind.google/research/evals/) | curated short-form parametric knowledge | cleaner factual recall instrument | no tools, long synthesis, or citations |
| [FActScore](https://arxiv.org/abs/2305.14251) | long-form atomic factual precision | claim decomposition and support checking | depends on knowledge source and automated judge |
| [LongFact / SAFE](https://deepmind.google/research/publications/85420/) | open-domain long-form factuality | search-augmented claim evaluation | search and evaluator errors remain |
| [FACTS suite](https://deepmind.google/blog/facts-benchmark-suite-systematically-evaluating-the-factuality-of-large-language-models/) | parametric, search, grounding, multimodal | separates four factuality surfaces | leaderboard protocols are not a private application |
| [FACTS Grounding](https://deepmind.google/blog/facts-grounding-a-new-benchmark-for-evaluating-the-factuality-of-large-language-models/) | long-context document grounding | detailed, fully grounded response tests | automated frontier-model judges need audit |
| [ALCE](https://arxiv.org/abs/2305.14627) | correctness and citation quality | citation entailment/completeness evaluation | retrieval corpus and tasks are controlled |
| [RAGTruth](https://arxiv.org/abs/2401.00396) | RAG hallucination spans | detector development and error taxonomy | does not represent every corpus or generator |
| [HaluEval](https://arxiv.org/abs/2305.11747) | generated and human-annotated hallucinations | broad detection experiments | synthetic construction can create artifacts |
| [POPE](https://arxiv.org/abs/2305.10355) and [CHAIR](https://arxiv.org/abs/1809.02156) | visual object hallucination | object existence in VLM answers/captions | not OCR, charts, spatial, or video completeness |

A release needs private, dated, application-specific evaluation even when all
public benchmark results are strong.

## 4. Metrics by stage

### 4.1 Retrieval and evidence

Measure:

- recall at $k$ for all necessary evidence;
- first-support rank and normalized discounted cumulative gain;
- authority-weighted recall;
- entity/date/version match rate;
- evidence sufficiency precision and recall;
- duplicate-origin rate;
- stale or unauthorized retrieval rate;
- parser/OCR/table extraction error; and
- poison/injection retrieval success.

Answer correctness cannot diagnose whether a wrong response came from missing
evidence or failure to use present evidence.

### 4.2 Claim factuality and grounding

Report:

- weighted atomic-claim precision;
- contradiction and unsupported-claim rates;
- material-claim error rate;
- answer completeness against the required claim set;
- correct partial-answer rate;
- false-premise challenge rate;
- correct clarification rate; and
- error severity, not only count.

Use micro and macro averages. A long answer otherwise dominates a short
critical answer, or each response hides how many claims it made.

### 4.3 Citation quality

Measure separately:

- source validity;
- claim–citation entailment precision;
- material-claim citation recall;
- citation placement;
- authority and freshness;
- source independence;
- quote accuracy; and
- unsupported claims that appear near unrelated citations.

### 4.4 Calibration and selective prediction

Track:

- correct, incorrect, abstain, ask, and escalate rates;
- error rate among answered cases;
- coverage at target error thresholds;
- risk–coverage curve and area;
- expected calibration error and Brier score where probabilities are valid;
- accuracy by confidence bin;
- severe-error rate by confidence bin;
- over-refusal on answerable cases; and
- under-refusal on unanswerable or insufficient-evidence cases.

Do not present lower raw accuracy from a cautious system without its much lower
error rate, and do not present higher accuracy from an aggressive guesser
without its abstention and severe-error rates.

### 4.5 Tools and actions

Measure:

- tool-selection and argument correctness;
- schema validity;
- execution success and timeout;
- result-use faithfulness;
- fabricated tool-result rate;
- read-after-write state correctness;
- authorization and policy violations;
- recovery from controlled failure; and
- claims of completion without backend confirmation.

### 4.6 Operational metrics

Reliability controls consume resources. Report:

- p50/p95/p99 latency by stage;
- tokens, searches, tool calls, and verifier calls;
- cost per successful answer and per prevented severe error;
- clarification and human-escalation load;
- cache hit and stale-cache rate;
- retry and loop-depth distributions;
- corpus/index lag; and
- availability when a dependency fails.

## 5. Label and judge protocol

### Human annotation

1. Give reviewers exact sources and claim spans.
2. Separate correctness, grounding, attribution, completeness, and usefulness.
3. Allow `supported`, `contradicted`, `insufficient`, `disputed`,
   `unverifiable`, and `not material`.
4. Blind model identity and randomize answer order.
5. Train with anchor examples and adjudicate high-impact disagreement.
6. Record expertise, language, and source-access limitations.
7. Measure agreement by slice rather than forcing consensus.

### LLM judges

Use structured rubrics, randomize order, freeze the judge version, and include
`both bad` and `unjudgeable`. Validate each judge against qualified humans,
especially for numbers, negation, citations, source conflicts, and adversarial
retrieved text. Use a different model family or multiple judges when
self-preference is plausible.

The verifier should receive only the information required for its check.
Giving it the generator's persuasive rationale can increase correlated error.

### Independent audit

Maintain three signals:

1. training or prompt-optimization reward;
2. validation metric used for selection; and
3. hidden audit metric reviewed only at a gate.

If the same model and rubric produce all three, there is no independent
evidence against reward hacking.

## 6. Experiment matrix and ablations

Compare matched versions under the same request set:

| Factor | Baseline | Treatment |
|---|---|---|
| checkpoint | prior model | candidate model |
| retrieval | none or current | new corpus/retriever/reranker |
| sufficiency | no gate | answerability gate |
| prompt | current | evidence contract |
| tools | unavailable | authoritative typed tool |
| decoding | standard | constrained or candidate selection |
| verification | none/current | claim and citation verifier |
| abstention | forced answer | calibrated selective policy |
| UI | source list | claim-local evidence |
| human review | current | severity-triggered review |

Run factorial subsets where affordable. A bundled improvement cannot establish
whether the checkpoint, search index, longer context, hidden retry, or verifier
caused the gain.

Repeat stochastic requests and report distributions. For research agents,
match search, token, time, and tool budgets. A system that searches ten times
more may be better, but it is a different cost–reliability point.

## 7. Release gates

Set thresholds before reading candidate results.

### Example gate structure

- no regression in critical material-claim error;
- zero unauthorized or fabricated tool-success events in the release suite;
- citation entailment and completeness above domain thresholds;
- unanswerable and insufficient-context error below threshold;
- required coverage at the target selective-risk level;
- no subgroup, language, or long-tail slice below its floor;
- poison/injection suite passes;
- p95 latency, cost, and escalation load within budget;
- source/index freshness service-level objective met;
- human audit confidence interval excludes the unacceptable error rate;
- rollback, incident trace, and prior-version routing are tested.

Do not average away a critical failure with thousands of easy correct cases.
Use hard blockers for authorization, fabricated execution, dangerous domain
claims, and provenance integrity.

## 8. Incident response

When a material false or unsupported answer is found:

1. preserve the request, response, sources, trace, and all component versions;
2. classify the failed reference and severity;
3. determine whether the source was absent, wrong, ignored, transformed, or
   mis-cited;
4. identify affected versions, tenants, languages, and time range;
5. disable or narrow the risky path, source, or claim type when necessary;
6. correct downstream state and notify affected owners under the incident
   policy;
7. add a minimized regression case and neighboring counterexamples;
8. fix the earliest reliable layer rather than only adding a prompt;
9. rerun the complete gate, including unrelated capability regressions; and
10. record why monitoring failed to detect the issue earlier.

Possible fixes include a source correction, parser repair, authority rule,
retrieval filter, tool binding, prompt change, verifier rule, threshold change,
model update, UI change, or human-review boundary. Do not assume every incident
requires model retraining.

## 9. Cost-tiered deployment patterns

### Low-stakes assistant

- strong base model;
- explicit uncertainty and clarification prompt;
- low-variance decoding for factual tasks;
- optional search;
- source display;
- sampled offline factuality evaluation.

Residual risk remains too high for unreviewed consequential decisions.

### Internal knowledge assistant

- permission-aware versioned corpus;
- hybrid retrieval plus reranking;
- sufficiency gate;
- evidence-only draft with claim citations;
- claim/citation verifier;
- calibrated abstention;
- feedback and regression trace.

### Open-web research system

- query planner and iterative search/read loop;
- source authority, diversity, date, and conflict policy;
- snapshots or stable locators;
- atomic-claim search verification;
- citation completeness and entailment;
- budget and loop controls;
- human audit for consequential reports.

### Structured decision support

- authoritative database/API as the truth source;
- typed calculations and business-rule engine;
- model used for intent and explanation, not silent value invention;
- state and authorization validation;
- severity-based abstention/escalation;
- qualified decision owner.

### High-stakes domain

- approved domain corpus and explicit scope;
- stringent material-claim gate;
- out-of-distribution and false-premise detection;
- calibrated abstention with conservative coverage;
- specialist tools and independent checks;
- qualified human decision and source review;
- post-deployment surveillance and incident reporting.

## 10. Clean acceptance checklist

### Scope and evidence

- [ ] Each claim type has a named authority, date rule, and allowed inference.
- [ ] Answer, partial, ask, abstain, and escalate are explicit outcomes.
- [ ] Permissions and untrusted-content boundaries are enforced before model use.
- [ ] Every volatile source retains date, version, and provenance.

### System

- [ ] Retrieval recall and evidence sufficiency are evaluated separately.
- [ ] Tool results are cryptographically or structurally bound to real events.
- [ ] Material claims are atomic and linked to exact evidence.
- [ ] Citation existence, entailment, completeness, quality, and freshness pass.
- [ ] Numerical and executable claims are recomputed or run.
- [ ] Verifier and generator failures are not assumed independent.

### Evaluation

- [ ] The private set represents production tasks and risk slices.
- [ ] Unanswerable, false-premise, insufficient, conflicting, stale, and poisoned cases exist.
- [ ] Repeated-run variance and confidence intervals are reported.
- [ ] Accuracy, error, abstention, coverage, and severity are separate.
- [ ] Human labels and LLM judges are calibrated and versioned.
- [ ] Cost, latency, availability, and human load meet the target.

### Operations

- [ ] Full component versions and traces permit failure reproduction.
- [ ] Monitoring detects corpus/index staleness and material claim errors.
- [ ] Correction, rollback, and incident workflows were exercised.
- [ ] Release gates were fixed before candidate evaluation.
- [ ] Public claims name the evaluated system, date, tools, and limitations.
