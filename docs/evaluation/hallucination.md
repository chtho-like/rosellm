# LLM Hallucination: Failure Taxonomy and Mitigation Map

**Verified through:** 2026-07-23.

There is no single “hallucination switch.” The practical solution is a layered
system that decides when external evidence is required, acquires and checks that
evidence, constrains what may be asserted, verifies each material claim, and
abstains or escalates when the remaining risk is too high.

The strongest general production pattern is:

```text
task and evidence contract
  -> answerability / freshness / risk routing
  -> retrieval, database, or executable tool
  -> evidence-sufficiency gate
  -> evidence-bound generation
  -> atomic-claim and citation verification
  -> answer, qualify, ask, abstain, or escalate
  -> trace, audit, evaluate, and improve
```

Model training, a better retriever, lower temperature, citations, and
self-checking can each help one part of this path. None proves end-to-end
truthfulness.

## 1. First define the failed reference

The word *hallucination* is often applied to every undesirable generation. That
collapses failures with different causes and fixes.

| Failure class | Operational definition | Example | Correct reference |
|---|---|---|---|
| **Open-world factual error** | a claim conflicts with the best available evidence about the world | a wrong office holder or release date | dated authoritative sources |
| **Unsupported factual claim** | evidence is absent or insufficient even when the claim might happen to be true | a biography detail with no support | source corpus plus evidence policy |
| **Intrinsic unfaithfulness** | output contradicts the supplied context | a summary reverses a study result | the supplied document |
| **Extrinsic addition** | output introduces material not licensed by the context | a meeting summary invents an owner | the supplied record and task contract |
| **Citation fabrication** | the source, title, author, identifier, or URL does not exist | a nonexistent court case | source registry or resolver |
| **Citation misattribution** | the source exists but does not entail the attached claim | a paper is cited for a result it never reports | exact source span |
| **Citation incompleteness** | material claims have no supporting citation | only easy background claims are sourced | required claim set |
| **Reasoning or calculation error** | correct premises lead to a wrong conclusion | arithmetic, unit, or logical mistake | calculator, program, proof, or rubric |
| **Tool hallucination** | a call, result, file state, database row, or test outcome is invented or altered | “all tests passed” without a run | immutable tool trace and resulting state |
| **Temporal error** | a formerly correct claim is stale or its valid date is missing | an old price is presented as current | effective date and freshness policy |
| **Entity or relation error** | attributes from similar entities are blended | two researchers' affiliations are merged | entity-resolved records |
| **Visual or audio grounding error** | an object, attribute, relation, text span, speaker, frame, or time is misperceived | an absent object is described | pixels, regions, frames, transcript, or sensor data |
| **Code hallucination** | an API, symbol, dependency, behavior, or environment assumption is invented | a nonexistent library method | pinned source, compiler, tests, and runtime |
| **False-premise compliance** | the system accepts an invalid premise instead of challenging it | explains why a nonexistent event occurred | premise verification |
| **Overclaim or omission** | wording is stronger than evidence, or a decisive limitation is left out | association is reported as causation | evidence class and coverage rubric |
| **Uncalibrated guess** | the system answers when its evidence or competence is below the allowed threshold | confident answer to an unanswerable question | selective-risk policy |

Not every creative invention is a failure. Fiction, brainstorming, hypothetical
examples, and role-play permit invented content when it is clearly framed and
does not masquerade as fact. Safety refusal, deception, bias, instruction
failure, and low usefulness also deserve separate labels even when they overlap
with factual errors.

### Factuality, faithfulness, and attribution are not interchangeable

- **Factuality** asks whether a claim is true in the world.
- **Faithfulness** asks whether a claim follows from the supplied evidence.
- **Attribution** asks whether the cited source exists, is correctly located,
  and supports the nearby claim.

A response can be factually true but unfaithful to the requested document. It
can be faithful to a false document. It can cite a real, high-quality source
that does not support the sentence. Production evaluation must score all three
when they matter.

## 2. Why unsupported generation occurs

### 2.1 The learning objective rewards plausible continuation

Autoregressive pretraining observes text, not a universal true/false label on
every sentence. Frequent and patterned strings are easier to learn than rare,
arbitrary facts. OpenAI's 2025
[analysis of hallucination incentives](https://openai.com/index/why-language-models-hallucinate/)
adds a second mechanism: accuracy-only evaluation rewards a lucky guess while
an abstention is guaranteed no credit. The model can therefore become a better
test taker without becoming appropriately cautious.

This does **not** mean every generated token is false, or that abstention is the
only solution. It means fluency probability is not a truth certificate and
that evaluation incentives must distinguish correct answers, errors, and
justified abstentions.

### 2.2 Knowledge is incomplete, compressed, conflicting, and time-bound

Training corpora contain mistakes, duplicated claims, unresolved disagreement,
long-tail entities, missing private facts, and facts that change after the
cutoff. Model parameters are not a lossless database with record-level
provenance and transactional updates. Increasing scale improves many averages
but does not turn arbitrary rare facts into deterministic lookup.

### 2.3 Post-training optimizes imperfect proxies

Human and AI raters can prefer confidence, detail, agreement, polished prose,
or longer answers over cautious factuality. A reward model can inherit those
preferences. Optimizing one judge too far can exploit its blind spots. A broad
helpfulness objective may teach the model to complete an answer even when the
correct response is “the evidence is insufficient.”

### 2.4 Context can be missing, noisy, conflicting, or ignored

Retrieval may return no answer, a stale version, a near-duplicate about the
wrong entity, or an adversarial passage. Even when the answer is present, long
or distractor-heavy contexts can cause the generator to overlook it. Google
Research's
[sufficient-context study](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/)
shows why “some retrieved text exists” is not an answerability guarantee:
insufficient context can increase confidence and wrong answers rather than
induce abstention.

### 2.5 Decoding exposes uncertainty and compounds earlier tokens

Sampling can select a low-probability wrong continuation; greedy decoding can
select the same high-probability misconception every time. Once an early entity
or number is wrong, later tokens are conditioned on it and may build a coherent
story around the error. Lowering temperature reduces variation but does not
repair a consistently wrong distribution.

### 2.6 The surrounding system can fail independently

Query rewriting, search, chunking, optical character recognition, database
mapping, reranking, context truncation, tool execution, caching, routing, and
UI citation attachment can each create or conceal an error. A production answer
is evidence about the whole deployed stack, not only the named checkpoint.

### 2.7 Ambiguous, adversarial, or impossible requests induce guessing

An underspecified entity, impossible date, false premise, missing attachment,
conflicting instruction, or unknowable private fact needs clarification or
abstention. If the interface demands an answer and the evaluation never rewards
questions, the system learns the wrong behavior.

## 3. Data, model, and training controls

These methods change what the model stores or how it behaves before a specific
request arrives. They reduce baseline risk but cannot supply fresh, private, or
transactional truth on their own.

| Control | Mechanism | Best use | Residual failure |
|---|---|---|---|
| Source curation and provenance | prefer authoritative, licensed, dated sources and retain origin metadata | improving the factual prior | authority can be wrong; mixture details are usually undisclosed |
| Quality, contradiction, and entity filtering | remove spam, resolve duplicates/entities, flag incompatible claims | reducing noisy supervision | filters introduce coverage and cultural bias |
| Exact and semantic deduplication | limit repeated errors and benchmark leakage | cleaner training/evaluation | deduplication is not fact verification |
| Temporal snapshots and effective dates | preserve when a claim was valid | time-sensitive domains | the checkpoint still becomes stale |
| Domain continued pretraining | increase exposure to specialist language and knowledge | stable domain concepts | can overwrite capabilities; still lacks record-level provenance |
| Retrieval-augmented pretraining | train a model to consult non-parametric memory | knowledge-intensive tasks | retriever and index become part of the failure surface |
| Grounded supervised fine-tuning | imitate answers that quote, cite, use tools, challenge premises, and abstain | teaching the desired response protocol | imitation does not verify unseen claims |
| Unanswerable and negative examples | explicitly train “insufficient evidence,” clarification, and refusal | selective answering | over-refusal and domain shift |
| Human-feedback alignment | reward honest, helpful, evidence-sensitive answers | broad behavioral improvement | rater preference is not an oracle |
| AI-feedback or constitutional training | use written principles and model critiques/preferences at scale | scalable norms such as honesty | judge correlation, principle gaps, reward hacking |
| Factuality preference optimization | rank atomic claims or complete answers with external factuality signals | open-ended truthfulness | verifier errors become training signal |
| Verifiable-reward RL | score math, code, database, search, or proof outcomes with executable checks | domains with hard validators | open-world claims rarely have a perfect verifier |
| Process supervision | reward valid intermediate actions, evidence collection, and checks | multi-step research and tools | correct-looking process can still end incorrectly |
| Rejection sampling / best-of-$N$ distillation | generate several candidates and retain verified or highly scored ones | higher-quality training and serving | selection cost and correlated candidates |
| Knowledge editing | change selected parameterized associations without full retraining | narrow corrections to open weights | locality, paraphrase, multi-hop, and scale remain fragile |
| Multimodal counterfactual training | pair images with hard negatives, false premises, regions, and grounded preferences | object/attribute/relation hallucination | perception and language priors remain entangled |

Classic evidence that post-training can improve truthfulness comes from
[InstructGPT](https://arxiv.org/abs/2203.02155): on its disclosed evaluation,
human-feedback training reduced additions on closed-domain tasks relative to
GPT-3 and improved TruthfulQA. This is a result for specific checkpoints and
data, not proof that generic preference optimization always improves facts.

[Fine-tuning Language Models for Factuality](https://arxiv.org/abs/2311.08401)
demonstrates a research pipeline that constructs factuality preferences with
retrieval or model confidence and optimizes them with Direct Preference
Optimization. [R-Tuning](https://arxiv.org/abs/2311.09677) instead constructs
refusal-aware examples so a model learns to distinguish questions inside and
outside its knowledge. Both illustrate the central design choice: training must
reward the desired answer-versus-abstain policy, not only fluent completion.

### Knowledge editing is a patch, not a live database

[ROME](https://arxiv.org/abs/2202.05262) and
[MEMIT](https://arxiv.org/abs/2210.07229) show that selected factual
associations in open models can be edited. Before using this operationally,
test:

- paraphrase generalization;
- neighboring facts that must remain unchanged;
- multi-hop consequences;
- conflicting and repeated edits;
- cross-language expressions;
- long-form generation rather than one-token completion; and
- rollback and checkpoint lineage.

When facts change frequently or require auditable access control, retrieval from
a versioned source of truth is usually easier to govern than repeated weight
editing.

## 4. Retrieval and grounding controls

[Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401)
combines parametric generation with retrieved non-parametric memory. It improves
updateability and provenance, but “RAG” names a family, not a complete
reliability design.

### 4.1 Build the corpus as an evidence system

1. Define which repositories, databases, websites, and document versions are
   authoritative for each claim type.
2. Preserve source ID, title, owner, effective date, ingestion time, access
   policy, language, and immutable content hash.
3. Parse headings, tables, lists, page numbers, code symbols, and document
   relationships rather than flattening everything into anonymous text.
4. Keep superseded records discoverable but clearly lower their authority for
   current-state questions.
5. Quarantine extraction failures and measure optical character recognition,
   table, and metadata quality.
6. Enforce permissions before retrieval; post-filtering an already exposed
   passage is too late.

### 4.2 Retrieve for recall, then rerank for precision

Useful components include:

- sparse lexical search for names, identifiers, and exact phrases;
- dense retrieval for paraphrase and semantic similarity;
- metadata and date filters;
- entity-aware lookup;
- reciprocal-rank or learned fusion;
- cross-encoder or LLM reranking;
- multi-query retrieval for ambiguous wording;
- query decomposition for multi-hop claims;
- neighboring-section and parent-document expansion; and
- diversity controls so ten near-duplicates do not masquerade as ten sources.

Chunk size is a trade-off. Small chunks improve localization but lose context;
large chunks retain context but add distractors. Anthropic's disclosed
[Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
prepends a short document-specific explanation to each chunk before both dense
and lexical indexing. It is one tested retrieval technique, not evidence about
every Claude production request.

### 4.3 Decide whether the evidence is sufficient

A separate answerability gate should ask:

- Is there a passage that directly addresses the question?
- Are the entity, time, jurisdiction, version, and unit matched?
- Does a multi-part question have evidence for every part?
- Do sources conflict, and if so can the conflict be explained?
- Is the source allowed and authoritative for this claim?
- Is the evidence fresh enough for the requested decision?
- Does the question require an external tool instead of prose retrieval?

The gate can route to another query, a structured database, broader search,
clarification, abstention, or human review. A similarity score alone is not an
answerability probability.

### 4.4 Use adaptive and corrective retrieval

Fixed top-$k$ retrieval wastes context on easy questions and fails silently on
hard ones.

- [Self-RAG](https://arxiv.org/abs/2310.11511) trains a model to decide when to
  retrieve and to emit reflection tokens about relevance and support.
- [Corrective RAG](https://arxiv.org/abs/2401.15884) evaluates retrieval quality,
  triggers different actions, extends weak results with web search, and filters
  documents before generation.
- [ReAct](https://arxiv.org/abs/2210.03629) interleaves reasoning with actions so
  search or a knowledge base can resolve uncertainty during a trajectory.
- Agentic research systems iterate query planning, browsing, reading,
  calculation, contradiction resolution, and backtracking rather than treating
  retrieval as one pre-generation call.

These approaches expand the control surface: loops need tool budgets, stop
conditions, source-quality policy, prompt-injection defense, replay, and
complete traces.

### 4.5 Ground generation at claim granularity

High-value patterns include:

1. extract evidence spans before drafting;
2. organize an evidence packet by requested claim;
3. tell the model which sources are authoritative and which are contextual;
4. require material claims to carry a source identifier;
5. make unsupported fields explicitly nullable;
6. separate sourced facts, calculations, interpretation, and recommendations;
7. preserve source dates and units;
8. generate short atomic statements that can be checked; and
9. verify after generation instead of assuming the prompt was obeyed.

Long context is useful only when the relevant evidence is present and used.
Dumping an entire corpus into a context window is not equivalent to retrieval,
answerability assessment, or verification.

### Why RAG still hallucinates

The retriever can miss the answer; retrieved text can be insufficient, stale,
poisoned, or about the wrong entity; the model can ignore or distort it; and
the citation layer can attach the wrong passage. The manually annotated
[RAGTruth](https://arxiv.org/abs/2401.00396) corpus exists precisely because
retrieval-augmented responses can remain unsupported or contradictory.

## 5. Structured data and executable tools

When a claim has a deterministic source or operation, ask a tool to produce the
value rather than asking the language model to remember or simulate it.

| Claim type | Preferred authority | Required validation |
|---|---|---|
| current account, inventory, or order state | transactional API or database | identity, authorization, timestamp, row/version |
| arithmetic, unit, or date calculation | typed calculator or code | input units, precision, overflow, executable result |
| market, weather, schedule, or other live data | versioned provider API | source time, market/location, missing-data handling |
| code behavior | checked-out source, compiler, linter, test, runtime | revision, command, exit status, logs, environment |
| theorem or symbolic claim | proof assistant, solver, or domain checker | formal statement, assumptions, kernel result |
| workflow completion | backend state and audit event | object ID, before/after state, idempotency |
| policy conformance | deterministic rules plus reviewed classifier | policy version, matched rule, false-positive audit |

The system must bind every narrated result to a real tool event. Store a
tool-call ID, arguments, tool version, result hash, time, status, and resulting
state. Do not allow a free-text model message to impersonate a tool result.
Schema validation prevents malformed calls; it does not make a valid argument
factually correct or an authorized action safe.

## 6. Prompt and context controls

Prompting is cheap and useful, especially when it defines a verifiable
contract. It is not a hard guarantee.

### Strong prompt patterns

- State the allowed evidence boundary: supplied documents only, approved
  databases, or named public sources.
- Instruct the system to challenge false premises and ask when the entity,
  time, jurisdiction, version, or desired standard is ambiguous.
- Define the allowed outcomes: answer, qualify, ask, abstain, or escalate.
- Require `not found`, `conflicting`, or `insufficient evidence` instead of
  filling a missing field.
- Separate quotations/evidence from synthesis and label inference explicitly.
- Require dates, units, source IDs, and exact citation placement.
- Use a fixed output schema with nullable fields and an evidence array.
- Provide examples of correct abstention, source conflict, and partial answer,
  not only successful complete answers.
- Treat retrieved pages, emails, documents, and tool output as untrusted data
  that cannot replace higher-priority instructions.

### Weak prompt patterns when used alone

- “Do not hallucinate.”
- “Be 100% accurate.”
- “Think step by step.”
- “Double-check your answer.”
- “Always provide citations.”
- “Answer as an expert.”

These statements do not supply missing evidence or an independent verifier.
They can alter style and caution, but must be measured on the target
distribution.

## 7. Decoding and candidate-selection controls

### Temperature and deterministic decoding

Lower temperature or greedy decoding reduces sample variance and is appropriate
for many extraction and analytical tasks. It cannot fix a high-probability
falsehood, stale knowledge, bad context, or a wrong tool result. Some hosted
systems may remain nondeterministic because routing, kernels, mixtures, or
server-side processing change.

### Constrained decoding

A grammar, finite-state machine, typed tool schema, or enumerated choice can
make invalid output strings impossible. Use it for:

- JSON/XML structure;
- identifiers and enums;
- dates, units, and bounded numeric formats;
- valid function-call syntax;
- extractive spans or allowed labels; and
- workflows whose next action comes from a finite authorized set.

It guarantees membership in the allowed language, not semantic correctness.
`{"case_id":"A123"}` can be perfectly valid JSON and still identify the wrong
case.

### Sampling, consensus, and ranking

- **Self-consistency** samples several reasoning paths and selects a consensus;
  it is effective when independent paths tend to converge on a checkable answer.
- **Best-of-$N$** uses a reward model or verifier to select among full
  candidates.
- **Ensembles** combine different model families, prompts, or retrievers.
- **Debate/critique** exposes disagreements for a verifier or human.

These methods trade compute for reliability only when the selector is better
than the candidates on the target error. Correlated models can repeat the same
misconception, and majority vote can confidently select it.

### Contrastive factuality decoding

Research methods modify token scores without updating weights:

- [Context-Aware Decoding](https://arxiv.org/abs/2305.14739) contrasts output
  probabilities with and without the supplied context to emphasize evidence.
- [DoLa](https://arxiv.org/abs/2309.03883) contrasts later and earlier
  Transformer-layer logits to surface factual knowledge.
- Google's 2025
  [SLED](https://research.google/blog/making-llms-more-accurate-by-using-all-of-their-layers/)
  aggregates layer information during decoding.

These are promising checkpoint-level controls with model- and task-dependent
costs. They do not provide freshness, source provenance, or a general
production truth guarantee.

## 8. Verification and post-generation repair

### 8.1 Decompose before judging

Split a response into atomic, independently checkable claims. Preserve:

- the exact answer span;
- claim type, entity, time, unit, and modality;
- attached citations and evidence spans;
- whether it is quotation, calculation, inference, recommendation, or
  uncertainty statement; and
- materiality, because a wrong caveat may matter more than ten correct
  background facts.

[FActScore](https://arxiv.org/abs/2305.14251) operationalizes long-form factual
precision as the fraction of atomic claims supported by a knowledge source.
[LongFact and SAFE](https://deepmind.google/research/publications/85420/) use a
search-augmented evaluator to decompose and check long-form claims. Automated
decomposition and search are themselves fallible, so retain evidence and audit
a stratified human sample.

### 8.2 Match the verifier to the claim

| Claim | Verification method | Important caveat |
|---|---|---|
| supported by supplied text | natural-language inference plus exact span | entailment models fail on numbers, negation, and long context |
| open-world factual claim | search and source-resolution agent | search ranking and source quality are not truth |
| citation | existence, location, entailment, completeness, quality | a real URL is insufficient |
| number or date | typed recomputation from cited inputs | source units and effective dates must match |
| code | compile, execute, hidden/property/mutation tests | weak tests can certify wrong code |
| database or workflow state | backend query and invariant check | read-after-write and eventual consistency matter |
| visual claim | region/frame evidence, OCR, specialist detector | detectors and OCR also hallucinate or miss |
| medical/legal/scientific interpretation | domain rubric and qualified human review | authoritative sources may disagree or change |

### 8.3 Verify citations along multiple axes

[ALCE](https://arxiv.org/abs/2305.14627) separates answer quality from citation
quality. A production citation audit should measure at least:

1. **validity:** the source and location exist;
2. **entailment / precision:** the cited span supports the nearby claim;
3. **completeness / recall:** every material externally verifiable claim has
   support;
4. **placement:** the citation is attached to the right clause;
5. **source quality:** the source is appropriate and authoritative;
6. **diversity / independence:** multiple citations are not copies of one
   unsupported origin;
7. **freshness:** the source is valid for the claimed date; and
8. **faithful transformation:** units, qualifiers, populations, and causal
   wording survive synthesis.

### 8.4 Revise only from verified evidence

- [RARR](https://arxiv.org/abs/2210.08726) searches for attribution and
  minimally revises unsupported output.
- [Chain-of-Verification](https://arxiv.org/abs/2309.11495) drafts, creates
  verification questions, answers them independently, and produces a revised
  response.
- Program repair can use compiler/test feedback; research agents can reopen
  sources and repair only failed claims.

The verifier, evidence, and revision policy must be independent enough to add
information. A generic “reflect again” prompt can preserve or amplify the same
error. The study
[Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)
found that intrinsic self-correction without external feedback can degrade
reasoning. Other work finds gains under particular prompts and temperatures.
The correct production conclusion is to test self-revision as a component, not
assume it is an oracle.

## 9. Uncertainty, calibration, and abstention

The system should optimize a risk–coverage frontier:

- **coverage:** the fraction of requests it answers;
- **selective risk:** the error rate among answered requests;
- **abstention quality:** whether withheld cases are actually the dangerous
  ones; and
- **utility:** the asymmetric cost of a wrong answer, missing answer, delay, and
  human escalation.

Useful uncertainty signals include:

| Signal | What it observes | Limitation |
|---|---|---|
| token log probability / margin | local decoder confidence | length/tokenization sensitive; fluent falsehoods can be high probability |
| verbal confidence | model's stated uncertainty | wording can be uncalibrated or persuasive |
| $P(\mathrm{True})$ / $P(\mathrm{IK})$ | self-evaluation of an answer or whether it knows | task transfer and prompt calibration vary |
| sample disagreement | answer instability across repeated draws | misses consistent misconceptions |
| semantic entropy | uncertainty over meanings rather than wordings | requires multiple samples and semantic clustering |
| hidden-state probe | internal representation correlated with correctness | needs model access, labels, and distribution-specific validation |
| retrieval sufficiency | whether evidence covers the question | evaluator can confuse relevance with entailment |
| verifier or ensemble score | independent evidence/checker result | correlated errors and judge drift |
| conformal calibration | threshold calibrated on exchangeable held-out data | guarantee depends on score, labels, and distribution assumptions |

[SelfCheckGPT](https://arxiv.org/abs/2303.08896) uses disagreement among
black-box samples. The Nature study on
[semantic entropy](https://www.nature.com/articles/s41586-024-07421-0)
clusters semantically equivalent generations before measuring uncertainty, so
paraphrases do not falsely appear diverse. Both target confabulation-like
instability, not every consistently wrong claim.

[Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221)
studies $P(\mathrm{True})$ and $P(\mathrm{IK})$ self-evaluation, while
[conformal abstention](https://arxiv.org/abs/2405.01563) calibrates an
abstention rule with formal error-control goals under stated assumptions.
Natural-language hedges should still be audited: Google's
[MetaFaith](https://research.google/pubs/metafaith-faithful-natural-language-uncertainty-expression-in-llms/)
reports that ordinary prompts often fail to make verbal uncertainty faithfully
track intrinsic uncertainty.

### Five valid outcomes

A reliable system does not force every request into “answer”:

1. **answer** with adequate evidence;
2. **qualify** a partially supported or disputed answer;
3. **ask** for missing identity, scope, source, time, or attachment;
4. **abstain** when expected error exceeds the domain threshold; or
5. **escalate** to a qualified human or controlled workflow.

Thresholds must be calibrated per domain and severity. A creative suggestion, a
customer account balance, and a drug interaction do not share an acceptable
false-answer cost.

## 10. Product and human-interface controls

The interface shapes reliance as much as the model.

- Show sources beside the exact supported claim, not as an undifferentiated
  footer.
- Expose a short supporting passage, source owner, date, and version.
- Distinguish retrieved fact, model inference, calculation, and recommendation.
- Label incomplete searches, inaccessible sources, and conflicts.
- Avoid decorative confidence percentages that were not empirically
  calibrated.
- Make “not enough evidence” and clarification normal outcomes rather than
  product failures.
- Provide a correction path that captures the claim, evidence, model/system
  version, and user impact.
- Require human sign-off for high-stakes decisions and define what the reviewer
  must inspect; a generic “human in the loop” label is not a control.
- Preserve original source access so the user can independently verify.
- Do not let a citation badge imply endorsement or truth.

## 11. Multimodal, code, and agent-specific extensions

### Multimodal systems

Evaluate object existence, attributes, counts, spatial relations, optical
character recognition, chart values, document layout, speaker identity,
temporal order, and coordinate grounding separately. Useful controls include:

- higher-resolution crops or region tools;
- optical character recognition with bounding boxes and confidence;
- counterfactual image-question pairs and false-premise training;
- region/frame citations;
- specialist detectors for count or geometry;
- cross-frame identity tracking; and
- a text-only ablation to detect answers supplied by language priors rather
  than the image.

Google's
[HALVA](https://research.google/blog/halva-hallucination-attenuated-language-and-vision-assistant/)
is one disclosed contrastive-training approach for attenuating visual
hallucination. A lower object-hallucination score does not establish accurate
OCR, charts, video, or world knowledge.

### Code systems

Ground the request in the exact repository and dependency versions; search
symbols before inventing APIs; use constrained edits; compile; run targeted and
regression tests; inspect the diff; and bind every claim about success to a real
command result. Hidden tests, property tests, fuzzing, static analysis, and
mutation testing reduce the chance that weak visible tests reward plausible but
wrong code.

### Agents

Agents add action and state hallucinations. Preserve:

- every tool schema and version;
- exact arguments and returned observations;
- permission and authorization decisions;
- retries, errors, and rejected actions;
- state identifiers before and after mutation;
- policy, prompt, retriever, and model versions; and
- a final backend-state verification independent of the agent's narration.

An agent that says “done” is not evidence that the external state changed.

## 12. What commonly fails when used as a silver bullet

| Claim | Why it is false or incomplete |
|---|---|
| “Use a larger model.” | averages may improve, but rare facts, stale knowledge, false premises, and system errors remain |
| “Set temperature to zero.” | variance falls; consistent falsehoods remain |
| “Add a long system prompt.” | instructions neither provide missing evidence nor enforce compliance |
| “Use RAG.” | retrieval can be irrelevant or insufficient, and generation can ignore it |
| “Put the whole corpus in a long context.” | presence, discovery, attention, conflict resolution, and answerability are distinct |
| “Require citations.” | models can fabricate, misplace, or launder citations |
| “Use valid JSON / Structured Outputs.” | syntax is guaranteed, not values |
| “Ask the model to critique itself.” | the same knowledge and bias may reproduce or worsen the error |
| “Use several agents or majority vote.” | correlated systems can agree on the same misconception |
| “Fine-tune the documents into the model.” | provenance and freshness become harder, and updates require retraining |
| “Use a knowledge graph.” | extraction, entity linking, schema, query, and graph freshness can fail |
| “Let an LLM judge grounding.” | judges have position, verbosity, self-preference, injection, and factuality errors |
| “Add a disclaimer.” | warning text does not reduce the underlying error rate |
| “A human reviews it.” | review fails without source access, a checklist, time, authority, and sampled quality audits |

## 13. A practical method-selection rule

Choose the authority before choosing the model.

| Need | Primary control stack |
|---|---|
| stable general explanation | strong model + targeted factuality evaluation + optional verification |
| internal-document question answering | permissioned hybrid RAG + sufficiency gate + claim citations + abstention |
| current open-web research | iterative search/read agent + source hierarchy + dates + citation entailment + human audit |
| current structured value | authoritative API/SQL + typed calculation + state/time validation |
| math or formal reasoning | executable verifier or proof kernel + candidate search + checked final answer |
| code change | repository retrieval + compiler/tests + diff/state verification |
| medical, legal, financial, or safety-critical support | curated domain sources + calibrated abstention + qualified human decision maker |
| creative generation | clear fictional frame; factual verification only for claims presented as real |

The companion
[production operations chapter](hallucination-operations.md) turns this map
into an executable architecture and evaluation gate. The
[vendor evidence chapter](hallucination-vendors.md) shows which parts leading
labs have actually disclosed.

## Primary-source index

1. Lewis et al.,
   [“Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”](https://arxiv.org/abs/2005.11401),
   2020.
2. Lin, Hilton, and Evans,
   [“TruthfulQA”](https://arxiv.org/abs/2109.07958), 2021/2022.
3. Nakano et al., [“WebGPT”](https://arxiv.org/abs/2112.09332), 2021.
4. Ouyang et al.,
   [“Training Language Models to Follow Instructions with Human Feedback”](https://arxiv.org/abs/2203.02155),
   2022.
5. Kadavath et al.,
   [“Language Models (Mostly) Know What They Know”](https://arxiv.org/abs/2207.05221),
   2022.
6. Gao et al.,
   [“RARR: Researching and Revising What Language Models Say”](https://arxiv.org/abs/2210.08726),
   2022.
7. Yao et al., [“ReAct”](https://arxiv.org/abs/2210.03629), 2022/2023.
8. Manakul, Liusie, and Gales,
   [“SelfCheckGPT”](https://arxiv.org/abs/2303.08896), 2023.
9. Min et al., [“FActScore”](https://arxiv.org/abs/2305.14251), 2023.
10. Gao et al.,
    [“Enabling Large Language Models to Generate Text with Citations”](https://arxiv.org/abs/2305.14627),
    2023.
11. Shi et al.,
    [“Trusting Your Evidence: Hallucinate Less with Context-Aware Decoding”](https://arxiv.org/abs/2305.14739),
    2023.
12. Chuang et al., [“DoLa”](https://arxiv.org/abs/2309.03883), 2023.
13. Dhuliawala et al.,
    [“Chain-of-Verification Reduces Hallucination”](https://arxiv.org/abs/2309.11495),
    2023.
14. Asai et al., [“Self-RAG”](https://arxiv.org/abs/2310.11511), 2023.
15. Huang et al.,
    [“Large Language Models Cannot Self-Correct Reasoning Yet”](https://arxiv.org/abs/2310.01798),
    2023.
16. Tian et al.,
    [“Fine-tuning Language Models for Factuality”](https://arxiv.org/abs/2311.08401),
    2023.
17. Zhang et al., [“R-Tuning”](https://arxiv.org/abs/2311.09677), 2023.
18. Niu et al., [“RAGTruth”](https://arxiv.org/abs/2401.00396), 2024.
19. Yan et al., [“Corrective RAG”](https://arxiv.org/abs/2401.15884), 2024.
20. Wei et al.,
    [“Long-form Factuality in Large Language Models”](https://deepmind.google/research/publications/85420/),
    2024.
21. Farquhar et al.,
    [“Detecting Hallucinations in Large Language Models Using Semantic Entropy”](https://www.nature.com/articles/s41586-024-07421-0),
    2024.
22. Yadkori et al.,
    [“Mitigating LLM Hallucinations via Conformal Abstention”](https://arxiv.org/abs/2405.01563),
    2024.
23. Kalai et al.,
    [“Why Language Models Hallucinate”](https://openai.com/index/why-language-models-hallucinate/),
    2025.
24. Google DeepMind,
    [FACTS Benchmark Suite](https://deepmind.google/blog/facts-benchmark-suite-systematically-evaluating-the-factuality-of-large-language-models/),
    2025.
