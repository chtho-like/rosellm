# Improving LLM Instruction Following: Method and System Map

**Verified through:** 2026-07-23.

Instruction following is not a single latent skill and there is no universal
“obedience” loss. A reliable system must identify the authorized instruction,
retain it across the relevant horizon, resolve conflicts, execute the requested
task, avoid unrequested behavior, and prove that each material constraint was
satisfied.

The strongest general pattern is:

```text
task and authority contract
  -> instruction extraction and conflict resolution
  -> capability, risk, and feasibility routing
  -> plan with explicit constraint coverage
  -> model, tool, and environment execution
  -> deterministic and semantic verification
  -> accept, repair, clarify, refuse, or escalate
  -> trace failures into data, rewards, evaluations, and product controls
```

This chapter catalogs the model-training, inference, decoding, agent, product,
and operational controls that can improve that path. The companion
[vendor evidence audit](instruction-following-vendors.md) asks which controls
leading developers have actually disclosed. The
[operations chapter](instruction-following-operations.md) turns the map into an
evaluation and deployment contract.

## 1. Define the behavior before selecting a method

At least the following surfaces require separate labels and tests.

| Surface | Desired behavior | Typical failure |
|---|---|---|
| **Task interpretation** | infer the requested operation, object, and completion condition | answers a nearby question |
| **Positive constraints** | include every requested item | omits a required field or topic |
| **Negative constraints** | avoid every prohibited item or action | edits an out-of-scope file |
| **Ordering** | perform or present steps in the required sequence | acts before approval or validation |
| **Cardinality and length** | produce exactly the requested count or bound | returns six bullets instead of five |
| **Format and grammar** | satisfy syntax, schema, sections, and delimiters | invalid JSON or extra prose |
| **Style and audience** | follow tone, language, reading level, and terminology | defaults to a preferred house style |
| **Scope and non-interference** | change only authorized targets and preserve accepted state | performs an unrelated cleanup |
| **Literalness versus intent** | infer omitted details only where the contract permits | “helpfully” rewrites a literal translation |
| **Multi-turn retention** | retain still-active instructions and accepted edits | forgets a turn-2 rule at turn 20 |
| **Revision consistency** | incorporate new changes without regressing earlier ones | reintroduces text that the user removed |
| **Instruction hierarchy** | obey the highest-authority applicable instruction | follows malicious text from a tool result |
| **Tool selection and arguments** | call the right tool, with permitted and valid arguments | calls a nonexistent or irrelevant tool |
| **Tool and policy following** | respect state, authorization, preconditions, and domain rules | completes the task by an unauthorized route |
| **Error recovery** | react correctly to missing data, tool errors, and failed checks | repeats a failed call or claims success |
| **Multilingual transfer** | preserve constraints across languages and scripts | translates the content but drops a format rule |
| **Multimodal grounding** | apply text rules to the correct region, frame, or audio span | edits or describes the wrong object |
| **Stability** | repeat compliant behavior under the production sampling regime | passes once and fails on later identical trials |
| **Correct non-compliance** | reject lower-priority, unsafe, impossible, or unauthorized requests | obeys everything indiscriminately |

Task correctness and instruction adherence are orthogonal. An answer may be
substantively correct but violate the requested format. A perfectly formatted
answer may solve the wrong task. A tool call may be syntactically valid but
unauthorized. Report each axis.

## 2. Why models fail to follow instructions

### 2.1 Pretraining predicts text rather than obeying a contract

Autoregressive pretraining rewards plausible continuation. It does not supply a
privileged representation of “the current binding rule,” a transaction
boundary, or an all-constraints acceptance test. Instruction tuning must teach
these behaviors, while the surrounding system must enforce the parts that
should never depend on token probabilities.

### 2.2 Training mixtures contain competing behavioral priors

Chat, essays, code, role-play, safety examples, terse extraction, detailed
explanations, and tool trajectories reward different styles. A preference
model may favor helpfulness, length, confidence, or polish over literal
compliance. Safety data can cause over-refusal; helpfulness data can cause scope
expansion.

### 2.3 Constraints compose nonlinearly

A model that follows each rule separately may fail when rules interact:

- “exactly three bullets” conflicts with four required topics;
- a JSON schema conflicts with “include a prose explanation”;
- “do not edit tests” interacts with “make all tests pass”;
- a later correction supersedes one field but not the whole earlier plan; and
- a tool result contains text that looks like a new instruction.

The conjunction is the task. Average per-rule accuracy can hide a response that
is unusable because one critical prohibition failed.

Even under the simplifying assumption that $m$ equally difficult constraints
are independent and each passes with probability $p$, the whole prompt passes
with probability only

$$
p_{\mathrm{all}}=p^m.
$$

At $p=0.95$, ten constraints yield only about $59.9\%$ all-constraints
reliability. Real rules are not independent, but this calculation explains why
apparently strong atomic performance can collapse on compositional prompts.

### 2.4 Attention and context management lose governing state

Long conversations contain obsolete drafts, examples, quoted text, tool
results, and repeated constraints. Recency, middle-position weakness,
truncation, retrieval, and lossy compaction can hide the still-binding rule.
The model cannot follow an instruction that the harness removed or mislabeled.

### 2.5 The model and serving stack may disagree

Chat templates, role tokens, tool serializers, stop strings, constrained
decoders, context windows, reasoning modes, and provider-specific parsers
change behavior. Fine-tuning a model on one tool-call syntax and serving it with
another can look like a reasoning failure even when the mismatch is entirely
in the harness.

### 2.6 Evaluation and rewards use imperfect proxies

An LLM judge may prefer verbosity or share the candidate's bias. A format
checker can miss semantic omissions. A final-state reward can reinforce
invalid intermediate actions when the result happens to be correct. Public
prompts can leak into training. Optimizing one saturated benchmark can make the
score rise without improving the production distribution.

## 3. Instruction-data design

### 3.1 Build a broad task mixture

Early instruction tuning established that breadth matters. T0's
[multitask prompted training](https://arxiv.org/abs/2110.08207) reformulated
supervised datasets with natural-language prompts. The
[FLAN scaling study](https://arxiv.org/abs/2210.11416) varied task count, model
scale, and chain-of-thought data; its 1,800-task mixture improved unseen-task
generalization across several model families.

A production mixture should cover:

- transformation, extraction, classification, generation, reasoning, and
  decision-support tasks;
- short and long answers;
- positive, negative, ordered, conditional, and exception-bearing rules;
- exact counts, schemas, tables, code, and natural-language formats;
- clarification, refusal, partial completion, and failure reporting;
- single-turn, multi-turn, revision, and stateful tool trajectories;
- multiple languages, scripts, domains, and modalities; and
- benign role conflicts plus adversarial prompt injection.

Balance task families deliberately. Raw volume from a common conversational
style can erase rarer but important literal, terse, or non-action behaviors.

### 3.2 Use human demonstrations for the target contract

[InstructGPT](https://arxiv.org/abs/2203.02155) used labeler-written
demonstrations before preference modeling and reinforcement learning. Human
examples remain valuable when the behavior depends on organizational policy,
subtle scope, domain-specific authorization, or a quality bar that a synthetic
teacher does not reliably represent.

Annotation records should preserve:

- atomic instruction and authority labels;
- required and forbidden outputs or actions;
- applicability conditions and exceptions;
- acceptable clarifications or refusals;
- exact tool and state transitions;
- why near-miss candidates fail; and
- disagreement rather than forcing a false consensus.

### 3.3 Generate synthetic instructions, then verify them

[Self-Instruct](https://arxiv.org/abs/2212.10560) bootstraps tasks and
demonstrations from a small seed set. [WizardLM's
Evol-Instruct](https://arxiv.org/abs/2304.12244) repeatedly rewrites seed
instructions to increase breadth, depth, constraints, and reasoning demands.
[Instruction
Backtranslation](https://arxiv.org/abs/2308.06259) runs the process in the
opposite direction: a seed model writes instructions for human-authored web
documents, then self-curation filters instruction–document pairs before
fine-tuning. The three approaches expand different parts of the distribution:
new tasks, harder constraints, and higher-quality target text.
Synthetic generation makes scale and targeted coverage affordable, but it can
amplify a teacher's style, errors, and preference for easily judged tasks.

Use a generate–filter–verify loop:

1. sample a seed by under-covered surface;
2. transform one controlled dimension at a time;
3. reject infeasible or contradictory instructions unless conflict handling is
   the intended lesson;
4. generate multiple candidate demonstrations;
5. run deterministic validators where possible;
6. use a separate semantic judge and sampled human audit;
7. deduplicate instructions, answers, and underlying templates; and
8. reserve transformation families, not only exact strings, for evaluation.

### 3.4 Train explicitly on complex constraint composition

[Conifer](https://arxiv.org/abs/2404.02823) constructs multi-level constrained
instructions with model-driven refinement, an easy-to-hard curriculum, and
process feedback. [FollowBench](https://arxiv.org/abs/2310.20410) and
[ComplexBench](https://arxiv.org/abs/2407.03978) provide useful taxonomies:
content, situation, style, format, examples, dependencies, and compositions.

Useful training transformations include:

- add one constraint while preserving all earlier constraints;
- negate an existing constraint;
- introduce an exception or conditional branch;
- require an exact count, order, or cross-field relationship;
- make two individually valid rules interact;
- perturb names, numbers, language, and surface form without changing logic;
- create a near-miss answer that violates exactly one rule; and
- ask the model to identify an infeasible conjunction before generating.

Do not train only on ever-longer lists. Real complexity includes dependencies,
scope, precedence, state change, and exceptions.

### 3.5 Include negative and contrastive examples

A dataset containing only ideal answers does not expose the decision boundary.
Include pairs that differ by:

- one missing required item;
- one forbidden addition;
- wrong order;
- correct format but wrong task;
- correct task but invalid schema;
- unauthorized tool use;
- obsolete versus current instruction;
- lower-priority text that should be ignored;
- excessive explanation when only an identifier is allowed; and
- over-literal execution when clarification is required.

Contrastive data is useful for reward models, Direct Preference Optimization
(DPO), rejection sampling, and explicit error classifiers.

### 3.6 Collect on-policy failures

Static synthetic data misses how the current model actually fails. Periodically
sample the deployed or candidate policy on real or representative tasks,
cluster failures by violated instruction, and add the hardest informative
cases to training and evaluation.

Keep train, development, release-gate, and incident-regression sets separate.
Do not immediately train on the only copy of a severe production failure; keep
an immutable holdout variant that proves the fix generalizes.

### 3.7 Preserve role, turn, and tool structure

Training serialization should retain system, developer, user, assistant, and
tool roles; tool call IDs; tool results; refusal/clarification turns; and
termination state. Flattening everything into undifferentiated text teaches the
model that a quoted webpage or tool result has the same authority as the
developer contract.

The chat template, tokenizer, loss mask, and production serializer must agree.
Test exact token sequences for every role and tool transition.

### 3.8 Put instruction structure into pretraining when scale permits

Most practical projects start from an existing base model and add SFT, but
instruction-shaped data can enter earlier. [Instruction
Pre-Training](https://arxiv.org/abs/2406.14491) uses a synthesizer to augment
raw corpora with instruction–response pairs at pretraining scale; its
experiments use 200 million pairs over more than 40 task categories in both
from-scratch and continued-pretraining settings.

This can make task structure less of a thin late-stage behavior, but it raises
the same risks at much larger scale:

- synthetic-task contamination and teacher-style monoculture;
- loss of raw language-modeling, knowledge, or domain quality;
- catastrophic forgetting during continued pretraining;
- mismatch between pretraining serialization and production roles; and
- far more expensive correction when a behavioral prior is wrong.

Mix raw, instruction-shaped, and domain data deliberately; use conservative
learning-rate schedules for continued pretraining; preserve base and
instruction anchor suites; and still apply post-training for authority,
preference, safety, and application-specific behavior.

## 4. Model-training controls

### 4.1 Supervised instruction tuning

For a tokenized conversation $x_{1:T}$, a common response-only objective is

$$
\mathcal L_{\mathrm{SFT}}(\theta)
=-\sum_{t=1}^{T}m_t\log p_\theta(x_t\mid x_{<t}),
$$

where $m_t=1$ only for policy-authored target tokens. Instructions, user
content, and tool observations remain context but normally do not receive
assistant-token imitation loss.

Important design choices include:

- full response versus selected-turn loss;
- packing without role-boundary corruption;
- task and language sampling weights;
- maximum length and truncation policy;
- oversampling rare prohibitions and hierarchy conflicts;
- training exact chat/tool templates used at inference;
- mixing terse and explanatory targets; and
- checkpoints before narrow tuning erases general capabilities.

SFT teaches an imitation prior. It does not guarantee compliance outside the
demonstration distribution.

### 4.2 Progressive curricula

Move from atomic constraints to combinations, dependencies, hierarchy
conflicts, multi-turn retention, tools, and noisy environments. Difficulty can
be estimated from:

- constraint count and dependency depth;
- current all-constraints pass rate;
- disagreement among validators;
- number of turns or state transitions;
- adversarial-content strength; and
- whether the model can recover after a controlled failure.

Continually remove universally trivial samples, but retain a stable anchor set
so advanced training does not regress basic format, language, or refusal
behavior.

### 4.3 Preference optimization and human or AI feedback

For each prompt, collect candidates that expose distinct compliance errors,
then rank by task success, rule adherence, scope, safety, and usefulness.
Possible optimizers include reward-model plus PPO, DPO-family objectives,
rejection-sampling fine-tuning, and reinforcement learning from AI feedback
(RLAIF).

A preference rubric must state whether:

- one critical prohibition outweighs several minor successes;
- unnecessary content is a failure;
- a clarification beats a speculative completion;
- correct non-compliance beats unsafe obedience;
- format and semantics are scored separately; and
- response length is normalized.

Generic “which answer is better?” preferences often reward prose quality
instead of literal adherence.

[Constitutional AI](https://arxiv.org/abs/2212.08073) makes principles explicit:
the model first critiques and revises responses under a written constitution,
SFT learns from the revisions, and AI comparisons train a preference model for
RLAIF. This is useful when many rules can be stated but human pairwise labels
are scarce. The constitution, critique model, and preference model can still
encode omissions or systematic bias, so human audit and behavioral tests
remain necessary.

### 4.4 Rule-based and verifiable rewards

Many constraints are mechanically checkable:

- exact count, prefix, suffix, language, or delimiter;
- JSON Schema, regular expression, or grammar;
- required and forbidden phrases;
- file-diff scope;
- tool name and argument schema;
- ordered event log;
- unit tests, static analysis, or database invariants; and
- final environment state.

Let $z_i(y)\in\{0,1\}$ indicate whether response or trajectory $y$ satisfies
constraint $i$. A weighted reward might use

$$
r(y)=r_{\mathrm{task}}(y)
+\sum_i w_i z_i(y)
-\sum_j \lambda_j v_j(y),
$$

where $v_j$ represents prohibited actions, scope expansion, or policy
violations. A hard-conjunction gate can additionally require

$$
z_{\mathrm{all}}(y)=\prod_{i\in\mathcal C_{\mathrm{critical}}}z_i(y).
$$

The first signal supplies dense learning; the second prevents a model from
trading one catastrophic failure for many easy successes.

Validators can be wrong. Audit parser edge cases, Unicode, equivalent
expressions, gaming, and rewards that accidentally favor empty or truncated
outputs.

[IF-RLVR](https://arxiv.org/abs/2507.02833) provides a concrete generalization
test: train reinforcement learning with verifiable rewards on 29
hand-annotated constraint types, then evaluate 58 different out-of-domain
constraints. Its central warning is that high scores on a small familiar
constraint set can be benchmark overfitting rather than a transferable
instruction-following skill. Hold out whole verifier and transformation
families, not only prompt strings.

### 4.5 Process supervision and intermediate state

Outcome-only reward can approve an invalid process that reaches the right
answer by chance. For agents and complex edits, supervise or verify:

- extraction of active instructions;
- conflict and feasibility analysis;
- plan-to-constraint coverage;
- tool selection and authorization;
- state observations and completion checks; and
- final output.

Process supervision costs more and can overconstrain legitimate strategies.
Use hard checks for security and state transitions; allow multiple valid
reasoning paths where the process need not be unique.

### 4.6 Instruction-hierarchy training

OpenAI's [Instruction Hierarchy
paper](https://arxiv.org/abs/2404.13208) generates examples in which lower-trust
text conflicts with higher-trust instructions and trains the model to ignore
the conflict selectively. The reported GPT-3.5 experiment used SFT and
reinforcement learning from human feedback and improved both trained and
held-out attacks with small ordinary-capability costs.

Hierarchy training should include:

- system/developer/user/tool conflict;
- quoted, translated, encoded, or role-played attacks;
- indirect injection in web pages, files, images, and tool results;
- extraction attempts against hidden instructions;
- benign lower-priority content that remains useful;
- absent conflict, to measure over-refusal; and
- domain authorization separate from linguistic role priority.

Role labels alone are not a guarantee. The
[Control Illusion](https://arxiv.org/abs/2502.15851) evaluation finds that
models can follow favored constraint types even when those constraints have
lower declared priority. Training and release tests need semantic conflicts,
not only message-role formatting.

Prompt-injection defenses make the data-versus-instruction distinction more
explicit. [StruQ](https://arxiv.org/abs/2402.06363) pairs a secure front end
that separates prompt and data channels with fine-tuning examples that teach
the model to ignore instructions inside the data channel.
[SecAlign](https://arxiv.org/abs/2410.05451) instead creates injected inputs
with secure and insecure responses, then uses preference optimization to favor
the legitimate instruction. Both are research systems, not proofs that role
tokens alone make arbitrary production agents secure.

### 4.7 Multi-turn and revision training

Train conversations in which instructions:

- persist unchanged;
- expire after one operation;
- are explicitly replaced;
- are locally amended while other clauses remain;
- conflict with an obsolete draft;
- refer to accepted state created many turns earlier; and
- survive tool calls, errors, and context compression.

Use turn-position balancing so governing rules appear near the beginning,
middle, and end. [Multi-IF](https://arxiv.org/abs/2410.15553) demonstrates why
single-turn accuracy is insufficient: all tested systems degraded across its
three turns, with additional multilingual weakness.

Meta's Llama 2 report describes **Ghost Attention**: synthesize dialogues in
which an instruction is present in every turn, train with that full context,
then drop the repeated instruction from intermediate user messages so the
model learns to retain the original constraint. It is an instructive
multi-turn data technique, although the report also shows that the effect
weakens over longer conversations.

### 4.8 Tool- and function-calling training

High-quality tool data must teach more than JSON syntax:

1. whether a tool is needed;
2. which available tool is relevant;
3. when no tool is applicable;
4. valid and authorized arguments;
5. parallel versus sequential calls;
6. use of real tool results;
7. recovery from errors and missing fields;
8. state verification; and
9. a faithful final response.

[APIGen](https://arxiv.org/abs/2406.18518) synthesizes data over 3,673
executable APIs and filters it through format checks, actual execution, and
semantic verification. [ToolACE](https://arxiv.org/abs/2409.00920) expands the
API pool and uses multi-agent synthesis plus rule- and model-based validation.
These pipelines illustrate a general principle: executable interfaces permit
stronger data verification than unconstrained chat.

[Toolformer](https://arxiv.org/abs/2302.04761) shows self-supervised insertion
and filtering of API calls by language-model loss, while
[Gorilla](https://arxiv.org/abs/2305.15334) fine-tunes retrieval-aware API
calling and measures hallucinated tools. Those methods address tool choice and
call formation; authorization, transaction safety, and faithful use of the
returned state still require separate training and system controls.

Include irrelevant-tool and no-tool cases. Otherwise a model can maximize
training success by calling something on every request.

### 4.9 Multilingual and multimodal instruction tuning

Translate meanings, not only strings. Preserve:

- counts and structural constraints;
- formal versus informal address;
- locale-specific units and dates;
- script and punctuation rules;
- code-switching and cross-language references; and
- the difference between instructions and content to transform.

For multimodal systems, attach constraints to stable image regions, frames,
timestamps, document coordinates, or audio speakers. Include counterfactual
rules about absent objects and visually embedded prompt injections.

### 4.10 Specialist training, merging, and distillation

A developer can train specialists for precise constraints, tools, coding,
writing, safety, and other domains, then merge them or distill them into a
unified policy. This improves targeted coverage but introduces:

- conflicting teacher preferences;
- regression during parameter merging;
- domain-router errors;
- loss of rare behaviors in distillation; and
- difficulty attributing a final gain to one specialist.

Use a stable cross-domain anchor suite and report every specialist's
contribution through matched ablations.

### 4.11 Parameter-efficient adaptation

Adapters, low-rank updates, prefix or soft-prompt tuning, and other
parameter-efficient methods can create instruction specialists without
updating every base parameter. They reduce training and storage cost, not the
need for representative data, reward design, and regression testing.

## 5. Inference-time controls

### 5.1 Write an executable instruction contract

Prompts work best when they make the task decidable:

- identify the object, operation, scope, and completion condition;
- number atomic positive and negative constraints;
- mark priority and applicability;
- give exact schemas, allowed values, and examples;
- distinguish content to process from instructions to obey;
- state whether clarification or refusal is allowed;
- specify tool and authorization boundaries; and
- avoid hidden contradictions.

Longer prompts are not automatically better. Repeated, prose-heavy rules create
more opportunities for conflict and omission.

### 5.2 Extract and normalize constraints before acting

For complex tasks, first construct an internal ledger such as:

```json
{
  "task": "update documentation",
  "required": ["add one section", "run the documentation build"],
  "forbidden": ["modify source code", "stage research/"],
  "order": ["inspect", "edit", "validate", "publish"],
  "scope": ["docs/", "mkdocs.yml"],
  "completion": ["strict build passes", "remote commit matches"],
  "ambiguities": []
}
```

The ledger is a planning aid, not a correctness guarantee. Validate its
extraction against the original request and keep security policy outside
model-editable state.

### 5.3 Decompose, plan, and bind steps to constraints

Map every planned action to at least one requirement and every prohibition to
a guard. Detect:

- uncovered requirements;
- actions without authorization;
- impossible combinations;
- missing information;
- destructive or irreversible steps; and
- points that require user or human approval.

Reasoning or planning can improve multi-step completion, but excessive
deliberation can reinterpret a simple literal request. Route by task complexity
instead of forcing one reasoning mode everywhere.

### 5.4 Use candidate search and external selection

Generate multiple candidates when the validator is reliable and the added cost
is justified. Select with:

- deterministic constraint checkers;
- task-specific executable tests;
- pairwise or listwise reward models;
- independent LLM judges with randomized order; or
- human review for high-impact ambiguity.

Best-of-$N$ helps only if the candidates vary and the selector recognizes the
failure. Correlated candidates plus a biased judge can make confidence rise
without increasing compliance.

### 5.5 Critique and repair with localized feedback

Verify each constraint, then repair only failed portions. A useful loop is:

```text
draft
  -> rule-by-rule verdict with evidence
  -> list exact failures and affected spans/actions
  -> minimal revision
  -> rerun all checks, including earlier passes
```

Do not ask vaguely “Did you follow the instructions?” The same model can
rubber-stamp its answer. Supply objective failures such as a JSON Pointer,
missing section, unauthorized diff, wrong event order, or failed test.

### 5.6 Constrained decoding

At each decoding step, a grammar engine can mask tokens that would make the
prefix impossible under a regular expression, finite-state machine, context-free
grammar, or JSON Schema. This can guarantee supported syntactic constraints.
[Grammar-Constrained Decoding](https://aclanthology.org/2023.emnlp-main.674/)
demonstrates the approach without task-specific fine-tuning.

Use it for:

- JSON, XML, SQL subsets, domain-specific languages, and tool calls;
- enumerations and finite identifiers;
- exact structural wrappers; and
- extraction whose values can be constrained to source spans.

It does not prove:

- semantic correctness;
- correct tool selection;
- authorization;
- factual field values;
- completeness; or
- that the schema itself represents a feasible task.

Strict grammar can also reduce reasoning freedom or force a model to populate a
bad schema. Use a two-stage plan-then-format design when semantic reasoning and
strict output syntax compete.

### 5.7 Deterministic validators and policy engines

Move non-negotiable checks out of the model:

- schema and type validation;
- allowlists and capability tokens;
- file/path and database-row scope;
- diff-size and forbidden-pattern gates;
- order and state-machine transitions;
- unit tests and business invariants;
- rate, cost, and time limits; and
- human approval for irreversible actions.

The model may propose. The policy engine decides whether the action is
authorized.

### 5.8 Persistent state and context management

Store accepted instructions in a versioned task ledger separate from ordinary
dialogue. On compaction:

1. retain authoritative instructions verbatim where feasible;
2. mark superseded and expired clauses;
3. preserve accepted outputs and unresolved decisions;
4. keep tool state and identifiers;
5. hash or version the summary;
6. test reconstruction against the original constraints; and
7. retrieve the relevant ledger entry before each action.

Do not rely on semantic vector search alone for governing rules. Exact IDs,
priority, scope, and version matter.

### 5.9 Model, mode, and route selection

Different checkpoints or modes may specialize in:

- literal low-latency transformation;
- complex constraint composition;
- long-context retention;
- tool use;
- coding;
- multilingual work; or
- safety and policy.

Route high-risk or low-confidence cases to a stronger model, external verifier,
or human. Record the exact model and mode; a changing alias is not an auditable
deployment.

### 5.10 Select demonstrations and optimize the whole harness

Few-shot examples define behavior by showing it. Select demonstrations by task,
constraint type, language, difficulty, and failure mode; include both compliant
and explicitly labeled near misses where the model can interpret the label.
Order and lexical similarity can dominate the intended principle, so validate
retrieval and ordering as versioned system components.

Automatic methods can search the prompt and pipeline against a development
metric. [Automatic Prompt Engineer](https://arxiv.org/abs/2211.01910) generates
and selects instruction candidates; [Optimization by
PROmpting](https://arxiv.org/abs/2309.03409) iteratively proposes prompts from
earlier candidates and scores; and [DSPy](https://arxiv.org/abs/2310.03714)
compiles declarative model pipelines by selecting demonstrations and other
parameters.

Optimize on a representative development set, then freeze and evaluate on
unseen transformation families. Otherwise prompt search simply overfits a
benchmark, leaks hidden cases, or exploits a biased judge.

### 5.11 Detect ambiguity and learn when to clarify

Instruction following sometimes requires asking before acting. Train and test
matched triples:

- an unambiguous request that should be executed without friction;
- an ambiguous request whose plausible interpretations materially differ; and
- the same request after a concise disambiguating answer.

[CLAMBER](https://arxiv.org/abs/2405.12063) separates ambiguity identification
from clarifying-question quality and finds that generic chain-of-thought and
few-shot prompting offer only marginal help on its distribution.
[Future-turn modeling](https://arxiv.org/abs/2410.13788) scores a clarification
using the later conversation it enables, then trains a policy to ask only when
useful.

Optimize task utility after clarification, not raw question frequency. Penalize
silent guessing, irrelevant or compound questions, questions that request
already supplied information, and over-clarification of harmless details.
Irreversible actions need a lower threshold for asking than reversible drafts.

### 5.12 White-box activation steering

Research systems with access to model internals can add learned activation
directions during inference. [Instruction-specific activation
steering](https://arxiv.org/abs/2410.12877) computes contrasts between inputs
with and without a constraint, then applies the resulting vectors to formats,
length, and word-inclusion behavior. [Contrastive Activation
Addition](https://arxiv.org/abs/2312.06681) demonstrates a broader related
recipe on Llama 2 residual streams.

These methods can alter behavior without a full fine-tune, but are less
auditable than an external policy rule, can entangle unrelated features,
require matched no-steering baselines plus layer and strength sweeps, and do
not create a hard instruction or authorization guarantee.

## 6. Product and agent controls

### 6.1 Separate authority from content

Use explicit typed channels for system policy, developer configuration, user
request, retrieved content, tool result, and assistant output. Mark untrusted
data and prevent it from silently becoming an instruction.

### 6.2 Make scope and authorization concrete

Expose only necessary tools and resources. Prefer:

- least-privilege credentials;
- read-only defaults;
- explicit change sets;
- dry runs and previews;
- idempotency keys;
- transaction boundaries;
- confirmation for material external effects; and
- read-after-write verification.

Instruction-following quality cannot compensate for overbroad authority.

### 6.3 Keep validators outside the conversational loop

A deterministic checker should receive the candidate artifact, original
contract, and immutable environment observation. Do not let the model rewrite
the test or silently omit failed checks.

### 6.4 Render adherence and uncertainty honestly

For consequential workflows, expose:

- what was requested and completed;
- what was intentionally not changed;
- checks actually run and their result IDs;
- unresolved ambiguities;
- partial or failed steps;
- model/tool versions; and
- approvals still required.

Do not claim success from an intended action, a syntactically valid call, or a
submitted request alone.

## 7. What commonly fails as a silver bullet

| Claim | Why it is incomplete |
|---|---|
| “Use a larger model.” | averages improve, but hierarchy, exact counts, scope, and state still fail |
| “Write a longer system prompt.” | length increases conflict and forgetting; prose is not enforcement |
| “Put IMPORTANT in all caps.” | salience is not authority or a release gate |
| “Give one example.” | the model may imitate surface form and miss the governing rule |
| “Ask for chain of thought.” | reasoning can help planning but also over-interpret literal tasks |
| “Set temperature to zero.” | variability falls; deterministic non-compliance remains |
| “Use JSON mode.” | valid JSON is weaker than schema conformance and semantic correctness |
| “Use Structured Outputs.” | syntax can be guaranteed while values, task choice, and authorization are wrong |
| “Make the model self-check.” | shared blind spots produce confident rubber-stamping |
| “Use multiple agents.” | they can share the same mistaken interpretation and expand scope |
| “Fine-tune on more conversations.” | an imbalanced mixture can strengthen unwanted defaults |
| “Score average constraints.” | one critical prohibition can disappear inside a high mean |
| “Retry until it passes.” | unbounded retries hide cost, bias evaluation, and can repeat external actions |
| “The tool call succeeded.” | execution success does not prove the right tool, arguments, result use, or final state |
| “A human is in the loop.” | review fails without the contract, evidence, authority, time, and an audited checklist |

## 8. Method-selection guide

| Requirement | Primary control stack |
|---|---|
| exact short format | targeted SFT + grammar/schema decoding + deterministic validation |
| complex natural-language constraints | compositional data + preference/RL + constraint extraction + per-rule verifier |
| minimal or no extra behavior | negative examples + scope reward + diff/action allowlist |
| long multi-turn revision | turn-position training + versioned task ledger + regression checks |
| system/developer priority | hierarchy-conflict training + typed trust channels + injection tests |
| function calling | executable verified trajectories + no-tool cases + strict arguments + state validation |
| coding changes | repository-scoped plan + diff gate + compiler/tests + state verification |
| multilingual instructions | multilingual constraint data + locale-aware validators + language-slice release gates |
| multimodal editing or agents | grounded region/action data + counterfactual negatives + tool/state checks |
| ambiguous consequential request | ambiguity training + ask/proceed router + concise question + post-clarification task check |
| recurring prompt distribution | representative dev set + automatic prompt/demonstration search + hidden-family evaluation |
| irreversible high-impact action | least privilege + independent verification + explicit human authorization |

The most reliable design combines learned generalization with deterministic
enforcement. Train the model to understand and plan under instructions; use
software to enforce the constraints that can be formalized; use human judgment
where the contract is consequential and irreducibly ambiguous.

## Primary-source index

- Sanh et al., [T0 / Multitask Prompted
  Training](https://arxiv.org/abs/2110.08207), 2021.
- Chung et al., [Scaling Instruction-Finetuned Language
  Models](https://arxiv.org/abs/2210.11416), 2022.
- Ouyang et al.,
  [InstructGPT](https://arxiv.org/abs/2203.02155), 2022.
- Wang et al.,
  [Self-Instruct](https://arxiv.org/abs/2212.10560), 2022.
- Xu et al.,
  [WizardLM / Evol-Instruct](https://arxiv.org/abs/2304.12244), 2023.
- Li et al., [Instruction
  Backtranslation](https://arxiv.org/abs/2308.06259), 2023.
- Cheng et al., [Instruction
  Pre-Training](https://arxiv.org/abs/2406.14491), 2024.
- Sun et al., [Conifer](https://arxiv.org/abs/2404.02823), 2024.
- Bai et al., [Constitutional AI](https://arxiv.org/abs/2212.08073), 2022.
- Rafailov et al., [Direct Preference
  Optimization](https://arxiv.org/abs/2305.18290), 2023.
- Wallace et al., [The Instruction
  Hierarchy](https://arxiv.org/abs/2404.13208), 2024.
- Chen et al., [StruQ](https://arxiv.org/abs/2402.06363), 2024.
- Chen et al., [SecAlign](https://arxiv.org/abs/2410.05451), 2024.
- Geng et al., [Control
  Illusion](https://arxiv.org/abs/2502.15851), 2025.
- Touvron et al., [Llama 2 and Ghost
  Attention](https://arxiv.org/abs/2307.09288), 2023.
- Geng et al., [Grammar-Constrained
  Decoding](https://aclanthology.org/2023.emnlp-main.674/), 2023.
- Schick et al., [Toolformer](https://arxiv.org/abs/2302.04761), 2023.
- Patil et al., [Gorilla](https://arxiv.org/abs/2305.15334), 2023.
- Rimsky et al., [Contrastive Activation
  Addition](https://arxiv.org/abs/2312.06681), 2023.
- Stolfo et al., [Instruction-specific activation
  steering](https://arxiv.org/abs/2410.12877), 2024.
- Zhou et al., [Automatic Prompt
  Engineer](https://arxiv.org/abs/2211.01910), 2022.
- Yang et al., [Optimization by
  PROmpting](https://arxiv.org/abs/2309.03409), 2023.
- Khattab et al., [DSPy](https://arxiv.org/abs/2310.03714), 2023.
- Zhang et al., [CLAMBER](https://arxiv.org/abs/2405.12063), 2024.
- Zhang et al., [Future-turn clarification
  training](https://arxiv.org/abs/2410.13788), 2024.
- Liu et al., [APIGen](https://arxiv.org/abs/2406.18518), 2024.
- Liu et al., [ToolACE](https://arxiv.org/abs/2409.00920), 2024.
- Zhou et al., [IFEval](https://arxiv.org/abs/2311.07911), 2023.
- Jiang et al., [FollowBench](https://arxiv.org/abs/2310.20410), 2023.
- Qin et al., [InFoBench](https://arxiv.org/abs/2401.03601), 2024.
- Wen et al., [ComplexBench](https://arxiv.org/abs/2407.03978), 2024.
- He et al., [Multi-IF](https://arxiv.org/abs/2410.15553), 2024.
- Pyatkin et al., [IFBench](https://arxiv.org/abs/2507.02833), 2025.
- Scale AI, [MultiChallenge](https://labs.scale.com/papers/multichallenge),
  2025.
- UC Berkeley, [Berkeley Function-Calling Leaderboard
  V4](https://gorilla.cs.berkeley.edu/leaderboard), verified 2026-07-23.
