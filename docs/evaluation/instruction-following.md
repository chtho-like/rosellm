# Instruction Following, Steerability, and Production Reliability

**Verified through:** 2026-07-23.

Instruction following is not one scalar capability. A model can satisfy a word
count while forgetting a rule from five turns earlier; call the correct tool
while violating a business policy; return valid JSON whose values are wrong;
or correctly ignore a user instruction because it conflicts with a higher
priority developer rule. This chapter separates those cases and explains why a
model can rank highly on an instruction-following leaderboard yet still feel
unreliable in a real product.

Use this topic in dependency order:

1. This page defines the behavioral surfaces, explains end-to-end attribution,
   maps benchmarks, and preserves a dated GPT-versus-DeepSeek comparison.
2. The [instruction-following method map](instruction-following-methods.md)
   catalogs data, SFT, preference/RL, hierarchy, tool, inference, decoder,
   memory, verification, and product controls.
3. The [vendor evidence audit](instruction-following-vendors.md) separates
   named training disclosures from research prototypes, API controls, product
   orchestration, and unknown proprietary details.
4. The [production operations
   chapter](instruction-following-operations.md) defines the instruction
   contract, test suite, metrics, release gates, telemetry, and incident loop.

Within this page, first define the surface being tested, locate the failure in
the model or surrounding system, choose a matched benchmark, and treat forum
reports as failure-mode discovery rather than prevalence measurement.

The general [research standard](../research-method.md) defines the evidence
labels used below. The [agent evaluation chapter](../agentic-rl/evaluation-and-safety.md)
covers stochastic environments, judges, confidence intervals, and deployment
gates in more detail. For factual errors, unsupported claims, citations,
retrieval sufficiency, and abstention, start with the
[reliability and factuality map](index.md); a response can follow every
instruction and still be false.

## 1. What practitioners mean by “follows instructions”

A useful evaluation begins with the question:

> Which instruction, supplied by whom, must be retained for how long, while
> doing which task, through which interface?

At least eleven different capabilities are routinely collapsed into the same
phrase.

| Surface | Representative test | Common failure |
|---|---|---|
| Task completion | “Reconcile these invoices and identify the mismatch.” | Polished prose without completing the requested operation |
| Atomic constraints | “Use exactly three sentences and include `alpha` twice.” | One locally checkable constraint is missed |
| Negative constraints | “Do not contact support; do not modify tests.” | A strong default habit overrides the prohibition |
| Ordered instructions | “Validate first, migrate second, then report.” | Correct actions occur in the wrong order |
| Content requirements | “Include owner, amount, evidence, and due date.” | The answer is substantively good but omits one required field |
| Style and verbosity | “Return only the identifier; no explanation.” | Helpful preamble, headings, or unsolicited advice appears |
| Multi-turn retention | A rule introduced at turn 2 still applies at turn 20 | Recency dominates an older global rule |
| Versioned editing | “Change only section B; retain every accepted edit.” | Earlier edits regress or unrelated content changes |
| Instruction hierarchy | System policy conflicts with user or tool text | The model obeys the wrong source, or refuses everything |
| Tool and policy following | Select a tool, form valid arguments, obey domain rules | Correct-looking answer without the required tool or authorization |
| Calibration | “If the record is absent, say unavailable.” | The model fills the gap or overconfidently guesses |

Multilingual and multimodal variants add another axis: the model must retrieve
the governing text rule while reasoning over a different language, an image,
audio, or video. Consistency is also separate from best-case capability. A
workflow that succeeds once and fails on two of the next nine identical runs is
not production-reliable even if its best sample is excellent.

### Correct non-compliance exists

“Obey everything” is not the target. If a user asks a model to ignore a
developer policy, or an untrusted web page tells an agent to reveal a secret,
the correct behavior is to ignore the lower-trust instruction. OpenAI describes
the priority order as **system > developer > user > tool** in its 2026
[Instruction Hierarchy work](https://openai.com/index/instruction-hierarchy-challenge/).
Safety refusal, instruction-hierarchy resolution, and ordinary user
steerability therefore need separate labels.

### Helpfulness can conflict with literalness

Preference training often rewards explanations, initiative, polished prose,
and inferred intent. Those traits can hurt a request for minimal output, narrow
edits, strict non-action, or literal translation. “More helpful” and “more
obedient” are correlated only when the user's objective and the preference
model's default are aligned.

## 2. The observed behavior is an end-to-end system property

The user does not observe a bare checkpoint. The path is usually:

```text
system/developer/user/tool messages
  -> product policy and prompt assembly
  -> chat template, tokenization, retrieval, and context truncation
  -> model post-training and current inference mode
  -> sampling or grammar-constrained decoding
  -> agent loop, retries, validation, and state mutation
  -> displayed answer or external action
```

This decomposition prevents several category errors.

1. **Model behavior:** supervised fine-tuning, preference optimization, and
   reinforcement learning teach the checkpoint which constraints to notice and
   how to trade them off.
2. **Inference behavior:** thinking mode, reasoning effort, temperature, token
   budget, and stop conditions can change adherence. More test-time reasoning
   is not guaranteed to improve a literal constraint.
3. **Decoder enforcement:** a grammar can make invalid JSON tokens impossible.
   It cannot guarantee that a valid field contains the right fact.
4. **Harness behavior:** an agent may forget an instruction after compaction,
   omit a tool schema, retry selectively, or mutate a file even when the raw
   model intended otherwise.
5. **Product behavior:** memory, custom instructions, routing, hidden policies,
   search, moderation, and changing aliases can make two surfaces with the same
   vendor name behave differently.

OpenAI's 2024 [Structured Outputs disclosure](https://openai.com/index/introducing-structured-outputs-in-the-api/)
is a clean example. Its trained model scored 93% on an internal complex-schema
test, while dynamic constrained decoding raised schema conformance to 100%.
That last seven-point gain is an inference-system guarantee, not proof that the
language model became semantically perfect. DeepSeek's May 2024 API changelog
similarly distinguished an 85% JSON parse rate from 97% after adding appropriate
regular expressions.

## 3. Why GPT models have often felt unusually steerable

The strongest public explanation is not a special Transformer block. It is the
combination of post-training targets, developer-facing evaluation, role
hierarchy, and serving controls.

### 3.1 Disclosed post-training targets are close to developer pain

OpenAI's GPT-4.1 release lists an internal evaluation derived from developer
feedback with six explicit categories:

- format following;
- negative instructions;
- ordered instructions;
- content requirements;
- ranking; and
- overconfidence or abstention when data is unavailable.

The release also measured unrelated edits in code: 9% for GPT-4o versus 2% for
GPT-4.1 in OpenAI's internal test. These targets align closely with the practical
meaning of “do exactly this and do not improvise.” See the April 2025
[GPT-4.1 release](https://openai.com/index/gpt-4-1/), instruction-following and
real-world-example sections.

### 3.2 Multi-turn state is trained and evaluated separately

On OpenAI's reported settings, GPT-4.1 scored 87.4% on IFEval versus 81.0% for
GPT-4o and improved by 10.5 absolute points on Scale MultiChallenge. The GPT-5
developer release later reported 69.6% on MultiChallenge using an o3-mini judge,
64.0% on a hard internal API instruction-following evaluation, and 99.0% on
COLLIE. These are vendor-reported release results, not a current cross-vendor
leaderboard, but they show that OpenAI treated single-turn constraints and
multi-turn retention as different optimization targets. See
[Introducing GPT-5 for developers](https://openai.com/index/introducing-gpt-5-for-developers/).

### 3.3 Instruction hierarchy is a first-class training objective

OpenAI's 2024 [Instruction Hierarchy paper](https://openai.com/index/the-instruction-hierarchy/)
frames prompt injection as following the wrong instruction source. Its 2026
IH-Challenge work trains on objectively gradable role conflicts and reports
improvements on held-out hierarchy, prompt-injection, and safety-steerability
tests. Ordinary format obedience does not establish this property.

### 3.4 The API provides control surfaces around the model

Structured Outputs, forced or selected tools, grammar-constrained custom tools,
explicit developer messages, reasoning-effort controls, and a verbosity setting
reduce ambiguity or enforce part of the output language. The end-user impression
“GPT follows my format” can therefore combine genuine checkpoint steerability
with deterministic serving machinery.

This also explains a trade-off reported in the GPT-4.1 release: early testers
found the model more literal. Literalness improves compliance with explicit
constraints but may reduce useful inference when the request is underspecified.

## 4. DeepSeek: historical weakness, real progress, and remaining gaps

The public evidence supports a more precise statement than either “DeepSeek
cannot follow instructions” or “the gap is gone.”

### 4.1 The weakness was visible enough to become an explicit release target

DeepSeek's official [API changelog](https://api-docs.deepseek.com/updates/)
states that the May 2024 DeepSeek-V2 update raised IFEval prompt-level accuracy
from 63.9% to 77.6%, optimized system-message following, and improved an
internal JSON parse rate from 78% to 85%. The September 2024 V2.5 release again
called out writing and instruction following, although its published ArenaHard,
AlpacaEval, MT-Bench, and AlignBench scores were broader chat-preference
measures rather than pure constraint tests.

DeepSeek-R1's January 2025 report is also revealing. On its reported IFEval
Prompt Strict setting, R1 scored 83.3, below DeepSeek-V3 at 86.1, Claude 3.5
Sonnet at 86.5, and slightly below GPT-4o at 84.3. The same report says the
pure-RL R1-Zero checkpoint suffered endless repetition, poor readability, and
language mixing; R1 added cold-start data, supervised stages, and preference
alignment to address such behavior. Reasoning ability and controllable chat
behavior were not the same optimization problem. See the
[DeepSeek-R1 report and model repository](https://github.com/deepseek-ai/DeepSeek-R1).

### 4.2 V4 explicitly trains an instruction-following expert

The April 2026 [DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348)
discloses a two-stage post-training system. Separate domain experts for
mathematics, coding, agents, instruction following, and other areas each receive
supervised fine-tuning followed by Group Relative Policy Optimization (GRPO)
with domain-specific rewards. A unified model then learns from those experts by
on-policy distillation. This is direct evidence that instruction following is
now a dedicated training track rather than an incidental by-product of stronger
reasoning.

### 4.3 DeepSeek's own report is positive and candid

The V4 report contains several non-public internal evaluations, so their prompts,
selection process, and vendor control must be treated cautiously.

| Internal V4 evaluation | Disclosed result | What it does and does not show |
|---|---:|---|
| Chinese creative-writing instruction following vs Gemini 3.1 Pro | DeepSeek 60.0% win rate | Strong evidence for that vendor-curated Chinese writing distribution, not general API reliability |
| High-complexity instruction following and multi-turn writing vs Claude Opus 4.5 | DeepSeek 45.9%, Opus 52.0%, tie 2.0% over 196 cases | Directly targets the harder behavior users notice; still private and not a GPT comparison |
| Chinese white-collar tasks, instruction-following dimension | DeepSeek V4 Pro Max 87.76, Opus 4.6 Max 88.88 | Near parity on 30 tool-equipped professional tasks; scale and rubric are internal |

The authors explicitly state that V4 Pro Max occasionally overlooks particular
formatting constraints, slightly trails Opus on instruction following, is less
effective at condensing large inputs into succinct summaries, and sometimes
over-thinks or misreads vague coding prompts. That admission matches a common
user perception more closely than a saturated single-turn benchmark does.

### 4.4 Current independent precise-constraint scores are competitive

The following is a **captured leaderboard snapshot**, not a RoseLLM rerun. We
read the live Artificial Analysis IFBench data on 2026-07-22. IFBench scores are
prompt-level loose accuracy on difficult program-verifiable constraints; they
do not measure all the surfaces in Section 1.

| Selected live IFBench row | Score |
|---|---:|
| Grok 4.3, medium reasoning | 83.33% |
| MiniMax-M3 | 82.86% |
| DeepSeek V4 Flash, max reasoning | 79.18% |
| Gemini 3.1 Pro Preview | 77.14% |
| DeepSeek V4 Pro, max reasoning | 76.46% |
| GPT-5.5, xhigh reasoning | 75.85% |
| GPT-5.6 Sol, max reasoning | 72.65% |
| Claude Fable 5 with fallback | 63.47% |

Source: the live [Artificial Analysis IFBench leaderboard](https://artificialanalysis.ai/evaluations/ifbench).
Its contents are time-sensitive. The selected rows deliberately show that this
narrow benchmark does **not** support a claim that current GPT models universally
dominate DeepSeek. It also shows why one should not infer global quality from
model size or reasoning effort: Flash outranks Pro here, and the leading Grok
medium-effort row outranks its high-effort row.

The defensible current conclusion is therefore:

- DeepSeek has improved substantially and is competitive on hard, verifiable,
  single-response constraints;
- DeepSeek's own V4 evidence still identifies small deficits on complex,
  multi-turn, concise, and formatting-sensitive tasks;
- public evidence does not provide one current, apples-to-apples GPT-versus-V4
  score covering hierarchy, multi-turn retention, tools, non-interference,
  verbosity, and product behavior; and
- a production decision remains application- and harness-specific.

## 5. What each benchmark actually measures

No public leaderboard is “the real instruction-following ranking.” A useful
suite covers orthogonal failure surfaces.

| Benchmark | Primary surface | Scoring | Main strength | Main limitation |
|---|---|---|---|---|
| [IFEval](https://arxiv.org/abs/2311.07911) | About 500 prompts using 25 verifiable instruction types | Programmatic strict/loose, prompt/instruction level | Objective, cheap, reproducible | Public, narrow, increasingly saturated; easy to mistake formatting for general reliability |
| [IFBench](https://github.com/allenai/IFBench) | 58 new out-of-distribution constraints combined with held-out WildChat prompts | Programmatic; paper normally reports prompt-level loose accuracy | Harder and more diverse precise constraints; open code and data | Mostly synthetic/verifiable constraints; standard score is not long-horizon product behavior |
| [InFoBench](https://arxiv.org/abs/2401.03601) | 500 instructions decomposed into about 2,250 fine-grained requirements | LLM-based decomposition and Decomposed Requirements Following Ratio | Makes omitted requirements visible instead of assigning one opaque answer score | Decomposition and semantic grading inherit judge error |
| [FollowBench](https://github.com/YJiangcm/FollowBench) | Content, situation, style, format, and example constraints at increasing levels | LLM judge with constraint-evolution context | Diagnoses degradation as constraints accumulate | Judge quality and open-ended rubrics matter |
| [ComplexBench](https://github.com/thu-coai/ComplexBench) | Multiple constraints and their dependency/composition structure | Rules plus LLM evaluators | Tests composition rather than isolated rules | Evaluation is more complex and partly judge-dependent |
| [CFBench](https://aclanthology.org/2025.acl-long.1581/) | 1,000 Chinese prompts across 200-plus real scenarios, 50-plus tasks, and a 10-category constraint taxonomy | Multi-dimensional constraint, instruction, and fulfillment scoring with requirement priority | Tests realistic Chinese constraint composition beyond a small rule set | Language and judge-dependent open-ended scoring limit generalization |
| [IFScale](https://distylai.github.io/IFScale/) | Adherence as the number of simultaneous instructions grows | Constraint-specific evaluation | Exposes scaling curves hidden by one aggregate | Still a controlled constraint proxy |
| [MultiChallenge](https://labs.scale.com/papers/multichallenge) | Instruction retention, inferred user memory, versioned editing, and self-coherence across turns | Instance rubrics with an LLM judge | Much closer to conversational failure | Scores are judge-sensitive; OpenAI reports its default GPT-4o judge sometimes mis-scored responses |
| [Multi-IF](https://arxiv.org/abs/2410.15553) | Three-turn instruction following across eight languages | Hybrid programmatic/LLM/human construction and evaluation | Reveals turn and language degradation | Limited dialogue depth and benchmark language set |
| [LIFBench](https://arxiv.org/abs/2411.07037) | 2,766 expanded instructions over three long-context scenarios, 11 tasks, and six length intervals | Rubric-based automated LIFEval without human or LLM judging | Exposes length-dependent performance and stability hidden in short prompts | Synthetic expansion and controlled lengths may not match a live product's compaction |
| [SysBench](https://proceedings.iclr.cc/paper_files/paper/2025/file/b917f916e7eed84ffe8f5e63492b2be8-Paper-Conference.pdf) | 500 system messages, each paired with five user turns, covering constraint violations, priority conflicts, and instability | Checklist-guided LLM verifier at three granularities | Separates higher-priority system compliance from ordinary user obedience | Judge error and a five-turn setting do not cover every tool injection or domain authorization rule |
| [MMMT-IF](https://research.google/pubs/mmmt-if-a-challenging-multi-modal-multi-turn-instruction-following-foundation-model-benchmark/) | Global text rules dispersed through image-grounded multi-turn dialogue | Program-verifiable Programmatic Instruction Following metrics | Separates rule retrieval from visual reasoning and tests repeated-run robustness | Narrow multimodal task design and older model set |
| [IH-Challenge](https://openai.com/index/instruction-hierarchy-challenge/) and System IFEval | Conflicts across system, developer, user, and tool roles | Programmatic | Tests obedience to the correct source and prompt-injection robustness | Simple conflicts do not cover every real malicious document or agent trajectory |
| [BFCL V4](https://gorilla.cs.berkeley.edu/leaderboard) | Single-turn, multi-turn, memory, hallucination, and tool-call correctness | Abstract-syntax-tree, execution, and task-specific checks | Reproducible tool-interface diagnostics with released responses | Tool syntax success is not complete domain-policy compliance |
| [$\tau^2$-bench](https://github.com/sierra-research/tau2-bench) | Stateful user-agent-tool interaction under a domain policy | Final state and task checks | Tests whether an agent completes work while following operating rules | Simulator, policy, user model, and harness affect the score |
| [LiveBench](https://github.com/LiveBench/LiveBench) instruction category | Dynamic objective instruction tasks | Programmatic ground truth, no LLM judge | Periodic releases reduce contamination and preserve discrimination | The instruction category remains a task sample, not an end-to-end product audit |

### Preference leaderboards are not direct instruction tests

Chatbot Arena, ArenaHard, AlpacaEval, and MT-Bench can reveal whether humans or
judges prefer an answer. Preference can include correctness, prose quality,
length, tone, and brand/style effects. A preferred answer may have silently
violated a negative instruction, while a terse perfectly compliant answer may
lose on perceived helpfulness. Use these resources as broad chat-quality
signals, not substitutes for constraint-level evaluation.

### Aggregate scores hide catastrophic “one missed rule” failures

Report at least both:

- **constraint-level accuracy:** the fraction of individual rules satisfied;
- **all-constraints or prompt-level accuracy:** the fraction of tasks on which
  every required rule passes.

A response that satisfies nine of ten rules scores 90% under the first metric
and 0% under the second. The second often matches production reality when one
forbidden edit, missing field, or unauthorized action invalidates the result.

## 6. What forums and practitioner reports can legitimately tell us

Forum posts are valuable for discovering recurring failure modes. They are poor
measurements of model-family prevalence because users self-select, prompts are
usually unavailable, model aliases change, and successful routine runs are
underreported.

### DeepSeek V4 community themes

Recent reports are mixed rather than uniformly negative.

- Several users report that V4 Flash is fast and economical for coding agents,
  long-context refactors, and specification-driven work; one July 2026 user
  called V4 Pro sufficient and said it followed instructions well. See the
  positive [V4 Pro report](https://www.reddit.com/r/DeepSeek/comments/1v2dwel/deepseek_v4_pro_is_amazing/)
  and an [OpenCode/V4 Flash thread](https://www.reddit.com/r/DeepSeek/comments/1uknpj9/switched_to_open_code_deepseek_v4_flash_is_the/).
- Other users repeatedly describe “babysitting,” omitted prompt clauses,
  premature claims of completion, role-play/style drift, or simple instructions
  becoming worse in thinking mode. See the mixed
  [Flash/Pro instruction thread](https://www.reddit.com/r/DeepSeek/comments/1uagzvt/flash_and_pro_not_following_instructions/)
  and the [reasoner-versus-chat discussion](https://www.reddit.com/r/DeepSeek/comments/1sv9lzt/deepseek_v4_how_can_we_get_more_out_of/).
- Some practitioners find non-thinking mode better for simple literal tasks and
  high/max effort better when retrieving many constraints or files. That is a
  useful test hypothesis, not a universal rule.
- Role-play communities emphasize persistent character knowledge, style, and
  “blind spots,” which are poorly represented by IFEval. Coding communities
  emphasize prohibited file reads, plan-mode boundaries, tool use, and claims
  of test completion. Their contradictory verdicts often reflect different
  target capabilities.

### GPT community themes

GPT models also receive current complaints about custom instructions or coding
rules being ignored, “helpful” reinterpretation, loss after context compaction,
and behavior changes across product updates. Examples include a June 2026
[Codex AGENTS.md report](https://www.reddit.com/r/codex/comments/1u9t02o/gpt_54_and_55_are_ignoring_agentsmd_instructions/)
and a [translation/adaptation complaint](https://www.reddit.com/r/ChatGPTcomplaints/comments/1tuupks/gpt55_keeps_helpfully_ignoring_instructions_i/).
These reports refute “GPT never disobeys,” but they do not establish a 40%
population failure rate or isolate the checkpoint from memory, compaction, or
application behavior.

### Normalize these confounders before believing a comparison

1. exact immutable model ID and provider;
2. web application, first-party API, or third-party gateway;
3. thinking enabled/disabled and reasoning effort;
4. temperature, top-p, maximum output, stop conditions, and retries;
5. system/developer prompt, chat template, and full message order;
6. context length, retrieval, summarization, and compaction state;
7. native checkpoint versus a quantized or distilled local model;
8. native tools versus prompt-emulated tools and constrained decoding;
9. safety policy or content filter; and
10. one impressive/failing sample versus repeated trials.

DeepSeek's current [Thinking Mode documentation](https://api-docs.deepseek.com/guides/thinking_mode/)
states that thinking defaults to enabled for V4, regular requests default to
high effort, some complex agent requests are automatically moved to max, and
the old `deepseek-chat`/`deepseek-reasoner` aliases temporarily route to V4
Flash non-thinking/thinking modes before retirement. A comparison that records
only “DeepSeek” is therefore not reproducible.

## 7. How to read an instruction-following claim in a tech report

Before accepting a score, record the following.

| Question | Why it changes the interpretation |
|---|---|
| Is this atomic formatting, open-ended task completion, multi-turn retention, hierarchy, or tool policy? | These capabilities are not interchangeable |
| What exact checkpoint, provider, date, mode, and reasoning effort were used? | Aliases and inference effort can change behavior |
| Is the dataset public, private, recently refreshed, or sampled from production? | Contamination and selection bias differ |
| Is scoring deterministic, human, or an LLM judge? | Rules miss semantics; judges add preference and self-bias |
| What is the denominator and metric: per-rule, all-rules, pass@1, best-of-N, win rate, or Elo? | Identical-looking percentages can mean different events |
| Were failures, refusals, truncations, and API errors counted as incorrect? | Filtering can inflate reliability |
| Were prompts, raw responses, graders, and code released? | Reproduction and error auditing require artifacts |
| Was JSON or tool syntax grammar-constrained? | Decoder enforcement is not raw model adherence |
| Were the compared prompts and budgets identical? | Extra reasoning tokens and retries buy capability |
| Does the report include negative results and slice breakdowns? | Aggregate wins can conceal the user's critical failure class |

Vendor reports are useful evidence of training priorities and internal product
distributions. They are not neutral leaderboards. Independent leaderboards are
useful for standardized comparisons. They are not the user's production
distribution. Both are needed.

## 8. A production evaluation that reflects actual experience

For a meaningful GPT-versus-DeepSeek decision, build a private suite from real
failures and accepted outputs. A practical first version contains 100–200 tasks:

- 20–30 atomic and compositional hard constraints;
- 15–25 negative instructions and “do nothing outside scope” cases;
- 15–25 long-context or multi-turn retention cases;
- 10–20 versioned editing and no-regression cases;
- 10–20 system/developer/user/tool conflicts and prompt injections;
- 15–25 tool-selection, argument-schema, error-recovery, and domain-policy
  cases; and
- multilingual or multimodal slices proportional to the application.

For each case, separate:

1. **task oracle:** was the substantive outcome correct?
2. **constraint oracles:** did every positive, negative, order, and scope rule
   pass?
3. **action oracle:** were only authorized tools and mutations used?
4. **calibration oracle:** did the model abstain when required?
5. **efficiency record:** latency, input/output/reasoning tokens, tool calls,
   retries, and cost.

Run both deterministic-looking settings and production settings. Use at least
three repeated trials for high-risk tasks; five or more is better when sampling,
tools, or simulated users add variance. Report:

- first-attempt all-constraints pass rate;
- per-constraint accuracy by failure type;
- task-completion rate;
- unauthorized or extraneous action rate;
- repair rate after one explicit critique;
- regression rate across turns;
- output-length ratio against the requested length;
- mean and tail latency, token use, tool calls, and cost; and
- confidence intervals or paired bootstrap uncertainty.

Evaluate V4 Flash and Pro separately, with thinking disabled and with high/max
effort where applicable. Evaluate each GPT checkpoint with the exact reasoning
effort, verbosity, tool mode, and schema enforcement used in production. Do not
let one model use grammar constraints, retries, or a stronger harness while
calling the result a checkpoint comparison.

The final ship criterion should be slice-specific. A lower-cost model can be a
good default for reversible drafting while a more reliable model handles
irreversible actions, complex policy, or final verification. Model routing is
often a better engineering answer than declaring one universal winner.

## 9. Bottom line

The historical impression that GPT models were easier to steer than early
DeepSeek chat/reasoning models has real support: DeepSeek repeatedly targeted
instruction following in release notes, R1 underperformed its own V3 on strict
IFEval, and reasoning-oriented training exposed readability and control
trade-offs.

The 2026 picture is more nuanced. DeepSeek V4 now includes a dedicated
instruction-following expert in post-training and is competitive with or ahead
of current GPT rows on the narrow IFBench precise-constraint leaderboard. At
the same time, DeepSeek's own report and current practitioner discussions still
identify complex multi-turn retention, succinctness, formatting details,
over-thinking, and agent-policy adherence as places requiring supervision.

GPT's perceived advantage is best understood as **model post-training plus
instruction hierarchy plus developer controls plus a mature serving and agent
stack**, not as an architectural law. It remains fallible, especially when
custom instructions, memory, compaction, safety policy, or helpful defaults
conflict. The most honest current answer is not one leaderboard rank but a
vector of matched, repeated measurements on the user's actual workflow.

## References

1. Jeffrey Zhou et al.,
   [“Instruction-Following Evaluation for Large Language Models”](https://arxiv.org/abs/2311.07911),
   2023.
2. Valentina Pyatkin et al.,
   [“Generalizing Verifiable Instruction Following” / IFBench](https://arxiv.org/abs/2507.02833),
   NeurIPS 2025; [official code](https://github.com/allenai/IFBench).
3. Ai2,
   [“Why Artificial Analysis uses Ai2's IFBench instruction-following eval”](https://allenai.org/blog/ifbench-artificial-analysis),
   2026.
4. Ved Sirdeshmukh et al.,
   [“MultiChallenge”](https://labs.scale.com/papers/multichallenge), 2025.
5. OpenAI, [“Introducing GPT-4.1 in the API”](https://openai.com/index/gpt-4-1/),
   2025.
6. OpenAI,
   [“Introducing GPT-5 for developers”](https://openai.com/index/introducing-gpt-5-for-developers/),
   2025.
7. OpenAI,
   [“Improving instruction hierarchy in frontier LLMs”](https://openai.com/index/instruction-hierarchy-challenge/),
   2026.
8. DeepSeek,
   [API changelog](https://api-docs.deepseek.com/updates/), verified
   2026-07-22.
9. DeepSeek,
   [“DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning”](https://arxiv.org/abs/2501.12948),
   v2, 2026.
10. DeepSeek,
    [“DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence”](https://arxiv.org/abs/2606.19348),
    2026, especially Sections 1 and 5.4 and Tables 13–14.
11. UC Berkeley,
    [Berkeley Function-Calling Leaderboard V4](https://gorilla.cs.berkeley.edu/leaderboard),
    verified 2026-07-22.
12. LiveBench,
    [official repository and evaluation protocol](https://github.com/LiveBench/LiveBench),
    verified 2026-07-22.
13. Elliot Epstein et al.,
    [“MMMT-IF”](https://research.google/pubs/mmmt-if-a-challenging-multi-modal-multi-turn-instruction-following-foundation-model-benchmark/),
    2024.
14. Yiwei Qin et al.,
    [“InFoBench”](https://arxiv.org/abs/2401.03601), 2024.
15. Tao Zhang et al.,
    [“CFBench”](https://aclanthology.org/2025.acl-long.1581/), ACL 2025.
16. Xiaodong Wu et al.,
    [“LIFBench”](https://arxiv.org/abs/2411.07037), 2024.
17. Yanzhao Qin et al.,
    [“SysBench”](https://proceedings.iclr.cc/paper_files/paper/2025/file/b917f916e7eed84ffe8f5e63492b2be8-Paper-Conference.pdf),
    ICLR 2025.
