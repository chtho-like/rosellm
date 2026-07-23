# What Leading Model Vendors Disclose About Hallucination Control

**Verified through:** 2026-07-23.

This is an evidence audit, not a ranking. It asks which hallucination controls a
vendor has publicly confirmed for a named model, product, experiment, or
customer-facing platform. It does not fill gaps in proprietary training or
serving stacks by analogy.

Evidence labels follow the [research standard](../research-method.md):

- **D — disclosed:** a first-party source or author paper states the practice;
- **C — confirmed artifact:** released code, configuration, prompt, data, or
  weights directly establish it;
- **R — reproduced:** RoseLLM retained a reproducible run and artifacts;
- **I — inferred:** the conclusion depends on named evidence and assumptions;
- **U — unknown:** public evidence is absent or insufficient.

The object label is equally important:

- **model training** describes how a checkpoint was trained;
- **evaluation** describes a test, not necessarily a mitigation;
- **product** describes a deployed user surface around one or more models;
- **API/cloud control** is functionality a customer may choose to add;
- **research prototype** is not automatically in a released model or product.

An Azure, Bedrock, Vertex AI, or NeMo guardrail does not prove that every model
served by that platform was trained with the same method. A product citation
does not prove claim–source entailment. A model trained to call search does not
prove that every live answer searched.

## 1. Cross-vendor conclusions

### The publicly visible production pattern

Leading systems repeatedly expose five layers:

1. **post-training for honesty, factuality, tool use, or abstention;**
2. **search, retrieval, databases, code, and other external tools;**
3. **citations or source-local provenance;**
4. **verifiers, groundedness checks, model cards, and private evaluations;**
5. **product policies that ask, abstain, block, correct, or escalate.**

The exact online router, source ranker, reward decomposition, verifier
threshold, cache policy, incident rule, and model mixture are usually **U**.
Public evidence therefore supports a layered industry pattern, not a complete
reverse-engineered recipe for any frontier product.

### Particularly concrete disclosed training examples

| Vendor / model | Disclosed control | Why the evidence is unusually useful |
|---|---|---|
| OpenAI InstructGPT | human demonstrations, preference model, PPO, and truthfulness/closed-domain hallucination evaluation | connects a named post-training pipeline to measured truthfulness changes |
| OpenAI WebGPT / deep research | browser/search trajectories, source-visible preferences, candidate selection, and later end-to-end research-agent RL | makes evidence collection part of the learned policy |
| Microsoft Phi-4 | self-knowledge probes, answer/refuse SFT, bogus unanswerable questions, and DPO ordering `correct > refuse > wrong` | directly targets the answer-versus-abstain boundary |
| Meta Llama 3 | generate factual probes from pretraining text, sample repeatedly, and train refusals where the model does not reliably know | explicitly teaches knowledge limits rather than adding facts |
| Cohere Command A | separate RAG and tool-use SFT/RL experts, then merge and polish them | exposes a grounded/tool trajectory as a dedicated training specialization |
| Tencent Hunyuan-A13B | hallucination-focused reward model, reference-aware and reference-free hallucination detectors, and online RL | exposes a multi-scorer factuality pipeline rather than only a search product |
| Alibaba Qwen3 | a RAG-specific reward that favors accurate use of retrieved context | explicitly names hallucination reduction in a released model's RL objective |
| Moonshot Kimi K2 | sentence-level grounded-factuality judge within a multi-task RL Gym | names a factuality reward component for a released model family |
| Kimi-Researcher | end-to-end RL over multi-turn real search and browsing | trains evidence acquisition and recovery rather than one-shot RAG |
| DeepSeek-V3.2 | multi-agent construction and verification of 50,275 search tasks, followed by search-environment RL and generative judging | combines verifiable evidence-seeking data with a named factual-reliability objective |
| Zhipu GLM-4.5 | cross-page search questions and trajectory rewards based on final-answer accuracy | gives a concrete search-agent recipe while also exposing outcome-only limits |
| xAI Grok 4.1 | post-training aimed at factual hallucinations plus atomic-fact evaluation on sampled production queries | connects post-training to a live-distribution audit, though the loss remains undisclosed |
| NVIDIA Nemotron 3 Nano experiment | on-policy preference pairs penalizing calls to nonexistent tools | detailed tool-hallucination ablation, explicitly not the final release recipe |

### Common disclosure gaps

For most vendors, the following remain **U**:

- exact factuality or abstention data and mixture weights;
- online source authority and diversity rules;
- retrieval, reranking, and evidence-sufficiency thresholds;
- claim-level citation entailment checks;
- how model, search, and product routes change by query;
- cache freshness and contradiction handling;
- severe-error monitoring and incident circuit breakers;
- complete reward composition and independent audit sets; and
- causal ablations separating checkpoint, prompt, search, retry, and verifier.

The absence of disclosure is not proof that a control is absent. It prevents a
public claim that the control is used.

## 2. OpenAI

### Factuality, calibrated failure, and tool-use post-training

OpenAI's public record spans several generations; it should not be compressed
into a claim that one historical recipe is the current production stack.

**[D — model training and evaluation]**

- [InstructGPT](https://arxiv.org/abs/2203.02155) used human demonstrations,
  preference comparisons, a reward model, proximal policy optimization (PPO),
  and explicit TruthfulQA and closed-domain hallucination evaluations. It is
  evidence that post-training can change truthfulness, not a complete
  description of current ChatGPT.
- The [GPT-5 System Card](https://cdn.openai.com/gpt-5-system-card.pdf),
  Section 3.7, says factuality was a post-training focus, including effective
  browsing and factual answers without browsing. On a representative
  production-prompt evaluation, the reported fraction of responses containing
  a factual error was 26% lower for `main` than GPT-4o and 65% lower for
  `thinking` than o3. The audit extracted claims and checked them with a
  web-enabled model grader, then human-reviewed a sample; 75% agreement is a
  grader–human agreement rate, not model accuracy.
- The same card's “deception” work includes environments in which a task is
  impossible, a browser fails, an attachment is absent, or the request is
  underspecified. The desired behavior is to report the limitation instead of
  fabricating a tool result or claiming completion. This is honest-failure
  training, not a universal fact checker.
- The [o3 and o4-mini System
  Card](https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf)
  describes large-scale reinforcement learning and learned use of web search,
  Python, files, and images during reasoning.

**[D — evaluation and production-feedback audit]** The [GPT-5.6
System Card](https://deploymentsafety.openai.com/gpt-5-6/gpt-5-6.pdf),
Section 6.1, evaluates de-identified historical conversations that users had
flagged for factual errors. It separately measures whether a new answer has any
error and whether it repeats the specific reported error. That confirms a
production-failure-to-evaluation loop; it does not show direct online learning
from an individual report.

**[U]** The factuality data, reward terms and weights, browser curriculum,
router, release gates, and live thresholds for current products are not public.

### WebGPT, search, and deep research

The [WebGPT paper](https://arxiv.org/abs/2112.09332) gives a historically
precise research recipe: a constrained browser, about 6,000 demonstrations,
21,500 pairwise comparisons, behavior cloning, a reward model, PPO, and
best-of-$N$ candidate selection. The strongest reported 175B system used
behavior cloning plus best-of-64 and returned source citations. **[D —
research prototype]** It does not establish the current ChatGPT Search stack.

OpenAI says [ChatGPT search](https://openai.com/index/introducing-chatgpt-search/)
used a fine-tuned GPT-4o, synthetic data including distilled outputs from
o1-preview, third-party search, partner content, and links to web sources.
**[D — product disclosure]**

[Deep research](https://openai.com/index/introducing-deep-research/) is
described as end-to-end reinforcement learning on difficult browsing and
reasoning tasks, with learned planning, repeated search, backtracking, Python
use, and citations. **[D — product/model training]** The same official source
lists residual failures: hallucinated facts or inferences, difficulty
distinguishing authoritative information from rumors, weak uncertainty
calibration, and citation-formatting errors. Those limitations are as
important as the training disclosure.

Current APIs expose model-selected
[web search](https://developers.openai.com/api/docs/guides/tools-web-search),
[file search](https://developers.openai.com/api/docs/guides/tools-file-search),
consulted-source lists, inline URL annotations, and deep-research tool traces.
**[C — API/product]** A retrieved URL or valid pointer does not establish
source quality, completeness, or claim–source entailment.

### Make abstention score better than guessing

OpenAI's [Model Spec](https://model-spec.openai.com/2025-04-11.html) asks
models to express uncertainty and clarify missing information. **[D —
behavioral specification]** A specification is a training target and product
norm, not evidence that every answer complies.

The research note [Why language models
hallucinate](https://openai.com/index/why-language-models-hallucinate/) argues
that next-token learning leaves rare facts statistically uncertain and that
accuracy-only leaderboards reward guessing. Its SimpleQA example reports
GPT-5-thinking-mini at 22% correct, 26% wrong, and 52% abstain, versus o4-mini
at 24% correct, 75% wrong, and 1% abstain. **[D — research/evaluation]** The
proposal is to penalize confident errors more heavily and give abstention
credit; it is not disclosure of every production loss.

OpenAI's [Confessions
prototype](https://openai.com/index/how-confessions-can-keep-language-models-honest/)
separates the task answer from a second report in which the model admits
errors, uncertainty, or reward hacking, and trains an independent reward model
only on the confession. **[D — research prototype]** It is explicitly not a
deployed ChatGPT feature.

Structured Outputs constrains a response to a supplied JSON Schema. **[D —
API control]** It prevents many protocol and extra-field failures, but it does
not validate factual field values.

## 3. Anthropic

### Honesty as a training objective, with undisclosed implementation detail

Anthropic's current
[constitution](https://www.anthropic.com/news/claude-new-constitution)
is used across multiple training stages. Claude generates material for
understanding the constitution, dialogues, candidate answers, and rankings
that are then used in training. The constitution calls for honesty, calibrated
confidence, acknowledgement of uncertainty, and no fabricated sources.
**[D — model-training norm]**

This continues a lineage from
[Constitutional AI](https://arxiv.org/abs/2212.08073): the 2022 prototype had
the model critique and revise responses for SFT, then used AI comparisons to
train a preference model and reinforcement learning from AI feedback (RLAIF).
**[D — historical research recipe]** That paper primarily targeted
harmlessness, and Anthropic says its process has evolved; it is not the full
2026 factuality recipe.

The [Fable 5 and Mythos 5 System
Card](https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf),
Section 6.3.3, states a direct behavioral objective: answer accurately when
confident, decline when uncertain, and do not invent facts, citations, or
capabilities. **[D — model training]** Exact data, losses, reward weights, and
stage composition remain **U**.

Anthropic also evaluates closed-book questions as correct, incorrect, or
abstained and reports a `correct - incorrect` net score, so a bad guess is worse
than abstention. It tests false premises, missing attachments, unavailable
tools, missing citations, multiple languages, and pressure to accept an
incorrect premise. Two selected difficult hallucination sets still showed
only 87% and 82% non-hallucination rates for Mythos 5, demonstrating residual
failure. **[D — evaluation]** The sets are intentionally hard and are not
production incidence estimates.

### Production research, search, retrieval, and citations

Anthropic's [multi-agent research-system engineering
report](https://www.anthropic.com/engineering/multi-agent-research-system)
describes a lead agent that delegates parallel searches to subagents, adapts
the plan as evidence arrives, and passes the final report to a separate
CitationAgent that locates citations in the collected documents. Human testing
found a preference for search-engine-optimized content farms, so the team
added heuristics favoring authoritative and primary sources. **[D/C —
production product and artifact]**

The research system is evaluated by model judges on factual accuracy, citation
accuracy, completeness, source quality, and tool efficiency, alongside manual
testing for rare hallucinations and source bias that automated tests missed.
The disclosed internal improvement is not a universal accuracy rate, and
multiple agents can propagate a shared bad premise.

Claude API
[web search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)
lets the model decide when and how often to search, supports domain controls,
and requires citations. A later tool version lets Claude write and run code to
filter search results before placing relevant material in context. **[C —
API/product]** Filtering improves relevance and context efficiency; it does not
guarantee source authority or factual entailment.

The [Citations API](https://platform.claude.com/docs/en/build-with-claude/citations)
returns an exact passage plus page or character ranges in supplied documents.
**[C — API]** It confirms that a pointer resolves to input text, not that the
text is true, that the claim follows from it, or that all material claims are
covered.

### A useful retrieval technique, not evidence of Claude's private stack

[Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
prepends document-level context to each chunk before embedding and BM25
indexing. In Anthropic's experiment, embedding plus BM25 reduced top-20
retrieval failure from 5.7% to 2.9%; reranking reduced it to 1.9%. **[D —
customer method/research experiment]** This is a concrete method for builders,
not confirmation that Claude.ai internally uses the same RAG pipeline.

Anthropic's developer guidance recommends explicitly allowing “I don't know,”
extracting verbatim evidence before answering long documents, checking every
claim against a quotation, withdrawing unsupported claims, best-of-$N$, and
iterative revision. **[D — customer guidance]** These are not disclosures of
the production model stack.

**Still unknown:** current pretraining fact selection, factuality rewards,
search ranking, citation-entailment thresholds, release gates, model routing,
and live error rates.

## 4. Google and Google DeepMind

### Factuality-focused adaptation and learned search

The [Gemini 1 Technical
Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf),
Section 5.1.6, explicitly describes factuality-focused adaptation for three
behaviors: closed-book factuality, attribution to user-provided context, and
hedging on unanswerable questions. In the reported final adaptation stage,
error fell from 6.7% to 3.8%, attribution score (AIS) rose from 40.2% to 60.0%,
and hedging rose from 0% to 69.3%. **[D — model training and evaluation]**
The data, objective, and mixture weights are not disclosed.

The [Gemini 2.5 Technical
Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
states that Gemini 2.0 was trained to call Google Search natively and that 2.5
learns interleaved thought, search, follow-up queries, and fact checking. It
also describes more reinforcement-learning compute, verifiable rewards,
model-generated rewards, and complex multi-step tool environments. **[D —
model training]** Those broad reward classes do not disclose a dedicated
factuality loss, weight, or search curriculum.

Historical DeepMind prototypes expose more algorithmic detail:

- [GopherCite](https://arxiv.org/abs/2203.11147) used human-preference
  reinforcement learning for open-book question answering, selected evidence
  passages, quoted them, and could abstain. On the least-certain third of
  questions, abstention raised the reported human-rated quality to 90% on
  Natural Questions and 80% on ELI5. **[D — research prototype]**
- [Sparrow](https://arxiv.org/abs/2209.14375) combined Google Search, a
  preference reward model, a 23-rule reward model, an evidence-support
  auxiliary objective, eight-candidate reranking, actor-critic learning, and
  self-play. Human raters judged 78% of its evidence-supported claims as
  supported. **[D — research prototype]**

Neither prototype is proof of current Gemini architecture; GopherCite also
shows that apparently supportive evidence can still support a false answer.

### Grounding, output-side checking, and user-visible verification

Gemini API [Grounding with Google
Search](https://ai.google.dev/gemini-api/docs/google-search?hl=en) lets the
model decide whether to search, issue one or more queries, consume results, and
return query/result metadata with text-span-aligned URL citations. It can be
combined with URL context and code execution. **[C — API/product]**

Google Cloud's [Check Grounding
API](https://cloud.google.com/generative-ai-app-builder/docs/check-grounding?hl=en)
compares an answer with up to 200 supplied facts and returns overall and
claim-level support scores, source spans, and citations; its documented
criterion requires a whole sentence to be entailed by the facts. **[C —
API/cloud control]** It checks support relative to supplied evidence, not
whether that evidence is true.

[Gemini Deep
Research](https://blog.google/products-and-platforms/products/gemini/google-gemini-deep-research/)
shows the user an editable plan, then iterates search, reading, and follow-up
search before producing a linked report. The Gemini Apps
[double-check feature](https://support.google.com/gemini/answer/14143489?hl=en)
performs a new search for similar or contradictory material and colors
statements accordingly. **[C — product]** Google explicitly warns that a green
link may not have been used to generate the original answer, similarity is not
entailment, and the feature can be wrong.

### Retrieval sufficiency, specialized grounding, and evaluation

Google research finds that RAG can make a model more confidently wrong when
the retrieved context is insufficient; its
[sufficient-context work](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/)
separates whether the evidence is adequate from whether an answer looks
plausible. **[D — research]**

[DataGemma](https://research.google/blog/grounding-ai-in-reality-with-a-little-help-from-data-commons/)
explores two specialized routes over Data Commons: retrieval-interleaved
generation, in which the model emits calls for exact statistics while writing,
and RAG over a natural-language corpus generated from the database. The Gemma
variants are fine-tuned to decide when to ask the external source. **[D —
research prototype]**

The [FACTS Grounding
benchmark](https://deepmind.google/blog/facts-grounding-a-new-benchmark-for-evaluating-the-factuality-of-large-language-models/)
first checks task compliance, then uses cross-family model judges calibrated on
a held-out human set to judge complete grounding over 1,719 long-document
examples. The later [FACTS
Suite](https://deepmind.google/blog/facts-benchmark-suite-systematically-evaluating-the-factuality-of-large-language-models/)
separates parametric, search, multimodal, and supplied-context factuality and
gives every model the same search tool. At its release, every tested model
scored below 70% overall. **[D — evaluation]** A benchmark is not proof of a
production gate or an estimate of live error incidence.

Google DeepMind's [SAFE and LongFact
work](https://deepmind.google/research/publications/85420/) decomposes a long
answer into atomic claims, generates search queries for each, judges support,
and combines precision with coverage. On roughly 16,000 facts, SAFE agreed
with crowd workers 72% of the time; on 100 disagreements, an adjudication
favored SAFE 76% of the time. **[D — research evaluator]** There is no public
evidence that SAFE is the Gemini production checker or training reward.

**Still unknown:** the current Gemini factuality data and reward mixture,
search/ranking implementation, claim thresholds, release gates, live
incidence, and how thumbs-down reports enter training.

## 5. Microsoft and Azure

### Phi-4: train the answer-versus-refuse boundary

The [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905), Section 4.4
and Appendix A.1, provides one of the clearest public abstention recipes.

**[D — model training]**

1. Run a base model several times on factual trivia questions and estimate
   whether it reliably answers correctly.
2. For knowledge the model usually has, create SFT examples mapping the
   question to the correct answer.
3. For knowledge it usually lacks, create examples mapping the question to a
   refusal.
4. Add plausible-looking but unanswerable synthetic questions.
5. Construct DPO preferences so a correct answer beats refusal, while refusal
   beats a wrong answer.

On the report's SimpleQA analysis, wrong answers fell from **90.0%** in the base
model to **15.8%** in the final model, while abstention rose from **6.8%** to
**81.1%**; correct answers stayed around 3%. The principal effect was safer
selective answering, not acquisition of more trivia knowledge. The report also
notes that an F1-style score can punish this trade-off.

**[U]** The complete factuality-data size, thresholds, mixture weights, and
transfer to current Microsoft products are not public.

### Azure groundedness and retrieval controls

**[D — API/cloud control]**

- Azure AI Content Safety
  [groundedness detection](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/content-filter-groundedness)
  compares an answer with supplied sources. It supports a binary mode and a
  reasoning mode that localizes unsupported spans; a correction option can be
  configured.
- Azure OpenAI On Your Data returns source text, title, URL, path, chunk ID,
  retrieval score, and filtering information through its
  [API response fields](https://learn.microsoft.com/en-us/azure/ai-services/openai/references/on-your-data).
- Microsoft's own
  [best-practices page](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/how-to/on-your-data-best-practices?view=foundry-classic)
  warns that `inScope=true` is not a hard guarantee: models may still answer
  outside the retrieved domain and citations may be missing.
- Microsoft 365 Copilot describes permission-aware organizational grounding
  through the
  [Semantic Index and Microsoft Graph](https://learn.microsoft.com/en-us/microsoftsearch/semantic-index-for-copilot),
  with web search for current information.

These are deployed or customer-configurable system layers. They do not disclose
the internal training of every hosted model or prove that every attached
citation entails its claim.

### Structural controls

Azure
[Structured Outputs](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs)
constrains a response to a JSON Schema. **[D — API/cloud control]** This is
valuable for tool names, arguments, required fields, and types. It does not make
the field values factually correct.

**Still unknown:** current Copilot model routing, retriever and reranker,
citation-entailment checks, online thresholds, and incident policies.

## 6. Meta

### Llama 3 factuality and abstention training

The [Llama 3 model-family report](https://arxiv.org/abs/2407.21783), Sections
3.1.2, 4, 4.3.5, and 4.3.6, discloses several relevant controls.

**[D — model training]**

- Pretraining data undergoes deduplication, quality filtering, and knowledge
  classification; scaling experiments inform the data mixture.
- Post-training iterates SFT, a reward model, rejection sampling, and DPO, with
  roughly 10–30 candidate responses sampled per prompt before reward-model
  selection.
- For factuality, Meta generates factual questions from pretraining passages,
  samples the model repeatedly, and checks correctness and informativeness
  against the passage. Questions that elicit informative but persistently wrong
  answers become refusal examples. The stated objective is to teach the model
  the boundary of what it knows, not add new facts during post-training.
- Llama 3 is trained on human and synthetic trajectories for Brave Search,
  Python, and Wolfram Alpha, including ReAct-style and multi-step tool use.

**[U]** Public sources omit the factual-probe dataset size, thresholds, loss
weights, and whether Llama 4 retained the same factuality pipeline.

### Product retrieval and an event-level circuit breaker

**[D — product]** Meta AI's Llama 3 launch describes use of
[real-time web information](https://about.fb.com/news/2024/04/meta-ai-assistant-built-with-llama-3/).
Later news partnerships add publisher material and article links.

Meta also disclosed a concrete production failure response after the July 2024
attempted assassination of Donald Trump. During the initial breaking-news
period, it configured Meta AI not to answer related questions because reliable
information was sparse, then updated the system as facts became available.
Meta's
[post-incident account](https://about.fb.com/news/2024/07/review-of-fact-checking-label-and-meta-ai-responses/)
also acknowledges that erroneous responses still occurred. This is direct
evidence of a topic-level temporary abstention/circuit breaker, not evidence of
perfect execution.

The [Llama 4 release](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
discloses a light SFT, online RL, and light DPO pipeline with easy-example
filtering. **[D — model training]** It does not specifically disclose a
hallucination mitigation recipe.

**Still unknown:** current Meta AI model routing, search ranking, evidence
sufficiency, citation checking, live thresholds, and whether the Llama 3
knowledge-probe method continues in later checkpoints.

## 7. Amazon and AWS

### Amazon Nova

The [Amazon Nova technical report and model
card](https://www.amazon.science/publications/the-amazon-nova-family-of-models-technical-report-and-model-card)
describes quality-filtered public, licensed, and proprietary data; SFT; a human
preference reward model; DPO/PPO; truthfulness and robustness data in
responsible-AI work; feedback of risk examples into later SFT/RLHF; and runtime
input/output moderation. **[D — model training]**

The report also evaluates:

- erroneous tool calls when no applicable tool exists through the BFCL
  Irrelevance slice; and
- a specific RAG experiment on CRAG with disclosed parsing, chunking, top-20
  retrieval, and GPT-4 Turbo judging.

These are useful evaluations. The report does not disclose a Phi-4- or
Llama-3-like general factual-abstention training recipe, and the CRAG
experiment is not proof of the Bedrock production retrieval stack.

### Bedrock and Amazon Q product controls

**[D — API/cloud control or product]**

- Bedrock
  [Contextual Grounding Checks](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-contextual-grounding-check.html)
  score whether a response is supported by the supplied source and relevant to
  the query, with configurable thresholds. AWS notes important scope limits:
  conversational QA is not the intended case, source-external additions are
  considered ungrounded, and streaming can expose text before a later check.
- Bedrock
  [Automated Reasoning](https://docs.aws.amazon.com/bedrock/latest/userguide/automated-reasoning-checks-concepts.html)
  translates natural-language policy into logical rules and uses a solver to
  check consistency. A valid result covers only the formalized policy; the
  translation and any out-of-policy fact may still be wrong.
- Bedrock Knowledge Bases
  [RetrieveAndGenerate](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_RetrieveAndGenerate.html)
  supports retrieval, reranking, query decomposition, and source-chunk
  citations.
- Amazon Q Business
  [hallucination reduction](https://docs.aws.amazon.com/amazonq/latest/qbusiness-ug/hallucination-reduction.html)
  evaluates RAG responses and may modify an answer when it detects a
  high-confidence inconsistency; it also provides granular source attribution.

**[U]** The detectors, thresholds, current model routes, and correction
algorithm are not publicly reconstructible.

## 8. Cohere

### Command A trains RAG and tool use as expert specializations

The [Command A Technical Report](https://cohere.com/research/papers/command-a-technical-report.pdf),
Sections 3.1–3.3, discloses an unusually explicit grounded-training pipeline.

**[D — model training]**

1. Train separate SFT experts for code, safety, RAG, math, multilingual work,
   and general long context, then merge them.
2. Train and merge corresponding RL experts.
3. Polish the merged model with best-of-$N$, offline preference optimization,
   and online RL.
4. For RAG and tool use, build trajectories containing reasoning, JSON tool
   calls, tool results, and a final answer that cites the tool output.
5. Use human and synthetic examples, SFT, and offline Cohere Preference
   Group Optimization; multi-reviewer annotation requires a majority.

The report evaluates a private enterprise RAG set with more than 10,000
snippets and human ground truth, including answerable and unanswerable cases.
Vendor-reported results are useful for the named system, not an independent
universal RAG ranking.

Cohere's own [RAG documentation](https://docs.cohere.com/v1/docs/retrieval-augmented-generation-rag)
states that retrieval and citations cannot guarantee accuracy or eliminate
hallucination; sources may be stale, wrong, or biased. **[D — API guidance]**

`strict_tools` and
[Structured Outputs](https://docs.cohere.com/v2/docs/structured-outputs)
constrain schemas, tool names, arguments, and types. **[D — API control]** They
do not verify factual values or faithful use of tool results.

**Still unknown:** the RAG data volume, citation reward, full loss composition,
current product retrieval, and online claim-level error rates.

## 9. Mistral AI

### Model and product disclosures

Mistral's [Large 2 release](https://mistral.ai/news/mistral-large-2407/)
states that fine-tuning focused on reducing hallucinations and making the model
more cautious and discerning. **[D — model training]** No data construction,
loss, refusal threshold, factuality benchmark, or ablation is disclosed, so the
claim cannot support a more specific recipe.

Mistral API documentation says its models are trained to answer from documents
and expose source-bound
[citations](https://docs.mistral.ai/studio-api/conversations/citations).
Le Chat combines pretrained knowledge with web/news information and citations;
the [AFP partnership](https://mistral.ai/news/mistral-afp/) supplies a
verifiable news archive. **[D — API/product]**

Mistral's 2026
[Search Toolkit](https://mistral.ai/news/search-toolkit/) separates ingestion,
retrieval, and evaluation and exposes hybrid lexical/vector search. Its
[RAG-judge guide](https://mistral.ai/fr/news/llm-as-rag-judge/) evaluates
context relevance, groundedness, and answer relevance. **[D — customer method
and product guidance]** These sources do not establish the exact live Le Chat
ranker or verifier.

Structured output constrains JSON Schema. **[D — API control]** It does not
prove factual content.

**Still unknown:** factuality training details, live source ranking,
citation-entailment verification, routing, and thresholds.

## 10. xAI

### Grok factuality post-training and production-distribution evaluation

xAI's [Grok 4.1 release](https://x.ai/news/grok-4-1) says post-training
specifically targeted factual hallucinations for information-seeking prompts.
It evaluates stratified real production queries and a 500-biography FActScore
set with web search enabled, decomposing answers into atomic claims and
separating major and minor factual errors. **[D — model training and
evaluation]**

The complete training data, reward, abstention objective, and causal ablation
are **U**.

### Search, sources, and a released production prompt

xAI released a revision-pinned
[Ask Grok system prompt](https://github.com/xai-org/grok-prompts/blob/a7c186f5ccac95875c0041aed60398f6ecb6d6c7/ask_grok_system_prompt.j2)
that instructs the product to use real-time search for facts and primary
sources, open pages to verify search information, diversify sources on complex
or disputed topics, and express uncertainty. **[C — product artifact]** A real
prompt is stronger evidence than generic advice, but it does not prove
compliance or source correctness.

The older [Grok-1 model card](https://x.ai/news/grok/model-card) already states
that deployment uses search and databases to improve factuality while external
information still does not eliminate hallucination. Current APIs expose
[web search](https://docs.x.ai/developers/tools/web-search),
X search, code execution,
[collection retrieval](https://docs.x.ai/developers/tools/collections-search),
and [citations](https://docs.x.ai/developers/tools/citations).
**[D — API/product]** The citation documentation notes that listed URLs are
sources encountered by the model; that alone does not prove every URL informed
the final claim.

Structured output and strict tool arguments constrain protocol, not semantic
truth. Safety refusal and deception evaluations in Grok system cards are also
not interchangeable with open-world factual hallucination.

**Still unknown:** live search/reranking internals, citation entailment, route
selection, verifier thresholds, and incident policy.

## 11. NVIDIA

### A detailed tool-hallucination experiment that was not the release recipe

The [Nemotron 3 Nano Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf),
Appendix C, defines tool hallucination as invoking a tool that was not declared.
It samples 32 on-policy solutions for 1,000 mathematics and 1,000 STEM
questions, then constructs about 50,000 preference examples in which undeclared
tool calls are rejected. **[D — research experiment]**

The DPO ablation reduced the report's AIME25 tool-hallucination rate from 1.25%
to zero and GPQA from 8.33% to 0.7%, while improving accuracy. The report
explicitly says the final released model did **not** use that DPO recipe because
RL produced similar results. This is strong method evidence and weak evidence
for the final checkpoint recipe; it covers tool calls, not general factuality.

### NeMo Guardrails

NeMo Guardrails exposes
[fact-checking rails](https://docs.nvidia.com/nemo/guardrails/configure-guardrails/guardrail-catalog/fact-checking)
that compare a RAG answer with retrieved chunks, use a self-check prompt or
specialized models, apply thresholds, and optionally fail closed when a check
does not complete. It also supports consistency-oriented hallucination
evaluation. **[D — open customer-side control]**

This does not prove that NVIDIA-hosted models use the rail by default or that
the selected judge reliably follows it.

## 12. DeepSeek

### Verifiable search questions and generative reward models

The [DeepSeek-V3.2
report](https://arxiv.org/abs/2512.02556), Section 3.2.3, describes a search
training-data pipeline aimed at long-tail entities. One agent searches and
generates questions; multiple answer agents respond; a verifier with search
access checks answers over several rounds. A sample is retained only when the
correct answer can be verified, wrong candidates can be falsified, and search
materially helps. The disclosed corpus has **50,275 search tasks**. **[D —
model training]**

DeepSeek trains specialists for search, general agents, and code, then mixes
them in Group Relative Policy Optimization (GRPO). Verifiable tasks use
rule-based outcome, length, and language-consistency rewards; open tasks use
per-prompt rubrics and a generative reward model (GenRM). The search objective
explicitly includes factual reliability. Training connects to a real web
search API, code environment, and Jupyter while preserving reasoning and tool
history across calls. **[D — model training]**

The [DeepSeek-V4 report](https://arxiv.org/abs/2606.19348) further describes
rule/test rewards for easy-to-verify tasks and rubric-guided RL plus a
reinforcement-trained generative judge and online multi-teacher distillation
for harder tasks. Its service description says the web/app uses primarily RAG
in non-thinking mode and iterative search/fetch in thinking mode. **[D —
model training and product]** The latter does not establish behavior for every
open checkpoint or API call.

DeepSeek's [Anthropic-compatible API
documentation](https://api-docs.deepseek.com/guides/anthropic_api/) says the
generic `citations` field is ignored and `search_result` input is unsupported,
although a web-search tool result is supported. **[D — API boundary]** This
prevents an unsupported claim that the compatibility layer offers a complete
citation contract.

**Still unknown:** epistemic-refusal thresholds, calibrated confidence,
claim-level citation rewards, production hallucination rates, citation
coverage, and live alerting.

## 13. Alibaba Qwen

### Factuality verification and a RAG-specific reward

The [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) describes more
than one million post-training examples. Offline RL factuality verification
uses structured data, code sandboxes, instruction verifiers, and multiple
critics; the online reward model includes truthfulness, defined to include
factual accuracy, contextual support, and absence of false or unsupported
content. **[D — model training]**

The [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), Section 4.4,
states that general RL spans more than 20 task families and that agents
interact with real environments. For RAG it introduces a task-specific reward
that favors an accurate answer adapted to the retrieved context in order to
minimize hallucination. The reward stack includes deterministic rules, a
reference-aware Qwen2.5-72B judge, and a reference-free scalar reward model.
**[D — model training]** This is one of the most direct published
RAG-specific anti-hallucination reward statements.

For a reasoning stage, Qwen3 retains only verifiable questions and removes
wrong, duplicate, guessed, internally inconsistent, and language-mixed
trajectories; the disclosed set contains 3,995 query–verifier pairs. This
small, curated stage is not the model's entire post-training set.

### End-to-end research agents and product citations

[Tongyi DeepResearch](https://arxiv.org/abs/2510.24701) uses Search, Visit,
Python, Scholar, and file-parsing tools. Agentic continued pretraining expands
the context from 32K to 128K; the team samples a web knowledge graph, injects
uncertainty, synthesizes verifiable questions, and runs on-policy GRPO using
binary correct-answer rewards, continually removing universally easy or
impossible tasks. The
[official repository](https://github.com/Alibaba-NLP/DeepResearch) is a
released artifact. **[D/C — model training and artifact]**

The Alibaba Cloud [Web Search
Agent](https://help.aliyun.com/en/model-studio/web-search-agent-guide) exposes
source filtering, query rewriting, vertical sources, automatic or forced
search, and optional numbered source links. **[D — product]** This does not
show that the base checkpoint learned citation entailment.

Qwen2.5 explicitly cautions that a reward model performing well on a standalone
benchmark may still perform poorly when optimized downstream because of
Goodhart-style overoptimization. **[D — disclosed limitation]**

**Still unknown:** evidence-based refusal thresholds, calibrated confidence,
citation-coverage rewards, production hallucination telemetry, and current
online gates.

## 14. Moonshot AI / Kimi

### Sentence-level unsupported-claim detection inside the reward

The [Kimi K2 Technical Report](https://arxiv.org/abs/2507.20534) describes more
than 3,000 real Model Context Protocol (MCP) tools, about 20,000 synthetic
tools, thousands of agents, stateful simulation, and real sandboxes. Rubric
judges filter trajectories. **[D — model training]**

More unusually, K2 trains a sentence-level faithfulness judge to detect factual
claims unsupported by the provided context and uses that judge directly as a
reward model. A self-critique rubric separately evaluates helpfulness,
reasoning, factuality, and safety. Online rollouts from verifiable tasks
continually update the critic, while math, code, and instruction-following use
answer verifiers, tests/sandboxes, or rules plus model judges. **[D — model
training]** This is stronger evidence than a generic claim that the product
“uses RAG.”

The report also discloses an important failure mode: preference rubrics may
favor confident, singular answers and penalize hedging or qualification,
creating overconfidence. Calibrated uncertainty is listed as future work.
**[D — disclosed limitation]**

### Search-agent reinforcement learning

[Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/) connects to
live search, a text browser, and code tools; automatically synthesizes and
validates questions; filters ambiguous, wrong, and overly easy tasks; and uses
REINFORCE with final-answer correctness, allowing more than 50 interaction
steps. **[D — research/product model training]** The reward concerns the final
answer, not necessarily every citation.

Kimi's [Search product
documentation](https://www.kimi.com/help/features/search) says Agentic Search
is built with end-to-end agentic RL, decides when to search, filters for
relevance, authority, and freshness across more than 100 vetted source types,
and attaches source links to search-based answers. **[D — product claim]** The
public record does not provide a citation-entailment reward or an independent
reproduction.

K2 reports offline grounding and faithfulness evaluations including FACTS
Grounding, HHEM/Vectara, and FaithJudge. **[D — evaluation]** Production
claim-level monitoring, calibrated output probabilities, and citation
precision remain **U**.

## 15. Zhipu AI / GLM

### Multi-page search data and trajectory-level outcome rewards

The [GLM-4.5 Technical Report](https://arxiv.org/abs/2508.06471), Section 3.3,
constructs search questions from multi-hop knowledge graphs. Human annotators
extract and selectively mask information across multiple pages so that the
answer requires cross-page search. **[D — model training]**

The entire search trajectory is rewarded by final-answer accuracy; a malformed
tool call terminates the trajectory with zero reward. The team iterates
self-distillation, and the reported ablation shows that allowing more browsing
rounds continues to improve BrowseComp. Holistic RL also includes factual
correctness in human feedback and varies AI-judge rubrics according to whether
objective ground truth exists. **[D — model training]**

Zhipu's [Web Search API](https://docs.z.ai/guides/tools/web-search) returns
structured title, URL, and summary fields and can prompt a model to emit
`[Source: ref_n]`. **[D — API/product]** This exposes inspectable sources but
does not establish that every generated claim is entailed.

The [GLM-4.5V/4.6V
report](https://arxiv.org/abs/2507.01006) states a critical limit of
outcome-only rewards: a coincidentally correct final answer can reinforce a
hallucinated intermediate rationale, motivating process-level hallucination
detection. It also reports reward hacking and collapse under weak verifiers.
**[D — disclosed limitation]**

**Still unknown:** epistemic abstention, probability calibration, citation
coverage, production claim-level rates, and live monitoring.

## 16. Baidu ERNIE

The [ERNIE 5.0 report](https://arxiv.org/abs/2602.04705) describes a unified
verifier, an unbiased replay buffer, training–inference consistency controls,
success/entropy-aware masking of already learned samples, and annealed
step-by-step hints for difficult zero-reward tasks. It reports SimpleQA,
ChineseSimpleQA, and BrowseComp-ZH and attributes post-training gains to
factual recall and answer calibration. **[D — model training/evaluation]**
It does not disclose a fact-specific verifier or a calibrated-refusal
algorithm.

Baidu Cloud's [knowledge-base
documentation](https://cloud.baidu.com/doc/qianfan/s/Imh4stpo0) presents RAG
as a product control for stale knowledge and hallucination. **[D —
API/cloud control]** Claim-level rewards, refusal thresholds, and production
monitoring remain **U**.

## 17. Tencent Hunyuan

The [Hunyuan-A13B Technical
Report](https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan-A13B/main/report/Hunyuan_A13B_Technical_Report.pdf),
Section 3.2, provides one of the most explicit Chinese-vendor recipes.
Knowledge-QA data passes multiple validation and critic filters; long-context
training uses a hallucination-focused reward model plus online RL; knowledge
QA combines hallucination detectors with and without reference answers and a
user-experience model; finance, law, and medicine use consistency rewards to
find unstable samples. The pipeline spans 16 subtasks and more than 30 scoring
services. **[D — model training]**

The report evaluates end-to-end RAG on FRAMES. Tencent Cloud's
[TokenHub search
documentation](https://cloud.tencent.com/document/product/1823/132358)
returns search results and source annotations, but permits a fallback to the
model's internal knowledge when search fails. **[D — product]** A search
switch therefore does not guarantee that a particular response was grounded.

The reward functions, citation entailment, calibrated refusal, and production
monitoring thresholds remain **U**.

## 18. ByteDance Seed

The [Seed-Thinking-v1.5
report](https://arxiv.org/abs/2504.13914) removes bad questions, escalates
reference answers disputed by multiple strong models to experts, uses
Seed-Verifier for semantic equivalence, and applies a separate
Seed-Thinking-Verifier to reward-hacking and uncertain-boundary cases.
Non-verifiable tasks use pairwise generative reward modeling. **[D — model
training]**

The report says SFT reduces hallucination, but its SimpleQA result is 12.9% and
it explicitly identifies factual memory as weak. The [Seed2.0 Model
Card](https://arxiv.org/abs/2607.00248) expands factual, long-form factuality,
and search evaluations without enough detail to carry the Seed 1.5 recipe
forward. **[D — evaluation; U — transfer to a later recipe]** Search-grounded
rewards, citation training, refusal thresholds, and production telemetry are
not public.

## 19. MiniMax

The [MiniMax-01 report](https://arxiv.org/abs/2501.08313) describes a
truthfulness pipeline of multi-response sampling, claim decomposition and
clustering, crowd verification, LLM comparison, and truthfulness scoring. It
also trains a search router with SFT and removes questions the model already
knows, calibrating search decisions around the checkpoint's knowledge
boundary. **[D — model training]**

The [MiniMax-M2 series
report](https://arxiv.org/abs/2605.26494) requires multiple sources and
cross-checking for open-web tasks. Each problem has an evidence specification;
a trajectory is accepted only when its answer actually rests on retrieved
evidence rather than model memory. Report rubrics also score transparency,
uncertainty, and risk disclosure. **[D — model training]**

MiniMax's [prompting
guide](https://platform.minimax.io/docs/token-plan/prompting-best-practices)
recommends allowing refusal, requiring quotations/citations, and constraining
source, time, and version. **[D — deployment guidance]** It is not evidence of
intrinsic calibration in every checkpoint.

Citation-entailment rewards, numerical confidence, online refusal thresholds,
and production hallucination monitoring remain **U**.

## 20. How to compare a new vendor claim

For every future disclosure, record:

1. **object:** checkpoint training, research prototype, product, API, cloud
   option, or evaluation;
2. **model and date:** immutable version, mode, tools, and verified-through
   date;
3. **failure class:** open-world fact, supplied-context grounding, citation,
   tool, multimodal, code, or calibration;
4. **intervention:** data, loss/reward, retrieval, tool, decoder, verifier,
   abstention, UI, monitoring, or human control;
5. **measurement:** answerable/unanswerable mix, abstention policy, atomic
   claims, sources, judge, repeated runs, and budget;
6. **trade-off:** accuracy, wrong-answer rate, coverage, cost, latency, and
   over-refusal;
7. **causal evidence:** matched ablation or only a bundled release comparison;
8. **residual failures:** what the source explicitly says remains; and
9. **unknowns:** details that public evidence cannot establish.

This prevents a customer feature from becoming a claim about proprietary
training, a benchmark from becoming a production guarantee, or a citation
badge from becoming proof of truth.
