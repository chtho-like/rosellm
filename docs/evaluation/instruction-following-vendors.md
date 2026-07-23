# What Leading Model Vendors Disclose About Instruction Following

**Verified through:** 2026-07-23.

This is an evidence audit, not a ranking. It asks which instruction-following
controls a developer has publicly confirmed for a named model, research
prototype, product, or customer-facing API. It does not infer proprietary
training from a benchmark score or a product feature.

Evidence labels follow the [research standard](../research-method.md):

- **D — disclosed:** a first-party source or author paper states the practice;
- **C — confirmed artifact:** released code, prompt, configuration, data, or
  weights directly establish it;
- **R — reproduced:** RoseLLM retained the run, configuration, and artifacts;
- **I — inferred:** the conclusion depends on named evidence and assumptions;
- **U — unknown:** public evidence is absent or insufficient.

The object label is equally important:

- **model training** changes a checkpoint;
- **research prototype** demonstrates a method but is not automatically
  deployed;
- **evaluation** measures behavior but does not reveal its cause;
- **API or decoding control** constrains inference around a model;
- **product orchestration** adds prompts, routing, memory, tools, and policy;
- **customer customization** is a capability offered to builders, not evidence
  about the vendor's base model.

A schema-constrained API does not prove that the model weights learned perfect
format following. A tool benchmark does not establish domain authorization.
A product prompt does not prove compliance. A release score does not reveal
the SFT data, reward, or live router.

## 1. Cross-vendor conclusions

### The publicly visible industry stack

Leading developers repeatedly expose seven layers:

1. broad instruction SFT over real, licensed, human, and synthetic tasks;
2. targeted data for complex constraints, system prompts, multiple turns,
   tools, failure recovery, and negative examples;
3. preferences, rejection sampling, DPO-family optimization, or online RL;
4. rule, execution, or model-based rewards for objectively and semantically
   judged requirements;
5. specialist training, curricula, merging, or distillation;
6. role channels, schemas, constrained decoding, tool controls, memory, and
   agent harnesses; and
7. private evaluations, partner feedback, production failures, canaries, and
   regression gates.

The exact data mixture, reward composition, hierarchy loss, online router,
compaction policy, retry budget, release threshold, and production violation
rate are usually **U**. Public evidence supports this layered pattern, not a
complete reverse-engineered stack.

This edition audits 27 named developers with sufficiently specific primary
evidence: OpenAI, Anthropic, Google, Meta, Microsoft, Amazon, Cohere, xAI,
Mistral, NVIDIA, Alibaba, Moonshot, Zhipu, Tencent, DeepSeek, Baidu,
ByteDance, MiniMax, Apple, IBM, AI21 Labs, 01.AI, Baichuan, Huawei, iFlytek,
StepFun, and Xiaomi. Section 29 records several prominent cases where a
specific instruction-following recipe remains unknown. Coverage is broad, not
a claim that every commercial model developer has been exhaustively audited.

### Particularly concrete disclosed examples

| Vendor / model | Disclosed practice | Why it is unusually informative |
|---|---|---|
| OpenAI GPT-4.1 | developer-feedback tasks for format, negative, ordered, content, ranking, and unavailable-data behavior | links named practical failure classes to training and private evaluation |
| OpenAI Instruction Hierarchy / IH-Challenge | synthesize or programmatically grade role conflicts and train selective obedience | targets following the correct source, not maximum obedience |
| Meta Llama 3 | repeated SFT, reward-model candidate selection, edited preferences, DPO, system-prompt and tool curricula | exposes an iterative data-generation and preference loop |
| Meta Llama 4 | Maverick uses lightweight SFT, multimodal online RL with medium-to-hard filtering, and lightweight DPO; Behemoth separately uses a pass-at-$k$ curriculum and mixed capability batches | distinguishes the released model's recipe from its still-training teacher |
| Google Gemini 1 | representative real-world single/multi-turn prompts, SFT demonstrations, feedback, reward model, and RLHF | ties a deployed family to a conventional but named post-training pipeline |
| Cohere Command A | a general instruction initializer plus six capability specialists, parameter merging, best-of-$N$, offline preference, and online RLHF | exposes a detailed general-to-specialist merge and regression strategy |
| DeepSeek V4 | dedicated instruction-following expert with SFT and GRPO, then on-policy distillation into a unified model | separates precise adherence from general reasoning specialists |
| Alibaba Qwen | broad SFT followed by multi-stage RL with instruction-specific rewards and verifiers | exposes a scalable rule/judge reward path for a released family |
| Moonshot Kimi K2 | rubric-based RL Gym with instruction, tool, and environment feedback | connects complex agent behavior to named reward and verification layers |
| Tencent Hunyuan | many task-specific scoring services and instruction or hallucination reward models in online RL | demonstrates multi-scorer specialization rather than one universal reward |
| Apple 2024 foundation models | teacher-committee rejection sampling plus mirror-descent RLHF with a leave-one-out advantage estimator | directly attributes significant instruction-following gains to two named post-training algorithms |

### Important counterexamples

- Microsoft Phi-4's DPO stages improved many reasoning qualities while its
  reported IFEval score fell from the SFT checkpoint. Preference optimization
  does not automatically improve literal compliance.
- Google found that standard instruction tuning could make a model less
  willing to obey deliberately arbitrary label mappings. Strong semantic
  priors and local in-context instructions can conflict.
- A grammar can deliver 100% supported-schema conformance while the model still
  selects the wrong tool, supplies a wrong value, or violates authorization.
- More reasoning can help complex composition and hurt short literal tasks.
- Strong safety or hierarchy training can over-refuse; preference tuning can
  favor verbose helpfulness over terse obedience.
- Partner testimonials, arena preference, and aggregate chat-quality scores are
  not instruction-level evaluations.

## 2. OpenAI

### Developer-derived instruction data and evaluations

OpenAI's [GPT-4.1 release](https://openai.com/index/gpt-4-1/) says training and
private evaluation drew from developer feedback in six categories:

- format following;
- negative instructions;
- ordered instructions;
- content requirements;
- ranking; and
- overconfidence or abstention when data is unavailable.

It also targeted retrieval from earlier conversation turns and multi-turn
retention. **[D — model training and evaluation]**

On OpenAI's reported settings, GPT-4.1 scored 87.4% on IFEval versus 81.0% for
GPT-4o, improved the default-grader MultiChallenge result from 27.8% to 38.3%,
and improved the private hard API instruction evaluation from 29.2% to 49.1%.
In an internal coding-edit test, unrelated edits fell from 9% to 2%. These are
vendor evaluations for named settings, not current cross-vendor prevalence
estimates.

[GPT-5 for developers](https://openai.com/index/introducing-gpt-5-for-developers/)
says the model was trained with real coding tasks from early customers and
improved detailed instruction following, tool-error handling, sequential and
parallel tool use, and progress preambles. OpenAI reports 69.6% on
MultiChallenge with an o3-mini grader, 64.0% on its private hard evaluation,
and 99.0% on COLLIE. **[D — model training and evaluation]** The release also
shows that API GPT-5 modes and ChatGPT routing are different objects.

### Train the model to follow the correct source

The 2024 [Instruction
Hierarchy](https://openai.com/index/the-instruction-hierarchy/) research
generates privileged-role conflicts and trains GPT-3.5 with SFT and
reinforcement learning from human feedback to ignore lower-priority
instructions selectively. It improved both trained and unseen attacks with
small general-capability costs. **[D — research prototype]**

The o1 [System Card](https://openai.com/index/openai-o1-system-card/) says o1
training included role conflicts with the order system over developer over
user. Reported role-conflict scores improved over GPT-4o, including
system/developer and system/user cases, while a phrase-protection slice
regressed. **[D — model training/evaluation]** This is evidence that hierarchy
training has trade-offs, not a proof of perfect priority handling.

The 2026
[IH-Challenge](https://openai.com/index/instruction-hierarchy-challenge/)
uses role-conflict tasks that Python can grade and that cannot be solved by
refusing everything. Large-scale RL on a GPT-5 Mini research variant improved
held-out hierarchy, safety steerability, CyberSecEval 2, and internal prompt
injection evaluations. **[D — research prototype]** OpenAI does not state that
every current production model uses this exact dataset or reward.

### Training plus inference constraints

OpenAI's [Structured
Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)
disclosure separates two effects:

- schema-understanding training raised `gpt-4o-2024-08-06` to 93% on an
  internal complex-schema evaluation; and
- dynamic constrained decoding raised supported-schema conformance to 100%.

**[D — model training plus API/decoder]** The decoder guarantees supported
shape, not semantic values, task choice, tool relevance, or authorization.

GPT-5 also exposes reasoning effort, verbosity, custom plaintext tools, and
regular-expression or context-free-grammar constraints. **[D — API controls]**
An explicit user constraint is documented to override the general verbosity
setting.

**Still unknown:** current GPT-5.6 SFT data, preference and RL rewards, mixture
weights, hierarchy loss, production feedback selection, routing, compaction,
and live violation rates. The Model Spec is a behavioral standard, not by
itself proof of a training recipe.

## 3. Anthropic

### Constitutional and character training

[Constitutional
AI](https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback)
first has a model critique and revise answers under written principles, then
uses AI comparisons to train a preference model and RLAIF. **[D — historical
research recipe]** Its main objective was harmlessness with less evasiveness;
Anthropic says current processes have evolved.

[Claude's Character](https://www.anthropic.com/research/claude-character)
describes a Claude 3 training stage in which researchers define desired
character traits, generate synthetic situations and responses, have the model
rank responses, train a preference model, and iteratively review the result.
**[D — model training]** This shapes default behavior and judgment; it is not a
hard guarantee for every literal constraint.

The current Claude constitution further states norms for helpfulness, honesty,
calibration, and appropriate deference. Exact current SFT, preference, and RL
composition remain **U**.

### Tool context, reflection, and constrained output

Anthropic's [Advanced Tool
Use](https://www.anthropic.com/engineering/advanced-tool-use) is
customer-visible inference and context engineering:

- Tool Search loads only relevant tools dynamically;
- Tool Use Examples supply realistic valid invocations;
- programmatic tool calling compresses intermediate work.

On Anthropic's internal MCP evaluations, Tool Search improved Opus 4 from 49%
to 74% and Opus 4.5 from 79.5% to 88.1%; examples improved a complex-parameter
test from 72% to 90%. **[D — API/product evaluation]** These gains do not
disclose base-model training.

The [think tool](https://www.anthropic.com/engineering/claude-think-tool)
creates an explicit intermediate reflection point for policy-heavy and
sequential tool tasks. Anthropic later recommends extended thinking for most
general cases and retains the tool for selected policy workflows. **[D —
inference/product control]**

Claude [strict tool
use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)
uses grammar-constrained sampling to guarantee supported tool names and
argument schemas. **[D — API/decoder]** It does not select the correct tool or
validate field meaning.

### Evaluate outcomes and traces

Anthropic's [agent evaluation
guide](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
recommends repeated trials, complete traces, outcome/state graders, and
separating model failures from harness failures. It warns that a grader
requiring one exact operation sequence can reject a valid alternative path.
**[D — production evaluation guidance]**

The [Claude Opus 4.7
release](https://www.anthropic.com/news/claude-opus-4-7) describes more literal
behavior and warns that prompts tuned to older models may require adjustment.
Partner-reported workflow and tool-error gains are useful deployment signals,
not neutral instruction-following measurements.

**Still unknown:** current Claude 4.x instruction SFT data, RLAIF/RL rewards,
hierarchy training, role-conflict objectives, product prompts, and online
release thresholds.

## 4. Google and Google DeepMind

### FLAN and the general instruction-tuning recipe

Google's [FLAN
work](https://research.google/blog/introducing-flan-more-generalizable-language-models-with-instruction-fine-tuning/)
templates many tasks as natural-language instructions and holds out task
clusters to test unseen-task transfer. The later [FLAN
Collection](https://research.google/blog/the-flan-collection-advancing-open-source-methods-for-instruction-tuning/)
mixes zero-shot, few-shot, chain-of-thought, and inverted-input templates while
balancing task sources. **[D — research/open model training]**

These studies establish task diversity, template diversity, scaling, and
mixture balance as real instruction-tuning levers. They are not a full recipe
for current Gemini.

### Gemini post-training

The [Gemini 1 Technical
Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)
describes:

1. representative real-world single- and multi-turn prompts from vendor,
   licensed, and synthetic sources;
2. human demonstrations for supervised fine-tuning;
3. multiple candidate answers and human feedback;
4. a reward model; and
5. reinforcement learning from human feedback.

**[D — model training]** This is direct evidence for a deployed family, though
the quantities, mixture, losses, and complete rewards are not public.

The [Gemini 1.5
report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)
adds data filtering, conditional pretraining tags, SFT, and RLHF. Its
helpfulness objective is to fulfill a request safely whenever possible and
refuse only when no policy-compliant answer exists. Long-context safety tests
include prompt injection around a needle. **[D — model training and
evaluation]**

The [Gemini 2.5
report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
says thinking models learn to use inference compute through RL, Gemini 2.0 was
trained for native tools and search, and 2.5 learns interleaved reasoning,
search, follow-up queries, and verification. Safety behavior uses custom data
inspired by Constitutional AI with human revision of incorrect refusals and
violations. **[D — model training]**

### Tool and structured-output protocols

Gemini [function
calling](https://ai.google.dev/gemini-api/docs/generate-content/function-calling)
exposes:

- `ANY`, which requires at least one schema-constrained function call;
- `VALIDATED`, which permits text or a constrained call;
- `AUTO`, which lets the model choose; and
- `NONE`, which disables calls.

Gemini [structured
output](https://ai.google.dev/gemini-api/docs/generate-content/structured-output)
constrains JSON syntax and supported schema. **[D — API/decoder]** Neither
guarantees semantic values, tool relevance, or business permission.

For Gemini 3 function-calling workflows, Vertex [thought
signatures](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures)
from required prior steps must be returned unchanged; missing required
signatures can produce HTTP 400, while the official SDKs handle them
automatically. **[D — API protocol]** This is context orchestration, not a
checkpoint-training disclosure.

### A useful instruction-tuning trade-off

Google reports that FLAN-PaLM was less willing than PaLM to obey deliberately
arbitrary label mappings in context; [symbol
tuning](https://research.google/blog/symbol-tuning-improves-in-context-learning-in-language-models/)
uses semantically meaningless labels to reduce reliance on prior task
semantics. **[D — research]** Standard instruction tuning can strengthen the
model's interpretation of what a task “should” mean while weakening obedience
to an unusual local mapping.

**Still unknown:** Gemini 3/3.5 instruction data, reward weights, hierarchy
algorithm, live product routing, memory/compaction, and release thresholds.

## 5. Meta

### Llama 3: iterative SFT, candidate selection, and DPO

The [Llama 3 model-family report](https://arxiv.org/abs/2407.21783) describes
six post-training rounds. Each uses SFT and DPO, with new preference data and
model outputs feeding later rounds. **[D — model training]**

The disclosed loop includes:

- human pairwise ranking of multi-turn answers;
- frequent human editing of the preferred answer, yielding
  `edited > chosen > rejected`;
- a reward model;
- about 10–30 candidates per prompt for rejection sampling;
- SFT on selected outputs;
- DPO, selected over PPO partly because it was cheaper and gave better IFEval;
- masking special formatting tokens in the DPO loss; and
- a 0.2 next-token negative-log-likelihood term for stability.

Later data includes system prompts controlling length, format, tone, and
persona. Annotators test consistency across multiple turns, and the resulting
preferences feed the reward model, rejection sampling, SFT, and DPO.

### Tool and function-call curricula

Llama 3 tool training combines human message-level labels and synthetic
trajectories, moving from single calls to dialogue and multi-step use. It
includes:

- tool versus no-tool negatives;
- function documentation;
- nested, parallel, and multi-turn calls;
- Python, Wolfram Alpha, and search; and
- final responses based on results.

**[D — model training]** The report says rejection sampling did not help this
tool data, so it was not used there. That negative result prevents the generic
claim that every good post-training component helps every domain.

### Llama 4

The [Llama 4
release](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) says
Maverick used lightweight SFT, multimodal online RL with continuous
medium-to-hard prompt filtering, and lightweight DPO. **[D — released-model
training]**

Separately, the still-training Behemoth teacher uses pass-at-$k$ difficulty
curricula, dynamically removes zero-advantage prompts, mixes capability
batches, and varies system instructions in reasoning and code training to
preserve instruction following. **[D — teacher model training]** These details
must not be projected onto every Scout or Maverick stage.

Prompt Guard and Llama Guard are separate product/open components. They do not
prove an instruction hierarchy inside Llama weights.

**Still unknown:** Llama 4 data and reward weights, live Meta AI prompts and
routing, hierarchy training, product memory, and online violation thresholds.

## 6. Microsoft

### Phi-4 shows that DPO is not automatically literal instruction training

The [Phi-4 Technical
Report](https://www.microsoft.com/en-us/research/publication/phi-4-technical-report/)
describes roughly eight billion post-training tokens in a broad ChatML SFT
mixture, followed by pivotal-token DPO and judge-guided DPO. The latter uses
about 850,000 preference pairs from GPT-4o, GPT-4 Turbo, and Phi candidates,
with a GPT-4o judge evaluating accuracy, style, and detail. **[D — model
training]**

Phi-4's reported IFEval score fell from 66.2 at SFT to 63.0 in the final model.
The report says strict instruction following was not the main priority,
training was mostly single-turn, and the model could be verbose. This is a
valuable counterexample: broad preference optimization and stronger reasoning
do not guarantee better exact adherence.

### Microsoft research prototypes

[WizardLM / Evol-Instruct](https://www.microsoft.com/en-us/research/publication/wizardlm-empowering-large-language-models-to-follow-complex-instructions/)
recursively increases instruction breadth, depth, complexity, and constraints
before SFT. **[D — research prototype]**

[Orca
AgentInstruct](https://www.microsoft.com/en-us/research/blog/orca-agentinstruct-agentic-flows-can-be-effective-synthetic-data-generators/)
uses specialized agents, tools, reflection, and verification across more than
100 subcategories to synthesize about 25 million pairs and releases one
million. Fine-tuning Mistral-7B produced broad reported gains but a substantial
summarization regression. **[D — research prototype]** Synthetic
specialization needs cross-domain regression tests.

Azure [Structured
Outputs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs)
is a hosted OpenAI control, not evidence about a Microsoft-trained base model.

**Still unknown:** current Microsoft closed-model instruction data, preference
or RL rewards, hierarchy training, and Copilot product routing.

## 7. Amazon and AWS

The [Nova 2 Technical
Report](https://cdn.amazon.science/c5/3d/84514a224666b5be6de4b43ef4aa/nova-2-0-technical-report2.pdf)
reports IFBench and MultiChallenge results but does not disclose the base
model's instruction SFT or RL recipe. **[D — evaluation; U — causal training
method]**

AWS offers [Nova 2
customization](https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-model-training-job.html)
in which customers can use multi-turn SFT for schema, policy, and tools, then
reinforcement fine-tuning with verifiable or model judges. The documentation
notes that SFT primarily changes behavior rather than acting as a reliable
knowledge-ingestion mechanism. Nova 1 also supports customer SFT and DPO.
**[D — customer customization]** This does not disclose the Nova base recipe.

Amazon's [Roles and Rules
research](https://www.amazon.science/publications/rnr-teaching-large-language-models-to-follow-roles-and-rules)
automatically builds diverse role/rule tasks from instruction datasets and
reports more than 25 percentage points of all-rule adherence improvement on
selected mixtures without a standard instruction-following regression.
**[D — research prototype]**

Bedrock
[ToolChoice](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_ToolChoice.html)
can allow automatic selection, require some tool, or require a named tool.
Nova structured-output guidance says constrained decoding reduced tool-use
format errors by more than 95%. **[D — API/decoder]** Correct structure does
not establish semantic applicability.

Amazon's [IHEval](https://www.amazon.science/publications/iheval-evaluating-language-models-on-following-the-instruction-hierarchy)
contains 3,538 examples over nine tasks and finds weak open-model performance
under conflicts. **[D — evaluation]** A benchmark is not a deployed fix.

**Still unknown:** Nova's actual instruction SFT, reward, hierarchy method,
production prompts, and release gates.

## 8. Cohere

### Command A builds specialists on a general instruction model

The [Command A Technical
Report](https://cohere.com/research/papers/command-a-technical-report.pdf)
discloses one of the most detailed current pipelines. **[D — model training]**

1. Train a general instruction-following Instruct initializer with SFT and
   offline preference optimization, selecting Self-Reward Preference
   Optimization (SRPO).
2. Train six SFT specialists: code, safety, RAG, math, multilingual, and
   general long context.
3. Combine them through a weighted linear parameter merge.
4. Train six corresponding RL specialists using pairwise or verifiable rewards
   and merge them.
5. Polish the unified model with best-of-$N$ SFT, offline preference
   optimization, and online RLHF.

The general initializer, rather than a seventh instruction specialist, supplies
the common instruction behavior. Synthetic enterprise prompts are sampled at
multiple temperatures, human-rated across turns, and rewritten when every
candidate is poor. System/preamble data explicitly varies response language,
JSON versus no Markdown, forbidden terms, and other controls before entering
both SFT and preference training.

### Reward, revision, and regression control

Cohere compares multiple preference objectives and selects its own Self-Reward
Preference Optimization (SRPO), which trains a generator and self-improver and
supports revision at inference. Its reward-model process relabels about four
million lower-quality pairs with an ensemble before using roughly 350,000
higher-quality examples.

All specialists share a general instruction-following initializer as a
cross-domain regularizer. Leave-one-out merges detect specialist conflicts,
especially long-context regressions. Final instruction preference pairs require
the chosen response to satisfy every criterion; one failed criterion makes a
candidate rejected. The polishing sequence alternates best-of-four SFT, offline
SRPO, and online Cohere Preference Group Optimization to limit regression and
reward hacking.

Reported IFEval and InFoBench numbers are vendor results and use settings that
are not uniformly comparable across every model row. They support the named
checkpoint, not a neutral universal ranking.

Command A's tool/RAG trajectories include custom instructions, reasoning,
sequential and parallel calls, results, citations, and multi-reviewer
preferences. **[D — model training]**

Cohere [Structured
Outputs](https://docs.cohere.com/docs/structured-outputs) and strict tools
constrain JSON shape, tool names, required arguments, and types. **[D —
API/decoder]** They do not validate semantic values or tool choice.

**Still unknown:** the complete instruction data, reward weights, parameter
merge weights, online thresholds, and production routing.

## 9. xAI

The [Grok-1 model card](https://x.ai/news/grok/model-card) says the model was
fine-tuned with substantial human and Grok-0 feedback but gives little
instruction-specific detail. **[D — high-level model training]**

[Grok 4.1](https://x.ai/news/grok-4-1) uses large-scale RL and frontier
agentic-reasoning models as reward models for non-verifiable style,
personality, helpfulness, and alignment. A silent live-traffic experiment and
blind pairwise user evaluation reported a 64.78% preference rate over the prior
production model. **[D — model training and product evaluation]** Preference
is not a pure instruction-following score.

[Grok 4.1 Fast](https://x.ai/news/grok-4-1-fast) is trained with RL in simulated
multi-tool environments across many domains for long-horizon, multi-turn work
and a two-million-token context. xAI reports 100% on the telecom slice of
$\tau^2$-bench and 72% on BFCL V4. **[D — model training and evaluation]**

[Grok Code Fast
1](https://x.ai/news/grok-code-fast-1) uses curated pull-request and coding
tasks, repeated partner feedback, and explicit grep, terminal, and file-editing
tools. **[D — specialist model training]**

The revisioned [grok-prompts
repository](https://github.com/xai-org/grok-prompts) exposes real product
instructions about high-authority policy, resisting override, and tool/search
behavior. **[C — product artifact]** It does not prove a weight-level hierarchy.

xAI [Structured
Outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs)
supports schema-constrained output and strict tool arguments, with documented
best-effort limitations for some features. **[D — API/decoder]**

**Still unknown:** Grok 4.x general instruction SFT/RL data, reward weights,
hierarchy training, live prompt selection, and product thresholds.

## 10. Mistral AI

The [Mixtral 8x7B
release](https://mistral.ai/news/mixtral-of-experts/) describes instruction
fine-tuning followed by DPO on paired feedback and characterizes the resulting
checkpoint as a careful instruction follower. **[D — model training]** The
instruction data, preference construction, and objective details are not
public.

[Mistral Large
2](https://mistral.ai/news/mistral-large-2407/) is described as substantially
better at precise instruction following and long multi-turn interaction, with
specific training for parallel and sequential function calls and cautious
behavior when information is insufficient. **[D — high-level model
training/behavior]** No data volume, loss, reward, or matched ablation is
disclosed.

Mistral's function-calling API can:

- allow automatic tool selection;
- require some tool;
- disable tools;
- require a named tool; and
- disable parallel calls.

The application must execute the call and return the actual result.
**[D — API/product control]**

Custom structured output can enforce a supported JSON Schema, while ordinary
JSON mode only guarantees valid JSON; Mistral's own limitation documentation
draws this distinction. **[D — API/decoder]**

Mistral's [prompt and skill
management](https://mistral.ai/news/manage-prompts-and-skills-in-studio/)
adds immutable versions, tests, approval, rollback, owners, labels, audit, and
telemetry lineage. **[D — production governance]** This can stabilize a
deployed system without changing the checkpoint.

**Still unknown:** current Large, Medium, and Ministral SFT/preferences/RL,
hierarchy training, production feedback, and live routing.

## 11. NVIDIA

### Nemotron 3 Nano: verified instruction data and unified RL

The [Nemotron 3 Nano Technical
Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
describes an instruction SFT pipeline that combines IFEval/IFBench-style
constraints with simulated single- and multi-turn users. Candidate answers from
several teacher models are retained only when every turn passes a verifier;
an LLM judge removes superficial or reward-hacking compliance. **[D — model
training]**

Its unified reinforcement learning includes:

- roughly 46,000 program-verifiable instruction tasks;
- roughly 3,000 MultiChallenge-inspired subtle multi-turn tasks judged by an
  LLM;
- roughly 9,000 JSON-Schema tasks whose reward checks exact schema but
  explicitly does not reward semantic content; and
- many other environments trained together to limit irreversible
  specialization regressions.

The report uses synchronous GRPO with masked importance sampling and discloses
the rollout batch structure. **[D — model training]** The JSON environment is
an unusually explicit statement of the syntax–semantics boundary.

NVIDIA also trains a large generative reward model to reason over two answers
before outputting helpfulness and a ranking, with position swapping to reduce
bias, then applies RLHF and group-relative length control to limit unnecessary
verbosity.

### Nemotron 3 Super and Ultra

The [Nemotron 3 Super Technical
Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
describes more than seven million SFT examples, long-context stages, unified
multi-environment RLVR, a separate software-engineering RL stage, and an RLHF
stage specifically improving instruction following, robustness, and
interaction quality. Multi-turn rubrics and over-refusal/jailbreak training are
included. **[D — model training]**

The [Nemotron 3 Ultra Technical
Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)
builds a dedicated instruction-following/factuality teacher through
domain-focused RLVR. Its mixture covers strict format, mid-conversation
instruction changes, long-horizon coherence, abstention, and RLHF; checks are
programmatic or model-based, and abstention reward is dynamically calibrated.
The specialist is then combined with general, agentic, STEM, chat, and other
teachers through asynchronous multi-teacher on-policy distillation. **[D —
model training]**

Vendor-reported IFBench and MultiChallenge gains support the named checkpoints
and protocols. They do not make scores across different harnesses automatically
comparable.

NeMo [Guardrails](https://docs.nvidia.com/nemo/guardrails/latest/about-nemo-guardrails-library/rail-types)
can apply input, retrieval, dialogue, execution, output, tool-argument, and
tool-result rails. **[D — customer-side product control]** Its effectiveness
depends on the chosen checkers and application evaluation; it is not an
intrinsic property of every NVIDIA model.

**Still unknown:** complete blends and reward weights, hosted configuration,
production routing, and live thresholds.

## 12. Alibaba Qwen

### Qwen3's four-stage post-training

The [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) discloses:

1. long-chain-of-thought cold start;
2. reasoning reinforcement learning;
3. thinking/non-thinking fusion; and
4. general reinforcement learning.

**[D — model training]**

Stage 3 non-thinking SFT explicitly includes code, mathematics, instruction
following, multilingual work, creative writing, question answering, and
role-play. Automatically generated checklists help evaluate examples.
Thinking-control markers are inserted at different turns so the latest
`/think` or `/no_think` instruction governs subsequent behavior.

Stage 4 spans more than 20 task families and directly optimizes content,
format, length, structured output, preference, agents, and RAG. Rewards combine:

- deterministic rules;
- a reference-aware Qwen2.5-72B-Instruct judge; and
- a reference-free scalar reward model trained from human preferences.

Agent data contains complete multi-turn interactions with real execution
feedback. **[D — model training]**

Qwen reports substantial IFEval, Multi-IF, ThinkFollow, and ToolUse gains
between stages for Qwen3-32B, alongside small regressions on some mathematics
and coding evaluations. Instruction and general-capability optimization must
be balanced.

### Qwen2.5 and coding agents

The [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) describes more
than one million SFT examples, about 150,000 DPO pairs selected with execution,
answer matching, and human/automatic checks, and online GRPO over truthfulness,
usefulness, brevity, relevance, harmlessness, and bias. **[D — model
training]**

[Qwen3-Coder](https://qwenlm.github.io/blog/qwen3-coder/) adds executable code
RL and long-horizon agent RL over 20,000 parallel environments. **[D —
specialist model training]**

The released [Qwen3
quickstart](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md)
confirms hard thinking-mode controls in the chat template, while Qwen-Agent
supports parallel, multi-step, and multi-turn functions. **[C — released
artifact/API]** These artifacts do not disclose the production hierarchy or
reward.

**Still unknown:** instruction sample counts and full taxonomy, reward weights,
annotation scale, production system hierarchy, and use of real user feedback.

## 13. Moonshot AI / Kimi

The [Kimi K2 Technical Report](https://arxiv.org/abs/2507.20534) describes an
agent SFT factory with:

- more than 3,000 real Model Context Protocol tools;
- more than 20,000 synthetic tools;
- thousands of system-prompt-defined agents;
- task-specific rubrics;
- user-persona and stateful tool simulators;
- injected success, partial failure, and boundary conditions;
- real code sandboxes; and
- large-scale rejection sampling that retains trajectories passing model
  judges.

**[D — model training]**

Instruction RL uses a hybrid verifier:

- code or deterministic checks for length, style, and formal constraints;
- an LLM judge for semantic constraints; and
- a separate hack check for answers that claim compliance without actually
  satisfying the rubric.

Instruction generators combine expert-conditioned prompts/rubrics,
AutoIF-style augmentation, and targeted generation of current-model failures
and edge cases. Subjective alignment uses a self-critic with core,
anti-hacking, and human-specific rubrics, first bootstrapped with preference
SFT and then updated from on-policy verifiable rollouts. **[D — model
training]**

K2 also uses task-specific token budgets, truncation penalties, an auxiliary
high-quality pretraining loss to limit forgetting, and temperature decay.

The released [K2 Vendor
Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier) compares 4,000 tool
requests across vendor APIs and exposes deployment concerns such as tool-call
ID normalization and guided encoding. **[C — evaluation artifact]** A base
model does not by itself hard-guarantee JSON across every serving stack.

**Still unknown:** instruction prompt and trajectory counts, reward weights,
training steps, and production hierarchy.

## 14. Zhipu AI / GLM

The [GLM-4.5 Technical Report](https://arxiv.org/abs/2508.06471) describes
separate reasoning, agent, and general specialists followed by
self-distillation into one model. Millions of SFT examples reach 128K context.
The pipeline filters duplicates, overly short and truncated answers, invalid
reasoning formats, wrong objective answers, reward-model failures, invalid
tool protocols, and failed terminal states. **[D — model training]**

Its agent factory collects real frameworks, APIs, MCP tools, and synthetic
tools; generates single- and multi-step tasks; simulates users to create
multi-turn interactions; and retains successful trajectories through multiple
judges.

General RL combines rules, RLHF, and RLAIF. A holistic RL set of about 5,000
prompts covers seven top-level, 33 second-level, and 139 third-level categories,
with human reward labels for instruction following, safety, and factuality plus
model rubrics.

### Dedicated instruction and function-call RL

GLM-4.5's instruction RL spans seven major and 151 minor constraint
categories. Feedback combines deterministic rules, a trained reward model, and
a critique model under GRPO. The report shows a SysBench instruction
satisfaction improvement over roughly 1,000 steps without an observed reward
hack on that audit. **[D — model training]**

Function-call RL has two levels:

- step-wise reward requires exact format, function name, arguments, and each
  field; and
- end-to-end reward requires valid calls plus task completion in MCP/AgentGym
  under environment rules or an LLM judge.

Invalid tool format terminates a rollout with zero reward. A pathology stage
targets rare language mixing, repetition, and format errors, while XML-like
calls reduce JSON escaping burden. **[D — model training]**

**Still unknown:** instruction sample count, reward weights, optimizer details,
real-user feedback, and production hierarchy.

## 15. Tencent Hunyuan

The [Hunyuan-A13B Technical
Report](https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan-A13B/main/report/Hunyuan_A13B_Technical_Report.pdf)
describes reasoning SFT/RL, all-scenarios SFT/general RL, and a dual
chain-of-thought stage. The all-scenarios mixture covers language, writing,
multilingual work, complex constraints, role-play, question answering,
multi-turn dialogue, and agents. **[D — model training]**

The data pipeline:

- removes ambiguous instructions;
- uses response scoring and expert rewrites;
- systematically varies complex constraints, long-context and agent
  requirements;
- verifies formal constraints with rules;
- builds multi-turn data from open, vendor, controlled-synthetic, and
  pseudo-dialogue sources;
- uses user, planner, tool, agent, and checker roles in an agent factory; and
- varies more than 30 system-instruction types, actions, tools, and response
  formats into about 20,000 format combinations.

General RL uses a generative reward model that can compare candidates and
references, inspect reasoning, call tools, and check length and constraints.
The system has 16 subtopics and more than 30 scoring services. **[D — model
training]**

Domain-specific rewards include:

- paired generative preference plus formal checks for writing;
- marker, order, tool, parameter, and value correctness for agents;
- unstable-dialogue mining for multi-turn work;
- constraint extraction, satisfaction tools, a critic, and reward model for
  complex instructions;
- understanding, consistency, and empathy for role-play; and
- specialized long-context and safety rewards.

The released [Hunyuan-A13B
quickstart](https://github.com/Tencent-Hunyuan/Hunyuan-A13B#use-with-transformers)
supports thinking and non-thinking controls through `enable_thinking` and
`/think` or `/no_think`. **[C — model/API artifact]** Vendor-reported IFEval,
SysBench, BFCL, and agent scores use different protocols from other reports and
should not be ranked by raw values alone.

**Still unknown:** per-domain sample counts, reward weights, RL optimizer and
steps, and human-annotation scale.

## 16. DeepSeek

### V4 makes instruction following a separate expert

DeepSeek's [V4
release](https://api-docs.deepseek.com/news/news260424/) and
[technical report](https://arxiv.org/abs/2606.19348) describe high-quality SFT
and GRPO for separate domain specialists, including a
dedicated instruction-following expert. More than ten specialist teachers are
then combined through full-vocabulary on-policy reverse-KL distillation rather
than the final mixed-RL stage used in V3.2. **[D — model training]**

Hard-to-verify tasks use rubric-guided RL data and a generative reward model,
with minimal/diverse human annotations described at a high level. Exact
instruction data and rewards are not disclosed.

V4 applies length penalties by non-thinking, high-thinking, and max-thinking
mode. A special tool schema and interleaved reasoning protocol preserve
reasoning through user/tool turns but clear it for a new ordinary user request.
DeepSeek warns that frameworks mislabeling tool results as user messages can
break this context and recommends non-thinking mode in that integration.
**[D — model/API protocol]**

Vendor evaluations report strengths in Chinese creative instruction following
and a remaining gap against Claude Opus 4.5 on a small complex
instruction/multi-turn writing set. The report also acknowledges occasional
format misses and weaker concise summarization.

### R1 and historical release lessons

The [DeepSeek-R1 repository and
report](https://github.com/deepseek-ai/DeepSeek-R1) describe R1-Zero problems
with repetition, readability, and language mixing. Production R1 adds
cold-start data, reasoning RL, rejection-sampled SFT, and final mixed RL with
rule rewards for reasoning and reward models for helpfulness/safety. General
instruction preferences are introduced late because longer preference
optimization showed reward hacking. **[D — model training]**

R1 did not originally include native tool-use or structured-output training.
Historical API release notes separately report system-message and JSON
improvements; those product/version results should not be projected onto every
checkpoint.

**Still unknown:** V4 instruction SFT size and taxonomy, reward weights, real
user-feedback use, and complete system hierarchy.

## 17. Baidu ERNIE

The [ERNIE 4.5 Technical
Report](https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)
describes 2.3 million SFT examples over ten domains, including mathematics,
code, logic, information processing, writing, multilingual tasks, knowledge QA,
multi-turn/role behavior, and safety. It separates reasoning and non-reasoning
data and may retain multiple reasoning paths for one query. **[D — model
training]**

Its unified reward stack uses:

- rule verifiers;
- reference-guided LLM judges;
- code sandboxes;
- reference-guided discriminative reward models;
- checklist-aware verifiers for objectively decidable criteria;
- dynamic generative reward models; and
- discriminative reward models for open tasks.

Progressive PPO moves from logic to mathematics/code and then mixed general
reasoning/non-reasoning tasks. Unified Preference Optimization combines a DPO
pairwise term with PPO; online variants form preference pairs by rejection
sampling multiple current-policy answers. Prompts that are all-correct,
all-wrong, or have very low within-group reward variance are filtered.
**[D — model training]**

ERNIEKit releases SFT and DPO components, while Baidu Cloud function calling
offers automatic, none, required, or named tool choice. The API returns a
function and arguments but does not execute them; Baidu also documents that a
prompt conflict can make tool choice imperfect. **[C/D — artifact and API
boundary]**

**Still unknown:** instruction slice counts, checklist construction, reward
weights, human-feedback volume, and flagship tool-training details.

## 18. ByteDance Seed

### Seed-OSS

The [Seed-OSS repository and model
card](https://github.com/ByteDance-Seed/seed-oss) release base variants with
and without synthetic instruction data. The synthetic-instruction variant
improves many reported evaluations; the instruction model reports 85.8 on
IFEval. The release also provides a native tool parser and trained reasoning
budgets. **[C/D — released artifact and evaluation]**

The card only gives general SFT/RLHF/PPO detail for safety. It does not disclose
a complete general instruction-following recipe.

### Seed2.0

The [Seed2.0 Model
Card](https://raw.githubusercontent.com/ByteDance-Seed/Seed2.0/master/Seed2.0%20Model%20Card.pdf)
makes reliable complex instruction execution an explicit objective and says
large-scale product feedback shapes optimization priorities: Doubao emphasizes
instruction robustness, long-tail, and long context, while Trae contributes
coding priorities. **[D — model/product feedback priority]** This does not
establish that raw user conversations are directly trained on.

Seed's private Chinese complex-instruction evaluation contains 912 cases over
17 weighted dimensions, including format, conditions, content, wording, tone,
emoji, few-shot patterns, and Chinese/English length. Seed2.0 Pro improves the
reported total over Seed1.8 while format barely improves and content regresses.
**[D — evaluation]** Aggregate gains do not prove every surface improved.

An automated model-on-model diagnostic records tokens, format adherence, turns,
emoji, outputs, traces, and tool trajectories to generate behavioral reports
for iteration. More than 15 cross-dependent constraints produce substantially
more failures in the WorldTravel analysis.

**Still unknown:** Seed2.0 architecture, SFT/RL recipe, rewards, product-feedback
sampling and authorization, and hierarchy training.

## 19. MiniMax

### MiniMax-01

The [MiniMax-01 report](https://arxiv.org/abs/2501.08313) describes millions of
high-quality prompts labeled by task, domain, and difficulty, covering long
context, code, mathematics/logic, writing, function calling, knowledge, and
safety. **[D — model training]**

Its reward framework evaluates correctness, truthfulness, helpfulness, and
harmlessness. Helpfulness combines automatic rule/constraint checks with
human coherence, depth, relevance, and style judgments. Domain specialists
iterate SFT and RL; multi-temperature rejection sampling uses hierarchical
rewards and diversity filters. Offline DPO selects best/worst candidates, while
online modified GRPO focuses on medium-success and SFT-unseen prompts with
importance clipping and KL control.

The report gives a detailed five-stage long-context SFT, DPO, and online-RL
curriculum. It also says Hailuo interactions contribute heavily to an internal
test and that teams collect multi-level instruction failures rapidly; it does
not establish unqualified direct training on raw interactions.

### MiniMax-M2 and Forge

The [MiniMax-M2 series
report](https://arxiv.org/abs/2605.26494) trains general conversation/writing,
multi-turn instruction and rubric following, cross-turn context, long context,
tool and tool-free behavior, role-play, code, and cowork/agent trajectories.
Rule and model verifiers filter SFT data. **[D — model training]**

Role-play RLHF uses explicit and implicit product feedback with disclosed
causal-inference and stratified bias-removal steps, plus entropy monitoring for
reward hacking. A composite RL reward penalizes language mixing and invalid
tool formats, rewards structured intermediate behavior and parallel completion,
and mixes reasoning, code, agent, and general domains to limit forgetting.

[Forge](https://www.minimaxi.com/news/forge-%E5%A4%A7%E8%A7%84%E6%A8%A1%E5%8E%9F%E7%94%9F-agent-rl-%E7%B3%BB%E7%BB%9F)
supports many agent scaffolds and tool formats at large rollout scale.
MiniMax's XML tool syntax and parser are released product/protocol artifacts,
not proof of general semantic compliance.

**Still unknown:** instruction sample counts, reward weights, complete feedback
sampling/privacy detail, and system hierarchy.

## 20. Apple

Apple's 2024 [on-device and server foundation-model
disclosure](https://machinelearning.apple.com/research/introducing-apple-foundation-models)
uses a hybrid of human-annotated and synthetic post-training data. A
teacher-committee rejection-sampling fine-tuning algorithm is followed by RLHF
using mirror-descent policy optimization and a leave-one-out advantage
estimator. Apple explicitly says both algorithms significantly improve
instruction-following quality. **[D — model training]**

Feature-specific low-rank adapters are dynamically loaded over a shared
foundation model. Summarization data comes from larger server teachers and is
filtered by rejection sampling; product evaluations sample 750 responses per
use case and classify the whole response as poor if any dimension is poor.
**[D — model adaptation and evaluation]** This resembles an all-constraints
product criterion but is not a universal model guarantee.

Apple says private user data and interactions are not used to train these
foundation models. The exact SFT mixture, reward-model data, RLHF reward,
adapter router, and current 2025/2026 production recipe remain **U**; the
disclosed pipeline applies to the named 2024 family.

## 21. IBM

The [Granite 3.0 8B Instruct model
card](https://www.ibm.com/docs/en/watsonx/w-and-w/2.2.0?topic=models-granite-30-8b-instruct-model-card)
describes an instruction checkpoint built from Granite-3.0-8B-Base. Its SFT
mixture combines permissively licensed public datasets, internally generated
targeted synthetic data, and a small amount of human-authored data. It uses a
structured chat format, SFT, reinforcement-learning alignment, and model
merging. **[D — model training]**

The card does not disclose the RL algorithm, reward composition, mixture
weights, hierarchy training, or production routing. Granite customer
fine-tuning and watsonx guardrails are separate customization or system
controls, not evidence about the base recipe.

## 22. AI21 Labs

The [Jamba-1.5 technical
report](https://arxiv.org/abs/2408.12570) describes SFT over high-quality
conversation, skill, and long-context data. Its synthetic pipeline generates
prompts for a target distribution, samples model responses, automatically
validates or ranks them, and applies post-editing. **[D — model training]**

For steerability, AI21 adds fine-grained, automatically verifiable constraints
to document tasks, uses constraint checks plus a general reward model for
rejection sampling, and moves shared constraints into the system message to
form multi-turn examples. This is unusually concrete evidence for
constraint-aware data construction. The report discusses PPO and DPO as
available preference methods but does **not** confirm that Jamba-1.5 used
either; claiming deployed PPO or DPO would therefore be unsupported.

**Still unknown:** exact data volumes and mixtures, post-editing scale,
reward-model training, hierarchy loss, and current hosted-product controls.

## 23. 01.AI / Yi

The released [Yi
repository](https://github.com/01-ai/Yi) states that the Yi-1.5 Chat family was
trained with SFT and no subsequent reinforcement-learning stage. Released
weights, chat templates, and fine-tuning examples provide artifact-level
evidence for the serialization and deployment surface. **[C/D — open-model
artifact and disclosure]**

This is a useful negative result: a capable instruction checkpoint need not
have a published RL stage. The exact SFT corpus, size, quality filters, complex
constraint coverage, and hierarchy training are **U**, and the statement
should not be projected onto later closed Yi APIs.

## 24. Baichuan

The [Baichuan 2 technical
report](https://statics.baichuan-ai.com/paper/Baichuan2-technical-report.pdf)
describes more than 100,000 SFT samples organized into six primary, 30
secondary, and more than 200 tertiary prompt categories. Its reward-model data
uses Baichuan 2 responses from different model sizes and SFT/PPO stages, then
PPO jointly uses actor, reference, reward, and critic models for 350
iterations. The report publishes several clipping, KL, and learning-rate
settings. **[D — model training]**

This is detailed evidence for the 2023 Baichuan 2 7B/13B Chat family, not for
Baichuan 4 or a current commercial API. The instruction-category counts do not
by themselves prove multi-turn, hierarchy, tool, or production reliability.

## 25. Huawei

[PanGu-Coder2](https://arxiv.org/abs/2307.14936) expands CodeAlpaca-20K with
Evol-Instruct to 100,000 coding instructions and rule-filters the result to
68,000. Student and teacher models sample at multiple temperatures; candidates
are ranked using compilation errors, runtime errors, partial or complete unit
test success, and a teacher heuristic. Training combines a ranking loss with
cross-entropy on the teacher response. **[D — specialist model training]**

This is a strong executable-verifier recipe for a 15B text-to-code model. It
does not establish that general or industry PanGu language models use the same
instruction data, preference objective, or hierarchy method. Huawei Cloud
fine-tuning is a customer capability, not disclosure of the base model.

## 26. iFlytek

[SciLit-LLM](https://arxiv.org/abs/2408.06574), built from
iFlytekSpark-13B, first continues pretraining on more than ten million papers
and patents, then performs domain SFT. Each SFT item has
instruction/input/output fields; instructions combine Self-Instruct generation
and human writing, while experts create the target outputs. A paper-polishing
slice additionally uses few-shot and chain-of-thought generation before
instruction fine-tuning. **[D — derivative specialist model training]**

This proves a concrete derivative route, not the alignment recipe of the core
commercial Spark or SparkDesk families. Their instruction SFT mixture,
preference/RL method, rewards, role hierarchy, and version mapping remain
**U**.

## 27. StepFun

StepFun releases a
[Step-3.5-Flash-SFT](https://huggingface.co/datasets/stepfun-ai/Step-3.5-Flash-SFT)
artifact with 1.62 million records and about 64.8 GB of general multi-turn
training data. Turns preserve role, content, and loss masks; some assistant
turns also retain `reasoning_content`. The dataset card targets multi-turn
instruction following, chat, code, and reasoning-style training. **[C —
released data artifact]**

The [Step-3.5-Flash model
card](https://huggingface.co/stepfun-ai/Step-3.5-Flash) separately describes
scalable RL. The shared name does not prove that every current dataset shard
was used for the released checkpoint, and no instruction-specific RL reward or
mixture is disclosed. Treat that causal link as **U**.

## 28. Xiaomi

The [MiMo-V2-Flash technical
report](https://arxiv.org/abs/2601.02780) begins general SFT with millions of
conversation, reasoning, code, and agent instruction–response examples in both
thinking and non-thinking modes. It trains specialist teachers for search,
code, tools, mathematics, general reasoning, and safety. **[D — model
training]**

Multi-teacher On-Policy Distillation gives the student token-level reverse-KL
rewards from teachers together with outcome rewards. Open-ended helpfulness and
safety use detailed rubrics, reference answers, and LLM judges. This is strong
evidence for a specialist-to-unified post-training route, but the report does
not disclose a dedicated system-over-user hierarchy reward or a
constraint-specific instruction mixture.

## 29. Explicit coverage gaps

The absence of a specific public recipe is itself a result:

| Developer / model | What is public | What remains U |
|---|---|---|
| [Databricks DBRX Instruct](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) | described as instruction-fine-tuned and post-trained, with human quality/safety feedback during development | instruction SFT mixture, preference objective, DPO/PPO/RLHF algorithm, and constraint generation |
| [SenseTime SenseChat / SenseNova](https://www.sensetime.com/) | product capability claims and vendor evaluations | a named model's instruction SFT, preference/RL, verifier, or hierarchy recipe |
| [Ant Ling 2.0](https://arxiv.org/abs/2510.22115) | instruct checkpoints and reasoning-oriented post-training such as Evo-CoT, DFT, and LPO | which data or rewards specifically target ordinary or complex instruction following |

Reasoning RL, a high IFEval score, or a customer fine-tuning interface is not
enough to fill these gaps. Add a developer only when a named model, artifact,
or system control supports the instruction-following claim.

## 30. How to compare a new vendor claim

For each future disclosure, record:

1. **object:** model training, research prototype, API/decoder, product,
   customer customization, or evaluation;
2. **model and date:** immutable checkpoint, provider, mode, tools, and
   verified-through date;
3. **surface:** atomic, negative, order, content, format, scope, multi-turn,
   hierarchy, tool, multilingual, multimodal, or end-to-end task;
4. **data:** real, human, licensed, synthetic, teacher, on-policy, product
   feedback, and filtering;
5. **training:** SFT, rejection sampling, preference, DPO, RLHF/RLAIF, RLVR,
   specialist, merge, or distillation;
6. **system control:** role channel, prompt, schema, constrained decoder, tool
   policy, memory, router, verifier, or guardrail;
7. **measurement:** per-constraint versus all-constraints, turns, tools,
   retries, reasoning budget, judge, and denominator;
8. **trade-off:** task quality, over-refusal, verbosity, latency, cost, and
   unrelated-capability regression;
9. **causal evidence:** matched ablation or bundled release comparison; and
10. **unknowns:** what public evidence cannot establish.

This prevents a high benchmark score from becoming a claim about private
training, a schema feature from becoming semantic reliability, or a product
guardrail from becoming a checkpoint property.
