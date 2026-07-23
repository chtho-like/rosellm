# RoseLLM: Learn the Whole Large Language Model Stack

RoseLLM is a textbook that can be executed and a codebase that can be studied.
Its scope is the complete lifecycle of a large language model: data, modeling,
pretraining, post-training, reinforcement learning, evaluation, serving,
systems, agents, and safety.

## Two ways to use this repository

### Follow the curriculum

Start with the [LLM learning roadmap](learning-roadmap.md). It orders the topics
by dependency and attaches concrete mastery checks. It also identifies where
the repository already provides a derivation, source lab, or primary-source
reading path and where prerequisite textbook study is still required.

The first complete specialization is the
[Agentic Reinforcement Learning curriculum](agentic-rl/index.md). It begins with
probability and Markov decision processes, then reaches long-horizon agent
training, distributed rollout systems, and source-level implementation. Read
the [long-horizon capability synthesis](agentic-rl/long-horizon-capability.md)
to separate what laboratories train into model weights from what context
management, tools, memory, verification, and multi-agent runtimes add. Read
the [research standard](research-method.md) before using the
[frontier-lab and open-industry evidence matrix](agentic-rl/case-studies/index.md).

For a quantitative orientation before entering the case studies, use the
[model training token ledger](model-training-token-ledger.md). It separates
fresh pretraining, continued pretraining, post-training, corpus size, and
context length across DeepSeek, Kimi, GLM, Qwen, other Chinese lineages, and
international reference families. The companion
[DeepSeek release timeline](deepseek-release-timeline.md) distinguishes
mainline checkpoints, specialist branches, API aliases, and temporary modes.

The [Multimodal Foundation Models section](multimodal/index.md) is a second
evidence-grounded specialization. It traces image, video, document, and audio
representations through encoders, projectors, shared backbones, media decoders,
and production cascades. Its Kimi and DeepSeek studies distinguish input
understanding from media generation and technical disclosure from labels such
as “native multimodal.”

The [Coding-Agent Systems section](coding-agents/index.md) separates models,
agent harnesses, interaction surfaces, and hosted control planes. Its first
source-level reconstruction follows both Reasonix Git roots and compares the
current Go system with OpenCode, Pi Coding Agent, OpenAI Codex, and Claude Code
across provider coupling, agent loops, persistence, extension boundaries,
permissions, sandboxing, and core-source visibility.

The [Instruction Following and Steerability chapter](evaluation/instruction-following.md)
separates atomic constraints, multi-turn retention, instruction hierarchy,
tool-policy compliance, structured decoding, and end-to-end product behavior.
Its companion [method map](evaluation/instruction-following-methods.md),
[vendor evidence audit](evaluation/instruction-following-vendors.md), and
[production operations chapter](evaluation/instruction-following-operations.md)
span data construction, post-training, inference, agent controls, the disclosed
practices of 27 leading developers, and a rule-level release system. The
overview maps the major public benchmarks and preserves a dated
GPT-versus-DeepSeek comparison.

### Where a new reader should begin

Use this dependency order rather than jumping directly to the newest model
report:

1. **Orientation and prerequisites:** this page, then the roadmap's five-pass
   loop and Levels 0–6. Use the [annotated bibliography](agentic-rl/bibliography.md)
   as a reading side rail.
2. **Coding-agent architecture:** systems map, then the Reasonix history and
   cross-project source comparison. Keep model capability separate from harness
   and product behavior.
3. **Multimodal architecture:** representation/fusion first, then Kimi and
   DeepSeek lineages, and finally image-generation architectures.
4. **Model evaluation:** instruction-following taxonomy, benchmark map, and
   private production-evaluation design. Keep checkpoint behavior separate from
   constrained decoding and agent-harness behavior.
5. **Agentic RL foundations:** curriculum map, terminology, long-horizon
   capability synthesis, history, mathematical foundations, step-by-step
   derivations, and algorithm families.
6. **The actual training system:** data and environments, end-to-end pipeline,
   inference prerequisites, rollout/training systems, evaluation and safety,
   then the source-level lab.
7. **Research reconstruction:** research standard, evidence matrix, then
   DeepSeek → GLM → Kimi → OpenAI/Anthropic/Google → Qwen/Meta/Mistral → open
   industry and community.

Keep the [glossary](glossary.md) open while reading. It is a lookup tool, not a
final chapter.

### Trace concepts into code

Use the implementation as a set of progressively more realistic laboratories:

| Layer | Repository area | Questions it makes concrete |
|---|---|---|
| Transformer training | `rosellm/rosetrainer/` | How do attention, loss, checkpoints, tensor parallelism, and DDP fit together? |
| LLM inference | `rosellm/roseinfer/` | How do paged KV caches, prefix reuse, continuous batching, prefill/decode disaggregation, and streaming work? |
| GPU kernels | `notebooks/cuda/` | Where do memory traffic, tiling, fusion, and numerical precision determine performance? |
| Distributed primitives | `notebooks/` | What do all-reduce, reduce-scatter, bucketing, and sharding actually do? |
| Multimodal models | `docs/multimodal/` | How do media become model positions, when is integration “native,” and how can a language-centered model emit images or speech? |
| Model evaluation | `docs/evaluation/` | Which factuality, grounding, citation, calibration, tool-integrity, or instruction surface failed, what is the reference, and which guarantee came from the model or surrounding system? |
| Coding-agent systems | `docs/coding-agents/` | How do models, agent loops, tools, context, clients, permissions, and sandboxes combine into coding products? |
| Agentic RL | `docs/agentic-rl/` and `rosellm/roserlhf/` | How does an interactive trajectory become an unbiased, stable policy update? |

## The standard of evidence

This knowledge base separates five different kinds of statements:

- **Disclosed fact:** explicitly stated by a model developer, paper, model card,
  repository, or dataset card.
- **Confirmed artifact:** directly observable in released weights,
  configuration, data manifests, or source code even when prose does not state
  it explicitly.
- **Reproduced result:** independently produced with a recorded environment,
  command, configuration, and artifact.
- **Inference:** a conclusion supported by disclosed facts but not itself
  disclosed. The assumptions must be written next to it.
- **Unknown:** information that is not public or cannot be verified.

This distinction is essential in frontier-model case studies. A public model
architecture does not reveal the private data mixture; an API behavior does not
prove the training recipe; and benchmark scores do not disclose a production
agent pipeline. See the [research standard](research-method.md) for the full
protocol.

## Current focus

The knowledge base currently has four deep, source-cited focus areas:

- **Model evaluation and reliability:** factuality and grounding controls,
  vendor-disclosed practices, claim-level production operations,
  instruction-following training and control methods, 27-developer evidence
  audit, benchmark methodology, and application-specific acceptance suites;

- **Coding-agent systems:** runtime lineage, model/provider coupling, agent
  loops, tools, state, extensibility, permissions, sandboxing, and open-source
  boundaries, beginning with a Reasonix/OpenCode/Pi/Codex/Claude Code study;
- **Multimodal foundation models:** modality representation, fusion, native
  integration, understanding versus generation, and the Kimi/DeepSeek model
  lineages; and
- **Agentic Reinforcement Learning (Agentic RL):**
  reinforcement learning in which a **Large Language Model (LLM)** policy
  interacts with a stateful environment over multiple turns.

The Agentic RL treatment covers both the scientific object and the engineering
system:

1. the **Partially Observable Markov Decision Process (POMDP)** and
   policy-gradient formulation;
2. prompts, observations, actions, tools, memory, and state transitions;
3. offline demonstrations, online rollouts, rejection sampling, and curriculum
   construction;
4. outcome, process, preference, safety, and cost rewards;
5. PPO, GRPO, RLOO, REINFORCE-style methods, and agent-specific credit
   assignment;
6. asynchronous generation, version control for policies, replay, staleness,
   fault-tolerant environments, and distributed optimization;
7. evaluation against contamination, judge bias, reward hacking, and
   non-deterministic environments; and
8. disclosed practices from DeepSeek, Zhipu GLM, Moonshot Kimi, and other labs.

The [Model Evaluation, Factuality, and Reliability
map](evaluation/index.md) adds a separate production path for unsupported
generation. It distinguishes open-world factuality, source faithfulness,
citation entailment, calibration, tool-result integrity, and instruction
following; catalogs controls from data and post-training through retrieval,
tools, verification, abstention, monitoring, and human review; and audits which
parts leading vendors have actually disclosed.

## Contribution standard

Contributions should read as self-contained technical writing for their intended
audience. Citations should point to primary sources when available, and
quantitative claims should identify the exact table, section, model version, and
evaluation setting. Follow the
[documentation and mathematical rendering standard](documentation-quality.md)
for Markdown, TeX, generated-site, and browser acceptance checks.
