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
the [research standard](research-method.md) before using the
[frontier-lab and open-industry evidence matrix](agentic-rl/case-studies/index.md).

The [Multimodal Foundation Models section](multimodal/index.md) is a second
evidence-grounded specialization. It traces image, video, document, and audio
representations through encoders, projectors, shared backbones, media decoders,
and production cascades. Its Kimi and DeepSeek studies distinguish input
understanding from media generation and technical disclosure from labels such
as “native multimodal.”

### Where a new reader should begin

Use this dependency order rather than jumping directly to the newest model
report:

1. **Orientation and prerequisites:** this page, then the roadmap's five-pass
   loop and Levels 0–6. Use the [annotated bibliography](agentic-rl/bibliography.md)
   as a reading side rail.
2. **Multimodal architecture:** representation/fusion first, then Kimi and
   DeepSeek lineages, and finally image-generation architectures.
3. **Agentic RL foundations:** curriculum map, terminology, history,
   mathematical foundations, step-by-step derivations, and algorithm families.
4. **The actual training system:** data and environments, end-to-end pipeline,
   inference prerequisites, rollout/training systems, evaluation and safety,
   then the source-level lab.
5. **Research reconstruction:** research standard, evidence matrix, then
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

The knowledge base currently has two deep, source-cited focus areas:

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

## Contribution standard

Contributions should read as self-contained technical writing for their intended
audience. Citations should point to primary sources when available, and
quantitative claims should identify the exact table, section, model version, and
evaluation setting. Follow the
[documentation and mathematical rendering standard](documentation-quality.md)
for Markdown, TeX, generated-site, and browser acceptance checks.
