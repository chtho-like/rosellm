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

The [frontier-lab research corpus](frontier-labs/index.md) is the broad research
companion: 3,050 auditable public records for DeepSeek, Moonshot/Kimi,
Zhipu/GLM, OpenAI, Anthropic, and Google DeepMind/Gemini, spanning far more than
Agentic RL. It includes reproducible inventories and archives, lab-by-lab deep
syntheses, a [cross-lab comparison](frontier-labs/cross-lab.md), a generated
[coverage ledger](frontier-labs/coverage.md), and the complete
[bibliography](frontier-labs/bibliography.md).

### Where a new reader should begin

Use this dependency order rather than jumping directly to the newest model
report:

1. **Orientation and prerequisites:** this page, then the roadmap's five-pass
   loop and Levels 0–6. Use the [annotated bibliography](agentic-rl/bibliography.md)
   as a reading side rail.
2. **Agentic RL foundations:** curriculum map, terminology, history,
   mathematical foundations, step-by-step derivations, and algorithm families.
3. **The actual training system:** data and environments, end-to-end pipeline,
   inference prerequisites, rollout/training systems, evaluation and safety,
   then the source-level lab.
4. **Research reconstruction:** research standard, evidence matrix, then
   DeepSeek → GLM → Kimi → OpenAI/Anthropic/Google → Qwen/Meta/Mistral → open
   industry and community.
5. **Full frontier-lab literature:** use the
   [six-lab corpus entry point](frontier-labs/index.md), then read the
   organization reports and cross-lab synthesis before consulting the
   item-level coverage ledger.

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

## Current research focus

The active focus is **Agentic Reinforcement Learning (Agentic RL)**:
reinforcement learning in which a **Large Language Model (LLM)** policy
interacts with a stateful environment over multiple turns. The treatment covers
both the scientific object and the engineering system:

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

That specialization is no longer the boundary of the research archive. The
frontier-lab corpus separately covers pretraining, architecture, optimization,
multimodality, systems, code and mathematics, science, robotics,
interpretability, evaluation, safety, governance, and social impact.

## Contribution standard

Contributions should read as self-contained technical writing for their intended
audience. Citations should point to primary sources when available, and
quantitative claims should identify the exact table, section, model version, and
evaluation setting. Follow the
[documentation and mathematical rendering standard](documentation-quality.md)
for Markdown, TeX, generated-site, and browser acceptance checks.
