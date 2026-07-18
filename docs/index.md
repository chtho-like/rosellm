# RoseLLM: Learn the Whole Large Language Model Stack

RoseLLM is a textbook that can be executed and a codebase that can be studied.
Its scope is the complete lifecycle of a large language model: data, modeling,
pretraining, post-training, reinforcement learning, evaluation, serving,
systems, agents, and safety.

## Two ways to use this repository

### Follow the curriculum

Start with the [LLM learning roadmap](learning-roadmap.md). It orders the topics
by dependency, attaches concrete mastery checks, and points from each idea to a
paper, an equation, and eventually an implementation.

The first complete specialization is the
[Agentic Reinforcement Learning curriculum](agentic-rl/index.md). It begins with
probability and Markov decision processes, then reaches long-horizon agent
training, distributed rollout systems, and frontier-lab case studies.

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

This knowledge base separates four different kinds of statements:

- **Disclosed fact:** explicitly stated by a model developer, paper, model card,
  repository, or dataset card.
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

## Language and contribution policy

All repository prose, comments, user-facing strings, commit messages, and new
identifiers should be English. Citations should point to primary sources when
available, and quantitative claims should identify the exact table, section,
model version, and evaluation setting.
