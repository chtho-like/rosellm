# Large Language Model Learning Roadmap

This roadmap turns “learn everything about **Large Language Models (LLMs)**”
into a dependency graph. It is
not a list of buzzwords. Each level states what you should be able to derive,
implement, measure, and explain before moving on.

## How to study

For every topic, use the same five-pass loop:

1. **Intuition:** explain the problem without notation.
2. **Mathematics:** derive the objective and its gradients, including shapes and
   assumptions.
3. **Minimal code:** implement the mechanism without a framework abstraction.
4. **Production code:** locate batching, parallelism, numerical, and failure
   handling concerns.
5. **Evidence:** reproduce a small experiment and state what the result does and
   does not establish.

Reading alone is not mastery. A topic is complete when you can predict a
failure, instrument it, and explain the trace from input bytes to hardware work.

## Level 0 — Mathematical and software prerequisites

### Mathematics

- Linear algebra: bases, matrix products, eigendecomposition, **Singular Value
  Decomposition (SVD)**, norms,
  projections, and tensor contractions.
- Calculus: partial derivatives, Jacobian-vector products, chain rule, and
  constrained optimization.
- Probability: random variables, expectation, variance, conditional
  probability, maximum likelihood, **Kullback–Leibler (KL) divergence**,
  entropy, and Monte Carlo
  estimators.
- Optimization: **Stochastic Gradient Descent (SGD)**, momentum, Adam/AdamW,
  learning-rate schedules, clipping,
  conditioning, and mixed-precision loss scaling.
- Information theory: cross-entropy, coding length, mutual information, and why
  next-token likelihood is measured in nats or bits.

### Systems

- Python and PyTorch execution, autograd graphs, tensor layouts, and profiling.
- **Graphics Processing Unit (GPU)** execution: warps, memory hierarchy,
  arithmetic intensity, kernels, and
  synchronization.
- Distributed computing: latency versus bandwidth, collectives, consistency,
  failures, and deterministic replay.

**Mastery check:** derive softmax cross-entropy from maximum likelihood;
implement its stable forward and backward passes; explain every tensor stride;
and compare the implementation with a fused kernel.

## Level 1 — Text, data, and tokenization

- Unicode, normalization, document boundaries, and serialization.
- Web crawls, licensed corpora, code, books, academic text, conversations, and
  synthetic data.
- Filtering, language identification, quality models, safety filtering,
  personally identifiable information, copyright, and governance.
- Exact and fuzzy deduplication, **Minimum Hashing (MinHash)** and
  **Locality-Sensitive Hashing (LSH)**, contamination analysis, and split
  integrity.
- **Byte Pair Encoding (BPE)**, byte-level BPE, Unigram, byte fallback,
  vocabulary allocation, and the
  interaction between tokenization and multilingual/code efficiency.
- Mixture design, temperature sampling, curricula, epoch accounting, and data
  provenance.

**Mastery check:** build a versioned corpus manifest; train a tokenizer; measure
fertility by domain and language; and prove that train/evaluation overlap is
below an explicit threshold.

## Level 2 — Transformer foundations

- Embeddings and tied output heads.
- Scaled dot-product attention, causal masks, multi-head attention,
  **Multi-Query Attention (MQA)**, **Grouped-Query Attention (GQA)**, and
  latent-attention variants.
- Absolute, relative, rotary, **Attention with Linear Biases (ALiBi)**, and
  extrapolated position representations.
- Pre-norm/post-norm, LayerNorm, **Root Mean Square Layer Normalization
  (RMSNorm)**, residual paths, **Multilayer Perceptrons (MLPs)**,
  **Gated Linear Units (GLUs)**, **Swish-Gated Linear Units (SwiGLUs)**, and
  initialization.
- Dense versus mixture-of-experts models: routing, load balancing, expert
  parallelism, capacity, auxiliary losses, and expert specialization.
- Long-context mechanisms, sparse attention, recurrence, state-space models,
  retrieval, and memory compression.

**Mastery check:** derive attention forward/backward complexity; implement a
decoder block; load compatible pretrained weights; and match reference logits
within a stated numerical tolerance.

## Level 3 — Pretraining science

- Autoregressive and denoising objectives.
- Parameter, token, and compute scaling laws; compute-optimal allocation; data
  constraints; and inference-aware model design.
- Batch size, sequence packing, optimizer states, warmup, decay, gradient noise,
  clipping, weight decay, and stability diagnostics.
- 32-bit floating point (FP32), TensorFloat-32 (TF32), 16-bit floating point
  (FP16), Brain Floating Point 16 (BF16), 8-bit floating point (FP8),
  quantized training, master weights, and stochastic
  rounding.
- Checkpointing, evaluation during training, data ablations, and loss-to-capability
  correlations.
- Dense and **Mixture-of-Experts (MoE)** distributed training with data, tensor,
  pipeline, sequence,
  context, and expert parallelism.

**Mastery check:** write a compute and memory budget, predict tokens/second,
train a small Generative Pre-trained Transformer (GPT), explain every divergence
event, and produce a reproducible model/data/checkpoint card.

## Level 4 — Supervised post-training

- Instruction and chat schema design, system/user/assistant/tool roles, and loss
  masking.
- Human demonstrations, expert data, synthetic instruction generation,
  distillation, rejection sampling, and self-instruct pipelines.
- Capability mixture design, sampling weights, length distributions, and
  multi-turn formatting.
- Full fine-tuning, parameter-efficient methods, catastrophic forgetting, and
  model merging.
- Evaluation of helpfulness, style, knowledge, reasoning, tool use, and safety.

**Mastery check:** construct a traceable **Supervised Fine-Tuning (SFT)**
dataset, inspect masked labels token
by token, fine-tune a small model, and attribute an observed regression to data
or optimization rather than intuition.

## Level 5 — Preference learning and alignment

- Preference elicitation, annotator instructions, disagreement, calibration,
  and active sampling.
- Bradley–Terry models, reward models, pairwise and listwise ranking, and reward
  uncertainty.
- **Reinforcement Learning from Human Feedback (RLHF)** with **Proximal Policy
  Optimization (PPO)**, KL control, value learning, **Generalized Advantage
  Estimation (GAE)**, and reward normalization.
- **Direct Preference Optimization (DPO)**-family offline preference objectives
  and their assumptions.
- Artificial Intelligence (AI) feedback, constitutional methods,
  critique/revision, process supervision, and scalable oversight.
- Reward hacking, overoptimization, Goodhart's law, sycophancy, and alignment
  taxes.

**Mastery check:** derive the reward-model likelihood, DPO loss, and PPO clipped
surrogate; then train toy versions and show where each estimator is biased or
high variance.

## Level 6 — Reasoning and reinforcement learning with verifiable rewards

- Chain-of-thought data, outcome versus process supervision, search, sampling,
  and verifier-guided selection.
- Math, code, theorem proving, games, and other domains with executable or
  rule-based rewards.
- Group-relative baselines, rejection sampling, curriculum scheduling, entropy
  dynamics, response-length bias, and pass@k-oriented objectives.
- Distillation versus online discovery; when Reinforcement Learning (RL)
  improves search and when it merely changes sampling behavior.

**Mastery check:** implement REINFORCE, **REINFORCE Leave-One-Out (RLOO)**, PPO,
and **Group Relative Policy Optimization (GRPO)** on the same small
task; control sample count and compute; and compare estimator variance,
effective sample size, entropy, and held-out success.

## Level 7 — Agentic reinforcement learning

An agentic episode includes state-changing environment interaction rather than
only one prompt and one response. Study the complete
[Agentic RL curriculum](agentic-rl/index.md):

- **Partially Observable Markov Decision Process (POMDP)** formulation and
  multiple time scales;
- planning, memory, tool use, perception, reflection, and self-improvement;
- environment and task generation;
- multi-turn rollout collection and trajectory storage;
- sparse reward and hierarchical credit assignment;
- asynchronous rollout/training systems and off-policy correction;
- evaluation under stochasticity and adversarial conditions; and
- real model-family training disclosures across
  [DeepSeek, Zhipu AI's General Language Model (GLM), Kimi, OpenAI, Anthropic,
  Google DeepMind, Qwen, Meta, Mistral, ByteDance, NVIDIA, Microsoft, xAI, and
  open projects](agentic-rl/case-studies/index.md).

**Mastery check:** train a model in a sandboxed multi-step environment, preserve
the exact sampled tokens and policy version, assign credit at token/turn/episode
levels, and reproduce an improvement on a hidden task split without reward or
environment leakage. Then audit one frontier claim by separating architecture,
data, post-training, scaffold, tools, routing, and inference-time compute; mark
every undisclosed field as unknown instead of completing the recipe by analogy.

## Level 8 — Inference systems

- Prefill/decode asymmetry, **Key-Value (KV) cache** layout, paged attention,
  speculative decoding, and disaggregated serving.
- Continuous and dynamic batching, scheduling, fairness, admission control,
  preemption, and tail latency.
- Tensor/pipeline/expert parallel inference, communication overlap, and
  multi-node reliability.
- Weight-only and activation quantization, calibration, kernels, and quality
  measurement.
- Streaming, structured generation, constrained decoding, sampling, and
  tokenizer boundaries.

**Mastery check:** account for every byte in a KV cache, predict the prefill and
decode rooflines, and explain p50/p99 latency changes using an instrumented
trace rather than aggregate throughput alone.

## Level 9 — Evaluation, safety, and operations

- Static benchmarks, dynamic benchmarks, capability elicitation, contamination,
  and statistical uncertainty.
- Large Language Model judges, pairwise evaluation,
  position/verbosity/self-preference bias, and
  human validation.
- Red teaming, misuse, prompt injection, tool authorization, sandboxing,
  monitoring, incident response, and rollback.
- Model/data/system cards, experiment tracking, lineage, governance, privacy,
  and deployment gates.
- Cost, energy, latency, throughput, utilization, and reliability as first-class
  evaluation dimensions.

**Mastery check:** design an evaluation suite with threat model, hidden splits,
confidence intervals, cost accounting, and explicit ship/no-ship criteria.

## Level 10 — Frontier research

- Multimodal and embodied agents.
- Self-play, automated curricula, environment generation, and open-endedness.
- Test-time training, persistent memory, continual learning, and non-stationary
  objectives.
- Multi-agent cooperation/competition, mechanism design, and social learning.
- Interpretability of learned reasoning and agent policies.
- Robust scalable oversight and control of long-horizon autonomous systems.

At this level, “knowing the literature” is not enough. You should be able to
state an unresolved question, construct a falsifiable experiment, quantify its
failure modes, and release enough evidence for an independent reproduction.
