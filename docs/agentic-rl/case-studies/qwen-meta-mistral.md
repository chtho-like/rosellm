# Qwen, Meta, and Mistral: Three Paths from Alignment to Agentic Reinforcement Learning

**Verified through:** 2026-07-19. **Sources:** official technical reports,
model cards, repositories, configuration artifacts, and first-party release or
engineering posts. Benchmark results are vendor-reported unless explicitly
marked reproduced.

## Reader's terminology key

The chapter defines uncommon terms before using their abbreviations:

- **Large Language Model (LLM):** an autoregressive neural network trained to
  predict tokens and then adapted to follow instructions.
- **Artificial Intelligence (AI)** and **Generative Pre-trained Transformer
  (GPT):** the broad field and the model-family term used in several proper
  checkpoint names.
- **Reinforcement Learning (RL):** optimization of a policy from rewards
  obtained after it takes actions. For an LLM, a token is an action and a
  multi-token answer or tool interaction is a trajectory.
- **Supervised Fine-Tuning (SFT):** next-token imitation on curated target
  answers or complete interaction trajectories.
- **Chain of Thought (CoT):** intermediate natural-language reasoning tokens.
- **Direct Preference Optimization (DPO):** an offline loss that increases the
  probability of a preferred response relative to a rejected response and a
  reference policy.
- **Reinforcement Learning from Human Feedback (RLHF):** policy training from
  human preferences, normally mediated by a learned reward model.
- **Reinforcement Learning with Verifiable Rewards (RLVR):** RL whose outcome
  can be checked mechanically, for example by symbolic answer matching, a
  compiler, or unit tests.
- **Proximal Policy Optimization (PPO):** actor-critic RL that limits policy
  movement with clipped probability ratios, a trust-region penalty, or both.
- **Group Relative Policy Optimization (GRPO):** critic-free RL that estimates
  an answer's advantage by comparing several answers to the same prompt.
- **Group Sequence Policy Optimization (GSPO):** Qwen's sequence-ratio variant
  of group-relative optimization.
- **Soft Adaptive Policy Optimization (SAPO):** Qwen's smooth, sign-dependent
  importance-ratio gate.
- **Single-Rollout Asynchronous Optimization (SAO):** a one-rollout actor-critic
  algorithm with direct double-sided importance masking.
- **Generalized Advantage Estimation (GAE):** a discounted mixture of
  temporal-difference errors used to assign credit over a trajectory.
- **Reward Model (RM):** a learned scalar scorer for candidate behavior.
- **Tool-Integrated Reasoning (TIR):** reasoning that calls an interpreter or
  another tool and incorporates its returned observation.
- **Continual Pretraining (CPT):** additional next-token training of an existing
  base model on a new mixture or domain.
- **Mixture of Experts (MoE):** sparse feed-forward computation in which a
  router sends each token to only a few of many experts. In names such as
  30B-A3B or 750B-A40B, the suffix means approximately 3B or 40B parameters
  are activated per token; exact counting conventions may exclude embeddings
  or auxiliary modules.
- **Grouped-Query Attention (GQA):** multiple query heads sharing fewer
  key/value heads to reduce memory and decoding bandwidth.
- **Multi-head Latent Attention (MLA):** compression of attention keys and
  values into a lower-dimensional latent state.
- **Multi-Token Prediction (MTP):** auxiliary prediction of more than the next
  token, also useful for speculative decoding.
- **Query-Key-Value (QKV):** the three projected representations used by
  attention.
- **Swish Gated Linear Unit (SwiGLU):** a gated feed-forward activation used by
  all three model families.
- **Byte-Level Byte Pair Encoding (BBPE):** a subword tokenizer learned over
  bytes, so arbitrary text remains representable.
- **Root Mean Square Normalization (RMSNorm):** normalization by root mean
  square rather than mean and variance.
- **Rotary Position Embedding (RoPE):** position-dependent rotation of query
  and key coordinates.
- **Yet another RoPE extensioN (YaRN):** a RoPE scaling method for extending
  context beyond the original training length.
- **Key-Value (KV) cache:** attention keys and values retained during
  autoregressive generation.
- **Fill in the Middle (FIM):** code training that predicts a missing region
  from its prefix and suffix.
- **Kullback-Leibler (KL) divergence:** a measure of policy distribution
  change, often used as a penalty.
- **Exponential Moving Average (EMA):** a smoothed parameter copy updated from
  recent checkpoints.
- **Model Context Protocol (MCP):** a standard interface through which agents
  discover and call external tools or data services.
- **Software Engineering (SWE) agent:** a model-driven loop that reads a
  repository, edits files, executes commands and tests, and revises its patch.
- **Vision-Language (VL)** and **Vision-Language-Action (VLA):** models joining
  text with perception, or with both perception and embodied action.
- **User Experience (UX):** the interaction/design specialty named in the
  Qwen3-Coder-Next expert pipeline.
- **Application Programming Interface (API):** a programmatic service
  interface; an API-visible model need not have downloadable weights.
- **Brain Floating Point 16 (BF16)** and **Floating Point 8 (FP8):**
  numerical formats used to reduce memory and compute.
- **Graphics Processing Unit (GPU):** the accelerator used for training and
  serving; **Floating-Point Operation (FLOP)** is a compute-counting unit.
- **Optical Character Recognition (OCR):** conversion of rendered text in
  images or documents into machine-readable tokens.
- **Portable Document Format (PDF):** a document format used in tool and file
  interaction examples.
- **F1 score:** the harmonic mean of precision and recall.
- **American Invitational Mathematics Examination (AIME):** a common
  verifier-friendly mathematical reasoning benchmark.
- **General Language Model (GLM):** Zhipu AI's model-family name.
- **Massachusetts Institute of Technology (MIT) license:** the license family
  modified by some Mistral releases.
- **Neural Collective Communications Library (NCCL):** NVIDIA's GPU-to-GPU
  communication library.
- **Data, Tensor, Pipeline, and Expert Parallelism (DP, TP, PP, EP):** splitting
  examples, tensors, layers, or experts across accelerators.

## Evidence labels

- **[D] Disclosed:** stated directly in a primary source.
- **[C] Confirmed artifact:** visible in released configuration, weights, or
  source code.
- **[R] Reproduced:** rerun by this repository with recorded artifacts.
- **[I] Inference:** a reasoned conclusion, with assumptions stated.
- **[U] Unknown:** absent, ambiguous, contradictory, or not verifiable.

The distinction is essential. A model can expose weights and inference code
while withholding the corpus, synthetic-data generators, reward models,
optimizer state, RL prompts, environment images, and production scheduler.
"Open weights" is therefore not synonymous with "reproducible training."

## 1. Executive conclusions

1. **Qwen exposes the broadest public sequence of optimizer experiments.**
   Qwen2.5 uses DPO and GRPO; Qwen3 separates reasoning and general RL; GSPO
   aligns clipping with sequence rewards; SAPO replaces hard clipping with a
   smooth gate; Qwen-AgentWorld trains an environment model and then uses it to
   generate controlled agent-RL experience.
2. **Meta's best documented pipeline is not an agent-RL recipe.** Llama 2 gives
   unusually complete PPO/RLHF hyperparameters. Llama 3.1 instead scales
   synthetic data, rejection sampling, SFT, and DPO, with separately annotated
   tool trajectories. Llama 4 discloses online RL only at a high level.
3. **Mistral publishes the clearest single reasoning-RL system paper.**
   Magistral gives its modified GRPO objective, reward components, asynchronous
   trainer/generator/verifier design, data reduction counts, batch curriculum,
   and negative results. Devstral separately shows real SWE trajectory
   collection but withholds the policy optimizer.
4. **Agentic RL is an environment-and-verifier operation, not merely a policy
   loss.** The practical work is constructing resettable sandboxes, mining
   learnable tasks, preserving exact tool observations, detecting reward
   hacking, and moving rollout log-probabilities and policy versions back to
   the trainer.
5. **Do not retroactively copy a later recipe onto an earlier checkpoint.**
   GSPO's paper says it contributed to later Qwen3 models, not the original
   April 2025 release. SAPO is associated with Qwen3-VL, not every Qwen3.5+
   model. Magistral's 2025 GRPO settings are not documented as the recipe for
   Mistral Small 4 or Medium 3.5. Llama 3's initial PPO disclosure and Llama
   3.1's final SFT/rejection-sampling/DPO recipe describe different releases.

## 2. A zero-to-research mental model

### 2.1 What an LLM policy actually optimizes

Given a prompt \(x\), an LLM policy \(\pi_\theta\) samples a response
\(y=(y_1,\ldots,y_T)\):

\[
\pi_\theta(y\mid x)
=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t}).
\]

In ordinary SFT, a fixed target response supplies a token-level loss. In online
RL, the policy must first generate a response or interact with an environment,
then receive a reward. For a tool agent, the trajectory is better written

\[
\tau=(s_0,a_0,o_0,s_1,a_1,o_1,\ldots,s_T),
\]

where \(a_i\) is model-generated reasoning or a tool call and \(o_i\) is the
environment observation. A terminal reward may say only whether the final
answer, patch, or workflow succeeded.

The fundamental estimator is

\[
\nabla_\theta J(\theta)
\approx
\mathbb E\left[
\sum_t \hat A_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right],
\]

where \(\hat A_t\) says whether action \(a_t\) was better or worse than a
baseline. The central differences among PPO, GRPO, GSPO, SAPO, and SAO are:

1. how that baseline or advantage is estimated;
2. whether off-policy correction is token-level or sequence-level;
3. how large a policy update is trusted;
4. whether a learned value model is required; and
5. how rollout collection overlaps with training.

### 2.2 Why groups, critics, and verifiers exist

- A **group baseline** samples \(G\) answers for one prompt. If rewards are
  \(R_1,\ldots,R_G\), an elementary relative advantage is
  \((R_i-\bar R)/(\operatorname{std}(R)+\varepsilon)\). It avoids a critic but
  consumes several rollouts and must wait for slow group members.
- A **critic** \(V_\phi(s)\) predicts future return from a state. It allows one
  rollout per prompt and token/time-dependent credit, but it adds a large model
  that can become unstable or stale.
- A **verifier** computes reward. Exact-answer and unit-test verifiers are
  precise but narrow. Learned judges cover open-ended behavior but can be
  exploited, drift, or encode label bias.
- An **environment** returns observations after actions. If it cannot be reset
  exactly, produces nondeterministic state, leaks answers, or permits network
  shortcuts, the policy may optimize the benchmark rather than the intended
  skill.

### 2.3 Why asynchronous RL is difficult

Long agent trajectories have a heavy-tailed duration distribution. Waiting for
the slowest rollout wastes accelerators; training immediately makes completed
rollouts stale because policy weights advance while other sequences are still
generating. If

\[
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{rollout}}(a_t\mid s_t)},
\]

then \(r_t\) measures how far the current trainer has moved from the policy that
generated token \(a_t\). Production systems trade off:

- generator utilization;
- policy freshness;
- queue size;
- maximum trajectory length;
- exact rollout log-probability transport;
- weight-broadcast latency; and
- the bias/variance introduced by clipping or masking stale samples.

The [mathematical foundations](../mathematical-foundations.md),
[algorithms](../algorithms.md), and [systems](../systems.md) chapters derive
these components independently. This chapter asks which components each vendor
actually disclosed.

## 3. Comparative lineage

| Date | Vendor release | Main transition | Agentic-RL significance |
|---|---|---|---|
| 2023-02 to 2023-07 | Meta Llama / Llama 2 | scaled dense pretraining, then iterative RLHF | full PPO/RM baseline |
| 2023-09 | Mistral 7B | GQA plus sliding-window attention | efficient open base; recipe mostly unknown |
| 2023-12 to 2024-04 | Mixtral 8x7B / 8x22B | sparse MoE | SFT/DPO and native function calls; no online RL disclosed |
| 2024-04 to 2024-07 | Meta Llama 3 / 3.1 | 15.6T tokens, synthetic-data flywheel | rejection sampling/SFT/DPO; explicit human tool trajectories |
| 2024-09 to 2024-12 | Qwen2.5 family | 18T tokens, specialist data | DPO then online GRPO; detailed Math RLVR |
| 2025-03 | QwQ-32B | reasoning cold start plus outcome RL | verifier-first math/code, then general RL |
| 2025-04 | Qwen3 | dense/MoE unified thinking modes | four-stage reasoning/general RL and agent tasks |
| 2025-04 | Meta Llama 4 | multimodal MoE | SFT, online RL, DPO; exact RL recipe unknown |
| 2025-05 to 2025-06 | Devstral / Magistral | real SWE traces; reasoning RL | environment SFT + policy optimization; fully described modified GRPO |
| 2025-07 | Qwen3-Coder | 480B/35B code MoE | executable-code RL and 20K parallel agent environments |
| 2025-07 to 2025-11 | GSPO / SAPO | sequence and smooth trust regions | stability for sparse MoE RL |
| 2025-12 | Mistral 3 / Devstral 2 | larger open MoE and code agents | strong agent interfaces; updated recipe unknown |
| 2026-02 to 2026-06 | Qwen3.5–3.7 | native multimodal agent models | scaled asynchronous RL infrastructure; later recipe increasingly opaque |
| 2026-03 to 2026-04 | Mistral Small 4 / Medium 3.5 | one instruct/reasoning/agent model | configurable reasoning; recipe withheld |
| 2026-04 to 2026-07 | Meta Muse Spark / 1.1 | post-Llama proprietary reasoning agents | thinking-time penalty and multi-agent training at high level |
| 2026-06 | Qwen-AgentWorld | language world model | >10M environment trajectories and controlled simulated RL |
| 2026-07 | SAO report | one-rollout asynchronous actor-critic | authors report deployment in GLM-5.2; controlled tests use Qwen |

## 4. Qwen / Alibaba

### 4.1 Qwen2.5: specialist data plus offline and online alignment

Primary source: [Qwen2.5 technical report](https://arxiv.org/abs/2412.15115),
Sections 2–4, pp. 2–7.

#### Architecture [D]

Qwen2.5 uses a dense decoder-only Transformer with GQA, SwiGLU, RoPE, QKV bias,
and pre-RMSNorm. Its BBPE tokenizer has 151,643 normal tokens plus 22 control
tokens, including two tool tokens.

| Model | Layers | Query / KV heads | Tied embedding | Context / generation | License in report |
|---|---:|---:|---|---:|---|
| 0.5B | 24 | 14 / 2 | yes | 32K / 8K | Apache 2.0 |
| 1.5B | 28 | 12 / 2 | yes | 32K / 8K | Apache 2.0 |
| 3B | 36 | 16 / 2 | yes | 32K / 8K | Qwen Research |
| 7B | 28 | 28 / 4 | no | 128K / 8K | Apache 2.0 |
| 14B | 48 | 40 / 8 | no | 128K / 8K | Apache 2.0 |
| 32B | 64 | 40 / 8 | no | 128K / 8K | Apache 2.0 |
| 72B | 80 | 64 / 8 | no | 128K / 8K | Qwen |

#### Pretraining data operation [D]

- Training exposure grows from Qwen2's roughly 7T tokens to **18T tokens**.
- Qwen2-Instruct models score and filter multilingual data.
- Qwen2.5-Math and Qwen2.5-Coder corpora are mixed into the general model.
- Qwen2-72B-Instruct and Qwen2-Math-72B-Instruct synthesize mathematics, code,
  and knowledge data.
- A proprietary general RM and Qwen2-Math-RM-72B filter synthetic material.
- Dense probes from 44M to 14B parameters and MoE probes from 44M to 1B active
  parameters are trained on 0.8B–600B tokens to predict learning rate and batch
  size. The final predicted settings are not published.

Long context is staged. General checkpoints move from 4K to 32K. Qwen2.5-Turbo
trains at 32K, 65,536, 131,072, and 262,144, using 40% maximum-length and 60%
shorter samples per stage. RoPE base changes from 10K to 1M. YaRN and
dual-chunk attention extend inference to one million tokens for Turbo and
131K for other long-context variants.

#### Post-training [D]

SFT uses more than one million examples, two epochs, sequence length 32,768,
learning rate \(7\times10^{-6}\rightarrow7\times10^{-7}\), weight decay .1,
and gradient clip 1.

Offline alignment:

1. sample new prompts;
2. resample responses from the SFT model;
3. check code by execution and answers by matching;
4. pair passing responses with failing responses;
5. apply human and automated review; and
6. train on about **150K DPO pairs** for one epoch with the Online Merging
   Optimizer and learning rate \(7\times10^{-7}\).

Online alignment uses GRPO. Separate reward models target truthfulness,
helpfulness, concision, relevance, harmlessness, and bias. Prompts combine
public and proprietary sources; candidate policies include SFT, DPO, and prior
RL checkpoints at several temperatures. High score-variance prompts are
prioritized. Each prompt receives eight responses; both global batch and
samples per episode are 2,048.

**[U]** Final pretraining schedule, exact mixture proportions, GRPO clipping,
KL term, rollout temperature, number of RL updates, reward aggregation,
hardware, and total compute.

### 4.2 Qwen2.5-Math: an industrial RLVR pipeline

Primary source: [Qwen2.5-Math](https://arxiv.org/abs/2409.12122), Sections
2–3, pp. 4–8.

#### Corpus and continual pretraining [D]

- Qwen Math Corpus v1 contains about 700B tokens; v2 exceeds **1T tokens**.
- Inputs include mathematical web pages, code, encyclopedias, examinations,
  arXiv-like material, and synthetic data.
- An iterative FastText classifier, Qwen2-0.5B-Instruct quality filter,
  deduplication, recall, synthesis, and mixture balancing construct the corpus.
- Qwen2-Math-Instruct generates additional specialist data.
- 1.5B, 7B, and 72B models receive CPT at 4K context.

#### SFT trace factory [D]

CoT and TIR train for three epochs at 4,096 tokens. The 72B model uses batch 256
and peak learning rate \(5\times10^{-6}\); 1.5B/7B use batch 128 and
\(2\times10^{-5}\); all decay toward \(7\times10^{-7}\).

- Initial CoT queries: 580K English plus 500K Chinese.
- After iterative RM rejection sampling: about 2M English plus 500K Chinese
  responses.
- TIR: 190K annotated, 205K synthetic, and 75K Chinese-translated examples.
- Online rejection fine-tuning varies temperature and difficulty, executes
  tools, uses majority voting, and deduplicates retained traces.

#### Reward and policy optimization [D]

Qwen2 RM data has 206K English prompts with six candidates each. Qwen2.5 RM
data expands to 361K English and 257K Chinese prompts, also with six
candidates. A two-linear-layer scalar head is trained with listwise pairwise
logistic ranking across valid positive/negative pairs.

Each potential RL prompt is sampled eight times. Only prompts with two to five
correct attempts are retained, leaving **66K prompts**: neither already solved
nor effectively impossible for the current policy.

The combined reward is

\[
R=\sigma(0.5R_{\mathrm{RM}})+(R_{\mathrm{verifier}}-1),
\qquad
R_{\mathrm{verifier}}\in\{0,1\}.
\]

GRPO samples 32 responses per prompt. Samples per episode are 4,096 for 7B and
2,048 for 72B; global batch is 512. Learning rates are \(10^{-5}\) and
\(5\times10^{-6}\); KL coefficient is \(10^{-3}\). Interpreter-output tokens
are masked from policy loss because the environment, not the model, generated
them.

**Why this matters:** the data loop and the RL loop are inseparable. Pass-rate
filtering determines which prompts have usable variance; execution determines
labels; token masking prevents crediting the policy for observations it did not
choose.

**[U]** Complete prompt corpus, proprietary synthesis prompts, final model
mixtures, exact number of policy updates, production hardware, and orchestration
code.

### 4.3 QwQ-32B: outcome-verifier reasoning

Primary source: [official QwQ-32B release](https://qwenlm.github.io/blog/qwq-32b/).

The Apache-2.0 checkpoint follows [D]:

1. a reasoning cold start;
2. first-stage outcome RL on mathematics and code, using exact answer
   verification and code execution rather than a learned RM; and
3. a smaller general RL stage combining a general RM with rule verifiers,
   improving instruction following, preferences, and agent behavior without
   material math/code loss.

This is a transition from preference-only alignment to reward grounded in an
external process. **[U]** Optimizer, group size, prompts, sampling temperature,
updates, learning rate, hardware, and compute.

### 4.4 Qwen3: four-stage reasoning and general RL

Primary source: [Qwen3 technical report](https://arxiv.org/abs/2505.09388),
Sections 2–4, pp. 2–12.

#### Architecture and pretraining [D]

Qwen3 retains GQA, SwiGLU, RoPE, and pre-RMSNorm, removes QKV bias, and adds
query/key normalization. Its BBPE vocabulary is 151,669.

| Model | Layers | Query / KV heads | Experts total / active | Context |
|---|---:|---:|---:|---:|
| 0.6B | 28 | 16 / 8 | dense | 32K |
| 1.7B | 28 | 16 / 8 | dense | 32K |
| 4B | 36 | 32 / 8 | dense | 128K |
| 8B | 36 | 32 / 8 | dense | 128K |
| 14B | 40 | 40 / 8 | dense | 128K |
| 32B | 64 | 64 / 8 | dense | 128K |
| 30B-A3B | 48 | 32 / 4 | 128 / 8 | 128K |
| 235B-A22B | 94 | 64 / 4 | 128 / 8 | 128K |

The MoE has no shared experts and uses a global-batch load-balancing loss.
Training exposure is **36T tokens across 119 languages and dialects**:

1. more than 30T at 4,096;
2. about 5T higher-quality science, technology, engineering, mathematics,
   code, reasoning, and synthetic data at 4,096; and
3. hundreds of billions for long context at 32,768, with 75% 16K–32K and 25%
   4K–16K sequences.

Qwen2.5-VL optical-character-recognition output contributes trillions of text
tokens; Qwen2.5-Math/Coder generate trillions of specialist synthetic tokens.
RoPE base changes 10K to 1M, with YaRN and dual-chunk attention.

#### Stage 1: long-CoT cold start [D]

Math, code, logic, and science prompts have references or executable tests.
Qwen2.5-72B removes easy and unverifiable prompts and balances domains.
QwQ-32B generates \(N\) candidates. Human and automated filters remove wrong,
repetitive, inconsistent, language-mixed, stylistically abnormal, and
validation-like traces. The number of retained examples is intentionally
small, but not published.

#### Stage 2: reasoning RL [D]

- Exactly **3,995 query-verifier pairs**.
- GRPO, large batch, many rollouts, and off-policy updates.
- 170 policy updates.
- The flagship model's AIME 2024 score rises from 70.1 to 85.1 in the
  reported setup.

The report does not provide clipping, KL, batch, rollout count, learning rate,
or hardware.

#### Stage 3: thinking-mode fusion [D]

Stage-2 rejection samples and diverse non-thinking instructions are joined.
The control tokens /think and /no_think choose behavior. Budget control emerges
from the mixture; the report does not claim a separate explicit
reasoning-length objective.

#### Stage 4: general RL [D]

More than 20 task families cover instruction following, formatting,
preferences, agents, and retrieval-augmented generation. Agent samples are
complete multi-turn interactions with real environment feedback. Rewards come
from:

- deterministic rules;
- a reference-aware Qwen2.5-72B judge; and
- a learned reference-free RM.

Reported ablations show better function calling and tool use but small
regressions on some reasoning evaluations. The generality trade-off is explicit.

#### Strong-to-weak distillation [D]

Smaller models first imitate off-policy teacher responses, then generate
on-policy student traces while matching logits from Qwen3-32B or
Qwen3-235B-A22B. In one controlled Qwen3-8B comparison, direct four-stage RL
uses 17,920 GPU-hours and on-policy distillation 1,800 GPU-hours while improving
the listed scores. That is evidence for this experiment, not a universal 10x
law.

### 4.5 GRPO to GSPO to SAPO

#### GRPO baseline

For a prompt with \(G\) sampled responses and terminal rewards \(R_i\), common
GRPO computes

\[
\hat A_i=
\frac{R_i-\operatorname{mean}(R)}
{\operatorname{std}(R)+\varepsilon}
\]

and applies the same sequence advantage to each sampled token. A token-level
PPO-like ratio is

\[
r_{i,t}(\theta)=
\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
{\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})}.
\]

The mismatch is conceptual: reward is assigned to a whole answer, but clipping
can independently retain or discard tokens.

#### GSPO: clip one sequence as one action

Primary source: [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071),
Sections 4–5, pp. 4–5.

GSPO defines a length-normalized sequence ratio

\[
s_i(\theta)=
\left(
\frac{\pi_\theta(y_i\mid x)}
{\pi_{\mathrm{old}}(y_i\mid x)}
\right)^{1/|y_i|}
=
\exp\left[
\frac1{|y_i|}
\sum_t\log r_{i,t}(\theta)
\right]
\]

and optimizes

\[
J_{\mathrm{GSPO}}
=
\mathbb E\left[
\frac1G\sum_i
\min\left(
s_i\hat A_i,\,
\operatorname{clip}(s_i,1-\epsilon,1+\epsilon)\hat A_i
\right)
\right].
\]

The entire answer enters or leaves the trust region together. Controlled
Qwen3-30B-A3B experiments use four minibatches. GSPO lower/upper bounds are
\(3\times10^{-4}\) and \(4\times10^{-4}\), versus GRPO's .2/.27.

The paper finds that about 10% of expert routes in the 48-layer MoE can differ
after one update on the same sample. Token-level GRPO requires routing replay
to reproduce rollout paths; sequence-level GSPO remains stable without it and
is more tolerant of inference/training numerical differences.

**Evidence boundary:** the report says GSPO contributed to "latest Qwen3
models." It does not prove use in the original April 2025 Qwen3 checkpoint.

#### SAPO: replace a hard wall with a smooth gate

Primary source: [Soft Adaptive Policy Optimization](https://arxiv.org/abs/2511.20347),
Section 3, pp. 4–5.

\[
J_{\mathrm{SAPO}}
=
\mathbb E\left[
\frac1G\sum_i\frac1{|y_i|}\sum_t
f_{i,t}(r_{i,t})\hat A_{i,t}
\right],
\qquad
f(x;\tau)=\frac4{\tau}\sigma\!\left(\tau(x-1)\right).
\]

Writing \(p=\sigma(\tau(r-1))\), the soft gate is \(4p(1-p)\), while the full
log-policy gradient also contains the importance ratio and advantage:

\[
\nabla_\theta\!\left[f(r;\tau)\hat A\right]
=
4p(1-p)\,r\,\hat A\,\nabla_\theta\log\pi_\theta.
\]

The gate decays smoothly as a token moves far from the rollout policy instead
of abruptly clipping or masking it. Positive and negative advantages use
different temperatures. The authors motivate
\(\tau_{\mathrm{negative}}>\tau_{\mathrm{positive}}\): decreasing a sampled
token's probability redistributes mass across many unselected vocabulary
items.

Controlled Qwen3-30B-A3B experiments use four minibatches,
\(\tau_{\mathrm{positive}}=1.0\), and
\(\tau_{\mathrm{negative}}=1.05\). The paper associates SAPO with the Qwen3-VL
series. It is not evidence that all Qwen3.5, 3.6, or 3.7 text models use SAPO.

### 4.6 Qwen3-Coder: executable tasks and environment scaling

Primary source: [Qwen3-Coder release](https://qwenlm.github.io/blog/qwen3-coder/).

#### Disclosed recipe [D]

- 480B total / 35B active parameters.
- 256K native context, extendable to one million with YaRN.
- 7.5T pretraining tokens, 70% code.
- Qwen2.5-Coder cleans and rewrites code data.
- Executable code RL uses automatically scaled test cases.
- Long-horizon agent RL trains planning, tool calls, environment feedback, and
  iterative revision.
- **20,000 independent environments** run in parallel on Alibaba Cloud.

This is a production-scale disclosure of environment concurrency, not a full
recipe. **[U]** Optimizer, task count, rollout count, reward formula, sampling,
updates, compute, and hardware topology.

### 4.7 Qwen3-Coder-Next: a reproducible task-construction blueprint

Primary source: [Qwen3-Coder-Next technical report](https://github.com/QwenLM/Qwen3-Coder/blob/main/qwen3_coder_next_tech_report.pdf).

The 80B-total / 3B-active hybrid-attention MoE follows:

~~~text
continued pretraining
  -> supervised fine-tuning
  -> WebDev, UX, single-turn-RL, and SWE-agent-RL experts
  -> distillation into one checkpoint
~~~

#### Environment acquisition [D]

- Mine real issue-related GitHub pull requests.
- Decontaminate benchmark overlap.
- Recover buggy patch, fix patch, and tests.
- Let an agent build a Docker image and executable verifier.
- Use a dedicated model to repair the environment.
- Run a separate quality-assurance agent.
- Add controlled synthetic bugs from SWE-Smith-like generators.

Appendix counts are **807,693 real pull-request instances across 52,960
repositories** and **851,898 synthetic issues across 5,019 used repositories**.
The report summarizes about 800K final verified tasks over more than nine
languages; source counts and final retained counts must not be conflated.

Alibaba's MegaFlow uses Kubernetes plus Argo stages for rollout, evaluation,
and postprocessing, co-locating the agent and environment containers.

#### Continued pretraining and SFT [D]

- Natural GitHub/Common Crawl and targeted sources.
- Roughly 600B repository-level tokens.
- Overall continued-pretraining exposure described only as "trillions."
- Context expands 32,768 to 262,144.
- Next-token prediction plus FIM, best-fit packing, and repetitive-token
  masking.
- Multi-turn traces from SWE-agent, OpenHands, Claude Code, and related
  scaffolds, with strict failure and format filters.
- SFT mixes proprietary corpora, verified agent traces, and
  documentation-grounded question answering.
- A Mini-SWE user simulator executes interaction; a pairwise judge compares
  every candidate pair. Candidate count and SFT count are unknown.

#### RL and anti-hacking [D]

Single-turn code RL uses execution and unit tests. Missing tests can be
synthesized and checked by majority consensus across independent solutions.

SWE-agent RL:

- keeps SFT and RL task sets disjoint;
- filters tasks by model pass rate;
- gives terminal completion reward;
- penalizes unfinished trajectories;
- applies token-level tool-format penalties;
- removes remotes, branches, and tags from environments; and
- blocks suspicious tool calls that combine a repository link with network
  keywords.

Average agent length grows about 50 to 130 turns; evaluation allows 300.
**[U]** Exact policy optimizer, hyperparameters, reward weights, updates, and
distillation loss.

### 4.8 Qwen3.5 to Qwen3.7: stronger agent systems, weaker recipe disclosure

#### Qwen3.5 [D]

Primary source: [Qwen3.5 release](https://qwen.ai/blog?id=qwen3.5).

- 397B total / 17B active.
- Qwen3-Next hybrid of Gated DeltaNet and gated attention, highly sparse MoE,
  and MTP.
- Native early-fusion multimodality, 201 languages, 250K vocabulary, and
  one-million-token hosted context.
- "Tens of trillions" of training tokens; exact exposure is not given.
- Native FP8 activations, routing, and matrix multiplication, retaining BF16
  for sensitive layers.
- Reported about 50% activation-memory reduction and more than 10% speedup.
- Post-training improvements attributed mainly to scaling RL tasks and
  environments.
- Fully disaggregated asynchronous training/inference, routing replay,
  speculative decoding, multi-turn locks, bounded staleness/data skew, and
  million-scale scaffolds/environments.
- Reported 3–5x RL-system speedup.

**[U]** RL algorithm, mixture, exact environments, trajectories, rollout
sampling, rewards, updates, hardware, and total compute.

#### Qwen3.6 [D/U]

- [Qwen3.6-Plus](https://qwen.ai/blog?id=qwen3.6) is a proprietary real-world
  agent model.
- [Qwen3.6-35B-A3B](https://qwen.ai/blog?id=qwen3.6-35b-a3b) and
  [Qwen3.6-27B](https://qwen.ai/blog?id=qwen3.6-27b) release open weights.
- Their releases document capability and deployment, not a new training
  recipe. Shared architecture lineage does not establish shared post-training.

#### Qwen3.7 [D/U]

Primary sources: [Qwen3.7-Max](https://qwen.ai/blog?id=qwen3.7) and
[Qwen3.7-Plus](https://qwen.ai/blog?id=qwen3.7-plus).

Qwen3.7-Max describes task, harness, and verifier as separable dimensions.
Combining them produces cross-harness and cross-verifier RL, reducing
overfitting to one agent wrapper or reward implementation. The release says its
Dynamic Cumulative Survival Games scale training-task temporal complexity to
sequential decision trajectories exceeding 1,000 steps. Separately, it
demonstrates a 35-hour autonomous kernel-optimization run with 1,158 tool calls.
The former is a horizon disclosure and the latter a capability demo; neither
states how many training trajectories were collected.

An SWE reward-hacking monitor accumulated more than 80 hours and 10K tool calls,
added 13 heuristic rules, and flagged 1,618 cases. Those are monitoring-system
statistics, not a disclosed policy-training dataset size.

Qwen3.7-Max is proprietary; Qwen3.7-Plus adds multimodal agent behavior.
**[U]** Parameters, architecture, tokens, optimizer, trajectory count, reward
weights, updates, and training compute.

#### Qwen3-Max-Thinking [D/U]

The [official release](https://qwen.ai/blog?id=qwen3-max-thinking) states that
parameters and RL compute were scaled, then tool fine-tuning and diverse
rule/model-feedback tasks added adaptive Search, Memory, and Code Interpreter
use. Multi-round "take-experience" inference accumulates earlier attempts as
test-time context. No parameter, data, RL, or compute counts are published.

### 4.9 Qwen-AgentWorld: learn the environment, then train the agent

Primary source: [Qwen-AgentWorld](https://qwen.ai/blog?id=qwen-agentworld).

The model predicts the next observation after an agent action in seven domains:
MCP, Search, Terminal, SWE, Web, desktop operating system, and Android. The
three graphical domains are represented as renderable markup rather than
pixels.

#### World-model training [D]

- More than **10M real environment-interaction trajectories**.
- CPT draws on container sandboxes, MCP servers, Android/web/desktop emulators,
  open traces, internal trajectories, and specialist world-knowledge corpora.
- A turn-level information-theoretic mask uses four surface statistics to
  choose action-observation pairs that contain environment information; masked
  turns remain context but do not contribute loss.
- SFT activates explicit next-state reasoning through rejection-sampled
  thinking traces; final SFT count is **7,094**.
- RL uses GSPO and combines a multi-dimensional rubric judge with exact
  per-domain rules.

AgentWorldBench uses ground-truth observations from trajectories generated by
five frontier models across nine benchmarks. It scores format, factuality,
consistency, realism, and overall quality.

#### Controlled simulated RL [D]

The world model can replace the live environment during policy rollouts:

\[
\text{policy action}
\;\longrightarrow\;
\text{world-model observation}
\;\longrightarrow\;
\text{next policy action}.
\]

- 4,000 unseen OpenClaw environments produce +4.3 on Claw-Eval and +7.1 on
  QwenClawBench in the reported comparison.
- An ordinary Qwen3.6-Plus simulator gives negligible benefit, making simulator
  quality a measured bottleneck.
- Controlled MCP simulations inject pagination, intermittent errors, partial
  results, and batch failures. They improve Tool Decathlon +3.7 and MCPMark
  +12.3; uncontrolled simulation can regress.
- 1,000 fictional search environments each contain a self-consistent
  relational database of 300–500 invented rows.
- Controlled simulation reaches 50.3 item-level F1 at step 60 versus 45.6 for
  live-search RL in the reported WideSearch setup.
- Scores average three independent rollouts with 256K maximum sequence length.

The result does **not** mean simulated environments replace real environments.
The source frames simulation as a complementary controllable axis. State detail
and simulator fidelity determine whether simulated RL works.

Only Qwen-AgentWorld-35B-A3B is explicitly confirmed open. The stronger
397B-A17B evaluation model must not be described as released without separate
evidence.

### 4.10 Embodied branch: Qwen-VLA

The [Qwen-VLA release](https://qwen.ai/blog?id=qwenvla) reports about 7.2M
text-to-action trajectories representing more than 14,000 hours across six
templates and six single-arm robots. Training proceeds:

1. freeze the vision-language backbone and train the action decoder on
   language/embodiment prompts;
2. unfreeze both during CPT on multimodal data;
3. branch into multitask and real-robot SFT; and
4. run PPO in SimplerEnv on closed-loop task success.

Simulation-only RL transfers to unseen environments and robot embodiments in
the reported tests. This is embodied policy learning, adjacent to but distinct
from text-only tool agents.

## 5. Meta: from PPO-based RLHF to synthetic data, tools, and online RL

### 5.1 Llama 2: the classical actor-critic baseline

Primary source: [Llama 2](https://arxiv.org/abs/2307.09288), Sections 2–3,
pp. 5–18.

#### Pretraining [D]

- Dense 7B, 13B, 34B, and 70B models; 4K context.
- Pre-RMSNorm, SwiGLU, RoPE, and a 32K SentencePiece tokenizer.
- GQA in 34B and 70B.
- 2T tokens from public sources.
- AdamW with \(\beta_1=.9,\beta_2=.95,\epsilon=10^{-5}\), weight decay .1,
  gradient clipping 1, 2,000-step warmup, and cosine decay to 10% of peak.
- Peak learning rate \(3\times10^{-4}\) for 7B/13B and
  \(1.5\times10^{-4}\) for 34B/70B.
- Global batch 4M tokens.

Training used Meta's Research SuperCluster and production clusters with A100
graphics processing units. Total disclosed pretraining cost is **3,311,616
A100 GPU-hours**:

| Model | GPU-hours |
|---|---:|
| 7B | 184,320 |
| 13B | 368,640 |
| 34B | 1,038,336 |
| 70B | 1,720,320 |

#### SFT and preference acquisition [D]

- 27,540 human-written SFT examples.
- Two SFT epochs, learning rate \(2\times10^{-5}\), weight decay .1, batch 64,
  sequence length 4,096, and prompt-token loss masking.
- 1,418,091 Meta comparisons; **2,919,326 total** with public datasets.
- Separate helpfulness and safety RMs.
- Pairwise logistic ranking plus a discrete preference-strength margin.
- RM training for one epoch, batch 512 pairs, 3% warmup; learning rate
  \(5\times10^{-6}\) for 70B and \(10^{-5}\) for smaller models.

The collection process is iterative. Later policies expose new failure modes;
new comparisons train better RMs; the better RMs select or reward later policy
outputs.

#### Rejection sampling and PPO [D]

Five RLHF rounds are labeled V1–V5. Rejection sampling is used through V4 and
PPO is applied later. Only 70B performs rejection-sampling generation; smaller
models imitate selected 70B outputs.

The PPO reward can be summarized as

\[
R(x,y)=
\operatorname{whiten}\left(\operatorname{logit}(R_c(x,y))\right)
-\beta D_{\mathrm{KL}}
\left(\pi_\theta(\cdot\mid x)\Vert\pi_0(\cdot\mid x)\right).
\]

The safety RM supplies \(R_c\) for safety-tagged prompts or if the safety score
is below .15; otherwise the helpfulness RM supplies it. Meta reports threshold
precision .89 and recall .55. That is high precision but only 55% recall on the
labeled unsafe examples, so it must not be paraphrased as a low-false-negative
setting.

PPO hyperparameters:

- AdamW \(\beta_1=.9,\beta_2=.95\);
- learning rate \(10^{-6}\);
- batch 512, minibatch 64;
- ratio clip .2;
- one gradient update per minibatch;
- KL coefficient .01 for 7B/13B and .005 for 34B/70B;
- 200–400 iterations; and
- about 330 seconds per 70B iteration.

Generation consolidates Fully Sharded Data Parallel shards to reduce latency.
Llama 2 was not explicitly trained on native tools; its report only observes
emergent tool-like syntax.

**[U]** Full corpus, exact rejection-sampling candidate count, complete
preference prompts, reward-model weights, full training code, and total
development compute.

### 5.2 Llama 3 and 3.1: the synthetic-data flywheel

Primary source: [Llama 3 technical report](https://arxiv.org/abs/2407.21783).
All detailed report results are for Llama 3.1.

#### Architecture and pretraining [D]

| Model | Layers | Width | Feed-forward width | Query / KV heads | Peak learning rate |
|---|---:|---:|---:|---:|---:|
| 8B | 32 | 4,096 | 14,336 | 32 / 8 | \(3\times10^{-4}\) |
| 70B | 80 | 8,192 | 28,672 | 64 / 8 | \(1.5\times10^{-4}\) |
| 405B | 126 | 16,384 | 53,248 | 128 / 8 | \(8\times10^{-5}\) |

All are standard dense Transformers with SwiGLU, RoPE base 500K, GQA, and a
128K-token vocabulary. Meta explicitly chose dense models over MoE to reduce
training complexity.

The 405B model is exposed to **15.6T tokens**:

- 50% general knowledge;
- 25% mathematics and reasoning;
- 17% code; and
- 8% multilingual.

Training compute is \(3.8\times10^{25}\) floating-point operations. The initial
405B schedule uses AdamW, 8K-step warmup to \(8\times10^{-5}\), then cosine
decay to \(8\times10^{-7}\) over 1.2M steps. Batch grows:

1. 4M tokens and 4,096 sequence length;
2. 8M and 8,192 after 252M training tokens; and
3. 16M after 2.87T tokens.

Six context-extension stages move 8K to 128K over about 800B tokens. A final
40M-token high-quality annealing stage decays learning rate to zero and is
followed by checkpoint averaging.

Infrastructure:

- up to 16K H100 GPUs;
- Grand Teton servers;
- a 24K-GPU cluster based on Remote Direct Memory Access over Converged
  Ethernet, of which 16K are used for the run;
- 240 petabytes of storage, 2 terabytes/second sustainable and 7
  terabytes/second peak;
- TP, context parallelism, PP, and DP; and
- 54 days with 466 interruptions, 419 unexpected.

The report lists about 43% Model FLOP Utilization at 8,192 GPUs, 41% at 16,384,
and 38% for long context.

#### Six post-training rounds [D]

The final Llama 3.1 procedure is:

~~~text
reward model initialization
  -> rejection sampling
  -> supervised fine-tuning
  -> direct preference optimization
  -> repeat with a stronger checkpoint and cleaner data
~~~

- SFT: learning rate \(10^{-5}\), about 8.5K–9K steps.
- DPO: learning rate \(10^{-5}\), \(\beta=.1\).
- Formatting tokens are masked from DPO loss.
- Chosen-response next-token loss is added with coefficient .2.
- PPO was explored; DPO was selected for lower compute and stronger instruction
  following in the authors' comparison.
- Rejection sampling draws 10–30 candidates.
- Paged attention more than doubles rejection-sampling throughput.
- The
  [Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)
  reports more than **25M synthetic fine-tuning examples**.

Preference examples are about 81.99% general, 6.93% code, 5.19% multilingual,
and 5.89% combined reasoning/tools. SFT mixture is 52.66% general, 14.89% code,
3.01% multilingual, 8.14% exams, 21.19% reasoning/tools, and .11% long context.
The report gives percentages, not the underlying row count for each table.

Quality control combines topic classification, RM and Llama-based scores,
difficulty estimates, and semantic deduplication. Reasoning traces use answer
filters, self-verification, outcome and process RMs, tree search for difficult
problems, Python execution, and explicit corrections from failed solutions.

Only .1% synthetic long-context SFT is needed in the reported optimum. DPO can
remain short-context without losing the long-SFT benefit.

#### Tool training [D]

Primary section: [Llama 3 report, Section 4.3.5, pp. 24–26](https://arxiv.org/pdf/2407.21783#page=25).

Core tools are Brave Search, Python, and Wolfram Alpha. The model can plan
several calls, reason over returned observations, and invoke zero-shot functions
defined only by signatures and documentation.

Data operation differs from ordinary preference collection:

1. annotate at individual assistant-message level so humans can separately
   judge tool selection and reasoning over observations;
2. do not rank or edit tool outputs generated by the environment;
3. bootstrap basic behavior with synthetic data from earlier Llama 3
   checkpoints;
4. start with single-call data;
5. progress to multi-call dialogue, file interactions, and difficult scenarios;
   and
6. add human annotations after synthetic fine-tuning reduces edit burden.

Single-step tool examples have the structure system prompt → user prompt → tool
call → tool output → final answer. About 30% are filtered for error and
formatting problems. Multi-step examples interleave reasoning and calls. File
tasks cover plain text, Word, PDF, PowerPoint, spreadsheet, comma-separated
value, tab-separated value, Python, JavaScript object notation, notebook,
hypertext markup, and extensible markup files.

Function-call synthesis uses public function definitions and multi-agent API
generation, teaching unseen, nested, parallel, and multi-turn calls.

Meta does **not** perform rejection sampling for tool use because it observes no
gain on its tool benchmarks. This is a valuable negative result: a data method
that helps answer quality need not help interactive actions.

#### Version boundary [D]

The April 2024 [Llama 3 announcement](https://ai.meta.com/blog/meta-llama-3/)
states that the initial 8B/70B chat models combine SFT, rejection sampling,
PPO, and DPO. The July 2024 technical report's final Llama 3.1/405B recipe is
iterative rejection sampling, SFT, and DPO, with PPO only explored. Both are
primary-source claims about different releases; neither should overwrite the
other.

### 5.3 Llama 3.2 and 3.3

Official sources:
[Llama 3.2 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md)
and
[Llama 3.3 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md).

- Llama 3.2 1B/3B are pruned and knowledge-distilled from Llama 3.1 8B/70B,
  then post-trained with SFT, rejection sampling, and DPO [D].
- They support 128K and agentic retrieval in the published use cases.
- Llama 3.3 70B reports more than 15T pretraining tokens and more than 25M
  synthetic fine-tuning examples [D].
- No materially new optimizer or agent-environment recipe is disclosed [U].

### 5.4 Llama 4: multimodal MoE and high-level online RL

Primary sources:
[Llama 4 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md)
and
[official launch](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).

| Model | Total / active | Experts | Context | Training tokens | GPU-hours |
|---|---:|---:|---:|---:|---:|
| Scout | 109B / 17B | 16 | 10M | about 40T | about 5M H100 |
| Maverick | about 400B / 17B | 128 | 1M | about 22T | about 2.38M H100 |

Architecture alternates dense and MoE layers. Maverick uses one shared expert
and one of 128 routed experts per token. Text and vision are early-fused; a
MetaCLIP vision encoder is aligned while the Llama backbone is initially
frozen. More than 200 pretraining languages are represented, with more than
100 exceeding one billion tokens. The report describes FP8 teacher training
and up to 32K GPUs for the Behemoth teacher.

Post-training is disclosed as [D]:

1. lightweight SFT;
2. online RL; and
3. lightweight DPO.

More than 50% of easy prompts are removed; difficulty increases continuously
and medium/hard samples are filtered. Maverick is codistilled from the
unreleased Behemoth model with a dynamic hard/soft target. The exact loss is not
published.

Behemoth is described as about 2T total / 288B active with 16 experts and still
training at launch. Its large-RL experiments use a pass-at-\(k\) difficulty
curriculum, zero-advantage filtering, mixed capabilities/system prompts, fully
asynchronous infrastructure, and flexible GPU allocation, with a reported
roughly 10x systems-efficiency improvement.

**Evidence boundary:** these are Behemoth/large-RL disclosures. They do not
establish that every tactic was identically applied to Scout and Maverick.

**[U]** Scout/Maverick online-RL algorithm, reward functions, environment
counts, rollout counts, update count, learning rates, and RL compute.

### 5.5 The post-Llama boundary: Muse Spark

The official [Llama model repository](https://github.com/meta-llama/llama-models)
still ends at Llama 4 at this cutoff. Meta's new proprietary frontier family is
Muse, not "Llama 5."

#### Muse Spark [D/U]

The [April 2026 release](https://ai.meta.com/blog/introducing-muse-spark-msl/)
describes a natively multimodal reasoning model with tools and multi-agent
orchestration:

- a rebuilt architecture, optimizer, data curation, and pretraining stack;
- smooth log-linear pass-at-1 and pass-at-16 improvement as RL steps scale;
- generalization gains on held-out tasks;
- an RL objective maximizing correctness subject to a thinking-time penalty;
- a reported phase transition from length growth to "thought compression" and
  then renewed capability growth; and
- parallel test-time agents that add compute without proportional latency.

More than 1,000 physicians curate health-training data, but that is a domain
subset, not total post-training size. **[U]** Parameters, pretraining tokens,
RL algorithm, reward coefficients, environment count, hardware, and total
compute.

#### Muse Spark 1.1 [D/U]

The [July 2026 release](https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/)
trains for personal tool workflows and complex coding harnesses. The model
zero-shot generalizes to new native tools, MCP servers, and skills. It is
trained to act as either main agent or subagent, plan, delegate in parallel,
escalate, compact context, and operate across diverse multi-turn harnesses.

It is available through Meta's Model API and Meta AI products; weights and a
training report are not released. The source does not provide parameters,
tokens, optimizer, environments, trajectory count, reward, or compute.

## 6. Mistral: efficient bases, real code trajectories, and explicit reasoning RL

### 6.1 Mistral 7B: architecture disclosure without a training recipe

Primary source: [Mistral 7B](https://arxiv.org/abs/2310.06825), Sections 2–4.

The Apache-2.0 model has [D]:

- 7B parameters, 32 layers, width 4,096;
- feed-forward width 14,336;
- 32 query heads, 8 KV heads, head dimension 128;
- 32K vocabulary;
- 4,096-token sliding attention window; and
- 8,192 trained context in the report's architecture table.

GQA reduces KV bandwidth. Sliding-window attention lets layer \(k\) propagate
information approximately \(kW\) positions, giving a theoretical 131K
receptive span across 32 layers. That is not full 131K dense attention. A
rolling buffer caps KV memory at \(W\); at 32K it gives an 8x cache reduction.
Chunked prefill processes long prompts without constructing full quadratic
attention.

The Instruct model is fine-tuned for chat, but the report gives no SFT count or
optimizer. **[U]** Training tokens, source mixture, deduplication, pretraining
optimizer, hardware, GPU-hours, and instruction-tuning recipe.

### 6.2 Mixtral: sparse scaling and offline preference alignment

The [Mixtral 8x7B report](https://arxiv.org/abs/2401.04088) replaces every
feed-forward block with eight SwiGLU experts. A learned router selects two:

\[
\operatorname{MoE}(x)
=
\sum_{i=1}^{8}
\operatorname{softmax}(\operatorname{Top2}(xW_g))_i E_i(x).
\]

It has about 47B total / 13B active parameters, 32 layers, width 4,096,
feed-forward width 14,336, 32 query heads, 8 KV heads, 32K vocabulary, and 32K
full context [D]. Expert parallelism dispatches tokens across devices; uneven
routing creates load-balancing and communication costs.

Mixtral-Instruct uses SFT and DPO [D]. The report does not give data counts,
learning rates, preference construction, or hardware [U].

[Mixtral 8x22B](https://mistral.ai/news/mixtral-8x22b/) extends the pattern to
141B total / 39B active, 64K context, native function calling, and Apache 2.0
base/instruct weights [D]. Its full pretraining and post-training recipe is
unknown.

### 6.3 Mistral Small 3: the non-RL control point

The [official Mistral Small 3 release](https://mistral.ai/news/mistral-small-3/)
explicitly states that the 24B model is **neither trained with RL nor synthetic
data** [D]. It supports low-latency function calling, but tool capability alone
is not proof of agentic RL.

[Mistral Small 3.1](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
adds vision and 128K context. The released configuration confirms 40 text
layers, width 5,120, feed-forward width 32,768, 32 query / 8 KV heads, and
131,072 vocabulary [C]. It becomes the base for Magistral Small and the original
Devstral.

### 6.4 Magistral: modified GRPO from first principles

Primary source: [Magistral](https://arxiv.org/abs/2506.10910).

The report asks two controlled questions:

1. How far can a large instruct checkpoint go with pure RL and no reasoning
   distillation?
2. Can a 24B model improve beyond a strong teacher-trace SFT checkpoint?

Magistral Medium starts from Mistral Medium 3 Instruct and uses pure RL.
Magistral Small starts from 24B Mistral Small 3 Instruct, performs SFT on Medium
reasoning traces, then uses RL.

#### Starting GRPO objective

For prompt \(q\), sampled responses \(o_i\), and token ratios

\[
r_{i,t}(\theta)=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})},
\]

standard GRPO uses group-normalized reward advantage and a PPO-style clipped
surrogate, optionally with a KL penalty to a reference model.

#### Five production modifications [D]

Source: [Magistral Section 2.1, pp. 2–3](https://arxiv.org/pdf/2506.10910#page=3).

1. **Remove the KL penalty and reference model.** Mistral finds the constraint
   does not prevent substantial reasoning-policy divergence and its compute
   cost is unjustified.
2. **Normalize over all generated group tokens.** Sum token losses across all
   responses and divide by \(\sum_i|o_i|\), avoiding an implicit equal weight
   for short and long responses.
3. **Center, but do not standardize, within the prompt group.**
   \(\hat A_i=R_i-\mu_{\mathrm{group}}\).
4. **Standardize advantages across the minibatch.**
5. **Drop zero-variance groups.** All-correct and all-wrong groups contribute no
   useful relative signal and only shrink/noisify the effective gradient.

The resulting objective is

\[
J(\theta)
=
\mathbb E\left[
\frac{1}{\sum_i|o_i|}
\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
\min\left(
r_{i,t}\hat A_i^{\mathrm{mb}},
\operatorname{clip}
\left(r_{i,t},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}}\right)
\hat A_i^{\mathrm{mb}}
\right)
\right],
\]

subject to at least two group rewards differing.

Mistral manually adjusts \(\epsilon_{\mathrm{high}}\) from .26 to .28 during
training to preserve group entropy. This clip-higher setting gives rare,
low-probability useful tokens room to increase.

### 6.5 Magistral's compositional reward

Source: [Magistral Section 2.2, pp. 3–5](https://arxiv.org/pdf/2506.10910#page=4).

#### Format gate [D]

The response must:

- begin with one opening reasoning tag and contain exactly one matching closing
  tag;
- place a mathematical final answer inside a box after reasoning; or
- place code in a typed fenced block after reasoning.

Failure gives reward zero and stops later grading. Valid format gives .1.

#### Verified correctness [D]

Mathematics:

- extract the last boxed answer;
- normalize reference and generated expressions;
- combine several parsers with SymPy symbolic comparison; and
- add .9 if correct, yielding total 1.0 before other shaping.

Code:

- extract the first code block;
- compile C++20 with a 10-second timeout when applicable;
- precompile the common standard-library header;
- select 20 tests, shared within a response group;
- allow 4 seconds and 300 megabytes per test; and
- add .9 only if every selected test passes.

#### Soft length penalty [D]

\[
R_{\mathrm{length}}(y)=
\begin{cases}
0,
& |y|\le l_{\max}-l_{\mathrm{cache}},\\[2mm]
-0.1
\dfrac{|y|-l_{\max}+l_{\mathrm{cache}}}{l_{\mathrm{cache}}},
& l_{\max}-l_{\mathrm{cache}}<|y|\le l_{\max},\\[3mm]
-0.1,
& |y|>l_{\max}.
\end{cases}
\]

It warns the policy before a hard truncation while limiting the shaping
magnitude.

#### Language-consistency reward [D]

Ten percent of English tasks are translated into French, Spanish, Italian,
German, Chinese, and Russian. LaTeX and code are removed, then FastText
classifies prompt, reasoning, and answer. Matching languages add .1. The model
generalizes same-language reasoning beyond the translated training set.

The system prompt itself matters. A phrase allowing informal and long
reasoning raises entropy and exploration in the authors' ablations. System
prompts are therefore part of the training recipe, not merely deployment
decoration.

### 6.6 Magistral's asynchronous trainer-generator-verifier system

Source: [Magistral Section 3, pp. 5–6](https://arxiv.org/pdf/2506.10910#page=6).

Three worker types run continuously:

- **trainers** store the optimized policy and take gradient steps;
- **generators** sample completions and record rollout log-probabilities; and
- **verifiers** execute the reward logic.

The longest completions take up to five times longer than the shortest. A
synchronous group would idle generators and bias throughput toward short
answers. Magistral instead:

1. keeps generators running without waiting;
2. verifies each completed trajectory;
3. sends completions through a fixed permutation to trainer DP groups;
4. takes a gradient step when each group has enough sequences; and
5. broadcasts consolidated new weights directly GPU-to-GPU with NCCL.

The broadcast takes under five seconds even for large world sizes. A generator
can replace weights **mid-generation**. Its existing KV cache is not recomputed,
so later tokens are produced by new weights conditioned on hidden states from
an older policy. The authors find recomputation unnecessary, possibly because
the clipped loss corrects modest off-policy drift.

A bounded queue limits staleness if training becomes the bottleneck. Batches
contain a fixed number of completions, minibatches also use completion counts,
and microbatches use fixed token capacity. A greedy descending-length bin pack
reduces padding by **19%**.

This pipeline illustrates the practical distinction:

\[
\text{freshest possible policy}
\neq
\text{highest hardware utilization}.
\]

Magistral deliberately accepts bounded policy/cache inconsistency to keep
generators saturated.

### 6.7 Magistral data curation

Source: [Magistral Section 4, pp. 6–7](https://arxiv.org/pdf/2506.10910#page=7).

#### Mathematics [D]

| Stage | Problems |
|---|---:|
| initial noisy collection | 699K |
| complete, verifiable, format-filtered | 501K |
| final difficulty-filtered | **38K** |

Processing:

1. remove incomplete, proof-only, and multi-part tasks that cannot be checked
   robustly;
2. rewrite multiple-choice prompts as statement problems;
3. use Mistral Large 2 to sample 16 solutions per prompt;
4. remove never-solved and frequently solved problems;
5. train a 24B model on the first curated set with online RL;
6. use that stronger model for another 16-sample pass over the original set;
7. again remove easy and still-unsolved tasks; and
8. remove likely wrong labels when a majority agrees with one another but not
   the reference.

The second grader matters: the weaker first model can misclassify genuinely
difficult tasks as impossible.

#### Code [D]

- Gather contest problems, solutions, and tests.
- Remove tasks with neither solutions nor adequate tests.
- Execute every solution against available tests.
- Discard low-agreement tests.
- If solutions agree on an output but all disagree with a test, correct the
  test to the modal output.
- Synthesize missing tests and run the same validation.
- Duplicate prompts for Python and C++ where applicable.
- Final set: **35K code problems**.

The final RL pool is therefore only about 73K math+code prompts. Large-scale RL
does not require an enormous unique prompt count if each task is difficult,
verifiable, sampled repeatedly, and updated online.

### 6.8 Magistral Medium: pure RL without reasoning SFT

Source: [Magistral Section 5.2, pp. 8–9](https://arxiv.org/pdf/2506.10910#page=8).

Starting checkpoint: Mistral Medium 3 Instruct. Training has several stages:

1. increase difficulty by adding more complex prompts or removing fully solved
   ones;
2. increase the non-penalized response limit 16K → 24K → 32K; and
3. reduce concurrency, batch, and minibatch as KV-cache cost grows.

Batch and minibatch fall 8K → 4K → 2K. The report does not publish the number
of stages/updates or learning rate.

Vendor-reported results:

The paper evaluates with temperature .7. It uses top-\(p=1.0\) for mathematics
and Graduate-Level Google-Proof Question Answering and top-\(p=.95\) for code,
with a 40K-token cap for AIME and LiveCodeBench and 32K for the other reported
tasks. AIME values average 64 samples per problem and report both pass@1 and
majority@64; LiveCodeBench averages 16 samples. These protocol differences
matter when comparing the table with results from another harness.

| Benchmark | Medium 3 | Magistral Medium |
|---|---:|---:|
| American Invitational Mathematics Examination 2024 pass@1 | 26.8 | 73.6 |
| same benchmark majority@64 | 43.4 | 90.0 |
| MATH-500 | 91.0 | 94.3 |
| Graduate-Level Google-Proof Question Answering Diamond | 59.6 | 70.8 |
| LiveCodeBench v5 | 29.1 | 59.4 |
| Aider Polyglot | 28.9 | 47.1 |

These are evidence that this RL stack improves the selected checkpoint under
the paper's harness, not an independently reproduced cross-vendor ranking.

### 6.9 Magistral Small: teacher cold start plus RL

Source: [Magistral Section 5.3, p. 9](https://arxiv.org/pdf/2506.10910#page=9).

Trace construction [D]:

- collect correct answers generated during Medium RL;
- remove early checkpoints with short reasoning;
- cap traces per problem to avoid easy-task domination;
- upsample lower-pass-rate prompts;
- generate more traces on OpenThoughts and OpenR1 code prompts;
- apply further filtering; and
- add 10% general instruction data to preserve non-reasoning behavior.

Mistral Small 3 Instruct is fine-tuned for four epochs. The checkpoint with the
best American Invitational Mathematics Examination 2024 score seeds RL.

RL settings [D]:

- batch 2,048 sequences;
- maximum non-penalized response length 32K;
- sampling temperature 1.0; and
- \(\epsilon_{\mathrm{high}}=.3\), larger because cold-start SFT reduces policy
  entropy.

The report compares SFT-only, RL-only, and SFT+RL; the combined path is best.
This contradicts a simplistic claim that small models cannot improve past
teacher distillation with RL.

### 6.10 Magistral ablations and negative results

Source: [Magistral Sections 6–8, pp. 11–16](https://arxiv.org/pdf/2506.10910#page=12).

#### Batch, minibatch, and staleness [D]

A 3B model is trained with fixed \(n_{\mathrm{async}}=4096\) and batch/minibatch
values in \(\{1024,2048,4096,8192\}\). Performance is similar when batch equals
minibatch and batch is sufficiently large. More than two minibatches per batch
degrades sharply. Batches at or below 1,024 are less stable.

Final runs therefore maintain

\[
\frac{n_{\mathrm{async}}}{n_{\mathrm{batch}}}\le2,
\qquad
n_{\mathrm{batch}}=n_{\mathrm{minibatch}}.
\]

Minibatch, group, and no advantage normalization give similar evaluation and
length curves; Mistral selects minibatch normalization.

#### What did not work [D]

- **Fractional code reward:** rewarding the fraction of tests passed discards
  three times fewer samples and learns faster, but finishes about two points
  lower on LiveCodeBench after a 250-step 24B ablation. Partial tests can reward
  semantically wrong solutions or verifier inconsistencies.
- **Entropy bonus:** the same coefficient drives entropy down on math-only data
  and explosively up on math+code. Clip-higher is more stable.
- **KL penalty:** a fixed or EMA reference inhibits the intended reasoning
  distribution shift. Manual clip-high adjustment is simpler.

#### Capability preservation [D]

Text-only RL does not degrade the retained vision encoder. Reported multimodal
changes include +5 on Massive Multi-discipline Multimodal Understanding, +4.4
on its Pro Standard variant, and +12 on the Pro Vision variant. Internal
function calling changes 87.2 → 87.4 and instruction following 86.8 → 87.4.

An alternate Medium experiment first SFTs on about **1.3M OpenThoughts/OpenR1
generations**, including DeepSeek-R1 traces, then runs RL on the hardest data.
American Invitational Mathematics Examination 2025 rises more than 12 points,
but Graduate-Level Google-Proof Question Answering Diamond falls 72.9 → 71.0.
RL is capability reallocation, not monotonic improvement on every metric.

### 6.11 Devstral: real agent trajectories, underspecified optimization

Primary source:
[Devstral technical report](https://arxiv.org/abs/2509.25193), Sections 2–4.

The original Devstral-Small is a dense 24B, 40-layer GQA model based on Mistral
Small 3, with a 128K extension [D].

#### Trajectory collection [D]

An agent runs in SWE-Gym through the OpenHands CodeAct scaffold:

1. receive a real software issue and repository state;
2. alternate CoT reasoning with bash/file-edit actions;
3. receive command output and file-state observations;
4. execute unit tests on the final patch; and
5. use test quality to filter the trajectory.

Natural-language examples are mixed in to preserve general capabilities.

#### Two-stage post-training and iteration [D]

1. train on a large rollout subset passing a first heuristic quality filter;
2. fine-tune on only trajectories passing the strictest filters;
3. generate new rollouts from the improved checkpoint; and
4. train further with "policy optimization."

The 2507 refresh adds pseudo-scaffolds, both extensible-markup and native
function-call prompt formats, and tighter trajectory filters. Evaluation uses
OpenHands at a pinned commit, bash and file-edit tools, no web browsing or
retrieval, and up to three attempts with temperature 0, then .1 and .1.

**[U]** The report never identifies the policy-optimization algorithm, reward
formula, training task count, rollout rounds, learning rate, batch, update count,
hardware, or compute. Calling it PPO, GRPO, or Magistral-style RL would be an
unsupported inference.

### 6.12 Devstral 2 and the 2026 unified models

#### Devstral 2 [D/U]

The [December 2025 release](https://mistral.ai/news/devstral-2-vibe-cli/)
provides 123B and 24B models with 256K context. Both explore codebases, edit
multiple files, run tools, and revise failures. The official 123B configuration
in the
[Devstral 2 model repository](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512)
confirms 88 layers, width 12,288, feed-forward width 28,672, 96 query / 8 KV
heads, and YaRN scaling [C].

No updated environment/training report is published [U]. The original
Devstral-Small recipe cannot simply be copied to Devstral 2.

#### Mistral Large 3 [D/U]

The [Mistral 3 launch](https://mistral.ai/news/mistral-3/) describes Mistral
Large 3 as 675B total / 41B active, trained from scratch on 3,000 NVIDIA H200
GPUs. Base and instruction weights use Apache 2.0. Exact tokens, data mixture,
optimizer, post-training method, and compute are unknown.

#### Mistral Small 4 [D/C/U]

The [March 2026 release](https://mistral.ai/news/mistral-small-4/) unifies
Mistral Small instruction following, Magistral reasoning, Devstral agentic
coding, and Pixtral vision:

- 119B total;
- 128 experts, 4 active;
- about 6B active in the Transformer, or 8B including embedding/output;
- 256K context;
- native text+image input and function calls;
- configurable reasoning effort; and
- Apache 2.0 weights.

The official
[model card and configuration](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)
report 6.5B activated, illustrating parameter-count convention differences.
The configuration confirms a Mistral-4 MoE and MLA-like latent attention [C].
No training report establishes which Magistral or Devstral data/optimizer was
reused [U].

#### Mistral Medium 3.5 and Leanstral [D/U]

[Mistral Medium 3.5](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B)
is a dense 128B, 256K multimodal model with configurable reasoning, native
function calls, and a modified MIT license. It replaces earlier Medium,
Magistral, and Devstral roles in Mistral products. Its full recipe is unknown.

[Leanstral 1.5](https://huggingface.co/mistralai/Leanstral-1.5-119B-A6B)
specializes the Small 4 architecture for Lean 4 proof engineering and tool use:
119B total, 6.5B active, 128 experts/4 active, 256K, Apache 2.0. Its training
data, verifier loop, RL method, and compute are not disclosed.

**Generation boundary:** Magistral is the detailed 2025 algorithm report.
Small 4, Medium 3.5, and Leanstral are later capability releases. Similar
behavior and shared ancestry are not evidence of identical RL.

## 7. Cross-case note: SAO, Qwen controls, and the GLM-5.2 evidence boundary

Primary source:
[Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508).
The paper is by Zhenyu Hou, Yujiang Li, Jie Tang, and Yuxiao Dong; it states
that work by Hou and Li occurred while interning at Z.AI.

### 7.1 What is actually established

The abstract says SAO was successfully deployed in the agentic-RL pipeline for
training the open **GLM-5.2 (750B-A40B)** model [D]. The official
[GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2) reports 753B
parameters; the SAO paper rounds the architecture to 750B total / 40B active.

The [official GLM-5.2 release](https://z.ai/blog/glm-5.2) and model card do not
name SAO, and no dedicated GLM-5.2 technical report is cited. The strongest
defensible statement is:

> SAO was deployed somewhere in GLM-5.2's agentic-RL pipeline.

It is **not** proven to be the only optimizer, final optimizer, or optimizer for
every GLM-5.2 capability.

### 7.2 Direct double-sided importance sampling

SAO generates one rollout per prompt and sends it to training immediately.
Stored rollout log-probabilities give

\[
r_t(\theta)
=
\exp\left[
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\mathrm{rollout}}(a_t\mid s_t)
\right].
\]

The objective is

\[
L(\theta)=
\mathbb E_t\left[
f(r_t;\epsilon_l,\epsilon_h)
\hat A_t
\log\pi_\theta(a_t\mid s_t)
\right],
\]

with

\[
f(x;\epsilon_l,\epsilon_h)=
\begin{cases}
x,&1-\epsilon_l<x<1+\epsilon_h,\\
0,&\text{otherwise}.
\end{cases}
\]

PPO normally saturates a ratio at the clipping boundary. SAO instead masks a
token completely when its ratio lies outside either boundary. It also uses the
rollout log-probability directly, avoiding a separately recomputed historical
policy.

### 7.3 Critic design and skip-observation GAE

One rollout cannot estimate a within-prompt group baseline, so SAO restores a
value model [D]:

- two critic updates per actor update, \(K=2\);
- freeze critic attention parameters;
- optimize critic MoE projection parameters; and
- scale value-model pretraining, without publishing its data/count.

For a trajectory
\([a_0,o_0,a_1,o_1,\ldots]\), environment observation tokens are skipped in
GAE. At the end of action \(a_i\),

\[
\delta_i
=
r_i+\gamma V(a_{i+1,0})-V(a_{i,N}),
\]

\[
\hat A(a_{i,N})
=
\delta_i+\gamma\lambda\hat A(a_{i+1,0}).
\]

The observation affects the next state but is not treated as a policy action.
This is the actor-critic analogue of masking interpreter output in
Qwen2.5-Math.

### 7.4 Controlled experiments are Qwen experiments

The detailed tests use **Qwen3-30B-A3B-Thinking-2507**, not GLM-5.2 [D].

Math:

- three-epoch SFT on GPT-OSS-120B traces;
- asynchronous RL batch 128, group size 1, maximum 128K;
- actor learning rate \(10^{-6}\);
- critic learning rate \(5\times10^{-6}\);
- \(\epsilon_l=.3,\epsilon_h=5.0\);
- length-adaptive policy GAE, for trajectory length \(l\),
  \(\lambda_{\mathrm{policy}}=1-1/(1.5\,l)\), critic \(\lambda=1\), critic
  warmup 10, and \(K=2\).

Coding:

- starts directly from the Qwen thinking checkpoint;
- OpenHands, maximum 300 turns / 128K; and
- \(\epsilon_l=.8,\epsilon_h=3.0\).

Reported results:

The mathematics evaluations use pass@1, temperature 1, top-\(p=1\), a 128K
token cap, and as many as 50 turns. AIME, Harvard-MIT Mathematics Tournament,
and International Mathematical Olympiad AnswerBench values average 16 runs;
BeyondAIME averages four. They are controlled results under this harness, not
single deterministic scores.

| Evaluation | GRPO | SAO |
|---|---:|---:|
| American Invitational Mathematics Examination 2025 | 84.2 | 97.3 |
| BeyondAIME | 54.8 | 74.8 |
| Harvard-MIT Mathematics Tournament Nov. 2025 | 76.0 | 88.3 |
| International Mathematical Olympiad AnswerBench | 55.8 | 74.0 |

SWE-bench Verified is 23.0 for the base checkpoint, 27.0 for GRPO plus direct
importance sampling, and 29.8 for SAO. Vanilla GRPO collapses near update 160
in the plotted run; SAO remains stable to roughly 1,000.

Ablations on American Invitational Mathematics Examination / BeyondAIME:

- full SAO: 97.3 / 74.8;
- one critic update: 95.0 / 69.8;
- do not freeze critic attention: 90.6 / 74.5;
- an alternative value-model baseline without direct importance sampling:
  91.3 / 69.0; and
- running-mean advantage: 79.8 / 55.3.

### 7.5 What remains unknown for GLM-5.2

The Qwen controlled settings must not be transferred to GLM-5.2. Unknown:

- SAO's stage scope and whether another optimizer follows it;
- GLM prompts, task families, environments, and reward functions;
- actor/critic initialization and value-pretraining corpus;
- rollout count, queue size, batch, clipping thresholds, and learning rates;
- update count, hardware, runtime, and compute; and
- the proportion of GLM-5.2 behavior attributable to SAO.

For the fuller Zhipu lineage, use the dedicated [GLM case study](glm.md).

## 8. Comparative anatomy of real training operations

### 8.1 Data and environment construction

| Operation | Qwen | Meta | Mistral |
|---|---|---|---|
| Broad pretraining scale | 18T Qwen2.5; 36T Qwen3; "tens of trillions" Qwen3.5 | 2T Llama 2; 15.6T Llama 3.1; about 22T/40T Llama 4 variants | early token counts unknown; later flagship counts mostly unknown |
| Synthetic-data engine | specialist Qwen teachers, OCR, code/math execution, world-model trajectories | flagship self-generation, rejection sampling, process/outcome RMs, tree search | Magistral Medium traces; Devstral/OpenHands rollouts |
| Learnability filter | response-score variance; 2–5/8 math successes; repeated pass-rate filtering | RM/difficulty scoring and semantic deduplication | 16-sample two-pass math grading; remove zero-variance groups |
| Executable environment | Python, compilers, SWE containers, MCP, web/desktop/Android emulators | Python/search/Wolfram and file tools; environment-RL counts unknown | code tests; SWE-Gym/OpenHands; Docker-like code workflows |
| Scaled agent infrastructure | 20K Qwen3-Coder environments; >10M AgentWorld trajectories | tool annotations detailed; later agent scale proprietary | asynchronous Magistral; Devstral real issue trajectories |
| Anti-hacking | strip repository network escape routes; Qwen3.7 monitor/rules | safety/post-training and system guardrails, details incomplete | test consensus/correction; Devstral filters, details incomplete |

The common loop is:

~~~text
mine or generate task
  -> instantiate resettable environment and hidden verifier
  -> run several candidate policies
  -> execute, score, and inspect failures
  -> remove too-easy, impossible, leaked, or unstable tasks
  -> train with SFT, preference loss, or online RL
  -> regenerate with the improved policy
  -> repeat while monitoring held-out environments
~~~

### 8.2 Reward design

| Signal | Strength | Failure mode | Public example |
|---|---|---|---|
| exact symbolic answer | cheap, objective | narrow parser, bad reference | Qwen2.5-Math; Magistral |
| all-tests-pass | grounded in behavior | weak/incorrect tests, environment escape | Qwen Coder; Magistral; Devstral |
| partial tests | denser feedback | rewards semantically wrong partial behavior | Magistral negative ablation |
| learned scalar RM | covers open-ended preference | bias, drift, reward hacking | Llama 2; Qwen2.5 |
| rubric LLM judge | flexible dimensions and references | evaluator leakage and judge dependence | Qwen3 general RL; AgentWorld |
| format reward/penalty | makes parsing reliable | surface compliance without competence | Magistral; Qwen Coder-Next |
| length/time penalty | limits cost and truncation | prematurely compresses reasoning | Magistral; Muse Spark |
| language consistency | preserves user language | classifier errors, surface gaming | Magistral |

A robust industrial reward is usually a composition with early gates:

\[
R(\tau)
=
\mathbf 1[\text{valid format and environment}]
\left(
w_{\mathrm{task}}R_{\mathrm{task}}
+w_{\mathrm{judge}}R_{\mathrm{judge}}
+w_{\mathrm{cost}}R_{\mathrm{cost}}
\right).
\]

This equation is a conceptual abstraction [I], not one vendor's exact loss.
The gate prevents expensive or unreliable grading of malformed trajectories.

### 8.3 Policy optimization and systems

| Pipeline | Baseline | Off-policy control | Collection behavior |
|---|---|---|---|
| Llama 2 PPO | learned critic | ratio clip + KL to reference | staged synchronous RLHF |
| Qwen/DeepSeek-style GRPO | response group | token ratio clip, sometimes KL | waits for group statistics |
| GSPO | response group | one length-normalized sequence ratio | aligns sequence reward and clipping |
| SAPO | response group | smooth token gate, sign-dependent temperature | avoids discontinuous trust boundary |
| Magistral GRPO | group mean + minibatch normalization | asymmetric token clipping, no KL | fully asynchronous, mid-generation weights |
| SAO | learned critic | direct two-sided token mask | one rollout immediately enters training |

No row is universally best:

- Group methods avoid value-model cost but multiply rollouts and create a
  straggler barrier.
- Critics support single rollouts and token/time credit but can destabilize.
- Sequence ratios match sequence rewards but may hide isolated pathological
  tokens.
- Token gates can target local staleness but may fragment a sequence's update.
- Asynchrony improves utilization while increasing policy/cache staleness.

### 8.4 Capability preservation is an explicit objective

Specialist RL can narrow a model. Public countermeasures include:

- Qwen3 general RL across more than 20 task families;
- 10% general instructions in Magistral Small cold-start SFT;
- natural-language data in Devstral;
- broad synthetic and preference mixtures in Llama;
- distillation back into a unified student; and
- held-out evaluations for tools, instructions, vision, and general reasoning.

The evidence still shows trade-offs. Qwen3's general RL slightly regresses some
reasoning tests. Magistral's alternate SFT+RL improves math/code while its
Graduate-Level Google-Proof Question Answering score falls. A single aggregate
reward is not a guarantee of Pareto improvement.

## 9. Openness and reproducibility

| Release family | Weights | Architecture/config | Data counts | RL equations/hyperparameters | Full reproduction |
|---|---|---|---|---|---|
| Qwen2.5 / Qwen3 | many open | strong | aggregate counts | partial to strong in specialist reports | no |
| Qwen3.5–3.7 | mixed open/proprietary | strong for open checkpoints | increasingly vague | mostly unknown | no |
| Qwen-AgentWorld | 35B-A3B open | strong | >10M + 7,094 | GSPO and reward type, many details unknown | no |
| Llama 2 | custom-license weights | strong | strong aggregate counts | unusually detailed PPO | no |
| Llama 3 / 4 | custom-license weights | strong | strong aggregates | DPO detailed in 3.1; online RL vague in 4 | no |
| Muse | no public weights | sparse | almost none | qualitative objective only | no |
| Mistral 7B / Mixtral | Apache weights | strong | absent | absent or offline method name | no |
| Magistral Small | Apache weights | inherited config | strong RL-prompt counts | strongest Mistral disclosure | no |
| Devstral | open weights | strong | trajectory method, no count | optimizer unnamed | no |
| Small 4 / Medium 3.5 | open weights | strong | absent | absent | no |

To reproduce a frontier agentic-RL result, weights are only one artifact. A
complete release would need:

1. raw and filtered prompt identifiers;
2. synthesis prompts, teacher versions, and sampling parameters;
3. environment images, reset logic, network policy, and tool schemas;
4. verifier source, hidden tests, timeouts, and judge prompts;
5. every SFT/preference/RL mixture and mask;
6. rollout log-probabilities, policy versions, queue policy, and staleness
   limit;
7. model, optimizer, scheduler, precision, parallelism, and random seeds;
8. reward models, value models, and checkpoint selection rules;
9. hardware topology, failures/restarts, wall time, and accelerator hours; and
10. evaluation scaffold, inference budget, and aggregation.

No Qwen, Meta, or Mistral frontier release in this chapter publishes all ten.

## 10. Operational lessons

1. **Treat task difficulty as a moving property of the policy.** A static
   dataset becomes all-correct or all-wrong. Re-score pass rates and refresh the
   curriculum after each material checkpoint change.
2. **Store action ownership.** Model tokens, tool outputs, user-simulator
   messages, and environment observations require different loss masks.
3. **Version everything.** Every trajectory should record policy checkpoint,
   tokenizer, system prompt, harness, tool schema, environment image, verifier,
   sampling parameters, and rollout log-probabilities.
4. **Use hidden validation and adversarial environments.** Public tests invite
   hard-coding; unrestricted networks invite answer retrieval; writable git
   metadata invites benchmark shortcuts.
5. **Measure reward variance before training.** All-equal response groups
   produce no relative signal. Low verifier agreement indicates an environment
   bug, not necessarily a difficult task.
6. **Budget the KV cache, not only parameters.** Long agent trajectories can
   make rollout memory the binding constraint, forcing lower concurrency and
   batch sizes as in Magistral.
7. **Separate algorithmic from systems gains.** A policy method can look better
   because it drops stragglers, changes effective batch, sees fresher data, or
   uses more environment attempts.
8. **Report the denominator.** "800K tasks," "10M trajectories," "20K
   environments," and "2,048 sequences per batch" describe different units.
9. **Preserve negative results.** Magistral's partial reward, entropy bonus,
   and KL failures are as reusable as its final objective.
10. **Audit retroactive attribution.** A later optimizer paper, capability
    demo, or shared architecture never proves how an earlier checkpoint was
    trained.

## 11. A research audit checklist

Before accepting a vendor training claim, ask:

- Is the source a model report, model card, released config, engineering post,
  or only a product demo?
- Does "tokens" mean unique corpus size or repeated training exposure?
- Are parameter counts total, active, non-embedding, or checkpoint tensors?
- Are task counts raw mined instances, constructed environments, retained
  prompts, trajectories, or policy samples?
- Is an optimizer named, mathematically specified, or merely described as RL?
- Are detailed hyperparameters from the actual flagship or a small controlled
  experiment?
- Are benchmarks pass@1, majority vote, best-of-\(N\), or multi-agent?
- Does an agent score include a custom scaffold, tools, context compaction, or
  retries?
- Is a claimed release open-weight, source-available, API-only, or fully
  reproducible?
- What is explicitly unknown?

Those questions prevent the two most common errors in frontier-model analysis:
turning a plausible implementation into a historical fact, and turning an open
checkpoint into a claim of open training.
