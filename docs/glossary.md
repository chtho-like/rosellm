# Glossary: Full Names, Meanings, and Relationships

This glossary is a decoding aid. Every chapter must still expand an uncommon
abbreviation at its first meaningful use and immediately explain what the
concept does. A full name answers “what do the letters mean?”; the explanation
answers the more important question, “what does this object do here?”

## Learning paradigms

| Term | Full name | Plain meaning and boundary |
|---|---|---|
| **AI** | Artificial Intelligence | The broad field of machines performing tasks associated with intelligence. Machine learning is one approach within it. |
| **LM / LLM** | Language Model / Large Language Model | A model assigns probabilities to token sequences. “Large” has no fixed parameter threshold. |
| **VLM / VLA** | Vision-Language Model / Vision-Language-Action model | A VLM processes images and language; a VLA additionally produces actions, commonly for robotics. |
| **RL** | Reinforcement Learning | Learning a behavior policy from scalar rewards produced through interaction. |
| **agentic RL** | Agentic Reinforcement Learning | RL in which an LLM acts over multiple steps, often using tools and receiving delayed environment feedback. It is a setting, not one algorithm. |
| **SFT** | Supervised Fine-Tuning | Maximum-likelihood training on supplied target responses or trajectories. It imitates demonstrations rather than directly maximizing a reward. |
| **RLHF** | Reinforcement Learning from Human Feedback | RL in which human judgments directly or indirectly define reward, often through a learned reward model. |
| **RLAIF** | Reinforcement Learning from AI Feedback | RL whose labels, critiques, or rewards are produced by another AI system. |
| **RLVR** | Reinforcement Learning with Verifiable Rewards | RL using mechanically checkable rewards such as unit tests, exact answers, or proof verification. “From verifiable rewards” is also used. |
| **BC / IL** | Behavioral Cloning / Imitation Learning | BC is supervised imitation of demonstrated actions; IL is the broader family. |
| **KD / OPD** | Knowledge Distillation / On-Policy Distillation | KD trains a student from a teacher. OPD evaluates the teacher on states sampled by the student, reducing state-distribution mismatch. |
| **CoT** | Chain of Thought | Intermediate language or latent reasoning before an answer. Long-CoT denotes deliberately extended traces. |

## MDP, POMDP, and trajectories

### Markov Decision Process (MDP)

An **MDP** is the mathematical model

\[
\mathcal M=(\mathcal S,\mathcal A,P,r,\gamma,\rho_0),
\]

where \(\mathcal S\) is the state space, \(\mathcal A\) the action space,
\(P(s'\mid s,a)\) the transition law, \(r\) the reward, \(\gamma\) the discount
factor, and \(\rho_0\) the initial-state distribution. “Markov” means the
current state contains everything needed to predict the next state once the
action is known.

### Partially Observable Markov Decision Process (POMDP)

A **POMDP** is an MDP in which the agent cannot directly observe the complete
state. Instead it receives \(o_t\sim O(\cdot\mid s_t)\) and must infer what
matters from its interaction history or learned memory. Tool-using LLM agents
are normally closer to POMDPs: hidden web state, omitted files, asynchronous
events, and truncated context make the true environment state only partially
visible.

| Term | Meaning |
|---|---|
| **state** \(s_t\) | All environment information that makes future dynamics independent of the earlier past, conditional on the next action. It may be hidden. |
| **observation** \(o_t\) | Information exposed to the agent, such as a prompt, terminal output, or screenshot. |
| **action** \(a_t\) | A decision affecting the environment: a token, message, tool call, or structured command. |
| **policy** \(\pi_\theta(a\mid h)\) | A parameterized probability distribution over actions given observable history \(h\). |
| **trajectory** \(\tau\) | One ordered record of observations, actions, rewards, and termination signals. |
| **episode / horizon** | One complete or truncated interaction / its maximum or realized number of decision steps. |
| **return** \(G_t\) | Discounted future reward, \(G_t=\sum_{k\ge0}\gamma^k r_{t+k}\). |
| **value** \(V^\pi(s)\) | Expected return from a state under policy \(\pi\). |
| **action value** \(Q^\pi(s,a)\) | Expected return after action \(a\), then following \(\pi\). |
| **advantage** \(A^\pi(s,a)\) | How much better an action is than the policy average: \(Q^\pi(s,a)-V^\pi(s)\). |
| **credit assignment** | Determining which earlier actions deserve responsibility for a delayed outcome. |
| **on-policy / off-policy** | Data comes from the current policy / from a teacher, replay buffer, older checkpoint, or other behavior policy. |

## Policy optimization and estimators

| Term | Full name | Plain meaning and relationship |
|---|---|---|
| **REINFORCE** | A Monte Carlo policy-gradient estimator; not an acronym in the original paper | Multiplies sampled log-probability gradients by returns or advantages. It is simple and often high variance. |
| **PG** | Policy Gradient | Direct differentiation of expected return with respect to policy parameters. |
| **TRPO** | Trust Region Policy Optimization | Constrains updates by a Kullback–Leibler divergence trust region. |
| **PPO** | Proximal Policy Optimization | Reuses rollouts while clipping or penalizing policy-ratio movement. Actor-critic PPO learns a value function. |
| **GAE** | Generalized Advantage Estimation | A geometrically weighted mixture of temporal-difference residuals, trading bias against variance through \(\lambda\). |
| **TD** | Temporal Difference | Bootstraps from a later value estimate instead of waiting for the complete return. |
| **RLOO** | REINFORCE Leave-One-Out | Uses the other responses to the same prompt as a reward baseline; no critic is required. |
| **GRPO** | Group Relative Policy Optimization | DeepSeek's publicly introduced critic-free PPO-like method; it derives relative advantages from a response group for one prompt. |
| **Dr. GRPO** | A named correction in *Understanding R1-Zero-Like Training* | Removes response-length and group-standard-deviation normalizations identified as bias sources. The label has no standard expanded phrase. |
| **DAPO** | Decoupled Clip and Dynamic sAmpling Policy Optimization | A GRPO-family recipe with asymmetric clipping, dynamic sampling, token-level loss, and overlong-reward shaping. |
| **GSPO** | Group Sequence Policy Optimization | Uses a sequence-level importance ratio rather than independently clipping token ratios. |
| **SAO** | Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning | Consumes each long trajectory when it finishes, uses direct double-sided importance masking, and restores a critic for group-size-one training. The 2026 authors report deployment in GLM-5.2. |
| **DIS** | Direct Double-Sided Importance Sampling | SAO's direct rollout-to-current-policy importance ratio with strict two-sided rejection outside a permitted interval. It is masking, not PPO's saturated clipping. |
| **DPO** | Direct Preference Optimization | Turns preference pairs into a classification-style policy objective without an online RL loop or separate scalar reward model. |
| **IPO** | Identity Preference Optimization | A preference objective designed to avoid some overfitting behavior of logistic DPO. |
| **KTO** | Kahneman–Tversky Optimization | A prospect-theory-inspired objective that can train from unpaired desirable/undesirable labels. |
| **ORPO** | Odds Ratio Preference Optimization | Adds an odds-ratio preference term to supervised likelihood, normally without a separate reference model. |
| **PRIME** | Process Reinforcement through IMplicit rEwards | Learns token-level process rewards from outcome-labeled rollouts. |
| **V-trace** | A truncated importance-sampling return estimator | Corrects trajectories from lagging actors toward the current learner policy. |
| **IMPALA** | Importance Weighted Actor-Learner Architecture | A distributed actor/learner architecture that introduced V-trace. |

## Reward and evaluation vocabulary

| Term | Full name | Meaning |
|---|---|---|
| **RM / ORM / PRM** | Reward Model / Outcome Reward Model / Process Reward Model | An RM predicts quality; an ORM scores the final outcome; a PRM scores intermediate steps. |
| **GRM** | Generative Reward Model | Generates a critique or structured judgment instead of only a scalar. |
| **LLM judge** | Large Language Model judge | A model prompted or trained to evaluate another output; it is fallible and not automatically ground truth. |
| **KL** | Kullback–Leibler divergence | A directional mismatch, \(D_{KL}(p\Vert q)=\mathbb E_p[\log p-\log q]\), often used to limit policy drift. |
| **TV** | Total Variation distance | A symmetric distance, \(\tfrac12\sum_x|p(x)-q(x)|\). |
| **pass@k** | pass at \(k\) | Probability that at least one of \(k\) sampled candidates passes; it requires a sampling protocol. |
| **IoU / F1** | Intersection over Union / harmonic mean of precision and recall | Overlap metric for regions / class metric \(2PR/(P+R)\). |

## Transformer architecture

| Term | Full name | Meaning |
|---|---|---|
| **MHA / MQA / GQA** | Multi-Head / Multi-Query / Grouped-Query Attention | MHA has separate heads; MQA shares one key/value head; GQA shares fewer key/value heads across groups of query heads. |
| **MLA** | Multi-head Latent Attention | DeepSeek's compressed latent key/value representation for lower inference cache and bandwidth. |
| **DSA** | DeepSeek Sparse Attention | DeepSeek's long-context mechanism that selects a limited set of relevant tokens. |
| **KDA** | Kimi Delta Attention | Moonshot's gated delta-rule linear attention. |
| **MoBA** | Mixture of Block Attention | Moonshot's parameter-free sparse routing of each query to selected context blocks. |
| **AttnRes** | Attention Residuals | Moonshot's learned attention over earlier residual streams instead of a fixed cumulative sum. |
| **MoE** | Mixture of Experts | Routes each token to a small subset of feed-forward experts, increasing total parameters without proportional active compute. |
| **FFN / MLP** | Feed-Forward Network / Multilayer Perceptron | The per-token nonlinear sublayer / the general stack of linear layers and nonlinearities. |
| **MTP** | Multi-Token Prediction | Predicts several future tokens from one position, potentially improving representations and speculative decoding. |
| **FIM** | Fill-in-the-Middle | Generates a missing span conditioned on both its prefix and suffix. |
| **RoPE** | Rotary Position Embedding | Encodes relative position by rotating query/key channels. |
| **YaRN** | Yet another RoPE extensioN method | Extends RoPE context through frequency scaling and attention-temperature adjustment. |
| **ALiBi** | Attention with Linear Biases | Adds head-specific distance biases to attention logits. |
| **RMSNorm** | Root Mean Square Layer Normalization | Normalizes magnitude without subtracting the mean. |
| **SwiGLU / GeGLU** | Swish-Gated / Gaussian Error Linear Unit Gated Linear Unit | Two gated feed-forward activations. |
| **QK-Norm** | Query-Key Normalization | Normalizes attention queries and keys before the dot product. |
| **KV cache** | Key-Value cache | Stores earlier attention keys/values to avoid recomputing them during decoding. |
| **QAT** | Quantization-Aware Training | Simulates low precision during training so the model adapts before quantized inference. |

## Data and representation

| Term | Full name | Meaning |
|---|---|---|
| **BPE / BBPE** | Byte Pair Encoding / Byte-Level Byte Pair Encoding | Frequent-pair subword merging / the byte-based variant that can represent any input. |
| **OCR** | Optical Character Recognition | Converts pixels or scanned pages into text. |
| **LSH / MinHash** | Locality-Sensitive Hashing / Minimum Hashing | Approximate similarity indexing / Jaccard-similarity estimation, often used in deduplication. |
| **PII** | Personally Identifiable Information | Data that can identify or link to a person. |
| **AST** | Abstract Syntax Tree | A tree representation of program syntax. |

## Distributed training and serving

| Term | Full name | Meaning |
|---|---|---|
| **GPU** | Graphics Processing Unit | The dominant parallel accelerator for modern LLM work. |
| **FLOP / FLOPs** | Floating-Point Operation / Floating-Point Operations | Arithmetic work; “per second” must be stated explicitly when it denotes a rate. |
| **MFU** | Model FLOPs Utilization | Achieved model arithmetic divided by theoretical accelerator peak under a stated precision convention. |
| **DP / DDP** | Data Parallelism / Distributed Data Parallel | Replicates parameters, splits examples, and synchronizes gradients. |
| **TP / PP** | Tensor Parallelism / Pipeline Parallelism | Shards within layers / places consecutive layer groups on different devices. |
| **EP / SP / CP** | Expert / Sequence / Context Parallelism | Shards experts / sequence work / long-context attention across devices. |
| **FSDP** | Fully Sharded Data Parallel | Shards parameters, gradients, and optimizer state across data-parallel workers. |
| **ZeRO** | Zero Redundancy Optimizer | Removes replicated optimizer state, gradients, and then parameters in stages. |
| **RDMA** | Remote Direct Memory Access | Moves data between registered memory on machines with little CPU involvement. |
| **RPC / HTTP / API** | Remote Procedure Call / Hypertext Transfer Protocol / Application Programming Interface | Common ways to invoke a remote component or exposed service. |
| **TITO** | Token-In, Token-Out | GLM-5's asynchronous design that exchanges exact tokens and behavior log-probabilities to reduce rollout/training mismatch. |
| **TTFT / ITL / QPS** | Time to First Token / Inter-Token Latency / Queries per Second | First-token delay / decode spacing / request throughput. |
| **1F1B** | One Forward, One Backward | A pipeline schedule alternating forward and backward microbatches after warmup. |
| **WAL** | Write-Ahead Log | A durable record written before mutation for replay and recovery. |

## Precision and hardware

| Term | Full name | Meaning |
|---|---|---|
| **FP32 / FP16 / FP8 / FP4** | 32-/16-/8-/4-bit floating point | Bit width alone does not specify exponent layout, mantissa layout, or scaling. |
| **BF16 / TF32** | Brain Floating Point 16 / TensorFloat-32 | A 16-bit format with FP32-like exponent range / NVIDIA's reduced-mantissa matrix format. |
| **E4M3 / E5M2** | 4 exponent + 3 mantissa bits / 5 exponent + 2 mantissa bits | Two FP8 range/precision trade-offs. |
| **MXFP4 / MXFP8** | Microscaling 4-/8-bit floating point | Low-precision values sharing scale metadata in small blocks. |
| **A100, A800, H100, H800** | NVIDIA product names, not acronyms | Accelerator variants with distinct compute, memory, and interconnect limits. |
| **NVLink / NVSwitch** | NVIDIA device interconnect / its switch fabric | High-bandwidth GPU links / an all-to-all fabric connecting them. |
| **RoCE** | RDMA over Converged Ethernet | Remote Direct Memory Access transported over Ethernet. |

## Agents and software

| Term | Full name | Meaning |
|---|---|---|
| **agent scaffold** | — | Runtime prompt construction, tool schema, context management, retries, stopping, and persistence around the model. |
| **MCP** | Model Context Protocol | An open protocol exposing tools and contextual resources to AI applications. |
| **GUI / CLI** | Graphical User Interface / Command-Line Interface | Visual point-and-type interaction / textual command interaction. |
| **sandbox** | — | Isolation limiting filesystem, network, process, or credential access. |
| **prompt injection** | — | Untrusted content that attempts to redirect an agent away from authorized instructions. |
| **reward hacking** | — | Raising the measured reward through unintended behavior rather than the desired outcome. |
| **policy lag** | — | Difference between the rollout-generating policy and the newer learner policy. |
| **JSON / XML** | JavaScript Object Notation / Extensible Markup Language | Two structured text formats. |
| **UUID / SHA-256** | Universally Unique Identifier / Secure Hash Algorithm 256-bit | Decentralized identifiers / cryptographic artifact digests. |
| **FAIL_TO_PASS / PASS_TO_PASS** | failure-to-passing / passing-to-passing tests | Tests fixed by a patch / already-passing tests that detect regressions. |
| **SWE** | Software Engineering | The domain named in benchmarks such as SWE-bench. |

## First-use example

Bad:

> We train with GRPO in a POMDP and use OPD afterward.

Good:

> We train with **Group Relative Policy Optimization (GRPO)**, a critic-free
> method that compares responses sampled for the same prompt. The interaction
> is a **Partially Observable Markov Decision Process (POMDP)** because the
> agent observes tool results rather than the complete environment state. We
> then apply **On-Policy Distillation (OPD)** on states sampled by the student.
