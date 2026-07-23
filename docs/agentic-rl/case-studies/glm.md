# Zhipu AI / GLM: From Blank Infilling to Compaction-Aware Agentic Reinforcement Learning

**Verified through:** 2026-07-23. Sources are official papers, repositories,
configs, model cards, and Zhipu/Z.ai releases. Vendor benchmarks are labeled as
such.

## Reader's terminology key

- **General Language Model (GLM):** the original research name for the
  autoregressive blank-infilling foundation; later it became a product-family
  name rather than one unchanged architecture.
- **Supervised Fine-Tuning (SFT):** imitation learning on curated answers or
  tool trajectories.
- **Reinforcement Learning from Human Feedback (RLHF):** policy optimization
  from human preferences, usually mediated by a learned reward model.
- **Proximal Policy Optimization (PPO):** actor–critic policy optimization with
  bounded/clipped probability-ratio movement.
- **Generalized Advantage Estimation (GAE):** a weighted mixture of
  temporal-difference residuals for lower-variance credit assignment.
- **Group Relative Policy Optimization (GRPO):** critic-free optimization from
  within-prompt response-group comparisons.
- **Single-Rollout Asynchronous Optimization (SAO):** one-rollout-per-prompt
  actor–critic training with direct double-sided importance masking; its authors
  report deployment in GLM-5.2.
- **CompactionRL:** critic-based Proximal Policy Optimization that jointly
  trains ordinary agent actions and model-generated context summaries under the
  final task reward; its authors report deployment in GLM-5.2.
- **Token-In, Token-Out (TITO):** GLM-5's exact-token/log-probability transport
  between rollout and training services.
- **Mixture of Experts (MoE):** sparse activation of a few feed-forward experts
  per token.
- **Multi-head Latent Attention (MLA), Multi-Token Prediction (MTP), and
  DeepSeek Sparse Attention (DSA):** respectively compressed attention state,
  prediction of multiple future tokens, and selected-token sparse attention.
- **On-Policy Distillation (OPD):** teacher supervision evaluated on states
  sampled by the student.
- **Key-Value (KV) cache:** stored attention keys and values reused during
  decoding.

## 1. Three related lineages

Do not collapse every GLM-branded paper into one production chain.

1. **Foundation models:** GLM → GLM-130B → ChatGLM → GLM-4 → GLM-4.5 →
   GLM-5 → GLM-5.2.
2. **Agent-training research:** AgentTuning → AutoWebGLM → WebRL → AutoGLM →
   production recipes disclosed in GLM-4.5/5/5.2.
3. **Perception/action:** CogAgent, GLM-4V/4.5V/4.6V, GUI grounding and
   multimodal tools.

AgentTuning used Llama-2, not ChatGLM. WebRL experimentally used GLM-4-9B, but
its exact algorithm is not proven to be the commercial GLM-4 recipe. The first
unusually detailed production agentic-RL disclosure is GLM-4.5; GLM-5 deepens
it; GLM-5.2 changes the optimizer for irregular long-horizon compacted traces.

```text
instruction SFT + opaque RLHF
  -> native tool protocol/function calling
  -> tool trajectories + executable environments
  -> reasoning/agent/general experts
  -> agentic RL with environment rewards
  -> self/cross-stage on-policy distillation
  -> asynchronous long-horizon rollout
  -> compaction-aware critic PPO + online anti-hacking
```

## 2. GLM: blank-infilling foundation

Primary source: [GLM](https://arxiv.org/abs/2103.10360), Section 2 and Figure 2,
pp. 2–4.

GLM removes continuous spans, replaces each with a mask, and autoregressively
reconstructs spans in random order:

$$
p(B\mid A)=
\prod_{i=1}^{m}\prod_{t=1}^{|B_{\pi_i}|}
p(b_{\pi_i,t}\mid A,B_{\pi_{<i}},b_{\pi_i,<t}).
$$

The attention mask has:

- Part A: corrupted input with bidirectional visibility;
- Part B: autoregressive visibility to A, completed spans, and the current span
  prefix.

Two-dimensional positions encode the removed mask location and within-span
offset. Different masking patterns support understanding, seq2seq generation,
and left-to-right modeling. This is pretraining, not RL or agent interaction.

## 3. GLM-130B: bilingual large-scale pretraining

Sources: [report](https://arxiv.org/abs/2210.02414), Sections 2–3 and Appendix
Table 11; [repository](https://github.com/THUDM/GLM-130B).

### Architecture/data [D]

- 130B dense; 70 layers; hidden 12,288; 96 heads; FFN 32,768; context 2,048.
- FP16, DeepNorm post-LN, RoPE, GeGLU.
- 400B tokens: about 200B English and 200B Chinese.
- Raw pools: ~1.2T Pile English tokens, ~1.0T WuDao Chinese tokens, plus 250GB
  Chinese web text.
- 95% self-supervised blank infilling: 30% short `[MASK]`, 70% prefix `[gMASK]`.
- Short spans use Poisson $\lambda=3$ and mask 15%.
- 5% multitask instruction pretraining from 74 prompted datasets.

### Training operation [D]

- May 6–July 3, 2022.
- 96 DGX-A100 nodes × 8 40GB A100 = 768 GPUs.
- DP 24 × TP 4 × PP 8.
- Global batch warmup 192 → 4,224.
- AdamW .9/.95, decay .1.
- LR $10^{-7}\to8\times10^{-5}$, cosine to $8\times10^{-6}$.
- Dropout .1; gradient clip 1.0.
- Hardware utilization 43.3%; model FLOP utilization 32.5%.
- DeepNorm and 0.1 embedding-gradient shrinkage for stability.

This report is a strong distributed-pretraining reference, not agentic RL.

## 4. ChatGLM generations

### ChatGLM-130B and ChatGLM-6B

Sources: [family report](https://arxiv.org/abs/2406.12793), pp. 2–4;
[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B).

**ChatGLM-130B [D]:** GLM-130B post-trained with manually constructed
prompt-response data, SFT, and RLHF. **[U]** SFT/preference count, label
operation, reward model, PPO, compute, and duration.

**ChatGLM-6B [C/D]:** 6.2B; 28 layers; hidden 4,096; 32 heads; FFN 16,384;
vocabulary 130,528; 2D positions; ~1T bilingual pretraining tokens; ~2K context;
SFT, “feedback bootstrap,” and RLHF at high level. The exact feedback/RL
operation is **[U]**. INT4 deployment at about 6GB drove adoption but is not an
agent-training fact.

### ChatGLM2-6B

[Repository](https://github.com/THUDM/ChatGLM2-6B):

- trained from scratch on 1.4T bilingual tokens;
- 28 layers, hidden 4,096, 32 query heads, 2 MQA KV groups;
- FFN 13,696; vocabulary 65,024; FlashAttention;
- base context 32K, dialogue alignment at 8K;
- human-preference alignment disclosed only at high level.

The transition is MQA and longer context, not yet an agentic-RL report.

### ChatGLM3-6B: native agent protocol

[Repository](https://github.com/THUDM/ChatGLM3): 28 layers, hidden 4,096, 32 Q
heads/2 MQA KV groups, FFN 13,696, standard 8K and separate 32K checkpoint.

This release openly supports native function calls, code interpreter, complex
agent workflows, and structured tool protocol. It demonstrates an agent
interface, not end-to-end environment RL. Data tokens/training steps are **[U]**.

## 5. GLM-4 and All Tools

Sources: [technical report](https://arxiv.org/abs/2406.12793), pp. 4–7 and
Tables 7–9; [repository](https://github.com/zai-org/GLM-4).

### Proprietary GLM-4 [D/U]

- ~10T pretraining tokens.
- Mostly English/Chinese plus smaller 24-language collection.
- Web, Wikipedia, books, code, papers; exact/fuzzy dedup + quality filtering.
- ~150K BBPE vocabulary merging Chinese/multilingual tokens into `cl100k_base`.
- RMSNorm, SwiGLU, 2D RoPE, GQA, QKV-only bias, FFN ~10/3 hidden.
- 128K context.
- SFT, RLHF, safety alignment; preferences cover safety, factuality, relevance,
  helpfulness, and overall human preference.
- Prompts from internal/third-party authentic human sources.

**[U]** parameters/layers/hardware, exact mixture, SFT/preference counts, reward
architecture, PPO settings, annotation cost.

### GLM-4 All Tools [D]

Additional alignment for intent, planning, browser, Python, CogView3, custom
functions/APIs, knowledge retrieval, and recursive tool execution/feedback. The
report describes the interaction loop but not trajectory count or explicit
agent-RL environments. The browser information score 78.08 versus 67.12 for the
compared GPT-4 setting is vendor-run (Table 9).

### GLM-4-9B [C/D]

- 40 layers; hidden 4,096; 32 query heads/2 MQA KV; FFN 13,696;
  vocabulary 151,552.
- 10T multilingual tokens at native 8K.
- Chat checkpoint 128K; experimental checkpoint 1M.
- Report says same post-training pipeline/data as GLM-4-0520.
- Browsing, Python, retrieval, function calling.

## 6. GLM-4-0414 and GLM-Z1

Primary source: [GLM-4 repository](https://github.com/zai-org/GLM-4).

### GLM-4-32B-0414 [C/D]

- dense 32B-class; 61 layers; hidden 6,144; 48 query/2 KV heads; FFN 23,040;
  vocabulary 151,552;
- native 32K, YaRN recommended for 128K;
- 15T high-quality tokens including synthetic reasoning;
- chat post-training uses human preferences, rejection sampling, and RL for
  instruction following, engineering code, and function calls;
- exact RL algorithm/environments **[U]**.

### GLM-Z1-32B-0414 [D]

Derived with cold-start data; extended math/code/logic RL; general-domain RL
with pairwise-ranking feedback. `GLM-Z1-Rumination-32B` adds search during long
reasoning plus rubric/answer grading.

### GLM-Z1-Air

[Official documentation](https://docs.bigmodel.cn/cn/guide/models/text/glm-z1)
states cold start, extended math/code/logic RL, and pairwise-ranking general RL;
128K context and up to 32K output for Air. Parameter count, architecture,
pretraining tokens, rewards, optimizer, steps, and hardware are **[U]**. Do not
repeat unsupported third-party “32B” claims for proprietary Z1-Air.

## 7. GLM-4.5: first detailed production agentic-RL factory

Sources: [technical report](https://arxiv.org/abs/2508.06471),
[repository](https://github.com/zai-org/GLM-4.5),
[flagship config](https://huggingface.co/zai-org/GLM-4.5/blob/main/config.json),
[Air config](https://huggingface.co/zai-org/GLM-4.5-Air/blob/main/config.json).

### 7.1 Architecture [D/C]

| Model | Total / active | Layers | Hidden | Routed / active | Shared | Attention |
|---|---:|---:|---:|---:|---:|---|
| GLM-4.5 | 355B / 32B | 3 dense + 89 MoE + 1 MTP | 5,120 | 160 / 8 | 1 | GQA 96 Q / 8 KV |
| GLM-4.5-Air | 106B / 12B | 1 dense + 45 MoE + 1 MTP | 4,096 | 128 / 8 | 1 | GQA 96 Q / 8 KV |

QK normalization, partial RoPE, sigmoid-gate loss-free balancing. Flagship dense
FFN 12,288/expert 1,536; Air dense 10,944/expert 1,408. Public context 131,072.
The report excludes embeddings/output in parameter accounting; hosting totals
around 358B are not necessarily a contradiction.

### 7.2 Pretraining/data [D]

Headline 23T tokens; Figure 3 components total about 23.1T:

- 15T general;
- 7T code/reasoning;
- 500B repository code;
- 500B synthetic reasoning;
- 100B long-context/agent data.

Operations: web quality buckets with repeated highest-quality data; MinHash and
semantic dedup; FineWeb2-derived multilingual data; language-specific code
classifiers; FIM; code-related web retrieval; LLM scoring for math/science.

Mid-training adds repository files/issues/PRs/commits, teacher-generated
reasoning, context 4K → 32K → 128K, long documents, synthetic agent trajectories,
and best-fit packing.

Optimization:

- Muon for most parameters; special handling for embeddings/bias/RMSNorm;
- 5 Newton–Schulz steps; momentum .95; update RMS .2;
- LR $2.5\times10^{-4}\to2.5\times10^{-5}$;
- token batch 16M → 64M over first 500B;
- weight decay .1; no dropout; RoPE base 10K → 1M;
- MTP loss .3 first 15T then .1.

GPU count, duration, and total FLOPs are **[U]**.

### 7.3 Post-training [D]

Stage 1 trains reasoning, agent, and general experts; Stage 2 unifies them by
self-distillation into a model with direct and thinking modes.

- “Millions” of SFT samples up to 128K.
- Expert-model data across reasoning, general chat, agents, and long context.
- Filters: repetition, truncation, reasoning quality, correctness, reward-model
  quality, tool protocol, terminal state.
- Remove roughly easiest 50% prompts: reported +2–4%.
- Four candidates + filters on hard prompts: another reported +1–2%.

### 7.4 Agent trajectory factory [D]

1. Collect real frameworks, APIs, MCP servers, and tools.
2. Synthesize more tool sets.
3. Generate one/multi-step tasks.
4. Let an LLM generate trajectories.
5. Simulate users as needed.
6. Run multiple judge agents.
7. Retain successful trajectories.
8. Validate XML-like tool protocol, parameters, and terminal state.

XML-like calls reduce escaping failures for embedded code compared with JSON.

### 7.5 Agentic RL [D with equation caveat]

**Web:** multihop questions from knowledge graphs; selective human obfuscation
across pages; final-answer reward.

**Software engineering:** real issues/PRs; executable tests; distributed hardened
sandbox; test reward; invalid format stops with zero reward.

Sample $K$ traces per prompt and center outcome rewards:

$$
A_i=r(x,y_i)-\frac1K\sum_jr(x,y_j).
$$

Only generated tokens are optimized. The operation iterates:

```text
agent RL -> collect successful/high-quality trajectories
         -> self-distill with SFT
         -> construct harder prompts/environments
         -> repeat RL
```

More allowed turns improves reported outcomes, demonstrating inference-time
interaction scaling.

**Critical caveat:** the report's printed objective contains only the sum of
centered rewards, which is identically zero:

$$
\sum_i(r_i-\bar r)=0.
$$

It omits the necessary score-function/log-probability term. A plausible intended
gradient is

$$
\mathbb E\left[\frac1K\sum_iA_i\sum_t
\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})\right],
$$

possibly with clipping/importance correction, but exact implementation is not
recoverable from the printed GLM-4.5 equation. Do not reproduce it uncritically.

### 7.6 Other RL and infrastructure [D]

- ~5,000 holistic preference prompts.
- Capability taxonomy: 7 major, 33 intermediate, 139 fine categories.
- Human preference RM + AI rubrics.
- Instruction taxonomy: 7 major, 151 minor categories.
- Function-call reward requires exact schema/name/arguments/fields.
- End-to-end tools use rule or judge completion rewards.
- MCP/AgentGym environments and simulated users.

[Slime](https://github.com/THUDM/slime) connects Megatron, SGLang engines/router,
data buffer, and custom environments/rewards. Reasoning/general RL commonly use
synchronous colocation; long software rollouts use disaggregated asynchrony.
BF16 training, online blockwise FP8 rollout, dedicated pools, periodic weight
sync, concurrent Docker environments, HTTP interfaces, dynamic sampling. Total
production GPU count is **[U]**.

## 8. GLM-4.6 and GLM-4.7

### GLM-4.6

[Official release](https://z.ai/blog/glm-4.6): same published 355B/32B family,
config retains 92 transformer layers, context about 200K (`202752`). Improved
code/reasoning/tools/search. Vendor CC-Bench reports about 15% fewer tokens than
4.5 and 48.6% win rate versus Claude Sonnet 4 under that setup; trajectories are
released at [CC-Bench](https://huggingface.co/datasets/zai-org/CC-Bench-trajectories).

**[U]** extra pretraining, post-training samples, environments, rewards, steps,
compute. Same architecture does not prove same training.

### GLM-4.7

[Official release](https://z.ai/blog/glm-4.7): same flagship scale/~200K;
interleaved thinking before responses/tools, preserved thinking across turns,
turn-level toggle. Vendor scores include SWE-bench 73.8, multilingual 66.7,
Terminal-Bench 2.0 41.0, HLE 42.8; protocol remains source-specific.

### GLM-4.7-Flash [C]

[Config](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json):
~30B/3B active; 47 layers; hidden 2,048; 64 routed/4 active + 1 shared; expert
FFN 1,536; MLA query rank 768/KV rank 512; 20 heads; context 202,752. Later GLM-5
DSA experiments on this shape are not proof of the original released checkpoint's
training process.

## 9. GLM-5: full agentic engineering pipeline

Sources: [technical report](https://arxiv.org/abs/2602.15763),
[repository](https://github.com/zai-org/GLM-5),
[config](https://huggingface.co/zai-org/GLM-5/blob/main/config.json),
[release](https://z.ai/blog/glm-5).

### 9.1 Architecture/accounting [D/C/U]

Headline 744B total/40B active, 256 experts, ~200K, MLA + DeepSeek Sparse
Attention, shared-parameter MTP.

Public config: hidden 6,144; 3 dense initial layers; 8 routed active + 1 shared;
dense FFN 12,288; expert FFN 2,048; 64 MLA heads; query rank 2,048; KV rank 512;
Q/K head 256 (192 non-RoPE + 64 RoPE); V head 256; DSA index 32×128, top-k
2,048; vocabulary 154,880; context 202,752; one MTP.

Report says 80 layers; config says 78; hosting can show ~754B rather than 744B.
The accounting/convention difference is unresolved **[U]**.

### 9.2 Pretraining [D]

- 27T main pretraining.
- 1T mid-training at 32K.
- 500B at 128K.
- 50B at 200K.
- Sum 28.55T, rounded headline 28.5T.

Data operations: DCLM-style embedding web classifier; world-knowledge classifier;
unique-token code-quality signal; Software Heritage metadata repair;
language-specific lower-resource code filtering; LLM scoring for math/science;
the math/science slice explicitly excludes synthetic/AI/template-generated data;
~10M issue–PR pairs; ~160B unique filtered repository tokens; repositories,
diffs, issues, PRs; natural long documents and synthetic long agent trajectories.

### 9.3 MLA/Muon Split/MTP/DSA [D]

- Muon Split orthogonalizes each head's projections separately.
- Three shared-parameter MTP layers during training.
- Vendor-reported acceptance length 2.76 versus 2.55 for compared V3.2.

DSA conversion after dense mid-training:

1. freeze base; warm indexer 1,000 steps;
2. 14 × 202,752 tokens/step (~2.84M); max indexer LR $5\times10^{-3}$;
3. joint sparse adaptation 20B tokens;
4. top-k 2,048;
5. freeze indexer during RL;
6. use deterministic `torch.topk`; nondeterministic top-k caused rapid RL
   degradation and entropy collapse.

Reported long-sequence attention-compute saving ~1.5–2×.

### 9.4 SFT and loss masks [D]

- General, reasoning, code-agent, long-context SFT to 202,752 tokens.
- Interleaved/preserved/turn-level thinking.
- Hard prompts filtered against GLM-4.7.
- Coding/agent trajectories from execution environments.
- Expert RL + rejection sampling improve trajectory quality.
- Incorrect action segments stay in context but receive zero loss, allowing the
  later correction to be trained without reinforcing the error:

```text
correct policy action       loss mask 1
tool/environment result     loss mask 0
incorrect action segment    loss mask 0
later corrected action      loss mask 1
```

### 9.5 Sequential RL [D]

```text
reasoning RL -> agentic RL -> general RL
             -> on-policy cross-stage distillation
```

Cross-stage distillation mitigates forgetting.

Reasoning RL:

- GRPO + IcePop-like train/inference mismatch suppression;
- no KL regularization;
- mismatch bound $\beta=2$;
- PPO clip $\epsilon_l=.2,\epsilon_h=.28$;
- group 32, batch 32, fully on-policy;
- math, science, code, tool-integrated reasoning;
- difficulty: rarely solved by 4.7 but solvable by strong teachers;
- domain binary judges.

IcePop suppresses token gradients when the rollout/training probability ratio
falls outside $[1/\beta,\beta]$.

### 9.6 Agent environments [D]

**Software engineering:** >10,000 verifiable environments over thousands of
repos in Python, Java, Go, C, C++, JS, TS, PHP, Ruby; RepoLaunch; FAIL_TO_PASS and
PASS_TO_PASS.

**Terminal:** draft task → construction agent creates Harbor/Docker/tests →
refinement agent validates task/rubric → web-derived tasks self-validate. Scale:
thousands of tasks; >90% Docker-build success.

**Search:** discovered URLs form a >2M-page world-knowledge graph; sample
low/medium-frequency entities and multihop neighborhoods; reject if solved
without tools in one of eight attempts or too easy for current agents;
bidirectionally verify unique answer/evidence.

**Context management:** vendor BrowseComp 55.3 → 62.0 by retaining latest five
interactions; hierarchical discard at 32K reaches 75.9.

**Slides/artifacts:** SFT then multilevel-reward RL, rejection and loss-masked
fine-tuning; rewards over static markup, rendered image, and content; explicit
reward-hacking discussion.

### 9.7 Asynchronous optimization [D/U]

- separate inference/training GPU pools;
- central multi-task orchestrator, >1,000 concurrent rollouts;
- token-in/token-out gateway preserves exact IDs/metadata;
- rollout log-probabilities define behavior policy.

$$
r_t(\theta)=\exp[
\log\pi_\theta(a_t\mid s_t)-
\log\pi_{\text{rollout}}(a_t\mid s_t)].
$$

Double-sided rejection:

$$
f(r_t)=
\begin{cases}
r_t,&1-\epsilon_l<r_t<1+\epsilon_h,\\
0,&\text{otherwise}.
\end{cases}
$$

- record rollout weight versions;
- drop trajectories older than an undisclosed lag $\tau$;
- exclude environment crashes;
- after GRPO filtering, if >half a group remains, repeat valid samples to pad;
  otherwise drop group;
- route all turns in one rollout to same DP rank for KV locality;
- reset optimizer state after some rollout-weight refreshes.

Exact sync interval, $\tau$, and hardware are **[U]**.

### 9.8 On-policy cross-stage distillation [D]

Teacher evaluates student-generated tokens; advantage derives from teacher–
student log-probability difference. Group 1, batch 1,024; teacher logits served
through inference infrastructure. Because student produces the state/action
distribution, this is not offline teacher imitation.

## 10. GLM-5.1

Sources: [release](https://z.ai/blog/glm-5.1),
[docs](https://docs.z.ai/guides/llm/glm-5.1),
[config](https://huggingface.co/zai-org/GLM-5.1/blob/main/config.json).

Same 744B/40B, ~200K family. Release emphasizes hundreds of optimization rounds,
thousands of calls, up to eight-hour tasks, and long coding/system case studies.
These are product evaluations, not training disclosures. Added tokens, RL
environments, algorithm changes, reward, and compute are **[U]**.

## 11. GLM-5.2: latest disclosed generation

Sources: [official release](https://z.ai/blog/glm-5.2) (2026-06-16),
[config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json),
[IndexCache paper](https://arxiv.org/abs/2603.12201),
[CompactionRL paper](https://arxiv.org/abs/2607.05378),
[SAO paper](https://arxiv.org/abs/2607.07508),
[GLM-5 repository](https://github.com/zai-org/GLM-5).

No official GLM-5.3 source was found by the cutoff.

### 11.1 Architecture and 1M context [D/C]

- the public checkpoint is counted as 753B by Hugging Face; the SAO paper
  rounds the model to 750B total/40B active (`750B-A40B`); earlier Z.ai
  material describes the family as 744B/40B, so parameter-count convention is
  material and should always be named;
- context 1,048,576; public config still 78 hidden layers;
- RoPE theta 8M; DSA top-k 2,048.

**IndexShare:** one DSA indexer per four-layer block. The first layer computes
top-k indices; next three reuse them. Claimed per-token FLOP reduction is 2.9× at
1M. Introduced during 128K mid-training.

**MTP:** share top-k indices and KV cache across prediction steps; shared step
parameters; add rejection sampling and end-to-end total-variation loss.

| Seven-step ablation | Acceptance length |
|---|---:|
| baseline | 4.56 |
| + IndexShare + KVShare | 5.10 |
| + rejection sampling | 5.29 |
| + end-to-end TV loss | 5.47 |

Vendor-reported +20% over baseline.

### 11.2 Compaction-aware critic PPO [D/U]

Super-long trajectories compact into a variable number of sub-traces with very
different lengths. Group-relative optimization becomes awkward when one rollout
yields two trainable sub-traces and another ten. The official release says
GLM-5.2 moved to:

- critic-based PPO;
- individual-rollout learning;
- token-level advantage estimation;
- all compacted sub-traces retained;
- token-level loss to handle length imbalance.

Li et al.'s 2026
[CompactionRL](https://arxiv.org/abs/2607.05378) states that the method was
deployed in GLM-5.2's reinforcement-learning (RL) pipeline **[D]**. Its
controlled experiments use GLM-4.7-Flash (30B-A3B) and a
Supervised Fine-Tuning (SFT) checkpoint derived from GLM-4.5-Air (106B-A30B),
not GLM-5.2 itself. This distinction is essential: the paper discloses a
mechanism and smaller-scale configuration, not the flagship's complete
resolved run.

#### Trainable compaction [D]

Let the current interaction history contain assistant actions and environment
observations. When the unused context budget drops below
$T_{\mathrm{comp}}$, the policy samples a summary

$$
S_t\sim\pi_\theta\!\left(
\cdot\mid\operatorname{concat}(h_t,q_{\mathrm{sum}})
\right)
$$

and reconstructs context from the system prompt, original user goal, summary,
and the most recent action-observation pairs. The paper keeps the last two
pairs by default. It treats each action plus its observation as atomic so a
tool call is not separated from its result.

The summary is not an external fixed model. Summary tokens and ordinary
execution tokens are sampled by the same trainable policy and receive the same
terminal task reward. The authors do not add a handcrafted summary-quality
reward. Holding the execution model fixed while swapping three summary models
moved reported SWE-bench Verified pass@1 from 49.0% to 55.5%, establishing that
the summary policy alone can materially change downstream success under this
scaffold **[D]**.

#### Token loss and cross-trajectory GAE [D]

One complete rollout becomes a variable sequence

$$
\tau=(\sigma_1,\ldots,\sigma_K)
$$

of execution and summary segments. Treating these segments as independent
group members would overrepresent episodes with more compactions. CompactionRL
therefore uses group-size-one critic PPO and normalizes the policy loss across
all generated tokens rather than averaging segments.

A naive segment-local Generalized Advantage Estimation (GAE) calculation makes
terminal reward appear artificially close to every earlier segment. If
$N_{>s}$ trainable tokens occur after segment $s$, CompactionRL applies

$$
\widehat A_{s,i}
=(\gamma\lambda)^{N_{>s}}A^{\mathrm{loc}}_{s,i},
$$

so credit reflects approximate distance to the end of the concatenated
rollout.

The disclosed experimental configuration is:

| Item | Published value |
|---|---|
| global batch / group | 128 / 1 |
| policy / critic learning rate | $2\times10^{-6}$ / $3\times10^{-6}$ |
| critic initialization | same checkpoint as actor; 50 value-pretraining steps |
| actor/critic updates | one / two per batch |
| peak context | 64K for 30B-A3B; 80K for 106B-A30B |
| response/compaction limit | 10,240 tokens per assistant response; at most three compactions |
| evaluation | temperature 1.0, top-$p=1.0$, at most 250 turns |

On the 106B-A30B experiment, removing token-level loss normalization changes
reported compacted evaluation from 66.8% to 60.0% on SWE-bench Verified and
24.5% to 21.3% on Terminal-Bench 2.0; removing cross-trajectory GAE changes
them to 63.0% and 22.5% **[D]**. These are vendor-author ablations on a sampled
SWE-bench set and a named harness, not independent product measurements.

The paper also reports an important negative boundary: CompactionRL gains do
not consistently transfer to single-window evaluation with compaction disabled,
and its cross-trajectory GAE is still an approximation. **[U]** for GLM-5.2:
critic architecture, resolved $\gamma/\lambda$, clipping, value/entropy
coefficients, batch, optimizer, task mixture, and how CompactionRL interacts
with SAO. “Uses PPO” and “deployed in the pipeline” do not reconstruct the
flagship update.

### 11.3 SAO: the missing algorithmic link [D/U]

The user-visible GLM-5.2 release predates a more precise primary source. Hou,
Li, Tang, and Dong's 2026 paper
[Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)
introduces **Single-Rollout Asynchronous Optimization (SAO)** and states in the
abstract that SAO was deployed in the agentic-RL pipeline used to train the
open GLM-5.2 model. Two authors performed the work while interning at Z.AI.
This is direct author disclosure **[D]**, not proof that every GLM-5.2 RL stage
used SAO.

#### Why group optimization becomes a systems problem

**Group Relative Policy Optimization (GRPO)** samples $G$ answers for one
prompt and needs the group before it can compute relative advantages. If
rollout durations are $T_1,\ldots,T_G$, a synchronous group becomes ready at

$$
T_{\text{ready}}=\max_i T_i,
$$

so finished workers wait for the longest member. Long coding episodes make the
tail large. Meanwhile the learner may update, so a late group can contain
tokens produced by stale rollout-policy versions. SAO sets $G=1$: each
trajectory enters training as soon as it finishes. That removes the
within-prompt group barrier but also removes GRPO's group-mean baseline.

#### Direct Double-Sided Importance Sampling

SAO uses **Direct Double-Sided Importance Sampling (DIS)**. For an action token
recorded with rollout-engine log-probability
$\log\pi_{\text{rollout}}(a_t\mid s_t)$, it recomputes

$$
r_t(\theta)=\exp\!\left[
\log\pi_\theta(a_t\mid s_t)-
\log\pi_{\text{rollout}}(a_t\mid s_t)
\right].
$$

It drops a separate $\pi_{\text{old}}$ and gates the token with

$$
f(r;\epsilon_l,\epsilon_h)=
\begin{cases}
r,&1-\epsilon_l<r<1+\epsilon_h,\\
0,&\text{otherwise}.
\end{cases}
$$

The paper prints

$$
L(\theta)=\widehat{\mathbb E}_t\!\left[
f(r_t;\epsilon_l,\epsilon_h)\widehat A_t
\log\pi_\theta(a_t\mid s_t)
\right].
$$

This is strict **two-sided masking**, not ordinary Proximal Policy Optimization
(PPO) clipping. PPO saturates only the direction in which the objective would
improve too far; DIS gives *zero* gradient contribution outside the interval
regardless of advantage sign. The paper does not mark whether $f(r_t)$ is
stop-gradient. A literal autodifferentiation through both $r_t$ and
$\log\pi_\theta$ adds an extra derivative term, so a source-level
reproduction must resolve this implementation detail rather than silently
assuming one interpretation **[U]**.

#### Restoring a critic at group size one

With no same-prompt comparison group, SAO learns a value model
$V_\phi(s_t)$:

1. run $K=2$ critic updates for every actor update in the reported
   experiments;
2. freeze the critic's full-attention parameters and optimize its
   Mixture-of-Experts (MoE) projections, because full attention updates had
   much larger gradient norms;
3. scale value-model pretraining to reduce cold-start error, although the paper
   does not disclose the corpus size; and
4. estimate token advantages with a length-adaptive form of Generalized
   Advantage Estimation (GAE).

For an agent trace $[a_0,o_0,a_1,o_1,\ldots]$, where $a_i$ is a model action
and $o_i$ is an environment observation, skip-observation GAE bridges from
the last token $a_{i,N}$ directly to the first token of the next model action:

$$
\begin{aligned}
\delta_i
  &=r_i+\gamma V_\phi(a_{i+1,0})-V_\phi(a_{i,N}),\\
\widehat A(a_{i,N})
  &=\delta_i+\gamma\lambda\widehat A(a_{i+1,0}).
\end{aligned}
$$

Observation tokens are not sampled by the policy and therefore receive no
policy loss or artificial token-to-token value transition. The *next action's
state still includes the observation in its context*; “skip observation” must
not be misread as hiding tool feedback from the model.

#### What the experiments establish—and what they do not

The controlled experiments use **Qwen3-30B-A3B**, not GLM-5.2:

| Setting | Published experimental value |
|---|---|
| math/tool initialization | three-epoch SFT of Qwen3-30B-A3B-Thinking-2507 on GPT-OSS-120B-generated tool traces |
| asynchronous batch/group | 128 trajectories / one rollout per prompt |
| maximum length | 128K tokens |
| actor learning rate | $10^{-6}$ |
| reasoning DIS interval | $(1-0.3,1+5.0)=(0.7,6.0)$ |
| coding DIS interval | $(1-0.8,1+3.0)=(0.2,4.0)$ |
| critic | learning rate $5\times10^{-6}$, 10-step warmup, two updates per actor update |
| coding scaffold | OpenHands, at most 300 turns, 128K context |

Vendor-author-reported outcomes are 97.3 versus 84.2 on AIME 2025, 74.8
versus 54.8 on BeyondAIME, 88.3 versus 76.0 on HMMT November 2025, and 74.0
versus 55.8 on IMOAnswerBench for SAO versus the reported GRPO baseline.
SWE-bench Verified is 29.8 for SAO, 27.0 for GRPO+DIS, and 23.0 for the base
checkpoint. The paper reports vanilla GRPO collapse around 160 steps, while
SAO remains stable for roughly 1,000; these are one paper's controlled results,
not independent evidence of GLM-5.2's gain.

The correct production statement is narrow:

- **[D]** SAO was deployed somewhere in GLM-5.2's agentic-RL pipeline.
- **[I]** Its single-rollout/critic design naturally explains the official
  release's move to individual-rollout learning for irregular compacted traces.
- **[U]** which GLM-5.2 stages, domains, rewards, task counts, rollout counts,
  coefficients, batch sizes, compute, and ablation deltas used SAO.
- **[U]** whether SAO was the only actor optimizer or the final consolidation
  optimizer; the release separately discloses parallel OPD for expert merging.

Transferring the Qwen experiment's $K=2$, learning rates, or very wide DIS
intervals to a 753B checkpoint would turn disclosed evidence into fiction.

### 11.4 Parallel on-policy expert distillation [D]

- Slime supports white-box/black-box rollout, compact traces, and sub-agent
  workflows.
- More than ten expert models merged into the final checkpoint by parallel OPD.
- Complete OPD reportedly takes about two days.
- KV-cache FP8 and flexible rollout/training resource organization.
- Hardware count/type **[U]**.

### 11.5 Online anti-hacking [D]

Coding rewards can be exploited by reading evaluation artifacts, copying
answers, recovering upstream commits, downloading target source, or chained tool
leakage. GLM-5.2 uses:

1. high-recall rule filter for suspicious actions;
2. LLM intent judge for precision.

During RL and evaluation, calls are monitored; detected hacks are blocked;
dummy information is returned; the rollout continues. Continuing avoids turning
every suspicion into terminal zero, preserves later correction data, and reduces
collapse risk. The hard external block—not a negative reward alone—prevents the
prohibited action.

### 11.6 Vendor-reported results

Terminal-Bench 2.1 81.0, SWE-bench Pro 62.1, NL2Repo 48.9, HLE 40.5 / 54.7 with
tools, GPQA-Diamond 91.2, MCP-Atlas public 76.8, Tool-Decathlon 48.2. The release
has harness/token/sandbox/judge footnotes; retain them in any score comparison.

## 12. Adjacent agent-training research

### AgentTuning

[Paper](https://arxiv.org/abs/2310.12823): 35,341 attempted trajectories; 1,866
successful across six tasks; GPT-4-0613 generation; reward-one filtering; mix
AgentInstruct with ShareGPT-like data at ratio .2. Experiments use Llama-2, so
this is a general research result, not ChatGLM production evidence.

### AutoWebGLM

[Paper](https://arxiv.org/abs/2404.03648), Section 4/Appendix A; based on
ChatGLM3-6B. Humans select tasks and record browser operations; GPT-4 fills
step intentions; curriculum SFT goes atomic → complex. DPO samples 20 responses
per prompt and keeps mixed success/failure prompts; ~13K preferences; LR
$10^{-6}$, batch 64, $\beta=.15$, auxiliary SFT weight .8. Rejection
fine-tuning: ~15K MiniWoB++ trajectories/66K steps and 240 WebArena/2K; LR
$10^{-5}$, batch 32.

### WebRL

[Paper](https://arxiv.org/abs/2411.02337); GLM-4-9B experiment. Binary environment
reward; outcome RM emits YES/NO from instruction/action history/final HTML;
12,200 RM examples from 1,186 source tasks + rollouts; LR $5\times10^{-6}$,
batch 128, four epochs. Eight curriculum phases; each creates 500 GPT-4o-derived
tasks from current failures. Critic/feasibility filter; KL-constrained
actor–critic with GAE $\lambda=.5,\gamma=.9$; replay with tightening
perplexity; actor/critic LR $10^{-6}$, batch 128. Reported success 6.1 → 43.0.

### AutoGLM

[Paper](https://arxiv.org/abs/2411.00820): separates planning and grounding,
browser/Android environments, progressive weak-to-strong self-evolving online
curriculum RL with GLM-4-9B planners. Strong research evidence; not proof of an
identical flagship recipe.

## 13. Multimodal/GUI branch

### CogAgent

[Paper](https://arxiv.org/abs/2312.08914),
[repository](https://github.com/zai-org/CogAgent): 18B dual-resolution visual
architecture up to 1,120×1,120, GUI grounding/navigation; later 9B checkpoint
based on GLM-4V-9B.

### GLM-4.5V / RLCS

[Report](https://arxiv.org/abs/2507.01006): 220M OCR images, 40M natural grounding
boxes, GUI data; pretraining seq 8,192, batch 1,536, 120K steps; SFT seq 32,768,
batch 32; RLVR + RLHF; GUI reward combines action correctness and IoU; grounding
IoU; QA exact/semantic; GRPO + RLCS difficulty-aware resampling. Report warns a
weak verifier in one domain can collapse other domains; selected setup uses
large batches, no KL/entropy loss, top-p 1.

### GLM-4.6V

[Release](https://z.ai/blog/glm-4.6v): synthetic agent data, URL multimodal values
inside extended MCP, draft → image selection → polish, visual feedback for
correction. CogView3 is an external tool available to GLM-4 All Tools, not proof
that its generator training is part of the language model.

## 14. Reconstructed disclosed production workflow

```text
web/code/science/books/repos/issues/PRs
  -> exact/fuzzy/semantic dedup
  -> domain/language quality classifiers
  -> staged base pretraining
  -> repository/reasoning/long-context mid-training
  -> cold-start SFT for reasoning and agent protocols
  -> executable environments + synthesized tools/tasks
  -> multiple specialist trajectories
  -> verify tests/answers/formats/terminal state
  -> reasoning RL
  -> web/SWE/terminal/tool agentic RL
  -> general preference/instruction RL
  -> self/cross-stage/parallel on-policy distillation
  -> harder environments and new rollouts
  -> anti-hack/safety/regression evaluation
  -> serving-compatible deployment
```

Complete raw sources, private/copyrighted proportions, annotation staffing/cost,
most SFT/preference counts, exact reward-model sizes, safety taxonomy, most RL
steps, post-130B GPU totals, base-training duration/FLOPs, and production API
routing remain **[U]**.

## 15. Source-level map

### Model implementations

- [Transformers GLM-4 MoE](https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm4_moe)
- [Transformers GLM MoE DSA](https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm_moe_dsa)
- [vLLM GLM-4.5 MTP](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/glm4_moe_mtp.py)
- [SGLang GLM-4 MoE](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/glm4_moe.py)

### Slime

Repository: [THUDM/slime](https://github.com/THUDM/slime). The paths below were
inspected at commit `fb42ae456fac8166afb604f13b30d22bb3c75053`:

- [`slime/ray/rollout.py`](https://github.com/THUDM/slime/blob/fb42ae456fac8166afb604f13b30d22bb3c75053/slime/ray/rollout.py):
  SGLang engines/router, generation, buffer, metrics.
- [`slime/backends/megatron_utils/loss.py`](https://github.com/THUDM/slime/blob/fb42ae456fac8166afb604f13b30d22bb3c75053/slime/backends/megatron_utils/loss.py):
  advantages, PPO loss, rollout/train mismatch, token importance masks.
- [`slime/utils/ppo_utils.py`](https://github.com/THUDM/slime/blob/fb42ae456fac8166afb604f13b30d22bb3c75053/slime/utils/ppo_utils.py):
  approximate KL, clipped PPO, CISPO, sequence masking, GRPO utilities.
- [`slime/agent/trajectory.py`](https://github.com/THUDM/slime/blob/fb42ae456fac8166afb604f13b30d22bb3c75053/slime/agent/trajectory.py):
  token-in/token-out traces, branching, drift realignment, exact masks.
- [`slime/rollout/on_policy_distillation.py`](https://github.com/THUDM/slime/blob/fb42ae456fac8166afb604f13b30d22bb3c75053/slime/rollout/on_policy_distillation.py):
  teacher log-probs and OPD samples.
- [`examples/coding_agent_rl/generate.py`](https://github.com/THUDM/slime/blob/fb42ae456fac8166afb604f13b30d22bb3c75053/examples/coding_agent_rl/generate.py):
  sandbox boot, coding agent, patch extraction/evaluation.
- [`examples/coding_agent_rl/swe.py`](https://github.com/THUDM/slime/blob/fb42ae456fac8166afb604f13b30d22bb3c75053/examples/coding_agent_rl/swe.py):
  clean SWE rewards and failing/passing tests.
- [`examples/search-r1/generate_with_search.py`](https://github.com/THUDM/slime/blob/fb42ae456fac8166afb604f13b30d22bb3c75053/examples/search-r1/generate_with_search.py):
  multi-turn search and exact token/log-prob tracking.
- [`slime/rollout/fully_async_rollout.py`](https://github.com/THUDM/slime/blob/fb42ae456fac8166afb604f13b30d22bb3c75053/slime/rollout/fully_async_rollout.py):
  asynchronous in-flight queue.

Current Slime evolves beyond each model release. It is an executable reference,
not automatically a frozen copy of GLM-4.5/5/5.2 internal training.
