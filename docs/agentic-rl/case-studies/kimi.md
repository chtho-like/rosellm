# Moonshot AI / Kimi: Long Context, Muon, Tool Reinforcement Learning, and Agent Swarms

**Verified through:** 2026-07-22. Sources are Moonshot papers, repositories,
model cards, and official product/research posts. Vendor benchmark claims are
not independent reproductions.

For a component-level explanation of MoonViT-3D, visual token projection, K2
suffixes, K3's “native vision” boundary, and media generation, see the
[Kimi multimodal architecture chapter](../../multimodal/kimi.md).

## Reader's terminology key

- **Chain of Thought (CoT):** intermediate reasoning before an answer;
  long-CoT deliberately allows extended traces.
- **Mixture of Experts (MoE):** sparse routing that activates only a subset of
  feed-forward experts for each token.
- **Key-Value (KV) cache:** stored attention keys/values reused during decoding;
  Mooncake manages this cache across memory tiers and machines.
- **Remote Direct Memory Access (RDMA):** network transfer directly between
  registered memory regions with little central-processing-unit involvement.
- **Mixture of Block Attention (MoBA):** Moonshot's sparse routing of a query to
  selected context blocks.
- **Kimi Delta Attention (KDA):** Moonshot's gated delta-rule linear attention.
- **Multi-head Latent Attention (MLA):** a compressed latent key/value attention
  representation used in Kimi's hybrid architectures.
- **Attention Residuals (AttnRes):** learned attention over earlier residual
  streams instead of an unweighted cumulative residual sum.
- **Parallel-Agent Reinforcement Learning (PARL):** K2.5 training for a learned
  orchestrator that launches and coordinates parallel sub-agents.
- **On-Policy Distillation (OPD):** teacher supervision on the student's own
  sampled state distribution.
- **Muon:** an optimizer name, not an acronym; it orthogonalizes matrix-shaped
  updates, and MuonClip adds Moonshot's stability control for K2.

## 1. Executive synthesis

Moonshot's public development has five connected strands:

1. **Long-context product/serving:** early 128K APIs, Mooncake's disaggregated
   KV-cache serving, and MoBA sparse long-context research.
2. **Long-CoT RL:** Kimi k1.5 documents verifiable long-context RL, anti-hacking
   filters, curricula, partial rollouts, and a value-model-free objective.
3. **Trillion-parameter MoE:** Moonlight scales Muon; K2 introduces MuonClip and
   reports stable 1.04T-MoE training over 15.5T tokens.
4. **Trainable agents:** K2 synthesizes tools/environments/trajectories; K2
   Thinking interleaves reasoning/tools; K2.5 adds multimodal agent RL and PARL
   for a learned multi-agent orchestrator.
5. **Architecture beyond conventional attention:** Kimi Linear introduces KDA,
   Attention Residuals changes depth routing, and K3 announces KDA + AttnRes +
   LatentMoE at 2.8T and 1M context.

Transparency boundary:

- k1.5, Moonlight, K2, Kimi Linear, and K2.5 have substantial reports.
- K2-0905, K2 Thinking, K2.6, and K2.7 Code have model cards/posts without a
  comparably complete training report.
- K3 was announced 2026-07-17; full weights were promised for 2026-07-27 and
  its complete training report was pending at the cutoff.

## 2. Chronology

| Date | Generation/line | Architecture/context | Disclosure |
|---|---|---|---|
| 2024-01-31 | `moonshot-v1` API | unknown params; up to 128K | very limited |
| 2024-06-26 | Mooncake | serving system | detailed systems paper |
| 2025-01-20 | Kimi k1.5 | proprietary multimodal Transformer; 128K; params unknown | detailed RL/data/system report |
| 2025-02-18 | MoBA | parameter-free sparse attention; research to 1M | detailed paper/code |
| 2025-02-23 | Moonlight | 15.29B/2.24B active excluding embeddings (~16B/3B including) | detailed optimizer/pretraining |
| 2025-04-10 | Kimi-VL | 16B/2.8B active multimodal MoE; 128K | detailed report |
| 2025-07-11 | Kimi K2 | 1.04T/32.6B active; 61 layers; 384 experts; 128K | detailed report |
| 2025-09-05 | K2-Instruct-0905 | K2 family; 256K | model card |
| 2025-10 | Kimi Linear | 48B/3B research MoE; hybrid KDA/MLA; 1M checkpoint | detailed report |
| 2025-11-06 | K2 Thinking | 1T/32B; 256K; interleaved reasoning/tools; INT4 | post/model card |
| 2026-01-27 | K2.5 | 1T/32B + ~400M MoonViT; 256K | detailed report |
| 2026-02-09 | Agent Swarm | K2.5 PARL orchestration | report + post |
| 2026-03 | Attention Residuals | attention over residual streams | paper/code |
| 2026-04-20 | K2.6 | K2.5-family architecture; 256K | post/model card |
| 2026-06-12 | K2.7 Code | built on K2.6; coding focus | model card/release |
| 2026-07-17 | K3 | 2.8T; 896 experts/16 selected; 1M; KDA/AttnRes/LatentMoE | announcement; report pending |

## 3. Early Kimi and Mooncake

### Early model [D/U]

[Official API announcement](https://platform.kimi.com/blog/posts/kimi-latest) and
[product page](https://kimi.moonshot.cn/download/app) offered `moonshot-v1` up
to 128K (consumer description: about 200K Chinese characters). Architecture,
parameters, pretraining corpus, post-training, and RL are **[U]**. Do not retrofit
later K2 architecture or k1.5 recipe onto this model.

### Mooncake [D]

Sources: [paper](https://arxiv.org/abs/2407.00079),
[repository](https://github.com/kvcache-ai/Mooncake).

Mooncake is serving infrastructure, not model training:

- disaggregated prefill/decode pools;
- distributed KV cache across GPU, CPU DRAM, and SSD;
- RDMA KV-chunk movement;
- prefix-locality-aware scheduling;
- layerwise asynchronous KV transfer;
- chunked prefill;
- hot replication, cold swapping, overload rejection;
- global Conductor optimizing locality, TTFT, and inter-token latency.

The paper reports up to 525% simulated throughput gain and 75% more requests in
one real workload. Results are topology/workload-specific; public hardware tests
use dummy Llama-2-70B-like workloads/A800 nodes and do not reveal production
Kimi architecture.

## 4. Kimi k1.5: long-CoT RL foundation

Sources: [paper](https://arxiv.org/abs/2501.12599),
[PDF](https://arxiv.org/pdf/2501.12599),
[repository](https://github.com/MoonshotAI/Kimi-k1.5).

### 4.1 Architecture/pretraining [D/U]

- Multimodal autoregressive Transformer with vision encoder and interleaved
  text/image input.
- Context 131,072.
- **[U]** parameters, dimensions/experts, pretraining-token total, cluster,
  duration, and cost.

Appendix B, pp. 21–24:

- English, Chinese, code, math/reasoning, general knowledge.
- Rule cleaning → fastText classification → embedding near-dedup → LLM quality.
- Code cleaned similarly to BigCode, markup downsampled, 32 major languages
  deliberately upsampled.
- Math from web/PDF with specialized OCR and learned filtering.
- Knowledge from exercises, textbooks, papers with educational/document labels.
- Multimodal captions, interleaved docs, OCR, knowledge, general VQA.
- Synthetic captions capped rather than allowed to dominate.

Stages:

1. language-first;
2. frozen language model while training vision tower separately;
3. unfreeze/joint train, vision-text rises to 30% mixture;
4. cooldown with high-quality and rejection-sampled synthetic QA;
5. context 4K → 32K → 131,072.

At final long-context stage, 40% uses full attention with natural long docs plus
synthetic long QA/summarization; 60% uses partial attention with uniform cooldown
data. RoPE base 1,000,000.

### 4.2 SFT [D]

PDF p. 8:

- ~1M text and ~1M text-vision examples.
- Listed text: 500K general QA, 200K code, 200K math/science, 5K creative,
  20K long context = 925K; remaining ~75K uncategorized.
- Epoch 1: 32K, LR $2\times10^{-5}\to2\times10^{-6}$.
- Epoch 2: 128K, rewarm $10^{-5}$, decay $10^{-6}$.
- Packed examples.

### 4.3 RL data and anti-hacking [D]

Pipeline:

$$
\text{pretraining}\to\text{vanilla SFT}\to
\text{long-CoT SFT warmup}\to\text{RL}.
$$

Prompts cover STEM, code, and general domains; balance difficulty; require
automatic evaluation. Estimate difficulty by sampling ten high-temperature SFT
answers and measuring pass rate. Exclude multiple-choice, true/false, and proof
formats that are easily hacked.

The **no-CoT guessing test** asks the model to answer without reasoning; if it
guesses correctly within eight attempts, remove the problem. This filters
memorization, leakage, and shallow reward paths.

The small long-CoT warmup set contains verified planning, evaluation,
reflection, and exploration traces.

### 4.4 Rewards [D]

Terminal binary outcome:

$$
r(x,y,y^*)\in\{0,1\}.
$$

- code: executable tests;
- structured math: exact/rule verification;
- free form: learned answer matcher.

No disclosed ground-truth intermediate reasoning reward. Search traces are
flattened into autoregressive context, so next-token learning and outcome RL
teach an implicit search pattern.

### 4.5 Value-model-free policy optimization [D]

PDF pp. 4–6 defines an online mirror-descent-like objective with relative
entropy. From $K$ old-policy responses:

$$
A_i=r_i-\frac1K\sum_jr_j.
$$

The policy minimizes a squared difference between new-policy sequence
log-ratio and an analytically reward-tilted target.

- no value network;
- no MCTS;
- reset optimizer between RL iterations;
- older-policy responses can be used through log-ratio objective;
- gradual length reward favors shorter correct and penalizes longer incorrect;
- easy-to-hard curriculum;
- task priority roughly $1-\text{success rate}$.

The rationale for avoiding token/state values is important: a locally failing
reasoning branch can be useful exploration if the model later detects/repairs
it. Premature negative token credit can suppress trial-and-error.

### 4.6 Concrete code-data operation [D]

For code problems:

1. generate 50 tests with CYaRon;
2. sample ten candidate ground-truth submissions;
3. retain a test when at least 7/10 agree;
4. admit a problem when at least 9/10 submissions pass retained tests.

From 1,000 contest problems: 614 lack special judges; 463 generate at least 40
valid cases; 323 survive the entire pipeline.

Math reward models: ~800K conventional scalar examples and ~800K CoT-labeled
examples. Manual spot-check reports 84.4% scalar versus 98.5% CoT model accuracy.

Vision RL covers real science, geolocation, charts, procedural spatial/geometry/
object tasks, and rendered text/code/structured data.

### 4.7 Long-to-short [D]

- weight averaging;
- shortest-correct rejection sampling over eight candidates;
- DPO preferring shortest correct over longer/incorrect (correct negative can be
  selected if ≥1.5× longer);
- second RL with stronger length penalty and lower maximum rollout.

Vendor result: short-CoT AIME 60.8 at 3,272 average tokens versus long-CoT 77.5.

### 4.8 RL infrastructure [D]

PDF pp. 8–11: synchronous iterative rollout/controller/replay/trainer/reward/
code-execution system.

**Partial rollout continuation:** fixed per-iteration output-token budget;
unfinished state saved/resumed next iteration; only newly generated segment is
on-policy/trainable; earlier tokens remain context; repetition detector prevents
infinite loops.

Megatron training + vLLM inference colocated in Kubernetes pod sharing GPUs:
training → inference under one minute; inference → training about ten seconds;
checkpoint conversion/transfer through shared memory and Mooncake/RDMA.

Sandbox: Kubernetes, `crun`, cgroup reuse, `tmpfs` overlay; reported startup
0.04s versus Docker 0.12s; throughput 120 versus 27 containers/s on 16 cores.

## 5. Moonlight: scaling Muon

Sources: [paper](https://arxiv.org/abs/2502.16982),
[repository](https://github.com/MoonshotAI/Moonlight).

### Muon mechanics [D]

For matrices, compute momentum then approximate orthogonalization with five
Newton–Schulz iterations. Moonlight adds:

- weight decay for weight/output RMS growth;
- shape multiplier approximately $0.2\sqrt{\max(A,B)}$ for an $A\times B$
  matrix to match AdamW-like update RMS;
- AdamW retained for embeddings, norms, output head.

Distributed Muon:

1. ZeRO-1 shards optimizer state.
2. Reduce-scatter gradients.
3. Update local momentum.
4. Gather full matrices for orthogonalization.
5. Discard nonlocal update pieces.
6. All-gather updated parameters.

Muon uses one momentum buffer versus AdamW's two moments. Report estimates
1–1.25× AdamW optimizer communication and 1–3% forward/backward latency.

### Moonlight model/training [D]

- 15.29B total/2.24B active excluding embeddings; ~16B/3B including.
- 8K; DeepSeek-V3-small-like MoE; no MTP.
- stable auxiliary-loss-free routing-bias update; router gate scale 2.446.
- 5.7T tokens.

Schedule:

1. 0–33B: warm to $4.2\times10^{-4}$, 2,000 steps, batch 2,048 sequences.
2. 33B–5.2T: cosine to $4.2\times10^{-5}$; batch 2,048 until 200B then 4,096.
3. 5.2–5.7T: math/code/reasoning cooldown; rewarm $10^{-4}$ over 100 steps,
   linearly decay to zero.

Weight decay .1; expert-bias rate $10^{-3}$ first two stages then zero.
Scaling fits across 399M–1.5B dense models estimate Muon needs ~52% AdamW FLOPs
for equal loss. This is a fit, not universal 2× wall-clock proof.

**[U]** exact corpus, cluster, duration, cost, full SFT mixture.

## 6. MoBA: sparse attention research

Sources: [paper](https://arxiv.org/abs/2502.13189),
[repository](https://github.com/MoonshotAI/MoBA).

MoBA blocks keys/values, represents a block by mean key, scores all block means
per query, selects top-k plus the current causal block, and runs ordinary
attention only inside selected blocks. Implementation sorts queries by selected
blocks, applies variable-length FlashAttention, and combines with online softmax.

Qualifications:

- no learned gate, but continued training is required;
- ~90% MoBA tokens then 10% full-attention tokens;
- final few full-attention layers improve SFT;
- 1M experiment extends Llama-family 8B through 128K/256K/512K/1M with ~100B
  activation tokens;
- block 4,096, top-12, MoBA in 29 layers/full attention last 3;
- MoBA prefill can pair with full-attention generation.

Vendor experiments report up to 6.5× 1M prefill. K2/K2.5 disclose MLA, so MoBA
research is not proof those products used MoBA.

## 7. Kimi K2: trillion-parameter MoE and agent RL

Sources: [technical report](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf),
[repository](https://github.com/MoonshotAI/Kimi-K2),
[launch](https://www.kimi.com/blog/kimi-k2).

### 7.1 Architecture [D]

- 1.04T total; 32.6B active.
- 61 layers; first dense.
- hidden 7,168; 384 routed experts; 8 routed + 1 shared active; expert 2,048.
- 64 attention heads; MLA; SwiGLU; 160K vocabulary; 128K context.
- effective expert sparsity 48.

Compared with DeepSeek-V3: more experts, fewer active routed experts/fewer dense
layers, 64 vs 128 heads, no expert grouping. K2 ablation says 128 heads cost 83%
more inference FLOPs at 128K for 0.5–1.2% validation-loss gain.

### 7.2 MuonClip [D]

QK-Clip measures maximum pre-softmax logit per head. If above $\tau=100$,
rescale selected query/key components, treating rotary/non-rotary keys
differently. A 53B/9B active precursor saw unbounded logits >1,000; K2 reports
15.5T training without a loss spike. This is vendor evidence, not an independent
universal stability guarantee.

### 7.3 Data and semantic rephrasing [D/U]

- 15.5T processed tokens across web/code/math/knowledge.
- Rewrite prompts vary style, perspective, exposition.
- Long documents rewritten chunk-by-chunk autoregressively.
- Semantic fidelity filters.
- Math rewritten as learning notes; selected material translated to English.
- Large corpora rewritten at most twice.

SimpleQA ablation: raw repeated 10 epochs 23.76; one rewrite repeated 27.39; ten
independent rewrites one epoch 28.94. Exact proportions, licenses, original/
synthetic ratio, dedup thresholds, unique tokens are **[U]**.

### 7.4 Schedule/system [D/U]

- main sequence 4,096; global batch ~67M tokens; decay .1; 500 warmup;
- LR $2\times10^{-4}$ through 10T, cosine to $2\times10^{-5}$ over 5.5T;
- late annealing/long context: ~400B at 4K + 60B at 32K, LR
  $2\times10^{-5}\to7\times10^{-6}$; wording does not prove these add to
  15.5T;
- YaRN to 128K.

H800 nodes: eight NVLink GPUs, 2TB CPU, eight 400Gbps RoCE NICs; PP16, virtual
stages, EP16, ZeRO-1. A 256-GPU model-parallel group holds ~6TB BF16 params +
FP32 gradients. Optimizer ~30GB/GPU; CPU offload across 32-node group.

Expert all-to-all overlaps 1F1B; weight-gradient computation decoupled from
backward. No DualPipe due duplicated storage. Activation recompute + selected
FP8 E4M3 activation storage (not claimed FP8 matrix compute) + CPU activation
offload.

Total GPUs, duration, energy, failures, and cost are **[U]**.

### 7.5 Agentic SFT factory [D]

1. Build tool repository.
2. Generate agent identities and rubric-scored tasks.
3. Generate tool-use trajectories.

- >3,000 real MCP tools;
- >20,000 synthetic tools;
- hierarchical domain evolution;
- thousands of agents;
- tens of thousands of trajectories;
- simulated user persona;
- stateful tool simulator with stochastic failure;
- LLM rubric judge;
- real sandboxes/tests for coding;
- K1.5/in-house experts generate candidates, filtered by LLM + humans.

### 7.6 RL Gym/rewards [D]

- math/STEM/logic: rule or expert verification;
- code: open/synthetic/pretraining-derived human tests and GitHub issues/PRs;
- instruction following: deterministic verifier + LLM judge + hack detector;
- grounded factuality: sentence-level judge;
- safety: evolved attacks, targets, rubrics;
- subjective tasks: self-critique reward model.

Self-critique: K2 samples candidates; K2 critic ranks with core, prescriptive,
human rubrics; bootstrap critic with preference SFT; continue grounding via
on-policy rollouts from verifiable domains.

Policy optimization mostly retains k1.5's squared log-ratio objective, adding
per-task output budgets/truncation penalties, high-quality pretraining loss to
reduce drift, and temperature decay. More than 10,000 concurrent Kubernetes
code sandboxes are reported.

### 7.7 Weight transition/evaluation [D]

Pipeline-wise parameter streaming updates the 1T model into inference in under
30 seconds; partial rollouts and unified Gym remain.

Vendor examples: SWE-bench Verified 65.8 single attempt, multilingual 47.3,
LiveCodeBench 53.7, AIME 49.5, GPQA 75.1, Tau2 66.1, ACEBench 76.5. The report
uses non-thinking model, normally 128K/8K output (SWE 16K); protocol is part of
the result.

## 8. K2-Instruct-0905

[Official model card](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905):
same 1T/32B family; context 256K; agent coding/frontend emphasis. Five-run
vendor means include SWE Verified $69.2\pm.63$, multilingual $55.9\pm.72$,
Multi-SWE 33.5, TerminalBench 44.5, SWE-Dev 66.6. Harness derives from SWE-agent;
unreachable Git objects and SWE-Dev tests were removed.

**[U]** new pretraining, post-training data, RL, optimizer, compute. Same
architecture does not imply a simple SFT-only update.

## 9. K2 Thinking: reasoning interleaved with tools

Sources: [launch](https://www.kimi.com/blog/kimi-k2-thinking),
[model card](https://huggingface.co/moonshotai/Kimi-K2-Thinking).

**[D]** 1T/32B; 256K; reasoning interleaved with function calls; ~200–300
sequential calls in search tasks; test-time scaling over thinking and tools;
native INT4 quantization-aware post-training; weight-only INT4 for MoE; all
published benchmarks use INT4.

Protocol examples:

- temperature 1, 256K;
- 96K reasoning cap on HLE/AIME/HMMT/GPQA;
- up to 300 search steps, 24K reasoning each;
- HLE up to 120 steps, 48K each;
- earlier tool outputs hidden on overflow;
- heavy mode: eight rollouts + reflective aggregation.

Vendor scores include HLE-with-tools 44.9, BrowseComp 60.2, SWE Verified 71.3.
K2.5 later says Toggle RL reduced K2 Thinking tokens 25–30% with negligible
loss. **[U]** training counts, data, reward mix, rollout K, hyperparameters,
compute, exact initialization.

## 10. Kimi Linear and KDA

Sources: [paper](https://arxiv.org/abs/2510.26692),
[repository](https://github.com/MoonshotAI/Kimi-Linear).

### Kimi Delta Attention [D]

KDA combines per-channel decay $\alpha_t$, delta-rule writes $\beta_t$,
normalized Q/K, short convolution + Swish, low-rank decay/write gates, RMSNorm,
and sigmoid output gate:

$$
S_t=\operatorname{diag}(\alpha_t)S_{t-1}
+\beta_t(v_t-S_{t-1}k_t)k_t^\top.
$$

The delta writes only the value not predicted from memory; diagonal decay lets
channels forget at different rates.

Released 48B/3B MoE: 256 routed experts; 8 routed + 1 shared; dense first layer;
3:1 KDA:MLA; recurrent components dimension 128. MLA preserves exact retrieval;
KDA reduces long sequential cost.

### Training [D]

- initial 1.4T at 4K; MuonClip; WSD schedule; peak LR $1.1\times10^{-3}$;
  global batch ~32M;
- released checkpoint through 5.7T; long context to 1M;
- broad SFT, reasoning SFT, RL over math/code/STEM;
- k1.5-family objective + high-quality pretraining loss + truncated importance
  sampling + dynamic KL + minibatching.

Report claims up to 75% lower KV cache, ~6× faster 1M decode, 2.9× prefill.
Kimi Linear is a validated K3 precursor, not proof of identical K3 layer ratios
or hyperparameters.

## 11. K2.5: multimodality and trainable orchestration

Sources: [paper](https://arxiv.org/abs/2602.02276),
[model card](https://huggingface.co/moonshotai/Kimi-K2.5),
[launch](https://www.kimi.com/blog/kimi-k2-5).

### 11.1 Architecture/continual pretraining [D/I]

- K2 language backbone: ~1T/32B, 61 layers, 384 experts, 8+1 active, MLA, 256K.
- ~400M MoonViT-3D initialized from SigLIP-SO-400M.
- Native-resolution NaViT processing; group four frames and temporal pool → 4×
  lower video-token rate.
- Starts near final K2 and processes ~15T additional mixed visual/text tokens,
  with more code/unique tokens and limited source repeats.
- **[I]** K2 15.5T + ~15T continuation suggests ~30T cumulative processed
  exposure, not 30T unique data.

Early fusion 10:90 visual/text outperforms mid 20:80 and late 50:50 in fixed-token
ablation across text/vision/multimodal. ViT receives ~1T caption-style tokens
aligned against Moonlight-16B-A3B; short projector connects to backbone.

Schedule table: ViT 1T at 4K; joint ~15T at 4K; long-context stages described as
500B and 200B as length grows 32K → 262,144; YaRN.

### 11.2 Visual data [D]

- captions, interleaved image-text, OCR, visual knowledge;
- perception/grounding, video, computer-use trajectories;
- cap synthetic captions;
- retrieve academic material and reformulate as questions;
- screenshot-to-HTML/React/SVG;
- desktop/mobile/web screenshots plus human action traces;
- hour-long video, boxes/points/contours/segmentation;
- filter/dedup media.

S3-like object storage dynamically blends, tokenizes, masks, augments, and packs.
Coordinate-preserving augmentation supports grounding; tiered caches and
deterministic resume support recovery.

### 11.3 “Zero-vision” SFT [D]

Main SFT uses text-only examples despite multimodal pretraining. Images are
manipulated through an IPython tool; the report says human visual trajectories
generalized worse. Text SFT + multimodal RL activates flexible visual tools.
Candidates come from K2, K2 Thinking, proprietary experts plus human/prompt/
automatic verification.

### 11.4 Joint text-vision RL [D]

Tasks: grounding/counting, charts/docs, vision STEM, OCR, points/boxes,
segmentation, synthetic puzzles.

Rewards:

- IoU/soft F1 localization;
- Gaussian point matching;
- segmentation IoU;
- normalized edit distance OCR;
- absolute-difference count reward;
- K2 verification for puzzles.

Text/visual experts combine with general reward models. Vendor ablation reports
text MMLU-Pro 84.7 → 86.4 and GPQA 84.3 → 86.4 after visual RL.

Objective changes toward token-level clipped ratios: ratios outside interval
receive zero gradient regardless of advantage sign. MuonClip remains. General
reward models cover chat, code, search, artifact creation with multiple rubrics.

### 11.5 Toggle efficient RL [D]

Alternate unconstrained reasoning-growth and budget-constrained efficiency.
Budgets are task-wise percentiles of correct rollout lengths, not one global
cap. Applied retrospectively to K2 Thinking: 25–30% fewer tokens with negligible
reported loss.

### 11.6 PARL / Agent Swarm [D]

Parallel-Agent Reinforcement Learning trains the orchestrator while subagents
remain frozen.

- orchestrator decides whether to spawn and what to delegate;
- subagents from intermediate checkpoints;
- subagent tokens excluded from orchestrator gradient;
- curriculum small subagents → larger;
- dynamic inference allocation.

Reward:

$$
r=\lambda_1r_{\text{parallel}}+
\lambda_2r_{\text{finish}}+r_{\text{performance}}.
$$

Parallel shaping prevents serial collapse; finish shaping prevents meaningless
spawns/uncompleted work; both anneal toward zero so task performance dominates.

Training prompts include wide/deep search, large-document analysis, batch
processing without explicitly ordering parallelism. Critical steps count main
work plus max branch depth, approximating wall time rather than total work.

Vendor results: BrowseComp 60.6 single → 78.4 swarm; WideSearch 72.7 → 79.0;
internal deep research 41.6 → 58.3; ~3–4.5× wall-clock. Product initial cap: 100
subagents and 1,500 calls. Independent bounded contexts return relevant results
to orchestrator—context sharding, not one shared transcript.

### 11.7 Infrastructure [D/U]

- modular Toolset, Judge, prompt enhancement, environments;
- white/black-box tools;
- each task an async coroutine; recursive subtask rollouts;
- up to 100K concurrent agent tasks;
- environment/sandbox pools;
- rollout log-probs for train/inference mismatch;
- gateway for proprietary/black-box models;
- decoupled encoder process replicates small ViT, load balances visual forward,
  drops activations/recomputes backward; ~90% reported text-only efficiency.

**[U]** exact modality ratios, RL/SFT totals, reward weights, cluster, duration,
cost.

## 12. K2.6

Sources: [launch](https://www.kimi.com/blog/kimi-k2-6),
[model card](https://huggingface.co/moonshotai/Kimi-K2.6).

Same disclosed K2.5 family: 1T/32B, 61 layers, 384 experts, MLA, ~400M vision,
256K. Product emphasis: long-horizon coding/design/proactive execution; up to
300 subagents and 4,000 coordinated steps; multi-hour/multi-day case studies.

Vendor scores: HLE tools 54.0, BrowseComp 83.2/86.3 swarm, OSWorld 73.1,
TerminalBench 66.7, SWE-Pro 58.6, SWE Verified 80.2. Typical evaluation uses
temperature 1, top-p 1, 262,144 context, up to 98,304 reasoning tokens with
benchmark-specific overflow rules.

**[U]** new data, environments, rewards, sample counts, optimizer, compute.

## 13. Attention Residuals

Sources: [paper](https://arxiv.org/abs/2603.15031),
[repository](https://github.com/MoonshotAI/Attention-Residuals).

Ordinary residual accumulation

$$
h_\ell=\sum_{i<\ell}v_i
$$

is replaced by learned normalized routing

$$
h_\ell=\sum_{i<\ell}\alpha_{i\to\ell}v_i,
\qquad\alpha=\operatorname{softmax}(q_\ell^\top k_i).
$$

A blockwise variant attends over block representatives, reducing residual
attention memory from roughly $O(Ld)$ to $O(Nd)$. Around eight blocks recover
most reported benefit. Scaling fits suggest conventional residual baseline
needs ~1.25× compute; 48B/3B Kimi Linear experiment reports GPQA +7.5 among other
gains. K3 adopts AttnRes, but exact K3 block/key/query details are **[U]**.

## 14. K2.7 Code

Sources: [model card](https://huggingface.co/moonshotai/Kimi-K2.7-Code),
[release notes](https://www.kimi.com/code/docs/en/kimi-code/whats-new.html),
[resource page](https://www.kimi.com/resources/kimi-k2-7-code).

Built on K2.6; same disclosed 1T/32B, 61 layers, 384 experts, 256K; text/image/
video input; thinking-only; INT4; vendor reports ~30% fewer thinking tokens.
Vendor evaluations: KimiCodeBench V2 62.0, Program 53.6, MLSLite 35.1, Claw24/7
46.9, MCP Atlas 76.0, MCPMark Verified 81.1, with source-specific setups.

**[U]** corpus, coding-RL operation, trajectory volume, reward, optimizer,
duration, cost.

## 15. K3: latest public endpoint at cutoff

Primary source: [official announcement](https://www.kimi.com/blog/kimi-k3), dated
2026-07-17 on the research page.

### Disclosed architecture [D]

- 2.8T total parameters;
- 896 experts, 16 selected;
- 1M context; native vision;
- Kimi Delta Attention, Attention Residuals, Stable LatentMoE;
- Quantile Balancing, Per-Head Muon, SiTU, Gated MLA;
- quantization-aware training from SFT;
- MXFP4 weights and MXFP8 activations.

Stable LatentMoE/Quantile Balancing are described at high level: expert-balance
signals from router-score quantiles, avoiding heuristic bias update and
sensitive balance hyperparameters; balanced expert execution uses static shapes
without host sync. Recommended deployments use supernodes with ≥64 accelerators.
KDA prefix-cache support was contributed to vLLM.

**[U]** active parameters. `2.8T × 16/896` is invalid because attention,
embeddings, shared components, and LatentMoE internals are omitted.

### Efficiency/agents [D as vendor claims]

Announcement claims ~2.5× scaling efficiency over K2 but does not define
loss/FLOP, throughput, total cost, or another unit. Product demonstrations
include 24-hour kernel work, multi-day chip design, and research runs with many
papers/equations/code/web fetches and self-improvement rounds. They demonstrate
intended operating scale, not the training environments that produced it.

### Release boundary [D/U]

At 2026-07-22, K3 was available in products/API, full weights were promised for
2026-07-27, and architecture/training/evaluation reports were promised later.
Therefore pretraining tokens/data, active params, optimizer, post-training, RL,
rewards, environments, hardware, duration, and cost remain **[U]**.

## 16. Reconstructed disclosed Moonshot workflow

### Corpus

- ingest web, code, PDF/textbooks/exercises/papers, images/video/screenshots,
  repos/issues/PRs;
- rules, language/domain classifiers, learned quality, embedding near-dedup,
  OCR/document classifiers;
- cap dominating synthetic categories and upsample valuable domains/languages;
- semantically rewrite selected knowledge/math;
- generate tests, grounding, screenshot-code, GUI, and tool-schema data.

### Pretraining

- short → long context activation;
- Muon/MuonClip matrices + AdamW non-matrices;
- sparse MoE with pipeline/expert/data parallelism;
- stable load balancing without large auxiliary loss;
- recompute, compressed activations, CPU offload;
- high-quality cooldown and long-context mid-training.

### SFT and environment construction

- generate candidates from earlier/strong specialist models;
- verify programmatically, with judges, and humans;
- broad instruction/code/reasoning/long/multimodal/tool data;
- async Gym interface;
- prioritize executable/terminal/sandbox/exact rewards;
- code-test agreement, learned matchers, rubric judges, hack/no-CoT tests;
- thousands to tens of thousands of concurrent environments.

### Online RL

- multiple samples/task and group-centered terminal outcomes;
- no value network in k1.5/K2;
- curricula and failure-prioritized sampling;
- partial continuation;
- gradual output/length control;
- high-quality pretraining loss against drift;
- K2.5 token-ratio clipping and Toggle budgets.

### Agent/multi-agent RL

- synthesize tools/agents/users/tasks/rubrics/trajectories;
- interleave reasoning and tool calls;
- freeze subagents and train orchestrator with PARL;
- anneal parallel/finish shaping toward task reward;
- independent agent contexts and compressed results.

### Deployment

- collocated rapid policy refresh where useful;
- pipeline-stream trillion-parameter checkpoints;
- disaggregated prefill/decode and distributed KV;
- low precision/QAT;
- harness-specific context management for long runs.

## 17. Critical unknowns

1. k1.5 architecture size.
2. K2 corpus proportions and unique-token count.
3. K2/K2.5 cluster size, duration, energy, cost.
4. SFT/RL token totals for most models.
5. Most exact reward weights, rollout counts, KL/clip coefficients, judge
   calibration.
6. K2-0905/K2 Thinking/K2.6/K2.7 update recipes.
7. K3 active parameters and training process at cutoff.
8. Cross-model benchmarks use different tool budgets, context rules, reasoning
   caps, and harnesses.
9. Product cases are not controlled robustness/training-efficacy experiments.
10. Processed tokens are not unique data; K2.5 continuation implies ~30T
    cumulative exposure only as an inference.
