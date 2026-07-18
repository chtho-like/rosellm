# DeepSeek: From Domain Data to Million-Token Agentic RL

**Verified through:** 2026-07-19. **Sources:** DeepSeek papers, official model
cards, repositories, and release notes. Benchmark results are vendor-reported
unless marked reproduced.

## 1. Executive conclusions

1. **DeepSeekMath introduced GRPO publicly, not DeepSeek-R1.** Its 2024 report
   provides the original DeepSeek formulation and concrete math-RL recipe.
2. **R1-Zero is the no-SFT experiment.** The production DeepSeek-R1 pipeline
   uses cold-start SFT, reasoning RL, rejection-sampled SFT, and a second mixed
   RL stage.
3. **Agentic RL becomes a first-class production pipeline in V3.1/V3.2.** V3.2
   discloses 85,267 coding/search/general/code-interpreter tasks and more than
   1,800 environments.
4. **V4 changes final capability consolidation.** More than ten specialists are
   trained with GRPO, but the final student is merged with full-vocabulary
   multi-teacher on-policy distillation (OPD), replacing V3.2's final mixed-RL
   consolidation.
5. DeepSeek releases many weights, reports, reference inference implementations,
   and selected kernels/systems. It does **not** release a complete pretraining
   or RL stack, raw corpus, reward datasets/models, or production orchestration.

## 2. Lineage

| Date | Release | Main transition | Agentic-RL significance |
|---|---|---|---|
| 2024-01-05 | DeepSeek LLM | Dense 7B/67B, 2T tokens | SFT + DPO; no online RL disclosed |
| 2024-01-25 | DeepSeek Coder | Repository-level code, FIM, 16K | Executable-code data foundation |
| 2024-02-05 | DeepSeekMath | Domain continued pretraining + GRPO | First public DeepSeek GRPO |
| 2024-05-07 | DeepSeek-V2 | 236B/21B MoE, MLA | Two-stage GRPO and hybrid rollout/training |
| 2024-06-17 | DeepSeek-Coder-V2 | V2 branch + 6T code-heavy tokens | Code/math verifier-based RL |
| 2024-09-05 | DeepSeek-V2.5 | Chat/coder capability combination | Update recipe unknown |
| 2024-12-26 | DeepSeek-V3 | 671B/37B, FP8, MTP, loss-free balancing | Specialist distillation + mixed GRPO |
| 2025-01-20 | DeepSeek-R1 | long-CoT RL | R1-Zero experiment + four-stage production R1 |
| 2025-03-24 | V3-0324 | post-training refresh | Tool/coding gains; recipe unknown |
| 2025-05-28 | R1-0528 | more post-training compute | Stronger reasoning/tool behavior; recipe partial |
| 2025-08-21 | V3.1 | hybrid thinking + agent training | Explicit search/coding/tool post-training |
| 2025-09-29 | V3.2-Exp | DeepSeek Sparse Attention | Specialists + mixed GRPO disclosed |
| 2025-12-01 | V3.2 | scaled agent environments | 85,267 tasks; >10% pretraining-equivalent post-training budget |
| 2026-04-24 | V4 Preview | 284B/13B Flash; 1.6T/49B Pro; 1M context | GRPO specialists consolidated by OPD |

## 3. DeepSeek LLM: dense baseline and global deduplication

Primary source: [DeepSeek LLM](https://arxiv.org/abs/2401.02954), Sections 2
and 4, pp. 4–6 and 12–13.

### Architecture and pretraining [D]

- Dense decoder-only 7B and 67B models; 4,096-token context.
- 7B: 30 layers, hidden width 4,096, 32 attention/KV heads.
- 67B: 95 layers, hidden width 8,192, 64 query heads and 8 KV heads (GQA).
- RMSNorm pre-normalization, SwiGLU, and RoPE.
- 2T training tokens.
- BBPE tokenizer trained on about 24 GB of multilingual text; 100,015 realized
  tokens in a nominal 102,400-token setup.
- AdamW \((\beta_1=.9,\beta_2=.95)\), weight decay .1, 2,000-step warmup,
  gradient clipping 1.0.
- Data/tensor/sequence/pipeline parallelism, FlashAttention, ZeRO-1, BF16
  forward/backward, FP32 gradients, asynchronous five-minute checkpoints.

### Data operation [D]

Global deduplication across 91 Common Crawl dumps removed **89.8%** of material,
compared with **22.2%** when deduplicating one dump independently. The number is
important for pipeline design: repeated web snapshots dominate raw volume, so
per-dump deduplication dramatically overstates unique corpus size.

### Alignment [D]

- 1.5M SFT examples.
- 1.2M helpful: 31.2% general, 46.6% math, 22.2% code.
- 300K safety examples.
- SFT: four epochs for 7B and two for 67B; learning rates \(10^{-5}\) and
  \(5\times10^{-6}\).
- DPO: one epoch, learning rate \(5\times10^{-6}\), batch 512.
- No online RL stage is reported.

**[U]** Source-level corpus mixture, DPO pair count, GPU count/hours/cost, and
training code remain undisclosed.

## 4. DeepSeek Coder: repository-level training data

Primary sources: [paper](https://arxiv.org/abs/2401.14196), Sections 2–3,
pp. 3–9; [repository](https://github.com/deepseek-ai/DeepSeek-Coder).

### Models and exposure [D]

- Dense 1.3B, 6.7B, and 33B models.
- 1.8T tokens at 4K, then 200B at 16K.
- Mixture: 87% code, 10% code-related English, 3% unrelated Chinese text.
- 50% document-level FIM in prefix–suffix–middle order before packing.
- 33B uses GQA.
- 16K extension: 1,000 steps, batch 512, RoPE scale 4/base 100,000.

### Repository acquisition [D]

1. Collect public GitHub repositories created before February 2023 in 87
   languages.
2. Apply repository/file filters similar to StarCoder.
3. Parse `import`, `using`, and `include` dependencies.
4. Topologically order files where the dependency graph permits.
5. Prefix repository paths and concatenate files as repository-level examples.
6. Apply repository-level near-deduplication.
7. Run compiler checks, learned quality scoring, and heuristics.
8. Decontaminate against HumanEval, MBPP, GSM8K, and MATH by exact 10-gram
   matching.

Only **32.8%** of initial code survives filtering. The disclosed final material
contains 797.92 GB and 603.173M files.

### Instruction tuning [D]

- Human instructions plus Alpaca-family data.
- About 2B instruction tokens.
- Batch about 4M tokens; learning rate \(10^{-5}\); 100 warmup steps.
- No RL stage.

This stage contributes realistic repository context and executable filtering,
but tool use and environment interaction are not yet policy-optimized.

## 5. DeepSeekMath: domain recovery and the first GRPO

Primary source: [DeepSeekMath](https://arxiv.org/abs/2402.03300), Sections 2–4.

### Math corpus construction [D]

1. Use OpenWebMath as positive seed data.
2. Train fastText on 500K positive and 500K negative Common Crawl examples.
3. Score about 40B deduplicated HTML pages.
4. Select the highest-scoring 40B-token subset after 40/80/120/160B trials.
5. Identify domains with more than 10% recall.
6. Human-label useful URL-path patterns inside those domains.
7. Recover matching pages and repeat classifier/domain discovery.
8. Repeat four times; 98% of final data was found by iteration three.
9. Apply 10-gram benchmark decontamination.

The underlying DeepSeekMath corpus contains 35.5M pages and 120B tokens.

### Actual training exposure [D]

DeepSeekMath-Base 7B starts from DeepSeek-Coder-Base-v1.5 and consumes **500B
sampled tokens**, not 120B unique tokens:

- 56% DeepSeekMath corpus;
- 4% AlgebraicStack;
- 10% arXiv;
- 20% GitHub code;
- 10% English/Chinese Common Crawl.

This distinction—underlying corpus versus repeated training exposure—must be
preserved in every model table.

### SFT and GRPO [D]

- 776K SFT examples: chain-of-thought, program-of-thought, and tool-integrated
  reasoning in English and Chinese.
- SFT: 500 steps, batch 256, max 4K, constant LR \(5\times10^{-5}\).
- RL prompts: about 144K GSM8K/MATH-style questions.
- 64 outputs per question; max response 1,024.
- Training batch 1,024; policy LR \(10^{-6}\); KL coefficient .04.
- One policy update per exploration batch.
- Outcome-reward and process-reward variants.
- Iterative RL refreshes the reward model and retains 10% historical replay.

GRPO removes PPO's learned critic and uses relative rewards in a prompt group.
Its public origin in the DeepSeek line is here, a year before R1.

## 6. DeepSeek-V2: MLA, fine-grained MoE, and online RL

Primary sources: [paper](https://arxiv.org/abs/2405.04434), Sections 3–4;
[repository](https://github.com/deepseek-ai/DeepSeek-V2).

### Architecture [D]

- 236B total, 21B active per token; 60 layers; hidden 5,120.
- MLA: 128 query heads of size 128; KV latent 512; query latent 1,536;
  decoupled RoPE dimension 64.
- First FFN dense; later layers use DeepSeekMoE.
- Each MoE layer has 2 shared and 160 routed experts; 6 routed experts active;
  expert intermediate 1,536.
- 128K after YaRN extension.

### Pretraining and systems [D]

- 8.1T tokens; base sequence 4K.
- AdamW .9/.95, weight decay .1, peak LR \(2.4\times10^{-4}\), 2K warmup;
  LR drops at 60% and 90%.
- Batch grows from 2,304 to 9,216 sequences over the first 225B tokens.
- H800 nodes with eight GPUs, NVLink/NVSwitch, and InfiniBand.
- 16-way zero-bubble pipeline; experts across eight nodes; ZeRO-1; no tensor
  parallelism.
- 172.8K H800 GPU-hours per trillion tokens, versus 300.6K for DeepSeek-67B.
- **[D→calc]** \(172.8\text{K}\times8.1\approx1.40\text{M}\) H800 GPU-hours
  for base pretraining, excluding context extension, post-training, experiments,
  and data processing.

### Context and alignment [D]

- YaRN scale 40, \(\alpha=1\), \(\beta=32\), target 160K.
- 1,000 extension steps at 32K, batch 576.
- 1.5M SFT examples, two epochs, LR \(5\times10^{-6}\).
- Two RL stages with GRPO:
  1. code/math reasoning with rule or learned reasoning rewards;
  2. helpfulness, safety, and rule rewards together.
- Code preferences use compiler feedback; math uses ground truth.
- Reward models initialize from SFT and include pointwise/pairwise forms.
- Hybrid training/inference engine with vLLM and CPU offload generates large
  online batches; the report finds online RL better than compared offline paths.

**[U]** Prompt count, group size, reward weights, post-training compute/cost, and
exact corpus percentages.

## 7. DeepSeek-Coder-V2: verifiers and learned code reward

Primary source: [DeepSeek-Coder-V2](https://arxiv.org/abs/2406.11931).

### Lineage/data [D]

Coder-V2 starts from an intermediate V2 checkpoint after 4.2T tokens—not the
finished 8.1T model—and consumes 6T more, for 10.2T exposure on this branch:

- 60% code;
- 10% math;
- 30% natural language.

Underlying code data is about 1.17T tokens: 821B source, 185B code-related text,
70B web code, and 94B from iterative GitHub recovery. Repositories are dated
before November 2023 and cover 338 languages.

### Models and post-training [D]

- 16B/2.4B active and 236B/21B active; 128K.
- 16B uses 50% FIM; 236B uses next-token prediction.
- Extension: 1,000 steps at 32K/batch 1,152, then 1,000 at 128K/batch 288.
- 20K code + 30K math instructions plus V2 general instructions.
- About 300M unique instruction tokens sampled to 1B exposure.
- SFT batch 1M tokens, LR \(5\times10^{-6}\), 100 warmup steps.
- RL uses roughly 40K code/math prompts.
- Math reward is ground truth.
- Raw compiler 0/1 feedback was judged noisy; compiler outcomes supervise a
  learned code reward model, then GRPO optimizes against it.

This is a useful example of the verifier/reward-model boundary: executable
signals can still be too noisy or incomplete to serve directly as the scalar
policy reward.

## 8. V2.5: capability combination with an undisclosed recipe

Sources: [V2.5 announcement](https://api-docs.deepseek.com/news/news0905) and
[V2.5-1210](https://api-docs.deepseek.com/news/news1210).

**[D]** The September model combines V2-0628 conversational behavior with
Coder-V2-0724 coding strengths; the December update improves math/coding. The
broad 236B/21B, 128K V2-family architecture remains.

**[U]** Merge algorithm, data mixture, SFT/RL steps, preference volume, reward
models, compute, and cost. Do not call it weight averaging, model soup,
distillation, or continued pretraining without evidence.

## 9. DeepSeek-V3: architecture/system co-design and specialist post-training

Primary sources: [paper](https://arxiv.org/abs/2412.19437), Sections 2–4 and
Table 1; [repository](https://github.com/deepseek-ai/DeepSeek-V3).

### Architecture [D]

- 671B main-model parameters, 37B active; 61 layers, hidden 7,168.
- Hosted files can total 685B because one-depth MTP adds 14B.
- MLA: 128 heads; KV latent 512; query latent 1,536; RoPE head 64.
- First three FFNs dense.
- Later MoE: 1 shared, 256 routed, 8 active; expert intermediate 2,048.
- Auxiliary-loss-free routing bias plus sequence-level balance loss \(10^{-4}\).
- One-depth multi-token prediction.

### Data and optimization [D/U]

- 14.8T processed training tokens **[D]**; unique corpus size **[U]**.
- Higher math, code, multilingual, and quality emphasis than V2, but exact
  source mixture is **[U]**.
- 128K BBPE vocabulary; cross-document packed attention; FIM probability .1.
- AdamW .9/.95, weight decay .1.
- 2K warmup to \(2.2\times10^{-4}\); constant through 10T; cosine over 4.3T to
  \(2.2\times10^{-5}\); final 500B at \(2.2\times10^{-5}\) for 333B then
  \(7.3\times10^{-6}\) for 167B.
- Batch grows 3,072 → 15,360 sequences over first 469B.
- MTP loss .3 for first 10T and .1 for final 4.8T.

### Context, infrastructure, and disclosed cost [D]

- YaRN: 1,000 steps at 32K/batch 1,920, then 1,000 at 128K/batch 480, LR
  \(7.3\times10^{-6}\).
- 2,048 H800 GPUs; FP8 mixed precision; DualPipe; eight-node expert parallel;
  ZeRO-1; no tensor parallelism.
- Pretraining 2.664M H800-hours, estimated $5.328M.
- Context extension 119K hours/$238K.
- Post-training 5K hours/$10K.
- Total 2.788M hours/$5.576M at $2/H800-hour.

The dollar value is a rental-equivalent for listed runs, not audited total
development cost; it excludes prior research, ablations, data work, and failed
runs.

### Post-training [D]

- 1.5M SFT examples; two epochs; cosine LR \(5\times10^{-6}\to10^{-6}\).
- Domain reasoning experts trained with SFT and RL generate long-CoT material.
- R1-family outputs are rejection sampled.
- Non-reasoning data begins from V2.5 and is human-verified.
- Rules for deterministic math/code; model rewards for writing, role-play, and
  open QA.
- GRPO mixes code, math, writing, role-play, QA, helpfulness, and safety.
- Constitutional/self-reward procedures use V3 voting.
- R1 distillation improves reasoning but lengthens responses, exposing a
  capability/style tradeoff.

## 10. DeepSeek-R1 and R1-Zero

Primary sources: [paper](https://arxiv.org/abs/2501.12948), Sections 2–3 and
Appendices B.1–B.6; [repository](https://github.com/deepseek-ai/DeepSeek-R1);
[announcement](https://api-docs.deepseek.com/news/news250120).

The revised report uploaded 2026-01-04 is much longer than the original January
2025 release. Appendix-only facts should be identified as revised-report
disclosures in historical analysis.

### 10.1 R1-Zero: no-SFT experiment [D]

Initialization: DeepSeek-V3-Base; no cold-start SFT; same 671B/37B family.

Rollout/update configuration:

- policy LR \(3\times10^{-6}\);
- KL coefficient .001;
- reported GRPO clip ratio 10;
- temperature 1;
- 16 responses per question;
- max 32,768 tokens until step 8,200, then 65,536;
- 32 unique prompts/update → 512 sampled responses;
- reference policy refreshed every 400 steps;
- each rollout collection generates 8,192 outputs in 16 minibatches;
- one inner epoch;
- 10,400 policy steps, about 1.6 reported epochs.

Reward:

\[
R=R_{\text{accuracy}}+R_{\text{format}}
\]

with equal weights. Math uses answer parsing/matching; code uses compiler/hidden
tests; logic uses deterministic verification; format enforces
`<think>...</think><answer>...</answer>`.

DeepSeek explicitly avoids neural outcome/process reward models in this stage
because of reward hacking and repeated retraining cost. The result develops
longer reasoning and self-reflection, but also readability and language-mixing
problems.

### 10.2 Production R1: four stages [D]

#### Stage 1 — cold-start reasoning SFT

- “Thousands” of long-CoT examples.
- R1-Zero trajectories sampled at temperature 1.
- Filter correctness, readability, repetition, and language consistency.
- SymPy checks symbolic expressions.
- DeepSeek-V3 rewrites low-quality outputs.

#### Stage 2 — reasoning RL

Essentially R1-Zero settings: LR \(3\times10^{-6}\), KL .001, clip ratio 10,
temperature 1, 16 outputs, 32 questions/step, max 32,768, batch 512, reference
refresh every 400 steps. A language-consistency reward measures fraction of
words in the target language.

#### Stage 3 — rejection-sampled SFT

804,745 examples:

| Domain | Count | Mean tokens where disclosed |
|---|---:|---:|
| Math | 395,285 | 6,094 |
| Code | 211,129 | 7,435.7 |
| STEM | 10,124 | — |
| Logic | 10,395 | — |
| General | 177,812 | — |
| Total | 804,745 | 5,355.3 overall |

Roughly 600K reasoning and 200K non-reasoning examples. Train 2–3 epochs, max
32,768, batch 128, cosine \(5\times10^{-5}\to5\times10^{-6}\).

#### Stage 4 — mixed RL

- Rule rewards for reasoning.
- Helpfulness and safety reward models for general prompts.
- Format/language rewards.
- Temperature .7; 1,700 steps.
- General instruction/preference data enters only during the final 400 steps;
  longer preference-reward optimization caused reward hacking.

### 10.3 Prompt/reward operations [D]

Approximate domains: 26K math, 17K algorithmic code, 8K bug fixing in prose,
22K STEM, 15K logic, 66K helpfulness, 12K harmlessness. Table/prose are ambiguous
about whether the 8K bug-fix prompts are included in the 17K code total; retain
the contradiction.

Code prompt pipeline:

- 5,151 Codeforces and 2,504 AtCoder problems;
- V2.5 generates additional tests;
- correct submissions identify invalid tests;
- incorrect submissions help choose discriminative tests;
- GitHub issues provide executable bug-fix tasks.

Reward models:

- helpfulness: 66K preference pairs, R1 backbone + scalar head, batch 256, LR
  \(6\times10^{-6}\), one epoch, max 8,192;
- safety: 106K prompts with safe/unsafe point labels, same optimization;
- prompts include public sources and opt-in user data.

### 10.4 Infrastructure and cost [D]

- vLLM actor rollout;
- separate reward/reference inference services;
- asynchronous compilers, answer matchers, format verifiers;
- PPO/GRPO/DPO-capable trainer;
- length-sorted best-fit packing, DualPipe, hot-expert redundancy, MTP
  speculative rollout;
- host/disk offload between phases.

At $2/H800-hour:

- R1-Zero: 512 H800 × ~198 h ≈ 101K hours, $202K;
- SFT data creation: 5K hours, $10K;
- R1: 512 H800 × ~80 h ≈ 41K hours, $82K;
- total: 147K hours, $294K.

This is post-training on an already trained V3, not total R1 development cost.

### 10.5 Distillation [D]

The roughly 800K examples fine-tune Qwen/Llama bases for 2–3 epochs, max 32,768,
batch 64. Initial learning rates: Qwen 1.5B \(10^{-4}\), 7B \(8\times10^{-5}\),
14B \(7\times10^{-5}\), 32B \(6\times10^{-5}\); Llama 8B
\(5\times10^{-5}\), 70B \(2\times10^{-5}\).

### 10.6 Explicit limitations [D]

The original R1 recipe did not natively train tool use, struggled with structured
output, and could not scale software-engineering RL because environments were
slow and difficult to validate. These limitations motivate V3.1/V3.2.

## 11. Interim production checkpoints

### V3-0324

[Official release](https://api-docs.deepseek.com/news/news250325) discloses
improved reasoning, frontend code, Chinese writing, and tool/function calling.
**[U]** Dataset, RL steps, rewards, and cost.

### R1-0528

[Announcement](https://api-docs.deepseek.com/news/news250528) and
[model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) disclose more
post-training compute and algorithmic optimization, a 64K generation limit, and
better reasoning/function/coding-agent results. Average reported AIME response
length roughly doubles from 12K to 23K. **[U]** data, rollouts, rewards, GRPO
settings, GPU hours, and cost.

### V3.1 / V3.1-Terminus

Sources: [V3.1 release](https://api-docs.deepseek.com/news/news250821),
[model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/README.md),
[Terminus](https://api-docs.deepseek.com/news/news250922).

- V3.1-Base continues V3-Base with ~630B tokens at 32K and ~209B at 128K,
  totaling ~839–840B.
- One checkpoint exposes thinking/non-thinking modes through the chat template.
- Explicit tool-use, search-agent, and coding-agent post-training.
- Terminus is a reliability/language/code/search-agent refresh.
- **[U]** SFT/RL sizes, algorithm settings, environments, rollout tokens, cost.

## 12. V3.2-Exp: sparse attention and unified post-training

Primary source: [V3.2-Exp report/code](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp),
Sections 2–3.

### DeepSeek Sparse Attention [D]

- A lightweight lightning indexer scores earlier positions.
- Each query attends to top-ranked KV entries in the MLA/MQA representation.
- Top-k 2,048.
- The only architectural change from V3.1-Terminus.

Conversion:

1. Freeze main model; train indexer only, LR \(10^{-3}\), 1,000 steps,
   \(16\times128\text{K}\) tokens/step → 2.1B tokens.
2. Joint sparse continuation, main LM LR \(7.3\times10^{-6}\), 15,000 steps,
   \(480\times128\text{K}\) tokens/step → 943.7B tokens. Indexer KL gradients
   are detached from the main model.

Total conversion exposure is about 945.8B tokens.

### Post-training [D]

- Specialists: writing/general QA, math, competitive code, logic, agentic code,
  agentic search; separate thinking/non-thinking specialists.
- Large-scale domain RL.
- Specialist trajectories distilled into one checkpoint.
- A final mixed GRPO stage combines reasoning, agent, and alignment domains to
  reduce forgetting.
- Reasoning/agent rewards: verified outcome, length penalty, language
  consistency.
- Open tasks: prompt-specific rubric evaluated by a generative reward model.

## 13. V3.2: production agentic-RL factory

Primary sources: [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf),
Sections 2–4 and Table 1; [release](https://api-docs.deepseek.com/news/news251201);
[model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2).

### Specialists and mixed GRPO [D]

Named specialists: writing, general QA, math, programming, general logic,
general agent tasks, agentic coding, and agentic search, with thinking and
non-thinking modes. Each receives large-scale RL, trajectories are distilled,
then thousands of mixed-GRPO steps close the remaining specialist/student gap.

- Reasoning/agent rewards: rules, length penalty, language consistency.
- Open-ended tasks: prompt-specific rubric + generative reward model.
- All domains train together rather than sequentially to reduce forgetting.

### On-policy fidelity controls [D]

- corrected KL estimation with importance sampling;
- domain-dependent KL strengths;
- masking of strongly divergent negative-advantage sequences;
- **Keep Routing:** reuse MoE routes chosen at rollout during learner replay;
- **Keep Sampling Mask:** respect rollout top-p/top-k action support in training.

These address a deep implementation issue: inference and training distributions
can differ because of quantization, kernels, routing, sampling masks, and policy
lag even when the checkpoint name is the same.

### Thinking with tools [D]

Reasoning state is retained across tool calls/results. A new real user message
discards earlier hidden reasoning. Cold-start data combines non-agent reasoning
patterns with non-thinking tool-call patterns, followed by RL.

### Agent data [D]

| Domain | Tasks/prompts | Operation |
|---|---:|---|
| Coding agent | 24,667 | real repositories and extracted software tasks |
| Search agent | 50,275 | real + synthetic long-tail questions |
| General agent | 4,417 | synthesized databases, tools, solutions, verifiers |
| Code interpreter | 5,908 | real/extracted executable tasks |
| **Total** | **85,267** | more than 1,800 environments overall |

Do not misstate this as 85K environments.

#### Search synthesis

1. Mine long-tail entities from web corpora.
2. A question agent searches and proposes problems.
3. Heterogeneous answer agents attempt solutions.
4. A verifier performs multiple checks.
5. Keep only verified ground truth with verifiably failed alternative attempts.
6. Use rubrics/generative rewards for residual qualities.

#### Coding environments

1. Mine millions of GitHub issue/PR pairs.
2. Apply heuristics and LLM quality filters.
3. Recover gold patches/tests.
4. Setup agent installs dependencies and runs tests.
5. Parse JUnit-like structured results.
6. Accept only when gold patch converts at least one failing test to passing and
   creates no regression from passing to failing.
7. Cover Python, Java, JavaScript/TypeScript, C/C++, Go, and PHP.

#### General-agent synthesis

- 1,827 environments with tool schemas/databases.
- Generate task, reference solution, and verifier.
- Require the solution to interact through tools.
- Iteratively increase difficulty.
- Retain RL tasks with non-zero pass@100.

DeepSeek states V3.2 post-training compute exceeded **10% of pretraining cost**,
but gives no absolute GPU-hour or dollar figure.

## 14. DeepSeek-V4 Preview: million-token models and OPD

Primary sources: [V4 Pro report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf),
especially Sections 2, 4, and 5; [official release](https://api-docs.deepseek.com/news/news260424/).

### 14.1 Models [D]

| Item | V4-Flash | V4-Pro |
|---|---:|---:|
| Total / active parameters | 284B / 13B | 1.6T / 49B |
| Layers | 43 | 61 |
| Hidden width | 4,096 | 7,168 |
| Routed experts / active | 256 / 6 | 384 / 6 |
| Expert intermediate | 2,048 | 3,072 |
| CSA top-k | 512 | 1,024 |
| Pretraining exposure | 32T | 33T |
| Maximum batch | 75.5M tokens | 94.4M tokens |
| Context | 1M | 1M |

Both use one shared expert, MTP depth one, manifold-constrained
Hyper-Connections expansion four, and 20 Sinkhorn iterations. Every block is
MoE; the first three use hash routing.

### 14.2 Architecture [D]

**Manifold-Constrained Hyper-Connections (mHC):** expand the residual stream into
four paths and constrain mixing to a doubly stochastic matrix through Sinkhorn
normalization. This adds learned residual routing while bounding the unstable
mixing seen in unconstrained Hyper-Connections.

**Compressed Sparse Attention (CSA):** compress each four KV positions, apply
DSA selection over compressed entries, and retain a 128-token sliding local
window.

**Heavily Compressed Attention (HCA):** compress every 128 KV positions and use
dense attention over the shortened sequence. CSA and HCA layers interleave.

At 1M context the report estimates KV cache at roughly 2% of a BF16 GQA8,
head-dimension-128 baseline.

**MoE:** routing affinity changes from sigmoid to
`sqrt(softplus(...))`; loss-free balancing remains with sequence balance
\(10^{-4}\); node restrictions are removed; the first three layers hash-route.

### 14.3 Data and pretraining [D/U]

- Begins from the V3 data pipeline; filters batched auto-generated and templated
  web content.
- Math/code remain core; agent material enters mid-training.
- More multilingual/long-tail cultural, scientific, technical, and long-document
  data.
- More than 32T processed tokens; exact proportions, licenses, unique counts,
  dedup thresholds, and agentic volume are **[U]**.
- 128K vocabulary, FIM/token splitting, sample-isolated packed attention.

Optimizer:

- Muon for most matrices; AdamW for embeddings, output head, RMSNorm.
- AdamW .9/.95, epsilon \(10^{-20}\), decay .1.
- Muon momentum .95, decay .1, update RMS .18.

Flash: 32T, 2K warmup, peak LR \(2.7\times10^{-4}\), cosine final
\(2.7\times10^{-5}\). Pro: 33T, peak \(2\times10^{-4}\), final
\(2\times10^{-5}\). Context curriculum is 4K → 16K → 64K → 1M; Flash uses
dense attention for the first 1T before sparse conversion, while Pro retains a
longer dense stage. MTP loss .3 then .1 at LR decay.

Stability:

- Anticipatory Routing uses delayed historical state to precompute future routes
  only after a loss-spike detector and rollback; about 20% overhead while active,
  negligible claimed run-wide because activation is rare.
- SwiGLU linear branch clamped to [-10, 10], gate upper bound 10.

### 14.4 Specialist GRPO and on-policy distillation [D]

1. Build more than ten domain specialists.
2. Fine-tune each.
3. Train each with GRPO and domain prompts/rewards.
4. Train separate non-think, think/high, and think-max effort specialists with
   different context/length controls.
5. Merge them into one student with multi-teacher OPD, replacing V3.2's final
   mixed GRPO stage.

Easy-to-verify tasks use rules/tests. Hard-to-verify tasks use rubric-guided
generative reward models; conventional scalar RMs are removed. RL is also
applied to the generative judge.

For student \(\pi_\theta\) and expert \(\pi_{E_i}\):

\[
L_{\mathrm{OPD}}=
\sum_iw_iD_{\mathrm{KL}}(\pi_\theta\Vert\pi_{E_i}).
\]

The student generates the trajectories, preserving student-state coverage.
DeepSeek computes full-vocabulary reverse KL rather than only a sampled-token
approximation.

Infrastructure:

- teacher weights in centralized distributed storage with ZeRO-like on-demand
  sharding;
- cache teacher final hidden states instead of >100K-vocabulary logits;
- reconstruct logits with teacher output head during learning;
- sort minibatch examples by teacher so only one teacher head is resident;
- TileLang kernel for exact teacher/student KL.

Teacher weighting, disagreements, complete reward mixture, and sample counts are
**[U]**.

### 14.5 Tool protocol, FP4, and rollout fidelity [D]

- XML-style DSML tool calls reduce escaping/schema failure.
- Tool conversations preserve complete reasoning across tools/results/later user
  messages; ordinary chat still discards reasoning at the next user message.
- “Quick Instruction” special tokens reuse KV cache for routing, search-query
  generation, domain/authority classification, titles, and URL read decisions.
- FP4 MXFP4 expert weights and FP4 CSA indexer Q/K; index scores FP32 → BF16.
- Reported selector speedup 2× at 99.7% KV-entry recall.
- RL rollout uses real FP4 deployment weights rather than simulated
  quantization, improving actor/deployment numerical fidelity.

### 14.6 Million-token fault tolerance and DSec [D]

- token-granular write-ahead log for every generation;
- persist unfinished KV cache on preemption;
- reconstruct KV from logged tokens after hardware failure;
- avoid restarting incomplete generations, which would select for shorter
  trajectories;
- split lightweight metadata from heavy token arrays; globally shuffle/pack
  metadata and load token arrays through shared memory per minibatch.

DeepSeek Elastic Compute (DSec) is a Rust platform over 3FS:

- API server, per-host edge agent, cluster watcher;
- hundreds of thousands of concurrent sandboxes;
- prewarmed functions, Docker-compatible containers, Firecracker microVMs, and
  QEMU VMs behind one Python API;
- EROFS/3FS lazy layers and copy-on-write overlays;
- globally ordered trajectory logs, preemption-safe resumption, replay without
  repeating non-idempotent commands, provenance and deterministic replay.

DSec is described, not released as complete public source in the cited report.
V4 hardware count, duration, energy, and dollar cost are **[U]**.

## 15. Reconstructed disclosed workflow

```text
raw web/code/math/documents
  -> global deduplication and quality filtering
  -> domain recovery/classification
  -> benchmark decontamination
  -> tokenization, packing, FIM
  -> base pretraining
  -> domain/agent/long-context mid-training
  -> sparse-attention conversion where applicable
  -> specialist SFT
  -> specialist GRPO with rules/tests/generative judges
  -> trajectory generation and rejection/distillation
  -> mixed GRPO consolidation [V3/V3.2]
     or multi-teacher OPD      [V4]
  -> quantization-aware adaptation
  -> deployment-equivalent rollouts
  -> agent/chat/API serving
```

This is supported at stage level. Exact live API routing, hidden prompts/tools,
online adaptation, user-data mixture, safety filters, and production topology
remain **[U]** unless explicitly disclosed.

## 16. Source-level study map

- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3): reference
  `inference/model.py` for MLA, MoE, MTP, and FP8 handling.
- [V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp): DSA inference and
  report.
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA): dense/sparse MLA kernels.
- [DeepEP](https://github.com/deepseek-ai/DeepEP): expert-parallel all-to-all.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM): FP8 GEMM.
- [DualPipe](https://github.com/deepseek-ai/DualPipe): bidirectional pipeline.
- [3FS](https://github.com/deepseek-ai/3FS): distributed storage.
- [V4-Pro inference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference):
  CSA/HCA reference files.
- [DeepSeek-Prover-V2](https://github.com/deepseek-ai/DeepSeek-Prover-V2): Lean
  theorem decomposition, proof search, binary verification.
- [DeepSeekMath-V2](https://github.com/deepseek-ai/DeepSeek-Math-V2): verifier
  and generator co-improvement for mathematical proof.

| Artifact | Public? |
|---|---|
| Major weights | generally yes |
| Lightweight/reference inference | yes |
| Selected kernels/communication/storage | yes |
| Exact pretraining framework | no |
| Complete RL orchestration | no |
| DSec implementation | not in cited release |
| Raw corpus | no |
| R1/V3.2 prompt/reward data | no |
| Reward models/data | generally no |
| Optimizer/checkpoint state | no |

Pin exact revisions before making source-code claims.

## 17. Specialized branches

### DeepSeek-Prover-V2

[Official repository/report](https://github.com/deepseek-ai/DeepSeek-Prover-V2):
7B and 671B Lean 4 models; the 671B branch starts from V3-Base. V3 decomposes
difficult theorems into subgoals, a smaller 7B prover searches them, successful
subgoals are recomposed, and informal reasoning plus formal proof becomes
cold-start data. SFT is followed by binary Lean-verification RL.

### DeepSeekMath-V2

[Official repository/report](https://github.com/deepseek-ai/DeepSeek-Math-V2):
starts from V3.2-Exp-Base; trains verifier and generator; scales verification
compute to label proof attempts beyond a fixed verifier; automatically generated
verified proofs improve both. Its data/reward approach later informs
V3.2-Speciale.

## 18. Claims to avoid

- “R1 was trained entirely without SFT.” Only R1-Zero was.
- “DeepSeek spent $5.6M to develop V3.” That is a listed-run rental-equivalent.
- “R1 cost only $294K end to end.” That is disclosed post-training on V3.
- “V2.5 used merge method X.” The method is unknown.
- “V3 had 14.8T unique tokens.” It had 14.8T processed exposure.
- “V3.2 had 85K environments.” It had 85,267 tasks and >1,800 environments.
- “V4's final model was trained by mixed GRPO.” Specialists use GRPO; final
  consolidation is OPD.
- “V4 High/Max training used 128K/384K.” Those are evaluation contexts, not
  disclosed training hyperparameters.
- “DeepSeek is fully open source.” It is more accurate to say open weights,
  reports, reference inference, and selected systems—not an end-to-end
  reproducible training release.
