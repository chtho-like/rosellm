# Model Training Token Ledger

**Verified through:** 2026-07-23. **Scope:** publicly disclosed text and
multimodal foundation-model training exposure, with continued pretraining and
post-training kept separate. Values are vendor-reported unless explicitly
marked as a calculation or inference.

This ledger answers a deceptively difficult question: “How many tokens was this
model trained on?” A single number can describe a fresh base-model run, a
continued-pretraining branch, a corpus before sampling, repeated exposure to the
same material, multimodal positions, or even the maximum input length. Those
quantities are not interchangeable.

For DeepSeek release dates, including specialist branches and API-only updates,
read the companion [DeepSeek release timeline](deepseek-release-timeline.md).
For training mechanisms behind the numbers, continue to the detailed
[DeepSeek](agentic-rl/case-studies/deepseek.md),
[Kimi](agentic-rl/case-studies/kimi.md),
[GLM](agentic-rl/case-studies/glm.md), and
[Qwen](agentic-rl/case-studies/qwen-meta-mistral.md) case studies.

## Executive comparison

The most useful current anchors are:

| Vendor lineage | Latest generation with a usable disclosure | Public training-token statement | Correct interpretation |
|---|---|---:|---|
| DeepSeek | V4-Flash / V4-Pro | 32T / 33T | separate V4 pretraining exposures; exact unique data is unknown |
| Alibaba Qwen | Qwen3 | about 36T | family pretraining exposure across three stages |
| Zhipu AI / Z.ai | GLM-5 | 27T + 1T + 500B + 50B = 28.55T | main pretraining plus three long-context/mid-training stages |
| Moonshot Kimi | K2 | 15.5T | fresh K2 processed exposure |
| Moonshot Kimi | K2.5 | about 15T additional | continued pretraining from a near-final K2 checkpoint; about 30T cumulative lineage exposure is an inference |
| Moonshot Kimi | K3 | unknown | architecture and product were announced, but the pretraining-token total was not public at the cutoff |
| Alibaba Qwen | Qwen3.5 | “tens of trillions” | qualitative scale only; no exact total |
| Zhipu AI / Z.ai | GLM-5.1 / 5.2 | unknown added amount | later post-training/product generations do not disclose a replacement base-token total |

DeepSeek-V4 therefore did not jump from V3's 14.8T to “more than 32T” merely
because its context window became one million tokens. The
[V4 report snapshot](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/a7aaed80dd2df27620eb534454253ea25eb11c7a/DeepSeek_V4.pdf)
states that V4-Flash processes 32T tokens and V4-Pro 33T tokens during
pretraining. The **one-million-token** figure is the supported context length:
the maximum sequence capacity at inference, not the size of the training run.

## Counting rules

### Units and evidence labels

- **B** means $10^9$ tokens and **T** means $10^{12}$ tokens.
- **[D] Disclosed:** a primary source states the value.
- **[C] Confirmed artifact:** a released model card, configuration, registry, or
  source artifact establishes it.
- **[I] Inferred:** arithmetic or lineage reasoning over disclosed stages; the
  vendor did not publish the resulting total as one number.
- **[U] Unknown:** public sources do not establish an exact value.

### What the number normally measures

**Training-token exposure** is the number of token positions processed by the
optimizer. It is not necessarily the number of unique tokens in the underlying
corpus. If one token appears in five epochs, it contributes five processed
tokens. Synthetic rewrites and repeated high-quality samples also count each
time they are processed.

Keep five quantities separate:

1. **Corpus size:** tokens available after or before filtering.
2. **Fresh pretraining exposure:** a model trained from initialization.
3. **Continual Pretraining (CPT):** additional next-token training of an
   existing checkpoint.
4. **Post-training:** Supervised Fine-Tuning (SFT), preference optimization,
   Reinforcement Learning (RL), and distillation.
5. **Context length:** positions accepted in one sequence or request.

Tokenizer vocabularies and segmentation differ, so token totals are only
approximate cross-vendor measures of data exposure. They do not directly
measure information content, compute, quality, deduplication, or model
capability.

### Multimodal qualification

A “token” in a multimodal report may be:

- an ordinary text token;
- a continuous visual embedding occupying a language-model sequence position;
- a discrete image-code token;
- a caption target used to train only a vision encoder; or
- a mixed sequence position processed by the joint model.

This ledger preserves the report's unit and identifies stages that do not update
the language backbone.

## DeepSeek

### Text and code foundation lineage

| Model or branch | Training-token disclosure | Accounting note | Evidence |
|---|---:|---|---|
| DeepSeek-Coder v1 | 1.8T at 4K + 200B at 16K = 2T | fresh code-model training; the later 2B instruction tokens are post-training | [D] [paper](https://arxiv.org/abs/2401.14196), [repository](https://github.com/deepseek-ai/DeepSeek-Coder) |
| DeepSeek-LLM 7B / 67B | 2T | fresh bilingual base training | [D] [report](https://arxiv.org/abs/2401.02954) |
| DeepSeekMoE 16B | 2T | fresh MoE experiment on the DeepSeek bilingual corpus | [D] [report](https://arxiv.org/abs/2401.06066) |
| DeepSeekMath 7B | +500B CPT | initialized from Coder-v1.5; the recovered math corpus is 120B, not the processed total | [D] [report](https://arxiv.org/abs/2402.03300) |
| DeepSeek-V2-Lite | 5.7T | separate smaller V2-family base run | [D] [V2 report](https://arxiv.org/abs/2405.04434), Appendix C |
| DeepSeek-V2 | 8.1T | fresh V2 base exposure | [D] [V2 report](https://arxiv.org/abs/2405.04434) |
| DeepSeek-Coder-V2 | 4.2T inherited point + 6T CPT = 10.2T branch exposure | starts from an intermediate V2 checkpoint, not the finished 8.1T V2 | [D] [Coder-V2 report](https://arxiv.org/abs/2406.11931) |
| DeepSeek-V2.5 / V2.5-1210 | unknown added amount | combines V2-0628 and Coder-V2-0724 capabilities; merge/training recipe is undisclosed | [U] [release](https://api-docs.deepseek.com/news/news0905/), [update](https://api-docs.deepseek.com/news/news1210/) |
| DeepSeek-V3 | 14.8T | processed base-pretraining exposure; unique corpus size is unknown | [D] [V3 report](https://arxiv.org/abs/2412.19437) |
| DeepSeek-R1-Zero / R1 | no separately disclosed broad pretraining | both are post-trained from V3-Base; do not add an invented base-token count | [U] [R1 report](https://arxiv.org/abs/2501.12948) |
| V3-0324 / R1-0528 | unknown added amount | post-training updates; no replacement broad-pretraining total | [U] [DeepSeek change log](https://api-docs.deepseek.com/updates/) |
| DeepSeek-V3.1-Base | +630B at 32K + 209B at 128K = about 839B CPT | **[I]** 14.8T + 0.839T gives about 15.64T lineage exposure | [D/I] [model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base) |
| DeepSeek-V3.2-Exp conversion | 2.1B indexer-only + 943.7B joint sparse continuation = about 945.8B | **[I]** adding V3, V3.1, and this conversion gives about 16.59T; production V3.2 may include undisclosed work | [D/I] [report and repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| DeepSeek-V3.2 / V3.2-Speciale | no separately disclosed additional broad total | production release follows the V3.2-Exp lineage; Speciale was a temporary reasoning variant | [U] [release log](https://api-docs.deepseek.com/updates/) |
| DeepSeek-V4-Flash | 32T | V4 pretraining exposure; exact unique-token count unknown | [D] [V4 report snapshot](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/a7aaed80dd2df27620eb534454253ea25eb11c7a/DeepSeek_V4.pdf) |
| DeepSeek-V4-Pro | 33T | V4 pretraining exposure; not a one-trillion-token CPT claim over Flash | [D] [V4 report snapshot](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/a7aaed80dd2df27620eb534454253ea25eb11c7a/DeepSeek_V4.pdf) |

Two cumulative figures deserve special caution:

- **V3.1 about 15.64T** and **V3.2-Exp about 16.59T** are ledger arithmetic,
  not vendor headlines. Stage totals are rounded, and the exact starting
  checkpoint or unreported intervening work may change the real number.
- R1's SFT and RL create a different policy from V3-Base but do not constitute
  a newly disclosed 14.8T-scale pretraining run.

### Vision, theorem proving, generation, and OCR branches

| Branch | Public token information | Boundary |
|---|---:|---|
| DeepSeek-VL 1.3B / 7B | language checkpoints had processed about 500B / 2T text tokens; added joint vision-language exposure is not given as one total | inherited text exposure is not multimodal exposure; [report](https://arxiv.org/abs/2403.05525) |
| DeepSeek-VL2 Tiny / Small / 27B | per model: 2B alignment + about 796.5–808.9B joint pretraining + about 19.5–20B SFT | total stage exposure is about 818–831B, but includes different objectives and model-specific values; [report](https://arxiv.org/abs/2412.10302) |
| Janus / JanusFlow / Janus-Pro | training steps and mixture ratios are disclosed; a directly comparable token total is not | image-code positions and text tokens are mixed; [Janus repository](https://github.com/deepseek-ai/Janus) |
| DeepSeek-Prover V1 / V1.5 | V1.5 reports 9B SFT tokens; broad base exposure is inherited | 9B is formal-proof post-training, not fresh base pretraining; [V1.5 report](https://arxiv.org/abs/2408.08152) |
| DeepSeek-Prover-V2 | based on V3-Base for 671B and Prover-V1.5 for 7B; added broad-token total unknown | cold start and RL are disclosed at a high level, not as pretraining tokens; [repository](https://github.com/deepseek-ai/DeepSeek-Prover-V2) |
| DeepSeekMath-V2 | built on V3.2-Exp-Base; added amount unknown | verifier/generator post-training must not be added to the base ledger without a disclosed token total; [repository](https://github.com/deepseek-ai/DeepSeek-Math-V2) |
| DeepSeek-OCR / OCR 2 | exact total unknown | reported throughput or visual compression ratio is not a training-token count; [OCR](https://github.com/deepseek-ai/DeepSeek-OCR), [OCR 2](https://github.com/deepseek-ai/DeepSeek-OCR-2) |

## Moonshot AI / Kimi

| Generation or branch | Training-token disclosure | Accounting note | Evidence |
|---|---:|---|---|
| `moonshot-v1` | unknown | product/API launch disclosed context, not training scale | [U] [official post](https://platform.kimi.com/blog/posts/kimi-latest) |
| Kimi k1.5 | unknown broad pretraining | report is detailed about long-context RL and data operations but withholds the foundation-model total | [U] [report](https://arxiv.org/abs/2501.12599) |
| Moonlight | 5.7T | fresh 16B-class MoE research run; released checkpoint includes the 5.2–5.7T cooldown | [D] [report](https://arxiv.org/abs/2502.16982) |
| Kimi-VL | 4.4T-stage program after selecting Moonlight at 5.2T | 2T + 0.1T trains the standalone vision encoder; joint language-backbone stages are 1.4T + 0.6T + 0.3T = 2.3T | [D] [report](https://arxiv.org/abs/2504.07491) |
| Kimi K2 | 15.5T | fresh 1.04T-total / 32.6B-active MoE pretraining exposure | [D] [report](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) |
| K2-Instruct-0905 / K2 Thinking / K2.6 / K2.7 Code | unknown added broad amount | capability and post-training releases do not disclose a replacement base total | [U] [K2 repository](https://github.com/MoonshotAI/Kimi-K2) |
| Kimi Linear | 1.4T first stage; released checkpoint through 5.7T | separate research architecture, not a K2 or K3 token-total disclosure | [D] [report](https://arxiv.org/abs/2510.26692) |
| Kimi K2.5 | about 15T additional joint mixed tokens | starts near the end of K2; **[I]** cumulative K2-line exposure is roughly 30T, not 30T unique data | [D/I] [report](https://arxiv.org/abs/2602.02276) |
| Kimi K2.5 vision stage | about 1T caption-style tokens | trains/aligned MoonViT with Moonlight before the joint K2.5 stage; do not silently add it to language-backbone exposure | [D] [K2.5 report](https://arxiv.org/abs/2602.02276) |
| Kimi K3 | unknown | announced 2026-07-17; training report and exact exposure were pending at the cutoff | [U] [official announcement](https://www.kimi.com/blog/kimi-k3) |

K2.5 illustrates why “latest model token count” can be misleading. Saying
“K2.5 used 15T” is technically incomplete: that is a continued-pretraining
stage on a near-final K2 backbone, not a from-scratch total.

## Zhipu AI / Z.ai GLM

| Generation | Training-token disclosure | Accounting note | Evidence |
|---|---:|---|---|
| GLM-130B | 400B | about half Chinese and half English | [D] [report](https://arxiv.org/abs/2210.02414) |
| ChatGLM-6B | about 1T | bilingual base exposure | [D/C] [family report](https://arxiv.org/abs/2406.12793), [repository](https://github.com/THUDM/ChatGLM-6B) |
| ChatGLM2-6B | 1.4T | trained from scratch | [D] [repository](https://github.com/THUDM/ChatGLM2-6B) |
| ChatGLM3-6B | unknown | architecture and agent protocol are public; new training exposure is not | [U] [repository](https://github.com/THUDM/ChatGLM3) |
| proprietary GLM-4 | about 10T | family report; exact flagship parameter count withheld | [D] [report](https://arxiv.org/abs/2406.12793) |
| GLM-4-9B | 10T | open 9B branch | [D] [GLM-4 repository](https://github.com/zai-org/GLM-4) |
| GLM-4-32B-0414 | 15T | includes synthetic reasoning material | [D] [GLM-4 repository](https://github.com/zai-org/GLM-4) |
| GLM-Z1 family | unknown added broad amount | reasoning/post-training derivative; do not count it as a fresh 15T run | [U] [GLM-4 repository](https://github.com/zai-org/GLM-4) |
| GLM-4.5 / 4.5-Air | headline 23T; components total about 23.1T | 15T general + 7T code/reasoning + 500B repository code + 500B synthetic reasoning + 100B long/agent | [D] [report](https://arxiv.org/abs/2508.06471) |
| GLM-4.6 / 4.7 | unknown added broad amount | later model updates disclose capabilities and post-training, not a replacement pretraining total | [U] [GLM-4.5 repository](https://github.com/zai-org/GLM-4.5) |
| GLM-5 | 27T + 1T + 500B + 50B = 28.55T | 27T main; then 32K, 128K, and 200K stages; vendor headline rounds to 28.5T | [D] [report](https://arxiv.org/abs/2602.15763) |
| GLM-5.1 / 5.2 | unknown added broad amount | later post-training and agent releases; no exact replacement total | [U] [GLM case study](agentic-rl/case-studies/glm.md) |

The apparent 23T versus 23.1T discrepancy for GLM-4.5 is rounding, not evidence
of an extra hidden stage. Preserve both the headline and component sum.

## Alibaba Qwen

| Generation or branch | Training-token disclosure | Accounting note | Evidence |
|---|---:|---|---|
| Qwen 1.8B | 2.2T | model-size-specific fresh training | [D] [official Qwen table](https://qwenlm.github.io/blog/qwen/) |
| Qwen 7B | 2.4T | model-size-specific fresh training | [D] [official Qwen table](https://qwenlm.github.io/blog/qwen/) |
| Qwen 14B / 72B | 3.0T | model-size-specific fresh training | [D] [official Qwen table](https://qwenlm.github.io/blog/qwen/) |
| Qwen1.5 | exact family-wide total not stated in its launch post | later secondary summaries often collapse this to about 3T, but the release does not provide a clean per-size ledger | [U] [official launch](https://qwenlm.github.io/blog/qwen1.5/) |
| CodeQwen1.5-7B | about 3T code-related tokens | specialist code run, not the general Qwen1.5 total | [D] [official launch](https://qwenlm.github.io/blog/codeqwen1.5/) |
| Qwen2 | about 7T | later Qwen2.5 report identifies the previous generation as 7T | [D] [Qwen2.5 report](https://arxiv.org/abs/2412.15115) |
| Qwen2.5 | up to 18T | general family pretraining exposure | [D] [report](https://arxiv.org/abs/2412.15115) |
| Qwen2.5-Coder | +5.5T CPT | continues from Qwen2.5 architecture; not a separate from-scratch 5.5T total | [D] [report](https://arxiv.org/abs/2409.12186) |
| QwQ / Qwen2.5-Math post-training | unknown new broad amount | reasoning and math derivatives inherit Qwen2.5-family bases | [U] [Qwen2.5 family launch](https://qwenlm.github.io/blog/qwen2.5/) |
| Qwen3 | about 36T | >30T general + about 5T high-quality reasoning/code/STEM + hundreds of billions long context | [D] [official launch](https://qwenlm.github.io/blog/qwen3/) |
| Qwen3-Next | 15T | separate architecture run on a uniformly sampled subset of the Qwen3 corpus | [D] [official release](https://qwen.ai/blog?id=qwen3-next) |
| Qwen3-Coder | 7.5T | specialist pretraining, about 70% code | [D] [official launch](https://qwenlm.github.io/blog/qwen3-coder/) |
| Qwen3-Coder-Next | “trillions” of CPT; about 600B repository-level slice | exact full continued-pretraining exposure unknown | [D/U] [technical report](https://github.com/QwenLM/Qwen3-Coder/blob/main/qwen3_coder_next_tech_report.pdf) |
| Qwen3-Max | 36T | proprietary trillion-parameter flagship run | [D] [official release](https://qwen.ai/blog?id=qwen3-max) |
| Qwen3.5 | “tens of trillions” | exact exposure unknown | [D/U] [official release](https://qwen.ai/blog?id=qwen3.5) |
| Qwen3.6 / 3.7 | unknown added broad amount | model releases do not provide a replacement exact total | [U] [Qwen lineage case study](agentic-rl/case-studies/qwen-meta-mistral.md) |

“Up to 18T” in the Qwen2.5 launch is a family headline. It should not be
silently converted into a guarantee that every size saw an identical schedule
unless its model card says so.

## Other Chinese model lineages

This section provides representative primary-source anchors. It is not a claim
that every commercial checkpoint from each vendor has a public recipe.

| Lineage | Model | Training-token disclosure | Evidence and boundary |
|---|---|---:|---|
| Baichuan | Baichuan-7B | 1.2T | [model report](https://arxiv.org/abs/2309.10305) |
| Baichuan | Baichuan 2 7B / 13B | 2.6T | [report](https://arxiv.org/abs/2309.10305) |
| Baichuan | Baichuan-M1 | 20T | medical/general model trained from scratch; [report](https://arxiv.org/abs/2502.12671) |
| 01.AI Yi | Yi 6B / 34B | about 3T processed; 3.1T constructed corpus | corpus size and exposure are close but not identical labels; [report](https://arxiv.org/abs/2403.04652) |
| 01.AI Yi | Yi-1.5 | +500B CPT | continued training, not a fresh 500B run; [repository](https://github.com/01-ai/Yi-1.5) |
| InternLM | InternLM2 family | about 2.0–2.6T depending on size | model-size-specific totals; [report](https://arxiv.org/abs/2403.17297) |
| InternLM | InternLM3-8B | 4T | [official repository](https://github.com/InternLM/InternLM) |
| Tencent Hunyuan | Hunyuan-Large | 7T, including about 1.5T synthetic | [report](https://arxiv.org/abs/2411.02265) |
| Tencent Hunyuan | Hunyuan-A13B | 20T main + 300B annealing | long-context stage is additional but not fully quantified; [technical report](https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf) |
| MiniMax | MiniMax-Text-01 | **[I]** about 11.758T from disclosed stages | 7.2T + 3.2T + 1T + 300B + 32B + 26B; report does not headline the sum; [report](https://arxiv.org/abs/2501.08313) |
| MiniMax | MiniMax-VL-01 | +512B vision-language tokens | continued multimodal training; [report](https://arxiv.org/abs/2501.08313) |
| Xiaomi MiMo | MiMo-7B | 25T | fresh 7B reasoning-oriented base run; [report](https://arxiv.org/abs/2505.07608) |
| Xiaomi MiMo | MiMo-VL | 2.4T multimodal pretraining | separate visual-language program; [report](https://arxiv.org/abs/2506.03569) |
| Skywork | Skywork-13B | more than 3.2T | bilingual base run; [report](https://arxiv.org/abs/2310.19341) |
| Baidu ERNIE | ERNIE 4.5 / 5.0 | exact token total unknown | architecture and multimodal training objectives are disclosed, but not a comparable exposure total; [ERNIE 5.0 report](https://arxiv.org/abs/2602.04705) |
| ByteDance Seed | Seed flagship generations | exact broad total unknown | papers disclose selected post-training/system recipes, not a full base-token ledger |
| StepFun | Step series | exact broad total unknown | public capability/model releases do not establish an auditable family-wide total |

The MiniMax sum is a useful but fragile derived value: it assumes the listed
training stages are sequential and non-overlapping. It must remain **[I]**, not
be repeated as a vendor-claimed headline.

## International reference points

| Vendor | Generation | Training-token disclosure | Evidence and boundary |
|---|---|---:|---|
| OpenAI | GPT-3 | 300B | [paper](https://arxiv.org/abs/2005.14165) |
| OpenAI | GPT-3.5, GPT-4, GPT-4o, o-series, GPT-5 family | unknown | no exact comparable base-pretraining total in public model reports |
| Meta | Llama 1 7B / 13B | 1T | [Llama paper](https://arxiv.org/abs/2302.13971) |
| Meta | Llama 1 33B / 65B | 1.4T | [Llama paper](https://arxiv.org/abs/2302.13971) |
| Meta | Llama 2 | 2T | all released sizes; [report](https://arxiv.org/abs/2307.09288) |
| Meta | Llama 3.1 405B | 15.6T | [report](https://arxiv.org/abs/2407.21783) |
| Meta | Llama 3.3 70B | more than 15T | [model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md) |
| Meta | Llama 4 Scout / Maverick | about 40T / 22T | model-specific totals; [model card](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md) |
| Google | Gemma 1 2B / 7B | 3T / 6T | [report](https://arxiv.org/abs/2403.08295) |
| Google | Gemma 2 2B / 9B / 27B | 2T / 8T / 13T | [report](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf) |
| Google | Gemma 3 1B / 4B / 12B / 27B | 2T / 4T / 12T / 14T | [official model card](https://ai.google.dev/gemma/docs/core/model_card_3) |
| Google | Gemini generations | unknown exact total | corpus categories and safety/evaluation evidence do not expose a comparable token count |
| Mistral AI | Mistral, Mixtral, Large, Small, Medium, Magistral, Devstral | generally unknown | open weights/configurations do not reveal exact broad-pretraining exposure |
| Anthropic | Claude generations | unknown | no exact comparable base-pretraining total |
| xAI | Grok generations | unknown | no exact comparable base-pretraining total |

## What the numbers do and do not imply

1. **More tokens is not automatically more unique data.** K2's semantic
   rewriting, GLM/Qwen synthetic data, and repeated high-quality subsets can
   increase processed exposure without increasing source-document count by the
   same factor.
2. **A continuation is not a fresh model total.** K2.5's about 15T, Coder-V2's
   6T, Qwen2.5-Coder's 5.5T, and V3.1's 839B all begin from existing
   checkpoints.
3. **Context length is a different axis.** A one-million-token context says
   how much one request can hold, not how many tokens trained the model.
4. **Model-size rows can differ.** Qwen 1 and Gemma publish different exposure
   by parameter size; family headlines should not erase those distinctions.
5. **Tokens do not determine compute alone.** Dense versus sparse MoE
   activation, sequence length, multimodal encoders, MTP, optimizer,
   checkpointing, hardware utilization, and failed runs all affect cost.
6. **Unknown is a result.** For Claude, Gemini, GPT-4+, Mistral flagships,
   ERNIE, Seed, Step, and many later update checkpoints, an exact public total
   does not exist. Third-party estimates should not be promoted to disclosures.

## Audit checklist

Before copying any number from this page, retain:

- exact model/checkpoint name;
- fresh pretraining versus CPT versus post-training;
- text-only versus mixed/visual positions;
- corpus size versus processed exposure;
- disclosed versus calculated/inferred status;
- tokenizer and model-size differences; and
- source date and report version.

Without those fields, a clean-looking comparison table becomes less accurate
than leaving the value unknown.
