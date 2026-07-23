# DeepSeek Release Timeline

**Verified through:** 2026-07-23. This timeline covers named first-party
DeepSeek model families, open checkpoints, and material API model updates. It
excludes third-party quantizations, community conversions, serving-framework
ports, kernels, infrastructure projects, and ordinary bug fixes.

Read this page together with the
[model training token ledger](model-training-token-ledger.md). A release date
does not disclose a training recipe, and a product alias does not necessarily
identify a new checkpoint.

## Dating and scope rules

DeepSeek's early repositories do not all contain a formal release log. This
page therefore labels the event type:

- **release:** an official first-party announcement or repository news entry;
- **weights:** the official registry exposes the checkpoint;
- **report:** the first technical report became public;
- **API update:** a hosted alias changed its backing model; and
- **temporary variant:** a time-limited endpoint or mode, not a new base family.

For early weights without an announcement date, the date comes from the
official Hugging Face registry's `createdAt` metadata. That is artifact evidence
of availability, not a claim about a press launch. Where a report and weights
appeared on different dates, both are shown.

## Main text and reasoning lineage

| Date | Model or event | Event type | Training-token status | What changed |
|---|---|---|---|---|
| 2023-11-29 | DeepSeek-LLM 7B / 67B | weights/repository | 2T [D] | dense bilingual base/chat family; [registry](https://huggingface.co/api/models/deepseek-ai/deepseek-llm-67b-base), [report](https://arxiv.org/abs/2401.02954) |
| 2024-01-08–09 | DeepSeekMoE 16B Base / Chat | weights | 2T [D] | first public fine-grained/shared-expert MoE family; base and chat registry records appeared on consecutive days; [base registry](https://huggingface.co/api/models/deepseek-ai/deepseek-moe-16b-base), [repository/report](https://github.com/deepseek-ai/DeepSeek-MoE) |
| 2024-05-06 | DeepSeek-V2 | release | 8.1T [D] | 236B-total / 21B-active MoE, Multi-head Latent Attention (MLA), 128K; [official repository news](https://github.com/deepseek-ai/DeepSeek-V2) |
| 2024-05-16 | DeepSeek-V2-Lite | release | 5.7T [D] | 16B-total / 2.4B-active smaller V2 family; [official repository news](https://github.com/deepseek-ai/DeepSeek-V2) |
| 2024-05-17 | DeepSeek-V2-0517 | API update | inherits V2 | first dated V2 hosted update in the current change log; [change log](https://api-docs.deepseek.com/updates/) |
| 2024-06-28 | DeepSeek-V2-0628 | API update | added amount unknown [U] | improved reasoning and role-play behavior; [change log](https://api-docs.deepseek.com/updates/) |
| 2024-09-05 | DeepSeek-V2.5 | release/API update | added amount unknown [U] | combines V2-0628 conversational and Coder-V2-0724 coding capabilities; [announcement](https://api-docs.deepseek.com/news/news0905/) |
| 2024-11-20 | DeepSeek-R1-Lite-Preview | web-only preview | training exposure unknown [U] | first public reasoning preview with visible reasoning and inference scaling; no weights or API were released; [official announcement](https://api-docs.deepseek.com/news/news1120) |
| 2024-12-10 | DeepSeek-V2.5-1210 | API update | added amount unknown [U] | math, coding, writing, file and webpage handling update; [announcement](https://api-docs.deepseek.com/news/news1210/) |
| 2024-12-26 | DeepSeek-V3 | release | 14.8T [D] | 671B-total / 37B-active MoE, FP8 training, Multi-Token Prediction (MTP), loss-free routing; [repository](https://github.com/deepseek-ai/DeepSeek-V3), [report](https://arxiv.org/abs/2412.19437) |
| 2025-01-20 | DeepSeek-R1-Zero and DeepSeek-R1 | release | inherit V3-Base; new broad total unknown [U] | no-SFT RL experiment plus production cold-start/SFT/RL pipeline; [official release](https://api-docs.deepseek.com/news/news250120) |
| 2025-01-20 | six DeepSeek-R1-Distill checkpoints | same-day release | post-training only | Qwen2.5- and Llama-based 1.5B, 7B, 8B, 14B, 32B, and 70B students; [repository](https://github.com/deepseek-ai/DeepSeek-R1) |
| 2025-03-24 | DeepSeek-V3-0324 | release/API update | added amount unknown [U] | post-training refresh for reasoning, code, writing, and function calling; [change log](https://api-docs.deepseek.com/updates/) |
| 2025-05-28 | DeepSeek-R1-0528 | release/API update | added amount unknown [U] | additional post-training compute and stronger reasoning; [change log](https://api-docs.deepseek.com/updates/) |
| 2025-08-21 | DeepSeek-V3.1 | release/API update | V3.1-Base adds about 839B CPT [D] | hybrid thinking/non-thinking, tool-agent training, 128K; [change log](https://api-docs.deepseek.com/updates/), [base model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base) |
| 2025-09-22 | DeepSeek-V3.1-Terminus | release/API update | added amount unknown [U] | language consistency and agent behavior update; [change log](https://api-docs.deepseek.com/updates/) |
| 2025-09-29 | DeepSeek-V3.2-Exp | release/API update | about 945.8B sparse conversion exposure [D] | experimental DeepSeek Sparse Attention (DSA); [repository/report](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| 2025-12-01 | DeepSeek-V3.2 | release | added broad amount unknown [U] | production sparse-attention and agent/reasoning model; [official transparency page](https://www.deepseek.com/en/transparency/), [report](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf) |
| 2025-12-01 | DeepSeek-V3.2-Speciale | temporary variant | not a separate base count | high-compute reasoning endpoint released alongside V3.2 for a limited period; [change log](https://api-docs.deepseek.com/updates/) |
| 2026-04-24 | DeepSeek-V4-Flash / V4-Pro | preview release | 32T / 33T [D] | million-token context; new attention, hyper-connection, optimizer, post-training, and agent systems; [official release](https://api-docs.deepseek.com/news/news260424/), [transparency page](https://www.deepseek.com/en/transparency/), [immutable report snapshot](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/a7aaed80dd2df27620eb534454253ea25eb11c7a/DeepSeek_V4.pdf) |

### Mainline interpretation

```text
DeepSeek-LLM
  -> DeepSeekMoE -> V2 / V2-Lite
  -> V2.5
      -> R1-Lite-Preview
  -> V3
      -> R1-Zero / R1 / R1-Distill
      -> V3-0324
      -> V3.1 -> V3.1-Terminus -> V3.2-Exp -> V3.2
  -> V4-Flash / V4-Pro
```

This is a capability and checkpoint lineage, not a statement that every arrow
is ordinary continued pretraining. V2.5 is a capability combination with an
undisclosed merge/training recipe; R1 is post-training over V3-Base; V4 is a
new reported pretraining program.

## Code, mathematics, and theorem proving

| Date | Model or event | Event type | Token/accounting status | Evidence |
|---|---|---|---|---|
| 2023-10-28 | DeepSeek-Coder v1 weights | weights | 2T fresh code-model training [D] | [registry](https://huggingface.co/api/models/deepseek-ai/deepseek-coder-33b-base), [repository](https://github.com/deepseek-ai/DeepSeek-Coder) |
| 2024-01-25 | DeepSeek-Coder paper and Coder-v1.5 weights | report/weights | v1 is 2T; exact v1.5 lineage total is not cleanly restated | [paper](https://arxiv.org/abs/2401.14196), [v1.5 registry](https://huggingface.co/api/models/deepseek-ai/deepseek-coder-7b-base-v1.5) |
| 2024-02-05 | DeepSeekMath 7B | release/weights/report | +500B CPT over Coder-v1.5 [D] | [repository](https://github.com/deepseek-ai/DeepSeek-Math), [report](https://arxiv.org/abs/2402.03300) |
| 2024-05-23 | DeepSeek-Prover V1 report | report | base inherited; added broad total unknown [U] | [report](https://arxiv.org/abs/2405.14333) |
| 2024-06-14 | DeepSeek-Coder-V2-0614 | API update/weights | V2 intermediate 4.2T + 6T CPT = 10.2T branch [D] | [change log](https://api-docs.deepseek.com/updates/), [registry](https://huggingface.co/api/models/deepseek-ai/DeepSeek-Coder-V2-Base) |
| 2024-06-17 | DeepSeek-Coder-V2 report | report | same 10.2T branch | [report](https://arxiv.org/abs/2406.11931) |
| 2024-07-24 | DeepSeek-Coder-V2-0724 | API update | added amount unknown [U] | [change log](https://api-docs.deepseek.com/updates/) |
| 2024-08-15 | DeepSeek-Prover-V1.5 | weights/repository/report | 9B formal-proof SFT tokens; inherited base | [repository](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5), [report](https://arxiv.org/abs/2408.08152) |
| 2024-08-16 | DeepSeek-Prover V1 weights | weights | inherited base; added broad total unknown [U] | [registry](https://huggingface.co/api/models/deepseek-ai/DeepSeek-Prover-V1) |
| 2025-04-30 | DeepSeek-Prover-V2 7B / 671B | weights/repository/report | based on Prover-V1.5 / V3-Base; added broad total unknown [U] | [repository](https://github.com/deepseek-ai/DeepSeek-Prover-V2), [report](https://arxiv.org/abs/2504.21801) |
| 2025-11-27 | DeepSeekMath-V2 | weights/repository/report | based on V3.2-Exp-Base; added amount unknown [U] | [repository](https://github.com/deepseek-ai/DeepSeek-Math-V2), [report](https://arxiv.org/abs/2511.22570) |

DeepSeek-R1-Distill is not placed in this table a second time: all six students
were one coordinated 2025-01-20 reasoning release and appear in the mainline
table.

## Vision-language, unified generation, and OCR

| Date | Model or event | Event type | Token/accounting status | Evidence |
|---|---|---|---|---|
| 2024-03-11 | DeepSeek-VL 1.3B / 7B | official family release | starts from about 500B / 2T text checkpoints; added joint total unknown | [repository news](https://github.com/deepseek-ai/DeepSeek-VL), [report](https://arxiv.org/abs/2403.05525) |
| 2024-10-18 | Janus 1.3B | weights/repository | exact comparable total unknown [U] | [registry](https://huggingface.co/api/models/deepseek-ai/Janus-1.3B), [repository](https://github.com/deepseek-ai/Janus) |
| 2024-11-13 | JanusFlow 1.3B | official repository news | exact comparable total unknown [U] | [Janus news](https://github.com/deepseek-ai/Janus), [report](https://arxiv.org/abs/2411.07975) |
| 2024-12-13 | DeepSeek-VL2 Tiny / Small / 27B | official family release | about 818–831B across alignment, joint pretraining, and SFT depending on size | [repository news](https://github.com/deepseek-ai/DeepSeek-VL2), [report](https://arxiv.org/abs/2412.10302) |
| 2025-01-27 | Janus-Pro 1B / 7B | official repository news | training steps/mixtures disclosed; comparable token total unknown [U] | [Janus news](https://github.com/deepseek-ai/Janus), [report](https://arxiv.org/abs/2501.17811) |
| 2025-10-20 | DeepSeek-OCR | official release | exact total unknown [U] | [repository release log](https://github.com/deepseek-ai/DeepSeek-OCR), [report](https://arxiv.org/abs/2510.18234) |
| 2026-01-27 | DeepSeek-OCR 2 | official release | exact total unknown [U] | [OCR release log](https://github.com/deepseek-ai/DeepSeek-OCR), [OCR 2 repository](https://github.com/deepseek-ai/DeepSeek-OCR-2), [report](https://arxiv.org/abs/2601.20552) |

These branches should not be forced into the V2 → V3 → V4 text-model chain.
DeepSeek-VL and VL2 understand images; Janus adds image generation; OCR models
specialize in document compression and recognition. Shared language backbones
do not make their training exposures directly comparable.

## Product aliases and names that are not separate generations

| Name | Correct interpretation at the cutoff |
|---|---|
| `deepseek-chat` | backward-compatible API alias whose backing checkpoint changes; the alias itself is not one stable model generation |
| `deepseek-reasoner` | reasoning-mode API alias; it has pointed to R1/V3-family and later hybrid-thinking behavior depending on release date |
| DeepSeek-R1-Lite-Preview | web-only preview that preceded R1; no open checkpoint or exact training-token total was disclosed |
| DeepSeek-V4-Pro-Max | maximum reasoning-effort mode of V4-Pro, not a separately disclosed checkpoint or pretraining run |
| V3.2-Speciale | temporary high-compute reasoning variant, not a new base family |
| R1-Distill-Qwen/Llama | six fine-tuned student checkpoints released together; their foundation pretraining belongs to Qwen/Llama, not DeepSeek |
| `Base`, `Chat`, `Instruct`, `RL`, `Zero` | training-stage or behavior suffixes; only count as a separate generation when the source reports a distinct checkpoint and recipe |
| `BF16`, `FP8`, `INT4`, `AWQ`, `GPTQ`, `GGUF` | precision or packaging variants; a conversion is not a new trained model unless the vendor reports additional training |

The DeepSeek change log should be read as a dated routing ledger. Calling
`deepseek-chat` in May 2024 and April 2026 does not imply that the same weights
served both requests.

## Condensed chronology

```text
2023  Coder v1 -- DeepSeek-LLM
2024  MoE -- Coder-v1.5 -- Math -- VL -- V2/Lite -- Prover --
      Coder-V2 -- Prover-V1.5 -- V2.5 -- Janus -- JanusFlow --
      VL2 -- V3
2025  R1/R1-Zero/Distill -- Janus-Pro -- V3-0324 -- Prover-V2 --
      R1-0528 -- V3.1 -- Terminus -- V3.2-Exp -- OCR --
      Math-V2 -- V3.2/Speciale
2026  OCR 2 -- V4-Flash/V4-Pro
```

## Remaining unknowns

1. Exact first-party announcement dates for some early weights whose official
   repositories provide no dated news item.
2. Added training exposure for V2.5, dated V2/V3/R1 updates, production V3.2,
   and most specialist post-training branches.
3. Unique-token counts, complete corpus identities and licenses, and sampling
   repetition for every flagship.
4. Whether all live product surfaces route identically at every moment; API
   aliases can change without creating a new open checkpoint.
5. Full pretraining, ablation, failed-run, and post-training compute for most
   generations.

An unknown field is intentionally retained rather than reconstructed from
model names, benchmark behavior, or community packaging.
