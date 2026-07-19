# OpenAI official cards and safety-report gap audit

Audit cutoff: 2026-07-19

This is an independent, read-only comparison against `research/literature/candidates/{official-pages,openai-rss,page-evidence,inventory-page-evidence}.jsonl` and the then-current `research/literature/inventory/openai.jsonl`. It does not modify `openai.jsonl`.

## Scope and evidence

Included artifacts are work-level model/system cards, Preparedness or governance frameworks, technical safety/evaluation reports, the public malicious-use report series, and substantive first-party incident responses. Product launch marketing, generic policy commentary, recruiting pages, events, and ordinary release notes are excluded.

Primary discovery and verification sources:

- OpenAI official sitemap: <https://openai.com/sitemap.xml>
- OpenAI official RSS: <https://openai.com/news/rss.xml>
- OpenAI Deployment Safety Hub sitemap: <https://deploymentsafety.openai.com/sitemap.xml>
- Canonical OpenAI publication pages and their explicit `Read the paper`, `Read the report`, `View PDF`, or `Previous version` links
- OpenAI CDN PDFs; arXiv PDFs only where the official OpenAI page or Deployment Safety Hub explicitly links them

The Deployment Safety Hub sitemap currently emits `http://localhost:4321/...` locations. This audit normalized only the top-level card slugs to `https://deploymentsafety.openai.com/<slug>`, then verified the canonical title and downloadable artifact. It did not assume that `/<slug>/<slug>.pdf` is valid: many migrated cards redirect that guessed path to HTML, so their actual linked CDN/arXiv PDF is recorded instead.

Machine-readable evidence is in `research/literature/candidates/openai-cards-audit.jsonl`. Every line has exactly `title`, `date`, `url`, `pdf_url`, `source_url`, `status`, and `notes`.

## Result

| Status | Count | Meaning |
|---|---:|---|
| `missing_record` | 6 | A distinct official card/framework is not represented in `openai.jsonl` |
| `present_missing_pdf` | 27 | The record exists, but an explicit official/officially-linked PDF is absent from `pdf_url` |
| `present_wrong_pdf` | 2 | The record exists, but its current PDF is wrong or dead |
| `present_html_only` | 13 | The official HTML publication is the substantive report; no separate PDF was verified |
| `present_complete` | 16 | The record and its official PDF are already represented |
| **Total audited** | **64** | Focused card, framework, safety/evaluation, threat-report, and incident corpus |

## Six missing records

| Date | Missing work | Official artifact | Why it is distinct |
|---|---|---|---|
| 2026-06-26 | GPT-5.6 Preview System Card | [card](https://deploymentsafety.openai.com/gpt-5-6-preview) · [PDF](https://deploymentsafety.openai.com/gpt-5-6-preview/gpt-5-6-preview.pdf) | The inventory has a preview product item, not this safety card. |
| 2026-06-03 | GPT-Rosalind-5.5 System Card | [card](https://deploymentsafety.openai.com/gpt-rosalind-5-5) · [PDF](https://deploymentsafety.openai.com/gpt-rosalind-5-5/gpt-rosalind-5-5.pdf) | Related Rosalind launch/capability pages do not contain the standalone card record. |
| 2026-04-21 | ChatGPT Images 2.0 System Card | [card](https://deploymentsafety.openai.com/chatgpt-images-2-0) · [PDF](https://deploymentsafety.openai.com/chatgpt-images-2-0/chatgpt-images-2-0.pdf) | The sitemap/RSS snapshots contain the product announcement, not the system card. |
| 2024-09-12 | OpenAI o1-preview and o1-mini System Card | [official previous-version context](https://openai.com/index/openai-o1-system-card) · [PDF](https://cdn.openai.com/o1-preview-system-card-20240917.pdf) | The December o1 card is a later version; the original September preview card has different checkpoints and evaluations. |
| 2023-12-18 | Preparedness Framework (Beta) | [context page](https://openai.com/index/frontier-risk-and-preparedness) · [PDF](https://cdn.openai.com/openai-preparedness-framework-beta.pdf) | The inventory has the October team/challenge page, but not the subsequently published beta framework as a report. |
| 2023-03-14 | GPT-4 System Card | [PDF](https://cdn.openai.com/papers/gpt-4-system-card.pdf) | The existing GPT-4 entry is the separate Technical Report (`gpt-4.pdf`); the 60-page system card is not interchangeable with it. |

The three 2026 Deployment Safety cards are also absent as distinct works from the existing candidate snapshots. The ChatGPT Images 2.0 slug appears there only through its launch page; the GPT-5.6 Preview and GPT-Rosalind-5.5 card slugs do not appear.

## PDF repair backlog

### System and model cards/addenda (14)

- [GPT-5.2-Codex](https://cdn.openai.com/pdf/ac7c37ae-7f4c-4442-b741-2eabdeaf77e0/oai_5_2_Codex.pdf)
- [GPT-5.1-Codex-Max](https://cdn.openai.com/pdf/2a7d98b1-57e5-4147-8d0e-683894d782ae/5p1_codex_max_card_03.pdf)
- [GPT-5.1 Instant/Thinking addendum](https://cdn.openai.com/pdf/4173ec8d-1229-47db-96de-06d87147e07e/5_1_system_card.pdf)
- [GPT-5 Sensitive Conversations addendum](https://cdn.openai.com/pdf/3da476af-b937-47fb-9931-88a851620101/addendum-to-gpt-5-system-card-sensitive-conversations.pdf)
- [Sora 2](https://cdn.openai.com/pdf/50d5973c-c4ff-4c2d-986f-c72b5d0ff069/sora_2_system_card.pdf)
- [GPT-5-Codex](https://cdn.openai.com/pdf/97cc5669-7a25-4e63-b15f-5fd5bdc4d149/gpt-5-codex-system-card.pdf)
- [gpt-oss model card](https://arxiv.org/pdf/2508.10925)
- [ChatGPT Agent](https://cdn.openai.com/pdf/839e66fc-602c-48bf-81d3-b21eacc3459d/chatgpt_agent_system_card.pdf)
- [o3 Operator addendum](https://cdn.openai.com/pdf/4375e605-f9a6-438d-bcc8-190599c183a6/o3_cua_system_card.pdf)
- [o3/o4-mini Codex addendum](https://cdn.openai.com/pdf/8df7697b-c1b2-4222-be00-1fd3298f351d/codex_system_card.pdf)
- [GPT-4o native image generation addendum](https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf)
- [GPT-4.5](https://cdn.openai.com/gpt-4-5-system-card-2272025.pdf) — replace the dead `https://cdn.openai.com/gpt-4-5-system-card.pdf` URL, which currently returns 404.
- [o3-mini](https://cdn.openai.com/o3-mini-system-card-feb10.pdf)
- [Operator](https://cdn.openai.com/operator_system_card.pdf)

### Frameworks and technical safety/evaluation papers (9)

- [Frontier Governance Framework](https://cdn.openai.com/pdf/e37d949b-8c9f-4d76-b99e-4272f4631a7e/openai-frontier-governance-framework.pdf)
- [Predicting LLM Safety Before Release by Simulating Deployment](https://arxiv.org/pdf/2607.07184)
- [IH-Challenge](https://cdn.openai.com/pdf/14e541fa-7e48-4d79-9cbf-61c3cde3e263/ih-challenge-paper.pdf)
- [Reasoning Models Struggle to Control their Chains of Thought](https://arxiv.org/pdf/2603.05706)
- [Monitoring Monitorability](https://arxiv.org/pdf/2512.18311)
- [gpt-oss-safeguard technical report](https://cdn.openai.com/pdf/08b7dee4-8bc6-4955-a219-7793fb69090c/Technical_report__Research_Preview_of_gpt_oss_safeguard.pdf)
- [Estimating Worst-Case Frontier Risks of Open-Weight LLMs](https://cdn.openai.com/pdf/231bf018-659a-494d-976c-2efdfc72b652/oai_gpt-oss_Model_Safety.pdf)
- [The Instruction Hierarchy](https://arxiv.org/pdf/2404.13208)
- [Safe-Completions](https://cdn.openai.com/pdf/be60c07b-6bc2-4f54-bcee-4141e1d6c69a/gpt-5-safe_completions.pdf) — this is a provenance repair: the existing third-party journal mirror contains the same paper, but the official OpenAI page links this first-party CDN copy.

### Public malicious-use report series (6)

- [February 2026](https://cdn.openai.com/pdf/df438d70-e3fe-4a6c-a403-ff632def8f79/disrupting-malicious-uses-of-ai.pdf)
- [October 2025](https://cdn.openai.com/threat-intelligence-reports/7d662b68-952f-4dfd-a2f2-fe55b041cc4a/disrupting-malicious-uses-of-ai-october-2025.pdf)
- [June 2025](https://cdn.openai.com/threat-intelligence-reports/5f73af09-a3a3-4a55-992e-069237681620/disrupting-malicious-uses-of-ai-june-2025.pdf)
- [February 2025](https://cdn.openai.com/threat-intelligence-reports/disrupting-malicious-uses-of-our-models-february-2025-update.pdf)
- [October 2024](https://cdn.openai.com/threat-intelligence-reports/influence-and-cyber-operations-an-update_October-2024.pdf) — also fills the blank inventory date with 2024-10-09.
- [May 2024](https://cdn.openai.com/threat-intelligence-reports/threat-intel-report-may-2024.pdf)

## HTML-native reports and incidents

Thirteen entries are intentionally not called PDF gaps. They include Sora's 2024 HTML system card; the OpenAI-side Anthropic joint safety evaluation; early-warning biorisk, red-teaming, coding-agent monitoring, and evaluation-methodology publications; four malicious-use/case pages without a separate report download; and the Mixpanel, Axios, and TanStack incident responses. Their official HTML pages are already represented in `openai.jsonl`.

## Suggested ingestion order

1. Add the six missing records, preserving GPT-4 Technical Report versus GPT-4 System Card and December o1 versus September o1-preview as separate works.
2. Repair the wrong Safe-Completions PDF and dead GPT-4.5 PDF URL, then attach the 27 verified missing PDFs.
3. Fill the October 2024 threat-report date.
4. Keep HTML-native publications with `pdf_url: null`; do not synthesize guessed `/<slug>/<slug>.pdf` links.
