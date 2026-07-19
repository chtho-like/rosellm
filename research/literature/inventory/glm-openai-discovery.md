# GLM / Zhipu AI and OpenAI discovery ledger

Snapshot date: **2026-07-19** (`Asia/Shanghai`). This ledger describes the
public, reproducible universe used for `glm.jsonl` and `openai.jsonl`. It is a
coverage claim over public sources, not a claim that private, unpublished, or
silently removed work can be enumerated.

No paper PDF was downloaded while producing these two inventories. Direct PDF
URLs were recorded and, where noted below, checked with metadata/HEAD requests.
Primary HTML records, official indexes, sitemaps, RSS, repository READMEs, and
bibliographic metadata were inspected.

## Final inventory counts

| Organization | Records | `core` | `direct` | `affiliated` | PDF URL | No PDF URL | arXiv | DOI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zhipu AI / Z.ai | 95 | 26 | 39 | 30 | 51 | 44 | 44 | 35 |
| OpenAI | 657 | 79 | 456 | 122 | 178 | 479 | 89 | 227 |
| **Total** | **752** | **105** | **495** | **152** | **229** | **523** | **133** | **262** |

Type distribution:

| Organization | Type counts |
|---|---|
| Zhipu AI / Z.ai | `research_paper` 54; `model_card` 21; `technical_report` 10; `benchmark` 6; `blog_with_report` 2; `other` 2 |
| OpenAI | `blog_with_report` 366; `research_paper` 236; `system_card` 34; `benchmark` 9; `technical_report` 9; `dataset` 2; `model_card` 1 |

The high no-PDF count is intentional. A first-party web-native research page,
system card, incident report, model card, or safety report remains an inventory
work even when the publisher does not expose a separate PDF.

## OpenAI: official-source universe

### Sitemap and research taxonomy

The official [OpenAI sitemap](https://openai.com/sitemap.xml) was expanded into
the seven research-relevant child classifications below. Counts are counts in
each child feed before unioning, so overlaps are expected:

| Child classification | Items |
|---|---:|
| `publication` | 192 |
| `research` | 43 |
| `safety` | 103 |
| `milestone` | 36 |
| `conclusion` | 21 |
| `release` | 78 |
| `engineering` | 18 |

Their normalized URL union contained **406** candidate pages. Twelve were not
works and were excluded: nine pure index/filter landing pages, the GPT-4o and
o1 external-tester acknowledgement subpages, and the employee raising-concerns
policy page. The remaining **394** official pages are represented in the final
inventory. The seven candidates that did not join to RSS were retained with a
`null` date rather than an inferred date.

Evidence quality is explicitly separated:

- **387/394** selected pages had title/date metadata joined from OpenAI's
  official RSS (`rss_metadata_only` in `page-evidence.jsonl`).
- **7/394** were sitemap-only (`rss_unmatched`); their titles were recoverable
  from the official URL/page identity, but dates were left unknown.
- Direct automated HTML fetching from `openai.com` returned an access challenge
  to the corpus scanner. These 394 records therefore use the first-party URL
  plus sitemap/RSS metadata, not an archived HTML-body claim. The limitation is
  recorded per item.

### Complete official RSS and omitted-category recovery

The complete [OpenAI RSS feed](https://openai.com/news/rss.xml) contained
**1,039** items spanning 2015-2026. It was not treated as 1,039 research papers:
customer stories, commercial announcements, events, and unrelated corporate
posts were excluded. After normalized-URL subtraction against the 406 sitemap
candidates, **38** additional first-party work pages were admitted. The
selection covered:

- every outside-sitemap item categorized `Research`, `Publication`,
  `Safety & Alignment`, `Safety`, `Security`, `Engineering`, or `Release`; and
- explicit incident/follow-up patterns needed for a complete safety history:
  malicious-use disruption reports, the GPT-4o sycophancy rollback and deeper
  follow-up, the March 2023 outage, the Mixpanel incident, Codex Security,
  DALL-E 2 research-preview follow-up, OpenAI Five Benchmark, GPT-5 medical
  research, and national/cyber-security research reports.

This prevents the sitemap research filters from silently dropping incident
reports or later follow-ups merely because OpenAI categorized them as Product,
Company, Global Affairs, or Security.

### System/model cards, preparedness, and PDF evidence

Cards are separate works when the card is distinct from a launch page. The
official [OpenAI Publication index](https://openai.com/research/index/publication/)
and [Deployment Safety Hub](https://deploymentsafety.openai.com/) were used to
cover system cards. In particular, dedicated **GPT-5.6** and **GPT-Live** card
records were added in addition to their launch pages.

Verified first-party PDF routes include, among others:

- [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf)
- [GPT-4 system card](https://cdn.openai.com/papers/gpt-4-system-card.pdf)
- [GPT-4V system card](https://cdn.openai.com/papers/GPTV_System_Card.pdf)
- [GPT-4o system card](https://cdn.openai.com/gpt-4o-system-card.pdf)
- [DALL-E 3 system card](https://cdn.openai.com/papers/DALL_E_3_System_Card.pdf)
- [OpenAI o1 system card](https://cdn.openai.com/o1-system-card-20241205.pdf)
- [Deep research system card](https://cdn.openai.com/deep-research-system-card.pdf)
- [OpenAI o3 and o4-mini system card](https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf)
- [GPT-4.5 system card](https://cdn.openai.com/gpt-4-5-system-card.pdf)
- [GPT-5 system card](https://cdn.openai.com/gpt-5-system-card.pdf)
- [Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf)
- [GPT-5.2 system-card update](https://cdn.openai.com/pdf/3a4153c8-c748-4b71-8e31-aecbde944f8d/oai_5_2_system-card.pdf)
- Deployment Safety PDFs for
  [GPT-5.3-Codex](https://deploymentsafety.openai.com/gpt-5-3-codex/gpt-5-3-codex.pdf),
  [GPT-5.3 Instant](https://deploymentsafety.openai.com/gpt-5-3-instant/gpt-5-3-instant.pdf),
  [GPT-5.4 Thinking](https://deploymentsafety.openai.com/gpt-5-4-thinking/gpt-5-4-thinking.pdf),
  [GPT-5.5](https://deploymentsafety.openai.com/gpt-5-5/gpt-5-5.pdf),
  [GPT-5.5 Instant](https://deploymentsafety.openai.com/gpt-5-5-instant/gpt-5-5-instant.pdf),
  [GPT-5.6](https://deploymentsafety.openai.com/gpt-5-6/gpt-5-6.pdf), and
  [GPT-Live](https://deploymentsafety.openai.com/gpt-live/gpt-live.pdf).

A PDF URL was recorded only when a direct first-party or repository/publisher
route was available. A web-rendered system card without a separately verified
PDF remains a card with `pdf_url: null`.

### Focused official-card follow-up

A separate 64-item audit of the OpenAI and Deployment Safety sitemaps repaired
29 existing PDF fields and added six distinct artifacts that the broader page
inventory had conflated with launch pages or later versions: GPT-5.6 Preview,
GPT-Rosalind-5.5, ChatGPT Images 2.0, the September 2024 o1-preview/o1-mini
system card, Preparedness Framework Beta, and the standalone GPT-4 System
Card. The machine audit and decision log are retained in
`candidates/openai-cards-audit.jsonl` and `inventory/openai-cards-audit.md`.
All 51 non-null routes in that focused audit returned HTTP 200 with PDF content
during verification; HTML-native reports remain explicitly PDF-null.

## OpenAI: broad work-level affiliation discovery

OpenAlex was used only as a discovery and work-level affiliation index. Every
admitted record was followed to a DOI, arXiv, or publisher landing page, which
is used as the primary record. Queries (date-bounded to the snapshot) were:

```text
raw_affiliation_strings.search:OpenAI,
  from_publication_date:2015-01-01,
  to_publication_date:2026-07-19                         -> 1,231 works

institutions.id:I4210161460,
  from_publication_date:2015-01-01,
  to_publication_date:2026-07-19                         -> 1,258 works
```

The URL union contained **1,296** OpenAlex works. This apparent size is badly
inflated by false mappings: papers crediting ChatGPT/Codex as an author,
manuscript grammar-use disclosures, OpenAI Gym mentions, DALL-E image credits,
software-version deposits, and strings such as “OpenAI System” were mapped to
the company institution.

Admission therefore required all of the following:

1. a compact, affiliation-shaped raw string explicitly naming `OpenAI`, such as
   `OpenAI`, `OpenAI, San Francisco, CA, USA`, or a compact multi-institution
   line ending in OpenAI;
2. that string attached to a non-model, human-named author;
3. an eligible scholarly work type and publication date no later than the
   snapshot; and
4. a DOI, arXiv, or publisher primary record after discovery.

Long acknowledgements/disclosures, tool-use statements, model pseudonyms, and
AI-agent authors were excluded. **276** records passed the pre-dedup gate;
DOI, arXiv-location, and normalized-title deduplication left **237** inventory
works carrying explicit work-level evidence (some merge into an official
OpenAI title rather than creating a duplicate). Eighty-four arXiv identifiers
were recovered from OpenAlex location URLs and then followed to the primary
arXiv page; the index's institution mapping alone was never sufficient.

Broad mathematics, physics, law, biology, systems, and social-science papers are
`affiliated` unless their title/object directly concerns OpenAI systems or
methods. An explicit raw string is evidence of the published affiliation claim,
not an independent employment audit; unusual but primary-record-backed claims
remain visible rather than being silently rewritten.

## Zhipu AI / Z.ai / GLM: official-source universe

### Official research index

The [Zhipu AI/Z.ai Research index](https://www.zhipuai.cn/zh/research) states
that it covers research about the GLM model family. Its server-rendered payload
was paged until `hasMore=false`. The snapshot exposed **16** items. All 16 are
represented: 14 remain distinct official web/model-card entries, while the
GLM-5 release and GLM-5 technical-report paths merge into the same primary
arXiv work/provenance chain where work-level deduplication requires it.

The index includes current base-model, multimodal, agent, inference-systems,
ASR/TTS, OCR, and technical-report items. Its embedded first-party links were
followed to Z.ai blog pages, official GitHub repositories, or arXiv.

### Official repositories and primary arXiv records

The public GitHub API returned **51** repositories for
[zai-org](https://github.com/zai-org) and **130** for
[THUDM](https://github.com/THUDM); 180/181 default-branch READMEs were readable.
README citation links were followed to primary arXiv HTML records. The final
inventory contains **40** first-party-repository-linked arXiv works, spanning:

- the GLM lineage (GLM, GLM-130B, ChatGLM/GLM-4, GLM-4.5, GLM-5);
- GLM-OCR, GLM-TTS, GLM-4-Voice, GLM-V, AutoGLM/MobileRL;
- CogView/CogVLM/CogVideo/CogAgent, CodeGeeX, ImageReward/VisionReward;
- WebGLM, benchmarks, diffusion/video systems, inference/cache systems, and
  RL/post-training research.

Repository-native model cards or evaluation artifacts without a distinct
paper were retained, including ChatGLM-6B/2/3, CodeGeeX2/4, VisualGLM, GLM-Edge,
CogView4, RealVideo, GLM-SIMPLE-EVALS, and the `slime` training framework.

### THUDM boundary

`THUDM` was **not** treated as synonymous with Zhipu AI. A THUDM work was
admitted only when at least one of these held:

- the repository or work-level raw affiliation explicitly named Zhipu AI/Z.ai;
- a current first-party Z.ai repository linked it as a direct project paper; or
- it is the foundational GLM lineage paper, recorded as `core` with an explicit
  note that this lineage inclusion does not convert unrelated Tsinghua work
  into Zhipu-affiliated research.

Consequently, broad THUDM agent/RL/LLM repositories without Zhipu evidence were
excluded even when topically relevant. Conversely, `IndexCache`, `ReST-RL`,
`XDAI`, CogDL-related work, and other explicitly supported/affiliated records
were eligible under their actual evidence.

## Zhipu AI / Z.ai: broad work-level affiliation discovery

Discovery-only OpenAlex queries were:

```text
raw_affiliation_strings.search:Zhipu,
  from_publication_date:2019-01-01,
  to_publication_date:2026-07-19                         -> 51 works

raw_affiliation_strings.search:Z.ai,
  from_publication_date:2019-01-01,
  to_publication_date:2026-07-19                         -> 5 works

institutions.id:I4401726915,
  from_publication_date:2019-01-01,
  to_publication_date:2026-07-19                         -> 31 works
```

The union contained **56** works. Raw-string filtering rejected similarly named
but unrelated companies (for example Nanjing, Zhejiang, Qingxin, Ningbo, and
medical/materials entities containing “Zhipu”) and model/agent pseudo-authors.
After DOI/title deduplication, **35** final inventory works carry explicit
work-level Zhipu AI/Z.ai affiliation evidence; some are merged with a direct
repo-linked paper. Publisher/DOI/arXiv pages, not OpenAlex, are primary URLs.

## Deduplication and typing

Deduplication order was DOI, arXiv identifier/location, then Unicode-normalized
title inside an organization. When an official page and bibliographic record
had an exact normalized title, provenance and identifiers were merged into one
work. Distinct artifacts (for example a launch post and its separate system
card, or a rollback notice and its follow-up) remain distinct.

Only the validator enums are used:

```text
technical_report, research_paper, system_card, model_card,
dataset, benchmark, blog_with_report, other
```

IDs are lowercase filesystem-safe slugs. Unknown dates and PDF URLs are `null`.
All records use `retrieved_at: 2026-07-19`.

## Ambiguities and current gaps

- A public search cannot prove completeness for unpublished, private, removed,
  robots-blocked, or unindexed work. This is an auditable public-universe
  snapshot, not a metaphysical “all.”
- OpenAI HTML bodies were blocked for bulk retrieval. Sitemap/RSS metadata and
  primary URLs are marked separately from verified CDN/Deployment Safety PDFs.
- The 1,039-item OpenAI RSS is a discovery universe, not a paper count. Excluded
  marketing/customer/corporate items can be re-audited from the retained
  candidate snapshot.
- OpenAlex institution assignment is noisy, especially after 2025 because AI
  models are increasingly credited as authors. Strict raw-string gates reduce
  but cannot independently prove an author's employment claim.
- Zhipu's current Research index exposes only 16 items and does not provide a
  complete historical archive. Official GitHub READMEs and primary arXiv/DOI
  records supply the older lineage.
- Hugging Face API access timed out from this environment. Model-family cards
  linked from the official Research index and repositories are included, but a
  checkpoint-by-checkpoint mirror of every official hosting revision is not
  claimed.
- Repository transfers between `THUDM` and `zai-org` can change canonical URLs.
  Current first-party paths and the evidence basis are recorded per work.
- PDF URLs were not downloaded in this subtask. Availability is based on direct
  arXiv/publisher/CDN routes and metadata/HEAD verification where possible.

### Post-discovery public-full-text audit

The corpus download pass exposed five ACM routes that returned HTTP 403 to the
non-browser client. Exact-title primary-source follow-up recovered arXiv full
texts for P2TAG (`2407.15431`), OAG-Bench (`2402.15810`), AutoWebGLM
(`2404.03648`), and OAG-BERT (`2103.02410`). The WWW 2025 OpenReview record for
Ask, Acquire, Understand supplies a public paper route, but that endpoint also
returned HTTP 403 to the corpus client; it remains a recorded, unresolved local
artifact gap rather than being omitted.

## Validation

The two JSONL files were copied alone into a temporary inventory directory and
validated with:

```bash
.venv/bin/python scripts/literature_corpus.py \
  --inventory /tmp/rosellm-glm-openai-validate.<random> validate
```

Result after the focused card follow-up: **752 records validated; 752
selected**. Shared-inventory validation is
reported separately because it also depends on concurrently produced
organization files.
