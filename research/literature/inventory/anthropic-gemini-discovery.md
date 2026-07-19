# Anthropic and Google DeepMind / Gemini literature discovery log

Cutoff: **2026-07-19** (Asia/Shanghai). Retrieval date stored in every record: **2026-07-19**.

This log documents the evidence trail behind `anthropic.jsonl` and `gemini.jsonl`. The inventory is publication-oriented: papers, technical/model reports, system/model cards, datasets, benchmarks, and substantial first-party research disclosures are included. Product pages, navigation pages, repositories, and release notes are not separate publications unless they expose a distinct report or card.

Temporary audit downloads were used outside the repository to inspect PDF titles, dates, page counts, and exact-file hashes. Those files are not part of the tracked inventory or archive pipeline. A non-null `pdf_url` records the best public direct-file route found; some publisher or repository endpoints can still gate automated access or change later.

## Result and coverage

| Organization | Total | Core | Direct | Affiliated | arXiv ID | DOI | PDF URL recorded | No PDF URL found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anthropic | 244 | 21 | 188 | 35 | 38 | 37 | 119 | 125 |
| Google DeepMind / Gemini | 1,985 | 24 | 18 | 1,943 | 699 | 1,619 | 1,242 | 743 |
| **Combined** | **2,229** | **45** | **206** | **1,978** | **737** | **1,656** | **1,361** | **868** |

Type counts:

- Anthropic: `blog_with_report=105`, `research_paper=69`, `technical_report=42`, `system_card=16`, `model_card=1`, `other=11`.
- Google DeepMind / Gemini: `research_paper=1926`, `model_card=41`, `technical_report=17`, `dataset=1`.

PDF coverage by tier:

| Organization | Core | Direct | Affiliated |
| --- | ---: | ---: | ---: |
| Anthropic | 21 / 21 | 82 / 188 | 16 / 35 |
| Google DeepMind / Gemini | 16 / 24 | 15 / 18 | 1,211 / 1,943 |

The three tiers mean:

- `core`: first-party model documentation or model-specific risk reporting. Anthropic core consists of the 17 Claude system/model-card works plus four model-specific risk reports. Google DeepMind core consists of 23 Gemini-family model cards plus the Gemini 3 Pro Frontier Safety Framework report.
- `direct`: first-party research centered on the organization or its model family but not a core card/risk report. Every Google DeepMind `direct` record was individually checked for Gemini as a main system in the title, abstract, report, or first-party page; a mere comparison against Gemini is not sufficient.
- `affiliated`: broader work. Scholarly/technical/dataset records require explicit work-level organization evidence. Google DeepMind's 18 non-Gemini first-party model cards (Gemma, Veo, Imagen, and Lyria families) are also `affiliated`, because they are official Google DeepMind artifacts but not Gemini-core documentation.

This separation is deliberate: `gemini.jsonl` is a Google DeepMind inventory with a narrow Gemini-direct/core subset, not a claim that all 1,985 records are Gemini papers.

## Official-surface boundary and source counts

### Anthropic

The 2026-07-19 [Anthropic sitemap](https://www.anthropic.com/sitemap.xml) contained 497 URLs. Its official-surface audit resolved as follows:

| Surface | Snapshot count | Inventory decision |
| --- | ---: | --- |
| `/research/` paths | 147 | Five are team index pages; the other 142 are substantive research pages and all 142 are represented. |
| `/research/team/` pages | 5 | Navigation/team indexes, not separate publications. |
| `/engineering/` pages | 25 | Reviewed for linked reports; not admitted solely for being engineering articles. |
| System/model-card index rows | 17 | All 17 represented and all have official PDF URLs. |
| First-party Responsible Scaling Policy versions | 9 | Versions 1.0, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2, 3.3, and 3.4 are distinct records. |

The 17 historical Claude documentation works are represented as 16 `system_card` records and one `model_card` record (`Claude 2 Model Card`). Current and historical official PDF aliases/revisions are retained in `source_pages` and explained in `notes`.

The Responsible Scaling Policy sweep also retained same-version redlines as supporting sources and covered the surrounding first-party safety outputs, including the ASL-3 Deployment Safeguards Report, RSP Noncompliance Reporting and Anti-Retaliation Policy, Frontier Safety Roadmap, model-specific alignment/sabotage risk reports, and the Summer 2025 pilot sabotage-risk report and reviews.

### Google DeepMind

The 2026-07-19 [Google DeepMind sitemap](https://deepmind.google/sitemap.xml) contained 708 URLs. Its official-surface audit resolved as follows:

| Surface | Snapshot count | Inventory decision |
| --- | ---: | --- |
| `/research/publications/` paths | 260 | One index plus 259 publication leaves; all 259 leaves are represented. |
| Publications index visible total | 259 | The live page said `259 publications`, matching the 259 sitemap leaves exactly. |
| `/models/model-cards/` sitemap paths | 12 | One index plus 11 leaves; all 11 leaves are represented. |
| Live model-card index entries | 41 | All 41 linked primary card URLs are represented: 23 Gemini core cards and 18 broader Google DeepMind cards. |

The live model-card index linked 14 Google DeepMind HTML card leaves even though only 11 leaves had reached the sitemap snapshot; the remaining cards were direct Google-hosted PDFs or `ai.google.dev` model cards. The index, not the sitemap alone, therefore defines the card boundary.

An earlier interactive observation displayed 252 publications while the page was partially hydrated/paginated. It is recorded as a transient UI state, not a boundary count. The verified final equality is **259 visible publications = 259 sitemap publication leaves = 259 represented leaves**.

Google DeepMind's current `/models/gemini...` landing pages are primarily product/navigation surfaces. They were not converted into publication rows unless they linked a distinct report, paper, evaluation, or model card.

## Evidence and metadata precedence

Evidence was resolved in this order:

1. First-party research/publication pages, system/model-card indexes, policy/version indexes, and organization-hosted PDFs.
2. Publisher pages, arXiv records, DOI metadata, and public institutional/author repositories.
3. OpenAlex work records as a discovery and work-level affiliation-evidence layer.

Official title/byline/date information takes precedence for first-party reports. For scholarly works, publisher/arXiv metadata takes precedence over OpenAlex when they disagree. `date` normally records the earliest public version found; where only a publication/update month is available, the first day of that month is used. Two Google DeepMind artifacts remain undated because neither their live index nor their PDF supplies a stable publication date: the Gemini 1.5 model-card appendix and the Veo 3 technical report.

The type normalization used this mapping while retaining the original source subtype in `notes`:

| Source description | Inventory type |
| --- | --- |
| journal article, conference paper, preprint, or scholarly paper | `research_paper` |
| technical, safety, policy, or model-family report | `technical_report` |
| Anthropic research article with a substantial linked or web-native report | `blog_with_report` |
| Claude system card | `system_card` |
| Claude 2 or Google DeepMind model card | `model_card` |
| standalone released dataset | `dataset` |
| official disclosure not fitting the enumerated publication types | `other` |

IDs are normalized to lowercase `[a-z0-9._-]+`; DOI and arXiv identifiers are unversioned canonical identities; dates use `YYYY-MM-DD`; and all records retain `retrieved_at=2026-07-19`.

## Affiliation discovery and query accounting

The retained affiliation boundary is work-level, not author-history or topic based:

- All 35 Anthropic `affiliated` records contain explicit work-level raw affiliation text naming Anthropic.
- Google DeepMind has 1,925 affiliated scholarly/technical/dataset records whose inclusion is supported by explicit work-level Google DeepMind/DeepMind affiliation evidence, plus 18 broader first-party model cards admitted from the official card index.
- OpenAlex evidence is preserved in `affiliation_evidence` and the corresponding work URLs remain in `source_pages`.

The exact original OpenAlex free-text/raw-affiliation query sequence and pre-deduplication candidate counts were not preserved during the initial scrape. This audit therefore worked backward row by row from retained OpenAlex work URLs and raw affiliation strings; it does **not** invent a query count after the fact.

The live normalized institution entities were checked but were not used as inclusion totals:

- [Google DeepMind, I4210090411](https://openalex.org/I4210090411) reported 10,649 works, far broader than this cutoff inventory.
- [Anthropic, I4387930290](https://openalex.org/I4387930290) reported zero works despite many publisher/OpenAlex records with explicit raw Anthropic affiliations.

Those institution counts demonstrate why normalized-entity counts are unsuitable as publication boundaries here. The reproducible source/query counts retained by this audit are the official-surface counts above: one 497-URL Anthropic sitemap snapshot, one 708-URL Google DeepMind sitemap snapshot, 142 Anthropic research leaves, 17 Anthropic card rows, nine RSP versions, 259 Google DeepMind publication leaves, and 41 Google DeepMind model-card entries.

## Duplicate and version decisions

### Anthropic

- Exact official aliases for Claude Opus 4.5, Sonnet 3.7, Haiku 4.5, Sonnet 4.5, Opus 4.1, and Opus 4.7 were hash-checked and merged into one work record per card.
- Non-byte-identical official updates of Fable/Mythos 5, Opus 4.8, Mythos Preview, Sonnet 4.6, Opus 4.6, Claude 4, and Claude 3 remain one evolving work per card; all observed revision URLs are retained with revision notes.
- The nine named Responsible Scaling Policy versions remain distinct works. Same-version aliases/redlines are supporting sources, not extra records.
- Five scraper artifacts caused by globally linked system-card PDFs on unrelated research pages were removed; the underlying research pages remain represented by their proper web records.
- Claude 2 was corrected to `Claude 2 Model Card`, `model_card`, and `2023-07-01` from the official historical card index.

### Google DeepMind / Gemini

- Twenty-nine candidates that treated Gemini/another AI system as an author, plus one unrelated `DeepMind Lab` false positive, were removed.
- Three same-work version pairs were merged: Machine Unlearning, Don't Do What Doesn't Matter, and the Madingley ecosystem work.
- The Gemini Robotics 1.5 technical report and model card point to the same Google-hosted PDF; the model card begins on page 30. They are represented once as a `model_card`, with all report/card sources retained.
- The handle record titled `Regret bounds for kernel-based reinforcement learning` was merged into canonical arXiv `2004.05599`, whose later title is `Kernel-Based Reinforcement Learning: A Finite-Time Analysis`.
- TxGemma was moved from `direct` to `affiliated`; it is broader Google DeepMind work, not Gemini-direct.
- All 18 remaining `direct` records carry an explicit Gemini-direct rationale in `notes`.

Five genuinely co-affiliated works intentionally appear once in each organization's inventory, with organization-specific work-level evidence: `Human-AI Complementarity: A Goal for Amplified Oversight`, `The impact of advanced AI systems on democracy`, `SoK: Watermarking for AI-Generated Content`, `Learned Neural Physics Simulation for Articulated 3D Human Pose Reconstruction`, and `Report of the 1st Workshop on Generative AI and Law`. Their repeated DOI/title across the two files is cross-organization membership, not a within-organization duplicate.

## Public-full-text audit

The URL audit replaced obvious DOI/HTML landing pages and one incorrect image with direct public full-text routes where available. Examples include:

- Anthropic `anthropic-doi-10.1162-coli.a.572`: arXiv `2408.01416` PDF.
- Anthropic `anthropic-doi-10.1145-3586183.3606801`: public MIT CSAIL PDF.
- Anthropic `anthropic-doi-10.1002-pro6.1247`: PMC full-text PDF route, with the browser interstitial limitation noted.
- Google DeepMind `gdm-doi-10.2172-2475542`: direct OSTI PDF.
- RSS proceedings records: direct `roboticsproceedings.org` PDFs instead of DOI landings.
- Oxford/PMLR/arXiv/Cambridge/JMLR/author-hosted fallbacks for records whose repository handle was only an HTML landing page.
- `gdm-doi-10.1016-j.media.2016.08.008`: replaced a journal figure JPEG with the matching arXiv `1603.00275` full paper.

PLOS printable-file endpoints and EMS numeric file endpoints were retained after live checks returned `application/pdf`, even though their URLs do not end in `.pdf`.

One legacy CU Scholar dissertation route returned HTTP 403 to the automated audit, while its associated Zenodo record returned HTTP 410. The direct legacy full-text URL is retained with that limitation stated in the record; the repository landing page remains a source.

`PDF URL recorded` is therefore a metadata-coverage count, not a guarantee that every publisher will permit every future unauthenticated client to download the file.

## Explicit exclusions

- Anthropic team indexes and Google DeepMind publication/model-card index pages are discovery surfaces, not publications.
- Engineering or product pages without a distinct research artifact are excluded as standalone publications.
- A title, abstract, or bibliography that merely mentions Claude, Gemini, Anthropic, or DeepMind is insufficient for `direct` or `affiliated` inclusion.
- AI systems entered as authors/tools, biographies containing a lab name without a work-level affiliation, and the unrelated `DeepMind Lab` entity are excluded.
- Journal/conference/preprint versions and official URL aliases of one underlying work are merged rather than counted twice.
- Google DeepMind product/model landing pages are excluded unless they expose a distinct report, card, benchmark, or publication.

## Known limitations

- Both official sites are dynamic. Sitemaps can lag page indexes, and client-side hydration can temporarily expose incomplete totals; the 252-to-259 publication observation is the concrete example.
- OpenAlex affiliation indexing is incomplete and noisy. The original query/candidate ledger was not preserved, so this audit reports only the row-level evidence and official counts that can be reproduced honestly.
- Corporate/team bylines are used when a first-party report has no stable individual-author list; individual blog authors are not substituted for a report's team byline.
- Some PDF hosts use browser interstitials, subscription checks, expiring redirects, anti-bot responses, or legacy repository routes. `pdf_url` may need later refresh even when the bibliographic identity is stable.
- Counts are cutoff snapshots, not claims about releases or index updates after 2026-07-19.

## Validation

- 244 Anthropic objects and 1,985 Google DeepMind / Gemini objects parse as JSONL.
- All 2,229 records contain the required fields and valid ID, type, tier, date, URL, arXiv, and DOI shapes.
- No duplicate arXiv ID, DOI, normalized title, or organization/id pair remains within either organization's inventory. The five intentional cross-organization co-affiliated works are documented above.
- Official coverage scripts found `142/142` Anthropic research leaves, `259/259` Google DeepMind publication leaves, and `11/11` sitemap-listed Google DeepMind model-card leaves represented.
- Every one of the 41 primary model-card URLs from the live Google DeepMind card index appears in the inventory.
- Isolated validation command:

  `python scripts/literature_corpus.py --inventory <temporary-two-file-directory> validate`

  Output: `validated 2229 records; selected 2229`.

- Repository-wide validation command:

  `python scripts/literature_corpus.py validate`

  Output at handoff: `validated 3050 records; selected 3050`.
