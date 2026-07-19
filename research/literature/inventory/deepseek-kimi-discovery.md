# DeepSeek and Moonshot AI / Kimi literature discovery log

Cutoff: **2026-07-19** (Asia/Shanghai). Retrieval date stored in every record: **2026-07-19**.

This log documents the evidence trail behind `deepseek.jsonl` and `kimi.jsonl`. The inventory is publication-oriented: papers, technical/model reports, benchmarks, and substantial first-party research disclosures are included; repositories that only ship software, weights, or short release notes are not treated as separate publications.

No PDF was downloaded during discovery or validation. `pdf_url` records a public full-text location only when one was found.

## Result and coverage

| Organization | Total | Core | Direct | Affiliated | arXiv ID | DOI | Public PDF URL | No public PDF found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek | 37 | 22 | 9 | 6 | 33 | 10 | 35 | 2 |
| Moonshot AI / Kimi | 32 | 13 | 11 | 8 | 23 | 9 | 24 | 8 |
| **Combined** | **69** | **35** | **20** | **14** | **56** | **19** | **59** | **10** |

Type counts:

- DeepSeek: `technical_report=22`, `research_paper=15`.
- Moonshot AI / Kimi: `technical_report=9`, `research_paper=14`, `benchmark=3`, `blog_with_report=6`.

The three tiers mean:

- `core`: a first-party model/report explicitly branded DeepSeek, Kimi, or Moonshot and centered on the lab's model family.
- `direct`: first-party methods, infrastructure, evaluation, or research that supports the lab's models but is not itself a flagship model report.
- `affiliated`: broader research with explicit lab affiliation for at least one author, or an official lab-hosted legacy research artifact, but not a central lab publication.

## Evidence and metadata precedence

Evidence was resolved in this order:

1. First-party laboratory research pages, official repositories, and lab-hosted report/model-card files.
2. Publisher paper pages and arXiv title pages/Atom metadata.
3. Crossref records, especially author affiliation arrays deposited by publishers.
4. OpenAlex as a discovery index only; every candidate was checked against a publisher, arXiv, or first-party page before inclusion.

Titles, author arrays, and arXiv v1 dates were populated from the live arXiv Atom API for all arXiv-bearing records. Publisher-deposited Crossref metadata supplied DOI-only titles/authors/dates and affiliation evidence. For `kimi-2407.02906` and `kimi-2506.13737`, the later AAAI author lists take precedence over earlier arXiv lists because those publication versions add the Moonshot-affiliated author. Unbylined first-party web publications use the visible corporate/team byline (`Kimi Team` or `DeepSeek-AI`).

`date` normally records the earliest public version used by this corpus: arXiv v1 when available, otherwise the first-party release/article date or publisher date. When a model/report release predates arXiv (for example Kimi K2, Kimi-Dev, Kimi K2.5, DeepSeek-V3.2, and DeepSeek-V4), the official release date is retained and the divergence is stated in `notes`.

Conference/journal versions of the same underlying work are merged into one record by arXiv ID/title, with the DOI retained. For example, the Nature version of DeepSeek-R1 is not a second record.

## Discovery ledger

### First-party laboratory surfaces

DeepSeek:

- [DeepSeek transparency center](https://www.deepseek.com/en/transparency/) — authoritative release/report provenance for DeepSeek-V3.2 and DeepSeek-V4.
- [DeepSeek Hugging Face papers](https://huggingface.co/deepseek-ai/papers) — first-party paper index (27 entries visible during retrieval).
- [DeepSeek GitHub organization](https://github.com/deepseek-ai) and `https://api.github.com/orgs/deepseek-ai/repos?per_page=100&type=public` — 35 public repositories at retrieval.
- Each repository's default-branch README was checked through `https://raw.githubusercontent.com/deepseek-ai/{repo}/{default_branch}/README.md` for arXiv IDs, report files, publisher links, and duplicate/version relationships.

Moonshot AI / Kimi:

- [Kimi research index](https://www.kimi.com/en/blog/) — 19 first-party research entries visible at retrieval, from Mooncake through Kimi K3.
- [Moonshot AI Hugging Face papers](https://huggingface.co/moonshotai/papers) — first-party paper index (14 entries visible during retrieval).
- [MoonshotAI GitHub organization](https://github.com/MoonshotAI) and `https://api.github.com/orgs/MoonshotAI/repos?per_page=100&type=public` — 38 public repositories at retrieval.
- Each repository's default-branch README was checked through `https://raw.githubusercontent.com/MoonshotAI/{repo}/{default_branch}/README.md` for arXiv IDs, report files, and publication links.
- [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/) was checked directly because it is a lab-hosted report outside arXiv.

### arXiv

Discovery searches included:

- `https://export.arxiv.org/api/query?search_query=all%3A%22DeepSeek-AI%22&start=0&max_results=200`
- `https://export.arxiv.org/api/query?search_query=all%3A%22MoonshotAI%22&start=0&max_results=200`
- exact-title follow-ups using `search_query=ti:%22{title}%22`, especially for DOI-only affiliation candidates.

Identity enrichment used batched calls of the form:

`https://export.arxiv.org/api/query?id_list={comma-separated-arxiv-ids}&max_results={batch-size}`

This recovered three exact-title preprints initially found through publisher metadata: `2411.00337` (EV charging forecasting), `2401.08281` (Faiss), and `2403.15105` (SAGraph). Whitespace was normalized, version suffixes were removed, and the spurious author token `:` was discarded if present.

Free-text occurrences of “DeepSeek”, “Kimi”, or “Moonshot” were never sufficient evidence: papers that only compare against or cite a model were excluded.

### Crossref

Affiliation discovery used:

- `https://api.crossref.org/works?query.affiliation=DeepSeek&filter=from-pub-date%3A2020-01-01%2Cuntil-pub-date%3A2026-07-19&rows=1000`
- `https://api.crossref.org/works?query.affiliation=Moonshot+AI&filter=from-pub-date%3A2020-01-01%2Cuntil-pub-date%3A2026-07-19&rows=1000`

The returned author affiliation arrays were then filtered for actual organization strings such as `DeepSeek-AI`, `DeepSeek AI`, `Beijing Deepseek Artificial Intelligence Fundamental Technology Research Company Ltd.`, or `Moonshot AI`. Final DOI metadata was read from `https://api.crossref.org/works/{DOI}`.

Crossref affiliation search is fuzzy. One DeepSeek result merely mentioned DeepSeek in an author's long biography, and the Moonshot query returned a very large generic “moonshot” result set. Those were rejected unless the structured author affiliation and a publisher/primary page confirmed the lab.

### OpenAlex

Institution lookup found [DeepSeek (I4405257960)](https://openalex.org/I4405257960) and [Moonshot AI (I4405260227)](https://openalex.org/I4405260227), but both institution entities reported zero linked works. The more useful discovery paths were:

- `https://api.openalex.org/works?filter=raw_affiliation_strings.search%3ADeepSeek&per-page=200` — 74 candidates at retrieval.
- `https://api.openalex.org/works?filter=raw_affiliation_strings.search%3AMoonshot+AI&per-page=200` — 24 candidates at retrieval.

These result counts are not publication counts. They include core papers, duplicates, malformed metadata, model/tool acknowledgements entered as affiliations, AI systems entered as coauthors, and—under “Moonshot”—papers from Alphabet's X, the Moonshot Factory. Only candidates with corroborating primary evidence were retained.

### Publisher and exact-title checks

Publisher pages/Crossref records were used to verify the explicit DeepSeek affiliations on Fire-Flyer AI-HPC, Insights into DeepSeek-V3, Janus, JanusFlow, EV charging forecasting, GUI test migration, The Faiss Library, CCAgent, and the Science policy perspective. Publisher pages similarly verified Moonshot affiliations on VisionLLaMA, Single Image Rolling Shutter Removal, The Best of Both Worlds, Image Quality Assessment, ANSMET, SAGraph, ExtendAttack, and Affinity Contrastive Learning.

Semantic Scholar's public API returned HTTP 429 during the sweep, so it was not used as an evidence source. The official/publisher/arXiv/Crossref/OpenAlex chain above was sufficient to resolve the retained set.

## Inclusion boundary and notable decisions

- DeepSeek-V3.2-Exp is retained separately from production DeepSeek-V3.2 because the official repository publishes a distinct report and explicitly distinguishes the experimental release.
- DeepSeek-V4 is included through the official transparency page/report even though its official 2026-04-24 release date precedes the `2606.*` arXiv identifier. The discrepancy is explicit in the record.
- Kimi K3, Kimi K2.6, Kimi K2 Thinking, Kimi-K2-Instruct-0905, Agent Swarm, Vendor Verifier, and PerceptionBench are retained as substantive first-party research disclosures even though no distinct arXiv paper was found. Their record type marks the boundary (`blog_with_report` or `benchmark`).
- Kimi K2.6's article links the earlier K2.5 report and a third-party system-card PDF, not a K2.6-specific paper; therefore its `pdf_url` remains null.
- Kimi-Researcher has a substantial first-party project report but no distinct arXiv record or own report PDF at the cutoff.
- DreamCraft3D is `affiliated`, not `core`: DeepSeek hosts and labels the official ICLR implementation, but the paper is not a DeepSeek model report.
- Broad author-affiliated work outside LLMs is retained in `affiliated` when publisher-deposited evidence names the organization; examples include EV forecasting, Faiss, VisionLLaMA, and Affinity Contrastive Learning.

## Explicit exclusions and false positives

- DeepSeek model/checkpoint updates without a distinct report—V2.5, V3-0324, R1-0528, and V3.1—were not made separate records.
- DeepSeek repositories that provide implementation notes/software but no distinct publication were excluded as publications: 3FS, DeepEP, DeepGEMM, DualPipe, EPLB, FlashMLA, LPLB, TileKernels, smallpond, and profile-data. The relevant V3/DualPipe or V3.2 papers remain included where applicable.
- Moonshot repositories that are software, SDKs, CLIs, kernels, or operational benchmarks without a standalone research publication were excluded, including batched-benchmark, checkpoint-engine, FlashKDA, kimi-agent SDK/CLI repositories, moonpalace, pykaos, and walle. FlashKDA's engineering deep-dive is not treated as a standalone paper.
- ShapeGPT was excluded: OpenAlex alone attached Wen Liu to “Deepseek”, while Crossref and the official project identify Tencent; the affiliation is disputed and lacks sufficient primary support.
- “Rethinking Chain-of-Thought Data” and “One Sample to Rule Them All” were excluded after full-text/title checks showed that DeepSeek was a cited model rather than an author affiliation.
- OpenAlex/Zenodo records listing `DeepSeek`, `Kimi`, ChatGPT, Claude, or Gemini as AI coauthors/tools were excluded.
- Records from `X, the Moonshot Factory` were excluded; it is Alphabet X, not Moonshot AI.
- Gated Delta Networks was treated as an external architectural precursor rather than a Moonshot-authored paper.
- Nature/IEEE/ACM/CVPR/ICCV/AAAI versions of the same arXiv work were merged rather than double counted.

## Public-full-text follow-up and remaining gaps

The download audit found two author-hosted copies that were not exposed by the
publisher DOI pages. They are now the inventory's public full-text locations:

- `deepseek-doi-10.1145-3650212.3680327` — the corresponding author's
  `2024-issta-migratepro.pdf` copy.
- `kimi-doi-10.1145-3695053.3731013` — the Tsinghua author group's
  `ansmet-isca25.pdf` copy.

The remaining records with no public PDF found are:

DeepSeek (2):

- `deepseek-doi-10.1145-3746252.3761392` — CCAgent.
- `deepseek-doi-10.1126-science.ady7922` — China's emerging regulation toward an open future for AI.

Moonshot AI / Kimi (8):

- `kimi-researcher-report`
- `kimi-k2-instruct-0905-note`
- `kimi-k2-thinking-note`
- `kimi-vendor-verifier-note`
- `kimi-agent-swarm-note`
- `kimi-k2.6-note`
- `kimi-perceptionbench-note`
- `kimi-k3-note`

“No public PDF found” means no arXiv full text or clearly public direct PDF was located. A DOI landing page may still provide subscription/institutional access.

## Known limitations

- Affiliation indexing is intrinsically incomplete and noisy. Crossref depends on publisher deposits, OpenAlex's normalized institutions had no linked works for either lab, and arXiv does not expose normalized affiliation fields in its Atom API.
- First-party research indexes can omit older or supporting papers; that is why official repositories, publisher metadata, and raw-affiliation searches were also swept.
- Corporate/team bylines are retained for unbylined official web reports rather than inventing individual authors.
- Kimi K3 and PerceptionBench were only two to three days old at the cutoff; a later paper or repository may supersede the current web-report records.
- Counts are cutoff snapshots, not claims about later releases.

## Validation

- 37 DeepSeek JSON objects and 32 Moonshot AI / Kimi JSON objects parse as JSONL.
- All 69 records contain every required field and non-empty identity fields (`id`, `org`, `title`, `authors`).
- IDs, type/tier enums, dates, URLs, arXiv IDs, DOI normalization, and within-corpus duplicate titles/identifiers pass the repository's `scripts/literature_corpus.py` validator when run on these two inventory files.
- No duplicate arXiv ID, DOI, normalized title, or organization/id pair exists within the two-file set.
- Discovery itself did not download PDFs. The separate corpus pipeline later
  downloaded and validated every currently listed public PDF and archived the
  HTML-native Kimi reports.
