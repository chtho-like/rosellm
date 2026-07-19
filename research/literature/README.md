# Frontier-Lab Literature Corpus

This directory is the reproducible source registry and local archive for
research associated with DeepSeek, Moonshot AI/Kimi, Zhipu AI/GLM, OpenAI,
Anthropic, and Gemini/Google DeepMind.  It deliberately covers far more than
agentic reinforcement learning.

The repository tracks the inventories, discovery logs, corpus tooling, and
coverage audits.  Downloaded PDFs and HTML snapshots live under `archive/` and
are ignored by Git because they are large and may have redistribution
restrictions.  Every archived artifact is represented in a generated manifest
with its resolved URL, byte size, and SHA-256 digest.

## What "all" means here

No public search system can prove that it has found unpublished work, private
reports, silently removed pages, or papers whose affiliation is omitted.
Accordingly, completeness is treated as an auditable claim over a defined
public universe rather than as a slogan.

Each organization has three inclusion tiers:

1. `core`: model-family technical reports, system cards, model cards, and
   first-party reports whose main object is one of the named systems;
2. `direct`: papers about architectures, data, training, inference, agents,
   evaluation, safety, interpretability, or applications that directly support
   or analyze those systems; and
3. `affiliated`: broader research with an author affiliation to the relevant
   institution at the time of publication.

For Gemini, `core` and `direct` mean Gemini-specific work.  The `affiliated`
tier is Google DeepMind research, not every publication from every Google
division.  Cross-organization papers may appear in more than one inventory;
their DOI or arXiv identifier is used to identify the shared work.

The target public universe is the union of:

- official research and publication indexes;
- official system-card, model-card, repository, and model-hosting indexes;
- official organization sitemaps and release archives;
- bibliographic records whose paper itself shows the relevant affiliation;
- correction, replacement, and withdrawn-version notices; and
- discovery-only secondary indexes, followed back to a primary source before
  admission.

Coverage is considered demonstrated only when the discovery log records the
source snapshot date, pagination boundary or item count, queries used,
deduplication result, and unresolved gaps.  Search-engine result counts are not
coverage evidence.

## Inventory format

Each line of `inventory/*.jsonl` is one JSON object with these fields:

| Field | Meaning |
|---|---|
| `id` | Stable lowercase slug within the organization |
| `org` | Canonical organization label |
| `title` | Published title |
| `authors` | Ordered author names; an organization author is allowed |
| `date` | ISO date when known, otherwise year or `null` |
| `type` | `technical_report`, `research_paper`, `system_card`, `model_card`, `dataset`, `benchmark`, `blog_with_report`, or `other` |
| `tier` | `core`, `direct`, or `affiliated` |
| `arxiv_id` | Base arXiv identifier without a version suffix, or `null` |
| `doi` | DOI without a `doi.org` prefix, or `null` |
| `primary_url` | Canonical first-party or publisher record |
| `pdf_url` | Direct report/paper PDF when publicly available |
| `source_pages` | Pages that prove discovery and provenance |
| `affiliation_evidence` | Concise evidence for organizational attribution |
| `topics` | Controlled and free-form topic labels |
| `notes` | Version, supersession, access, or ambiguity notes |
| `retrieved_at` | Date on which the record was verified |

Unknown values are `null`, never an invented guess.  A missing public PDF is
not a reason to omit a record: the inventory retains it and the coverage audit
marks the artifact gap.

## Corpus workflow

The corpus driver has no third-party dependency for validation or downloading.
Full-text extraction uses the small, pinned research environment:

```bash
python3 -m pip install -r requirements-research.txt
```

### Offline validation

The complete offline gate validates inventory schema and duplicates, prints
artifact coverage, audits the research chapters against the inventory
and generated coverage ledger, and runs the focused literature-tool tests:

```bash
make research-check
```

Its component targets are `make research-validate`,
`make research-doc-audit`, `make research-equivalence-audit`, and
`make research-test`. The document audit treats
the six inventories and `coverage.json` as authoritative: every inventory ID
and primary URL must occur together in the mapped lab chapter, and the summary
table in `docs/frontier-labs/index.md` must match the ledger. These checks do
not use the network. Normal `make docs` and `make docs-render` runs depend on
that read-only audit, never on download or recovery targets.

The equivalence audit separately proves that every recovered arXiv URL already
owned by another record in the same organization has a matching explicit
publication/preprint or announcement/paper link in
`candidates/document-equivalences.jsonl`. This keeps record-level completeness
without silently inflating work-level counts.

### Download and extraction

```bash
python3 scripts/literature_corpus.py validate
python3 scripts/literature_corpus.py audit
python3 scripts/literature_corpus.py download --workers 8
python3 scripts/literature_corpus.py extract --repair-quality --workers 6
```

Useful narrower runs are:

```bash
python3 scripts/literature_corpus.py --org DeepSeek --tier core download
python3 scripts/literature_corpus.py --org Anthropic audit
```

Downloads are resumable.  A file is accepted as a PDF only if it has the PDF
magic header and passes `pdfinfo` when that binary is available.  The manifest
is regenerated deterministically after every run.

On macOS, `brew install poppler` adds `pdftotext` and `pdftoppm`. With
`--repair-quality`, the extractor invokes `pdftotext` only when the primary
pypdf text contains Unicode replacement characters or pypdf fails; the
alternate is accepted only if it reduces damage without discarding most of the
document. The original PDF remains authoritative either way.

### Human-readable PDF library

The canonical archive deliberately keeps the stable machine path
`archive/pdf/<org>/<inventory-id>.pdf`. Download, extraction, recovery,
promotion, equivalence, and coverage tooling all use that identity, so a title
correction never silently moves the authoritative artifact.

For Finder, shell browsing, and local search, generate a separate readable
view after downloading PDFs:

```bash
make research-library
```

The command builds Git-ignored copy-on-write clones under
`library/<org>/<tier>/<year>/`. A typical filename is:

```text
DeepSeek-R1--Incentivizing-Reasoning-Capability-in-LLMs-via-Reinforcement-Learning--2025--TR--deepseek-2501.12948.pdf
```

Major fields use `--`; punctuation and whitespace inside a field collapse to
`-`. The optional first field is a curated common name from
`display-names.jsonl`, never a guessed acronym. Type codes are `TR`, `PAPER`,
`SC`, `MC`, `DATA`, `BENCH`, `REPORT`, and `OTHER`. The stable inventory ID is
always retained as the final field so same-title records, publication/preprint
pairs, and case-insensitive filesystems cannot silently collide.

Names are capped at 180 UTF-8 bytes. Only the human title component may be
shortened; the exact title remains in the inventory and the generated
`library/index.jsonl`. The generator normalizes Unicode and HTML markup,
rejects unknown or duplicate alias entries, preflights every case-folded path,
verifies canonical and version-ledger SHA-256 values, and refuses to replace a
directory that has a missing, copied, or unsafe generator marker or contains
unindexed user files. Use the read-only preview before rebuilding when
reviewing a naming change:

```bash
python scripts/literature_library.py build --dry-run
```

On macOS/APFS the default `clone` mode gives every readable PDF an independent
inode while initially sharing the canonical data blocks. Editing a browsing
copy therefore cannot write through to the authoritative archive, and an
unchanged library does not consume a second physical copy of the corpus.
Linux requires a reflink and refuses to fall back to a byte copy.
`--link-mode hardlink` remains available only as an explicit portability
escape hatch; it has write-through semantics and should be treated as
read-only. Both `archive/` and `library/` remain local, reconstructible, and
ignored by Git.

Preserved historical PDFs from `candidates/document-versions.jsonl` appear
beside their canonical record with a final `--rev-<SHA-prefix>` field. This
keeps alternate official revisions readable without pretending that they are
separate inventory records.

### Recovery passes

Recovery is conservative, resumable, and evidence-first. Each automated pass
writes or merges its own JSONL ledger and may install a fully validated PDF in
the ignored local archive, but none edits an authoritative inventory:

1. `make research-recover-openalex` handles records whose ordinary download
   failed. It resolves the exact OpenAlex work identity, or its DOI identity,
   and probes independently listed open-access locations.
2. `make research-recover-arxiv` searches both failed and `missing_pdf_url`
   records, explicitly excluding `blog_with_report`. It queries arXiv by the
   inventory title and requires a normalized-title score of at least `0.96`;
   an exact DOI match is also accepted as identity. The chosen PDF still has
   to pass the corpus PDF checks.
3. `make research-recover-pmc` considers local-artifact gaps for DOI-bearing
   `research_paper` records. It requires an exact Europe PMC DOI result with a
   PMCID and `hasPDF=Y`, accepts only the official NCBI PMC OA PDF route, and,
   if that legacy route fails, requires the newest exact-PMCID canonical PDF
   object in the public PMC AWS dataset.

Run all three in that order with:

```bash
make research-recover
```

The aggregate target executes the passes sequentially so that they do not race
while installing canonical archive files. The resulting evidence is stored in
`candidates/oa-recovery.jsonl`, `candidates/secondary-recovery.jsonl`, and
`candidates/pmc-recovery.jsonl`.

Some official or primary routes require human review: for example, a
first-party model-card PDF, an official proceedings copy, or a title variant
whose author set must be checked. Those decisions belong in
`candidates/manual-recovery.jsonl`, including the inventory identity, selected
and provenance URLs, review basis, explicit title match, PDF-magic and
`pdfinfo` results, byte and page counts, SHA-256 digest, and local path when
available. Manual evidence does not bypass promotion validation.

When one official report page exposes multiple PDF routes or revisions, the
inventory keeps one canonical record and
`candidates/document-versions.jsonl` records each validated alternate,
distinguishing byte-identical aliases from byte-distinct official revisions.
Alternate revision files are preserved under `archive/versions/` instead of
being miscounted as separate papers or left as orphan manifest rows.

`candidates/document-equivalences.jsonl` serves a different purpose: it links
an inventory publication or announcement to an explicitly verified related
manifestation, such as a preprint or paper. When both citable records are in
the inventory, they remain separate in record-level totals; the relation
supports work-level accounting without silent deduplication. The ledger also
records exceptional relations such as a withdrawn preprint with no public PDF,
without fabricating an artifact. Both ledgers are read by the generated report.

### Review and promote recovery evidence

After all recovery jobs have finished, promotion is a separate, reviewable
operation. The Make target and the bare command are both dry-runs:

```bash
make research-promote-recovery
python3 scripts/literature_recovery_promotion.py
```

The promoter reads `candidates/oa-recovery.jsonl`,
`candidates/secondary-recovery.jsonl`, and
`candidates/manual-recovery.jsonl`, plus
`candidates/pmc-recovery.jsonl`. Every `status=recovered` row must still map to
the same inventory identity, and its canonical local artifact must match PDF
magic, mandatory `pdfinfo`, byte count, and SHA-256 evidence. It validates the
entire proposed inventory before writing and may only change `pdf_url`,
`arxiv_id`, and `source_pages`; IDs, titles, authors, organization, tier, and
affiliation evidence remain immutable. If a recovered arXiv URL is already
owned by another record in the same organization, the promoter keeps that
record as the identifier owner, updates only the recovered record's PDF and
provenance links, and reports `existing_arxiv_owner` for a later explicit
version/equivalence decision. After reviewing the complete dry-run, apply the
same plan explicitly:

```bash
python3 scripts/literature_recovery_promotion.py --apply
```

Repeated application is idempotent. Do not use `--apply` while any recovery
ledger is still being generated.

The equivalent explicit Make target applies the already reviewed plan and then
revalidates the inventories:

```bash
make research-apply-recovery
```

### Regenerate reports and audit documentation

After an applied promotion or an intentional ledger update, rebuild all tracked
coverage outputs and rerun the documentation gate:

```bash
make research-report
make research-doc-audit
make research-test
```

`research-report` deterministically regenerates
`research/literature/coverage.json`, `docs/frontier-labs/coverage.md`, and
`docs/frontier-labs/bibliography.md`. Its source snapshot includes official
page evidence, every recovery ledger, `document-versions.jsonl`, and
`document-equivalences.jsonl`, so unresolved artifacts and record/work
relationships remain visible rather than being hidden by successful recovery.

## Analytical deliverables

The final synthesis is organized separately from the machine inventory so that
factual records and interpretation do not blur together.  It will include:

- one chronological and thematic map per organization;
- model-family and method genealogies;
- architecture, data, optimization, post-training, inference, systems,
  multimodal, evaluation, safety, interpretability, and application analyses;
- disclosure-versus-unknown matrices using the repository evidence standard;
- cross-lab comparisons that normalize benchmark protocol differences; and
- a coverage ledger listing every unresolved source, inaccessible artifact,
  affiliation ambiguity, and suspected omission.

The existing `docs/agentic-rl/` material is a verified thematic seed, not the
boundary of this corpus.
