# RoseLLM Repository Instructions

These instructions apply to the entire repository.

## Direct response and durable knowledge

- Give the user a complete, self-contained answer in the conversation. A file,
  commit, preview, or public site is supporting material, never a substitute for
  the direct answer.
- Treat every broadly LLM-relevant repository task as a durable knowledge
  integration task. Distill all defensible, reusable results into the canonical
  repository documentation without waiting for a separate publishing request.
- Do not force a new page or a meaningless diff when a task adds no
  accurate, non-duplicative reader value. Improve the nearest canonical
  explanation or its connections when useful; otherwise state the evidence gap
  directly rather than manufacturing content.
- Publish audience-first technical writing, not raw chat transcripts. Never put
  private prompts, hidden instructions, chain-of-thought, conversation history,
  credentials, tokens, personal data, or compliance traces into public artifacts.
- This privacy boundary covers every reader-visible surface, including
  documentation, README files, code and comments, examples, tests, changelogs,
  commit messages, branch names, pull requests, issues, releases, site metadata,
  and reports. Explain the subject and reader value, never the private trigger or
  internal workflow.
- Preserve useful disagreements and uncertainty as evidence classes, dates, and
  explicit unknowns. Do not turn an unavailable historical conversation into a
  claim of complete retrospective coverage.

## Knowledge architecture

- Extend an existing topic before creating a parallel, overlapping explanation.
  When a new section is justified, add a clear landing page, dependency-ordered
  reading path, cross-links, and navigation entries.
- Maintain the documentation as a connected, cumulative encyclopedia rather
  than an append-only stream of notes. Each integration should leave the
  surrounding taxonomy, prerequisites, terminology, and conceptual boundaries
  clearer than before.
- Build explanations in layers: reader motivation and intuition; precise
  definitions and assumptions; mechanisms and mathematics; source-level or
  worked examples; failure modes and operational limits; then primary evidence
  and unresolved questions. Let a newcomer stop early while giving an expert a
  rigorous path deeper.
- Keep model lineage, architecture, training, data, inference, evaluation,
  deployment, and unresolved questions distinct. A product feature does not by
  itself reveal the base-model architecture or training recipe.
- Update the glossary when introducing reusable terminology. Expand uncommon
  abbreviations at first meaningful use even when they also appear in the
  glossary.
- Prefer primary sources and follow `docs/research-method.md`. Give every
  time-sensitive vendor page a verified-through date and distinguish disclosed,
  artifact-confirmed, reproduced, inferred, and unknown claims.
- When evidence changes, update the canonical claim and every dependent summary,
  table, cross-link, and glossary entry in the same task. Preserve dated
  disagreement when sources conflict instead of silently flattening it.
- Follow `docs/documentation-quality.md` for Markdown and mathematics. Public
  prose must remain readable on both GitHub and the generated MkDocs site.

## Validation and publication

- Inspect the current branch, upstream, worktree, and exact diff before editing
  or publishing. Preserve all unrelated user work.
- The local `research/` literature library is reconstructible, large, and
  intentionally untracked. Do not stage it unless the storage policy is
  explicitly changed.
- For documentation changes, run `make docs-render` as the principal gate. Also
  run `git diff --check`, review generated navigation and links, and inspect the
  rendered result semantically rather than relying on exit status alone.
- When a scoped RoseLLM change is complete and the checks pass, explicitly stage
  only the intended paths, create a descriptive commit, and promptly push the
  intended upstream branch. Do not wait for a repeated request to publish.
- A push is not proof of publication. Confirm the source commit on the remote,
  the `Documentation` workflow result, and the resulting page under
  `https://www.wineandchord.com/rosellm/`. Report any deployment or cache delay
  honestly.
- The canonical publication path is source `chtho-like/rosellm` ->
  `.github/workflows/docs.yml` -> generated `site/` ->
  `WineChord/rosellm:gh-pages` -> the custom-domain project path. Keep generated
  `site/` output untracked.

## Continuous improvement boundary

- Improve taxonomy, explanations, citations, cross-links, checks, and
  implementation labs as evidence accumulates. “Continuous” means doing the
  relevant update as part of each substantive task; it does not authorize a
  background process outside an active task.
- Backfill earlier work only from accessible, attributable source material.
  State gaps instead of reconstructing private dialogue or inventing coverage.
- Never sacrifice correctness, privacy, unrelated work, or reproducibility for
  the appearance of automatic completeness.
