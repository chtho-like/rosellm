# Research, Evidence, and Citation Standard

Frontier-model writing is unusually vulnerable to confident but unsupported
claims. Model developers disclose only part of their process, names are reused
across product and research versions, benchmark settings differ, and later
checkpoints silently replace earlier ones. This document defines how RoseLLM
keeps technical claims auditable.

## Evidence classes

Every consequential claim in a model case study should be classifiable as one
of the following.

| Label | Meaning | Acceptable wording |
|---|---|---|
| **D — Disclosed** | A primary source states the claim directly. | “The report states …” |
| **R — Reproduced** | We reran the procedure and retained configuration, logs, code revision, and artifacts. | “Our reproduction measured …” |
| **I — Inferred** | The conclusion follows from listed evidence plus explicit assumptions, but the source does not state it. | “This suggests … under assumptions A and B.” |
| **U — Unknown** | Public evidence is absent, ambiguous, contradictory, or insufficient. | “The public sources do not disclose …” |

The labels are ordered by kind, not credibility. A carefully qualified inference
can be useful; it must never be presented as a disclosed production procedure.

## Source hierarchy

Prefer sources in this order for claims about a model or method:

1. paper or technical report from the authors;
2. official model card, repository, release notes, API documentation, or dataset
   card;
3. first-party engineering blog or recorded presentation;
4. peer-reviewed independent reproduction;
5. well-documented third-party reproduction with code and artifacts;
6. secondary reporting, used only for context and linked to its underlying
   evidence when possible.

A survey is excellent for discovery and taxonomy, but the original paper should
support algorithmic and quantitative claims. A leaderboard is not a substitute
for a benchmark protocol. A model's generated explanation of its own training
is not evidence about that training.

## Required citation precision

A useful citation lets a reader verify the exact claim without searching an
entire document. Include, when available:

- stable URL, DOI, or arXiv identifier and version;
- section, appendix, table, figure, or page;
- model/checkpoint name and release date;
- base versus instruct/reasoning/API variant;
- evaluation regime: zero/few-shot, sampling temperature, number of samples,
  tools, context budget, and judge;
- unit and denominator for every numerical value.

For example, “67.3” is not a complete result. State whether it is accuracy,
pass@1, Elo, win rate, or another metric; identify the test set and model
variant; and record whether tools or majority voting were used.

## Quantitative claim checklist

Before adding a number, answer all applicable questions:

- Is the parameter count total, non-embedding, trainable, or active per token?
- Is the token count raw, filtered, unique, sampled, or repeated? Does it include
  continued pretraining or post-training tokens?
- Is context length the training length, validated length, or advertised API
  limit?
- Is compute reported in FLOP, accelerator-hours, wall-clock time, or monetary
  cost? What hardware and utilization assumptions apply?
- Does a dollar figure cover only a final training run, or also data, ablations,
  failed runs, salaries, and infrastructure?
- Is a benchmark result single-sample, pass@k, consensus, or best-of-N?
- Was the evaluated model public, an internal checkpoint, or a changing API?

If the source does not answer a question material to interpretation, mark it
unknown instead of supplying a plausible value.

## Reconstructing a training pipeline

Case studies use the following table to prevent gaps from being filled by
imagination:

| Stage | Evidence to seek | Typical unknowns |
|---|---|---|
| Base architecture | dimensions, attention, MoE routing, tokenizer, context | exact implementation and kernel choices |
| Pretraining data | token count, domains, filtering, mixture, deduplication | corpus identities, mixture weights, licenses |
| Pretraining optimization | optimizer, schedule, batch, precision, stability methods | failed runs, full hyperparameter schedule |
| Mid-training | context extension, domain adaptation, continued pretraining | exact data overlap and stage boundaries |
| SFT | task categories, source types, sample count, loss masking | prompts, annotator process, mixture weights |
| Preference/reward modeling | labels, judges, verifiers, reward composition | rubric, calibration, held-out agreement |
| RL | algorithm, rollout count, group size, clipping/KL, curriculum | infrastructure, complete hyperparameters, selection bias |
| Distillation | teacher, sampling, filtering, student objective | rejected samples and contamination controls |
| Evaluation | benchmarks, prompts, sampling, tools, judge | private sets and cherry-picking controls |
| Deployment | serving model, routing, tools, safety layers, monitoring | production topology and live policy updates |

The final row is often almost entirely unknown. A public technical report is a
training disclosure, not necessarily a description of the vendor's live
product stack.

## Reproduction records

A result labeled **R** must preserve:

- repository revision and uncommitted diff;
- environment and dependency lock;
- hardware topology, driver, and accelerator details;
- data manifest and immutable hashes where licensing permits;
- exact command and resolved configuration;
- random seeds and determinism settings;
- raw logs, checkpoints, and evaluation outputs;
- expected statistical variation and known deviations from the source.

If an artifact cannot be redistributed, retain a manifest that describes how an
authorized researcher can reconstruct it.

## Freshness and contradiction policy

Every vendor case study has a **verified through** date. Newer versions are not
silently folded into older rows. When primary sources disagree:

1. quote neither source at length;
2. record both claims and their dates;
3. check whether they describe different variants or evaluation settings;
4. prefer the later correction only when it explicitly supersedes the earlier
   source; and
5. otherwise retain the contradiction as unresolved.

## Citation format used in this repository

Guides use readable inline links plus a numbered references section. A first
mention should normally include authors or organization, title, year, and a
stable link. Repeated citations may use the short title. For source-code claims,
link to a revision-pinned file and line range rather than a moving default
branch.

This standard is intentionally stricter than ordinary tutorial writing. The
goal is not to maximize citation count; it is to make every important claim
traceable to the strongest available evidence.
