# RoseLLM

RoseLLM is both an implementation lab and a rigorous, source-cited learning
repository for large language models.

The codebase is organized around three complementary goals:

1. **Learn the stack from first principles.** The documentation starts with the
   required mathematics and builds toward modern pretraining, post-training,
   agentic reinforcement learning, distributed training, and inference systems.
2. **Read executable implementations.** Minimal PyTorch, Triton, and CUDA
   implementations expose the mechanics that production frameworks often hide.
3. **Connect papers to real systems.** Research notes identify exactly what a
   paper or model developer disclosed, what can be reproduced, what is a
   defensible inference, and what remains unknown.

## Start here

If you are starting from zero, use this order:

1. Read the [documentation home](docs/index.md) and the five-pass method in the
   [LLM learning roadmap](docs/learning-roadmap.md). Treat Levels 0–6 as a
   dependency and mastery checklist; the repository does not yet replace a full
   textbook for every prerequisite.
2. Open the [Agentic RL curriculum](docs/agentic-rl/index.md), then read
   terminology, history, mathematics, derivations, algorithms, data,
   end-to-end training, systems, evaluation, and the source-level lab in that
   order.
3. Learn the [research and citation standard](docs/research-method.md), then use
   the [evidence matrix](docs/agentic-rl/case-studies/index.md) to study
   DeepSeek, GLM, Kimi, the western frontier labs, and open industry without
   confusing public evidence with plausible inference.
4. For the full literature project, open the
   [six-lab frontier research corpus](docs/frontier-labs/index.md): it tracks
   3,050 auditable public records across DeepSeek, Kimi, GLM, OpenAI,
   Anthropic, and Google DeepMind/Gemini. The generated
   [coverage ledger](research/literature/coverage.json) is the source of truth
   for current PDF and extracted-text counts and unresolved gaps; the tracked
   candidate ledgers preserve recovery evidence and explicit version and
   equivalence relationships.

The implementation entry points are [training](rosellm/rosetrainer/),
[inference](rosellm/roseinfer/), [Agentic RL](rosellm/roserlhf/), and the
[kernel notebooks](notebooks/cuda/).

The first long-form curriculum is **Agentic Reinforcement Learning**. It covers
the formal POMDP view, policy-gradient mathematics, data and environment
construction, verifiable and learned rewards, multi-turn credit assignment,
rollout/training infrastructure, evaluation, safety, and evidence-backed case
studies of frontier model families.

## Documentation

Technical claims should cite primary sources whenever a primary source exists.
[Documentation and mathematical rendering rules](docs/documentation-quality.md)
define the acceptance gate for Markdown, TeX, generated HTML, and browser
output.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-docs.txt
make docs
make docs-render
python -m mkdocs serve
```

The documentation site is configured with MkDocs Material. Every guide is also
readable directly on GitHub.

## Literature corpus operations

The [literature corpus guide](research/literature/README.md) documents the
complete reproducible workflow. Offline validation, documentation coverage,
and focused unit tests can be run together without contacting external
services:

```bash
make research-check
```

Artifact acquisition and recovery are deliberately explicit network
operations. The aggregate recovery target runs OpenAlex open-access lookup,
strict arXiv title recovery for failed or missing non-blog artifacts, and the
Europe PMC/PMC AWS fallback in that order. Manually verified official routes
are recorded in the same evidence-first workflow.

```bash
make research-download
make research-recover
make research-promote-recovery  # validates and previews; does not edit inventories
make research-apply-recovery    # explicit inventory promotion
make research-extract
make research-library           # readable copy-on-write view; shared initial blocks
make research-report
make research-doc-audit
```

Canonical artifacts keep their stable inventory-ID paths for recovery and
audit reproducibility. `make research-library` builds a separate Git-ignored
`research/literature/library/` tree whose PDF names contain curated common
names, readable titles, year, document type, and stable record ID. The default
copy-on-write clones use independent inodes while initially sharing the
canonical data blocks, so browsing-layer edits cannot write through to the
authoritative archive.

`make docs` and `make docs-render` perform the read-only corpus documentation
audit, but never download papers, contact recovery services, or promote
inventory changes.

## Package status

RoseLLM is an experimental learning and research project. Its APIs may change
while the training and inference stacks are being developed.

## License

[MIT](LICENSE)
