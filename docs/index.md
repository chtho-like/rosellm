# RoseLLM Documentation

Welcome to the RoseLLM docs. This site collects deep dives and guides for the core subsystems used in training and serving large language models.

- Training engine, parallelism, and memory optimizations
- Mixed precision with dynamic loss scaling
- Gradient bucketing and communication overlap
- Position embeddings (RoPE) and related utilities

Quick start

- Install: `pip install -e .[dev]`
- Run tests: `make test`
- Build docs: `make docs`  (serving with `make docs-serve`)

Key guides

- Mixed precision: `dynamic-loss-scaling-deep-dive.md`
- Gradient bucketing: `gradient-bucketing-implementation-guide.md`
- ROPE: `rope-position-embeddings-deep-dive.md`
- Range-based buffer mapping: `range-based-parameter-buffer-mapping-interview-guide.md`

For examples, see the `examples/` directory and the project README.
