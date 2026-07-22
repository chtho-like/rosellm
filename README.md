# RoseLLM

**Learn the LLM stack by building it.** RoseLLM is an executable, source-cited
systems lab spanning transformer training, high-throughput inference, GPU
kernels, and agentic reinforcement learning.

[![Documentation](https://github.com/chtho-like/rosellm/actions/workflows/docs.yml/badge.svg)](https://github.com/chtho-like/rosellm/actions/workflows/docs.yml)
[![Project status: experimental](https://img.shields.io/badge/status-experimental-orange.svg)](#project-status)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[Live documentation](https://www.wineandchord.com/rosellm/) ·
[Documentation source](docs/index.md) ·
[Learning roadmap](docs/learning-roadmap.md) ·
[Multimodal architecture](docs/multimodal/index.md) ·
[Agentic RL curriculum](docs/agentic-rl/index.md) ·
[Serving benchmarks](benchmarks/serving/README.md)

RoseLLM connects ideas that are often taught or implemented in isolation. The
same repository lets you derive an objective, inspect its PyTorch
implementation, trace the serving path down to KV-cache operations, and audit
the primary evidence behind a frontier-model claim.

This is a learning and research codebase, not a drop-in production serving
framework. The implementations favor visibility, instrumentation, and
experimentation over API stability.

## What is inside

| Area | What you can study and run | Entry point |
|---|---|---|
| **RoseTrainer** | Decoder-only transformer training, Hugging Face weight conversion, checkpoints, mixed precision, Distributed Data Parallel (DDP), tensor parallelism, and fused layers | [`rosellm/rosetrainer/`](rosellm/rosetrainer/) |
| **RoseInfer** | Offline and online generation, continuous batching, paged KV caches, copy-on-write prefix reuse, chunked prefill, streaming, multiprocess serving, and prefill/decode disaggregation | [`rosellm/roseinfer/`](rosellm/roseinfer/) |
| **RoseRLHF** | Auditable trajectories, return and advantage estimators, PPO/GRPO-style clipping, SAO/DIS, GSPO, GSPO-token, and SAPO objectives | [`rosellm/roserlhf/`](rosellm/roserlhf/) |
| **Kernel & distributed labs** | CUDA and Triton implementations of attention, FlashAttention, GEMM, softmax, normalization, KV-cache operations, and collective communication | [`notebooks/`](notebooks/) |
| **Benchmarks** | Reproducible online/offline harnesses for RoseInfer, vLLM, SGLang, and TensorRT-LLM, plus profiling and plotting tools | [`benchmarks/serving/`](benchmarks/serving/) |
| **Knowledge base** | A dependency-ordered LLM roadmap, multimodal architecture and generation, a full Agentic RL curriculum, mathematical derivations, source labs, and evidence-graded frontier-model case studies | [`docs/`](docs/) |

## Quick start

### 1. Install the project

Use a dedicated environment. PyTorch 2.6 or newer is declared by the package;
for NVIDIA systems, install the wheel matching your CUDA environment before
installing RoseLLM.

```bash
git clone https://github.com/chtho-like/rosellm.git
cd rosellm

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Optional inference backends such as FlashInfer, FlashAttention, and Triton are
environment-specific and are not installed by the base package. RoseInfer
selects available backends at runtime; the CUDA fast paths require an NVIDIA
GPU.

### 2. Run the smallest end-to-end Agentic RL example

This CPU-friendly example samples multi-turn trajectories, applies a terminal
verifier, constructs group-relative advantages, aligns them to action tokens,
and performs clipped policy updates:

```bash
python examples/agentic_rl_toy.py
```

Then follow the same data path in the
[source-level Agentic RL lab](docs/agentic-rl/source-lab.md).

### 3. Generate with a Hugging Face GPT-2 checkpoint

```bash
python -m rosellm.roseinfer.cli_generate \
  --hf-model-id gpt2 \
  --device cuda \
  --prompt "Large language models are" \
  --max-new-tokens 64 \
  --top-k 40 \
  --top-p 0.95 \
  --do-sample \
  --stream
```

The direct generation CLI currently loads Hugging Face GPT-2 models. The
server path also supports Qwen3 model configurations.

### 4. Start the OpenAI-compatible server

The convenience script starts RoseInfer on `127.0.0.1:8888`, with GPT-2 as the
default model:

```bash
bash rosellm/openai_server.sh
```

To serve Qwen3-0.6B instead:

```bash
ROSEINFER_HF_MODEL_ID="Qwen/Qwen3-0.6B" \
  bash rosellm/openai_server.sh
```

Call the streaming chat endpoint from another terminal:

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Explain paged attention."}],
    "max_tokens": 128,
    "stream": true
  }'
```

RoseInfer exposes `/health`, `/generate`, `/v1/models`,
`/v1/completions`, and `/v1/chat/completions`. See
[`rosellm/openai_server.sh`](rosellm/openai_server.sh) for cache sizing and
startup controls.

## Read the stack in order

If you are here to learn rather than benchmark, use the repository as a
dependency graph:

1. Start with the [documentation home](docs/index.md) and the five-pass study
   method in the [LLM learning roadmap](docs/learning-roadmap.md).
2. Move from mathematics and transformer foundations through pretraining,
   alignment, inference systems, evaluation, agents, and safety.
3. Use the [multimodal foundation-model map](docs/multimodal/index.md) to trace
   images, video, documents, and audio from representation through shared
   reasoning and media generation.
4. Study the [Agentic RL curriculum](docs/agentic-rl/index.md) from terminology
   and POMDP foundations to algorithms, data, rollout systems, evaluation, and
   executable objectives.
5. Read the [research standard](docs/research-method.md) before the
   [frontier-model case studies](docs/agentic-rl/case-studies/index.md).

The research notes distinguish **disclosed facts**, **confirmed artifacts**,
**reproduced results**, **inferences**, and **unknowns**. Undisclosed training
details stay unknown instead of being filled in by analogy.

## Systems highlights

### Training

- Minimal decoder-only transformer implementations with explicit attention,
  loss, optimizer, checkpoint, and data paths.
- Single-process and DDP training entry points, tensor-parallel layers,
  activation checkpointing, and FP16/BF16 automatic mixed precision.
- GPT-2 and Qwen3 model/configuration adapters for comparing local
  implementations with Hugging Face checkpoints.

### Inference and serving

- Continuous and dynamic batching with configurable admission limits,
  prefill packing, and streaming intervals.
- Global paged KV-cache management, longest-prefix lookup, copy-on-write block
  reuse, cache budgeting, and rollover handling.
- Chunked prefill, CUDA graphs, fused KV append/MLP/sampling paths, and
  pluggable naive, FlashInfer, or FlashAttention backends.
- Optional engine processes and two-engine prefill/decode disaggregation.
- OpenAI-compatible text and chat completions with Server-Sent Events (SSE).

### Agentic reinforcement learning

- A trajectory contract that preserves task, environment, tokenizer, sampled
  token, log-probability, action-mask, and policy-version provenance.
- Discounted returns, GAE, leave-one-out and group-standardized advantages,
  turn-to-token credit assignment, and observation masking.
- Executable PPO/GRPO-style, SAO/DIS, GSPO, GSPO-token, and SAPO policy
  objectives with diagnostic outputs and focused tests.

### Kernels and measurement

- Progressive CUDA labs for attention, FlashAttention, GEMM, softmax,
  LayerNorm/RMSNorm, elementwise fusion, transpose, and histogram kernels.
- Triton kernels and verification harnesses for serving-critical operations.
- Online latency, offline throughput, Torch profiler, and Nsight Systems
  workflows for controlled backend comparisons.

## Repository map

```text
rosellm/
├── rosellm/
│   ├── roseinfer/       # inference engine and API server
│   ├── roserlhf/        # trajectories, advantages, and policy losses
│   └── rosetrainer/     # transformer models and training loops
├── docs/                # curriculum, derivations, and research notes
├── examples/            # small end-to-end learning examples
├── benchmarks/serving/  # backend comparison and profiling harnesses
├── notebooks/           # PyTorch, distributed, Triton, and CUDA labs
├── scripts/             # checks and microbenchmarks
└── tests/               # CPU, GPU, scheduler, server, and RL tests
```

## Documentation and validation

Install the pinned documentation toolchain and build the site locally:

```bash
python -m pip install -r requirements-docs.txt
make docs
make docs-render
python -m mkdocs serve -a 127.0.0.1:8000
```

`make docs` runs strict MkDocs and source/generated-HTML math checks.
`make docs-render` additionally opens every page in a real browser and verifies
the rendered formulas. The generated site is published at
[www.wineandchord.com/rosellm](https://www.wineandchord.com/rosellm/), and every
guide remains readable directly on GitHub.

Useful code checks:

```bash
make test-fast     # excludes slow, GPU, and distributed tests
make test-no-cuda  # hides CUDA and excludes GPU-marked tests
make lint          # flake8, mypy, Black, and isort checks
```

GPU kernels, fused backends, distributed runs, and serving benchmarks require
the corresponding CUDA hardware and optional dependencies.

## Project status

RoseLLM is experimental and under active development. APIs, checkpoint formats,
and benchmark interfaces may change. Treat it as a transparent implementation
and research lab; validate correctness, performance, and operational behavior
before adapting any component to a production system.

## Contributing

Keep changes inspectable and reproducible. Technical claims should cite primary
sources when available; quantitative results should identify the command,
environment, configuration, and artifact. Documentation changes must follow the
[writing and mathematical rendering standard](docs/documentation-quality.md).

## License

RoseLLM is released under the [MIT License](LICENSE).
