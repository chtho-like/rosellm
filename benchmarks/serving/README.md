# Serving Benchmarks (roseinfer vs vLLM vs SGLang vs TensorRT-LLM)

## Dependencies

These scripts assume the following Python packages are available:

- `openai`, `httpx`, `transformers`, `numpy`, `matplotlib`
- `vllm` (for vLLM runs)
- `sglang` (for SGLang runs)
- `tensorrt_llm` (optional, for TensorRT-LLM runs)
- Optional (for roseinfer self-compare variants): `flashinfer`, `flash-attn`
- Optional (for Nsight Systems profiling): `nsys` (Nsight Systems CLI)

## Online (OpenAI servers + trace replay)

```bash
python benchmarks/serving/online_compare.py --model gpt2 --gpu 0
```

This starts each backend's OpenAI-compatible server (one at a time), replays TraceA with a shared OpenAI client, and writes `online_results.json` under `outputs/benchmarks/serving/online_<timestamp>/`.

`online_results.json` includes `meta.versions` and wall-time info (`meta.run_start_time`, `meta.run_end_time`, `meta.wall_s`).

Notes:
- vLLM async scheduling is enabled by default for a fairer comparison vs SGLang/TensorRT-LLM; disable with `--vllm-no-async-scheduling`.
- vLLM attention backend defaults to FlashInfer; override with `--vllm-attention-backend`. When using FlashInfer attention on 12GB GPUs, set `--vllm-max-num-seqs 128` (the scripts auto-set this when `--vllm-attention-backend=flashinfer`).
- SGLang attention defaults to Triton; override with `--sglang-attention-backend`.

## Offline (token-id throughput)

```bash
python benchmarks/serving/offline_compare.py --model gpt2 --gpu 0
```

This runs each backend (one at a time) in its own process and writes `offline_results.json` under `outputs/benchmarks/serving/offline_<timestamp>/`.

Offline is aligned on token IDs (no tokenize/detokenize in the measured path):

- roseinfer: `prompt_token_ids` via `OnlineScheduler`
- vLLM: `prompt_token_ids` + `SamplingParams(detokenize=False)`
- SGLang: `Engine.generate(input_ids=...)` with `skip_tokenizer_init=True`
- TensorRT-LLM: `prompt_token_ids` + `SamplingParams(detokenize=False)`

`offline_results.json` includes `meta.versions` and wall-time info (`meta.run_start_time`, `meta.run_end_time`, `meta.wall_s`, `meta.backend_wall_s`).

Notes:
- vLLM async scheduling is enabled by default for a fairer comparison vs SGLang/TensorRT-LLM; disable with `--vllm-no-async-scheduling`.
- vLLM attention backend defaults to FlashInfer; override with `--vllm-attention-backend`. When using FlashInfer attention on 12GB GPUs, set `--vllm-max-num-seqs 128` (the scripts auto-set this when `--vllm-attention-backend=flashinfer`).
- SGLang attention defaults to Triton; override with `--sglang-attention-backend`.

## Plotting

```bash
python benchmarks/serving/plot_compare.py \
  --online outputs/benchmarks/serving/online_*/online_results.json \
  --offline outputs/benchmarks/serving/offline_*/offline_results.json
```

This writes figures (paper-style) under `--output-dir` (default: `outputs/benchmarks/serving/figures/`):

- `online_latency_compare.png` (2x2 overview: TTFT/TPOT/ITL/E2E; p90 curve + p50–p90 band; hollow markers are p99)
- `online_ttft_ms.png`, `online_tpot_ms.png`, `online_itl_ms.png`, `online_e2e_ms.png`
- `offline_throughput_compare.png`
- `online_summary.md`, `offline_summary.md`

## One-shot convenience script

```bash
bash scripts/bench_serving_compare.sh
```

## Self-compare (roseinfer variants only)

```bash
bash scripts/bench_roseinfer_self_compare.sh
```

By default this runs `roseinfer` with multiple prefill attention backends (naive / flashinfer / flash-attn when installed), and produces the same plot set under `outputs/benchmarks/serving/figures/`.

Useful knobs:

- A/B overlap scheduling: add `--roseinfer-compare-overlap-schedule` (overlap is enabled by default).

## Profiling (optional, separate stage)

Both online and offline benchmarks support an extra profiling stage that runs *after* the benchmark stage (so the benchmark JSON results used for plots/tables are not affected).

- Online: `python benchmarks/serving/online_compare.py --profile torch|nsys|both`
  - `--profile-only` skips trace replay and only collects profiles.
  - Profile artifacts are written under `outputs/benchmarks/serving/online_<ts>/profiles/` and indexed by `profile_manifest.json`.
- Offline: `python benchmarks/serving/offline_compare.py --profile torch|nsys|both`
  - `--profile-only` skips throughput runs and only collects profiles.
  - Profile artifacts are written under `outputs/benchmarks/serving/offline_<ts>/profiles/` and indexed by `profile_manifest.json`.

Notes:

- Nsight Systems capture uses `--capture-range=none` (captures from process start) to avoid missing CUDA kernels from early-spawned worker processes in multi-process backends.
- Torch profiling uses each backend's start/stop hooks (vLLM/SGLang/roseinfer via `/start_profile`/`/stop_profile`, TensorRT-LLM via `TLLM_PROFILE_START_STOP` + `TLLM_TORCH_PROFILE_TRACE`).
- TensorRT-LLM profiling enables NVTX annotations (`TLLM_LLMAPI_ENABLE_NVTX=1`) and GC ranges (`TLLM_PROFILE_RECORD_GC=1`); disable GC ranges with `--trtllm-no-profile-record-gc`.
