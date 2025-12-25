# Serving Benchmarks (roseinfer vs vLLM vs SGLang)

## Dependencies

These scripts assume the following Python packages are available:

- `openai`, `httpx`, `transformers`, `numpy`, `matplotlib`
- `vllm` (for vLLM runs)
- `sglang` (for SGLang runs)

## Online (OpenAI servers + trace replay)

```bash
python benchmarks/serving/online_compare.py --model gpt2 --gpu 0
```

This starts each backend's OpenAI-compatible server (one at a time), replays TraceA with a shared OpenAI client, and writes `online_results.json` under `outputs/benchmarks/serving/online_<timestamp>/`.

`online_results.json` includes `meta.versions` and wall-time info (`meta.run_start_time`, `meta.run_end_time`, `meta.wall_s`).

## Offline (token-id throughput)

```bash
python benchmarks/serving/offline_compare.py --model gpt2 --gpu 0
```

This runs each backend (one at a time) in its own process and writes `offline_results.json` under `outputs/benchmarks/serving/offline_<timestamp>/`.

Offline is aligned on token IDs (no tokenize/detokenize in the measured path):

- roseinfer: `prompt_token_ids` via `OnlineScheduler`
- vLLM: `prompt_token_ids` + `SamplingParams(detokenize=False)`
- SGLang: `Engine.generate(input_ids=...)` with `skip_tokenizer_init=True`

`offline_results.json` includes `meta.versions` and wall-time info (`meta.run_start_time`, `meta.run_end_time`, `meta.wall_s`, `meta.backend_wall_s`).

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
