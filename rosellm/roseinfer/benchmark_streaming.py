import argparse
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass
from typing import List

import torch

from rosellm.rosetrainer.hf_gpt2 import load_gpt2_from_hf_pretrained

from .engine import InferenceEngine
from .server import SchedulerManager


@dataclass(frozen=True)
class StreamResult:
    request_id: int
    submit_start: float
    submit_end: float
    tokenize_ts: float
    admit_ts: float
    first_token_ts: float
    finish_ts: float
    completion_text: str
    completion_tokens: int
    token_timestamps: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SchedulerManager streaming admission/latency.",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="gpt2",
        help="HF model ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 AMP on CUDA instead of float16.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to generate text from",
    )
    parser.add_argument(
        "--prompt-repeats",
        type=str,
        default=None,
        help=(
            "Comma-separated repeat counts for base prompt, cycled per request. "
            'Example: "1,1,1,64" produces a 3-short+1-long mix.'
        ),
    )
    parser.add_argument(
        "--unique-prompts",
        action="store_true",
        help="Append an index suffix to each prompt to avoid prefix cache hits.",
    )
    parser.add_argument(
        "--pretok",
        action="store_true",
        help="Pre-tokenize prompts and pass prompt_token_ids to SchedulerManager.add_request().",
    )
    parser.add_argument(
        "--tokenize-workers",
        type=int,
        default=0,
        help="Number of background tokenization worker threads in SchedulerManager (0: tokenize in add_request).",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=16,
        help="Number of streaming requests",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Warmup runs before measurement (for JIT / CUDA graph capture).",
    )
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=1,
        help="Number of measured runs.",
    )
    parser.add_argument(
        "--submit-interval-ms",
        type=float,
        default=0.0,
        help="Submit interval in milliseconds (0: burst).",
    )
    parser.add_argument(
        "--submit-schedule",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help=(
            "How to apply submit interval: "
            "'relative' sleeps after each submission; "
            "'absolute' targets t0 + i*interval (less sensitive to add_request overhead)."
        ),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Online scheduler max batch size (decode step)",
    )
    parser.add_argument(
        "--prefill-max-batch-size",
        type=int,
        default=None,
        help=(
            "Max pending requests to prefill per worker iteration "
            "(default: same as --max-batch-size)."
        ),
    )
    parser.add_argument(
        "--prefill-max-tokens",
        type=int,
        default=None,
        help=(
            "Max prompt tokens to prefill per worker iteration " "(default: unlimited)."
        ),
    )
    parser.add_argument(
        "--prefill-admission-policy",
        type=str,
        default="fifo",
        choices=["fifo", "pack"],
        help="Prefill admission policy (default: fifo).",
    )
    parser.add_argument(
        "--prefill-admission-lookahead",
        type=int,
        default=64,
        help="Pending lookahead window for pack admission.",
    )
    parser.add_argument(
        "--prefill-force-fifo-every",
        type=int,
        default=0,
        help="Force FIFO admission every N iterations (0: disable).",
    )
    parser.add_argument(
        "--max-active-requests",
        type=int,
        default=None,
        help="Max unfinished requests allowed in scheduler (default: unlimited).",
    )
    parser.add_argument(
        "--decode-first",
        action="store_true",
        help="Run one decode step before prefill admission when possible.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of new tokens to generate per request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling to generate text (or else greedy)",
    )
    parser.add_argument(
        "--no-stop-on-eos",
        action="store_true",
        help="Do not stop when EOS is generated",
    )
    parser.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="Disable prefix cache inside OnlineScheduler",
    )
    parser.add_argument(
        "--paged-attn",
        action="store_true",
        help="Use paged attention for decode(T=1).",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Use CUDA Graph for decode(T=1) when possible (CUDA only).",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Enable NVTX ranges (sets ROSEINFER_NVTX=1).",
    )
    return parser.parse_args()


def count_tokens(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    if len(vs) == 1:
        return vs[0]
    pos = (len(vs) - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vs[lo]
    w = pos - lo
    return vs[lo] * (1.0 - w) + vs[hi] * w


def run_once(
    *,
    engine: InferenceEngine,
    prompts: list[str],
    prompt_lens: list[int],
    prompt_token_ids_list: list[list[int]] | None,
    args: argparse.Namespace,
    record_token_timestamps: bool,
    print_summary: bool,
) -> None:
    mgr = SchedulerManager(
        engine,
        max_batch_size=int(args.max_batch_size),
        prefill_max_batch_size=args.prefill_max_batch_size,
        prefill_max_tokens=args.prefill_max_tokens,
        decode_first=args.decode_first,
        prefill_admission_policy=args.prefill_admission_policy,
        prefill_admission_lookahead=int(args.prefill_admission_lookahead),
        prefill_force_fifo_every=int(args.prefill_force_fifo_every),
        max_active_requests=args.max_active_requests,
        record_token_timestamps=record_token_timestamps,
        tokenize_workers=int(args.tokenize_workers),
    )
    if args.no_prefix_cache:
        mgr.scheduler.use_prefix_cache = False

    stop_on_eos = not args.no_stop_on_eos
    results: list[StreamResult] = []
    results_lock = threading.Lock()

    def consume_stream(
        request_id: int,
        submit_start: float,
        submit_end: float,
    ) -> None:
        first_token_ts: float | None = None
        pieces: list[str] = []
        for piece in mgr.stream_text(request_id):
            pieces.append(piece)
        finish_ts = time.perf_counter()
        tokenize_ts = mgr.pop_tokenize_timestamp(request_id)
        if tokenize_ts is None:
            tokenize_ts = submit_end
        admit_ts = mgr.pop_admit_timestamp(request_id)
        if admit_ts is None:
            admit_ts = tokenize_ts
        token_ts = mgr.pop_token_timestamps(request_id)
        if token_ts:
            first_token_ts = token_ts[0]
        if first_token_ts is None:
            first_token_ts = finish_ts
        completion_text = "".join(pieces)
        completion_tokens = len(token_ts)
        with results_lock:
            results.append(
                StreamResult(
                    request_id=request_id,
                    submit_start=submit_start,
                    submit_end=submit_end,
                    tokenize_ts=tokenize_ts,
                    admit_ts=admit_ts,
                    first_token_ts=first_token_ts,
                    finish_ts=finish_ts,
                    completion_text=completion_text,
                    completion_tokens=completion_tokens,
                    token_timestamps=token_ts,
                )
            )

    threads: list[threading.Thread] = []
    try:
        t_global0 = time.perf_counter()
        submit_interval_s = float(args.submit_interval_ms) / 1e3
        submit_schedule = str(args.submit_schedule)
        submit_lags: list[float] = []
        for i, p in enumerate(prompts):
            if submit_interval_s > 0 and submit_schedule == "absolute":
                target = t_global0 + float(i) * submit_interval_s
                now = time.perf_counter()
                if now < target:
                    time.sleep(target - now)
            prompt_token_ids = (
                prompt_token_ids_list[i] if prompt_token_ids_list is not None else None
            )
            submit_start = time.perf_counter()
            if submit_interval_s > 0 and submit_schedule == "absolute":
                target = t_global0 + float(i) * submit_interval_s
                submit_lags.append(max(0.0, submit_start - target))
            request_id = mgr.add_request(
                prompt=p,
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                stop_on_eos=stop_on_eos,
                do_sample=bool(args.do_sample),
            )
            submit_end = time.perf_counter()
            th = threading.Thread(
                target=consume_stream,
                args=(request_id, submit_start, submit_end),
                daemon=True,
            )
            threads.append(th)
            th.start()
            if submit_interval_s > 0 and submit_schedule == "relative":
                time.sleep(submit_interval_s)
        submit_wall = time.perf_counter() - t_global0
        for th in threads:
            th.join()

        if not print_summary:
            return

        add_lats = [r.submit_end - r.submit_start for r in results]
        tokenize_lats = [max(0.0, r.tokenize_ts - r.submit_end) for r in results]
        queue_waits = [max(0.0, r.admit_ts - r.tokenize_ts) for r in results]
        prefill_first = [max(0.0, r.first_token_ts - r.admit_ts) for r in results]
        ttfts = [r.first_token_ts - r.submit_start for r in results]
        totals = [r.finish_ts - r.submit_start for r in results]
        completion_tokens = [int(r.completion_tokens) for r in results]
        sum_tokens = int(sum(completion_tokens))

        tpots: list[float] = []
        itls: list[float] = []
        for r in results:
            ts = r.token_timestamps
            if len(ts) < 2:
                continue
            tpots.append((ts[-1] - ts[0]) / float(len(ts) - 1))
            for i in range(1, len(ts)):
                itls.append(ts[i] - ts[i - 1])

        start0 = min(r.submit_start for r in results) if results else 0.0
        end0 = max(r.finish_ts for r in results) if results else 0.0
        wall = max(1e-9, end0 - start0)

        print("=== streaming benchmark ===")
        print(f"Model: {args.hf_model_id}")
        print(f"Device: {args.device}")
        print(f"Pretok: {bool(args.pretok)}")
        print(f"Tokenize workers: {int(args.tokenize_workers)}")
        print(f"Paged attention: {bool(args.paged_attn)}")
        print(f"CUDA Graph: {bool(args.cuda_graph)}")
        print(f"NVTX: {bool(args.nvtx)}")
        print(f"Requests: {len(results)}")
        print(f"Prompt tokens (total): {sum(prompt_lens)}")
        print(f"Completion tokens (total): {sum_tokens}")
        print(f"Submit wall: {submit_wall:.6f} s")
        if submit_interval_s > 0:
            print(
                f"Submit interval/schedule: {float(args.submit_interval_ms):.3f} ms / {submit_schedule}"
            )
        if submit_lags:
            print(
                f"Submit lag p50/p95/p99: "
                f"{statistics.median(submit_lags)*1e3:.2f}/"
                f"{percentile(submit_lags, 95)*1e3:.2f}/"
                f"{percentile(submit_lags, 99)*1e3:.2f} ms"
            )
        print(
            f"add_request latency p50/p95/p99: "
            f"{statistics.median(add_lats)*1e3:.2f}/"
            f"{percentile(add_lats, 95)*1e3:.2f}/"
            f"{percentile(add_lats, 99)*1e3:.2f} ms"
        )
        print(
            f"Tokenize p50/p95/p99: "
            f"{statistics.median(tokenize_lats)*1e3:.2f}/"
            f"{percentile(tokenize_lats, 95)*1e3:.2f}/"
            f"{percentile(tokenize_lats, 99)*1e3:.2f} ms"
        )
        print(
            f"Queue wait (post-tok) p50/p95/p99: "
            f"{statistics.median(queue_waits)*1e3:.2f}/"
            f"{percentile(queue_waits, 95)*1e3:.2f}/"
            f"{percentile(queue_waits, 99)*1e3:.2f} ms"
        )
        print(
            f"Prefill->first token p50/p95/p99: "
            f"{statistics.median(prefill_first)*1e3:.2f}/"
            f"{percentile(prefill_first, 95)*1e3:.2f}/"
            f"{percentile(prefill_first, 99)*1e3:.2f} ms"
        )
        print(
            f"TTFT p50/p95/p99: "
            f"{statistics.median(ttfts)*1e3:.2f}/"
            f"{percentile(ttfts, 95)*1e3:.2f}/"
            f"{percentile(ttfts, 99)*1e3:.2f} ms"
        )
        tpot_p50 = statistics.median(tpots) if tpots else 0.0
        itl_p50 = statistics.median(itls) if itls else 0.0
        print(
            f"TPOT p50/p95/p99: "
            f"{tpot_p50*1e3:.2f}/"
            f"{percentile(tpots, 95)*1e3:.2f}/"
            f"{percentile(tpots, 99)*1e3:.2f} ms/token"
        )
        print(
            f"ITL p50/p95/p99: "
            f"{itl_p50*1e3:.2f}/"
            f"{percentile(itls, 95)*1e3:.2f}/"
            f"{percentile(itls, 99)*1e3:.2f} ms"
        )
        print(
            f"Latency p50/p95/p99: "
            f"{statistics.median(totals)*1e3:.2f}/"
            f"{percentile(totals, 95)*1e3:.2f}/"
            f"{percentile(totals, 99)*1e3:.2f} ms"
        )
        print(f"Throughput (completion,total): {sum_tokens / wall:.2f} tokens/s")
    finally:
        mgr.close()


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.repeat_runs <= 0:
        raise ValueError("--repeat-runs must be >= 1")
    if float(args.submit_interval_ms) < 0:
        raise ValueError("--submit-interval-ms must be >= 0")
    if int(args.tokenize_workers) < 0:
        raise ValueError("--tokenize-workers must be >= 0")
    if args.nvtx and args.device == "cuda":
        os.environ["ROSEINFER_NVTX"] = "1"
    if args.cuda_graph and not args.paged_attn:
        print("[warn] --cuda-graph is most effective with --paged-attn (decode(T=1))")
    if args.device == "cuda":
        dtype = torch.bfloat16 if args.bf16 else torch.float16
    else:
        dtype = torch.float32
    repeats: list[int] | None = None
    if args.prompt_repeats is not None:
        repeats = [int(x.strip()) for x in str(args.prompt_repeats).split(",") if x]
        if not repeats or any(r <= 0 for r in repeats):
            raise ValueError("--prompt-repeats must contain positive integers")

    prompts: list[str] = []
    for i in range(int(args.num_requests)):
        rep = repeats[i % len(repeats)] if repeats is not None else 1
        p = " ".join([args.prompt] * rep) if rep > 1 else args.prompt
        if args.unique_prompts:
            p = f"{p} [{i}]"
        prompts.append(p)
    model, cfg, tokenizer = load_gpt2_from_hf_pretrained(
        args.hf_model_id,
        device=torch.device(args.device),
        dtype=dtype,
    )
    prompt_token_ids_list: list[list[int]] | None = None
    if args.pretok:
        max_pos = int(cfg.max_position_embeddings)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        eos_id = int(eos_id) if eos_id is not None else 0
        prompt_token_ids_list = []
        for p in prompts:
            ids = tokenizer.encode(p, add_special_tokens=False)
            if not ids:
                ids = [eos_id]
            if len(ids) > max_pos:
                ids = ids[-max_pos:]
            prompt_token_ids_list.append(ids)
        prompt_lens = [len(ids) for ids in prompt_token_ids_list]
    else:
        prompt_lens = [count_tokens(tokenizer, p) for p in prompts]
    block_size = 64
    blocks_per_request = [
        math.ceil((l + int(args.max_new_tokens)) / block_size) for l in prompt_lens
    ]
    required_blocks_per_layer = sum(blocks_per_request)
    if not args.no_prefix_cache and int(args.max_new_tokens) > 0:
        if prompt_token_ids_list is None:
            unique_prompt_lens = {p: count_tokens(tokenizer, p) for p in set(prompts)}
        else:
            unique_prompt_lens = {}
            for p, ids in zip(prompts, prompt_token_ids_list):
                unique_prompt_lens.setdefault(p, len(ids))
        prefix_extra_blocks = sum(
            1 for l in unique_prompt_lens.values() if (l % block_size) != 0
        )
        required_blocks_per_layer += prefix_extra_blocks
    max_context = cfg.max_position_embeddings
    kv_cache_max_concurrency = max(
        1,
        math.ceil(required_blocks_per_layer * block_size / max_context),
    )
    engine = InferenceEngine(
        model=model,
        config=cfg,
        tokenizer=tokenizer,
        device=args.device,
        use_amp=not args.no_amp,
        bf16=args.bf16,
        kv_cache_max_concurrency=kv_cache_max_concurrency,
        prefix_cache_max_entries=len(set(prompts)),
        use_paged_attention=bool(args.paged_attn),
        use_cuda_graph=bool(args.cuda_graph),
    )
    for i in range(int(args.warmup_runs)):
        print(f"=== warmup {i + 1}/{int(args.warmup_runs)} ===")
        run_once(
            engine=engine,
            prompts=prompts,
            prompt_lens=prompt_lens,
            prompt_token_ids_list=prompt_token_ids_list,
            args=args,
            record_token_timestamps=False,
            print_summary=False,
        )
    for i in range(int(args.repeat_runs)):
        if int(args.repeat_runs) > 1:
            print(f"=== run {i + 1}/{int(args.repeat_runs)} ===")
        run_once(
            engine=engine,
            prompts=prompts,
            prompt_lens=prompt_lens,
            prompt_token_ids_list=prompt_token_ids_list,
            args=args,
            record_token_timestamps=True,
            print_summary=True,
        )


if __name__ == "__main__":
    main()
