import argparse
import math
import os
import statistics
import time
from pathlib import Path
from typing import List, Optional

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from rosellm.rosetrainer.dataset import build_tokenizer

from .engine import InferenceEngine, OfflineScheduler, OnlineScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the scheduler",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        required=True,
        help="Tokenizer name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
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
        "--prompts",
        type=str,
        nargs="+",
        help="Prompts to generate text from",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=16,
        help="Number of requests to benchmark",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
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
        "--mode",
        type=str,
        default="online",
        choices=["naive", "online", "offline", "all"],
        help="Mode to run the benchmark",
    )
    parser.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="Disable prefix cache",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiler",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="profiles",
        help="Directory to save profiler output",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs (ignored when --profile is set).",
    )
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=3,
        help="Number of measured runs (ignored when --profile is set).",
    )
    parser.add_argument(
        "--print-per-run",
        action="store_true",
        help="Print per-run stats in repeat mode.",
    )
    parser.add_argument(
        "--keep-prefix-cache-across-runs",
        action="store_true",
        help="Keep prefix cache entries between runs (default resets for fairness).",
    )
    parser.add_argument(
        "--paged-attn",
        action="store_true",
        help="Use paged attention",
    )
    return parser.parse_args()


def build_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts is not None:
        return args.prompts
    return [args.prompt for _ in range(args.num_requests)]


def count_tokens(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids)


def maybe_sync_cuda(engine: InferenceEngine) -> None:
    if engine.device.type != "cuda":
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device=engine.device)


def report_stats(
    mode: str,
    engine: InferenceEngine,
    prompts: List[str],
    outputs: List[str],
    elapsed: float,
    prefill_elapsed: Optional[float] = None,
    decode_elapsed: Optional[float] = None,
    prompt_lens: Optional[List[int]] = None,
) -> None:
    assert len(prompts) == len(outputs)
    tokenizer = engine.tokenizer
    if prompt_lens is None:
        prompt_lens = [count_tokens(tokenizer, p) for p in prompts]
    prompt_tokens = sum(prompt_lens)
    completion_tokens = 0
    for t_prompt, out in zip(prompt_lens, outputs):
        t_out = count_tokens(tokenizer, out)
        if t_out > t_prompt:
            completion_tokens += t_out - t_prompt
    total_tokens = prompt_tokens + completion_tokens
    if elapsed <= 0:
        elapsed = 1e-6
    print(f"=== {mode} ===")
    print(f"Requests: {len(prompts)}")
    if prefill_elapsed is not None and decode_elapsed is not None:
        print(f"Elapsed (prefill/add): {prefill_elapsed:.6f} seconds")
        print(f"Elapsed (decode/run): {decode_elapsed:.6f} seconds")
        print(f"Elapsed (total): {elapsed:.6f} seconds")
    else:
        print(f"Elapsed: {elapsed:.6f} seconds")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Throughput (completion,total): {completion_tokens / elapsed:.2f} tokens/s")
    if decode_elapsed is not None and decode_elapsed > 0:
        print(
            f"Throughput (completion,decode): "
            f"{completion_tokens / decode_elapsed:.2f} tokens/s"
        )
    print(f"Throughput (total,total): {total_tokens / elapsed:.2f} tokens/s")
    print()


def maybe_reset_prefix_cache(engine: InferenceEngine, args: argparse.Namespace) -> None:
    if args.no_prefix_cache:
        return
    if args.keep_prefix_cache_across_runs:
        return
    engine.prefix_cache.clear()


def summarize_runs(
    mode: str,
    results: List[dict],
    warmup_runs: int,
    repeat_runs: int,
) -> None:
    if not results:
        return
    decode_times = [r["decode_elapsed"] for r in results]
    total_times = [r["elapsed"] for r in results]
    tps_decode = [r["completion_tokens"] / r["decode_elapsed"] for r in results]
    tps_total = [r["completion_tokens"] / r["elapsed"] for r in results]
    print(f"=== {mode} summary ===")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Measured runs: {repeat_runs}")
    print(
        f"Decode time p50/mean: "
        f"{statistics.median(decode_times):.6f}/"
        f"{statistics.mean(decode_times):.6f} s"
    )
    print(
        f"Total time p50/mean: "
        f"{statistics.median(total_times):.6f}/"
        f"{statistics.mean(total_times):.6f} s"
    )
    print(
        f"Throughput(completion,decode) p50/mean: "
        f"{statistics.median(tps_decode):.2f}/"
        f"{statistics.mean(tps_decode):.2f} tokens/s"
    )
    print(
        f"Throughput(completion,total) p50/mean: "
        f"{statistics.median(tps_total):.2f}/"
        f"{statistics.mean(tps_total):.2f} tokens/s"
    )
    print()


def benchmark_naive(
    engine: InferenceEngine,
    prompts: List[str],
    prompt_lens: List[int],
    args: argparse.Namespace,
) -> None:
    outputs: List[str] = []
    stop_on_eos = not args.no_stop_on_eos
    maybe_sync_cuda(engine)
    t0 = time.perf_counter()
    for p in prompts:
        text = engine.generate(
            prompt=p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_on_eos=stop_on_eos,
            do_sample=args.do_sample,
        )
        outputs.append(text)
    maybe_sync_cuda(engine)
    t1 = time.perf_counter()
    report_stats("naive", engine, prompts, outputs, t1 - t0, prompt_lens=prompt_lens)


def benchmark_offline(
    engine: InferenceEngine,
    prompts: List[str],
    prompt_lens: List[int],
    args: argparse.Namespace,
) -> None:
    def run_once() -> tuple[List[str], float, float]:
        scheduler = OfflineScheduler(
            engine,
            use_prefix_cache=not args.no_prefix_cache,
        )
        stop_on_eos = not args.no_stop_on_eos
        request_ids: List[int] = []
        maybe_sync_cuda(engine)
        t0 = time.perf_counter()
        for p in prompts:
            rid = scheduler.add_request(
                p,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                stop_on_eos=stop_on_eos,
                do_sample=args.do_sample,
            )
            request_ids.append(rid)
        maybe_sync_cuda(engine)
        t1 = time.perf_counter()
        prefill_elapsed = t1 - t0

        maybe_sync_cuda(engine)
        t2 = time.perf_counter()
        outputs_by_id = scheduler.run()
        maybe_sync_cuda(engine)
        outputs: List[str] = []
        for rid in request_ids:
            text = outputs_by_id[rid]
            outputs.append(text)
        t3 = time.perf_counter()
        decode_elapsed = t3 - t2
        return outputs, prefill_elapsed, decode_elapsed

    if args.profile:
        scheduler = OfflineScheduler(
            engine,
            use_prefix_cache=not args.no_prefix_cache,
        )
        stop_on_eos = not args.no_stop_on_eos
        request_ids: List[int] = []
        maybe_sync_cuda(engine)
        t0 = time.perf_counter()
        for p in prompts:
            rid = scheduler.add_request(
                p,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                stop_on_eos=stop_on_eos,
                do_sample=args.do_sample,
            )
            request_ids.append(rid)
        maybe_sync_cuda(engine)
        t1 = time.perf_counter()
        prefill_elapsed = t1 - t0

        maybe_sync_cuda(engine)
        t2 = time.perf_counter()
        out_dir = Path(args.profile_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = os.fspath(out_dir / "offline_run.json")
        sched = schedule(wait=1, warmup=2, active=3, repeat=1)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
        ) as prof:
            outputs_by_id = scheduler.run()
            prof.step()
            maybe_sync_cuda(engine)
        outputs: List[str] = []
        for rid in request_ids:
            outputs.append(outputs_by_id[rid])
        t3 = time.perf_counter()
        decode_elapsed = t3 - t2
        report_stats(
            "offline",
            engine,
            prompts,
            outputs,
            prefill_elapsed + decode_elapsed,
            prefill_elapsed=prefill_elapsed,
            decode_elapsed=decode_elapsed,
            prompt_lens=prompt_lens,
        )
        prof.export_chrome_trace(trace_path)
        print(f"[profile] wrote: {trace_path}")
        return

    warmup_runs = max(0, int(args.warmup_runs))
    repeat_runs = max(1, int(args.repeat_runs))
    for _ in range(warmup_runs):
        maybe_reset_prefix_cache(engine, args)
        run_once()

    results: List[dict] = []
    for i in range(repeat_runs):
        maybe_reset_prefix_cache(engine, args)
        outputs, prefill_elapsed, decode_elapsed = run_once()
        elapsed = prefill_elapsed + decode_elapsed
        completion_tokens = 0
        for t_prompt, out in zip(prompt_lens, outputs):
            t_out = count_tokens(engine.tokenizer, out)
            completion_tokens += max(0, t_out - t_prompt)
        total_tokens = sum(prompt_lens) + completion_tokens
        results.append(
            dict(
                run=i,
                prefill_elapsed=prefill_elapsed,
                decode_elapsed=decode_elapsed,
                elapsed=elapsed,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )
        if args.print_per_run:
            report_stats(
                "offline",
                engine,
                prompts,
                outputs,
                elapsed,
                prefill_elapsed=prefill_elapsed,
                decode_elapsed=decode_elapsed,
                prompt_lens=prompt_lens,
            )
    summarize_runs("offline", results, warmup_runs, repeat_runs)


def benchmark_online(
    engine: InferenceEngine,
    prompts: List[str],
    prompt_lens: List[int],
    args: argparse.Namespace,
) -> None:
    def run_once() -> tuple[List[str], float, float]:
        scheduler = OnlineScheduler(
            engine,
            max_batch_size=args.max_batch_size,
            use_prefix_cache=not args.no_prefix_cache,
        )
        stop_on_eos = not args.no_stop_on_eos
        request_ids: List[int] = []
        maybe_sync_cuda(engine)
        t0 = time.perf_counter()
        for p in prompts:
            rid = scheduler.add_request(
                p,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                stop_on_eos=stop_on_eos,
                do_sample=args.do_sample,
            )
            request_ids.append(rid)
        maybe_sync_cuda(engine)
        t1 = time.perf_counter()
        prefill_elapsed = t1 - t0

        maybe_sync_cuda(engine)
        t2 = time.perf_counter()
        while scheduler.has_unfinished():
            scheduler.step()
        maybe_sync_cuda(engine)
        outputs: List[str] = []
        for rid in request_ids:
            outputs.append(scheduler.pop_response(rid))
        t3 = time.perf_counter()
        decode_elapsed = t3 - t2
        return outputs, prefill_elapsed, decode_elapsed

    if args.profile:
        scheduler = OnlineScheduler(
            engine,
            max_batch_size=args.max_batch_size,
            use_prefix_cache=not args.no_prefix_cache,
        )
        stop_on_eos = not args.no_stop_on_eos
        request_ids: List[int] = []
        maybe_sync_cuda(engine)
        t0 = time.perf_counter()
        for p in prompts:
            rid = scheduler.add_request(
                p,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                stop_on_eos=stop_on_eos,
                do_sample=args.do_sample,
            )
            request_ids.append(rid)
        maybe_sync_cuda(engine)
        t1 = time.perf_counter()
        prefill_elapsed = t1 - t0

        maybe_sync_cuda(engine)
        t2 = time.perf_counter()
        out_dir = Path(args.profile_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = os.fspath(out_dir / "online_decode.json")
        sched = schedule(wait=1, warmup=2, active=3, repeat=1)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
        ) as prof:
            while scheduler.has_unfinished():
                scheduler.step()
                prof.step()
            maybe_sync_cuda(engine)
        outputs: List[str] = []
        for rid in request_ids:
            outputs.append(scheduler.pop_response(rid))
        t3 = time.perf_counter()
        decode_elapsed = t3 - t2
        report_stats(
            "online",
            engine,
            prompts,
            outputs,
            prefill_elapsed + decode_elapsed,
            prefill_elapsed=prefill_elapsed,
            decode_elapsed=decode_elapsed,
            prompt_lens=prompt_lens,
        )
        prof.export_chrome_trace(trace_path)
        print(f"[profile] wrote: {trace_path}")
        return

    warmup_runs = max(0, int(args.warmup_runs))
    repeat_runs = max(1, int(args.repeat_runs))
    for _ in range(warmup_runs):
        maybe_reset_prefix_cache(engine, args)
        run_once()

    results: List[dict] = []
    for i in range(repeat_runs):
        maybe_reset_prefix_cache(engine, args)
        outputs, prefill_elapsed, decode_elapsed = run_once()
        elapsed = prefill_elapsed + decode_elapsed
        completion_tokens = 0
        for t_prompt, out in zip(prompt_lens, outputs):
            t_out = count_tokens(engine.tokenizer, out)
            completion_tokens += max(0, t_out - t_prompt)
        total_tokens = sum(prompt_lens) + completion_tokens
        results.append(
            dict(
                run=i,
                prefill_elapsed=prefill_elapsed,
                decode_elapsed=decode_elapsed,
                elapsed=elapsed,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )
        if args.print_per_run:
            report_stats(
                "online",
                engine,
                prompts,
                outputs,
                elapsed,
                prefill_elapsed=prefill_elapsed,
                decode_elapsed=decode_elapsed,
                prompt_lens=prompt_lens,
            )
    summarize_runs("online", results, warmup_runs, repeat_runs)


def main() -> None:
    args = parse_args()
    prompts = build_prompts(args)
    block_size = 64
    tokenizer = build_tokenizer(args.tokenizer_name)
    prompt_lens = [count_tokens(tokenizer, p) for p in prompts]
    blocks_per_request = [
        math.ceil((l + args.max_new_tokens) / block_size) for l in prompt_lens
    ]
    required_blocks_per_layer = sum(blocks_per_request)
    if not args.no_prefix_cache and args.max_new_tokens > 0:
        unique_prompt_lens = {p: count_tokens(tokenizer, p) for p in set(prompts)}
        prefix_extra_blocks = sum(
            1 for l in unique_prompt_lens.values() if (l % block_size) != 0
        )
        required_blocks_per_layer += prefix_extra_blocks
    ckpt = torch.load(
        args.checkpoint_path,
        map_location="meta",
        weights_only=True,
    )
    cfg_dict = ckpt.get("config") or {}
    max_context = int(cfg_dict.get("max_position_embeddings", 1024))
    kv_cache_max_concurrency = max(
        1,
        math.ceil(required_blocks_per_layer * block_size / max_context),
    )
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        use_amp=not args.no_amp,
        bf16=args.bf16,
        kv_cache_max_concurrency=kv_cache_max_concurrency,
        prefix_cache_max_entries=len(set(prompts)),
        use_paged_attention=args.paged_attn,
    )
    if args.mode in ("naive", "all"):
        benchmark_naive(engine, prompts, prompt_lens, args)
    if args.mode in ("offline", "all"):
        benchmark_offline(engine, prompts, prompt_lens, args)
    if args.mode in ("online", "all"):
        benchmark_online(engine, prompts, prompt_lens, args)


if __name__ == "__main__":
    main()
