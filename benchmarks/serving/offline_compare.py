from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class OfflineResult:
    backend: str
    model: str
    device: str
    num_prompts: int
    input_len: int
    output_len: int
    total_input_tokens: int
    total_output_tokens: int
    prefill_s: float | None
    decode_s: float | None
    total_s: float
    request_throughput_rps: float
    output_throughput_tps: float
    total_throughput_tps: float
    extra: dict[str, Any]


def _parse_cpu_set(spec: str) -> list[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s.strip())
            hi = int(hi_s.strip())
            if hi < lo:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError("empty CPU set")
    return sorted(cpus)


def _format_cpu_set(cpus: Sequence[int]) -> str:
    return ",".join(str(c) for c in cpus)


def _default_server_cpus() -> list[int]:
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        return cpus
    return cpus[: len(cpus) // 2]


def _iso_now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _try_git_rev() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", errors="replace").strip() or None
    except Exception:
        return None


def _collect_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    try:
        import importlib.metadata as md

        for pkg in (
            "rosellm",
            "vllm",
            "sglang",
            "torch",
            "transformers",
            "fastapi",
            "uvicorn",
            "openai",
            "httpx",
        ):
            try:
                versions[pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                versions[pkg] = "not installed"
    except Exception:
        pass
    git_rev = _try_git_rev()
    if git_rev:
        versions["git_rev"] = git_rev
    return versions


def _resolve_vocab_size(model_id: str) -> int:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        v = int(getattr(tok, "vocab_size", 0) or 0)
        if v > 0:
            return v
    except Exception:
        pass
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id)
        v = int(getattr(cfg, "vocab_size", 0) or 0)
        if v > 0:
            return v
    except Exception:
        pass
    # GPT-2 default.
    return 50257


def _make_prompt_token_ids(
    *,
    num_prompts: int,
    input_len: int,
    seed: int,
    vocab_size: int,
) -> list[list[int]]:
    if num_prompts <= 0:
        return []
    if input_len <= 0:
        raise ValueError("input_len must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    rng = random.Random(int(seed))
    return [
        [rng.randint(0, vocab_size - 1) for _ in range(int(input_len))]
        for _ in range(int(num_prompts))
    ]


def _dtype_flag_vllm(dtype: str) -> dict[str, Any]:
    dtype = dtype.lower()
    if dtype == "auto":
        return {}
    if dtype == "fp16":
        return {"dtype": "float16"}
    if dtype == "bf16":
        return {"dtype": "bfloat16"}
    if dtype == "fp32":
        return {"dtype": "float32"}
    raise ValueError(f"unsupported dtype for vLLM: {dtype}")


def _dtype_flag_sglang(dtype: str) -> dict[str, Any]:
    dtype = dtype.lower()
    if dtype == "auto":
        return {}
    if dtype == "fp16":
        return {"dtype": "float16"}
    if dtype == "bf16":
        return {"dtype": "bfloat16"}
    if dtype == "fp32":
        return {"dtype": "float32"}
    raise ValueError(f"unsupported dtype for SGLang: {dtype}")


def _run_roseinfer(args: argparse.Namespace) -> OfflineResult:
    import torch

    from rosellm.roseinfer.engine import InferenceEngine, OnlineRequest, OnlineScheduler
    from rosellm.rosetrainer.hf_gpt2 import load_gpt2_from_hf_pretrained

    device = torch.device(args.device)
    dtype_name = str(args.dtype).lower()
    use_amp = device.type == "cuda" and (dtype_name != "fp32") and not args.no_amp
    if use_amp:
        dtype = torch.bfloat16 if (dtype_name == "bf16" or args.bf16) else torch.float16
    else:
        dtype = torch.float32

    model, config, tokenizer = load_gpt2_from_hf_pretrained(
        args.model,
        device=device,
        dtype=dtype,
    )
    engine = InferenceEngine(
        checkpoint_path=None,
        tokenizer_name=args.model,
        device=args.device,
        use_amp=use_amp,
        bf16=args.bf16,
        model=model,
        config=config,
        tokenizer=tokenizer,
        kv_cache_max_concurrency=max(1, int(args.num_prompts)),
        prefix_cache_max_entries=256,
    )
    prompt_token_ids = _make_prompt_token_ids(
        num_prompts=int(args.num_prompts),
        input_len=int(args.input_len),
        seed=int(args.seed),
        vocab_size=int(tokenizer.vocab_size),
    )

    def make_requests(n: int, *, output_len: int) -> list[OnlineRequest]:
        reqs: list[OnlineRequest] = []
        for token_ids in prompt_token_ids[:n]:
            reqs.append(
                OnlineRequest(
                    prompt="",
                    prompt_token_ids=token_ids,
                    max_new_tokens=int(output_len),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                    stop_on_eos=not bool(args.ignore_eos),
                    do_sample=True,
                )
            )
        return reqs

    if int(args.warmup_prompts) > 0:
        scheduler = OnlineScheduler(
            engine,
            max_batch_size=int(args.max_batch_size),
            use_prefix_cache=False,
        )
        warmup_reqs = make_requests(
            min(int(args.warmup_prompts), int(args.num_prompts)),
            output_len=min(16, int(args.output_len)),
        )
        scheduler.add_requests(warmup_reqs)
        while scheduler.has_unfinished():
            scheduler.step()
        for sess in scheduler._sessions.values():  # type: ignore[attr-defined]
            sess.release_kv_blocks()

    scheduler = OnlineScheduler(
        engine,
        max_batch_size=int(args.max_batch_size),
        use_prefix_cache=False,
    )
    reqs = make_requests(int(args.num_prompts), output_len=int(args.output_len))
    total_input_tokens = sum(len(r.prompt_token_ids or []) for r in reqs)

    t0 = time.perf_counter()
    scheduler.add_requests(reqs)
    t1 = time.perf_counter()
    while scheduler.has_unfinished():
        scheduler.step()
    t2 = time.perf_counter()

    total_output_tokens = 0
    for sess in scheduler._sessions.values():  # type: ignore[attr-defined]
        total_output_tokens += len(getattr(sess, "generated_ids", []))

    for sess in scheduler._sessions.values():  # type: ignore[attr-defined]
        sess.release_kv_blocks()

    prefill_s = t1 - t0
    decode_s = t2 - t1
    total_s = t2 - t0
    return OfflineResult(
        backend="roseinfer",
        model=args.model,
        device=args.device,
        num_prompts=int(args.num_prompts),
        input_len=int(args.input_len),
        output_len=int(args.output_len),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        prefill_s=float(prefill_s),
        decode_s=float(decode_s),
        total_s=float(total_s),
        request_throughput_rps=float(int(args.num_prompts) / max(total_s, 1e-9)),
        output_throughput_tps=float(total_output_tokens / max(total_s, 1e-9)),
        total_throughput_tps=float(
            (total_input_tokens + total_output_tokens) / max(total_s, 1e-9)
        ),
        extra={
            "prefill_throughput_tps": float(total_input_tokens / max(prefill_s, 1e-9)),
            "decode_throughput_tps": float(total_output_tokens / max(decode_s, 1e-9)),
        },
    )


def _run_vllm(args: argparse.Namespace) -> OfflineResult:
    from vllm import LLM, SamplingParams

    num_prompts = int(args.num_prompts)
    input_len = int(args.input_len)
    output_len = int(args.output_len)
    vocab_size = _resolve_vocab_size(args.model)
    prompt_token_ids = _make_prompt_token_ids(
        num_prompts=num_prompts,
        input_len=input_len,
        seed=int(args.seed),
        vocab_size=vocab_size,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=int(args.tensor_parallel_size),
        trust_remote_code=True,
        **_dtype_flag_vllm(str(args.dtype)),
    )

    if not bool(args.skip_warmup) and prompt_token_ids:
        warmup = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            ignore_eos=True,
            detokenize=False,
        )
        try:
            llm.generate(
                [{"prompt_token_ids": prompt_token_ids[0]}], warmup, use_tqdm=False
            )
        except TypeError:
            llm.generate([{"prompt_token_ids": prompt_token_ids[0]}], warmup)

    prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
    sparams = SamplingParams(
        max_tokens=output_len,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        ignore_eos=bool(args.ignore_eos),
        detokenize=False,
    )

    t0 = time.perf_counter()
    try:
        outputs = llm.generate(prompts, sparams, use_tqdm=False)
    except TypeError:
        outputs = llm.generate(prompts, sparams)
    t1 = time.perf_counter()

    total_input_tokens = sum(len(ids) for ids in prompt_token_ids)
    total_output_tokens = 0
    for out in outputs:
        # vLLM returns RequestOutput objects; each has .outputs[0].token_ids
        try:
            total_output_tokens += len(out.outputs[0].token_ids)  # type: ignore[attr-defined]
        except Exception:
            pass
    total_s = t1 - t0
    return OfflineResult(
        backend="vllm",
        model=args.model,
        device="cuda",
        num_prompts=num_prompts,
        input_len=input_len,
        output_len=output_len,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        prefill_s=None,
        decode_s=None,
        total_s=float(total_s),
        request_throughput_rps=float(num_prompts / max(total_s, 1e-9)),
        output_throughput_tps=float(total_output_tokens / max(total_s, 1e-9)),
        total_throughput_tps=float(
            (total_input_tokens + total_output_tokens) / max(total_s, 1e-9)
        ),
        extra={},
    )


def _run_sglang(args: argparse.Namespace) -> OfflineResult:
    from sglang.srt.entrypoints.engine import Engine

    num_prompts = int(args.num_prompts)
    input_len = int(args.input_len)
    output_len = int(args.output_len)
    vocab_size = _resolve_vocab_size(args.model)
    prompt_token_ids = _make_prompt_token_ids(
        num_prompts=num_prompts,
        input_len=input_len,
        seed=int(args.seed),
        vocab_size=vocab_size,
    )

    engine_kwargs: dict[str, Any] = {
        "model_path": args.model,
        "tokenizer_path": args.model,
        "skip_tokenizer_init": True,
        "tp_size": int(args.tensor_parallel_size),
        "device": "cuda",
        "random_seed": int(args.seed),
        "log_level": "error",
    }
    engine_kwargs.update(_dtype_flag_sglang(str(args.dtype)))
    engine = Engine(**engine_kwargs)
    try:
        if not bool(args.skip_warmup) and prompt_token_ids:
            warm_n = min(int(args.warmup_prompts), num_prompts, 16)
            if warm_n > 0:
                warm_sampling = [
                    {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 0,
                        "max_new_tokens": min(16, output_len),
                        "ignore_eos": True,
                    }
                    for _ in range(warm_n)
                ]
                engine.generate(
                    input_ids=prompt_token_ids[:warm_n],
                    sampling_params=warm_sampling,
                )
                time.sleep(0.2)

        sampling = [
            {
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "max_new_tokens": int(output_len),
                "ignore_eos": bool(args.ignore_eos),
            }
            for _ in range(num_prompts)
        ]
        t0 = time.perf_counter()
        outputs = engine.generate(
            input_ids=prompt_token_ids,
            sampling_params=sampling,
        )
        t1 = time.perf_counter()
    finally:
        engine.shutdown()

    total_input_tokens = sum(len(ids) for ids in prompt_token_ids)
    total_output_tokens = sum(int(o["meta_info"]["completion_tokens"]) for o in outputs)
    total_s = float(t1 - t0)

    return OfflineResult(
        backend="sglang",
        model=args.model,
        device="cuda",
        num_prompts=num_prompts,
        input_len=int(args.input_len),
        output_len=int(args.output_len),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        prefill_s=None,
        decode_s=None,
        total_s=total_s,
        request_throughput_rps=float(num_prompts / max(total_s, 1e-9)),
        output_throughput_tps=float(total_output_tokens / max(total_s, 1e-9)),
        total_throughput_tps=float(
            (total_input_tokens + total_output_tokens) / max(total_s, 1e-9)
        ),
        extra={},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline throughput benchmark: roseinfer vs vLLM vs sglang.",
    )
    parser.add_argument("--model", type=str, default="gpt2", help="HF model ID.")
    parser.add_argument(
        "--device", type=str, default="cuda", help='roseinfer device: "cuda" or "cpu".'
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES for vLLM/sglang/roseinfer.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=256, help="Number of prompts."
    )
    parser.add_argument(
        "--input-len", type=int, default=512, help="Prompt length in tokens."
    )
    parser.add_argument(
        "--output-len", type=int, default=128, help="Output length in tokens."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="DType for vLLM/SGLang and roseinfer AMP (best-effort).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS to force max_tokens output.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="TP size for vLLM/sglang."
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=256,
        help="roseinfer OnlineScheduler max_batch_size.",
    )
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        default=8,
        help="Warmup prompts for roseinfer/SGLang.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip vLLM/SGLang warmup.",
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable AMP for roseinfer."
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bf16 AMP for roseinfer."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmarks/serving",
        help="Directory to store intermediate files and combined results.",
    )
    parser.add_argument(
        "--server-cpus",
        type=str,
        default=None,
        help='CPU set for offline benchmark subprocesses (e.g. "0-15"). Default: first half.',
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="roseinfer,vllm,sglang",
        help="Comma-separated backends to run in compare mode.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["roseinfer", "vllm", "sglang"],
        help="Run a single backend and print JSON to stdout (internal).",
    )
    return parser.parse_args()


def _run_single_backend(args: argparse.Namespace) -> OfflineResult:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.backend == "roseinfer":
        return _run_roseinfer(args)
    if args.backend == "vllm":
        return _run_vllm(args)
    if args.backend == "sglang":
        return _run_sglang(args)
    raise ValueError(f"unknown backend: {args.backend}")


def _run_compare(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"offline_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_start_time = _iso_now()
    run_wall_t0 = time.perf_counter()
    versions = _collect_versions()
    print(f"[meta] start={run_start_time}, versions={versions}")

    server_cpus = (
        _parse_cpu_set(args.server_cpus) if args.server_cpus else _default_server_cpus()
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = str(len(server_cpus))
    env["MKL_NUM_THREADS"] = str(len(server_cpus))

    results: list[OfflineResult] = []
    backend_wall_s: dict[str, float] = {}
    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        cmd = [
            "taskset",
            "-c",
            _format_cpu_set(server_cpus),
            sys.executable,
            str(Path(__file__).resolve()),
            "--backend",
            backend,
            "--model",
            args.model,
            "--device",
            args.device,
            "--gpu",
            args.gpu,
            "--dtype",
            str(args.dtype),
            "--num-prompts",
            str(int(args.num_prompts)),
            "--input-len",
            str(int(args.input_len)),
            "--output-len",
            str(int(args.output_len)),
            "--temperature",
            str(float(args.temperature)),
            "--top-p",
            str(float(args.top_p)),
            "--top-k",
            str(int(args.top_k)),
            "--seed",
            str(int(args.seed)),
            "--tensor-parallel-size",
            str(int(args.tensor_parallel_size)),
            "--max-batch-size",
            str(int(args.max_batch_size)),
            "--warmup-prompts",
            str(int(args.warmup_prompts)),
            "--output-dir",
            str(run_dir),
        ]
        if args.ignore_eos:
            cmd.append("--ignore-eos")
        if args.skip_warmup:
            cmd.append("--skip-warmup")
        if args.no_amp:
            cmd.append("--no-amp")
        if args.bf16:
            cmd.append("--bf16")

        backend_t0 = time.perf_counter()
        raw = subprocess.check_output(cmd, env=env)
        backend_t1 = time.perf_counter()
        backend_wall_s[backend] = float(backend_t1 - backend_t0)
        raw_text = raw.decode("utf-8", errors="replace")
        parsed: dict[str, Any] | None = None
        for line in reversed(raw_text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            raise RuntimeError(f"failed to parse backend JSON output for {backend}")
        results.append(OfflineResult(**parsed))
        r = results[-1]
        print(
            f"[{backend}] output_throughput={r.output_throughput_tps:.2f} tok/s, "
            f"request_throughput={r.request_throughput_rps:.2f} req/s, "
            f"wall={backend_wall_s[backend]:.2f}s"
        )

    run_wall_s = float(time.perf_counter() - run_wall_t0)
    run_end_time = _iso_now()

    out_json = run_dir / "offline_results.json"
    payload = {
        "meta": {
            "model": args.model,
            "device": args.device,
            "gpu": args.gpu,
            "dtype": str(args.dtype),
            "seed": int(args.seed),
            "num_prompts": int(args.num_prompts),
            "input_len": int(args.input_len),
            "output_len": int(args.output_len),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "ignore_eos": bool(args.ignore_eos),
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "max_batch_size": int(args.max_batch_size),
            "warmup_prompts": int(args.warmup_prompts),
            "skip_warmup": bool(args.skip_warmup),
            "no_amp": bool(args.no_amp),
            "bf16": bool(args.bf16),
            "server_cpus": server_cpus,
            "backends": [b.strip() for b in args.backends.split(",") if b.strip()],
            "run_start_time": run_start_time,
            "run_end_time": run_end_time,
            "wall_s": run_wall_s,
            "backend_wall_s": backend_wall_s,
            "versions": versions,
        },
        "results": [asdict(r) for r in results],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Total wall time: {run_wall_s:.2f}s")


def main() -> None:
    args = parse_args()
    if args.backend is not None:
        result = _run_single_backend(args)
        sys.stdout.write(json.dumps(asdict(result)) + "\n")
        return
    _run_compare(args)


if __name__ == "__main__":
    main()
