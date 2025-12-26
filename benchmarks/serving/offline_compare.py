from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _append_pythonpath(env: dict[str, str], path: str) -> None:
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = path if not old else f"{path}{os.pathsep}{old}"


def _maybe_add_sglang_source_pythonpath_env(env: dict[str, str]) -> None:
    if _module_available("sglang") and _module_available("sgl_kernel"):
        return
    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        repo_root / ".vscode" / "sglang" / "python",
        repo_root / ".vscode" / "sglang" / "sgl-kernel" / "python",
    )
    for candidate in candidates:
        if candidate.is_dir():
            _append_pythonpath(env, str(candidate))


def _maybe_add_sglang_source_sys_path() -> None:
    if _module_available("sglang") and _module_available("sgl_kernel"):
        return
    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        repo_root / ".vscode" / "sglang" / "python",
        repo_root / ".vscode" / "sglang" / "sgl-kernel" / "python",
    )
    for candidate in reversed(candidates):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


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

    from rosellm.roseinfer.engine import (
        ChunkedOnlineScheduler,
        InferenceEngine,
        OnlineRequest,
        OnlineScheduler,
    )
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
        use_paged_attention=bool(args.roseinfer_paged_attn),
        use_cuda_graph=bool(args.roseinfer_cuda_graph),
        prefill_attn_backend=str(args.roseinfer_prefill_attn_backend),
        decode_attn_backend=str(args.roseinfer_decode_attn_backend),
        use_fused_ops=bool(args.roseinfer_fused_ops),
        use_fused_mlp=bool(args.roseinfer_fused_mlp),
        use_fused_sampler=bool(args.roseinfer_fused_sampler),
        use_fused_kv_append=bool(args.roseinfer_fused_kv_append),
    )
    prompt_token_ids = _make_prompt_token_ids(
        num_prompts=int(args.num_prompts),
        input_len=int(args.input_len),
        seed=int(args.seed),
        vocab_size=int(tokenizer.vocab_size),
    )
    use_chunked = bool(args.roseinfer_chunked_prefill)
    prefill_chunk_size = int(args.roseinfer_prefill_chunk_size)
    use_prefix_cache = bool(args.roseinfer_prefix_cache)
    if use_chunked and not bool(args.roseinfer_paged_attn):
        print("[warn] disabling roseinfer chunked prefill (requires paged attention)")
        use_chunked = False
    if use_chunked and (device.type != "cuda" or not torch.cuda.is_available()):
        print("[warn] disabling roseinfer chunked prefill (requires CUDA)")
        use_chunked = False
    if use_chunked and not use_amp:
        print("[warn] disabling roseinfer chunked prefill (requires fp16/bf16 AMP)")
        use_chunked = False
    if use_chunked and prefill_chunk_size <= 0:
        raise ValueError("--roseinfer-prefill-chunk-size must be >= 1")

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
        if use_chunked:
            scheduler = ChunkedOnlineScheduler(
                engine,
                max_batch_size=int(args.max_batch_size),
                prefill_chunk_size=prefill_chunk_size,
                use_prefix_cache=use_prefix_cache,
            )
        else:
            scheduler = OnlineScheduler(
                engine,
                max_batch_size=int(args.max_batch_size),
                use_prefix_cache=use_prefix_cache,
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

    if use_chunked:
        scheduler = ChunkedOnlineScheduler(
            engine,
            max_batch_size=int(args.max_batch_size),
            prefill_chunk_size=prefill_chunk_size,
            use_prefix_cache=use_prefix_cache,
        )
    else:
        scheduler = OnlineScheduler(
            engine,
            max_batch_size=int(args.max_batch_size),
            use_prefix_cache=use_prefix_cache,
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

    prefill_s = None if use_chunked else float(t1 - t0)
    decode_s = None if use_chunked else float(t2 - t1)
    total_s = t2 - t0
    return OfflineResult(
        backend=str(args.roseinfer_label),
        model=args.model,
        device=args.device,
        num_prompts=int(args.num_prompts),
        input_len=int(args.input_len),
        output_len=int(args.output_len),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        prefill_s=prefill_s,
        decode_s=decode_s,
        total_s=float(total_s),
        request_throughput_rps=float(int(args.num_prompts) / max(total_s, 1e-9)),
        output_throughput_tps=float(total_output_tokens / max(total_s, 1e-9)),
        total_throughput_tps=float(
            (total_input_tokens + total_output_tokens) / max(total_s, 1e-9)
        ),
        extra={
            "prefill_chunked": use_chunked,
            "prefill_chunk_size": prefill_chunk_size if use_chunked else None,
            "prefix_cache": use_prefix_cache,
            "fused_ops": bool(args.roseinfer_fused_ops),
            "fused_sampler": bool(args.roseinfer_fused_sampler),
            "fused_kv_append": bool(args.roseinfer_fused_kv_append),
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
        warmup_top_k = 0
        # vLLM (older versions) uses -1 to disable top-k.
        if warmup_top_k == 0:
            warmup_top_k = -1
        warmup = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=int(warmup_top_k),
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
    top_k = int(args.top_k)
    if top_k == 0:
        top_k = -1
    sparams = SamplingParams(
        max_tokens=output_len,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=top_k,
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
    _maybe_add_sglang_source_sys_path()
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
    if getattr(args, "sglang_attention_backend", None):
        engine_kwargs["attention_backend"] = str(args.sglang_attention_backend)
    if getattr(args, "sglang_sampling_backend", None):
        engine_kwargs["sampling_backend"] = str(args.sglang_sampling_backend)
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
        "--sglang-attention-backend",
        type=str,
        default="triton",
        choices=["flashinfer", "triton", "torch_native", "fa3", "flashmla"],
        help="SGLang attention backend (default: triton for broad compatibility).",
    )
    parser.add_argument(
        "--sglang-sampling-backend",
        type=str,
        default="flashinfer",
        choices=["flashinfer", "pytorch"],
        help="SGLang sampling backend (default: flashinfer).",
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
        "--roseinfer-prefill-attn-backend",
        type=str,
        default="auto",
        choices=["auto", "naive", "flashinfer", "flashattn"],
        help="Prefill attention backend for roseinfer (default: auto).",
    )
    parser.add_argument(
        "--roseinfer-prefill-attn-backends",
        type=str,
        default=None,
        help=(
            "Comma-separated prefill attention backends to compare for roseinfer. "
            "If set, expands 'roseinfer' in --backends (e.g. naive,flashinfer,flashattn)."
        ),
    )
    parser.add_argument(
        "--roseinfer-decode-attn-backend",
        type=str,
        default="auto",
        choices=["auto", "naive", "flashinfer", "flashattn"],
        help="Decode attention backend for roseinfer dense past_kv path (default: auto).",
    )
    parser.add_argument(
        "--roseinfer-paged-attn",
        dest="roseinfer_paged_attn",
        action="store_true",
        help="Use paged attention decode(T=1) for roseinfer (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-paged-attn",
        dest="roseinfer_paged_attn",
        action="store_false",
        help="Disable paged attention for roseinfer.",
    )
    parser.set_defaults(roseinfer_paged_attn=True)
    parser.add_argument(
        "--roseinfer-cuda-graph",
        dest="roseinfer_cuda_graph",
        action="store_true",
        help="Use CUDA graphs for paged attention decode(T=1) in roseinfer (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-cuda-graph",
        dest="roseinfer_cuda_graph",
        action="store_false",
        help="Disable CUDA graphs for roseinfer.",
    )
    parser.set_defaults(roseinfer_cuda_graph=True)
    parser.add_argument(
        "--roseinfer-chunked-prefill",
        dest="roseinfer_chunked_prefill",
        action="store_true",
        help="Enable chunked prefill for roseinfer (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-chunked-prefill",
        dest="roseinfer_chunked_prefill",
        action="store_false",
        help="Disable chunked prefill for roseinfer.",
    )
    parser.set_defaults(roseinfer_chunked_prefill=True)
    parser.add_argument(
        "--roseinfer-prefill-chunk-size",
        type=int,
        default=256,
        help="Chunk size for roseinfer chunked prefill (default: 256).",
    )
    parser.add_argument(
        "--roseinfer-prefix-cache",
        dest="roseinfer_prefix_cache",
        action="store_true",
        help="Enable prefix caching for roseinfer (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-prefix-cache",
        dest="roseinfer_prefix_cache",
        action="store_false",
        help="Disable prefix caching for roseinfer.",
    )
    parser.set_defaults(roseinfer_prefix_cache=True)
    parser.add_argument(
        "--roseinfer-fused-ops",
        dest="roseinfer_fused_ops",
        action="store_true",
        help="Enable fused ops (e.g., add+LayerNorm) for roseinfer (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-fused-ops",
        dest="roseinfer_fused_ops",
        action="store_false",
        help="Disable fused ops for roseinfer.",
    )
    parser.set_defaults(roseinfer_fused_ops=True)
    parser.add_argument(
        "--roseinfer-compare-fused-ops",
        action="store_true",
        help="Run roseinfer twice: fused ops on/off (for A/B benchmark).",
    )
    parser.add_argument(
        "--roseinfer-fused-mlp",
        dest="roseinfer_fused_mlp",
        action="store_true",
        help="Enable fused MLP epilogue for roseinfer (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-fused-mlp",
        dest="roseinfer_fused_mlp",
        action="store_false",
        help="Disable fused MLP epilogue for roseinfer.",
    )
    parser.set_defaults(roseinfer_fused_mlp=True)
    parser.add_argument(
        "--roseinfer-compare-fused-mlp",
        action="store_true",
        help="Run roseinfer twice: fused MLP on/off (for A/B benchmark).",
    )
    parser.add_argument(
        "--roseinfer-fused-sampler",
        dest="roseinfer_fused_sampler",
        action="store_true",
        help="Enable fused sampler for roseinfer (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-fused-sampler",
        dest="roseinfer_fused_sampler",
        action="store_false",
        help="Disable fused sampler for roseinfer.",
    )
    parser.set_defaults(roseinfer_fused_sampler=True)
    parser.add_argument(
        "--roseinfer-compare-fused-sampler",
        action="store_true",
        help="Run roseinfer twice: fused sampler on/off (for A/B benchmark).",
    )
    parser.add_argument(
        "--roseinfer-fused-kv-append",
        dest="roseinfer_fused_kv_append",
        action="store_true",
        help="Enable fused KV append for roseinfer (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-fused-kv-append",
        dest="roseinfer_fused_kv_append",
        action="store_false",
        help="Disable fused KV append for roseinfer.",
    )
    parser.set_defaults(roseinfer_fused_kv_append=True)
    parser.add_argument(
        "--roseinfer-compare-fused-kv-append",
        action="store_true",
        help="Run roseinfer twice: fused KV append on/off (for A/B benchmark).",
    )
    parser.add_argument(
        "--roseinfer-label",
        type=str,
        default="roseinfer",
        help="Backend label for roseinfer output (internal).",
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
    _maybe_add_sglang_source_pythonpath_env(env)

    results: list[OfflineResult] = []
    backend_wall_s: dict[str, float] = {}
    base_backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    roseinfer_prefill_backends = None
    if args.roseinfer_prefill_attn_backends:
        roseinfer_prefill_backends = [
            x.strip()
            for x in str(args.roseinfer_prefill_attn_backends).split(",")
            if x.strip()
        ]
        if not roseinfer_prefill_backends:
            roseinfer_prefill_backends = None

    @dataclass(frozen=True)
    class RunSpec:
        base_backend: str
        label: str
        roseinfer_prefill_backend: str | None = None
        roseinfer_fused_ops: bool | None = None
        roseinfer_fused_mlp: bool | None = None
        roseinfer_fused_sampler: bool | None = None
        roseinfer_fused_kv_append: bool | None = None

    run_specs: list[RunSpec] = []
    for backend in base_backends:
        if backend != "roseinfer":
            run_specs.append(RunSpec(base_backend=backend, label=backend))
            continue
        variants = roseinfer_prefill_backends or [
            str(args.roseinfer_prefill_attn_backend)
        ]
        base_fused_ops = bool(args.roseinfer_fused_ops)
        base_fused_mlp = bool(args.roseinfer_fused_mlp)
        base_fused_sampler = bool(args.roseinfer_fused_sampler)
        base_fused_kv_append = bool(args.roseinfer_fused_kv_append)
        cfgs: list[tuple[bool, bool, bool, bool]] = []
        seen: set[tuple[bool, bool, bool, bool]] = set()

        def add_cfg(
            fused_ops: bool,
            fused_mlp: bool,
            fused_sampler: bool,
            fused_kv_append: bool,
        ) -> None:
            cfg = (
                bool(fused_ops),
                bool(fused_mlp),
                bool(fused_sampler),
                bool(fused_kv_append),
            )
            if cfg in seen:
                return
            seen.add(cfg)
            cfgs.append(cfg)

        add_cfg(
            base_fused_ops, base_fused_mlp, base_fused_sampler, base_fused_kv_append
        )
        if bool(getattr(args, "roseinfer_compare_fused_ops", False)):
            add_cfg(
                (not base_fused_ops),
                base_fused_mlp,
                base_fused_sampler,
                base_fused_kv_append,
            )
        if bool(getattr(args, "roseinfer_compare_fused_mlp", False)):
            add_cfg(
                base_fused_ops,
                (not base_fused_mlp),
                base_fused_sampler,
                base_fused_kv_append,
            )
        if bool(getattr(args, "roseinfer_compare_fused_sampler", False)):
            add_cfg(
                base_fused_ops,
                base_fused_mlp,
                (not base_fused_sampler),
                base_fused_kv_append,
            )
        if bool(getattr(args, "roseinfer_compare_fused_kv_append", False)):
            add_cfg(
                base_fused_ops,
                base_fused_mlp,
                base_fused_sampler,
                (not base_fused_kv_append),
            )
        for prefill_backend in variants:
            prefill_backend = str(prefill_backend)
            if prefill_backend == "flashinfer" and not _module_available("flashinfer"):
                print(
                    "[warn] flashinfer not installed; skipping roseinfer prefill backend 'flashinfer'"
                )
                continue
            if prefill_backend == "flashattn" and not _module_available("flash_attn"):
                print(
                    "[warn] flash-attn not installed; skipping roseinfer prefill backend 'flashattn'"
                )
                continue
            base_label = (
                "roseinfer"
                if prefill_backend == "auto"
                else f"roseinfer+{prefill_backend}"
            )
            for fused_ops, fused_mlp, fused_sampler, fused_kv_append in cfgs:
                label = base_label
                if not fused_ops:
                    label += "+nofuse"
                if not fused_mlp:
                    label += "+nomlp"
                if not fused_sampler:
                    label += "+nosampler"
                if not fused_kv_append:
                    label += "+nokv"
                run_specs.append(
                    RunSpec(
                        base_backend="roseinfer",
                        label=label,
                        roseinfer_prefill_backend=prefill_backend,
                        roseinfer_fused_ops=bool(fused_ops),
                        roseinfer_fused_mlp=bool(fused_mlp),
                        roseinfer_fused_sampler=bool(fused_sampler),
                        roseinfer_fused_kv_append=bool(fused_kv_append),
                    )
                )

    if not run_specs:
        raise ValueError("no backends to run (all candidates were skipped)")

    for spec in run_specs:
        base_backend = spec.base_backend
        label = spec.label
        prefill_backend = spec.roseinfer_prefill_backend
        fused_ops = spec.roseinfer_fused_ops
        fused_mlp = spec.roseinfer_fused_mlp
        fused_sampler = spec.roseinfer_fused_sampler
        fused_kv_append = spec.roseinfer_fused_kv_append
        cmd = [
            "taskset",
            "-c",
            _format_cpu_set(server_cpus),
            sys.executable,
            str(Path(__file__).resolve()),
            "--backend",
            base_backend,
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
        if base_backend == "roseinfer":
            cmd += [
                "--roseinfer-prefill-attn-backend",
                str(prefill_backend or args.roseinfer_prefill_attn_backend),
                "--roseinfer-decode-attn-backend",
                str(args.roseinfer_decode_attn_backend),
                "--roseinfer-label",
                str(label),
            ]
            cmd.append(
                "--roseinfer-paged-attn"
                if bool(args.roseinfer_paged_attn)
                else "--roseinfer-no-paged-attn"
            )
            cmd.append(
                "--roseinfer-cuda-graph"
                if bool(args.roseinfer_cuda_graph)
                else "--roseinfer-no-cuda-graph"
            )
            cmd.append(
                "--roseinfer-chunked-prefill"
                if bool(args.roseinfer_chunked_prefill)
                else "--roseinfer-no-chunked-prefill"
            )
            cmd += [
                "--roseinfer-prefill-chunk-size",
                str(int(args.roseinfer_prefill_chunk_size)),
            ]
            cmd.append(
                "--roseinfer-prefix-cache"
                if bool(args.roseinfer_prefix_cache)
                else "--roseinfer-no-prefix-cache"
            )
            cmd.append(
                "--roseinfer-fused-ops"
                if bool(fused_ops)
                else "--roseinfer-no-fused-ops"
            )
            cmd.append(
                "--roseinfer-fused-mlp"
                if bool(fused_mlp)
                else "--roseinfer-no-fused-mlp"
            )
            cmd.append(
                "--roseinfer-fused-sampler"
                if bool(fused_sampler)
                else "--roseinfer-no-fused-sampler"
            )
            cmd.append(
                "--roseinfer-fused-kv-append"
                if bool(fused_kv_append)
                else "--roseinfer-no-fused-kv-append"
            )

        backend_t0 = time.perf_counter()
        raw = subprocess.check_output(cmd, env=env)
        backend_t1 = time.perf_counter()
        backend_wall_s[label] = float(backend_t1 - backend_t0)
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
            raise RuntimeError(f"failed to parse backend JSON output for {label}")
        results.append(OfflineResult(**parsed))
        r = results[-1]
        print(
            f"[{label}] output_throughput={r.output_throughput_tps:.2f} tok/s, "
            f"request_throughput={r.request_throughput_rps:.2f} req/s, "
            f"wall={backend_wall_s[label]:.2f}s"
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
            "roseinfer_prefill_attn_backend": str(args.roseinfer_prefill_attn_backend),
            "roseinfer_prefill_attn_backends": (
                [str(x) for x in roseinfer_prefill_backends]
                if roseinfer_prefill_backends is not None
                else None
            ),
            "roseinfer_decode_attn_backend": str(args.roseinfer_decode_attn_backend),
            "roseinfer_paged_attn": bool(args.roseinfer_paged_attn),
            "roseinfer_cuda_graph": bool(args.roseinfer_cuda_graph),
            "roseinfer_chunked_prefill": bool(args.roseinfer_chunked_prefill),
            "roseinfer_prefill_chunk_size": int(args.roseinfer_prefill_chunk_size),
            "roseinfer_prefix_cache": bool(args.roseinfer_prefix_cache),
            "roseinfer_fused_ops": bool(args.roseinfer_fused_ops),
            "roseinfer_compare_fused_ops": bool(args.roseinfer_compare_fused_ops),
            "roseinfer_fused_mlp": bool(args.roseinfer_fused_mlp),
            "roseinfer_compare_fused_mlp": bool(args.roseinfer_compare_fused_mlp),
            "roseinfer_fused_sampler": bool(args.roseinfer_fused_sampler),
            "roseinfer_compare_fused_sampler": bool(
                args.roseinfer_compare_fused_sampler
            ),
            "roseinfer_fused_kv_append": bool(args.roseinfer_fused_kv_append),
            "roseinfer_compare_fused_kv_append": bool(
                args.roseinfer_compare_fused_kv_append
            ),
            "server_cpus": server_cpus,
            "backends": [spec.label for spec in run_specs],
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
