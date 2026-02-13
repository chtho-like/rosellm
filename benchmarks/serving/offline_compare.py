from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


def _maybe_add_trtllm_source_pythonpath_env(env: dict[str, str]) -> None:
    if _module_available("tensorrt_llm"):
        return
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / ".vscode" / "TensorRT-LLM"
    if candidate.is_dir():
        _append_pythonpath(env, str(candidate))


def _maybe_add_trtllm_source_sys_path() -> None:
    if _module_available("tensorrt_llm"):
        return
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / ".vscode" / "TensorRT-LLM"
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _resolve_trtllm_python(args: argparse.Namespace) -> str | None:
    if getattr(args, "trtllm_python", None):
        return str(args.trtllm_python)
    env_val = os.environ.get("ROSELLM_TRTLLM_PYTHON") or os.environ.get("TRTLLM_PYTHON")
    if env_val:
        return str(env_val)
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / ".venv-trtllm" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return None


def _resolve_vllm_python(args: argparse.Namespace) -> str | None:
    if getattr(args, "vllm_python", None):
        return str(args.vllm_python)
    env_val = os.environ.get("ROSELLM_VLLM_PYTHON") or os.environ.get("VLLM_PYTHON")
    if env_val:
        return str(env_val)
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / ".conda-vllm" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return None


def _resolve_sglang_python(args: argparse.Namespace) -> str | None:
    if getattr(args, "sglang_python", None):
        return str(args.sglang_python)
    env_val = os.environ.get("ROSELLM_SGLANG_PYTHON") or os.environ.get("SGLANG_PYTHON")
    if env_val:
        return str(env_val)
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / ".conda-sglang" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return None


def _trtllm_runtime_env(*, python_exe: str, base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    # NOTE: venv `python` is often a symlink to system Python; avoid `.resolve()`.
    venv_root = Path(python_exe).absolute().parent.parent
    site_pkgs = next(venv_root.glob("lib/python*/site-packages"), None)
    if site_pkgs is None:
        return env

    lib_dirs: list[str] = []
    tensorrt_libs = site_pkgs / "tensorrt_libs"
    if tensorrt_libs.is_dir():
        lib_dirs.append(str(tensorrt_libs))
    nvidia_root = site_pkgs / "nvidia"
    if nvidia_root.is_dir():
        for child in nvidia_root.iterdir():
            lib_dir = child / "lib"
            if lib_dir.is_dir():
                lib_dirs.append(str(lib_dir))

    old = env.get("LD_LIBRARY_PATH", "")
    joined = ":".join(lib_dirs)
    env["LD_LIBRARY_PATH"] = joined if not old else f"{joined}:{old}"
    return env


def _trtllm_available(*, python_exe: str | None = None) -> bool:
    cache: dict[str, bool] = getattr(_trtllm_available, "_cache", {})
    cache_key = python_exe or "<current>"
    if cache_key in cache:
        return cache[cache_key]

    ok = False
    if _module_available("tensorrt_llm"):
        ok = True
    elif python_exe:
        env = _trtllm_runtime_env(python_exe=python_exe, base_env=os.environ.copy())
        try:
            subprocess.check_output(
                [python_exe, "-c", "import tensorrt_llm"],
                env=env,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            ok = True
        except Exception:
            ok = False
    elif shutil.which("trtllm-serve") is not None:
        ok = True
    else:
        repo_root = Path(__file__).resolve().parents[2]
        candidate_root = repo_root / ".vscode" / "TensorRT-LLM"
        if candidate_root.is_dir():
            env = os.environ.copy()
            _append_pythonpath(env, str(candidate_root))
            try:
                subprocess.check_output(
                    [sys.executable, "-c", "import tensorrt_llm"],
                    env=env,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
                ok = True
            except Exception:
                ok = False

    cache[cache_key] = ok
    setattr(_trtllm_available, "_cache", cache)
    return ok


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


def _effective_warmup_prompts(
    *,
    requested: int,
    num_prompts: int,
    max_batch_size: int,
    full_batch: bool,
) -> int:
    if int(requested) <= 0:
        return 0
    warm_n = min(int(requested), int(num_prompts))
    if bool(full_batch):
        warm_n = max(warm_n, min(int(num_prompts), int(max_batch_size)))
    return int(warm_n)


def _auto_split_cpu_affinity(cpus: Sequence[int]) -> tuple[list[int], list[int]]:
    cpus = list(cpus)
    if not cpus:
        return [], []
    if len(cpus) <= 2:
        return cpus[:1], (cpus[1:] or cpus[:1])
    # For offline MP benchmarks, keep a small slice for the driver/API process and
    # give the engine process the remaining majority.
    api_ct = max(1, len(cpus) // 4)
    api_ct = min(api_ct, 4)
    api = cpus[:api_ct]
    engine = cpus[api_ct:] or cpus[:1]
    return api, engine


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


def _collect_versions(
    *,
    vllm_python: str | None = None,
    sglang_python: str | None = None,
    trtllm_python: str | None = None,
) -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    try:
        import importlib.metadata as md

        for pkg in (
            "rosellm",
            "vllm",
            "sglang",
            "tensorrt_llm",
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

    def probe_pkg_version(python_exe: str, pkg: str) -> str | None:
        try:
            out = subprocess.check_output(
                [
                    python_exe,
                    "-c",
                    (
                        "import importlib.metadata as md; "
                        f"print(md.version('{pkg}'))"
                    ),
                ],
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            probed = out.decode("utf-8", errors="replace").strip()
            return probed or None
        except Exception:
            return None

    def probe_python_version(python_exe: str) -> str | None:
        try:
            out = subprocess.check_output(
                [python_exe, "-c", "import sys; print(sys.version.split()[0])"],
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            probed = out.decode("utf-8", errors="replace").strip()
            return probed or None
        except Exception:
            return None

    if vllm_python and Path(str(vllm_python)).is_file():
        probed = probe_pkg_version(str(vllm_python), "vllm")
        if probed:
            versions["vllm"] = probed
        py_ver = probe_python_version(str(vllm_python))
        if py_ver:
            versions["vllm_python_version"] = py_ver

    if sglang_python and Path(str(sglang_python)).is_file():
        probed = probe_pkg_version(str(sglang_python), "sglang")
        if probed:
            versions["sglang"] = probed
        py_ver = probe_python_version(str(sglang_python))
        if py_ver:
            versions["sglang_python_version"] = py_ver

    if trtllm_python and Path(str(trtllm_python)).is_file():
        probed = probe_pkg_version(str(trtllm_python), "tensorrt_llm")
        if probed:
            versions["tensorrt_llm"] = probed
        py_ver = probe_python_version(str(trtllm_python))
        if py_ver:
            versions["trtllm_python_version"] = py_ver
    git_rev = _try_git_rev()
    if git_rev:
        versions["git_rev"] = git_rev
    versions["python_executable"] = sys.executable
    if vllm_python:
        versions["vllm_python"] = str(vllm_python)
    if sglang_python:
        versions["sglang_python"] = str(sglang_python)
    if trtllm_python:
        versions["trtllm_python"] = str(trtllm_python)
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


def _resolve_vllm_max_num_seqs(args: argparse.Namespace) -> int | None:
    max_num_seqs = getattr(args, "vllm_max_num_seqs", None)
    if max_num_seqs is not None:
        return int(max_num_seqs)
    vllm_attention_backend = str(getattr(args, "vllm_attention_backend", "auto")).lower()
    if vllm_attention_backend == "flashinfer":
        # vLLM v0.13.0 may OOM during sampler warmup on 12GB GPUs when
        # `attention_backend=flashinfer` and the default max_num_seqs=256.
        return 128
    return None


def _dtype_flag_trtllm(dtype: str) -> dict[str, Any]:
    dtype = dtype.lower()
    if dtype == "auto":
        return {}
    if dtype == "fp16":
        return {"dtype": "float16"}
    if dtype == "bf16":
        return {"dtype": "bfloat16"}
    if dtype == "fp32":
        return {"dtype": "float32"}
    raise ValueError(f"unsupported dtype for TensorRT-LLM: {dtype}")


@contextmanager
def _maybe_cuda_profiler(enabled: bool):
    if not enabled:
        yield
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.profiler.start()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()


@contextmanager
def _maybe_torch_profiler(enabled: bool, *, out_dir: Path, trace_name: str):
    if not enabled:
        yield
        return
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(out_dir / f"{trace_name}.trace.json"))


def _run_roseinfer(args: argparse.Namespace) -> OfflineResult:
    import torch

    from transformers import AutoConfig

    profile_mode = str(getattr(args, "profile", "none")).lower()
    if profile_mode == "both":
        raise ValueError("--profile=both is only supported in compare mode")
    torch_profile = profile_mode == "torch"
    nsys_profile = profile_mode == "nsys"
    profile_dir = getattr(args, "profile_dir", None)
    if (torch_profile or nsys_profile) and profile_dir is None:
        raise ValueError("--profile-dir is required when --profile!=none")
    profile_out_dir = (
        Path(str(profile_dir)).expanduser().resolve()
        if profile_dir is not None
        else None
    )

    from rosellm.roseinfer.engine import (
        ChunkedOnlineScheduler,
        InferenceEngine,
        OnlineRequest,
        OnlineScheduler,
    )

    device = torch.device(args.device)
    dtype_name = str(args.dtype).lower()
    use_amp = device.type == "cuda" and (dtype_name != "fp32") and not args.no_amp
    if use_amp:
        dtype = torch.bfloat16 if (dtype_name == "bf16" or args.bf16) else torch.float16
    else:
        dtype = torch.float32

    hf_cfg = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=False,
    )
    model_type = str(getattr(hf_cfg, "model_type", "") or "").lower()
    if model_type == "gpt2":
        from rosellm.rosetrainer.hf_gpt2 import load_gpt2_from_hf_pretrained

        model, config, tokenizer = load_gpt2_from_hf_pretrained(
            args.model,
            device=device,
            dtype=dtype,
        )
    elif model_type == "qwen3":
        from rosellm.rosetrainer.hf_qwen3 import load_qwen3_from_hf_pretrained

        model, config, tokenizer = load_qwen3_from_hf_pretrained(
            args.model,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(
            "unsupported roseinfer HF model_type="
            f"{model_type!r} (supported: gpt2, qwen3)"
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
        kv_cache_max_concurrency=max(
            1,
            min(int(args.num_prompts), int(args.max_batch_size)),
        ),
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
    overlap_schedule = bool(getattr(args, "roseinfer_overlap_schedule", True))
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

    warm_n = _effective_warmup_prompts(
        requested=int(args.warmup_prompts),
        num_prompts=int(args.num_prompts),
        max_batch_size=int(args.max_batch_size),
        full_batch=bool(getattr(args, "warmup_full_batch", True)),
    )
    if warm_n > 0:
        warmup_token_ids = _make_prompt_token_ids(
            num_prompts=warm_n,
            input_len=int(args.input_len),
            seed=int(args.seed) + 1,
            vocab_size=int(tokenizer.vocab_size),
        )

        if use_chunked:
            scheduler = ChunkedOnlineScheduler(
                engine,
                max_batch_size=int(args.max_batch_size),
                prefill_chunk_size=prefill_chunk_size,
                use_prefix_cache=use_prefix_cache,
                overlap_schedule=overlap_schedule,
            )
        else:
            scheduler = OnlineScheduler(
                engine,
                max_batch_size=int(args.max_batch_size),
                use_prefix_cache=use_prefix_cache,
                overlap_schedule=overlap_schedule,
            )
        warmup_reqs: list[OnlineRequest] = []
        for token_ids in warmup_token_ids:
            warmup_reqs.append(
                OnlineRequest(
                    prompt="",
                    prompt_token_ids=token_ids,
                    max_new_tokens=min(16, int(args.output_len)),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                    stop_on_eos=not bool(args.ignore_eos),
                    do_sample=True,
                )
            )
        scheduler.add_requests(warmup_reqs)
        while scheduler.has_unfinished():
            scheduler.step()
        for sess in scheduler._sessions.values():  # type: ignore[attr-defined]
            sess.release_kv_blocks()
        if use_prefix_cache:
            engine.prefix_cache.clear()

    if use_chunked:
        scheduler = ChunkedOnlineScheduler(
            engine,
            max_batch_size=int(args.max_batch_size),
            prefill_chunk_size=prefill_chunk_size,
            use_prefix_cache=use_prefix_cache,
            overlap_schedule=overlap_schedule,
        )
    else:
        scheduler = OnlineScheduler(
            engine,
            max_batch_size=int(args.max_batch_size),
            use_prefix_cache=use_prefix_cache,
            overlap_schedule=overlap_schedule,
        )
    reqs = make_requests(int(args.num_prompts), output_len=int(args.output_len))
    total_input_tokens = sum(len(r.prompt_token_ids or []) for r in reqs)

    with _maybe_torch_profiler(
        torch_profile,
        out_dir=profile_out_dir or Path("."),
        trace_name="trace",
    ), _maybe_cuda_profiler(nsys_profile):
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
            "python_executable": sys.executable,
            "prefill_chunked": use_chunked,
            "prefill_chunk_size": prefill_chunk_size if use_chunked else None,
            "prefix_cache": use_prefix_cache,
            "fused_ops": bool(args.roseinfer_fused_ops),
            "fused_sampler": bool(args.roseinfer_fused_sampler),
            "fused_kv_append": bool(args.roseinfer_fused_kv_append),
            "overlap_schedule": overlap_schedule,
        },
    )


def _run_roseinfer_mp(args: argparse.Namespace) -> OfflineResult:
    import importlib.util

    import torch

    orig_affinity: set[int] | None = None
    try:
        orig_affinity = set(os.sched_getaffinity(0))
    except Exception:
        orig_affinity = None

    orig_omp = os.environ.get("OMP_NUM_THREADS")
    orig_mkl = os.environ.get("MKL_NUM_THREADS")
    orig_torch_threads = torch.get_num_threads()
    try:
        orig_torch_interop_threads = torch.get_num_interop_threads()
    except Exception:
        orig_torch_interop_threads = None

    profile_mode = str(getattr(args, "profile", "none")).lower()
    if profile_mode == "both":
        raise ValueError("--profile=both is only supported in compare mode")
    torch_profile = profile_mode == "torch"
    nsys_profile = profile_mode == "nsys"
    profile_dir = getattr(args, "profile_dir", None)
    if (torch_profile or nsys_profile) and profile_dir is None:
        raise ValueError("--profile-dir is required when --profile!=none")
    profile_out_dir = (
        Path(str(profile_dir)).expanduser().resolve()
        if profile_dir is not None
        else None
    )

    from rosellm.roseinfer.mp import EngineProcessArgs, MPTokenManager

    device = torch.device(args.device)
    dtype_name = str(args.dtype).lower()
    use_amp = device.type == "cuda" and (dtype_name != "fp32") and not args.no_amp
    if device.type != "cuda" or not torch.cuda.is_available():
        if bool(args.roseinfer_paged_attn):
            print("[warn] disabling roseinfer paged attention (requires CUDA)")
        if bool(args.roseinfer_cuda_graph):
            print("[warn] disabling roseinfer cuda graphs (requires CUDA)")
        if bool(args.roseinfer_chunked_prefill):
            print("[warn] disabling roseinfer chunked prefill (requires CUDA)")
    use_chunked = bool(args.roseinfer_chunked_prefill)
    prefill_chunk_size = int(args.roseinfer_prefill_chunk_size)
    use_prefix_cache = bool(args.roseinfer_prefix_cache)
    overlap_schedule = bool(getattr(args, "roseinfer_overlap_schedule", True))
    if use_chunked and not bool(args.roseinfer_paged_attn):
        print("[warn] disabling roseinfer chunked prefill (requires paged attention)")
        use_chunked = False
    if use_chunked and (device.type != "cuda" or not torch.cuda.is_available()):
        use_chunked = False
    if use_chunked and not use_amp:
        print("[warn] disabling roseinfer chunked prefill (requires fp16/bf16 AMP)")
        use_chunked = False
    if use_chunked and importlib.util.find_spec("flashinfer") is None:
        print("[warn] disabling roseinfer chunked prefill (requires flashinfer)")
        use_chunked = False
    if use_chunked and prefill_chunk_size <= 0:
        raise ValueError("--roseinfer-prefill-chunk-size must be >= 1")

    mp_ipc = str(getattr(args, "roseinfer_mp_ipc", "pipe")).lower()
    mp_batch_send = bool(getattr(args, "roseinfer_mp_batch_send", True))
    mp_thread_cap = bool(getattr(args, "roseinfer_mp_thread_cap", True))
    mp_affinity = bool(getattr(args, "roseinfer_mp_affinity", True))
    mp_max_recv_per_iter = int(getattr(args, "roseinfer_mp_max_recv_per_iter", 64))
    mp_fill_target = bool(getattr(args, "roseinfer_mp_fill_target", True))
    mp_kv_cache_max_concurrency = int(
        getattr(args, "roseinfer_mp_kv_cache_max_concurrency", 0)
    )
    if mp_kv_cache_max_concurrency <= 0:
        # Default to the effective concurrent batch, not the total prompt count,
        # to avoid over-allocating KV cache memory.
        mp_kv_cache_max_concurrency = max(
            1, min(int(args.max_batch_size), int(args.num_prompts))
        )
    mp_flat_events = bool(getattr(args, "roseinfer_mp_flat_events", True))
    mp_finish_only = bool(getattr(args, "roseinfer_mp_finish_only", True))
    mp_fast_finish_counts = bool(getattr(args, "roseinfer_mp_fast_finish_counts", True))

    api_cpus: list[int] | None = None
    engine_cpus: list[int] | None = None
    if mp_affinity:
        available = sorted(os.sched_getaffinity(0))
        api_cpus, engine_cpus = _auto_split_cpu_affinity(available)

    engine_torch_threads: int | None = None
    if mp_thread_cap and device.type == "cuda":
        engine_torch_threads = 1
    if engine_torch_threads is not None:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    engine_args = EngineProcessArgs(
        checkpoint_path=None,
        hf_model_id=args.model,
        tokenizer_name=args.model,
        device=args.device,
        no_amp=bool(args.no_amp),
        bf16=bool(args.bf16),
        prefill_attn_backend=str(args.roseinfer_prefill_attn_backend),
        decode_attn_backend=str(args.roseinfer_decode_attn_backend),
        paged_attn=bool(args.roseinfer_paged_attn),
        cuda_graph=bool(args.roseinfer_cuda_graph),
        fused_ops=bool(args.roseinfer_fused_ops),
        fused_mlp=bool(args.roseinfer_fused_mlp),
        fused_sampler=bool(args.roseinfer_fused_sampler),
        fused_kv_append=bool(args.roseinfer_fused_kv_append),
        chunked_prefill=bool(use_chunked),
        prefill_chunk_size=int(prefill_chunk_size),
        prefix_cache=bool(use_prefix_cache),
        overlap_schedule=bool(overlap_schedule),
        max_batch_size=int(args.max_batch_size),
        kv_cache_max_concurrency=int(mp_kv_cache_max_concurrency),
        prefix_cache_max_entries=256,
        mp_torch_num_threads=engine_torch_threads,
        mp_torch_num_interop_threads=engine_torch_threads,
        mp_cpu_affinity=tuple(engine_cpus) if engine_cpus else None,
        mp_fill_target=bool(mp_fill_target),
        mp_max_recv_per_iter=int(mp_max_recv_per_iter),
        mp_emit_token_events=not bool(mp_finish_only),
        mp_flat_events=bool(mp_flat_events),
        mp_fast_finish_counts=bool(mp_fast_finish_counts),
    )

    vocab_size = _resolve_vocab_size(args.model)
    prompt_token_ids = _make_prompt_token_ids(
        num_prompts=int(args.num_prompts),
        input_len=int(args.input_len),
        seed=int(args.seed),
        vocab_size=int(vocab_size),
    )
    total_input_tokens = sum(len(ids) for ids in prompt_token_ids)

    mgr = MPTokenManager(
        engine_args=engine_args, ipc_mode=mp_ipc, start_timeout_s=600.0
    )
    try:
        if api_cpus:
            try:
                os.sched_setaffinity(0, set(int(c) for c in api_cpus))
            except Exception:
                pass

        warm_n = _effective_warmup_prompts(
            requested=int(args.warmup_prompts),
            num_prompts=int(args.num_prompts),
            max_batch_size=int(args.max_batch_size),
            full_batch=bool(getattr(args, "warmup_full_batch", True)),
        )
        if warm_n > 0 and prompt_token_ids:
            warm_len = min(16, int(args.output_len))
            warm_prompt_token_ids = _make_prompt_token_ids(
                num_prompts=warm_n,
                input_len=int(args.input_len),
                seed=int(args.seed) + 1,
                vocab_size=int(vocab_size),
            )
            if mp_batch_send:
                warm_rids = mgr.add_requests_token_ids(
                    warm_prompt_token_ids,
                    max_new_tokens=warm_len,
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                    stop_on_eos=not bool(args.ignore_eos),
                    do_sample=True,
                )
            else:
                warm_rids = [
                    mgr.add_request_token_ids(
                        ids,
                        max_new_tokens=warm_len,
                        temperature=float(args.temperature),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        stop_on_eos=not bool(args.ignore_eos),
                        do_sample=True,
                    )
                    for ids in warm_prompt_token_ids
                ]
            mgr.wait_finished(warm_rids, timeout_s=300.0)

        if torch_profile:
            assert profile_out_dir is not None
            mgr.start_profile(tool="torch", output_dir=str(profile_out_dir))
        if nsys_profile:
            mgr.start_profile(tool="cuda")
        t0 = time.perf_counter()
        if mp_batch_send:
            rids = mgr.add_requests_token_ids(
                prompt_token_ids,
                max_new_tokens=int(args.output_len),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                stop_on_eos=not bool(args.ignore_eos),
                do_sample=True,
            )
        else:
            rids = [
                mgr.add_request_token_ids(
                    ids,
                    max_new_tokens=int(args.output_len),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                    stop_on_eos=not bool(args.ignore_eos),
                    do_sample=True,
                )
                for ids in prompt_token_ids
            ]
        counts = mgr.wait_finished(rids, timeout_s=600.0)
        t1 = time.perf_counter()
        if nsys_profile:
            mgr.stop_profile(tool="cuda")
        if torch_profile:
            mgr.stop_profile(tool="torch")
    finally:
        mgr.close()
        if orig_affinity is not None:
            try:
                os.sched_setaffinity(0, orig_affinity)
            except Exception:
                pass
        if orig_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = orig_omp
        if orig_mkl is None:
            os.environ.pop("MKL_NUM_THREADS", None)
        else:
            os.environ["MKL_NUM_THREADS"] = orig_mkl
        try:
            torch.set_num_threads(int(orig_torch_threads))
        except Exception:
            pass
        if orig_torch_interop_threads is not None:
            try:
                torch.set_num_interop_threads(int(orig_torch_interop_threads))
            except Exception:
                pass

    total_output_tokens = int(sum(counts.values()))
    total_s = float(t1 - t0)
    return OfflineResult(
        backend=str(args.roseinfer_label),
        model=args.model,
        device=args.device,
        num_prompts=int(args.num_prompts),
        input_len=int(args.input_len),
        output_len=int(args.output_len),
        total_input_tokens=int(total_input_tokens),
        total_output_tokens=int(total_output_tokens),
        prefill_s=None,
        decode_s=None,
        total_s=float(total_s),
        request_throughput_rps=float(int(args.num_prompts) / max(total_s, 1e-9)),
        output_throughput_tps=float(total_output_tokens / max(total_s, 1e-9)),
        total_throughput_tps=float(
            (total_input_tokens + total_output_tokens) / max(total_s, 1e-9)
        ),
        extra={
            "python_executable": sys.executable,
            "mp": True,
            "mp_ipc": mp_ipc,
            "mp_batch_send": bool(mp_batch_send),
            "mp_thread_cap": bool(mp_thread_cap),
            "mp_affinity": bool(mp_affinity),
            "mp_max_recv_per_iter": int(mp_max_recv_per_iter),
            "mp_fill_target": bool(mp_fill_target),
            "kv_cache_max_concurrency": int(mp_kv_cache_max_concurrency),
            "mp_flat_events": bool(mp_flat_events),
            "mp_finish_only": bool(mp_finish_only),
            "mp_fast_finish_counts": bool(mp_fast_finish_counts),
            "prefill_chunked": use_chunked,
            "prefill_chunk_size": prefill_chunk_size if use_chunked else None,
            "prefix_cache": use_prefix_cache,
            "fused_ops": bool(args.roseinfer_fused_ops),
            "fused_sampler": bool(args.roseinfer_fused_sampler),
            "fused_kv_append": bool(args.roseinfer_fused_kv_append),
            "overlap_schedule": overlap_schedule,
        },
    )


def _run_vllm(args: argparse.Namespace) -> OfflineResult:
    from vllm import LLM, SamplingParams

    profile_mode = str(getattr(args, "profile", "none")).lower()
    if profile_mode == "both":
        raise ValueError("--profile=both is only supported in compare mode")
    torch_profile = profile_mode == "torch"
    nsys_profile = profile_mode == "nsys"
    profile_dir = getattr(args, "profile_dir", None)
    if (torch_profile or nsys_profile) and profile_dir is None:
        raise ValueError("--profile-dir is required when --profile!=none")
    profile_out_dir = (
        Path(str(profile_dir)).expanduser().resolve()
        if profile_dir is not None
        else None
    )

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

    vllm_async_scheduling = bool(getattr(args, "vllm_async_scheduling", True))
    vllm_attention_backend = str(getattr(args, "vllm_attention_backend", "auto"))
    llm_kwargs: dict[str, Any] = dict(
        model=args.model,
        tensor_parallel_size=int(args.tensor_parallel_size),
        trust_remote_code=True,
        async_scheduling=vllm_async_scheduling,
        **_dtype_flag_vllm(str(args.dtype)),
    )
    max_seq_len = max(16, input_len + output_len + 8)
    try:
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        max_pos = int(
            getattr(
                hf_cfg,
                "max_position_embeddings",
                getattr(hf_cfg, "n_positions", 0),
            )
            or 0
        )
    except Exception:
        max_pos = 0
    if max_pos >= 8192:
        # Long-context models (e.g. Qwen3) may default to 32k/64k context lengths.
        # For this offline benchmark (fixed input/output lengths), clamp vLLM's
        # max_model_len to avoid allocating an enormous KV cache that can OOM on
        # small GPUs.
        llm_kwargs["max_model_len"] = int(max_seq_len)
    if vllm_attention_backend and vllm_attention_backend.lower() != "auto":
        llm_kwargs["attention_backend"] = vllm_attention_backend
    vllm_max_num_seqs = _resolve_vllm_max_num_seqs(args)
    if vllm_max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = int(vllm_max_num_seqs)
    vllm_async_scheduling_supported = True
    vllm_attention_backend_supported = True
    try:
        llm = LLM(**llm_kwargs)
    except TypeError as first_err:
        # Keep some compatibility with older vLLM versions that may not support
        # newer EngineArgs/LLM keyword arguments.
        base_kwargs = dict(llm_kwargs)
        attempts: list[tuple[dict[str, Any], bool, bool]] = []

        if "max_model_len" in base_kwargs:
            no_len = dict(base_kwargs)
            no_len.pop("max_model_len", None)
            attempts.append((no_len, True, True))
        if "attention_backend" in base_kwargs:
            no_attn = dict(base_kwargs)
            no_attn.pop("attention_backend", None)
            attempts.append((no_attn, False, True))
        if "async_scheduling" in base_kwargs:
            no_async = dict(base_kwargs)
            no_async.pop("async_scheduling", None)
            attempts.append((no_async, True, False))
        if "attention_backend" in base_kwargs and "async_scheduling" in base_kwargs:
            no_both = dict(base_kwargs)
            no_both.pop("attention_backend", None)
            no_both.pop("async_scheduling", None)
            attempts.append((no_both, False, False))
        if "max_model_len" in base_kwargs and "attention_backend" in base_kwargs:
            no_len_attn = dict(base_kwargs)
            no_len_attn.pop("max_model_len", None)
            no_len_attn.pop("attention_backend", None)
            attempts.append((no_len_attn, False, True))
        if "max_model_len" in base_kwargs and "async_scheduling" in base_kwargs:
            no_len_async = dict(base_kwargs)
            no_len_async.pop("max_model_len", None)
            no_len_async.pop("async_scheduling", None)
            attempts.append((no_len_async, True, False))
        if (
            "max_model_len" in base_kwargs
            and "attention_backend" in base_kwargs
            and "async_scheduling" in base_kwargs
        ):
            no_len_both = dict(base_kwargs)
            no_len_both.pop("max_model_len", None)
            no_len_both.pop("attention_backend", None)
            no_len_both.pop("async_scheduling", None)
            attempts.append((no_len_both, False, False))

        last_err: TypeError | None = None
        for kwargs, attn_ok, async_ok in attempts:
            try:
                llm = LLM(**kwargs)
                vllm_attention_backend_supported = attn_ok
                vllm_async_scheduling_supported = async_ok
                break
            except TypeError as e:
                last_err = e
                continue
        else:
            raise (last_err or first_err)

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

    with _maybe_torch_profiler(
        torch_profile,
        out_dir=profile_out_dir or Path("."),
        trace_name="trace",
    ), _maybe_cuda_profiler(nsys_profile):
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
        extra={
            "python_executable": sys.executable,
            "async_scheduling": vllm_async_scheduling,
            "async_scheduling_supported": vllm_async_scheduling_supported,
            "attention_backend": vllm_attention_backend,
            "attention_backend_supported": vllm_attention_backend_supported,
            "max_num_seqs": vllm_max_num_seqs,
        },
    )


def _run_sglang(args: argparse.Namespace) -> OfflineResult:
    _maybe_add_sglang_source_sys_path()
    from sglang.srt.entrypoints.engine import Engine

    profile_mode = str(getattr(args, "profile", "none")).lower()
    if profile_mode == "both":
        raise ValueError("--profile=both is only supported in compare mode")
    torch_profile = profile_mode == "torch"
    nsys_profile = profile_mode == "nsys"
    profile_dir = getattr(args, "profile_dir", None)
    if (torch_profile or nsys_profile) and profile_dir is None:
        raise ValueError("--profile-dir is required when --profile!=none")
    profile_out_dir = (
        Path(str(profile_dir)).expanduser().resolve()
        if profile_dir is not None
        else None
    )

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
        if torch_profile:
            assert profile_out_dir is not None
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                engine.tokenizer_manager.start_profile(
                    output_dir=str(profile_out_dir),
                    activities=["CPU", "GPU"],
                )
            )
        elif nsys_profile:
            assert profile_out_dir is not None
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                engine.tokenizer_manager.start_profile(
                    output_dir=str(profile_out_dir),
                    activities=["CUDA_PROFILER"],
                )
            )

        t0 = time.perf_counter()
        outputs = engine.generate(
            input_ids=prompt_token_ids,
            sampling_params=sampling,
        )
        t1 = time.perf_counter()

        if torch_profile or nsys_profile:
            engine.tokenizer_manager.stop_profile()
            if torch_profile:
                assert profile_out_dir is not None
                deadline = time.time() + 30.0
                while time.time() < deadline:
                    if any(profile_out_dir.rglob("*.trace.json.gz")):
                        break
                    time.sleep(0.25)
            else:
                time.sleep(0.25)
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
        extra={"python_executable": sys.executable},
    )


def _run_trtllm(args: argparse.Namespace) -> OfflineResult:
    _maybe_add_trtllm_source_sys_path()
    from tensorrt_llm import SamplingParams
    from tensorrt_llm.commands.serve import get_llm_args
    from transformers import AutoTokenizer

    profile_mode = str(getattr(args, "profile", "none")).lower()
    if profile_mode == "both":
        raise ValueError("--profile=both is only supported in compare mode")
    torch_profile = profile_mode == "torch"
    nsys_profile = profile_mode == "nsys"
    profile_dir = getattr(args, "profile_dir", None)
    if (torch_profile or nsys_profile) and profile_dir is None:
        raise ValueError("--profile-dir is required when --profile!=none")
    profile_out_dir = (
        Path(str(profile_dir)).expanduser().resolve()
        if profile_dir is not None
        else None
    )

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
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eos_id = tok.eos_token_id
    if eos_id is None:
        eos_id = max(0, int(vocab_size) - 1)
    pad_id = tok.pad_token_id

    backend = str(getattr(args, "trtllm_backend", "tensorrt")).lower()
    max_seq_len = max(16, input_len + output_len + 8)
    max_num_tokens = max(1, int(args.max_batch_size)) * int(max_seq_len)
    llm_args, _ = get_llm_args(
        model=args.model,
        tokenizer=None,
        backend=backend,
        max_batch_size=int(args.max_batch_size),
        max_num_tokens=int(max_num_tokens),
        max_seq_len=int(max_seq_len),
        tensor_parallel_size=int(args.tensor_parallel_size),
        trust_remote_code=True,
    )
    llm_args["skip_tokenizer_init"] = True
    llm_args.update(_dtype_flag_trtllm(str(args.dtype)))
    if backend == "pytorch":
        from tensorrt_llm import LLM as PyTorchLLM

        llm = PyTorchLLM(**llm_args)
    elif backend in ("tensorrt", "trt"):
        from tensorrt_llm._tensorrt_engine import LLM as TrtLLM

        llm_args.pop("backend", None)
        llm = TrtLLM(**llm_args)
    else:
        raise ValueError(f"unsupported TensorRT-LLM backend: {backend}")
    try:
        if not bool(args.skip_warmup) and prompt_token_ids:
            warmup = SamplingParams(
                end_id=int(eos_id),
                pad_id=(int(pad_id) if pad_id is not None else None),
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                ignore_eos=True,
                detokenize=False,
            )
            llm.generate(
                [{"prompt_token_ids": prompt_token_ids[0]}],
                warmup,
                use_tqdm=False,
            )

        prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
        top_k_opt: int | None = int(args.top_k)
        if top_k_opt <= 0:
            top_k_opt = None
        sparams = SamplingParams(
            end_id=int(eos_id),
            pad_id=(int(pad_id) if pad_id is not None else None),
            max_tokens=output_len,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=top_k_opt,
            ignore_eos=bool(args.ignore_eos),
            detokenize=False,
        )
        with _maybe_torch_profiler(
            torch_profile,
            out_dir=profile_out_dir or Path("."),
            trace_name="trace",
        ), _maybe_cuda_profiler(nsys_profile):
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sparams, use_tqdm=False)
            t1 = time.perf_counter()
    finally:
        llm.shutdown()

    total_input_tokens = sum(len(ids) for ids in prompt_token_ids)
    total_output_tokens = 0
    for out in outputs if isinstance(outputs, list) else [outputs]:
        try:
            total_output_tokens += len(out.outputs[0].token_ids)  # type: ignore[attr-defined]
        except Exception:
            pass
    total_s = float(t1 - t0)

    return OfflineResult(
        backend="trtllm",
        model=args.model,
        device="cuda",
        num_prompts=num_prompts,
        input_len=input_len,
        output_len=output_len,
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
        extra={"python_executable": sys.executable},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline throughput benchmark: roseinfer vs vLLM vs sglang vs TensorRT-LLM.",
    )
    parser.add_argument("--model", type=str, default="gpt2", help="HF model ID.")
    parser.add_argument(
        "--device", type=str, default="cuda", help='roseinfer device: "cuda" or "cpu".'
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES for vLLM/sglang/trtllm/roseinfer.",
    )
    parser.add_argument(
        "--sglang-attention-backend",
        type=str,
        default="triton",
        choices=["flashinfer", "triton", "torch_native", "fa3", "flashmla"],
        help="SGLang attention backend (default: triton).",
    )
    parser.add_argument(
        "--sglang-sampling-backend",
        type=str,
        default="flashinfer",
        choices=["flashinfer", "pytorch"],
        help="SGLang sampling backend (default: flashinfer).",
    )
    parser.add_argument(
        "--vllm-python",
        type=str,
        default=None,
        help=(
            "Python executable for the vLLM backend. "
            "Default: use current interpreter or auto-detect `./.conda-vllm/bin/python`."
        ),
    )
    parser.add_argument(
        "--vllm-attention-backend",
        type=str,
        default="flashinfer",
        choices=["auto", "flashinfer", "flash_attn", "triton_attn"],
        help="vLLM attention backend (default: flashinfer).",
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=None,
        help=(
            "vLLM max_num_seqs. Default: use vLLM default; if unset and "
            "--vllm-attention-backend=flashinfer, auto-set to 128 to avoid OOM on 12GB GPUs."
        ),
    )
    parser.add_argument(
        "--vllm-async-scheduling",
        dest="vllm_async_scheduling",
        action="store_true",
        help=(
            "vLLM: enable async scheduling (overlap CPU scheduling with GPU work) "
            "for a more apples-to-apples comparison vs sglang/TRT-LLM (default: enabled)."
        ),
    )
    parser.add_argument(
        "--vllm-no-async-scheduling",
        dest="vllm_async_scheduling",
        action="store_false",
        help="vLLM: disable async scheduling.",
    )
    parser.set_defaults(vllm_async_scheduling=True)
    parser.add_argument(
        "--sglang-python",
        type=str,
        default=None,
        help=(
            "Python executable for the SGLang backend. "
            "Default: use current interpreter or auto-detect `./.conda-sglang/bin/python`."
        ),
    )
    parser.add_argument(
        "--trtllm-python",
        type=str,
        default=None,
        help=(
            "Python executable for the TensorRT-LLM backend. "
            "Default: auto-detect `./.venv-trtllm/bin/python` or use current interpreter."
        ),
    )
    parser.add_argument(
        "--trtllm-backend",
        type=str,
        default="tensorrt",
        choices=["pytorch", "tensorrt", "trt"],
        help="TensorRT-LLM backend for offline benchmark (default: tensorrt).",
    )
    parser.add_argument(
        "--trtllm-worker-single-process",
        action="store_true",
        help=(
            "TensorRT-LLM: set env TLLM_WORKER_USE_SINGLE_PROCESS=1 for TP=1 "
            "(may hurt streaming performance; default: disabled)."
        ),
    )
    parser.add_argument(
        "--trtllm-profile-record-gc",
        dest="trtllm_profile_record_gc",
        action="store_true",
        help="TensorRT-LLM profiling: annotate Python GC with NVTX ranges (default: enabled).",
    )
    parser.add_argument(
        "--trtllm-no-profile-record-gc",
        dest="trtllm_profile_record_gc",
        action="store_false",
        help="TensorRT-LLM profiling: disable GC NVTX ranges.",
    )
    parser.set_defaults(trtllm_profile_record_gc=True)
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
        help="DType for vLLM/SGLang/TensorRT-LLM and roseinfer AMP (best-effort).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling.")
    ignore_eos_group = parser.add_mutually_exclusive_group()
    ignore_eos_group.add_argument(
        "--ignore-eos",
        dest="ignore_eos",
        action="store_true",
        help="Ignore EOS early-stop (default: enabled for offline throughput).",
    )
    ignore_eos_group.add_argument(
        "--disable-ignore-eos",
        dest="ignore_eos",
        action="store_false",
        help=(
            "Enable EOS early-stop (not recommended for offline throughput numbers; "
            "output length becomes data-dependent)."
        ),
    )
    parser.set_defaults(ignore_eos=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="TP size for vLLM/sglang/trtllm.",
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
        "--warmup-full-batch",
        dest="warmup_full_batch",
        action="store_true",
        help=(
            "Ensure warmup covers at least one full batch "
            "(>= min(num_prompts, max_batch_size)) to avoid JIT skew "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-warmup-full-batch",
        dest="warmup_full_batch",
        action="store_false",
        help="Warm up only --warmup-prompts prompts (may include JIT in timed runs).",
    )
    parser.set_defaults(warmup_full_batch=True)
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
        choices=["auto", "auto2", "naive", "flashinfer", "flashattn"],
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
        "--roseinfer-overlap-schedule",
        dest="roseinfer_overlap_schedule",
        action="store_true",
        help="Enable CPU/GPU overlap scheduling for roseinfer (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-overlap-schedule",
        dest="roseinfer_overlap_schedule",
        action="store_false",
        help="Disable CPU/GPU overlap scheduling for roseinfer.",
    )
    parser.set_defaults(roseinfer_overlap_schedule=True)
    parser.add_argument(
        "--roseinfer-compare-overlap-schedule",
        action="store_true",
        help="Run roseinfer twice: overlap scheduling on/off (for A/B benchmark).",
    )
    parser.add_argument(
        "--roseinfer-mp-ipc",
        type=str,
        default="pipe",
        help="roseinfer_mp: IPC transport between API and engine: queue|pipe (default: pipe).",
    )
    parser.add_argument(
        "--roseinfer-mp-batch-send",
        dest="roseinfer_mp_batch_send",
        action="store_true",
        help="roseinfer_mp: send prompt batches in a single IPC command (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-mp-batch-send",
        dest="roseinfer_mp_batch_send",
        action="store_false",
        help="roseinfer_mp: disable batch send (send one request per IPC command).",
    )
    parser.set_defaults(roseinfer_mp_batch_send=True)
    parser.add_argument(
        "--roseinfer-mp-max-recv-per-iter",
        type=int,
        default=64,
        help=(
            "roseinfer_mp: max commands drained per engine loop iteration when busy; "
            "0 disables the budget (default: 64)."
        ),
    )
    parser.add_argument(
        "--roseinfer-mp-fill-target",
        dest="roseinfer_mp_fill_target",
        action="store_true",
        help=(
            "roseinfer_mp: bypass cmd budget while ramping up below target concurrency "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--roseinfer-no-mp-fill-target",
        dest="roseinfer_mp_fill_target",
        action="store_false",
        help="roseinfer_mp: disable ramp-up fill-to-target behavior.",
    )
    parser.set_defaults(roseinfer_mp_fill_target=True)
    parser.add_argument(
        "--roseinfer-mp-kv-cache-max-concurrency",
        type=int,
        default=0,
        help=(
            "roseinfer_mp: KV cache max concurrency for the engine process; "
            "0 selects an automatic value based on the workload (default: 0)."
        ),
    )
    parser.add_argument(
        "--roseinfer-mp-flat-events",
        dest="roseinfer_mp_flat_events",
        action="store_true",
        help="roseinfer_mp: use flat token-pair events (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-mp-flat-events",
        dest="roseinfer_mp_flat_events",
        action="store_false",
        help="roseinfer_mp: disable flat token-pair events.",
    )
    parser.set_defaults(roseinfer_mp_flat_events=True)
    parser.add_argument(
        "--roseinfer-mp-finish-only",
        dest="roseinfer_mp_finish_only",
        action="store_true",
        help="roseinfer_mp: only send per-request token counts at finish (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-mp-finish-only",
        dest="roseinfer_mp_finish_only",
        action="store_false",
        help="roseinfer_mp: stream token events during generation (slower IPC).",
    )
    parser.set_defaults(roseinfer_mp_finish_only=True)
    parser.add_argument(
        "--roseinfer-mp-fast-finish-counts",
        dest="roseinfer_mp_fast_finish_counts",
        action="store_true",
        help="roseinfer_mp: compute final token counts via scheduler step_count (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-mp-fast-finish-counts",
        dest="roseinfer_mp_fast_finish_counts",
        action="store_false",
        help="roseinfer_mp: fall back to incremental token counting (slower).",
    )
    parser.set_defaults(roseinfer_mp_fast_finish_counts=True)
    parser.add_argument(
        "--roseinfer-mp-thread-cap",
        dest="roseinfer_mp_thread_cap",
        action="store_true",
        help="roseinfer_mp: cap torch threads to 1 (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-mp-thread-cap",
        dest="roseinfer_mp_thread_cap",
        action="store_false",
        help="roseinfer_mp: disable torch thread capping.",
    )
    parser.set_defaults(roseinfer_mp_thread_cap=True)
    parser.add_argument(
        "--roseinfer-mp-affinity",
        dest="roseinfer_mp_affinity",
        action="store_true",
        help="roseinfer_mp: split CPU affinity between API and engine (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-mp-affinity",
        dest="roseinfer_mp_affinity",
        action="store_false",
        help="roseinfer_mp: disable CPU affinity split.",
    )
    parser.set_defaults(roseinfer_mp_affinity=True)
    parser.add_argument(
        "--roseinfer-compare-mp-ablations",
        action="store_true",
        help="Add roseinfer_mp ablation variants (disable one mp optimization at a time).",
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
        "--profile",
        type=str,
        default="none",
        choices=["none", "torch", "nsys", "both"],
        help=(
            "Extra profiling stage (separate run; not included in offline_results.json). "
            "Use --profile-only to skip the throughput benchmark."
        ),
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Skip throughput benchmark; only run the profiling stage.",
    )
    parser.add_argument(
        "--profile-num-prompts",
        type=int,
        default=8,
        help="Number of prompts for the profiling stage (default: 8).",
    )
    parser.add_argument(
        "--profile-input-len",
        type=int,
        default=256,
        help="Prompt length for the profiling stage (default: 256).",
    )
    parser.add_argument(
        "--profile-output-len",
        type=int,
        default=32,
        help="Output length for the profiling stage (default: 32).",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="(Internal) Output dir for a single-backend profiling run.",
    )
    parser.add_argument(
        "--profile-nsys-cuda-flush-interval-ms",
        type=int,
        default=None,
        help=(
            "Profiling stage: optional value for `nsys profile --cuda-flush-interval=...` "
            "(in ms)."
        ),
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
        default="roseinfer_mp,roseinfer,vllm,sglang,trtllm",
        help="Comma-separated backends to run in compare mode (roseinfer,roseinfer_mp,vllm,sglang,trtllm).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["roseinfer", "roseinfer_mp", "vllm", "sglang", "trtllm"],
        help="Run a single backend and print JSON to stdout (internal).",
    )
    return parser.parse_args()


def _run_single_backend(args: argparse.Namespace) -> OfflineResult:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.backend == "trtllm" and bool(getattr(args, "trtllm_worker_single_process", False)):
        os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    profile_mode = str(getattr(args, "profile", "none")).lower()
    if args.backend == "trtllm" and profile_mode != "none":
        os.environ["TLLM_LLMAPI_ENABLE_NVTX"] = "1"
        if bool(getattr(args, "trtllm_profile_record_gc", True)):
            os.environ["TLLM_PROFILE_RECORD_GC"] = "1"
        else:
            os.environ.pop("TLLM_PROFILE_RECORD_GC", None)
    if args.backend == "roseinfer":
        return _run_roseinfer(args)
    if args.backend == "roseinfer_mp":
        return _run_roseinfer_mp(args)
    if args.backend == "vllm":
        return _run_vllm(args)
    if args.backend == "sglang":
        return _run_sglang(args)
    if args.backend == "trtllm":
        return _run_trtllm(args)
    raise ValueError(f"unknown backend: {args.backend}")


def _run_compare(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"offline_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_start_time = _iso_now()
    run_wall_t0 = time.perf_counter()
    vllm_python = _resolve_vllm_python(args)
    sglang_python = _resolve_sglang_python(args)
    trtllm_python = _resolve_trtllm_python(args)
    versions = _collect_versions(
        vllm_python=vllm_python,
        sglang_python=sglang_python,
        trtllm_python=trtllm_python,
    )
    print(f"[meta] start={run_start_time}, versions={versions}")

    server_cpus = (
        _parse_cpu_set(args.server_cpus) if args.server_cpus else _default_server_cpus()
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = str(len(server_cpus))
    env["MKL_NUM_THREADS"] = str(len(server_cpus))
    if sglang_python is None:
        _maybe_add_sglang_source_pythonpath_env(env)
    if trtllm_python is None:
        _maybe_add_trtllm_source_pythonpath_env(env)

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
        roseinfer_overlap_schedule: bool | None = None
        roseinfer_mp_ipc: str | None = None
        roseinfer_mp_batch_send: bool | None = None
        roseinfer_mp_thread_cap: bool | None = None
        roseinfer_mp_affinity: bool | None = None
        roseinfer_mp_max_recv_per_iter: int | None = None
        roseinfer_mp_fill_target: bool | None = None
        roseinfer_mp_kv_cache_max_concurrency: int | None = None
        roseinfer_mp_flat_events: bool | None = None
        roseinfer_mp_finish_only: bool | None = None
        roseinfer_mp_fast_finish_counts: bool | None = None

    run_specs: list[RunSpec] = []
    has_roseinfer_mp = "roseinfer_mp" in base_backends
    for backend in base_backends:
        if backend not in ("roseinfer", "roseinfer_mp"):
            if backend == "trtllm" and not _trtllm_available(python_exe=trtllm_python):
                print("[warn] tensorrt_llm not available; skipping backend 'trtllm'")
                continue
            run_specs.append(RunSpec(base_backend=backend, label=backend))
            continue
        variants = roseinfer_prefill_backends or [
            str(args.roseinfer_prefill_attn_backend)
        ]
        base_fused_ops = bool(args.roseinfer_fused_ops)
        base_fused_mlp = bool(args.roseinfer_fused_mlp)
        base_fused_sampler = bool(args.roseinfer_fused_sampler)
        base_fused_kv_append = bool(args.roseinfer_fused_kv_append)
        base_overlap = bool(getattr(args, "roseinfer_overlap_schedule", True))
        base_mp_ipc = str(getattr(args, "roseinfer_mp_ipc", "pipe"))
        base_mp_batch_send = bool(getattr(args, "roseinfer_mp_batch_send", True))
        base_mp_thread_cap = bool(getattr(args, "roseinfer_mp_thread_cap", True))
        base_mp_affinity = bool(getattr(args, "roseinfer_mp_affinity", True))
        base_mp_max_recv = int(getattr(args, "roseinfer_mp_max_recv_per_iter", 64))
        base_mp_fill_target = bool(getattr(args, "roseinfer_mp_fill_target", True))
        base_mp_kv_conc = int(getattr(args, "roseinfer_mp_kv_cache_max_concurrency", 0))
        base_mp_flat_events = bool(getattr(args, "roseinfer_mp_flat_events", True))
        base_mp_finish_only = bool(getattr(args, "roseinfer_mp_finish_only", True))
        base_mp_fast_finish_counts = bool(
            getattr(args, "roseinfer_mp_fast_finish_counts", True)
        )
        cfgs: list[tuple[bool, bool, bool, bool, bool]] = []
        seen: set[tuple[bool, bool, bool, bool, bool]] = set()

        def add_cfg(
            fused_ops: bool,
            fused_mlp: bool,
            fused_sampler: bool,
            fused_kv_append: bool,
            overlap_schedule: bool,
        ) -> None:
            cfg = (
                bool(fused_ops),
                bool(fused_mlp),
                bool(fused_sampler),
                bool(fused_kv_append),
                bool(overlap_schedule),
            )
            if cfg in seen:
                return
            seen.add(cfg)
            cfgs.append(cfg)

        add_cfg(
            base_fused_ops,
            base_fused_mlp,
            base_fused_sampler,
            base_fused_kv_append,
            base_overlap,
        )
        if bool(getattr(args, "roseinfer_compare_fused_ops", False)):
            add_cfg(
                (not base_fused_ops),
                base_fused_mlp,
                base_fused_sampler,
                base_fused_kv_append,
                base_overlap,
            )
        if bool(getattr(args, "roseinfer_compare_fused_mlp", False)):
            add_cfg(
                base_fused_ops,
                (not base_fused_mlp),
                base_fused_sampler,
                base_fused_kv_append,
                base_overlap,
            )
        if bool(getattr(args, "roseinfer_compare_fused_sampler", False)):
            add_cfg(
                base_fused_ops,
                base_fused_mlp,
                (not base_fused_sampler),
                base_fused_kv_append,
                base_overlap,
            )
        if bool(getattr(args, "roseinfer_compare_fused_kv_append", False)):
            add_cfg(
                base_fused_ops,
                base_fused_mlp,
                base_fused_sampler,
                (not base_fused_kv_append),
                base_overlap,
            )
        if bool(getattr(args, "roseinfer_compare_overlap_schedule", False)):
            add_cfg(
                base_fused_ops,
                base_fused_mlp,
                base_fused_sampler,
                base_fused_kv_append,
                (not base_overlap),
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
            for fused_ops, fused_mlp, fused_sampler, fused_kv_append, overlap in cfgs:
                label = base_label
                if backend == "roseinfer" and has_roseinfer_mp:
                    label += "+inproc"
                if not fused_ops:
                    label += "+nofuse"
                if not fused_mlp:
                    label += "+nomlp"
                if not fused_sampler:
                    label += "+nosampler"
                if not fused_kv_append:
                    label += "+nokv"
                if not overlap:
                    label += "+nooverlap"
                run_specs.append(
                    RunSpec(
                        base_backend=backend,
                        label=label,
                        roseinfer_prefill_backend=prefill_backend,
                        roseinfer_fused_ops=bool(fused_ops),
                        roseinfer_fused_mlp=bool(fused_mlp),
                        roseinfer_fused_sampler=bool(fused_sampler),
                        roseinfer_fused_kv_append=bool(fused_kv_append),
                        roseinfer_overlap_schedule=bool(overlap),
                        roseinfer_mp_ipc=(
                            base_mp_ipc if backend == "roseinfer_mp" else None
                        ),
                        roseinfer_mp_batch_send=(
                            base_mp_batch_send if backend == "roseinfer_mp" else None
                        ),
                        roseinfer_mp_thread_cap=(
                            base_mp_thread_cap if backend == "roseinfer_mp" else None
                        ),
                        roseinfer_mp_affinity=(
                            base_mp_affinity if backend == "roseinfer_mp" else None
                        ),
                        roseinfer_mp_max_recv_per_iter=(
                            base_mp_max_recv if backend == "roseinfer_mp" else None
                        ),
                        roseinfer_mp_fill_target=(
                            base_mp_fill_target if backend == "roseinfer_mp" else None
                        ),
                        roseinfer_mp_kv_cache_max_concurrency=(
                            base_mp_kv_conc if backend == "roseinfer_mp" else None
                        ),
                        roseinfer_mp_flat_events=(
                            base_mp_flat_events if backend == "roseinfer_mp" else None
                        ),
                        roseinfer_mp_finish_only=(
                            base_mp_finish_only if backend == "roseinfer_mp" else None
                        ),
                        roseinfer_mp_fast_finish_counts=(
                            base_mp_fast_finish_counts
                            if backend == "roseinfer_mp"
                            else None
                        ),
                    )
                )

    if bool(getattr(args, "roseinfer_compare_mp_ablations", False)):
        extra: list[RunSpec] = []
        seen_labels = {spec.label for spec in run_specs}
        for spec in list(run_specs):
            if spec.base_backend != "roseinfer_mp":
                continue

            def add_variant(*, suffix: str, **updates: Any) -> None:
                label = f"{spec.label}{suffix}"
                if label in seen_labels:
                    return
                seen_labels.add(label)
                payload = asdict(spec)
                payload.update(updates)
                payload["label"] = label
                extra.append(RunSpec(**payload))

            if bool(spec.roseinfer_mp_thread_cap):
                add_variant(suffix="+nothr", roseinfer_mp_thread_cap=False)
            if bool(spec.roseinfer_mp_affinity):
                add_variant(suffix="+noaff", roseinfer_mp_affinity=False)
            if bool(spec.roseinfer_mp_batch_send):
                add_variant(suffix="+nobatch", roseinfer_mp_batch_send=False)
            if bool(spec.roseinfer_mp_fill_target):
                add_variant(suffix="+nofill", roseinfer_mp_fill_target=False)
            if int(spec.roseinfer_mp_max_recv_per_iter or 0) != 0:
                add_variant(suffix="+nodrain", roseinfer_mp_max_recv_per_iter=0)
            if int(spec.roseinfer_mp_kv_cache_max_concurrency or 0) == 0:
                base_auto_kv = max(
                    1, min(int(args.max_batch_size), int(args.num_prompts))
                )
                max_kv = max(1, int(args.max_batch_size))
                if max_kv != base_auto_kv:
                    add_variant(
                        suffix=f"+kv{max_kv}",
                        roseinfer_mp_kv_cache_max_concurrency=int(max_kv),
                    )
            if str(spec.roseinfer_mp_ipc or "").lower() != "queue":
                add_variant(suffix="+queueipc", roseinfer_mp_ipc="queue")
            if bool(spec.roseinfer_mp_finish_only):
                add_variant(suffix="+streamtok", roseinfer_mp_finish_only=False)
                if bool(spec.roseinfer_mp_fast_finish_counts):
                    add_variant(
                        suffix="+slowcnt",
                        roseinfer_mp_fast_finish_counts=False,
                    )

        run_specs.extend(extra)

    if not run_specs:
        raise ValueError("no backends to run (all candidates were skipped)")

    profile_mode = str(getattr(args, "profile", "none")).lower()
    profile_only = bool(getattr(args, "profile_only", False))
    if profile_only and profile_mode == "none":
        raise ValueError("--profile-only requires --profile torch|nsys|both")
    if profile_mode not in ("none", "torch", "nsys", "both"):
        raise ValueError("--profile must be one of: none, torch, nsys, both")

    def run_profile_stage() -> None:
        if profile_mode == "none":
            return
        tools = ["torch", "nsys"] if profile_mode == "both" else [profile_mode]
        profile_num_prompts = int(getattr(args, "profile_num_prompts", 8))
        profile_input_len = int(getattr(args, "profile_input_len", 256))
        profile_output_len = int(getattr(args, "profile_output_len", 32))
        if profile_num_prompts <= 0:
            raise ValueError("--profile-num-prompts must be >= 1")
        if profile_input_len <= 0:
            raise ValueError("--profile-input-len must be >= 1")
        if profile_output_len <= 0:
            raise ValueError("--profile-output-len must be >= 1")

        profiles_root = run_dir / "profiles"
        profiles_root.mkdir(parents=True, exist_ok=True)

        profile_start_time = _iso_now()
        profile_t0 = time.perf_counter()
        manifest: dict[str, Any] = {
            "meta": {
                "profile": profile_mode,
                "profile_only": profile_only,
                "profile_num_prompts": profile_num_prompts,
                "profile_input_len": profile_input_len,
                "profile_output_len": profile_output_len,
                "server_cpus": server_cpus,
                "tools": tools,
                "backends": [spec.label for spec in run_specs],
                "profile_start_time": profile_start_time,
                "vllm_async_scheduling": bool(
                    getattr(args, "vllm_async_scheduling", True)
                ),
                "trtllm_profile_record_gc": bool(
                    getattr(args, "trtllm_profile_record_gc", True)
                ),
                "versions": versions,
            },
            "runs": [],
        }

        cpu_str = _format_cpu_set(server_cpus)
        script_path = str(Path(__file__).resolve())
        for tool in tools:
            for spec in run_specs:
                base_backend = spec.base_backend
                label = spec.label
                out_dir = profiles_root / tool / label
                out_dir.mkdir(parents=True, exist_ok=True)

                profile_env = env.copy()
                profile_env["ROSEINFER_NVTX"] = "1"

                python_exe = (
                    trtllm_python
                    if base_backend == "trtllm" and trtllm_python is not None
                    else vllm_python
                    if base_backend == "vllm" and vllm_python is not None
                    else sglang_python
                    if base_backend == "sglang" and sglang_python is not None
                    else sys.executable
                )
                if python_exe != sys.executable:
                    profile_env.pop("PYTHONPATH", None)
                if base_backend == "trtllm" and trtllm_python is not None:
                    profile_env = _trtllm_runtime_env(
                        python_exe=trtllm_python, base_env=profile_env
                    )
                if base_backend == "trtllm":
                    profile_env["TLLM_LLMAPI_ENABLE_NVTX"] = "1"
                    if bool(getattr(args, "trtllm_profile_record_gc", True)):
                        profile_env["TLLM_PROFILE_RECORD_GC"] = "1"
                    else:
                        profile_env.pop("TLLM_PROFILE_RECORD_GC", None)
                    if bool(getattr(args, "trtllm_worker_single_process", False)):
                        profile_env["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
                    else:
                        profile_env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)

                py_cmd = [
                    python_exe,
                    script_path,
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
                    "--trtllm-backend",
                    str(getattr(args, "trtllm_backend", "tensorrt")),
                    "--num-prompts",
                    str(profile_num_prompts),
                    "--input-len",
                    str(profile_input_len),
                    "--output-len",
                    str(profile_output_len),
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
                    "--profile",
                    tool,
                    "--profile-dir",
                    str(out_dir),
                ]
                if args.ignore_eos:
                    py_cmd.append("--ignore-eos")
                else:
                    py_cmd.append("--disable-ignore-eos")
                if args.skip_warmup:
                    py_cmd.append("--skip-warmup")
                if args.no_amp:
                    py_cmd.append("--no-amp")
                if args.bf16:
                    py_cmd.append("--bf16")
                if base_backend == "vllm":
                    py_cmd += [
                        "--vllm-attention-backend",
                        str(getattr(args, "vllm_attention_backend", "auto")),
                    ]
                    vllm_max_num_seqs = _resolve_vllm_max_num_seqs(args)
                    if vllm_max_num_seqs is not None:
                        py_cmd += [
                            "--vllm-max-num-seqs",
                            str(int(vllm_max_num_seqs)),
                        ]
                    py_cmd.append(
                        "--vllm-async-scheduling"
                        if bool(getattr(args, "vllm_async_scheduling", True))
                        else "--vllm-no-async-scheduling"
                    )
                if base_backend == "sglang":
                    py_cmd += [
                        "--sglang-attention-backend",
                        str(getattr(args, "sglang_attention_backend", "triton")),
                        "--sglang-sampling-backend",
                        str(getattr(args, "sglang_sampling_backend", "flashinfer")),
                    ]
                if base_backend in ("roseinfer", "roseinfer_mp"):
                    py_cmd += [
                        "--roseinfer-prefill-attn-backend",
                        str(
                            spec.roseinfer_prefill_backend
                            or args.roseinfer_prefill_attn_backend
                        ),
                        "--roseinfer-decode-attn-backend",
                        str(args.roseinfer_decode_attn_backend),
                        "--roseinfer-label",
                        str(label),
                    ]
                    py_cmd.append(
                        "--roseinfer-paged-attn"
                        if bool(args.roseinfer_paged_attn)
                        else "--roseinfer-no-paged-attn"
                    )
                    py_cmd.append(
                        "--roseinfer-cuda-graph"
                        if bool(args.roseinfer_cuda_graph)
                        else "--roseinfer-no-cuda-graph"
                    )
                    py_cmd.append(
                        "--roseinfer-chunked-prefill"
                        if bool(args.roseinfer_chunked_prefill)
                        else "--roseinfer-no-chunked-prefill"
                    )
                    py_cmd += [
                        "--roseinfer-prefill-chunk-size",
                        str(int(args.roseinfer_prefill_chunk_size)),
                    ]
                    py_cmd.append(
                        "--roseinfer-prefix-cache"
                        if bool(args.roseinfer_prefix_cache)
                        else "--roseinfer-no-prefix-cache"
                    )
                    py_cmd.append(
                        "--roseinfer-fused-ops"
                        if bool(spec.roseinfer_fused_ops)
                        else "--roseinfer-no-fused-ops"
                    )
                    py_cmd.append(
                        "--roseinfer-fused-mlp"
                        if bool(spec.roseinfer_fused_mlp)
                        else "--roseinfer-no-fused-mlp"
                    )
                    py_cmd.append(
                        "--roseinfer-fused-sampler"
                        if bool(spec.roseinfer_fused_sampler)
                        else "--roseinfer-no-fused-sampler"
                    )
                    py_cmd.append(
                        "--roseinfer-fused-kv-append"
                        if bool(spec.roseinfer_fused_kv_append)
                        else "--roseinfer-no-fused-kv-append"
                    )
                    py_cmd.append(
                        "--roseinfer-overlap-schedule"
                        if bool(spec.roseinfer_overlap_schedule)
                        else "--roseinfer-no-overlap-schedule"
                    )
                    if base_backend == "roseinfer_mp":
                        py_cmd += ["--roseinfer-mp-ipc", str(spec.roseinfer_mp_ipc)]
                        py_cmd.append(
                            "--roseinfer-mp-batch-send"
                            if bool(spec.roseinfer_mp_batch_send)
                            else "--roseinfer-no-mp-batch-send"
                        )
                        py_cmd += [
                            "--roseinfer-mp-max-recv-per-iter",
                            str(int(spec.roseinfer_mp_max_recv_per_iter or 0)),
                        ]
                        py_cmd.append(
                            "--roseinfer-mp-fill-target"
                            if bool(spec.roseinfer_mp_fill_target)
                            else "--roseinfer-no-mp-fill-target"
                        )
                        py_cmd += [
                            "--roseinfer-mp-kv-cache-max-concurrency",
                            str(int(spec.roseinfer_mp_kv_cache_max_concurrency or 0)),
                        ]
                        py_cmd.append(
                            "--roseinfer-mp-flat-events"
                            if bool(spec.roseinfer_mp_flat_events)
                            else "--roseinfer-no-mp-flat-events"
                        )
                        py_cmd.append(
                            "--roseinfer-mp-finish-only"
                            if bool(spec.roseinfer_mp_finish_only)
                            else "--roseinfer-no-mp-finish-only"
                        )
                        py_cmd.append(
                            "--roseinfer-mp-fast-finish-counts"
                            if bool(spec.roseinfer_mp_fast_finish_counts)
                            else "--roseinfer-no-mp-fast-finish-counts"
                        )
                        py_cmd.append(
                            "--roseinfer-mp-thread-cap"
                            if bool(spec.roseinfer_mp_thread_cap)
                            else "--roseinfer-no-mp-thread-cap"
                        )
                        py_cmd.append(
                            "--roseinfer-mp-affinity"
                            if bool(spec.roseinfer_mp_affinity)
                            else "--roseinfer-no-mp-affinity"
                        )

                cmd: list[str] = ["taskset", "-c", cpu_str]
                nsys_prefix = None
                if tool == "nsys":
                    nsys_prefix = out_dir / label
                    nsys_cuda_flush_ms = getattr(
                        args, "profile_nsys_cuda_flush_interval_ms", None
                    )
                    # See `benchmarks/serving/online_compare.py`: multi-process backends
                    # can start worker processes before `cudaProfilerStart`, causing
                    # capture-range=cudaProfilerApi to miss GPU kernels. Prefer capturing
                    # from process start for reliable process-tree profiling.
                    cmd += [
                        "nsys",
                        "profile",
                        "--force-overwrite=true",
                        "--capture-range=none",
                        "--trace=cuda,nvtx,osrt",
                        *(
                            [f"--cuda-flush-interval={int(nsys_cuda_flush_ms)}"]
                            if nsys_cuda_flush_ms is not None
                            else []
                        ),
                        "--sample=none",
                        "--trace-fork-before-exec=true",
                        "-o",
                        str(nsys_prefix),
                    ]
                cmd += py_cmd

                t0 = time.perf_counter()
                raw = b""
                retcode: int | None = None
                try:
                    raw = subprocess.check_output(
                        cmd, env=profile_env, stderr=subprocess.STDOUT
                    )
                except subprocess.CalledProcessError as exc:
                    raw = exc.output or b""
                    retcode = int(exc.returncode)
                    if tool != "nsys" or retcode not in (0, 143):
                        raise
                wall_s = float(time.perf_counter() - t0)
                (out_dir / "stdout.log").write_bytes(raw)

                parsed: dict[str, Any] | None = None
                raw_text = raw.decode("utf-8", errors="replace")
                for line in reversed(raw_text.splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
                if parsed is not None:
                    (out_dir / "result.json").write_text(
                        json.dumps(parsed, indent=2), encoding="utf-8"
                    )

                manifest["runs"].append(
                    {
                        "tool": tool,
                        "backend": label,
                        "base_backend": base_backend,
                        "cmd": cmd,
                        "wall_s": wall_s,
                        "output_dir": str(out_dir),
                        "nsys_output_prefix": str(nsys_prefix) if nsys_prefix else None,
                    }
                )
                print(f"[profile:{tool}] {label} wall={wall_s:.2f}s -> {out_dir}")

        profile_wall_s = float(time.perf_counter() - profile_t0)
        profile_end_time = _iso_now()
        manifest["meta"]["profile_end_time"] = profile_end_time
        manifest["meta"]["profile_wall_s"] = profile_wall_s
        out_manifest = run_dir / "profile_manifest.json"
        out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote: {out_manifest}")

    if profile_only:
        run_profile_stage()
        return

    for spec in run_specs:
        base_backend = spec.base_backend
        label = spec.label
        prefill_backend = spec.roseinfer_prefill_backend
        fused_ops = spec.roseinfer_fused_ops
        fused_mlp = spec.roseinfer_fused_mlp
        fused_sampler = spec.roseinfer_fused_sampler
        fused_kv_append = spec.roseinfer_fused_kv_append
        overlap_schedule = spec.roseinfer_overlap_schedule
        python_exe = (
            trtllm_python
            if base_backend == "trtllm" and trtllm_python is not None
            else vllm_python
            if base_backend == "vllm" and vllm_python is not None
            else sglang_python
            if base_backend == "sglang" and sglang_python is not None
            else sys.executable
        )
        run_env = (
            _trtllm_runtime_env(python_exe=trtllm_python, base_env=env)
            if base_backend == "trtllm" and trtllm_python is not None
            else env
        )
        if python_exe != sys.executable:
            run_env = dict(run_env)
            run_env.pop("PYTHONPATH", None)
        if base_backend == "trtllm":
            if bool(getattr(args, "trtllm_worker_single_process", False)):
                run_env["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
            else:
                run_env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)

        cmd = [
            "taskset",
            "-c",
            _format_cpu_set(server_cpus),
            python_exe,
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
            "--trtllm-backend",
            str(getattr(args, "trtllm_backend", "tensorrt")),
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
        else:
            cmd.append("--disable-ignore-eos")
        if args.skip_warmup:
            cmd.append("--skip-warmup")
        if args.no_amp:
            cmd.append("--no-amp")
        if args.bf16:
            cmd.append("--bf16")
        if base_backend == "vllm":
            cmd += [
                "--vllm-attention-backend",
                str(getattr(args, "vllm_attention_backend", "auto")),
            ]
            vllm_max_num_seqs = _resolve_vllm_max_num_seqs(args)
            if vllm_max_num_seqs is not None:
                cmd += [
                    "--vllm-max-num-seqs",
                    str(int(vllm_max_num_seqs)),
                ]
            cmd.append(
                "--vllm-async-scheduling"
                if bool(getattr(args, "vllm_async_scheduling", True))
                else "--vllm-no-async-scheduling"
            )
        if base_backend == "sglang":
            cmd += [
                "--sglang-attention-backend",
                str(getattr(args, "sglang_attention_backend", "triton")),
                "--sglang-sampling-backend",
                str(getattr(args, "sglang_sampling_backend", "flashinfer")),
            ]
        if base_backend in ("roseinfer", "roseinfer_mp"):
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
            cmd.append(
                "--roseinfer-overlap-schedule"
                if bool(overlap_schedule)
                else "--roseinfer-no-overlap-schedule"
            )
            if base_backend == "roseinfer_mp":
                cmd += ["--roseinfer-mp-ipc", str(spec.roseinfer_mp_ipc)]
                cmd.append(
                    "--roseinfer-mp-batch-send"
                    if bool(spec.roseinfer_mp_batch_send)
                    else "--roseinfer-no-mp-batch-send"
                )
                cmd += [
                    "--roseinfer-mp-max-recv-per-iter",
                    str(int(spec.roseinfer_mp_max_recv_per_iter or 0)),
                ]
                cmd.append(
                    "--roseinfer-mp-fill-target"
                    if bool(spec.roseinfer_mp_fill_target)
                    else "--roseinfer-no-mp-fill-target"
                )
                cmd += [
                    "--roseinfer-mp-kv-cache-max-concurrency",
                    str(int(spec.roseinfer_mp_kv_cache_max_concurrency or 0)),
                ]
                cmd.append(
                    "--roseinfer-mp-flat-events"
                    if bool(spec.roseinfer_mp_flat_events)
                    else "--roseinfer-no-mp-flat-events"
                )
                cmd.append(
                    "--roseinfer-mp-finish-only"
                    if bool(spec.roseinfer_mp_finish_only)
                    else "--roseinfer-no-mp-finish-only"
                )
                cmd.append(
                    "--roseinfer-mp-fast-finish-counts"
                    if bool(spec.roseinfer_mp_fast_finish_counts)
                    else "--roseinfer-no-mp-fast-finish-counts"
                )
                cmd.append(
                    "--roseinfer-mp-thread-cap"
                    if bool(spec.roseinfer_mp_thread_cap)
                    else "--roseinfer-no-mp-thread-cap"
                )
                cmd.append(
                    "--roseinfer-mp-affinity"
                    if bool(spec.roseinfer_mp_affinity)
                    else "--roseinfer-no-mp-affinity"
                )

        backend_t0 = time.perf_counter()
        raw = b""
        retcode: int | None = None
        try:
            raw = subprocess.check_output(cmd, env=run_env, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raw = exc.output or b""
            retcode = int(exc.returncode)
        backend_t1 = time.perf_counter()
        backend_wall_s[label] = float(backend_t1 - backend_t0)
        raw_text = raw.decode("utf-8", errors="replace")
        if retcode is not None:
            tail = "\n".join(raw_text.splitlines()[-80:]).strip()
            results.append(
                OfflineResult(
                    backend=str(label),
                    model=args.model,
                    device=args.device,
                    num_prompts=int(args.num_prompts),
                    input_len=int(args.input_len),
                    output_len=int(args.output_len),
                    total_input_tokens=0,
                    total_output_tokens=0,
                    prefill_s=None,
                    decode_s=None,
                    total_s=float(backend_wall_s[label]),
                    request_throughput_rps=0.0,
                    output_throughput_tps=0.0,
                    total_throughput_tps=0.0,
                    extra={
                        "error": tail or f"backend exited with status {retcode}",
                        "exit_code": retcode,
                    },
                )
            )
            print(f"[{label}] error=exit {retcode}, wall={backend_wall_s[label]:.2f}s")
            continue

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
            tail = "\n".join(raw_text.splitlines()[-80:]).strip()
            results.append(
                OfflineResult(
                    backend=str(label),
                    model=args.model,
                    device=args.device,
                    num_prompts=int(args.num_prompts),
                    input_len=int(args.input_len),
                    output_len=int(args.output_len),
                    total_input_tokens=0,
                    total_output_tokens=0,
                    prefill_s=None,
                    decode_s=None,
                    total_s=float(backend_wall_s[label]),
                    request_throughput_rps=0.0,
                    output_throughput_tps=0.0,
                    total_throughput_tps=0.0,
                    extra={"error": tail or "failed to parse backend JSON output"},
                )
            )
            print(f"[{label}] error=parse, wall={backend_wall_s[label]:.2f}s")
            continue

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
            "vllm_async_scheduling": bool(getattr(args, "vllm_async_scheduling", True)),
            "vllm_attention_backend": str(
                getattr(args, "vllm_attention_backend", "auto")
            ),
            "vllm_max_num_seqs": _resolve_vllm_max_num_seqs(args),
            "sglang_attention_backend": str(
                getattr(args, "sglang_attention_backend", "auto")
            ),
            "sglang_sampling_backend": str(
                getattr(args, "sglang_sampling_backend", "auto")
            ),
            "trtllm_worker_single_process": bool(
                getattr(args, "trtllm_worker_single_process", False)
            ),
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "max_batch_size": int(args.max_batch_size),
            "warmup_prompts": int(args.warmup_prompts),
            "warmup_full_batch": bool(getattr(args, "warmup_full_batch", True)),
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
            "roseinfer_overlap_schedule": bool(args.roseinfer_overlap_schedule),
            "roseinfer_compare_overlap_schedule": bool(
                args.roseinfer_compare_overlap_schedule
            ),
            "roseinfer_mp_ipc": str(getattr(args, "roseinfer_mp_ipc", "pipe")),
            "roseinfer_mp_batch_send": bool(
                getattr(args, "roseinfer_mp_batch_send", True)
            ),
            "roseinfer_mp_max_recv_per_iter": int(
                getattr(args, "roseinfer_mp_max_recv_per_iter", 64)
            ),
            "roseinfer_mp_fill_target": bool(
                getattr(args, "roseinfer_mp_fill_target", True)
            ),
            "roseinfer_mp_kv_cache_max_concurrency": int(
                getattr(args, "roseinfer_mp_kv_cache_max_concurrency", 0)
            ),
            "roseinfer_mp_thread_cap": bool(
                getattr(args, "roseinfer_mp_thread_cap", True)
            ),
            "roseinfer_mp_affinity": bool(getattr(args, "roseinfer_mp_affinity", True)),
            "roseinfer_mp_flat_events": bool(
                getattr(args, "roseinfer_mp_flat_events", True)
            ),
            "roseinfer_mp_finish_only": bool(
                getattr(args, "roseinfer_mp_finish_only", True)
            ),
            "roseinfer_mp_fast_finish_counts": bool(
                getattr(args, "roseinfer_mp_fast_finish_counts", True)
            ),
            "roseinfer_compare_mp_ablations": bool(
                getattr(args, "roseinfer_compare_mp_ablations", False)
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

    run_profile_stage()


def main() -> None:
    args = parse_args()
    if args.backend is not None:
        result = _run_single_backend(args)
        sys.stdout.write(json.dumps(asdict(result)) + "\n")
        return
    _run_compare(args)


if __name__ == "__main__":
    main()
