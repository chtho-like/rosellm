from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import importlib.util
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer

TRACE_A_URL = (
    "https://raw.githubusercontent.com/alibaba-edu/"
    "qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"
)

_DUMMY_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "sk-rosellm"


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _append_pythonpath(env: dict[str, str], path: str) -> None:
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = path if not old else f"{path}{os.pathsep}{old}"


def _maybe_add_sglang_source_pythonpath(env: dict[str, str]) -> None:
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


def _maybe_add_trtllm_source_pythonpath(env: dict[str, str]) -> None:
    if _module_available("tensorrt_llm"):
        return
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / ".vscode" / "TensorRT-LLM"
    if candidate.is_dir():
        _append_pythonpath(env, str(candidate))


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

    if (
        vllm_python
        and versions.get("vllm") == "not installed"
        and Path(str(vllm_python)).is_file()
    ):
        try:
            out = subprocess.check_output(
                [
                    str(vllm_python),
                    "-c",
                    "import importlib.metadata as md; print(md.version('vllm'))",
                ],
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            probed = out.decode("utf-8", errors="replace").strip()
            if probed:
                versions["vllm"] = probed
        except Exception:
            pass

    if (
        sglang_python
        and versions.get("sglang") == "not installed"
        and Path(str(sglang_python)).is_file()
    ):
        try:
            out = subprocess.check_output(
                [
                    str(sglang_python),
                    "-c",
                    "import importlib.metadata as md; print(md.version('sglang'))",
                ],
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            probed = out.decode("utf-8", errors="replace").strip()
            if probed:
                versions["sglang"] = probed
        except Exception:
            pass

    if (
        trtllm_python
        and versions.get("tensorrt_llm") == "not installed"
        and Path(str(trtllm_python)).is_file()
    ):
        try:
            out = subprocess.check_output(
                [
                    str(trtllm_python),
                    "-c",
                    "import importlib.metadata as md; print(md.version('tensorrt_llm'))",
                ],
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            probed = out.decode("utf-8", errors="replace").strip()
            if probed:
                versions["tensorrt_llm"] = probed
        except Exception:
            pass
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


@dataclass(frozen=True)
class TraceItem:
    timestamp_s: float
    input_len: int
    output_len: int
    prompt: str


@dataclass(frozen=True)
class RequestMetrics:
    input_len: int
    output_len: int
    start_s: float
    end_s: float
    ttft_s: float | None
    e2e_s: float
    tpot_s: float | None
    num_stream_events: int
    error: str | None


@dataclass(frozen=True)
class SummaryStats:
    mean: float
    p50: float
    p90: float
    p99: float
    max: float


@dataclass(frozen=True)
class OnlineBenchmarkSummary:
    backend: str
    scale: float
    num_requests: int
    num_success: int
    num_error: int
    success_rate: float
    total_output_events: int
    duration_s: float
    req_per_s: float
    tok_per_s: float
    ttft_ms: SummaryStats
    e2e_ms: SummaryStats
    tpot_ms: SummaryStats
    itl_ms: SummaryStats


def _percentiles(values: Sequence[float]) -> SummaryStats:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot compute percentiles for empty list")
    return SummaryStats(
        mean=float(arr.mean()),
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p99=float(np.percentile(arr, 99)),
        max=float(arr.max()),
    )


def _parse_csv_floats(csv: str) -> list[float]:
    out: list[float] = []
    for piece in csv.split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.append(float(piece))
    return out


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


def _default_cpu_sets() -> tuple[list[int], list[int]]:
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 4:
        # keep at least one core for each side
        mid = max(1, len(cpus) // 2)
        return cpus[:mid], cpus[mid:] or cpus[:1]
    mid = len(cpus) // 2
    return cpus[:mid], cpus[mid:]


def _download_if_needed(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (controlled URL)
    tmp.replace(dst)
    return dst


def _read_trace_a_jsonl(
    file_path: Path,
    *,
    n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    rows.sort(key=lambda r: float(r["timestamp"]))
    return rows


def _resolve_model_max_context(model_id: str, tokenizer: Any) -> int:
    max_len = int(getattr(tokenizer, "model_max_length", 0) or 0)
    if 0 < max_len < 10**6:
        return max_len
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id)
        for attr in (
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "seq_length",
        ):
            v = getattr(cfg, attr, None)
            if v is not None and int(v) > 0:
                return int(v)
    except Exception:
        pass
    return 1024


def _make_base_prompt_ids(tokenizer: Any, n_tokens: int, *, seed: int) -> list[int]:
    rng = random.Random(seed)
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        raise ValueError("tokenizer.vocab_size must be positive")
    token_ids = [rng.randint(0, max(1, vocab_size // 2)) for _ in range(n_tokens)]
    # GPT2 tokenizer should be round-trip stable: encode(decode(ids)) == ids.
    prompt = tokenizer.decode(token_ids)
    roundtrip = tokenizer.encode(prompt, add_special_tokens=False)
    if len(roundtrip) != n_tokens:
        # fall back: use the round-tripped ids and pad/truncate.
        if len(roundtrip) < n_tokens:
            need = n_tokens - len(roundtrip)
            roundtrip.extend(
                [rng.randint(0, max(1, vocab_size // 2)) for _ in range(need)]
            )
        else:
            roundtrip = roundtrip[:n_tokens]
        prompt = tokenizer.decode(roundtrip)
        roundtrip2 = tokenizer.encode(prompt, add_special_tokens=False)
        if len(roundtrip2) != n_tokens:
            raise ValueError(
                "failed to construct a base prompt with stable token length"
            )
        token_ids = roundtrip2
    return token_ids


def _build_trace_items(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_ctx: int,
    scale: float,
    start_offset_s: float,
    max_input_len: int | None,
    max_output_len: int | None,
    prompt_overhead_tokens: int,
    seed: int,
) -> list[TraceItem]:
    if not rows:
        return []
    if max_ctx <= 0:
        raise ValueError("max_ctx must be positive")
    overhead = max(0, int(prompt_overhead_tokens))
    # Some servers treat max context as a strict upper bound (total_tokens < max_ctx),
    # so keep a 1-token safety margin by default.
    effective_ctx = max(2, int(max_ctx) - overhead - 1)
    min_ts = float(min(r["timestamp"] for r in rows))
    raw = []
    for r in rows:
        in_len = int(r["input_length"])
        out_len = int(r["output_length"])
        if max_output_len is not None:
            out_len = min(out_len, int(max_output_len))
        out_len = max(1, min(out_len, effective_ctx - 1))
        in_budget = effective_ctx - out_len
        in_len = max(1, min(in_len, in_budget))
        if max_input_len is not None:
            in_len = max(1, min(in_len, int(max_input_len)))
        if in_len + out_len > effective_ctx:
            out_len = max(1, effective_ctx - in_len)
        raw.append(
            (
                (float(r["timestamp"]) - min_ts) * float(scale) + float(start_offset_s),
                in_len,
                out_len,
            )
        )
    max_in = max(in_len for _, in_len, _ in raw)
    base_ids = _make_base_prompt_ids(tokenizer, max_in, seed=seed)
    out: list[TraceItem] = []
    for ts_s, in_len, out_len in raw:
        prompt = tokenizer.decode(base_ids[:in_len])
        out.append(
            TraceItem(
                timestamp_s=ts_s, input_len=in_len, output_len=out_len, prompt=prompt
            )
        )
    out.sort(key=lambda x: x.timestamp_s)
    return out


async def _wait_ready(base_url: str, *, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(base_url=base_url, timeout=2.0) as client:
        while True:
            try:
                r = await client.get("/health")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            try:
                r = await client.get("/v1/models")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            if time.time() >= deadline:
                raise TimeoutError(
                    f"server not ready after {timeout_s:.1f}s: {base_url}"
                )
            await asyncio.sleep(0.2)


async def _benchmark_one(
    client: AsyncOpenAI,
    *,
    model: str,
    trace: TraceItem,
    temperature: float,
    top_p: float,
    top_k: int,
    ignore_eos: bool,
) -> tuple[RequestMetrics, list[float]]:
    await asyncio.sleep(max(0.0, trace.timestamp_s - time.perf_counter()))
    start = time.perf_counter()
    try:
        response = await client.completions.create(
            model=model,
            stream=True,
            prompt=trace.prompt,
            max_tokens=trace.output_len,
            temperature=temperature,
            top_p=top_p,
            extra_body={
                "top_k": int(top_k),
                "ignore_eos": bool(ignore_eos),
            },
        )
        token_tics: list[float] = []
        async for chunk in response:
            try:
                content = chunk.choices[0].text
            except Exception:
                content = None
            if content:
                token_tics.append(time.perf_counter())
        end = time.perf_counter()
        ttft_s = (token_tics[0] - start) if token_tics else None
        if ttft_s is not None and len(token_tics) >= 2:
            itl = [token_tics[i] - token_tics[i - 1] for i in range(1, len(token_tics))]
            tpot_s = float(sum(itl) / len(itl))
        else:
            tpot_s = None
            itl = []
        return (
            RequestMetrics(
                input_len=trace.input_len,
                output_len=trace.output_len,
                start_s=start,
                end_s=end,
                ttft_s=ttft_s,
                e2e_s=end - start,
                tpot_s=tpot_s,
                num_stream_events=len(token_tics),
                error=None,
            ),
            itl,
        )
    except Exception as exc:
        end = time.perf_counter()
        return (
            RequestMetrics(
                input_len=trace.input_len,
                output_len=trace.output_len,
                start_s=start,
                end_s=end,
                ttft_s=None,
                e2e_s=end - start,
                tpot_s=None,
                num_stream_events=0,
                error=str(exc),
            ),
            [],
        )


async def run_trace_benchmark(
    *,
    base_url: str,
    model: str,
    traces: list[TraceItem],
    temperature: float,
    top_p: float,
    top_k: int,
    ignore_eos: bool,
) -> tuple[list[RequestMetrics], list[float]]:
    start0 = time.perf_counter()
    async with AsyncOpenAI(base_url=base_url, api_key=_DUMMY_OPENAI_API_KEY) as client:
        tasks = [
            _benchmark_one(
                client,
                model=model,
                trace=TraceItem(
                    timestamp_s=start0 + t.timestamp_s,
                    input_len=t.input_len,
                    output_len=t.output_len,
                    prompt=t.prompt,
                ),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                ignore_eos=ignore_eos,
            )
            for t in traces
        ]
        results = await asyncio.gather(*tasks)
    req_metrics = [r[0] for r in results]
    itl_all = [dt for _, itl in results for dt in itl]
    return req_metrics, itl_all


async def _warmup(
    *,
    base_url: str,
    model: str,
    prompt: str,
    output_len: int,
    num_requests: int,
    temperature: float,
    top_p: float,
    top_k: int,
    ignore_eos: bool,
) -> None:
    if num_requests <= 0:
        return
    async with AsyncOpenAI(base_url=base_url, api_key=_DUMMY_OPENAI_API_KEY) as client:
        for _ in range(num_requests):
            response = await client.completions.create(
                model=model,
                stream=True,
                prompt=prompt,
                max_tokens=int(output_len),
                temperature=float(temperature),
                top_p=float(top_p),
                extra_body={
                    "top_k": int(top_k),
                    "ignore_eos": bool(ignore_eos),
                },
            )
            async for _ in response:
                pass


def _summarize(
    *,
    backend: str,
    scale: float,
    req_metrics: list[RequestMetrics],
    itl_all: list[float],
) -> OnlineBenchmarkSummary:
    ok = [m for m in req_metrics if m.error is None]
    num_success = len(ok)
    num_error = len(req_metrics) - num_success
    success_rate = num_success / max(len(req_metrics), 1)
    ttft_ms = [m.ttft_s * 1000 for m in ok if m.ttft_s is not None]
    e2e_ms = [m.e2e_s * 1000 for m in ok]
    tpot_ms = [m.tpot_s * 1000 for m in ok if m.tpot_s is not None]
    itl_ms = [dt * 1000 for dt in itl_all]

    def nan_stats() -> SummaryStats:
        nan = float("nan")
        return SummaryStats(mean=nan, p50=nan, p90=nan, p99=nan, max=nan)

    total_events = sum(m.num_stream_events for m in ok)
    if ok:
        duration_s = (max(m.end_s for m in ok) - min(m.start_s for m in ok)) or 1e-9
    else:
        duration_s = (
            (max(m.end_s for m in req_metrics) - min(m.start_s for m in req_metrics))
            if req_metrics
            else 1e-9
        ) or 1e-9
    req_per_s = len(ok) / duration_s
    tok_per_s = total_events / duration_s
    return OnlineBenchmarkSummary(
        backend=backend,
        scale=float(scale),
        num_requests=len(req_metrics),
        num_success=num_success,
        num_error=num_error,
        success_rate=float(success_rate),
        total_output_events=total_events,
        duration_s=float(duration_s),
        req_per_s=float(req_per_s),
        tok_per_s=float(tok_per_s),
        ttft_ms=_percentiles(ttft_ms) if ttft_ms else nan_stats(),
        e2e_ms=_percentiles(e2e_ms) if e2e_ms else nan_stats(),
        tpot_ms=_percentiles(tpot_ms) if tpot_ms else nan_stats(),
        itl_ms=_percentiles(itl_ms) if itl_ms else nan_stats(),
    )


def _start_process(
    *,
    cmd: list[str],
    env: dict[str, str],
    cpu_set: list[int],
    log_path: Path,
) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_str = _format_cpu_set(cpu_set)
    if cmd and cmd[0] == "nsys":
        full_cmd = cmd
    else:
        full_cmd = ["taskset", "-c", cpu_str, *cmd]
    log_f = log_path.open("wb")
    return subprocess.Popen(
        full_cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )


def _terminate_process(p: subprocess.Popen[bytes], *, timeout_s: float = 15.0) -> None:
    if p.poll() is not None:
        return
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=timeout_s)


def _find_trtllm_server_pids(*, port: int) -> set[int]:
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        return set()
    port = int(port)
    pids: set[int] = set()
    try:
        for conn in psutil.net_connections(kind="tcp"):
            if (
                conn.pid
                and conn.status == psutil.CONN_LISTEN
                and conn.laddr
                and int(conn.laddr.port) == port
            ):
                pids.add(int(conn.pid))
    except Exception:
        pass
    needle_port = str(port)
    needle_mod = "tensorrt_llm.commands.serve"
    for proc in psutil.process_iter(attrs=["pid", "cmdline"]):
        cmd = proc.info.get("cmdline") or []
        if not cmd:
            continue
        if needle_mod not in " ".join(cmd):
            continue
        if needle_port not in cmd:
            continue
        pids.add(int(proc.info["pid"]))
    return pids


def _terminate_trtllm_server(*, port: int, timeout_s: float = 20.0) -> None:
    pids = _find_trtllm_server_pids(port=port)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        remaining = _find_trtllm_server_pids(port=port)
        if not remaining:
            return
        time.sleep(0.25)
    for pid in _find_trtllm_server_pids(port=port):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def _terminate_server_by_port(*, port: int, timeout_s: float = 20.0) -> None:
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        return
    port = int(port)
    try:
        listen_conns = [
            c
            for c in psutil.net_connections(kind="tcp")
            if c.pid
            and c.status == psutil.CONN_LISTEN
            and c.laddr
            and int(c.laddr.port) == port
        ]
    except Exception:
        listen_conns = []

    root_pids = {int(c.pid) for c in listen_conns if c.pid}
    if not root_pids:
        return

    def _send_tree(pid: int, sig: int) -> None:
        try:
            root = psutil.Process(int(pid))
        except Exception:
            return
        for child in root.children(recursive=True):
            try:
                child.send_signal(sig)
            except Exception:
                continue
        try:
            root.send_signal(sig)
        except Exception:
            return

    for pid in root_pids:
        _send_tree(pid, signal.SIGTERM)

    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        still_listening: set[int] = set()
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if (
                    conn.pid
                    and conn.status == psutil.CONN_LISTEN
                    and conn.laddr
                    and int(conn.laddr.port) == port
                ):
                    still_listening.add(int(conn.pid))
        except Exception:
            break
        if not still_listening:
            return
        time.sleep(0.25)

    for pid in root_pids:
        _send_tree(pid, signal.SIGKILL)


def _pick_free_port(host: str, preferred: int, used: set[int]) -> int:
    port = int(preferred)
    for _ in range(256):
        if port in used:
            port += 1
            continue
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
        except OSError:
            port += 1
            continue
        finally:
            sock.close()
        used.add(port)
        return port
    raise RuntimeError(f"could not find a free TCP port near {preferred}")


def _roseinfer_server_cmd(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    async_streaming: bool | None = None,
    prefill_attn_backend: str | None = None,
    decode_attn_backend: str | None = None,
    fused_ops: bool | None = None,
    fused_mlp: bool | None = None,
    fused_sampler: bool | None = None,
    fused_kv_append: bool | None = None,
    overlap_schedule: bool | None = None,
    engine_process: bool | None = None,
    max_batch_size: int | None = None,
    gc_freeze: bool | None = None,
    fast_sse: bool | None = None,
    kv_cache_max_concurrency: int | None = None,
    prefix_cache_max_entries: int | None = None,
    mp_ipc: str | None = None,
    mp_thread_cap: bool | None = None,
    mp_affinity: bool | None = None,
    mp_max_recv_per_iter: int | None = None,
    mp_fill_target: bool | None = None,
    mp_flat_events: bool | None = None,
    mp_async_admit: bool | None = None,
    mp_tokenize_workers: int | None = None,
) -> list[str]:
    prefill_attn_backend = (
        str(prefill_attn_backend)
        if prefill_attn_backend is not None
        else str(args.roseinfer_prefill_attn_backend)
    )
    decode_attn_backend = (
        str(decode_attn_backend)
        if decode_attn_backend is not None
        else str(args.roseinfer_decode_attn_backend)
    )
    paged_attn = bool(args.roseinfer_paged_attn)
    cuda_graph = bool(args.roseinfer_cuda_graph)
    chunked_prefill = bool(args.roseinfer_chunked_prefill)
    prefix_cache = bool(args.roseinfer_prefix_cache)
    kv_cache_max_concurrency = (
        int(kv_cache_max_concurrency)
        if kv_cache_max_concurrency is not None
        else int(getattr(args, "roseinfer_kv_cache_max_concurrency", 0))
    )
    prefix_cache_max_entries = (
        int(prefix_cache_max_entries)
        if prefix_cache_max_entries is not None
        else int(getattr(args, "roseinfer_prefix_cache_max_entries", 256))
    )
    fused_ops = (
        bool(fused_ops) if fused_ops is not None else bool(args.roseinfer_fused_ops)
    )
    fused_mlp = (
        bool(fused_mlp) if fused_mlp is not None else bool(args.roseinfer_fused_mlp)
    )
    fused_sampler = (
        bool(fused_sampler)
        if fused_sampler is not None
        else bool(args.roseinfer_fused_sampler)
    )
    fused_kv_append = (
        bool(fused_kv_append)
        if fused_kv_append is not None
        else bool(args.roseinfer_fused_kv_append)
    )
    overlap_schedule = (
        bool(overlap_schedule)
        if overlap_schedule is not None
        else bool(getattr(args, "roseinfer_overlap_schedule", True))
    )
    async_streaming = (
        bool(async_streaming)
        if async_streaming is not None
        else bool(getattr(args, "roseinfer_async_streaming", True))
    )
    engine_process = (
        bool(engine_process)
        if engine_process is not None
        else bool(getattr(args, "roseinfer_engine_process", True))
    )
    max_batch_size = (
        int(max_batch_size)
        if max_batch_size is not None
        else int(getattr(args, "roseinfer_max_batch_size", 8))
    )
    gc_freeze = (
        bool(gc_freeze)
        if gc_freeze is not None
        else bool(getattr(args, "roseinfer_gc_freeze", True))
    )
    fast_sse = (
        bool(fast_sse)
        if fast_sse is not None
        else bool(getattr(args, "roseinfer_fast_sse", True))
    )
    mp_ipc = (
        str(mp_ipc)
        if mp_ipc is not None
        else str(getattr(args, "roseinfer_mp_ipc", "pipe"))
    )
    mp_thread_cap = (
        bool(mp_thread_cap)
        if mp_thread_cap is not None
        else bool(getattr(args, "roseinfer_mp_thread_cap", True))
    )
    mp_affinity = (
        bool(mp_affinity)
        if mp_affinity is not None
        else bool(getattr(args, "roseinfer_mp_affinity", True))
    )
    mp_max_recv_per_iter = (
        int(mp_max_recv_per_iter)
        if mp_max_recv_per_iter is not None
        else int(getattr(args, "roseinfer_mp_max_recv_per_iter", 64))
    )
    mp_fill_target = (
        bool(mp_fill_target)
        if mp_fill_target is not None
        else bool(getattr(args, "roseinfer_mp_fill_target", True))
    )
    mp_flat_events = (
        bool(mp_flat_events)
        if mp_flat_events is not None
        else bool(getattr(args, "roseinfer_mp_flat_events", True))
    )
    mp_async_admit = (
        bool(mp_async_admit)
        if mp_async_admit is not None
        else bool(getattr(args, "roseinfer_mp_async_admit", False))
    )
    mp_tokenize_workers = (
        int(mp_tokenize_workers)
        if mp_tokenize_workers is not None
        else int(getattr(args, "roseinfer_mp_tokenize_workers", 0))
    )
    cmd = [
        sys.executable,
        "-m",
        "rosellm.roseinfer.server",
        "--hf-model-id",
        args.model,
        "--tokenizer-name",
        args.model,
        "--device",
        args.device,
        "--host",
        host,
        "--port",
        str(port),
        "--stream-interval",
        "1",
        "--max-batch-size",
        str(int(max_batch_size)),
        "--prefill-attn-backend",
        prefill_attn_backend,
        "--decode-attn-backend",
        decode_attn_backend,
    ]
    if not gc_freeze:
        cmd += ["--no-gc-freeze"]
    if not fast_sse:
        cmd += ["--no-fast-sse"]
    if args.max_inflight_requests is not None:
        cmd += ["--max-inflight-requests", str(args.max_inflight_requests)]
    cmd += ["--kv-cache-max-concurrency", str(int(kv_cache_max_concurrency))]
    cmd += ["--prefix-cache-max-entries", str(int(prefix_cache_max_entries))]
    if args.dtype == "fp32":
        cmd += ["--no-amp"]
    elif args.dtype == "bf16":
        cmd += ["--bf16"]
    if args.no_amp:
        cmd += ["--no-amp"]
    if args.bf16:
        cmd += ["--bf16"]
    if not async_streaming:
        cmd += ["--no-async-streaming"]
    cmd += ["--paged-attn" if paged_attn else "--no-paged-attn"]
    cmd += ["--cuda-graph" if cuda_graph else "--no-cuda-graph"]
    cmd += ["--chunked-prefill" if chunked_prefill else "--no-chunked-prefill"]
    if chunked_prefill:
        cmd += ["--prefill-chunk-size", str(int(args.roseinfer_prefill_chunk_size))]
    cmd += ["--prefix-cache" if prefix_cache else "--no-prefix-cache"]
    cmd += ["--fused-ops" if fused_ops else "--no-fused-ops"]
    cmd += ["--fused-mlp" if fused_mlp else "--no-fused-mlp"]
    cmd += ["--fused-sampler" if fused_sampler else "--no-fused-sampler"]
    cmd += ["--fused-kv-append" if fused_kv_append else "--no-fused-kv-append"]
    cmd += ["--overlap-schedule" if overlap_schedule else "--no-overlap-schedule"]
    cmd += ["--engine-process" if engine_process else "--no-engine-process"]
    if engine_process:
        cmd += ["--mp-ipc", str(mp_ipc)]
        cmd += ["--mp-max-recv-per-iter", str(int(mp_max_recv_per_iter))]
        cmd += ["--mp-fill-target" if mp_fill_target else "--no-mp-fill-target"]
        cmd += ["--mp-flat-events" if mp_flat_events else "--no-mp-flat-events"]
        cmd += ["--mp-thread-cap" if mp_thread_cap else "--no-mp-thread-cap"]
        cmd += ["--mp-affinity" if mp_affinity else "--no-mp-affinity"]
        cmd += ["--mp-async-admit" if mp_async_admit else "--no-mp-async-admit"]
        if int(mp_tokenize_workers) > 0:
            cmd += ["--mp-tokenize-workers", str(int(mp_tokenize_workers))]
    return cmd


def _dtype_flag_vllm(dtype: str) -> list[str]:
    dtype = dtype.lower()
    if dtype == "auto":
        return []
    if dtype == "fp16":
        return ["--dtype", "float16"]
    if dtype == "bf16":
        return ["--dtype", "bfloat16"]
    if dtype == "fp32":
        return ["--dtype", "float32"]
    raise ValueError(f"unsupported dtype for vLLM: {dtype}")


def _dtype_flag_sglang(dtype: str) -> list[str]:
    dtype = dtype.lower()
    if dtype == "auto":
        return []
    if dtype == "fp16":
        return ["--dtype", "float16"]
    if dtype == "bf16":
        return ["--dtype", "bfloat16"]
    if dtype == "fp32":
        return ["--dtype", "float32"]
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


def _vllm_server_cmd(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    max_context_len: int,
    python_exe: str | None = None,
    enable_layerwise_nvtx_tracing: bool = False,
) -> list[str]:
    py = python_exe or sys.executable
    cmd = [
        py,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(int(max_context_len)),
        "--disable-log-requests",
        "--disable-log-stats",
    ]
    if bool(getattr(args, "vllm_async_scheduling", True)):
        cmd.append("--async-scheduling")
    else:
        cmd.append("--no-async-scheduling")
    vllm_attention_backend = str(getattr(args, "vllm_attention_backend", "auto"))
    if vllm_attention_backend and vllm_attention_backend.lower() != "auto":
        cmd += ["--attention-backend", vllm_attention_backend]
    vllm_max_num_seqs = _resolve_vllm_max_num_seqs(args)
    if vllm_max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(int(vllm_max_num_seqs))]
    if enable_layerwise_nvtx_tracing:
        cmd.append("--enable-layerwise-nvtx-tracing")
    cmd += _dtype_flag_vllm(str(args.dtype))
    return cmd


def _trtllm_server_cmd(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    max_context_len: int,
    python_exe: str | None = None,
    backend: str | None = None,
) -> list[str]:
    py = python_exe or sys.executable
    cmd = [
        py,
        "-m",
        "tensorrt_llm.commands.serve",
        args.model,
        "--host",
        host,
        "--port",
        str(port),
        "--backend",
        str(
            backend
            if backend is not None
            else getattr(args, "trtllm_backend", "pytorch")
        ),
        "--max_seq_len",
        str(int(max_context_len)),
        "--log_level",
        "error",
    ]
    return cmd


def _sglang_server_cmd(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    max_context_len: int,
    python_exe: str | None = None,
    enable_layerwise_nvtx_marker: bool = False,
) -> list[str]:
    py = python_exe or sys.executable
    cmd = [
        py,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--host",
        host,
        "--port",
        str(port),
        "--context-length",
        str(int(max_context_len)),
        "--stream-interval",
        "1",
        "--stream-output",
    ]
    if getattr(args, "sglang_attention_backend", None):
        cmd += ["--attention-backend", str(args.sglang_attention_backend)]
    if getattr(args, "sglang_sampling_backend", None):
        cmd += ["--sampling-backend", str(args.sglang_sampling_backend)]
    if enable_layerwise_nvtx_marker:
        cmd.append("--enable-layerwise-nvtx-marker")
    cmd += _dtype_flag_sglang(str(args.dtype))
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online serving benchmark: roseinfer vs vLLM vs sglang via OpenAI servers.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HF model ID (used for all backends).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device for roseinfer ("cuda" or "cpu").',
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="DType hint for server env vars (best-effort).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help='CUDA_VISIBLE_DEVICES for all servers (e.g. "0").',
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
            "vLLM --max-num-seqs. Default: use vLLM default; if unset and "
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
        "--trtllm-backend",
        type=str,
        default="tensorrt",
        choices=["pytorch", "tensorrt", "trt", "_autodeploy"],
        help="TensorRT-LLM backend for OpenAI server (default: tensorrt).",
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
        "--trtllm-worker-single-process",
        action="store_true",
        help=(
            "TensorRT-LLM: set env TLLM_WORKER_USE_SINGLE_PROCESS=1 for TP=1 "
            "(may hurt streaming performance; default: disabled)."
        ),
    )
    parser.add_argument(
        "--trtllm-profile-start-stop",
        type=str,
        default="1-20",
        help=(
            "TensorRT-LLM profiling iteration spans for TLLM_PROFILE_START_STOP "
            '(format: "start-stop[,start-stop...]" or "iter[,iter...]"; default: 1-20). '
            "Used in the extra profiling stage."
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
        "--trace-a-url",
        type=str,
        default=TRACE_A_URL,
        help="TraceA JSONL URL.",
    )
    parser.add_argument(
        "--trace-a-path",
        type=str,
        default=None,
        help="Local TraceA JSONL path (skip download).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of requests to replay from the trace.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="0.4,0.5,0.6,0.7,0.8,1.6",
        help="Comma-separated timestamp scales (smaller = heavier load).",
    )
    parser.add_argument(
        "--start-offset-s",
        type=float,
        default=1.0,
        help="Delay before the first request (seconds).",
    )
    parser.add_argument(
        "--prompt-overhead-tokens",
        type=int,
        default=8,
        help="Safety margin for special tokens (subtracted from max context).",
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=128,
        help="Clamp trace output_length to this value (for faster runs).",
    )
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=None,
        help="Clamp trace input_length after context fitting (optional).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (non-OpenAI field; sent via extra_body).",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Force generating exactly max_tokens by ignoring EOS.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dummy prompt generation.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for servers.",
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=18000,
        help="Base port (backends use base+idx).",
    )
    parser.add_argument(
        "--server-cpus",
        type=str,
        default=None,
        help='CPU set for server processes (e.g. "0-15"). Default: first half.',
    )
    parser.add_argument(
        "--client-cpus",
        type=str,
        default=None,
        help='CPU set for client process (e.g. "16-31"). Default: second half.',
    )
    parser.add_argument(
        "--max-inflight-requests",
        type=int,
        default=None,
        help="roseinfer --max-inflight-requests (default: unlimited).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP for roseinfer.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 AMP for roseinfer.",
    )
    parser.add_argument(
        "--roseinfer-async-streaming",
        dest="roseinfer_async_streaming",
        action="store_true",
        help="Enable roseinfer async token/SSE streaming (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-async-streaming",
        dest="roseinfer_async_streaming",
        action="store_false",
        help="Disable roseinfer async token/SSE streaming.",
    )
    parser.set_defaults(roseinfer_async_streaming=True)
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
        "--roseinfer-kv-cache-max-concurrency",
        type=int,
        default=0,
        help="roseinfer --kv-cache-max-concurrency (default: 0 = auto).",
    )
    parser.add_argument(
        "--roseinfer-max-batch-size",
        type=int,
        default=8,
        help="roseinfer --max-batch-size (default: 8).",
    )
    parser.add_argument(
        "--roseinfer-gc-freeze",
        dest="roseinfer_gc_freeze",
        action="store_true",
        help="roseinfer --gc-freeze (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-gc-freeze",
        dest="roseinfer_gc_freeze",
        action="store_false",
        help="roseinfer --no-gc-freeze.",
    )
    parser.set_defaults(roseinfer_gc_freeze=True)
    parser.add_argument(
        "--roseinfer-fast-sse",
        dest="roseinfer_fast_sse",
        action="store_true",
        help="roseinfer --fast-sse (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-fast-sse",
        dest="roseinfer_fast_sse",
        action="store_false",
        help="roseinfer --no-fast-sse.",
    )
    parser.set_defaults(roseinfer_fast_sse=True)
    parser.add_argument(
        "--roseinfer-prefix-cache-max-entries",
        type=int,
        default=256,
        help="roseinfer --prefix-cache-max-entries (default: 256).",
    )
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
        "--roseinfer-engine-process",
        dest="roseinfer_engine_process",
        action="store_true",
        help="Run roseinfer engine in a worker process (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-engine-process",
        dest="roseinfer_engine_process",
        action="store_false",
        help="Disable roseinfer engine worker process (run in-process).",
    )
    parser.set_defaults(roseinfer_engine_process=True)
    parser.add_argument(
        "--roseinfer-compare-engine-process",
        action="store_true",
        help="Run roseinfer twice: engine-process on/off (for A/B benchmark).",
    )
    parser.add_argument(
        "--roseinfer-mp-ipc",
        type=str,
        default="pipe",
        help="roseinfer multiprocess IPC transport: queue|pipe (default: pipe).",
    )
    parser.add_argument(
        "--roseinfer-mp-max-recv-per-iter",
        type=int,
        default=64,
        help=(
            "roseinfer mp: max commands drained per engine loop iteration when busy; "
            "0 disables the budget (default: 64)."
        ),
    )
    parser.add_argument(
        "--roseinfer-mp-fill-target",
        dest="roseinfer_mp_fill_target",
        action="store_true",
        help="roseinfer mp: bypass cmd budget while ramping up below target concurrency (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-mp-fill-target",
        dest="roseinfer_mp_fill_target",
        action="store_false",
        help="roseinfer mp: disable ramp-up fill-to-target behavior.",
    )
    parser.set_defaults(roseinfer_mp_fill_target=True)
    parser.add_argument(
        "--roseinfer-mp-flat-events",
        dest="roseinfer_mp_flat_events",
        action="store_true",
        help="roseinfer mp: use flat token-pair IPC events (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-mp-flat-events",
        dest="roseinfer_mp_flat_events",
        action="store_false",
        help="roseinfer mp: disable flat token-pair IPC events.",
    )
    parser.set_defaults(roseinfer_mp_flat_events=True)
    parser.add_argument(
        "--roseinfer-mp-thread-cap",
        dest="roseinfer_mp_thread_cap",
        action="store_true",
        help="roseinfer mp: cap torch threads to 1 (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-mp-thread-cap",
        dest="roseinfer_mp_thread_cap",
        action="store_false",
        help="roseinfer mp: disable torch thread capping.",
    )
    parser.set_defaults(roseinfer_mp_thread_cap=True)
    parser.add_argument(
        "--roseinfer-mp-affinity",
        dest="roseinfer_mp_affinity",
        action="store_true",
        help="roseinfer mp: split CPU affinity between API and engine (default: enabled).",
    )
    parser.add_argument(
        "--roseinfer-no-mp-affinity",
        dest="roseinfer_mp_affinity",
        action="store_false",
        help="roseinfer mp: disable CPU affinity split.",
    )
    parser.set_defaults(roseinfer_mp_affinity=True)
    parser.add_argument(
        "--roseinfer-mp-async-admit",
        dest="roseinfer_mp_async_admit",
        action="store_true",
        help="roseinfer mp: enable API-side async admit threads (--mp-async-admit).",
    )
    parser.add_argument(
        "--roseinfer-no-mp-async-admit",
        dest="roseinfer_mp_async_admit",
        action="store_false",
        help="roseinfer mp: disable API-side async admit threads.",
    )
    parser.set_defaults(roseinfer_mp_async_admit=True)
    parser.add_argument(
        "--roseinfer-mp-tokenize-workers",
        type=int,
        default=4,
        help=(
            "roseinfer mp: API-side tokenization worker threads (--mp-tokenize-workers) "
            "when --mp-async-admit is enabled (default: 4)."
        ),
    )
    parser.add_argument(
        "--roseinfer-compare-mp-ablations",
        action="store_true",
        help=(
            "Add roseinfer mp ablation variants (disable one mp optimization at a time). "
            "Use with --roseinfer-engine-process."
        ),
    )
    parser.add_argument(
        "--timeout-ready-s",
        type=float,
        default=120.0,
        help="Max seconds to wait for each server to be ready.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=5,
        help="Warmup requests per backend (0 to disable).",
    )
    parser.add_argument(
        "--warmup-input-len",
        type=int,
        default=128,
        help="Warmup prompt length in tokens (approx).",
    )
    parser.add_argument(
        "--warmup-output-len",
        type=int,
        default=32,
        help="Warmup output length in tokens.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmarks/serving",
        help="Output directory for JSON results and logs.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="none",
        choices=["none", "torch", "nsys", "both"],
        help=(
            "Extra profiling stage (separate server run; not included in online_results.json). "
            "Use --profile-only to skip the trace benchmark."
        ),
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Skip trace benchmark; only run the profiling stage.",
    )
    parser.add_argument(
        "--profile-n",
        type=int,
        default=32,
        help="Number of requests for the profiling stage (default: 32).",
    )
    parser.add_argument(
        "--profile-scale",
        type=float,
        default=0.4,
        help="Trace time scale for the profiling stage (default: 0.4).",
    )
    parser.add_argument(
        "--profile-output-len",
        type=int,
        default=None,
        help=(
            "Output length clamp for the profiling stage (default: use --max-output-len)."
        ),
    )
    parser.add_argument(
        "--profile-torch-minimal",
        dest="profile_torch_minimal",
        action="store_true",
        help=(
            "Profiling stage: capture a short, representative torch-profiler window to "
            "keep traces small (default: enabled)."
        ),
    )
    parser.add_argument(
        "--profile-torch-full",
        dest="profile_torch_minimal",
        action="store_false",
        help="Profiling stage: capture a full torch-profiler trace (may be very large).",
    )
    parser.set_defaults(profile_torch_minimal=True)
    parser.add_argument(
        "--profile-torch-delay-steps",
        type=int,
        default=0,
        help=(
            "Profiling stage (torch, minimal): skip this many engine steps before "
            "recording (default: 0)."
        ),
    )
    parser.add_argument(
        "--profile-torch-num-steps",
        type=int,
        default=20,
        help=(
            "Profiling stage (torch, minimal): record this many engine steps "
            "(default: 20)."
        ),
    )
    parser.add_argument(
        "--profile-torch-with-stack",
        dest="profile_torch_with_stack",
        action="store_true",
        help="Profiling stage: collect Python/C++ stacks in torch profiler.",
    )
    parser.add_argument(
        "--profile-torch-no-with-stack",
        dest="profile_torch_with_stack",
        action="store_false",
        help="Profiling stage: disable stack collection in torch profiler (default).",
    )
    parser.set_defaults(profile_torch_with_stack=False)
    parser.add_argument(
        "--profile-torch-record-shapes",
        dest="profile_torch_record_shapes",
        action="store_true",
        help="Profiling stage: record tensor shapes in torch profiler.",
    )
    parser.add_argument(
        "--profile-torch-no-record-shapes",
        dest="profile_torch_record_shapes",
        action="store_false",
        help="Profiling stage: disable recording shapes in torch profiler (default).",
    )
    parser.set_defaults(profile_torch_record_shapes=False)
    parser.add_argument(
        "--profile-torch-with-memory",
        dest="profile_torch_with_memory",
        action="store_true",
        help="Profiling stage: enable memory profiling when supported.",
    )
    parser.add_argument(
        "--profile-torch-no-with-memory",
        dest="profile_torch_with_memory",
        action="store_false",
        help="Profiling stage: disable memory profiling when supported (default: disabled).",
    )
    parser.set_defaults(profile_torch_with_memory=False)
    parser.add_argument(
        "--profile-vllm-nvtx-scopes",
        dest="profile_vllm_nvtx_scopes",
        action="store_true",
        help="Profiling stage: enable vLLM layerwise NVTX tracing for nsys (default: enabled).",
    )
    parser.add_argument(
        "--profile-no-vllm-nvtx-scopes",
        dest="profile_vllm_nvtx_scopes",
        action="store_false",
        help="Profiling stage: disable vLLM layerwise NVTX tracing for nsys.",
    )
    parser.set_defaults(profile_vllm_nvtx_scopes=True)
    parser.add_argument(
        "--profile-sglang-layerwise-nvtx",
        dest="profile_sglang_layerwise_nvtx",
        action="store_true",
        help="Profiling stage: enable SGLang layerwise NVTX markers for nsys (default: enabled).",
    )
    parser.add_argument(
        "--profile-no-sglang-layerwise-nvtx",
        dest="profile_sglang_layerwise_nvtx",
        action="store_false",
        help="Profiling stage: disable SGLang layerwise NVTX markers for nsys.",
    )
    parser.set_defaults(profile_sglang_layerwise_nvtx=True)
    parser.add_argument(
        "--profile-nsys-trace",
        type=str,
        default="cuda,nvtx,osrt",
        help=(
            "Profiling stage: value for `nsys profile --trace=...` "
            "(default: cuda,nvtx,osrt)."
        ),
    )
    parser.add_argument(
        "--profile-nsys-cuda-graph-trace",
        type=str,
        default="node",
        help=(
            "Profiling stage: value for `nsys profile --cuda-graph-trace=...` "
            "(default: node)."
        ),
    )
    parser.add_argument(
        "--profile-nsys-cpuctxsw",
        type=str,
        default=None,
        help=(
            "Profiling stage: optional value for `nsys profile --cpuctxsw=...` "
            "(e.g. none, process-tree, system-wide). When unset, nsys default is used."
        ),
    )
    parser.add_argument(
        "--profile-nsys-kill",
        type=str,
        default=None,
        help=(
            "Profiling stage: optional value for `nsys profile --kill=...` "
            "(e.g. none, sigterm, sigkill, or a signal number). "
            "When unset, nsys default is used."
        ),
    )
    parser.add_argument(
        "--profile-nsys-cuda-flush-interval-ms",
        type=int,
        default=None,
        help=(
            "Profiling stage: optional value for `nsys profile --cuda-flush-interval=...` "
            "in milliseconds. When unset, nsys default is used."
        ),
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="roseinfer,vllm,sglang,trtllm",
        help="Comma-separated backends to run (roseinfer,vllm,sglang,trtllm).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"online_{run_id}"
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

    profile_mode = str(getattr(args, "profile", "none")).lower()
    profile_only = bool(getattr(args, "profile_only", False))
    if profile_only and profile_mode == "none":
        raise ValueError("--profile-only requires --profile torch|nsys|both")
    if profile_mode not in ("none", "torch", "nsys", "both"):
        raise ValueError("--profile must be one of: none, torch, nsys, both")

    if args.server_cpus is None or args.client_cpus is None:
        server_default, client_default = _default_cpu_sets()
        server_cpus = (
            server_default
            if args.server_cpus is None
            else _parse_cpu_set(args.server_cpus)
        )
        client_cpus = (
            client_default
            if args.client_cpus is None
            else _parse_cpu_set(args.client_cpus)
        )
    else:
        server_cpus = _parse_cpu_set(args.server_cpus)
        client_cpus = _parse_cpu_set(args.client_cpus)

    os.sched_setaffinity(0, set(client_cpus))

    trace_path = (
        Path(args.trace_a_path).expanduser().resolve()
        if args.trace_a_path is not None
        else _download_if_needed(
            args.trace_a_url,
            Path("~/.cache/rosellm/benchmarks/traceA.jsonl").expanduser(),
        )
    )
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
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
        roseinfer_async_streaming: bool | None = None
        roseinfer_engine_process: bool | None = None
        roseinfer_max_batch_size: int | None = None
        roseinfer_gc_freeze: bool | None = None
        roseinfer_fast_sse: bool | None = None
        roseinfer_kv_cache_max_concurrency: int | None = None
        roseinfer_prefix_cache_max_entries: int | None = None
        roseinfer_fused_ops: bool | None = None
        roseinfer_fused_mlp: bool | None = None
        roseinfer_fused_sampler: bool | None = None
        roseinfer_fused_kv_append: bool | None = None
        roseinfer_overlap_schedule: bool | None = None
        roseinfer_mp_ipc: str | None = None
        roseinfer_mp_thread_cap: bool | None = None
        roseinfer_mp_affinity: bool | None = None
        roseinfer_mp_max_recv_per_iter: int | None = None
        roseinfer_mp_fill_target: bool | None = None
        roseinfer_mp_flat_events: bool | None = None
        roseinfer_mp_async_admit: bool | None = None
        roseinfer_mp_tokenize_workers: int | None = None

    run_specs: list[RunSpec] = []
    for backend in backends:
        if backend != "roseinfer":
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
        base_engine_process = bool(getattr(args, "roseinfer_engine_process", True))
        base_async_streaming = bool(getattr(args, "roseinfer_async_streaming", True))
        base_max_batch_size = int(getattr(args, "roseinfer_max_batch_size", 8))
        base_gc_freeze = bool(getattr(args, "roseinfer_gc_freeze", True))
        base_fast_sse = bool(getattr(args, "roseinfer_fast_sse", True))
        base_kv_conc = int(getattr(args, "roseinfer_kv_cache_max_concurrency", 0))
        base_prefix_entries = int(
            getattr(args, "roseinfer_prefix_cache_max_entries", 256)
        )
        base_mp_ipc = str(getattr(args, "roseinfer_mp_ipc", "pipe"))
        base_mp_thread_cap = bool(getattr(args, "roseinfer_mp_thread_cap", True))
        base_mp_affinity = bool(getattr(args, "roseinfer_mp_affinity", True))
        base_mp_max_recv = int(getattr(args, "roseinfer_mp_max_recv_per_iter", 64))
        base_mp_fill_target = bool(getattr(args, "roseinfer_mp_fill_target", True))
        base_mp_flat_events = bool(getattr(args, "roseinfer_mp_flat_events", True))
        base_mp_async_admit = bool(getattr(args, "roseinfer_mp_async_admit", False))
        base_mp_tokenize_workers = int(
            getattr(args, "roseinfer_mp_tokenize_workers", 0)
        )
        engine_cfgs: list[bool] = [base_engine_process]
        if bool(getattr(args, "roseinfer_compare_engine_process", False)):
            engine_cfgs.append(not base_engine_process)
        engine_cfgs = list(dict.fromkeys(engine_cfgs))
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
            for engine_process in engine_cfgs:
                for (
                    fused_ops,
                    fused_mlp,
                    fused_sampler,
                    fused_kv_append,
                    overlap_schedule,
                ) in cfgs:
                    label = base_label
                    if not bool(engine_process):
                        label += "+inproc"
                    if not fused_ops:
                        label += "+nofuse"
                    if not fused_mlp:
                        label += "+nomlp"
                    if not fused_sampler:
                        label += "+nosampler"
                    if not fused_kv_append:
                        label += "+nokv"
                    if not overlap_schedule:
                        label += "+nooverlap"
                    if base_max_batch_size != 8:
                        label += f"+batch{base_max_batch_size}"
                    if not base_gc_freeze:
                        label += "+nogc"
                    if not base_fast_sse:
                        label += "+nofastsse"
                    default_mp_async_admit = True
                    default_mp_tokenize_workers = 4
                    if base_mp_async_admit != default_mp_async_admit:
                        label += "+nompasync"
                    if base_mp_async_admit:
                        tok_workers = (
                            int(base_mp_tokenize_workers)
                            if int(base_mp_tokenize_workers) > 0
                            else 1
                        )
                        if tok_workers != int(default_mp_tokenize_workers):
                            label += f"+tok{tok_workers}"
                    run_specs.append(
                        RunSpec(
                            base_backend="roseinfer",
                            label=label,
                            roseinfer_prefill_backend=prefill_backend,
                            roseinfer_async_streaming=bool(base_async_streaming),
                            roseinfer_engine_process=bool(engine_process),
                            roseinfer_max_batch_size=int(base_max_batch_size),
                            roseinfer_gc_freeze=bool(base_gc_freeze),
                            roseinfer_fast_sse=bool(base_fast_sse),
                            roseinfer_kv_cache_max_concurrency=int(base_kv_conc),
                            roseinfer_prefix_cache_max_entries=int(base_prefix_entries),
                            roseinfer_fused_ops=bool(fused_ops),
                            roseinfer_fused_mlp=bool(fused_mlp),
                            roseinfer_fused_sampler=bool(fused_sampler),
                            roseinfer_fused_kv_append=bool(fused_kv_append),
                            roseinfer_overlap_schedule=bool(overlap_schedule),
                            roseinfer_mp_ipc=base_mp_ipc,
                            roseinfer_mp_thread_cap=base_mp_thread_cap,
                            roseinfer_mp_affinity=base_mp_affinity,
                            roseinfer_mp_max_recv_per_iter=base_mp_max_recv,
                            roseinfer_mp_fill_target=base_mp_fill_target,
                            roseinfer_mp_flat_events=base_mp_flat_events,
                            roseinfer_mp_async_admit=base_mp_async_admit,
                            roseinfer_mp_tokenize_workers=base_mp_tokenize_workers,
                        )
                    )

    if bool(getattr(args, "roseinfer_compare_mp_ablations", False)):
        extra: list[RunSpec] = []
        seen_labels = {spec.label for spec in run_specs}
        for spec in list(run_specs):
            if spec.base_backend != "roseinfer" or not bool(
                spec.roseinfer_engine_process
            ):
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
            if bool(spec.roseinfer_mp_fill_target):
                add_variant(suffix="+nofill", roseinfer_mp_fill_target=False)
            if int(spec.roseinfer_mp_max_recv_per_iter or 0) != 0:
                add_variant(suffix="+nodrain", roseinfer_mp_max_recv_per_iter=0)
            if str(spec.roseinfer_mp_ipc or "").lower() != "queue":
                add_variant(suffix="+queueipc", roseinfer_mp_ipc="queue")
            if bool(spec.roseinfer_mp_flat_events):
                add_variant(suffix="+noflat", roseinfer_mp_flat_events=False)
            if bool(spec.roseinfer_async_streaming):
                add_variant(suffix="+noasync", roseinfer_async_streaming=False)
            if (
                int(spec.roseinfer_kv_cache_max_concurrency or 0) == 0
                and args.max_inflight_requests is not None
                and int(args.max_inflight_requests) != 256
            ):
                add_variant(
                    suffix="+kv256",
                    roseinfer_kv_cache_max_concurrency=256,
                )

        run_specs.extend(extra)

    if not run_specs:
        raise ValueError("no backends to run (all candidates were skipped)")
    scales = _parse_csv_floats(args.scales)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    max_ctx = _resolve_model_max_context(args.model, tokenizer)
    rows = _read_trace_a_jsonl(trace_path, n=int(args.n))

    summaries: list[OnlineBenchmarkSummary] = []
    raw_per_backend: dict[str, dict[str, Any]] = {}

    def run_profile_stage() -> None:
        if profile_mode == "none":
            return
        tools = ["torch", "nsys"] if profile_mode == "both" else [profile_mode]
        profile_n = int(getattr(args, "profile_n", 32))
        profile_scale = float(getattr(args, "profile_scale", 0.4))
        profile_output_len: int | None = getattr(args, "profile_output_len", None)
        if profile_output_len is None:
            profile_output_len = (
                int(args.max_output_len) if args.max_output_len is not None else None
            )
        if profile_n <= 0:
            raise ValueError("--profile-n must be >= 1")
        if profile_output_len is None:
            raise ValueError(
                "--profile-output-len must be set when --max-output-len is unset"
            )
        if profile_output_len <= 0:
            raise ValueError("--profile-output-len must be >= 1")
        if not tools:
            return

        traces = _build_trace_items(
            rows=rows,
            tokenizer=tokenizer,
            max_ctx=max_ctx,
            scale=profile_scale,
            start_offset_s=float(args.start_offset_s),
            max_input_len=(
                int(args.max_input_len) if args.max_input_len is not None else None
            ),
            max_output_len=profile_output_len,
            prompt_overhead_tokens=int(args.prompt_overhead_tokens),
            seed=int(args.seed),
        )
        traces = traces[:profile_n]
        if not traces:
            raise ValueError(
                "no profiling traces built; check --profile-n and trace input"
            )

        profiles_root = run_dir / "profiles"
        profiles_root.mkdir(parents=True, exist_ok=True)
        profile_start_time = _iso_now()
        profile_t0 = time.perf_counter()

        manifest: dict[str, Any] = {
            "meta": {
                "profile": profile_mode,
                "profile_only": profile_only,
                "profile_n": profile_n,
                "profile_scale": profile_scale,
                "profile_output_len": profile_output_len,
                "profile_start_offset_s": float(args.start_offset_s),
                "profile_nsys_trace": str(
                    getattr(args, "profile_nsys_trace", "cuda,nvtx,osrt")
                ),
                "profile_nsys_cuda_graph_trace": str(
                    getattr(args, "profile_nsys_cuda_graph_trace", "node")
                ),
                "profile_nsys_cpuctxsw": (
                    str(getattr(args, "profile_nsys_cpuctxsw"))
                    if getattr(args, "profile_nsys_cpuctxsw", None) is not None
                    else None
                ),
                "profile_nsys_kill": (
                    str(getattr(args, "profile_nsys_kill"))
                    if getattr(args, "profile_nsys_kill", None) is not None
                    else None
                ),
                "profile_nsys_cuda_flush_interval_ms": (
                    int(getattr(args, "profile_nsys_cuda_flush_interval_ms"))
                    if getattr(args, "profile_nsys_cuda_flush_interval_ms", None) is not None
                    else None
                ),
                "profile_torch_minimal": bool(
                    getattr(args, "profile_torch_minimal", True)
                ),
                "profile_torch_delay_steps": int(
                    getattr(args, "profile_torch_delay_steps", 0)
                ),
                "profile_torch_num_steps": int(getattr(args, "profile_torch_num_steps", 0)),
                "profile_torch_with_stack": bool(
                    getattr(args, "profile_torch_with_stack", True)
                ),
                "profile_torch_record_shapes": bool(
                    getattr(args, "profile_torch_record_shapes", True)
                ),
                "profile_torch_with_memory": bool(
                    getattr(args, "profile_torch_with_memory", True)
                ),
                "profile_vllm_nvtx_scopes": bool(
                    getattr(args, "profile_vllm_nvtx_scopes", True)
                ),
                "profile_sglang_layerwise_nvtx": bool(
                    getattr(args, "profile_sglang_layerwise_nvtx", True)
                ),
                "tools": tools,
                "backends": [spec.label for spec in run_specs],
                "server_cpus": server_cpus,
                "client_cpus": client_cpus,
                "profile_start_time": profile_start_time,
                "warmup_requests": int(args.warmup_requests),
                "trtllm_worker_single_process": bool(
                    getattr(args, "trtllm_worker_single_process", False)
                ),
                "trtllm_profile_start_stop": str(
                    getattr(args, "trtllm_profile_start_stop", "1-20")
                ),
                "trtllm_profile_record_gc": bool(
                    getattr(args, "trtllm_profile_record_gc", True)
                ),
                "vllm_async_scheduling": bool(
                    getattr(args, "vllm_async_scheduling", True)
                ),
                "versions": versions,
            },
            "runs": [],
        }

        def http_post(url: str, *, json_body: dict[str, Any] | None = None) -> None:
            r = httpx.post(url, json=json_body, timeout=300.0)
            if r.status_code != 200:
                raise RuntimeError(f"POST {url} failed: {r.status_code} {r.text}")

        def http_post_ok(
            url: str, *, json_body: dict[str, Any] | None = None
        ) -> str | None:
            try:
                http_post(url, json_body=json_body)
                return None
            except Exception as exc:
                return str(exc)

        used_ports: set[int] = set()
        for tool in tools:
            for i, spec in enumerate(run_specs):
                base_backend = spec.base_backend
                backend = spec.label
                prefill_backend = spec.roseinfer_prefill_backend
                engine_process = spec.roseinfer_engine_process
                fused_ops = spec.roseinfer_fused_ops
                fused_mlp = spec.roseinfer_fused_mlp
                fused_sampler = spec.roseinfer_fused_sampler
                fused_kv_append = spec.roseinfer_fused_kv_append
                overlap_schedule = spec.roseinfer_overlap_schedule
                mp_ipc = spec.roseinfer_mp_ipc
                mp_thread_cap = spec.roseinfer_mp_thread_cap
                mp_affinity = spec.roseinfer_mp_affinity
                mp_max_recv_per_iter = spec.roseinfer_mp_max_recv_per_iter
                mp_fill_target = spec.roseinfer_mp_fill_target
                mp_flat_events = spec.roseinfer_mp_flat_events
                mp_async_admit = spec.roseinfer_mp_async_admit
                mp_tokenize_workers = spec.roseinfer_mp_tokenize_workers

                port = _pick_free_port(args.host, int(args.port_base) + i, used_ports)
                root_url = f"http://{args.host}:{port}"
                base_url = f"{root_url}/v1"
                out_dir = profiles_root / tool / backend
                out_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / f"{backend}.profile_{tool}.server.log"

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
                env["OMP_NUM_THREADS"] = str(len(server_cpus))
                env["MKL_NUM_THREADS"] = str(len(server_cpus))
                env["TOKENIZERS_PARALLELISM"] = "false"
                env["ROSEINFER_NVTX"] = "1"

                if base_backend == "vllm":
                    if tool == "torch":
                        env["VLLM_TORCH_PROFILER_DIR"] = str(out_dir)
                        env["VLLM_TORCH_PROFILER_WITH_STACK"] = (
                            "1"
                            if bool(getattr(args, "profile_torch_with_stack", True))
                            else "0"
                        )
                        env["VLLM_TORCH_PROFILER_RECORD_SHAPES"] = (
                            "1"
                            if bool(getattr(args, "profile_torch_record_shapes", True))
                            else "0"
                        )
                        env["VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"] = (
                            "1"
                            if bool(getattr(args, "profile_torch_with_memory", True))
                            else "0"
                        )
                        if bool(getattr(args, "profile_torch_minimal", True)):
                            delay_steps = int(getattr(args, "profile_torch_delay_steps", 0))
                            num_steps = int(getattr(args, "profile_torch_num_steps", 0))
                            env["VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM"] = "1"
                            env["VLLM_PROFILER_DELAY_ITERS"] = str(max(0, delay_steps))
                            env["VLLM_PROFILER_MAX_ITERS"] = str(max(0, num_steps))
                elif base_backend == "trtllm":
                    env["TLLM_LLMAPI_ENABLE_NVTX"] = "1"
                    env["TLLM_PROFILE_START_STOP"] = str(args.trtllm_profile_start_stop)
                    if bool(getattr(args, "trtllm_profile_record_gc", True)):
                        env["TLLM_PROFILE_RECORD_GC"] = "1"
                    else:
                        env.pop("TLLM_PROFILE_RECORD_GC", None)
                    if bool(getattr(args, "trtllm_worker_single_process", False)):
                        env["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
                    else:
                        env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
                    if tool == "torch":
                        env["TLLM_TORCH_PROFILE_TRACE"] = str(out_dir / "trace.json")
                elif base_backend == "roseinfer":
                    if tool == "torch":
                        env["ROSEINFER_TORCH_PROFILE_WITH_STACK"] = (
                            "1"
                            if bool(getattr(args, "profile_torch_with_stack", True))
                            else "0"
                        )
                        env["ROSEINFER_TORCH_PROFILE_RECORD_SHAPES"] = (
                            "1"
                            if bool(getattr(args, "profile_torch_record_shapes", True))
                            else "0"
                        )
                        env["ROSEINFER_TORCH_PROFILE_WITH_PROFILE_MEMORY"] = (
                            "1"
                            if bool(getattr(args, "profile_torch_with_memory", True))
                            else "0"
                        )
                        if bool(getattr(args, "profile_torch_minimal", True)):
                            env["ROSEINFER_TORCH_PROFILE_DELAY_STEPS"] = str(
                                max(0, int(getattr(args, "profile_torch_delay_steps", 0)))
                            )
                            env["ROSEINFER_TORCH_PROFILE_NUM_STEPS"] = str(
                                max(0, int(getattr(args, "profile_torch_num_steps", 0)))
                            )

                if base_backend == "roseinfer":
                    cmd = _roseinfer_server_cmd(
                        args,
                        host=args.host,
                        port=port,
                        async_streaming=spec.roseinfer_async_streaming,
                        prefill_attn_backend=prefill_backend,
                        engine_process=engine_process,
                        max_batch_size=spec.roseinfer_max_batch_size,
                        gc_freeze=spec.roseinfer_gc_freeze,
                        fast_sse=spec.roseinfer_fast_sse,
                        kv_cache_max_concurrency=spec.roseinfer_kv_cache_max_concurrency,
                        prefix_cache_max_entries=spec.roseinfer_prefix_cache_max_entries,
                        fused_ops=fused_ops,
                        fused_mlp=fused_mlp,
                        fused_sampler=fused_sampler,
                        fused_kv_append=fused_kv_append,
                        overlap_schedule=overlap_schedule,
                        mp_ipc=mp_ipc,
                        mp_thread_cap=mp_thread_cap,
                        mp_affinity=mp_affinity,
                        mp_max_recv_per_iter=mp_max_recv_per_iter,
                        mp_fill_target=mp_fill_target,
                        mp_flat_events=mp_flat_events,
                        mp_async_admit=mp_async_admit,
                        mp_tokenize_workers=mp_tokenize_workers,
                    )
                elif base_backend == "vllm":
                    if vllm_python is not None:
                        env.pop("PYTHONPATH", None)
                    cmd = _vllm_server_cmd(
                        args,
                        host=args.host,
                        port=port,
                        max_context_len=max_ctx,
                        python_exe=vllm_python,
                        enable_layerwise_nvtx_tracing=(
                            tool == "nsys"
                            and bool(getattr(args, "profile_vllm_nvtx_scopes", True))
                        ),
                    )
                elif base_backend == "sglang":
                    if sglang_python is None:
                        _maybe_add_sglang_source_pythonpath(env)
                    else:
                        env.pop("PYTHONPATH", None)
                    cmd = _sglang_server_cmd(
                        args,
                        host=args.host,
                        port=port,
                        max_context_len=max_ctx,
                        python_exe=sglang_python,
                        enable_layerwise_nvtx_marker=(
                            tool == "nsys"
                            and bool(getattr(args, "profile_sglang_layerwise_nvtx", True))
                        ),
                    )
                elif base_backend == "trtllm":
                    if trtllm_python:
                        env.pop("PYTHONPATH", None)
                        env = _trtllm_runtime_env(
                            python_exe=trtllm_python, base_env=env
                        )
                    else:
                        _maybe_add_trtllm_source_pythonpath(env)
                    cmd = _trtllm_server_cmd(
                        args,
                        host=args.host,
                        port=port,
                        max_context_len=max_ctx,
                        python_exe=trtllm_python,
                        backend=str(getattr(args, "trtllm_backend", "tensorrt")),
                    )
                else:
                    raise ValueError(f"unknown backend: {base_backend}")

                nsys_prefix = None
                nsys_capture_range = None
                if tool == "nsys":
                    nsys_prefix = out_dir / backend
                    # NOTE: Many backends (roseinfer multiprocess / SGLang 3-stage) spawn
                    # worker processes early. `capture-range=cudaProfilerApi` can miss CUDA
                    # kernels from already-running children; profiling the full process tree
                    # is more reliable for cross-backend comparisons.
                    nsys_capture_range = "none"
                    nsys_trace = str(
                        getattr(args, "profile_nsys_trace", "cuda,nvtx,osrt")
                    )
                    nsys_cuda_graph_trace = str(
                        getattr(args, "profile_nsys_cuda_graph_trace", "node")
                    )
                    nsys_cpuctxsw = getattr(args, "profile_nsys_cpuctxsw", None)
                    nsys_kill = getattr(args, "profile_nsys_kill", None)
                    nsys_cuda_flush_ms = getattr(
                        args, "profile_nsys_cuda_flush_interval_ms", None
                    )
                    cpu_str = _format_cpu_set(server_cpus)
                    cmd = [
                        "nsys",
                        "profile",
                        "--force-overwrite=true",
                        f"--capture-range={nsys_capture_range}",
                        f"--trace={nsys_trace}",
                        f"--cuda-graph-trace={nsys_cuda_graph_trace}",
                        *(
                            [f"--kill={str(nsys_kill)}"]
                            if nsys_kill is not None
                            else []
                        ),
                        *(
                            [f"--cuda-flush-interval={int(nsys_cuda_flush_ms)}"]
                            if nsys_cuda_flush_ms is not None
                            else []
                        ),
                        *(
                            [f"--cpuctxsw={str(nsys_cpuctxsw)}"]
                            if nsys_cpuctxsw is not None
                            else []
                        ),
                        "--sample=none",
                        "--trace-fork-before-exec=true",
                        "-o",
                        str(nsys_prefix),
                        "taskset",
                        "-c",
                        cpu_str,
                        *cmd,
                    ]

                server = _start_process(
                    cmd=cmd, env=env, cpu_set=server_cpus, log_path=log_path
                )
                stage_t0 = time.perf_counter()
                run_record: dict[str, Any] | None = None
                try:
                    profile_err: str | None = None
                    try:
                        asyncio.run(
                            _wait_ready(root_url, timeout_s=float(args.timeout_ready_s))
                        )
                    except Exception as exc:
                        profile_err = f"server not ready: {exc}"

                    warmup_n = int(args.warmup_requests)
                    if profile_err is None and warmup_n > 0:
                        try:
                            effective_ctx = max(
                                2,
                                int(max_ctx) - int(args.prompt_overhead_tokens) - 1,
                            )
                            warmup_in = int(
                                min(
                                    int(args.warmup_input_len),
                                    max(
                                        1,
                                        effective_ctx - int(args.warmup_output_len),
                                    ),
                                )
                            )
                            warmup_ids = _make_base_prompt_ids(
                                tokenizer, warmup_in, seed=int(args.seed)
                            )
                            warmup_prompt = tokenizer.decode(warmup_ids)
                            asyncio.run(
                                _warmup(
                                    base_url=base_url,
                                    model=args.model,
                                    prompt=warmup_prompt,
                                    output_len=int(args.warmup_output_len),
                                    num_requests=warmup_n,
                                    temperature=float(args.temperature),
                                    top_p=float(args.top_p),
                                    top_k=int(args.top_k),
                                    ignore_eos=bool(args.ignore_eos),
                                )
                            )
                        except Exception as exc:
                            profile_err = f"warmup failed: {exc}"

                    if profile_err is None:
                        if base_backend == "vllm":
                            if tool == "torch":
                                err = http_post_ok(f"{root_url}/start_profile")
                            else:
                                err = None
                        elif base_backend == "sglang":
                            if tool == "torch":
                                activities = ["CPU", "GPU"]
                                if bool(getattr(args, "profile_torch_with_memory", True)):
                                    activities.append("MEM")
                                body = {
                                    "output_dir": str(out_dir),
                                    "activities": activities,
                                    "with_stack": bool(
                                        getattr(args, "profile_torch_with_stack", True)
                                    ),
                                    "record_shapes": bool(
                                        getattr(args, "profile_torch_record_shapes", True)
                                    ),
                                }
                                if bool(getattr(args, "profile_torch_minimal", True)):
                                    delay_steps = max(
                                        0,
                                        int(getattr(args, "profile_torch_delay_steps", 0)),
                                    )
                                    num_steps = max(
                                        0,
                                        int(getattr(args, "profile_torch_num_steps", 0)),
                                    )
                                    if delay_steps > 0:
                                        body["start_step"] = delay_steps
                                    if num_steps > 0:
                                        body["num_steps"] = num_steps
                                    body["profile_prefix"] = backend
                                err = http_post_ok(
                                    f"{root_url}/start_profile",
                                    json_body=body,
                                )
                            else:
                                # With `nsys_capture_range=none`, avoid extra CUDA profiler
                                # start/stop calls (they can confuse multi-process capture).
                                err = None
                        elif base_backend == "roseinfer":
                            if tool == "torch":
                                err = http_post_ok(
                                    f"{root_url}/start_profile",
                                    json_body={
                                        "tool": "torch",
                                        "output_dir": str(out_dir),
                                    },
                                )
                            else:
                                # With `nsys_capture_range=none`, let Nsight Systems capture
                                # from process start; don't use cudaProfilerStart/Stop.
                                err = None
                        else:
                            err = None

                        if err is not None:
                            profile_err = f"start_profile failed: {err}"

                    if profile_err is None:
                        try:
                            asyncio.run(
                                run_trace_benchmark(
                                    base_url=base_url,
                                    model=args.model,
                                    traces=traces,
                                    temperature=float(args.temperature),
                                    top_p=float(args.top_p),
                                    top_k=int(args.top_k),
                                    ignore_eos=bool(args.ignore_eos),
                                )
                            )
                        except Exception as exc:
                            profile_err = f"trace benchmark failed: {exc}"

                    if (
                        profile_err is None
                        and base_backend == "trtllm"
                        and tool == "torch"
                    ):
                        trace_path = out_dir / "trace.json"
                        deadline = time.time() + 15.0
                        while time.time() < deadline and not trace_path.exists():
                            time.sleep(0.25)
                        if not trace_path.exists():
                            profile_err = (
                                "torch trace not produced (TensorRT backend); "
                                "use nsys for GPU profiling"
                            )

                    if profile_err is None:
                        if base_backend == "vllm":
                            if tool == "torch":
                                err = http_post_ok(f"{root_url}/stop_profile")
                                if err is not None:
                                    profile_err = f"stop_profile failed: {err}"
                                else:
                                    # vLLM may flush traces asynchronously.
                                    time.sleep(5.0)
                        elif base_backend == "sglang":
                            if tool == "torch":
                                torch_minimal = bool(
                                    getattr(args, "profile_torch_minimal", True)
                                )
                                torch_num_steps = int(getattr(args, "profile_torch_num_steps", 0))
                                if torch_minimal and torch_num_steps > 0:
                                    # With num_steps, SGLang auto-stops profiling internally.
                                    time.sleep(5.0)
                                else:
                                    err = http_post_ok(f"{root_url}/stop_profile")
                                    if err is not None:
                                        profile_err = f"stop_profile failed: {err}"
                                    else:
                                        # SGLang exports traces asynchronously; give it a moment
                                        # before terminating the server process.
                                        time.sleep(5.0)
                            else:
                                # nsys capture-range=none; nothing to stop explicitly.
                                time.sleep(1.0)
                        elif base_backend == "roseinfer":
                            if tool != "nsys":
                                err = http_post_ok(f"{root_url}/stop_profile")
                                if err is not None:
                                    profile_err = f"stop_profile failed: {err}"
                            else:
                                # nsys capture-range=none; nothing to stop explicitly.
                                pass

                    wall_s = float(time.perf_counter() - stage_t0)
                    run_record = {
                        "tool": tool,
                        "backend": backend,
                        "base_backend": base_backend,
                        "port": port,
                        "cmd": cmd,
                        "log_path": str(log_path),
                        "output_dir": str(out_dir),
                        "nsys_output_prefix": (
                            str(nsys_prefix) if nsys_prefix is not None else None
                        ),
                        "wall_s": wall_s,
                        "profile_error": profile_err,
                    }
                    manifest["runs"].append(run_record)
                    if profile_err is None:
                        print(
                            f"[profile:{tool}] {backend} wall={wall_s:.2f}s -> {out_dir}"
                        )
                    else:
                        print(
                            f"[profile:{tool}] {backend} error={profile_err} -> {out_dir}"
                        )
                finally:
                    shutdown_timeout_s = 60.0 if tool == "nsys" else 15.0
                    if tool == "nsys" and base_backend == "trtllm":
                        shutdown_timeout_s = 300.0
                    if tool == "nsys" and base_backend == "trtllm":
                        # TRT-LLM may detach its Uvicorn server into a new session/process
                        # group; stop it by port so nsys can finalize and write the report.
                        _terminate_trtllm_server(port=port, timeout_s=20.0)
                        try:
                            server.wait(timeout=shutdown_timeout_s)
                        except subprocess.TimeoutExpired:
                            server.kill()
                            server.wait(timeout=shutdown_timeout_s)
                    elif tool == "nsys":
                        # Prefer terminating the *server process* by port and letting nsys
                        # finalize normally. Sending SIGTERM directly to nsys can lead to
                        # missing CUDA kernel activity for multi-process backends.
                        _terminate_server_by_port(port=port, timeout_s=20.0)
                        try:
                            server.wait(timeout=shutdown_timeout_s)
                        except subprocess.TimeoutExpired:
                            server.kill()
                            server.wait(timeout=shutdown_timeout_s)
                    else:
                        _terminate_process(server, timeout_s=shutdown_timeout_s)
                    if (
                        tool == "nsys"
                        and run_record is not None
                        and run_record.get("profile_error") is None
                        and nsys_prefix is not None
                    ):
                        rep_path = Path(f"{nsys_prefix}.nsys-rep")
                        deadline = time.time() + 15.0
                        while time.time() < deadline and not rep_path.exists():
                            time.sleep(0.25)
                        if not rep_path.exists():
                            run_record["profile_error"] = (
                                f"nsys report not produced: {rep_path}"
                            )
                            print(
                                f"[profile:{tool}] {backend} error={run_record['profile_error']} -> {out_dir}"
                            )

        manifest["meta"]["profile_end_time"] = _iso_now()
        manifest["meta"]["profile_wall_s"] = float(time.perf_counter() - profile_t0)
        out_manifest = run_dir / "profile_manifest.json"
        out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote: {out_manifest}")

    if profile_only:
        run_profile_stage()
        return

    used_ports: set[int] = set()
    for i, spec in enumerate(run_specs):
        base_backend = spec.base_backend
        backend = spec.label
        prefill_backend = spec.roseinfer_prefill_backend
        engine_process = spec.roseinfer_engine_process
        fused_ops = spec.roseinfer_fused_ops
        fused_mlp = spec.roseinfer_fused_mlp
        fused_sampler = spec.roseinfer_fused_sampler
        fused_kv_append = spec.roseinfer_fused_kv_append
        overlap_schedule = spec.roseinfer_overlap_schedule
        mp_ipc = spec.roseinfer_mp_ipc
        mp_thread_cap = spec.roseinfer_mp_thread_cap
        mp_affinity = spec.roseinfer_mp_affinity
        mp_max_recv_per_iter = spec.roseinfer_mp_max_recv_per_iter
        mp_fill_target = spec.roseinfer_mp_fill_target
        mp_flat_events = spec.roseinfer_mp_flat_events
        mp_async_admit = spec.roseinfer_mp_async_admit
        mp_tokenize_workers = spec.roseinfer_mp_tokenize_workers
        backend_wall_t0 = time.perf_counter()
        backend_start_time = _iso_now()
        port = _pick_free_port(args.host, int(args.port_base) + i, used_ports)
        base_url = f"http://{args.host}:{port}/v1"
        log_path = run_dir / f"{backend}.server.log"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["OMP_NUM_THREADS"] = str(len(server_cpus))
        env["MKL_NUM_THREADS"] = str(len(server_cpus))
        env["TOKENIZERS_PARALLELISM"] = "false"

        if base_backend == "roseinfer":
            cmd = _roseinfer_server_cmd(
                args,
                host=args.host,
                port=port,
                async_streaming=spec.roseinfer_async_streaming,
                prefill_attn_backend=prefill_backend,
                engine_process=engine_process,
                max_batch_size=spec.roseinfer_max_batch_size,
                gc_freeze=spec.roseinfer_gc_freeze,
                fast_sse=spec.roseinfer_fast_sse,
                kv_cache_max_concurrency=spec.roseinfer_kv_cache_max_concurrency,
                prefix_cache_max_entries=spec.roseinfer_prefix_cache_max_entries,
                fused_ops=fused_ops,
                fused_mlp=fused_mlp,
                fused_sampler=fused_sampler,
                fused_kv_append=fused_kv_append,
                overlap_schedule=overlap_schedule,
                mp_ipc=mp_ipc,
                mp_thread_cap=mp_thread_cap,
                mp_affinity=mp_affinity,
                mp_max_recv_per_iter=mp_max_recv_per_iter,
                mp_fill_target=mp_fill_target,
                mp_flat_events=mp_flat_events,
                mp_async_admit=mp_async_admit,
                mp_tokenize_workers=mp_tokenize_workers,
            )
        elif base_backend == "vllm":
            if vllm_python is not None:
                env.pop("PYTHONPATH", None)
            cmd = _vllm_server_cmd(
                args,
                host=args.host,
                port=port,
                max_context_len=max_ctx,
                python_exe=vllm_python,
            )
        elif base_backend == "sglang":
            if sglang_python is None:
                _maybe_add_sglang_source_pythonpath(env)
            else:
                env.pop("PYTHONPATH", None)
            cmd = _sglang_server_cmd(
                args,
                host=args.host,
                port=port,
                max_context_len=max_ctx,
                python_exe=sglang_python,
            )
        elif base_backend == "trtllm":
            if bool(getattr(args, "trtllm_worker_single_process", False)):
                env["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
            else:
                env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
            if trtllm_python:
                env.pop("PYTHONPATH", None)
                env = _trtllm_runtime_env(python_exe=trtllm_python, base_env=env)
            else:
                _maybe_add_trtllm_source_pythonpath(env)
            cmd = _trtllm_server_cmd(
                args,
                host=args.host,
                port=port,
                max_context_len=max_ctx,
                python_exe=trtllm_python,
            )
        else:
            raise ValueError(f"unknown backend: {base_backend}")

        server = _start_process(
            cmd=cmd, env=env, cpu_set=server_cpus, log_path=log_path
        )
        try:
            ready_t0 = time.perf_counter()
            asyncio.run(
                _wait_ready(
                    f"http://{args.host}:{port}", timeout_s=float(args.timeout_ready_s)
                )
            )
            ready_s = float(time.perf_counter() - ready_t0)
            effective_ctx = max(
                2,
                int(max_ctx) - int(args.prompt_overhead_tokens) - 1,
            )
            warmup_in = int(
                min(
                    int(args.warmup_input_len),
                    max(1, effective_ctx - int(args.warmup_output_len)),
                )
            )
            warmup_ids = _make_base_prompt_ids(
                tokenizer, warmup_in, seed=int(args.seed)
            )
            warmup_prompt = tokenizer.decode(warmup_ids)
            warmup_t0 = time.perf_counter()
            asyncio.run(
                _warmup(
                    base_url=base_url,
                    model=args.model,
                    prompt=warmup_prompt,
                    output_len=int(args.warmup_output_len),
                    num_requests=int(args.warmup_requests),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    ignore_eos=bool(args.ignore_eos),
                )
            )
            warmup_s = float(time.perf_counter() - warmup_t0)
            backend_out: dict[str, Any] = {
                "backend": backend,
                "base_url": base_url,
                "cmd": cmd,
                "log_path": str(log_path),
                "runs": [],
            }

            scale_wall_s: dict[str, float] = {}
            for scale in scales:
                traces = _build_trace_items(
                    rows=rows,
                    tokenizer=tokenizer,
                    max_ctx=max_ctx,
                    scale=scale,
                    start_offset_s=float(args.start_offset_s),
                    max_input_len=(
                        int(args.max_input_len)
                        if args.max_input_len is not None
                        else None
                    ),
                    max_output_len=(
                        int(args.max_output_len)
                        if args.max_output_len is not None
                        else None
                    ),
                    prompt_overhead_tokens=int(args.prompt_overhead_tokens),
                    seed=int(args.seed),
                )
                scale_t0 = time.perf_counter()
                req_metrics, itl_all = asyncio.run(
                    run_trace_benchmark(
                        base_url=base_url,
                        model=args.model,
                        traces=traces,
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        top_k=int(args.top_k),
                        ignore_eos=bool(args.ignore_eos),
                    )
                )
                scale_wall_s[str(scale)] = float(time.perf_counter() - scale_t0)
                summary = _summarize(
                    backend=backend,
                    scale=float(scale),
                    req_metrics=req_metrics,
                    itl_all=itl_all,
                )
                summaries.append(summary)
                backend_out["runs"].append(
                    {
                        "scale": float(scale),
                        "summary": asdict(summary),
                        "request_metrics": [asdict(m) for m in req_metrics],
                    }
                )
                print(
                    f"[{backend} scale={scale}] "
                    f"ok={summary.num_success}/{summary.num_requests}, "
                    f"TTFT p90={summary.ttft_ms.p90:.2f}ms, "
                    f"E2E p90={summary.e2e_ms.p90:.2f}ms, "
                    f"TPOT p90={summary.tpot_ms.p90:.2f}ms, "
                    f"ITL p90={summary.itl_ms.p90:.2f}ms"
                )

            backend_out["timing"] = {
                "backend_start_time": backend_start_time,
                "backend_end_time": _iso_now(),
                "server_ready_s": ready_s,
                "warmup_s": warmup_s,
                "scale_wall_s": scale_wall_s,
                "backend_wall_s": float(time.perf_counter() - backend_wall_t0),
            }
            raw_per_backend[backend] = backend_out
        finally:
            _terminate_process(server)

    run_wall_s = float(time.perf_counter() - run_wall_t0)
    run_end_time = _iso_now()
    out_json = run_dir / "online_results.json"
    payload = {
        "meta": {
            "model": args.model,
            "device": args.device,
            "gpu": args.gpu,
            "dtype": str(args.dtype),
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
            "seed": int(args.seed),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "ignore_eos": bool(args.ignore_eos),
            "n": int(args.n),
            "scales": scales,
            "max_output_len": int(args.max_output_len),
            "max_input_len": (
                int(args.max_input_len) if args.max_input_len is not None else None
            ),
            "prompt_overhead_tokens": int(args.prompt_overhead_tokens),
            "start_offset_s": float(args.start_offset_s),
            "warmup_requests": int(args.warmup_requests),
            "warmup_input_len": int(args.warmup_input_len),
            "warmup_output_len": int(args.warmup_output_len),
            "timeout_ready_s": float(args.timeout_ready_s),
            "backends": [spec.label for spec in run_specs],
            "server_cpus": server_cpus,
            "client_cpus": client_cpus,
            "max_inflight_requests": (
                int(args.max_inflight_requests)
                if args.max_inflight_requests is not None
                else None
            ),
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
            "roseinfer_max_batch_size": int(
                getattr(args, "roseinfer_max_batch_size", 8)
            ),
            "roseinfer_gc_freeze": bool(getattr(args, "roseinfer_gc_freeze", True)),
            "roseinfer_fast_sse": bool(getattr(args, "roseinfer_fast_sse", True)),
            "roseinfer_kv_cache_max_concurrency": int(
                getattr(args, "roseinfer_kv_cache_max_concurrency", 0)
            ),
            "roseinfer_prefix_cache_max_entries": int(
                getattr(args, "roseinfer_prefix_cache_max_entries", 256)
            ),
            "roseinfer_fused_ops": bool(args.roseinfer_fused_ops),
            "roseinfer_compare_fused_ops": bool(
                getattr(args, "roseinfer_compare_fused_ops", False)
            ),
            "roseinfer_fused_mlp": bool(args.roseinfer_fused_mlp),
            "roseinfer_compare_fused_mlp": bool(
                getattr(args, "roseinfer_compare_fused_mlp", False)
            ),
            "roseinfer_fused_sampler": bool(args.roseinfer_fused_sampler),
            "roseinfer_compare_fused_sampler": bool(
                getattr(args, "roseinfer_compare_fused_sampler", False)
            ),
            "roseinfer_fused_kv_append": bool(args.roseinfer_fused_kv_append),
            "roseinfer_compare_fused_kv_append": bool(
                getattr(args, "roseinfer_compare_fused_kv_append", False)
            ),
            "roseinfer_overlap_schedule": bool(
                getattr(args, "roseinfer_overlap_schedule", True)
            ),
            "roseinfer_compare_overlap_schedule": bool(
                getattr(args, "roseinfer_compare_overlap_schedule", False)
            ),
            "roseinfer_engine_process": bool(
                getattr(args, "roseinfer_engine_process", True)
            ),
            "roseinfer_compare_engine_process": bool(
                getattr(args, "roseinfer_compare_engine_process", False)
            ),
            "roseinfer_async_streaming": bool(
                getattr(args, "roseinfer_async_streaming", True)
            ),
            "roseinfer_mp_ipc": str(getattr(args, "roseinfer_mp_ipc", "pipe")),
            "roseinfer_mp_max_recv_per_iter": int(
                getattr(args, "roseinfer_mp_max_recv_per_iter", 64)
            ),
            "roseinfer_mp_fill_target": bool(
                getattr(args, "roseinfer_mp_fill_target", True)
            ),
            "roseinfer_mp_thread_cap": bool(
                getattr(args, "roseinfer_mp_thread_cap", True)
            ),
            "roseinfer_mp_affinity": bool(getattr(args, "roseinfer_mp_affinity", True)),
            "roseinfer_mp_flat_events": bool(
                getattr(args, "roseinfer_mp_flat_events", True)
            ),
            "roseinfer_mp_async_admit": bool(
                getattr(args, "roseinfer_mp_async_admit", False)
            ),
            "roseinfer_mp_tokenize_workers": int(
                getattr(args, "roseinfer_mp_tokenize_workers", 0)
            ),
            "roseinfer_compare_mp_ablations": bool(
                getattr(args, "roseinfer_compare_mp_ablations", False)
            ),
            "trace_path": str(trace_path),
            "trtllm_worker_single_process": bool(
                getattr(args, "trtllm_worker_single_process", False)
            ),
            "run_start_time": run_start_time,
            "run_end_time": run_end_time,
            "wall_s": run_wall_s,
            "versions": versions,
        },
        "summaries": [asdict(s) for s in summaries],
        "by_backend": raw_per_backend,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Total wall time: {run_wall_s:.2f}s")

    run_profile_stage()


if __name__ == "__main__":
    main()
