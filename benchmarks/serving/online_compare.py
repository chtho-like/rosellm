from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import importlib.util
import json
import os
import random
import signal
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
    if not ttft_ms:
        raise ValueError("no TTFT samples collected; did the server stream tokens?")
    if not tpot_ms:
        raise ValueError("no TPOT samples collected; did the server stream >1 token?")
    if not itl_ms:
        raise ValueError("no ITL samples collected; did the server stream >1 token?")

    total_events = sum(m.num_stream_events for m in ok)
    duration_s = (max(m.end_s for m in ok) - min(m.start_s for m in ok)) or 1e-9
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
        ttft_ms=_percentiles(ttft_ms),
        e2e_ms=_percentiles(e2e_ms),
        tpot_ms=_percentiles(tpot_ms),
        itl_ms=_percentiles(itl_ms),
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


def _roseinfer_server_cmd(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    prefill_attn_backend: str | None = None,
    decode_attn_backend: str | None = None,
    fused_ops: bool | None = None,
    fused_mlp: bool | None = None,
    fused_sampler: bool | None = None,
    fused_kv_append: bool | None = None,
    engine_process: bool | None = None,
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
    engine_process = (
        bool(engine_process)
        if engine_process is not None
        else bool(getattr(args, "roseinfer_engine_process", True))
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
        "--prefill-attn-backend",
        prefill_attn_backend,
        "--decode-attn-backend",
        decode_attn_backend,
    ]
    if args.max_inflight_requests is not None:
        cmd += ["--max-inflight-requests", str(args.max_inflight_requests)]
    if args.dtype == "fp32":
        cmd += ["--no-amp"]
    elif args.dtype == "bf16":
        cmd += ["--bf16"]
    if args.no_amp:
        cmd += ["--no-amp"]
    if args.bf16:
        cmd += ["--bf16"]
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
    cmd += ["--engine-process" if engine_process else "--no-engine-process"]
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


def _vllm_server_cmd(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    max_context_len: int,
) -> list[str]:
    cmd = [
        sys.executable,
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
    cmd += _dtype_flag_vllm(str(args.dtype))
    return cmd


def _sglang_server_cmd(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    max_context_len: int,
) -> list[str]:
    cmd = [
        sys.executable,
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
        "--backends",
        type=str,
        default="roseinfer,vllm,sglang",
        help="Comma-separated backends to run.",
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
    versions = _collect_versions()
    print(f"[meta] start={run_start_time}, versions={versions}")

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
        roseinfer_engine_process: bool | None = None
        roseinfer_fused_ops: bool | None = None
        roseinfer_fused_mlp: bool | None = None
        roseinfer_fused_sampler: bool | None = None
        roseinfer_fused_kv_append: bool | None = None

    run_specs: list[RunSpec] = []
    for backend in backends:
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
        base_engine_process = bool(getattr(args, "roseinfer_engine_process", True))
        engine_cfgs: list[bool] = [base_engine_process]
        if bool(getattr(args, "roseinfer_compare_engine_process", False)):
            engine_cfgs.append(not base_engine_process)
        engine_cfgs = list(dict.fromkeys(engine_cfgs))
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
            for engine_process in engine_cfgs:
                for fused_ops, fused_mlp, fused_sampler, fused_kv_append in cfgs:
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
                    run_specs.append(
                        RunSpec(
                            base_backend="roseinfer",
                            label=label,
                            roseinfer_prefill_backend=prefill_backend,
                            roseinfer_engine_process=bool(engine_process),
                            roseinfer_fused_ops=bool(fused_ops),
                            roseinfer_fused_mlp=bool(fused_mlp),
                            roseinfer_fused_sampler=bool(fused_sampler),
                            roseinfer_fused_kv_append=bool(fused_kv_append),
                        )
                    )

    if not run_specs:
        raise ValueError("no backends to run (all candidates were skipped)")
    scales = _parse_csv_floats(args.scales)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    max_ctx = _resolve_model_max_context(args.model, tokenizer)
    rows = _read_trace_a_jsonl(trace_path, n=int(args.n))

    summaries: list[OnlineBenchmarkSummary] = []
    raw_per_backend: dict[str, dict[str, Any]] = {}

    for i, spec in enumerate(run_specs):
        base_backend = spec.base_backend
        backend = spec.label
        prefill_backend = spec.roseinfer_prefill_backend
        engine_process = spec.roseinfer_engine_process
        fused_ops = spec.roseinfer_fused_ops
        fused_mlp = spec.roseinfer_fused_mlp
        fused_sampler = spec.roseinfer_fused_sampler
        fused_kv_append = spec.roseinfer_fused_kv_append
        backend_wall_t0 = time.perf_counter()
        backend_start_time = _iso_now()
        port = int(args.port_base) + i
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
                prefill_attn_backend=prefill_backend,
                engine_process=engine_process,
                fused_ops=fused_ops,
                fused_mlp=fused_mlp,
                fused_sampler=fused_sampler,
                fused_kv_append=fused_kv_append,
            )
        elif base_backend == "vllm":
            cmd = _vllm_server_cmd(
                args, host=args.host, port=port, max_context_len=max_ctx
            )
        elif base_backend == "sglang":
            _maybe_add_sglang_source_pythonpath(env)
            cmd = _sglang_server_cmd(
                args, host=args.host, port=port, max_context_len=max_ctx
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
            "trace_path": str(trace_path),
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


if __name__ == "__main__":
    main()
