from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import queue
import socket
import struct
import threading
import time
import traceback
from array import array
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, AsyncIterator, Deque, Dict, Iterator, Optional, Sequence

import torch

from rosellm.roseinfer.detokenizer import BaseDetokenizer, build_detokenizer
from rosellm.roseinfer.engine import (
    ChunkedOnlineScheduler,
    InferenceEngine,
    OnlineRequest,
    OnlineScheduler,
)
from rosellm.roseinfer.errors import SchedulerManagerOverloadedError
from rosellm.roseinfer.hybrid_queue import HybridQueue


@dataclass(frozen=True)
class _MPAdmitTask:
    request_id: int
    prompt: str
    prompt_token_ids: list[int] | None
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    stop_on_eos: bool
    do_sample: bool


@dataclass(frozen=True)
class ToyEngineSpec:
    config: Any
    seed: int = 0


@dataclass(frozen=True)
class EngineProcessArgs:
    checkpoint_path: str | None
    hf_model_id: str | None
    tokenizer_name: str
    device: str
    no_amp: bool
    bf16: bool
    prefill_attn_backend: str
    decode_attn_backend: str
    paged_attn: bool
    cuda_graph: bool
    fused_ops: bool
    fused_mlp: bool
    fused_sampler: bool
    fused_kv_append: bool
    chunked_prefill: bool
    prefill_chunk_size: int
    prefix_cache: bool
    overlap_schedule: bool
    max_batch_size: int
    kv_cache_max_concurrency: int = 256
    kv_cache_max_tokens: int | None = None
    kv_cache_mem_fraction: float | None = None
    prefix_cache_max_entries: int = 256
    toy: ToyEngineSpec | None = None
    mp_torch_num_threads: int | None = None
    mp_torch_num_interop_threads: int | None = None
    mp_cpu_affinity: tuple[int, ...] | None = None
    gc_freeze: bool = True
    gc_warn_ms: float = 0.0
    gc_log_all: bool = False
    mp_fill_target: bool = True
    mp_max_recv_per_iter: int = 64
    mp_emit_token_events: bool = True
    mp_flat_events: bool = True
    mp_fast_finish_counts: bool = True


@dataclass(frozen=True)
class AddRequestCmd:
    request_id: int
    prompt_token_ids: list[int]
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    stop_on_eos: bool
    do_sample: bool


@dataclass(frozen=True)
class AddBatchCmd:
    """Batch of AddRequestCmd with shared sampling parameters.

    This is primarily used by offline benchmarks to amortize IPC overhead.
    """

    request_ids: array
    prompt_lens: array
    prompt_token_ids: array
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    stop_on_eos: bool
    do_sample: bool


@dataclass(frozen=True)
class CancelRequestCmd:
    request_id: int


@dataclass(frozen=True)
class StartProfileCmd:
    profile_id: int
    tool: str
    output_dir: str | None = None


@dataclass(frozen=True)
class StopProfileCmd:
    profile_id: int
    tool: str


@dataclass(frozen=True)
class ShutdownCmd:
    pass


EngineCommand = (
    AddRequestCmd
    | AddBatchCmd
    | CancelRequestCmd
    | StartProfileCmd
    | StopProfileCmd
    | ShutdownCmd
)


@dataclass(frozen=True)
class ReadyEvent:
    eos_token_id: int | None
    max_context: int


@dataclass(frozen=True)
class TokensEvent:
    tokens: dict[int, list[int]]
    finished_ids: list[int]


@dataclass(frozen=True)
class TokenPairsEvent:
    """Flat token stream (rid, token_id) pairs.

    This avoids per-request nested dict/list allocations and is friendlier to
    compact binary encoding for the pipe transport.
    """

    rids: array
    token_ids: array
    finished_ids: array


@dataclass(frozen=True)
class TokenCountsEvent:
    """Final token counts per request (used by MPTokenManager offline benchmarks)."""

    request_ids: array
    token_counts: array


@dataclass(frozen=True)
class ErrorEvent:
    request_id: int | None
    message: str


@dataclass(frozen=True)
class ProfileEvent:
    profile_id: int
    action: str
    tool: str
    ok: bool
    message: str | None = None


EngineEvent = (
    ReadyEvent
    | TokensEvent
    | TokenPairsEvent
    | TokenCountsEvent
    | ErrorEvent
    | ProfileEvent
)


_CMD_ADD = 1
_CMD_CANCEL = 2
_CMD_START_PROFILE = 3
_CMD_STOP_PROFILE = 4
_CMD_SHUTDOWN = 5
_CMD_ADD_BATCH = 6

_EVT_READY = 101
_EVT_TOKENS = 102
_EVT_ERROR = 103
_EVT_PROFILE = 104
_EVT_TOKEN_PAIRS = 105
_EVT_TOKEN_COUNTS = 106

_TOOL_TORCH = 1
_TOOL_CUDA = 2

_PROFILE_START = 1
_PROFILE_STOP = 2


@contextmanager
def _maybe_nvtx_range(name: str, enabled: bool) -> Iterator[None]:
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def _maybe_set_pipe_socket_buffers(conns: Sequence[Connection], buf_bytes: int) -> None:
    if int(buf_bytes) <= 0:
        return
    for conn in conns:
        try:
            sock = socket.fromfd(conn.fileno(), socket.AF_UNIX, socket.SOCK_STREAM)
        except OSError:
            continue
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(buf_bytes))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(buf_bytes))
        except OSError:
            pass
        finally:
            try:
                sock.close()
            except Exception:
                pass


def _pack_u32(values: Sequence[int]) -> bytes:
    if isinstance(values, array) and values.typecode == "I":
        return values.tobytes()
    return array("I", values).tobytes()


def _unpack_u32(data: bytes | memoryview) -> array:
    arr = array("I")
    arr.frombytes(data)
    return arr


def _encode_cmd(cmd: EngineCommand) -> bytes:
    if isinstance(cmd, AddRequestCmd):
        toks = _pack_u32(cmd.prompt_token_ids)
        header = struct.pack(
            "<BII f I f BB I",
            _CMD_ADD,
            int(cmd.request_id),
            int(cmd.max_new_tokens),
            float(cmd.temperature),
            int(cmd.top_k),
            float(cmd.top_p),
            1 if bool(cmd.stop_on_eos) else 0,
            1 if bool(cmd.do_sample) else 0,
            int(len(cmd.prompt_token_ids)),
        )
        return header + toks
    if isinstance(cmd, AddBatchCmd):
        if cmd.request_ids.typecode != "I":
            raise ValueError("AddBatchCmd request_ids must be array('I')")
        if cmd.prompt_lens.typecode != "I":
            raise ValueError("AddBatchCmd prompt_lens must be array('I')")
        if cmd.prompt_token_ids.typecode != "I":
            raise ValueError("AddBatchCmd prompt_token_ids must be array('I')")
        n_req = int(len(cmd.request_ids))
        if int(len(cmd.prompt_lens)) != n_req:
            raise ValueError("AddBatchCmd length mismatch")
        total_tokens = int(len(cmd.prompt_token_ids))
        if int(sum(int(x) for x in cmd.prompt_lens)) != total_tokens:
            raise ValueError("AddBatchCmd token length mismatch")
        header = struct.pack(
            "<BII f I f BB I",
            _CMD_ADD_BATCH,
            n_req,
            int(cmd.max_new_tokens),
            float(cmd.temperature),
            int(cmd.top_k),
            float(cmd.top_p),
            1 if bool(cmd.stop_on_eos) else 0,
            1 if bool(cmd.do_sample) else 0,
            total_tokens,
        )
        return b"".join(
            [
                header,
                cmd.request_ids.tobytes(),
                cmd.prompt_lens.tobytes(),
                cmd.prompt_token_ids.tobytes(),
            ]
        )
    if isinstance(cmd, CancelRequestCmd):
        return struct.pack("<BI", _CMD_CANCEL, int(cmd.request_id))
    if isinstance(cmd, StartProfileCmd):
        tool = str(cmd.tool).lower()
        tool_code = _TOOL_TORCH if tool == "torch" else _TOOL_CUDA
        out = (str(cmd.output_dir) if cmd.output_dir is not None else "").encode(
            "utf-8"
        )
        header = struct.pack(
            "<BIBI",
            _CMD_START_PROFILE,
            int(cmd.profile_id),
            int(tool_code),
            int(len(out)),
        )
        return header + out
    if isinstance(cmd, StopProfileCmd):
        tool = str(cmd.tool).lower()
        tool_code = _TOOL_TORCH if tool == "torch" else _TOOL_CUDA
        return struct.pack(
            "<BIB", _CMD_STOP_PROFILE, int(cmd.profile_id), int(tool_code)
        )
    if isinstance(cmd, ShutdownCmd):
        return struct.pack("<B", _CMD_SHUTDOWN)
    raise TypeError(f"unsupported cmd type: {type(cmd)!r}")


def _decode_cmd(data: bytes) -> EngineCommand:
    if not data:
        raise ValueError("empty command message")
    mv = memoryview(data)
    cmd_type = int(mv[0])
    off = 1
    if cmd_type == _CMD_ADD:
        (
            request_id,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            stop_on_eos,
            do_sample,
            prompt_len,
        ) = struct.unpack_from("<II f I f BB I", mv, off)
        off += struct.calcsize("<II f I f BB I")
        token_arr = _unpack_u32(mv[off : off + int(prompt_len) * 4])
        prompt_token_ids = token_arr.tolist()
        return AddRequestCmd(
            request_id=int(request_id),
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            stop_on_eos=bool(stop_on_eos),
            do_sample=bool(do_sample),
        )
    if cmd_type == _CMD_ADD_BATCH:
        (
            n_req,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            stop_on_eos,
            do_sample,
            total_tokens,
        ) = struct.unpack_from("<II f I f BB I", mv, off)
        off += struct.calcsize("<II f I f BB I")
        n_req_i = int(n_req)
        total_tokens_i = int(total_tokens)
        request_ids = _unpack_u32(mv[off : off + n_req_i * 4])
        off += n_req_i * 4
        prompt_lens = _unpack_u32(mv[off : off + n_req_i * 4])
        off += n_req_i * 4
        prompt_token_ids = _unpack_u32(mv[off : off + total_tokens_i * 4])
        if int(sum(prompt_lens)) != total_tokens_i:
            raise ValueError("AddBatchCmd token length mismatch")
        return AddBatchCmd(
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            stop_on_eos=bool(stop_on_eos),
            do_sample=bool(do_sample),
        )
    if cmd_type == _CMD_CANCEL:
        (rid,) = struct.unpack_from("<I", mv, off)
        return CancelRequestCmd(request_id=int(rid))
    if cmd_type == _CMD_START_PROFILE:
        profile_id, tool_code, out_len = struct.unpack_from("<IBI", mv, off)
        off += struct.calcsize("<IBI")
        out_dir = mv[off : off + int(out_len)].tobytes().decode("utf-8")
        tool = "torch" if int(tool_code) == _TOOL_TORCH else "cuda"
        return StartProfileCmd(
            profile_id=int(profile_id),
            tool=tool,
            output_dir=out_dir or None,
        )
    if cmd_type == _CMD_STOP_PROFILE:
        profile_id, tool_code = struct.unpack_from("<IB", mv, off)
        tool = "torch" if int(tool_code) == _TOOL_TORCH else "cuda"
        return StopProfileCmd(profile_id=int(profile_id), tool=tool)
    if cmd_type == _CMD_SHUTDOWN:
        return ShutdownCmd()
    raise ValueError(f"unknown cmd type code: {cmd_type}")


def _enqueue_add_batch(
    cmd: AddBatchCmd,
    pending: Deque[AddRequestCmd],
    pending_by_rid: dict[int, AddRequestCmd],
) -> None:
    offset = 0
    for rid, n_tok in zip(cmd.request_ids, cmd.prompt_lens):
        n_i = int(n_tok)
        toks = cmd.prompt_token_ids[offset : offset + n_i].tolist()
        offset += n_i
        req = AddRequestCmd(
            request_id=int(rid),
            prompt_token_ids=toks,
            max_new_tokens=int(cmd.max_new_tokens),
            temperature=float(cmd.temperature),
            top_k=int(cmd.top_k),
            top_p=float(cmd.top_p),
            stop_on_eos=bool(cmd.stop_on_eos),
            do_sample=bool(cmd.do_sample),
        )
        pending.append(req)
        pending_by_rid[int(req.request_id)] = req


def _encode_evt(evt: EngineEvent) -> bytes:
    if isinstance(evt, ReadyEvent):
        eos = int(evt.eos_token_id) if evt.eos_token_id is not None else -1
        return struct.pack("<BiI", _EVT_READY, eos, int(evt.max_context))
    if isinstance(evt, TokensEvent):
        parts: list[bytes] = [struct.pack("<BI", _EVT_TOKENS, int(len(evt.tokens)))]
        for rid, toks in evt.tokens.items():
            tok_bytes = _pack_u32(toks)
            parts.append(struct.pack("<II", int(rid), int(len(toks))))
            parts.append(tok_bytes)
        parts.append(struct.pack("<I", int(len(evt.finished_ids))))
        if evt.finished_ids:
            parts.append(_pack_u32(evt.finished_ids))
        return b"".join(parts)
    if isinstance(evt, TokenPairsEvent):
        n_pairs = int(len(evt.rids))
        if int(len(evt.token_ids)) != n_pairs:
            raise ValueError("TokenPairsEvent length mismatch")
        header = struct.pack("<BI", _EVT_TOKEN_PAIRS, n_pairs)
        parts = [
            header,
            evt.rids.tobytes(),
            evt.token_ids.tobytes(),
            struct.pack("<I", int(len(evt.finished_ids))),
        ]
        if evt.finished_ids:
            parts.append(evt.finished_ids.tobytes())
        return b"".join(parts)
    if isinstance(evt, TokenCountsEvent):
        n = int(len(evt.request_ids))
        if int(len(evt.token_counts)) != n:
            raise ValueError("TokenCountsEvent length mismatch")
        header = struct.pack("<BI", _EVT_TOKEN_COUNTS, n)
        return header + evt.request_ids.tobytes() + evt.token_counts.tobytes()
    if isinstance(evt, ErrorEvent):
        rid = int(evt.request_id) if evt.request_id is not None else -1
        msg = str(evt.message).encode("utf-8")
        return struct.pack("<BiI", _EVT_ERROR, rid, int(len(msg))) + msg
    if isinstance(evt, ProfileEvent):
        tool = str(evt.tool).lower()
        tool_code = _TOOL_TORCH if tool == "torch" else _TOOL_CUDA
        action = str(evt.action).lower()
        action_code = _PROFILE_START if action == "start" else _PROFILE_STOP
        msg = (str(evt.message) if evt.message else "").encode("utf-8")
        header = struct.pack(
            "<BIBBBI",
            _EVT_PROFILE,
            int(evt.profile_id),
            int(action_code),
            int(tool_code),
            1 if bool(evt.ok) else 0,
            int(len(msg)),
        )
        return header + msg
    raise TypeError(f"unsupported event type: {type(evt)!r}")


def _decode_evt(data: bytes) -> EngineEvent:
    if not data:
        raise ValueError("empty event message")
    mv = memoryview(data)
    evt_type = int(mv[0])
    off = 1
    if evt_type == _EVT_READY:
        eos, max_ctx = struct.unpack_from("<iI", mv, off)
        return ReadyEvent(
            eos_token_id=None if int(eos) < 0 else int(eos),
            max_context=int(max_ctx),
        )
    if evt_type == _EVT_TOKENS:
        (n_records,) = struct.unpack_from("<I", mv, off)
        off += 4
        tokens: dict[int, list[int]] = {}
        for _ in range(int(n_records)):
            rid, n_toks = struct.unpack_from("<II", mv, off)
            off += 8
            n_toks_i = int(n_toks)
            tok_bytes = mv[off : off + n_toks_i * 4].tobytes()
            off += n_toks_i * 4
            arr = _unpack_u32(tok_bytes)
            tokens[int(rid)] = [int(x) for x in arr]
        (n_finished,) = struct.unpack_from("<I", mv, off)
        off += 4
        finished_ids: list[int] = []
        if int(n_finished) > 0:
            raw = mv[off : off + int(n_finished) * 4].tobytes()
            arr = _unpack_u32(raw)
            finished_ids = [int(x) for x in arr]
        return TokensEvent(tokens=tokens, finished_ids=finished_ids)
    if evt_type == _EVT_TOKEN_PAIRS:
        (n_pairs,) = struct.unpack_from("<I", mv, off)
        off += 4
        n_pairs_i = int(n_pairs)
        rids = _unpack_u32(mv[off : off + n_pairs_i * 4])
        off += n_pairs_i * 4
        token_ids = _unpack_u32(mv[off : off + n_pairs_i * 4])
        off += n_pairs_i * 4
        (n_finished,) = struct.unpack_from("<I", mv, off)
        off += 4
        finished_ids = array("I")
        if int(n_finished) > 0:
            finished_ids = _unpack_u32(mv[off : off + int(n_finished) * 4])
        return TokenPairsEvent(
            rids=rids, token_ids=token_ids, finished_ids=finished_ids
        )
    if evt_type == _EVT_TOKEN_COUNTS:
        (n,) = struct.unpack_from("<I", mv, off)
        off += 4
        n_i = int(n)
        request_ids = _unpack_u32(mv[off : off + n_i * 4])
        off += n_i * 4
        token_counts = _unpack_u32(mv[off : off + n_i * 4])
        return TokenCountsEvent(request_ids=request_ids, token_counts=token_counts)
    if evt_type == _EVT_ERROR:
        rid, msg_len = struct.unpack_from("<iI", mv, off)
        off += struct.calcsize("<iI")
        msg = mv[off : off + int(msg_len)].tobytes().decode("utf-8")
        return ErrorEvent(request_id=None if int(rid) < 0 else int(rid), message=msg)
    if evt_type == _EVT_PROFILE:
        profile_id, action_code, tool_code, ok, msg_len = struct.unpack_from(
            "<IBBBI", mv, off
        )
        off += struct.calcsize("<IBBBI")
        msg = mv[off : off + int(msg_len)].tobytes().decode("utf-8")
        action = "start" if int(action_code) == _PROFILE_START else "stop"
        tool = "torch" if int(tool_code) == _TOOL_TORCH else "cuda"
        return ProfileEvent(
            profile_id=int(profile_id),
            action=action,
            tool=tool,
            ok=bool(ok),
            message=msg or None,
        )
    raise ValueError(f"unknown event type code: {evt_type}")


def _maybe_set_torch_threads(
    *,
    num_threads: int | None,
    num_interop_threads: int | None,
) -> None:
    if num_threads is not None and int(num_threads) > 0:
        torch.set_num_threads(int(num_threads))
    if num_interop_threads is not None and int(num_interop_threads) > 0:
        try:
            torch.set_num_interop_threads(int(num_interop_threads))
        except Exception:
            pass


def _maybe_set_cpu_affinity(cpus: Sequence[int] | None) -> None:
    if not cpus:
        return
    try:
        os.sched_setaffinity(0, set(int(c) for c in cpus))
    except Exception:
        pass


def _freeze_gc_heap() -> None:
    import gc

    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.freeze()


def _build_engine(args: EngineProcessArgs) -> InferenceEngine:
    device = torch.device(args.device)
    if args.toy is not None:
        from rosellm.roseinfer.simple_tokenizer import ByteTokenizer
        from rosellm.rosetrainer.model import GPTModel

        torch.manual_seed(int(args.toy.seed))
        model = GPTModel(args.toy.config)
        tokenizer = ByteTokenizer(vocab_size=int(args.toy.config.vocab_size))
        return InferenceEngine(
            checkpoint_path=None,
            tokenizer_name=args.tokenizer_name,
            device=args.device,
            use_amp=False,
            use_paged_attention=False,
            use_cuda_graph=False,
            prefill_attn_backend="naive",
            decode_attn_backend="naive",
            use_fused_ops=False,
            use_fused_mlp=False,
            use_fused_sampler=False,
            use_fused_kv_append=False,
            kv_cache_max_concurrency=max(4, int(args.max_batch_size)),
            prefix_cache_max_entries=0,
            model=model,
            config=args.toy.config,
            tokenizer=tokenizer,
        )
    if args.hf_model_id is not None:
        from transformers import AutoConfig

        use_amp = (not bool(args.no_amp)) and device.type == "cuda"
        if use_amp:
            dtype = torch.bfloat16 if bool(args.bf16) else torch.float16
        else:
            dtype = torch.float32
        model_type = str(
            getattr(
                AutoConfig.from_pretrained(
                    args.hf_model_id,
                    trust_remote_code=False,
                ),
                "model_type",
                "",
            )
            or ""
        ).lower()
        if model_type == "gpt2":
            from rosellm.rosetrainer.hf_gpt2 import load_gpt2_from_hf_pretrained

            model, config, tokenizer = load_gpt2_from_hf_pretrained(
                args.hf_model_id,
                device=device,
                dtype=dtype,
            )
        elif model_type == "qwen3":
            from rosellm.rosetrainer.hf_qwen3 import load_qwen3_from_hf_pretrained

            model, config, tokenizer = load_qwen3_from_hf_pretrained(
                args.hf_model_id,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(
                "unsupported hf_model_id model_type="
                f"{model_type!r} (supported: gpt2, qwen3)"
            )
        return InferenceEngine(
            checkpoint_path=None,
            tokenizer_name=args.tokenizer_name,
            device=args.device,
            use_amp=use_amp,
            bf16=bool(args.bf16),
            kv_cache_max_concurrency=int(args.kv_cache_max_concurrency),
            kv_cache_max_tokens=args.kv_cache_max_tokens,
            kv_cache_mem_fraction=args.kv_cache_mem_fraction,
            prefix_cache_max_entries=int(args.prefix_cache_max_entries),
            use_paged_attention=bool(args.paged_attn),
            use_cuda_graph=bool(args.cuda_graph),
            prefill_attn_backend=str(args.prefill_attn_backend),
            decode_attn_backend=str(args.decode_attn_backend),
            use_fused_ops=bool(args.fused_ops),
            use_fused_mlp=bool(args.fused_mlp),
            use_fused_sampler=bool(args.fused_sampler),
            use_fused_kv_append=bool(args.fused_kv_append),
            model=model,
            config=config,
            tokenizer=tokenizer,
        )
    if args.checkpoint_path is None:
        raise ValueError("checkpoint_path must be set when hf_model_id is None")
    return InferenceEngine(
        checkpoint_path=args.checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        use_amp=not bool(args.no_amp),
        bf16=bool(args.bf16),
        kv_cache_max_concurrency=int(args.kv_cache_max_concurrency),
        kv_cache_max_tokens=args.kv_cache_max_tokens,
        kv_cache_mem_fraction=args.kv_cache_mem_fraction,
        prefix_cache_max_entries=int(args.prefix_cache_max_entries),
        use_paged_attention=bool(args.paged_attn),
        use_cuda_graph=bool(args.cuda_graph),
        prefill_attn_backend=str(args.prefill_attn_backend),
        decode_attn_backend=str(args.decode_attn_backend),
        use_fused_ops=bool(args.fused_ops),
        use_fused_mlp=bool(args.fused_mlp),
        use_fused_sampler=bool(args.fused_sampler),
        use_fused_kv_append=bool(args.fused_kv_append),
    )


def _engine_process_main(
    args: EngineProcessArgs,
    cmd_q: "mp.Queue[EngineCommand]",
    evt_q: "mp.Queue[EngineEvent]",
) -> None:
    torch.set_grad_enabled(False)
    try:
        gc_warn_ms = float(getattr(args, "gc_warn_ms", 0.0) or 0.0)
        gc_log_all = bool(getattr(args, "gc_log_all", False))
        if gc_warn_ms > 0.0 or gc_log_all:
            from rosellm.roseinfer.gc_observer import install_gc_observer

            install_gc_observer(
                warn_ms=gc_warn_ms,
                log_all=gc_log_all,
                prefix=f"engine pid={os.getpid()}",
                nvtx=os.environ.get("ROSEINFER_NVTX") == "1",
            )
        _maybe_set_cpu_affinity(args.mp_cpu_affinity)
        _maybe_set_torch_threads(
            num_threads=args.mp_torch_num_threads,
            num_interop_threads=args.mp_torch_num_interop_threads,
        )
        engine = _build_engine(args)
        if bool(args.chunked_prefill):
            scheduler: OnlineScheduler | ChunkedOnlineScheduler = (
                ChunkedOnlineScheduler(
                    engine,
                    max_batch_size=int(args.max_batch_size),
                    prefill_chunk_size=int(args.prefill_chunk_size),
                    prefill_max_batch_size=int(args.max_batch_size),
                    use_prefix_cache=bool(args.prefix_cache),
                    overlap_schedule=bool(args.overlap_schedule),
                )
            )
        else:
            scheduler = OnlineScheduler(
                engine,
                max_batch_size=int(args.max_batch_size),
                use_prefix_cache=bool(args.prefix_cache),
                overlap_schedule=bool(args.overlap_schedule),
            )
        engine.warmup_paged_attention_decode()
        if bool(getattr(args, "gc_freeze", True)):
            _freeze_gc_heap()
        evt_q.put(
            ReadyEvent(
                eos_token_id=engine.eos_token_id,
                max_context=int(engine.config.max_position_embeddings),
            )
        )

        pending: "Deque[AddRequestCmd]" = deque()
        pending_by_rid: dict[int, AddRequestCmd] = {}
        active: set[int] = set()
        token_counts: dict[int, int] | None = None
        emit_tokens = bool(getattr(args, "mp_emit_token_events", True))
        flat_events = bool(getattr(args, "mp_flat_events", True))
        fast_finish_counts = bool(getattr(args, "mp_fast_finish_counts", True))
        if (not emit_tokens) and (not fast_finish_counts):
            token_counts = {}
        torch_prof: Any | None = None
        torch_prof_dir: str | None = None
        cuda_prof_active = False
        nvtx = bool(
            torch.cuda.is_available() and os.environ.get("ROSEINFER_NVTX") == "1"
        )

        def start_profile(cmd: StartProfileCmd) -> None:
            nonlocal torch_prof, torch_prof_dir, cuda_prof_active
            tool = str(cmd.tool).lower()
            if tool == "torch":
                if torch_prof is not None:
                    raise RuntimeError("torch profiler already running")
                if cmd.output_dir is None:
                    raise ValueError("output_dir is required for torch profiler")
                out_dir = str(cmd.output_dir)
                os.makedirs(out_dir, exist_ok=True)
                activities = [torch.profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with_stack = os.environ.get("ROSEINFER_TORCH_PROFILE_WITH_STACK") == "1"
                record_shapes = (
                    os.environ.get("ROSEINFER_TORCH_PROFILE_RECORD_SHAPES", "1") == "1"
                )
                profile_memory = (
                    os.environ.get("ROSEINFER_TORCH_PROFILE_WITH_PROFILE_MEMORY", "1")
                    == "1"
                )
                delay_steps = max(
                    0, int(os.environ.get("ROSEINFER_TORCH_PROFILE_DELAY_STEPS", "0"))
                )
                num_steps = max(
                    0, int(os.environ.get("ROSEINFER_TORCH_PROFILE_NUM_STEPS", "0"))
                )
                schedule = (
                    torch.profiler.schedule(
                        wait=delay_steps, warmup=0, active=num_steps, repeat=1
                    )
                    if num_steps > 0
                    else None
                )
                prof = torch.profiler.profile(
                    activities=activities,
                    record_shapes=record_shapes,
                    profile_memory=profile_memory,
                    with_stack=with_stack,
                    schedule=schedule,
                )
                prof.__enter__()
                torch_prof = prof
                torch_prof_dir = out_dir
                return
            if tool == "cuda":
                if cuda_prof_active:
                    raise RuntimeError("cuda profiler already running")
                if torch.cuda.is_available():
                    torch.cuda.profiler.start()
                cuda_prof_active = True
                return
            raise ValueError("tool must be torch|cuda")

        def stop_profile(cmd: StopProfileCmd) -> None:
            nonlocal torch_prof, torch_prof_dir, cuda_prof_active
            tool = str(cmd.tool).lower()
            if tool == "torch":
                if torch_prof is None or torch_prof_dir is None:
                    raise RuntimeError("torch profiler not running")
                prof = torch_prof
                out_dir = torch_prof_dir
                prof.__exit__(None, None, None)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof.export_chrome_trace(os.path.join(out_dir, "trace.json"))
                torch_prof = None
                torch_prof_dir = None
                return
            if tool == "cuda":
                if not cuda_prof_active:
                    raise RuntimeError("cuda profiler not running")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
                cuda_prof_active = False
                return
            raise ValueError("tool must be torch|cuda")

        def cancel_request(request_id: int) -> None:
            pending_by_rid.pop(request_id, None)
            active.discard(int(request_id))
            if token_counts is not None:
                token_counts.pop(int(request_id), None)
            scheduler.discard_request(request_id)

        max_recv = int(getattr(args, "mp_max_recv_per_iter", 64))
        if max_recv < 0:
            max_recv = 0
        bootstrap_target = min(
            int(args.max_batch_size),
            int(getattr(args, "kv_cache_max_concurrency", int(args.max_batch_size))),
        )
        fill_target = bool(getattr(args, "mp_fill_target", True))

        while True:
            if not pending_by_rid and not scheduler.has_unfinished():
                cmd = cmd_q.get()
                if isinstance(cmd, ShutdownCmd):
                    return
                if isinstance(cmd, StartProfileCmd):
                    ok = True
                    msg: str | None = None
                    try:
                        start_profile(cmd)
                    except Exception as exc:
                        ok = False
                        msg = str(exc)
                    evt_q.put(
                        ProfileEvent(
                            profile_id=int(cmd.profile_id),
                            action="start",
                            tool=str(cmd.tool),
                            ok=ok,
                            message=msg,
                        )
                    )
                    continue
                if isinstance(cmd, StopProfileCmd):
                    ok = True
                    msg: str | None = None
                    try:
                        stop_profile(cmd)
                    except Exception as exc:
                        ok = False
                        msg = str(exc)
                    evt_q.put(
                        ProfileEvent(
                            profile_id=int(cmd.profile_id),
                            action="stop",
                            tool=str(cmd.tool),
                            ok=ok,
                            message=msg,
                        )
                    )
                    continue
                if isinstance(cmd, CancelRequestCmd):
                    cancel_request(int(cmd.request_id))
                    continue
                if isinstance(cmd, AddBatchCmd):
                    _enqueue_add_batch(cmd, pending, pending_by_rid)
                    continue
                if isinstance(cmd, AddRequestCmd):
                    pending.append(cmd)
                    pending_by_rid[int(cmd.request_id)] = cmd
            else:
                try:
                    recv_ct = 0
                    bootstrap = (not active) and (not scheduler.has_unfinished())
                    with _maybe_nvtx_range("roseinfer.mp.drain_cmds", nvtx):
                        while True:
                            if (
                                bootstrap
                                and bootstrap_target > 0
                                and len(pending) >= bootstrap_target
                            ):
                                break
                            fill_needed = bool(
                                (not bootstrap)
                                and fill_target
                                and bootstrap_target > 0
                                and (len(active) + len(pending)) < bootstrap_target
                            )
                            if (
                                (not bootstrap)
                                and (not fill_needed)
                                and max_recv != 0
                                and recv_ct >= max_recv
                            ):
                                break
                            cmd = cmd_q.get_nowait()
                            recv_ct += 1
                            if isinstance(cmd, ShutdownCmd):
                                return
                            if isinstance(cmd, CancelRequestCmd):
                                cancel_request(int(cmd.request_id))
                            elif isinstance(cmd, StartProfileCmd):
                                ok = True
                                msg: str | None = None
                                try:
                                    start_profile(cmd)
                                except Exception as exc:
                                    ok = False
                                    msg = str(exc)
                                evt_q.put(
                                    ProfileEvent(
                                        profile_id=int(cmd.profile_id),
                                        action="start",
                                        tool=str(cmd.tool),
                                        ok=ok,
                                        message=msg,
                                    )
                                )
                            elif isinstance(cmd, StopProfileCmd):
                                ok = True
                                msg: str | None = None
                                try:
                                    stop_profile(cmd)
                                except Exception as exc:
                                    ok = False
                                    msg = str(exc)
                                evt_q.put(
                                    ProfileEvent(
                                        profile_id=int(cmd.profile_id),
                                        action="stop",
                                        tool=str(cmd.tool),
                                        ok=ok,
                                        message=msg,
                                    )
                                )
                            elif isinstance(cmd, AddBatchCmd):
                                _enqueue_add_batch(cmd, pending, pending_by_rid)
                            elif isinstance(cmd, AddRequestCmd):
                                pending.append(cmd)
                                pending_by_rid[int(cmd.request_id)] = cmd
                except queue.Empty:
                    pass

            tokens_out: dict[int, list[int]] | None = None
            pair_rids: array | None = None
            pair_toks: array | None = None
            finished_counts_rids: list[int] | None = None
            finished_counts: list[int] | None = None
            finished: set[int] | None = set() if emit_tokens else None

            if emit_tokens:
                if flat_events:
                    pair_rids = array("I")
                    pair_toks = array("I")
                else:
                    tokens_out = {}
            else:
                finished_counts_rids = []
                finished_counts = []

            kv_conc = int(getattr(args, "kv_cache_max_concurrency", 0) or 0)
            admit_limit = int(args.max_batch_size)
            if kv_conc > 0:
                admit_limit = min(admit_limit, max(0, int(kv_conc - len(active))))
            batch: list[OnlineRequest] = []
            while pending and len(batch) < admit_limit:
                item = pending.popleft()
                if pending_by_rid.pop(int(item.request_id), None) is None:
                    continue
                batch.append(
                    OnlineRequest(
                        prompt="",
                        prompt_token_ids=item.prompt_token_ids,
                        max_new_tokens=int(item.max_new_tokens),
                        temperature=float(item.temperature),
                        top_k=int(item.top_k),
                        top_p=float(item.top_p),
                        stop_on_eos=bool(item.stop_on_eos),
                        do_sample=bool(item.do_sample),
                        request_id=int(item.request_id),
                    )
                )

            if batch:
                with _maybe_nvtx_range("roseinfer.mp.add_requests", nvtx):
                    rids = scheduler.add_requests(batch)
                if torch_prof is not None:
                    step_fn = getattr(torch_prof, "step", None)
                    if callable(step_fn):
                        step_fn()
                for rid in rids:
                    active.add(int(rid))
                    if token_counts is not None:
                        token_counts.setdefault(int(rid), 0)
                        token_ids = scheduler.get_generated_ids(rid)
                        if token_ids:
                            token_counts[int(rid)] = int(
                                token_counts.get(int(rid), 0) + len(token_ids)
                            )
                    if emit_tokens:
                        token_ids = scheduler.get_generated_ids(rid)
                        if token_ids:
                            if pair_rids is not None and pair_toks is not None:
                                for token_id in token_ids:
                                    pair_rids.append(int(rid))
                                    pair_toks.append(int(token_id))
                            else:
                                assert tokens_out is not None
                                tokens_out[int(rid)] = [int(t) for t in token_ids]
                    if scheduler.is_finished(rid):
                        if finished is not None:
                            finished.add(int(rid))
                        if not emit_tokens:
                            assert (
                                finished_counts_rids is not None
                                and finished_counts is not None
                            )
                            if token_counts is not None:
                                finished_counts_rids.append(int(rid))
                                finished_counts.append(
                                    int(token_counts.get(int(rid), 0))
                                )
                            else:
                                finished_counts_rids.append(int(rid))
                                finished_counts.append(
                                    int(scheduler.get_step_count(int(rid)))
                                )
                        scheduler.discard_request(int(rid))
                        active.discard(int(rid))
                        if token_counts is not None:
                            token_counts.pop(int(rid), None)

            if scheduler.has_unfinished():
                with _maybe_nvtx_range("roseinfer.mp.step", nvtx):
                    step_tokens = scheduler.step()
                if torch_prof is not None:
                    step_fn = getattr(torch_prof, "step", None)
                    if callable(step_fn):
                        step_fn()
                for rid, token_id in step_tokens.items():
                    if token_counts is not None:
                        token_counts[int(rid)] = int(token_counts.get(int(rid), 0) + 1)
                    if emit_tokens:
                        if pair_rids is not None and pair_toks is not None:
                            pair_rids.append(int(rid))
                            pair_toks.append(int(token_id))
                        else:
                            assert tokens_out is not None
                            tokens_out.setdefault(int(rid), []).append(int(token_id))
                for rid in scheduler.pop_finished_ids():
                    if finished is not None:
                        finished.add(int(rid))
                    if not emit_tokens:
                        assert (
                            finished_counts_rids is not None
                            and finished_counts is not None
                        )
                        if token_counts is not None:
                            finished_counts_rids.append(int(rid))
                            finished_counts.append(int(token_counts.get(int(rid), 0)))
                        else:
                            finished_counts_rids.append(int(rid))
                            finished_counts.append(
                                int(scheduler.get_step_count(int(rid)))
                            )
                    scheduler.discard_request(int(rid))
                    active.discard(int(rid))
                    if token_counts is not None:
                        token_counts.pop(int(rid), None)

            if emit_tokens:
                if pair_rids is not None and pair_toks is not None:
                    if pair_rids or finished:
                        evt_q.put(
                            TokenPairsEvent(
                                rids=pair_rids,
                                token_ids=pair_toks,
                                finished_ids=array("I", sorted(finished or set())),
                            )
                        )
                elif (tokens_out or finished) and tokens_out is not None:
                    evt_q.put(
                        TokensEvent(
                            tokens=tokens_out,
                            finished_ids=sorted(finished or set()),
                        )
                    )
            elif finished_counts_rids:
                evt_q.put(
                    TokenCountsEvent(
                        request_ids=array("I", finished_counts_rids),
                        token_counts=array("I", finished_counts or []),
                    )
                )
    except KeyboardInterrupt:
        return
    except Exception as exc:
        traceback.print_exc()
        try:
            evt_q.put(ErrorEvent(request_id=None, message=str(exc)))
        except Exception:
            pass


def _engine_process_main_pipe(
    args: EngineProcessArgs,
    cmd_recv: Connection,
    evt_send: Connection,
) -> None:
    def recv_blocking() -> EngineCommand:
        return _decode_cmd(cmd_recv.recv_bytes())

    def send_evt(evt: EngineEvent) -> None:
        evt_send.send_bytes(_encode_evt(evt))

    torch.set_grad_enabled(False)
    try:
        gc_warn_ms = float(getattr(args, "gc_warn_ms", 0.0) or 0.0)
        gc_log_all = bool(getattr(args, "gc_log_all", False))
        if gc_warn_ms > 0.0 or gc_log_all:
            from rosellm.roseinfer.gc_observer import install_gc_observer

            install_gc_observer(
                warn_ms=gc_warn_ms,
                log_all=gc_log_all,
                prefix=f"engine pid={os.getpid()}",
                nvtx=os.environ.get("ROSEINFER_NVTX") == "1",
            )
        _maybe_set_cpu_affinity(args.mp_cpu_affinity)
        _maybe_set_torch_threads(
            num_threads=args.mp_torch_num_threads,
            num_interop_threads=args.mp_torch_num_interop_threads,
        )
        engine = _build_engine(args)
        if bool(args.chunked_prefill):
            scheduler: OnlineScheduler | ChunkedOnlineScheduler = (
                ChunkedOnlineScheduler(
                    engine,
                    max_batch_size=int(args.max_batch_size),
                    prefill_chunk_size=int(args.prefill_chunk_size),
                    prefill_max_batch_size=int(args.max_batch_size),
                    use_prefix_cache=bool(args.prefix_cache),
                    overlap_schedule=bool(args.overlap_schedule),
                )
            )
        else:
            scheduler = OnlineScheduler(
                engine,
                max_batch_size=int(args.max_batch_size),
                use_prefix_cache=bool(args.prefix_cache),
                overlap_schedule=bool(args.overlap_schedule),
            )
        engine.warmup_paged_attention_decode()
        if bool(getattr(args, "gc_freeze", True)):
            _freeze_gc_heap()
        send_evt(
            ReadyEvent(
                eos_token_id=engine.eos_token_id,
                max_context=int(engine.config.max_position_embeddings),
            )
        )

        pending: "Deque[AddRequestCmd]" = deque()
        pending_by_rid: dict[int, AddRequestCmd] = {}
        active: set[int] = set()
        token_counts: dict[int, int] | None = None
        emit_tokens = bool(getattr(args, "mp_emit_token_events", True))
        flat_events = bool(getattr(args, "mp_flat_events", True))
        fast_finish_counts = bool(getattr(args, "mp_fast_finish_counts", True))
        if (not emit_tokens) and (not fast_finish_counts):
            token_counts = {}
        torch_prof: Any | None = None
        torch_prof_dir: str | None = None
        cuda_prof_active = False
        nvtx = bool(
            torch.cuda.is_available() and os.environ.get("ROSEINFER_NVTX") == "1"
        )

        def start_profile(cmd: StartProfileCmd) -> None:
            nonlocal torch_prof, torch_prof_dir, cuda_prof_active
            tool = str(cmd.tool).lower()
            if tool == "torch":
                if torch_prof is not None:
                    raise RuntimeError("torch profiler already running")
                if cmd.output_dir is None:
                    raise ValueError("output_dir is required for torch profiler")
                out_dir = str(cmd.output_dir)
                os.makedirs(out_dir, exist_ok=True)
                activities = [torch.profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with_stack = os.environ.get("ROSEINFER_TORCH_PROFILE_WITH_STACK") == "1"
                record_shapes = (
                    os.environ.get("ROSEINFER_TORCH_PROFILE_RECORD_SHAPES", "1") == "1"
                )
                profile_memory = (
                    os.environ.get("ROSEINFER_TORCH_PROFILE_WITH_PROFILE_MEMORY", "1")
                    == "1"
                )
                delay_steps = max(
                    0, int(os.environ.get("ROSEINFER_TORCH_PROFILE_DELAY_STEPS", "0"))
                )
                num_steps = max(
                    0, int(os.environ.get("ROSEINFER_TORCH_PROFILE_NUM_STEPS", "0"))
                )
                schedule = (
                    torch.profiler.schedule(
                        wait=delay_steps, warmup=0, active=num_steps, repeat=1
                    )
                    if num_steps > 0
                    else None
                )
                prof = torch.profiler.profile(
                    activities=activities,
                    record_shapes=record_shapes,
                    profile_memory=profile_memory,
                    with_stack=with_stack,
                    schedule=schedule,
                )
                prof.__enter__()
                torch_prof = prof
                torch_prof_dir = out_dir
                return
            if tool == "cuda":
                if cuda_prof_active:
                    raise RuntimeError("cuda profiler already running")
                if torch.cuda.is_available():
                    torch.cuda.profiler.start()
                cuda_prof_active = True
                return
            raise ValueError("tool must be torch|cuda")

        def stop_profile(cmd: StopProfileCmd) -> None:
            nonlocal torch_prof, torch_prof_dir, cuda_prof_active
            tool = str(cmd.tool).lower()
            if tool == "torch":
                if torch_prof is None or torch_prof_dir is None:
                    raise RuntimeError("torch profiler not running")
                prof = torch_prof
                out_dir = torch_prof_dir
                prof.__exit__(None, None, None)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof.export_chrome_trace(os.path.join(out_dir, "trace.json"))
                torch_prof = None
                torch_prof_dir = None
                return
            if tool == "cuda":
                if not cuda_prof_active:
                    raise RuntimeError("cuda profiler not running")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
                cuda_prof_active = False
                return
            raise ValueError("tool must be torch|cuda")

        def cancel_request(request_id: int) -> None:
            pending_by_rid.pop(request_id, None)
            active.discard(int(request_id))
            if token_counts is not None:
                token_counts.pop(int(request_id), None)
            scheduler.discard_request(request_id)

        max_recv = int(getattr(args, "mp_max_recv_per_iter", 64))
        if max_recv < 0:
            max_recv = 0
        bootstrap_target = min(
            int(args.max_batch_size),
            int(getattr(args, "kv_cache_max_concurrency", int(args.max_batch_size))),
        )
        fill_target = bool(getattr(args, "mp_fill_target", True))

        while True:
            if not pending_by_rid and not scheduler.has_unfinished():
                cmd = recv_blocking()
                if isinstance(cmd, ShutdownCmd):
                    return
                if isinstance(cmd, StartProfileCmd):
                    ok = True
                    msg: str | None = None
                    try:
                        start_profile(cmd)
                    except Exception as exc:
                        ok = False
                        msg = str(exc)
                    send_evt(
                        ProfileEvent(
                            profile_id=int(cmd.profile_id),
                            action="start",
                            tool=str(cmd.tool),
                            ok=ok,
                            message=msg,
                        )
                    )
                    continue
                if isinstance(cmd, StopProfileCmd):
                    ok = True
                    msg: str | None = None
                    try:
                        stop_profile(cmd)
                    except Exception as exc:
                        ok = False
                        msg = str(exc)
                    send_evt(
                        ProfileEvent(
                            profile_id=int(cmd.profile_id),
                            action="stop",
                            tool=str(cmd.tool),
                            ok=ok,
                            message=msg,
                        )
                    )
                    continue
                if isinstance(cmd, CancelRequestCmd):
                    cancel_request(int(cmd.request_id))
                    continue
                if isinstance(cmd, AddBatchCmd):
                    _enqueue_add_batch(cmd, pending, pending_by_rid)
                    continue
                if isinstance(cmd, AddRequestCmd):
                    pending.append(cmd)
                    pending_by_rid[int(cmd.request_id)] = cmd
            else:
                recv_ct = 0
                bootstrap = (not active) and (not scheduler.has_unfinished())
                with _maybe_nvtx_range("roseinfer.mp.drain_cmds", nvtx):
                    while True:
                        if (
                            bootstrap
                            and bootstrap_target > 0
                            and len(pending) >= bootstrap_target
                        ):
                            break
                        fill_needed = bool(
                            (not bootstrap)
                            and fill_target
                            and bootstrap_target > 0
                            and (len(active) + len(pending)) < bootstrap_target
                        )
                        if (
                            (not bootstrap)
                            and (not fill_needed)
                            and max_recv != 0
                            and recv_ct >= max_recv
                        ):
                            break
                        if not cmd_recv.poll(0.0):
                            break
                        cmd = _decode_cmd(cmd_recv.recv_bytes())
                        recv_ct += 1
                        if isinstance(cmd, ShutdownCmd):
                            return
                        if isinstance(cmd, CancelRequestCmd):
                            cancel_request(int(cmd.request_id))
                        elif isinstance(cmd, StartProfileCmd):
                            ok = True
                            msg: str | None = None
                            try:
                                start_profile(cmd)
                            except Exception as exc:
                                ok = False
                                msg = str(exc)
                            send_evt(
                                ProfileEvent(
                                    profile_id=int(cmd.profile_id),
                                    action="start",
                                    tool=str(cmd.tool),
                                    ok=ok,
                                    message=msg,
                                )
                            )
                        elif isinstance(cmd, StopProfileCmd):
                            ok = True
                            msg: str | None = None
                            try:
                                stop_profile(cmd)
                            except Exception as exc:
                                ok = False
                                msg = str(exc)
                            send_evt(
                                ProfileEvent(
                                    profile_id=int(cmd.profile_id),
                                    action="stop",
                                    tool=str(cmd.tool),
                                    ok=ok,
                                    message=msg,
                                )
                            )
                        elif isinstance(cmd, AddBatchCmd):
                            _enqueue_add_batch(cmd, pending, pending_by_rid)
                        elif isinstance(cmd, AddRequestCmd):
                            pending.append(cmd)
                            pending_by_rid[int(cmd.request_id)] = cmd

            tokens_out: dict[int, list[int]] | None = None
            pair_rids: array | None = None
            pair_toks: array | None = None
            finished_counts_rids: list[int] | None = None
            finished_counts: list[int] | None = None
            finished: set[int] | None = set() if emit_tokens else None

            if emit_tokens:
                if flat_events:
                    pair_rids = array("I")
                    pair_toks = array("I")
                else:
                    tokens_out = {}
            else:
                finished_counts_rids = []
                finished_counts = []

            kv_conc = int(getattr(args, "kv_cache_max_concurrency", 0) or 0)
            admit_limit = int(args.max_batch_size)
            if kv_conc > 0:
                admit_limit = min(admit_limit, max(0, int(kv_conc - len(active))))
            batch: list[OnlineRequest] = []
            while pending and len(batch) < admit_limit:
                item = pending.popleft()
                if pending_by_rid.pop(int(item.request_id), None) is None:
                    continue
                batch.append(
                    OnlineRequest(
                        prompt="",
                        prompt_token_ids=item.prompt_token_ids,
                        max_new_tokens=int(item.max_new_tokens),
                        temperature=float(item.temperature),
                        top_k=int(item.top_k),
                        top_p=float(item.top_p),
                        stop_on_eos=bool(item.stop_on_eos),
                        do_sample=bool(item.do_sample),
                        request_id=int(item.request_id),
                    )
                )

            if batch:
                with _maybe_nvtx_range("roseinfer.mp.add_requests", nvtx):
                    rids = scheduler.add_requests(batch)
                if torch_prof is not None:
                    step_fn = getattr(torch_prof, "step", None)
                    if callable(step_fn):
                        step_fn()
                for rid in rids:
                    active.add(int(rid))
                    if token_counts is not None:
                        token_counts.setdefault(int(rid), 0)
                        token_ids = scheduler.get_generated_ids(rid)
                        if token_ids:
                            token_counts[int(rid)] = int(
                                token_counts.get(int(rid), 0) + len(token_ids)
                            )
                    if emit_tokens:
                        token_ids = scheduler.get_generated_ids(rid)
                        if token_ids:
                            if pair_rids is not None and pair_toks is not None:
                                for token_id in token_ids:
                                    pair_rids.append(int(rid))
                                    pair_toks.append(int(token_id))
                            else:
                                assert tokens_out is not None
                                tokens_out[int(rid)] = [int(t) for t in token_ids]
                    if scheduler.is_finished(rid):
                        if finished is not None:
                            finished.add(int(rid))
                        if not emit_tokens:
                            assert (
                                finished_counts_rids is not None
                                and finished_counts is not None
                            )
                            if token_counts is not None:
                                finished_counts_rids.append(int(rid))
                                finished_counts.append(
                                    int(token_counts.get(int(rid), 0))
                                )
                            else:
                                finished_counts_rids.append(int(rid))
                                finished_counts.append(
                                    int(scheduler.get_step_count(int(rid)))
                                )
                        scheduler.discard_request(int(rid))
                        active.discard(int(rid))
                        if token_counts is not None:
                            token_counts.pop(int(rid), None)

            if scheduler.has_unfinished():
                with _maybe_nvtx_range("roseinfer.mp.step", nvtx):
                    step_tokens = scheduler.step()
                if torch_prof is not None:
                    step_fn = getattr(torch_prof, "step", None)
                    if callable(step_fn):
                        step_fn()
                for rid, token_id in step_tokens.items():
                    if token_counts is not None:
                        token_counts[int(rid)] = int(token_counts.get(int(rid), 0) + 1)
                    if emit_tokens:
                        if pair_rids is not None and pair_toks is not None:
                            pair_rids.append(int(rid))
                            pair_toks.append(int(token_id))
                        else:
                            assert tokens_out is not None
                            tokens_out.setdefault(int(rid), []).append(int(token_id))
                for rid in scheduler.pop_finished_ids():
                    if finished is not None:
                        finished.add(int(rid))
                    if not emit_tokens:
                        assert (
                            finished_counts_rids is not None
                            and finished_counts is not None
                        )
                        if token_counts is not None:
                            finished_counts_rids.append(int(rid))
                            finished_counts.append(int(token_counts.get(int(rid), 0)))
                        else:
                            finished_counts_rids.append(int(rid))
                            finished_counts.append(
                                int(scheduler.get_step_count(int(rid)))
                            )
                    scheduler.discard_request(int(rid))
                    active.discard(int(rid))
                    if token_counts is not None:
                        token_counts.pop(int(rid), None)

            if emit_tokens:
                if pair_rids is not None and pair_toks is not None:
                    if pair_rids or finished:
                        send_evt(
                            TokenPairsEvent(
                                rids=pair_rids,
                                token_ids=pair_toks,
                                finished_ids=array("I", sorted(finished or set())),
                            )
                        )
                elif (tokens_out or finished) and tokens_out is not None:
                    send_evt(
                        TokensEvent(
                            tokens=tokens_out,
                            finished_ids=sorted(finished or set()),
                        )
                    )
            elif finished_counts_rids:
                send_evt(
                    TokenCountsEvent(
                        request_ids=array("I", finished_counts_rids),
                        token_counts=array("I", finished_counts or []),
                    )
                )
    except KeyboardInterrupt:
        return
    except Exception as exc:
        traceback.print_exc()
        try:
            send_evt(ErrorEvent(request_id=None, message=str(exc)))
        except Exception:
            pass


class _StreamState:
    __slots__ = ("buf", "tokens_since_flush", "sent_any")

    def __init__(self) -> None:
        self.buf: list[str] = []
        self.tokens_since_flush = 0
        self.sent_any = False


class MPSchedulerManager:
    def __init__(
        self,
        tokenizer: Any,
        *,
        engine_args: EngineProcessArgs,
        stream_interval: int = 1,
        max_inflight_requests: int | None = None,
        start_timeout_s: float = 300.0,
        ipc_mode: str = "pipe",
        async_admit: bool = False,
        tokenize_workers: int = 0,
    ) -> None:
        if int(stream_interval) <= 0:
            raise ValueError("stream_interval must be >= 1")
        ipc_mode = str(ipc_mode).lower()
        if ipc_mode not in ("queue", "pipe"):
            raise ValueError("ipc_mode must be queue|pipe")
        tokenize_workers = int(tokenize_workers)
        if tokenize_workers < 0:
            raise ValueError("tokenize_workers must be >= 0")
        self.tokenizer = tokenizer
        self._make_detok = lambda: build_detokenizer(
            self.tokenizer, tokenizer_name=engine_args.tokenizer_name
        )
        self._stream_interval = int(stream_interval)
        self._async_admit = bool(async_admit)
        self._tokenize_workers = tokenize_workers
        self._max_inflight_requests = (
            int(max_inflight_requests) if max_inflight_requests is not None else None
        )
        if self._max_inflight_requests is not None and self._max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be >= 1")
        self._lock = threading.Lock()
        self._running = True
        self._next_request_id = 0
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._queues: Dict[int, "HybridQueue[Optional[str]]"] = {}
        self._detoks: Dict[int, BaseDetokenizer] = {}
        self._stream_states: Dict[int, _StreamState] = {}
        self._max_context: int | None = None
        self._profile_seq: int = 0
        self._profile_waiters: dict[int, threading.Event] = {}
        self._profile_results: dict[int, ProfileEvent] = {}
        self._cuda_prof_active = False
        self._admit_q: "queue.Queue[_MPAdmitTask | None] | None" = None
        self._admit_threads: list[threading.Thread] = []

        ctx = mp.get_context("spawn")
        self._ipc_mode = ipc_mode
        self._cmd_lock = threading.Lock()
        self._cmd_q: "mp.Queue[EngineCommand] | None" = None
        self._evt_q: "mp.Queue[EngineEvent] | None" = None
        self._cmd_send: Connection | None = None
        self._evt_recv: Connection | None = None
        self._proc: mp.Process
        if ipc_mode == "queue":
            self._cmd_q = ctx.Queue()
            self._evt_q = ctx.Queue()
            self._proc = ctx.Process(
                target=_engine_process_main,
                args=(engine_args, self._cmd_q, self._evt_q),
                daemon=True,
            )
        else:
            cmd_recv, cmd_send = ctx.Pipe(duplex=True)
            evt_recv, evt_send = ctx.Pipe(duplex=True)
            _maybe_set_pipe_socket_buffers(
                (cmd_recv, cmd_send, evt_recv, evt_send),
                int(os.environ.get("ROSEINFER_MP_PIPE_BUF_BYTES", "8388608")),
            )
            self._cmd_send = cmd_send
            self._evt_recv = evt_recv
            self._proc = ctx.Process(
                target=_engine_process_main_pipe,
                args=(engine_args, cmd_recv, evt_send),
                daemon=True,
            )
        self._proc.start()
        if ipc_mode == "pipe":
            try:
                cmd_recv.close()
            except Exception:
                pass
            try:
                evt_send.close()
            except Exception:
                pass

        ready = self._await_ready(timeout_s=float(start_timeout_s))
        self._max_context = int(ready.max_context)

        self._worker = threading.Thread(
            target=self._event_loop,
            name="roseinfer-mp-dispatch",
            daemon=True,
        )
        self._worker.start()

        if self._async_admit:
            self._admit_q = queue.Queue()
            worker_ct = int(self._tokenize_workers) if self._tokenize_workers > 0 else 1
            for i in range(worker_ct):
                th = threading.Thread(
                    target=self._admit_loop,
                    name=f"roseinfer-mp-admit-{i}",
                    daemon=True,
                )
                self._admit_threads.append(th)
                th.start()

    def set_asyncio_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._async_loop = loop
        with self._lock:
            queues = list(self._queues.values())
        for q in queues:
            q.set_asyncio_loop(loop)

    def _await_ready(self, *, timeout_s: float) -> ReadyEvent:
        start = time.perf_counter()
        while True:
            left = timeout_s - (time.perf_counter() - start)
            if left <= 0:
                raise TimeoutError("engine process failed to start within timeout")
            if self._ipc_mode == "queue":
                evt_q = self._evt_q
                if evt_q is None:
                    raise RuntimeError("missing event queue")
                evt = evt_q.get(timeout=left)
            else:
                evt_recv = self._evt_recv
                if evt_recv is None:
                    raise RuntimeError("missing event pipe")
                if not evt_recv.poll(left):
                    raise TimeoutError("engine process failed to start within timeout")
                evt = _decode_evt(evt_recv.recv_bytes())
            if isinstance(evt, ReadyEvent):
                return evt
            if isinstance(evt, ErrorEvent):
                raise RuntimeError(f"engine process failed: {evt.message}")

    def close(self) -> None:
        proc = self._proc
        worker = self._worker
        admit_threads = list(self._admit_threads)
        admit_q = self._admit_q
        if admit_q is not None:
            for _ in admit_threads:
                admit_q.put(None)
        with self._lock:
            if not self._running:
                return
            self._running = False
            waiters = list(self._profile_waiters.values())
            self._profile_waiters.clear()
            self._profile_results.clear()
            queues = list(self._queues.values())
            self._queues.clear()
            self._detoks.clear()
            self._stream_states.clear()
            cuda_active = bool(self._cuda_prof_active)
            self._cuda_prof_active = False
        for ev in waiters:
            ev.set()
        for q in queues:
            q.put(None)
        if cuda_active and torch.cuda.is_available():
            try:
                torch.cuda.profiler.stop()
            except Exception:
                pass
        try:
            if self._ipc_mode == "queue":
                cmd_q = self._cmd_q
                if cmd_q is not None:
                    cmd_q.put(ShutdownCmd())
            else:
                cmd_send = self._cmd_send
                if cmd_send is not None:
                    with self._cmd_lock:
                        cmd_send.send_bytes(_encode_cmd(ShutdownCmd()))
        except Exception:
            pass
        if self._ipc_mode == "pipe":
            try:
                if self._cmd_send is not None:
                    self._cmd_send.close()
            except Exception:
                pass
            try:
                if self._evt_recv is not None:
                    self._evt_recv.close()
            except Exception:
                pass
        worker.join(timeout=1.0)
        for th in admit_threads:
            th.join(timeout=1.0)
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)

    def _cancel_backend(self, request_id: int) -> None:
        try:
            if self._ipc_mode == "queue":
                cmd_q = self._cmd_q
                if cmd_q is None:
                    return
                cmd_q.put(CancelRequestCmd(request_id=int(request_id)))
            else:
                cmd_send = self._cmd_send
                if cmd_send is None:
                    return
                with self._cmd_lock:
                    cmd_send.send_bytes(
                        _encode_cmd(CancelRequestCmd(request_id=int(request_id)))
                    )
        except Exception:
            pass

    def add_request(
        self,
        prompt: str,
        prompt_token_ids: list[int] | None = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        do_sample: bool = False,
    ) -> int:
        with self._lock:
            if not self._running:
                raise RuntimeError("MPSchedulerManager is closed")
            if self._max_inflight_requests is not None and (
                len(self._queues) >= self._max_inflight_requests
            ):
                raise SchedulerManagerOverloadedError("too many inflight requests")
            request_id = self._next_request_id
            self._next_request_id += 1
            q: "HybridQueue[Optional[str]]" = HybridQueue()
            loop = self._async_loop
            if loop is not None:
                q.set_asyncio_loop(loop)
            detok = self._make_detok()
            self._queues[request_id] = q
            self._detoks[request_id] = detok
            self._stream_states[request_id] = _StreamState()

        if self._async_admit:
            admit_q = self._admit_q
            if admit_q is None:
                raise RuntimeError("async admit queue is not initialized")
            admit_q.put(
                _MPAdmitTask(
                    request_id=int(request_id),
                    prompt=str(prompt),
                    prompt_token_ids=list(prompt_token_ids)
                    if prompt_token_ids is not None
                    else None,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    stop_on_eos=bool(stop_on_eos),
                    do_sample=bool(do_sample),
                )
            )
            return request_id

        if prompt_token_ids is None:
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            token_ids = list(prompt_token_ids)
        if not token_ids:
            eos = getattr(self.tokenizer, "eos_token_id", 0)
            token_ids = [int(eos) if eos is not None else 0]
        max_ctx = self._max_context
        if max_ctx is not None and len(token_ids) > max_ctx:
            token_ids = token_ids[-max_ctx:]
        detok.start_prompt(token_ids)
        try:
            cmd = AddRequestCmd(
                request_id=int(request_id),
                prompt_token_ids=token_ids,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                stop_on_eos=bool(stop_on_eos),
                do_sample=bool(do_sample),
            )
            if self._ipc_mode == "queue":
                cmd_q = self._cmd_q
                if cmd_q is None:
                    raise RuntimeError("missing command queue")
                cmd_q.put(cmd)
            else:
                cmd_send = self._cmd_send
                if cmd_send is None:
                    raise RuntimeError("missing command pipe")
                payload = _encode_cmd(cmd)
                with self._cmd_lock:
                    cmd_send.send_bytes(payload)
        except Exception:
            self._cancel_backend(request_id)
            with self._lock:
                q2 = self._queues.pop(request_id, None)
                self._detoks.pop(request_id, None)
                self._stream_states.pop(request_id, None)
            if q2 is not None:
                q2.put(None)
            raise
        return request_id

    def _admit_loop(self) -> None:
        q = self._admit_q
        if q is None:
            return
        try:
            while True:
                task = q.get()
                if task is None:
                    return
                rid = int(task.request_id)
                with self._lock:
                    if not self._running or rid not in self._queues:
                        continue
                    detok = self._detoks.get(rid)
                    out_q = self._queues.get(rid)
                if detok is None or out_q is None:
                    continue

                try:
                    if task.prompt_token_ids is None:
                        token_ids = self.tokenizer.encode(
                            task.prompt,
                            add_special_tokens=False,
                        )
                    else:
                        token_ids = list(task.prompt_token_ids)
                    if not token_ids:
                        eos = getattr(self.tokenizer, "eos_token_id", 0)
                        token_ids = [int(eos) if eos is not None else 0]
                    max_ctx = self._max_context
                    if max_ctx is not None and len(token_ids) > max_ctx:
                        token_ids = token_ids[-max_ctx:]

                    with self._lock:
                        if not self._running or rid not in self._queues:
                            continue

                    detok.start_prompt(token_ids)

                    with self._lock:
                        if not self._running or rid not in self._queues:
                            continue

                    cmd = AddRequestCmd(
                        request_id=rid,
                        prompt_token_ids=token_ids,
                        max_new_tokens=int(task.max_new_tokens),
                        temperature=float(task.temperature),
                        top_k=int(task.top_k),
                        top_p=float(task.top_p),
                        stop_on_eos=bool(task.stop_on_eos),
                        do_sample=bool(task.do_sample),
                    )
                    if self._ipc_mode == "queue":
                        cmd_q = self._cmd_q
                        if cmd_q is None:
                            raise RuntimeError("missing command queue")
                        cmd_q.put(cmd)
                    else:
                        cmd_send = self._cmd_send
                        if cmd_send is None:
                            raise RuntimeError("missing command pipe")
                        payload = _encode_cmd(cmd)
                        with self._cmd_lock:
                            cmd_send.send_bytes(payload)
                except Exception:
                    self._cancel_backend(rid)
                    out_q.put(None)
                    with self._lock:
                        self._queues.pop(rid, None)
                        self._detoks.pop(rid, None)
                        self._stream_states.pop(rid, None)
        except Exception:
            traceback.print_exc()

    def stream_text(self, request_id: int) -> Iterator[str]:
        with self._lock:
            q = self._queues.get(request_id)
        if q is None:
            return
        try:
            while True:
                piece = q.get()
                if piece is None:
                    break
                yield piece
        finally:
            with self._lock:
                self._queues.pop(request_id, None)
                self._detoks.pop(request_id, None)
                self._stream_states.pop(request_id, None)

    async def astream_text(self, request_id: int) -> AsyncIterator[str]:
        with self._lock:
            q = self._queues.get(request_id)
        if q is None:
            return
        try:
            while True:
                piece = await q.aget()
                if piece is None:
                    break
                yield piece
        finally:
            with self._lock:
                self._queues.pop(request_id, None)
                self._detoks.pop(request_id, None)
                self._stream_states.pop(request_id, None)
            self._cancel_backend(request_id)

    def start_profile(
        self,
        *,
        tool: str,
        output_dir: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        tool = str(tool).lower()
        if tool not in ("torch", "cuda"):
            raise ValueError("tool must be torch|cuda")
        if tool == "torch" and output_dir is None:
            raise ValueError("output_dir is required for torch profiler")

        waiter = threading.Event()
        with self._lock:
            if not self._running:
                raise RuntimeError("MPSchedulerManager is closed")
            if tool == "cuda" and self._cuda_prof_active:
                raise RuntimeError("cuda profiler already running")
            profile_id = int(self._profile_seq)
            self._profile_seq += 1
            self._profile_waiters[profile_id] = waiter
        cmd = StartProfileCmd(
            profile_id=profile_id,
            tool=tool,
            output_dir=str(output_dir) if output_dir is not None else None,
        )
        if self._ipc_mode == "queue":
            cmd_q = self._cmd_q
            if cmd_q is None:
                raise RuntimeError("missing command queue")
            cmd_q.put(cmd)
        else:
            cmd_send = self._cmd_send
            if cmd_send is None:
                raise RuntimeError("missing command pipe")
            payload = _encode_cmd(cmd)
            with self._cmd_lock:
                cmd_send.send_bytes(payload)

        try:
            if not waiter.wait(timeout=timeout_s):
                raise TimeoutError("start_profile timed out")
            with self._lock:
                evt = self._profile_results.pop(profile_id, None)
            if evt is None:
                raise RuntimeError("missing profile ack event")
            if not bool(evt.ok):
                raise RuntimeError(evt.message or "start_profile failed")
            if tool == "cuda" and torch.cuda.is_available():
                # Ensure the API process has an active CUDA context; otherwise
                # `cudaProfilerStart` may be a no-op and Nsight Systems capture-range
                # won't trigger for multiprocess profiling.
                try:
                    torch.cuda.current_device()
                except Exception:
                    pass
                torch.cuda.profiler.start()
                with self._lock:
                    self._cuda_prof_active = True
        except Exception:
            with self._lock:
                self._profile_waiters.pop(profile_id, None)
                self._profile_results.pop(profile_id, None)
            raise

    def stop_profile(
        self,
        *,
        tool: str,
        timeout_s: float = 300.0,
    ) -> None:
        tool = str(tool).lower()
        if tool not in ("torch", "cuda"):
            raise ValueError("tool must be torch|cuda")

        waiter = threading.Event()
        with self._lock:
            if not self._running:
                raise RuntimeError("MPSchedulerManager is closed")
            profile_id = int(self._profile_seq)
            self._profile_seq += 1
            self._profile_waiters[profile_id] = waiter
        cmd = StopProfileCmd(profile_id=profile_id, tool=tool)
        if self._ipc_mode == "queue":
            cmd_q = self._cmd_q
            if cmd_q is None:
                raise RuntimeError("missing command queue")
            cmd_q.put(cmd)
        else:
            cmd_send = self._cmd_send
            if cmd_send is None:
                raise RuntimeError("missing command pipe")
            payload = _encode_cmd(cmd)
            with self._cmd_lock:
                cmd_send.send_bytes(payload)
        if not waiter.wait(timeout=timeout_s):
            raise TimeoutError("stop_profile timed out")
        with self._lock:
            evt = self._profile_results.pop(profile_id, None)
        if evt is None:
            raise RuntimeError("missing profile ack event")
        if not bool(evt.ok):
            raise RuntimeError(evt.message or "stop_profile failed")
        if tool == "cuda":
            with self._lock:
                cuda_active = bool(self._cuda_prof_active)
                self._cuda_prof_active = False
            if cuda_active and torch.cuda.is_available():
                torch.cuda.profiler.stop()

    def _event_loop(self) -> None:
        try:
            while True:
                if self._ipc_mode == "queue":
                    evt_q = self._evt_q
                    if evt_q is None:
                        return
                    evt = evt_q.get()
                else:
                    evt_recv = self._evt_recv
                    if evt_recv is None:
                        return
                    try:
                        evt = _decode_evt(evt_recv.recv_bytes())
                    except EOFError:
                        return
                if evt is None:
                    return
                if isinstance(evt, ReadyEvent):
                    continue
                if isinstance(evt, ProfileEvent):
                    with self._lock:
                        waiter = self._profile_waiters.pop(int(evt.profile_id), None)
                        if waiter is not None:
                            self._profile_results[int(evt.profile_id)] = evt
                            waiter.set()
                    continue
                if isinstance(evt, ErrorEvent):
                    with self._lock:
                        waiters = list(self._profile_waiters.values())
                        self._profile_waiters.clear()
                        self._profile_results.clear()
                        queues = list(self._queues.values())
                        self._queues.clear()
                        self._detoks.clear()
                        self._stream_states.clear()
                        self._running = False
                    for ev in waiters:
                        ev.set()
                    for q in queues:
                        q.put(None)
                    return
                if isinstance(evt, TokensEvent):
                    token_records: list[
                        tuple[
                            int,
                            list[int],
                            "queue.Queue[Optional[str]] | None",
                            BaseDetokenizer | None,
                            _StreamState | None,
                        ]
                    ] = []
                    finished_records: list[
                        tuple[
                            int,
                            "queue.Queue[Optional[str]] | None",
                            BaseDetokenizer | None,
                            _StreamState | None,
                        ]
                    ] = []
                    with self._lock:
                        if not self._running:
                            return
                        for rid, token_ids in evt.tokens.items():
                            token_records.append(
                                (
                                    int(rid),
                                    token_ids,
                                    self._queues.get(int(rid)),
                                    self._detoks.get(int(rid)),
                                    self._stream_states.get(int(rid)),
                                )
                            )
                        for rid in evt.finished_ids:
                            finished_records.append(
                                (
                                    int(rid),
                                    self._queues.get(int(rid)),
                                    self._detoks.get(int(rid)),
                                    self._stream_states.get(int(rid)),
                                )
                            )

                    for rid, token_ids, q, detok, state in token_records:
                        if q is None or detok is None or state is None:
                            self._cancel_backend(rid)
                            continue
                        for token_id in token_ids:
                            piece = detok.on_token(int(token_id))
                            if piece:
                                state.buf.append(piece)
                            state.tokens_since_flush += 1
                            if state.buf and (
                                not state.sent_any
                                or state.tokens_since_flush >= self._stream_interval
                            ):
                                q.put("".join(state.buf))
                                state.buf.clear()
                                state.tokens_since_flush = 0
                                state.sent_any = True

                    for rid, q, detok, state in finished_records:
                        if q is None or state is None:
                            continue
                        if detok is not None:
                            tail = detok.flush()
                            if tail:
                                state.buf.append(tail)
                        if state.buf:
                            q.put("".join(state.buf))
                            state.buf.clear()
                            state.tokens_since_flush = 0
                            state.sent_any = True
                        q.put(None)
                    continue

                if isinstance(evt, TokenPairsEvent):
                    records: dict[
                        int,
                        tuple[
                            "queue.Queue[Optional[str]] | None",
                            BaseDetokenizer | None,
                            _StreamState | None,
                        ],
                    ] = {}
                    with self._lock:
                        if not self._running:
                            return
                        for rid in evt.rids:
                            rid_i = int(rid)
                            if rid_i in records:
                                continue
                            records[rid_i] = (
                                self._queues.get(rid_i),
                                self._detoks.get(rid_i),
                                self._stream_states.get(rid_i),
                            )
                        for rid in evt.finished_ids:
                            rid_i = int(rid)
                            if rid_i in records:
                                continue
                            records[rid_i] = (
                                self._queues.get(rid_i),
                                self._detoks.get(rid_i),
                                self._stream_states.get(rid_i),
                            )

                    for rid, token_id in zip(evt.rids, evt.token_ids):
                        q, detok, state = records.get(int(rid), (None, None, None))
                        if q is None or detok is None or state is None:
                            self._cancel_backend(int(rid))
                            continue
                        piece = detok.on_token(int(token_id))
                        if piece:
                            state.buf.append(piece)
                        state.tokens_since_flush += 1
                        if state.buf and (
                            not state.sent_any
                            or state.tokens_since_flush >= self._stream_interval
                        ):
                            q.put("".join(state.buf))
                            state.buf.clear()
                            state.tokens_since_flush = 0
                            state.sent_any = True

                    for rid in evt.finished_ids:
                        q, detok, state = records.get(int(rid), (None, None, None))
                        if q is None or state is None:
                            continue
                        if detok is not None:
                            tail = detok.flush()
                            if tail:
                                state.buf.append(tail)
                        if state.buf:
                            q.put("".join(state.buf))
                            state.buf.clear()
                            state.tokens_since_flush = 0
                            state.sent_any = True
                        q.put(None)
                    continue
        except Exception:
            traceback.print_exc()
            with self._lock:
                waiters = list(self._profile_waiters.values())
                self._profile_waiters.clear()
                self._profile_results.clear()
                queues = list(self._queues.values())
                self._queues.clear()
                self._detoks.clear()
                self._stream_states.clear()
                self._running = False
            for ev in waiters:
                ev.set()
            for q in queues:
                q.put(None)


class MPTokenManager:
    """A lightweight multiprocess client that only tracks token ids (no detokenization).

    This is mainly intended for offline throughput benchmarks where detokenization
    would skew comparisons.
    """

    def __init__(
        self,
        *,
        engine_args: EngineProcessArgs,
        max_inflight_requests: int | None = None,
        start_timeout_s: float = 300.0,
        ipc_mode: str = "pipe",
    ) -> None:
        ipc_mode = str(ipc_mode).lower()
        if ipc_mode not in ("queue", "pipe"):
            raise ValueError("ipc_mode must be queue|pipe")
        self._ipc_mode = ipc_mode
        self._cmd_lock = threading.Lock()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._running = True
        self._next_request_id = 0
        self._max_inflight_requests = (
            int(max_inflight_requests) if max_inflight_requests is not None else None
        )
        if self._max_inflight_requests is not None and self._max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be >= 1")
        self._max_context: int | None = None
        self._token_counts: dict[int, int] = {}
        self._finished: set[int] = set()
        self._error: str | None = None
        self._profile_seq: int = 0
        self._profile_waiters: dict[int, threading.Event] = {}
        self._profile_results: dict[int, ProfileEvent] = {}

        ctx = mp.get_context("spawn")
        self._cmd_q: "mp.Queue[EngineCommand] | None" = None
        self._evt_q: "mp.Queue[EngineEvent] | None" = None
        self._cmd_send: Connection | None = None
        self._evt_recv: Connection | None = None
        if ipc_mode == "queue":
            self._cmd_q = ctx.Queue()
            self._evt_q = ctx.Queue()
            self._proc = ctx.Process(
                target=_engine_process_main,
                args=(engine_args, self._cmd_q, self._evt_q),
                daemon=True,
            )
        else:
            cmd_recv, cmd_send = ctx.Pipe(duplex=True)
            evt_recv, evt_send = ctx.Pipe(duplex=True)
            _maybe_set_pipe_socket_buffers(
                (cmd_recv, cmd_send, evt_recv, evt_send),
                int(os.environ.get("ROSEINFER_MP_PIPE_BUF_BYTES", "8388608")),
            )
            self._cmd_send = cmd_send
            self._evt_recv = evt_recv
            self._proc = ctx.Process(
                target=_engine_process_main_pipe,
                args=(engine_args, cmd_recv, evt_send),
                daemon=True,
            )
        self._proc.start()
        if ipc_mode == "pipe":
            try:
                cmd_recv.close()
            except Exception:
                pass
            try:
                evt_send.close()
            except Exception:
                pass

        ready = self._await_ready(timeout_s=float(start_timeout_s))
        self._max_context = int(ready.max_context)

        self._worker = threading.Thread(
            target=self._event_loop,
            name="roseinfer-mp-token-dispatch",
            daemon=True,
        )
        self._worker.start()

    def _await_ready(self, *, timeout_s: float) -> ReadyEvent:
        start = time.perf_counter()
        while True:
            left = timeout_s - (time.perf_counter() - start)
            if left <= 0:
                raise TimeoutError("engine process failed to start within timeout")
            if self._ipc_mode == "queue":
                evt_q = self._evt_q
                if evt_q is None:
                    raise RuntimeError("missing event queue")
                evt = evt_q.get(timeout=left)
            else:
                evt_recv = self._evt_recv
                if evt_recv is None:
                    raise RuntimeError("missing event pipe")
                if not evt_recv.poll(left):
                    raise TimeoutError("engine process failed to start within timeout")
                evt = _decode_evt(evt_recv.recv_bytes())
            if isinstance(evt, ReadyEvent):
                return evt
            if isinstance(evt, ErrorEvent):
                raise RuntimeError(f"engine process failed: {evt.message}")

    def close(self) -> None:
        proc = self._proc
        worker = self._worker
        with self._lock:
            if not self._running:
                return
            self._running = False
            waiters = list(self._profile_waiters.values())
            self._profile_waiters.clear()
            self._profile_results.clear()
            self._cond.notify_all()
        for ev in waiters:
            ev.set()
        try:
            self._send_cmd(ShutdownCmd())
        except Exception:
            pass
        if self._ipc_mode == "pipe":
            try:
                if self._cmd_send is not None:
                    self._cmd_send.close()
            except Exception:
                pass
            try:
                if self._evt_recv is not None:
                    self._evt_recv.close()
            except Exception:
                pass
        worker.join(timeout=1.0)
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)

    def _send_cmd(self, cmd: EngineCommand) -> None:
        if self._ipc_mode == "queue":
            cmd_q = self._cmd_q
            if cmd_q is None:
                raise RuntimeError("missing command queue")
            cmd_q.put(cmd)
            return
        cmd_send = self._cmd_send
        if cmd_send is None:
            raise RuntimeError("missing command pipe")
        payload = _encode_cmd(cmd)
        with self._cmd_lock:
            cmd_send.send_bytes(payload)

    def add_request_token_ids(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        do_sample: bool = False,
    ) -> int:
        token_ids = list(prompt_token_ids)
        if not token_ids:
            token_ids = [0]
        max_ctx = self._max_context
        if max_ctx is not None and len(token_ids) > max_ctx:
            token_ids = token_ids[-max_ctx:]

        with self._lock:
            if not self._running:
                raise RuntimeError("MPTokenManager is closed")
            if self._max_inflight_requests is not None and (
                len(self._token_counts) - len(self._finished)
                >= self._max_inflight_requests
            ):
                raise SchedulerManagerOverloadedError("too many inflight requests")
            request_id = int(self._next_request_id)
            self._next_request_id += 1
            self._token_counts[request_id] = 0

        cmd = AddRequestCmd(
            request_id=int(request_id),
            prompt_token_ids=token_ids,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            stop_on_eos=bool(stop_on_eos),
            do_sample=bool(do_sample),
        )
        try:
            self._send_cmd(cmd)
        except Exception:
            with self._lock:
                self._token_counts.pop(request_id, None)
                self._finished.discard(request_id)
            raise
        return request_id

    def add_requests_token_ids(
        self,
        prompt_token_ids_batch: Sequence[Sequence[int]],
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        do_sample: bool = False,
    ) -> list[int]:
        token_batch: list[list[int]] = []
        for ids in prompt_token_ids_batch:
            token_ids = list(ids)
            if not token_ids:
                token_ids = [0]
            max_ctx = self._max_context
            if max_ctx is not None and len(token_ids) > max_ctx:
                token_ids = token_ids[-max_ctx:]
            token_batch.append(token_ids)

        n = len(token_batch)
        if n == 0:
            return []

        request_ids: list[int] = []
        with self._lock:
            if not self._running:
                raise RuntimeError("MPTokenManager is closed")
            inflight = len(self._token_counts) - len(self._finished)
            if self._max_inflight_requests is not None and (
                inflight + n > self._max_inflight_requests
            ):
                raise SchedulerManagerOverloadedError("too many inflight requests")
            for _ in range(n):
                request_id = int(self._next_request_id)
                self._next_request_id += 1
                request_ids.append(request_id)
                self._token_counts[request_id] = 0

        cmd = AddBatchCmd(
            request_ids=array("I", [int(r) for r in request_ids]),
            prompt_lens=array("I", [int(len(x)) for x in token_batch]),
            prompt_token_ids=array("I", [int(t) for ids in token_batch for t in ids]),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            stop_on_eos=bool(stop_on_eos),
            do_sample=bool(do_sample),
        )
        try:
            self._send_cmd(cmd)
        except Exception:
            with self._lock:
                for rid in request_ids:
                    self._token_counts.pop(int(rid), None)
                    self._finished.discard(int(rid))
            raise
        return request_ids

    def wait_finished(
        self,
        request_ids: Sequence[int],
        *,
        timeout_s: float | None = None,
    ) -> dict[int, int]:
        want = {int(r) for r in request_ids}
        deadline = (
            (time.perf_counter() + float(timeout_s)) if timeout_s is not None else None
        )
        with self._cond:
            while True:
                if self._error is not None:
                    raise RuntimeError(self._error)
                if want.issubset(self._finished):
                    return {rid: int(self._token_counts.get(rid, 0)) for rid in want}
                if not self._running:
                    raise RuntimeError("MPTokenManager stopped")
                if deadline is None:
                    self._cond.wait(timeout=0.1)
                    continue
                left = float(deadline - time.perf_counter())
                if left <= 0:
                    raise TimeoutError("wait_finished timed out")
                self._cond.wait(timeout=min(0.1, left))

    def start_profile(
        self,
        *,
        tool: str,
        output_dir: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        tool = str(tool).lower()
        if tool not in ("torch", "cuda"):
            raise ValueError("tool must be torch|cuda")
        if tool == "torch" and output_dir is None:
            raise ValueError("output_dir is required for torch profiler")

        waiter = threading.Event()
        with self._lock:
            if not self._running:
                raise RuntimeError("MPTokenManager is closed")
            profile_id = int(self._profile_seq)
            self._profile_seq += 1
            self._profile_waiters[profile_id] = waiter

        self._send_cmd(
            StartProfileCmd(
                profile_id=profile_id,
                tool=tool,
                output_dir=str(output_dir) if output_dir is not None else None,
            )
        )
        if not waiter.wait(timeout=timeout_s):
            raise TimeoutError("start_profile timed out")
        with self._lock:
            evt = self._profile_results.pop(profile_id, None)
        if evt is None:
            raise RuntimeError("missing profile ack event")
        if not bool(evt.ok):
            raise RuntimeError(evt.message or "start_profile failed")

    def stop_profile(
        self,
        *,
        tool: str,
        timeout_s: float = 300.0,
    ) -> None:
        tool = str(tool).lower()
        if tool not in ("torch", "cuda"):
            raise ValueError("tool must be torch|cuda")

        waiter = threading.Event()
        with self._lock:
            if not self._running:
                raise RuntimeError("MPTokenManager is closed")
            profile_id = int(self._profile_seq)
            self._profile_seq += 1
            self._profile_waiters[profile_id] = waiter

        self._send_cmd(StopProfileCmd(profile_id=profile_id, tool=tool))
        if not waiter.wait(timeout=timeout_s):
            raise TimeoutError("stop_profile timed out")
        with self._lock:
            evt = self._profile_results.pop(profile_id, None)
        if evt is None:
            raise RuntimeError("missing profile ack event")
        if not bool(evt.ok):
            raise RuntimeError(evt.message or "stop_profile failed")

    def _event_loop(self) -> None:
        try:
            while True:
                if self._ipc_mode == "queue":
                    evt_q = self._evt_q
                    if evt_q is None:
                        return
                    evt = evt_q.get()
                else:
                    evt_recv = self._evt_recv
                    if evt_recv is None:
                        return
                    try:
                        evt = _decode_evt(evt_recv.recv_bytes())
                    except EOFError:
                        return
                if evt is None:
                    return
                if isinstance(evt, ReadyEvent):
                    continue
                if isinstance(evt, ProfileEvent):
                    with self._lock:
                        waiter = self._profile_waiters.pop(int(evt.profile_id), None)
                        if waiter is not None:
                            self._profile_results[int(evt.profile_id)] = evt
                            waiter.set()
                    continue
                if isinstance(evt, ErrorEvent):
                    with self._cond:
                        self._error = str(evt.message)
                        waiters = list(self._profile_waiters.values())
                        self._profile_waiters.clear()
                        self._profile_results.clear()
                        self._running = False
                        self._cond.notify_all()
                    for ev in waiters:
                        ev.set()
                    return
                with self._cond:
                    if isinstance(evt, TokensEvent):
                        for rid, toks in evt.tokens.items():
                            self._token_counts[int(rid)] = int(
                                self._token_counts.get(int(rid), 0) + len(toks)
                            )
                        for rid in evt.finished_ids:
                            self._finished.add(int(rid))
                        self._cond.notify_all()
                        continue
                    if isinstance(evt, TokenPairsEvent):
                        for rid in evt.rids:
                            self._token_counts[int(rid)] = int(
                                self._token_counts.get(int(rid), 0) + 1
                            )
                        for rid in evt.finished_ids:
                            self._finished.add(int(rid))
                        self._cond.notify_all()
                        continue
                    if isinstance(evt, TokenCountsEvent):
                        for rid, count in zip(evt.request_ids, evt.token_counts):
                            self._token_counts[int(rid)] = int(count)
                            self._finished.add(int(rid))
                        self._cond.notify_all()
                        continue
        except Exception:
            traceback.print_exc()
            with self._cond:
                self._error = "event loop crashed"
                waiters = list(self._profile_waiters.values())
                self._profile_waiters.clear()
                self._profile_results.clear()
                self._running = False
                self._cond.notify_all()
            for ev in waiters:
                ev.set()
