import argparse
import asyncio
import json
import os
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterator, List, Literal, Optional, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .detokenizer import BaseDetokenizer
from .engine import (
    ChunkedOnlineScheduler,
    InferenceEngine,
    OnlineRequest,
    OnlineScheduler,
)
from .errors import SchedulerManagerOverloadedError
from .hybrid_queue import HybridQueue
from .mp import EngineProcessArgs, MPSchedulerManager

try:
    import orjson  # type: ignore[import-not-found]
except Exception:
    orjson = None


def _json_dumps_bytes(obj: object) -> bytes:
    if orjson is not None:
        return orjson.dumps(obj)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    stop_on_eos: bool = True
    do_sample: bool = False
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class ChatCompletionChunkResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[UsageInfo] = None


class CompletionChunkChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str] = None


class CompletionChunk(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChunkChoice]


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    ignore_eos: bool = False
    stream: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    ignore_eos: bool = False
    stream: bool = False


class StartProfileRequest(BaseModel):
    tool: Literal["torch", "cuda"] = "torch"
    output_dir: Optional[str] = None


@dataclass(frozen=True)
class _PendingRequest:
    request_id: int
    prompt: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    stop_on_eos: bool
    do_sample: bool


@dataclass(frozen=True)
class _TokenizeTask:
    request_id: int
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    stop_on_eos: bool
    do_sample: bool


class _StreamState:
    __slots__ = ("buf", "tokens_since_flush", "sent_any")

    def __init__(self) -> None:
        self.buf: list[str] = []
        self.tokens_since_flush = 0
        self.sent_any = False


PrefillAdmissionPolicy = Literal["fifo", "pack"]


def _cap_prefill_max_reqs(
    max_reqs: int,
    *,
    max_active_requests: Optional[int],
    active_unfinished: int,
) -> int:
    if max_reqs <= 0:
        raise ValueError("max_reqs must be positive")
    if active_unfinished < 0:
        raise ValueError("active_unfinished must be non-negative")
    if max_active_requests is None:
        return max_reqs
    if max_active_requests <= 0:
        raise ValueError("max_active_requests must be positive")
    slots = max_active_requests - active_unfinished
    if slots <= 0:
        return 0
    return min(max_reqs, slots)


def _take_pending_for_prefill(
    pending_buf: "deque[_PendingRequest]",
    pending_q: "queue.Queue[_PendingRequest]",
    *,
    max_reqs: int,
    max_tokens: Optional[int],
    max_context: int,
    admission_policy: PrefillAdmissionPolicy,
    lookahead: int,
    force_fifo: bool,
) -> list[_PendingRequest]:
    if max_reqs <= 0:
        raise ValueError("max_reqs must be positive")
    if max_context <= 0:
        raise ValueError("max_context must be positive")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if lookahead <= 0:
        raise ValueError("lookahead must be positive")

    if force_fifo or admission_policy == "fifo" or max_tokens is None:
        out: list[_PendingRequest] = []
        tokens_used = 0
        while len(out) < max_reqs:
            if pending_buf:
                req = pending_buf.popleft()
            else:
                try:
                    req = pending_q.get_nowait()
                except queue.Empty:
                    break

            cost = min(len(req.prompt_token_ids), max_context)
            if max_tokens is not None:
                if not out and cost > max_tokens:
                    out.append(req)
                    break
                if out and tokens_used + cost > max_tokens:
                    pending_buf.appendleft(req)
                    break

            out.append(req)
            tokens_used += cost
        return out

    # admission_policy == "pack" and max_tokens is not None.
    window: list[_PendingRequest] = []
    while len(window) < lookahead:
        if pending_buf:
            window.append(pending_buf.popleft())
            continue
        try:
            window.append(pending_q.get_nowait())
        except queue.Empty:
            break
    if not window:
        return []

    costs = [min(len(req.prompt_token_ids), max_context) for req in window]
    order = sorted(range(len(window)), key=lambda i: (costs[i], i))
    selected = [False for _ in window]
    tokens_used = 0
    selected_count = 0
    for idx in order:
        if selected_count >= max_reqs:
            break
        cost = costs[idx]
        if cost > max_tokens:
            continue
        if tokens_used + cost > max_tokens:
            continue
        selected[idx] = True
        tokens_used += cost
        selected_count += 1
    if selected_count == 0:
        selected[0] = True

    out: list[_PendingRequest] = []
    for idx, req in enumerate(window):
        if selected[idx]:
            out.append(req)
            if len(out) >= max_reqs:
                break
    for idx in range(len(window) - 1, -1, -1):
        if not selected[idx]:
            pending_buf.appendleft(window[idx])
    return out


class SchedulerManager:
    def __init__(
        self,
        engine: InferenceEngine,
        max_batch_size: int = 8,
        chunked_prefill: bool = False,
        prefill_chunk_size: int = 256,
        prefix_cache: bool = True,
        overlap_schedule: bool = True,
        prefill_max_batch_size: Optional[int] = None,
        prefill_max_tokens: Optional[int] = None,
        stream_interval: int = 1,
        record_token_timestamps: bool = False,
        tokenize_workers: int = 0,
        decode_first: bool = False,
        prefill_admission_policy: PrefillAdmissionPolicy = "fifo",
        prefill_admission_lookahead: int = 64,
        prefill_force_fifo_every: int = 0,
        max_active_requests: Optional[int] = None,
        max_inflight_requests: Optional[int] = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if prefill_max_batch_size is None:
            prefill_max_batch_size = max_batch_size
        self._prefill_max_batch_size = int(prefill_max_batch_size)
        if self._prefill_max_batch_size <= 0:
            raise ValueError("prefill_max_batch_size must be positive")
        self._prefill_max_tokens = (
            int(prefill_max_tokens) if prefill_max_tokens is not None else None
        )
        if self._prefill_max_tokens is not None and self._prefill_max_tokens <= 0:
            raise ValueError("prefill_max_tokens must be positive")
        self._stream_interval = int(stream_interval)
        if self._stream_interval <= 0:
            raise ValueError("stream_interval must be positive")
        self._tokenize_workers = int(tokenize_workers)
        if self._tokenize_workers < 0:
            raise ValueError("tokenize_workers must be non-negative")
        self._decode_first = bool(decode_first)
        if prefill_admission_policy not in ("fifo", "pack"):
            raise ValueError("prefill_admission_policy must be fifo|pack")
        self._prefill_admission_policy = prefill_admission_policy
        self._prefill_admission_lookahead = int(prefill_admission_lookahead)
        if self._prefill_admission_lookahead <= 0:
            raise ValueError("prefill_admission_lookahead must be positive")
        self._prefill_force_fifo_every = int(prefill_force_fifo_every)
        if self._prefill_force_fifo_every < 0:
            raise ValueError("prefill_force_fifo_every must be non-negative")
        self._prefill_iter = 0
        self._max_active_requests = (
            int(max_active_requests) if max_active_requests is not None else None
        )
        if self._max_active_requests is not None and self._max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive")
        self._max_inflight_requests = (
            int(max_inflight_requests) if max_inflight_requests is not None else None
        )
        if self._max_inflight_requests is not None and self._max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be positive")

        self.engine = engine
        if bool(chunked_prefill):
            self.scheduler = ChunkedOnlineScheduler(
                engine,
                max_batch_size=int(max_batch_size),
                prefill_chunk_size=int(prefill_chunk_size),
                prefill_max_batch_size=int(prefill_max_batch_size),
                use_prefix_cache=bool(prefix_cache),
                overlap_schedule=bool(overlap_schedule),
            )
        else:
            self.scheduler = OnlineScheduler(
                engine,
                max_batch_size=int(max_batch_size),
                use_prefix_cache=bool(prefix_cache),
                overlap_schedule=bool(overlap_schedule),
            )
        self.engine.warmup_paged_attention_decode()
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._queues: Dict[int, "HybridQueue[Optional[str]]"] = {}
        self._detoks: Dict[int, BaseDetokenizer] = {}
        self._stream_states: Dict[int, _StreamState] = {}
        self._record_token_timestamps = bool(record_token_timestamps)
        self._token_timestamps: Dict[int, list[float]] = {}
        self._admit_timestamps: Dict[int, float] = {}
        self._tokenize_timestamps: Dict[int, float] = {}
        self._pending: "queue.Queue[_PendingRequest]" = queue.Queue()
        self._pending_buf: "deque[_PendingRequest]" = deque()
        self._tokenize_q: "queue.Queue[_TokenizeTask | None]" | None = (
            queue.Queue() if self._tokenize_workers > 0 else None
        )
        self._tokenize_threads: list[threading.Thread] = []
        self._next_request_id: int = 0
        self._running = True
        self._torch_prof: object | None = None
        self._torch_prof_dir: str | None = None
        self._cuda_prof_active = False
        if self._tokenize_q is not None:
            for i in range(self._tokenize_workers):
                th = threading.Thread(
                    target=self._tokenize_loop,
                    name=f"roseinfer-tokenize-{i}",
                    daemon=True,
                )
                self._tokenize_threads.append(th)
                th.start()
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self._worker.start()

    def set_asyncio_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._async_loop = loop
        with self._lock:
            queues = list(self._queues.values())
        for q in queues:
            q.set_asyncio_loop(loop)

    def close(self) -> None:
        worker = self._worker
        tok_q: "queue.Queue[_TokenizeTask | None] | None" = None
        tok_threads: list[threading.Thread] = []
        request_ids: list[int] = []
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._wakeup.set()
            tok_q = self._tokenize_q
            tok_threads = list(self._tokenize_threads)
            request_ids = list(self._queues.keys())
            for rid in request_ids:
                q = self._queues.get(rid)
                if q is not None:
                    q.put(None)
            self._queues.clear()
            self._detoks.clear()
            self._stream_states.clear()
            self._token_timestamps.clear()
            self._admit_timestamps.clear()
            self._tokenize_timestamps.clear()
            self._pending_buf.clear()
            self._tokenize_q = None
            self._tokenize_threads.clear()
        if tok_q is not None:
            for _ in tok_threads:
                tok_q.put(None)
            for th in tok_threads:
                th.join(timeout=1.0)
        worker.join(timeout=1.0)
        if worker.is_alive():
            return
        for rid in request_ids:
            self.scheduler.discard_request(rid)

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
        with self._lock:
            if not self._running:
                raise RuntimeError("SchedulerManager is closed")
            self._profile_apply("start", tool, output_dir)

    def stop_profile(
        self,
        *,
        tool: str,
        timeout_s: float = 300.0,
    ) -> None:
        tool = str(tool).lower()
        if tool not in ("torch", "cuda"):
            raise ValueError("tool must be torch|cuda")
        with self._lock:
            if not self._running:
                raise RuntimeError("SchedulerManager is closed")
            self._profile_apply("stop", tool, None)

    def _profile_apply(self, action: str, tool: str, output_dir: str | None) -> None:
        import torch

        tool = str(tool).lower()
        if action == "start":
            if tool == "torch":
                if self._torch_prof is not None:
                    raise RuntimeError("torch profiler already running")
                if output_dir is None:
                    raise ValueError("output_dir is required for torch profiler")
                os.makedirs(str(output_dir), exist_ok=True)
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
                self._torch_prof = prof
                self._torch_prof_dir = str(output_dir)
                return
            if tool == "cuda":
                if self._cuda_prof_active:
                    raise RuntimeError("cuda profiler already running")
                if torch.cuda.is_available():
                    torch.cuda.profiler.start()
                self._cuda_prof_active = True
                return
            raise ValueError("tool must be torch|cuda")

        if action == "stop":
            if tool == "torch":
                if self._torch_prof is None or self._torch_prof_dir is None:
                    raise RuntimeError("torch profiler not running")
                prof = self._torch_prof
                out_dir = self._torch_prof_dir
                prof.__exit__(None, None, None)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof.export_chrome_trace(os.path.join(out_dir, "trace.json"))
                self._torch_prof = None
                self._torch_prof_dir = None
                return
            if tool == "cuda":
                if not self._cuda_prof_active:
                    raise RuntimeError("cuda profiler not running")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
                self._cuda_prof_active = False
                return
            raise ValueError("tool must be torch|cuda")

        raise ValueError("action must be start|stop")

    def _torch_prof_step(self) -> None:
        with self._lock:
            prof = self._torch_prof
        if prof is None:
            return
        step_fn = getattr(prof, "step", None)
        if not callable(step_fn):
            return
        try:
            step_fn()
        except Exception:
            traceback.print_exc()

    def add_request(
        self,
        prompt: str,
        prompt_token_ids: Optional[list[int]] = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        do_sample: bool = False,
    ) -> int:
        with self._lock:
            if not self._running:
                raise RuntimeError("SchedulerManager is closed")
            if self._max_inflight_requests is not None and (
                len(self._queues) >= self._max_inflight_requests
            ):
                raise SchedulerManagerOverloadedError("too many inflight requests")
            detok = self.engine._make_detok()
            request_id = self._next_request_id
            self._next_request_id += 1
            q: "HybridQueue[Optional[str]]" = HybridQueue()
            loop = self._async_loop
            if loop is not None:
                q.set_asyncio_loop(loop)
            self._queues[request_id] = q
            self._detoks[request_id] = detok
            self._stream_states[request_id] = _StreamState()
            if self._record_token_timestamps:
                self._token_timestamps[request_id] = []

        tok_q = self._tokenize_q
        if prompt_token_ids is None and tok_q is not None:
            tok_q.put(
                _TokenizeTask(
                    request_id=request_id,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    stop_on_eos=stop_on_eos,
                    do_sample=do_sample,
                )
            )
            self._wakeup.set()
            return request_id

        if prompt_token_ids is None:
            token_ids = self.engine.tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
        else:
            token_ids = list(prompt_token_ids)
        if not token_ids:
            token_ids = [self.engine.eos_token_id]
        max_pos = int(self.engine.config.max_position_embeddings)
        if len(token_ids) > max_pos:
            token_ids = token_ids[-max_pos:]
        detok.start_prompt(token_ids)
        if self._record_token_timestamps:
            with self._lock:
                self._tokenize_timestamps[request_id] = time.perf_counter()
        self._pending.put(
            _PendingRequest(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=list(token_ids),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_on_eos=stop_on_eos,
                do_sample=do_sample,
            )
        )
        self._wakeup.set()
        return request_id

    def pop_token_timestamps(
        self,
        request_id: int,
    ) -> list[float]:
        with self._lock:
            out = self._token_timestamps.pop(request_id, None)
        return list(out) if out is not None else []

    def pop_admit_timestamp(
        self,
        request_id: int,
    ) -> float | None:
        with self._lock:
            out = self._admit_timestamps.pop(request_id, None)
        return float(out) if out is not None else None

    def pop_tokenize_timestamp(
        self,
        request_id: int,
    ) -> float | None:
        with self._lock:
            out = self._tokenize_timestamps.pop(request_id, None)
        return float(out) if out is not None else None

    def _tokenize_loop(self) -> None:
        tok_q = self._tokenize_q
        if tok_q is None:
            return
        try:
            while True:
                task = tok_q.get()
                if task is None:
                    return
                with self._lock:
                    if not self._running:
                        return
                    detok = self._detoks.get(task.request_id)
                    out_q = self._queues.get(task.request_id)
                if detok is None or out_q is None:
                    continue

                try:
                    token_ids = self.engine.tokenizer.encode(
                        task.prompt,
                        add_special_tokens=False,
                    )
                    if not token_ids:
                        token_ids = [self.engine.eos_token_id]
                    max_pos = int(self.engine.config.max_position_embeddings)
                    if len(token_ids) > max_pos:
                        token_ids = token_ids[-max_pos:]
                    detok.start_prompt(token_ids)
                    if self._record_token_timestamps:
                        with self._lock:
                            self._tokenize_timestamps[
                                task.request_id
                            ] = time.perf_counter()
                    self._pending.put(
                        _PendingRequest(
                            request_id=task.request_id,
                            prompt=task.prompt,
                            prompt_token_ids=list(token_ids),
                            max_new_tokens=task.max_new_tokens,
                            temperature=task.temperature,
                            top_k=task.top_k,
                            top_p=task.top_p,
                            stop_on_eos=task.stop_on_eos,
                            do_sample=task.do_sample,
                        )
                    )
                    self._wakeup.set()
                except Exception:
                    traceback.print_exc()
                    out_q.put(None)
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

    def _worker_loop(self) -> None:
        try:
            while True:

                def run_decode_once() -> None:
                    if not self.scheduler.has_unfinished():
                        return
                    step_tokens = self.scheduler.step()
                    self._torch_prof_step()
                    finished_ids = self.scheduler.pop_finished_ids()

                    step_records: list[
                        tuple[
                            int,
                            int,
                            "HybridQueue[Optional[str]] | None",
                            BaseDetokenizer | None,
                            _StreamState | None,
                            list[float] | None,
                        ]
                    ] = []
                    finished_records: list[
                        tuple[
                            int,
                            "HybridQueue[Optional[str]] | None",
                            BaseDetokenizer | None,
                            _StreamState | None,
                        ]
                    ] = []
                    with self._lock:
                        for rid, token_id in step_tokens.items():
                            step_records.append(
                                (
                                    rid,
                                    int(token_id),
                                    self._queues.get(rid),
                                    self._detoks.get(rid),
                                    self._stream_states.get(rid),
                                    (
                                        self._token_timestamps.get(rid)
                                        if self._record_token_timestamps
                                        else None
                                    ),
                                )
                            )
                        for rid in finished_ids:
                            finished_records.append(
                                (
                                    int(rid),
                                    self._queues.get(rid),
                                    self._detoks.get(rid),
                                    self._stream_states.get(rid),
                                )
                            )

                    for rid, token_id, q, detok, state, token_ts in step_records:
                        if q is None or detok is None or state is None:
                            self.scheduler.discard_request(rid)
                            continue
                        if token_ts is not None:
                            token_ts.append(time.perf_counter())
                        piece = detok.on_token(token_id)
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
                        self.scheduler.discard_request(rid)
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

                with self._lock:
                    if not self._running:
                        break
                    max_new = self._prefill_max_batch_size
                    max_tokens = self._prefill_max_tokens
                    max_context = int(self.engine.config.max_position_embeddings)
                    decode_first = self._decode_first
                    admission_policy = self._prefill_admission_policy
                    lookahead = self._prefill_admission_lookahead
                    force_fifo_every = self._prefill_force_fifo_every
                    max_active = self._max_active_requests

                self._prefill_iter += 1
                force_fifo = force_fifo_every > 0 and (
                    self._prefill_iter % force_fifo_every == 0
                )

                did_decode = False
                if decode_first and self.scheduler.has_unfinished():
                    run_decode_once()
                    did_decode = True

                admit_cap = _cap_prefill_max_reqs(
                    max_new,
                    max_active_requests=max_active,
                    active_unfinished=self.scheduler.num_unfinished(),
                )
                if admit_cap > 0:
                    pending = _take_pending_for_prefill(
                        self._pending_buf,
                        self._pending,
                        max_reqs=admit_cap,
                        max_tokens=max_tokens,
                        max_context=max_context,
                        admission_policy=admission_policy,
                        lookahead=lookahead,
                        force_fifo=force_fifo,
                    )
                else:
                    pending = []
                batch: list[OnlineRequest] = []
                for req in pending:
                    with self._lock:
                        if not self._running:
                            break
                        q = self._queues.get(req.request_id)
                        detok = self._detoks.get(req.request_id)
                    if q is None or detok is None:
                        continue
                    batch.append(
                        OnlineRequest(
                            prompt=req.prompt,
                            max_new_tokens=req.max_new_tokens,
                            temperature=req.temperature,
                            top_k=req.top_k,
                            top_p=req.top_p,
                            stop_on_eos=req.stop_on_eos,
                            do_sample=req.do_sample,
                            prompt_token_ids=req.prompt_token_ids,
                            request_id=req.request_id,
                        )
                    )
                admit_ts = (
                    time.perf_counter()
                    if (batch and self._record_token_timestamps)
                    else 0.0
                )
                rids = self.scheduler.add_requests(batch) if batch else []
                if batch:
                    self._torch_prof_step()
                if rids and self._record_token_timestamps:
                    with self._lock:
                        for rid in rids:
                            self._admit_timestamps[rid] = admit_ts
                for rid in rids:
                    with self._lock:
                        q = self._queues.get(rid)
                        detok = self._detoks.get(rid)
                        state = self._stream_states.get(rid)
                        token_ts = (
                            self._token_timestamps.get(rid)
                            if self._record_token_timestamps
                            else None
                        )
                    if q is None or detok is None or state is None:
                        self.scheduler.discard_request(rid)
                        continue
                    for tid in self.scheduler.get_generated_ids(rid):
                        if token_ts is not None:
                            token_ts.append(time.perf_counter())
                        piece = detok.on_token(int(tid))
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
                    if self.scheduler.is_finished(rid):
                        tail = detok.flush()
                        if tail:
                            state.buf.append(tail)
                        if state.buf:
                            q.put("".join(state.buf))
                            state.buf.clear()
                            state.tokens_since_flush = 0
                            state.sent_any = True
                        q.put(None)
                        self.scheduler.discard_request(rid)

                if not did_decode:
                    run_decode_once()

                if (
                    not pending
                    and not self._pending_buf
                    and not self.scheduler.has_unfinished()
                ):
                    self._wakeup.wait()
                    self._wakeup.clear()
        except Exception:
            traceback.print_exc()
            with self._lock:
                queues = list(self._queues.values())
                self._running = False
                self._wakeup.set()
            for q in queues:
                q.put(None)


def format_messages_as_prompt(messages: List[ChatMessage]) -> str:
    lines: list[str] = []
    for m in messages:
        if m.role == "system":
            lines.append(f"[system]\n{m.content}\n")
        elif m.role == "user":
            lines.append(f"User: \n{m.content}\n")
        elif m.role == "assistant":
            lines.append(f"Assistant: \n{m.content}\n")
    lines.append("Assistant:")
    return "".join(lines)


def estimate_usage(
    tokenizer,
    prompt: str,
    completion: str,
) -> UsageInfo:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    prompt_tokens = len(prompt_ids)
    completion_tokens = len(completion_ids)
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def create_app(
    tokenizer,
    sched_manager: "SchedulerManager | MPSchedulerManager",
    *,
    served_model_name: str | None = None,
    async_streaming: bool = True,
    fast_sse: bool = True,
) -> FastAPI:
    app = FastAPI(title="roseinfer", version="0.1.0")
    app.add_event_handler("shutdown", sched_manager.close)

    async def _startup() -> None:
        sched_manager.set_asyncio_loop(asyncio.get_running_loop())

    app.add_event_handler("startup", _startup)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict:
        model_id = served_model_name or "roseinfer"
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "rosellm",
                }
            ],
        }

    @app.post("/start_profile")
    async def start_profile(body: StartProfileRequest) -> dict[str, str]:
        tool = str(body.tool).lower()
        try:
            if tool == "torch":
                if body.output_dir is None:
                    raise ValueError("output_dir is required for torch profiler")
                sched_manager.start_profile(
                    tool="torch", output_dir=str(body.output_dir)
                )
            elif tool == "cuda":
                sched_manager.start_profile(tool="cuda")
            else:
                raise ValueError("tool must be torch|cuda")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.post("/stop_profile")
    async def stop_profile() -> dict[str, str]:
        torch_err: str | None = None
        try:
            sched_manager.stop_profile(tool="torch")
            return {"status": "ok"}
        except Exception as exc:
            torch_err = str(exc)

        try:
            sched_manager.stop_profile(tool="cuda")
            return {"status": "ok"}
        except Exception as exc:
            cuda_err = str(exc)

        raise HTTPException(
            status_code=400,
            detail=f"torch: {torch_err or 'unknown'}; cuda: {cuda_err or 'unknown'}",
        )

    @app.post("/generate", response_model=GenerateResponse)
    def generate(
        body: GenerateRequest,
    ) -> GenerateResponse | StreamingResponse:
        if body.stream:
            try:
                request_id = sched_manager.add_request(
                    prompt=body.prompt,
                    max_new_tokens=body.max_new_tokens,
                    temperature=body.temperature,
                    top_k=body.top_k,
                    top_p=body.top_p,
                    stop_on_eos=body.stop_on_eos,
                    do_sample=body.do_sample,
                )
            except SchedulerManagerOverloadedError as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc

            if async_streaming:

                async def token_stream() -> AsyncIterator[bytes]:
                    async for piece in sched_manager.astream_text(request_id):
                        yield piece.encode("utf-8")

            else:

                def token_stream() -> Iterator[bytes]:
                    for piece in sched_manager.stream_text(request_id):
                        yield piece.encode("utf-8")

            return StreamingResponse(
                token_stream(),
                media_type="text/plain; charset=utf-8",
            )
        try:
            request_id = sched_manager.add_request(
                prompt=body.prompt,
                max_new_tokens=body.max_new_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                stop_on_eos=True,
                do_sample=True,
            )
        except SchedulerManagerOverloadedError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        completion = "".join(sched_manager.stream_text(request_id))
        return GenerateResponse(text=body.prompt + completion)

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(
        body: ChatCompletionRequest,
    ) -> ChatCompletionResponse | StreamingResponse:
        prompt = format_messages_as_prompt(body.messages)
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = body.model or "roseinfer"
        stop_on_eos = not bool(body.ignore_eos)
        if body.stream:
            try:
                request_id = sched_manager.add_request(
                    prompt=prompt,
                    max_new_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_k=body.top_k,
                    top_p=body.top_p,
                    stop_on_eos=stop_on_eos,
                    do_sample=bool(body.do_sample),
                )
            except SchedulerManagerOverloadedError as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc

            if async_streaming:

                async def event_stream() -> AsyncIterator[bytes]:
                    first_chunk = ChatCompletionChunk(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(
                                    role="assistant",
                                    content="",
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json()}\n\n".encode("utf-8")
                    async for piece in sched_manager.astream_text(request_id):
                        if not piece:
                            continue
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            object="chat.completion.chunk",
                            created=created,
                            model=model_name,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(
                                        role=None,
                                        content=piece,
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")
                    final_chunk = ChatCompletionChunk(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(
                                    role=None,
                                    content=None,
                                ),
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"

            else:

                def event_stream() -> Iterator[bytes]:
                    first_chunk = ChatCompletionChunk(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(
                                    role="assistant",
                                    content="",
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json()}\n\n".encode("utf-8")
                    for piece in sched_manager.stream_text(request_id):
                        if not piece:
                            continue
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            object="chat.completion.chunk",
                            created=created,
                            model=model_name,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(
                                        role=None,
                                        content=piece,
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")
                    final_chunk = ChatCompletionChunk(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(
                                    role=None,
                                    content=None,
                                ),
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
            )
        try:
            request_id = sched_manager.add_request(
                prompt=prompt,
                max_new_tokens=body.max_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                stop_on_eos=stop_on_eos,
                do_sample=bool(body.do_sample),
            )
        except SchedulerManagerOverloadedError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        completion = "".join(sched_manager.stream_text(request_id))
        usage = estimate_usage(tokenizer, prompt, completion)
        resp = ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=created,
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )
        return resp

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(
        body: CompletionRequest,
    ) -> CompletionResponse | StreamingResponse:
        created = int(time.time())
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        model_name = body.model or "roseinfer"
        stop_on_eos = not bool(body.ignore_eos)
        if body.stream:
            try:
                request_id = sched_manager.add_request(
                    prompt=body.prompt,
                    max_new_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_k=body.top_k,
                    top_p=body.top_p,
                    stop_on_eos=stop_on_eos,
                    do_sample=bool(body.do_sample),
                )
            except SchedulerManagerOverloadedError as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc

            if fast_sse:
                created_b = str(int(created)).encode("utf-8")
                completion_id_json = _json_dumps_bytes(completion_id)
                model_name_json = _json_dumps_bytes(model_name)
                sse_prefix = (
                    b'data: {"id":'
                    + completion_id_json
                    + b',"object":"text_completion","created":'
                    + created_b
                    + b',"model":'
                    + model_name_json
                    + b',"choices":[{"text":'
                )
                sse_mid = b',"index":0,"finish_reason":null}]}' + b"\n\n"
                sse_final = b',"index":0,"finish_reason":"stop"}]}' + b"\n\n"

                if async_streaming:

                    async def event_stream() -> AsyncIterator[bytes]:
                        async for piece in sched_manager.astream_text(request_id):
                            if not piece:
                                continue
                            yield sse_prefix + _json_dumps_bytes(piece) + sse_mid
                        yield sse_prefix + _json_dumps_bytes("") + sse_final
                        yield b"data: [DONE]\n\n"

                else:

                    def event_stream() -> Iterator[bytes]:
                        for piece in sched_manager.stream_text(request_id):
                            if not piece:
                                continue
                            yield sse_prefix + _json_dumps_bytes(piece) + sse_mid
                        yield sse_prefix + _json_dumps_bytes("") + sse_final
                        yield b"data: [DONE]\n\n"

            elif async_streaming:

                async def event_stream() -> AsyncIterator[bytes]:
                    async for piece in sched_manager.astream_text(request_id):
                        if not piece:
                            continue
                        chunk = CompletionChunk(
                            id=completion_id,
                            object="text_completion",
                            created=created,
                            model=model_name,
                            choices=[
                                CompletionChunkChoice(
                                    index=0,
                                    text=piece,
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")
                    final_chunk = CompletionChunk(
                        id=completion_id,
                        object="text_completion",
                        created=created,
                        model=model_name,
                        choices=[
                            CompletionChunkChoice(
                                index=0,
                                text="",
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"

            else:

                def event_stream() -> Iterator[bytes]:
                    for piece in sched_manager.stream_text(request_id):
                        if not piece:
                            continue
                        chunk = CompletionChunk(
                            id=completion_id,
                            object="text_completion",
                            created=created,
                            model=model_name,
                            choices=[
                                CompletionChunkChoice(
                                    index=0,
                                    text=piece,
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")
                    final_chunk = CompletionChunk(
                        id=completion_id,
                        object="text_completion",
                        created=created,
                        model=model_name,
                        choices=[
                            CompletionChunkChoice(
                                index=0,
                                text="",
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
            )
        try:
            request_id = sched_manager.add_request(
                prompt=body.prompt,
                max_new_tokens=body.max_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                stop_on_eos=stop_on_eos,
                do_sample=bool(body.do_sample),
            )
        except SchedulerManagerOverloadedError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        completion_text = "".join(sched_manager.stream_text(request_id))
        usage = estimate_usage(tokenizer, body.prompt, completion_text)
        return CompletionResponse(
            id=completion_id,
            object="text_completion",
            created=created,
            model=model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    text=completion_text,
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the inference server",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Tokenizer name",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=None,
        help=(
            "Load model weights from Hugging Face (GPT-2 only). "
            "If set, --checkpoint-path/--tokenizer-name become optional."
        ),
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
        "--prefill-attn-backend",
        type=str,
        default="auto",
        choices=["auto", "auto2", "naive", "flashinfer", "flashattn"],
        help="Prefill attention backend (default: auto).",
    )
    parser.add_argument(
        "--decode-attn-backend",
        type=str,
        default="auto",
        choices=["auto", "naive", "flashinfer", "flashattn"],
        help="Decode attention backend for dense past_kv path (default: auto).",
    )
    parser.add_argument(
        "--paged-attn",
        dest="paged_attn",
        action="store_true",
        help="Use paged attention for decode(T=1).",
    )
    parser.add_argument(
        "--no-paged-attn",
        dest="paged_attn",
        action="store_false",
        help="Disable paged attention.",
    )
    parser.set_defaults(paged_attn=True)
    parser.add_argument(
        "--cuda-graph",
        dest="cuda_graph",
        action="store_true",
        help="Use CUDA graphs for paged attention decode(T=1).",
    )
    parser.add_argument(
        "--no-cuda-graph",
        dest="cuda_graph",
        action="store_false",
        help="Disable CUDA graphs.",
    )
    parser.set_defaults(cuda_graph=True)
    parser.add_argument(
        "--chunked-prefill",
        dest="chunked_prefill",
        action="store_true",
        help="Enable chunked prefill (incremental prompt ingestion).",
    )
    parser.add_argument(
        "--no-chunked-prefill",
        dest="chunked_prefill",
        action="store_false",
        help="Disable chunked prefill.",
    )
    parser.set_defaults(chunked_prefill=True)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=256,
        help="Max tokens per prefill chunk when --chunked-prefill is set.",
    )
    parser.add_argument(
        "--stop-on-eos",
        dest="stop_on_eos",
        action="store_true",
        help="Stop on EOS token",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling to generate text (or else greedy)",
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
        "--async-streaming",
        dest="async_streaming",
        action="store_true",
        help="Use async token/SSE streaming to avoid one thread per stream (default: enabled).",
    )
    parser.add_argument(
        "--no-async-streaming",
        dest="async_streaming",
        action="store_false",
        help="Disable async token/SSE streaming (falls back to sync generators).",
    )
    parser.set_defaults(async_streaming=True)
    parser.add_argument(
        "--max-inflight-requests",
        type=int,
        default=None,
        help="Max inflight requests accepted by SchedulerManager (default: unlimited).",
    )
    parser.add_argument(
        "--kv-cache-max-concurrency",
        type=int,
        default=0,
        help=(
            "Max concurrent sessions for KV cache allocation; 0 selects an automatic value "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--prefix-cache-max-entries",
        type=int,
        default=256,
        help="Max entries in prefix cache (default: 256).",
    )
    parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Flush streaming output every N generated tokens (default: 1).",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Max batch size for online scheduler decode/prefill (default: 8).",
    )
    parser.add_argument(
        "--gc-freeze",
        dest="gc_freeze",
        action="store_true",
        help="Freeze Python GC heap after startup to reduce GC jitter (default: enabled).",
    )
    parser.add_argument(
        "--no-gc-freeze",
        dest="gc_freeze",
        action="store_false",
        help="Disable GC freeze (may increase tail latency due to GC jitter).",
    )
    parser.set_defaults(gc_freeze=True)
    parser.add_argument(
        "--gc-warn-ms",
        type=float,
        default=0.0,
        help="Log GC pauses >= this threshold in ms (default: 0 = disabled).",
    )
    parser.add_argument(
        "--gc-log-all",
        action="store_true",
        help="Log every GC cycle (debug; may be noisy).",
    )
    parser.add_argument(
        "--fast-sse",
        dest="fast_sse",
        action="store_true",
        help="Use a faster JSON builder for OpenAI SSE streaming (default: enabled).",
    )
    parser.add_argument(
        "--no-fast-sse",
        dest="fast_sse",
        action="store_false",
        help="Disable fast SSE JSON builder (may increase CPU overhead in streaming).",
    )
    parser.set_defaults(fast_sse=True)
    parser.add_argument(
        "--prefix-cache",
        dest="prefix_cache",
        action="store_true",
        help="Enable prefix caching (default: enabled).",
    )
    parser.add_argument(
        "--no-prefix-cache",
        dest="prefix_cache",
        action="store_false",
        help="Disable prefix caching.",
    )
    parser.set_defaults(prefix_cache=True)
    parser.add_argument(
        "--overlap-schedule",
        dest="overlap_schedule",
        action="store_true",
        help="Enable CPU/GPU overlap scheduling (default: enabled).",
    )
    parser.add_argument(
        "--no-overlap-schedule",
        dest="overlap_schedule",
        action="store_false",
        help="Disable CPU/GPU overlap scheduling.",
    )
    parser.set_defaults(overlap_schedule=True)
    parser.add_argument(
        "--fused-ops",
        dest="fused_ops",
        action="store_true",
        help="Enable fused inference ops (e.g., add+LayerNorm) (default: enabled).",
    )
    parser.add_argument(
        "--no-fused-ops",
        dest="fused_ops",
        action="store_false",
        help="Disable fused inference ops.",
    )
    parser.set_defaults(fused_ops=True)
    parser.add_argument(
        "--fused-mlp",
        dest="fused_mlp",
        action="store_true",
        help="Enable fused MLP epilogue (bias+GELU/residual) (default: enabled).",
    )
    parser.add_argument(
        "--no-fused-mlp",
        dest="fused_mlp",
        action="store_false",
        help="Disable fused MLP epilogue.",
    )
    parser.set_defaults(fused_mlp=True)
    parser.add_argument(
        "--fused-sampler",
        dest="fused_sampler",
        action="store_true",
        help="Enable fused GPU sampler (default: enabled).",
    )
    parser.add_argument(
        "--no-fused-sampler",
        dest="fused_sampler",
        action="store_false",
        help="Disable fused sampler (falls back to torch ops).",
    )
    parser.set_defaults(fused_sampler=True)
    parser.add_argument(
        "--fused-kv-append",
        dest="fused_kv_append",
        action="store_true",
        help="Fuse KV append into paged decode kernel (default: enabled).",
    )
    parser.add_argument(
        "--no-fused-kv-append",
        dest="fused_kv_append",
        action="store_false",
        help="Disable fused KV append (uses separate append kernel).",
    )
    parser.set_defaults(fused_kv_append=True)
    parser.add_argument(
        "--engine-process",
        dest="engine_process",
        action="store_true",
        help="Run engine/scheduler in a dedicated worker process (default: enabled).",
    )
    parser.add_argument(
        "--no-engine-process",
        dest="engine_process",
        action="store_false",
        help="Disable engine worker process; run everything in one process.",
    )
    parser.set_defaults(engine_process=True)
    parser.add_argument(
        "--mp-ipc",
        type=str,
        default="pipe",
        help="IPC transport between API and engine process: queue|pipe (default: pipe).",
    )
    parser.add_argument(
        "--mp-max-recv-per-iter",
        type=int,
        default=64,
        help=(
            "Max number of commands to drain per engine loop iteration when busy; "
            "0 disables the budget (default: 64)."
        ),
    )
    parser.add_argument(
        "--mp-fill-target",
        dest="mp_fill_target",
        action="store_true",
        help=(
            "When the engine is ramping up below the target concurrency, temporarily "
            "bypass the cmd drain budget to fill to target (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-mp-fill-target",
        dest="mp_fill_target",
        action="store_false",
        help="Disable ramp-up fill-to-target behavior for multiprocess mode.",
    )
    parser.set_defaults(mp_fill_target=True)
    parser.add_argument(
        "--mp-flat-events",
        dest="mp_flat_events",
        action="store_true",
        help="Use flat token-pair events in engine IPC (default: enabled).",
    )
    parser.add_argument(
        "--no-mp-flat-events",
        dest="mp_flat_events",
        action="store_false",
        help="Disable flat token-pair events (use legacy dict-of-lists IPC).",
    )
    parser.set_defaults(mp_flat_events=True)
    parser.add_argument(
        "--mp-thread-cap",
        dest="mp_thread_cap",
        action="store_true",
        help="Cap torch intra/inter-op threads to 1 in API/engine processes (default: enabled for CUDA).",
    )
    parser.add_argument(
        "--no-mp-thread-cap",
        dest="mp_thread_cap",
        action="store_false",
        help="Disable torch thread capping for multiprocess mode.",
    )
    parser.set_defaults(mp_thread_cap=True)
    parser.add_argument(
        "--mp-affinity",
        dest="mp_affinity",
        action="store_true",
        help="Split CPU affinity between API and engine processes (default: enabled).",
    )
    parser.add_argument(
        "--no-mp-affinity",
        dest="mp_affinity",
        action="store_false",
        help="Disable CPU affinity splitting for multiprocess mode.",
    )
    parser.set_defaults(mp_affinity=True)
    parser.add_argument(
        "--mp-api-cpus",
        type=str,
        default=None,
        help="CPU set for API process, e.g. '0-7,16-23' (default: auto-split).",
    )
    parser.add_argument(
        "--mp-engine-cpus",
        type=str,
        default=None,
        help="CPU set for engine process, e.g. '8-15,24-31' (default: auto-split).",
    )
    parser.add_argument(
        "--mp-async-admit",
        dest="mp_async_admit",
        action="store_true",
        help=(
            "Offload prompt tokenization + engine command submission to background threads "
            "in the API process (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-mp-async-admit",
        dest="mp_async_admit",
        action="store_false",
        help="Disable async admit threads for multiprocess mode.",
    )
    parser.set_defaults(mp_async_admit=True)
    parser.add_argument(
        "--mp-tokenize-workers",
        type=int,
        default=4,
        help=(
            "Tokenization worker threads in the API process when --mp-async-admit is set "
            "(default: 4)."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to listen on",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on",
    )
    args = parser.parse_args()
    if args.hf_model_id is None:
        if args.checkpoint_path is None:
            parser.error("--checkpoint-path is required when --hf-model-id is not set")
        if args.tokenizer_name is None:
            parser.error("--tokenizer-name is required when --hf-model-id is not set")
    else:
        if args.tokenizer_name is None:
            args.tokenizer_name = args.hf_model_id
    return args


def _parse_cpu_set(spec: str) -> list[int]:
    cpus: set[int] = set()
    for part in str(spec).split(","):
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


def _cpu_core_groups(cpus: Sequence[int]) -> list[list[int]]:
    """Group logical CPUs by physical core (thread siblings).

    This helps avoid binding two processes to different hyperthreads of the same
    core, which can regress latency/throughput under load.
    """

    cpus = sorted(int(c) for c in cpus)
    available = set(cpus)
    groups: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for cpu in cpus:
        sibs: list[int]
        try:
            path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            with open(path, "r", encoding="utf-8") as f:
                sibs = _parse_cpu_set(f.read().strip())
        except Exception:
            sibs = [cpu]
        sibs = [c for c in sibs if c in available]
        if not sibs:
            sibs = [cpu]
        key = tuple(sorted(sibs))
        if key in seen:
            continue
        seen.add(key)
        groups.append(list(key))
    groups.sort(key=lambda g: min(g))
    return groups


def _auto_split_cpu_affinity(cpus: Sequence[int]) -> tuple[list[int], list[int]]:
    cpus = list(cpus)
    if not cpus:
        return [], []
    if len(cpus) <= 2:
        return cpus[:1], (cpus[1:] or cpus[:1])
    if len(cpus) <= 8:
        # On small CPU counts it is easy to starve either side; split roughly evenly.
        engine_ct = max(1, len(cpus) // 2)
    elif len(cpus) <= 16:
        # Still small enough that the engine can be CPU-sensitive (IPC + scheduler).
        engine_ct = max(4, len(cpus) // 2)
    else:
        # The engine process is GPU-bound, but still needs enough CPU for:
        # - scheduler bookkeeping
        # - IPC encode/decode
        # - sampling + KV index math
        # Give it a meaningful slice so MP doesn't regress vs in-proc.
        engine_ct = max(6, min(16, len(cpus) // 3))
        # Keep at least 2 CPUs for API-side tokenize/detokenize + HTTP/SSE.
        engine_ct = min(engine_ct, max(1, len(cpus) - 2))
    if len(cpus) - engine_ct <= 0:
        engine_ct = max(1, len(cpus) - 1)

    groups = _cpu_core_groups(cpus)
    smt_groups = [g for g in groups if len(g) > 1]
    solo_groups = [g for g in groups if len(g) == 1]

    engine_set: set[int] = set()
    # Prefer SMT cores (often "P-cores" on hybrid CPUs) for the engine process.
    for g in reversed(smt_groups):
        if len(engine_set) >= engine_ct:
            break
        engine_set.update(g)
    for g in reversed(solo_groups):
        if len(engine_set) >= engine_ct:
            break
        engine_set.update(g)

    engine = sorted(engine_set) or cpus[:1]
    api = [c for c in cpus if c not in set(engine)]
    if not api:
        api = engine[:1]
    return api, engine


def _resolve_mp_cpu_sets(
    *,
    mp_affinity: bool,
    api_spec: str | None,
    engine_spec: str | None,
) -> tuple[list[int] | None, list[int] | None]:
    if not bool(mp_affinity):
        return None, None
    available = sorted(os.sched_getaffinity(0))
    if api_spec or engine_spec:
        api = _parse_cpu_set(api_spec) if api_spec else []
        engine = _parse_cpu_set(engine_spec) if engine_spec else []
        if api and engine:
            overlap = set(api).intersection(engine)
            if overlap:
                raise ValueError(f"mp cpu sets overlap: {sorted(overlap)}")
            return api, engine
        if api and not engine:
            rest = [c for c in available if c not in set(api)]
            return api, rest or api[:1]
        if engine and not api:
            rest = [c for c in available if c not in set(engine)]
            return rest or engine[:1], engine
        return None, None
    return _auto_split_cpu_affinity(available)


def main() -> None:
    import torch
    import uvicorn

    args = parse_args()
    if args.max_inflight_requests is not None and int(args.max_inflight_requests) <= 0:
        raise ValueError("--max-inflight-requests must be >= 1")
    if int(args.stream_interval) <= 0:
        raise ValueError("--stream-interval must be >= 1")
    if int(args.max_batch_size) <= 0:
        raise ValueError("--max-batch-size must be >= 1")
    if int(args.prefill_chunk_size) <= 0:
        raise ValueError("--prefill-chunk-size must be >= 1")
    device = torch.device(args.device)
    paged_attn = bool(args.paged_attn)
    cuda_graph = bool(args.cuda_graph)
    chunked_prefill = bool(args.chunked_prefill)
    if device.type != "cuda" or not torch.cuda.is_available():
        if paged_attn:
            print("[warn] disabling --paged-attn (requires CUDA)")
        if cuda_graph:
            print("[warn] disabling --cuda-graph (requires CUDA)")
        if chunked_prefill:
            print("[warn] disabling --chunked-prefill (requires CUDA)")
        paged_attn = False
        cuda_graph = False
        chunked_prefill = False
    if not paged_attn:
        if cuda_graph:
            print("[warn] disabling --cuda-graph (requires --paged-attn)")
            cuda_graph = False
        if chunked_prefill:
            print("[warn] disabling --chunked-prefill (requires --paged-attn)")
            chunked_prefill = False
    if chunked_prefill:
        import importlib.util

        if bool(args.no_amp):
            print("[warn] disabling --chunked-prefill (requires fp16/bf16 AMP)")
            chunked_prefill = False
        elif importlib.util.find_spec("flashinfer") is None:
            print("[warn] disabling --chunked-prefill (requires flashinfer)")
            chunked_prefill = False

    served_model_name = args.tokenizer_name or "roseinfer"
    max_batch_size = int(args.max_batch_size)
    kv_cache_max_concurrency = int(getattr(args, "kv_cache_max_concurrency", 0))
    if kv_cache_max_concurrency <= 0:
        if args.max_inflight_requests is not None:
            kv_cache_max_concurrency = int(args.max_inflight_requests)
        else:
            kv_cache_max_concurrency = 256
    if args.max_inflight_requests is not None and kv_cache_max_concurrency < int(
        args.max_inflight_requests
    ):
        raise ValueError(
            "--kv-cache-max-concurrency must be >= --max-inflight-requests when set"
        )
    use_engine_process = bool(args.engine_process)
    if use_engine_process:
        from rosellm.rosetrainer.dataset import build_tokenizer

        tokenizer = build_tokenizer(args.tokenizer_name)
        if args.hf_model_id is not None:
            served_model_name = args.hf_model_id

        api_cpus, engine_cpus = _resolve_mp_cpu_sets(
            mp_affinity=bool(getattr(args, "mp_affinity", True)),
            api_spec=getattr(args, "mp_api_cpus", None),
            engine_spec=getattr(args, "mp_engine_cpus", None),
        )

        if bool(getattr(args, "mp_thread_cap", True)) and device.type == "cuda":
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        engine_torch_threads = (
            1
            if bool(getattr(args, "mp_thread_cap", True)) and device.type == "cuda"
            else None
        )
        engine_args = EngineProcessArgs(
            checkpoint_path=args.checkpoint_path if args.hf_model_id is None else None,
            hf_model_id=args.hf_model_id,
            tokenizer_name=args.tokenizer_name,
            device=args.device,
            no_amp=bool(args.no_amp),
            bf16=bool(args.bf16),
            prefill_attn_backend=str(args.prefill_attn_backend),
            decode_attn_backend=str(args.decode_attn_backend),
            paged_attn=bool(paged_attn),
            cuda_graph=bool(cuda_graph),
            fused_ops=bool(args.fused_ops),
            fused_mlp=bool(args.fused_mlp),
            fused_sampler=bool(args.fused_sampler),
            fused_kv_append=bool(args.fused_kv_append),
            chunked_prefill=bool(chunked_prefill),
            prefill_chunk_size=int(args.prefill_chunk_size),
            prefix_cache=bool(args.prefix_cache),
            overlap_schedule=bool(args.overlap_schedule),
            max_batch_size=int(max_batch_size),
            kv_cache_max_concurrency=int(kv_cache_max_concurrency),
            prefix_cache_max_entries=int(
                getattr(args, "prefix_cache_max_entries", 256)
            ),
            mp_torch_num_threads=engine_torch_threads,
            mp_torch_num_interop_threads=engine_torch_threads,
            mp_cpu_affinity=tuple(engine_cpus) if engine_cpus else None,
            gc_freeze=bool(getattr(args, "gc_freeze", True)),
            gc_warn_ms=float(getattr(args, "gc_warn_ms", 0.0) or 0.0),
            gc_log_all=bool(getattr(args, "gc_log_all", False)),
            mp_fill_target=bool(getattr(args, "mp_fill_target", True)),
            mp_max_recv_per_iter=int(getattr(args, "mp_max_recv_per_iter", 64)),
            mp_flat_events=bool(getattr(args, "mp_flat_events", True)),
        )
        sched_manager: SchedulerManager | MPSchedulerManager = MPSchedulerManager(
            tokenizer,
            engine_args=engine_args,
            stream_interval=int(args.stream_interval),
            max_inflight_requests=args.max_inflight_requests,
            ipc_mode=str(getattr(args, "mp_ipc", "pipe")),
            async_admit=bool(getattr(args, "mp_async_admit", False)),
            tokenize_workers=int(getattr(args, "mp_tokenize_workers", 0)),
        )
        if api_cpus:
            try:
                os.sched_setaffinity(0, set(int(c) for c in api_cpus))
            except Exception:
                pass
    else:
        if args.hf_model_id is None:
            engine = InferenceEngine(
                checkpoint_path=args.checkpoint_path,
                tokenizer_name=args.tokenizer_name,
                device=args.device,
                use_amp=not args.no_amp,
                bf16=args.bf16,
                kv_cache_max_concurrency=int(kv_cache_max_concurrency),
                prefix_cache_max_entries=int(
                    getattr(args, "prefix_cache_max_entries", 256)
                ),
                use_paged_attention=paged_attn,
                use_cuda_graph=cuda_graph,
                prefill_attn_backend=str(args.prefill_attn_backend),
                decode_attn_backend=str(args.decode_attn_backend),
                use_fused_ops=bool(args.fused_ops),
                use_fused_mlp=bool(args.fused_mlp),
                use_fused_sampler=bool(args.fused_sampler),
                use_fused_kv_append=bool(args.fused_kv_append),
            )
            served_model_name = args.tokenizer_name
        else:
            from rosellm.rosetrainer.hf_gpt2 import load_gpt2_from_hf_pretrained

            use_amp = bool(not args.no_amp) and device.type == "cuda"
            if use_amp:
                dtype = torch.bfloat16 if args.bf16 else torch.float16
            else:
                dtype = torch.float32
            model, config, tokenizer = load_gpt2_from_hf_pretrained(
                args.hf_model_id,
                device=device,
                dtype=dtype,
            )
            engine = InferenceEngine(
                checkpoint_path=None,
                tokenizer_name=args.tokenizer_name or args.hf_model_id,
                device=args.device,
                use_amp=use_amp,
                bf16=args.bf16,
                kv_cache_max_concurrency=int(kv_cache_max_concurrency),
                prefix_cache_max_entries=int(
                    getattr(args, "prefix_cache_max_entries", 256)
                ),
                use_paged_attention=paged_attn,
                use_cuda_graph=cuda_graph,
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
            served_model_name = args.hf_model_id
        tokenizer = engine.tokenizer
        sched_manager = SchedulerManager(
            engine,
            max_batch_size=int(max_batch_size),
            max_inflight_requests=args.max_inflight_requests,
            stream_interval=int(args.stream_interval),
            chunked_prefill=chunked_prefill,
            prefill_chunk_size=int(args.prefill_chunk_size),
            prefix_cache=bool(args.prefix_cache),
            overlap_schedule=bool(args.overlap_schedule),
        )

    app = create_app(
        tokenizer,
        sched_manager,
        served_model_name=served_model_name,
        async_streaming=bool(getattr(args, "async_streaming", True)),
        fast_sse=bool(getattr(args, "fast_sse", True)),
    )
    gc_warn_ms = float(getattr(args, "gc_warn_ms", 0.0) or 0.0)
    gc_log_all = bool(getattr(args, "gc_log_all", False))
    if gc_warn_ms > 0.0 or gc_log_all:
        from rosellm.roseinfer.gc_observer import install_gc_observer

        install_gc_observer(
            warn_ms=gc_warn_ms,
            log_all=gc_log_all,
            prefix=f"api pid={os.getpid()}",
            nvtx=os.environ.get("ROSEINFER_NVTX") == "1",
        )
    if bool(getattr(args, "gc_freeze", True)):
        import gc

        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        gc.freeze()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
