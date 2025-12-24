import argparse
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterator, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .detokenizer import BaseDetokenizer
from .engine import InferenceEngine, OnlineRequest, OnlineScheduler


class SchedulerManagerOverloadedError(RuntimeError):
    pass


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


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = (64,)
    temperature: float = (1.0,)
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False


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
        self.scheduler = OnlineScheduler(
            engine,
            max_batch_size=int(max_batch_size),
        )
        self.engine.warmup_paged_attention_decode()
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._queues: Dict[int, "queue.Queue[Optional[str]]"] = {}
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
            q: "queue.Queue[Optional[str]]" = queue.Queue()
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

    def _worker_loop(self) -> None:
        try:
            while True:

                def run_decode_once() -> None:
                    if not self.scheduler.has_unfinished():
                        return
                    step_tokens = self.scheduler.step()
                    finished_ids = self.scheduler.pop_finished_ids()

                    step_records: list[
                        tuple[
                            int,
                            int,
                            "queue.Queue[Optional[str]] | None",
                            BaseDetokenizer | None,
                            _StreamState | None,
                            list[float] | None,
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
    engine: InferenceEngine,
    prompt: str,
    completion: str,
) -> UsageInfo:
    tokenizer = engine.tokenizer
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
    engine: InferenceEngine,
    *,
    max_inflight_requests: Optional[int] = None,
    stream_interval: int = 1,
) -> FastAPI:
    app = FastAPI(title="roseinfer", version="0.1.0")
    sched_manager = SchedulerManager(
        engine,
        max_batch_size=8,
        max_inflight_requests=max_inflight_requests,
        stream_interval=stream_interval,
    )
    app.add_event_handler("shutdown", sched_manager.close)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

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

            def token_stream() -> Iterator[bytes]:
                for piece in sched_manager.stream_text(request_id):
                    yield piece.encode("utf-8")

            return StreamingResponse(
                token_stream(),
                media_type="text/plain; charset=utf-8",
            )
        text = engine.generate(
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            stop_on_eos=True,
            do_sample=True,
        )
        return GenerateResponse(text=text)

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(
        body: ChatCompletionRequest,
    ) -> ChatCompletionResponse | StreamingResponse:
        prompt = format_messages_as_prompt(body.messages)
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = body.model or "roseinfer"
        if body.stream:
            try:
                request_id = sched_manager.add_request(
                    prompt=prompt,
                    max_new_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_k=body.top_k,
                    top_p=body.top_p,
                    stop_on_eos=True,
                    do_sample=True,
                )
            except SchedulerManagerOverloadedError as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc

            def event_stream():
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
        text = engine.generate(
            prompt=prompt,
            max_new_tokens=body.max_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            stop_on_eos=True,
            do_sample=True,
        )
        usage = estimate_usage(engine, prompt, text)
        resp = ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=created,
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )
        return resp

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the inference server",
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
        "--max-inflight-requests",
        type=int,
        default=None,
        help="Max inflight requests accepted by SchedulerManager (default: unlimited).",
    )
    parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Flush streaming output every N generated tokens (default: 1).",
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
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    if args.max_inflight_requests is not None and int(args.max_inflight_requests) <= 0:
        raise ValueError("--max-inflight-requests must be >= 1")
    if int(args.stream_interval) <= 0:
        raise ValueError("--stream-interval must be >= 1")
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        use_amp=not args.no_amp,
        bf16=args.bf16,
    )
    app = create_app(
        engine,
        max_inflight_requests=args.max_inflight_requests,
        stream_interval=int(args.stream_interval),
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
