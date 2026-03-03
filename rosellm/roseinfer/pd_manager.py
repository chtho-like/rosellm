from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Deque, Dict, Iterator, Optional

import torch

from rosellm.roseinfer.detokenizer import BaseDetokenizer
from rosellm.roseinfer.engine import (
    InferenceEngine,
    InferenceSession,
    OnlineRequest,
    OnlineScheduler,
)
from rosellm.roseinfer.errors import SchedulerManagerOverloadedError
from rosellm.roseinfer.hybrid_queue import HybridQueue


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


@dataclass(frozen=True)
class _PrefillResult:
    request_id: int
    prompt_token_ids: list[int]
    initial_token_ids: list[int]
    finished: bool
    src_session: InferenceSession | None
    error: str | None = None


class _StreamState:
    __slots__ = ("buf", "tokens_since_flush", "sent_any")

    def __init__(self) -> None:
        self.buf: list[str] = []
        self.tokens_since_flush = 0
        self.sent_any = False


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


class PDDisaggregatedSchedulerManager:
    """Prefill/decode disaggregation inside a single process.

    - Prefill runs on `prefill_engine` in a background thread.
    - Decode runs on `decode_engine` in a background thread.
    - KV blocks are copied from prefill -> decode before decode starts.

    This targets tail ITL stability: decode work is no longer interrupted by
    prefill compute in the decode loop.
    """

    def __init__(
        self,
        prefill_engine: InferenceEngine,
        decode_engine: InferenceEngine,
        *,
        max_batch_size: int = 8,
        prefill_max_batch_size: int | None = None,
        prefill_max_tokens: int | None = None,
        stream_interval: int = 1,
        record_token_timestamps: bool = False,
        tokenize_workers: int = 0,
        max_active_requests: int | None = None,
        max_inflight_requests: int | None = None,
        prefix_cache: bool = True,
        overlap_schedule: bool = True,
    ) -> None:
        if int(max_batch_size) <= 0:
            raise ValueError("max_batch_size must be positive")
        if prefill_max_batch_size is None:
            prefill_max_batch_size = int(max_batch_size)
        if int(prefill_max_batch_size) <= 0:
            raise ValueError("prefill_max_batch_size must be positive")
        if prefill_max_tokens is not None and int(prefill_max_tokens) <= 0:
            raise ValueError("prefill_max_tokens must be positive")
        if int(stream_interval) <= 0:
            raise ValueError("stream_interval must be positive")
        if int(tokenize_workers) < 0:
            raise ValueError("tokenize_workers must be non-negative")
        if max_active_requests is not None and int(max_active_requests) <= 0:
            raise ValueError("max_active_requests must be positive")
        if max_inflight_requests is not None and int(max_inflight_requests) <= 0:
            raise ValueError("max_inflight_requests must be positive")

        if prefill_engine.tokenizer is not decode_engine.tokenizer:
            raise ValueError("prefill/decode engines must share the same tokenizer")

        self.prefill_engine = prefill_engine
        self.decode_engine = decode_engine

        self._prefill_max_batch_size = int(prefill_max_batch_size)
        self._prefill_max_tokens = (
            int(prefill_max_tokens) if prefill_max_tokens else None
        )
        self._stream_interval = int(stream_interval)
        self._tokenize_workers = int(tokenize_workers)
        self._record_token_timestamps = bool(record_token_timestamps)
        self._max_active_requests = (
            int(max_active_requests) if max_active_requests is not None else None
        )
        self._max_inflight_requests = (
            int(max_inflight_requests) if max_inflight_requests is not None else None
        )

        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._running = True

        self._queues: Dict[int, "HybridQueue[Optional[str]]"] = {}
        self._detoks: Dict[int, BaseDetokenizer] = {}
        self._stream_states: Dict[int, _StreamState] = {}
        self._token_timestamps: Dict[int, list[float]] = {}
        self._admit_timestamps: Dict[int, float] = {}
        self._tokenize_timestamps: Dict[int, float] = {}
        self._next_request_id = 0
        self._canceled: set[int] = set()

        self._pending: "queue.Queue[_PendingRequest]" = queue.Queue()
        self._pending_buf: Deque[_PendingRequest] = deque()
        self._prefill_results: "queue.Queue[_PrefillResult]" = queue.Queue()
        self._release_prefill: "queue.Queue[int]" = queue.Queue()

        self._tokenize_q: "queue.Queue[_TokenizeTask | None] | None" = (
            queue.Queue() if self._tokenize_workers > 0 else None
        )
        self._tokenize_threads: list[threading.Thread] = []

        self._prefill_sched = OnlineScheduler(
            self.prefill_engine,
            max_batch_size=max(1, int(max_batch_size)),
            use_prefix_cache=bool(prefix_cache),
            overlap_schedule=False,
        )
        self._decode_sched = OnlineScheduler(
            self.decode_engine,
            max_batch_size=int(max_batch_size),
            use_prefix_cache=False,
            overlap_schedule=bool(overlap_schedule),
        )
        self.decode_engine.warmup_paged_attention_decode()

        self._prefill_thread = threading.Thread(
            target=self._prefill_loop,
            name="roseinfer-pd-prefill",
            daemon=True,
        )
        self._decode_thread = threading.Thread(
            target=self._decode_loop,
            name="roseinfer-pd-decode",
            daemon=True,
        )

        self._pending_release_events: dict[int, torch.cuda.Event] = {}

        self._torch_prof: object | None = None
        self._torch_prof_dir: str | None = None
        self._cuda_prof_active = False

        if self._tokenize_q is not None:
            for i in range(self._tokenize_workers):
                th = threading.Thread(
                    target=self._tokenize_loop,
                    name=f"roseinfer-pd-tokenize-{i}",
                    daemon=True,
                )
                self._tokenize_threads.append(th)
                th.start()

        self._prefill_thread.start()
        self._decode_thread.start()

    def set_asyncio_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._async_loop = loop
        with self._lock:
            queues = list(self._queues.values())
        for q in queues:
            q.set_asyncio_loop(loop)

    def close(self) -> None:
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
            self._canceled.clear()
            self._pending_buf.clear()
            self._tokenize_q = None
            self._tokenize_threads.clear()

        if tok_q is not None:
            for _ in tok_threads:
                tok_q.put(None)
            for th in tok_threads:
                th.join(timeout=1.0)

        self._prefill_thread.join(timeout=1.0)
        self._decode_thread.join(timeout=1.0)

        for rid in request_ids:
            self._decode_sched.discard_request(rid)
            self._prefill_sched.discard_request(rid)

    def start_profile(
        self,
        *,
        tool: str,
        output_dir: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        del timeout_s
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
        del timeout_s
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
            detok = self.decode_engine._make_detok()
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
            token_ids = self.decode_engine.tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
        else:
            token_ids = list(prompt_token_ids)
        if not token_ids:
            token_ids = [self.decode_engine.eos_token_id]
        max_pos = int(self.decode_engine.config.max_position_embeddings)
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

    def pop_token_timestamps(self, request_id: int) -> list[float]:
        with self._lock:
            out = self._token_timestamps.pop(request_id, None)
        return list(out) if out is not None else []

    def pop_admit_timestamp(self, request_id: int) -> float | None:
        with self._lock:
            out = self._admit_timestamps.pop(request_id, None)
        return float(out) if out is not None else None

    def pop_tokenize_timestamp(self, request_id: int) -> float | None:
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
                    token_ids = self.decode_engine.tokenizer.encode(
                        task.prompt,
                        add_special_tokens=False,
                    )
                    if not token_ids:
                        token_ids = [self.decode_engine.eos_token_id]
                    max_pos = int(self.decode_engine.config.max_position_embeddings)
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
                self._canceled.add(int(request_id))

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
                self._canceled.add(int(request_id))

    def _prefill_loop(self) -> None:
        try:
            if self.prefill_engine.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.set_device(self.prefill_engine.device)
            while True:
                # Release prefills whose KV copy has completed.
                while True:
                    try:
                        rid = self._release_prefill.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        self._prefill_sched.discard_request(int(rid))
                    except Exception:
                        traceback.print_exc()

                with self._lock:
                    if not self._running:
                        return
                    max_reqs = int(self._prefill_max_batch_size)
                    max_tokens = self._prefill_max_tokens
                    max_ctx = int(self.decode_engine.config.max_position_embeddings)
                    max_active = self._max_active_requests

                    decode_unfinished = int(self._decode_sched.num_unfinished())
                    prefill_unfinished = int(self._prefill_sched.num_unfinished())
                    active_unfinished = decode_unfinished + prefill_unfinished

                admit_cap = _cap_prefill_max_reqs(
                    max_reqs,
                    max_active_requests=max_active,
                    active_unfinished=active_unfinished,
                )
                if admit_cap <= 0:
                    time.sleep(0.001)
                    continue

                if not self._pending_buf:
                    try:
                        req0 = self._pending.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    self._pending_buf.append(req0)

                batch: list[_PendingRequest] = []
                tokens_used = 0
                while len(batch) < admit_cap:
                    if self._pending_buf:
                        req = self._pending_buf.popleft()
                    else:
                        try:
                            req = self._pending.get_nowait()
                        except queue.Empty:
                            break

                    with self._lock:
                        if req.request_id in self._canceled:
                            continue
                        detok = self._detoks.get(req.request_id)
                        out_q = self._queues.get(req.request_id)
                    if detok is None or out_q is None:
                        continue

                    cost = min(len(req.prompt_token_ids), max_ctx)
                    if max_tokens is not None:
                        if batch and tokens_used + cost > int(max_tokens):
                            self._pending_buf.appendleft(req)
                            break
                    batch.append(req)
                    tokens_used += cost

                if not batch:
                    continue

                reqs = [
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
                    for req in batch
                ]
                try:
                    if self._record_token_timestamps:
                        admit_ts = time.perf_counter()
                    else:
                        admit_ts = 0.0
                    rids = self._prefill_sched.add_requests(reqs)
                    if self._record_token_timestamps:
                        with self._lock:
                            for rid in rids:
                                self._admit_timestamps[int(rid)] = float(admit_ts)
                except Exception as exc:
                    msg = f"prefill failed: {exc}"
                    for req in batch:
                        self._prefill_results.put(
                            _PrefillResult(
                                request_id=req.request_id,
                                prompt_token_ids=req.prompt_token_ids,
                                initial_token_ids=[],
                                finished=True,
                                src_session=None,
                                error=msg,
                            )
                        )
                    self._wakeup.set()
                    continue

                req_by_id = {int(req.request_id): req for req in batch}
                for rid in rids:
                    rid = int(rid)
                    req = req_by_id.get(rid)
                    if req is None:
                        continue
                    initial_ids = self._prefill_sched.get_generated_ids(rid)
                    sess = self._prefill_sched._sessions.get(rid)
                    finished = self._prefill_sched.is_finished(rid)

                    if finished:
                        self._prefill_sched.discard_request(rid)
                        self._prefill_results.put(
                            _PrefillResult(
                                request_id=rid,
                                prompt_token_ids=req.prompt_token_ids,
                                initial_token_ids=initial_ids,
                                finished=True,
                                src_session=None,
                            )
                        )
                        continue

                    self._prefill_results.put(
                        _PrefillResult(
                            request_id=rid,
                            prompt_token_ids=req.prompt_token_ids,
                            initial_token_ids=initial_ids,
                            finished=False,
                            src_session=sess,
                        )
                    )

                self._wakeup.set()
        except Exception:
            traceback.print_exc()
            with self._lock:
                queues = list(self._queues.values())
                self._running = False
                self._wakeup.set()
            for q in queues:
                q.put(None)

    def _emit_token_id(
        self,
        *,
        request_id: int,
        token_id: int,
        q: "HybridQueue[Optional[str]]",
        detok: BaseDetokenizer,
        state: _StreamState,
        token_ts: list[float] | None,
    ) -> None:
        if token_ts is not None:
            token_ts.append(time.perf_counter())
        piece = detok.on_token(int(token_id))
        if piece:
            state.buf.append(piece)
        state.tokens_since_flush += 1
        if state.buf and (
            not state.sent_any or state.tokens_since_flush >= self._stream_interval
        ):
            q.put("".join(state.buf))
            state.buf.clear()
            state.tokens_since_flush = 0
            state.sent_any = True

    def _finish_request(
        self,
        *,
        request_id: int,
        q: "HybridQueue[Optional[str]]",
        detok: BaseDetokenizer,
        state: _StreamState,
    ) -> None:
        tail = detok.flush()
        if tail:
            state.buf.append(tail)
        if state.buf:
            q.put("".join(state.buf))
            state.buf.clear()
            state.tokens_since_flush = 0
            state.sent_any = True
        q.put(None)

    def _handle_prefill_result(self, res: _PrefillResult) -> None:
        rid = int(res.request_id)
        with self._lock:
            q = self._queues.get(rid)
            detok = self._detoks.get(rid)
            state = self._stream_states.get(rid)
            token_ts = (
                self._token_timestamps.get(rid)
                if self._record_token_timestamps
                else None
            )
            canceled = rid in self._canceled

        if q is None or detok is None or state is None or canceled:
            if res.src_session is not None:
                self._release_prefill.put(rid)
            return
        if res.error is not None:
            self._finish_request(
                request_id=rid,
                q=q,
                detok=detok,
                state=state,
            )
            return

        for tid in res.initial_token_ids:
            self._emit_token_id(
                request_id=rid,
                token_id=int(tid),
                q=q,
                detok=detok,
                state=state,
                token_ts=token_ts,
            )

        if res.finished:
            self._finish_request(
                request_id=rid,
                q=q,
                detok=detok,
                state=state,
            )
            return

        src_sess = res.src_session
        if src_sess is None:
            self._finish_request(
                request_id=rid,
                q=q,
                detok=detok,
                state=state,
            )
            return

        # Copy KV blocks into the decode engine and adopt the session.
        dst_sess = InferenceSession(self.decode_engine)
        dst_sess.input_ids = self.decode_engine._encode_prompt_token_ids(
            res.prompt_token_ids
        )
        dst_sess.prompt_length = int(src_sess.prompt_length)
        dst_sess.set_generation_config(
            max_new_tokens=int(src_sess.max_new_tokens),
            temperature=float(src_sess.temperature),
            top_k=int(src_sess.top_k),
            top_p=float(src_sess.top_p),
            do_sample=bool(src_sess.do_sample),
            stop_on_eos=bool(src_sess.stop_on_eos),
        )
        dst_sess.generated_ids = list(src_sess.generated_ids)
        dst_sess.step_count = int(src_sess.step_count)
        dst_sess.committed_step_count = int(src_sess.committed_step_count)

        src_kvm = self.prefill_engine.kv_manager
        dst_kvm = self.decode_engine.kv_manager
        dst_sess.block_ids = dst_kvm.clone_blocks_from(
            src=src_kvm,
            src_block_ids=src_sess.block_ids,
        )
        dst_sess.clear_paged_block_table_cache()

        copy_evt: torch.cuda.Event | None = None
        if self.decode_engine.device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(self.decode_engine.device):
                copy_evt = torch.cuda.Event()
                copy_evt.record()

        self._decode_sched.adopt_session(rid, dst_sess)
        if copy_evt is None:
            self._release_prefill.put(rid)
        else:
            self._pending_release_events[rid] = copy_evt

    def _decode_loop(self) -> None:
        try:
            if self.decode_engine.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.set_device(self.decode_engine.device)
            while True:
                with self._lock:
                    if not self._running:
                        return

                did_work = False

                # Release prefill KV once device copy is complete.
                if self._pending_release_events:
                    for rid, evt in list(self._pending_release_events.items()):
                        if evt.query():
                            self._pending_release_events.pop(rid, None)
                            self._release_prefill.put(int(rid))
                            did_work = True

                # Ingest prefill results.
                while True:
                    try:
                        res = self._prefill_results.get_nowait()
                    except queue.Empty:
                        break
                    self._handle_prefill_result(res)
                    did_work = True

                # Run one decode step if possible.
                if self._decode_sched.has_unfinished():
                    step_tokens = self._decode_sched.step()
                    self._torch_prof_step()
                    finished_ids = self._decode_sched.pop_finished_ids()

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
                                    int(rid),
                                    int(token_id),
                                    self._queues.get(int(rid)),
                                    self._detoks.get(int(rid)),
                                    self._stream_states.get(int(rid)),
                                    (
                                        self._token_timestamps.get(int(rid))
                                        if self._record_token_timestamps
                                        else None
                                    ),
                                )
                            )
                        for rid in finished_ids:
                            finished_records.append(
                                (
                                    int(rid),
                                    self._queues.get(int(rid)),
                                    self._detoks.get(int(rid)),
                                    self._stream_states.get(int(rid)),
                                )
                            )

                    for rid, token_id, q, detok, state, token_ts in step_records:
                        if q is None or detok is None or state is None:
                            self._decode_sched.discard_request(rid)
                            continue
                        self._emit_token_id(
                            request_id=rid,
                            token_id=token_id,
                            q=q,
                            detok=detok,
                            state=state,
                            token_ts=token_ts,
                        )

                    for rid, q, detok, state in finished_records:
                        self._decode_sched.discard_request(rid)
                        if q is None or state is None:
                            continue
                        if detok is not None:
                            self._finish_request(
                                request_id=rid,
                                q=q,
                                detok=detok,
                                state=state,
                            )
                    did_work = True

                if did_work:
                    continue

                try:
                    res = self._prefill_results.get(timeout=0.05)
                except queue.Empty:
                    continue
                self._handle_prefill_result(res)
        except Exception:
            traceback.print_exc()
            with self._lock:
                queues = list(self._queues.values())
                self._running = False
                self._wakeup.set()
            for q in queues:
                q.put(None)
