from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterator, Optional

import torch

from rosellm.roseinfer.detokenizer import BaseDetokenizer, build_detokenizer
from rosellm.roseinfer.engine import (
    ChunkedOnlineScheduler,
    InferenceEngine,
    OnlineRequest,
    OnlineScheduler,
)
from rosellm.roseinfer.errors import SchedulerManagerOverloadedError


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
    toy: ToyEngineSpec | None = None


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
    AddRequestCmd | CancelRequestCmd | StartProfileCmd | StopProfileCmd | ShutdownCmd
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


EngineEvent = ReadyEvent | TokensEvent | ErrorEvent | ProfileEvent


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
        from rosellm.rosetrainer.hf_gpt2 import load_gpt2_from_hf_pretrained

        use_amp = (not bool(args.no_amp)) and device.type == "cuda"
        if use_amp:
            dtype = torch.bfloat16 if bool(args.bf16) else torch.float16
        else:
            dtype = torch.float32
        model, config, tokenizer = load_gpt2_from_hf_pretrained(
            args.hf_model_id,
            device=device,
            dtype=dtype,
        )
        return InferenceEngine(
            checkpoint_path=None,
            tokenizer_name=args.tokenizer_name,
            device=args.device,
            use_amp=use_amp,
            bf16=bool(args.bf16),
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
        evt_q.put(
            ReadyEvent(
                eos_token_id=engine.eos_token_id,
                max_context=int(engine.config.max_position_embeddings),
            )
        )

        pending: "Deque[AddRequestCmd]" = deque()
        pending_by_rid: dict[int, AddRequestCmd] = {}
        torch_prof: Any | None = None
        torch_prof_dir: str | None = None
        cuda_prof_active = False

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
                prof = torch.profiler.profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
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
                torch_prof = None
                torch_prof_dir = None
                prof.__exit__(None, None, None)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof.export_chrome_trace(os.path.join(out_dir, "trace.json"))
                return
            if tool == "cuda":
                if not cuda_prof_active:
                    raise RuntimeError("cuda profiler not running")
                cuda_prof_active = False
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.profiler.stop()
                return
            raise ValueError("tool must be torch|cuda")

        def cancel_request(request_id: int) -> None:
            pending_by_rid.pop(request_id, None)
            scheduler.discard_request(request_id)

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
                if isinstance(cmd, AddRequestCmd):
                    pending.append(cmd)
                    pending_by_rid[int(cmd.request_id)] = cmd
            else:
                try:
                    while True:
                        cmd = cmd_q.get_nowait()
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
                        elif isinstance(cmd, AddRequestCmd):
                            pending.append(cmd)
                            pending_by_rid[int(cmd.request_id)] = cmd
                except queue.Empty:
                    pass

            tokens_out: dict[int, list[int]] = {}
            finished: set[int] = set()

            batch: list[OnlineRequest] = []
            while pending and len(batch) < int(args.max_batch_size):
                item = pending.popleft()
                if pending_by_rid.pop(int(item.request_id), None) is None:
                    continue
                batch.append(
                    OnlineRequest(
                        prompt="",
                        prompt_token_ids=list(item.prompt_token_ids),
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
                rids = scheduler.add_requests(batch)
                for rid in rids:
                    token_ids = scheduler.get_generated_ids(rid)
                    if token_ids:
                        tokens_out[int(rid)] = [int(t) for t in token_ids]
                    if scheduler.is_finished(rid):
                        finished.add(int(rid))
                        scheduler.discard_request(int(rid))

            if scheduler.has_unfinished():
                step_tokens = scheduler.step()
                for rid, token_id in step_tokens.items():
                    tokens_out.setdefault(int(rid), []).append(int(token_id))
                for rid in scheduler.pop_finished_ids():
                    finished.add(int(rid))
                    scheduler.discard_request(int(rid))

            if tokens_out or finished:
                evt_q.put(
                    TokensEvent(
                        tokens=tokens_out,
                        finished_ids=sorted(finished),
                    )
                )
    except Exception as exc:
        traceback.print_exc()
        try:
            evt_q.put(ErrorEvent(request_id=None, message=str(exc)))
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
    ) -> None:
        if int(stream_interval) <= 0:
            raise ValueError("stream_interval must be >= 1")
        self.tokenizer = tokenizer
        self._make_detok = lambda: build_detokenizer(
            self.tokenizer, tokenizer_name=engine_args.tokenizer_name
        )
        self._stream_interval = int(stream_interval)
        self._max_inflight_requests = (
            int(max_inflight_requests) if max_inflight_requests is not None else None
        )
        if self._max_inflight_requests is not None and self._max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be >= 1")
        self._lock = threading.Lock()
        self._running = True
        self._next_request_id = 0
        self._queues: Dict[int, "queue.Queue[Optional[str]]"] = {}
        self._detoks: Dict[int, BaseDetokenizer] = {}
        self._stream_states: Dict[int, _StreamState] = {}
        self._max_context: int | None = None
        self._profile_seq: int = 0
        self._profile_waiters: dict[int, threading.Event] = {}
        self._profile_results: dict[int, ProfileEvent] = {}

        ctx = mp.get_context("spawn")
        self._cmd_q: "mp.Queue[EngineCommand]" = ctx.Queue()
        self._evt_q: "mp.Queue[EngineEvent]" = ctx.Queue()
        self._proc = ctx.Process(
            target=_engine_process_main,
            args=(engine_args, self._cmd_q, self._evt_q),
            daemon=True,
        )
        self._proc.start()

        ready = self._await_ready(timeout_s=float(start_timeout_s))
        self._max_context = int(ready.max_context)

        self._worker = threading.Thread(
            target=self._event_loop,
            name="roseinfer-mp-dispatch",
            daemon=True,
        )
        self._worker.start()

    def _await_ready(self, *, timeout_s: float) -> ReadyEvent:
        start = time.perf_counter()
        while True:
            left = timeout_s - (time.perf_counter() - start)
            if left <= 0:
                raise TimeoutError("engine process failed to start within timeout")
            evt = self._evt_q.get(timeout=left)
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
            queues = list(self._queues.values())
            self._queues.clear()
            self._detoks.clear()
            self._stream_states.clear()
        for ev in waiters:
            ev.set()
        for q in queues:
            q.put(None)
        try:
            self._cmd_q.put(ShutdownCmd())
        except Exception:
            pass
        worker.join(timeout=1.0)
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)

    def _cancel_backend(self, request_id: int) -> None:
        try:
            self._cmd_q.put(CancelRequestCmd(request_id=int(request_id)))
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
            q: "queue.Queue[Optional[str]]" = queue.Queue()
            detok = self._make_detok()
            self._queues[request_id] = q
            self._detoks[request_id] = detok
            self._stream_states[request_id] = _StreamState()

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
            self._cmd_q.put(
                AddRequestCmd(
                    request_id=int(request_id),
                    prompt_token_ids=token_ids,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    stop_on_eos=bool(stop_on_eos),
                    do_sample=bool(do_sample),
                )
            )
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
            self._cancel_backend(request_id)

    def start_profile(
        self,
        *,
        tool: str,
        output_dir: str | None = None,
        timeout_s: float = 30.0,
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
            profile_id = int(self._profile_seq)
            self._profile_seq += 1
            self._profile_waiters[profile_id] = waiter
        self._cmd_q.put(
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
        timeout_s: float = 30.0,
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
        self._cmd_q.put(StopProfileCmd(profile_id=profile_id, tool=tool))
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
                evt = self._evt_q.get()
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
                if not isinstance(evt, TokensEvent):
                    continue

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
                                [int(t) for t in token_ids],
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
