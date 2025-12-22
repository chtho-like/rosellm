import argparse
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterator, List, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .detokenizer import BaseDetokenizer
from .engine import InferenceEngine, OnlineRequest, OnlineScheduler


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


def _take_pending_for_prefill(
    pending_buf: "deque[_PendingRequest]",
    pending_q: "queue.Queue[_PendingRequest]",
    *,
    max_reqs: int,
    max_tokens: Optional[int],
    max_context: int,
) -> list[_PendingRequest]:
    if max_reqs <= 0:
        raise ValueError("max_reqs must be positive")
    if max_context <= 0:
        raise ValueError("max_context must be positive")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

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


class SchedulerManager:
    def __init__(
        self,
        engine: InferenceEngine,
        max_batch_size: int = 8,
        prefill_max_batch_size: Optional[int] = None,
        prefill_max_tokens: Optional[int] = None,
        record_token_timestamps: bool = False,
        decode_first: bool = False,
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
        self._decode_first = bool(decode_first)

        self.engine = engine
        self.scheduler = OnlineScheduler(
            engine,
            max_batch_size=int(max_batch_size),
        )
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._queues: Dict[int, "queue.Queue[Optional[str]]"] = {}
        self._detoks: Dict[int, BaseDetokenizer] = {}
        self._record_token_timestamps = bool(record_token_timestamps)
        self._token_timestamps: Dict[int, list[float]] = {}
        self._pending: "queue.Queue[_PendingRequest]" = queue.Queue()
        self._pending_buf: "deque[_PendingRequest]" = deque()
        self._next_request_id: int = 0
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self._worker.start()

    def close(self) -> None:
        worker = self._worker
        request_ids: list[int] = []
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._wakeup.set()
            request_ids = list(self._queues.keys())
            for rid in request_ids:
                q = self._queues.get(rid)
                if q is not None:
                    q.put(None)
            self._queues.clear()
            self._detoks.clear()
            self._token_timestamps.clear()
            self._pending_buf.clear()
        worker.join(timeout=1.0)
        if worker.is_alive():
            return
        for rid in request_ids:
            self.scheduler.discard_request(rid)

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eos: bool = True,
        do_sample: bool = False,
    ) -> int:
        token_ids = self.engine.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        if not token_ids:
            token_ids = [self.engine.eos_token_id]
        detok = self.engine._make_detok()
        detok.start_prompt(token_ids)
        with self._lock:
            if not self._running:
                raise RuntimeError("SchedulerManager is closed")
            request_id = self._next_request_id
            self._next_request_id += 1
            q: "queue.Queue[Optional[str]]" = queue.Queue()
            self._queues[request_id] = q
            self._detoks[request_id] = detok
            if self._record_token_timestamps:
                self._token_timestamps[request_id] = []
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

    def _worker_loop(self) -> None:
        try:
            while True:

                def run_decode_once() -> None:
                    if not self.scheduler.has_unfinished():
                        return
                    step_tokens = self.scheduler.step()
                    finished_ids = self.scheduler.pop_finished_ids()

                    for rid, token_id in step_tokens.items():
                        with self._lock:
                            q = self._queues.get(rid)
                            detok = self._detoks.get(rid)
                            token_ts = (
                                self._token_timestamps.get(rid)
                                if self._record_token_timestamps
                                else None
                            )
                        if q is None or detok is None:
                            self.scheduler.discard_request(rid)
                            continue
                        if token_ts is not None:
                            token_ts.append(time.perf_counter())
                        piece = detok.on_token(int(token_id))
                        if piece:
                            q.put(piece)

                    for rid in finished_ids:
                        with self._lock:
                            q = self._queues.get(rid)
                            detok = self._detoks.get(rid)
                        self.scheduler.discard_request(rid)
                        if q is None:
                            continue
                        if detok is not None:
                            tail = detok.flush()
                            if tail:
                                q.put(tail)
                        q.put(None)

                with self._lock:
                    if not self._running:
                        break
                    max_new = self._prefill_max_batch_size
                    max_tokens = self._prefill_max_tokens
                    max_context = int(self.engine.config.max_position_embeddings)
                    decode_first = self._decode_first

                did_decode = False
                if decode_first and self.scheduler.has_unfinished():
                    run_decode_once()
                    did_decode = True

                pending = _take_pending_for_prefill(
                    self._pending_buf,
                    self._pending,
                    max_reqs=max_new,
                    max_tokens=max_tokens,
                    max_context=max_context,
                )
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
                rids = self.scheduler.add_requests(batch) if batch else []
                for rid in rids:
                    with self._lock:
                        q = self._queues.get(rid)
                        detok = self._detoks.get(rid)
                        token_ts = (
                            self._token_timestamps.get(rid)
                            if self._record_token_timestamps
                            else None
                        )
                    if q is None or detok is None:
                        self.scheduler.discard_request(rid)
                        continue
                    for tid in self.scheduler.get_generated_ids(rid):
                        if token_ts is not None:
                            token_ts.append(time.perf_counter())
                        piece = detok.on_token(int(tid))
                        if piece:
                            q.put(piece)
                    if self.scheduler.is_finished(rid):
                        tail = detok.flush()
                        if tail:
                            q.put(tail)
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


def create_app(engine: InferenceEngine) -> FastAPI:
    app = FastAPI(title="roseinfer", version="0.1.0")
    sched_manager = SchedulerManager(engine, max_batch_size=8)
    app.add_event_handler("shutdown", sched_manager.close)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(
        body: GenerateRequest,
    ) -> GenerateResponse | StreamingResponse:
        if body.stream:
            request_id = sched_manager.add_request(
                prompt=body.prompt,
                max_new_tokens=body.max_new_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                stop_on_eos=body.stop_on_eos,
                do_sample=body.do_sample,
            )

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
            request_id = sched_manager.add_request(
                prompt=prompt,
                max_new_tokens=body.max_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                stop_on_eos=True,
                do_sample=True,
            )

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
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        use_amp=not args.no_amp,
        bf16=args.bf16,
    )
    app = create_app(engine)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
