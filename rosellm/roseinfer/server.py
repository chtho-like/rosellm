import argparse
import queue
import threading
import time
import uuid
from typing import Dict, Iterator, List, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .detokenizer import BaseDetokenizer
from .engine import InferenceEngine, OnlineScheduler


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


class SchedulerManager:
    def __init__(
        self,
        engine: InferenceEngine,
        max_batch_size: int = 8,
    ) -> None:
        self.engine = engine
        self.scheduler = OnlineScheduler(
            engine,
            max_batch_size=max_batch_size,
        )
        self._lock = threading.Lock()
        self._queues: Dict[int, "queue.Queue[Optional[str]]"] = {}
        self._detoks: Dict[int, BaseDetokenizer] = {}
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self._worker.start()

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
            request_id = self.scheduler.add_request(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_on_eos=stop_on_eos,
                do_sample=do_sample,
            )
            q: "queue.Queue[Optional[str]]" = queue.Queue()
            self._queues[request_id] = q
            self._detoks[request_id] = detok
            session = self.scheduler._sessions[request_id]
            for tid in session.generated_ids:
                piece = detok.on_token(int(tid))
                if piece:
                    q.put(piece)
            if session.finished:
                tail = detok.flush()
                if tail:
                    q.put(tail)
                q.put(None)
        return request_id

    def stream_text(self, request_id: int) -> Iterator[str]:
        q = self._queues[request_id]
        while True:
            piece = q.get()
            if piece is None:
                break
            yield piece

    def _worker_loop(self) -> None:
        while self._running:
            with self._lock:
                has_work = self.scheduler.has_unfinished()
                if has_work:
                    step_tokens = self.scheduler.step()
                else:
                    step_tokens = {}
            if not has_work:
                time.sleep(0.005)
                continue
            for rid, token_id in step_tokens.items():
                detok = self._detoks.get(rid)
                q = self._queues.get(rid)
                if detok is None or q is None:
                    continue
                piece = detok.on_token(int(token_id))
                if piece:
                    q.put(piece)
            finished_ids: list[int] = []
            with self._lock:
                for rid in list(self._queues.keys()):
                    if self.scheduler.is_finished(rid):
                        finished_ids.append(rid)
            for rid in finished_ids:
                detok = self._detoks.pop(rid, None)
                q = self._queues.pop(rid, None)
                if q is None:
                    continue
                if detok is not None:
                    tail = detok.flush()
                    if tail:
                        q.put(tail)
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
