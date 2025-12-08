import argparse

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine import InferenceEngine


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


def create_app(engine: InferenceEngine) -> FastAPI:
    app = FastAPI(title="roseinfer", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(
        body: GenerateRequest,
    ) -> GenerateResponse | StreamingResponse:
        if body.stream:

            def token_stream() -> bytes:
                chunks = engine.stream_generate(
                    prompt=body.prompt,
                    max_new_tokens=body.max_new_tokens,
                    temperature=body.temperature,
                    top_k=body.top_k,
                    top_p=body.top_p,
                    stop_on_eos=body.stop_on_eos,
                    do_sample=body.do_sample,
                )
                for piece in chunks:
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
            stop_on_eos=body.stop_on_eos,
            do_sample=body.do_sample,
        )
        return GenerateResponse(text=text)

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
