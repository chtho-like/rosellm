import argparse

from .engine import InferenceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a model",
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
        "--prompt",
        type=str,
        required=True,
        help="Prompt to generate text from",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        use_amp=not args.no_amp,
        bf16=args.bf16,
    )
    print(f"[roseinfer] device: {engine.device}")
    print(f"[roseinfer] use_amp: {engine.use_amp}")
    print(f"[roseinfer] prompt: {args.prompt}")
    output = engine.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(f"[roseinfer] output: {output}")


if __name__ == "__main__":
    main()
