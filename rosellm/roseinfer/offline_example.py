import argparse

from .engine import InferenceEngine, OfflineScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a model in batch mode",
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
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="Prompts to generate text from",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
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
        "--no-stop-on-eos",
        dest="stop_on_eos",
        action="store_false",
        help="Do not stop on EOS token",
    )
    parser.set_defaults(stop_on_eos=True)
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
        "--do-sample",
        action="store_true",
        help="Use sampling to generate text (or else greedy)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the output",
    )
    return parser.parse_args()


def offline_example(engine: InferenceEngine, args: argparse.Namespace) -> None:
    scheduler = OfflineScheduler(engine)
    request_ids: list[int] = []
    for p in args.prompts:
        rid = scheduler.add_request(
            p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )
        request_ids.append(rid)
    outputs = scheduler.run()
    for rid in request_ids:
        print(f"### request {rid}")
        print(outputs[rid])
        print()


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        use_amp=not args.no_amp,
        bf16=args.bf16,
    )
    offline_example(engine, args)


if __name__ == "__main__":
    main()
