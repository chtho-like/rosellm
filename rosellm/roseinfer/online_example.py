import argparse

from .engine import InferenceEngine, OnlineScheduler


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


def online_example(engine: InferenceEngine, args: argparse.Namespace) -> None:
    scheduler = OnlineScheduler(engine, max_batch_size=4)
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
    step_idx = 0
    r = None
    while scheduler.has_unfinished():
        step_idx += 1
        _ = scheduler.step()
        if step_idx == 2 and r is None:
            # simulate continuous batching
            r = scheduler.add_request("Hello, world!")

    for rid in request_ids:
        if scheduler.is_finished(rid):
            print(f"### request {rid}")
            print(scheduler.get_response(rid))
            print()
        else:
            print(f"### request {rid} is not finished")
            print()
    if r is not None:
        print(f"### request {r}")
        print(scheduler.get_response(r))
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
    online_example(engine, args)


if __name__ == "__main__":
    main()
