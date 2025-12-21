import argparse

import torch

from rosellm.roseinfer.engine import InferenceEngine, InferenceSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify paged attention correctness (dense vs paged) on decode.",
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
        "--prompt",
        type=str,
        default="Hello",
        help="Prompt to verify",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=16,
        help="Number of decode steps to compare (greedy).",
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


@torch.no_grad()
def run_greedy_decode_steps(
    engine: InferenceEngine,
    prompt: str,
    steps: int,
) -> tuple[list[int], str]:
    session = InferenceSession(engine)
    try:
        prompt_ids = engine._maybe_truncate(engine._encode_prompt(prompt))
        logits = session.prefill(prompt_ids)
        tok0 = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        session.generated_ids.append(tok0)
        session.step_count = 1
        token_ids: list[int] = [tok0]
        for _ in range(steps):
            next_logits = engine.decode_step_sessions([session])  # [1, V]
            tok = int(torch.argmax(next_logits, dim=-1).item())
            token_ids.append(tok)
            session.generated_ids.append(tok)
            session.step_count += 1
        full_ids = prompt_ids[0].tolist() + token_ids
        text = engine.tokenizer.decode(full_ids, skip_special_tokens=True)
        return token_ids, text
    finally:
        session.release_kv_blocks()


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        use_amp=not args.no_amp,
        bf16=args.bf16,
        kv_cache_max_concurrency=1,
        prefix_cache_max_entries=0,
        use_paged_attention=False,
    )
    dense_toks, dense_text = run_greedy_decode_steps(engine, args.prompt, args.steps)

    engine.prefix_cache.clear()
    engine.use_paged_attention = True
    paged_toks, paged_text = run_greedy_decode_steps(engine, args.prompt, args.steps)

    if dense_toks != paged_toks:
        for i, (a, b) in enumerate(zip(dense_toks, paged_toks)):
            if a != b:
                raise SystemExit(
                    f"Mismatch at step {i}: dense={a} paged={b}\n"
                    f"dense_text={dense_text}\n"
                    f"paged_text={paged_text}\n"
                )
        raise SystemExit("Mismatch: token sequences differ in length.")

    print("OK: dense == paged")
    print(dense_text)


if __name__ == "__main__":
    main()
