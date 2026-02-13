import argparse
import sys
from collections.abc import Callable

import torch

from rosellm.rosetrainer.hf_gpt2 import load_gpt2_from_hf_pretrained

from .engine import InferenceEngine

ROLE_SYSTEM = "System"
ROLE_USER = "User"
ROLE_ASSISTANT = "Assistant"
EXIT_COMMANDS = {"/exit", "/quit"}
CLEAR_COMMAND = "/clear"
USER_INPUT_PROMPT = "You> "
ASSISTANT_PREFIX = "Assistant> "
PROMPT_INDENT = "  "


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a model",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Tokenizer name",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=None,
        help=(
            "Load model weights from Hugging Face (GPT-2 only). "
            "If set, --checkpoint-path/--tokenizer-name become optional."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to generate text from",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive multi-turn chat in the shell",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="Optional system instruction used in interactive mode",
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
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
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
        "--do-sample",
        action="store_true",
        help="Use sampling to generate text (or else greedy)",
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
        "--stream",
        action="store_true",
        help="Stream the output",
    )
    args = parser.parse_args(argv)
    if args.hf_model_id is None:
        if args.checkpoint_path is None:
            parser.error("--checkpoint-path is required when --hf-model-id is not set")
        if args.tokenizer_name is None:
            parser.error("--tokenizer-name is required when --hf-model-id is not set")
    else:
        if args.tokenizer_name is None:
            args.tokenizer_name = args.hf_model_id
    if not args.interactive and args.prompt is None:
        parser.error(
            "--prompt is required unless --interactive is enabled.",
        )
    return args


def _build_chat_prompt(
    *,
    system_prompt: str,
    history: list[tuple[str, str]],
    user_input: str,
) -> str:
    lines: list[str] = []
    if system_prompt:
        lines.append(f"{ROLE_SYSTEM}:")
        for line in str(system_prompt).splitlines() or [""]:
            lines.append(f"{PROMPT_INDENT}{line}")
    for user_text, assistant_text in history:
        lines.append(f"{ROLE_USER}:")
        for line in str(user_text).splitlines() or [""]:
            lines.append(f"{PROMPT_INDENT}{line}")
        lines.append(f"{ROLE_ASSISTANT}:")
        for line in str(assistant_text).splitlines() or [""]:
            lines.append(f"{PROMPT_INDENT}{line}")
    lines.append(f"{ROLE_USER}:")
    for line in str(user_input).splitlines() or [""]:
        lines.append(f"{PROMPT_INDENT}{line}")
    lines.append(f"{ROLE_ASSISTANT}:")
    lines.append(PROMPT_INDENT)
    return "\n".join(lines)


def _strip_assistant_prefix(line: str) -> str:
    if line.startswith(f"{ROLE_ASSISTANT}:"):
        rest = line[len(f"{ROLE_ASSISTANT}:") :]
        return rest.lstrip()
    return line


def _is_stop_line(line: str) -> bool:
    return line.startswith(f"{ROLE_USER}:") or line.startswith(f"{ROLE_SYSTEM}:")


def _generate_chat_reply(
    engine: InferenceEngine,
    *,
    model_prompt: str,
    args: argparse.Namespace,
) -> str:
    iterator = engine.stream_generate(
        prompt=model_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_on_eos=args.stop_on_eos,
        do_sample=args.do_sample,
    )
    reply_parts: list[str] = []
    pending = ""
    printed_any = False
    stop = False
    try:
        for piece in iterator:
            if not piece:
                continue
            pending += piece
            while "\n" in pending:
                raw_line, pending = pending.split("\n", 1)
                if _is_stop_line(raw_line):
                    stop = True
                    break
                line = _strip_assistant_prefix(raw_line)
                if not printed_any:
                    line = line.lstrip()
                if not printed_any and not line:
                    continue
                if printed_any:
                    reply_parts.append("\n")
                    if args.stream:
                        sys.stdout.write("\n")
                reply_parts.append(line)
                if args.stream:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                printed_any = True
            if stop:
                break
        if not stop:
            if pending and _is_stop_line(pending):
                pending = ""
            line = _strip_assistant_prefix(pending)
            if not printed_any:
                line = line.lstrip()
            if line or printed_any:
                reply_parts.append(line)
                if args.stream:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                printed_any = printed_any or bool(line)
    finally:
        iterator.close()
    reply = "".join(reply_parts).rstrip()
    if args.stream:
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        print(reply)
    return reply


def _run_interactive_chat(
    engine: InferenceEngine,
    args: argparse.Namespace,
    *,
    read_input: Callable[[str], str] = input,
) -> None:
    print("[roseinfer] interactive mode enabled")
    print(
        f"[roseinfer] commands: {CLEAR_COMMAND} " f"{' '.join(sorted(EXIT_COMMANDS))}",
    )
    if args.system_prompt:
        print(f"[roseinfer] system prompt: {args.system_prompt}")
    history: list[tuple[str, str]] = []
    initial_user_input = args.prompt
    if isinstance(initial_user_input, str):
        initial_user_input = initial_user_input.strip()
    else:
        initial_user_input = ""
    while True:
        if initial_user_input:
            user_input = initial_user_input
            initial_user_input = ""
            print(f"{USER_INPUT_PROMPT}{user_input}")
        else:
            try:
                user_input = read_input(USER_INPUT_PROMPT).strip()
            except EOFError:
                print()
                print("[roseinfer] EOF received, exiting interactive mode.")
                return
            except KeyboardInterrupt:
                print()
                print("[roseinfer] interrupted, exiting interactive mode.")
                return
        if not user_input:
            continue
        if user_input in EXIT_COMMANDS:
            print("[roseinfer] exiting interactive mode.")
            return
        if user_input == CLEAR_COMMAND:
            history.clear()
            print("[roseinfer] chat history cleared.")
            continue
        model_prompt = _build_chat_prompt(
            system_prompt=args.system_prompt,
            history=history,
            user_input=user_input,
        )
        print(ASSISTANT_PREFIX, end="", flush=True)
        assistant_reply = _generate_chat_reply(
            engine,
            model_prompt=model_prompt,
            args=args,
        )
        history.append((user_input, assistant_reply))


def main() -> None:
    args = parse_args()
    if args.hf_model_id is None:
        engine = InferenceEngine(
            checkpoint_path=args.checkpoint_path,
            tokenizer_name=args.tokenizer_name,
            device=args.device,
            use_amp=not args.no_amp,
            bf16=args.bf16,
        )
    else:
        device = torch.device(args.device)
        use_amp = bool(not args.no_amp) and device.type == "cuda"
        if use_amp:
            dtype = torch.bfloat16 if args.bf16 else torch.float16
        else:
            dtype = torch.float32
        model, config, tokenizer = load_gpt2_from_hf_pretrained(
            args.hf_model_id,
            device=device,
            dtype=dtype,
        )
        engine = InferenceEngine(
            checkpoint_path=None,
            tokenizer_name=args.tokenizer_name,
            device=args.device,
            use_amp=use_amp,
            bf16=args.bf16,
            model=model,
            config=config,
            tokenizer=tokenizer,
        )
    print(f"[roseinfer] device: {engine.device}")
    print(f"[roseinfer] use_amp: {engine.use_amp}")
    if args.interactive:
        _run_interactive_chat(engine, args)
        return
    print(f"[roseinfer] prompt: {args.prompt}")
    if args.stream:
        print("[roseinfer] streaming output: ", end="", flush=True)
        for piece in engine.stream_generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_on_eos=args.stop_on_eos,
            do_sample=args.do_sample,
        ):
            if piece:
                print(piece, end="", flush=True)
        print()
    else:
        output = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            stop_on_eos=args.stop_on_eos,
            do_sample=args.do_sample,
        )
        print(f"[roseinfer] output: {output}")


if __name__ == "__main__":
    main()
