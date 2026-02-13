import argparse

import pytest

from rosellm.roseinfer.cli_generate import (
    _build_chat_prompt,
    _run_interactive_chat,
    parse_args,
)


class _FakeEngine:
    def __init__(self, replies: list[list[str]]) -> None:
        self._replies = replies
        self.prompts: list[str] = []
        self.kwargs_history: list[dict[str, object]] = []

    def stream_generate(
        self,
        prompt: str,
        **kwargs: object,
    ):
        self.prompts.append(prompt)
        self.kwargs_history.append(dict(kwargs))
        idx = len(self.prompts) - 1
        pieces = self._replies[idx]
        for piece in pieces:
            yield piece


def _build_args(
    *,
    stream: bool,
    system_prompt: str = "",
    prompt: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        max_new_tokens=32,
        temperature=0.7,
        top_k=16,
        top_p=0.9,
        stop_on_eos=True,
        do_sample=True,
        stream=stream,
        system_prompt=system_prompt,
        prompt=prompt,
    )


def test_parse_args_requires_prompt_without_interactive() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--checkpoint-path",
                "ckpt.pt",
                "--tokenizer-name",
                "gpt2",
            ]
        )


def test_parse_args_accepts_interactive_without_prompt() -> None:
    args = parse_args(
        [
            "--checkpoint-path",
            "ckpt.pt",
            "--tokenizer-name",
            "gpt2",
            "--interactive",
        ]
    )
    assert args.interactive is True
    assert args.prompt is None


def test_parse_args_requires_checkpoint_without_hf_model_id() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--tokenizer-name",
                "gpt2",
                "--prompt",
                "hi",
            ]
        )


def test_parse_args_requires_tokenizer_without_hf_model_id() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--checkpoint-path",
                "ckpt.pt",
                "--prompt",
                "hi",
            ]
        )


def test_parse_args_accepts_hf_model_id_without_checkpoint() -> None:
    args = parse_args(
        [
            "--hf-model-id",
            "gpt2",
            "--prompt",
            "hi",
        ]
    )
    assert args.hf_model_id == "gpt2"
    assert args.checkpoint_path is None
    assert args.tokenizer_name == "gpt2"


def test_parse_args_prefers_explicit_tokenizer_with_hf_model_id() -> None:
    args = parse_args(
        [
            "--hf-model-id",
            "gpt2",
            "--tokenizer-name",
            "distilgpt2",
            "--prompt",
            "hi",
        ]
    )
    assert args.hf_model_id == "gpt2"
    assert args.tokenizer_name == "distilgpt2"


def test_build_chat_prompt_includes_system_and_history() -> None:
    prompt = _build_chat_prompt(
        system_prompt="you are concise",
        history=[("hello", "hi"), ("how are you?", "great")],
        user_input="tell me more",
    )
    assert prompt.splitlines()[0] == "System:"
    assert prompt.splitlines()[1] == "  you are concise"
    assert "User:\n  hello" in prompt
    assert "Assistant:\n  hi" in prompt
    assert prompt.endswith("  ")


def test_interactive_chat_clear_resets_history(capsys: pytest.CaptureFixture) -> None:
    args = _build_args(
        stream=False,
        system_prompt="be helpful",
    )
    engine = _FakeEngine(
        replies=[
            ["hello there"],
            ["fresh context"],
        ]
    )
    user_inputs = iter(
        [
            "first turn",
            "/clear",
            "second turn",
            "/quit",
        ]
    )

    def _read_input(_prompt: str) -> str:
        return next(user_inputs)

    _run_interactive_chat(
        engine,
        args,
        read_input=_read_input,
    )

    assert len(engine.prompts) == 2
    assert "User:\n  first turn" in engine.prompts[0]
    assert "first turn" not in engine.prompts[1]
    assert "User:\n  second turn" in engine.prompts[1]
    out = capsys.readouterr().out
    assert "[roseinfer] chat history cleared." in out
    assert "Assistant> hello there" in out


def test_interactive_chat_stream_mode_writes_pieces(
    capsys: pytest.CaptureFixture,
) -> None:
    args = _build_args(stream=True)
    engine = _FakeEngine(replies=[["A", "B"]])
    user_inputs = iter(["hello", "/exit"])

    def _read_input(_prompt: str) -> str:
        return next(user_inputs)

    _run_interactive_chat(
        engine,
        args,
        read_input=_read_input,
    )

    assert len(engine.prompts) == 1
    assert engine.prompts[0].startswith("User:\n  hello")
    out = capsys.readouterr().out
    assert "Assistant> AB" in out
    kwargs = engine.kwargs_history[0]
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_k"] == 16


def test_interactive_chat_uses_prompt_as_first_turn() -> None:
    args = _build_args(
        stream=False,
        prompt="boot question",
    )
    engine = _FakeEngine(replies=[["boot answer"]])
    user_inputs = iter(["/quit"])

    def _read_input(_prompt: str) -> str:
        return next(user_inputs)

    _run_interactive_chat(
        engine,
        args,
        read_input=_read_input,
    )

    assert len(engine.prompts) == 1
    assert "User:\n  boot question" in engine.prompts[0]
