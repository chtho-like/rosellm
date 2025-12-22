import torch

from rosellm.roseinfer.engine import InferenceEngine, OnlineRequest, OnlineScheduler
from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.model import GPTModel


class _CountingTokenizer:
    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = int(vocab_size)
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.encode_calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls += 1
        del text, add_special_tokens
        return [1, 2, 3]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return ""


def test_online_scheduler_add_requests_prefix_cache_batches_prefill() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=128,
        max_position_embeddings=32,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )
    tok = _CountingTokenizer(vocab_size=128)
    model = GPTModel(cfg)
    forward_calls = 0
    orig_forward = model.forward

    def counting_forward(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal forward_calls
        forward_calls += 1
        return orig_forward(*args, **kwargs)

    model.forward = counting_forward  # type: ignore[method-assign]

    engine = InferenceEngine(
        model=model,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="dummy",
        device="cpu",
        use_amp=False,
        kv_cache_max_concurrency=8,
        prefix_cache_max_entries=8,
    )

    scheduler = OnlineScheduler(engine, max_batch_size=8, use_prefix_cache=True)
    scheduler.add_requests(
        [
            OnlineRequest(
                prompt="p0",
                prompt_token_ids=[1, 2, 3],
                max_new_tokens=1,
                stop_on_eos=False,
                do_sample=False,
                request_id=0,
            ),
            OnlineRequest(
                prompt="p1",
                prompt_token_ids=[1, 2, 3, 4],
                max_new_tokens=1,
                stop_on_eos=False,
                do_sample=False,
                request_id=1,
            ),
            OnlineRequest(
                prompt="p2",
                prompt_token_ids=[1, 2],
                max_new_tokens=1,
                stop_on_eos=False,
                do_sample=False,
                request_id=2,
            ),
        ]
    )
    assert tok.encode_calls == 0
    assert forward_calls == 1


def test_online_scheduler_add_requests_prefix_cache_dedups_prompts_in_batch() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=128,
        max_position_embeddings=32,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )
    tok = _CountingTokenizer(vocab_size=128)
    model = GPTModel(cfg)
    forward_calls = 0
    orig_forward = model.forward

    def counting_forward(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal forward_calls
        forward_calls += 1
        return orig_forward(*args, **kwargs)

    model.forward = counting_forward  # type: ignore[method-assign]

    engine = InferenceEngine(
        model=model,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="dummy",
        device="cpu",
        use_amp=False,
        kv_cache_max_concurrency=8,
        prefix_cache_max_entries=8,
    )

    scheduler = OnlineScheduler(engine, max_batch_size=8, use_prefix_cache=True)
    scheduler.add_requests(
        [
            OnlineRequest(
                prompt="same",
                prompt_token_ids=[1, 2, 3],
                max_new_tokens=1,
                stop_on_eos=False,
                do_sample=False,
                request_id=0,
            ),
            OnlineRequest(
                prompt="same",
                prompt_token_ids=[1, 2, 3],
                max_new_tokens=1,
                stop_on_eos=False,
                do_sample=False,
                request_id=1,
            ),
            OnlineRequest(
                prompt="same",
                prompt_token_ids=[1, 2, 3],
                max_new_tokens=1,
                stop_on_eos=False,
                do_sample=False,
                request_id=2,
            ),
        ]
    )
    assert tok.encode_calls == 0
    assert forward_calls == 1
