import torch

from rosellm.roseinfer.engine import InferenceEngine
from rosellm.roseinfer.server import SchedulerManager
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
        del add_special_tokens
        if not text:
            return []
        return [1, 2, 3]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return ""


def test_server_add_request_reuses_tokenization() -> None:
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
    engine = InferenceEngine(
        model=model,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="dummy",
        device="cpu",
        use_amp=False,
        kv_cache_max_concurrency=1,
        prefix_cache_max_entries=0,
    )

    mgr = SchedulerManager(engine, max_batch_size=1)
    try:
        _ = mgr.add_request("hello", max_new_tokens=1, stop_on_eos=False)
        assert tok.encode_calls == 1
    finally:
        mgr.close()


def test_server_add_request_accepts_prompt_token_ids() -> None:
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
    engine = InferenceEngine(
        model=model,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="dummy",
        device="cpu",
        use_amp=False,
        kv_cache_max_concurrency=1,
        prefix_cache_max_entries=0,
    )

    mgr = SchedulerManager(engine, max_batch_size=1)
    try:
        _ = mgr.add_request(
            "hello",
            prompt_token_ids=[1, 2, 3],
            max_new_tokens=1,
            stop_on_eos=False,
        )
        assert tok.encode_calls == 0
    finally:
        mgr.close()
