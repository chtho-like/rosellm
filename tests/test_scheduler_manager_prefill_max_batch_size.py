import pytest
import torch

from rosellm.roseinfer.engine import InferenceEngine
from rosellm.roseinfer.server import SchedulerManager
from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.model import GPTModel


class _DummyTokenizer:
    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = int(vocab_size)
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [1, 2, 3]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return ""


def _make_engine() -> InferenceEngine:
    cfg = GPTConfig(
        vocab_size=128,
        max_position_embeddings=32,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )
    tok = _DummyTokenizer(vocab_size=128)
    model = GPTModel(cfg)
    return InferenceEngine(
        model=model,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="dummy",
        device="cpu",
        use_amp=False,
        kv_cache_max_concurrency=4,
        prefix_cache_max_entries=0,
    )


def test_scheduler_manager_prefill_max_batch_size_validates_positive() -> None:
    torch.manual_seed(0)
    engine = _make_engine()
    with pytest.raises(ValueError, match="prefill_max_batch_size must be positive"):
        SchedulerManager(engine, max_batch_size=2, prefill_max_batch_size=0)


def test_scheduler_manager_prefill_max_batch_size_decouples_from_decode() -> None:
    torch.manual_seed(0)
    engine = _make_engine()
    mgr = SchedulerManager(engine, max_batch_size=2, prefill_max_batch_size=8)
    try:
        assert mgr.scheduler.max_batch_size == 2
        assert mgr._prefill_max_batch_size == 8
    finally:
        mgr.close()
