import torch

from rosellm.roseinfer.detokenizer import BaseDetokenizer
from rosellm.roseinfer.engine import InferenceEngine
from rosellm.roseinfer.server import SchedulerManager
from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.model import GPTModel


class _CountingTokenizer:
    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = int(vocab_size)
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if not text:
            return []
        return [1, 2, 3]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return ""


class _IdDetok(BaseDetokenizer):
    def reset(self) -> None:
        return None

    def start_prompt(self, prompt_ids: list[int]) -> None:
        del prompt_ids

    def on_token(self, token_id: int) -> str:
        return f"{int(token_id)},"


def _build_mgr(*, stream_interval: int) -> SchedulerManager:
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
    engine._make_detok = lambda: _IdDetok()  # type: ignore[method-assign]
    mgr = SchedulerManager(
        engine, max_batch_size=1, stream_interval=int(stream_interval)
    )
    mgr.scheduler.use_prefix_cache = False
    return mgr


def test_scheduler_manager_stream_interval_batches_output_pieces() -> None:
    mgr1 = _build_mgr(stream_interval=1)
    try:
        rid1 = mgr1.add_request("hello", max_new_tokens=8, stop_on_eos=False)
        pieces1 = list(mgr1.stream_text(rid1))
    finally:
        mgr1.close()

    mgr4 = _build_mgr(stream_interval=4)
    try:
        rid4 = mgr4.add_request("hello", max_new_tokens=8, stop_on_eos=False)
        pieces4 = list(mgr4.stream_text(rid4))
    finally:
        mgr4.close()

    assert "".join(pieces1) == "".join(pieces4)
    assert len(pieces4) < len(pieces1)
