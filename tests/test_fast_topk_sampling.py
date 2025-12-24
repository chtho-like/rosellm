import torch

from rosellm.roseinfer.engine import InferenceEngine
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


def _build_engine(*, vocab_size: int = 32) -> InferenceEngine:
    cfg = GPTConfig(
        vocab_size=vocab_size,
        max_position_embeddings=32,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )
    tok = _CountingTokenizer(vocab_size=vocab_size)
    model = GPTModel(cfg)
    return InferenceEngine(
        model=model,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="dummy",
        device="cpu",
        use_amp=False,
        kv_cache_max_concurrency=1,
        prefix_cache_max_entries=0,
    )


def test_top_k_logits_clamps_to_vocab_size() -> None:
    engine = _build_engine(vocab_size=16)
    logits = torch.randn(2, 16)
    out = engine._top_k_logits(logits, top_k=10_000)
    torch.testing.assert_close(out, logits)


def test_sample_next_token_batch_top_k_gt_vocab_does_not_crash() -> None:
    torch.manual_seed(0)
    engine = _build_engine(vocab_size=16)
    logits = torch.randn(4, 16)
    ids = engine._sample_next_token_batch(
        logits,
        temperature=1.0,
        top_k=10_000,
        top_p=0.9,
        do_sample=True,
    )
    assert tuple(ids.shape) == (4,)
    assert int(ids.min()) >= 0
    assert int(ids.max()) < 16
