import pytest
import torch

from rosellm.roseinfer.engine import InferenceEngine
from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.model import GPTModel

try:
    import flashinfer  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    flashinfer = None  # type: ignore[assignment]


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


def _build_engine(*, vocab_size: int = 64) -> InferenceEngine:
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
    model = GPTModel(cfg).to("cuda")
    return InferenceEngine(
        model=model,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="dummy",
        device="cuda",
        use_amp=False,
        kv_cache_max_concurrency=1,
        prefix_cache_max_entries=0,
        use_fused_sampler=True,
    )


def _generator_offset(gen: torch.Generator) -> int:
    state = gen.get_state().view(torch.int64)
    return int(state[1].item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashinfer_fused_sampler_top_k_zero_uses_torch_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rosellm.roseinfer import engine as engine_mod

    class _DummySampling:
        @staticmethod
        def top_k_top_p_sampling_from_logits(*args, **kwargs):
            raise AssertionError("flashinfer sampler should not be called for top_k=0")

    dummy_flashinfer = type(
        "_DummyFlashinfer",
        (),
        {"sampling": _DummySampling},
    )
    monkeypatch.setattr(engine_mod, "flashinfer", dummy_flashinfer)

    torch.manual_seed(0)
    engine = _build_engine(vocab_size=64)
    assert engine._sampling_generator is not None

    logits = torch.randn(4, 64, device=engine.device, dtype=torch.float16)
    ids = engine._sample_next_token_batch(
        logits,
        temperature=0.7,
        top_k=0,
        top_p=0.9,
        do_sample=True,
    )
    assert tuple(ids.shape) == (4,)
    assert int(ids.min()) >= 0
    assert int(ids.max()) < 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashinfer_fused_sampler_advances_generator_state() -> None:
    if flashinfer is None:
        pytest.skip("requires flashinfer")

    torch.manual_seed(0)
    engine = _build_engine(vocab_size=64)
    assert engine._sampling_generator is not None

    logits = torch.randn(4, 64, device=engine.device, dtype=torch.float16)
    off0 = _generator_offset(engine._sampling_generator)
    ids1 = engine._sample_next_token_batch(
        logits,
        temperature=0.7,
        top_k=10,
        top_p=0.9,
        do_sample=True,
    )
    off1 = _generator_offset(engine._sampling_generator)
    ids2 = engine._sample_next_token_batch(
        logits,
        temperature=0.7,
        top_k=10,
        top_p=0.9,
        do_sample=True,
    )
    off2 = _generator_offset(engine._sampling_generator)

    assert off1 != off0
    assert off2 != off1
    assert tuple(ids1.shape) == (4,)
    assert ids1.is_cuda
    assert ids1.dtype == torch.int32
    assert tuple(ids2.shape) == (4,)
