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
        del add_special_tokens
        if not text:
            return [self.eos_token_id]
        out: list[int] = []
        for b in text.encode("utf-8"):
            out.append(int(b % (self.vocab_size - 1)) + 1)
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(i) for i in ids)


def test_server_streaming_emits_tokens() -> None:
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
    tok = _DummyTokenizer(vocab_size=128)
    model = GPTModel(cfg)
    engine = InferenceEngine(
        model=model,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="dummy",
        device="cpu",
        use_amp=False,
        kv_cache_max_concurrency=4,
        prefix_cache_max_entries=0,
    )

    mgr = SchedulerManager(engine, max_batch_size=2)
    try:
        mgr.scheduler.use_prefix_cache = False
        rid = mgr.add_request(
            "hello",
            max_new_tokens=2,
            stop_on_eos=False,
            do_sample=False,
        )
        pieces = list(mgr.stream_text(rid))
        assert "".join(pieces) != ""
    finally:
        mgr.close()
