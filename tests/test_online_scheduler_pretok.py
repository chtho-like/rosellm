import torch

from rosellm.roseinfer.engine import InferenceEngine, OnlineScheduler
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


def test_online_scheduler_add_request_pretok_skips_encode() -> None:
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

    scheduler = OnlineScheduler(engine, max_batch_size=1, use_prefix_cache=False)
    scheduler.add_request(
        "hello",
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=2,
        stop_on_eos=False,
        do_sample=False,
    )
    assert tok.encode_calls == 0
