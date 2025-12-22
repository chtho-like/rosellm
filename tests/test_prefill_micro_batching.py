import torch

from rosellm.roseinfer.engine import InferenceEngine, InferenceSession
from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.model import GPTModel


class _DummyTokenizer:
    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = int(vocab_size)
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        raise RuntimeError("encode should not be called in this test")

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return ""


def test_prefill_micro_batching_matches_sequential_logits() -> None:
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

    token_ids_list = [
        [1, 2, 3, 4, 5],
        [1, 2],
    ]

    expected: list[torch.Tensor] = []
    for ids in token_ids_list:
        input_ids = engine._encode_prompt_token_ids(ids)
        logits, _, _ = engine.model(
            input_ids=input_ids,
            attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=True,
        )
        expected.append(logits[:, -1, :])
    expected_last_logits = torch.cat(expected, dim=0)

    sessions: list[InferenceSession] = []
    for ids in token_ids_list:
        sess = InferenceSession(engine)
        sess.input_ids = engine._encode_prompt_token_ids(ids)
        sess.set_generation_config(
            max_new_tokens=2,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            do_sample=False,
            stop_on_eos=False,
        )
        sessions.append(sess)

    input_ids, attn_mask, lengths, _ = engine._encode_prompt_token_ids_batch(
        token_ids_list
    )
    got_last_logits = engine._prefill_register_kv_batch(
        sessions=sessions,
        input_ids=input_ids,
        attention_mask=attn_mask,
        lengths=lengths,
    )
    torch.testing.assert_close(
        got_last_logits,
        expected_last_logits,
        atol=1e-5,
        rtol=1e-5,
    )
    for sess in sessions:
        sess.release_kv_blocks()
