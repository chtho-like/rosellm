import pytest
import torch

from rosellm.roseinfer.engine import InferenceEngine, OnlineRequest, OnlineScheduler
from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.model import GPTModel


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefix_cache_longest_prefix_reuse_uses_decode_path() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=128,
        max_position_embeddings=128,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )

    class _Tok:
        vocab_size = 128
        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del text, add_special_tokens
            return [1]

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            del ids, skip_special_tokens
            return ""

    tok = _Tok()
    model = GPTModel(cfg)
    seq_lens: list[int] = []
    orig_forward = model.forward

    def counting_forward(*args, **kwargs):  # type: ignore[no-untyped-def]
        input_ids = kwargs.get("input_ids")
        if input_ids is not None:
            seq_lens.append(int(input_ids.size(1)))
        return orig_forward(*args, **kwargs)

    model.forward = counting_forward  # type: ignore[method-assign]

    engine = InferenceEngine(
        model=model,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="dummy",
        device="cuda",
        use_amp=False,
        kv_cache_max_concurrency=8,
        prefix_cache_max_entries=8,
        use_paged_attention=True,
        use_cuda_graph=False,
    )

    scheduler = OnlineScheduler(engine, max_batch_size=1, use_prefix_cache=True)
    base_ids = [7] * 64
    extended_ids = base_ids + [8]

    scheduler.add_requests(
        [
            OnlineRequest(
                prompt="base",
                prompt_token_ids=base_ids,
                max_new_tokens=1,
                stop_on_eos=False,
                do_sample=False,
                request_id=0,
            )
        ]
    )
    seq_lens.clear()

    scheduler.add_requests(
        [
            OnlineRequest(
                prompt="extended",
                prompt_token_ids=extended_ids,
                max_new_tokens=1,
                stop_on_eos=False,
                do_sample=False,
                request_id=1,
            )
        ]
    )
    assert seq_lens, "expected at least one model.forward call"
    assert all(l == 1 for l in seq_lens), seq_lens
