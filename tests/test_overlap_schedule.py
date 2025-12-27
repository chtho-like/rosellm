import pytest
import torch

from rosellm.roseinfer.engine import (
    ChunkedOnlineScheduler,
    InferenceEngine,
    OnlineRequest,
    OnlineScheduler,
)
from rosellm.roseinfer.simple_tokenizer import ByteTokenizer
from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.model import GPTModel


@pytest.mark.gpu
def test_overlap_online_scheduler_matches_sync_greedy_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for overlap scheduling")
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=256,
        max_position_embeddings=64,
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_ff=128,
        dropout=0.0,
    )
    tok = ByteTokenizer(vocab_size=int(cfg.vocab_size))

    def make_engine() -> InferenceEngine:
        torch.manual_seed(0)
        model = GPTModel(cfg)
        return InferenceEngine(
            model=model,
            config=cfg,
            tokenizer=tok,
            tokenizer_name="toy",
            device="cuda",
            use_amp=True,
            kv_cache_max_concurrency=8,
            prefix_cache_max_entries=0,
            use_paged_attention=True,
            use_cuda_graph=False,
            prefill_attn_backend="naive",
            decode_attn_backend="naive",
            use_fused_ops=False,
            use_fused_mlp=False,
            use_fused_sampler=False,
            use_fused_kv_append=False,
        )

    reqs = [
        OnlineRequest(
            prompt="",
            prompt_token_ids=[5, 6, 7, 8],
            max_new_tokens=8,
            stop_on_eos=False,
            do_sample=False,
            request_id=0,
        ),
        OnlineRequest(
            prompt="",
            prompt_token_ids=[9, 10, 11],
            max_new_tokens=8,
            stop_on_eos=False,
            do_sample=False,
            request_id=1,
        ),
    ]

    eng0 = make_engine()
    sched0 = OnlineScheduler(
        eng0, max_batch_size=2, use_prefix_cache=False, overlap_schedule=False
    )
    sched0.add_requests(reqs)
    while sched0.has_unfinished():
        sched0.step()
    out0 = {rid: sched0.get_generated_ids(rid) for rid in (0, 1)}

    eng1 = make_engine()
    sched1 = OnlineScheduler(
        eng1, max_batch_size=2, use_prefix_cache=False, overlap_schedule=True
    )
    sched1.add_requests(reqs)
    while sched1.has_unfinished():
        sched1.step()
    out1 = {rid: sched1.get_generated_ids(rid) for rid in (0, 1)}

    assert out0 == out1


@pytest.mark.gpu
def test_overlap_chunked_scheduler_matches_sync_greedy_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for overlap scheduling")
    torch.manual_seed(0)
    # Flashinfer paged prefill requires head_dim in a supported set (e.g., 64).
    cfg = GPTConfig(
        vocab_size=256,
        max_position_embeddings=128,
        n_layers=2,
        n_heads=2,
        d_model=128,
        d_ff=256,
        dropout=0.0,
    )
    tok = ByteTokenizer(vocab_size=int(cfg.vocab_size))

    def make_engine() -> InferenceEngine:
        torch.manual_seed(0)
        model = GPTModel(cfg)
        return InferenceEngine(
            model=model,
            config=cfg,
            tokenizer=tok,
            tokenizer_name="toy",
            device="cuda",
            use_amp=True,
            kv_cache_max_concurrency=8,
            prefix_cache_max_entries=0,
            use_paged_attention=True,
            use_cuda_graph=False,
            prefill_attn_backend="flashinfer",
            decode_attn_backend="naive",
            use_fused_ops=False,
            use_fused_mlp=False,
            use_fused_sampler=False,
            use_fused_kv_append=False,
        )

    reqs = [
        OnlineRequest(
            prompt="",
            prompt_token_ids=[(i + 1) % cfg.vocab_size for i in range(23)],
            max_new_tokens=6,
            stop_on_eos=False,
            do_sample=False,
            request_id=0,
        ),
        OnlineRequest(
            prompt="",
            prompt_token_ids=[(i + 7) % cfg.vocab_size for i in range(31)],
            max_new_tokens=6,
            stop_on_eos=False,
            do_sample=False,
            request_id=1,
        ),
    ]

    eng0 = make_engine()
    sched0 = ChunkedOnlineScheduler(
        eng0,
        max_batch_size=2,
        prefill_chunk_size=8,
        prefill_max_batch_size=2,
        use_prefix_cache=False,
        overlap_schedule=False,
    )
    sched0.add_requests(reqs)
    while sched0.has_unfinished():
        sched0.step()
    out0 = {rid: sched0.get_generated_ids(rid) for rid in (0, 1)}

    eng1 = make_engine()
    sched1 = ChunkedOnlineScheduler(
        eng1,
        max_batch_size=2,
        prefill_chunk_size=8,
        prefill_max_batch_size=2,
        use_prefix_cache=False,
        overlap_schedule=True,
    )
    sched1.add_requests(reqs)
    while sched1.has_unfinished():
        sched1.step()
    out1 = {rid: sched1.get_generated_ids(rid) for rid in (0, 1)}

    assert out0 == out1
