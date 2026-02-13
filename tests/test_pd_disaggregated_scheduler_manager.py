import torch

from rosellm.roseinfer.engine import (
    InferenceEngine,
    InferenceSession,
    OnlineRequest,
    OnlineScheduler,
)
from rosellm.roseinfer.pd_manager import PDDisaggregatedSchedulerManager
from rosellm.roseinfer.server import SchedulerManager
from rosellm.roseinfer.simple_tokenizer import ByteTokenizer
from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.model import GPTModel


def _make_toy_cfg() -> GPTConfig:
    return GPTConfig(
        vocab_size=128,
        max_position_embeddings=32,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )


def test_pd_disaggregated_scheduler_manager_streaming_cpu_matches_baseline() -> None:
    torch.manual_seed(0)
    cfg = _make_toy_cfg()
    tok = ByteTokenizer(vocab_size=128)

    prefill_model = GPTModel(cfg)
    decode_model = GPTModel(cfg)
    decode_model.load_state_dict(prefill_model.state_dict(), strict=True)

    prefill_engine = InferenceEngine(
        model=prefill_model,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="toy",
        device="cpu",
        use_amp=False,
        kv_cache_max_concurrency=8,
        prefix_cache_max_entries=0,
        use_paged_attention=False,
        use_cuda_graph=False,
        prefill_attn_backend="naive",
        decode_attn_backend="naive",
        use_fused_ops=False,
        use_fused_mlp=False,
        use_fused_sampler=False,
        use_fused_kv_append=False,
    )
    decode_engine = InferenceEngine(
        model=decode_model,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="toy",
        device="cpu",
        use_amp=False,
        kv_cache_max_concurrency=8,
        prefix_cache_max_entries=0,
        use_paged_attention=False,
        use_cuda_graph=False,
        prefill_attn_backend="naive",
        decode_attn_backend="naive",
        use_fused_ops=False,
        use_fused_mlp=False,
        use_fused_sampler=False,
        use_fused_kv_append=False,
    )

    pd_mgr = PDDisaggregatedSchedulerManager(
        prefill_engine,
        decode_engine,
        max_batch_size=2,
        stream_interval=1,
        prefix_cache=False,
        overlap_schedule=False,
    )
    baseline_engine = InferenceEngine(
        model=GPTModel(cfg),
        config=cfg,
        tokenizer=tok,
        tokenizer_name="toy",
        device="cpu",
        use_amp=False,
        kv_cache_max_concurrency=8,
        prefix_cache_max_entries=0,
        use_paged_attention=False,
        use_cuda_graph=False,
        prefill_attn_backend="naive",
        decode_attn_backend="naive",
        use_fused_ops=False,
        use_fused_mlp=False,
        use_fused_sampler=False,
        use_fused_kv_append=False,
    )
    baseline_engine.model.load_state_dict(prefill_model.state_dict(), strict=True)
    baseline_mgr = SchedulerManager(
        baseline_engine,
        max_batch_size=2,
        stream_interval=1,
    )
    try:
        rid_pd = pd_mgr.add_request(
            "hello",
            max_new_tokens=4,
            stop_on_eos=False,
            do_sample=False,
        )
        out_pd = "".join(list(pd_mgr.stream_text(rid_pd)))

        rid_base = baseline_mgr.add_request(
            "hello",
            max_new_tokens=4,
            stop_on_eos=False,
            do_sample=False,
        )
        out_base = "".join(list(baseline_mgr.stream_text(rid_base)))

        assert out_pd == out_base
        assert out_pd != ""
    finally:
        pd_mgr.close()
        baseline_mgr.close()


def test_kv_block_manager_clone_blocks_from_copies_kv() -> None:
    torch.manual_seed(0)
    cfg = _make_toy_cfg()
    tok = ByteTokenizer(vocab_size=128)

    model0 = GPTModel(cfg)
    model1 = GPTModel(cfg)
    model1.load_state_dict(model0.state_dict(), strict=True)

    eng0 = InferenceEngine(
        model=model0,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="toy",
        device="cpu",
        use_amp=False,
        kv_cache_max_concurrency=4,
        prefix_cache_max_entries=0,
        use_paged_attention=False,
        use_cuda_graph=False,
        prefill_attn_backend="naive",
        decode_attn_backend="naive",
        use_fused_ops=False,
        use_fused_mlp=False,
        use_fused_sampler=False,
        use_fused_kv_append=False,
    )
    eng1 = InferenceEngine(
        model=model1,
        config=cfg,
        tokenizer=tok,
        tokenizer_name="toy",
        device="cpu",
        use_amp=False,
        kv_cache_max_concurrency=4,
        prefix_cache_max_entries=0,
        use_paged_attention=False,
        use_cuda_graph=False,
        prefill_attn_backend="naive",
        decode_attn_backend="naive",
        use_fused_ops=False,
        use_fused_mlp=False,
        use_fused_sampler=False,
        use_fused_kv_append=False,
    )

    sched0 = OnlineScheduler(eng0, max_batch_size=1, use_prefix_cache=False)
    rids = sched0.add_requests(
        [
            OnlineRequest(
                prompt="hi",
                max_new_tokens=4,
                stop_on_eos=False,
                do_sample=False,
                request_id=0,
            )
        ]
    )
    assert rids == [0]
    src = sched0._sessions[0]
    assert not src.finished
    assert src.prompt_length > 0

    dst = InferenceSession(eng1)
    dst.prompt_length = int(src.prompt_length)

    src_kvm = eng0.kv_manager
    dst_kvm = eng1.kv_manager
    for layer_idx in range(int(src_kvm.num_layers)):
        dst.block_ids_per_layer[layer_idx] = dst_kvm.clone_blocks_from(
            src=src_kvm,
            layer_idx=layer_idx,
            src_block_ids=src.block_ids_per_layer[layer_idx],
        )

    total_len = int(src.prompt_length)
    for layer_idx in range(int(src_kvm.num_layers)):
        k0, v0 = src_kvm.gather_sequence(
            layer_idx,
            src.block_ids_per_layer[layer_idx],
            total_len,
        )
        k1, v1 = dst_kvm.gather_sequence(
            layer_idx,
            dst.block_ids_per_layer[layer_idx],
            total_len,
        )
        assert torch.allclose(k0, k1)
        assert torch.allclose(v0, v1)
