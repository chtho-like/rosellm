import torch

from rosellm.roseinfer.mp import EngineProcessArgs, MPSchedulerManager, ToyEngineSpec
from rosellm.roseinfer.simple_tokenizer import ByteTokenizer
from rosellm.rosetrainer.config import GPTConfig


def test_mp_scheduler_manager_streaming_cpu() -> None:
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
    tok = ByteTokenizer(vocab_size=128)
    engine_args = EngineProcessArgs(
        checkpoint_path=None,
        hf_model_id=None,
        tokenizer_name="toy",
        device="cpu",
        no_amp=True,
        bf16=False,
        prefill_attn_backend="naive",
        decode_attn_backend="naive",
        paged_attn=False,
        cuda_graph=False,
        fused_ops=False,
        fused_mlp=False,
        fused_sampler=False,
        fused_kv_append=False,
        chunked_prefill=False,
        prefill_chunk_size=256,
        prefix_cache=False,
        overlap_schedule=True,
        max_batch_size=2,
        toy=ToyEngineSpec(config=cfg, seed=0),
    )
    mgr = MPSchedulerManager(
        tok,
        engine_args=engine_args,
        stream_interval=1,
        start_timeout_s=30.0,
    )
    try:
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
