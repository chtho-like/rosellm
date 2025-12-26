import pytest
import torch

from rosellm.rosetrainer import paged_attention as pa


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_paged_attention_decode_triton_write_kv() -> None:
    if not pa.TRITON_AVAILABLE:
        pytest.skip("requires Triton")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16

    batch = 2
    heads = 2
    head_dim = 16
    block_size = 8
    max_blocks = 4
    num_blocks = batch * max_blocks

    q = torch.randn((batch, heads, head_dim), device=device, dtype=dtype)
    k_new = torch.randn_like(q)
    v_new = torch.randn_like(q)

    k_cache = torch.randn(
        (num_blocks, heads, block_size, head_dim), device=device, dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    block_table = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        .view(batch, max_blocks)
        .contiguous()
    )
    slot_mapping = torch.tensor([0, 1], device=device, dtype=torch.int32)
    context_lens = torch.tensor([3, 9], device=device, dtype=torch.int32)

    def _sentinelize(cache: torch.Tensor) -> None:
        for b in range(batch):
            ctx = int(context_lens[b].item())
            slot = int(slot_mapping[b].item())
            logical_blk = ctx // block_size
            blk = int(block_table[slot, logical_blk].item())
            pos = ctx % block_size
            cache[blk, :, pos, :].zero_()

    k0 = k_cache.clone()
    v0 = v_cache.clone()
    _sentinelize(k0)
    _sentinelize(v0)

    out_no = pa.paged_attention_decode_triton(
        q=q,
        k_new=k_new,
        v_new=v_new,
        k_cache_layer=k0,
        v_cache_layer=v0,
        block_table=block_table,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        scale=1.0,
        block_size=block_size,
        write_kv=False,
    )
    for b in range(batch):
        ctx = int(context_lens[b].item())
        slot = int(slot_mapping[b].item())
        logical_blk = ctx // block_size
        blk = int(block_table[slot, logical_blk].item())
        pos = ctx % block_size
        torch.testing.assert_close(
            k0[blk, :, pos, :], torch.zeros_like(k0[blk, :, pos, :]), rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            v0[blk, :, pos, :], torch.zeros_like(v0[blk, :, pos, :]), rtol=0.0, atol=0.0
        )

    k1 = k_cache.clone()
    v1 = v_cache.clone()
    _sentinelize(k1)
    _sentinelize(v1)

    out_yes = pa.paged_attention_decode_triton(
        q=q,
        k_new=k_new,
        v_new=v_new,
        k_cache_layer=k1,
        v_cache_layer=v1,
        block_table=block_table,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        scale=1.0,
        block_size=block_size,
        write_kv=True,
    )

    torch.testing.assert_close(out_yes, out_no, rtol=2e-2, atol=2e-2)
    for b in range(batch):
        ctx = int(context_lens[b].item())
        slot = int(slot_mapping[b].item())
        logical_blk = ctx // block_size
        blk = int(block_table[slot, logical_blk].item())
        pos = ctx % block_size
        torch.testing.assert_close(k1[blk, :, pos, :], k_new[b], rtol=0.0, atol=0.0)
        torch.testing.assert_close(v1[blk, :, pos, :], v_new[b], rtol=0.0, atol=0.0)
