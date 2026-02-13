import pytest
import torch

from rosellm.rosetrainer.paged_attention import (
    TRITON_AVAILABLE,
    paged_attention_decode_ref,
    paged_attention_decode_triton,
)


def _dense_decode_reference(
    *,
    q: torch.Tensor,  # [B, Hq, D]
    k_past: torch.Tensor,  # [B, Hkv, T, D]
    v_past: torch.Tensor,  # [B, Hkv, T, D]
    k_new: torch.Tensor,  # [B, Hkv, D]
    v_new: torch.Tensor,  # [B, Hkv, D]
    scale: float,
) -> torch.Tensor:  # [B, Hq, D]
    bsz, hq, d = q.shape
    hkv = int(k_past.size(1))
    if hq % hkv != 0:
        raise ValueError("hq must be divisible by hkv")
    group = hq // hkv
    k_past_q = k_past.repeat_interleave(group, dim=1)
    v_past_q = v_past.repeat_interleave(group, dim=1)
    k_new_q = k_new.repeat_interleave(group, dim=1).unsqueeze(2)
    v_new_q = v_new.repeat_interleave(group, dim=1).unsqueeze(2)
    k_all = torch.cat([k_past_q, k_new_q], dim=2)  # [B, Hq, T+1, D]
    v_all = torch.cat([v_past_q, v_new_q], dim=2)
    scores = torch.einsum("bhd,bhtd->bht", q.float(), k_all.float()) * float(scale)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bht,bhtd->bhd", probs, v_all.float())
    return out.to(dtype=q.dtype)


def test_paged_attention_decode_ref_gqa_matches_dense() -> None:
    torch.manual_seed(0)
    bsz = 2
    hq = 4
    hkv = 2
    d = 8
    block_size = 4
    max_blocks = 2
    context_len = 3
    scale = float(d**-0.5)

    q = torch.randn((bsz, hq, d), dtype=torch.float32)
    k_new = torch.randn((bsz, hkv, d), dtype=torch.float32)
    v_new = torch.randn((bsz, hkv, d), dtype=torch.float32)

    # Each sequence uses its own physical block via slot_mapping.
    k_cache = torch.zeros((bsz, hkv, block_size, d), dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    k_past = torch.randn((bsz, hkv, context_len, d), dtype=torch.float32)
    v_past = torch.randn((bsz, hkv, context_len, d), dtype=torch.float32)
    for b in range(bsz):
        k_cache[b, :, :context_len, :].copy_(k_past[b])
        v_cache[b, :, :context_len, :].copy_(v_past[b])

    block_table = torch.tensor(
        [[0, 0], [1, 1]],
        dtype=torch.int32,
    )
    slot_mapping = torch.tensor([0, 1], dtype=torch.int32)
    context_lens = torch.tensor([context_len, context_len], dtype=torch.int32)

    out_ref = paged_attention_decode_ref(
        q=q,
        k_new=k_new,
        v_new=v_new,
        k_cache_layer=k_cache,
        v_cache_layer=v_cache,
        block_table=block_table,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        scale=scale,
        block_size=block_size,
        write_kv=False,
    )
    out_dense = _dense_decode_reference(
        q=q,
        k_past=k_past,
        v_past=v_past,
        k_new=k_new,
        v_new=v_new,
        scale=scale,
    )

    max_err = (out_ref - out_dense).abs().max().item()
    assert max_err < 1e-4


def test_paged_attention_decode_ref_gqa_write_kv() -> None:
    torch.manual_seed(0)
    bsz = 2
    hq = 4
    hkv = 2
    d = 8
    block_size = 4
    context_len = 3
    scale = float(d**-0.5)

    q = torch.randn((bsz, hq, d), dtype=torch.float32)
    k_new = torch.randn((bsz, hkv, d), dtype=torch.float32)
    v_new = torch.randn((bsz, hkv, d), dtype=torch.float32)

    k_cache = torch.zeros((bsz, hkv, block_size, d), dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)

    block_table = torch.tensor(
        [[0, 0], [1, 1]],
        dtype=torch.int32,
    )
    slot_mapping = torch.tensor([0, 1], dtype=torch.int32)
    context_lens = torch.tensor([context_len, context_len], dtype=torch.int32)

    paged_attention_decode_ref(
        q=q,
        k_new=k_new,
        v_new=v_new,
        k_cache_layer=k_cache,
        v_cache_layer=v_cache,
        block_table=block_table,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        scale=scale,
        block_size=block_size,
        write_kv=True,
    )

    # write_pos == context_len == 3 -> block 0 pos 3
    assert torch.allclose(k_cache[0, :, context_len, :], k_new[0])
    assert torch.allclose(v_cache[0, :, context_len, :], v_new[0])
    assert torch.allclose(k_cache[1, :, context_len, :], k_new[1])
    assert torch.allclose(v_cache[1, :, context_len, :], v_new[1])


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not TRITON_AVAILABLE),
    reason="requires CUDA + Triton",
)
def test_paged_attention_decode_triton_gqa_matches_ref() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    bsz = 2
    hq = 4
    hkv = 2
    d = 8
    block_size = 16
    max_blocks = 2
    context_len = 7
    scale = float(d**-0.5)

    q = torch.randn((bsz, hq, d), device=device, dtype=torch.float16)
    k_new = torch.randn((bsz, hkv, d), device=device, dtype=torch.float16)
    v_new = torch.randn((bsz, hkv, d), device=device, dtype=torch.float16)

    k_cache = torch.zeros((bsz, hkv, block_size, d), device=device, dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    k_past = torch.randn((bsz, hkv, context_len, d), device=device, dtype=torch.float16)
    v_past = torch.randn((bsz, hkv, context_len, d), device=device, dtype=torch.float16)
    for b in range(bsz):
        k_cache[b, :, :context_len, :].copy_(k_past[b])
        v_cache[b, :, :context_len, :].copy_(v_past[b])

    block_table = torch.tensor(
        [[0, 0], [1, 1]],
        device=device,
        dtype=torch.int32,
    )
    slot_mapping = torch.tensor([0, 1], device=device, dtype=torch.int32)
    context_lens = torch.tensor([context_len, context_len], device=device, dtype=torch.int32)

    out_ref = paged_attention_decode_ref(
        q=q,
        k_new=k_new,
        v_new=v_new,
        k_cache_layer=k_cache,
        v_cache_layer=v_cache,
        block_table=block_table,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        scale=scale,
        block_size=block_size,
        write_kv=False,
    )
    out_tri = paged_attention_decode_triton(
        q=q,
        k_new=k_new,
        v_new=v_new,
        k_cache_layer=k_cache,
        v_cache_layer=v_cache,
        block_table=block_table,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        scale=scale,
        block_size=block_size,
        write_kv=False,
    )

    max_err = (out_ref - out_tri).abs().max().item()
    assert max_err < 5e-2

