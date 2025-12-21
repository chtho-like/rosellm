import math

import torch

from rosellm.rosetrainer.paged_attention import paged_attention_decode_ref


def _dense_decode(
    q: torch.Tensor,  # [B,H,D]
    k_new: torch.Tensor,  # [B,H,D]
    v_new: torch.Tensor,  # [B,H,D]
    k_cache_layer: torch.Tensor,  # [N_BLOCKS,H,BS,D]
    v_cache_layer: torch.Tensor,  # [N_BLOCKS,H,BS,D]
    block_table: torch.Tensor,  # [B,MAX_BLOCKS]
    context_lens: torch.Tensor,  # [B]
    *,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    bsz, n_heads, head_dim = q.shape
    out = torch.empty_like(q)
    num_blocks = int(block_table.size(1))
    for b in range(bsz):
        past_len = int(context_lens[b].item())
        pieces_k = []
        pieces_v = []
        for lb in range(num_blocks):
            start = lb * block_size
            take = max(0, min(block_size, past_len - start))
            if take == 0:
                continue
            phys = int(block_table[b, lb].item())
            pieces_k.append(k_cache_layer[phys, :, :take, :])  # [H,take,D]
            pieces_v.append(v_cache_layer[phys, :, :take, :])
        if pieces_k:
            past_k = torch.cat(pieces_k, dim=1)  # [H,L,D]
            past_v = torch.cat(pieces_v, dim=1)
        else:
            past_k = torch.empty(
                (n_heads, 0, head_dim),
                device=q.device,
                dtype=q.dtype,
            )
            past_v = torch.empty_like(past_k)
        k_all = torch.cat([past_k, k_new[b].unsqueeze(1)], dim=1)  # [H,L+1,D]
        v_all = torch.cat([past_v, v_new[b].unsqueeze(1)], dim=1)
        scores = (q[b].float().unsqueeze(1) * k_all.float()).sum(-1) * scale  # [H,L+1]
        w = torch.softmax(scores, dim=-1)
        out[b] = (w.unsqueeze(-1) * v_all.float()).sum(dim=1).to(dtype=q.dtype)
    return out


def test_paged_attention_decode_ref_matches_dense() -> None:
    torch.manual_seed(0)
    bsz, n_heads, head_dim = 3, 4, 8
    block_size = 4
    max_blocks = 8
    max_logical_blocks = 5
    context_lens = torch.tensor([0, 3, 11], dtype=torch.int32)  # include 0
    k_cache = torch.randn(
        max_blocks, n_heads, block_size, head_dim, dtype=torch.float16
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.randint(
        0,
        max_blocks,
        (bsz, max_logical_blocks),
        dtype=torch.int32,
    )
    q = torch.randn(bsz, n_heads, head_dim, dtype=torch.float16)
    k_new = torch.randn_like(q)
    v_new = torch.randn_like(q)
    scale = 1.0 / math.sqrt(float(head_dim))

    out_ref = paged_attention_decode_ref(
        q=q,
        k_new=k_new,
        v_new=v_new,
        k_cache_layer=k_cache,
        v_cache_layer=v_cache,
        block_table=block_table,
        context_lens=context_lens,
        scale=scale,
        block_size=block_size,
    )
    assert not torch.isnan(out_ref).any()

    out_dense = _dense_decode(
        q=q,
        k_new=k_new,
        v_new=v_new,
        k_cache_layer=k_cache,
        v_cache_layer=v_cache,
        block_table=block_table,
        context_lens=context_lens,
        scale=scale,
        block_size=block_size,
    )
    max_err = (out_ref - out_dense).abs().max().item()
    assert max_err < 1e-3
