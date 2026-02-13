from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

try:
    import flashinfer  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    flashinfer = None  # type: ignore[assignment]

try:
    from flash_attn import flash_attn_varlen_func  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    flash_attn_varlen_func = None  # type: ignore[assignment]


_DEFAULT_FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024


@dataclass
class _FlashInferPrefillCache:
    device: torch.device
    workspace: torch.Tensor
    wrapper: Any
    last_sig: tuple[Any, ...] | None = None


_FLASHINFER_PREFILL_CACHES: dict[int, _FlashInferPrefillCache] = {}


@dataclass
class _FlashInferPagedPrefillCache:
    device: torch.device
    workspace: torch.Tensor
    wrapper: Any
    last_sig: tuple[Any, ...] | None = None


_FLASHINFER_PAGED_PREFILL_CACHES: dict[int, _FlashInferPagedPrefillCache] = {}


def _flashinfer_prefill_plan(
    *,
    device: torch.device,
    batch_size: int,
    lengths: torch.Tensor,  # [B] int32 cuda
    num_heads: int,
    head_dim: int,
    sm_scale: float,
    causal: bool,
    q_dtype: torch.dtype,
) -> Any:
    if flashinfer is None:
        raise RuntimeError(
            "flashinfer is not installed; install it to use attn_backend='flashinfer'"
        )
    if device.type != "cuda":
        raise RuntimeError("flashinfer attention backend requires CUDA")
    if q_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"flashinfer attention backend requires fp16/bf16, got dtype={q_dtype}"
        )
    if lengths.dtype != torch.int32:
        raise ValueError("lengths must be int32")
    if lengths.dim() != 1 or int(lengths.numel()) != int(batch_size):
        raise ValueError("lengths shape mismatch")

    cache = _FLASHINFER_PREFILL_CACHES.get(device.index or 0)
    if cache is None or cache.device != device:
        workspace = torch.empty(
            (_DEFAULT_FLASHINFER_WORKSPACE_BYTES,),
            device=device,
            dtype=torch.uint8,
        )
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
        )
        cache = _FlashInferPrefillCache(
            device=device,
            workspace=workspace,
            wrapper=wrapper,
        )
        _FLASHINFER_PREFILL_CACHES[device.index or 0] = cache

    lengths_cpu = lengths.to(device="cpu", non_blocking=True)
    sig = (
        int(batch_size),
        tuple(int(x) for x in lengths_cpu.tolist()),
        int(num_heads),
        int(head_dim),
        bool(causal),
        float(sm_scale),
        str(q_dtype),
    )
    if cache.last_sig != sig:
        qo_indptr = torch.zeros(
            (batch_size + 1,),
            device=device,
            dtype=torch.int32,
        )
        qo_indptr[1:].copy_(
            torch.cumsum(lengths, dim=0, dtype=torch.int32),
            non_blocking=True,
        )
        # self-attention: kv has the same ragged layout.
        kv_indptr = qo_indptr
        cache.wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=int(num_heads),
            num_kv_heads=int(num_heads),
            head_dim_qk=int(head_dim),
            causal=bool(causal),
            sm_scale=float(sm_scale),
            q_data_type=q_dtype,
            kv_data_type=q_dtype,
        )
        cache.last_sig = sig
    return cache.wrapper


def _flashinfer_paged_prefill_plan(
    *,
    device: torch.device,
    qo_indptr: torch.Tensor,  # [B+1] int32 cuda
    kv_indptr: torch.Tensor,  # [B+1] int32 cuda
    kv_indices: torch.Tensor,  # [kv_indptr[-1]] int32 cuda
    kv_last_page_len: torch.Tensor,  # [B] int32 cuda
    num_heads: int,
    head_dim: int,
    page_size: int,
    sm_scale: float,
    causal: bool,
    q_dtype: torch.dtype,
) -> Any:
    if flashinfer is None:
        raise RuntimeError(
            "flashinfer is not installed; install it to use attn_backend='flashinfer_paged'"
        )
    if device.type != "cuda":
        raise RuntimeError("flashinfer paged prefill attention backend requires CUDA")
    if q_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "flashinfer paged prefill attention backend requires fp16/bf16, "
            f"got dtype={q_dtype}"
        )
    if qo_indptr.dtype != torch.int32:
        raise ValueError("qo_indptr must be int32")
    if kv_indptr.dtype != torch.int32:
        raise ValueError("kv_indptr must be int32")
    if kv_indices.dtype != torch.int32:
        raise ValueError("kv_indices must be int32")
    if kv_last_page_len.dtype != torch.int32:
        raise ValueError("kv_last_page_len must be int32")

    cache = _FLASHINFER_PAGED_PREFILL_CACHES.get(device.index or 0)
    if cache is None or cache.device != device:
        workspace = torch.empty(
            (_DEFAULT_FLASHINFER_WORKSPACE_BYTES,),
            device=device,
            dtype=torch.uint8,
        )
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            kv_layout="HND",
        )
        cache = _FlashInferPagedPrefillCache(
            device=device,
            workspace=workspace,
            wrapper=wrapper,
        )
        _FLASHINFER_PAGED_PREFILL_CACHES[device.index or 0] = cache

    qo_lens = (qo_indptr[1:] - qo_indptr[:-1]).to(device="cpu", non_blocking=True)
    kv_lens = flashinfer.get_seq_lens(kv_indptr, kv_last_page_len, int(page_size)).to(
        device="cpu", non_blocking=True
    )
    kv_indices_cpu = kv_indices.to(device="cpu", non_blocking=True)
    sig = (
        tuple(int(x) for x in qo_lens.tolist()),
        tuple(int(x) for x in kv_lens.tolist()),
        tuple(int(x) for x in kv_indices_cpu.tolist()),
        int(num_heads),
        int(head_dim),
        int(page_size),
        bool(causal),
        float(sm_scale),
        str(q_dtype),
    )
    if cache.last_sig != sig:
        cache.wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=kv_last_page_len,
            num_qo_heads=int(num_heads),
            num_kv_heads=int(num_heads),
            head_dim_qk=int(head_dim),
            page_size=int(page_size),
            causal=bool(causal),
            sm_scale=float(sm_scale),
            q_data_type=q_dtype,
            kv_data_type=q_dtype,
        )
        cache.last_sig = sig
    return cache.wrapper


def plan_flashinfer_paged_prefill_wrapper(
    *,
    device: torch.device,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_heads: int,
    head_dim: int,
    page_size: int,
    sm_scale: float,
    causal: bool,
    q_dtype: torch.dtype,
) -> Any:
    return _flashinfer_paged_prefill_plan(
        device=device,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=kv_last_page_len,
        num_heads=int(num_heads),
        head_dim=int(head_dim),
        page_size=int(page_size),
        sm_scale=float(sm_scale),
        causal=bool(causal),
        q_dtype=q_dtype,
    )


def prefill_attention_flashinfer_paged(
    *,
    q: torch.Tensor,  # [B, H, T, D]
    k: torch.Tensor,  # [B, H, T, D]
    v: torch.Tensor,  # [B, H, T, D]
    attention_mask: torch.Tensor | None,  # [B, T] or None
    paged_kv_cache: Any,
    layer_idx: int,
    sm_scale: float,
    causal: bool,
) -> torch.Tensor:
    if flashinfer is None:
        raise RuntimeError(
            "flashinfer is not installed; install it to use attn_backend='flashinfer_paged'"
        )
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be 4D [B, H, T, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have the same shape")
    bsz, n_heads, seq_len, head_dim = q.shape
    device = q.device
    if device.type != "cuda":
        raise RuntimeError("flashinfer paged prefill attention backend requires CUDA")

    page_size = int(getattr(paged_kv_cache, "block_size"))
    slot_mapping = getattr(paged_kv_cache, "slot_mapping")
    context_lens = getattr(paged_kv_cache, "context_lens")
    block_tables = getattr(paged_kv_cache, "block_tables")
    k_cache = getattr(paged_kv_cache, "k_cache")
    v_cache = getattr(paged_kv_cache, "v_cache")

    if not isinstance(block_tables, list) or not block_tables:
        raise ValueError("paged_kv_cache.block_tables must be a non-empty list")
    if layer_idx < 0 or layer_idx >= len(block_tables):
        raise ValueError("layer_idx out of range for paged_kv_cache.block_tables")

    if slot_mapping.dtype != torch.int32:
        slot_mapping = slot_mapping.to(torch.int32)
    slot_mapping = slot_mapping.contiguous()
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)
    context_lens = context_lens.contiguous()

    qo_indptr = getattr(paged_kv_cache, "prefill_qo_indptr", None)
    if qo_indptr is not None:
        if qo_indptr.dtype != torch.int32:
            raise ValueError("paged_kv_cache.prefill_qo_indptr must be int32")
        if qo_indptr.dim() != 1 or int(qo_indptr.numel()) != int(bsz) + 1:
            raise ValueError("paged_kv_cache.prefill_qo_indptr shape mismatch")
        lengths = qo_indptr[1:] - qo_indptr[:-1]
    elif attention_mask is None:
        lengths = torch.full((int(bsz),), int(seq_len), device=device, dtype=torch.int32)
    else:
        mask = attention_mask.to(device=device, dtype=torch.bool, non_blocking=True)
        if (
            mask.dim() != 2
            or int(mask.size(0)) != int(bsz)
            or int(mask.size(1)) != int(seq_len)
        ):
            raise ValueError("attention_mask must have shape [B, T]")
        lengths = mask.sum(dim=1, dtype=torch.int32)

    if int(lengths.sum().item()) == 0:
        return torch.zeros_like(q)

    if qo_indptr is None:
        qo_indptr = torch.empty((int(bsz) + 1,), device=device, dtype=torch.int32)
        qo_indptr[0] = 0
        qo_indptr[1:].copy_(torch.cumsum(lengths, dim=0, dtype=torch.int32))
    nnz_qo = int(qo_indptr[-1].item())

    # kv lens for each request is context_lens (should already include appended tokens).
    kv_lens = context_lens.to(device=device, dtype=torch.int32, non_blocking=True)
    if kv_lens.numel() != int(bsz):
        raise ValueError("paged_kv_cache.context_lens must have shape [B]")

    kv_indptr = getattr(paged_kv_cache, "prefill_kv_indptr", None)
    kv_indices = getattr(paged_kv_cache, "prefill_kv_indices", None)
    kv_last_page_len = getattr(paged_kv_cache, "prefill_kv_last_page_len", None)
    if kv_indptr is not None or kv_indices is not None or kv_last_page_len is not None:
        if kv_indptr is None or kv_indices is None or kv_last_page_len is None:
            raise ValueError("incomplete paged_kv_cache prefill KV metadata")
        if kv_indptr.dtype != torch.int32:
            raise ValueError("paged_kv_cache.prefill_kv_indptr must be int32")
        if kv_indices.dtype != torch.int32:
            raise ValueError("paged_kv_cache.prefill_kv_indices must be int32")
        if kv_last_page_len.dtype != torch.int32:
            raise ValueError("paged_kv_cache.prefill_kv_last_page_len must be int32")
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()
        kv_last_page_len = kv_last_page_len.contiguous()
    else:
        # pages per request
        num_pages = (kv_lens + page_size - 1) // page_size  # [B]
        kv_indptr = torch.empty((int(bsz) + 1,), device=device, dtype=torch.int32)
        kv_indptr[0] = 0
        kv_indptr[1:].copy_(torch.cumsum(num_pages, dim=0, dtype=torch.int32))

        # kv_last_page_len: [B]
        kv_last_page_len = kv_lens - (num_pages - 1) * page_size
        kv_last_page_len = kv_last_page_len.to(torch.int32).contiguous()

        # Gather kv_indices from per-layer block table.
        block_table = block_tables[layer_idx]
        if block_table.dtype != torch.int32:
            block_table = block_table.to(torch.int32)
        block_table = block_table.contiguous()

        kv_indices_parts: list[torch.Tensor] = []
        for b in range(int(bsz)):
            n = int(num_pages[b].item())
            if n <= 0:
                continue
            slot = int(slot_mapping[b].item())
            kv_indices_parts.append(block_table[slot, :n])
        kv_indices = (
            torch.cat(kv_indices_parts, dim=0)
            if kv_indices_parts
            else torch.empty((0,), device=device, dtype=torch.int32)
        ).contiguous()
        if int(kv_indices.numel()) != int(kv_indptr[-1].item()):
            raise RuntimeError("kv_indices size mismatch (block table gather)")

    # Pack q/k/v by attention_mask to ragged tensor layout.
    q_flat = (
        q.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )
    k_flat = (
        k.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )
    v_flat = (
        v.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )
    if attention_mask is None:
        idx = torch.arange(int(bsz * seq_len), device=device, dtype=torch.long)
    else:
        idx = (
            attention_mask.to(device=device, dtype=torch.bool, non_blocking=True)
            .reshape(-1)
            .nonzero(as_tuple=False)
            .squeeze(-1)
            .to(dtype=torch.long)
        )
    q_ragged = q_flat.index_select(0, idx)
    k_ragged = k_flat.index_select(0, idx)
    v_ragged = v_flat.index_select(0, idx)
    if int(q_ragged.size(0)) != nnz_qo:
        raise RuntimeError("ragged packing mismatch (q nnz != qo_indptr[-1])")

    # Append KV into paged cache before running attention.
    batch_idx = getattr(paged_kv_cache, "prefill_batch_idx", None)
    pos = getattr(paged_kv_cache, "prefill_pos", None)
    if batch_idx is not None or pos is not None:
        if batch_idx is None or pos is None:
            raise ValueError("incomplete paged_kv_cache prefill token positions")
        batch_idx = batch_idx.contiguous()
        pos = pos.contiguous()
    else:
        batch_idx, pos = flashinfer.get_batch_indices_positions(
            qo_indptr,
            kv_lens,
            nnz_qo,
        )
    k_layer = k_cache[layer_idx]
    v_layer = v_cache[layer_idx]
    flashinfer.append_paged_kv_cache(
        k_ragged,
        v_ragged,
        batch_idx,
        pos,
        (k_layer, v_layer),
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        kv_layout="HND",
    )

    wrapper = getattr(paged_kv_cache, "prefill_wrapper", None)
    if wrapper is None:
        wrapper = _flashinfer_paged_prefill_plan(
            device=device,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last_page_len,
            num_heads=int(n_heads),
            head_dim=int(head_dim),
            page_size=int(page_size),
            sm_scale=float(sm_scale),
            causal=bool(causal),
            q_dtype=q.dtype,
        )
    o_ragged = wrapper.run(q_ragged, (k_layer, v_layer))
    o_flat = q_flat.new_zeros((int(bsz * seq_len), int(n_heads), int(head_dim)))
    o_flat.index_copy_(0, idx, o_ragged)
    return (
        o_flat.view(int(bsz), int(seq_len), int(n_heads), int(head_dim))
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def prefill_attention_flashinfer_paged_varlen(
    *,
    q: torch.Tensor,  # [N, H, D]
    k: torch.Tensor,  # [N, H, D]
    v: torch.Tensor,  # [N, H, D]
    paged_kv_cache: Any,
    layer_idx: int,
    sm_scale: float,
    causal: bool,
) -> torch.Tensor:
    if flashinfer is None:
        raise RuntimeError(
            "flashinfer is not installed; install it to use paged varlen prefill"
        )
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q/k/v must be 3D [N, H, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have the same shape")
    nnz_qo, n_heads, head_dim = q.shape
    device = q.device
    if device.type != "cuda":
        raise RuntimeError("paged varlen prefill requires CUDA")

    page_size = int(getattr(paged_kv_cache, "block_size"))
    context_lens = getattr(paged_kv_cache, "context_lens")
    k_cache = getattr(paged_kv_cache, "k_cache")
    v_cache = getattr(paged_kv_cache, "v_cache")

    qo_indptr = getattr(paged_kv_cache, "prefill_qo_indptr", None)
    kv_indptr = getattr(paged_kv_cache, "prefill_kv_indptr", None)
    kv_indices = getattr(paged_kv_cache, "prefill_kv_indices", None)
    kv_last_page_len = getattr(paged_kv_cache, "prefill_kv_last_page_len", None)
    if qo_indptr is None:
        raise ValueError("paged_kv_cache.prefill_qo_indptr is required for varlen")
    if kv_indptr is None or kv_indices is None or kv_last_page_len is None:
        raise ValueError("paged_kv_cache prefill KV metadata is required for varlen")
    if qo_indptr.dtype != torch.int32:
        raise ValueError("paged_kv_cache.prefill_qo_indptr must be int32")
    if kv_indptr.dtype != torch.int32:
        raise ValueError("paged_kv_cache.prefill_kv_indptr must be int32")
    if kv_indices.dtype != torch.int32:
        raise ValueError("paged_kv_cache.prefill_kv_indices must be int32")
    if kv_last_page_len.dtype != torch.int32:
        raise ValueError("paged_kv_cache.prefill_kv_last_page_len must be int32")
    qo_indptr = qo_indptr.contiguous()
    kv_indptr = kv_indptr.contiguous()
    kv_indices = kv_indices.contiguous()
    kv_last_page_len = kv_last_page_len.contiguous()

    bsz = int(qo_indptr.numel()) - 1
    if bsz <= 0:
        raise ValueError("empty qo_indptr")
    if int(qo_indptr[-1].item()) != int(nnz_qo):
        raise ValueError("q size mismatch (nnz != qo_indptr[-1])")

    kv_lens = context_lens.to(device=device, dtype=torch.int32, non_blocking=True)
    if int(kv_lens.numel()) != int(bsz):
        raise ValueError("paged_kv_cache.context_lens must have shape [B]")

    batch_idx = getattr(paged_kv_cache, "prefill_batch_idx", None)
    pos = getattr(paged_kv_cache, "prefill_pos", None)
    if batch_idx is not None or pos is not None:
        if batch_idx is None or pos is None:
            raise ValueError("incomplete paged_kv_cache prefill token positions")
        batch_idx = batch_idx.contiguous()
        pos = pos.contiguous()
    else:
        batch_idx, pos = flashinfer.get_batch_indices_positions(
            qo_indptr,
            kv_lens,
            int(nnz_qo),
        )

    k_layer = k_cache[layer_idx]
    v_layer = v_cache[layer_idx]
    flashinfer.append_paged_kv_cache(
        k,
        v,
        batch_idx,
        pos,
        (k_layer, v_layer),
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        kv_layout="HND",
    )

    wrapper = getattr(paged_kv_cache, "prefill_wrapper", None)
    if wrapper is None:
        wrapper = _flashinfer_paged_prefill_plan(
            device=device,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last_page_len,
            num_heads=int(n_heads),
            head_dim=int(head_dim),
            page_size=int(page_size),
            sm_scale=float(sm_scale),
            causal=bool(causal),
            q_dtype=q.dtype,
        )
    return wrapper.run(q, (k_layer, v_layer))


def _ragged_token_indices(
    *,
    attention_mask: Optional[torch.Tensor],  # [B, T] or None
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_mask is None:
        lengths = torch.full(
            (batch_size,),
            int(seq_len),
            device=device,
            dtype=torch.int32,
        )
        idx = torch.arange(
            int(batch_size * seq_len),
            device=device,
            dtype=torch.long,
        )
        return idx, lengths

    mask = attention_mask.to(device=device, dtype=torch.bool, non_blocking=True)
    if (
        mask.dim() != 2
        or int(mask.size(0)) != int(batch_size)
        or int(mask.size(1)) != int(seq_len)
    ):
        raise ValueError("attention_mask must have shape [B, T]")
    lengths = mask.sum(dim=1, dtype=torch.int32)
    idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1).to(dtype=torch.long)
    if idx.numel() != int(lengths.sum().item()):
        raise RuntimeError("ragged packing mismatch (mask nonzero != sum(lengths))")
    return idx, lengths


def prefill_attention_flashinfer(
    *,
    q: torch.Tensor,  # [B, H, T, D]
    k: torch.Tensor,  # [B, H, T, D]
    v: torch.Tensor,  # [B, H, T, D]
    attention_mask: Optional[torch.Tensor],  # [B, T] or None
    sm_scale: float,
    causal: bool,
) -> torch.Tensor:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be 4D [B, H, T, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have the same shape")
    bsz, n_heads, seq_len, head_dim = q.shape
    device = q.device
    if attention_mask is None:
        lengths = torch.full(
            (int(bsz),),
            int(seq_len),
            device=device,
            dtype=torch.int32,
        )
        idx = None
    else:
        idx, lengths = _ragged_token_indices(
            attention_mask=attention_mask,
            batch_size=int(bsz),
            seq_len=int(seq_len),
            device=device,
        )
        if idx.numel() == 0:
            return torch.zeros_like(q)

    wrapper = _flashinfer_prefill_plan(
        device=device,
        batch_size=int(bsz),
        lengths=lengths,
        num_heads=int(n_heads),
        head_dim=int(head_dim),
        sm_scale=float(sm_scale),
        causal=bool(causal),
        q_dtype=q.dtype,
    )

    # [B, T, H, D] -> [B*T, H, D]
    q_flat = (
        q.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )
    k_flat = (
        k.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )
    v_flat = (
        v.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )

    if idx is None:
        o_flat = wrapper.run(q_flat, k_flat, v_flat)
    else:
        q_ragged = q_flat.index_select(0, idx)
        k_ragged = k_flat.index_select(0, idx)
        v_ragged = v_flat.index_select(0, idx)

        o_ragged = wrapper.run(q_ragged, k_ragged, v_ragged)
        o_flat = q_flat.new_zeros((int(bsz * seq_len), int(n_heads), int(head_dim)))
        o_flat.index_copy_(0, idx, o_ragged)
    # [B, T, H, D] -> [B, H, T, D]
    return (
        o_flat.view(int(bsz), int(seq_len), int(n_heads), int(head_dim))
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def prefill_attention_flashattn(
    *,
    q: torch.Tensor,  # [B, H, T, D]
    k: torch.Tensor,  # [B, H, T, D]
    v: torch.Tensor,  # [B, H, T, D]
    attention_mask: Optional[torch.Tensor],  # [B, T] or None
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    if flash_attn_varlen_func is None:
        raise RuntimeError(
            "flash-attn is not installed; install it to use attn_backend='flashattn'"
        )
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be 4D [B, H, T, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have the same shape")
    bsz, n_heads, seq_len, head_dim = q.shape
    device = q.device
    if device.type != "cuda":
        raise RuntimeError("flash-attn attention backend requires CUDA")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"flash-attn attention backend requires fp16/bf16, got dtype={q.dtype}"
        )
    if attention_mask is None:
        lengths = torch.full(
            (int(bsz),),
            int(seq_len),
            device=device,
            dtype=torch.int32,
        )
        idx = None
    else:
        idx, lengths = _ragged_token_indices(
            attention_mask=attention_mask,
            batch_size=int(bsz),
            seq_len=int(seq_len),
            device=device,
        )
        if idx.numel() == 0:
            return torch.zeros_like(q)
    cu_seqlens = torch.zeros(
        (int(bsz) + 1,),
        device=device,
        dtype=torch.int32,
    )
    cu_seqlens[1:].copy_(
        torch.cumsum(lengths, dim=0, dtype=torch.int32),
        non_blocking=True,
    )
    max_seqlen = int(lengths.max().item())

    q_flat = (
        q.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )
    k_flat = (
        k.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )
    v_flat = (
        v.permute(0, 2, 1, 3)
        .contiguous()
        .view(int(bsz * seq_len), int(n_heads), int(head_dim))
    )

    if idx is None:
        o_flat = flash_attn_varlen_func(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=bool(causal),
            softmax_scale=float(softmax_scale),
        )
    else:
        q_ragged = q_flat.index_select(0, idx)
        k_ragged = k_flat.index_select(0, idx)
        v_ragged = v_flat.index_select(0, idx)
        o_ragged = flash_attn_varlen_func(
            q_ragged,
            k_ragged,
            v_ragged,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=bool(causal),
            softmax_scale=float(softmax_scale),
        )
        o_flat = q_flat.new_zeros((int(bsz * seq_len), int(n_heads), int(head_dim)))
        o_flat.index_copy_(0, idx, o_ragged)

    return (
        o_flat.view(int(bsz), int(seq_len), int(n_heads), int(head_dim))
        .permute(0, 2, 1, 3)
        .contiguous()
    )
