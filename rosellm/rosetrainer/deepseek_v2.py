from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from rosellm.rosetrainer.attention_backends import (
    prefill_attention_flashattn,
    prefill_attention_flashinfer,
    prefill_attention_flashinfer_paged,
)
from rosellm.rosetrainer.paged_attention import (
    TRITON_AVAILABLE,
    PagedKVCache,
    paged_attention_decode_ref,
    paged_attention_decode_triton,
)


paged_attention_decode = (
    paged_attention_decode_triton if TRITON_AVAILABLE else paged_attention_decode_ref
)


def _act_fn(name: str):
    act = str(name or "silu").lower()
    if act in ("silu", "swish"):
        return F.silu
    if act == "gelu":
        return F.gelu
    raise ValueError(f"unsupported hidden_act: {name}")


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(hidden_size)))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        variance = hs.pow(2).mean(dim=-1, keepdim=True)
        hs = hs * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hs.to(dtype=input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(int(unsqueeze_dim))
    sin = sin.unsqueeze(int(unsqueeze_dim))
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, *, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        head_dim = int(head_dim)
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        inv_freq = 1.0 / (
            float(base)
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float32)
                / float(head_dim)
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,  # [B, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=x.device)
        inv_freq_expanded = inv_freq[None, :, None].expand(
            int(position_ids.size(0)), -1, 1
        )
        pos = position_ids[:, None, :].to(dtype=torch.float32)
        device_type = x.device.type
        if not isinstance(device_type, str) or device_type == "mps":
            device_type = "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ pos).transpose(1, 2)  # [B, T, D/2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [B, T, D]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        hs = int(getattr(config, "hidden_size")) if hidden_size is None else int(hidden_size)
        inter = (
            int(getattr(config, "intermediate_size"))
            if intermediate_size is None
            else int(intermediate_size)
        )
        self.gate_proj = nn.Linear(hs, inter, bias=False)
        self.up_proj = nn.Linear(hs, inter, bias=False)
        self.down_proj = nn.Linear(inter, hs, bias=False)
        self.act_fn = _act_fn(str(getattr(config, "hidden_act", "silu")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.top_k = int(getattr(config, "num_experts_per_tok"))
        self.n_routed_experts = int(getattr(config, "n_routed_experts"))
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))
        self.scoring_func = str(getattr(config, "scoring_func", "softmax"))
        self.topk_method = str(getattr(config, "topk_method", "greedy"))
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", False))
        gating_dim = int(getattr(config, "hidden_size"))
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, gating_dim)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # hidden_states: [B, T, H]
        bsz, seq_len, h = hidden_states.shape
        hs = hidden_states.view(-1, h)
        logits = F.linear(hs.to(torch.float32), self.weight.to(torch.float32), None)
        if self.scoring_func != "softmax":
            raise NotImplementedError(
                f"unsupported MoE gating scoring_func={self.scoring_func!r}"
            )
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        if self.topk_method != "greedy":
            raise NotImplementedError(
                f"unsupported MoE gating topk_method={self.topk_method!r}"
            )
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denom = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denom
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        aux_loss = None
        return topk_idx, topk_weight, aux_loss


class DeepseekV2MoE(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.num_experts_per_tok = int(getattr(config, "num_experts_per_tok"))
        self.n_routed_experts = int(getattr(config, "n_routed_experts"))

        ep_size = int(getattr(config, "ep_size", 1) or 1)
        if ep_size > 1:
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError("ep_size>1 requires torch.distributed to be initialized")
            if ep_size != int(dist.get_world_size()):
                raise ValueError("config.ep_size must equal dist world_size")
            self.ep_size = ep_size
            self.ep_rank = int(dist.get_rank())
            if self.n_routed_experts % ep_size != 0:
                raise ValueError("n_routed_experts must be divisible by ep_size")
            self.experts_per_rank = self.n_routed_experts // ep_size
            self.experts = nn.ModuleList(
                [
                    (
                        DeepseekV2MLP(
                            config,
                            intermediate_size=int(getattr(config, "moe_intermediate_size")),
                        )
                        if (
                            i >= self.ep_rank * self.experts_per_rank
                            and i < (self.ep_rank + 1) * self.experts_per_rank
                        )
                        else None
                    )
                    for i in range(self.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.ep_rank = 0
            self.experts_per_rank = self.n_routed_experts
            self.experts = nn.ModuleList(
                [
                    DeepseekV2MLP(
                        config,
                        intermediate_size=int(getattr(config, "moe_intermediate_size")),
                    )
                    for _ in range(self.n_routed_experts)
                ]
            )

        self.gate = MoEGate(config)
        self.n_shared_experts = getattr(config, "n_shared_experts", None)
        if self.n_shared_experts is not None:
            inter = int(getattr(config, "moe_intermediate_size")) * int(self.n_shared_experts)
            self.shared_experts = DeepseekV2MLP(config, intermediate_size=inter)
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, _aux_loss = self.gate(hidden_states)
        hs = hidden_states.view(-1, hidden_states.shape[-1])
        if self.training or hs.requires_grad:
            raise RuntimeError("DeepseekV2MoE only supports inference in roseinfer")
        y = self.moe_infer(hs, topk_idx, topk_weight).view(*orig_shape)
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        # x: [N, H], topk_ids: [N, K], topk_weight: [N, K]
        cnts = topk_ids.new_zeros((int(topk_ids.size(0)), len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)

        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // int(topk_ids.size(1))]
        sorted_tokens_shape = sorted_tokens.shape

        gatherd_idxs: np.ndarray | None = None
        input_split_sizes: list[int] | None = None
        output_splits: list[int] | None = None

        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
            output_splits = (
                tokens_per_expert_group.view(self.ep_size, -1)
                .sum(1)
                .cpu()
                .numpy()
                .tolist()
            )
            gathered_tokens = sorted_tokens.new_empty(
                int(tokens_per_expert_group.sum(dim=0).cpu().item()),
                int(sorted_tokens.shape[1]),
            )
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
            dist.all_to_all(
                list(gathered_tokens.split(output_splits)),
                list(sorted_tokens.split(input_split_sizes)),
            )
            tokens_per_expert_post_gather = tokens_per_expert_group.view(
                self.ep_size, self.experts_per_rank
            ).sum(dim=0)
            gatherd_idxs_np = np.zeros(shape=(int(gathered_tokens.shape[0]),), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs_np[s : s + int(k)] = i % self.experts_per_rank
                s += int(k)
            gatherd_idxs_np = gatherd_idxs_np.argsort()
            gatherd_idxs = gatherd_idxs_np
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather

        tokens_per_expert_np = tokens_per_expert.cpu().numpy()
        outputs: list[torch.Tensor] = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert_np.tolist()):
            end_idx = start_idx + int(num_tokens)
            if int(num_tokens) == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            if expert is None:
                raise RuntimeError("missing local expert module")
            tokens_for_expert = sorted_tokens[start_idx:end_idx]
            outputs.append(expert(tokens_for_expert))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
        if self.ep_size > 1:
            assert gatherd_idxs is not None
            assert input_split_sizes is not None
            assert output_splits is not None
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
            dist.all_to_all(
                list(gathered_tokens.split(input_split_sizes)),
                list(new_x.split(output_splits)),
            )
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class DeepseekV2Attention(nn.Module):
    def __init__(self, config: Any, *, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.num_heads = int(getattr(config, "num_attention_heads"))
        self.kv_lora_rank = int(getattr(config, "kv_lora_rank"))
        self.qk_rope_head_dim = int(getattr(config, "qk_rope_head_dim"))
        self.qk_nope_head_dim = int(getattr(config, "qk_nope_head_dim"))
        self.v_head_dim = int(getattr(config, "v_head_dim"))
        self.q_head_dim = int(self.qk_nope_head_dim + self.qk_rope_head_dim)
        self.scaling = float(self.q_head_dim**-0.5)
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))
        eps = float(getattr(config, "rms_norm_eps", 1e-6))

        bias = bool(getattr(config, "attention_bias", False))
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.q_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(self.kv_lora_rank, eps=eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=bias,
        )
        self.rotary_emb = DeepseekV2RotaryEmbedding(
            head_dim=self.qk_rope_head_dim,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, H]
        *,
        attention_mask: Optional[torch.Tensor],  # [B, T] or None
        position_ids: torch.Tensor,  # [B, T]
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]],
        return_kv: bool,
        paged_kv_cache: Optional[PagedKVCache],
        attn_backend: str | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(
            int(bsz), int(seq_len), int(self.num_heads), int(self.q_head_dim)
        )
        q = q.transpose(1, 2)  # [B, H, T, Dq]
        q_nope, q_pe = torch.split(
            q,
            [int(self.qk_nope_head_dim), int(self.qk_rope_head_dim)],
            dim=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv,
            [int(self.kv_lora_rank), int(self.qk_rope_head_dim)],
            dim=-1,
        )
        k_pe = k_pe.view(int(bsz), int(seq_len), 1, int(self.qk_rope_head_dim)).transpose(
            1, 2
        )  # [B, 1, T, Drope]
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(
                int(bsz),
                int(seq_len),
                int(self.num_heads),
                int(self.qk_nope_head_dim + self.v_head_dim),
            )
            .transpose(1, 2)
        )  # [B, H, T, Dk_nope + Dv]
        k_nope, v = torch.split(
            kv,
            [int(self.qk_nope_head_dim), int(self.v_head_dim)],
            dim=-1,
        )

        cos, sin = self.rotary_emb(hidden_states, position_ids=position_ids)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        query_states = k_pe.new_empty(int(bsz), int(self.num_heads), int(seq_len), int(self.q_head_dim))
        query_states[:, :, :, : int(self.qk_nope_head_dim)] = q_nope
        query_states[:, :, :, int(self.qk_nope_head_dim) :] = q_pe

        key_states = k_pe.new_empty(int(bsz), int(self.num_heads), int(seq_len), int(self.q_head_dim))
        key_states[:, :, :, : int(self.qk_nope_head_dim)] = k_nope
        key_states[:, :, :, int(self.qk_nope_head_dim) :] = k_pe

        # Pad V so KV cache head_dim matches q_head_dim (paged kernels assume a
        # single D for K/V).
        if int(self.v_head_dim) != int(self.q_head_dim):
            v = F.pad(v, (0, int(self.q_head_dim - self.v_head_dim)))

        backend = str(attn_backend or "naive").lower()
        if paged_kv_cache is not None:
            if past_kv is not None:
                raise ValueError("past_kv and paged_kv_cache cannot be used together")
            if int(seq_len) != 1:
                if backend not in ("flashinfer_paged", "flashinfer-paged"):
                    raise ValueError(
                        "paged prefill requires attn_backend='flashinfer_paged' "
                        f"(got attn_backend={attn_backend})"
                    )
                if self.training or hidden_states.requires_grad:
                    raise ValueError("paged attention only supports inference")
                attn_out = prefill_attention_flashinfer_paged(
                    q=query_states,
                    k=key_states,
                    v=v,
                    attention_mask=attention_mask,
                    paged_kv_cache=paged_kv_cache,
                    layer_idx=int(self.layer_idx),
                    sm_scale=float(self.scaling),
                    causal=True,
                )
                # [B, H, T, D] -> [B, T, H*Dv]
                if int(self.v_head_dim) != int(self.q_head_dim):
                    attn_out = attn_out[..., : int(self.v_head_dim)]
                attn_out = attn_out.transpose(1, 2).contiguous()
                attn_out = attn_out.view(int(bsz), int(seq_len), int(self.num_heads) * int(self.v_head_dim))
                out = self.o_proj(attn_out)
                if return_kv:
                    return out, (key_states, v)
                return out

            q_step = query_states.squeeze(-2)  # [B, H, D]
            k_new = key_states.squeeze(-2)  # [B, H, D]
            v_new = v.squeeze(-2)  # [B, H, D]
            out_h = paged_attention_decode(
                q=q_step,
                k_new=k_new,
                v_new=v_new,
                k_cache_layer=paged_kv_cache.k_cache[self.layer_idx],
                v_cache_layer=paged_kv_cache.v_cache[self.layer_idx],
                block_table=paged_kv_cache.block_tables[self.layer_idx],
                slot_mapping=paged_kv_cache.slot_mapping,
                context_lens=paged_kv_cache.context_lens,
                scale=float(self.scaling),
                block_size=int(paged_kv_cache.block_size),
                write_kv=bool(paged_kv_cache.write_kv),
            )  # [B, H, D]
            attn_out = out_h.unsqueeze(2)  # [B, H, 1, D]
            if int(self.v_head_dim) != int(self.q_head_dim):
                attn_out = attn_out[..., : int(self.v_head_dim)]
            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_out = attn_out.view(int(bsz), int(seq_len), int(self.num_heads) * int(self.v_head_dim))
            out = self.o_proj(attn_out)
            if return_kv:
                return out, (key_states, v)
            return out

        if past_kv is not None:
            past_k, past_v = past_kv
            key_states = torch.cat([past_k, key_states], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # Prefill attention backends (past_kv must be None).
        if backend in ("flashinfer", "flashattn") and past_kv is None:
            if self.training or hidden_states.requires_grad:
                raise ValueError(f"{backend} attention backend only supports inference")
            if backend == "flashinfer":
                attn_out = prefill_attention_flashinfer(
                    q=query_states,
                    k=key_states,
                    v=v,
                    attention_mask=attention_mask,
                    sm_scale=float(self.scaling),
                    causal=True,
                )
            else:
                attn_out = prefill_attention_flashattn(
                    q=query_states,
                    k=key_states,
                    v=v,
                    attention_mask=attention_mask,
                    softmax_scale=float(self.scaling),
                    causal=True,
                )
        else:
            attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) * float(
                self.scaling
            )
            full_seq_len = int(key_states.size(-2))
            causal = torch.arange(
                full_seq_len,
                device=attn_scores.device,
            )[None, :] <= (
                full_seq_len - int(seq_len)
                + torch.arange(int(seq_len), device=attn_scores.device)[:, None]
            )
            attn_scores = attn_scores.masked_fill(~causal, float("-inf"))
            if attention_mask is not None:
                mask_value = torch.finfo(attn_scores.dtype).min
                attn_mask = attention_mask[:, None, None, :full_seq_len]
                attn_scores = attn_scores.masked_fill(attn_mask == 0, mask_value)
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_out = torch.matmul(attn_weights, v)

        # [B, H, T, D] -> [B, T, H*Dv]
        if int(self.v_head_dim) != int(self.q_head_dim):
            attn_out = attn_out[..., : int(self.v_head_dim)]
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(int(bsz), int(seq_len), int(self.num_heads) * int(self.v_head_dim))
        out = self.o_proj(attn_out)
        if not return_kv:
            return out
        return out, (key_states, v)


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: Any, *, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = DeepseekV2RMSNorm(int(getattr(config, "hidden_size")), eps=eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(int(getattr(config, "hidden_size")), eps=eps)
        self.self_attn = DeepseekV2Attention(config, layer_idx=self.layer_idx)
        is_moe = (
            getattr(config, "n_routed_experts", None) is not None
            and self.layer_idx >= int(getattr(config, "first_k_dense_replace", 0) or 0)
            and (self.layer_idx % int(getattr(config, "moe_layer_freq", 1) or 1) == 0)
        )
        self.mlp = DeepseekV2MoE(config) if is_moe else DeepseekV2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]],
        return_kv: bool,
        paged_kv_cache: Optional[PagedKVCache],
        attn_backend: str | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | torch.Tensor:
        residual = hidden_states
        hs = self.input_layernorm(hidden_states)
        if return_kv:
            attn_out, present = self.self_attn(
                hs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_kv=past_kv,
                return_kv=True,
                paged_kv_cache=paged_kv_cache,
                attn_backend=attn_backend,
            )
            hidden_states = residual + attn_out
            mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = hidden_states + mlp_out
            return hidden_states, present
        attn_out = self.self_attn(
            hs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_kv=past_kv,
            return_kv=False,
            paged_kv_cache=paged_kv_cache,
            attn_backend=attn_backend,
        )
        hidden_states = residual + attn_out
        mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + mlp_out
        return hidden_states


class DeepseekV2Model(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            int(getattr(config, "vocab_size")),
            int(getattr(config, "hidden_size")),
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(config, layer_idx=i)
                for i in range(int(getattr(config, "num_hidden_layers")))
            ]
        )
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.norm = DeepseekV2RMSNorm(int(getattr(config, "hidden_size")), eps=eps)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T]
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        paged_kv_cache: Optional[PagedKVCache] = None,
        attn_backend: str | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        if position_ids is None:
            seq_len = int(input_ids.size(1))
            position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
            if int(position_ids.size(0)) != int(input_ids.size(0)):
                position_ids = position_ids.expand(int(input_ids.size(0)), -1).contiguous()

        hidden_states = self.embed_tokens(input_ids)
        presents: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = None
            if past_key_values is not None and i < len(past_key_values):
                past_kv = past_key_values[i]
            if use_cache:
                hidden_states, present = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_kv=past_kv,
                    return_kv=True,
                    paged_kv_cache=paged_kv_cache,
                    attn_backend=attn_backend,
                )
                assert presents is not None
                presents.append(present)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_kv=past_kv,
                    return_kv=False,
                    paged_kv_cache=paged_kv_cache,
                    attn_backend=attn_backend,
                )
        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class DeepseekV2ForCausalLM(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.model = DeepseekV2Model(config)
        self.lm_head = nn.Linear(
            int(getattr(config, "hidden_size")),
            int(getattr(config, "vocab_size")),
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T]
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        paged_kv_cache: Optional[PagedKVCache] = None,
        attn_backend: str | None = None,
        logits_positions: Optional[torch.Tensor] = None,
    ):
        hidden_states, presents = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            paged_kv_cache=paged_kv_cache,
            attn_backend=attn_backend,
        )
        if logits_positions is not None:
            if labels is not None:
                raise ValueError("logits_positions is not supported with labels")
            bsz = int(hidden_states.size(0))
            seq_len = int(hidden_states.size(1))
            logits_positions = logits_positions.to(device=hidden_states.device, dtype=torch.long)
            if logits_positions.dim() != 1 or int(logits_positions.numel()) != bsz:
                raise ValueError("logits_positions must be a 1D tensor of shape [B]")
            if int(logits_positions.min().item()) < 0 or int(logits_positions.max().item()) >= seq_len:
                raise ValueError("logits_positions out of range")
            row = torch.arange(bsz, device=hidden_states.device, dtype=torch.long)
            hs = hidden_states[row, logits_positions, :]
            logits = self.lm_head(hs).unsqueeze(1)
        else:
            logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, int(getattr(self.config, "vocab_size"))),
                shift_labels.view(-1),
            )
        if use_cache:
            return logits, loss, presents
        return logits, loss

