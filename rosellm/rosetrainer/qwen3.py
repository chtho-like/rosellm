from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rosellm.rosetrainer.attention_backends import (
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


class Qwen3RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float = 1e-6,
    ) -> None:
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


def repeat_kv(
    hidden_states: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    n_rep = int(n_rep)
    if n_rep == 1:
        return hidden_states
    hs = hidden_states[:, :, None, :, :].expand(
        batch,
        num_kv_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hs.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if getattr(config, "rope_scaling", None) is not None:
            raise ValueError("rope_scaling is not supported yet")
        head_dim = int(getattr(config, "head_dim", 0) or 0)
        if head_dim <= 0:
            head_dim = int(config.hidden_size) // int(config.num_attention_heads)
        base = float(getattr(config, "rope_theta", 10000.0))
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.int64,
                    device=device,
                ).to(dtype=torch.float32)
                / float(head_dim)
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,  # [B, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq
        inv_freq_expanded = (
            inv_freq[None, :, None]
            .float()
            .expand(int(position_ids.size(0)), -1, 1)
            .to(device=x.device)
        )
        pos = position_ids[:, None, :].float()
        device_type = x.device.type
        if not isinstance(device_type, str) or device_type == "mps":
            device_type = "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ pos).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * float(self.attention_scaling)
            sin = emb.sin() * float(self.attention_scaling)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = _act_fn(str(getattr(config, "hidden_act", "silu")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(config, "head_dim", 0) or (self.hidden_size // self.num_attention_heads)
        )
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = float(self.head_dim**-0.5)

        bias = bool(getattr(config, "attention_bias", False))
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=bias,
        )
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=eps)

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,  # [B, T, H]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(
            int(bsz),
            int(seq_len),
            int(self.num_attention_heads),
            int(self.head_dim),
        )
        k = self.k_proj(hidden_states).view(
            int(bsz),
            int(seq_len),
            int(self.num_key_value_heads),
            int(self.head_dim),
        )
        v = self.v_proj(hidden_states).view(
            int(bsz),
            int(seq_len),
            int(self.num_key_value_heads),
            int(self.head_dim),
        )
        q = self.q_norm(q).transpose(1, 2)  # [B, Hq, T, D]
        k = self.k_norm(k).transpose(1, 2)  # [B, Hkv, T, D]
        v = v.transpose(1, 2)  # [B, Hkv, T, D]
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, H]
        *,
        cos: torch.Tensor,  # [B, T, D]
        sin: torch.Tensor,  # [B, T, D]
        attention_mask: Optional[torch.Tensor],  # [B, T] or [B, S]
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False,
        paged_kv_cache: Optional[PagedKVCache] = None,
        position_ids: Optional[torch.Tensor] = None,  # unused; cos/sin already carry positions
        attn_backend: str | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | torch.Tensor:
        del position_ids
        q, k, v = self._project_qkv(hidden_states)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        bsz, _, seq_len, _ = q.shape

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
                    raise ValueError(
                        "flashinfer_paged attention backend only supports inference"
                    )
                attn_out = prefill_attention_flashinfer_paged(
                    q=q,
                    k=k,
                    v=v,
                    attention_mask=attention_mask,
                    paged_kv_cache=paged_kv_cache,
                    layer_idx=int(self.layer_idx),
                    sm_scale=float(self.scaling),
                    causal=True,
                )
                attn_out = attn_out.transpose(1, 2).contiguous()
                attn_out = attn_out.view(int(bsz), int(seq_len), -1)
                out = self.o_proj(attn_out)
                if return_kv:
                    return out, (k, v)
                return out

            q_step = q.squeeze(-2)  # [B, Hq, D]
            k_new = k.squeeze(-2)  # [B, Hkv, D]
            v_new = v.squeeze(-2)  # [B, Hkv, D]
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
            )  # [B, Hq, D]
            attn_out = out_h.unsqueeze(2)  # [B, Hq, 1, D]
            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_out = attn_out.view(int(bsz), int(seq_len), -1)
            out = self.o_proj(attn_out)
            if return_kv:
                return out, (k, v)
            return out

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        if backend == "flashinfer" and past_kv is None:
            if self.training or hidden_states.requires_grad:
                raise ValueError("flashinfer attention backend only supports inference")
            attn_out = prefill_attention_flashinfer(
                q=q,
                k=k,
                v=v,
                attention_mask=attention_mask,
                sm_scale=float(self.scaling),
                causal=True,
            )
        else:
            key_states = repeat_kv(k, self.num_key_value_groups)
            value_states = repeat_kv(v, self.num_key_value_groups)
            attn_scores = torch.matmul(
                q,
                key_states.transpose(2, 3),
            ) * float(self.scaling)
            full_seq_len = int(key_states.size(-2))
            causal = torch.arange(
                full_seq_len,
                device=attn_scores.device,
            )[None, :] <= (
                full_seq_len - int(seq_len)
                + torch.arange(
                    int(seq_len),
                    device=attn_scores.device,
                )[:, None]
            )
            attn_scores = attn_scores.masked_fill(~causal, float("-inf"))
            if attention_mask is not None:
                mask_value = torch.finfo(attn_scores.dtype).min
                attn_mask = attention_mask[:, None, None, :full_seq_len]
                attn_scores = attn_scores.masked_fill(attn_mask == 0, mask_value)
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
                q.dtype
            )
            attn_out = torch.matmul(attn_weights, value_states)

        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(int(bsz), int(seq_len), -1)
        out = self.o_proj(attn_out)
        if not return_kv:
            return out
        return out, (k, v)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Any, *, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = Qwen3RMSNorm(int(config.hidden_size), eps=eps)
        self.post_attention_layernorm = Qwen3RMSNorm(int(config.hidden_size), eps=eps)
        self.self_attn = Qwen3Attention(config, layer_idx=self.layer_idx)
        self.mlp = Qwen3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]],
        return_kv: bool,
        paged_kv_cache: Optional[PagedKVCache],
        attn_backend: str | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if return_kv:
            attn_out, present = self.self_attn(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                past_kv=past_kv,
                return_kv=True,
                paged_kv_cache=paged_kv_cache,
                attn_backend=attn_backend,
            )
            hidden_states = residual + attn_out
        else:
            attn_out = self.self_attn(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                past_kv=past_kv,
                return_kv=False,
                paged_kv_cache=paged_kv_cache,
                attn_backend=attn_backend,
            )
            hidden_states = residual + attn_out
            present = None

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if not return_kv:
            return hidden_states
        assert present is not None
        return hidden_states, present


class Qwen3Model(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(int(config.vocab_size), int(config.hidden_size))
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx=i)
                for i in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = Qwen3RMSNorm(
            int(config.hidden_size),
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config)

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
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            if attention_mask is not None:
                pos = attention_mask.to(dtype=torch.long).cumsum(-1) - 1
                pos.masked_fill_(attention_mask == 0, 0)
                position_ids = pos
            else:
                past_len = 0
                if past_key_values is not None and past_key_values:
                    past_len = int(past_key_values[0][0].size(-2))
                position_ids = torch.arange(
                    past_len,
                    past_len + int(seq_len),
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0)
        else:
            position_ids = position_ids.to(device=device, dtype=torch.long)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        presents: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None

        for layer_idx, layer in enumerate(self.layers):
            past_kv = None
            if past_key_values is not None and layer_idx < len(past_key_values):
                past_kv = past_key_values[layer_idx]
            if use_cache:
                hidden_states, present = layer(
                    hidden_states,
                    cos=cos,
                    sin=sin,
                    attention_mask=attention_mask,
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
                    cos=cos,
                    sin=sin,
                    attention_mask=attention_mask,
                    past_kv=past_kv,
                    return_kv=False,
                    paged_kv_cache=paged_kv_cache,
                    attn_backend=attn_backend,
                )
        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(
            int(config.hidden_size),
            int(config.vocab_size),
            bias=False,
        )
        if bool(getattr(config, "tie_word_embeddings", True)):
            self.lm_head.weight = self.model.embed_tokens.weight

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
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, int(self.config.vocab_size)),
                shift_labels.view(-1),
            )
        if use_cache:
            return logits, loss, presents
        return logits, loss
