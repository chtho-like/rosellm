import math
import os
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

from rosellm.rosetrainer.attention_backends import (
    prefill_attention_flashattn,
    prefill_attention_flashinfer,
    prefill_attention_flashinfer_paged,
    prefill_attention_flashinfer_paged_varlen,
)
from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.fused_layernorm import add_layer_norm_, layer_norm
from rosellm.rosetrainer.fused_mlp import add_bias_residual_, bias_gelu_new_
from rosellm.rosetrainer.paged_attention import (
    TRITON_AVAILABLE,
    PagedKVCache,
    paged_attention_decode_ref,
    paged_attention_decode_triton,
)
from rosellm.rosetrainer.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    init_tensor_parallel,
)

paged_attention_decode = (
    paged_attention_decode_triton if TRITON_AVAILABLE else paged_attention_decode_ref
)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        use_tp_cfg = getattr(config, "use_tensor_parallel", False)
        self.use_tp = use_tp_cfg and dist.is_available() and dist.is_initialized()
        if self.use_tp:
            init_tensor_parallel()
            tp_world_size = dist.get_world_size()
            if self.n_heads % tp_world_size != 0:
                raise ValueError("n_heads must be divisible by tp_world_size")
            self.tp_world_size = tp_world_size
            self.local_heads = self.n_heads // tp_world_size
            self.qkv_proj = ColumnParallelLinear(
                in_features=config.d_model,
                out_features=3 * config.d_model,
                bias=True,
                gather_output=False,
            )
            self.out_proj = RowParallelLinear(
                in_features=config.d_model,
                out_features=config.d_model,
                bias=True,
                input_is_parallel=True,
            )
        else:
            self.tp_world_size = 1
            self.local_heads = self.n_heads
            self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
            self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj.gpt2_residual = True
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config.max_position_embeddings, config.max_position_embeddings
                )
            )
            .unsqueeze(0)
            .unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # [B, T]
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False,
        paged_kv_cache: Optional[PagedKVCache] = None,
        layer_idx: int = 0,
        attn_backend: str | None = None,
    ):
        bsz, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(bsz, seq_len, 3, self.local_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]
        if paged_kv_cache is not None:
            if past_kv is not None:
                raise ValueError("past_kv and paged_kv_cache cannot be used together")
            backend = (attn_backend or "naive").lower()
            if seq_len != 1:
                if backend not in ("flashinfer_paged", "flashinfer-paged"):
                    raise ValueError(
                        "paged prefill requires attn_backend='flashinfer_paged' "
                        f"(got attn_backend={attn_backend})"
                    )
                if self.training or x.requires_grad:
                    raise ValueError(
                        "flashinfer_paged attention backend only supports inference"
                    )
                attn_output = prefill_attention_flashinfer_paged(
                    q=q,
                    k=k,
                    v=v,
                    attention_mask=attention_mask,
                    paged_kv_cache=paged_kv_cache,
                    layer_idx=int(layer_idx),
                    sm_scale=float(self.d_head**-0.5),
                    causal=True,
                )
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(
                    bsz,
                    seq_len,
                    self.local_heads * self.d_head,
                )
                out = self.out_proj(attn_output)
                out = self.dropout(out)
                if return_kv:
                    return out, (k, v)
                return out
            out_h = paged_attention_decode(
                q=q.squeeze(-2),  # [B, H, D]
                k_new=k.squeeze(-2),  # [B, H, D]
                v_new=v.squeeze(-2),
                k_cache_layer=paged_kv_cache.k_cache[layer_idx],
                v_cache_layer=paged_kv_cache.v_cache[layer_idx],
                block_table=paged_kv_cache.block_tables[layer_idx],
                slot_mapping=paged_kv_cache.slot_mapping,
                context_lens=paged_kv_cache.context_lens,
                scale=self.d_head**-0.5,
                block_size=paged_kv_cache.block_size,
                write_kv=bool(paged_kv_cache.write_kv),
            )  # [B, H, D]
            attn_output = out_h.unsqueeze(2)  # [B, H, 1, D]
            # [B, 1, H, D]
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(
                bsz,
                seq_len,
                self.local_heads * self.d_head,
            )
            out = self.out_proj(attn_output)
            out = self.dropout(out)
            if return_kv:
                return out, (k, v)
            return out
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            full_seq_len = k.size(-2)
        else:
            full_seq_len = seq_len

        backend = (attn_backend or "naive").lower()
        scale = float(self.d_head**-0.5)
        if backend in ("naive", "eager"):
            attn_scores = q @ k.transpose(-2, -1) * scale
            causal_mask = self.mask[
                :,
                :,
                full_seq_len - seq_len : full_seq_len,
                :full_seq_len,
            ]
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
            if attention_mask is not None:  # padding mask
                attn_mask = attention_mask[:, None, None, :]
                mask_value = torch.finfo(attn_scores.dtype).min
                attn_scores = attn_scores.masked_fill(attn_mask == 0, mask_value)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = attn_weights @ v
        elif backend in ("flashinfer", "flashattn"):
            if self.training or x.requires_grad:
                raise ValueError(f"{backend} attention backend only supports inference")
            if past_kv is not None:
                raise ValueError(
                    f"{backend} attention backend does not support past_kv"
                )
            if attention_mask is not None and int(attention_mask.size(-1)) != int(
                seq_len
            ):
                raise ValueError("attention_mask must match x sequence length")
            if backend == "flashinfer":
                attn_output = prefill_attention_flashinfer(
                    q=q,
                    k=k,
                    v=v,
                    attention_mask=attention_mask,
                    sm_scale=scale,
                    causal=True,
                )
            else:
                attn_output = prefill_attention_flashattn(
                    q=q,
                    k=k,
                    v=v,
                    attention_mask=attention_mask,
                    softmax_scale=scale,
                    causal=True,
                )
        else:
            raise ValueError(f"unknown attention backend: {attn_backend}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, self.local_heads * self.d_head)
        out = self.out_proj(attn_output)
        out = self.dropout(out)
        if return_kv:
            return out, (k, v)
        return out

    def forward_varlen(
        self,
        x: torch.Tensor,  # [N, D]
        *,
        paged_kv_cache: PagedKVCache,
        layer_idx: int,
        attn_backend: str | None = None,
    ) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("varlen attention expects x to be 2D [N, D]")
        if paged_kv_cache is None:
            raise ValueError("paged_kv_cache is required for varlen attention")
        backend = (attn_backend or "naive").lower()
        if backend not in ("flashinfer_paged", "flashinfer-paged"):
            raise ValueError(
                "paged varlen prefill requires attn_backend='flashinfer_paged' "
                f"(got attn_backend={attn_backend})"
            )
        if self.training or x.requires_grad:
            raise ValueError("paged varlen attention only supports inference")

        n_tokens, _ = x.size()
        qkv = self.qkv_proj(x)  # [N, 3*D]
        head_dim = int(self.d_head)
        local_heads = int(self.local_heads)
        proj_dim = int(local_heads * head_dim)
        q_proj, k_proj, v_proj = qkv.split(proj_dim, dim=-1)
        q = q_proj.view(int(n_tokens), local_heads, head_dim)
        k = k_proj.view(int(n_tokens), local_heads, head_dim)
        v = v_proj.view(int(n_tokens), local_heads, head_dim)
        attn_out = prefill_attention_flashinfer_paged_varlen(
            q=q,
            k=k,
            v=v,
            paged_kv_cache=paged_kv_cache,
            layer_idx=int(layer_idx),
            sm_scale=float(self.d_head**-0.5),
            causal=True,
        )  # [N, H, D]
        attn_out = attn_out.contiguous().view(
            int(n_tokens),
            int(local_heads * head_dim),
        )
        out = self.out_proj(attn_out)
        out = self.dropout(out)
        return out


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    return (
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
    )


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        use_tp = getattr(config, "use_tensor_parallel", False)
        if use_tp and dist.is_available() and dist.is_initialized():
            init_tensor_parallel()
            self.fc1 = ColumnParallelLinear(
                in_features=config.d_model,
                out_features=config.d_ff,
                bias=True,
                gather_output=False,
            )
            self.fc2 = RowParallelLinear(
                in_features=config.d_ff,
                out_features=config.d_model,
                bias=True,
                input_is_parallel=True,
            )
        else:
            self.fc1 = nn.Linear(config.d_model, config.d_ff)
            self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.fc2.gpt2_residual = True
        self.dropout = nn.Dropout(config.dropout)
        self.activation = getattr(config, "activation", "gelu")

    def forward(
        self,
        x: torch.Tensor,
        *,
        residual: torch.Tensor | None = None,
        use_fused_mlp: bool = False,
    ) -> torch.Tensor:
        wants_residual = (
            bool(use_fused_mlp)
            and residual is not None
            and (not self.training)
            and (not x.requires_grad)
        )
        fused_ok = (
            wants_residual
            and x.is_cuda
            and residual.is_cuda
            and residual.dtype == x.dtype
            and x.is_contiguous()
            and residual.is_contiguous()
            and isinstance(self.fc1, nn.Linear)
            and isinstance(self.fc2, nn.Linear)
            and self.fc1.bias is not None
            and self.fc2.bias is not None
            and self.activation == "gelu_new"
        )
        if fused_ok:
            x2 = x.view(-1, int(x.size(-1)))
            residual2 = residual.view(-1, int(residual.size(-1)))
            h = torch.matmul(x2, self.fc1.weight.t())
            bias_gelu_new_(h, self.fc1.bias)
            y = torch.matmul(h, self.fc2.weight.t())
            add_bias_residual_(residual2, y, self.fc2.bias)
            return residual
        x = self.fc1(x)
        if self.activation == "gelu_new":
            x = gelu_new(x)
        else:
            x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if wants_residual:
            residual.add_(x)
            return residual
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False,
        paged_kv_cache: Optional[PagedKVCache] = None,
        layer_idx: int = 0,
        attn_backend: str | None = None,
        use_fused_ops: bool = False,
        use_fused_mlp: bool = False,
    ):
        fused_ok = bool(use_fused_ops) and (not self.training) and (not x.requires_grad)
        if fused_ok:
            x_ln1 = layer_norm(
                x,
                self.ln1.weight,
                self.ln1.bias,
                eps=float(self.ln1.eps),
            )
        else:
            x_ln1 = self.ln1(x)
        if return_kv:
            attn_out, present_kv = self.attn(
                x_ln1,
                attention_mask=attention_mask,
                past_kv=past_kv,
                return_kv=True,
                paged_kv_cache=paged_kv_cache,
                layer_idx=layer_idx,
                attn_backend=attn_backend,
            )
        else:
            attn_out = self.attn(
                x_ln1,
                attention_mask=attention_mask,
                past_kv=past_kv,
                return_kv=False,
                paged_kv_cache=paged_kv_cache,
                layer_idx=layer_idx,
                attn_backend=attn_backend,
            )
            present_kv = None
        if fused_ok:
            x_ln2 = add_layer_norm_(
                x,
                attn_out,
                self.ln2.weight,
                self.ln2.bias,
                eps=float(self.ln2.eps),
            )
        else:
            x = x + attn_out
            x_ln2 = self.ln2(x)
        if fused_ok and bool(use_fused_mlp):
            x = self.mlp(
                x_ln2,
                residual=x,
                use_fused_mlp=True,
            )
        else:
            mlp_out = self.mlp(x_ln2)
            if fused_ok:
                x.add_(mlp_out)
            else:
                x = x + mlp_out
        if return_kv:
            return x, present_kv
        return x

    def forward_varlen(
        self,
        x: torch.Tensor,  # [N, D]
        *,
        paged_kv_cache: PagedKVCache,
        layer_idx: int = 0,
        attn_backend: str | None = None,
        use_fused_ops: bool = False,
        use_fused_mlp: bool = False,
    ) -> torch.Tensor:
        fused_ok = bool(use_fused_ops) and (not self.training) and (not x.requires_grad)
        if fused_ok:
            x_ln1 = layer_norm(
                x,
                self.ln1.weight,
                self.ln1.bias,
                eps=float(self.ln1.eps),
            )
        else:
            x_ln1 = self.ln1(x)

        attn_out = self.attn.forward_varlen(
            x_ln1,
            paged_kv_cache=paged_kv_cache,
            layer_idx=int(layer_idx),
            attn_backend=attn_backend,
        )
        if fused_ok:
            x_ln2 = add_layer_norm_(
                x,
                attn_out,
                self.ln2.weight,
                self.ln2.bias,
                eps=float(self.ln2.eps),
            )
        else:
            x = x + attn_out
            x_ln2 = self.ln2(x)

        if fused_ok and bool(use_fused_mlp):
            x = self.mlp(
                x_ln2,
                residual=x,
                use_fused_mlp=True,
            )
        else:
            mlp_out = self.mlp(x_ln2)
            if fused_ok:
                x.add_(mlp_out)
            else:
                x = x + mlp_out
        return x


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.use_fused_ops = os.environ.get("ROSELLM_FUSED_OPS", "1") != "0"
        self.use_fused_mlp = os.environ.get("ROSELLM_FUSED_MLP", "1") != "0"
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tie_weights()
        self.apply(self._init_weights)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(
            module,
            (
                nn.Linear,
                ColumnParallelLinear,
                RowParallelLinear,
            ),
        ):
            std = 0.02
            if getattr(module, "gpt2_residual", False):
                n_layers = float(self.config.n_layers)
                std = std / math.sqrt(2.0 * n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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
        logits_positions: Optional[torch.Tensor] = None,  # [B]
    ):
        bsz, seq_len = input_ids.size()
        device = input_ids.device
        token_emb = self.token_embedding(input_ids)  # [B, T, D]
        if position_ids is None:
            if past_key_values is not None and len(past_key_values) > 0:
                past_k = past_key_values[0][0]
                past_len = past_k.size(-2)
            else:
                past_len = 0
            position_ids = torch.arange(
                past_len,
                past_len + seq_len,
                device=device,
            ).unsqueeze(
                0
            )  # [1, T]
        else:
            position_ids = position_ids.to(device=device, dtype=torch.long)
        pos_emb = self.position_embedding(position_ids)  # [1, T, D]
        pos_emb = pos_emb.expand(bsz, seq_len, -1)  # [B, T, D]
        x = token_emb + pos_emb
        x = self.dropout(x)
        use_ckpt = (
            getattr(
                self.config,
                "use_activation_checkpoint",
                False,
            )
            and self.training
            and not use_cache
        )
        presents: list[tuple[Any, Any]] | None = [] if use_cache else None
        if use_ckpt:
            for block in self.blocks:

                def block_forward(
                    *inputs,
                    _block=block,
                ):
                    return _block(*inputs)

                x = ckpt(block_forward, x, attention_mask, use_reentrant=False)
        else:
            for layer_idx, block in enumerate(self.blocks):
                if past_key_values is not None and layer_idx < len(past_key_values):
                    past_kv = past_key_values[layer_idx]
                else:
                    past_kv = None
                if use_cache:
                    x, present_kv = block(
                        x,
                        attention_mask=attention_mask,
                        past_kv=past_kv,
                        return_kv=True,
                        paged_kv_cache=paged_kv_cache,
                        layer_idx=layer_idx,
                        attn_backend=attn_backend,
                        use_fused_ops=bool(self.use_fused_ops),
                        use_fused_mlp=bool(self.use_fused_mlp),
                    )
                    presents.append(present_kv)
                else:
                    x = block(
                        x,
                        attention_mask=attention_mask,
                        past_kv=past_kv,
                        paged_kv_cache=paged_kv_cache,
                        layer_idx=layer_idx,
                        attn_backend=attn_backend,
                        use_fused_ops=bool(self.use_fused_ops),
                        use_fused_mlp=bool(self.use_fused_mlp),
                    )
        if (
            bool(self.use_fused_ops)
            and (not self.training)
            and (not x.requires_grad)
            and x.is_cuda
        ):
            x = layer_norm(
                x,
                self.ln_f.weight,
                self.ln_f.bias,
                eps=float(self.ln_f.eps),
            )
        else:
            x = self.ln_f(x)
        if logits_positions is not None:
            if labels is not None:
                raise ValueError("logits_positions is not supported with labels")
            logits_positions = logits_positions.to(device=device, dtype=torch.long)
            if logits_positions.dim() != 1 or int(logits_positions.numel()) != bsz:
                raise ValueError("logits_positions must be a 1D tensor of shape [B]")
            if int(logits_positions.min().item()) < 0:
                raise ValueError("logits_positions must be >= 0")
            if int(logits_positions.max().item()) >= int(seq_len):
                raise ValueError("logits_positions must be < seq_len")
            row = torch.arange(bsz, device=device, dtype=torch.long)
            x_sel = x[row, logits_positions, :]  # [B, D]
            logits = self.lm_head(x_sel).unsqueeze(1)  # [B, 1, V]
        else:
            logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
            shift_labels = labels[:, 1:].contiguous()  # [B, T-1]
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),  # [B*(T-1), V]
                shift_labels.view(-1),  # [B*(T-1)]
            )
        if use_cache:
            return logits, loss, presents
        return logits, loss  # [B, T, V], []

    def forward_varlen(
        self,
        *,
        input_ids: torch.Tensor,  # [N]
        position_ids: torch.Tensor,  # [N]
        paged_kv_cache: PagedKVCache,
        attn_backend: str | None = None,
    ) -> torch.Tensor:  # [N, V]
        if input_ids.dim() != 1:
            raise ValueError("varlen forward expects input_ids to be 1D [N]")
        if position_ids.dim() != 1:
            raise ValueError("varlen forward expects position_ids to be 1D [N]")
        if input_ids.numel() != position_ids.numel():
            raise ValueError("input_ids/position_ids size mismatch")
        if paged_kv_cache is None:
            raise ValueError("paged_kv_cache is required for varlen forward")

        device = input_ids.device
        token_emb = self.token_embedding(input_ids)  # [N, D]
        position_ids = position_ids.to(device=device, dtype=torch.long)
        pos_emb = self.position_embedding(position_ids)  # [N, D]
        x = token_emb + pos_emb
        x = self.dropout(x)
        for layer_idx, block in enumerate(self.blocks):
            x = block.forward_varlen(
                x,
                paged_kv_cache=paged_kv_cache,
                layer_idx=int(layer_idx),
                attn_backend=attn_backend,
                use_fused_ops=bool(self.use_fused_ops),
                use_fused_mlp=bool(self.use_fused_mlp),
            )
        if (
            bool(self.use_fused_ops)
            and (not self.training)
            and (not x.requires_grad)
            and x.is_cuda
        ):
            x = layer_norm(
                x,
                self.ln_f.weight,
                self.ln_f.bias,
                eps=float(self.ln_f.eps),
            )
        else:
            x = self.ln_f(x)
        logits = self.lm_head(x)  # [N, V]
        return logits
