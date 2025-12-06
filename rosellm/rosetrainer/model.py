import math
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

from rosellm.rosetrainer.config import GPTConfig
from rosellm.rosetrainer.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    init_tensor_parallel,
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
    ):
        bsz, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(bsz, seq_len, 3, self.local_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            full_seq_len = k.size(-2)
        else:
            full_seq_len = seq_len

        attn_scores = q @ k.transpose(-2, -1) * self.d_head**-0.5
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
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, self.local_heads * self.d_head)
        out = self.out_proj(attn_output)
        out = self.dropout(out)
        if return_kv:
            return out, (k, v)
        return out


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

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
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
    ):
        if return_kv:
            attn_out, present_kv = self.attn(
                self.ln1(x),
                attention_mask=attention_mask,
                past_kv=past_kv,
                return_kv=True,
            )
        else:
            attn_out = self.attn(self.ln1(x), attention_mask=attention_mask)
            present_kv = None
        x = x + attn_out
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        if return_kv:
            return x, present_kv
        return x


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
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
    ):
        bsz, seq_len = input_ids.size()
        device = input_ids.device
        token_emb = self.token_embedding(input_ids)  # [B, T, D]
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
            and not use_cache,
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
                    )
                    presents.append(present_kv)
                else:
                    x = block(x, attention_mask=attention_mask, past_kv=past_kv)
        x = self.ln_f(x)
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
