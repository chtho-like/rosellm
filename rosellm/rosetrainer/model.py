from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig
from tensor_parallel import (
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
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(bsz, seq_len, 3, self.local_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = q @ k.transpose(-2, -1) * self.d_head**-0.5
        causal_mask = self.mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
        if attention_mask is not None:  # padding mask
            attn_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, self.local_heads * self.d_head)
        out = self.out_proj(attn_output)
        out = self.dropout(out)
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

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        attn_out = self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + attn_out
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
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

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T]
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len = input_ids.size()
        device = input_ids.device
        token_emb = self.token_embedding(input_ids)  # [B, T, D]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
        pos_emb = self.position_embedding(position_ids)  # [1, T, D]
        pos_emb = pos_emb.expand(bsz, seq_len, -1)  # [B, T, D]
        x = token_emb + pos_emb
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
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
        return logits, loss  # [B, T, V], []
