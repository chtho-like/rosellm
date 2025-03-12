from typing import Optional, Tuple, Unpack

import torch
import torch.nn as nn

from rosellm.config import ModelConfig
from rosellm.loss import causal_lm_loss
from rosellm.models.cache_utils import Cache
from rosellm.models.flash_attention_utils import FlashAttentionKwargs


class Qwen2Config(ModelConfig):
    """E.g. For Qwen2-0.5B
    {
      "architectures": [
        "Qwen2ForCausalLM"
      ],
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151643,
      "hidden_act": "silu",
      "hidden_size": 896,
      "initializer_range": 0.02,
      "intermediate_size": 4864,
      "max_position_embeddings": 32768,
      "max_window_layers": 24,
      "model_type": "qwen2",
      "num_attention_heads": 14,
      "num_hidden_layers": 24,
      "num_key_value_heads": 2,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000.0,
      "sliding_window": 32768,
      "tie_word_embeddings": true,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.40.1",
      "use_cache": true,
      "use_mrope": false,
      "use_sliding_window": false,
      "vocab_size": 151936
    }
    """

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.tie_word_embeddings = tie_word_embeddings

    @classmethod
    def from_dict(
        cls,
        config_dict: dict,
    ):
        return cls(**config_dict)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids=None,
    unsqueeze_dim=1,
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            # E.g. 896 // 14 = 64 for Qwen2-0.5B
            config.hidden_size // config.num_attention_heads,
        )
        self.num_key_value_groups = (
            # E.g. 14 // 2 = 7 for Qwen2-0.5B
            config.num_attention_heads
            // config.num_key_value_heads
        )
        # a.k.a \frac{1}{\sqrt{d}}
        # E.g. 64**-0.5 = 0.125 for Qwen2-0.5B
        self.scaling = self.head_dim**-0.5
        # E.g. 0.0 for Qwen2-0.5B
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            # E.g. 896 for Qwen2-0.5B
            config.hidden_size,
            # E.g. 14 * 64 = 896 for Qwen2-0.5B
            config.num_attention_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            # E.g. 896 for Qwen2-0.5B
            config.hidden_size,
            # E.g. 2 * 64 = 128 for Qwen2-0.5B
            config.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            # E.g. 896 for Qwen2-0.5B
            config.hidden_size,
            # E.g. 2 * 64 = 128 for Qwen2-0.5B
            config.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            # E.g. 14 * 64 = 896 for Qwen2-0.5B
            config.num_attention_heads * self.head_dim,
            # E.g. 896 for Qwen2-0.5B
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        # [batch_size, seq_len, hidden_size]
        # E.g. [2, 1024, 896] for Qwen2-0.5B
        hidden_states: torch.Tensor,
        # cos: [1, 1, seq_len, head_dim/2]
        # sin: [1, 1, seq_len, head_dim/2]
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor]],
    ]:
        # [batch_size, seq_len]
        # E.g. [2, 1024] for Qwen2-0.5B
        input_shape = hidden_states.shape[:-1]
        # [batch_size, seq_len, -1, head_dim]
        # E.g. [2, 1024, -1, 64] for Qwen2-0.5B
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = (
            # [batch_size, seq_len, num_attention_heads, head_dim]
            self.q_proj(hidden_states).view(hidden_shape)
            # [batch_size, num_attention_heads, seq_len, head_dim]
            .transpose(1, 2)
        )
        key_states = (
            # [batch_size, seq_len, num_key_value_heads, head_dim]
            self.k_proj(hidden_states).view(hidden_shape)
            # [batch_size, num_key_value_heads, seq_len, head_dim]
            .transpose(1, 2)
        )
        value_states = (
            # [batch_size, seq_len, num_key_value_heads, head_dim]
            self.v_proj(hidden_states).view(hidden_shape)
            # [batch_size, num_key_value_heads, seq_len, head_dim]
            .transpose(1, 2)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )
        return (
            torch.tensor(0.0),
            None,
            None,
        )


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
    ):
        super().__init__()


class Qwen2RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(
            config=config,
            layer_idx=layer_idx,
        )
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
    ):
        super().__init__()


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,  # E.g. 151936 for Qwen2-0.5B
            config.hidden_size,  # E.g. 896 for Qwen2-0.5B
        )
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, layer_idx)
                # E.g. 24 layers for Qwen2-0.5B
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(
            # E.g. 896 for Qwen2-0.5B
            config.hidden_size,
            # E.g. 1e-6 for Qwen2-0.5B
            eps=config.rms_norm_eps,
        )
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False


class Qwen2ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        args: Optional[ModelConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.args = args
        self.loss_type = causal_lm_loss
        self.vocab_size = config.vocab_size
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.post_init()

    @classmethod
    def _from_config(
        cls,
        config: Qwen2Config,
        args: Optional[ModelConfig] = None,
    ):
        return cls(config, args)

    def post_init(self):
        pass
