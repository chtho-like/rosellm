from typing import Optional, Tuple, Unpack

import torch
import torch.nn as nn

from rosellm.config import ModelConfig
from rosellm.loss import causal_lm_loss
from rosellm.models.cache_utils import Cache
from rosellm.models.flash_attention_utils import FlashAttentionKwargs
from rosellm.models.rope_utils import ROPE_INIT_FUNCTIONS


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
        """
        Possible structure in json config:
        "rope_scaling": {
            "rope_type": "default"
        }
        or
        "rope_scaling": {
            "rope_type": "linear",
            "factor": 2.0
        }
        or
        "rope_scaling": {
            "rope_type": "dynamic",
            "factor": 2.0,
            "original_max_position_embeddings": 4096
        }
        or
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.0
        }
        or
        "rope_scaling": {
            "rope_type": "longrope",
            "factor": 4.0
        }
        or
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        }
        """
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
    # [batch_size, num_attention_heads, seq_len, head_dim]
    # E.g. [2, 14, 1024, 64] for Qwen2-0.5B
    q: torch.Tensor,
    # [batch_size, num_key_value_heads, seq_len, head_dim]
    # E.g. [2, 2, 1024, 64] for Qwen2-0.5B
    k: torch.Tensor,
    # [batch_size, seq_len, head_dim]
    # E.g. [2, 1024, 64] for Qwen2-0.5B
    cos: torch.Tensor,
    # [batch_size, seq_len, head_dim]
    # E.g. [2, 1024, 64] for Qwen2-0.5B
    sin: torch.Tensor,
    unsqueeze_dim=1,
):
    # [batch_size, 1, seq_len, head_dim]
    # E.g. [2, 1, 1024, 64] for Qwen2-0.5B
    cos = cos.unsqueeze(unsqueeze_dim)
    # [batch_size, 1, seq_len, head_dim]
    # E.g. [2, 1, 1024, 64] for Qwen2-0.5B
    sin = sin.unsqueeze(unsqueeze_dim)
    # [batch_size, num_attention_heads, seq_len, head_dim]
    # E.g. [2, 14, 1024, 64] for Qwen2-0.5B
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # [batch_size, num_key_value_heads, seq_len, head_dim]
    # E.g. [2, 2, 1024, 64] for Qwen2-0.5B
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
        # [batch_size, seq_len, head_dim]
        # E.g. [2, 1024, 64] for Qwen2-0.5B
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
    inv_freq: torch.Tensor

    def __init__(
        self,
        config: Qwen2Config,
    ):
        super().__init__()
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "default")
        else:
            self.rope_type = "default"
        # E.g. 32768 for Qwen2-0.5B
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        # shape of inv_freq:
        # [head_dim/2]
        # E.g. [32] for Qwen2-0.5B
        inv_freq, self.attention_scaling = self.rope_init_fn(
            # E.g. 1000000.0 for Qwen2-0.5B
            config.rope_theta,
            # E.g. 896 // 14 = 64 for Qwen2-0.5B
            config.hidden_size // config.num_attention_heads,
        )
        self.register_buffer(
            "inv_freq",
            inv_freq,
            # The buffer will not be saved in state_dict().
            # Each time the model is loaded, the buffer will be
            # recomputed.
            persistent=False,
        )
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(
        self,
        # [batch_size, seq_len, hidden_size]
        # E.g. [2, 1024, 896] for Qwen2-0.5B
        x,
        # [batch_size, seq_len]
        # E.g. [2, 1024] for Qwen2-0.5B
        position_ids,
    ):
        inv_freq_expanded = (
            # [head_dim/2] => [1, head_dim/2, 1]
            # E.g. [32] => [1, 32, 1] for Qwen2-0.5B
            self.inv_freq[None, :, None].float()
            # [batch_size, head_dim/2, 1]
            # E.g. [2, 32, 1] for Qwen2-0.5B
            .expand(position_ids.shape[0], -1, 1)
        )
        # [batch_size, 1, seq_len]
        # E.g. [2, 1, 1024] for Qwen2-0.5B
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            # [batch_size, head_dim/2, 1]
            # @ [batch_size, 1, seq_len]
            # => [batch_size, head_dim/2, seq_len]
            # => [batch_size, seq_len, head_dim/2]
            # E.g. [2, 1024, 32] for Qwen2-0.5B
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            # [batch_size, seq_len, head_dim]
            # E.g. [2, 1024, 64] for Qwen2-0.5B
            emb = torch.cat((freqs, freqs), dim=-1)
            # [batch_size, seq_len, head_dim]
            # E.g. [2, 1024, 64] for Qwen2-0.5B
            cos = emb.cos()
            # [batch_size, seq_len, head_dim]
            # E.g. [2, 1024, 64] for Qwen2-0.5B
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn)
        # will apply a post-processing scaling factor.
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


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
