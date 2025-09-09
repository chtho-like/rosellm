"""Position embeddings module for RoseLLM.

This module provides various position embedding implementations including:
- Learned position embeddings
- Sinusoidal position embeddings
- Rotary Position Embeddings (RoPE)
- Attention with Linear Biases (ALiBi)
"""

from rosellm.rosetrainer.embeddings.position_embeddings import (
    ALiBiPositionEmbedding,
    LearnedPositionEmbedding,
    PositionEmbeddingFactory,
    PositionEmbeddingMixin,
    PositionEmbeddingType,
    SinusoidalPositionEmbedding,
)
from rosellm.rosetrainer.embeddings.rope import (
    FusedRoPE,
    RoPEConfig,
    RoPEInterpolationType,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_fused,
    precompute_rope_params,
    rotate_half,
)

__all__ = [
    # Position embeddings
    "PositionEmbeddingType",
    "PositionEmbeddingFactory",
    "PositionEmbeddingMixin",
    "LearnedPositionEmbedding",
    "SinusoidalPositionEmbedding",
    "ALiBiPositionEmbedding",
    # RoPE specific
    "RoPEConfig",
    "RoPEInterpolationType",
    "RotaryEmbedding",
    "FusedRoPE",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_fused",
    "rotate_half",
    "precompute_rope_params",
]
