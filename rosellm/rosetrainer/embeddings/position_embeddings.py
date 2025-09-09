"""Position embeddings integration module for RoseLLM.

This module provides a unified interface for different position embedding types,
including learned embeddings, sinusoidal embeddings, and rotary embeddings (RoPE).
"""

from enum import Enum
from typing import Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from rosellm.rosetrainer.embeddings.rope import FusedRoPE, RoPEConfig, RotaryEmbedding
from rosellm.rosetrainer.utils.logging import get_logger

logger = get_logger(__name__)


class PositionEmbeddingType(Enum):
    """Supported position embedding types."""

    NONE = "none"
    LEARNED = "learned"
    SINUSOIDAL = "sinusoidal"
    ROTARY = "rotary"
    ALIBI = "alibi"
    RELATIVE = "relative"


class LearnedPositionEmbedding(nn.Module):
    """Learned position embeddings (traditional transformer style)."""

    def __init__(
        self,
        max_position_embeddings: int,
        hidden_size: int,
        dropout: float = 0.0,
    ):
        """Initialize learned position embeddings.

        Args:
            max_position_embeddings: Maximum sequence length
            hidden_size: Dimension of embeddings
            dropout: Dropout probability
        """
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        position_ids: Optional[Tensor] = None,
        seq_length: Optional[int] = None,
        past_key_values_length: int = 0,
    ) -> Tensor:
        """Get position embeddings.

        Args:
            position_ids: Position indices [batch_size, seq_length]
            seq_length: Sequence length (if position_ids not provided)
            past_key_values_length: Length of past key values for caching

        Returns:
            Position embeddings [batch_size, seq_length, hidden_size]
        """
        if position_ids is None:
            if seq_length is None:
                raise ValueError("Either position_ids or seq_length must be provided")
            device = self.embedding.weight.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        embeddings = self.embedding(position_ids)
        result = self.dropout(embeddings)
        return result  # type: ignore[no-any-return]


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings (fixed, not learned)."""

    def __init__(
        self,
        max_position_embeddings: int,
        hidden_size: int,
        base: float = 10000.0,
    ):
        """Initialize sinusoidal position embeddings.

        Args:
            max_position_embeddings: Maximum sequence length
            hidden_size: Dimension of embeddings
            base: Base for frequency computation
        """
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.base = base

        # Precompute embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize sinusoidal embeddings."""
        position = torch.arange(self.max_position_embeddings).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2)
            * -(torch.log(torch.tensor(self.base)) / self.hidden_size)
        )

        embeddings = torch.zeros(self.max_position_embeddings, self.hidden_size)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("embeddings", embeddings, persistent=False)
        self.embeddings: Tensor = embeddings  # Type hint for pyright

    def forward(
        self,
        position_ids: Optional[Tensor] = None,
        seq_length: Optional[int] = None,
        past_key_values_length: int = 0,
    ) -> Tensor:
        """Get sinusoidal position embeddings.

        Args:
            position_ids: Position indices
            seq_length: Sequence length
            past_key_values_length: Length of past key values

        Returns:
            Position embeddings
        """
        if position_ids is not None:
            # Use provided position IDs
            # batch_size = position_ids.shape[0]
            embeddings = self.embeddings[position_ids]
        else:
            if seq_length is None:
                raise ValueError("Either position_ids or seq_length must be provided")

            # Generate sequential positions
            start_pos = past_key_values_length
            end_pos = start_pos + seq_length
            embeddings = self.embeddings[start_pos:end_pos]
            embeddings = embeddings.unsqueeze(0)  # Add batch dimension

        return embeddings


class ALiBiPositionEmbedding(nn.Module):
    """Attention with Linear Biases (ALiBi) position embedding.

    ALiBi adds position-dependent biases directly to attention scores
    instead of adding position embeddings to inputs.
    """

    def __init__(
        self,
        num_heads: int,
        max_position_embeddings: int = 2048,
    ):
        """Initialize ALiBi embeddings.

        Args:
            num_heads: Number of attention heads
            max_position_embeddings: Maximum sequence length
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings

        # Compute slopes for each head
        self._init_alibi_slopes()

    def _init_alibi_slopes(self) -> None:
        """Initialize ALiBi slopes for each attention head."""

        def get_slopes(n):
            """Get slopes for n attention heads."""

            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(torch.log2(torch.tensor(n)) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            log2_n = torch.log2(torch.tensor(n, dtype=torch.float32))
            if (log2_n == log2_n.floor()).item():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = int(2 ** torch.floor(log2_n).item())
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = torch.tensor(get_slopes(self.num_heads))
        self.register_buffer("slopes", slopes, persistent=False)
        self.slopes: Tensor = slopes  # Type hint for pyright

    def forward(
        self,
        attention_scores: Tensor,
        seq_length: int,
        key_length: Optional[int] = None,
    ) -> Tensor:
        """Apply ALiBi biases to attention scores.

        Args:
            attention_scores: Attention scores [batch, num_heads, seq_len, seq_len]
            seq_length: Query sequence length
            key_length: Key sequence length (defaults to seq_length)

        Returns:
            Attention scores with ALiBi biases applied
        """
        if key_length is None:
            key_length = seq_length

        # Create position bias matrix
        positions = torch.arange(seq_length, dtype=torch.float32).unsqueeze(
            1
        ) - torch.arange(key_length, dtype=torch.float32).unsqueeze(0)
        positions = positions.to(attention_scores.device)

        # Apply slopes
        alibi = positions.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(1)

        # Add to attention scores
        return attention_scores + alibi


class PositionEmbeddingFactory:
    """Factory for creating position embedding modules."""

    _registry: Dict[PositionEmbeddingType, Type[nn.Module]] = {
        PositionEmbeddingType.LEARNED: LearnedPositionEmbedding,
        PositionEmbeddingType.SINUSOIDAL: SinusoidalPositionEmbedding,
        PositionEmbeddingType.ROTARY: RotaryEmbedding,
        PositionEmbeddingType.ALIBI: ALiBiPositionEmbedding,
    }

    @classmethod
    def create(
        cls,
        embedding_type: Union[str, PositionEmbeddingType],
        **kwargs,
    ) -> Optional[nn.Module]:
        """Create a position embedding module.

        Args:
            embedding_type: Type of position embedding
            **kwargs: Arguments for the specific embedding type

        Returns:
            Position embedding module or None if type is NONE
        """
        # Convert string to enum if needed
        if isinstance(embedding_type, str):
            embedding_type = PositionEmbeddingType(embedding_type.lower())

        if embedding_type == PositionEmbeddingType.NONE:
            return None

        if embedding_type not in cls._registry:
            raise ValueError(f"Unknown position embedding type: {embedding_type}")

        # Special handling for RoPE
        if embedding_type == PositionEmbeddingType.ROTARY:
            # Check if we should use fused version
            use_fused = kwargs.pop("use_fused", True)

            # Create RoPE config if not provided
            if "config" in kwargs:
                config = kwargs["config"]
            else:
                config = RoPEConfig(**kwargs)

            if use_fused:
                return FusedRoPE(config)
            else:
                return RotaryEmbedding(config)

        # Create other embedding types
        embedding_class = cls._registry[embedding_type]
        return embedding_class(**kwargs)

    @classmethod
    def register(
        cls,
        embedding_type: PositionEmbeddingType,
        embedding_class: Type[nn.Module],
    ):
        """Register a new position embedding type.

        Args:
            embedding_type: Type identifier
            embedding_class: Module class to register
        """
        cls._registry[embedding_type] = embedding_class
        logger.info(f"Registered position embedding: {embedding_type.value}")


class PositionEmbeddingMixin:
    """Mixin class for models that use position embeddings.

    This mixin provides convenient methods for handling position embeddings
    in transformer models.
    """

    def __init__(self) -> None:
        """Initialize the mixin."""
        self.position_embedding: Optional[nn.Module] = None
        self.position_embedding_type: Optional[PositionEmbeddingType] = None

    def setup_position_embeddings(
        self,
        embedding_type: Union[str, PositionEmbeddingType],
        **kwargs,
    ):
        """Setup position embeddings for the model.

        Args:
            embedding_type: Type of position embedding
            **kwargs: Arguments for the embedding module
        """
        self.position_embedding = PositionEmbeddingFactory.create(
            embedding_type, **kwargs
        )

        if isinstance(embedding_type, str):
            self.position_embedding_type = PositionEmbeddingType(embedding_type.lower())
        else:
            self.position_embedding_type = embedding_type

        logger.info(f"Setup position embeddings: {self.position_embedding_type.value}")

    def get_position_embeddings(
        self,
        seq_length: int,
        position_ids: Optional[Tensor] = None,
        past_key_values_length: int = 0,
    ) -> Optional[Tensor]:
        """Get position embeddings.

        Args:
            seq_length: Sequence length
            position_ids: Optional position indices
            past_key_values_length: Length of cached key values

        Returns:
            Position embeddings or None if not using additive embeddings
        """
        if self.position_embedding is None:
            return None

        if self.position_embedding_type in [
            PositionEmbeddingType.LEARNED,
            PositionEmbeddingType.SINUSOIDAL,
        ]:
            result = self.position_embedding(
                position_ids=position_ids,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )
            return result  # type: ignore[no-any-return]

        # RoPE and ALiBi are applied differently (not additive)
        return None

    def apply_rotary_embeddings(
        self,
        query: Tensor,
        key: Tensor,
        position_ids: Optional[Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply rotary embeddings if configured.

        Args:
            query: Query tensor
            key: Key tensor
            position_ids: Optional position indices
            seq_len: Optional sequence length

        Returns:
            Tuple of (query, key) with RoPE applied if configured
        """
        if self.position_embedding_type != PositionEmbeddingType.ROTARY:
            return query, key

        if self.position_embedding is None:
            return query, key

        result = self.position_embedding(query, key, position_ids, seq_len)
        return result  # type: ignore[no-any-return]

    def apply_alibi_biases(
        self,
        attention_scores: Tensor,
        seq_length: int,
        key_length: Optional[int] = None,
    ) -> Tensor:
        """Apply ALiBi biases if configured.

        Args:
            attention_scores: Attention scores
            seq_length: Query sequence length
            key_length: Key sequence length

        Returns:
            Attention scores with ALiBi applied if configured
        """
        if self.position_embedding_type != PositionEmbeddingType.ALIBI:
            return attention_scores

        if self.position_embedding is None:
            return attention_scores

        result = self.position_embedding(attention_scores, seq_length, key_length)
        return result  # type: ignore[no-any-return]
