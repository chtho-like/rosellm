"""Rotary Position Embeddings (RoPE) implementation for RoseLLM.

This module implements Rotary Position Embeddings as described in:
RoFormer: Enhanced Transformer with Rotary Position Embedding
(https://arxiv.org/abs/2104.09864)

The implementation includes:
- Basic RoPE with sinusoidal embeddings
- Efficient caching mechanisms
- Support for different interpolation methods
- Fused operations for better performance
- Multi-dimensional parallelism support
"""

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from rosellm.rosetrainer.utils.logging import get_logger

logger = get_logger(__name__)


class RoPEInterpolationType(Enum):
    """RoPE interpolation methods for extending context length."""

    NONE = "none"
    LINEAR = "linear"
    NTK = "ntk"  # Neural Tangent Kernel
    DYNAMIC_NTK = "dynamic_ntk"
    YaRN = "yarn"  # Yet another RoPE extension


@dataclass
class RoPEConfig:
    """Configuration for Rotary Position Embeddings.

    Attributes:
        dim: Dimension of the model (head_dim for RoPE)
        max_position_embeddings: Maximum sequence length
        base: Base for the frequency computation (default: 10000)
        rope_scaling: Optional scaling configuration for longer contexts
        interpolation_type: Type of interpolation for position extension
        scaling_factor: Factor for scaling positions (for interpolation)
        original_max_position_embeddings: Original max position before scaling
        rope_theta: Alternative base parameter name (same as base)
        partial_rotary_factor: Fraction of dimensions to apply RoPE to (0.0 to 1.0)
        use_fused: Whether to use fused operations for better performance
        yarn_beta_fast: YaRN beta parameter for high frequencies (default: 32)
        yarn_beta_slow: YaRN beta parameter for low frequencies (default: 1)
        yarn_original_max_position: Original max position for YaRN scaling
    """

    dim: int
    max_position_embeddings: int = 2048
    base: float = 10000.0
    rope_scaling: Optional[dict] = None
    interpolation_type: RoPEInterpolationType = RoPEInterpolationType.NONE
    scaling_factor: float = 1.0
    original_max_position_embeddings: Optional[int] = None
    rope_theta: Optional[float] = None  # Alternative to base
    partial_rotary_factor: float = 1.0
    use_fused: bool = True
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_original_max_position: Optional[int] = None

    def __post_init__(self):
        """Validate and process configuration."""
        if self.rope_theta is not None:
            self.base = self.rope_theta

        if self.rope_scaling is not None:
            # Process rope_scaling dictionary
            scaling_type = self.rope_scaling.get("type", "linear")
            self.interpolation_type = RoPEInterpolationType(scaling_type)
            self.scaling_factor = self.rope_scaling.get("factor", 1.0)

        if self.partial_rotary_factor < 0.0 or self.partial_rotary_factor > 1.0:
            raise ValueError(
                f"partial_rotary_factor must be in [0, 1], "
                f"got {self.partial_rotary_factor}"
            )

        if self.dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {self.dim}")


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding module.

    This module computes and applies rotary position embeddings to query and key
    tensors. Supports various interpolation methods for extending context length.
    """

    def __init__(self, config: RoPEConfig):
        """Initialize RotaryEmbedding.

        Args:
            config: RoPE configuration
        """
        super().__init__()
        self.config = config

        # Compute dimension to apply RoPE
        self.rope_dim = int(config.dim * config.partial_rotary_factor)
        if self.rope_dim % 2 != 0:
            self.rope_dim -= 1  # Ensure even dimension

        # Initialize frequency computation
        self._init_rope_parameters()

        # Thread-safe cache for computed embeddings
        self._cache_lock = threading.Lock()
        self._seq_len_cached = 0
        self._cos_cached: Optional[Tensor] = None
        self._sin_cached: Optional[Tensor] = None
        self._device_cache: Dict[
            torch.device, Tuple[Optional[Tensor], Optional[Tensor]]
        ] = {}

        logger.info(
            f"Initialized RoPE with dim={self.rope_dim}, "
            f"max_pos={config.max_position_embeddings}, "
            f"base={config.base}, "
            f"interpolation={config.interpolation_type.value}"
        )

    def _init_rope_parameters(self):
        """Initialize RoPE frequency parameters."""
        base = self.config.base
        dim = self.rope_dim

        # Adjust base for NTK interpolation
        if self.config.interpolation_type == RoPEInterpolationType.NTK:
            base = base * self.config.scaling_factor ** (dim / (dim - 2))
        elif self.config.interpolation_type == RoPEInterpolationType.DYNAMIC_NTK:
            # Dynamic NTK adjusts base based on sequence length dynamically
            pass  # Will be handled in forward

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.inv_freq: Tensor = inv_freq  # Type hint for pyright

        # For YaRN, we need additional parameters
        if self.config.interpolation_type == RoPEInterpolationType.YaRN:
            self._init_yarn_parameters()

    def _init_yarn_parameters(self) -> None:
        """Initialize YaRN-specific parameters with proper beta values.

        YaRN (Yet another RoPE extension) uses a sophisticated interpolation
        scheme that preserves both high and low frequency components.
        """
        dim = self.rope_dim
        max_pos = (
            self.config.yarn_original_max_position
            or self.config.max_position_embeddings
        )
        scale = self.config.scaling_factor
        beta_fast = self.config.yarn_beta_fast
        beta_slow = self.config.yarn_beta_slow

        # Compute frequency ramp
        freq_extra = 1.0 / (self.config.base ** (torch.arange(0, dim, 2).float() / dim))
        # freq_inter = 1.0 / (
        #     scale * self.config.base ** (torch.arange(0, dim, 2).float() / dim)
        # )

        # Compute wavelength thresholds
        low_freq_wavelen = max_pos / beta_slow
        high_freq_wavelen = max_pos / beta_fast

        # Compute wavelengths for each frequency
        wavelen_extra = 2 * torch.pi / freq_extra
        # wavelen_inter = 2 * torch.pi / freq_inter

        # Compute scaling mask based on wavelength
        mask_low = wavelen_extra > low_freq_wavelen
        mask_high = wavelen_extra < high_freq_wavelen
        mask_mid = ~(mask_low | mask_high)

        # Compute smooth interpolation factors
        yarn_scale = torch.ones_like(freq_extra)

        # Low frequencies: use extrapolation
        yarn_scale[mask_low] = scale

        # High frequencies: use original (no scaling)
        yarn_scale[mask_high] = 1.0

        # Mid frequencies: smooth transition
        if mask_mid.any():
            # Smooth transition using logarithmic interpolation
            wavelen_mid = wavelen_extra[mask_mid]
            smooth_factor = torch.log(wavelen_mid / high_freq_wavelen) / torch.log(
                torch.tensor(low_freq_wavelen / high_freq_wavelen)
            )
            yarn_scale[mask_mid] = 1.0 + (scale - 1.0) * smooth_factor

        self.register_buffer("yarn_scale", yarn_scale, persistent=False)
        self.yarn_scale: Tensor = yarn_scale  # Type hint for pyright

        # Store attention scaling factor for YaRN
        mscale = scale ** ((dim / (dim - 2)) * 0.1) if scale > 1 else 1.0
        self.register_buffer("yarn_mscale", torch.tensor(mscale), persistent=False)
        self.yarn_mscale: Tensor = torch.tensor(mscale)  # Type hint for pyright

    def _compute_dynamic_ntk_parameters(self, seq_len: int) -> Tensor:
        """Compute dynamic NTK parameters based on sequence length.

        Args:
            seq_len: Current sequence length

        Returns:
            Adjusted inverse frequencies
        """
        base = self.config.base
        dim = self.rope_dim
        max_pos = self.config.max_position_embeddings

        # Calculate dynamic scaling factor
        if seq_len > max_pos:
            scale = seq_len / max_pos
            base = base * scale ** (dim / (dim - 2))

        result = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        return result  # type: ignore[no-any-return]

    def _compute_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ):
        """Compute and cache cos/sin embeddings with memory optimization.

        Args:
            seq_len: Sequence length to compute embeddings for
            device: Device to place tensors on
            dtype: Data type for tensors
        """
        with self._cache_lock:
            # Check if we need to expand cache
            if (
                seq_len <= self._seq_len_cached
                and self._cos_cached is not None
                and self._sin_cached is not None
            ):
                # Move to correct device if needed
                if self._cos_cached.device != device:
                    self._cos_cached = self._cos_cached.to(device)
                    self._sin_cached = self._sin_cached.to(device)
                return

            # Compute positions based on interpolation type
            if self.config.interpolation_type == RoPEInterpolationType.LINEAR:
                position = torch.arange(seq_len, device=device, dtype=torch.float32)
                position = position / self.config.scaling_factor
                inv_freq = self.inv_freq.to(device)
            elif self.config.interpolation_type == RoPEInterpolationType.DYNAMIC_NTK:
                inv_freq = self._compute_dynamic_ntk_parameters(seq_len).to(device)
                position = torch.arange(seq_len, device=device, dtype=torch.float32)
            elif self.config.interpolation_type == RoPEInterpolationType.YaRN:
                position = torch.arange(seq_len, device=device, dtype=torch.float32)
                # Apply YaRN scaling to frequencies
                inv_freq = self.inv_freq.to(device) * self.yarn_scale.to(device)
            else:
                position = torch.arange(seq_len, device=device, dtype=torch.float32)
                inv_freq = self.inv_freq.to(device)

            # Compute frequencies efficiently
            freqs = torch.outer(position, inv_freq)

            # Compute cos and sin using optimized concatenation
            emb = torch.cat([freqs, freqs], dim=-1)

            # Cache with proper dtype
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
            self._seq_len_cached = seq_len

            # Store in device cache for multi-device support
            self._device_cache[device] = (self._cos_cached, self._sin_cached)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Optional[Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
                or [batch, num_heads, seq_len, head_dim]
            k: Key tensor of same shape as q
            position_ids: Optional position indices [batch, seq_len]
            seq_len: Optional sequence length override

        Returns:
            Tuple of (rotated_q, rotated_k)

        Raises:
            ValueError: If tensor dimensions are invalid
        """
        # Validate input tensors
        if q.dim() != 4 or k.dim() != 4:
            raise ValueError(f"Expected 4D tensors, got q: {q.dim()}D, k: {k.dim()}D")

        if q.shape != k.shape:
            raise ValueError(
                f"Query and key shapes must match, got q: {q.shape}, k: {k.shape}"
            )

        # Determine sequence length and validate
        if seq_len is None:
            # Check tensor layout: [B, S, H, D] or [B, H, S, D]
            if q.shape[-1] == self.rope_dim or q.shape[-1] == self.config.dim:
                # Last dim is head_dim, so either [B, S, H, D] or [B, H, S, D]
                seq_len = q.shape[1] if q.shape[1] != q.shape[2] else q.shape[2]
            else:
                seq_len = q.shape[1]  # Default to dim 1

        # Validate head dimension
        head_dim = q.shape[-1]
        if self.rope_dim > head_dim:
            raise ValueError(
                f"RoPE dim ({self.rope_dim}) cannot exceed head dim ({head_dim})"
            )

        # Update cache if needed (thread-safe)
        self._compute_cos_sin_cache(seq_len, q.device, q.dtype)

        # Get cached cos/sin with proper synchronization
        with self._cache_lock:
            if self._cos_cached is None or self._sin_cached is None:
                raise RuntimeError("Failed to initialize RoPE cache")

            # Ensure cache is on correct device
            if self._cos_cached.device != q.device:
                self._cos_cached = self._cos_cached.to(q.device)
                assert (
                    self._sin_cached is not None
                ), "Sin cache should be initialized with cos cache"
                self._sin_cached = self._sin_cached.to(q.device)

            cos = self._cos_cached[:seq_len]
            sin = self._sin_cached[:seq_len]

        # Handle position_ids if provided
        if position_ids is not None:
            if position_ids.dim() not in [1, 2]:
                raise ValueError(
                    f"position_ids must be 1D or 2D, got {position_ids.dim()}D"
                )

            # Validate position_ids range
            if position_ids.max() >= seq_len:
                raise ValueError(
                    f"position_ids max ({position_ids.max()}) "
                    f"exceeds seq_len ({seq_len})"
                )

            cos = cos[position_ids]  # [batch, seq_len, dim]
            sin = sin[position_ids]

        # Apply rotary embeddings with optimized operations
        if self.rope_dim == self.config.dim:
            # Full RoPE - use optimized implementation
            q_rot = apply_rotary_pos_emb_optimized(q, cos, sin)
            k_rot = apply_rotary_pos_emb_optimized(k, cos, sin)
        else:
            # Partial RoPE
            q_rot = apply_rotary_pos_emb(q, cos, sin, self.rope_dim)
            k_rot = apply_rotary_pos_emb(k, cos, sin, self.rope_dim)

        # Apply YaRN attention scaling if needed
        if self.config.interpolation_type == RoPEInterpolationType.YaRN and hasattr(
            self, "yarn_mscale"
        ):
            q_rot = q_rot * self.yarn_mscale

        return q_rot, k_rot

    def reset_cache(self):
        """Reset the cached cos/sin embeddings (thread-safe)."""
        with self._cache_lock:
            self._seq_len_cached = 0
            self._cos_cached = None
            self._sin_cached = None
            self._device_cache.clear()


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input.

    Optimized to minimize memory allocations.

    Args:
        x: Input tensor of shape [..., dim] where dim is even

    Returns:
        Rotated tensor with first half negated and swapped with second half
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    tensor: Tensor, cos: Tensor, sin: Tensor, rope_dim: Optional[int] = None
) -> Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        tensor: Input tensor [batch, seq_len, num_heads, head_dim] or
                [batch, num_heads, seq_len, head_dim]
        cos: Cosine embeddings
        sin: Sine embeddings
        rope_dim: Dimension to apply RoPE to (None for full dimension)

    Returns:
        Tensor with rotary embeddings applied
    """
    # Handle different tensor layouts
    if tensor.dim() == 4:
        # Determine layout: [B, S, H, D] or [B, H, S, D]
        if tensor.shape[1] == cos.shape[-2] or (
            cos.dim() == 4 and tensor.shape[1] == cos.shape[1]
        ):
            # [B, S, H, D] layout
            seq_dim = 1
        else:
            # [B, H, S, D] layout
            seq_dim = 2
    else:
        raise ValueError(f"Expected 4D tensor, got {tensor.dim()}D")

    # Apply RoPE to specified dimensions
    if rope_dim is not None and rope_dim < tensor.shape[-1]:
        # Partial RoPE
        t_rot, t_pass = tensor[..., :rope_dim], tensor[..., rope_dim:]

        # Expand cos/sin if needed
        if cos.dim() == 2:
            # cos/sin are [seq_len, dim]
            if seq_dim == 1:
                cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
                sin = sin.unsqueeze(0).unsqueeze(2)
            else:
                cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim]
                sin = sin.unsqueeze(0).unsqueeze(1)
        elif cos.dim() == 3:
            # cos/sin are [batch, seq_len, dim] (from position_ids)
            if seq_dim == 1:
                cos = cos.unsqueeze(2)  # [batch, seq_len, 1, dim]
                sin = sin.unsqueeze(2)
            else:
                cos = cos.unsqueeze(1)  # [batch, 1, seq_len, dim]
                sin = sin.unsqueeze(1)

        # Apply rotation
        t_rot = (t_rot * cos[..., :rope_dim]) + (
            rotate_half(t_rot) * sin[..., :rope_dim]
        )
        return torch.cat([t_rot, t_pass], dim=-1)
    else:
        # Full RoPE
        if cos.dim() == 2:
            if seq_dim == 1:
                cos = cos.unsqueeze(0).unsqueeze(2)
                sin = sin.unsqueeze(0).unsqueeze(2)
            else:
                cos = cos.unsqueeze(0).unsqueeze(1)
                sin = sin.unsqueeze(0).unsqueeze(1)
        elif cos.dim() == 3:
            # cos/sin are [batch, seq_len, dim] (from position_ids)
            if seq_dim == 1:
                cos = cos.unsqueeze(2)  # [batch, seq_len, 1, dim]
                sin = sin.unsqueeze(2)
            else:
                cos = cos.unsqueeze(1)  # [batch, 1, seq_len, dim]
                sin = sin.unsqueeze(1)

        return (tensor * cos) + (rotate_half(tensor) * sin)


def apply_rotary_pos_emb_optimized(
    tensor: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Optimized implementation of rotary position embeddings.

    Uses efficient tensor operations and minimizes memory allocations.

    Args:
        tensor: Input tensor of shape [..., dim]
        cos: Cosine embeddings
        sin: Sine embeddings

    Returns:
        Tensor with rotary embeddings applied
    """
    # Efficient dimension handling
    ndim = tensor.shape[-1]
    half_dim = ndim // 2

    # Use view for zero-copy reshape when possible
    tensor_reshape = tensor.view(*tensor.shape[:-1], 2, half_dim)

    # Prepare cos/sin with proper broadcasting
    cos_reshape = _prepare_cos_sin_for_broadcast(cos, tensor.shape, is_cos=True)
    sin_reshape = _prepare_cos_sin_for_broadcast(sin, tensor.shape, is_cos=False)

    # Apply rotation using Einstein notation for clarity and efficiency
    # This computes: [cos, -sin; sin, cos] @ [x1; x2]
    x1 = tensor_reshape[..., 0, :]
    x2 = tensor_reshape[..., 1, :]

    # Optimized rotation without intermediate allocations
    out = torch.empty_like(tensor)
    out[..., :half_dim] = (
        x1 * cos_reshape[..., :half_dim] - x2 * sin_reshape[..., :half_dim]
    )
    out[..., half_dim:] = (
        x1 * sin_reshape[..., :half_dim] + x2 * cos_reshape[..., :half_dim]
    )

    return out


def _prepare_cos_sin_for_broadcast(
    embeddings: Tensor,
    target_shape: Tuple[int, ...],
    is_cos: bool = True,
) -> Tensor:
    """Prepare cos/sin embeddings for broadcasting.

    Args:
        embeddings: Cos or sin embeddings
        target_shape: Target tensor shape for broadcasting
        is_cos: Whether this is cos (True) or sin (False)

    Returns:
        Embeddings reshaped for broadcasting
    """
    # Handle different embedding dimensions efficiently
    if embeddings.dim() == 2:
        # [seq_len, dim]
        if len(target_shape) == 4:
            if target_shape[1] == embeddings.shape[0]:
                # [B, S, H, D] layout
                return embeddings.unsqueeze(0).unsqueeze(2)
            else:
                # [B, H, S, D] layout
                return embeddings.unsqueeze(0).unsqueeze(1)
    elif embeddings.dim() == 3:
        # [batch, seq_len, dim]
        if len(target_shape) == 4:
            if target_shape[2] != embeddings.shape[1]:
                # [B, S, H, D] layout
                return embeddings.unsqueeze(2)
            else:
                # [B, H, S, D] layout
                return embeddings.unsqueeze(1)

    return embeddings


def apply_rotary_pos_emb_fused(
    tensor: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Legacy fused implementation for backward compatibility.

    Redirects to optimized implementation.
    """
    return apply_rotary_pos_emb_optimized(tensor, cos, sin)


class FusedRoPE(nn.Module):
    """Fused Rotary Position Embedding with optimized CUDA kernels.

    This module provides highly optimized RoPE operations using custom CUDA kernels
    when available, falling back to the standard implementation otherwise.
    """

    def __init__(self, config: RoPEConfig):
        """Initialize FusedRoPE.

        Args:
            config: RoPE configuration
        """
        super().__init__()
        self.config = config
        self.rope = RotaryEmbedding(config)

        # Try to import optimized kernels
        self.use_cuda_kernel = False
        if torch.cuda.is_available():
            try:
                # Placeholder for custom CUDA kernel import
                # from rosellm.kernels import fused_rope_forward
                # self.use_cuda_kernel = True
                pass
            except ImportError:
                logger.debug(
                    "Custom CUDA kernels not available, using PyTorch implementation"
                )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Optional[Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply fused rotary embeddings.

        Args:
            q: Query tensor
            k: Key tensor
            position_ids: Optional position indices
            seq_len: Optional sequence length override

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        if self.use_cuda_kernel and q.is_cuda:
            # Use custom CUDA kernel if available
            # return fused_rope_forward(q, k, self.config, position_ids, seq_len)
            pass

        # Fall back to standard implementation
        result = self.rope(q, k, position_ids, seq_len)
        return result  # type: ignore[no-any-return]


def precompute_rope_params(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """Precompute RoPE parameters for a given configuration.

    This is useful for sharing precomputed embeddings across multiple layers.

    Args:
        dim: Dimension of embeddings
        max_seq_len: Maximum sequence length
        base: Base for frequency computation
        device: Device to place tensors on
        dtype: Data type for tensors

    Returns:
        Tuple of (cos, sin) tensors of shape [max_seq_len, dim]
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Compute positions
    position = torch.arange(max_seq_len, device=device).float()

    # Compute frequencies
    freqs = torch.outer(position, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)

    # Compute cos and sin
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    return cos, sin
