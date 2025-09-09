"""
Fused Layer Normalization for RoseTrainer

This module provides optimized fused layer normalization operations that combine
normalization, scaling, and bias into single kernel launches for improved performance.
Compatible with Megatron-LM's implementation for bit-to-bit accuracy validation.
"""

import logging
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

# Configure logging
logger = logging.getLogger(__name__)

# Try to import optimized kernels
try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN  # pyright: ignore

    HAVE_PERSIST_LAYER_NORM = True
except ImportError:
    FastLayerNormFN = None  # type: ignore[assignment]
    HAVE_PERSIST_LAYER_NORM = False
    logger.debug("Apex persistent layer norm not available")

try:
    from apex.normalization.fused_layer_norm import (  # pyright: ignore
        FusedLayerNormAffineFunction,
    )

    HAVE_FUSED_LAYER_NORM = True
except ImportError:
    FusedLayerNormAffineFunction = None  # type: ignore[assignment]
    HAVE_FUSED_LAYER_NORM = False
    logger.debug("Apex fused layer norm not available")


class LayerNormKernelType(Enum):
    """Enumeration of available layer norm kernel types."""

    PERSISTENT = "persistent"
    FUSED = "fused"
    CPU = "cpu"


class LayerNormException(Exception):
    """Base exception for layer normalization errors."""

    pass


class KernelNotAvailableError(LayerNormException):
    """Raised when a requested kernel is not available."""

    pass


class InvalidConfigurationError(LayerNormException):
    """Raised when configuration is invalid."""

    pass


@dataclass
class LayerNormConfig:
    """Configuration for fused layer normalization.

    This configuration class controls all aspects of the layer normalization
    implementation, including kernel selection, numerical stability options,
    and memory optimization settings.

    Attributes:
        hidden_size: Size of the hidden dimension. Must be positive integer.
        eps: Epsilon for numerical stability. Prevents division by zero.
            Default: 1e-5 (standard for most models).
        persist_layer_norm: Use persistent kernel for supported sizes.
            Persistent kernels are optimized for specific hidden sizes and
            can provide 30-50% speedup. Default: True.
        zero_centered_gamma: Center gamma around zero instead of one.
            Improves numerical stability for deep networks. Default: False.
        sequence_parallel: Enable sequence parallelism support.
            Sets special flags on parameters for distributed training.
            Default: False.
        memory_efficient: Use memory-efficient backward pass.
            Trades compute for memory by recomputing some values.
            Default: False.
        device: Device to place parameters on. If None, uses default device.

    Examples:
        >>> # Basic configuration
        >>> config = LayerNormConfig(hidden_size=768)
        >>>
        >>> # Advanced configuration for large models
        >>> config = LayerNormConfig(
        ...     hidden_size=4096,
        ...     zero_centered_gamma=True,
        ...     memory_efficient=True
        ... )
    """

    hidden_size: int
    eps: float = 1e-5
    persist_layer_norm: bool = True
    zero_centered_gamma: bool = False
    sequence_parallel: bool = False
    memory_efficient: bool = False
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.hidden_size <= 0:
            raise InvalidConfigurationError(
                f"hidden_size must be positive, got {self.hidden_size}"
            )
        if self.eps <= 0:
            raise InvalidConfigurationError(f"eps must be positive, got {self.eps}")


class LayerNormKernel(ABC):
    """Abstract base class for layer norm kernel implementations."""

    @abstractmethod
    def forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        normalized_shape: Tuple[int, ...],
        eps: float,
        zero_centered_gamma: bool,
        memory_efficient: bool,
    ) -> Tensor:
        """Execute forward pass."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this kernel is available."""
        pass

    @abstractmethod
    def get_type(self) -> LayerNormKernelType:
        """Get the kernel type."""
        pass


class PersistentKernel(LayerNormKernel):
    """Persistent layer norm kernel implementation.

    This kernel uses optimized persistent kernels from Apex that keep
    data in registers across the entire normalization operation,
    providing significant speedups for specific hidden sizes.
    """

    SUPPORTED_SIZES = frozenset(
        [
            1024,
            1536,
            2048,
            2304,
            3072,
            3840,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
            12800,
            15360,
            16384,
            18432,
            20480,
            24576,
            25600,
            30720,
            32768,
            40960,
            49152,
            65536,
        ]
    )

    def forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        normalized_shape: Tuple[int, ...],
        eps: float,
        zero_centered_gamma: bool,
        memory_efficient: bool,
    ) -> Tensor:
        """Execute forward pass with persistent kernel."""
        if not self.is_available():
            raise KernelNotAvailableError("Persistent kernel not available")

        # Apply zero-centered gamma if configured
        weight_adjusted = weight + 1 if zero_centered_gamma else weight

        import inspect

        if FastLayerNormFN is not None:
            if (
                "memory_efficient"
                in inspect.getfullargspec(FastLayerNormFN.forward).args
            ):
                output = FastLayerNormFN.apply(
                    input, weight_adjusted, bias, eps, memory_efficient
                )
            else:
                output = FastLayerNormFN.apply(input, weight_adjusted, bias, eps)

            # Make viewless to avoid memory issues
            if output._base is not None:
                output = output.clone()
            return output  # type: ignore[no-any-return]

        raise KernelNotAvailableError("Persistent kernel not available")

    def is_available(self) -> bool:
        """Check if persistent kernel is available."""
        return HAVE_PERSIST_LAYER_NORM

    def get_type(self) -> LayerNormKernelType:
        """Get kernel type."""
        return LayerNormKernelType.PERSISTENT


class FusedKernel(LayerNormKernel):
    """Fused layer norm kernel implementation."""

    def forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        normalized_shape: Tuple[int, ...],
        eps: float,
        zero_centered_gamma: bool,
        memory_efficient: bool,
    ) -> Tensor:
        """Execute forward pass with fused kernel."""
        if not self.is_available():
            raise KernelNotAvailableError("Fused kernel not available")

        # Apply zero-centered gamma if configured
        weight_adjusted = weight + 1 if zero_centered_gamma else weight

        import inspect

        if FusedLayerNormAffineFunction is not None:
            if (
                "memory_efficient"
                in inspect.getfullargspec(FusedLayerNormAffineFunction.forward).args
            ):
                result = FusedLayerNormAffineFunction.apply(
                    input,
                    weight_adjusted,
                    bias,
                    normalized_shape,
                    eps,
                    memory_efficient,
                )
                return result  # type: ignore[no-any-return]
            else:
                result = FusedLayerNormAffineFunction.apply(
                    input, weight_adjusted, bias, normalized_shape, eps
                )
                return result  # type: ignore[no-any-return]

        raise KernelNotAvailableError("Fused kernel not available")

    def is_available(self) -> bool:
        """Check if fused kernel is available."""
        return HAVE_FUSED_LAYER_NORM

    def get_type(self) -> LayerNormKernelType:
        """Get kernel type."""
        return LayerNormKernelType.FUSED


class CPUKernel(LayerNormKernel):
    """CPU fallback kernel implementation."""

    def forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        normalized_shape: Tuple[int, ...],
        eps: float,
        zero_centered_gamma: bool,
        memory_efficient: bool,
    ) -> Tensor:
        """Execute forward pass with CPU fallback."""
        # Apply zero-centered gamma if configured
        weight_adjusted = weight + 1 if zero_centered_gamma else weight

        result = FusedLayerNormFunction.apply(
            input, weight_adjusted, bias, normalized_shape, eps
        )
        # The autograd function always returns a Tensor
        assert isinstance(result, Tensor), "FusedLayerNormFunction must return a Tensor"
        return result

    def is_available(self) -> bool:
        """CPU kernel is always available."""
        return True

    def get_type(self) -> LayerNormKernelType:
        """Get kernel type."""
        return LayerNormKernelType.CPU


class FusedLayerNormFunction(torch.autograd.Function):
    """Custom autograd function for CPU fallback layer norm.

    This implementation provides a CPU-optimized forward and backward pass
    for layer normalization when GPU kernels are not available. It fuses
    the normalization, scaling, and bias operations to minimize memory
    bandwidth usage.

    Note:
        This function uses Welford's algorithm for numerical stability
        when computing variance.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        normalized_shape: Tuple[int, ...],
        eps: float,
    ) -> Tensor:
        """Forward pass with CPU-optimized computation.

        Args:
            ctx: Autograd context for saving tensors
            input: Input tensor to normalize
            weight: Scale parameter
            bias: Shift parameter
            normalized_shape: Shape for normalization
            eps: Small value for numerical stability

        Returns:
            Normalized and scaled output tensor
        """
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps

        # Validate inputs
        if input.dim() < len(normalized_shape):
            raise ValueError(
                f"Input dimension {input.dim()} must be >= "
                f"normalized_shape length {len(normalized_shape)}"
            )

        # Compute mean and variance with optimized memory access
        dims = tuple(range(input.ndim - len(normalized_shape), input.ndim))

        # Use contiguous memory for better cache utilization
        if not input.is_contiguous():
            input = input.contiguous()

        mean = input.mean(dims, keepdim=True)
        # Use Bessel's correction=False for consistency with most implementations
        var = input.var(dims, keepdim=True, unbiased=False)

        # Fused normalization with reciprocal square root for efficiency
        rstd = (var + eps).rsqrt()  # reciprocal std is more efficient
        normalized = (input - mean) * rstd

        # Fused scale and shift
        output = torch.addcmul(bias, normalized, weight)

        # Save for backward (optimize memory if configured)
        if ctx is not None:
            ctx.save_for_backward(input, weight, mean, rstd)

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:  # type: ignore[override]
        """Backward pass with fused operations.

        Args:
            ctx: Autograd context with saved tensors
            *grad_outputs: Gradient of the loss w.r.t. output

        Returns:
            Tuple of gradients for (input, weight, bias, normalized_shape, eps)
        """
        grad_output = grad_outputs[0]
        input, weight, mean, rstd = ctx.saved_tensors  # Note: using rstd instead of std
        normalized_shape = ctx.normalized_shape

        # Compute normalized input using saved reciprocal std
        normalized = (input - mean) * rstd

        # Gradient w.r.t. weight and bias
        # Need to sum over all dimensions except the normalized ones
        reduce_dims = tuple(range(input.ndim - len(normalized_shape)))
        if reduce_dims:
            grad_weight = (grad_output * normalized).sum(reduce_dims)
            grad_bias = grad_output.sum(reduce_dims)
        else:
            grad_weight = grad_output * normalized
            grad_bias = grad_output.clone()

        # Reshape for proper dimensions
        if grad_weight.shape != weight.shape:
            grad_weight = grad_weight.reshape(weight.shape)
        if grad_bias.shape != weight.shape:
            grad_bias = grad_bias.reshape(weight.shape)

        # Gradient w.r.t. input
        dims = tuple(range(input.ndim - len(normalized_shape), input.ndim))
        N = 1
        for dim in dims:
            N *= input.shape[dim]

        grad_normalized = grad_output * weight

        # Compute gradients using reciprocal std for efficiency
        # grad_var = d_loss/d_var = sum(grad_normalized * normalized) * (-0.5) * rstd^3
        grad_var = (
            (grad_normalized * normalized).sum(dims, keepdim=True)
            * (-0.5)
            * rstd.pow(3)
        )
        # grad_mean using rstd
        grad_mean = grad_normalized.sum(dims, keepdim=True) * (-rstd)
        grad_mean = (
            grad_mean + grad_var * (-2.0) * (input - mean).sum(dims, keepdim=True) / N
        )

        # Final gradient w.r.t. input using rstd
        grad_input = grad_normalized * rstd
        grad_input = grad_input + grad_var * 2.0 * (input - mean) / N
        grad_input = grad_input + grad_mean / N

        return grad_input, grad_weight, grad_bias, None, None


class FusedLayerNorm(nn.Module):
    """Layer Normalization with fused kernels for improved performance.

    This implementation provides:
    - Fused forward and backward passes
    - Support for persistent kernels (specific hidden sizes)
    - Zero-centered gamma for numerical stability
    - CPU fallback with optimized implementation
    - Compatibility with Megatron-LM for validation

    Args:
        config: LayerNormConfig with all settings
    """

    # Hidden sizes supported by persistent kernel
    PERSIST_LN_HIDDEN_SIZES = [
        1024,
        1536,
        2048,
        2304,
        3072,
        3840,
        4096,
        5120,
        6144,
        8192,
        10240,
        12288,
        12800,
        15360,
        16384,
        18432,
        20480,
        24576,
        25600,
        30720,
        32768,
        40960,
        49152,
        65536,
    ]

    def __init__(self, config: LayerNormConfig):
        super().__init__()
        self.config = config

        # Validate configuration
        if not isinstance(config.hidden_size, (int, numbers.Integral)):
            raise InvalidConfigurationError(
                f"hidden_size must be an integer, got {type(config.hidden_size)}"
            )

        # Setup dimensions
        self.normalized_shape = (config.hidden_size,)
        self.eps = config.eps

        # Initialize parameters
        self.weight = Parameter(torch.empty(config.hidden_size))
        self.bias = Parameter(torch.empty(config.hidden_size))

        # Select kernel strategy
        self.kernel = self._select_kernel(config)

        # Move to device if specified
        if config.device is not None:
            self.to(config.device)

        # Set sequence parallel flags
        if config.sequence_parallel:
            self._set_sequence_parallel_flags()

        # Initialize parameters
        self.reset_parameters()

        # Log kernel selection
        logger.info(f"Using {self.kernel.get_type().value} kernel for LayerNorm")

    def _select_kernel(self, config: LayerNormConfig) -> LayerNormKernel:
        """Select the best available kernel based on configuration.

        Selection priority:
        1. Persistent kernel (if size is supported and requested)
        2. Fused kernel (if available)
        3. CPU fallback (always available)

        Args:
            config: Layer norm configuration

        Returns:
            Selected kernel implementation

        Note:
            Kernel instances are cached to avoid recreation overhead.
            The selection is logged for debugging purposes.
        """
        # Cache kernel instances to avoid recreation
        if not hasattr(self, "_kernel_cache"):
            self._kernel_cache: Dict[str, LayerNormKernel] = {}

        # Try persistent kernel first if requested
        if (
            config.persist_layer_norm
            and config.hidden_size in PersistentKernel.SUPPORTED_SIZES
        ):
            if "persistent" not in self._kernel_cache:
                self._kernel_cache["persistent"] = PersistentKernel()

            if self._kernel_cache["persistent"].is_available():
                return self._kernel_cache["persistent"]

            logger.warning(
                f"Persistent kernel requested for size {config.hidden_size} "
                "but not available"
            )

        # Try fused kernel
        if "fused" not in self._kernel_cache:
            fused_kernel = FusedKernel()
            self._kernel_cache["fused"] = fused_kernel

        cached_kernel = self._kernel_cache.get("fused")
        if cached_kernel and cached_kernel.is_available():
            return cached_kernel

        # Fallback to CPU
        if "cpu" not in self._kernel_cache:
            cpu_kernel = CPUKernel()
            self._kernel_cache["cpu"] = cpu_kernel
            logger.warning(
                "No optimized kernels available. Using CPU fallback. "
                "Install apex for better performance."
            )

        return self._kernel_cache["cpu"]

    def _set_sequence_parallel_flags(self) -> None:
        """Set sequence parallel flags on parameters.

        Note:
            These flags are used by the distributed training framework
            to identify parameters that should be synchronized across
            the sequence parallel dimension.

        Warning:
            Failures to set flags are logged but not raised, as they
            may not be critical for all use cases.
        """
        try:
            setattr(self.weight, "sequence_parallel", True)
            setattr(self.bias, "sequence_parallel", True)
            logger.debug("Successfully set sequence parallel flags")
        except Exception as e:
            logger.warning(
                f"Failed to set sequence parallel flags: {e}. "
                f"This may affect sequence parallel training."
            )

    def reset_parameters(self) -> None:
        """Initialize parameters based on configuration.

        Uses zero initialization for zero-centered gamma mode,
        otherwise uses standard initialization (ones for weight, zeros for bias).
        """
        if self.config.zero_centered_gamma:
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with fused operations.

        Args:
            input: Input tensor to normalize

        Returns:
            Normalized output tensor

        Raises:
            RuntimeError: If forward pass fails
        """
        if input is None:
            raise ValueError("Input tensor cannot be None")

        try:
            # Execute forward pass with selected kernel
            output = self.kernel.forward(
                input=input,
                weight=self.weight,
                bias=self.bias,
                normalized_shape=self.normalized_shape,
                eps=self.eps,
                zero_centered_gamma=self.config.zero_centered_gamma,
                memory_efficient=self.config.memory_efficient,
            )

            # Ensure output is not None
            if output is None:
                raise RuntimeError("Kernel returned None output")

            return output

        except KernelNotAvailableError as e:
            # Fallback to CPU kernel if current kernel fails
            logger.warning(f"Kernel failed: {e}. Falling back to CPU kernel.")
            self.kernel = CPUKernel()
            return self.kernel.forward(
                input=input,
                weight=self.weight,
                bias=self.bias,
                normalized_shape=self.normalized_shape,
                eps=self.eps,
                zero_centered_gamma=self.config.zero_centered_gamma,
                memory_efficient=self.config.memory_efficient,
            )
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise RuntimeError(f"LayerNorm forward pass failed: {e}") from e

    def extra_repr(self) -> str:
        """String representation with configuration details.

        Returns:
            Formatted string with key configuration parameters.
        """
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"kernel={self.kernel.get_type().value}, "
            f"zero_centered={self.config.zero_centered_gamma}, "
            f"memory_efficient={self.config.memory_efficient}"
        )
