"""Multi-Tensor Operations for Range-Based Parameter Buffers.

This module provides optimized multi-tensor operations that work efficiently
with range-based parameter buffer mapping, enabling high-performance gradient
operations, clipping, and scaling across distributed ranks.

Key Features:
- Multi-tensor gradient scaling with range awareness
- Efficient gradient clipping across range boundaries
- Optimized norm computation for distributed parameters
- Memory-efficient tensor operations using buffer views
- Integration with existing gradient utilities
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter

from ..utils.gradient_utils import GradientClipConfig
from ..utils.multi_tensor_ops import MultiTensorOperator
from .range_aware_gradient_buffer import RangeAwareGradientBuffer
from .range_buffer_mapping import RangeBufferMapper

logger = logging.getLogger(__name__)

# Constants for performance optimization
MAX_TENSOR_BATCH_SIZE = 64  # Maximum tensors to process in one operation
MIN_TENSOR_SIZE_FOR_BATCHING = 1024  # Minimum size to benefit from batching
GRADIENT_NORM_CACHE_SIZE = 128  # Size of gradient norm cache


class RangeMultiTensorOperator:
    """Multi-tensor operator optimized for range-based parameter buffers.

    This class provides efficient multi-tensor operations that leverage
    range-based buffer organization for better memory access patterns
    and reduced communication overhead.

    Args:
        range_mapper: Range buffer mapper for parameter organization.
        gradient_buffer: Range-aware gradient buffer for efficient operations.
        device: Device for tensor operations.
        enable_benchmarking: Whether to enable operation benchmarking.
    """

    def __init__(
        self,
        range_mapper: Optional[RangeBufferMapper] = None,
        gradient_buffer: Optional[RangeAwareGradientBuffer] = None,
        device: Optional[torch.device] = None,
        enable_benchmarking: bool = False,
    ):
        self.range_mapper = range_mapper
        self.gradient_buffer = gradient_buffer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.enable_benchmarking = enable_benchmarking

        # Initialize base multi-tensor operator
        self.base_operator = MultiTensorOperator(
            device=self.device,
            enable_benchmarking=enable_benchmarking,
        )

        # Performance tracking
        self.operation_stats: Dict[str, Dict[str, float]] = {}
        self._norm_cache: Dict[int, float] = {}  # Cache for gradient norms
        self._norm_cache_step = -1  # Step counter for cache invalidation

        logger.debug(f"Initialized RangeMultiTensorOperator on device {self.device}")

    def scale_gradients(
        self,
        parameters: List[Parameter],
        scale_factor: float,
        use_ranges: bool = True,
    ) -> None:
        """Scale gradients using range-aware optimization.

        Args:
            parameters: List of parameters to scale gradients for.
            scale_factor: Scaling factor to apply.
            use_ranges: Whether to use range-based optimization.
        """
        if scale_factor == 1.0:
            return

        if not parameters:
            return

        # Use range-based scaling if available and enabled
        if use_ranges and self._can_use_range_optimization(parameters):
            self._scale_gradients_by_ranges(parameters, scale_factor)
        else:
            # Fallback to standard multi-tensor scaling
            self._scale_gradients_standard(parameters, scale_factor)

    def _can_use_range_optimization(self, parameters: List[Parameter]) -> bool:
        """Check if range-based optimization can be used."""
        try:
            if self.range_mapper is None or self.gradient_buffer is None:
                return False

            # Check if all parameters can be mapped to buffers
            for i, param in enumerate(parameters):
                if i >= len(self.range_mapper.parameters):
                    return False
                if self.range_mapper.get_parameter_buffer(i) is None:
                    return False

            return True
        except Exception as e:
            logger.debug(f"Range optimization check failed: {e}")
            return False

    def _scale_gradients_by_ranges(
        self, parameters: List[Parameter], scale_factor: float
    ) -> None:
        """Scale gradients using range buffer optimization."""
        if self.gradient_buffer is None:
            logger.warning("Gradient buffer is None, cannot use range-based scaling")
            return

        try:
            # Group parameters by their bucket ranges
            bucket_params: Dict[int, List[Parameter]] = {}
            for param in parameters:
                if param in self.gradient_buffer.param_to_bucket:
                    bucket_idx = self.gradient_buffer.param_to_bucket[param]
                    if bucket_idx not in bucket_params:
                        bucket_params[bucket_idx] = []
                    bucket_params[bucket_idx].append(param)

            if not bucket_params:
                logger.debug("No parameters found in bucket mapping")
                return

            # Scale gradients for each bucket
            scaled_buckets = 0
            for bucket_idx, bucket_params_list in bucket_params.items():
                if bucket_idx < len(self.gradient_buffer.buckets):
                    bucket = self.gradient_buffer.buckets[bucket_idx]
                    if bucket.grad_buffer is not None:
                        # Scale the entire bucket buffer at once
                        bucket.grad_buffer.mul_(scale_factor)
                        scaled_buckets += 1
                    else:
                        logger.debug(f"Bucket {bucket_idx} has no gradient buffer")
                else:
                    logger.warning(f"Bucket index {bucket_idx} out of range")

            if self.enable_benchmarking:
                logger.debug(
                    f"Scaled {scaled_buckets} buckets with factor {scale_factor}"
                )

        except Exception as e:
            logger.error(f"Range-based gradient scaling failed: {e}")
            raise RuntimeError(f"Gradient scaling operation failed: {e}") from e

    def _scale_gradients_standard(
        self, parameters: List[Parameter], scale_factor: float
    ) -> None:
        """Scale gradients using standard multi-tensor operations."""
        # Group tensors by device and dtype for efficient batching
        tensor_groups: Dict[Tuple[torch.device, torch.dtype], List[Tensor]] = {}

        for param in parameters:
            if param.grad is not None:
                key = (param.grad.device, param.grad.dtype)
                if key not in tensor_groups:
                    tensor_groups[key] = []
                tensor_groups[key].append(param.grad)

        # Apply scaling to each group
        for (device, dtype), tensors in tensor_groups.items():
            if tensors:
                self.base_operator.scale_tensors(tensors, scale_factor)

    def compute_gradient_norm(
        self,
        parameters: List[Parameter],
        norm_type: float = 2.0,
        use_ranges: bool = True,
        cache_step: Optional[int] = None,
    ) -> float:
        """Compute gradient norm with range-aware optimization.

        Args:
            parameters: List of parameters to compute norm for.
            norm_type: Type of norm to compute (1.0, 2.0, or inf).
            use_ranges: Whether to use range-based optimization.
            cache_step: Step number for caching (optional).

        Returns:
            Computed gradient norm.
        """
        # Check cache if step is provided
        if cache_step is not None and cache_step == self._norm_cache_step:
            cache_key = hash(tuple(id(p) for p in parameters))
            if cache_key in self._norm_cache:
                return self._norm_cache[cache_key]

        # Compute norm
        if use_ranges and self._can_use_range_optimization(parameters):
            norm = self._compute_gradient_norm_by_ranges(parameters, norm_type)
        else:
            norm = self._compute_gradient_norm_standard(parameters, norm_type)

        # Update cache if step is provided
        if cache_step is not None:
            if cache_step != self._norm_cache_step:
                # Clear cache for new step
                self._norm_cache.clear()
                self._norm_cache_step = cache_step

            cache_key = hash(tuple(id(p) for p in parameters))
            self._norm_cache[cache_key] = norm

        return norm

    def _compute_gradient_norm_by_ranges(
        self, parameters: List[Parameter], norm_type: float
    ) -> float:
        """Compute gradient norm using range buffer optimization."""
        if self.gradient_buffer is None:
            return 0.0

        # Collect unique buckets for the given parameters
        bucket_indices = set()
        for param in parameters:
            if param in self.gradient_buffer.param_to_bucket:
                bucket_indices.add(self.gradient_buffer.param_to_bucket[param])

        if not bucket_indices:
            return 0.0

        # Compute norm across bucket buffers
        total_norm = 0.0

        for bucket_idx in bucket_indices:
            if bucket_idx < len(self.gradient_buffer.buckets):
                bucket = self.gradient_buffer.buckets[bucket_idx]
                if bucket.grad_buffer is not None:
                    if norm_type == float("inf"):
                        bucket_norm = bucket.grad_buffer.abs().max().item()
                        total_norm = max(total_norm, bucket_norm)
                    elif norm_type == 1.0:
                        total_norm += bucket.grad_buffer.abs().sum().item()
                    else:  # L2 norm (most common)
                        total_norm += bucket.grad_buffer.pow(norm_type).sum().item()

        if norm_type != float("inf") and norm_type != 1.0:
            total_norm = total_norm ** (1.0 / norm_type)

        return total_norm

    def _compute_gradient_norm_standard(
        self, parameters: List[Parameter], norm_type: float
    ) -> float:
        """Compute gradient norm using standard operations."""
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            return 0.0

        norm_result = self.base_operator.calculate_norm(
            gradients, norm_type, per_tensor=False
        )
        if isinstance(norm_result, torch.Tensor):
            return float(norm_result.item())
        elif isinstance(norm_result, list) and len(norm_result) > 0:
            # Per-tensor norms - compute combined norm
            combined_norm = sum(
                (
                    t.item() ** norm_type
                    if isinstance(t, torch.Tensor)
                    else float(t) ** norm_type
                )
                for t in norm_result
            )
            return float(
                combined_norm ** (1.0 / norm_type) if norm_type != 0 else combined_norm
            )
        else:
            return 0.0

    def clip_gradients(
        self,
        parameters: List[Parameter],
        clip_config: GradientClipConfig,
        use_ranges: bool = True,
    ) -> Dict[str, float]:
        """Clip gradients with range-aware optimization.

        Args:
            parameters: List of parameters to clip gradients for.
            clip_config: Configuration for gradient clipping.
            use_ranges: Whether to use range-based optimization.

        Returns:
            Dictionary with clipping statistics.
        """
        if not parameters:
            return {"grad_norm": 0.0, "clipped_grad_norm": 0.0, "clip_ratio": 1.0}

        # Compute gradient norm before clipping
        grad_norm = self.compute_gradient_norm(
            parameters, norm_type=clip_config.norm_type, use_ranges=use_ranges
        )

        if grad_norm <= clip_config.max_norm or grad_norm == 0.0:
            # No clipping needed
            return {
                "grad_norm": grad_norm,
                "clipped_grad_norm": grad_norm,
                "clip_ratio": 1.0,
            }

        # Apply clipping
        clip_ratio = clip_config.max_norm / (grad_norm + 1e-8)

        if use_ranges and self._can_use_range_optimization(parameters):
            self._clip_gradients_by_ranges(parameters, clip_ratio)
        else:
            self._clip_gradients_standard(parameters, clip_ratio)

        clipped_norm = min(grad_norm, clip_config.max_norm)

        return {
            "grad_norm": grad_norm,
            "clipped_grad_norm": clipped_norm,
            "clip_ratio": clip_ratio,
        }

    def _clip_gradients_by_ranges(
        self, parameters: List[Parameter], clip_ratio: float
    ) -> None:
        """Clip gradients using range buffer optimization."""
        if self.gradient_buffer is None:
            return

        # Group parameters by buckets and apply clipping
        bucket_indices = set()
        for param in parameters:
            if param in self.gradient_buffer.param_to_bucket:
                bucket_indices.add(self.gradient_buffer.param_to_bucket[param])

        # Apply clipping to each affected bucket
        for bucket_idx in bucket_indices:
            if bucket_idx < len(self.gradient_buffer.buckets):
                bucket = self.gradient_buffer.buckets[bucket_idx]
                if bucket.grad_buffer is not None:
                    bucket.grad_buffer.mul_(clip_ratio)

    def _clip_gradients_standard(
        self, parameters: List[Parameter], clip_ratio: float
    ) -> None:
        """Clip gradients using standard operations."""
        gradients = [p.grad for p in parameters if p.grad is not None]
        if gradients:
            self.base_operator.scale_tensors(gradients, clip_ratio)

    def synchronize_buffers(self) -> None:
        """Synchronize parameter buffers after range-based operations."""
        if self.range_mapper is not None:
            try:
                self.range_mapper.copy_buffers_to_parameters()
            except Exception as e:
                logger.warning(
                    f"Failed to synchronize parameter buffers: {e}. "
                    f"Parameters may not reflect recent buffer operations."
                )
                # Don't raise here as this is a synchronization helper

    def zero_gradients(
        self,
        parameters: List[Parameter],
        use_ranges: bool = True,
        set_to_none: bool = False,
    ) -> None:
        """Zero gradients with range-aware optimization.

        Args:
            parameters: List of parameters to zero gradients for.
            use_ranges: Whether to use range-based optimization.
            set_to_none: Whether to set gradients to None instead of zero.
        """
        if use_ranges and self.gradient_buffer is not None:
            try:
                # Zero range-aware gradient buffer
                self.gradient_buffer.reset()
                return
            except Exception as e:
                logger.warning(
                    f"Failed to reset range gradient buffer: {e}. "
                    f"Falling back to standard gradient zeroing."
                )

        # Fallback to standard gradient zeroing
        for param in parameters:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def apply_weight_decay(
        self,
        parameters: List[Parameter],
        weight_decay: float,
        use_ranges: bool = True,
    ) -> None:
        """Apply weight decay with range-aware optimization.

        Args:
            parameters: List of parameters to apply weight decay to.
            weight_decay: Weight decay coefficient.
            use_ranges: Whether to use range-based optimization.
        """
        if weight_decay == 0.0:
            return

        if not parameters:
            return

        if use_ranges and self._can_use_range_optimization(parameters):
            self._apply_weight_decay_by_ranges(parameters, weight_decay)
        else:
            self._apply_weight_decay_standard(parameters, weight_decay)

    def _apply_weight_decay_by_ranges(
        self, parameters: List[Parameter], weight_decay: float
    ) -> None:
        """Apply weight decay using range buffer optimization."""
        if self.range_mapper is None:
            return

        # Apply weight decay to parameter buffers
        for dtype, buffer in self.range_mapper.buffers.items():
            if buffer.numel() > 0:
                # Apply weight decay directly to buffer
                # Note: This assumes the buffer contains parameter data
                buffer.data.mul_(1.0 - weight_decay)

    def _apply_weight_decay_standard(
        self, parameters: List[Parameter], weight_decay: float
    ) -> None:
        """Apply weight decay using standard operations."""
        param_tensors = [p for p in parameters if p.requires_grad]
        if param_tensors:
            # Group by device and dtype for efficiency
            tensor_groups: Dict[Tuple[torch.device, torch.dtype], List[Tensor]] = {}

            for param in param_tensors:
                key = (param.device, param.dtype)
                if key not in tensor_groups:
                    tensor_groups[key] = []
                tensor_groups[key].append(param)

            # Apply weight decay to each group
            for (device, dtype), tensors in tensor_groups.items():
                if tensors:
                    self.base_operator.scale_tensors(tensors, 1.0 - weight_decay)

    def get_operation_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for operations.

        Returns:
            Dictionary with operation timing and performance statistics.
        """
        return self.operation_stats.copy()

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.operation_stats.clear()
        self._norm_cache.clear()
        self._norm_cache_step = -1

    def __repr__(self) -> str:
        """String representation of the range multi-tensor operator."""
        return (
            f"RangeMultiTensorOperator("
            f"device={self.device}, "
            f"has_range_mapper={self.range_mapper is not None}, "
            f"has_gradient_buffer={self.gradient_buffer is not None})"
        )


def create_range_multi_tensor_operator(
    range_mapper: Optional[RangeBufferMapper] = None,
    gradient_buffer: Optional[RangeAwareGradientBuffer] = None,
    device: Optional[torch.device] = None,
    enable_benchmarking: bool = False,
) -> RangeMultiTensorOperator:
    """Factory function to create range multi-tensor operator.

    Args:
        range_mapper: Range buffer mapper for parameter organization.
        gradient_buffer: Range-aware gradient buffer.
        device: Device for operations.
        enable_benchmarking: Whether to enable benchmarking.

    Returns:
        Configured RangeMultiTensorOperator instance.
    """
    return RangeMultiTensorOperator(
        range_mapper=range_mapper,
        gradient_buffer=gradient_buffer,
        device=device,
        enable_benchmarking=enable_benchmarking,
    )


def multi_tensor_range_scale(
    tensors: List[Tensor],
    scale_factor: float,
    range_operator: Optional[RangeMultiTensorOperator] = None,
) -> None:
    """Scale multiple tensors using range-aware optimization.

    Args:
        tensors: List of tensors to scale.
        scale_factor: Scaling factor to apply.
        range_operator: Range multi-tensor operator (optional).
    """
    if scale_factor == 1.0:
        return

    if not tensors:
        return

    if range_operator is not None:
        # Use range-aware scaling if operator is available
        try:
            # Convert tensors to parameters for range operations
            parameters: List[Parameter] = []
            for tensor in tensors:
                if hasattr(tensor, "grad") and tensor.grad is not None:
                    # tensors with grad attribute should be Parameters
                    parameters.append(tensor)  # type: ignore[arg-type]

            if parameters:
                range_operator.scale_gradients(parameters, scale_factor)
                return
        except Exception as e:
            logger.warning(
                f"Range-aware scaling failed: {e}, falling back to standard scaling"
            )

    # Fallback to standard multi-tensor scaling
    for tensor in tensors:
        tensor.mul_(scale_factor)
