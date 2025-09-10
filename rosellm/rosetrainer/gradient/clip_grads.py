"""
Gradient clipping utilities for RoseTrainer.

This module provides comprehensive gradient clipping functionality with support for:
- L2 norm and max value clipping
- Multi-tensor operations for efficiency
- Distributed training with proper norm reduction
- Megatron-LM compatibility mode
- Model and expert parallel group support
"""

import logging
import warnings
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from rosellm.rosetrainer.utils.multi_tensor_ops import (
    MultiTensorOperator,
    get_default_operator,
)

# Constants for numerical stability
DEFAULT_EPSILON = 1e-6
DEFAULT_ADAPTIVE_SIGMA = 2.0

logger = logging.getLogger(__name__)


class ClipType(Enum):
    """Gradient clipping strategy."""

    NORM = "norm"  # L2 norm clipping
    VALUE = "value"  # Max value clipping
    ADAPTIVE = "adaptive"  # Adaptive clipping based on gradient statistics


class GradientClipper:
    """
    Advanced gradient clipping with multi-tensor support and distributed training.

    Features:
    - Efficient multi-tensor operations with automatic fallback
    - Distributed gradient norm reduction across model/expert parallel groups
    - Megatron-LM compatibility mode for consistent behavior
    - Per-parameter and global clipping strategies
    - Detailed clipping statistics and monitoring

    Args:
        max_norm: Maximum gradient norm for L2 clipping
        max_value: Maximum gradient value for value clipping
        clip_type: Type of clipping to apply
        model_parallel_group: Process group for model parallelism
        expert_parallel_group: Process group for expert parallelism
        use_multi_tensor: Enable multi-tensor operations
        megatron_compatible: Enable Megatron-LM compatibility mode
        check_for_nan_in_grad: Check for NaN values in gradients
        log_stats: Enable detailed clipping statistics
        epsilon: Small value to prevent division by zero in norm clipping
        adaptive_sigma: Multiplier for standard deviation in adaptive clipping
    """

    def __init__(
        self,
        max_norm: Optional[float] = None,
        max_value: Optional[float] = None,
        clip_type: ClipType = ClipType.NORM,
        model_parallel_group: Optional[dist.ProcessGroup] = None,
        expert_parallel_group: Optional[dist.ProcessGroup] = None,
        use_multi_tensor: bool = True,
        megatron_compatible: bool = False,
        check_for_nan_in_grad: bool = False,
        log_stats: bool = False,
        epsilon: float = DEFAULT_EPSILON,
        adaptive_sigma: float = DEFAULT_ADAPTIVE_SIGMA,
    ):
        self.max_norm = max_norm
        self.max_value = max_value
        self.clip_type = clip_type
        self.model_parallel_group = model_parallel_group
        self.expert_parallel_group = expert_parallel_group
        self.use_multi_tensor = use_multi_tensor
        self.megatron_compatible = megatron_compatible
        self.check_for_nan_in_grad = check_for_nan_in_grad
        self.log_stats = log_stats
        self.epsilon = epsilon
        self.adaptive_sigma = adaptive_sigma

        # Validate input parameters
        self._validate_parameters()

        # Multi-tensor operator
        self.operator: Optional[MultiTensorOperator] = None
        if use_multi_tensor:
            try:
                self.operator = get_default_operator()
            except Exception as e:
                warnings.warn(f"Failed to initialize multi-tensor operator: {e}")
                self.use_multi_tensor = False

        # Statistics tracking (thread-safe access should be handled by caller)
        self.stats: Dict[str, float] = {}
        # Adaptive clipping state
        self._adaptive_norm_history: List[float] = []
        self._adaptive_window_size: int = 100  # Moving window for adaptive statistics

    def _validate_parameters(self) -> None:
        """Validate constructor parameters."""
        if self.max_norm is not None and (
            not isinstance(self.max_norm, (int, float)) or self.max_norm <= 0
        ):
            raise ValueError(f"max_norm must be a positive number, got {self.max_norm}")

        if self.max_value is not None and (
            not isinstance(self.max_value, (int, float)) or self.max_value <= 0
        ):
            raise ValueError(
                f"max_value must be a positive number, got {self.max_value}"
            )

        if not isinstance(self.epsilon, (int, float)) or self.epsilon <= 0:
            raise ValueError(f"epsilon must be a positive number, got {self.epsilon}")

        if (
            not isinstance(self.adaptive_sigma, (int, float))
            or self.adaptive_sigma <= 0
        ):
            raise ValueError(
                f"adaptive_sigma must be a positive number, "
                f"got {self.adaptive_sigma}"
            )

    def clip_gradients(
        self,
        parameters: Union[torch.nn.Module, Iterable[torch.nn.Parameter]],
        aggregate_norm_fn: Optional[Callable[[float], float]] = None,
    ) -> Dict[str, float]:
        """
        Clip gradients of model parameters.

        Args:
            parameters: Model or list of parameters to clip
            aggregate_norm_fn: Custom function for aggregating norms

        Returns:
            Dictionary containing clipping statistics
        """
        if isinstance(parameters, torch.nn.Module):
            params = [p for p in parameters.parameters() if p.grad is not None]
        else:
            params = list(p for p in parameters if p.grad is not None)

        if not params:
            return {"total_norm": 0.0, "clipped": False}

        # Check for NaN values if requested
        if self.check_for_nan_in_grad:
            self._check_nan_gradients(params)

        # Apply clipping based on type
        if self.clip_type == ClipType.NORM:
            return self._clip_by_norm(params, aggregate_norm_fn)
        elif self.clip_type == ClipType.VALUE:
            return self._clip_by_value(params)
        elif self.clip_type == ClipType.ADAPTIVE:
            return self._clip_adaptive(params)
        else:
            raise ValueError(f"Unknown clip type: {self.clip_type}")

    def _clip_by_norm(
        self,
        parameters: List[torch.nn.Parameter],
        aggregate_norm_fn: Optional[Callable[[float], float]] = None,
    ) -> Dict[str, float]:
        """
        Clip gradients by L2 norm.

        Args:
            parameters: List of parameters to clip
            aggregate_norm_fn: Custom norm aggregation function

        Returns:
            Clipping statistics
        """
        if self.max_norm is None or self.max_norm <= 0:
            raise ValueError(
                f"max_norm must be positive for norm clipping, got {self.max_norm}"
            )

        # Compute total norm
        if self.use_multi_tensor and self.operator is not None:
            total_norm = self._compute_norm_multi_tensor(parameters)
        else:
            total_norm = self._compute_norm_single_tensor(parameters)

        # Aggregate norm across distributed groups if needed
        if aggregate_norm_fn is not None:
            total_norm = aggregate_norm_fn(total_norm)
        else:
            total_norm = self._aggregate_norm_distributed(total_norm)

        # Compute clipping coefficient
        clip_coef = self.max_norm / (total_norm + self.epsilon)
        clipped = clip_coef < 1.0

        # Apply clipping if needed
        if clipped:
            if self.use_multi_tensor and self.operator is not None:
                self._scale_gradients_multi_tensor(parameters, clip_coef)
            else:
                self._scale_gradients_single_tensor(parameters, clip_coef)

        # Update statistics
        stats = {
            "total_norm": total_norm,
            "clip_coef": clip_coef if clipped else 1.0,
            "clipped": clipped,
            "max_norm": self.max_norm,
        }

        if self.log_stats:
            self.stats.update(stats)
            self._log_statistics(stats)

        return stats

    def _clip_by_value(
        self,
        parameters: List[torch.nn.Parameter],
    ) -> Dict[str, float]:
        """
        Clip gradients by maximum value.

        Args:
            parameters: List of parameters to clip

        Returns:
            Clipping statistics
        """
        if self.max_value is None or self.max_value <= 0:
            raise ValueError(
                f"max_value must be positive for value clipping, got {self.max_value}"
            )

        num_clipped = 0
        max_grad = 0.0

        for param in parameters:
            if param.grad is None:
                continue

            grad_data = param.grad.data
            max_grad = max(max_grad, grad_data.abs().max().item())

            # Check if clipping is needed
            if (grad_data.abs() > self.max_value).any():
                num_clipped += 1
                # Clip values in-place
                grad_data.clamp_(-self.max_value, self.max_value)

        # Update statistics
        stats = {
            "max_grad": max_grad,
            "num_clipped": num_clipped,
            "clipped": num_clipped > 0,
            "max_value": self.max_value,
        }

        if self.log_stats:
            self.stats.update(stats)
            self._log_statistics(stats)

        return stats

    def _clip_adaptive(
        self,
        parameters: List[torch.nn.Parameter],
    ) -> Dict[str, float]:
        """
        Apply adaptive gradient clipping based on gradient statistics.

        This method dynamically adjusts the clipping threshold based on
        a moving window of gradient norms for more stable adaptation.

        Args:
            parameters: List of parameters to clip

        Returns:
            Clipping statistics
        """
        # Compute current total norm
        if self.use_multi_tensor and self.operator is not None:
            current_norm = self._compute_norm_multi_tensor(parameters)
        else:
            current_norm = self._compute_norm_single_tensor(parameters)
        if current_norm == 0.0:
            return {"total_norm": 0.0, "clipped": False}

        # Update moving window of norms
        self._adaptive_norm_history.append(current_norm)
        if len(self._adaptive_norm_history) > self._adaptive_window_size:
            self._adaptive_norm_history.pop(0)

        # Compute statistics from history
        if len(self._adaptive_norm_history) < 2:
            # Not enough history, use conservative clipping
            adaptive_threshold = self.max_norm or float("inf")
            mean_norm = current_norm
            std_norm = 0.0
        else:
            norms_tensor = torch.tensor(self._adaptive_norm_history)
            mean_norm = norms_tensor.mean().item()
            std_norm = norms_tensor.std().item()

            # Adaptive threshold (mean + sigma*std)
            adaptive_threshold = mean_norm + self.adaptive_sigma * std_norm

        # Compute the raw adaptive threshold for reporting
        if len(self._adaptive_norm_history) >= 2:
            raw_adaptive_threshold = mean_norm + self.adaptive_sigma * std_norm
        else:
            raw_adaptive_threshold = adaptive_threshold
        # Use constrained threshold for actual clipping
        original_max_norm = self.max_norm
        applied_threshold = min(adaptive_threshold, original_max_norm or float("inf"))
        self.max_norm = applied_threshold

        # Compute clipping coefficient directly to avoid recomputing norm
        clip_coef = self.max_norm / (current_norm + self.epsilon)
        clipped = clip_coef < 1.0

        # Apply clipping if needed
        if clipped:
            if self.use_multi_tensor and self.operator is not None:
                self._scale_gradients_multi_tensor(parameters, clip_coef)
            else:
                self._scale_gradients_single_tensor(parameters, clip_coef)

        # Restore original max_norm
        self.max_norm = original_max_norm

        # Prepare statistics
        stats = {
            "total_norm": current_norm,
            "clip_coef": clip_coef if clipped else 1.0,
            "clipped": clipped,
            # The actual applied threshold (constrained)
            "max_norm": applied_threshold,
            "mean_norm": mean_norm,
            "std_norm": std_norm,
            # The raw computed threshold
            "adaptive_threshold": raw_adaptive_threshold,
            "history_size": len(self._adaptive_norm_history),
        }

        if self.log_stats:
            self.stats.update(stats)
            self._log_statistics(stats)

        return stats

    def _compute_norm_multi_tensor(
        self,
        parameters: List[torch.nn.Parameter],
    ) -> float:
        """
        Compute L2 norm using multi-tensor operations.

        Args:
            parameters: List of parameters

        Returns:
            Total L2 norm
        """
        # Group gradients by dtype and device
        grouped_grads = self._group_gradients(parameters)

        total_norm = 0.0
        for (dtype, device), grads in grouped_grads.items():
            if self.operator is not None:
                try:
                    # Use multi-tensor L2 norm computation
                    group_norm = self.operator.calculate_norm(
                        grads, norm_type=2.0, per_tensor=False
                    )
                    if isinstance(group_norm, torch.Tensor):
                        total_norm += group_norm.item() ** 2
                    elif isinstance(group_norm, (int, float)):
                        total_norm += group_norm**2
                    else:
                        # Handle list case by fallback
                        raise ValueError("Unexpected norm format")
                except Exception:
                    # Fallback to single tensor
                    for grad in grads:
                        total_norm += grad.data.norm() ** 2
            else:
                # Single tensor computation
                for grad in grads:
                    total_norm += grad.data.norm() ** 2

        return float(total_norm**0.5)

    def _compute_norm_single_tensor(
        self,
        parameters: List[torch.nn.Parameter],
    ) -> float:
        """
        Compute L2 norm using single tensor operations with memory efficiency.

        Args:
            parameters: List of parameters

        Returns:
            Total L2 norm
        """
        # More memory-efficient approach using torch.norm with dim reduction
        device = None

        # Collect all gradient tensors and find common device/dtype
        grad_tensors = []
        for param in parameters:
            if param.grad is not None:
                grad_tensors.append(param.grad.data)
                if device is None:
                    device = param.grad.device
        if not grad_tensors:
            return 0.0

        # Use torch.stack if all tensors have same shape, else individual norms
        try:
            # Try more efficient batch computation for many small tensors
            if len(grad_tensors) > 10:
                norms_squared = torch.stack([t.norm() ** 2 for t in grad_tensors])
                return float(norms_squared.sum().sqrt())
            else:
                # Fall back to individual computation for few tensors
                total_norm_sq = sum(
                    param.grad.data.norm() ** 2
                    for param in parameters
                    if param.grad is not None
                )
                return float(total_norm_sq**0.5)
        except (RuntimeError, ValueError):
            # Fallback to original method if stacking fails
            total_norm_sq = sum(
                param.grad.data.norm() ** 2
                for param in parameters
                if param.grad is not None
            )
            return float(total_norm_sq**0.5)

    def _scale_gradients_multi_tensor(
        self,
        parameters: List[torch.nn.Parameter],
        scale: float,
    ) -> None:
        """
        Scale gradients using multi-tensor operations.

        Args:
            parameters: List of parameters
            scale: Scaling factor
        """
        # Group gradients by dtype and device
        grouped_grads = self._group_gradients(parameters)

        for (dtype, device), grads in grouped_grads.items():
            if self.operator is not None:
                try:
                    # Use multi-tensor scaling - convert scale to tensor
                    scale_tensor = torch.tensor(scale, device=device, dtype=dtype)
                    self.operator.scale_tensors(grads, scale_tensor, in_place=True)
                except Exception:
                    # Fallback to single tensor
                    for grad in grads:
                        grad.data.mul_(scale)
            else:
                # Single tensor scaling
                for grad in grads:
                    grad.data.mul_(scale)

    def _scale_gradients_single_tensor(
        self,
        parameters: List[torch.nn.Parameter],
        scale: float,
    ) -> None:
        """
        Scale gradients using single tensor operations.

        Args:
            parameters: List of parameters
            scale: Scaling factor
        """
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(scale)

    def _aggregate_norm_distributed(
        self,
        norm: float,
    ) -> float:
        """
        Aggregate gradient norm across distributed process groups.

        In Megatron-LM compatibility mode, this follows the exact
        reduction order used by Megatron for consistency.

        Args:
            norm: Local gradient norm

        Returns:
            Aggregated norm
        """
        if not dist.is_initialized():
            return norm

        norm_tensor = torch.tensor(
            [norm**2], device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Model parallel reduction
        if self.model_parallel_group is not None:
            dist.all_reduce(norm_tensor, group=self.model_parallel_group)

        # Expert parallel reduction (for MoE models)
        if self.expert_parallel_group is not None:
            dist.all_reduce(norm_tensor, group=self.expert_parallel_group)

        # In Megatron compatibility mode, follow exact reduction pattern
        if self.megatron_compatible:
            # Megatron reduces across TP then PP groups
            # This is handled by the model_parallel_group above
            pass

        return float(norm_tensor.item() ** 0.5)

    def _check_nan_gradients(
        self,
        parameters: List[torch.nn.Parameter],
    ) -> None:
        """
        Check for NaN and Inf values in gradients with detailed reporting.

        Args:
            parameters: List of parameters to check

        Raises:
            ValueError: If NaN or Inf values are found
        """
        nan_params = []
        inf_params = []

        for i, param in enumerate(parameters):
            if param.grad is not None:
                grad_data = param.grad.data

                if torch.isnan(grad_data).any():
                    nan_count = torch.isnan(grad_data).sum().item()
                    param_info = {
                        "index": i,
                        "shape": tuple(grad_data.shape),
                        "dtype": grad_data.dtype,
                        "device": grad_data.device,
                        "nan_count": nan_count,
                        "total_elements": grad_data.numel(),
                    }
                    nan_params.append(param_info)

                if torch.isinf(grad_data).any():
                    inf_count = torch.isinf(grad_data).sum().item()
                    param_info = {
                        "index": i,
                        "shape": tuple(grad_data.shape),
                        "dtype": grad_data.dtype,
                        "device": grad_data.device,
                        "inf_count": inf_count,
                        "total_elements": grad_data.numel(),
                    }
                    inf_params.append(param_info)

        if nan_params:
            param_descriptions = [
                f"param_{p['index']} "
                f"({p['nan_count']}/{p['total_elements']} elements)"
                for p in nan_params[:5]  # Limit to first 5
            ]
            error_msg = (
                f"NaN gradients found in {len(nan_params)} parameter(s): "
                + ", ".join(param_descriptions)
            )
            if len(nan_params) > 5:
                error_msg += f" and {len(nan_params) - 5} more"
            raise ValueError(error_msg)

        if inf_params:
            param_descriptions = [
                f"param_{p['index']} "
                f"({p['inf_count']}/{p['total_elements']} elements)"
                for p in inf_params[:5]  # Limit to first 5
            ]
            error_msg = (
                f"Inf gradients found in {len(inf_params)} parameter(s): "
                + ", ".join(param_descriptions)
            )
            if len(inf_params) > 5:
                error_msg += f" and {len(inf_params) - 5} more"
            raise ValueError(error_msg)

    def _group_gradients(
        self,
        parameters: List[torch.nn.Parameter],
    ) -> Dict[Tuple[torch.dtype, torch.device], List[torch.Tensor]]:
        """
        Group gradients by dtype and device for efficient processing.

        Args:
            parameters: List of parameters

        Returns:
            Grouped gradients
        """
        grouped: Dict[Tuple[torch.dtype, torch.device], List[torch.Tensor]] = {}
        for param in parameters:
            if param.grad is not None:
                key = (param.grad.dtype, param.grad.device)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(param.grad)

        return grouped

    def _log_statistics(
        self,
        stats: Dict[str, float],
    ) -> None:
        """
        Log clipping statistics for monitoring.

        Args:
            stats: Statistics dictionary
        """
        if self.clip_type == ClipType.NORM:
            logger.info(
                "Gradient norm: %.4f, Clipped: %s, Scale: %.4f",
                stats.get("total_norm", 0),
                stats.get("clipped", False),
                stats.get("clip_coef", 1.0),
            )
        elif self.clip_type == ClipType.VALUE:
            logger.info(
                "Max gradient: %.4f, Clipped params: %d",
                stats.get("max_grad", 0),
                stats.get("num_clipped", 0),
            )
        elif self.clip_type == ClipType.ADAPTIVE:
            logger.info(
                "Adaptive threshold: %.4f, Mean norm: %.4f, Clipped: %s",
                stats.get("adaptive_threshold", 0),
                stats.get("mean_norm", 0),
                stats.get("clipped", False),
            )


def clip_grad_norm(
    parameters: Union[torch.nn.Module, Iterable[torch.nn.Parameter]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    model_parallel_group: Optional[dist.ProcessGroup] = None,
) -> float:
    """
    Clip gradient norm of model parameters.

    This function provides a PyTorch-compatible interface with additional
    support for distributed training and model parallelism.

    Args:
        parameters: Model or iterable of parameters
        max_norm: Maximum norm value
        norm_type: Type of norm (only 2.0 supported for efficiency)
        error_if_nonfinite: Raise error if norm is inf or nan
        model_parallel_group: Process group for model parallelism

    Returns:
        Total norm of the parameters

    Example:
        >>> model = torch.nn.Linear(10, 10)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> loss = model(torch.randn(32, 10)).sum()
        >>> loss.backward()
        >>> norm = clip_grad_norm(model.parameters(), max_norm=1.0)
        >>> optimizer.step()
    """
    if norm_type != 2.0:
        warnings.warn(f"Only L2 norm (norm_type=2.0) is optimized. Got {norm_type}")
        # Fallback to PyTorch implementation for other norms
        if isinstance(parameters, torch.nn.Module):
            params = list(parameters.parameters())
        else:
            params = list(parameters)
        return float(
            torch.nn.utils.clip_grad_norm_(
                params, max_norm, norm_type, error_if_nonfinite
            )
        )

    clipper = GradientClipper(
        max_norm=max_norm,
        clip_type=ClipType.NORM,
        model_parallel_group=model_parallel_group,
        check_for_nan_in_grad=error_if_nonfinite,
    )

    stats = clipper.clip_gradients(parameters)

    if error_if_nonfinite:
        total_norm = stats["total_norm"]
        if torch.isnan(torch.tensor(total_norm)) or torch.isinf(
            torch.tensor(total_norm)
        ):
            raise RuntimeError(f"Gradient norm is {total_norm}")

    return stats["total_norm"]


def clip_grad_value(
    parameters: Union[torch.nn.Module, Iterable[torch.nn.Parameter]],
    clip_value: float,
) -> int:
    """
    Clip gradient values of model parameters.

    Args:
        parameters: Model or iterable of parameters
        clip_value: Maximum absolute value

    Returns:
        Number of parameters that were clipped

    Example:
        >>> model = torch.nn.Linear(10, 10)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> loss = model(torch.randn(32, 10)).sum()
        >>> loss.backward()
        >>> num_clipped = clip_grad_value(model.parameters(), clip_value=0.5)
        >>> optimizer.step()
    """
    clipper = GradientClipper(
        max_value=clip_value,
        clip_type=ClipType.VALUE,
    )

    stats = clipper.clip_gradients(parameters)
    return int(stats.get("num_clipped", 0))
