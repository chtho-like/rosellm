"""
Advanced Gradient Utilities with Multi-Tensor Operations

This module provides optimized gradient handling utilities inspired by Megatron-LM's
gradient utilities, with fallback mechanisms for compatibility across different
PyTorch versions and hardware configurations.

Key Features:
- Multi-tensor gradient norm calculation with APEX fallback
- Model-parallel aware gradient operations
- Configuration-driven gradient clipping strategies
- Robust error handling and graceful degradation
- Memory-efficient gradient synchronization

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- APEX Multi-Tensor: https://github.com/NVIDIA/apex
"""

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from ..parallelism.parallel_state import (
        get_data_parallel_group,
        get_tensor_model_parallel_group,
    )
    from ..parallelism.parallel_state import is_initialized as parallel_initialized
except ImportError:
    # Fallback for absolute imports when module is imported differently
    from rosellm.rosetrainer.parallelism.parallel_state import (
        get_data_parallel_group,
        get_tensor_model_parallel_group,
    )
    from rosellm.rosetrainer.parallelism.parallel_state import (
        is_initialized as parallel_initialized,
    )

logger = logging.getLogger(__name__)


@dataclass
class GradientClipConfig:
    """Configuration for gradient clipping operations."""

    clip_type: str = "norm"  # "norm", "value", "none"
    max_norm: float = 1.0
    norm_type: float = 2.0
    error_if_nonfinite: bool = True
    model_parallel_reduce: bool = True
    use_multitensor: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.clip_type not in ["norm", "value", "none"]:
            raise ValueError(f"Invalid clip_type: {self.clip_type}")
        if self.max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {self.max_norm}")
        if self.norm_type <= 0:
            raise ValueError(f"norm_type must be positive, got {self.norm_type}")


def _try_import_apex_multitensor() -> Tuple[bool, Optional[Any]]:
    """
    Try to import APEX multi-tensor utilities with graceful fallback.

    Returns:
        Tuple of (success, multi_tensor_applier or None)
    """
    try:
        from apex.multi_tensor_apply import (  # type: ignore[import-untyped]
            multi_tensor_applier,
        )

        return True, multi_tensor_applier
    except (ImportError, ModuleNotFoundError, AttributeError):
        logger.debug("APEX multi_tensor_apply not available, using PyTorch fallback")
        return False, None


def _get_model_parameters(
    model: nn.Module, requires_grad_only: bool = True
) -> List[torch.Tensor]:
    """
    Extract model parameters as a list of tensors.

    Args:
        model: PyTorch model
        requires_grad_only: Only include parameters that require gradients

    Returns:
        List of parameter tensors
    """
    if requires_grad_only:
        return [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    else:
        return [p for p in model.parameters() if p.grad is not None]


def _get_gradient_tensors(parameters: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Extract gradient tensors from parameters.

    Args:
        parameters: List of parameter tensors

    Returns:
        List of gradient tensors
    """
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad)
    return gradients


def _get_default_device(
    parameters: Union[List[torch.Tensor], nn.Module],
    fallback_device: Optional[torch.device] = None,
) -> torch.device:
    """
    Get the default device from parameters or fallback options.

    Args:
        parameters: Model parameters or list of tensors
        fallback_device: Fallback device if none can be determined

    Returns:
        Device to use for tensor creation
    """
    # Try to get device from model parameters
    if isinstance(parameters, nn.Module):
        try:
            return next(parameters.parameters()).device
        except StopIteration:
            pass
    elif parameters and hasattr(parameters[0], "device"):
        return parameters[0].device

    # Use provided fallback
    if fallback_device is not None:
        return fallback_device
    # Default fallbacks
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        return torch.device("cpu")


def calculate_gradient_norm_multitensor(
    parameters: Union[List[torch.Tensor], nn.Module],
    norm_type: float = 2.0,
    use_multitensor: bool = True,
    model_parallel_reduce: bool = True,
) -> torch.Tensor:
    """
    Calculate gradient norm across multiple tensors with optimizations.

    This function provides an optimized gradient norm calculation with multi-tensor
    operations when available, falling back to standard PyTorch operations otherwise.

    Args:
        parameters: Model parameters or list of tensors
        norm_type: Type of norm to calculate (1, 2, inf, etc.)
        use_multitensor: Whether to use APEX multi-tensor operations if available
        model_parallel_reduce: Whether to reduce across model parallel groups

    Returns:
        Gradient norm as a scalar tensor

    Raises:
        ValueError: If norm_type is invalid
        RuntimeError: If gradient calculation fails unexpectedly
    """
    # Input validation
    if norm_type <= 0 and norm_type != float("inf"):
        raise ValueError(f"norm_type must be positive or inf, got {norm_type}")

    # Convert model to parameter list if needed
    if isinstance(parameters, nn.Module):
        param_list = _get_model_parameters(parameters, requires_grad_only=True)
    else:
        param_list = [p for p in parameters if p.grad is not None]

    if not param_list:
        logger.debug("No parameters with gradients found for norm calculation")
        device = _get_default_device(parameters)
        return torch.tensor(0.0, device=device)

    # Extract gradients
    gradients = _get_gradient_tensors(param_list)

    if not gradients:
        return torch.tensor(0.0, device=param_list[0].device)

    # Try to use multi-tensor operations if requested and available
    if use_multitensor:
        has_apex, multi_tensor_applier = _try_import_apex_multitensor()

        if has_apex and multi_tensor_applier is not None:
            try:
                # Use APEX multi-tensor L2 norm for efficiency
                if norm_type == 2.0:
                    total_norm = _calculate_norm_apex_multitensor(
                        gradients, multi_tensor_applier
                    )
                else:
                    # Fall back to standard calculation for non-L2 norms
                    total_norm = _calculate_norm_standard(gradients, norm_type)
            except Exception as e:
                logger.warning(
                    f"APEX multi-tensor norm calculation failed: {e}, "
                    "falling back to standard"
                )
                total_norm = _calculate_norm_standard(gradients, norm_type)
        else:
            total_norm = _calculate_norm_standard(gradients, norm_type)
    else:
        total_norm = _calculate_norm_standard(gradients, norm_type)

    # Reduce across model parallel groups if requested and available
    if model_parallel_reduce and parallel_initialized():
        total_norm = _reduce_across_model_parallel_groups(total_norm, norm_type)

    return total_norm


def _calculate_norm_apex_multitensor(
    gradients: List[torch.Tensor], multi_tensor_applier: Any
) -> torch.Tensor:
    """
    Calculate L2 norm using APEX multi-tensor operations.

    Args:
        gradients: List of gradient tensors
        multi_tensor_applier: APEX multi-tensor applier

    Returns:
        L2 norm as scalar tensor
    """
    if not gradients:
        return torch.tensor(0.0, device=torch.device("cpu"))

    try:
        # Import APEX L2 norm kernel
        import amp_C  # type: ignore[import-untyped, import-not-found]

        # Group gradients by device and dtype for efficiency
        grouped_grads = _group_tensors_by_device_dtype(gradients)

        if not grouped_grads:
            return torch.tensor(0.0, device=gradients[0].device)

        total_norm_squared = torch.tensor(
            0.0, device=gradients[0].device, dtype=gradients[0].dtype
        )

        for (device, dtype), grad_group in grouped_grads.items():
            if not grad_group:
                continue

            try:
                # Use multi-tensor L2 norm
                group_norm = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    torch.tensor(0.0, device=device, dtype=dtype),
                    [grad_group],
                    False,
                )

                if torch.isfinite(group_norm):
                    total_norm_squared += group_norm**2
                else:
                    logger.warning(
                        f"Non-finite norm from APEX multi-tensor: {group_norm}"
                    )
                    # Fall back to standard for this group
                    group_std_norm = _calculate_norm_standard(grad_group, 2.0)
                    total_norm_squared += group_std_norm**2

            except Exception as group_e:
                logger.debug(
                    f"APEX multi-tensor norm failed for device {device}: {group_e}"
                )
                # Fall back to standard for this group
                group_std_norm = _calculate_norm_standard(grad_group, 2.0)
                total_norm_squared += group_std_norm**2

        return torch.sqrt(total_norm_squared)

    except ImportError as ie:
        logger.debug(f"APEX import failed: {ie}")
        return _calculate_norm_standard(gradients, 2.0)
    except Exception as e:
        logger.debug(f"APEX multi-tensor norm failed: {e}")
        # Fall back to standard calculation
        return _calculate_norm_standard(gradients, 2.0)


def _calculate_norm_standard(
    gradients: List[torch.Tensor], norm_type: float
) -> torch.Tensor:
    """
    Calculate gradient norm using standard PyTorch operations with numerical stability.

    Args:
        gradients: List of gradient tensors
        norm_type: Type of norm to calculate

    Returns:
        Gradient norm as scalar tensor
    """
    if not gradients:
        return torch.tensor(0.0)

    device = gradients[0].device
    dtype = gradients[0].dtype

    # Filter out empty or NaN/Inf gradients for stability
    valid_gradients = []
    for grad in gradients:
        if grad.numel() > 0 and torch.isfinite(grad).any():
            valid_gradients.append(grad)
    if not valid_gradients:
        logger.warning("No valid finite gradients found for norm calculation")
        return torch.tensor(0.0, device=device, dtype=dtype)

    if norm_type == float("inf"):
        # Infinity norm
        total_norm = torch.tensor(0.0, device=device, dtype=dtype)
        for grad in valid_gradients:
            # Only consider finite values for inf norm
            finite_mask = torch.isfinite(grad)
            if finite_mask.any():
                grad_norm = grad[finite_mask].abs().max()
                total_norm = torch.max(total_norm, grad_norm)
    else:
        # P-norm calculation with numerical stability
        total_norm_pow = torch.tensor(0.0, device=device, dtype=dtype)

        for grad in valid_gradients:
            # Only consider finite values
            finite_mask = torch.isfinite(grad)
            if finite_mask.any():
                finite_grad = grad[finite_mask]
                grad_norm_pow = torch.norm(finite_grad, p=norm_type) ** norm_type
                total_norm_pow += grad_norm_pow

        # Handle numerical edge cases
        if total_norm_pow == 0:
            total_norm = torch.tensor(0.0, device=device, dtype=dtype)
        elif torch.isinf(total_norm_pow):
            logger.warning("Infinite norm detected, using fallback calculation")
            total_norm = torch.tensor(float("inf"), device=device, dtype=dtype)
        else:
            total_norm = total_norm_pow ** (1.0 / norm_type)

    return total_norm


def _group_tensors_by_device_dtype(
    tensors: List[torch.Tensor],
) -> Dict[Tuple[torch.device, torch.dtype], List[torch.Tensor]]:
    """
    Group tensors by device and dtype for efficient multi-tensor operations.

    Args:
        tensors: List of tensors to group

    Returns:
        Dictionary mapping (device, dtype) to list of tensors
    """
    groups: Dict[Tuple[torch.device, torch.dtype], List[torch.Tensor]] = {}

    for tensor in tensors:
        key = (tensor.device, tensor.dtype)
        if key not in groups:
            groups[key] = []
        groups[key].append(tensor)

    return groups


def _reduce_across_model_parallel_groups(
    norm: torch.Tensor, norm_type: float
) -> torch.Tensor:
    """
    Reduce gradient norm across model parallel groups.

    Args:
        norm: Local gradient norm
        norm_type: Type of norm being calculated

    Returns:
        Reduced gradient norm
    """
    # Get tensor parallel group
    tp_group = get_tensor_model_parallel_group()

    if tp_group is not None:
        if norm_type == float("inf"):
            # For infinity norm, take max across all ranks
            dist.all_reduce(norm, op=dist.ReduceOp.MAX, group=tp_group)
        else:
            # For p-norms, sum the p-th powers then take p-th root
            norm_pow = norm**norm_type
            dist.all_reduce(norm_pow, op=dist.ReduceOp.SUM, group=tp_group)
            norm = norm_pow ** (1.0 / norm_type)

    return norm


def apply_gradient_clipping(
    parameters: Union[List[torch.Tensor], nn.Module],
    config: GradientClipConfig,
) -> Dict[str, float]:
    """
    Apply gradient clipping with comprehensive configuration options.

    Args:
        parameters: Model parameters or list of tensors
        config: Gradient clipping configuration

    Returns:
        Dictionary with clipping statistics

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If gradient clipping fails
    """
    # Input validation
    if not isinstance(config, GradientClipConfig):
        raise ValueError("config must be a GradientClipConfig instance")

    if config.clip_type not in ["norm", "value", "none"]:
        raise ValueError(f"Invalid clip_type: {config.clip_type}")

    stats = {
        "grad_norm": 0.0,
        "clipped": False,
        "scale_factor": 1.0,
        "num_parameters": 0,
        "num_gradients": 0,
    }

    try:
        # Convert model to parameter list if needed
        if isinstance(parameters, nn.Module):
            param_list = _get_model_parameters(parameters, requires_grad_only=True)
        else:
            param_list = [p for p in parameters if p.grad is not None]

        stats["num_parameters"] = len(param_list)

        if not param_list:
            logger.warning("No parameters with gradients found for clipping")
            return stats

        gradients = _get_gradient_tensors(param_list)
        stats["num_gradients"] = len(gradients)

        if config.clip_type == "none":
            # Calculate norm but don't clip
            if gradients:
                grad_norm = calculate_gradient_norm_multitensor(
                    param_list,
                    norm_type=config.norm_type,
                    use_multitensor=config.use_multitensor,
                    model_parallel_reduce=config.model_parallel_reduce,
                )
                stats["grad_norm"] = float(grad_norm)
            return stats

        elif config.clip_type == "norm":
            return _apply_norm_clipping(param_list, config, stats)

        elif config.clip_type == "value":
            return _apply_value_clipping(param_list, config, stats)

        else:
            raise ValueError(f"Unknown clip_type: {config.clip_type}")

    except Exception as e:
        logger.error(f"Gradient clipping failed: {e}")
        if config.error_if_nonfinite:
            raise RuntimeError(f"Gradient clipping failed: {e}") from e
        return stats


def _apply_norm_clipping(
    parameters: List[torch.Tensor],
    config: GradientClipConfig,
    stats: Dict[str, float],
) -> Dict[str, float]:
    """Apply gradient norm clipping."""
    # Calculate total norm
    grad_norm = calculate_gradient_norm_multitensor(
        parameters,
        norm_type=config.norm_type,
        use_multitensor=config.use_multitensor,
        model_parallel_reduce=config.model_parallel_reduce,
    )

    stats["grad_norm"] = float(grad_norm)

    # Check for non-finite values
    if config.error_if_nonfinite and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
        raise RuntimeError(f"Non-finite gradient norm detected: {grad_norm}")

    # Calculate clipping factor
    max_norm = config.max_norm
    clip_coeff = max_norm / (grad_norm + 1e-6)  # Add epsilon for numerical stability

    if clip_coeff < 1.0:
        stats["clipped"] = True
        stats["scale_factor"] = float(clip_coeff)

        # Apply clipping
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(clip_coeff)

    return stats


def _apply_value_clipping(
    parameters: List[torch.Tensor],
    config: GradientClipConfig,
    stats: Dict[str, float],
) -> Dict[str, float]:
    """Apply gradient value clipping."""
    max_val = config.max_norm
    clipped_any = False

    for param in parameters:
        if param.grad is not None:
            # Check for non-finite values
            if config.error_if_nonfinite:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    raise RuntimeError("Non-finite gradient values detected")

            # Clip values
            original_grad = param.grad.clone()
            param.grad.clamp_(-max_val, max_val)

            if not torch.equal(original_grad, param.grad):
                clipped_any = True

    stats["clipped"] = clipped_any
    if clipped_any:
        # Recalculate norm after clipping
        post_clip_norm = calculate_gradient_norm_multitensor(
            parameters,
            norm_type=config.norm_type,
            use_multitensor=config.use_multitensor,
            model_parallel_reduce=config.model_parallel_reduce,
        )
        stats["grad_norm"] = float(post_clip_norm)

    return stats


def check_gradient_finite(
    parameters: Union[List[torch.nn.Parameter], nn.Module],
    raise_on_nonfinite: bool = True,
) -> Tuple[bool, Dict[str, int]]:
    """
    Check if all gradients are finite (not NaN or Inf).

    Args:
        parameters: Model parameters or list of tensors
        raise_on_nonfinite: Whether to raise exception on non-finite gradients

    Returns:
        Tuple of (all_finite, stats_dict)

    Raises:
        RuntimeError: If non-finite gradients found and raise_on_nonfinite is True
    """
    stats = {
        "total_parameters": 0,
        "parameters_with_grad": 0,
        "nan_parameters": 0,
        "inf_parameters": 0,
    }

    # Convert model to parameter list if needed
    if isinstance(parameters, nn.Module):
        param_list = list(parameters.parameters())
    else:
        param_list = list(parameters)

    stats["total_parameters"] = len(param_list)

    nan_count = 0
    inf_count = 0

    for param in param_list:
        if param.grad is not None:
            stats["parameters_with_grad"] += 1

            if torch.isnan(param.grad).any():
                nan_count += 1

            if torch.isinf(param.grad).any():
                inf_count += 1

    stats["nan_parameters"] = nan_count
    stats["inf_parameters"] = inf_count

    all_finite = nan_count == 0 and inf_count == 0

    if not all_finite and raise_on_nonfinite:
        raise RuntimeError(
            f"Non-finite gradients detected: {nan_count} NaN, "
            f"{inf_count} Inf parameters"
        )

    return all_finite, stats


def sync_gradients(
    model: nn.Module,
    data_parallel_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """
    Synchronize gradients across data parallel ranks.

    Args:
        model: PyTorch model
        data_parallel_group: Process group for data parallelism (optional)
    """
    if not parallel_initialized():
        return

    # Get data parallel group if not provided
    if data_parallel_group is None:
        data_parallel_group = get_data_parallel_group()

    if data_parallel_group is None:
        return  # No data parallelism

    # Synchronize gradients
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, group=data_parallel_group)
            # Average gradients
            if data_parallel_group is not None:
                param.grad.div_(dist.get_world_size(data_parallel_group))


@contextlib.contextmanager
def gradient_accumulation_context(
    model: nn.Module,
    accumulation_steps: int,
    sync_on_last_step: bool = True,
) -> Iterator[bool]:
    """
    Context manager for gradient accumulation with optional synchronization control.

    Args:
        model: PyTorch model
        accumulation_steps: Number of accumulation steps
        sync_on_last_step: Whether to sync gradients only on the last step

    Yields:
        Boolean indicating if this is the last accumulation step
    """
    # Use a module-level counter to avoid mypy issues
    if not hasattr(gradient_accumulation_context, "_step_counter"):
        setattr(gradient_accumulation_context, "_step_counter", 0)

    step = getattr(gradient_accumulation_context, "_step_counter", 0)
    is_last_step = (step + 1) % accumulation_steps == 0

    # Disable gradient synchronization if not the last step
    if sync_on_last_step and not is_last_step:
        if hasattr(model, "no_sync"):  # DDP model
            with model.no_sync():  # type: ignore[operator]
                yield is_last_step
        else:
            yield is_last_step
    else:
        yield is_last_step

    setattr(gradient_accumulation_context, "_step_counter", step + 1)


def get_gradient_stats(
    parameters: Union[List[torch.nn.Parameter], nn.Module],
    include_histograms: bool = False,
) -> Dict[str, Any]:
    """
    Get comprehensive gradient statistics for monitoring and debugging.

    Args:
        parameters: Model parameters or list of tensors
        include_histograms: Whether to include gradient histograms (expensive)

    Returns:
        Dictionary with gradient statistics
    """
    stats = {
        "total_parameters": 0,
        "parameters_with_grad": 0,
        "grad_norm_l1": 0.0,
        "grad_norm_l2": 0.0,
        "grad_norm_inf": 0.0,
        "grad_mean": 0.0,
        "grad_std": 0.0,
        "grad_min": 0.0,
        "grad_max": 0.0,
        "zero_grad_parameters": 0,
        "finite": True,
    }

    # Convert model to parameter list if needed
    if isinstance(parameters, nn.Module):
        param_list = list(parameters.parameters())
    else:
        param_list = list(parameters)

    stats["total_parameters"] = len(param_list)

    all_grads = []
    zero_grad_count = 0

    for param in param_list:
        if param.grad is not None:
            stats["parameters_with_grad"] += 1

            grad_flat = param.grad.flatten()

            # Check for zero gradients
            if torch.allclose(param.grad, torch.zeros_like(param.grad)):
                zero_grad_count += 1

            all_grads.append(grad_flat)

    stats["zero_grad_parameters"] = zero_grad_count

    if all_grads:
        # Optimized calculation without full concatenation
        # Calculate norms efficiently
        l1_norm_squared = torch.tensor(0.0, device=all_grads[0].device)
        l2_norm_squared = torch.tensor(0.0, device=all_grads[0].device)
        inf_norm = torch.tensor(0.0, device=all_grads[0].device)

        total_elements = 0
        sum_vals = torch.tensor(0.0, device=all_grads[0].device)
        sum_squared = torch.tensor(0.0, device=all_grads[0].device)
        min_val = torch.tensor(float("inf"), device=all_grads[0].device)
        max_val = torch.tensor(float("-inf"), device=all_grads[0].device)
        all_finite = True

        for grad in all_grads:
            # Update norms
            l1_norm_squared += torch.norm(grad, p=1)
            l2_norm_squared += torch.norm(grad, p=2) ** 2
            inf_norm = torch.max(inf_norm, torch.norm(grad, p=float("inf")))

            # Update statistics
            total_elements += grad.numel()
            sum_vals += grad.sum()
            sum_squared += (grad**2).sum()
            min_val = torch.min(min_val, grad.min())
            max_val = torch.max(max_val, grad.max())

            # Check finiteness
            if all_finite and not torch.isfinite(grad).all():
                all_finite = False

        stats["grad_norm_l1"] = float(l1_norm_squared)
        stats["grad_norm_l2"] = float(torch.sqrt(l2_norm_squared))
        stats["grad_norm_inf"] = float(inf_norm)

        # Calculate statistics
        mean_val = sum_vals / total_elements
        var_val = (sum_squared / total_elements) - (mean_val**2)
        stats["grad_mean"] = float(mean_val)
        stats["grad_std"] = float(torch.sqrt(torch.clamp(var_val, min=0.0)))
        stats["grad_min"] = float(min_val)
        stats["grad_max"] = float(max_val)

        # Check finiteness
        stats["finite"] = all_finite

        # Add histograms if requested
        if include_histograms:
            try:
                # Concatenate gradients only if histograms are needed
                all_grads_tensor = torch.cat(all_grads)
                hist_values, hist_bins = torch.histogram(
                    all_grads_tensor, bins=50, range=None
                )
                # Use separate variable for histogram data to avoid type issues
                histogram_data = {
                    "values": hist_values.cpu().numpy().tolist(),
                    "bins": hist_bins.cpu().numpy().tolist(),
                }
                stats["histogram"] = histogram_data  # type: ignore
            except Exception as e:
                logger.warning(f"Failed to compute gradient histogram: {e}")

    return stats
