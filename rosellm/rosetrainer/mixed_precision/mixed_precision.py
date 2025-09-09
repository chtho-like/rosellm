"""
Enhanced Mixed Precision Training Manager with Dynamic Loss Scaling

This module provides a comprehensive mixed precision training manager that integrates
dynamic loss scaling, gradient management, and optimization strategies for stable
and efficient training of large language models.

Key Features:
- Integration with both legacy and dynamic gradient scalers
- Automatic precision management (FP16, BF16, FP32)
- Advanced gradient handling with overflow detection
- Memory-efficient operations with APEX integration
- Comprehensive monitoring and logging

References:
- Mixed Precision Training: https://arxiv.org/abs/1710.03740
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- APEX Documentation: https://nvidia.github.io/apex/
"""

import logging
import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from ..utils.multi_tensor_ops import (
    MultiTensorOperator,
    multi_tensor_clip_grad_norm,
    multi_tensor_scale,
)
from .dynamic_scaler import DynamicGradScaler as EnhancedDynamicGradScaler
from .dynamic_scaler import DynamicScalerConfig
from .gradient_scaler import AbstractGradScaler, GradScalerConfig

logger = logging.getLogger(__name__)


class PrecisionType(str, Enum):
    """Supported precision types for mixed precision training."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    MIXED = "mixed"  # Auto-select based on hardware


class MixedPrecisionConfig:
    """
    Configuration for mixed precision training with dynamic scaling support.

    This configuration manages both precision settings and gradient scaling
    behavior for optimal training stability and performance.
    """

    def __init__(
        self,
        # Precision settings
        precision: Union[str, PrecisionType] = PrecisionType.FP16,
        autocast_enabled: bool = True,
        autocast_dtype: Optional[torch.dtype] = None,
        # Gradient scaling configuration
        use_dynamic_scaling: bool = True,
        scaler_config: Optional[Union[GradScalerConfig, DynamicScalerConfig]] = None,
        # Advanced settings
        loss_scale: Optional[float] = None,  # For backward compatibility
        clip_gradients: bool = True,
        gradient_clip_value: float = 1.0,
        # Performance optimization
        cache_enabled: bool = True,
        sync_batch_norm: bool = False,
        # Monitoring
        log_overflow_info: bool = True,
        track_scale_history: bool = False,
    ):
        """
        Initialize mixed precision configuration.

        Args:
            precision: Precision type to use
            autocast_enabled: Enable PyTorch autocast
            autocast_dtype: Override autocast dtype (auto-detected if None)
            use_dynamic_scaling: Use dynamic gradient scaling
            scaler_config: Configuration for gradient scaler
            loss_scale: Fixed loss scale (deprecated, use scaler_config)
            clip_gradients: Enable gradient clipping
            gradient_clip_value: Maximum gradient norm
            cache_enabled: Enable autocast cache
            sync_batch_norm: Use synchronized batch normalization
            log_overflow_info: Log overflow detection information
            track_scale_history: Track scaling history for analysis
        """
        # Normalize precision type
        if isinstance(precision, str):
            precision = PrecisionType(precision.lower())
        self.precision = precision

        # Autocast configuration
        self.autocast_enabled = autocast_enabled
        self.autocast_dtype = autocast_dtype or self._get_autocast_dtype()
        self.cache_enabled = cache_enabled

        # Gradient scaling setup
        self.use_dynamic_scaling = use_dynamic_scaling

        # Gradient clipping
        self.clip_gradients = clip_gradients
        self.gradient_clip_value = gradient_clip_value

        # Advanced settings
        self.sync_batch_norm = sync_batch_norm
        self.log_overflow_info = log_overflow_info
        self.track_scale_history = track_scale_history

        # Handle legacy loss_scale parameter and set scaler config
        if loss_scale is not None and scaler_config is None:
            warnings.warn(
                "loss_scale parameter is deprecated. Use scaler_config instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.scaler_config: Union[GradScalerConfig, DynamicScalerConfig] = (
                GradScalerConfig(scaler_type="constant", initial_scale=loss_scale)
            )
        else:
            self.scaler_config = scaler_config or self._get_default_scaler_config()

        # Validate configuration
        self._validate_config()

    def _get_autocast_dtype(self) -> torch.dtype:
        """Determine appropriate autocast dtype based on precision setting."""
        if self.precision == PrecisionType.FP16:
            return torch.float16
        elif self.precision == PrecisionType.BF16:
            return torch.bfloat16
        elif self.precision == PrecisionType.MIXED:
            # Auto-select based on hardware capabilities
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    return torch.float16
            else:
                return torch.float16
        else:
            return torch.float32

    def _get_default_scaler_config(
        self,
    ) -> Union[GradScalerConfig, DynamicScalerConfig]:
        """Get default scaler configuration based on settings."""
        if self.use_dynamic_scaling:
            return DynamicScalerConfig(
                log_scale_changes=self.log_overflow_info,
                detailed_overflow_info=self.log_overflow_info,
            )
        else:
            return GradScalerConfig(scaler_type="dynamic")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Check precision support
        if self.precision == PrecisionType.BF16:
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                warnings.warn(
                    "BF16 not supported on current hardware. Consider using FP16.",
                    UserWarning,
                )

        # Validate gradient clip value
        if self.clip_gradients and self.gradient_clip_value <= 0:
            raise ValueError(
                f"gradient_clip_value must be positive, got {self.gradient_clip_value}"
            )


class MixedPrecisionManager:
    """
    Comprehensive mixed precision training manager with dynamic loss scaling.

    This manager handles all aspects of mixed precision training including:
    - Automatic precision casting with autocast
    - Dynamic or static gradient scaling
    - Overflow detection and handling
    - Gradient clipping and optimization
    - Performance monitoring and logging
    """

    def __init__(
        self,
        config: Optional[MixedPrecisionConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize the mixed precision manager.

        Args:
            config: Mixed precision configuration
            device: Target device for operations
        """
        self.config = config or MixedPrecisionConfig()

        # Device management
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Initialize gradient scaler
        self.scaler = self._create_scaler()

        # Performance tracking
        self._overflow_count = 0
        self._successful_steps = 0
        self._total_steps = 0
        self._scale_history: List[float] = []

        # Autocast context configuration
        self._autocast_kwargs = self._get_autocast_kwargs()

        # Initialize multi-tensor operator for optimized operations
        self.multi_tensor_operator = MultiTensorOperator(
            device=self.device,
            enable_benchmarking=self.config.track_scale_history,
        )

        logger.info(
            f"MixedPrecisionManager initialized: "
            f"precision={self.config.precision.value}, "
            f"dynamic_scaling={self.config.use_dynamic_scaling}, device={device}"
        )

    def _create_scaler(
        self,
    ) -> Optional[Union[AbstractGradScaler, EnhancedDynamicGradScaler]]:
        """Create appropriate gradient scaler based on configuration."""
        if self.config.precision == PrecisionType.FP32:
            # No scaling needed for FP32
            return None

        if self.config.use_dynamic_scaling and isinstance(
            self.config.scaler_config, DynamicScalerConfig
        ):
            return EnhancedDynamicGradScaler(
                config=self.config.scaler_config, device=self.device
            )
        elif isinstance(self.config.scaler_config, GradScalerConfig):
            return self.config.scaler_config.create_scaler(device=str(self.device))
        else:
            # Fallback to basic dynamic scaler
            return EnhancedDynamicGradScaler(device=self.device)

    def _get_autocast_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for autocast context."""
        if not self.config.autocast_enabled or str(self.device).startswith("cpu"):
            return {"enabled": False}

        kwargs = {
            "enabled": True,
            "dtype": self.config.autocast_dtype,
            "cache_enabled": self.config.cache_enabled,
        }

        return kwargs

    def autocast_context(self):
        """
        Create autocast context for forward pass.

        Returns:
            Autocast context manager for mixed precision forward pass
        """
        if str(self.device).startswith("cuda"):
            return autocast(**self._autocast_kwargs)
        else:
            # CPU autocast for limited ops
            return autocast(enabled=self.config.autocast_enabled)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for backward pass.

        Args:
            loss: Loss tensor to scale

        Returns:
            Scaled loss tensor (or original if no scaling)
        """
        if self.scaler is not None:
            return self.scaler.scale_loss(loss)
        return loss

    def backward_step(
        self, loss: torch.Tensor, retain_graph: bool = False, create_graph: bool = False
    ) -> None:
        """
        Perform backward pass with appropriate loss scaling.

        Args:
            loss: Loss tensor for backward pass
            retain_graph: Whether to retain computation graph
            create_graph: Whether to create graph for higher-order derivatives
        """
        scaled_loss = self.scale_loss(loss)
        scaled_loss.backward(retain_graph=retain_graph, create_graph=create_graph)

    def unscale_gradients(
        self,
        parameters: Union[nn.Module, List[torch.Tensor]],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """
        Unscale gradients before optimizer step using optimized multi-tensor operations.

        Args:
            parameters: Model parameters or optimizer
            optimizer: Optimizer (for compatibility, uses parameters if provided)
        """
        if self.scaler is not None:
            # Get the inverse scale for unscaling
            inv_scale = (
                1.0 / self.scaler.scale if hasattr(self.scaler, "scale") else None
            )

            if inv_scale is not None:
                # Use multi-tensor operations for efficient unscaling
                if isinstance(parameters, nn.Module):
                    params_with_grad = [
                        p for p in parameters.parameters() if p.grad is not None
                    ]
                else:
                    # Cast to Any to handle tensor list type checking
                    params_list: Any = parameters
                    params_with_grad = [
                        p
                        for p in params_list
                        if hasattr(p, "grad") and p.grad is not None
                    ]

                if params_with_grad:
                    # Filter out None grads and ensure we have a list of tensors
                    grads: List[torch.Tensor] = []
                    for p in params_with_grad:
                        if p.grad is not None:
                            grads.append(p.grad)

                    if grads:
                        multi_tensor_scale(grads, inv_scale, self.multi_tensor_operator)
            else:
                # Fallback to original method
                target = optimizer if optimizer is not None else parameters
                if hasattr(self.scaler, "unscale_gradients"):
                    # Enhanced scaler with direct gradient unscaling
                    self.scaler.unscale_gradients(parameters)  # type: ignore
                elif hasattr(self.scaler, "unscale_"):
                    # PyTorch-style scaler
                    self.scaler.unscale_(target)  # type: ignore[attr-defined]

    def clip_gradients(
        self,
        parameters: Union[nn.Module, List[torch.Tensor]],
        max_norm: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """
        Clip gradients using optimized multi-tensor operations.

        Args:
            parameters: Model parameters to clip
            max_norm: Maximum gradient norm (uses config default if None)

        Returns:
            Total gradient norm before clipping (if calculated)
        """
        if not self.config.clip_gradients:
            return None

        clip_value = max_norm or self.config.gradient_clip_value

        # Convert to appropriate format for multi_tensor_clip_grad_norm
        if isinstance(parameters, nn.Module):
            # Use multi-tensor clipping for efficiency
            clip_stats = multi_tensor_clip_grad_norm(
                parameters,
                clip_value,
                norm_type=2.0,
                operator=self.multi_tensor_operator,
            )
        else:
            # Convert List[torch.Tensor] to List[torch.nn.Parameter] if needed
            params_to_clip = [
                p if isinstance(p, torch.nn.Parameter) else torch.nn.Parameter(p)
                for p in parameters
            ]
            # Use multi-tensor clipping for efficiency
            clip_stats = multi_tensor_clip_grad_norm(
                params_to_clip,
                clip_value,
                norm_type=2.0,
                operator=self.multi_tensor_operator,
            )

        return torch.tensor(clip_stats["total_norm"], device=self.device)

    def check_overflow_and_step(
        self,
        parameters: Union[nn.Module, List[torch.Tensor]],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> bool:
        """
        Check for gradient overflow and update scaler.

        Args:
            parameters: Model parameters to check
            optimizer: Optimizer (for compatibility)

        Returns:
            True if overflow was detected, False otherwise
        """
        self._total_steps += 1

        if self.scaler is None:
            # No scaling, no overflow possible
            self._successful_steps += 1
            return False

        # Check for overflow using appropriate method
        if hasattr(self.scaler, "check_overflow_and_update"):
            # Enhanced dynamic scaler
            has_overflow = (
                self.scaler.check_overflow_and_update(  # type: ignore[attr-defined]
                    parameters
                )
            )
        elif hasattr(self.scaler, "update"):
            # Legacy scaler - need to check manually
            has_overflow = self._check_overflow_legacy(parameters)
            self.scaler.update(has_overflow)  # type: ignore[attr-defined]
        else:
            # No overflow checking capability
            has_overflow = False

        # Update statistics
        if has_overflow:
            self._overflow_count += 1
            if self.config.log_overflow_info:
                current_scale = (
                    float(self.scaler.scale.item())
                    if hasattr(self.scaler, "scale")
                    else 0.0
                )
                logger.warning(
                    f"Gradient overflow detected at step {self._total_steps}, "
                    f"scale={current_scale:.1f}"
                )
        else:
            self._successful_steps += 1

        # Track scale history if enabled
        if (
            self.config.track_scale_history
            and hasattr(self.scaler, "scale")
            and len(self._scale_history) < 10000
        ):
            self._scale_history.append(float(self.scaler.scale.item()))

        return bool(has_overflow)

    def _check_overflow_legacy(
        self, parameters: Union[nn.Module, List[torch.Tensor]]
    ) -> bool:
        """Legacy overflow checking for older scaler implementations."""
        if isinstance(parameters, nn.Module):
            param_list = list(parameters.parameters())
        else:
            param_list = list(parameters)  # type: ignore[assignment, arg-type]

        for param in param_list:
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return True
        return False

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Union[nn.Module, List[torch.Tensor]],
        closure: Optional[Callable[[], float]] = None,
        unscale_gradients: bool = True,
        clip_gradients: bool = True,
    ) -> bool:
        """
        Perform a complete optimizer step with mixed precision handling.

        Args:
            optimizer: Optimizer to step
            parameters: Model parameters
            closure: Optional closure for optimizer
            unscale_gradients: Whether to unscale gradients before step
            clip_gradients: Whether to clip gradients before step

        Returns:
            True if step was successful (no overflow), False if skipped due to overflow
        """
        # Unscale gradients if requested and scaler exists
        if unscale_gradients and self.scaler is not None:
            self.unscale_gradients(parameters, optimizer)

        # Check for overflow before stepping
        has_overflow = self.check_overflow_and_step(parameters, optimizer)

        if has_overflow:
            # Skip optimizer step due to overflow
            return False

        # Clip gradients if requested
        if clip_gradients:
            self.clip_gradients(parameters)

        # Perform optimizer step
        if self.scaler is not None and hasattr(self.scaler, "step"):
            # PyTorch-style scaler
            self.scaler.step(optimizer, closure)  # type: ignore[attr-defined]
        else:
            # Manual step
            optimizer.step(closure)

        return True

    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for checkpointing.

        Returns:
            Dictionary containing manager state
        """
        state = {
            "overflow_count": self._overflow_count,
            "successful_steps": self._successful_steps,
            "total_steps": self._total_steps,
            "config": {
                "precision": self.config.precision.value,
                "use_dynamic_scaling": self.config.use_dynamic_scaling,
                "autocast_enabled": self.config.autocast_enabled,
            },
        }

        # Include scaler state if available
        if self.scaler is not None and hasattr(self.scaler, "state_dict"):
            state["scaler"] = self.scaler.state_dict()

        # Include scale history if tracking is enabled
        if self.config.track_scale_history and self._scale_history:
            state["scale_history"] = self._scale_history[-100:]  # Last 100 entries

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from checkpoint.

        Args:
            state_dict: State dictionary from checkpoint
        """
        # Load statistics
        self._overflow_count = state_dict.get("overflow_count", 0)
        self._successful_steps = state_dict.get("successful_steps", 0)
        self._total_steps = state_dict.get("total_steps", 0)

        # Load scaler state if available
        if "scaler" in state_dict and self.scaler is not None:
            if hasattr(self.scaler, "load_state_dict"):
                self.scaler.load_state_dict(state_dict["scaler"])

        # Load scale history if available
        if "scale_history" in state_dict:
            self._scale_history = state_dict["scale_history"]

        logger.info(
            f"MixedPrecisionManager state loaded: "
            f"total_steps={self._total_steps}, "
            f"overflows={self._overflow_count}, "
            f"success_rate={self.get_success_rate():.2%}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get training statistics and performance metrics.

        Returns:
            Dictionary with comprehensive training statistics
        """
        stats = {
            "total_steps": self._total_steps,
            "successful_steps": self._successful_steps,
            "overflow_count": self._overflow_count,
            "success_rate": self.get_success_rate(),
            "precision": self.config.precision.value,
            "dynamic_scaling": self.config.use_dynamic_scaling,
        }

        # Add scaler information if available
        if self.scaler is not None:
            if hasattr(self.scaler, "get_scale_info"):
                # Enhanced scaler with detailed info
                stats["scaler_info"] = self.scaler.get_scale_info()  # type: ignore[attr-defined]  # noqa: E501
            elif hasattr(self.scaler, "scale"):
                # Basic scaler info
                stats["current_scale"] = float(
                    self.scaler.scale.item()
                )  # type: ignore[attr-defined]

        # Add scale history statistics if available
        if self._scale_history:
            import statistics

            scale_stats = {
                "min_scale": min(self._scale_history),
                "max_scale": max(self._scale_history),
                "mean_scale": statistics.mean(self._scale_history),
                "std_scale": (
                    statistics.stdev(self._scale_history)
                    if len(self._scale_history) > 1
                    else 0.0
                ),
            }
            stats["scale_statistics"] = scale_stats  # type: ignore[assignment]

        return stats

    def get_success_rate(self) -> float:
        """Calculate the success rate (non-overflow steps / total steps)."""
        if self._total_steps == 0:
            return 1.0
        return self._successful_steps / self._total_steps

    def reset_statistics(self) -> None:
        """Reset performance statistics."""
        self._overflow_count = 0
        self._successful_steps = 0
        self._total_steps = 0
        self._scale_history.clear()
        logger.info("Mixed precision statistics reset")


# Utility functions for common use cases
def create_mixed_precision_manager(
    precision: str = "fp16",
    use_dynamic_scaling: bool = True,
    initial_scale: float = 2**16,
    device: Optional[Union[str, torch.device]] = None,
) -> MixedPrecisionManager:
    """
    Factory function to create a mixed precision manager with common settings.

    Args:
        precision: Precision type ("fp16", "bf16", "fp32", "mixed")
        use_dynamic_scaling: Whether to use dynamic loss scaling
        initial_scale: Initial loss scale for dynamic scaling
        device: Target device

    Returns:
        Configured MixedPrecisionManager
    """
    if use_dynamic_scaling:
        scaler_config: Union[GradScalerConfig, DynamicScalerConfig] = (
            DynamicScalerConfig(initial_scale=initial_scale)
        )
    else:
        scaler_config = GradScalerConfig(
            scaler_type="constant", initial_scale=initial_scale
        )

    config = MixedPrecisionConfig(
        precision=precision,
        use_dynamic_scaling=use_dynamic_scaling,
        scaler_config=scaler_config,
    )

    return MixedPrecisionManager(config=config, device=device)


def get_precision_context(precision: str, device: Optional[torch.device] = None):
    """
    Get appropriate precision context for forward pass.

    Args:
        precision: Precision type string
        device: Target device

    Returns:
        Context manager for precision casting
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    precision_type = PrecisionType(precision.lower())

    if precision_type == PrecisionType.FP32:
        return autocast(enabled=False)
    elif precision_type == PrecisionType.FP16:
        return autocast(dtype=torch.float16)
    elif precision_type == PrecisionType.BF16:
        return autocast(dtype=torch.bfloat16)
    elif precision_type == PrecisionType.MIXED:
        # Auto-select based on hardware
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return autocast(dtype=dtype)
    else:
        return autocast(enabled=False)
