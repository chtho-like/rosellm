import logging
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from ..config import PrecisionType
from ..mixed_precision import AbstractGradScaler, ConstantGradScaler, DynamicGradScaler

logger = logging.getLogger(__name__)

# References:
# [1] Micikevicius, P. et al. "Mixed Precision Training."
#     arXiv:1710.03740 (2017)
# [2] NVIDIA Apex implementation:
#     https://github.com/NVIDIA/apex
# [3] PyTorch AMP implementation:
#     https://pytorch.org/docs/stable/amp.html
# [4] Shoeybi, M. et al. "Megatron-LM: Training Multi-Billion Parameter Language Models
#     Using Model Parallelism." arXiv:1909.08053 (2019)
# [5] FP8 Training: https://arxiv.org/abs/2209.05433


class DynamicLossScaler:
    """
    Implements dynamic loss scaling for mixed precision training.
    Adjusts the loss scale automatically based on gradient overflow.
    """

    def __init__(
        self,
        init_scale: float = 2**16,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2**24,
    ) -> None:
        """
        Initialize the dynamic loss scaler.

        Args:
            init_scale: Initial loss scale.
            scale_factor: Factor by which to increase/decrease the loss scale.
            scale_window: Number of iterations before increasing the loss scale.
            min_scale: Minimum loss scale.
            max_scale: Maximum loss scale.
        """
        self.cur_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.consecutive_successful_steps = 0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss tensor."""
        return loss * self.cur_scale

    def unscale(self, values: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        """Unscale gradient values in-place."""
        if isinstance(values, torch.Tensor):
            values.div_(self.cur_scale)
        else:
            for val in values:
                if val is not None:
                    val.div_(self.cur_scale)

    def update_scale(self, overflow: bool) -> None:
        """
        Update the loss scale based on gradient overflow status.

        Args:
            overflow: Whether a gradient overflow occurred.
        """
        if overflow:
            # Decrease loss scale by scale_factor and reset counter
            self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
            self.consecutive_successful_steps = 0
            logger.warning(
                f"Gradient overflow, reducing loss scale to {self.cur_scale}"
            )
        else:
            # Increment counter of successful steps
            self.consecutive_successful_steps += 1

            # If we've had scale_window successful steps, increase the loss scale
            if self.consecutive_successful_steps >= self.scale_window:
                self.cur_scale = min(self.cur_scale * self.scale_factor, self.max_scale)
                self.consecutive_successful_steps = 0
                logger.info(
                    f"No overflow for {self.scale_window} steps, "
                    f"increasing loss scale to {self.cur_scale}"
                )

    def get_scale(self) -> float:
        """Get the current loss scale."""
        return self.cur_scale


def check_overflow(params: List[torch.nn.Parameter]) -> bool:
    """
    Check for gradient overflow in parameters.

    Args:
        params: List of parameters to check.

    Returns:
        True if overflow occurred, False otherwise.
    """
    for p in params:
        if p.grad is not None:
            grad = p.grad.data
            if grad.isnan().any() or grad.isinf().any():
                return True

    return False


class MixedPrecisionManager:
    """
    Comprehensive mixed precision training manager supporting multiple precision types.
    """

    def __init__(
        self,
        precision: Union[str, PrecisionType] = PrecisionType.FP32,
        loss_scale: Union[str, float] = "dynamic",
        init_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
        use_custom_scaler: bool = False,
        hysteresis: int = 2,
    ) -> None:
        """
        Initialize the mixed precision manager.

        Args:
            precision: Precision type to use
            loss_scale: Loss scaling strategy ('dynamic', 'static', or float value)
            init_scale: Initial loss scale for dynamic scaling
            growth_factor: Factor to grow loss scale
            backoff_factor: Factor to reduce loss scale on overflow
            growth_interval: Steps between scale increases
            enabled: Whether mixed precision is enabled
            use_custom_scaler: Whether to use custom gradient scaler
            hysteresis: Hysteresis for custom dynamic scaler
        """
        self.precision = (
            PrecisionType(precision) if isinstance(precision, str) else precision
        )
        self.enabled = enabled and self.precision != PrecisionType.FP32
        self.use_custom_scaler = use_custom_scaler

        # Initialize appropriate scaler based on precision type
        self.scaler: Optional[Union[GradScaler, AbstractGradScaler]] = None
        self.autocast_dtype = None

        if self.enabled:
            if self.precision == PrecisionType.FP16:
                self.autocast_dtype = torch.float16

                if use_custom_scaler:
                    # Use custom gradient scaler
                    if isinstance(loss_scale, str):
                        if loss_scale == "dynamic":
                            self.scaler = DynamicGradScaler(
                                initial_scale=init_scale,
                                min_scale=1.0,
                                growth_factor=growth_factor,
                                backoff_factor=backoff_factor,
                                growth_interval=growth_interval,
                                hysteresis=hysteresis,
                                device="cuda" if torch.cuda.is_available() else "cpu",
                            )
                        elif loss_scale == "static":
                            self.scaler = ConstantGradScaler(
                                initial_scale=init_scale,
                                device="cuda" if torch.cuda.is_available() else "cpu",
                            )
                        else:
                            self.scaler = None
                    elif isinstance(loss_scale, (int, float)):
                        self.scaler = ConstantGradScaler(
                            initial_scale=float(loss_scale),
                            device="cuda" if torch.cuda.is_available() else "cpu",
                        )
                    else:
                        # Default to dynamic
                        self.scaler = DynamicGradScaler(
                            initial_scale=init_scale,
                            growth_factor=growth_factor,
                            backoff_factor=backoff_factor,
                            growth_interval=growth_interval,
                            hysteresis=hysteresis,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                        )
                    logger.info("Initialized FP16 with custom gradient scaler")
                else:
                    # Use PyTorch native scaler
                    # Handle different loss scaling strategies
                    if isinstance(loss_scale, str):
                        if loss_scale == "dynamic":
                            scaler_enabled = True
                        elif loss_scale == "static":
                            scaler_enabled = True
                            init_scale = 2**16  # Use fixed scale for static
                        else:
                            scaler_enabled = False
                    elif isinstance(loss_scale, (int, float)):
                        scaler_enabled = True
                        init_scale = float(loss_scale)
                    else:
                        scaler_enabled = True  # Default to dynamic
                    self.scaler = GradScaler(
                        "cuda",
                        init_scale=init_scale,
                        growth_factor=growth_factor,
                        backoff_factor=backoff_factor,
                        growth_interval=growth_interval,
                        enabled=scaler_enabled,
                    )
                    logger.info("Initialized FP16 mixed precision training")

            elif self.precision == PrecisionType.BF16:
                self.autocast_dtype = torch.bfloat16
                # BF16 doesn't need loss scaling
                self.scaler = GradScaler("cuda", enabled=False)
                logger.info("Initialized BF16 mixed precision training")

            elif self.precision == PrecisionType.FP8:
                # FP8 support is experimental
                if not hasattr(torch, "float8_e4m3fn"):
                    warnings.warn(
                        "FP8 not supported in current PyTorch version, "
                        "falling back to FP16"
                    )
                    self.precision = PrecisionType.FP16
                    self.autocast_dtype = torch.float16
                    self.scaler = GradScaler("cuda", enabled=True)
                else:
                    self.autocast_dtype = torch.float8_e4m3fn
                    self.scaler = GradScaler("cuda", enabled=True)
                    logger.info("Initialized experimental FP8 mixed precision training")

            elif self.precision == PrecisionType.MIXED:
                # Auto-detect best precision based on hardware
                if torch.cuda.is_bf16_supported():
                    self.autocast_dtype = torch.bfloat16
                    self.scaler = GradScaler("cuda", enabled=False)
                    logger.info("Auto-selected BF16 for mixed precision")
                else:
                    self.autocast_dtype = torch.float16
                    self.scaler = GradScaler("cuda", enabled=True)
                    logger.info("Auto-selected FP16 for mixed precision")

    @contextmanager
    def autocast_context(self, device_type: str = "cuda"):
        """
        Context manager for automatic mixed precision.

        Args:
            device_type: Device type for autocast ('cuda' or 'cpu')
        """
        if self.enabled and self.autocast_dtype is not None:
            with autocast(device_type=device_type, dtype=self.autocast_dtype):
                yield
        else:
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale the loss for mixed precision training.

        Args:
            loss: Loss tensor to scale

        Returns:
            Scaled loss tensor
        """
        if self.scaler:
            if isinstance(self.scaler, AbstractGradScaler):
                return self.scaler.scale_loss(loss)
            elif hasattr(self.scaler, "is_enabled") and self.scaler.is_enabled():
                return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Perform optimizer step with mixed precision handling.

        Args:
            optimizer: Optimizer to step
        """
        if self.scaler:
            if isinstance(self.scaler, AbstractGradScaler):
                # Custom scaler doesn't have a step method, user manages optimizer step
                optimizer.step()
            else:
                # PyTorch native scaler
                self.scaler.step(optimizer)
                self.scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Unscale gradients before clipping.

        Args:
            optimizer: Optimizer with gradients to unscale
        """
        if self.scaler:
            if isinstance(self.scaler, AbstractGradScaler):
                # Get parameters from optimizer
                params = []
                for group in optimizer.param_groups:
                    params.extend(group["params"])
                self.scaler.unscale_gradients(params)
            elif hasattr(self.scaler, "is_enabled") and self.scaler.is_enabled():
                self.scaler.unscale_(optimizer)

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get state dict for checkpointing.

        Returns:
            State dictionary
        """
        state: Dict[str, Any] = {
            "precision": self.precision.value,
            "enabled": self.enabled,
        }
        if self.scaler:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from checkpoint.

        Args:
            state_dict: State dictionary to load
        """
        if "scaler" in state_dict and self.scaler:
            self.scaler.load_state_dict(state_dict["scaler"])

    def convert_model(self, model: nn.Module) -> nn.Module:
        """
        Convert model to appropriate precision.

        Args:
            model: Model to convert

        Returns:
            Converted model
        """
        if not self.enabled:
            return model

        if self.precision in [PrecisionType.FP16, PrecisionType.MIXED]:
            model = convert_model_to_fp16(model)
        elif self.precision == PrecisionType.BF16:
            model = convert_model_to_bf16(model)
        elif self.precision == PrecisionType.FP8:
            logger.warning("FP8 model conversion not yet implemented")

        return model


def convert_model_to_fp16(model: nn.Module, keep_norm_fp32: bool = True) -> nn.Module:
    """
    Convert model parameters to FP16.

    Args:
        model: PyTorch model to convert
        keep_norm_fp32: Keep normalization layers in FP32

    Returns:
        Converted model
    """
    for module in model.modules():
        # Skip normalization layers if requested
        if keep_norm_fp32 and (
            isinstance(
                module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            )
        ):
            continue

        # Convert parameters to FP16
        for _, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param.data = param.data.half()

    return model


def convert_model_to_bf16(model: nn.Module, keep_norm_fp32: bool = True) -> nn.Module:
    """
    Convert model parameters to BF16.

    Args:
        model: PyTorch model to convert
        keep_norm_fp32: Keep normalization layers in FP32

    Returns:
        Converted model
    """
    if not torch.cuda.is_bf16_supported():
        warnings.warn("BF16 not supported on current hardware")
        return model

    for module in model.modules():
        # Skip normalization layers if requested
        if keep_norm_fp32 and (
            isinstance(
                module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            )
        ):
            continue

        # Convert parameters to BF16
        for _, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param.data = param.data.to(torch.bfloat16)

    return model
