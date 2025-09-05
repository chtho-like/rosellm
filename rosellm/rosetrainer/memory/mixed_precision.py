from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn

# References:
# [1] Micikevicius, P. et al. "Mixed Precision Training." 
#     arXiv:1710.03740 (2017)
# [2] NVIDIA Apex implementation:
#     https://github.com/NVIDIA/apex
# [3] PyTorch AMP implementation:
#     https://pytorch.org/docs/stable/amp.html
# [4] Shoeybi, M. et al. "Megatron-LM: Training Multi-Billion Parameter Language Models
#     Using Model Parallelism." arXiv:1909.08053 (2019)


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
    ):
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
            print(f"Gradient overflow, reducing loss scale to {self.cur_scale}")
        else:
            # Increment counter of successful steps
            self.consecutive_successful_steps += 1

            # If we've had scale_window successful steps, increase the loss scale
            if self.consecutive_successful_steps >= self.scale_window:
                self.cur_scale = min(self.cur_scale * self.scale_factor, self.max_scale)
                self.consecutive_successful_steps = 0
                print(
                    f"No overflow for {self.scale_window} steps, increasing loss scale to {self.cur_scale}"
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


def convert_model_to_fp16(model: nn.Module) -> None:
    """
    Convert model parameters to FP16 except for normalization layers.

    Args:
        model: PyTorch model to convert.
    """
    for module in model.modules():
        # Skip certain layer types (layer norm, batch norm, etc.)
        if (
            isinstance(module, nn.LayerNorm)
            or isinstance(module, nn.BatchNorm1d)
            or isinstance(module, nn.BatchNorm2d)
            or isinstance(module, nn.BatchNorm3d)
        ):
            continue

        # Convert parameters to FP16
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                param.data = param.data.half()
