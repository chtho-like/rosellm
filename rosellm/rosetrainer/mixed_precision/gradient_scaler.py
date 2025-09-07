"""
Custom Gradient Scaler Implementation for Mixed Precision Training

This module implements gradient scalers compatible with Megatron-LM's
design patterns, providing both constant and dynamic loss scaling strategies
for mixed precision training.

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Mixed Precision Training: https://arxiv.org/abs/1710.03740
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AbstractGradScaler(ABC):
    """
    Abstract base class for gradient scalers.

    This class defines the interface for gradient scalers used in mixed precision
    training, following Megatron-LM's design patterns for compatibility.
    """

    def __init__(self, initial_scale: float, device: Optional[str] = None):
        """
        Initialize the gradient scaler.

        Args:
            initial_scale: Initial loss scale value (must be positive)
            device: Device to place scale tensor on ('cuda' or 'cpu').
                   If None, automatically selects cuda if available, otherwise cpu.

        Raises:
            AssertionError: If initial_scale is not positive
        """
        assert (
            initial_scale > 0.0
        ), f"initial_scale must be positive, got {initial_scale}"

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._scale = torch.tensor([initial_scale], dtype=torch.float, device=device)
        self._device = device

    @property
    def scale(self) -> torch.Tensor:
        """Get the current loss scale as a tensor."""
        return self._scale

    @property
    def inv_scale(self) -> torch.Tensor:
        """
        Get the inverse of the current scale for gradient unscaling.

        Returns:
            Inverse scale as a float32 tensor
        """
        return self._scale.double().reciprocal().float()

    @abstractmethod
    def update(self, found_inf: bool) -> None:
        """
        Update the loss scale based on gradient overflow status.

        Args:
            found_inf: Whether infinite or NaN gradients were found
        """
        pass

    @abstractmethod
    def state_dict(self) -> Dict:
        """
        Get state dictionary for checkpointing.

        Returns:
            Dictionary containing scaler state
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Load state from a checkpoint.

        Args:
            state_dict: State dictionary from checkpoint
        """
        pass

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale the loss tensor for mixed precision training.

        Args:
            loss: Loss tensor to scale

        Returns:
            Scaled loss tensor
        """
        return loss * self.scale

    def unscale_gradients(self, parameters: Union[nn.Module, list]) -> None:
        """
        Unscale gradients in-place.

        Args:
            parameters: Model or list of parameters with gradients to unscale
        """
        inv_scale = self.inv_scale

        if isinstance(parameters, nn.Module):
            param_list = [p for p in parameters.parameters() if p.grad is not None]
        else:
            param_list = [p for p in parameters if p.grad is not None]

        for param in param_list:
            if param.grad is not None:
                param.grad.mul_(inv_scale)


class ConstantGradScaler(AbstractGradScaler):
    """
    Constant loss scale gradient scaler (scale never changes).

    This scaler maintains a fixed loss scale throughout training, useful for
    stable training scenarios or when dynamic scaling is not desired.
    """

    def update(self, found_inf: bool) -> None:
        """
        No-op for constant scaler (scale never changes).

        Args:
            found_inf: Whether infinite or NaN gradients were found (ignored)
        """
        pass

    def state_dict(self) -> Dict:
        """
        Get state dictionary for checkpointing.

        Returns:
            Dictionary containing the constant scale value
        """
        return {"scale": self._scale.cpu()}

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Load state from checkpoint.

        Args:
            state_dict: State dictionary containing 'scale' key
        """
        self._scale = state_dict["scale"].to(self._device)


class DynamicGradScaler(AbstractGradScaler):
    """
    Dynamic loss scaling gradient scaler with hysteresis.

    This scaler automatically adjusts the loss scale based on gradient overflow
    detection, growing the scale during stable training and backing off when
    overflows occur, with hysteresis to prevent oscillation.
    """

    def __init__(
        self,
        initial_scale: float,
        min_scale: float = 1.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        hysteresis: int = 2,
        device: Optional[str] = None,
    ):
        """
        Initialize the dynamic gradient scaler.

        Args:
            initial_scale: Initial loss scale value
            min_scale: Minimum allowed scale value
            growth_factor: Factor to multiply scale by when growing
            backoff_factor: Factor to multiply scale by when backing off
            growth_interval: Number of steps without overflow before growth
            hysteresis: Number of consecutive overflows before backoff
            device: Device to place tensors on. If None, automatically selects
                   cuda if available, otherwise cpu.

        Raises:
            AssertionError: If any parameter is invalid
        """
        super().__init__(initial_scale, device)

        # Validation
        assert (
            min_scale > 0.0 and min_scale <= initial_scale
        ), f"min_scale must be positive and <= initial_scale, got {min_scale}"
        assert growth_factor > 1.0, f"growth_factor must be > 1.0, got {growth_factor}"
        assert (
            0.0 < backoff_factor < 1.0
        ), f"backoff_factor must be in (0, 1), got {backoff_factor}"
        assert (
            growth_interval > 0
        ), f"growth_interval must be positive, got {growth_interval}"
        assert hysteresis > 0, f"hysteresis must be positive, got {hysteresis}"

        # Scale bounds and factors - use self._device from parent class
        self.min_scale = torch.tensor(
            [min_scale], dtype=torch.float, device=self._device
        )
        self.growth_factor = torch.tensor(
            [growth_factor], dtype=torch.float, device=self._device
        )
        self.backoff_factor = torch.tensor(
            [backoff_factor], dtype=torch.float, device=self._device
        )

        # Hysteresis parameters
        self.growth_interval = growth_interval
        self.hysteresis = hysteresis

        # Trackers for dynamic scaling
        self._growth_tracker = 0
        self._hysteresis_tracker = hysteresis

    def update(self, found_inf: bool) -> None:
        """
        Update scale based on gradient overflow status.

        Implements hysteresis to prevent scale oscillation:
        - Requires multiple consecutive overflows before backing off
        - Grows scale after many successful iterations

        Args:
            found_inf: Whether infinite or NaN gradients were found
        """
        if found_inf:
            # Reset growth tracker on overflow
            self._growth_tracker = 0
            self._hysteresis_tracker -= 1

            if self._hysteresis_tracker <= 0:
                # Back off the scale after consecutive overflows
                self._scale = torch.max(
                    self._scale * self.backoff_factor, self.min_scale
                )
                self._hysteresis_tracker = self.hysteresis

                logger.info(
                    f"Gradient overflow detected, backing off scale to "
                    f"{self._scale.item():.1f}"
                )
        else:
            # Increment growth tracker for successful iteration
            self._growth_tracker += 1

            if self._growth_tracker >= self.growth_interval:
                # Grow the scale after many successful iterations
                self._scale = self._scale * self.growth_factor
                self._growth_tracker = 0
                self._hysteresis_tracker = self.hysteresis

                logger.debug(
                    f"No overflow for {self.growth_interval} steps, "
                    f"growing scale to {self._scale.item():.1f}"
                )

    def state_dict(self) -> Dict:
        """
        Get state dictionary for checkpointing.

        Returns:
            Dictionary containing scale and tracker states
        """
        return {
            "scale": self._scale.cpu(),
            "growth_tracker": self._growth_tracker,
            "hysteresis_tracker": self._hysteresis_tracker,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Load state from checkpoint.

        Args:
            state_dict: State dictionary with scale and tracker values
        """
        self._scale = state_dict["scale"].to(self._device)
        self._growth_tracker = state_dict["growth_tracker"]
        self._hysteresis_tracker = state_dict["hysteresis_tracker"]


@dataclass
class GradScalerConfig:
    """
    Configuration for gradient scaler.

    This dataclass encapsulates all configuration options for gradient scalers
    and provides a factory method to create the appropriate scaler instance.
    """

    scaler_type: str = "dynamic"  # "constant", "dynamic", "none"
    initial_scale: float = 2**16
    min_scale: float = 1.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    hysteresis: int = 2

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_types = ["constant", "dynamic", "none"]
        if self.scaler_type not in valid_types:
            raise ValueError(
                f"Invalid scaler_type: {self.scaler_type}. "
                f"Must be one of {valid_types}"
            )

        if self.initial_scale <= 0:
            raise ValueError(
                f"initial_scale must be positive, got {self.initial_scale}"
            )

        if self.min_scale <= 0:
            raise ValueError(f"min_scale must be positive, got {self.min_scale}")

        if self.growth_factor <= 1.0:
            raise ValueError(f"growth_factor must be > 1.0, got {self.growth_factor}")

        if not 0.0 < self.backoff_factor < 1.0:
            raise ValueError(
                f"backoff_factor must be in (0, 1), got {self.backoff_factor}"
            )

        if self.growth_interval <= 0:
            raise ValueError(
                f"growth_interval must be positive, got {self.growth_interval}"
            )

        if self.hysteresis <= 0:
            raise ValueError(f"hysteresis must be positive, got {self.hysteresis}")

    def create_scaler(
        self, device: Optional[str] = None
    ) -> Optional[AbstractGradScaler]:
        """
        Factory method to create the appropriate gradient scaler.

        Args:
            device: Device to place scaler tensors on. If None, automatically
                   selects cuda if available, otherwise cpu.

        Returns:
            Gradient scaler instance or None if scaler_type is "none"

        Raises:
            ValueError: If scaler_type is unknown
        """
        if self.scaler_type == "constant":
            return ConstantGradScaler(self.initial_scale, device)
        elif self.scaler_type == "dynamic":
            return DynamicGradScaler(
                self.initial_scale,
                self.min_scale,
                self.growth_factor,
                self.backoff_factor,
                self.growth_interval,
                self.hysteresis,
                device,
            )
        elif self.scaler_type == "none":
            return None
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")


def check_for_inf_and_nan(
    parameters: Union[list, nn.Module], scaler: Optional[AbstractGradScaler] = None
) -> bool:
    """
    Check for infinite or NaN values in gradients and update scaler.

    This function checks all gradients for non-finite values and optionally
    updates a gradient scaler based on the result.

    Args:
        parameters: Model or list of parameters to check
        scaler: Optional gradient scaler to update

    Returns:
        True if any infinite or NaN gradients were found, False otherwise
    """
    found_inf = False

    if isinstance(parameters, nn.Module):
        param_list = list(parameters.parameters())
    else:
        param_list = parameters

    for param in param_list:
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                found_inf = True
                break

    # Update scaler if provided
    if scaler is not None:
        scaler.update(found_inf)

    return found_inf
