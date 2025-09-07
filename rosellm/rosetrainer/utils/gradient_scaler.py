"""
Custom Gradient Scaler Implementation

This module provides an enhanced gradient scaler with monitoring capabilities
for mixed precision training.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.optim

logger = logging.getLogger(__name__)


class CustomGradientScaler:
    """
    Custom gradient scaler with enhanced monitoring and integration.

    This scaler provides compatibility with both standard PyTorch and
    custom gradient utilities, with additional monitoring capabilities.
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        """Initialize the custom gradient scaler.

        Args:
            init_scale: Initial scale factor
            growth_factor: Factor to multiply scale on successful iterations
            backoff_factor: Factor to multiply scale on overflow
            growth_interval: Number of iterations between scale increases
            enabled: Whether scaling is enabled
        """
        self._scale = torch.tensor(init_scale, dtype=torch.float32)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        self._growth_tracker = 0
        self._found_inf_history: List[bool] = []

    def scale(self, outputs: torch.Tensor) -> torch.Tensor:
        """Scale the outputs (loss) by the scale factor.

        Args:
            outputs: Tensor to scale (typically loss)

        Returns:
            Scaled tensor
        """
        if not self._enabled:
            return outputs
        return outputs * self._scale

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale the gradients in the optimizer's parameters.

        Args:
            optimizer: Optimizer containing parameters to unscale
        """
        if not self._enabled:
            return

        inv_scale = 1.0 / self._scale
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.mul_(inv_scale)

    def update(
        self,
        found_inf: Optional[bool] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Update the scale factor based on gradient overflow.

        Args:
            found_inf: Whether infinite/NaN gradients were found
            optimizer: Optional optimizer to check for overflow
        """
        if not self._enabled:
            return

        # Auto-detect overflow if not provided
        if found_inf is None and optimizer is not None:
            found_inf = False
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        if (
                            torch.isnan(param.grad).any()
                            or torch.isinf(param.grad).any()
                        ):
                            found_inf = True
                            break
                if found_inf:
                    break

        if found_inf is None:
            return

        self._found_inf_history.append(found_inf)

        if found_inf:
            # Backoff on overflow
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
            logger.debug(
                f"Gradient overflow detected, scale reduced to {self._scale.item()}"
            )
        else:
            # Increase scale if we've had enough successful iterations
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
                logger.debug(f"Scale increased to {self._scale.item()}")

    def step(self, optimizer: torch.optim.Optimizer, *args, **kwargs) -> Optional[Any]:
        """
        Unscale gradients and optionally step the optimizer.

        Args:
            optimizer: Optimizer to step
            *args: Additional arguments for optimizer.step()
            **kwargs: Additional keyword arguments for optimizer.step()

        Returns:
            The return value of optimizer.step() if gradients are finite,
            None otherwise
        """
        self.unscale_(optimizer)

        # Check for inf/nan
        found_inf = False
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        found_inf = True
                        break
            if found_inf:
                break

        if not found_inf:
            retval = optimizer.step(*args, **kwargs)
        else:
            retval = None

        self.update(found_inf)
        return retval

    def get_scale(self) -> float:
        """Get the current scale factor."""
        return float(self._scale.item()) if self._enabled else 1.0

    def get_growth_tracker(self) -> int:
        """Get the current growth tracker value."""
        return self._growth_tracker

    def get_overflow_history(self) -> List[bool]:
        """Get the history of overflow detections."""
        return self._found_inf_history.copy()

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            "scale": self._scale,
            "growth_tracker": self._growth_tracker,
            "found_inf_history": self._found_inf_history,
            "enabled": self._enabled,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self._scale = state_dict.get("scale", self._scale)
        self._growth_tracker = state_dict.get("growth_tracker", 0)
        self._found_inf_history = state_dict.get("found_inf_history", [])
        self._enabled = state_dict.get("enabled", self._enabled)

    def __repr__(self) -> str:
        return (
            f"CustomGradientScaler("
            f"scale={self.get_scale():.1f}, "
            f"growth_tracker={self._growth_tracker}, "
            f"enabled={self._enabled})"
        )
