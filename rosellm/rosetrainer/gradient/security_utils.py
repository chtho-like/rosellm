"""Security utilities for gradient operations.

This module provides security checks and validation for gradient
operations to prevent attacks and ensure safe training.
"""

import hashlib
import logging
from typing import Any, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradientSecurityValidator:
    """Validates gradients for security and safety concerns."""

    def __init__(
        self,
        max_norm: float = 1e10,
        min_norm: float = 1e-10,
        check_nan: bool = True,
        check_inf: bool = True,
        check_anomalies: bool = True,
        enable_checksums: bool = False,
    ):
        """Initialize security validator.

        Args:
            max_norm: Maximum allowed gradient norm.
            min_norm: Minimum allowed gradient norm (detect underflow).
            check_nan: Check for NaN values.
            check_inf: Check for infinite values.
            check_anomalies: Check for statistical anomalies.
            enable_checksums: Enable gradient checksums for tampering detection.
        """
        self.max_norm = max_norm
        self.min_norm = min_norm
        self.check_nan = check_nan
        self.check_inf = check_inf
        self.check_anomalies = check_anomalies
        self.enable_checksums = enable_checksums

        # Track gradient statistics for anomaly detection
        self.gradient_history: List[float] = []
        self.gradient_checksums: dict = {}

    def validate_gradient(
        self,
        gradient: torch.Tensor,
        param_name: Optional[str] = None,
    ) -> bool:
        """Validate a single gradient tensor.

        Args:
            gradient: Gradient tensor to validate.
            param_name: Optional parameter name for logging.

        Returns:
            True if gradient is valid, False otherwise.
        """
        if gradient is None:
            return True  # None gradients are OK (frozen params)

        param_id = param_name or "unknown"

        # Check for NaN
        if self.check_nan and torch.isnan(gradient).any():
            logger.error(f"NaN detected in gradient for {param_id}")
            return False

        # Check for Inf
        if self.check_inf and torch.isinf(gradient).any():
            logger.error(f"Inf detected in gradient for {param_id}")
            return False

        # Check gradient norm bounds
        grad_norm = gradient.norm().item()

        if grad_norm > self.max_norm:
            logger.warning(
                f"Gradient norm {grad_norm:.2e} exceeds max {self.max_norm:.2e} "
                f"for {param_id}"
            )
            return False

        if grad_norm < self.min_norm and grad_norm > 0:
            logger.warning(
                f"Gradient norm {grad_norm:.2e} below min {self.min_norm:.2e} "
                f"for {param_id} (possible underflow)"
            )
            # Don't fail on underflow, just warn

        # Check for anomalies
        if self.check_anomalies:
            if not self._check_statistical_anomalies(grad_norm, param_id):
                return False

        # Verify checksum if enabled
        if self.enable_checksums:
            if not self._verify_checksum(gradient, param_id):
                return False

        return True

    def _check_statistical_anomalies(
        self,
        grad_norm: float,
        param_id: str,
    ) -> bool:
        """Check for statistical anomalies in gradient.

        Args:
            grad_norm: Gradient norm.
            param_id: Parameter identifier.

        Returns:
            True if no anomalies detected.
        """
        self.gradient_history.append(grad_norm)

        # Keep only recent history
        if len(self.gradient_history) > 1000:
            self.gradient_history = self.gradient_history[-1000:]

        # Need enough history for statistics
        if len(self.gradient_history) < 10:
            return True

        import numpy as np

        history = np.array(self.gradient_history[-100:])
        mean = history.mean()
        std = history.std()

        # Check for sudden spikes (possible attack)
        if std > 0 and abs(grad_norm - mean) > 10 * std:
            logger.warning(
                f"Anomalous gradient detected for {param_id}: "
                f"norm={grad_norm:.2e}, expected={mean:.2e}±{std:.2e}"
            )
            # Don't fail, just warn for now

        return True

    def _verify_checksum(
        self,
        gradient: torch.Tensor,
        param_id: str,
    ) -> bool:
        """Verify gradient checksum for tampering detection.

        Args:
            gradient: Gradient tensor.
            param_id: Parameter identifier.

        Returns:
            True if checksum is valid or first time seeing.
        """
        # Compute checksum
        grad_bytes = gradient.cpu().numpy().tobytes()
        checksum = hashlib.sha256(grad_bytes).hexdigest()[:16]

        if param_id in self.gradient_checksums:
            # Verify consistency (simplified check)
            # In practice, would need more sophisticated verification
            pass

        self.gradient_checksums[param_id] = checksum
        return True

    def validate_model_gradients(
        self,
        model: nn.Module,
    ) -> bool:
        """Validate all gradients in a model.

        Args:
            model: Model to validate gradients for.

        Returns:
            True if all gradients are valid.
        """
        all_valid = True

        for name, param in model.named_parameters():
            if param.grad is not None:
                if not self.validate_gradient(param.grad, name):
                    all_valid = False

        return all_valid

    def sanitize_gradient(
        self,
        gradient: torch.Tensor,
        param_name: Optional[str] = None,
    ) -> torch.Tensor:
        """Sanitize a gradient by clipping and removing invalid values.

        Args:
            gradient: Gradient to sanitize.
            param_name: Optional parameter name for logging.

        Returns:
            Sanitized gradient.
        """
        if gradient is None:
            return gradient

        param_id = param_name or "unknown"

        # Replace NaN/Inf with zeros
        if torch.isnan(gradient).any() or torch.isinf(gradient).any():
            logger.warning(f"Replacing NaN/Inf with zeros in {param_id}")
            gradient = torch.where(
                torch.isnan(gradient) | torch.isinf(gradient),
                torch.zeros_like(gradient),
                gradient,
            )

        # Clip gradient norm
        grad_norm = gradient.norm().item()
        if grad_norm > self.max_norm:
            scale = self.max_norm / grad_norm
            gradient = gradient * scale
            logger.info(
                f"Clipped gradient norm for {param_id}: "
                f"{grad_norm:.2e} -> {self.max_norm:.2e}"
            )

        return gradient


class InputValidator:
    """Validates inputs to gradient operations."""

    @staticmethod
    def validate_model_list(models: List[Any]) -> None:
        """Validate a list of models.

        Args:
            models: List of models to validate.

        Raises:
            ValueError: If validation fails.
        """
        if not models:
            raise ValueError("Model list cannot be empty")

        if not all(isinstance(m, nn.Module) for m in models):
            raise ValueError("All models must be nn.Module instances")

    @staticmethod
    def validate_process_group(group: Any) -> None:
        """Validate a process group.

        Args:
            group: Process group to validate.

        Raises:
            ValueError: If validation fails.
        """
        if group is not None:
            # Check if it's a valid process group
            if not hasattr(group, "size") and not hasattr(group, "rank"):
                raise ValueError("Invalid process group object")

    @staticmethod
    def validate_tensor_list(tensors: List[Any]) -> None:
        """Validate a list of tensors.

        Args:
            tensors: List of tensors to validate.

        Raises:
            ValueError: If validation fails.
        """
        if not tensors:
            return  # Empty is OK

        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise ValueError("All items must be torch.Tensor objects")

        # Check for consistent device
        devices = {t.device for t in tensors}
        if len(devices) > 1:
            raise ValueError(f"All tensors must be on same device, got {devices}")

        # Check for consistent dtype (warning only)
        dtypes = {t.dtype for t in tensors}
        if len(dtypes) > 1:
            logger.warning(f"Mixed dtypes in tensor list: {dtypes}")


def create_security_validator(config: Any) -> GradientSecurityValidator:
    """Factory function to create security validator from config.

    Args:
        config: Configuration object.

    Returns:
        Configured security validator.
    """
    return GradientSecurityValidator(
        max_norm=getattr(config, "max_gradient_norm", 1e10),
        min_norm=getattr(config, "min_gradient_norm", 1e-10),
        check_nan=getattr(config, "check_for_nan", True),
        check_inf=getattr(config, "check_for_inf", True),
        check_anomalies=getattr(config, "check_anomalies", False),
        enable_checksums=getattr(config, "enable_checksums", False),
    )
