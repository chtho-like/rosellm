"""
Learning Rate and Weight Decay Scheduler for RoseTrainer

This module implements a comprehensive optimizer parameter scheduler compatible
with Megatron-LM's scheduling strategies. It supports multiple decay styles
and provides fine-grained control over learning rate warmup and decay.

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
"""

import logging
import math
from typing import Any, Dict, List, Literal, Optional, Tuple

from torch.optim import Optimizer

logger = logging.getLogger(__name__)

# Constants for loss scaling
DEFAULT_INIT_SCALE = 2**16
DEFAULT_MAX_SCALE = 2**24
DEFAULT_MIN_SCALE = 1.0
DEFAULT_SCALE_WINDOW = 2000
DEFAULT_SCALE_FACTOR = 2.0

# Mathematical constants
PI = math.pi
HALF = 0.5


class SchedulerError(Exception):
    """Base exception for scheduler-related errors."""

    pass


class InvalidSchedulerStateError(SchedulerError):
    """Raised when scheduler state is invalid or inconsistent."""

    pass


class OptimizerParamScheduler:
    """Optimizer parameter scheduler for learning rate and weight decay.

    This scheduler provides:
    - Multiple LR decay styles (linear, cosine, inverse-square-root, WSD, constant)
    - Linear warmup from init_lr to max_lr
    - Weight decay scheduling with linear/cosine increment
    - Per-parameter group LR/WD multipliers
    - Checkpoint save/restore support

    The scheduler is designed to be compatible with Megatron-LM's
    OptimizerParamScheduler for reproducible training.

    Args:
        optimizer: PyTorch optimizer to schedule
        init_lr: Initial learning rate for warmup start
        max_lr: Maximum learning rate (peak after warmup)
        min_lr: Minimum learning rate (final after decay)
        lr_warmup_steps: Number of warmup steps
        lr_decay_steps: Total number of decay steps
        lr_decay_style: Decay style for learning rate
        start_wd: Initial weight decay value
        end_wd: Final weight decay value
        wd_incr_steps: Number of weight decay increment steps
        wd_incr_style: Weight decay increment style
        wsd_decay_steps: Steps for WSD decay phase (only for WSD style)
        lr_wsd_decay_style: Decay style within WSD phase
    """

    def __init__(
        self,
        optimizer: Optimizer,
        init_lr: float = 0.0,
        max_lr: float = 1e-3,
        min_lr: float = 1e-5,
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 1000,
        lr_decay_style: Literal[
            "linear", "cosine", "inverse-square-root", "WSD", "constant"
        ] = "linear",
        start_wd: float = 0.0,
        end_wd: float = 0.0,
        wd_incr_steps: int = 0,
        wd_incr_style: Literal["linear", "cosine", "constant"] = "constant",
        wsd_decay_steps: Optional[int] = None,
        lr_wsd_decay_style: Optional[
            Literal["linear", "cosine", "exponential", "minus_sqrt"]
        ] = None,
    ) -> None:
        # Store optimizer
        self.optimizer = optimizer

        # Learning rate parameters
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr

        # Validate learning rate parameters
        if self.min_lr < 0.0:
            raise ValueError(f"min_lr must be non-negative, got {self.min_lr}")
        if self.max_lr < self.min_lr:
            raise ValueError(
                f"max_lr ({self.max_lr}) must be >= min_lr ({self.min_lr})"
            )
        if self.init_lr > self.max_lr:
            raise ValueError(
                f"init_lr ({self.init_lr}) must be <= max_lr ({self.max_lr})"
            )

        # Warmup and decay steps
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps

        # Validate step parameters
        if self.lr_decay_steps <= 0:
            raise ValueError(
                f"lr_decay_steps must be positive, got {self.lr_decay_steps}"
            )
        if self.lr_warmup_steps >= self.lr_decay_steps:
            raise ValueError(
                f"lr_warmup_steps ({self.lr_warmup_steps}) must be less than "
                f"lr_decay_steps ({self.lr_decay_steps})"
            )

        # Decay style
        self.lr_decay_style = lr_decay_style
        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style

        if self.lr_decay_style == "WSD":
            if self.wsd_decay_steps is None:
                raise ValueError(
                    "wsd_decay_steps is required when using WSD decay style"
                )
            if self.lr_wsd_decay_style is None:
                raise ValueError(
                    "lr_wsd_decay_style is required when using WSD decay style"
                )
            if self.wsd_decay_steps > self.lr_decay_steps:
                raise ValueError(
                    f"wsd_decay_steps ({self.wsd_decay_steps}) must be <= "
                    f"lr_decay_steps ({self.lr_decay_steps})"
                )

        # Weight decay parameters
        self.start_wd = start_wd
        self.end_wd = end_wd

        # Validate weight decay parameters
        if self.start_wd < 0.0:
            raise ValueError(f"start_wd must be non-negative, got {self.start_wd}")
        if self.end_wd < self.start_wd:
            raise ValueError(
                f"end_wd ({self.end_wd}) must be >= start_wd ({self.start_wd})"
            )

        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style

        # Step counter
        self.num_steps = 0

        # Cache for computed values to avoid redundant calculations
        self._lr_cache: Dict[Tuple[int, Optional[float], Optional[float]], float] = {}
        self._wd_cache: Dict[int, float] = {}
        self._cache_size_limit = 1000  # Prevent unbounded cache growth

        # Initialize learning rate and weight decay
        self.step(0)

    def get_wd(self) -> float:
        """Calculate current weight decay value.

        Returns:
            Current weight decay based on increment schedule
        """
        # Check cache first
        if self.num_steps in self._wd_cache:
            return self._wd_cache[self.num_steps]

        # Clear cache if it gets too large
        if len(self._wd_cache) > self._cache_size_limit:
            self._wd_cache.clear()

        if self.wd_incr_steps == 0 or self.num_steps > self.wd_incr_steps:
            result = self.end_wd
        elif self.wd_incr_style == "constant":
            if self.start_wd != self.end_wd:
                raise InvalidSchedulerStateError(
                    f"start_wd ({self.start_wd}) must equal end_wd ({self.end_wd}) "
                    f"for constant weight decay style"
                )
            result = self.end_wd
        else:
            # Calculate increment ratio
            incr_ratio = self.num_steps / self.wd_incr_steps
            if not 0.0 <= incr_ratio <= 1.0:
                raise InvalidSchedulerStateError(
                    f"Invalid increment ratio {incr_ratio}, must be in [0, 1]"
                )

            delta_wd = self.end_wd - self.start_wd

            if self.wd_incr_style == "linear":
                coeff = incr_ratio
            elif self.wd_incr_style == "cosine":
                coeff = HALF * (math.cos(PI * (1 - incr_ratio)) + 1.0)
            else:
                raise SchedulerError(f"Unsupported wd_incr_style: {self.wd_incr_style}")

            result = self.start_wd + coeff * delta_wd

        # Cache the result
        self._wd_cache[self.num_steps] = result
        return result

    def get_lr(self, param_group: Optional[Dict] = None) -> float:
        """Calculate current learning rate.

        Args:
            param_group: Optional parameter group dict for per-group LR

        Returns:
            Current learning rate based on warmup/decay schedule
        """
        # Use parameter group overrides if provided
        if param_group is not None:
            max_lr = param_group.get("max_lr", self.max_lr)
            min_lr = param_group.get("min_lr", self.min_lr)
        else:
            max_lr = self.max_lr
            min_lr = self.min_lr

        # Create cache key
        cache_key = (
            self.num_steps,
            max_lr if param_group else None,
            min_lr if param_group else None,
        )

        # Check cache first
        if cache_key in self._lr_cache:
            return self._lr_cache[cache_key]

        # Clear cache if it gets too large
        if len(self._lr_cache) > self._cache_size_limit:
            self._lr_cache.clear()

        # Linear warmup phase
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            warmup_ratio = self.num_steps / self.lr_warmup_steps
            result = self.init_lr + (max_lr - self.init_lr) * warmup_ratio
        # Constant learning rate
        elif self.lr_decay_style == "constant":
            result = max_lr
        # After decay steps, use minimum LR
        elif self.num_steps > self.lr_decay_steps:
            result = min_lr
        # Special handling for inverse-square-root
        elif self.lr_decay_style == "inverse-square-root":
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.num_steps, 1)
            lr = max_lr * math.sqrt(warmup_steps) / math.sqrt(num_steps)
            result = max(min_lr, lr)
        else:
            # Calculate decay ratio for other styles
            num_steps_ = self.num_steps - self.lr_warmup_steps
            decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
            decay_ratio = num_steps_ / decay_steps_

            if not 0.0 <= decay_ratio <= 1.0:
                raise InvalidSchedulerStateError(
                    f"Invalid decay ratio {decay_ratio}, must be in [0, 1]"
                )

            delta_lr = max_lr - min_lr

            # Calculate coefficient based on decay style
            if self.lr_decay_style == "linear":
                coeff = 1.0 - decay_ratio
            elif self.lr_decay_style == "cosine":
                coeff = HALF * (math.cos(PI * decay_ratio) + 1.0)
            elif self.lr_decay_style == "WSD":
                # Warmup-Stable-Decay: stable phase then decay
                if self.wsd_decay_steps is None:
                    raise InvalidSchedulerStateError(
                        "wsd_decay_steps is None but WSD decay style is selected"
                    )

                wsd_anneal_start = self.lr_decay_steps - self.wsd_decay_steps
                if self.num_steps <= wsd_anneal_start:
                    coeff = 1.0
                else:
                    wsd_steps = self.num_steps - wsd_anneal_start
                    wsd_decay_ratio = wsd_steps / self.wsd_decay_steps

                    if self.lr_wsd_decay_style == "linear":
                        coeff = 1.0 - wsd_decay_ratio
                    elif self.lr_wsd_decay_style == "cosine":
                        coeff = HALF * (math.cos(PI * wsd_decay_ratio) + 1.0)
                    elif self.lr_wsd_decay_style == "exponential":
                        coeff = (2.0 * math.pow(HALF, wsd_decay_ratio)) - 1.0
                    elif self.lr_wsd_decay_style == "minus_sqrt":
                        coeff = 1.0 - math.sqrt(wsd_decay_ratio)
                    else:
                        raise SchedulerError(
                            f"Unsupported lr_wsd_decay_style: {self.lr_wsd_decay_style}"
                        )
            else:
                raise SchedulerError(
                    f"Unsupported lr_decay_style: {self.lr_decay_style}"
                )

            result = min_lr + coeff * delta_lr

        # Cache the result
        self._lr_cache[cache_key] = result
        return float(result)

    def step(self, increment: int = 1) -> None:
        """Update learning rate and weight decay for all parameter groups.

        Args:
            increment: Number of steps to increment (default: 1)

        Raises:
            ValueError: If increment is negative
        """
        if increment < 0:
            raise ValueError(f"Step increment must be non-negative, got {increment}")

        # Clear caches when stepping to ensure fresh calculations
        if increment > 0:
            self._lr_cache.clear()
            self._wd_cache.clear()

        self.num_steps += increment

        try:
            new_wd = self.get_wd()

            for param_group in self.optimizer.param_groups:
                # Calculate new learning rate
                new_lr = self.get_lr(param_group)

                # Apply per-group multipliers if present
                param_group["lr"] = new_lr * param_group.get("lr_mult", 1.0)
                param_group["weight_decay"] = new_wd * param_group.get("wd_mult", 1.0)
        except Exception as e:
            logger.error(f"Error updating scheduler at step {self.num_steps}: {e}")
            raise SchedulerError(f"Failed to update scheduler: {e}") from e

    def state_dict(self) -> Dict:
        """Return scheduler state for checkpointing.

        Returns:
            Dictionary containing scheduler state
        """
        return {
            "num_steps": self.num_steps,
            "init_lr": self.init_lr,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_decay_style": self.lr_decay_style,
            "start_wd": self.start_wd,
            "end_wd": self.end_wd,
            "wd_incr_steps": self.wd_incr_steps,
            "wd_incr_style": self.wd_incr_style,
            "wsd_decay_steps": self.wsd_decay_steps,
            "lr_wsd_decay_style": self.lr_wsd_decay_style,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load scheduler state from checkpoint.

        Args:
            state_dict: Dictionary containing scheduler state

        Raises:
            InvalidSchedulerStateError: If critical config mismatch detected
        """
        if "num_steps" not in state_dict:
            raise InvalidSchedulerStateError(
                "State dict missing required 'num_steps' field"
            )

        self.num_steps = state_dict["num_steps"]

        # Clear caches when loading state
        self._lr_cache.clear()
        self._wd_cache.clear()

        # Validate that critical config matches
        critical_keys = [
            "lr_decay_style",
            "lr_decay_steps",
            "lr_warmup_steps",
        ]

        warnings = []
        errors = []

        for key in critical_keys:
            checkpoint_val = state_dict.get(key)
            current_val = getattr(self, key)
            if checkpoint_val is not None and checkpoint_val != current_val:
                errors.append(
                    f"{key}: checkpoint={checkpoint_val}, current={current_val}"
                )

        if errors:
            raise InvalidSchedulerStateError(
                f"Critical scheduler config mismatch: {'; '.join(errors)}"
            )

        # Warn about non-critical mismatches
        non_critical_keys = [
            "init_lr",
            "max_lr",
            "min_lr",
            "start_wd",
            "end_wd",
            "wd_incr_steps",
            "wd_incr_style",
        ]

        for key in non_critical_keys:
            checkpoint_val = state_dict.get(key)
            current_val = getattr(self, key)
            if checkpoint_val is not None and checkpoint_val != current_val:
                warnings.append(
                    f"{key}: checkpoint={checkpoint_val}, current={current_val}"
                )

        if warnings:
            logger.warning(
                f"Non-critical scheduler config differences: {'; '.join(warnings)}"
            )

        # Apply current step's LR and WD
        self.step(0)

    def get_last_lr(self) -> List[float]:
        """Get last computed learning rates for all parameter groups.

        Returns:
            List of learning rates for each parameter group
        """
        return [group["lr"] for group in self.optimizer.param_groups]


def create_scheduler(
    optimizer: Optimizer, config: Dict, scheduler_type: str = "default"
) -> OptimizerParamScheduler:
    """Factory function to create scheduler with common presets.

    Args:
        optimizer: PyTorch optimizer
        config: Configuration dict with scheduler parameters
        scheduler_type: Preset type ('default', 'warmup_linear', 'warmup_cosine',
                       'warmup_stable_decay', 'inverse_sqrt')

    Returns:
        Configured OptimizerParamScheduler instance

    Raises:
        ValueError: If scheduler_type is not recognized
    """
    base_config = {
        "optimizer": optimizer,
        "init_lr": config.get("init_lr", 0.0),
        "max_lr": config.get("max_lr", 1e-3),
        "min_lr": config.get("min_lr", 1e-5),
        "lr_warmup_steps": config.get("lr_warmup_steps", 0),
        "lr_decay_steps": config.get("lr_decay_steps", 1000),
        "start_wd": config.get("start_wd", 0.0),
        "end_wd": config.get("end_wd", 0.0),
        "wd_incr_steps": config.get("wd_incr_steps", 0),
        "wd_incr_style": config.get("wd_incr_style", "constant"),
    }

    presets: Dict[str, Dict[str, Any]] = {
        "default": {
            "lr_decay_style": "linear",
        },
        "warmup_linear": {
            "lr_decay_style": "linear",
            "init_lr": 0.0,
        },
        "warmup_cosine": {
            "lr_decay_style": "cosine",
            "init_lr": 0.0,
        },
        "warmup_stable_decay": {
            "lr_decay_style": "WSD",
            "wsd_decay_steps": config.get("wsd_decay_steps", 200),
            "lr_wsd_decay_style": config.get("lr_wsd_decay_style", "cosine"),
            "init_lr": 0.0,
        },
        "inverse_sqrt": {
            "lr_decay_style": "inverse-square-root",
            "init_lr": 0.0,
        },
    }

    if scheduler_type not in presets:
        raise ValueError(
            f"Unknown scheduler_type '{scheduler_type}'. "
            f"Available types: {list(presets.keys())}"
        )

    # Merge preset with base config
    preset_config = presets[scheduler_type]
    scheduler_config = {**base_config, **preset_config}

    # Override with any explicit config values
    for key in ["lr_decay_style", "wsd_decay_steps", "lr_wsd_decay_style"]:
        if key in config:
            scheduler_config[key] = config[key]

    return OptimizerParamScheduler(**scheduler_config)
