"""Microbatch calculator for pipeline parallelism.

This module provides utilities to calculate the number of microbatches and their sizes
for distributed training with pipeline parallelism. It follows Megatron-LM patterns
for compatibility and optimal performance.

Key Features:
- Dynamic microbatch size calculation
- Rampup/warmup support for gradual batch size increase
- Automatic adjustment based on memory constraints
- Thread-safe global state management with proper locking
- Memory-efficient history tracking with circular buffers
- Comprehensive input validation and error handling
"""

import logging
import math
import threading
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, List, Optional, Union

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Constants for better maintainability
DEFAULT_MEMORY_THRESHOLD = 0.9
MEMORY_SAFETY_MARGIN = 0.8
ADJUSTMENT_INTERVAL_MULTIPLIER = 10
MAX_HISTORY_SIZE = 1000
BYTES_PER_FP16_ELEMENT = 2
GB_TO_BYTES = 1024**3

# Thread-safe lock for global state
_GLOBAL_LOCK = threading.RLock()

# Global variables for microbatch calculator state
_GLOBAL_MICROBATCH_CALCULATOR: Optional["MicrobatchCalculatorBase"] = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR: Optional["MicrobatchCalculatorBase"] = None


class CalculatorType(Enum):
    """Enum for calculator types."""

    CONSTANT = "constant"
    RAMPUP = "rampup"
    ADAPTIVE = "adaptive"


class MicrobatchCalculatorError(Exception):
    """Base exception for microbatch calculator errors."""

    pass


class InvalidConfigurationError(MicrobatchCalculatorError):
    """Exception raised for invalid configuration."""

    pass


@dataclass
class AdjustmentRecord:
    """Record of a microbatch size adjustment."""

    consumed_samples: int
    micro_batch_size: int
    memory_usage: float
    timestamp: float


class MicrobatchCalculatorBase(ABC):
    """Abstract base class for microbatch calculators.

    This follows the Megatron-LM pattern for calculating microbatch sizes
    and counts during training with pipeline parallelism.
    """

    def __init__(
        self, global_batch_size: int, micro_batch_size: int, data_parallel_size: int
    ):
        """Initialize the microbatch calculator.

        Args:
            global_batch_size: Total batch size across all data parallel ranks
            micro_batch_size: Size of each microbatch
            data_parallel_size: Number of data parallel ranks

        Raises:
            InvalidConfigurationError: If configuration is invalid
        """
        # Comprehensive input validation
        self._validate_positive_integer(global_batch_size, "global_batch_size")
        self._validate_positive_integer(micro_batch_size, "micro_batch_size")
        self._validate_positive_integer(data_parallel_size, "data_parallel_size")

        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.data_parallel_size = data_parallel_size

        # Calculate derived values with validation
        if global_batch_size % data_parallel_size != 0:
            raise InvalidConfigurationError(
                f"global_batch_size ({global_batch_size}) must be divisible by "
                f"data_parallel_size ({data_parallel_size})"
            )

        self.global_batch_size_per_gpu = global_batch_size // data_parallel_size

        if self.global_batch_size_per_gpu % micro_batch_size != 0:
            raise InvalidConfigurationError(
                f"global_batch_size_per_gpu ({self.global_batch_size_per_gpu}) "
                f"must be divisible by micro_batch_size ({micro_batch_size})"
            )

        self.num_microbatches = self.global_batch_size_per_gpu // micro_batch_size
        if self.num_microbatches <= 0:
            raise InvalidConfigurationError(
                f"Calculated num_microbatches ({self.num_microbatches}) "
                f"must be positive"
            )

    @staticmethod
    def _validate_positive_integer(value: int, name: str) -> None:
        """Validate that a value is a positive integer.

        Args:
            value: Value to validate
            name: Parameter name for error messages

        Raises:
            InvalidConfigurationError: If validation fails
        """
        if not isinstance(value, int):
            raise InvalidConfigurationError(
                f"{name} must be an integer, got {type(value).__name__}"
            )
        if value <= 0:
            raise InvalidConfigurationError(f"{name} must be positive, got {value}")
        if value > 2**31 - 1:  # Prevent overflow
            raise InvalidConfigurationError(
                f"{name} too large (overflow risk), got {value}"
            )

    @abstractmethod
    def get_num_microbatches(self) -> int:
        """Get the number of microbatches for the current iteration."""
        pass

    @abstractmethod
    def get_micro_batch_size(self) -> int:
        """Get the size of each microbatch."""
        pass

    @abstractmethod
    def update(self, consumed_samples: int, consistency_check: bool = True) -> None:
        """Update the calculator state.

        Args:
            consumed_samples: Total number of samples consumed so far
            consistency_check: Whether to perform consistency checks across ranks
        """
        pass

    def get_current_global_batch_size(self) -> int:
        """Get the current effective global batch size."""
        return (
            self.get_num_microbatches()
            * self.micro_batch_size
            * self.data_parallel_size
        )

    def is_first_microbatch(self, microbatch_idx: int) -> bool:
        """Check if this is the first microbatch.

        Args:
            microbatch_idx: Index of the microbatch

        Returns:
            True if this is the first microbatch
        """
        return microbatch_idx == 0

    def is_last_microbatch(self, microbatch_idx: int) -> bool:
        """Check if this is the last microbatch.

        Args:
            microbatch_idx: Index of the microbatch

        Returns:
            True if this is the last microbatch
        """
        return microbatch_idx == self.get_num_microbatches() - 1


class ConstantNumMicrobatches(MicrobatchCalculatorBase):
    """Constant number of microbatches calculator.

    This calculator keeps a constant number of microbatches throughout training,
    which is the simplest and most common pattern.
    """

    def __init__(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        data_parallel_size: int,
        rampup_batch_size: Optional[List[int]] = None,
    ):
        """Initialize constant microbatch calculator.

        Args:
            global_batch_size: Total batch size across all data parallel ranks
            micro_batch_size: Size of each microbatch
            data_parallel_size: Number of data parallel ranks
            rampup_batch_size: Optional list of batch sizes for rampup (unused here)
        """
        super().__init__(global_batch_size, micro_batch_size, data_parallel_size)

        if rampup_batch_size is not None:
            warnings.warn("rampup_batch_size is ignored for ConstantNumMicrobatches")

    def get_num_microbatches(self) -> int:
        """Get the constant number of microbatches."""
        return self.num_microbatches

    def get_micro_batch_size(self) -> int:
        """Get the constant microbatch size."""
        return self.micro_batch_size

    def update(self, consumed_samples: int, consistency_check: bool = True) -> None:
        """No-op update for constant calculator."""
        pass


class RampupBatchSizeNumMicrobatches(MicrobatchCalculatorBase):
    """Rampup batch size calculator with warmup support.

    This calculator gradually increases the batch size during warmup phase,
    which can help with training stability and memory optimization.
    """

    def __init__(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        data_parallel_size: int,
        rampup_batch_size: List[int],
        start_global_batch_size: Optional[int] = None,
    ):
        """Initialize rampup microbatch calculator.

        Args:
            global_batch_size: Target batch size after rampup
            micro_batch_size: Size of each microbatch
            data_parallel_size: Number of data parallel ranks
            rampup_batch_size: List of batch sizes for rampup schedule
            start_global_batch_size: Initial batch size (defaults to first rampup value)
        """
        super().__init__(global_batch_size, micro_batch_size, data_parallel_size)

        if not rampup_batch_size or len(rampup_batch_size) == 0:
            raise InvalidConfigurationError(
                "rampup_batch_size must be provided and non-empty for rampup calculator"
            )

        self.rampup_batch_size = rampup_batch_size
        self.start_global_batch_size = start_global_batch_size or rampup_batch_size[0]

        # Validate rampup schedule
        self._validate_rampup_schedule(
            rampup_batch_size, data_parallel_size, micro_batch_size
        )

        # Current state
        self.current_global_batch_size = self.start_global_batch_size
        self.current_rampup_index = 0
        self.ramping_up = True

        self._update_current_microbatches()

    @staticmethod
    def _validate_rampup_schedule(
        rampup_batch_size: List[int], data_parallel_size: int, micro_batch_size: int
    ) -> None:
        """Validate rampup schedule.

        Args:
            rampup_batch_size: List of batch sizes for rampup
            data_parallel_size: Number of data parallel ranks
            micro_batch_size: Size of each microbatch

        Raises:
            InvalidConfigurationError: If schedule is invalid
        """
        for i, bs in enumerate(rampup_batch_size):
            if bs <= 0:
                raise InvalidConfigurationError(
                    f"rampup_batch_size[{i}] must be positive, got {bs}"
                )
            if bs % data_parallel_size != 0:
                raise InvalidConfigurationError(
                    f"rampup_batch_size[{i}] ({bs}) must be divisible by "
                    f"data_parallel_size ({data_parallel_size})"
                )
            bs_per_gpu = bs // data_parallel_size
            if bs_per_gpu % micro_batch_size != 0:
                raise InvalidConfigurationError(
                    f"rampup_batch_size[{i}] per GPU ({bs_per_gpu}) must be "
                    f"divisible by micro_batch_size ({micro_batch_size})"
                )

    def _update_current_microbatches(self) -> None:
        """Update current number of microbatches based on batch size."""
        bs_per_gpu = self.current_global_batch_size // self.data_parallel_size
        self.current_num_microbatches = bs_per_gpu // self.micro_batch_size

    def get_num_microbatches(self) -> int:
        """Get current number of microbatches."""
        return self.current_num_microbatches

    def get_micro_batch_size(self) -> int:
        """Get the constant microbatch size."""
        return self.micro_batch_size

    def update(self, consumed_samples: int, consistency_check: bool = True) -> None:
        """Update rampup state based on consumed samples.

        Args:
            consumed_samples: Total samples consumed so far
            consistency_check: Whether to verify consistency across ranks
        """
        if not self.ramping_up:
            return

        # Check if we should advance rampup
        if self.current_rampup_index < len(self.rampup_batch_size):
            next_batch_size = self.rampup_batch_size[self.current_rampup_index]

            # Simple rampup: advance after consuming current batch size samples
            if consumed_samples >= next_batch_size:
                self.current_global_batch_size = min(
                    next_batch_size, self.global_batch_size
                )
                self.current_rampup_index += 1
                self._update_current_microbatches()

                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        f"Ramping up batch size to {self.current_global_batch_size} "
                        f"at {consumed_samples} samples"
                    )

        # Check if rampup complete
        if self.current_global_batch_size >= self.global_batch_size:
            self.ramping_up = False
            self.current_global_batch_size = self.global_batch_size
            self._update_current_microbatches()
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Rampup complete at {consumed_samples} samples")

        # Consistency check across ranks
        if consistency_check and dist.is_initialized():
            tensor = torch.tensor(
                [self.current_global_batch_size, self.current_num_microbatches],
                dtype=torch.long,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            expected = torch.tensor(
                [self.current_global_batch_size, self.current_num_microbatches],
                dtype=torch.long,
                device=tensor.device,
            )
            if not torch.equal(tensor, expected * dist.get_world_size()):
                raise RuntimeError(
                    "Microbatch calculator state inconsistent across ranks. "
                    f"Expected {expected * dist.get_world_size()}, got {tensor}"
                )


class AdaptiveMicrobatchCalculator(MicrobatchCalculatorBase):
    """Adaptive microbatch calculator with memory-aware adjustment.

    This advanced calculator can dynamically adjust microbatch sizes
    based on available memory and training dynamics.
    """

    def __init__(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        data_parallel_size: int,
        min_micro_batch_size: int = 1,
        max_micro_batch_size: Optional[int] = None,
        memory_threshold: float = 0.9,
    ):
        """Initialize adaptive microbatch calculator.

        Args:
            global_batch_size: Total batch size
            micro_batch_size: Initial microbatch size
            data_parallel_size: Number of data parallel ranks
            min_micro_batch_size: Minimum allowed microbatch size
            max_micro_batch_size: Maximum allowed microbatch size
            memory_threshold: Memory usage threshold for adjustment (0-1)
        """
        super().__init__(global_batch_size, micro_batch_size, data_parallel_size)

        # Validate adaptive-specific parameters
        self._validate_positive_integer(min_micro_batch_size, "min_micro_batch_size")
        if max_micro_batch_size is not None:
            self._validate_positive_integer(
                max_micro_batch_size, "max_micro_batch_size"
            )

        self.min_micro_batch_size = min_micro_batch_size
        self.max_micro_batch_size = max_micro_batch_size or micro_batch_size * 2
        self.memory_threshold = memory_threshold

        if not 0 < memory_threshold <= 1:
            raise InvalidConfigurationError(
                f"memory_threshold must be in (0, 1], got {memory_threshold}"
            )
        if self.max_micro_batch_size < self.min_micro_batch_size:
            raise InvalidConfigurationError(
                f"max_micro_batch_size ({self.max_micro_batch_size}) must be >= "
                f"min_micro_batch_size ({self.min_micro_batch_size})"
            )

        self.current_micro_batch_size = micro_batch_size
        # Use deque for memory-efficient circular buffer
        self.adjustment_history: Deque[AdjustmentRecord] = deque(
            maxlen=MAX_HISTORY_SIZE
        )
        self._last_adjustment_samples = 0

    def get_num_microbatches(self) -> int:
        """Get current number of microbatches."""
        return self.global_batch_size_per_gpu // self.current_micro_batch_size

    def get_micro_batch_size(self) -> int:
        """Get current microbatch size."""
        return self.current_micro_batch_size

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage as fraction.

        Returns:
            Memory usage as fraction (0-1)
        """
        if not torch.cuda.is_available():
            return 0.0

        try:
            allocated = torch.cuda.memory_allocated() / GB_TO_BYTES  # noqa: F841
            reserved = torch.cuda.memory_reserved() / GB_TO_BYTES
            total = torch.cuda.get_device_properties(0).total_memory / GB_TO_BYTES

            if total <= 0:
                return 0.0

            return float(min(1.0, reserved / total))  # Cap at 1.0
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def _adjust_microbatch_size(self, memory_usage: float) -> None:
        """Adjust microbatch size based on memory usage."""
        if memory_usage > self.memory_threshold:
            # Reduce microbatch size if memory pressure
            new_size = max(
                self.min_micro_batch_size, self.current_micro_batch_size // 2
            )
            if new_size != self.current_micro_batch_size:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        f"Reducing microbatch size from "
                        f"{self.current_micro_batch_size} to {new_size} "
                        f"due to memory pressure ({memory_usage:.2%})"
                    )
                self.current_micro_batch_size = new_size
        elif memory_usage < self.memory_threshold * 0.7:
            # Increase microbatch size if memory available
            new_size = min(self.max_micro_batch_size, self.current_micro_batch_size * 2)
            # Ensure it divides evenly
            while (
                self.global_batch_size_per_gpu % new_size != 0
                and new_size > self.current_micro_batch_size
            ):
                new_size -= 1

            if new_size > self.current_micro_batch_size:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        f"Increasing microbatch size from "
                        f"{self.current_micro_batch_size} to {new_size} "
                        f"(memory usage: {memory_usage:.2%})"
                    )
                self.current_micro_batch_size = new_size

    def update(self, consumed_samples: int, consistency_check: bool = True) -> None:
        """Update calculator with memory-aware adjustment.

        Args:
            consumed_samples: Total samples consumed
            consistency_check: Whether to check consistency across ranks
        """
        import time

        memory_usage = self._get_memory_usage()
        # Store adjustment record in circular buffer
        self.adjustment_history.append(
            AdjustmentRecord(
                consumed_samples=consumed_samples,
                micro_batch_size=self.current_micro_batch_size,
                memory_usage=memory_usage,
                timestamp=time.time(),
            )
        )

        # Adjust every N samples to avoid thrashing
        adjustment_interval = self.global_batch_size * ADJUSTMENT_INTERVAL_MULTIPLIER
        if consumed_samples - self._last_adjustment_samples >= adjustment_interval:
            self._adjust_microbatch_size(memory_usage)
            self._last_adjustment_samples = consumed_samples

        # Ensure consistency across ranks
        if consistency_check and dist.is_initialized():
            tensor = torch.tensor(
                [self.current_micro_batch_size],
                dtype=torch.long,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)  # Use minimum across ranks
            self.current_micro_batch_size = int(tensor.item())


# Global state management functions


def initialize_microbatch_calculator(
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    rampup_batch_size: Optional[List[int]] = None,
    calculator_type: Union[str, CalculatorType] = CalculatorType.CONSTANT,
    **kwargs: Any,
) -> MicrobatchCalculatorBase:
    """Initialize the global microbatch calculator.

    Args:
        global_batch_size: Total batch size across all ranks
        micro_batch_size: Size of each microbatch
        data_parallel_size: Number of data parallel ranks
        rampup_batch_size: Optional rampup schedule
        calculator_type: Type of calculator ('constant', 'rampup', 'adaptive')
        **kwargs: Additional arguments passed to specific calculator types

    Returns:
        The initialized calculator instance

    Raises:
        InvalidConfigurationError: If configuration is invalid
        ValueError: If calculator type is unknown
    """
    global _GLOBAL_MICROBATCH_CALCULATOR

    # Convert string to enum if needed
    if isinstance(calculator_type, str):
        try:
            calculator_type = CalculatorType(calculator_type.lower())
        except ValueError:
            raise ValueError(
                f"Unknown calculator type: {calculator_type}. "
                f"Valid types: {[t.value for t in CalculatorType]}"
            )

    with _GLOBAL_LOCK:
        if _GLOBAL_MICROBATCH_CALCULATOR is not None:
            logger.warning(
                "Microbatch calculator already initialized. "
                "Destroying previous instance."
            )
            destroy_microbatch_calculator()

        try:
            calculator: MicrobatchCalculatorBase
            if calculator_type == CalculatorType.CONSTANT:
                calculator = ConstantNumMicrobatches(
                    global_batch_size,
                    micro_batch_size,
                    data_parallel_size,
                    rampup_batch_size,
                )
            elif calculator_type == CalculatorType.RAMPUP:
                if rampup_batch_size is None:
                    raise InvalidConfigurationError(
                        "rampup_batch_size required for rampup calculator"
                    )
                calculator = RampupBatchSizeNumMicrobatches(
                    global_batch_size,
                    micro_batch_size,
                    data_parallel_size,
                    rampup_batch_size,
                    **kwargs,
                )
            elif calculator_type == CalculatorType.ADAPTIVE:
                calculator = AdaptiveMicrobatchCalculator(
                    global_batch_size, micro_batch_size, data_parallel_size, **kwargs
                )
            else:
                raise ValueError(f"Unhandled calculator type: {calculator_type}")

            _GLOBAL_MICROBATCH_CALCULATOR = calculator

            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    f"Initialized {calculator_type.value} microbatch calculator: "
                    f"global_batch_size={global_batch_size}, "
                    f"micro_batch_size={micro_batch_size}, "
                    f"num_microbatches={calculator.get_num_microbatches()}"
                )

            return calculator

        except Exception:
            _GLOBAL_MICROBATCH_CALCULATOR = None
            raise


def get_microbatch_calculator() -> Optional[MicrobatchCalculatorBase]:
    """Get the global microbatch calculator.

    Returns:
        The global calculator instance or None if not initialized
    """
    with _GLOBAL_LOCK:
        return _GLOBAL_MICROBATCH_CALCULATOR


def get_num_microbatches() -> int:
    """Get number of microbatches from global calculator.

    Returns:
        Number of microbatches

    Raises:
        RuntimeError: If calculator not initialized
    """
    with _GLOBAL_LOCK:
        if _GLOBAL_MICROBATCH_CALCULATOR is None:
            raise RuntimeError(
                "Microbatch calculator not initialized. Call "
                "initialize_microbatch_calculator() first."
            )
        return _GLOBAL_MICROBATCH_CALCULATOR.get_num_microbatches()


def get_micro_batch_size() -> int:
    """Get microbatch size from global calculator.

    Returns:
        Size of each microbatch

    Raises:
        RuntimeError: If calculator not initialized
    """
    with _GLOBAL_LOCK:
        if _GLOBAL_MICROBATCH_CALCULATOR is None:
            raise RuntimeError(
                "Microbatch calculator not initialized. Call "
                "initialize_microbatch_calculator() first."
            )
        return _GLOBAL_MICROBATCH_CALCULATOR.get_micro_batch_size()


def update_microbatch_calculator(
    consumed_samples: int, consistency_check: bool = True
) -> None:
    """Update the global microbatch calculator state.

    Args:
        consumed_samples: Total samples consumed so far
        consistency_check: Whether to check consistency across ranks
    """
    with _GLOBAL_LOCK:
        if _GLOBAL_MICROBATCH_CALCULATOR is not None:
            _GLOBAL_MICROBATCH_CALCULATOR.update(consumed_samples, consistency_check)


def destroy_microbatch_calculator() -> None:
    """Destroy the global microbatch calculator.

    This should be called when done with the calculator to free resources.
    """
    global _GLOBAL_MICROBATCH_CALCULATOR
    with _GLOBAL_LOCK:
        _GLOBAL_MICROBATCH_CALCULATOR = None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Destroyed global microbatch calculator")


# Utility functions for automatic batch size calculation


def calculate_optimal_microbatch_size(
    model_size_gb: float,
    available_memory_gb: float,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    pipeline_parallel_size: int = 1,
    activation_checkpoint: bool = False,
    optimizer_type: str = "adam",
    precision: str = "fp16",
) -> int:
    """Calculate optimal microbatch size based on model and memory constraints.

    Args:
        model_size_gb: Model size in GB
        available_memory_gb: Available GPU memory in GB
        sequence_length: Sequence length
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        pipeline_parallel_size: Pipeline parallel size
        activation_checkpoint: Whether using activation checkpointing
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        precision: Training precision ('fp16', 'fp32', 'bf16')

    Returns:
        Suggested microbatch size

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate inputs
    if model_size_gb <= 0 or available_memory_gb <= 0:
        raise ValueError("Model size and available memory must be positive")
    if sequence_length <= 0 or hidden_size <= 0 or num_layers <= 0:
        raise ValueError("Model dimensions must be positive")
    if pipeline_parallel_size <= 0:
        raise ValueError("Pipeline parallel size must be positive")

    # Estimate memory per sample based on precision
    precision_bytes = {"fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8}
    bytes_per_element = precision_bytes.get(precision.lower(), 2)

    # Activation memory per layer per sample
    if activation_checkpoint:
        # Only need to store one layer at a time
        activation_memory = hidden_size * sequence_length * bytes_per_element * 2
    else:
        # Store all layer activations
        activation_memory = (
            hidden_size * sequence_length * bytes_per_element * num_layers * 2
        )

    activation_memory_gb = activation_memory / GB_TO_BYTES

    # Model memory (divided by pipeline parallel)
    model_memory_per_gpu = model_size_gb / pipeline_parallel_size

    # Optimizer states based on optimizer type
    optimizer_multipliers = {
        "adam": 2.0,  # momentum and variance
        "adamw": 2.0,  # momentum and variance
        "sgd": 1.0,  # only momentum
        "lamb": 2.0,  # similar to Adam
    }
    optimizer_multiplier = optimizer_multipliers.get(optimizer_type.lower(), 2.0)
    optimizer_memory = model_memory_per_gpu * optimizer_multiplier

    # Available for activations
    available_for_activations = (
        available_memory_gb - model_memory_per_gpu - optimizer_memory
    )
    available_for_activations *= MEMORY_SAFETY_MARGIN  # Safety margin

    # Calculate microbatch size
    if available_for_activations > 0:
        microbatch_size = int(available_for_activations / activation_memory_gb)
        # Round to power of 2 for efficiency
        microbatch_size = 2 ** int(math.log2(max(1, microbatch_size)))
        return max(1, microbatch_size)
    else:
        return 1


def get_microbatch_schedule(
    start_batch_size: int,
    target_batch_size: int,
    warmup_steps: int,
    schedule_type: str = "linear",
    ensure_divisible_by: Optional[int] = None,
) -> List[int]:
    """Generate a microbatch rampup schedule.

    Args:
        start_batch_size: Initial batch size
        target_batch_size: Target batch size after warmup
        warmup_steps: Number of warmup steps
        schedule_type: Schedule type ('linear', 'exponential', 'cosine', 'polynomial')
        ensure_divisible_by: Ensure all batch sizes are divisible by this value

    Returns:
        List of batch sizes for rampup

    Raises:
        ValueError: If parameters are invalid
    """
    # Input validation
    if start_batch_size <= 0 or target_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if start_batch_size > target_batch_size:
        raise ValueError("Start batch size cannot exceed target batch size")
    if warmup_steps < 0:
        raise ValueError("Warmup steps must be non-negative")

    if warmup_steps == 0:
        return [target_batch_size]

    schedule: List[int] = []

    if schedule_type == "linear":
        for i in range(warmup_steps):
            ratio = (i + 1) / warmup_steps
            batch_size = int(
                start_batch_size + (target_batch_size - start_batch_size) * ratio
            )
            schedule.append(batch_size)

    elif schedule_type == "exponential":
        for i in range(warmup_steps):
            ratio = (i + 1) / warmup_steps
            # Exponential growth
            batch_size = int(
                start_batch_size * (target_batch_size / start_batch_size) ** ratio
            )
            schedule.append(batch_size)

    elif schedule_type == "cosine":
        for i in range(warmup_steps):
            ratio = (i + 1) / warmup_steps
            # Cosine schedule
            cosine_val = (1 - math.cos(ratio * math.pi)) / 2
            batch_size = int(
                start_batch_size + (target_batch_size - start_batch_size) * cosine_val
            )
            schedule.append(batch_size)

    elif schedule_type == "polynomial":
        # Polynomial schedule with degree 2
        for i in range(warmup_steps):
            ratio = (i + 1) / warmup_steps
            # Quadratic growth
            poly_val = ratio**2
            batch_size = int(
                start_batch_size + (target_batch_size - start_batch_size) * poly_val
            )
            schedule.append(batch_size)
    else:
        raise ValueError(
            f"Unknown schedule type: {schedule_type}. "
            f"Valid types: linear, exponential, cosine, polynomial"
        )

    # Ensure monotonic increase and unique values
    schedule = sorted(list(set(schedule)))

    # Ensure divisibility if requested
    if ensure_divisible_by and ensure_divisible_by > 0:
        schedule = [
            (bs // ensure_divisible_by) * ensure_divisible_by for bs in schedule
        ]
        # Remove duplicates after rounding
        schedule = sorted(list(set(bs for bs in schedule if bs > 0)))

    # Ensure we end at target (if divisible or no constraint)
    if schedule and schedule[-1] != target_batch_size:
        if not ensure_divisible_by or target_batch_size % ensure_divisible_by == 0:
            schedule.append(target_batch_size)

    return schedule
