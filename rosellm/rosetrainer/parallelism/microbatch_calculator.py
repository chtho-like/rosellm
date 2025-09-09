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
from typing import Any, Deque, List, Optional, Protocol, Union

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Constants for better maintainability
DEFAULT_MEMORY_THRESHOLD = 0.9
MEMORY_SAFETY_MARGIN = 0.8
ADJUSTMENT_INTERVAL_MULTIPLIER = 10
MAX_HISTORY_SIZE = 1000
BYTES_PER_FP16_ELEMENT = 2
BYTES_PER_FP32_ELEMENT = 4
BYTES_PER_BF16_ELEMENT = 2
BYTES_PER_FP64_ELEMENT = 8
GB_TO_BYTES = 1024**3
MB_TO_BYTES = 1024**2

# Memory calculation constants
ACTIVATION_MEMORY_MULTIPLIER = 2.0  # Forward + backward activations
GRADIENT_MEMORY_MULTIPLIER = 1.0  # Gradient storage
OPTIMIZER_STATE_MULTIPLIERS = {
    "adam": 2.0,  # momentum + variance
    "adamw": 2.0,  # momentum + variance
    "sgd": 1.0,  # only momentum
    "lamb": 2.0,  # similar to Adam
    "adagrad": 1.0,  # accumulated gradients
    "rmsprop": 1.0,  # squared gradients
}

# Pipeline efficiency thresholds
MIN_PIPELINE_EFFICIENCY = 0.8
RECOMMENDED_MICROBATCHES_PER_STAGE = 2

# Batch size limits for safety
MAX_SAFE_BATCH_SIZE = 2**20  # Prevent integer overflow
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE_GROWTH_FACTOR = 2.0

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


def create_adjustment_record(
    consumed_samples: int, micro_batch_size: int, memory_usage: float, timestamp: float
) -> "AdjustmentRecord":
    """Create a validated adjustment record.

    Args:
        consumed_samples: Total samples processed when adjustment occurred
        micro_batch_size: Microbatch size at time of adjustment
        memory_usage: GPU memory usage fraction [0.0, 1.0]
        timestamp: Unix timestamp when adjustment was recorded

    Returns:
        Validated AdjustmentRecord instance

    Raises:
        ValueError: If any parameter is invalid
    """
    if consumed_samples < 0:
        raise ValueError(
            f"consumed_samples must be non-negative, got {consumed_samples}"
        )
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
    if not (0.0 <= memory_usage <= 1.0):
        raise ValueError(f"memory_usage must be in [0, 1], got {memory_usage}")
    if timestamp <= 0:
        raise ValueError(f"timestamp must be positive, got {timestamp}")

    return AdjustmentRecord(consumed_samples, micro_batch_size, memory_usage, timestamp)


@dataclass(frozen=True)
class AdjustmentRecord:
    """Record of a microbatch size adjustment.

    This is an immutable record for tracking microbatch adjustments
    over time, used primarily by the adaptive calculator.

    Use `create_adjustment_record()` factory function for validated instances.

    Attributes:
        consumed_samples: Total samples processed when adjustment occurred
        micro_batch_size: Microbatch size at time of adjustment
        memory_usage: GPU memory usage fraction [0.0, 1.0] at time of adjustment
        timestamp: Unix timestamp when adjustment was recorded
    """

    consumed_samples: int
    micro_batch_size: int
    memory_usage: float
    timestamp: float


class MicrobatchCalculatorProtocol(Protocol):
    """Protocol defining the interface for microbatch calculators.

    This protocol ensures type safety and clear contracts for
    all microbatch calculator implementations.
    """

    def get_num_microbatches(self) -> int:
        """Get the number of microbatches for the current iteration."""
        ...

    def get_micro_batch_size(self) -> int:
        """Get the size of each microbatch."""
        ...

    def update(self, consumed_samples: int, consistency_check: bool = True) -> None:
        """Update the calculator state based on training progress."""
        ...

    def get_current_global_batch_size(self) -> int:
        """Get the current effective global batch size."""
        ...


class MicrobatchCalculatorBase(ABC):
    """Abstract base class for microbatch calculators.

    This class provides the foundation for all microbatch calculation strategies,
    following the Megatron-LM pattern for distributed training with pipeline
    parallelism. It handles the core batch size arithmetic and provides common
    utilities for validation and state management.

    Key Responsibilities:
    - Validate input parameters and detect configuration errors early
    - Provide thread-safe access to microbatch calculations
    - Ensure consistency across different parallelism dimensions
    - Support dynamic adjustment strategies for adaptive calculators

    Performance Characteristics:
    - O(1) time complexity for all calculation methods
    - Minimal memory overhead with efficient state storage
    - Thread-safe operations suitable for multi-threaded training loops

    Thread Safety:
    - All methods are thread-safe unless otherwise noted
    - Subclasses should use appropriate locking for mutable state
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
        if value > MAX_SAFE_BATCH_SIZE:  # Prevent overflow
            raise InvalidConfigurationError(
                f"{name} too large (overflow risk), got {value}. "
                f"Maximum allowed: {MAX_SAFE_BATCH_SIZE}"
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
        # Thread-safe access to mutable state
        self._adjustment_lock = threading.RLock()

    def get_num_microbatches(self) -> int:
        """Get current number of microbatches."""
        return self.global_batch_size_per_gpu // self.current_micro_batch_size

    def get_micro_batch_size(self) -> int:
        """Get current microbatch size."""
        return self.current_micro_batch_size

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage as fraction with enhanced error handling.

        Returns:
            Memory usage as fraction (0-1), or 0.0 if CUDA unavailable or error occurs.

        Note:
            Uses reserved memory rather than allocated for more accurate
            measurement of actual memory pressure. Includes fallback mechanisms
            for different CUDA configurations.
        """
        if not torch.cuda.is_available():
            return 0.0

        try:
            # Get current device to avoid hardcoded device 0
            current_device = torch.cuda.current_device()

            # Try multiple memory measurement approaches for robustness
            try:
                reserved_bytes = torch.cuda.memory_reserved(current_device)
                total_bytes = torch.cuda.get_device_properties(
                    current_device
                ).total_memory
            except RuntimeError:
                # Fallback to allocated memory if reserved fails
                logger.debug(
                    "Reserved memory query failed, falling back to allocated memory"
                )
                reserved_bytes = torch.cuda.memory_allocated(current_device)
                total_bytes = torch.cuda.get_device_properties(
                    current_device
                ).total_memory

            if total_bytes <= 0:
                logger.warning(
                    f"Invalid total memory: {total_bytes} bytes on device "
                    f"{current_device}"
                )
                return 0.0

            usage_fraction = float(reserved_bytes) / float(total_bytes)

            # Validate result is reasonable
            if (
                usage_fraction < 0.0 or usage_fraction > 1.1
            ):  # Allow slight overflow for safety
                logger.warning(
                    f"Suspicious memory usage: {usage_fraction:.3f} on device "
                    f"{current_device}"
                )
                return min(1.0, max(0.0, usage_fraction))

            return min(1.0, max(0.0, usage_fraction))  # Clamp to [0, 1]

        except (RuntimeError, AttributeError, ZeroDivisionError) as e:
            logger.warning(
                f"Failed to get memory usage on device "
                f"{torch.cuda.current_device()}: {e}"
            )
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

        Note:
            This method is thread-safe and uses locking to protect mutable state.
        """
        import time

        with self._adjustment_lock:
            memory_usage = self._get_memory_usage()

            # Store adjustment record in circular buffer (using factory for validation)
            record = create_adjustment_record(
                consumed_samples=consumed_samples,
                micro_batch_size=self.current_micro_batch_size,
                memory_usage=memory_usage,
                timestamp=time.time(),
            )
            self.adjustment_history.append(record)

            # Adjust every N samples to avoid thrashing
            adjustment_interval = (
                self.global_batch_size * ADJUSTMENT_INTERVAL_MULTIPLIER
            )
            should_adjust = (
                consumed_samples - self._last_adjustment_samples >= adjustment_interval
            )

            if should_adjust:
                old_size = self.current_micro_batch_size
                self._adjust_microbatch_size(memory_usage)
                self._last_adjustment_samples = consumed_samples

                # Log adjustment if size changed
                if old_size != self.current_micro_batch_size and logger.isEnabledFor(
                    logging.DEBUG
                ):
                    logger.debug(
                        f"Microbatch size adjusted: {old_size} -> "
                        f"{self.current_micro_batch_size} (memory: {memory_usage:.2%})"
                    )

        # Ensure consistency across ranks (outside lock to avoid deadlock)
        if consistency_check and dist.is_initialized():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tensor = torch.tensor(
                    [self.current_micro_batch_size],
                    dtype=torch.long,
                    device=device,
                )
                dist.all_reduce(
                    tensor, op=dist.ReduceOp.MIN
                )  # Use minimum across ranks

                with self._adjustment_lock:
                    consistent_size = int(tensor.item())
                    if consistent_size != self.current_micro_batch_size:
                        logger.info(
                            f"Adjusting microbatch size for consistency: "
                            f"{self.current_micro_batch_size} -> {consistent_size}"
                        )
                        self.current_micro_batch_size = consistent_size

            except Exception as e:
                logger.warning(
                    f"Failed to synchronize microbatch size across ranks: {e}"
                )


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

    This function provides a more accurate estimation by considering:
    - Precision-aware memory calculations
    - Activation checkpointing memory savings
    - Pipeline parallelism effects
    - Overflow protection with safe integer arithmetic

    Args:
        model_size_gb: Model size in GB
        available_memory_gb: Available GPU memory in GB
        sequence_length: Sequence length
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        pipeline_parallel_size: Pipeline parallel size
        activation_checkpoint: Whether using activation checkpointing
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw', etc.)
        precision: Training precision ('fp16', 'fp32', 'bf16', 'fp64')

    Returns:
        Suggested microbatch size (guaranteed to be >= 1)

    Raises:
        ValueError: If parameters are invalid or would cause overflow
    """
    # Comprehensive input validation
    if not (0 < model_size_gb < 1000):  # Reasonable bounds
        raise ValueError(f"Model size must be in (0, 1000) GB, got {model_size_gb}")
    if not (0 < available_memory_gb < 1000):
        raise ValueError(
            f"Available memory must be in (0, 1000) GB, got {available_memory_gb}"
        )
    if not (1 <= sequence_length <= 1_000_000):  # Prevent overflow
        raise ValueError(f"Sequence length must be in [1, 1M], got {sequence_length}")
    if not (1 <= hidden_size <= 100_000):
        raise ValueError(f"Hidden size must be in [1, 100K], got {hidden_size}")
    if not (1 <= num_layers <= 1000):
        raise ValueError(f"Number of layers must be in [1, 1000], got {num_layers}")
    if not (1 <= pipeline_parallel_size <= 1000):
        raise ValueError(
            f"Pipeline parallel size must be in [1, 1000], got {pipeline_parallel_size}"
        )

    # Precision mapping with validation
    precision_bytes = {
        "fp16": BYTES_PER_FP16_ELEMENT,
        "bf16": BYTES_PER_BF16_ELEMENT,
        "fp32": BYTES_PER_FP32_ELEMENT,
        "fp64": BYTES_PER_FP64_ELEMENT,
    }
    precision_lower = precision.lower()
    if precision_lower not in precision_bytes:
        raise ValueError(
            f"Unknown precision: {precision}. "
            f"Valid options: {list(precision_bytes.keys())}"
        )

    bytes_per_element = precision_bytes[precision_lower]

    # Safe integer arithmetic to prevent overflow
    try:
        # Calculate activation memory per sample (in bytes)
        base_activation_elements = hidden_size * sequence_length

        if activation_checkpoint:
            # Store only sqrt(num_layers) checkpoints + 1 layer of activations
            checkpoint_layers = max(1, int(math.sqrt(num_layers)))
            activation_elements = base_activation_elements * checkpoint_layers
        else:
            # Store all layer activations
            activation_elements = base_activation_elements * num_layers

        # Include forward + backward activations
        total_activation_elements = int(
            activation_elements * ACTIVATION_MEMORY_MULTIPLIER
        )

        # Convert to GB with overflow protection
        if total_activation_elements > (
            2**50 // bytes_per_element
        ):  # Prevent overflow
            logger.warning(
                f"Activation memory calculation would overflow, using fallback"
            )
            return MIN_BATCH_SIZE

        activation_memory_bytes = total_activation_elements * bytes_per_element
        activation_memory_gb = activation_memory_bytes / GB_TO_BYTES

    except (OverflowError, ValueError) as e:
        logger.warning(f"Overflow in memory calculation: {e}")
        return MIN_BATCH_SIZE

    # Model memory per GPU (safely divided)
    model_memory_per_gpu = model_size_gb / max(1, pipeline_parallel_size)

    # Optimizer memory using safe lookup
    optimizer_multiplier = OPTIMIZER_STATE_MULTIPLIERS.get(
        optimizer_type.lower(), OPTIMIZER_STATE_MULTIPLIERS["adam"]  # Default to Adam
    )
    optimizer_memory = model_memory_per_gpu * optimizer_multiplier

    # Calculate available memory with safety margin
    total_fixed_memory = model_memory_per_gpu + optimizer_memory
    if total_fixed_memory >= available_memory_gb:
        logger.warning(
            f"Fixed memory ({total_fixed_memory:.2f} GB) exceeds "
            f"available memory ({available_memory_gb:.2f} GB)"
        )
        return MIN_BATCH_SIZE

    available_for_activations = available_memory_gb - total_fixed_memory
    available_for_activations *= MEMORY_SAFETY_MARGIN

    # Calculate microbatch size with proper bounds checking
    if available_for_activations <= 0 or activation_memory_gb <= 0:
        return MIN_BATCH_SIZE

    # Safe division and rounding
    raw_microbatch_size = available_for_activations / activation_memory_gb
    if raw_microbatch_size < 1:
        return MIN_BATCH_SIZE

    # Round down to nearest power of 2 for memory alignment efficiency
    microbatch_size = int(raw_microbatch_size)
    if microbatch_size > 1:
        # Find largest power of 2 <= microbatch_size
        power_of_2 = 1 << (microbatch_size.bit_length() - 1)
        microbatch_size = power_of_2

    # Final bounds check
    return max(
        MIN_BATCH_SIZE,
        min(microbatch_size, MAX_SAFE_BATCH_SIZE // (sequence_length or 1)),
    )


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
