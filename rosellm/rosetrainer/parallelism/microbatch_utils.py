"""Utility functions and helpers for microbatch calculation.

This module provides additional utilities for working with microbatch calculators,
including performance monitoring, debugging tools, and integration helpers.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .microbatch_calculator import MicrobatchCalculatorBase, get_microbatch_calculator

logger = logging.getLogger(__name__)


@dataclass
class MicrobatchMetrics:
    """Metrics for microbatch processing performance."""

    total_samples: int = 0
    total_microbatches: int = 0
    total_time_ms: float = 0.0
    memory_peaks: List[float] = field(default_factory=list)
    throughput_samples_per_sec: float = 0.0
    average_microbatch_time_ms: float = 0.0
    memory_efficiency: float = 0.0

    def update(
        self, samples: int, microbatches: int, time_ms: float, memory_gb: float
    ) -> None:
        """Update metrics with new measurements."""
        self.total_samples += samples
        self.total_microbatches += microbatches
        self.total_time_ms += time_ms
        self.memory_peaks.append(memory_gb)

        if self.total_time_ms > 0:
            self.throughput_samples_per_sec = self.total_samples / (
                self.total_time_ms / 1000
            )
            self.average_microbatch_time_ms = self.total_time_ms / max(
                1, self.total_microbatches
            )

        if self.memory_peaks:
            avg_memory = sum(self.memory_peaks) / len(self.memory_peaks)
            max_memory = max(self.memory_peaks)
            self.memory_efficiency = avg_memory / max_memory if max_memory > 0 else 0

    def __str__(self) -> str:
        """String representation of metrics."""
        return (
            f"MicrobatchMetrics(\n"
            f"  samples={self.total_samples:,}\n"
            f"  microbatches={self.total_microbatches:,}\n"
            f"  throughput={self.throughput_samples_per_sec:.1f} samples/sec\n"
            f"  avg_time={self.average_microbatch_time_ms:.2f}ms/microbatch\n"
            f"  memory_efficiency={self.memory_efficiency:.2%}\n"
            f")"
        )


class MicrobatchProfiler:
    """Profiler for monitoring microbatch processing performance."""

    def __init__(self, enabled: bool = True):
        """Initialize the profiler.

        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.metrics = MicrobatchMetrics()
        self._start_time: Optional[float] = None
        self._microbatch_count = 0

    def start_iteration(self) -> None:
        """Start timing a new iteration."""
        if self.enabled:
            self._start_time = time.perf_counter()
            self._microbatch_count = 0

    def record_microbatch(self) -> None:
        """Record a processed microbatch."""
        if self.enabled:
            self._microbatch_count += 1

    def end_iteration(self, samples: int) -> None:
        """End timing the current iteration.

        Args:
            samples: Number of samples processed in this iteration
        """
        if self.enabled and self._start_time is not None:
            elapsed_ms = (time.perf_counter() - self._start_time) * 1000
            memory_gb = self._get_memory_usage()

            self.metrics.update(
                samples=samples,
                microbatches=self._microbatch_count,
                time_ms=elapsed_ms,
                memory_gb=memory_gb,
            )

            self._start_time = None
            self._microbatch_count = 0

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024**3))
        return 0.0

    def get_metrics(self) -> MicrobatchMetrics:
        """Get current metrics."""
        return self.metrics

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = MicrobatchMetrics()


def validate_microbatch_config(
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    pipeline_parallel_size: int = 1,
    gradient_accumulation_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate and suggest microbatch configuration with performance optimizations.

    This function performs comprehensive validation with early returns and
    cached calculations for better performance.

    Args:
        global_batch_size: Total batch size across all ranks
        micro_batch_size: Size of each microbatch
        data_parallel_size: Number of data parallel ranks
        pipeline_parallel_size: Number of pipeline stages
        gradient_accumulation_steps: Optional expected accumulation steps

    Returns:
        Dictionary with validation results and suggestions

    Note:
        Uses optimized divisor calculation and early validation exits.
    """
    # Fast input validation with early returns
    if any(
        x <= 0
        for x in [
            global_batch_size,
            micro_batch_size,
            data_parallel_size,
            pipeline_parallel_size,
        ]
    ):
        return {
            "valid": False,
            "warnings": ["All size parameters must be positive"],
            "suggestions": ["Check input parameters"],
            "calculated_values": {},
        }

    results: Dict[str, Any] = {
        "valid": True,
        "warnings": [],
        "suggestions": [],
        "calculated_values": {},
    }

    # Check global batch size divisibility first (most common failure)
    if global_batch_size % data_parallel_size != 0:
        results["valid"] = False
        results["warnings"].append(
            f"Global batch size {global_batch_size} not divisible by "
            f"data parallel size {data_parallel_size}"
        )
        # Optimize suggestion calculation
        suggested_bs = (
            (global_batch_size + data_parallel_size - 1) // data_parallel_size
        ) * data_parallel_size
        results["suggestions"].append(f"Consider using batch size {suggested_bs}")

    # Calculate once and reuse
    batch_per_gpu = global_batch_size // data_parallel_size
    results["calculated_values"]["batch_per_gpu"] = batch_per_gpu

    # Check micro batch size divisibility
    if batch_per_gpu % micro_batch_size != 0:
        results["valid"] = False
        results["warnings"].append(
            f"Batch per GPU {batch_per_gpu} not divisible by "
            f"micro batch size {micro_batch_size}"
        )

        # Optimized divisor finding using mathematical approach
        closest_divisor = _find_closest_divisor(batch_per_gpu, micro_batch_size)
        results["suggestions"].append(f"Consider micro batch size {closest_divisor}")

    # Early exit if basic validation failed
    if not results["valid"]:
        results["calculated_values"]["num_microbatches"] = 0
        return results

    # Calculate derived values
    num_microbatches = batch_per_gpu // micro_batch_size
    results["calculated_values"]["num_microbatches"] = num_microbatches

    # Pipeline parallelism analysis (only if pipeline_parallel_size > 1)
    if pipeline_parallel_size > 1:
        _validate_pipeline_config(
            results,
            num_microbatches,
            pipeline_parallel_size,
            micro_batch_size,
            data_parallel_size,
        )

    # Gradient accumulation consistency check
    if (
        gradient_accumulation_steps is not None
        and gradient_accumulation_steps != num_microbatches
    ):
        results["warnings"].append(
            f"Gradient accumulation steps ({gradient_accumulation_steps}) "
            f"doesn't match calculated microbatches ({num_microbatches})"
        )

    return results


def _find_closest_divisor(n: int, target: int) -> int:
    """Find the divisor of n closest to target value.

    Uses optimized algorithm that checks divisors in order of proximity to target.
    """
    if n <= 0:
        return 1

    # Check target first
    if n % target == 0:
        return target

    # Check nearby values in expanding radius
    for delta in range(1, min(target, n - target) + 1):
        # Check target - delta
        candidate_low = target - delta
        if candidate_low > 0 and n % candidate_low == 0:
            return candidate_low

        # Check target + delta
        candidate_high = target + delta
        if candidate_high <= n and n % candidate_high == 0:
            return candidate_high

    # Fallback: find all divisors and pick closest (should be rare)
    divisors = [i for i in range(1, int(n**0.5) + 1) if n % i == 0]
    all_divisors = divisors + [n // d for d in divisors if d != n // d]
    return min(all_divisors, key=lambda x: abs(x - target))


def _validate_pipeline_config(
    results: Dict[str, Any],
    num_microbatches: int,
    pipeline_parallel_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
) -> None:
    """Validate pipeline parallelism configuration in-place."""
    # Minimum microbatches check
    if num_microbatches < pipeline_parallel_size:
        results["warnings"].append(
            f"Number of microbatches ({num_microbatches}) less than "
            f"pipeline stages ({pipeline_parallel_size}). "
            f"This will cause pipeline bubbles."
        )
        min_microbatches = pipeline_parallel_size * 2  # Recommended minimum
        suggested_global = min_microbatches * micro_batch_size * data_parallel_size
        results["suggestions"].append(
            f"Consider increasing batch size to at least {suggested_global}"
        )

    # Pipeline efficiency calculation
    # Bubble time = (P-1) * (F + B) where P = pipeline stages
    # Total time = M * (F + B) where M = microbatches
    # Efficiency = (Total - Bubble) / Total = 1 - (P-1)/M
    efficiency = 1.0 - (pipeline_parallel_size - 1) / max(1, num_microbatches)
    results["calculated_values"]["pipeline_efficiency"] = max(0.0, efficiency)

    # Pipeline efficiency warning
    from rosellm.rosetrainer.parallelism.microbatch_calculator import (
        MIN_PIPELINE_EFFICIENCY,
    )

    if efficiency < MIN_PIPELINE_EFFICIENCY:
        results["warnings"].append(
            f"Pipeline efficiency is low ({efficiency:.1%}). "
            "Consider increasing number of microbatches."
        )


def suggest_optimal_config(
    model_params: int,
    available_memory_gb: float,
    world_size: int,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    sequence_length: int = 2048,
    target_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Suggest optimal microbatch configuration based on constraints.

    Args:
        model_params: Number of model parameters
        available_memory_gb: Available GPU memory per device
        world_size: Total number of GPUs
        tensor_parallel_size: Tensor parallel degree
        pipeline_parallel_size: Pipeline parallel degree
        sequence_length: Sequence length
        target_batch_size: Optional target global batch size

    Returns:
        Dictionary with suggested configuration
    """
    # Calculate data parallel size
    data_parallel_size = world_size // (tensor_parallel_size * pipeline_parallel_size)

    # Estimate model memory (rough approximation)
    model_size_gb = (model_params * 2) / (1024**3)  # FP16
    model_per_gpu = model_size_gb / tensor_parallel_size

    # Estimate optimizer memory (Adam)
    optimizer_memory = model_per_gpu * 2

    # Available for activations
    activation_memory = available_memory_gb - model_per_gpu - optimizer_memory
    activation_memory *= 0.8  # Safety margin

    # Estimate microbatch size based on memory
    # Very rough estimate: assume square root scaling with sequence length
    memory_per_sample_gb = (sequence_length / 1024) * 0.001  # Very rough
    max_micro_batch = int(activation_memory / memory_per_sample_gb)
    max_micro_batch = max(1, min(max_micro_batch, 64))  # Reasonable bounds

    # Round to power of 2
    micro_batch_size = 2 ** (max_micro_batch.bit_length() - 1)

    # Calculate global batch size
    if target_batch_size is None:
        # Aim for 8-16 microbatches per GPU for good efficiency
        target_microbatches = min(16, max(8, pipeline_parallel_size * 2))
        global_batch_size = micro_batch_size * target_microbatches * data_parallel_size
    else:
        global_batch_size = target_batch_size

    # Validate and adjust
    validation = validate_microbatch_config(
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
    )

    config = {
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "data_parallel_size": data_parallel_size,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "num_microbatches": validation["calculated_values"].get("num_microbatches", 0),
        "estimated_memory_gb": (
            model_per_gpu + optimizer_memory + activation_memory * 0.5
        ),
        "validation": validation,
    }

    return config


def log_microbatch_info(
    calculator: Optional[MicrobatchCalculatorBase] = None,
    rank: int = 0,
) -> None:
    """Log detailed microbatch configuration information.

    Args:
        calculator: Microbatch calculator instance (uses global if None)
        rank: Process rank for conditional logging
    """
    if rank != 0:
        return

    if calculator is None:
        calculator = get_microbatch_calculator()

    if calculator is None:
        logger.warning("No microbatch calculator initialized")
        return

    num_mb = calculator.get_num_microbatches()
    mb_size = calculator.get_micro_batch_size()
    global_bs = calculator.get_current_global_batch_size()

    logger.info("=" * 60)
    logger.info("Microbatch Configuration")
    logger.info("=" * 60)
    logger.info(f"  Calculator Type: {calculator.__class__.__name__}")
    logger.info(f"  Global Batch Size: {global_bs}")
    logger.info(f"  Micro Batch Size: {mb_size}")
    logger.info(f"  Number of Microbatches: {num_mb}")
    logger.info(f"  Data Parallel Size: {calculator.data_parallel_size}")
    logger.info(f"  Batch per GPU: {calculator.global_batch_size_per_gpu}")

    # Additional info for specific calculator types
    if hasattr(calculator, "memory_threshold"):
        threshold = getattr(calculator, "memory_threshold", 0)
        logger.info(f"  Memory Threshold: {threshold:.1%}")
    if hasattr(calculator, "rampup_batch_size"):
        schedule = getattr(calculator, "rampup_batch_size", [])
        logger.info(f"  Rampup Schedule: {schedule}")

    logger.info("=" * 60)
