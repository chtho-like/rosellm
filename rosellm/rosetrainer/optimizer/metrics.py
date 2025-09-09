"""
Performance metrics and monitoring for distributed optimizer.

This module provides comprehensive metrics collection and analysis
for monitoring distributed training performance.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class OptimizerMetrics:
    """Container for optimizer performance metrics."""

    # Timing metrics (in seconds)
    gradient_reduction_time: float = 0.0
    parameter_update_time: float = 0.0
    broadcast_time: float = 0.0
    total_step_time: float = 0.0

    # Communication metrics
    num_all_reduces: int = 0
    num_broadcasts: int = 0
    bytes_communicated: int = 0

    # Gradient metrics
    gradient_norm: float = 0.0
    gradient_clips: int = 0
    max_gradient_norm: float = 0.0

    # Memory metrics
    peak_memory_mb: float = 0.0
    allocated_memory_mb: float = 0.0

    # Efficiency metrics
    communication_efficiency: float = 0.0  # Ratio of computation to communication time
    bucket_efficiency: float = 0.0  # Average bucket fill ratio

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timing": {
                "gradient_reduction_ms": self.gradient_reduction_time * 1000,
                "parameter_update_ms": self.parameter_update_time * 1000,
                "broadcast_ms": self.broadcast_time * 1000,
                "total_step_ms": self.total_step_time * 1000,
            },
            "communication": {
                "num_all_reduces": self.num_all_reduces,
                "num_broadcasts": self.num_broadcasts,
                "bytes_communicated_mb": self.bytes_communicated / (1024 * 1024),
            },
            "gradients": {
                "norm": self.gradient_norm,
                "clips": self.gradient_clips,
                "max_norm": self.max_gradient_norm,
            },
            "memory": {
                "peak_mb": self.peak_memory_mb,
                "allocated_mb": self.allocated_memory_mb,
            },
            "efficiency": {
                "communication": self.communication_efficiency,
                "bucket": self.bucket_efficiency,
            },
        }


class PerformanceMonitor:
    """
    Performance monitoring for distributed optimizer.

    Tracks and analyzes performance metrics across training steps
    to identify bottlenecks and optimization opportunities.
    """

    def __init__(self, window_size: int = 100, device: Optional[torch.device] = None):
        """
        Initialize performance monitor.

        Args:
            window_size: Size of sliding window for moving averages
            device: Device for memory tracking (None for CPU)
        """
        self.window_size = window_size
        self.device = device

        # Sliding windows for moving averages
        self.step_times: deque = deque(maxlen=window_size)
        self.gradient_norms: deque = deque(maxlen=window_size)
        self.reduction_times: deque = deque(maxlen=window_size)

        # Cumulative metrics
        self.total_steps: int = 0
        self.total_gradient_clips: int = 0
        self.total_bytes_communicated: int = 0

        # Current step metrics
        self.current_metrics = OptimizerMetrics()

        # Timing helpers
        self._timers: Dict[str, float] = {}

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._timers[name] = time.perf_counter()

    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time."""
        if name not in self._timers:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0

        elapsed = time.perf_counter() - self._timers[name]
        del self._timers[name]
        return elapsed

    def record_gradient_reduction(
        self, duration: float, num_params: int, bytes_reduced: int
    ) -> None:
        """Record gradient reduction metrics."""
        self.current_metrics.gradient_reduction_time = duration
        self.current_metrics.num_all_reduces += 1
        self.current_metrics.bytes_communicated += bytes_reduced
        self.total_bytes_communicated += bytes_reduced
        self.reduction_times.append(duration)

    def record_parameter_update(self, duration: float) -> None:
        """Record parameter update time."""
        self.current_metrics.parameter_update_time = duration

    def record_broadcast(self, duration: float, bytes_broadcast: int) -> None:
        """Record parameter broadcast metrics."""
        self.current_metrics.broadcast_time = duration
        self.current_metrics.num_broadcasts += 1
        self.current_metrics.bytes_communicated += bytes_broadcast
        self.total_bytes_communicated += bytes_broadcast

    def record_gradient_norm(self, norm: float, clipped: bool = False) -> None:
        """Record gradient norm and clipping."""
        self.current_metrics.gradient_norm = norm
        self.gradient_norms.append(norm)

        if clipped:
            self.current_metrics.gradient_clips += 1
            self.total_gradient_clips += 1

        if norm > self.current_metrics.max_gradient_norm:
            self.current_metrics.max_gradient_norm = norm

    def record_memory_usage(self) -> None:
        """Record current memory usage."""
        if self.device and self.device.type == "cuda":
            self.current_metrics.allocated_memory_mb = torch.cuda.memory_allocated(
                self.device
            ) / (1024 * 1024)
            self.current_metrics.peak_memory_mb = torch.cuda.max_memory_allocated(
                self.device
            ) / (1024 * 1024)

    def calculate_efficiency(
        self, computation_time: float, communication_time: float
    ) -> None:
        """Calculate communication efficiency."""
        if communication_time > 0:
            self.current_metrics.communication_efficiency = computation_time / (
                computation_time + communication_time
            )
        else:
            self.current_metrics.communication_efficiency = 1.0

    def step(self) -> None:
        """Complete a monitoring step and update statistics."""
        self.total_steps += 1

        # Record total step time
        if "step" in self._timers:
            self.current_metrics.total_step_time = self.end_timer("step")
            self.step_times.append(self.current_metrics.total_step_time)

        # Record memory usage
        self.record_memory_usage()

        # Calculate efficiency
        if self.current_metrics.gradient_reduction_time > 0:
            computation_time = (
                self.current_metrics.total_step_time
                - self.current_metrics.gradient_reduction_time
                - self.current_metrics.broadcast_time
            )
            communication_time = (
                self.current_metrics.gradient_reduction_time
                + self.current_metrics.broadcast_time
            )
            self.calculate_efficiency(computation_time, communication_time)

    def get_current_metrics(self) -> OptimizerMetrics:
        """Get metrics for the current step."""
        return self.current_metrics

    def get_average_metrics(self) -> Dict[str, float]:
        """Get moving average metrics."""
        avg_metrics = {}

        if self.step_times:
            avg_metrics["avg_step_time_ms"] = (
                sum(self.step_times) / len(self.step_times) * 1000
            )

        if self.gradient_norms:
            avg_metrics["avg_gradient_norm"] = sum(self.gradient_norms) / len(
                self.gradient_norms
            )

        if self.reduction_times:
            avg_metrics["avg_reduction_time_ms"] = (
                sum(self.reduction_times) / len(self.reduction_times) * 1000
            )

        if self.total_steps > 0:
            avg_metrics["gradient_clip_rate"] = (
                self.total_gradient_clips / self.total_steps
            )
            avg_metrics["avg_bytes_per_step_mb"] = (
                self.total_bytes_communicated / self.total_steps / (1024 * 1024)
            )

        return avg_metrics

    def reset(self) -> None:
        """Reset current step metrics."""
        self.current_metrics = OptimizerMetrics()
        self._timers.clear()

    def log_summary(self, step: Optional[int] = None) -> None:
        """Log a summary of current metrics."""
        metrics = self.current_metrics.to_dict()
        avg_metrics = self.get_average_metrics()

        step_str = f"Step {step}" if step is not None else "Current"

        logger.info(
            f"{step_str} - "
            f"Step time: {metrics['timing']['total_step_ms']:.2f}ms, "
            f"Gradient norm: {metrics['gradients']['norm']:.4f}, "
            f"Communication efficiency: {metrics['efficiency']['communication']:.2%}"
        )

        if avg_metrics:
            logger.info(
                f"Moving averages - "
                f"Step time: {avg_metrics.get('avg_step_time_ms', 0):.2f}ms, "
                f"Gradient norm: {avg_metrics.get('avg_gradient_norm', 0):.4f}, "
                f"Reduction time: {avg_metrics.get('avg_reduction_time_ms', 0):.2f}ms"
            )
