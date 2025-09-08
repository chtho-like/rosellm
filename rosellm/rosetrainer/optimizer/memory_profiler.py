"""Memory profiling utilities for distributed optimizer.

Provides detailed memory usage tracking and optimization recommendations.
"""

import gc
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Container for memory statistics."""

    allocated_mb: float
    reserved_mb: float
    active_mb: float
    inactive_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    num_alloc_retries: int
    num_ooms: int

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "allocated_mb": self.allocated_mb,
            "reserved_mb": self.reserved_mb,
            "active_mb": self.active_mb,
            "inactive_mb": self.inactive_mb,
            "peak_allocated_mb": self.peak_allocated_mb,
            "peak_reserved_mb": self.peak_reserved_mb,
            "num_alloc_retries": self.num_alloc_retries,
            "num_ooms": self.num_ooms,
        }

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"Memory Stats: Allocated={self.allocated_mb:.1f}MB, "
            f"Reserved={self.reserved_mb:.1f}MB, Peak={self.peak_allocated_mb:.1f}MB"
        )


class MemoryProfiler:
    """Profile and track memory usage for distributed optimizer."""

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize memory profiler.

        Args:
            device: Device to profile (defaults to current CUDA device).
        """
        self.device = (
            device or torch.cuda.current_device() if torch.cuda.is_available() else None
        )
        self.baseline_stats: Optional[MemoryStats] = None
        self.history: List[MemoryStats] = []
        self.enabled = torch.cuda.is_available()

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics.

        Returns:
            Current memory statistics.
        """
        if not self.enabled:
            return MemoryStats(0, 0, 0, 0, 0, 0, 0, 0)

        bytes_to_mb = 1.0 / (1024 * 1024)

        stats = MemoryStats(
            allocated_mb=torch.cuda.memory_allocated(self.device) * bytes_to_mb,
            reserved_mb=torch.cuda.memory_reserved(self.device) * bytes_to_mb,
            active_mb=torch.cuda.memory_allocated(self.device) * bytes_to_mb,
            inactive_mb=(
                torch.cuda.memory_reserved(self.device)
                - torch.cuda.memory_allocated(self.device)
            )
            * bytes_to_mb,
            peak_allocated_mb=(
                torch.cuda.max_memory_allocated(self.device) * bytes_to_mb
            ),
            peak_reserved_mb=torch.cuda.max_memory_reserved(self.device) * bytes_to_mb,
            num_alloc_retries=torch.cuda.memory_stats(self.device).get(
                "num_alloc_retries", 0
            ),
            num_ooms=torch.cuda.memory_stats(self.device).get("num_ooms", 0),
        )

        return stats

    def set_baseline(self) -> None:
        """Set current memory usage as baseline."""
        self.baseline_stats = self.get_current_stats()

    def get_delta_from_baseline(self) -> Optional[MemoryStats]:
        """Get memory delta from baseline.

        Returns:
            Memory statistics delta, or None if no baseline set.
        """
        if self.baseline_stats is None:
            return None

        current = self.get_current_stats()

        return MemoryStats(
            allocated_mb=current.allocated_mb - self.baseline_stats.allocated_mb,
            reserved_mb=current.reserved_mb - self.baseline_stats.reserved_mb,
            active_mb=current.active_mb - self.baseline_stats.active_mb,
            inactive_mb=current.inactive_mb - self.baseline_stats.inactive_mb,
            peak_allocated_mb=current.peak_allocated_mb,
            peak_reserved_mb=current.peak_reserved_mb,
            num_alloc_retries=(
                current.num_alloc_retries - self.baseline_stats.num_alloc_retries
            ),
            num_ooms=current.num_ooms - self.baseline_stats.num_ooms,
        )

    def record_snapshot(self) -> None:
        """Record current memory snapshot to history."""
        self.history.append(self.get_current_stats())

    def analyze_model_memory(
        self, model: nn.Module, include_buffers: bool = True
    ) -> Dict[str, float]:
        """Analyze memory usage of a model.

        Args:
            model: Model to analyze.
            include_buffers: Whether to include buffers in analysis.

        Returns:
            Dictionary with memory breakdown.
        """
        memory_breakdown = {
            "parameters_mb": 0.0,
            "gradients_mb": 0.0,
            "buffers_mb": 0.0,
            "total_mb": 0.0,
        }

        bytes_to_mb = 1.0 / (1024 * 1024)

        # Analyze parameters
        for param in model.parameters():
            param_bytes = param.numel() * param.element_size()
            memory_breakdown["parameters_mb"] += param_bytes * bytes_to_mb

            if param.grad is not None:
                grad_bytes = param.grad.numel() * param.grad.element_size()
                memory_breakdown["gradients_mb"] += grad_bytes * bytes_to_mb

        # Analyze buffers
        if include_buffers:
            for buffer in model.buffers():
                buffer_bytes = buffer.numel() * buffer.element_size()
                memory_breakdown["buffers_mb"] += buffer_bytes * bytes_to_mb

        memory_breakdown["total_mb"] = (
            memory_breakdown["parameters_mb"]
            + memory_breakdown["gradients_mb"]
            + memory_breakdown["buffers_mb"]
        )

        return memory_breakdown

    def estimate_optimizer_memory(
        self,
        num_parameters: int,
        optimizer_type: str = "Adam",
        dtype: torch.dtype = torch.float32,
    ) -> float:
        """Estimate optimizer state memory usage.

        Args:
            num_parameters: Number of model parameters.
            optimizer_type: Type of optimizer.
            dtype: Data type of parameters.

        Returns:
            Estimated memory usage in MB.
        """
        # bytes_per_element = 4 if dtype == torch.float32 else 2  # Not used

        # Optimizer state multipliers
        state_multipliers = {
            "SGD": 1,  # Momentum only
            "Adam": 2,  # First and second moments
            "AdamW": 2,  # First and second moments
            "RMSprop": 2,  # Square average and momentum
            "Adagrad": 1,  # Sum of squares
        }

        multiplier = state_multipliers.get(optimizer_type, 2)

        # States are typically stored in FP32 regardless of model dtype
        state_bytes = num_parameters * 4 * multiplier

        return state_bytes / (1024 * 1024)

    def get_memory_summary(self) -> str:
        """Get formatted memory summary.

        Returns:
            Formatted string with memory summary.
        """
        if not self.enabled:
            return "CUDA not available - memory profiling disabled"

        stats = self.get_current_stats()
        delta = self.get_delta_from_baseline()

        summary = [
            "Memory Summary:",
            f"  Current: {stats}",
        ]

        if delta is not None:
            summary.append(
                f"  Delta from baseline: Allocated={delta.allocated_mb:+.1f}MB, "
                f"Reserved={delta.reserved_mb:+.1f}MB"
            )

        if stats.num_ooms > 0:
            summary.append(f"  WARNING: {stats.num_ooms} OOM errors detected!")

        if stats.num_alloc_retries > 0:
            summary.append(f"  WARNING: {stats.num_alloc_retries} allocation retries")

        # Memory fragmentation analysis
        fragmentation = (stats.reserved_mb - stats.allocated_mb) / max(
            stats.reserved_mb, 1
        )
        if fragmentation > 0.3:
            summary.append(
                f"  WARNING: High memory fragmentation ({fragmentation:.1%}). "
                f"Consider calling torch.cuda.empty_cache()"
            )

        return "\n".join(summary)

    def optimize_memory(self) -> Dict[str, str]:
        """Get memory optimization recommendations.

        Returns:
            Dictionary of optimization recommendations.
        """
        recommendations = {}
        stats = self.get_current_stats()

        # Check for high memory usage
        if stats.allocated_mb > 0.9 * stats.reserved_mb:
            recommendations["high_usage"] = (
                "Memory usage is near capacity. Consider: "
                "1) Reducing batch size, "
                "2) Enabling gradient checkpointing, "
                "3) Using mixed precision training"
            )

        # Check for fragmentation
        fragmentation = (stats.reserved_mb - stats.allocated_mb) / max(
            stats.reserved_mb, 1
        )
        if fragmentation > 0.3:
            recommendations["fragmentation"] = (
                f"High memory fragmentation ({fragmentation:.1%}). "
                "Call torch.cuda.empty_cache() periodically"
            )

        # Check for OOMs
        if stats.num_ooms > 0:
            recommendations["oom"] = (
                f"Detected {stats.num_ooms} OOM errors. "
                "Reduce memory usage or enable CPU offloading"
            )

        # Check peak vs current
        if stats.peak_allocated_mb > 1.5 * stats.allocated_mb:
            recommendations["peak_usage"] = (
                "Peak memory significantly higher than current. "
                "Consider gradient accumulation to smooth memory usage"
            )

        return recommendations

    def cleanup(self) -> None:
        """Perform memory cleanup operations."""
        if self.enabled:
            # Clear cache
            torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            # Clear cache again after GC
            torch.cuda.empty_cache()

            logger.info("Memory cleanup completed")
