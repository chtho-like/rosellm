"""Timer configuration for performance profiling."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TimerLogLevel(Enum):
    """Log level for timer output."""

    OFF = 0  # No logging
    SUMMARY = 1  # Only summary at end
    INTERVAL = 2  # Log at intervals
    VERBOSE = 3  # Detailed logging


class TimerAggregation(Enum):
    """Aggregation method for distributed timers."""

    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    SUM = "sum"


@dataclass
class TimerConfig:
    """Configuration for performance timers.

    Attributes:
        enabled: Whether timing is enabled
        log_level: Level of timer output
        log_interval: Steps between timer logs
        synchronize_cuda: Whether to sync CUDA before timing
        use_barrier: Whether to use distributed barrier
        track_memory: Whether to track memory usage
        aggregation_method: How to aggregate across ranks
        precision: Decimal places for output
        warmup_steps: Steps to skip for warmup
        enabled_timers: List of timer names to enable (None = all)
        disabled_timers: List of timer names to disable
        output_file: Optional file to write timing results
        profile_compute: Enable compute profiling
        profile_communication: Enable communication profiling
    """

    enabled: bool = True
    log_level: TimerLogLevel = TimerLogLevel.INTERVAL
    log_interval: int = 100
    synchronize_cuda: bool = True
    use_barrier: bool = False
    track_memory: bool = False
    aggregation_method: TimerAggregation = TimerAggregation.MEAN
    precision: int = 3
    warmup_steps: int = 5
    enabled_timers: Optional[List[str]] = None
    disabled_timers: List[str] = field(default_factory=list)
    output_file: Optional[str] = None
    profile_compute: bool = True
    profile_communication: bool = True

    # Performance optimization settings
    max_history: int = 10000  # Maximum history to keep
    batch_size: int = 1000  # Batch size for statistics

    # Timer categories for organized output
    timer_categories: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "forward": ["forward-compute", "forward-comm"],
            "backward": ["backward-compute", "backward-comm"],
            "optimizer": ["optimizer-step", "gradient-clip"],
            "data": ["data-loader", "data-transfer"],
            "checkpoint": ["save-checkpoint", "load-checkpoint"],
            "misc": [],
        }
    )

    def should_log(self, step: int) -> bool:
        """Check if should log at this step."""
        if self.log_level == TimerLogLevel.OFF:
            return False
        if self.log_level == TimerLogLevel.SUMMARY:
            return False
        if step <= self.warmup_steps:
            return False
        return step % self.log_interval == 0

    def is_timer_enabled(self, name: str) -> bool:
        """Check if a specific timer is enabled."""
        if not self.enabled:
            return False
        if name in self.disabled_timers:
            return False
        if self.enabled_timers is not None:
            return name in self.enabled_timers
        return True

    def get_category(self, timer_name: str) -> str:
        """Get category for a timer."""
        for category, timers in self.timer_categories.items():
            if timer_name in timers:
                return category
        return "misc"
