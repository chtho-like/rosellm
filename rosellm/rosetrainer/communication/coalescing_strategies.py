"""
Coalescing Strategies for Gradient Communication

This module implements various strategies for gradient bucket coalescing,
allowing flexible optimization based on workload characteristics.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List

from rosellm.rosetrainer.optimizer.gradient_buffer import Bucket

logger = logging.getLogger(__name__)


class CoalescingStrategyType(Enum):
    """Types of coalescing strategies available."""

    SIZE_BASED = "size_based"
    LATENCY_BASED = "latency_based"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class CoalescingStrategy(ABC):
    """
    Abstract base class for coalescing strategies.

    Different strategies can be used to determine when and how to coalesce
    gradient buckets based on various criteria.
    """

    @abstractmethod
    def should_coalesce(
        self,
        buckets: List[Bucket],
        current_size_bytes: int,
        elapsed_time_ms: float,
    ) -> bool:
        """
        Determine if buckets should be coalesced now.

        Args:
            buckets: List of pending buckets
            current_size_bytes: Total size of pending buckets
            elapsed_time_ms: Time since coalescing started

        Returns:
            True if buckets should be coalesced now
        """
        pass

    @abstractmethod
    def group_buckets(
        self,
        buckets: List[Bucket],
        max_group_size_bytes: int,
    ) -> List[List[Bucket]]:
        """
        Group buckets for coalescing.

        Args:
            buckets: List of buckets to group
            max_group_size_bytes: Maximum size per group

        Returns:
            List of bucket groups
        """
        pass

    @abstractmethod
    def update_metrics(self, performance_data: Dict[str, float]):
        """
        Update strategy based on performance metrics.

        Args:
            performance_data: Performance metrics from recent operations
        """
        pass


class SizeBasedStrategy(CoalescingStrategy):
    """
    Coalescing strategy based on accumulated buffer size.

    This strategy coalesces buckets when the accumulated size reaches
    a threshold, optimizing for bandwidth utilization.
    """

    def __init__(
        self,
        target_size_mb: float = 50.0,
        min_buckets: int = 2,
    ):
        self.target_size_bytes = int(target_size_mb * 1024 * 1024)
        self.min_buckets = min_buckets
        self.performance_history: List[Dict[str, float]] = []

    def should_coalesce(
        self,
        buckets: List[Bucket],
        current_size_bytes: int,
        elapsed_time_ms: float,
    ) -> bool:
        """Coalesce when size threshold is reached."""
        if len(buckets) < self.min_buckets:
            return False

        return current_size_bytes >= self.target_size_bytes

    def group_buckets(
        self,
        buckets: List[Bucket],
        max_group_size_bytes: int,
    ) -> List[List[Bucket]]:
        """Group buckets by size, keeping groups under max size."""
        groups = []
        current_group: List[Bucket] = []
        current_size = 0

        # Sort buckets by size for better packing
        sorted_buckets = sorted(
            buckets,
            key=lambda b: (
                b.grad_buffer.numel() * b.grad_buffer.element_size()
                if b.grad_buffer is not None
                else 0
            ),
            reverse=True,
        )

        for bucket in sorted_buckets:
            bucket_size = (
                bucket.grad_buffer.numel() * bucket.grad_buffer.element_size()
                if bucket.grad_buffer is not None
                else 0
            )

            if current_group and current_size + bucket_size > max_group_size_bytes:
                groups.append(current_group)
                current_group = []
                current_size = 0

            current_group.append(bucket)
            current_size += bucket_size

        if current_group:
            groups.append(current_group)

        return groups

    def update_metrics(self, performance_data: Dict[str, float]):
        """Update target size based on performance."""
        self.performance_history.append(performance_data)

        # Adjust target size based on throughput
        if len(self.performance_history) >= 5:
            recent = self.performance_history[-5:]
            avg_throughput = sum(d.get("throughput_gbps", 0) for d in recent) / len(
                recent
            )

            # If throughput is good, try larger sizes
            if performance_data.get("throughput_gbps", 0) > avg_throughput * 1.1:
                self.target_size_bytes = int(
                    min(self.target_size_bytes * 1.2, 200 * 1024 * 1024)  # Max 200MB
                )
                logger.debug(
                    f"Increased target size to " f"{self.target_size_bytes / 1e6:.1f}MB"
                )


class LatencyBasedStrategy(CoalescingStrategy):
    """
    Coalescing strategy based on latency constraints.

    This strategy ensures coalescing doesn't introduce excessive latency,
    suitable for latency-sensitive workloads.
    """

    def __init__(
        self,
        max_latency_ms: float = 10.0,
        min_buckets: int = 2,
    ):
        self.max_latency_ms = max_latency_ms
        self.min_buckets = min_buckets
        self.avg_processing_time_ms = 1.0  # Initial estimate

    def should_coalesce(
        self,
        buckets: List[Bucket],
        current_size_bytes: int,
        elapsed_time_ms: float,
    ) -> bool:
        """Coalesce when approaching latency limit."""
        if len(buckets) < self.min_buckets:
            return False

        # Estimate time for current buckets
        estimated_time = len(buckets) * self.avg_processing_time_ms

        # Coalesce if we're close to latency limit or have enough buckets
        return (
            elapsed_time_ms + estimated_time >= self.max_latency_ms * 0.8
            or len(buckets) >= self.min_buckets * 2
        )

    def group_buckets(
        self,
        buckets: List[Bucket],
        max_group_size_bytes: int,
    ) -> List[List[Bucket]]:
        """Group buckets to minimize latency."""
        groups = []

        # Estimate buckets per group based on latency
        buckets_per_group = max(
            self.min_buckets, int(self.max_latency_ms / self.avg_processing_time_ms)
        )

        for i in range(0, len(buckets), buckets_per_group):
            group = buckets[i : i + buckets_per_group]

            # Check size constraint
            group_size = sum(
                (
                    b.grad_buffer.numel() * b.grad_buffer.element_size()
                    if b.grad_buffer is not None
                    else 0
                )
                for b in group
            )

            if group_size <= max_group_size_bytes:
                groups.append(group)
            else:
                # Split large group
                subgroup: List[Bucket] = []
                subgroup_size = 0
                for bucket in group:
                    bucket_size = (
                        bucket.grad_buffer.numel() * bucket.grad_buffer.element_size()
                        if bucket.grad_buffer is not None
                        else 0
                    )
                    if subgroup and subgroup_size + bucket_size > max_group_size_bytes:
                        groups.append(subgroup)
                        subgroup = []
                        subgroup_size = 0
                    subgroup.append(bucket)
                    subgroup_size += bucket_size
                if subgroup:
                    groups.append(subgroup)

        return groups

    def update_metrics(self, performance_data: Dict[str, float]):
        """Update processing time estimate."""
        if "time_ms" in performance_data and "num_ops" in performance_data:
            num_ops = performance_data["num_ops"]
            if num_ops > 0:
                time_per_op = performance_data["time_ms"] / num_ops
                # Exponential moving average
                self.avg_processing_time_ms = (
                    0.7 * self.avg_processing_time_ms + 0.3 * time_per_op
                )


class HybridStrategy(CoalescingStrategy):
    """
    Hybrid strategy combining size and latency considerations.

    This strategy balances between bandwidth utilization and latency,
    suitable for most general workloads.
    """

    def __init__(
        self,
        target_size_mb: float = 50.0,
        max_latency_ms: float = 10.0,
        min_buckets: int = 2,
    ):
        self.size_strategy = SizeBasedStrategy(target_size_mb, min_buckets)
        self.latency_strategy = LatencyBasedStrategy(max_latency_ms, min_buckets)
        self.size_weight = 0.5
        self.latency_weight = 0.5

    def should_coalesce(
        self,
        buckets: List[Bucket],
        current_size_bytes: int,
        elapsed_time_ms: float,
    ) -> bool:
        """Coalesce based on both size and latency."""
        # Coalesce if either strategy strongly suggests it
        size_vote = self.size_strategy.should_coalesce(
            buckets, current_size_bytes, elapsed_time_ms
        )
        latency_vote = self.latency_strategy.should_coalesce(
            buckets, current_size_bytes, elapsed_time_ms
        )

        # Weighted decision
        if self.size_weight > 0.7:
            return size_vote
        elif self.latency_weight > 0.7:
            return latency_vote
        else:
            return size_vote or latency_vote

    def group_buckets(
        self,
        buckets: List[Bucket],
        max_group_size_bytes: int,
    ) -> List[List[Bucket]]:
        """Group buckets using the dominant strategy."""
        if self.size_weight > self.latency_weight:
            return self.size_strategy.group_buckets(buckets, max_group_size_bytes)
        else:
            return self.latency_strategy.group_buckets(buckets, max_group_size_bytes)

    def update_metrics(self, performance_data: Dict[str, float]):
        """Update both strategies and adjust weights."""
        self.size_strategy.update_metrics(performance_data)
        self.latency_strategy.update_metrics(performance_data)

        # Adjust weights based on performance
        if "throughput_gbps" in performance_data and "time_ms" in performance_data:
            # If throughput is low, favor size-based
            if performance_data["throughput_gbps"] < 10.0:  # Example threshold
                self.size_weight = min(0.8, self.size_weight + 0.05)
                self.latency_weight = 1.0 - self.size_weight
            # If latency is high, favor latency-based
            elif performance_data["time_ms"] > 15.0:  # Example threshold
                self.latency_weight = min(0.8, self.latency_weight + 0.05)
                self.size_weight = 1.0 - self.latency_weight


class AdaptiveStrategy(CoalescingStrategy):
    """
    Adaptive strategy that automatically selects the best approach.

    This strategy monitors performance and automatically switches between
    different strategies based on workload characteristics.
    """

    def __init__(self) -> None:
        self.strategies = {
            CoalescingStrategyType.SIZE_BASED: SizeBasedStrategy(),
            CoalescingStrategyType.LATENCY_BASED: LatencyBasedStrategy(),
            CoalescingStrategyType.HYBRID: HybridStrategy(),
        }
        self.current_strategy = CoalescingStrategyType.HYBRID
        self.strategy_scores: Dict[CoalescingStrategyType, float] = {
            s: 0.0 for s in self.strategies
        }
        self.exploration_rate = 0.1
        self.iteration = 0

    def should_coalesce(
        self,
        buckets: List[Bucket],
        current_size_bytes: int,
        elapsed_time_ms: float,
    ) -> bool:
        """Use current strategy to decide."""
        strategy = self.strategies[self.current_strategy]
        return strategy.should_coalesce(  # type: ignore[no-any-return]
            buckets, current_size_bytes, elapsed_time_ms
        )

    def group_buckets(
        self,
        buckets: List[Bucket],
        max_group_size_bytes: int,
    ) -> List[List[Bucket]]:
        """Use current strategy to group buckets."""
        strategy = self.strategies[self.current_strategy]
        return strategy.group_buckets(  # type: ignore[no-any-return]
            buckets, max_group_size_bytes
        )

    def update_metrics(self, performance_data: Dict[str, float]):
        """Update strategy selection based on performance."""
        # Update current strategy
        self.strategies[self.current_strategy].update_metrics(performance_data)

        # Calculate score for current strategy
        score = self._calculate_score(performance_data)
        self.strategy_scores[self.current_strategy] = (
            0.9 * self.strategy_scores[self.current_strategy] + 0.1 * score
        )

        self.iteration += 1

        # Periodically explore or exploit
        if self.iteration % 10 == 0:
            import random

            if random.random() < self.exploration_rate:
                # Explore: try a different strategy
                self.current_strategy = random.choice(list(self.strategies.keys()))
                logger.debug(f"Exploring strategy: {self.current_strategy.value}")
            else:
                # Exploit: use best performing strategy
                best_strategy = max(self.strategy_scores.items(), key=lambda x: x[1])[0]
                if best_strategy != self.current_strategy:
                    self.current_strategy = best_strategy
                    logger.info(
                        f"Switched to best strategy: {self.current_strategy.value}"
                    )

            # Decay exploration rate
            self.exploration_rate *= 0.95
            self.exploration_rate = max(0.01, self.exploration_rate)

    def _calculate_score(self, performance_data: Dict[str, float]) -> float:
        """Calculate performance score for strategy evaluation."""
        score = 0.0

        # High throughput is good
        if "throughput_gbps" in performance_data:
            score += performance_data["throughput_gbps"] * 10

        # Low latency is good
        if "time_ms" in performance_data:
            score += max(0, 20 - performance_data["time_ms"])

        # Many operations coalesced is good
        if "num_ops" in performance_data:
            score += performance_data["num_ops"]

        return score


def create_strategy(
    strategy_type: CoalescingStrategyType, **kwargs
) -> CoalescingStrategy:
    """
    Factory function to create coalescing strategies.

    Args:
        strategy_type: Type of strategy to create
        **kwargs: Strategy-specific parameters

    Returns:
        Configured coalescing strategy
    """
    if strategy_type == CoalescingStrategyType.SIZE_BASED:
        return SizeBasedStrategy(
            target_size_mb=kwargs.get("target_size_mb", 50.0),
            min_buckets=kwargs.get("min_buckets", 2),
        )
    elif strategy_type == CoalescingStrategyType.LATENCY_BASED:
        return LatencyBasedStrategy(
            max_latency_ms=kwargs.get("max_latency_ms", 10.0),
            min_buckets=kwargs.get("min_buckets", 2),
        )
    elif strategy_type == CoalescingStrategyType.HYBRID:
        return HybridStrategy(
            target_size_mb=kwargs.get("target_size_mb", 50.0),
            max_latency_ms=kwargs.get("max_latency_ms", 10.0),
            min_buckets=kwargs.get("min_buckets", 2),
        )
    elif strategy_type == CoalescingStrategyType.ADAPTIVE:
        return AdaptiveStrategy()
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
