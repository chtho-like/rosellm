"""
Bucket Group Management for Advanced Gradient Communication

Implements multi-bucket coordination and advanced grouping strategies
for efficient distributed gradient communication in RoseTrainer.

This module extends the basic bucketing functionality by providing
hierarchical bucket organization and coordinated communication patterns.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch.distributed as dist

from .gradient_buckets import BucketManager, GradientBucket

logger = logging.getLogger(__name__)


class GroupStrategy(Enum):
    """Strategy for organizing buckets into groups."""

    PARALLEL = "parallel"  # All buckets communicate simultaneously
    SEQUENTIAL = "sequential"  # Buckets communicate one after another
    HIERARCHICAL = "hierarchical"  # Multi-level communication tree
    ADAPTIVE = "adaptive"  # Dynamically choose based on conditions


class PriorityLevel(Enum):
    """Priority levels for bucket groups."""

    CRITICAL = 0  # Highest priority - communicate first
    HIGH = 1  # High priority
    NORMAL = 2  # Normal priority
    LOW = 3  # Low priority - can be delayed
    BACKGROUND = 4  # Lowest priority - communicate when resources available


@dataclass
class BucketGroupConfig:
    """Configuration for bucket group management."""

    # Group organization
    group_strategy: GroupStrategy = GroupStrategy.ADAPTIVE
    max_groups: int = 8  # Maximum number of groups
    min_buckets_per_group: int = 1
    max_buckets_per_group: int = 10

    # Priority management
    enable_prioritization: bool = True
    priority_threshold_mb: float = 10.0  # Size threshold for high priority

    # Communication optimization
    overlap_groups: bool = True
    pipeline_communication: bool = True
    adaptive_batch_size: bool = True

    # Performance tuning
    max_concurrent_groups: int = 4
    group_timeout_ms: int = 60000  # 1 minute
    load_balancing: bool = True

    # Advanced features
    compression_per_group: bool = False
    gradient_accumulation_groups: bool = False
    cross_group_optimization: bool = True


class BucketGroup:
    """
    Manages a group of buckets for coordinated communication.

    This class provides hierarchical organization of gradient buckets,
    allowing for optimized communication patterns and resource management.
    """

    def __init__(
        self,
        group_id: int,
        priority: PriorityLevel = PriorityLevel.NORMAL,
        max_buckets: int = 10,
    ):
        """
        Initialize a bucket group.

        Args:
            group_id: Unique identifier for this group
            priority: Priority level for communication scheduling
            max_buckets: Maximum number of buckets in this group
        """
        self.group_id = group_id
        self.priority = priority
        self.max_buckets = max_buckets

        # Bucket management
        self.buckets: List[GradientBucket] = []
        self.bucket_ids: Set[int] = set()

        # Communication state
        self.is_communicating = False
        self.communication_handles: List[dist.Work] = []
        self.last_communication_time: Optional[float] = None

        # Performance tracking
        self.communication_times: List[float] = []
        self.total_size_communicated = 0
        self.successful_communications = 0
        self.failed_communications = 0

        # Group metadata
        self.creation_time = time.time()
        self.last_optimization_time = 0.0

    def can_add_bucket(self, bucket: GradientBucket) -> bool:
        """Check if a bucket can be added to this group."""
        return (
            len(self.buckets) < self.max_buckets
            and bucket.bucket_id not in self.bucket_ids
        )

    def add_bucket(self, bucket: GradientBucket) -> bool:
        """
        Add a bucket to this group.

        Args:
            bucket: Bucket to add

        Returns:
            True if bucket was successfully added
        """
        if not self.can_add_bucket(bucket):
            return False

        self.buckets.append(bucket)
        self.bucket_ids.add(bucket.bucket_id)

        logger.debug(
            f"Added bucket {bucket.bucket_id} to group {self.group_id} "
            f"({len(self.buckets)}/{self.max_buckets} buckets)"
        )

        return True

    def remove_bucket(self, bucket_id: int) -> bool:
        """Remove a bucket from this group."""
        for i, bucket in enumerate(self.buckets):
            if bucket.bucket_id == bucket_id:
                self.buckets.pop(i)
                self.bucket_ids.discard(bucket_id)
                logger.debug(f"Removed bucket {bucket_id} from group {self.group_id}")
                return True
        return False

    def get_total_size(self) -> int:
        """Get total size of all buckets in this group."""
        return sum(bucket.current_size_bytes for bucket in self.buckets)

    def get_total_gradients(self) -> int:
        """Get total number of gradients in this group."""
        return sum(len(bucket.gradients) for bucket in self.buckets)

    def start_communication(
        self,
        process_group: Optional[dist.ProcessGroup] = None,
        overlap: bool = True,
    ) -> List[dist.Work]:
        """
        Start asynchronous communication for all buckets in this group.

        Args:
            process_group: Distributed process group
            overlap: Whether to overlap communication within the group

        Returns:
            List of communication work handles
        """
        if self.is_communicating:
            logger.warning(f"Group {self.group_id} is already communicating")
            return []

        self.is_communicating = True
        self.last_communication_time = time.time()
        self.communication_handles.clear()

        # Start communication for all buckets
        for bucket in self.buckets:
            if bucket.gradients:  # Only communicate buckets with gradients
                handle = bucket.start_communication(
                    process_group=process_group,
                    predivide=True,  # Always predivide for group communication
                )
                if handle is not None:
                    self.communication_handles.append(handle)

        logger.debug(
            f"Started communication for group {self.group_id} "
            f"with {len(self.communication_handles)} active handles"
        )

        return self.communication_handles

    def wait_communication(self) -> float:
        """
        Wait for all communications in this group to complete.

        Returns:
            Total communication time
        """
        if not self.is_communicating:
            return 0.0

        total_time = 0.0

        # Wait for all bucket communications
        for i, bucket in enumerate(self.buckets):
            if bucket.communication_handle is not None:
                comm_time = bucket.wait_communication()
                total_time = max(total_time, comm_time)  # Use maximum time

        # Update group statistics
        if self.last_communication_time is not None:
            group_time = time.time() - self.last_communication_time
            self.communication_times.append(group_time)
            self.total_size_communicated += self.get_total_size()
            self.successful_communications += 1

        self.is_communicating = False
        self.communication_handles.clear()

        return total_time

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for this group."""
        total_size_mb = self.get_total_size() / (1024 * 1024)
        avg_comm_time = (
            sum(self.communication_times) / len(self.communication_times)
            if self.communication_times
            else 0.0
        )

        return {
            "group_id": self.group_id,
            "priority": self.priority.name,
            "num_buckets": len(self.buckets),
            "total_gradients": self.get_total_gradients(),
            "total_size_mb": total_size_mb,
            "utilization": len(self.buckets) / self.max_buckets,
            "avg_communication_time": avg_comm_time,
            "successful_communications": self.successful_communications,
            "failed_communications": self.failed_communications,
            "throughput_mbps": (
                (self.total_size_communicated / (1024 * 1024)) / avg_comm_time
                if avg_comm_time > 0
                else 0.0
            ),
        }

    def optimize(self) -> Dict[str, Any]:
        """
        Optimize bucket arrangement within this group.

        Returns:
            Optimization statistics and actions taken
        """
        optimization_start = time.time()
        actions_taken = []

        # Sort buckets by communication time (if available)
        if all(bucket.communication_times for bucket in self.buckets):
            # Sort by average communication time (ascending)
            self.buckets.sort(
                key=lambda b: sum(b.communication_times) / len(b.communication_times)
            )
            actions_taken.append("sorted_by_comm_time")

        # Balance bucket sizes
        total_size = self.get_total_size()
        if total_size > 0:
            target_size = total_size / len(self.buckets)
            imbalanced_buckets = [
                b
                for b in self.buckets
                if abs(b.current_size_bytes - target_size) > target_size * 0.3
            ]
            if imbalanced_buckets:
                actions_taken.append("identified_imbalanced_buckets")

        self.last_optimization_time = time.time()
        optimization_time = self.last_optimization_time - optimization_start

        return {
            "optimization_time": optimization_time,
            "actions_taken": actions_taken,
            "buckets_optimized": len(self.buckets),
        }


class BucketGroupManager:
    """
    Manages multiple bucket groups for hierarchical gradient communication.

    This class coordinates multiple bucket groups, implementing advanced
    communication strategies and resource management.
    """

    def __init__(
        self,
        config: BucketGroupConfig,
        bucket_manager: BucketManager,
    ):
        """
        Initialize the bucket group manager.

        Args:
            config: Group configuration
            bucket_manager: Associated bucket manager
        """
        self.config = config
        self.bucket_manager = bucket_manager

        # Group management
        self.groups: List[BucketGroup] = []
        self.group_by_priority: Dict[PriorityLevel, List[BucketGroup]] = {
            level: [] for level in PriorityLevel
        }
        self.next_group_id = 0

        # Communication coordination
        self.active_groups: Set[int] = set()
        self.communication_queue: List[
            Tuple[PriorityLevel, int]
        ] = []  # (priority, group_id)

        # Performance tracking
        self.total_group_communications = 0
        self.total_group_time = 0.0
        self.communication_history: List[Dict[str, Any]] = []

    def create_group(
        self, priority: PriorityLevel = PriorityLevel.NORMAL
    ) -> BucketGroup:
        """Create a new bucket group."""
        group = BucketGroup(
            group_id=self.next_group_id,
            priority=priority,
            max_buckets=self.config.max_buckets_per_group,
        )

        self.groups.append(group)
        self.group_by_priority[priority].append(group)
        self.next_group_id += 1

        logger.debug(f"Created group {group.group_id} with priority {priority.name}")
        return group

    def assign_buckets_to_groups(self) -> Dict[str, Any]:
        """
        Assign buckets to groups based on the configured strategy.

        Returns:
            Assignment statistics and information
        """
        assignment_start = time.time()

        # Clear existing assignments
        for group in self.groups:
            group.buckets.clear()
            group.bucket_ids.clear()

        buckets = self.bucket_manager.buckets
        if not buckets:
            return {"message": "No buckets to assign"}

        if self.config.group_strategy == GroupStrategy.PARALLEL:
            self._assign_parallel_strategy(buckets)
        elif self.config.group_strategy == GroupStrategy.SEQUENTIAL:
            self._assign_sequential_strategy(buckets)
        elif self.config.group_strategy == GroupStrategy.HIERARCHICAL:
            self._assign_hierarchical_strategy(buckets)
        else:  # ADAPTIVE
            self._assign_adaptive_strategy(buckets)

        assignment_time = time.time() - assignment_start

        return {
            "strategy": self.config.group_strategy.name,
            "assignment_time": assignment_time,
            "num_groups_used": sum(1 for g in self.groups if g.buckets),
            "total_buckets": len(buckets),
            "avg_buckets_per_group": (
                len(buckets) / sum(1 for g in self.groups if g.buckets)
                if any(g.buckets for g in self.groups)
                else 0
            ),
        }

    def _assign_parallel_strategy(self, buckets: List[GradientBucket]) -> None:
        """Assign buckets for parallel communication."""
        # Distribute buckets evenly across available groups
        buckets_per_group = max(1, len(buckets) // max(1, len(self.groups)))

        for i, bucket in enumerate(buckets):
            group_index = i // buckets_per_group

            # Create new group if needed
            while group_index >= len(self.groups):
                priority = self._determine_bucket_priority(bucket)
                self.create_group(priority)

            self.groups[group_index].add_bucket(bucket)

    def _assign_sequential_strategy(self, buckets: List[GradientBucket]) -> None:
        """Assign buckets for sequential communication."""
        # Sort buckets by size (largest first for better pipeline utilization)
        sorted_buckets = sorted(
            buckets, key=lambda b: b.current_size_bytes, reverse=True
        )

        current_group = None
        for bucket in sorted_buckets:
            # Create new group if current is full or doesn't exist
            if current_group is None or not current_group.can_add_bucket(bucket):
                priority = self._determine_bucket_priority(bucket)
                current_group = self.create_group(priority)

            current_group.add_bucket(bucket)

    def _assign_hierarchical_strategy(self, buckets: List[GradientBucket]) -> None:
        """Assign buckets using hierarchical strategy."""
        # Group by priority first
        priority_buckets: Dict[PriorityLevel, List[GradientBucket]] = {
            level: [] for level in PriorityLevel
        }

        for bucket in buckets:
            priority = self._determine_bucket_priority(bucket)
            priority_buckets[priority].append(bucket)

        # Create groups for each priority level
        for priority, bucket_list in priority_buckets.items():
            if bucket_list:
                # Create multiple groups if needed
                buckets_per_group = min(
                    self.config.max_buckets_per_group,
                    max(self.config.min_buckets_per_group, len(bucket_list) // 2),
                )

                for i in range(0, len(bucket_list), buckets_per_group):
                    group = self.create_group(priority)
                    batch = bucket_list[i : i + buckets_per_group]
                    for bucket in batch:
                        group.add_bucket(bucket)

    def _assign_adaptive_strategy(self, buckets: List[GradientBucket]) -> None:
        """Assign buckets using adaptive strategy based on current conditions."""
        # Analyze bucket characteristics
        total_size = sum(bucket.current_size_bytes for bucket in buckets)
        avg_size = total_size / len(buckets) if buckets else 0

        large_buckets = [b for b in buckets if b.current_size_bytes > avg_size * 1.5]
        small_buckets = [b for b in buckets if b.current_size_bytes < avg_size * 0.5]

        # Strategy selection based on bucket distribution
        if len(large_buckets) > len(buckets) * 0.7:
            # Mostly large buckets - use parallel strategy
            self._assign_parallel_strategy(buckets)
        elif len(small_buckets) > len(buckets) * 0.7:
            # Mostly small buckets - group more aggressively
            self._assign_sequential_strategy(buckets)
        else:
            # Mixed sizes - use hierarchical approach
            self._assign_hierarchical_strategy(buckets)

    def _determine_bucket_priority(self, bucket: GradientBucket) -> PriorityLevel:
        """Determine priority level for a bucket."""
        if not self.config.enable_prioritization:
            return PriorityLevel.NORMAL

        size_mb = bucket.current_size_bytes / (1024 * 1024)

        # Size-based prioritization
        if size_mb > self.config.priority_threshold_mb:
            return PriorityLevel.HIGH
        elif size_mb < self.config.priority_threshold_mb * 0.1:
            return PriorityLevel.LOW
        else:
            return PriorityLevel.NORMAL

    def synchronize_groups(
        self,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, Any]:
        """
        Synchronize all bucket groups according to the configuration.

        Args:
            process_group: Distributed process group

        Returns:
            Synchronization statistics
        """
        sync_start = time.time()

        # Filter groups that have buckets with gradients
        active_groups = [
            group
            for group in self.groups
            if group.buckets and any(bucket.gradients for bucket in group.buckets)
        ]

        if not active_groups:
            return {"message": "No active groups to synchronize"}

        # Sort groups by priority
        sorted_groups = sorted(active_groups, key=lambda g: g.priority.value)

        communication_stats = []

        if self.config.overlap_groups:
            # Start all communications simultaneously
            handles = []
            for group in sorted_groups:
                group_handles = group.start_communication(
                    process_group=process_group,
                    overlap=True,
                )
                handles.extend(group_handles)

            # Wait for all to complete
            max_time = 0.0
            for group in sorted_groups:
                comm_time = group.wait_communication()
                max_time = max(max_time, comm_time)
                communication_stats.append(
                    {
                        "group_id": group.group_id,
                        "communication_time": comm_time,
                        "buckets": len(group.buckets),
                    }
                )

        else:
            # Sequential group communication
            for group in sorted_groups:
                group.start_communication(
                    process_group=process_group,
                    overlap=True,
                )
                comm_time = group.wait_communication()
                communication_stats.append(
                    {
                        "group_id": group.group_id,
                        "communication_time": comm_time,
                        "buckets": len(group.buckets),
                    }
                )

        total_time = time.time() - sync_start

        # Update global statistics
        self.total_group_communications += 1
        self.total_group_time += total_time

        sync_result = {
            "total_time": total_time,
            "groups_synchronized": len(sorted_groups),
            "overlap_enabled": self.config.overlap_groups,
            "group_details": communication_stats,
        }

        self.communication_history.append(sync_result)

        return sync_result

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all groups."""
        group_stats = [group.get_statistics() for group in self.groups if group.buckets]

        return {
            "config": {
                "strategy": self.config.group_strategy.name,
                "max_groups": self.config.max_groups,
                "overlap_enabled": self.config.overlap_groups,
                "prioritization_enabled": self.config.enable_prioritization,
            },
            "groups": {
                "total": len(self.groups),
                "active": len(group_stats),
                "by_priority": {
                    level.name: len(groups)
                    for level, groups in self.group_by_priority.items()
                    if groups
                },
            },
            "performance": {
                "total_communications": self.total_group_communications,
                "avg_time_per_communication": (
                    self.total_group_time / self.total_group_communications
                    if self.total_group_communications > 0
                    else 0.0
                ),
            },
            "group_details": group_stats,
        }

    def optimize_groups(self) -> Dict[str, Any]:
        """Optimize all groups and group assignments."""
        optimization_start = time.time()

        optimization_results = []
        for group in self.groups:
            if group.buckets:
                result = group.optimize()
                result["group_id"] = group.group_id
                optimization_results.append(result)

        # Global optimization: rebalance groups if needed
        if self.config.load_balancing:
            rebalance_result = self._rebalance_groups()
            optimization_results.append(rebalance_result)

        total_time = time.time() - optimization_start

        return {
            "total_optimization_time": total_time,
            "groups_optimized": len(optimization_results),
            "optimization_details": optimization_results,
        }

    def _rebalance_groups(self) -> Dict[str, Any]:
        """Rebalance buckets across groups for better load distribution."""
        rebalance_start = time.time()

        # Calculate current load distribution
        group_loads = [
            (group.get_total_size(), group) for group in self.groups if group.buckets
        ]

        if len(group_loads) < 2:
            return {"message": "Not enough groups for rebalancing"}

        group_loads.sort(key=lambda x: x[0])  # Sort by load

        actions_taken = []

        # Move buckets from heavily loaded groups to lightly loaded ones
        heavy_group = group_loads[-1][1]  # Most loaded
        light_group = group_loads[0][1]  # Least loaded

        # Simple rebalancing: move one bucket if there's significant imbalance
        if (
            heavy_group.get_total_size() > light_group.get_total_size() * 2
            and len(heavy_group.buckets) > 1
        ):
            # Find smallest bucket in heavy group
            smallest_bucket = min(
                heavy_group.buckets, key=lambda b: b.current_size_bytes
            )

            if light_group.can_add_bucket(smallest_bucket):
                heavy_group.remove_bucket(smallest_bucket.bucket_id)
                light_group.add_bucket(smallest_bucket)
                actions_taken.append(
                    f"moved_bucket_{smallest_bucket.bucket_id}_from_group_"
                    f"{heavy_group.group_id}_to_{light_group.group_id}"
                )

        rebalance_time = time.time() - rebalance_start

        return {
            "rebalance_time": rebalance_time,
            "actions_taken": actions_taken,
            "groups_involved": len(group_loads),
        }
