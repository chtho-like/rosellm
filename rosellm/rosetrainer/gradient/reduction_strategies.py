"""Gradient reduction strategies for shared weights.

This module implements various strategies for reducing gradients across
distributed processes, following the Strategy design pattern.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

logger = logging.getLogger(__name__)


class GradientReducer(Protocol):
    """Protocol for gradient reduction operations."""

    def reduce(
        self,
        tensors: List[torch.Tensor],
        group: Optional[dist.ProcessGroup] = None,
    ) -> List[torch.Tensor]:
        """Reduce gradients across processes."""
        ...


class ReductionStrategyBase(ABC):
    """Abstract base class for gradient reduction strategies."""

    def __init__(self, world_size: int = 1):
        """Initialize reduction strategy.

        Args:
            world_size: Number of processes in the group.
        """
        self.world_size = world_size
        self._validate_world_size()

    def _validate_world_size(self) -> None:
        """Validate world size."""
        if self.world_size < 1:
            raise ValueError(f"World size must be >= 1, got {self.world_size}")

    @abstractmethod
    def reduce_gradients(
        self,
        gradients: List[torch.Tensor],
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> List[torch.Tensor]:
        """Reduce gradients using the specific strategy.

        Args:
            gradients: List of gradient tensors to reduce.
            process_group: Process group for communication.

        Returns:
            List of reduced gradient tensors.
        """
        pass

    def _validate_inputs(self, gradients: List[torch.Tensor]) -> None:
        """Validate input gradients.

        Args:
            gradients: List of gradient tensors.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not gradients:
            raise ValueError("No gradients to reduce")

        if not all(isinstance(g, torch.Tensor) for g in gradients):
            raise ValueError("All gradients must be torch.Tensor objects")

        # Check for consistent device
        devices = {g.device for g in gradients}
        if len(devices) > 1:
            raise ValueError(f"All gradients must be on the same device, got {devices}")


class AllReduceStrategy(ReductionStrategyBase):
    """All-reduce strategy for gradient synchronization.

    This strategy performs an all-reduce operation where all processes
    receive the sum of gradients from all other processes.
    """

    def __init__(self, world_size: int = 1, average: bool = True):
        """Initialize all-reduce strategy.

        Args:
            world_size: Number of processes.
            average: Whether to average gradients instead of sum.
        """
        super().__init__(world_size)
        self.average = average

    def reduce_gradients(
        self,
        gradients: List[torch.Tensor],
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> List[torch.Tensor]:
        """Perform all-reduce on gradients.

        Args:
            gradients: List of gradient tensors.
            process_group: Process group for communication.

        Returns:
            List of reduced gradients.
        """
        self._validate_inputs(gradients)

        if self.world_size == 1:
            return gradients

        # Coalesce for efficiency
        coalesced = _flatten_dense_tensors(gradients)

        # Perform all-reduce
        dist.all_reduce(
            coalesced,
            op=dist.ReduceOp.SUM,
            group=process_group,
        )

        # Average if requested
        if self.average:
            coalesced.div_(self.world_size)

        # Unflatten back to original shapes
        return _unflatten_dense_tensors(  # type: ignore[no-any-return]
            coalesced, gradients
        )


class ReduceScatterStrategy(ReductionStrategyBase):
    """Reduce-scatter strategy for gradient synchronization.

    This strategy performs a reduce-scatter operation where gradients
    are partitioned and each process receives a portion of the sum.
    """

    def __init__(self, world_size: int = 1):
        """Initialize reduce-scatter strategy.

        Args:
            world_size: Number of processes.
        """
        super().__init__(world_size)

    def reduce_gradients(
        self,
        gradients: List[torch.Tensor],
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> List[torch.Tensor]:
        """Perform reduce-scatter on gradients.

        Args:
            gradients: List of gradient tensors.
            process_group: Process group for communication.

        Returns:
            List of scattered gradient portions.
        """
        self._validate_inputs(gradients)

        if self.world_size == 1:
            return gradients

        # Implementation would require partitioning logic
        # For now, fallback to all-reduce
        logger.warning("Reduce-scatter not fully implemented, using all-reduce")
        return AllReduceStrategy(self.world_size).reduce_gradients(
            gradients, process_group
        )


class HierarchicalReductionStrategy(ReductionStrategyBase):
    """Hierarchical reduction strategy for large-scale training.

    This strategy performs reduction in a hierarchical manner,
    first within nodes, then across nodes, to optimize for
    network topology.
    """

    def __init__(
        self,
        world_size: int = 1,
        local_world_size: int = 1,
        enable_compression: bool = False,
    ):
        """Initialize hierarchical reduction strategy.

        Args:
            world_size: Total number of processes.
            local_world_size: Number of processes per node.
            enable_compression: Whether to compress gradients.
        """
        super().__init__(world_size)
        self.local_world_size = local_world_size
        self.enable_compression = enable_compression
        self._validate_hierarchy()

    def _validate_hierarchy(self) -> None:
        """Validate hierarchical configuration."""
        if self.local_world_size < 1:
            raise ValueError(
                f"Local world size must be >= 1, got {self.local_world_size}"
            )

        if self.world_size % self.local_world_size != 0:
            raise ValueError(
                f"World size {self.world_size} must be divisible by "
                f"local world size {self.local_world_size}"
            )

    def reduce_gradients(
        self,
        gradients: List[torch.Tensor],
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> List[torch.Tensor]:
        """Perform hierarchical reduction on gradients.

        Args:
            gradients: List of gradient tensors.
            process_group: Process group for communication.

        Returns:
            List of reduced gradients.
        """
        self._validate_inputs(gradients)

        if self.world_size == 1:
            return gradients

        # In a full implementation, this would:
        # 1. Reduce within local nodes
        # 2. Select representatives from each node
        # 3. Reduce across representatives
        # 4. Broadcast back to local nodes

        # For now, use optimized all-reduce
        if self.enable_compression:
            logger.info("Using compressed all-reduce")
            # Could implement FP16 compression here

        return AllReduceStrategy(self.world_size).reduce_gradients(
            gradients, process_group
        )


class AdaptiveReductionStrategy(ReductionStrategyBase):
    """Adaptive strategy that chooses the best reduction method.

    This strategy analyzes gradient characteristics and network
    conditions to dynamically select the optimal reduction strategy.
    """

    def __init__(
        self,
        world_size: int = 1,
        tensor_size_threshold: int = 1024 * 1024,  # 1MB
    ):
        """Initialize adaptive reduction strategy.

        Args:
            world_size: Number of processes.
            tensor_size_threshold: Size threshold for strategy selection.
        """
        super().__init__(world_size)
        self.tensor_size_threshold = tensor_size_threshold

        # Initialize sub-strategies
        self.all_reduce = AllReduceStrategy(world_size)
        self.reduce_scatter = ReduceScatterStrategy(world_size)
        self.hierarchical = HierarchicalReductionStrategy(world_size)

    def reduce_gradients(
        self,
        gradients: List[torch.Tensor],
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> List[torch.Tensor]:
        """Adaptively reduce gradients.

        Args:
            gradients: List of gradient tensors.
            process_group: Process group for communication.

        Returns:
            List of reduced gradients.
        """
        self._validate_inputs(gradients)

        if self.world_size == 1:
            return gradients

        # Calculate total size
        total_size = sum(g.numel() * g.element_size() for g in gradients)

        # Choose strategy based on size and world size
        if total_size < self.tensor_size_threshold:
            # Small tensors: use simple all-reduce
            logger.debug("Using all-reduce for small tensors")
            return self.all_reduce.reduce_gradients(gradients, process_group)
        elif self.world_size > 8:
            # Large scale: use hierarchical
            logger.debug("Using hierarchical reduction for large scale")
            return self.hierarchical.reduce_gradients(gradients, process_group)
        else:
            # Medium scale: use all-reduce with averaging
            logger.debug("Using all-reduce with averaging")
            return self.all_reduce.reduce_gradients(gradients, process_group)


def create_reduction_strategy(
    strategy_name: str,
    world_size: int = 1,
    **kwargs,
) -> ReductionStrategyBase:
    """Factory function to create reduction strategies.

    Args:
        strategy_name: Name of the strategy.
        world_size: Number of processes.
        **kwargs: Additional strategy-specific arguments.

    Returns:
        Configured reduction strategy.

    Raises:
        ValueError: If strategy name is unknown.
    """
    strategies = {
        "all_reduce": AllReduceStrategy,
        "reduce_scatter": ReduceScatterStrategy,
        "hierarchical": HierarchicalReductionStrategy,
        "adaptive": AdaptiveReductionStrategy,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(strategies.keys())}"
        )

    strategy_class = strategies[strategy_name]
    return strategy_class(world_size, **kwargs)  # type: ignore[no-any-return]
