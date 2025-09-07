"""
Parameter Partitioning Strategies for Distributed Optimizer.

This module implements the Strategy pattern for different parameter
partitioning approaches across distributed ranks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from torch.nn import Parameter


class PartitioningStrategy(ABC):
    """Abstract base class for parameter partitioning strategies."""

    @abstractmethod
    def partition(
        self, params: List[Parameter], world_size: int
    ) -> Dict[int, List[Parameter]]:
        """
        Partition parameters across ranks.

        Args:
            params: List of model parameters to partition
            world_size: Number of ranks to partition across

        Returns:
            Dictionary mapping rank to assigned parameters
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this partitioning strategy."""
        pass


class RoundRobinPartitioning(PartitioningStrategy):
    """Round-robin parameter partitioning for balanced distribution."""

    def partition(
        self, params: List[Parameter], world_size: int
    ) -> Dict[int, List[Parameter]]:
        """Partition parameters using round-robin assignment."""
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not params:
            return {i: [] for i in range(world_size)}

        rank_to_params: Dict[int, List[Parameter]] = {i: [] for i in range(world_size)}

        for idx, param in enumerate(params):
            assigned_rank = idx % world_size
            rank_to_params[assigned_rank].append(param)

        return rank_to_params

    def get_name(self) -> str:
        return "round_robin"


class SizeBalancedPartitioning(PartitioningStrategy):
    """Size-based parameter partitioning for memory-balanced distribution."""

    def partition(
        self, params: List[Parameter], world_size: int
    ) -> Dict[int, List[Parameter]]:
        """Partition parameters to balance memory usage across ranks."""
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not params:
            return {i: [] for i in range(world_size)}

        # Sort parameters by size (largest first)
        sorted_params = sorted(params, key=lambda p: p.numel(), reverse=True)

        # Track total size per rank
        rank_sizes = [0] * world_size
        rank_to_params: Dict[int, List[Parameter]] = {i: [] for i in range(world_size)}

        # Greedily assign each parameter to rank with smallest current size
        for param in sorted_params:
            # Find rank with minimum size
            min_rank = min(range(world_size), key=lambda r: rank_sizes[r])

            # Assign parameter to this rank
            rank_to_params[min_rank].append(param)
            rank_sizes[min_rank] += param.numel()

        return rank_to_params

    def get_name(self) -> str:
        return "size_balanced"


class LayerWisePartitioning(PartitioningStrategy):
    """Layer-wise parameter partitioning for model parallelism."""

    def partition(
        self, params: List[Parameter], world_size: int
    ) -> Dict[int, List[Parameter]]:
        """Partition parameters by grouping consecutive parameters (layers)."""
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not params:
            return {i: [] for i in range(world_size)}

        rank_to_params: Dict[int, List[Parameter]] = {i: [] for i in range(world_size)}

        # Calculate parameters per rank
        params_per_rank = len(params) // world_size
        remainder = len(params) % world_size

        start_idx = 0
        for rank in range(world_size):
            # Distribute remainder across first ranks
            end_idx = start_idx + params_per_rank + (1 if rank < remainder else 0)
            rank_to_params[rank] = params[start_idx:end_idx]
            start_idx = end_idx

        return rank_to_params

    def get_name(self) -> str:
        return "layer_wise"


class PartitioningStrategyFactory:
    """Factory for creating partitioning strategies."""

    _strategies: Dict[str, PartitioningStrategy] = {
        "round_robin": RoundRobinPartitioning(),
        "size_balanced": SizeBalancedPartitioning(),
        "layer_wise": LayerWisePartitioning(),
    }

    @classmethod
    def create(cls, strategy_name: str) -> PartitioningStrategy:
        """
        Create a partitioning strategy by name.

        Args:
            strategy_name: Name of the strategy
                ("round_robin", "size_balanced", "layer_wise")

        Returns:
            PartitioningStrategy instance

        Raises:
            ValueError: If strategy_name is not recognized
        """
        if strategy_name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown partitioning strategy: {strategy_name}. "
                f"Available strategies: {available}"
            )
        return cls._strategies[strategy_name]

    @classmethod
    def register(cls, name: str, strategy: PartitioningStrategy) -> None:
        """Register a custom partitioning strategy."""
        cls._strategies[name] = strategy

    @classmethod
    def list_strategies(cls) -> List[str]:
        """Get list of available strategy names."""
        return list(cls._strategies.keys())
