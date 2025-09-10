"""
Advanced Checkpoint Strategy Implementations with Factory Pattern

This module provides various checkpoint strategies using the Strategy pattern
with a Factory for creating appropriate strategies based on configuration.
Each strategy implements different approaches to distributed checkpointing
optimized for specific parallel training scenarios.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type

# Import removed - was unused


class CheckpointDecision(Protocol):
    """Protocol for checkpoint decision results."""

    should_checkpoint: bool
    reason: str
    metadata: Dict[str, Any]


@dataclass
class CheckpointContext:
    """Context information for making checkpoint decisions."""

    layer_idx: int
    total_layers: int
    memory_usage_mb: float
    rank: int
    world_size: int
    tp_rank: Optional[int] = None
    pp_rank: Optional[int] = None
    dp_rank: Optional[int] = None
    cp_rank: Optional[int] = None
    ep_rank: Optional[int] = None
    is_attention_layer: bool = False
    is_mlp_layer: bool = False
    is_expert_layer: bool = False
    computational_cost: float = 1.0
    memory_cost: float = 1.0

    @property
    def layer_position(self) -> float:
        """Get normalized layer position [0, 1]."""
        return self.layer_idx / max(self.total_layers - 1, 1)

    @property
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical."""
        return self.memory_usage_mb > 1024  # 1GB threshold

    @property
    def is_computationally_expensive(self) -> bool:
        """Check if layer is computationally expensive."""
        return self.computational_cost > 2.0


class CheckpointStrategy(ABC):
    """Abstract base class for checkpoint strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy with optional configuration."""
        self.config = config or {}
        self._cache: Dict[int, bool] = {}

    @abstractmethod
    def should_checkpoint(self, context: CheckpointContext) -> Tuple[bool, str]:
        """
        Determine if a layer should be checkpointed.

        Args:
            context: Context information for decision making

        Returns:
            Tuple of (should_checkpoint, reason_string)
        """
        pass

    def reset_cache(self) -> None:
        """Reset internal decision cache."""
        self._cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            "strategy_name": self.__class__.__name__,
            "cache_size": len(self._cache),
            "config": self.config,
        }


class CoordinatedStrategy(CheckpointStrategy):
    """All ranks make identical checkpoint decisions."""

    def should_checkpoint(self, context: CheckpointContext) -> Tuple[bool, str]:
        """Coordinated checkpoint decision across all ranks."""
        # Cache key based on layer index only (same for all ranks)
        cache_key = context.layer_idx

        if cache_key in self._cache:
            return self._cache[cache_key], "cached_decision"

        # Checkpoint every N layers
        checkpoint_interval = self.config.get("checkpoint_interval", 3)
        should_checkpoint = (context.layer_idx % checkpoint_interval) == 0

        # Additional criteria
        if context.is_memory_critical:
            should_checkpoint = True
            reason = "memory_critical"
        elif context.is_computationally_expensive:
            should_checkpoint = True
            reason = "computationally_expensive"
        else:
            reason = f"interval_{checkpoint_interval}"

        self._cache[cache_key] = should_checkpoint
        return should_checkpoint, reason


class LoadBalancedStrategy(CheckpointStrategy):
    """Distribute checkpoints across ranks for memory balance."""

    def should_checkpoint(self, context: CheckpointContext) -> Tuple[bool, str]:
        """Load-balanced checkpoint decision."""
        # Different ranks checkpoint different layers
        rank_offset = context.rank % 4
        layer_group = context.layer_idx % 4

        should_checkpoint = layer_group == rank_offset

        # Override for critical layers
        if context.is_memory_critical and context.rank == 0:
            should_checkpoint = True
            reason = "memory_critical_rank0"
        else:
            reason = f"load_balanced_group_{layer_group}"

        return should_checkpoint, reason


class HierarchicalStrategy(CheckpointStrategy):
    """Different strategies for different parallel dimensions."""

    def should_checkpoint(self, context: CheckpointContext) -> Tuple[bool, str]:
        """Hierarchical checkpoint decision based on parallel dimensions."""
        # Tensor parallel ranks always coordinate
        if context.tp_rank is not None and context.tp_rank == 0:
            # TP rank 0 makes decision for all TP ranks
            should_checkpoint = (context.layer_idx % 2) == 0
            reason = "tp_coordinator"
        elif context.pp_rank is not None:
            # Pipeline parallel uses stage-specific strategy
            stages_to_checkpoint = self.config.get("pipeline_stages", [0, -1])
            should_checkpoint = context.pp_rank in stages_to_checkpoint
            reason = f"pipeline_stage_{context.pp_rank}"
        else:
            # Default strategy
            should_checkpoint = context.layer_idx < (context.total_layers // 2)
            reason = "hierarchical_default"

        return should_checkpoint, reason


class AdaptiveStrategy(CheckpointStrategy):
    """Dynamically adapt strategy based on runtime conditions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive strategy with performance tracking."""
        super().__init__(config)
        self.performance_history: List[float] = []
        self.memory_history: List[float] = []
        self.checkpoint_ratio = 0.5  # Start with 50% checkpointing

    def should_checkpoint(self, context: CheckpointContext) -> Tuple[bool, str]:
        """Adaptive checkpoint decision based on runtime metrics."""
        # Update history
        self.memory_history.append(context.memory_usage_mb)
        if len(self.memory_history) > 100:
            self.memory_history.pop(0)

        # Adapt checkpoint ratio based on memory pressure
        if self.memory_history:
            avg_memory = sum(self.memory_history) / len(self.memory_history)
            memory_threshold = self.config.get("memory_threshold_mb", 2048)

            if avg_memory > memory_threshold:
                # Increase checkpointing
                self.checkpoint_ratio = min(0.8, self.checkpoint_ratio + 0.1)
                reason_prefix = "high_memory"
            elif avg_memory < memory_threshold * 0.5:
                # Decrease checkpointing
                self.checkpoint_ratio = max(0.2, self.checkpoint_ratio - 0.1)
                reason_prefix = "low_memory"
            else:
                reason_prefix = "balanced"
        else:
            reason_prefix = "initial"

        # Apply adaptive ratio
        should_checkpoint = context.layer_position <= self.checkpoint_ratio
        reason = f"{reason_prefix}_ratio_{self.checkpoint_ratio:.2f}"

        return should_checkpoint, reason


class ExpertAwareStrategy(CheckpointStrategy):
    """Strategy optimized for Mixture-of-Experts models."""

    def should_checkpoint(self, context: CheckpointContext) -> Tuple[bool, str]:
        """Expert-aware checkpoint decision for MoE models."""
        if context.is_expert_layer:
            # Expert layers have special handling
            if context.ep_rank is not None:
                # Checkpoint based on expert parallel rank
                expert_group_size = self.config.get("expert_group_size", 4)
                should_checkpoint = (context.ep_rank % expert_group_size) == 0
                reason = f"expert_group_{context.ep_rank}"
            else:
                # Default expert checkpointing
                should_checkpoint = True
                reason = "expert_layer_default"
        else:
            # Non-expert layers use standard strategy
            should_checkpoint = context.is_attention_layer
            reason = "attention_layer" if should_checkpoint else "other_layer"

        return should_checkpoint, reason


class PipelineAwareStrategy(CheckpointStrategy):
    """Strategy optimized for pipeline parallelism."""

    def should_checkpoint(self, context: CheckpointContext) -> Tuple[bool, str]:
        """Pipeline-aware checkpoint decision."""
        if context.pp_rank is None:
            # No pipeline parallelism
            return False, "no_pipeline"

        # Checkpoint at pipeline boundaries
        is_boundary = (
            context.layer_idx == 0 or context.layer_idx == context.total_layers - 1
        )

        if is_boundary:
            return True, "pipeline_boundary"

        # Checkpoint in bubble optimization zones
        bubble_optimization = self.config.get("bubble_optimization", True)
        if bubble_optimization:
            # Checkpoint middle stages more aggressively
            total_stages = self.config.get("total_pipeline_stages", 4)
            is_middle_stage = context.pp_rank > 0 and context.pp_rank < total_stages - 1

            if is_middle_stage and context.layer_idx % 2 == 0:
                return True, "bubble_optimization"

        return False, "pipeline_skip"


class CheckpointStrategyFactory:
    """Factory for creating checkpoint strategies."""

    _strategies = {
        "coordinated": CoordinatedStrategy,
        "load_balanced": LoadBalancedStrategy,
        "hierarchical": HierarchicalStrategy,
        "adaptive": AdaptiveStrategy,
        "expert_aware": ExpertAwareStrategy,
        "pipeline_aware": PipelineAwareStrategy,
    }

    @classmethod
    def create_strategy(
        cls, strategy_name: str, config: Optional[Dict[str, Any]] = None
    ) -> CheckpointStrategy:
        """
        Create a checkpoint strategy by name.

        Args:
            strategy_name: Name of the strategy to create
            config: Optional configuration for the strategy

        Returns:
            Instantiated checkpoint strategy

        Raises:
            ValueError: If strategy_name is not recognized
        """
        if strategy_name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available strategies: {available}"
            )

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(config)  # type: ignore[abstract]

    @classmethod
    def register_strategy(
        cls, name: str, strategy_class: Type[CheckpointStrategy]
    ) -> None:
        """
        Register a custom strategy class.

        Args:
            name: Name to register the strategy under
            strategy_class: Strategy class to register
        """
        if not issubclass(strategy_class, CheckpointStrategy):
            raise TypeError(
                f"Strategy class must inherit from CheckpointStrategy, "
                f"got {strategy_class}"
            )

        cls._strategies[name] = strategy_class

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names."""
        return list(cls._strategies.keys())


class CompositeStrategy(CheckpointStrategy):
    """Composite strategy that combines multiple strategies."""

    def __init__(
        self, strategies: List[CheckpointStrategy], combination_mode: str = "any"
    ):
        """
        Initialize composite strategy.

        Args:
            strategies: List of strategies to combine
            combination_mode: How to combine decisions ("any", "all", "majority")
        """
        super().__init__()
        self.strategies = strategies
        self.combination_mode = combination_mode

    def should_checkpoint(self, context: CheckpointContext) -> Tuple[bool, str]:
        """Combine decisions from multiple strategies."""
        decisions = []
        reasons = []

        for strategy in self.strategies:
            should_checkpoint, reason = strategy.should_checkpoint(context)
            decisions.append(should_checkpoint)
            reasons.append(f"{strategy.__class__.__name__}:{reason}")

        if self.combination_mode == "any":
            result = any(decisions)
        elif self.combination_mode == "all":
            result = all(decisions)
        elif self.combination_mode == "majority":
            result = sum(decisions) > len(decisions) / 2
        else:
            raise ValueError(f"Unknown combination mode: {self.combination_mode}")

        combined_reason = f"composite_{self.combination_mode}[{','.join(reasons)}]"
        return result, combined_reason


def create_auto_strategy(
    world_size: int, parallel_config: Optional[Dict[str, int]] = None
) -> CheckpointStrategy:
    """
    Automatically create an appropriate strategy based on configuration.

    Args:
        world_size: Total number of distributed processes
        parallel_config: Dictionary with parallel dimension sizes

    Returns:
        Appropriate checkpoint strategy for the configuration
    """
    if parallel_config is None:
        parallel_config = {}

    tp_size = parallel_config.get("tp_size", 1)
    pp_size = parallel_config.get("pp_size", 1)
    ep_size = parallel_config.get("ep_size", 1)

    # Select strategy based on parallelism configuration
    if ep_size > 1:
        # MoE model
        return ExpertAwareStrategy({"expert_group_size": min(4, ep_size)})
    elif pp_size > 1:
        # Pipeline parallelism
        return PipelineAwareStrategy(
            {"total_pipeline_stages": pp_size, "bubble_optimization": True}
        )
    elif world_size > 8:
        # Large scale training
        return LoadBalancedStrategy()
    elif tp_size > 1:
        # Tensor parallelism
        return HierarchicalStrategy()
    else:
        # Default coordinated strategy
        return CoordinatedStrategy({"checkpoint_interval": 3})
