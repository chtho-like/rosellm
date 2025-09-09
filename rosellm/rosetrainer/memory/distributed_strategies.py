"""
Advanced Distributed Checkpointing Strategies for RoseTrainer

This module implements sophisticated strategies for distributed activation checkpointing
that consider multiple parallelism dimensions, memory patterns, and communication costs.
It extends the base selective recomputation with distributed-aware decision making.

Key Features:
- Multi-dimensional parallelism-aware strategies
- Communication cost modeling and optimization
- Dynamic load balancing across distributed ranks
- Expert parallelism and MoE-specific optimizations
- Pipeline parallelism bubble minimization
- Memory pressure-based adaptive strategies
- CUDA Graph compatibility for distributed operations

References:
[1] Megatron-LM Distributed Training Strategies
[2] FairScale Distributed Memory Optimization
[3] DeepSpeed ZeRO and Pipeline Parallelism
[4] GPipe: Efficient Training of Giant Neural Networks
"""

import enum
import logging
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, cast

import torch
import torch.distributed as dist

from ..parallelism import parallel_state
from .distributed_checkpoint import (
    DistributedCheckpointConfig,
    DistributedCheckpointStrategy,
)
from .selective_recompute import (
    BaseSelectionStrategy,
    LayerProfile,
    SelectiveCheckpointConfig,
)

logger = logging.getLogger(__name__)


class CommunicationPattern(enum.Enum):
    """Communication patterns for distributed checkpointing."""

    ALL_TO_ALL = "all_to_all"
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    POINT_TO_POINT = "point_to_point"
    BROADCAST = "broadcast"
    REDUCE_SCATTER = "reduce_scatter"


@dataclass
class CommunicationCost:
    """Model for communication cost estimation."""

    pattern: CommunicationPattern
    data_size_bytes: int
    num_participants: int
    bandwidth_gbps: float = 100.0  # Default InfiniBand bandwidth
    latency_microseconds: float = 1.0  # Default network latency

    def estimate_time_seconds(self) -> float:
        """Estimate communication time in seconds."""
        # Convert data size to GB
        data_size_gb = self.data_size_bytes / (1024**3)

        # Basic bandwidth model (ignoring algorithmic complexity)
        bandwidth_time = data_size_gb / self.bandwidth_gbps

        # Add latency component
        latency_time = self.latency_microseconds / 1_000_000

        # Pattern-specific scaling
        if self.pattern == CommunicationPattern.ALL_TO_ALL:
            # O(n) complexity for most implementations
            scaling_factor = self.num_participants
        elif self.pattern in [
            CommunicationPattern.ALL_REDUCE,
            CommunicationPattern.ALL_GATHER,
        ]:
            # O(log n) for tree-based algorithms
            scaling_factor = int(math.log2(max(self.num_participants, 2)))
        elif self.pattern == CommunicationPattern.POINT_TO_POINT:
            # O(1) for direct communication
            scaling_factor = 1
        elif self.pattern == CommunicationPattern.BROADCAST:
            # O(log n) for tree-based broadcast
            scaling_factor = int(math.log2(max(self.num_participants, 2)))
        elif self.pattern == CommunicationPattern.REDUCE_SCATTER:
            # O(log n) for tree-based reduce-scatter
            scaling_factor = int(math.log2(max(self.num_participants, 2)))
        else:
            scaling_factor = 1

        return bandwidth_time * scaling_factor + latency_time


@dataclass
class DistributedLayerProfile(LayerProfile):
    """Extended layer profile for distributed training."""

    # Communication costs
    forward_communication_cost: float = 0.0  # seconds
    backward_communication_cost: float = 0.0  # seconds

    # Parallel dimension information
    tensor_parallel_rank: int = 0
    pipeline_parallel_rank: int = 0
    data_parallel_rank: int = 0
    context_parallel_rank: int = 0
    expert_parallel_rank: int = 0

    # Memory distribution
    memory_distribution: Dict[int, float] = field(
        default_factory=dict
    )  # rank -> memory_mb

    # Checkpoint coordination
    coordinated_checkpoint_count: int = 0
    local_checkpoint_count: int = 0
    coordination_overhead: float = 0.0  # seconds

    # Expert-specific information
    is_expert_layer: bool = False
    expert_id: Optional[int] = None
    expert_utilization: float = 0.0  # 0.0 to 1.0

    def get_total_communication_cost(self) -> float:
        """Get total communication cost for this layer."""
        return self.forward_communication_cost + self.backward_communication_cost

    def get_memory_imbalance_ratio(self) -> float:
        """Get memory imbalance ratio across ranks."""
        if not self.memory_distribution:
            return 1.0

        memory_values = list(self.memory_distribution.values())
        if not memory_values:
            return 1.0

        min_memory = min(memory_values)
        max_memory = max(memory_values)

        return max_memory / min_memory if min_memory > 0 else float("inf")


class DistributedCoordinatedStrategy(BaseSelectionStrategy):
    """Coordinated strategy that synchronizes decisions across specified
    parallel dimensions.
    """

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        super().__init__(config)

        # Get distributed configuration
        if hasattr(config, "distributed_config"):
            self.dist_config = cast(Any, config).distributed_config
        else:
            # Use default distributed config
            self.dist_config = DistributedCheckpointConfig()

        # Get parallel state
        if parallel_state.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

            self.tp_size = parallel_state.get_tensor_model_parallel_size()
            self.pp_size = parallel_state.get_pipeline_model_parallel_size()
            self.dp_size = parallel_state.get_data_parallel_size()
            self.cp_size = parallel_state.get_context_parallel_size()
            self.ep_size = parallel_state.get_expert_model_parallel_size()

            self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
            self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            self.dp_rank = parallel_state.get_data_parallel_rank()
            self.cp_rank = parallel_state.get_context_parallel_rank()
            self.ep_rank = parallel_state.get_expert_model_parallel_rank()

            self.tp_group = parallel_state.get_tensor_model_parallel_group()
            self.pp_group = parallel_state.get_pipeline_model_parallel_group()
            self.dp_group = parallel_state.get_data_parallel_group()
            self.cp_group = parallel_state.get_context_parallel_group()
            self.ep_group = parallel_state.get_expert_model_parallel_group()
        else:
            # Single rank setup
            self.world_size = 1
            self.rank = 0
            self.tp_size = self.pp_size = self.dp_size = self.cp_size = self.ep_size = 1
            self.tp_rank = self.pp_rank = self.dp_rank = self.cp_rank = self.ep_rank = 0
            self.tp_group = (
                self.pp_group
            ) = self.dp_group = self.cp_group = self.ep_group = None

        # Coordination cache
        self.coordination_cache: Dict[str, bool] = {}
        self.coordination_timeout = 5.0  # seconds

        # Thread safety
        self._lock = threading.RLock() if config.thread_safe else None

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Determine if layer should be checkpointed with coordination."""
        if not parallel_state.is_initialized():
            # Single rank - use base strategy logic
            return self._local_decision(layer_id, profile)

        # Check cache first
        if self._lock:
            with self._lock:
                if layer_id in self.coordination_cache:
                    return self.coordination_cache[layer_id]
        else:
            if layer_id in self.coordination_cache:
                return self.coordination_cache[layer_id]

        # Make coordinated decision
        decision = self._coordinated_decision(layer_id, profile)

        # Cache result
        if self._lock:
            with self._lock:
                self.coordination_cache[layer_id] = decision
        else:
            self.coordination_cache[layer_id] = decision

        return decision

    def _local_decision(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Make local checkpointing decision without coordination."""
        # Simple hash-based decision for consistency
        return hash(layer_id) % 2 == 0

    def _coordinated_decision(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Make coordinated decision across relevant parallel dimensions."""
        try:
            # Determine primary coordination group
            primary_group = None
            primary_rank = 0

            if (
                self.dist_config.coordinate_across_tp
                and self.tp_group is not None
                and self.tp_size > 1
            ):
                primary_group = self.tp_group
                primary_rank = 0  # TP rank 0 makes the decision
            elif (
                self.dist_config.coordinate_across_cp
                and self.cp_group is not None
                and self.cp_size > 1
            ):
                primary_group = self.cp_group
                primary_rank = 0
            elif (
                self.dist_config.coordinate_across_ep
                and self.ep_group is not None
                and self.ep_size > 1
            ):
                primary_group = self.ep_group
                primary_rank = 0

            if primary_group is None:
                return self._local_decision(layer_id, profile)

            # Make decision on primary rank
            if self.rank == primary_rank:
                decision = self._local_decision(layer_id, profile)
            else:
                decision = False  # Will be overwritten by broadcast

            # Broadcast decision
            decision_tensor = torch.tensor(int(decision), dtype=torch.int32)
            if torch.cuda.is_available():
                decision_tensor = decision_tensor.cuda()

            dist.broadcast(decision_tensor, src=primary_rank, group=primary_group)

            return bool(decision_tensor.item())

        except Exception as e:
            logger.warning(f"Coordination failed for layer {layer_id}: {e}")
            return self._local_decision(layer_id, profile)

    def update_selection(
        self, profiles: Dict[str, LayerProfile], step: int
    ) -> Set[str]:
        """Update selection with coordination."""
        # Clear cache periodically
        if step % 100 == 0:
            if self._lock:
                with self._lock:
                    self.coordination_cache.clear()
            else:
                self.coordination_cache.clear()

        return set()  # Decisions are made on-demand


class DistributedLoadBalancedStrategy(BaseSelectionStrategy):
    """Strategy that balances memory load across distributed ranks."""

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        super().__init__(config)

        # Get distributed configuration
        if hasattr(config, "distributed_config"):
            self.dist_config = cast(Any, config).distributed_config
        else:
            self.dist_config = DistributedCheckpointConfig()

        # Memory tracking
        self.rank_memory_usage: Dict[int, float] = {}
        self.memory_update_interval = 50  # steps
        self.last_memory_update = 0

        # Load balancing state
        self.checkpoint_assignments: Dict[
            str, Set[int]
        ] = {}  # layer_id -> set of ranks
        self.target_memory_per_rank = 0.0

        # Thread safety
        self._lock = threading.RLock() if config.thread_safe else None

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Determine checkpointing with load balancing."""
        if not parallel_state.is_initialized():
            return True

        rank = dist.get_rank()

        # Check if this rank is assigned to checkpoint this layer
        if self._lock:
            with self._lock:
                assigned_ranks = self.checkpoint_assignments.get(layer_id, set())
        else:
            assigned_ranks = self.checkpoint_assignments.get(layer_id, set())

        return rank in assigned_ranks

    def update_selection(
        self, profiles: Dict[str, LayerProfile], step: int
    ) -> Set[str]:
        """Update selection with load balancing."""
        if not parallel_state.is_initialized():
            return set(profiles.keys())

        # Update memory usage periodically
        if step - self.last_memory_update >= self.memory_update_interval:
            self._update_memory_usage()
            self._rebalance_checkpoints(profiles)
            self.last_memory_update = step

        # Return layers assigned to this rank
        rank = dist.get_rank()
        selected_layers = set()

        if self._lock:
            with self._lock:
                for layer_id, assigned_ranks in self.checkpoint_assignments.items():
                    if rank in assigned_ranks:
                        selected_layers.add(layer_id)
        else:
            for layer_id, assigned_ranks in self.checkpoint_assignments.items():
                if rank in assigned_ranks:
                    selected_layers.add(layer_id)

        return selected_layers

    def _update_memory_usage(self) -> None:
        """Update memory usage across all ranks."""
        try:
            # Get local memory usage
            local_memory = 0.0
            if torch.cuda.is_available():
                local_memory = torch.cuda.memory_allocated() / (1024**3)  # GB

            world_size = dist.get_world_size()

            # All-gather memory usage
            memory_tensor = torch.tensor(local_memory, dtype=torch.float32)
            if torch.cuda.is_available():
                memory_tensor = memory_tensor.cuda()

            gathered_memory = [
                torch.zeros_like(memory_tensor) for _ in range(world_size)
            ]
            dist.all_gather(gathered_memory, memory_tensor)

            # Update memory tracking
            if self._lock:
                with self._lock:
                    for rank, mem_tensor in enumerate(gathered_memory):
                        self.rank_memory_usage[rank] = mem_tensor.item()

                    # Calculate target memory per rank
                    total_memory = sum(self.rank_memory_usage.values())
                    self.target_memory_per_rank = (
                        total_memory / world_size if world_size > 0 else 0.0
                    )
            else:
                for rank, mem_tensor in enumerate(gathered_memory):
                    self.rank_memory_usage[rank] = mem_tensor.item()

                total_memory = sum(self.rank_memory_usage.values())
                self.target_memory_per_rank = (
                    total_memory / world_size if world_size > 0 else 0.0
                )

        except Exception as e:
            logger.error(f"Failed to update memory usage: {e}")

    def _rebalance_checkpoints(self, profiles: Dict[str, LayerProfile]) -> None:
        """Rebalance checkpoint assignments across ranks."""
        if not self.rank_memory_usage:
            return

        world_size = dist.get_world_size()

        # Sort layers by memory usage (descending)
        layer_memory = []
        for layer_id, profile in profiles.items():
            memory_usage = getattr(profile, "memory_usage", 0.0)
            layer_memory.append((layer_id, memory_usage))

        layer_memory.sort(key=lambda x: x[1], reverse=True)

        # Initialize rank loads
        rank_loads = {
            rank: self.rank_memory_usage.get(rank, 0.0) for rank in range(world_size)
        }

        # Assign layers using a greedy approach
        new_assignments: Dict[str, Set[int]] = {}

        for layer_id, memory_usage in layer_memory:
            # Find rank with minimum load
            min_rank = min(rank_loads.keys(), key=lambda r: rank_loads[r])

            # Assign layer to this rank
            new_assignments[layer_id] = {min_rank}
            rank_loads[min_rank] += memory_usage

        # Update assignments atomically
        if self._lock:
            with self._lock:
                self.checkpoint_assignments = new_assignments
        else:
            self.checkpoint_assignments = new_assignments


class DistributedHierarchicalStrategy(BaseSelectionStrategy):
    """Strategy that uses different approaches for different parallel dimensions."""

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        super().__init__(config)

        # Initialize sub-strategies for different dimensions
        self.tp_strategy = self._create_tp_strategy(config)
        self.pp_strategy = self._create_pp_strategy(config)
        self.dp_strategy = self._create_dp_strategy(config)
        self.cp_strategy = self._create_cp_strategy(config)
        self.ep_strategy = self._create_ep_strategy(config)

        # Strategy priorities (higher = more important)
        self.strategy_priorities = {
            "tp": 5,  # Highest priority for tensor parallel
            "ep": 4,  # Expert parallel
            "cp": 3,  # Context parallel
            "pp": 2,  # Pipeline parallel
            "dp": 1,  # Data parallel (lowest priority)
        }

    def _create_tp_strategy(
        self, config: SelectiveCheckpointConfig
    ) -> BaseSelectionStrategy:
        """Create strategy for tensor parallelism."""
        # For TP, coordinate checkpointing to ensure consistency
        return DistributedCoordinatedStrategy(config)

    def _create_pp_strategy(
        self, config: SelectiveCheckpointConfig
    ) -> BaseSelectionStrategy:
        """Create strategy for pipeline parallelism."""
        # For PP, use load balancing to distribute checkpoints across stages
        return DistributedLoadBalancedStrategy(config)

    def _create_dp_strategy(
        self, config: SelectiveCheckpointConfig
    ) -> BaseSelectionStrategy:
        """Create strategy for data parallelism."""
        # For DP, can use independent decisions since gradients are synchronized
        from .selective_recompute import HybridStrategy

        return HybridStrategy(config)

    def _create_cp_strategy(
        self, config: SelectiveCheckpointConfig
    ) -> BaseSelectionStrategy:
        """Create strategy for context parallelism."""
        # For CP, coordinate to ensure sequence consistency
        return DistributedCoordinatedStrategy(config)

    def _create_ep_strategy(
        self, config: SelectiveCheckpointConfig
    ) -> BaseSelectionStrategy:
        """Create strategy for expert parallelism."""
        # For EP, use expert-aware load balancing
        return DistributedExpertAwareStrategy(config)

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Use hierarchical decision making."""
        if not parallel_state.is_initialized():
            return True

        # Get parallel dimensions with non-trivial sizes
        active_dimensions = []

        tp_size = parallel_state.get_tensor_model_parallel_size()
        if tp_size > 1:
            active_dimensions.append(("tp", tp_size, self.tp_strategy))

        ep_size = parallel_state.get_expert_model_parallel_size()
        if ep_size > 1:
            active_dimensions.append(("ep", ep_size, self.ep_strategy))

        cp_size = parallel_state.get_context_parallel_size()
        if cp_size > 1:
            active_dimensions.append(("cp", cp_size, self.cp_strategy))

        pp_size = parallel_state.get_pipeline_model_parallel_size()
        if pp_size > 1:
            active_dimensions.append(("pp", pp_size, self.pp_strategy))

        dp_size = parallel_state.get_data_parallel_size()
        if dp_size > 1:
            active_dimensions.append(("dp", dp_size, self.dp_strategy))

        if not active_dimensions:
            return True  # Single rank

        # Sort by priority (highest first)
        active_dimensions.sort(
            key=lambda x: self.strategy_priorities[x[0]], reverse=True
        )

        # Use the highest priority dimension's strategy
        primary_dim, _, primary_strategy = active_dimensions[0]

        decision = primary_strategy.should_checkpoint(layer_id, profile)

        # For debugging
        logger.debug(
            f"Hierarchical decision for {layer_id}: {decision} "
            f"(primary dimension: {primary_dim})"
        )

        return decision

    def update_selection(
        self, profiles: Dict[str, LayerProfile], step: int
    ) -> Set[str]:
        """Update selection using hierarchical approach."""
        # Update all sub-strategies
        all_selections = []

        if hasattr(self.tp_strategy, "update_selection"):
            all_selections.append(self.tp_strategy.update_selection(profiles, step))
        if hasattr(self.pp_strategy, "update_selection"):
            all_selections.append(self.pp_strategy.update_selection(profiles, step))
        if hasattr(self.dp_strategy, "update_selection"):
            all_selections.append(self.dp_strategy.update_selection(profiles, step))
        if hasattr(self.cp_strategy, "update_selection"):
            all_selections.append(self.cp_strategy.update_selection(profiles, step))
        if hasattr(self.ep_strategy, "update_selection"):
            all_selections.append(self.ep_strategy.update_selection(profiles, step))

        # Combine selections (union of all)
        combined_selection = set()
        for selection in all_selections:
            combined_selection.update(selection)

        return combined_selection


class DistributedExpertAwareStrategy(BaseSelectionStrategy):
    """Strategy optimized for MoE models with expert parallelism."""

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        super().__init__(config)

        # Expert tracking
        self.expert_layers: Set[str] = set()
        self.non_expert_layers: Set[str] = set()
        self.expert_utilization: Dict[str, float] = {}

        # Load balancing for experts
        self.expert_memory_usage: Dict[int, Dict[str, float]] = defaultdict(
            dict
        )  # rank -> {layer -> memory}

        # Thread safety
        self._lock = threading.RLock() if config.thread_safe else None

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Expert-aware checkpointing decision."""
        # Classify layer type
        is_expert = self._is_expert_layer(layer_id)

        if is_expert:
            return self._expert_layer_decision(layer_id, profile)
        else:
            return self._non_expert_layer_decision(layer_id, profile)

    def _is_expert_layer(self, layer_id: str) -> bool:
        """Determine if layer is an expert layer."""
        # Common patterns for expert layers
        expert_patterns = ["expert", "moe", "mixture_of_experts", "ffn_expert"]
        return any(pattern in layer_id.lower() for pattern in expert_patterns)

    def _expert_layer_decision(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Make checkpointing decision for expert layers."""
        if not parallel_state.is_initialized():
            return True

        ep_size = parallel_state.get_expert_model_parallel_size()
        if ep_size <= 1:
            return True  # Not using expert parallelism

        ep_rank = parallel_state.get_expert_model_parallel_rank()

        # Balance expert checkpointing across expert ranks
        expert_hash = hash(layer_id) % ep_size

        # This expert rank is responsible for checkpointing this layer
        should_checkpoint = expert_hash == ep_rank

        # Consider expert utilization if available
        if isinstance(profile, DistributedLayerProfile):
            if profile.is_expert_layer and profile.expert_utilization < 0.1:
                # Low utilization expert - don't checkpoint to save memory
                should_checkpoint = False

        return should_checkpoint

    def _non_expert_layer_decision(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Make checkpointing decision for non-expert layers."""
        # For non-expert layers, coordinate across tensor parallel ranks
        if not parallel_state.is_initialized():
            return True

        tp_size = parallel_state.get_tensor_model_parallel_size()
        if tp_size > 1:
            # Coordinate across TP ranks
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            return tp_rank == 0  # Only TP rank 0 checkpoints

        return True

    def update_selection(
        self, profiles: Dict[str, LayerProfile], step: int
    ) -> Set[str]:
        """Update expert-aware selection."""
        selected_layers = set()

        # Update expert/non-expert classification
        if self._lock:
            with self._lock:
                self._update_layer_classification(profiles)
        else:
            self._update_layer_classification(profiles)

        # Select layers based on current strategy
        for layer_id, profile in profiles.items():
            if self.should_checkpoint(layer_id, profile):
                selected_layers.add(layer_id)

        return selected_layers

    def _update_layer_classification(self, profiles: Dict[str, LayerProfile]) -> None:
        """Update classification of expert vs non-expert layers."""
        for layer_id in profiles:
            if self._is_expert_layer(layer_id):
                self.expert_layers.add(layer_id)
                self.non_expert_layers.discard(layer_id)
            else:
                self.non_expert_layers.add(layer_id)
                self.expert_layers.discard(layer_id)


class DistributedPipelineAwareStrategy(BaseSelectionStrategy):
    """Strategy optimized for pipeline parallelism with bubble minimization."""

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        super().__init__(config)

        # Pipeline configuration
        if hasattr(config, "distributed_config"):
            self.dist_config = cast(Any, config).distributed_config
        else:
            self.dist_config = DistributedCheckpointConfig()

        # Pipeline state
        self.pipeline_stages: Dict[int, Set[str]] = defaultdict(
            set
        )  # stage -> layer_ids
        self.stage_memory_usage: Dict[int, float] = {}
        self.bubble_optimization = True

        # Critical path analysis
        self.critical_stages: Set[int] = set()
        self.stage_execution_times: Dict[int, float] = {}

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Pipeline-aware checkpointing decision."""
        if not parallel_state.is_initialized():
            return True

        pp_size = parallel_state.get_pipeline_model_parallel_size()
        if pp_size <= 1:
            return True  # Not using pipeline parallelism

        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

        # Use explicit stages if configured
        if self.dist_config.pipeline_checkpoint_stages:
            return pp_rank in self.dist_config.pipeline_checkpoint_stages

        # Bubble optimization: avoid checkpointing on critical path
        if self.bubble_optimization and pp_rank in self.critical_stages:
            return False

        # Default: checkpoint on middle stages to balance memory and bubbles
        middle_start = pp_size // 4
        middle_end = 3 * pp_size // 4

        return middle_start <= pp_rank < middle_end

    def update_selection(
        self, profiles: Dict[str, LayerProfile], step: int
    ) -> Set[str]:
        """Update pipeline-aware selection."""
        if not parallel_state.is_initialized():
            return set(profiles.keys())

        # Update pipeline stage analysis
        self._analyze_pipeline_stages(profiles)

        # Update critical path
        self._update_critical_path()

        # Select layers for this pipeline stage
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        selected_layers = set()

        for layer_id, profile in profiles.items():
            # Assign layer to pipeline stage (simplified)
            stage = self._get_layer_stage(layer_id)
            if stage == pp_rank and self.should_checkpoint(layer_id, profile):
                selected_layers.add(layer_id)

        return selected_layers

    def _analyze_pipeline_stages(self, profiles: Dict[str, LayerProfile]) -> None:
        """Analyze pipeline stages and their characteristics."""
        pp_size = parallel_state.get_pipeline_model_parallel_size()

        # Reset stage information
        self.pipeline_stages.clear()
        self.stage_memory_usage.clear()
        self.stage_execution_times.clear()

        # Assign layers to stages and collect statistics
        for layer_id, profile in profiles.items():
            stage = self._get_layer_stage(layer_id)
            if 0 <= stage < pp_size:
                self.pipeline_stages[stage].add(layer_id)
                self.stage_memory_usage[stage] = (
                    self.stage_memory_usage.get(stage, 0.0) + profile.memory_usage
                )
                self.stage_execution_times[stage] = (
                    self.stage_execution_times.get(stage, 0.0)
                    + profile.computation_time
                )

    def _get_layer_stage(self, layer_id: str) -> int:
        """Determine which pipeline stage a layer belongs to."""
        # Simplified layer-to-stage mapping based on layer name
        # In practice, this would be more sophisticated
        layer_hash = hash(layer_id)
        pp_size = parallel_state.get_pipeline_model_parallel_size()
        return abs(layer_hash) % pp_size

    def _update_critical_path(self) -> None:
        """Update critical path stages based on execution times."""
        if not self.stage_execution_times:
            return

        # Find stages with high execution times (potential bottlenecks)
        avg_time = sum(self.stage_execution_times.values()) / len(
            self.stage_execution_times
        )

        self.critical_stages.clear()
        for stage, exec_time in self.stage_execution_times.items():
            if exec_time > avg_time * 1.2:  # 20% above average
                self.critical_stages.add(stage)


class DistributedAdaptiveStrategy(BaseSelectionStrategy):
    """Adaptive strategy that dynamically selects the best approach based on
    runtime conditions.
    """

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        super().__init__(config)

        # Available sub-strategies
        self.strategies = {
            "coordinated": DistributedCoordinatedStrategy(config),
            "load_balanced": DistributedLoadBalancedStrategy(config),
            "hierarchical": DistributedHierarchicalStrategy(config),
            "expert_aware": DistributedExpertAwareStrategy(config),
            "pipeline_aware": DistributedPipelineAwareStrategy(config),
        }

        # Strategy performance tracking
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.strategy_memory_efficiency: Dict[str, List[float]] = defaultdict(list)

        # Current strategy selection
        self.current_strategy = "hierarchical"  # Default
        self.strategy_switch_interval = 200  # steps
        self.last_strategy_update = 0

        # Adaptation parameters
        self.performance_window = 50  # Keep last 50 measurements
        self.min_improvement_threshold = 0.05  # 5% improvement required

        # Thread safety
        self._lock = threading.RLock() if config.thread_safe else None

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Use current best strategy for checkpointing decision."""
        current_strategy = self.strategies[self.current_strategy]
        return current_strategy.should_checkpoint(layer_id, profile)

    def update_selection(
        self, profiles: Dict[str, LayerProfile], step: int
    ) -> Set[str]:
        """Adaptive selection with strategy evaluation."""
        # Evaluate and potentially switch strategies
        if step - self.last_strategy_update >= self.strategy_switch_interval:
            self._evaluate_and_switch_strategy(profiles, step)
            self.last_strategy_update = step

        # Use current strategy
        current_strategy = self.strategies[self.current_strategy]
        if hasattr(current_strategy, "update_selection"):
            return current_strategy.update_selection(profiles, step)

        return set()

    def _evaluate_and_switch_strategy(
        self, profiles: Dict[str, LayerProfile], step: int
    ) -> None:
        """Evaluate all strategies and switch to the best one."""
        if not profiles:
            return

        current_metrics = self._compute_current_metrics(profiles)

        # Record performance for current strategy
        if self._lock:
            with self._lock:
                self.strategy_performance[self.current_strategy].append(
                    current_metrics["performance"]
                )
                self.strategy_memory_efficiency[self.current_strategy].append(
                    current_metrics["memory_efficiency"]
                )

                # Trim history
                for strategy in self.strategy_performance:
                    if (
                        len(self.strategy_performance[strategy])
                        > self.performance_window
                    ):
                        self.strategy_performance[strategy] = self.strategy_performance[
                            strategy
                        ][-self.performance_window :]
                    if (
                        len(self.strategy_memory_efficiency[strategy])
                        > self.performance_window
                    ):
                        self.strategy_memory_efficiency[
                            strategy
                        ] = self.strategy_memory_efficiency[strategy][
                            -self.performance_window :
                        ]
        else:
            self.strategy_performance[self.current_strategy].append(
                current_metrics["performance"]
            )
            self.strategy_memory_efficiency[self.current_strategy].append(
                current_metrics["memory_efficiency"]
            )

        # Evaluate all strategies
        best_strategy = self._select_best_strategy()

        if best_strategy != self.current_strategy:
            logger.info(
                f"Switching distributed strategy from {self.current_strategy} "
                f"to {best_strategy}"
            )
            self.current_strategy = best_strategy

    def _compute_current_metrics(
        self, profiles: Dict[str, LayerProfile]
    ) -> Dict[str, float]:
        """Compute current performance and memory efficiency metrics."""
        total_memory = sum(profile.memory_usage for profile in profiles.values())
        total_compute_time = sum(
            profile.computation_time for profile in profiles.values()
        )
        total_recompute_time = sum(
            profile.recompute_time for profile in profiles.values()
        )

        # Performance metric (lower recompute overhead is better)
        if total_compute_time > 0:
            performance = 1.0 / (1.0 + total_recompute_time / total_compute_time)
        else:
            performance = 1.0

        # Memory efficiency (normalize by number of layers)
        memory_efficiency = total_memory / len(profiles) if profiles else 0.0

        return {
            "performance": performance,
            "memory_efficiency": memory_efficiency,
        }

    def _select_best_strategy(self) -> str:
        """Select the best strategy based on historical performance."""
        strategy_scores = {}

        for strategy_name in self.strategies:
            if strategy_name not in self.strategy_performance:
                continue

            perf_history = self.strategy_performance[strategy_name]
            memory_history = self.strategy_memory_efficiency[strategy_name]

            if not perf_history or not memory_history:
                continue

            # Compute average performance and memory efficiency
            avg_performance = sum(perf_history) / len(perf_history)
            avg_memory_efficiency = sum(memory_history) / len(memory_history)

            # Combined score (higher is better)
            # Weight performance more heavily than memory efficiency
            score = 0.7 * avg_performance + 0.3 * (1.0 / (1.0 + avg_memory_efficiency))

            strategy_scores[strategy_name] = score

        if not strategy_scores:
            return self.current_strategy

        # Select strategy with highest score
        best_strategy = max(strategy_scores.keys(), key=lambda k: strategy_scores[k])

        # Only switch if improvement is significant
        current_score = strategy_scores.get(self.current_strategy, 0.0)
        best_score = strategy_scores[best_strategy]

        if best_score > current_score * (1 + self.min_improvement_threshold):
            return best_strategy
        else:
            return self.current_strategy


# Factory functions for creating distributed strategies
def create_distributed_strategy(
    strategy_type: DistributedCheckpointStrategy,
    config: SelectiveCheckpointConfig,
) -> BaseSelectionStrategy:
    """Create a distributed checkpointing strategy.

    Args:
        strategy_type: Type of distributed strategy to create
        config: Selective checkpointing configuration

    Returns:
        Configured distributed strategy instance
    """
    if strategy_type == DistributedCheckpointStrategy.COORDINATED:
        return DistributedCoordinatedStrategy(config)
    elif strategy_type == DistributedCheckpointStrategy.LOAD_BALANCED:
        return DistributedLoadBalancedStrategy(config)
    elif strategy_type == DistributedCheckpointStrategy.HIERARCHICAL:
        return DistributedHierarchicalStrategy(config)
    elif strategy_type == DistributedCheckpointStrategy.EXPERT_AWARE:
        return DistributedExpertAwareStrategy(config)
    elif strategy_type == DistributedCheckpointStrategy.PIPELINE_AWARE:
        return DistributedPipelineAwareStrategy(config)
    elif strategy_type == DistributedCheckpointStrategy.ADAPTIVE:
        return DistributedAdaptiveStrategy(config)
    else:
        raise ValueError(f"Unknown distributed strategy type: {strategy_type}")


def estimate_communication_cost(
    layer_size_bytes: int,
    pattern: CommunicationPattern,
    num_participants: int,
    bandwidth_gbps: float = 100.0,
    latency_microseconds: float = 1.0,
) -> float:
    """Estimate communication cost for a given pattern.

    Args:
        layer_size_bytes: Size of data to communicate
        pattern: Communication pattern
        num_participants: Number of participating ranks
        bandwidth_gbps: Available bandwidth in Gbps
        latency_microseconds: Network latency in microseconds

    Returns:
        Estimated communication time in seconds
    """
    cost_model = CommunicationCost(
        pattern=pattern,
        data_size_bytes=layer_size_bytes,
        num_participants=num_participants,
        bandwidth_gbps=bandwidth_gbps,
        latency_microseconds=latency_microseconds,
    )

    return cost_model.estimate_time_seconds()
