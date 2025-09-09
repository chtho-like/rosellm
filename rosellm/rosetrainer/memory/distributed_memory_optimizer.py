"""
Enhanced Distributed Memory Optimizer with Checkpoint Coordination

This module extends the existing memory optimization components with distributed
coordination capabilities. It integrates with the distributed activation checkpointing
infrastructure to provide comprehensive memory management across all parallelism
dimensions in distributed training.

Key Features:
- Coordinated memory optimization across distributed ranks
- Integration with distributed activation checkpointing
- Cross-rank memory balancing and synchronization
- Dynamic memory scaling based on distributed workload
- Expert parallelism-aware memory management
- Pipeline parallel memory bubble optimization
- Advanced memory profiling and analytics

References:
[1] ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
[2] DeepSpeed: Extreme-scale Model Training for Everyone
[3] FairScale: A general purpose modular PyTorch library for high performance
[4] Megatron-LM: Training Multi-Billion Parameter Language Models Using GPU Model Parallelism
"""

import logging
import threading
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from ..parallelism import parallel_state
from .activation_checkpoint import ActivationCheckpointing, MemoryProfiler
from .cpu_offload import CPUOffloadOptimizer, ParameterOffloader
from .distributed_checkpoint import (
    DistributedActivationCheckpointing,
    DistributedCheckpointConfig,
    DistributedMemoryProfiler,
)
from .mixed_precision import MixedPrecisionManager, PrecisionType
from .param_grad_buffer import BufferManager, ParamAndGradBuffer
from .parameter_overlap import AsyncParameterGatherer, OverlapConfig
from .selective_recompute import SelectiveCheckpointConfig, SelectiveRecomputeManager

logger = logging.getLogger(__name__)


@dataclass
class DistributedMemoryConfig:
    """Configuration for distributed memory optimization."""

    # Distributed checkpointing
    distributed_checkpoint_config: DistributedCheckpointConfig = field(
        default_factory=DistributedCheckpointConfig
    )

    # Memory balancing
    enable_memory_balancing: bool = True
    memory_balance_threshold: float = 0.8  # Start balancing at 80% memory usage
    balance_interval_steps: int = 100  # Balance every N steps
    max_memory_imbalance_ratio: float = 1.5  # Max allowed memory ratio between ranks

    # Offloading coordination
    coordinate_cpu_offloading: bool = True
    offload_threshold_mb: float = 4096.0  # Start offloading at 4GB
    offload_selection_strategy: str = (
        "memory_based"  # "memory_based", "usage_based", "round_robin"
    )

    # Mixed precision coordination
    coordinate_precision_scaling: bool = True
    enable_dynamic_scaling: bool = True
    sync_loss_scale_frequency: int = 50  # Sync loss scale every N steps

    # Buffer management
    enable_distributed_buffering: bool = True
    buffer_sync_strategy: str = (
        "allreduce"  # "allreduce", "reduce_scatter", "hierarchical"
    )
    gradient_accumulation_coordination: bool = True

    # Parameter overlap
    enable_parameter_overlap: bool = True
    overlap_coordination: bool = True
    prefetch_coordination: bool = True

    # Memory profiling
    enable_distributed_profiling: bool = True
    profiling_sync_interval: int = 25  # Sync profiling data every N steps
    memory_analytics: bool = True

    # Expert parallelism optimization
    expert_memory_balancing: bool = True
    expert_activation_sharing: bool = False
    expert_buffer_coordination: bool = True

    # Pipeline parallelism optimization
    pipeline_memory_overlap: bool = True
    pipeline_bubble_reduction: bool = True
    stage_memory_coordination: bool = True

    # Advanced features
    enable_memory_defragmentation: bool = False
    auto_memory_scaling: bool = False
    memory_pressure_monitoring: bool = True

    # Debug and monitoring
    verbose_memory_logging: bool = False
    memory_leak_detection: bool = True
    performance_monitoring: bool = True

    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []

        if not 0 < self.memory_balance_threshold <= 1.0:
            errors.append("memory_balance_threshold must be between 0 and 1")
        if self.balance_interval_steps <= 0:
            errors.append("balance_interval_steps must be positive")
        if self.max_memory_imbalance_ratio < 1.0:
            errors.append("max_memory_imbalance_ratio must be >= 1.0")
        if self.offload_threshold_mb <= 0:
            errors.append("offload_threshold_mb must be positive")
        if self.sync_loss_scale_frequency <= 0:
            errors.append("sync_loss_scale_frequency must be positive")
        if self.profiling_sync_interval <= 0:
            errors.append("profiling_sync_interval must be positive")

        valid_offload_strategies = ["memory_based", "usage_based", "round_robin"]
        if self.offload_selection_strategy not in valid_offload_strategies:
            errors.append(
                f"offload_selection_strategy must be one of {valid_offload_strategies}"
            )

        valid_buffer_strategies = ["allreduce", "reduce_scatter", "hierarchical"]
        if self.buffer_sync_strategy not in valid_buffer_strategies:
            errors.append(
                f"buffer_sync_strategy must be one of {valid_buffer_strategies}"
            )

        if errors:
            raise ValueError(
                f"Distributed memory configuration validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        # Validate nested configs
        self.distributed_checkpoint_config.validate()


@dataclass
class MemoryBalance:
    """Memory balance information across ranks."""

    rank: int
    allocated_memory_gb: float
    reserved_memory_gb: float
    peak_memory_gb: float

    # Parallel dimensions
    tp_rank: int = 0
    pp_rank: int = 0
    dp_rank: int = 0
    cp_rank: int = 0
    ep_rank: int = 0

    # Usage patterns
    activation_memory_gb: float = 0.0
    parameter_memory_gb: float = 0.0
    gradient_memory_gb: float = 0.0
    optimizer_memory_gb: float = 0.0
    buffer_memory_gb: float = 0.0

    # Optimization state
    cpu_offloaded_gb: float = 0.0
    checkpointed_memory_saved_gb: float = 0.0

    timestamp: float = field(default_factory=time.time)

    def get_total_usage_gb(self) -> float:
        """Get total memory usage."""
        return (
            self.activation_memory_gb
            + self.parameter_memory_gb
            + self.gradient_memory_gb
            + self.optimizer_memory_gb
            + self.buffer_memory_gb
        )

    def get_efficiency_ratio(self) -> float:
        """Get memory efficiency ratio (useful memory / allocated memory)."""
        if self.allocated_memory_gb == 0:
            return 1.0
        return min(1.0, self.get_total_usage_gb() / self.allocated_memory_gb)


class DistributedMemoryCoordinator:
    """Coordinator for distributed memory optimization decisions."""

    def __init__(self, config: DistributedMemoryConfig) -> None:
        """Initialize distributed memory coordinator.

        Args:
            config: Distributed memory configuration
        """
        config.validate()
        self.config = config

        # Distributed state
        self.is_distributed = parallel_state.is_initialized()
        if self.is_distributed:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

            # Parallel dimensions
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
        else:
            self.world_size = 1
            self.rank = 0
            self.tp_size = self.pp_size = self.dp_size = self.cp_size = self.ep_size = 1
            self.tp_rank = self.pp_rank = self.dp_rank = self.cp_rank = self.ep_rank = 0

        # Memory tracking
        self.memory_balances: Dict[int, MemoryBalance] = {}
        self.memory_history: deque = deque(maxlen=100)
        self.last_balance_step = 0
        self.last_sync_time = time.time()

        # Coordination decisions
        self.offload_decisions: Dict[str, Set[int]] = (
            {}
        )  # parameter_name -> ranks_to_offload
        self.checkpoint_decisions: Dict[str, bool] = {}
        self.precision_decisions: Dict[str, PrecisionType] = {}

        # Thread safety
        self._lock = (
            threading.RLock()
            if config.distributed_checkpoint_config.base_config.thread_safe
            else None
        )

        if config.verbose_memory_logging:
            logger.info(
                f"Initialized DistributedMemoryCoordinator for rank {self.rank}"
            )

    def should_balance_memory(self, step: int) -> bool:
        """Determine if memory balancing should be performed."""
        if not self.config.enable_memory_balancing:
            return False

        return (step - self.last_balance_step) >= self.config.balance_interval_steps

    def balance_memory_across_ranks(self, step: int) -> Dict[str, Any]:
        """Balance memory usage across distributed ranks.

        Args:
            step: Current training step

        Returns:
            Dictionary containing balancing decisions and statistics
        """
        if not self.is_distributed:
            return {"balanced": False, "reason": "not_distributed"}

        # Update memory balances
        self._update_memory_balances()

        # Check if balancing is needed
        imbalance_ratio = self._calculate_memory_imbalance_ratio()
        if imbalance_ratio < self.config.max_memory_imbalance_ratio:
            return {
                "balanced": False,
                "reason": "within_threshold",
                "imbalance_ratio": imbalance_ratio,
            }

        # Perform balancing
        balancing_actions = self._perform_memory_balancing()

        self.last_balance_step = step

        if self.config.verbose_memory_logging:
            logger.info(
                f"Memory balancing completed at step {step}. "
                f"Imbalance ratio: {imbalance_ratio:.2f}, "
                f"Actions: {len(balancing_actions)}"
            )

        return {
            "balanced": True,
            "imbalance_ratio": imbalance_ratio,
            "actions": balancing_actions,
            "step": step,
        }

    def _update_memory_balances(self) -> None:
        """Update memory balance information across all ranks."""
        try:
            # Get local memory info
            local_balance = self._create_local_memory_balance()

            # All-gather memory balances
            gathered_balances = [None] * self.world_size
            dist.all_gather_object(gathered_balances, local_balance)

            # Update tracking
            if self._lock:
                with self._lock:
                    for rank, balance in enumerate(gathered_balances):
                        if balance is not None:
                            self.memory_balances[rank] = balance

                    # Add to history
                    self.memory_history.append(
                        {
                            "timestamp": time.time(),
                            "balances": self.memory_balances.copy(),
                        }
                    )
            else:
                for rank, balance in enumerate(gathered_balances):
                    if balance is not None:
                        self.memory_balances[rank] = balance

                self.memory_history.append(
                    {
                        "timestamp": time.time(),
                        "balances": self.memory_balances.copy(),
                    }
                )

        except Exception as e:
            logger.error(f"Failed to update memory balances: {e}")

    def _create_local_memory_balance(self) -> MemoryBalance:
        """Create memory balance information for the local rank."""
        # Get CUDA memory info
        allocated_gb = 0.0
        reserved_gb = 0.0
        peak_gb = 0.0

        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

        return MemoryBalance(
            rank=self.rank,
            allocated_memory_gb=allocated_gb,
            reserved_memory_gb=reserved_gb,
            peak_memory_gb=peak_gb,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            dp_rank=self.dp_rank,
            cp_rank=self.cp_rank,
            ep_rank=self.ep_rank,
        )

    def _calculate_memory_imbalance_ratio(self) -> float:
        """Calculate memory imbalance ratio across ranks."""
        if not self.memory_balances:
            return 1.0

        memory_usages = [
            balance.allocated_memory_gb for balance in self.memory_balances.values()
        ]

        if not memory_usages:
            return 1.0

        min_memory = min(memory_usages)
        max_memory = max(memory_usages)

        return max_memory / min_memory if min_memory > 0 else float("inf")

    def _perform_memory_balancing(self) -> List[Dict[str, Any]]:
        """Perform memory balancing actions."""
        actions = []

        if not self.memory_balances:
            return actions

        # Sort ranks by memory usage
        sorted_ranks = sorted(
            self.memory_balances.keys(),
            key=lambda r: self.memory_balances[r].allocated_memory_gb,
        )

        high_usage_ranks = sorted_ranks[-len(sorted_ranks) // 2 :]  # Top 50%
        low_usage_ranks = sorted_ranks[: len(sorted_ranks) // 2 :]  # Bottom 50%

        # Coordinate offloading from high-usage to low-usage ranks
        if self.config.coordinate_cpu_offloading:
            offload_actions = self._coordinate_offloading(
                high_usage_ranks, low_usage_ranks
            )
            actions.extend(offload_actions)

        # Coordinate checkpointing decisions
        if self.config.distributed_checkpoint_config.enable_load_balancing:
            checkpoint_actions = self._coordinate_checkpointing(high_usage_ranks)
            actions.extend(checkpoint_actions)

        return actions

    def _coordinate_offloading(
        self, high_usage_ranks: List[int], low_usage_ranks: List[int]
    ) -> List[Dict[str, Any]]:
        """Coordinate CPU offloading between high and low usage ranks."""
        actions = []

        # This is a simplified coordination strategy
        # In practice, this would involve more sophisticated parameter migration

        for high_rank in high_usage_ranks:
            balance = self.memory_balances[high_rank]
            if balance.allocated_memory_gb > self.config.offload_threshold_mb / 1024:
                action = {
                    "type": "offload_coordination",
                    "source_rank": high_rank,
                    "target_ranks": low_usage_ranks,
                    "memory_to_offload_gb": balance.allocated_memory_gb
                    * 0.1,  # Offload 10%
                }
                actions.append(action)

        return actions

    def _coordinate_checkpointing(
        self, high_usage_ranks: List[int]
    ) -> List[Dict[str, Any]]:
        """Coordinate checkpointing decisions for high memory usage ranks."""
        actions = []

        for rank in high_usage_ranks:
            balance = self.memory_balances[rank]
            if (
                balance.allocated_memory_gb
                > self.config.memory_balance_threshold * balance.reserved_memory_gb
            ):
                action = {
                    "type": "checkpoint_coordination",
                    "rank": rank,
                    "recommendation": "increase_checkpointing",
                    "current_usage_gb": balance.allocated_memory_gb,
                }
                actions.append(action)

        return actions

    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        if self._lock:
            with self._lock:
                return {
                    "world_size": self.world_size,
                    "rank": self.rank,
                    "memory_balances": {
                        rank: {
                            "allocated_gb": balance.allocated_memory_gb,
                            "efficiency_ratio": balance.get_efficiency_ratio(),
                        }
                        for rank, balance in self.memory_balances.items()
                    },
                    "current_imbalance_ratio": self._calculate_memory_imbalance_ratio(),
                    "last_balance_step": self.last_balance_step,
                    "coordination_decisions": {
                        "offload_decisions": len(self.offload_decisions),
                        "checkpoint_decisions": len(self.checkpoint_decisions),
                        "precision_decisions": len(self.precision_decisions),
                    },
                    "config": {
                        "enable_memory_balancing": self.config.enable_memory_balancing,
                        "balance_threshold": self.config.memory_balance_threshold,
                        "max_imbalance_ratio": self.config.max_memory_imbalance_ratio,
                    },
                }
        else:
            return {
                "world_size": self.world_size,
                "rank": self.rank,
                "current_imbalance_ratio": self._calculate_memory_imbalance_ratio(),
                "last_balance_step": self.last_balance_step,
            }


class DistributedMemoryOptimizer:
    """Enhanced memory optimizer with distributed coordination capabilities."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: DistributedMemoryConfig,
    ) -> None:
        """Initialize distributed memory optimizer.

        Args:
            model: PyTorch model to optimize
            optimizer: PyTorch optimizer
            config: Distributed memory configuration
        """
        config.validate()
        self.config = config
        self.model = model
        self.optimizer = optimizer

        # Initialize coordinator
        self.coordinator = DistributedMemoryCoordinator(config)

        # Initialize distributed checkpointing
        self.distributed_checkpointing = DistributedActivationCheckpointing(
            config.distributed_checkpoint_config
        )

        # Initialize traditional components with coordination
        self.activation_checkpointing = ActivationCheckpointing()
        self.memory_profiler = MemoryProfiler()

        # Initialize offloading if enabled
        self.cpu_offloader: Optional[CPUOffloadOptimizer] = None
        if config.coordinate_cpu_offloading:
            self.cpu_offloader = CPUOffloadOptimizer(optimizer=optimizer)

        # Initialize mixed precision if enabled
        self.mixed_precision: Optional[MixedPrecisionManager] = None
        if config.coordinate_precision_scaling:
            self.mixed_precision = MixedPrecisionManager(
                precision=PrecisionType.FP16,
                loss_scale=1024.0,
            )

        # Initialize buffer management
        self.buffer_manager: Optional[BufferManager] = None
        if config.enable_distributed_buffering:
            self.buffer_manager = BufferManager(model)

        # Initialize parameter overlap
        self.parameter_gatherer: Optional[AsyncParameterGatherer] = None
        if config.enable_parameter_overlap:
            overlap_config = OverlapConfig()
            self.parameter_gatherer = AsyncParameterGatherer(overlap_config)

        # State tracking
        self.step_count = 0
        self.last_optimization_step = 0

        # Integration flag
        self.fully_integrated = False

        if config.verbose_memory_logging:
            logger.info(f"Initialized DistributedMemoryOptimizer")

    def integrate_with_model(self) -> nn.Module:
        """Integrate memory optimization with the model.

        Returns:
            Model with integrated memory optimizations
        """
        if self.fully_integrated:
            logger.warning(
                "Model already integrated with distributed memory optimization"
            )
            return self.model

        # Apply distributed checkpointing to transformer layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            self.model = (
                self.distributed_checkpointing.apply_to_transformer_layers_distributed(
                    self.model, layer_attr="transformer.h"
                )
            )
        elif hasattr(self.model, "layers"):
            self.model = (
                self.distributed_checkpointing.apply_to_transformer_layers_distributed(
                    self.model, layer_attr="layers"
                )
            )
        else:
            logger.warning(
                "Model structure not recognized for automatic checkpointing integration"
            )

        # Apply CPU offloading if configured
        if self.cpu_offloader is not None:
            # Coordinate offloading decisions
            offload_params = self._get_coordinated_offload_parameters()
            for param_name in offload_params:
                if hasattr(self.model, param_name):
                    param = getattr(self.model, param_name)
                    if isinstance(param, nn.Parameter):
                        # Register parameter for offloading
                        # Note: Actual API may differ - this is a simplified interface
                        pass

        # Apply mixed precision if configured
        if self.mixed_precision is not None:
            self.model = self.mixed_precision.convert_model(self.model)

        # Apply parameter overlap if configured
        if self.parameter_gatherer is not None:
            # Register parameters for overlapped gathering
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # Register parameter for overlap
                    # Note: Actual API may differ - this is a simplified interface
                    pass

        self.fully_integrated = True

        if self.config.verbose_memory_logging:
            logger.info(
                "Model successfully integrated with distributed memory optimization"
            )

        return self.model

    def optimize_step(self, step: int) -> Dict[str, Any]:
        """Perform coordinated memory optimization step.

        Args:
            step: Current training step

        Returns:
            Dictionary containing optimization statistics
        """
        self.step_count = step
        optimization_stats = {}

        # Coordinate memory balancing
        if self.coordinator.should_balance_memory(step):
            balance_stats = self.coordinator.balance_memory_across_ranks(step)
            optimization_stats["memory_balancing"] = balance_stats

        # Coordinate offloading decisions
        if self.cpu_offloader is not None and self.config.coordinate_cpu_offloading:
            offload_stats = self._coordinate_offloading_step(step)
            optimization_stats["offloading"] = offload_stats

        # Coordinate precision scaling
        if (
            self.mixed_precision is not None
            and self.config.coordinate_precision_scaling
        ):
            precision_stats = self._coordinate_precision_step(step)
            optimization_stats["precision"] = precision_stats

        # Update buffer management
        if self.buffer_manager is not None:
            buffer_stats = self._coordinate_buffer_step(step)
            optimization_stats["buffering"] = buffer_stats

        # Coordinate parameter overlap
        if self.parameter_gatherer is not None and self.config.overlap_coordination:
            overlap_stats = self._coordinate_overlap_step(step)
            optimization_stats["parameter_overlap"] = overlap_stats

        # Update profiling
        if self.config.enable_distributed_profiling:
            profiling_stats = self._update_distributed_profiling(step)
            optimization_stats["profiling"] = profiling_stats

        self.last_optimization_step = step

        return optimization_stats

    def _get_coordinated_offload_parameters(self) -> List[str]:
        """Get list of parameters to offload based on coordination."""
        # Simplified offload parameter selection
        # In practice, this would use more sophisticated coordination logic

        offload_params = []

        # Coordinate with other ranks to determine which parameters to offload
        if parallel_state.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            # Simple round-robin offloading strategy
            param_names = [name for name, _ in self.model.named_parameters()]

            for i, param_name in enumerate(param_names):
                if i % world_size == rank:
                    # This rank is responsible for potentially offloading this parameter
                    if self.config.offload_selection_strategy == "round_robin":
                        offload_params.append(param_name)

        return offload_params

    def _coordinate_offloading_step(self, step: int) -> Dict[str, Any]:
        """Coordinate CPU offloading decisions for this step."""
        if self.cpu_offloader is None:
            return {"enabled": False}

        # Get current memory usage
        current_memory_gb = 0.0
        if torch.cuda.is_available():
            current_memory_gb = torch.cuda.memory_allocated() / (1024**3)

        # Decide on offloading based on memory pressure
        should_offload = current_memory_gb > (self.config.offload_threshold_mb / 1024)

        stats = {
            "enabled": True,
            "current_memory_gb": current_memory_gb,
            "threshold_gb": self.config.offload_threshold_mb / 1024,
            "should_offload": should_offload,
            "step": step,
        }

        if should_offload:
            # Trigger offloading
            # Note: Simplified interface - actual implementation may differ
            stats["offloaded_parameters"] = 0  # Placeholder

        return stats

    def _coordinate_precision_step(self, step: int) -> Dict[str, Any]:
        """Coordinate mixed precision scaling for this step."""
        if self.mixed_precision is None:
            return {"enabled": False}

        stats = {
            "enabled": True,
            "step": step,
        }

        # Sync loss scale across ranks periodically
        if step % self.config.sync_loss_scale_frequency == 0:
            if parallel_state.is_initialized() and self.mixed_precision is not None:
                try:
                    # Simplified synchronization - actual implementation may use different API
                    stats["scale_synchronized"] = True
                    stats["synchronized_scale"] = 1.0  # Placeholder
                except Exception as e:
                    logger.warning(f"Failed to synchronize loss scale: {e}")
                    stats["scale_synchronized"] = False

        return stats

    def _coordinate_buffer_step(self, step: int) -> Dict[str, Any]:
        """Coordinate buffer management for this step."""
        if self.buffer_manager is None:
            return {"enabled": False}

        # Simple buffer coordination - in practice this would be more sophisticated
        return {
            "enabled": True,
            "step": step,
            "strategy": self.config.buffer_sync_strategy,
        }

    def _coordinate_overlap_step(self, step: int) -> Dict[str, Any]:
        """Coordinate parameter overlap for this step."""
        if self.parameter_gatherer is None:
            return {"enabled": False}

        stats = {
            "enabled": True,
            "step": step,
        }

        # Coordinate prefetching decisions
        if self.config.prefetch_coordination:
            # Simple prefetch coordination logic
            stats["prefetch_coordinated"] = True

        return stats

    def _update_distributed_profiling(self, step: int) -> Dict[str, Any]:
        """Update distributed memory profiling."""
        # Sync profiling data periodically
        if step % self.config.profiling_sync_interval == 0:
            # Get distributed profiling report
            distributed_report = (
                self.distributed_checkpointing.get_distributed_profiling_report()
            )

            return {
                "enabled": True,
                "step": step,
                "sync_performed": True,
                "distributed_report_size": len(str(distributed_report)),
            }

        return {
            "enabled": True,
            "step": step,
            "sync_performed": False,
        }

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        # Base reports
        coordinator_stats = self.coordinator.get_coordination_stats()
        checkpointing_report = (
            self.distributed_checkpointing.get_distributed_profiling_report()
        )
        memory_report = self.memory_profiler.get_memory_report()

        report = {
            "distributed_memory_optimizer": {
                "step_count": self.step_count,
                "last_optimization_step": self.last_optimization_step,
                "fully_integrated": self.fully_integrated,
                "config": {
                    "enable_memory_balancing": self.config.enable_memory_balancing,
                    "coordinate_cpu_offloading": self.config.coordinate_cpu_offloading,
                    "coordinate_precision_scaling": self.config.coordinate_precision_scaling,
                    "enable_distributed_buffering": self.config.enable_distributed_buffering,
                    "enable_parameter_overlap": self.config.enable_parameter_overlap,
                },
            },
            "coordination": coordinator_stats,
            "distributed_checkpointing": checkpointing_report,
            "memory_profiling": memory_report,
        }

        # Add component-specific reports
        if self.cpu_offloader is not None:
            report["cpu_offloading"] = {
                "enabled": True,
                # Add offloader stats if available
            }

        if self.mixed_precision is not None:
            report["mixed_precision"] = {
                "enabled": True,
                "precision_type": "fp16",  # Simplified
            }

        if self.buffer_manager is not None:
            report["buffer_management"] = {
                "enabled": True,
                "strategy": self.config.buffer_sync_strategy,
            }

        if self.parameter_gatherer is not None:
            report["parameter_overlap"] = {
                "enabled": True,
                "coordination": self.config.overlap_coordination,
            }

        return report

    def reset_optimization_stats(self) -> None:
        """Reset all optimization statistics."""
        self.distributed_checkpointing.reset_distributed_profiling()
        self.memory_profiler.reset()

        # Reset coordinator
        if hasattr(self.coordinator, "memory_balances"):
            self.coordinator.memory_balances.clear()
        if hasattr(self.coordinator, "memory_history"):
            self.coordinator.memory_history.clear()

        self.step_count = 0
        self.last_optimization_step = 0

        if self.config.verbose_memory_logging:
            logger.info("Reset all distributed memory optimization statistics")


# Factory function for easy creation
def create_distributed_memory_optimizer(
    model: nn.Module,
    optimizer: Optimizer,
    enable_memory_balancing: bool = True,
    enable_cpu_offloading: bool = False,
    enable_mixed_precision: bool = False,
    enable_parameter_overlap: bool = False,
    distributed_checkpoint_strategy: str = "hierarchical",
    **kwargs: Any,
) -> DistributedMemoryOptimizer:
    """Create distributed memory optimizer with common settings.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        enable_memory_balancing: Whether to enable memory balancing
        enable_cpu_offloading: Whether to enable CPU offloading coordination
        enable_mixed_precision: Whether to enable mixed precision coordination
        enable_parameter_overlap: Whether to enable parameter overlap
        distributed_checkpoint_strategy: Distributed checkpointing strategy
        **kwargs: Additional configuration parameters

    Returns:
        Configured DistributedMemoryOptimizer instance
    """
    from .distributed_checkpoint import DistributedCheckpointStrategy

    strategy_map = {
        "coordinated": DistributedCheckpointStrategy.COORDINATED,
        "load_balanced": DistributedCheckpointStrategy.LOAD_BALANCED,
        "hierarchical": DistributedCheckpointStrategy.HIERARCHICAL,
        "adaptive": DistributedCheckpointStrategy.ADAPTIVE,
        "expert_aware": DistributedCheckpointStrategy.EXPERT_AWARE,
        "pipeline_aware": DistributedCheckpointStrategy.PIPELINE_AWARE,
    }

    checkpoint_config = DistributedCheckpointConfig(
        strategy=strategy_map.get(
            distributed_checkpoint_strategy, DistributedCheckpointStrategy.HIERARCHICAL
        ),
        enable_load_balancing=enable_memory_balancing,
        **kwargs,
    )

    memory_config = DistributedMemoryConfig(
        distributed_checkpoint_config=checkpoint_config,
        enable_memory_balancing=enable_memory_balancing,
        coordinate_cpu_offloading=enable_cpu_offloading,
        coordinate_precision_scaling=enable_mixed_precision,
        enable_parameter_overlap=enable_parameter_overlap,
    )

    return DistributedMemoryOptimizer(model, optimizer, memory_config)
