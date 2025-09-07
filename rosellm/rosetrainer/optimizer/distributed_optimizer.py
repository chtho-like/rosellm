"""
Distributed Optimizer with Gradient Bucketing

This module implements a distributed optimizer wrapper that provides efficient
gradient reduction across data parallel ranks with memory optimization through
gradient bucketing and communication-computation overlap.

Key Features:
- Wraps any base optimizer (Adam, AdamW, SGD, etc.)
- Gradient bucketing for reduced memory fragmentation
- Asynchronous gradient reduction with backward computation overlap
- Optimizer state partitioning across DP ranks for memory efficiency
- Support for mixed precision training
"""

import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch.nn import Parameter
from torch.optim import Optimizer

from rosellm.rosetrainer.optimizer.exceptions import (
    CommunicationError,
    ConfigurationError,
    SynchronizationError,
)
from rosellm.rosetrainer.optimizer.gradient_buffer import GradientBuffer
from rosellm.rosetrainer.optimizer.metrics import PerformanceMonitor
from rosellm.rosetrainer.optimizer.partitioning_strategies import (
    PartitioningStrategyFactory,
)
from rosellm.rosetrainer.parallelism import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_size,
)

logger = logging.getLogger(__name__)


class DistributedOptimizer(Optimizer):
    """
    Distributed optimizer wrapper with gradient bucketing and communication overlap.

    This optimizer wraps any base optimizer and adds distributed training capabilities:
    - Gradient bucketing to reduce communication operations
    - Asynchronous gradient reduction overlapped with backward pass
    - Optimizer state partitioning across data parallel ranks
    - Memory-efficient gradient buffer management

    Args:
        optimizer: Base optimizer instance to wrap
        models: Model or list of models whose parameters to optimize
        bucket_size_mb: Target bucket size in megabytes (default: 25MB)
        overlap_grad_reduce: Enable gradient reduction overlap (default: True)
        partition_optimizer_states: Partition states across DP ranks (default: True)
        gradient_accumulation_steps: Number of gradient accumulation steps
        clip_grad_norm: Maximum gradient norm for clipping (optional)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        models: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
        bucket_size_mb: float = 25.0,
        overlap_grad_reduce: bool = True,
        partition_optimizer_states: bool = True,
        partitioning_strategy: str = "round_robin",
        gradient_accumulation_steps: int = 1,
        clip_grad_norm: Optional[float] = None,
        enable_metrics: bool = False,
    ):
        # Validate inputs
        if gradient_accumulation_steps < 1:
            raise ConfigurationError(
                f"gradient_accumulation_steps must be >= 1, "
                f"got {gradient_accumulation_steps}"
            )
        if bucket_size_mb <= 0:
            raise ConfigurationError(
                f"bucket_size_mb must be positive, got {bucket_size_mb}"
            )
        if clip_grad_norm is not None and clip_grad_norm <= 0:
            raise ConfigurationError(
                f"clip_grad_norm must be positive or None, got {clip_grad_norm}"
            )
        if not isinstance(optimizer, Optimizer):
            raise ConfigurationError(
                f"optimizer must be a torch.optim.Optimizer instance, "
                f"got {type(optimizer)}"
            )

        self.base_optimizer = optimizer
        self.bucket_size_mb = bucket_size_mb
        self.overlap_grad_reduce = overlap_grad_reduce
        self.partition_optimizer_states = partition_optimizer_states
        partitioning_name = partitioning_strategy
        self.partitioning_strategy = PartitioningStrategyFactory.create(
            partitioning_name
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.clip_grad_norm = clip_grad_norm

        # Thread safety
        self._lock = threading.Lock()

        # Get process group info
        self.dp_process_group = get_data_parallel_group()
        self.dp_size = get_data_parallel_size()
        self.dp_rank = get_data_parallel_rank()

        # Track models and parameters
        if models is not None:
            self.models = [models] if not isinstance(models, list) else models
        else:
            # Extract models from optimizer param groups
            self.models = []

        # Get all parameters from optimizer
        self.all_params: List[Parameter] = []
        for group in self.base_optimizer.param_groups:
            self.all_params.extend(group["params"])

        # Initialize gradient buffer if using bucketing
        self.gradient_buffer: Optional[GradientBuffer] = None
        if self.dp_size > 1 and self.overlap_grad_reduce:
            self._init_gradient_buffer()

        # Initialize state partitioning
        self.param_to_rank: Dict[Parameter, int] = {}
        self.rank_to_params: Dict[int, List[Parameter]] = defaultdict(list)
        if self.partition_optimizer_states and self.dp_size > 1:
            self._partition_parameters()

        # Gradient accumulation counter
        self.accumulation_step = 0

        # Performance monitoring
        self.enable_metrics = enable_metrics
        self.performance_monitor: Optional[PerformanceMonitor] = None
        if enable_metrics:
            device = self.all_params[0].device if self.all_params else None
            self.performance_monitor = PerformanceMonitor(device=device)

        # Statistics tracking
        self.stats = {
            "num_bucket_reductions": 0,
            "num_gradient_clips": 0,
            "total_norm": 0.0,
        }

        logger.info(
            f"Initialized DistributedOptimizer: dp_size={self.dp_size}, "
            f"bucket_size_mb={bucket_size_mb}, overlap={overlap_grad_reduce}, "
            f"partition_states={partition_optimizer_states}"
        )

    def _init_gradient_buffer(self) -> None:
        """Initialize gradient buffer for bucketing"""
        # Determine dtype for gradient buffer
        dtype = torch.float32
        if len(self.all_params) > 0:
            dtype = self.all_params[0].dtype

        # Create gradient buffer
        self.gradient_buffer = GradientBuffer(
            params=self.all_params,
            bucket_size_mb=self.bucket_size_mb,
            dtype=dtype,
            device=self.all_params[0].device if self.all_params else None,
            process_group=self.dp_process_group,
        )

        logger.info(f"Gradient buffer initialized: {self.gradient_buffer}")

    def _partition_parameters(self) -> None:
        """Partition parameters across data parallel ranks for state sharding"""
        if not self.all_params:
            return

        # Use strategy pattern for partitioning
        rank_to_params = self.partitioning_strategy.partition(
            self.all_params, self.dp_size
        )

        # Build mappings
        self.rank_to_params = rank_to_params
        for rank, params in rank_to_params.items():
            for param in params:
                self.param_to_rank[param] = rank

        # Log partitioning info
        local_params = len(self.rank_to_params[self.dp_rank])
        total_params = len(self.all_params)
        strategy_name = self.partitioning_strategy.get_name()
        logger.info(
            f"Parameter partitioning ({strategy_name}): rank {self.dp_rank} owns "
            f"{local_params}/{total_params} parameters"
        )

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero out gradients, optionally setting to None"""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

        # Reset gradient buffer if using bucketing
        if self.gradient_buffer is not None:
            self.gradient_buffer.reset()

    def _clip_gradients(self) -> float:
        """Clip gradients and return total norm"""
        if self.clip_grad_norm is None:
            return 0.0

        # Compute total gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.all_params, self.clip_grad_norm
        )

        self.stats["num_gradient_clips"] += 1
        norm_value = (
            total_norm.item()
            if isinstance(total_norm, torch.Tensor)
            else float(total_norm)
        )
        self.stats["total_norm"] = norm_value

        if self.performance_monitor:
            self.performance_monitor.record_gradient_norm(norm_value, clipped=True)

        norm_val: float = (
            total_norm.item()
            if isinstance(total_norm, torch.Tensor)
            else float(total_norm)
        )
        return norm_val

    def _reduce_gradients(self) -> None:
        """Reduce gradients across data parallel ranks with error handling"""
        if self.dp_size <= 1:
            return

        if self.performance_monitor:
            self.performance_monitor.start_timer("gradient_reduction")

        try:
            if self.gradient_buffer is not None and self.overlap_grad_reduce:
                # Use bucketed reduction (already happening via hooks)
                self.gradient_buffer.synchronize_all_buckets()
                self.stats["num_bucket_reductions"] += len(self.gradient_buffer.buckets)
            else:
                # Fallback to parameter-wise all-reduce
                for param in self.all_params:
                    if param.grad is not None:
                        dist.all_reduce(param.grad, group=self.dp_process_group)
                        param.grad.div_(self.dp_size)

            # Synchronization barrier to ensure all ranks complete reduction
            if dist.is_initialized():
                try:
                    dist.barrier(group=self.dp_process_group)
                except Exception as barrier_error:
                    raise SynchronizationError(
                        "Rank synchronization failed after gradient "
                        f"reduction: {barrier_error}"
                    ) from barrier_error

        except SynchronizationError:
            raise
        except Exception as e:
            logger.error(f"Gradient reduction failed: {e}")
            raise CommunicationError(
                f"Failed to reduce gradients across ranks: {e}"
            ) from e
        finally:
            if self.performance_monitor:
                duration = self.performance_monitor.end_timer("gradient_reduction")
                bytes_reduced = sum(
                    p.grad.numel() * p.grad.element_size()
                    for p in self.all_params
                    if p.grad is not None
                )
                self.performance_monitor.record_gradient_reduction(
                    duration, len(self.all_params), bytes_reduced
                )

    def step(  # type: ignore[override]
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Increment accumulation counter
        self.accumulation_step += 1

        # Only step when we've accumulated enough gradients
        if self.accumulation_step < self.gradient_accumulation_steps:
            return loss

        # Reset accumulation counter
        self.accumulation_step = 0

        if self.performance_monitor:
            self.performance_monitor.start_timer("step")

        with self._lock:
            # Ensure all gradient reductions are complete
            self._reduce_gradients()

            # Scale gradients by accumulation steps
            if self.gradient_accumulation_steps > 1:
                for param in self.all_params:
                    if param.grad is not None:
                        param.grad.div_(self.gradient_accumulation_steps)

            # Clip gradients if needed
            if self.clip_grad_norm is not None:
                self._clip_gradients()

            # Perform optimizer step
            if self.performance_monitor:
                self.performance_monitor.start_timer("parameter_update")

            try:
                if self.partition_optimizer_states and self.dp_size > 1:
                    # Only update parameters assigned to this rank
                    self._partitioned_step()
                else:
                    # Standard optimizer step
                    self.base_optimizer.step()
            finally:
                if self.performance_monitor:
                    duration = self.performance_monitor.end_timer("parameter_update")
                    self.performance_monitor.record_parameter_update(duration)

        if self.performance_monitor:
            self.performance_monitor.step()

        return loss

    def _partitioned_step(self) -> None:
        """Perform optimizer step only on parameters assigned to this rank"""
        # Create temporary param groups with only local parameters
        local_param_groups = []
        for group in self.base_optimizer.param_groups:
            local_params = [
                p
                for p in group["params"]
                if self.param_to_rank.get(p, 0) == self.dp_rank
            ]
            if local_params:
                local_group = {**group, "params": local_params}
                local_param_groups.append(local_group)

        # Temporarily replace param groups
        original_groups = self.base_optimizer.param_groups
        self.base_optimizer.param_groups = local_param_groups

        # Perform step
        self.base_optimizer.step()

        # Restore original param groups
        self.base_optimizer.param_groups = original_groups

        # Broadcast updated parameters from owner ranks
        self._broadcast_parameters()

    def _broadcast_parameters(self) -> None:
        """Efficiently broadcast parameters from owner ranks to all ranks"""
        if self.dp_size <= 1:
            return

        if self.performance_monitor:
            self.performance_monitor.start_timer("broadcast")

        try:
            # Group parameters by owner rank for efficient broadcasting
            rank_to_broadcast_params: Dict[int, List[Parameter]] = defaultdict(list)
            for param in self.all_params:
                owner_rank = self.param_to_rank.get(param, 0)
                rank_to_broadcast_params[owner_rank].append(param)

            # Broadcast parameters in batches per rank
            for owner_rank, params in rank_to_broadcast_params.items():
                # Flatten parameters for this rank
                flat_params = torch.cat([p.data.flatten() for p in params])

                # Broadcast flattened tensor
                dist.broadcast(flat_params, src=owner_rank, group=self.dp_process_group)

                # Unflatten and copy back to parameters
                offset = 0
                for param in params:
                    param_size = param.numel()
                    param.data.copy_(
                        flat_params[offset : offset + param_size].view_as(param)
                    )
                    offset += param_size

            # Ensure all ranks have received updates
            if dist.is_initialized():
                try:
                    dist.barrier(group=self.dp_process_group)
                except Exception as barrier_error:
                    raise SynchronizationError(
                        f"Rank synchronization failed after parameter "
                        f"broadcast: {barrier_error}"
                    ) from barrier_error

        except SynchronizationError:
            raise
        except Exception as e:
            logger.error(f"Parameter broadcasting failed: {e}")
            raise CommunicationError(f"Failed to broadcast parameters: {e}") from e
        finally:
            if self.performance_monitor:
                duration = self.performance_monitor.end_timer("broadcast")
                bytes_broadcast = sum(
                    p.numel() * p.element_size() for p in self.all_params
                )
                self.performance_monitor.record_broadcast(duration, bytes_broadcast)

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state dict"""
        state_dict = {
            "base_optimizer": self.base_optimizer.state_dict(),
            "accumulation_step": self.accumulation_step,
            "stats": self.stats,
        }

        # Add configuration
        state_dict["config"] = {
            "bucket_size_mb": self.bucket_size_mb,
            "overlap_grad_reduce": self.overlap_grad_reduce,
            "partition_optimizer_states": self.partition_optimizer_states,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "clip_grad_norm": self.clip_grad_norm,
        }

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state dict"""
        # Load base optimizer state
        if "base_optimizer" in state_dict:
            self.base_optimizer.load_state_dict(state_dict["base_optimizer"])

        # Load other state
        if "accumulation_step" in state_dict:
            self.accumulation_step = state_dict["accumulation_step"]
        if "stats" in state_dict:
            self.stats.update(state_dict["stats"])

        # Verify configuration matches
        if "config" in state_dict:
            config = state_dict["config"]
            if config.get("bucket_size_mb") != self.bucket_size_mb:
                logger.warning(
                    f"Bucket size mismatch: saved={config.get('bucket_size_mb')}, "
                    f"current={self.bucket_size_mb}"
                )

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Return parameter groups from base optimizer"""
        groups: List[Dict[str, Any]] = self.base_optimizer.param_groups
        return groups

    @param_groups.setter
    def param_groups(self, value: List[Dict[str, Any]]) -> None:
        """Set parameter groups on base optimizer"""
        self.base_optimizer.param_groups = value

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a parameter group to the optimizer"""
        self.base_optimizer.add_param_group(param_group)

        # Update internal tracking
        if "params" in param_group:
            new_params = param_group["params"]
            self.all_params.extend(new_params)

            # Update partitioning if needed
            if self.partition_optimizer_states and self.dp_size > 1:
                for idx, param in enumerate(
                    new_params, start=len(self.all_params) - len(new_params)
                ):
                    assigned_rank = idx % self.dp_size
                    self.param_to_rank[param] = assigned_rank
                    self.rank_to_params[assigned_rank].append(param)

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        stats: Dict[str, Any] = dict(self.stats)

        # Add performance metrics if available
        if self.performance_monitor:
            stats[
                "performance"
            ] = self.performance_monitor.get_current_metrics().to_dict()
            stats["performance_avg"] = self.performance_monitor.get_average_metrics()

        # Add gradient buffer stats if available
        if self.gradient_buffer is not None:
            stats["gradient_buffer"] = self.gradient_buffer.get_bucket_info()

        # Add memory stats
        if self.partition_optimizer_states:
            stats["local_params"] = len(self.rank_to_params[self.dp_rank])
            stats["total_params"] = len(self.all_params)

        return stats

    def __repr__(self) -> str:
        return (
            f"DistributedOptimizer(base={self.base_optimizer.__class__.__name__}, "
            f"dp_size={self.dp_size}, bucket_size_mb={self.bucket_size_mb}, "
            f"overlap={self.overlap_grad_reduce}, "
            f"partition={self.partition_optimizer_states})"
        )
