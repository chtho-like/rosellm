"""Gradient synchronization strategies for multi-dimensional parallelism."""

import logging
import time
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

# from ..parallelism import parallel_state  # Not currently used
from ..utils.gradient_utils import calculate_gradient_norm_multitensor

logger = logging.getLogger(__name__)


class GradientSyncStrategy(ABC):
    """Abstract base class for gradient synchronization strategies."""

    def __init__(self, config: Any) -> None:
        """Initialize gradient sync strategy.

        Args:
            config: Gradient finalization configuration.
        """
        self.config = config
        self.stats: Dict[str, Any] = {}
        self._reduction_op_cache: Optional[Any] = None

    @abstractmethod
    def sync_gradients(
        self,
        model: nn.Module,
        process_groups: Dict[str, Optional[dist.ProcessGroup]],
    ) -> Dict[str, Any]:
        """Synchronize gradients across process groups.

        Args:
            model: Model with gradients to synchronize.
            process_groups: Dictionary of process groups for each dimension.

        Returns:
            Dictionary with synchronization statistics.
        """
        pass

    def _get_reduction_op(
        self,
    ) -> Any:  # Returns ReduceOp but type varies by PyTorch version
        """Get the reduction operation from config.

        Returns:
            DistReduce operation.
        """
        if self._reduction_op_cache is not None:
            return self._reduction_op_cache

        op_map = {
            "sum": dist.ReduceOp.SUM,
            "mean": dist.ReduceOp.SUM,  # Handle mean separately with division
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }
        self._reduction_op_cache = op_map.get(
            self.config.reduction_op, dist.ReduceOp.SUM
        )
        return self._reduction_op_cache

    def _scale_gradients(
        self, parameters: List[nn.Parameter], scale_factor: float
    ) -> None:
        """Scale gradients by a factor.

        Args:
            parameters: Parameters with gradients to scale.
            scale_factor: Factor to scale gradients by.
        """
        if scale_factor == 1.0:
            return

        # Vectorized operation for efficiency
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(scale_factor)

    def _get_reduction_order(
        self, process_groups: Dict[str, Optional[dist.ProcessGroup]]
    ) -> List[Tuple[str, Optional[dist.ProcessGroup]]]:
        """Get the order of dimensions for reduction.

        Args:
            process_groups: Dictionary of process groups.

        Returns:
            List of (dimension_name, process_group) tuples in reduction order.
        """
        if self.config.dimension_order == "tp-pp-dp-cp-ep":
            order = ["tp", "pp", "dp", "cp", "ep"]
        elif self.config.dimension_order == "dp-tp-pp-cp-ep":
            order = ["dp", "tp", "pp", "cp", "ep"]
        elif (
            self.config.dimension_order == "custom"
            and self.config.custom_dimension_order
        ):
            order = self.config.custom_dimension_order
        elif self.config.dimension_order == "hierarchical":
            # Flatten hierarchical levels for simple strategies
            order = []
            for level in self.config.hierarchical_levels:
                order.extend(level)
        else:
            # Default order
            order = ["tp", "pp", "dp", "cp", "ep"]

        return [(dim, process_groups.get(dim)) for dim in order]

    def _batch_all_reduce(
        self,
        tensors: List[torch.Tensor],
        process_group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Optional[Any]:
        """Perform batched all-reduce for efficiency.

        Args:
            tensors: List of tensors to reduce.
            process_group: Process group for reduction.
            async_op: Whether to perform async operation.

        Returns:
            Handle for async operation if async_op is True.
        """
        if not tensors:
            return None

        reduction_op = self._get_reduction_op()
        world_size = dist.get_world_size(process_group)

        # Batch small tensors together for efficiency
        small_tensors = [t for t in tensors if t.numel() < 1024]
        large_tensors = [t for t in tensors if t.numel() >= 1024]

        handles = []

        # Reduce large tensors individually
        for tensor in large_tensors:
            handle = dist.all_reduce(
                tensor, op=reduction_op, group=process_group, async_op=async_op
            )
            if async_op and handle:
                handles.append(handle)
            elif self.config.reduction_op == "mean":
                tensor.div_(world_size)

        # Batch small tensors
        if small_tensors:
            # Flatten and concatenate
            flat_tensors = [t.view(-1) for t in small_tensors]
            combined = torch.cat(flat_tensors)

            handle = dist.all_reduce(
                combined, op=reduction_op, group=process_group, async_op=async_op
            )

            if async_op and handle:
                handles.append((handle, combined, small_tensors))
            else:
                # Unpack immediately
                if self.config.reduction_op == "mean":
                    combined.div_(world_size)

                offset = 0
                for tensor in small_tensors:
                    numel = tensor.numel()
                    tensor.copy_(combined[offset : offset + numel].view_as(tensor))
                    offset += numel

        return handles if async_op else None


class SimpleGradientSync(GradientSyncStrategy):
    """Simple gradient synchronization using all-reduce."""

    def sync_gradients(
        self,
        model: nn.Module,
        process_groups: Dict[str, Optional[dist.ProcessGroup]],
    ) -> Dict[str, Any]:
        """Synchronize gradients with simple all-reduce.

        Args:
            model: Model with gradients to synchronize.
            process_groups: Dictionary of process groups for each dimension.

        Returns:
            Dictionary with synchronization statistics.
        """
        stats = {
            "sync_time": 0.0,
            "num_params_synced": 0,
            "total_gradient_norm": 0.0,
        }

        start_time = time.perf_counter()

        # Get parameters with gradients
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        stats["num_params_synced"] = len(params_with_grad)

        if not params_with_grad:
            return stats

        # Apply pre-divide factor
        if self.config.gradient_predivide_factor != 1.0:
            self._scale_gradients(
                params_with_grad, self.config.gradient_predivide_factor
            )

        # Determine reduction order based on configuration
        reduction_order = self._get_reduction_order(process_groups)

        # Perform reductions in order
        for dim_name, group in reduction_order:
            if group is not None:
                self._reduce_gradients_simple(params_with_grad, group, dim_name)

        # Apply post-divide factor
        if self.config.gradient_postdivide_factor != 1.0:
            self._scale_gradients(
                params_with_grad, self.config.gradient_postdivide_factor
            )

        # Calculate gradient norm if requested
        if self.config.check_gradient_norm:
            grad_norm = calculate_gradient_norm_multitensor(
                params_with_grad,
                norm_type=self.config.gradient_norm_type,
                model_parallel_reduce=False,  # Already reduced
            )
            stats["total_gradient_norm"] = float(grad_norm)

        stats["sync_time"] = time.perf_counter() - start_time
        return stats

    def _reduce_gradients_simple(
        self,
        parameters: List[nn.Parameter],
        process_group: dist.ProcessGroup,
        dim_name: str,
    ) -> None:
        """Reduce gradients across a process group.

        Args:
            parameters: Parameters with gradients to reduce.
            process_group: Process group for reduction.
            dim_name: Name of the dimension being reduced.
        """
        world_size = dist.get_world_size(process_group)
        if world_size == 1:
            return

        try:
            # Use batched all-reduce for efficiency
            gradients = [p.grad for p in parameters if p.grad is not None]
            self._batch_all_reduce(gradients, process_group, async_op=False)

        except RuntimeError as e:
            logger.error(f"Failed to reduce gradient for {dim_name}: {e}")
            raise RuntimeError(f"Gradient reduction failed for {dim_name}") from e


class BucketedGradientSync(GradientSyncStrategy):
    """Bucketed gradient synchronization for improved efficiency."""

    def __init__(self, config: Any) -> None:
        """Initialize bucketed gradient sync.

        Args:
            config: Gradient finalization configuration.
        """
        super().__init__(config)
        self.buckets: List[List[nn.Parameter]] = []
        self.bucket_buffers: List[Optional[torch.Tensor]] = []

    def sync_gradients(
        self,
        model: nn.Module,
        process_groups: Dict[str, Optional[dist.ProcessGroup]],
    ) -> Dict[str, Any]:
        """Synchronize gradients using bucketed all-reduce.

        Args:
            model: Model with gradients to synchronize.
            process_groups: Dictionary of process groups for each dimension.

        Returns:
            Dictionary with synchronization statistics.
        """
        stats = {
            "sync_time": 0.0,
            "num_params_synced": 0,
            "num_buckets": 0,
            "total_gradient_norm": 0.0,
        }

        start_time = time.perf_counter()

        # Get parameters with gradients
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        stats["num_params_synced"] = len(params_with_grad)

        if not params_with_grad:
            return stats

        # Create buckets if not already created
        if not self.buckets:
            self._create_buckets(params_with_grad)

        stats["num_buckets"] = len(self.buckets)

        # Apply pre-divide factor
        if self.config.gradient_predivide_factor != 1.0:
            self._scale_gradients(
                params_with_grad, self.config.gradient_predivide_factor
            )

        # Determine reduction order
        reduction_order = self._get_reduction_order(process_groups)

        # Perform bucketed reductions
        for dim_name, group in reduction_order:
            if group is not None:
                self._reduce_gradients_bucketed(group, dim_name)

        # Apply post-divide factor
        if self.config.gradient_postdivide_factor != 1.0:
            self._scale_gradients(
                params_with_grad, self.config.gradient_postdivide_factor
            )

        # Calculate gradient norm if requested
        if self.config.check_gradient_norm:
            grad_norm = calculate_gradient_norm_multitensor(
                params_with_grad,
                norm_type=self.config.gradient_norm_type,
                model_parallel_reduce=False,
            )
            stats["total_gradient_norm"] = float(grad_norm)

        stats["sync_time"] = time.perf_counter() - start_time
        return stats

    def _create_buckets(self, parameters: List[nn.Parameter]) -> None:
        """Create parameter buckets for efficient communication.

        Args:
            parameters: Parameters to bucket.
        """
        bucket_size_bytes = int(self.config.bucket_size_mb * 1024 * 1024)
        bucket_cap_bytes = int(self.config.bucket_cap_mb * 1024 * 1024)

        current_bucket: List[nn.Parameter] = []
        current_size = 0

        for param in parameters:
            param_size = param.numel() * param.element_size()

            # Check if adding this param exceeds bucket cap
            if current_size + param_size > bucket_cap_bytes and current_bucket:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0

            current_bucket.append(param)
            current_size += param_size

            # Check if bucket is full
            if current_size >= bucket_size_bytes:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0

        # Add remaining parameters
        if current_bucket:
            self.buckets.append(current_bucket)

        # Create contiguous buffers if configured
        if self.config.use_contiguous_buffers:
            self._create_contiguous_buffers()

    def _create_contiguous_buffers(self) -> None:
        """Create contiguous buffers for each bucket."""
        self.bucket_buffers = []

        for bucket in self.buckets:
            if not bucket:
                self.bucket_buffers.append(None)
                continue

            # Calculate total size
            total_numel = sum(p.grad.numel() for p in bucket if p.grad is not None)
            if total_numel == 0:
                self.bucket_buffers.append(None)
                continue

            # Create buffer with same dtype and device as first parameter
            dtype = bucket[0].dtype
            device = bucket[0].device

            # Use FP16 compression if configured
            if self.config.fp16_compression and dtype == torch.float32:
                dtype = torch.float16

            buffer = torch.zeros(total_numel, dtype=dtype, device=device)
            self.bucket_buffers.append(buffer)

    def _reduce_gradients_bucketed(
        self,
        process_group: dist.ProcessGroup,
        dim_name: str,
    ) -> None:
        """Reduce gradients using buckets.

        Args:
            process_group: Process group for reduction.
            dim_name: Name of the dimension being reduced.
        """
        world_size = dist.get_world_size(process_group)
        if world_size == 1:
            return

        reduction_op = self._get_reduction_op()
        handles = []

        for bucket_idx, bucket in enumerate(self.buckets):
            if not bucket:
                continue

            # Use contiguous buffer if available
            if (
                self.config.use_contiguous_buffers
                and self.bucket_buffers[bucket_idx] is not None
            ):
                buffer = self.bucket_buffers[bucket_idx]
                assert buffer is not None  # Type narrowing

                # Pack gradients into buffer
                offset = 0
                for param in bucket:
                    if param.grad is not None:
                        numel = param.grad.numel()
                        if (
                            self.config.fp16_compression
                            and param.dtype == torch.float32
                        ):
                            buffer[offset : offset + numel].copy_(
                                param.grad.view(-1).half()
                            )
                        else:
                            buffer[offset : offset + numel].copy_(param.grad.view(-1))
                        offset += numel

                # Launch async all-reduce
                if self.config.enable_async_grad_sync:
                    handle = dist.all_reduce(
                        buffer, op=reduction_op, group=process_group, async_op=True
                    )
                    handles.append((handle, buffer, bucket))
                else:
                    dist.all_reduce(buffer, op=reduction_op, group=process_group)
                    self._unpack_buffer(buffer, bucket, world_size)
            else:
                # Reduce individual gradients
                for param in bucket:
                    if param.grad is not None:
                        if self.config.enable_async_grad_sync:
                            handle = dist.all_reduce(
                                param.grad,
                                op=reduction_op,
                                group=process_group,
                                async_op=True,
                            )
                            handles.append((handle, param.grad, [param]))
                        else:
                            dist.all_reduce(
                                param.grad, op=reduction_op, group=process_group
                            )
                            if self.config.reduction_op == "mean":
                                param.grad.div_(world_size)

        # Wait for async operations and unpack buffers
        if handles:
            try:
                for item in handles:
                    if isinstance(item, tuple):
                        handle, data, params = item
                        handle.wait(
                            timeout=timedelta(seconds=self.config.sync_timeout_seconds)
                        )
                        if isinstance(data, torch.Tensor) and len(params) > 1:
                            # Unpack buffer
                            self._unpack_buffer(data, params, world_size)
                        elif self.config.reduction_op == "mean":
                            # Scale single gradient
                            data.div_(world_size)
                    else:
                        # Single handle - shouldn't happen with current logic
                        item.wait(
                            timeout=timedelta(seconds=self.config.sync_timeout_seconds)
                        )
            except RuntimeError as e:
                logger.error(f"Async operation timeout: {e}")
                # Attempt to recover by syncing
                dist.barrier(group=process_group)
                raise

    def _unpack_buffer(
        self,
        buffer: torch.Tensor,
        parameters: List[nn.Parameter],
        world_size: int,
    ) -> None:
        """Unpack gradients from contiguous buffer.

        Args:
            buffer: Contiguous buffer with reduced gradients.
            parameters: Parameters to unpack into.
            world_size: World size for mean reduction.
        """
        offset = 0
        for param in parameters:
            if param.grad is not None:
                numel = param.grad.numel()
                grad_slice = buffer[offset : offset + numel].view_as(param.grad)

                # Handle FP16 decompression
                if self.config.fp16_compression and param.dtype == torch.float32:
                    param.grad.copy_(grad_slice.float())
                else:
                    param.grad.copy_(grad_slice)

                # Handle mean reduction
                if self.config.reduction_op == "mean":
                    param.grad.div_(world_size)

                offset += numel


class HierarchicalGradientSync(GradientSyncStrategy):
    """Hierarchical gradient synchronization for large-scale training."""

    def sync_gradients(
        self,
        model: nn.Module,
        process_groups: Dict[str, Optional[dist.ProcessGroup]],
    ) -> Dict[str, Any]:
        """Synchronize gradients hierarchically across dimensions.

        Args:
            model: Model with gradients to synchronize.
            process_groups: Dictionary of process groups for each dimension.

        Returns:
            Dictionary with synchronization statistics.
        """
        stats: Dict[str, Any] = {
            "sync_time": 0.0,
            "num_params_synced": 0,
            "num_levels": 0,
            "level_times": [],
            "total_gradient_norm": 0.0,
        }

        start_time = time.perf_counter()

        # Get parameters with gradients
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        stats["num_params_synced"] = len(params_with_grad)

        if not params_with_grad:
            return stats

        # Apply pre-divide factor
        if self.config.gradient_predivide_factor != 1.0:
            self._scale_gradients(
                params_with_grad, self.config.gradient_predivide_factor
            )

        # Perform hierarchical reduction
        stats["num_levels"] = len(self.config.hierarchical_levels)

        for level_idx, level_dims in enumerate(self.config.hierarchical_levels):
            level_start = time.perf_counter()

            # Get process groups for this level
            level_groups = []
            for dim in level_dims:
                group = process_groups.get(dim)
                if group is not None:
                    level_groups.append((dim, group))

            # Reduce across all dimensions in this level
            if level_groups:
                self._reduce_level(params_with_grad, level_groups)

            level_time = time.perf_counter() - level_start
            stats["level_times"].append(level_time)

            if self.config.verbose:
                logger.info(
                    f"Hierarchical level {level_idx} ({level_dims}) "
                    f"completed in {level_time:.3f}s"
                )

        # Apply post-divide factor
        if self.config.gradient_postdivide_factor != 1.0:
            self._scale_gradients(
                params_with_grad, self.config.gradient_postdivide_factor
            )

        # Calculate gradient norm if requested
        if self.config.check_gradient_norm:
            grad_norm = calculate_gradient_norm_multitensor(
                params_with_grad,
                norm_type=self.config.gradient_norm_type,
                model_parallel_reduce=False,
            )
            stats["total_gradient_norm"] = float(grad_norm)

        stats["sync_time"] = time.perf_counter() - start_time
        return stats

    def _reduce_level(
        self,
        parameters: List[nn.Parameter],
        level_groups: List[Tuple[str, dist.ProcessGroup]],
    ) -> None:
        """Reduce gradients across all dimensions in a hierarchical level.

        Args:
            parameters: Parameters with gradients to reduce.
            level_groups: List of (dimension_name, process_group) for this level.
        """
        reduction_op = self._get_reduction_op()

        # Handle special cases for different dimensions
        for dim_name, group in level_groups:
            world_size = dist.get_world_size(group)
            if world_size == 1:
                continue

            if (
                dim_name == "ep"
                and self.config.expert_parallel_sync_type == "all_to_all"
            ):
                # Special handling for expert parallelism
                self._reduce_expert_parallel(parameters, group)
            elif dim_name == "cp" and self.config.context_parallel_sync_type == "ring":
                # Special handling for context parallelism
                self._reduce_context_parallel_ring(parameters, group)
            else:
                # Standard all-reduce
                for param in parameters:
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=reduction_op, group=group)
                        if self.config.reduction_op == "mean":
                            param.grad.div_(world_size)

    def _reduce_expert_parallel(
        self,
        parameters: List[nn.Parameter],
        process_group: dist.ProcessGroup,
    ) -> None:
        """Reduce gradients for expert parallelism using all-to-all.

        Args:
            parameters: Parameters with gradients to reduce.
            process_group: Expert parallel process group.
        """
        world_size = dist.get_world_size(process_group)
        if world_size == 1:
            return

        # rank = dist.get_rank(process_group)  # Not used currently

        # Separate expert and non-expert parameters
        expert_params = []
        non_expert_params = []

        for param in parameters:
            if param.grad is None:
                continue
            # Check if parameter is marked as expert (via attribute)
            if hasattr(param, "is_expert_param") and getattr(
                param, "is_expert_param", False
            ):
                expert_params.append(param)
            else:
                non_expert_params.append(param)

        # Standard all-reduce for non-expert parameters
        if non_expert_params:
            non_expert_grads = [p.grad for p in non_expert_params if p.grad is not None]
            self._batch_all_reduce(non_expert_grads, process_group, async_op=False)

        # All-to-all communication for expert parameters
        if expert_params and self.config.expert_parallel_sync_type == "all_to_all":
            for param in expert_params:
                if param.grad is not None:
                    # Split gradient for all-to-all
                    grad_chunks = param.grad.chunk(world_size, dim=0)
                    output_chunks = [torch.empty_like(chunk) for chunk in grad_chunks]

                    # Perform all-to-all
                    dist.all_to_all(
                        output_chunks, list(grad_chunks), group=process_group
                    )

                    # Concatenate and average
                    param.grad = torch.cat(output_chunks, dim=0)
                    if self.config.reduction_op == "mean":
                        param.grad.div_(world_size)
        elif expert_params:
            # Fallback to all-reduce for expert params
            expert_grads = [p.grad for p in expert_params if p.grad is not None]
            self._batch_all_reduce(expert_grads, process_group, async_op=False)

    def _reduce_context_parallel_ring(
        self,
        parameters: List[nn.Parameter],
        process_group: dist.ProcessGroup,
    ) -> None:
        """Reduce gradients for context parallelism using ring reduction.

        Args:
            parameters: Parameters with gradients to reduce.
            process_group: Context parallel process group.
        """
        world_size = dist.get_world_size(process_group)
        if world_size == 1:
            return

        rank = dist.get_rank(process_group)

        if self.config.context_parallel_sync_type == "ring":
            # Implement ring-based reduction for better scalability
            for param in parameters:
                if param.grad is None:
                    continue

                # Ring reduce implementation
                grad = param.grad
                recv_buff = torch.empty_like(grad)

                # Perform ring reduce in log(world_size) steps
                for step in range(
                    int(torch.log2(torch.tensor(world_size)).ceil().item())
                ):
                    # Calculate source and destination ranks
                    distance = 2**step
                    if rank % (2 * distance) == 0:
                        # Send to rank + distance
                        dest = (rank + distance) % world_size
                        if dest < world_size:
                            dist.send(grad, dest, group=process_group)
                    elif (rank - distance) % (2 * distance) == 0:
                        # Receive from rank - distance
                        src = (rank - distance) % world_size
                        if src >= 0:
                            dist.recv(recv_buff, src, group=process_group)
                            grad.add_(recv_buff)

                # Final averaging
                if self.config.reduction_op == "mean":
                    grad.div_(world_size)

                # Broadcast final result to all ranks
                dist.broadcast(grad, 0, group=process_group)
        else:
            # Fallback to standard all-reduce
            gradients = [p.grad for p in parameters if p.grad is not None]
            self._batch_all_reduce(gradients, process_group, async_op=False)
