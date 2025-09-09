"""
Sequence Parallel Implementation for RoseLLM

This module implements sequence parallelism following Megatron-LM's design patterns.
Sequence parallelism distributes activations along the sequence dimension across
tensor parallel ranks, significantly reducing activation memory requirements.

Key Features:
- Scatter/gather operations for sequence dimension distribution
- Autograd functions with proper forward/backward implementations
- Integration with existing tensor parallelism infrastructure
- Memory-efficient communication patterns

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- "Reducing Activation Recomputation in Large Transformer Models"
  (Korthikanti et al., 2022)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.autograd import Function

from .parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
    is_initialized,
)

# Configure logging
logger = logging.getLogger(__name__)


# ----------------------
# Configuration Classes
# ----------------------


class CommunicationOp(Enum):
    """Types of communication operations in sequence parallelism."""

    SCATTER = "scatter"
    GATHER = "gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"


@dataclass
class SequenceParallelConfig:
    """
    Configuration for sequence parallel operations.

    Attributes:
        enable_memory_profiling: Track memory usage during operations
        enable_communication_stats: Collect communication timing statistics
        debug_mode: Enable detailed debug logging
        optimize_memory: Use memory-efficient implementations
        communication_overlap: Enable computation-communication overlap
        gradient_accumulation_fusion: Fuse gradient accumulation with communication
    """

    enable_memory_profiling: bool = False
    enable_communication_stats: bool = False
    debug_mode: bool = False
    optimize_memory: bool = True
    communication_overlap: bool = False
    gradient_accumulation_fusion: bool = False

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.communication_overlap and not torch.cuda.is_available():
            raise ValueError("Communication overlap requires CUDA")
        if self.gradient_accumulation_fusion and not self.optimize_memory:
            logger.warning(
                "Gradient accumulation fusion works best with memory "
                "optimization enabled"
            )


# Global configuration
_SEQUENCE_PARALLEL_CONFIG = SequenceParallelConfig()


def set_sequence_parallel_config(config: SequenceParallelConfig) -> None:
    """Set global sequence parallel configuration."""
    global _SEQUENCE_PARALLEL_CONFIG
    config.validate()
    _SEQUENCE_PARALLEL_CONFIG = config
    logger.info(f"Sequence parallel config updated: {config}")


def get_sequence_parallel_config() -> SequenceParallelConfig:
    """Get global sequence parallel configuration."""
    return _SEQUENCE_PARALLEL_CONFIG


# ----------------------
# Error Handling
# ----------------------


class SequenceParallelError(Exception):
    """Base exception for sequence parallel operations."""

    pass


class TensorShapeError(SequenceParallelError):
    """Exception raised for invalid tensor shapes."""

    pass


class CommunicationError(SequenceParallelError):
    """Exception raised for communication failures."""

    pass


def validate_tensor_for_sequence_parallel(
    tensor: torch.Tensor,
    expected_dims: Optional[int] = None,
    min_seq_len: Optional[int] = None,
    operation: Optional[str] = None,
) -> None:
    """
    Validate tensor for sequence parallel operations.

    Args:
        tensor: Tensor to validate
        expected_dims: Expected number of dimensions
        min_seq_len: Minimum sequence length required
        operation: Name of operation for error messages

    Raises:
        TensorShapeError: If tensor shape is invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise TensorShapeError(f"Expected torch.Tensor, got {type(tensor)}")

    if expected_dims is not None and tensor.ndim != expected_dims:
        raise TensorShapeError(
            f"{operation or 'Operation'} expected {expected_dims}D tensor, "
            f"got {tensor.ndim}D tensor with shape {tensor.shape}"
        )

    if min_seq_len is not None and tensor.shape[0] < min_seq_len:
        raise TensorShapeError(
            f"{operation or 'Operation'} requires sequence length >= {min_seq_len}, "
            f"got {tensor.shape[0]}"
        )


# ----------------------
# Memory Profiling
# ----------------------


class MemoryProfiler:
    """Profile memory usage during sequence parallel operations."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.stats: Dict[str, Dict[str, float]] = {}

    def record(self, operation: str, phase: str = "forward") -> None:
        """Record memory statistics for an operation."""
        if not self.enabled or not torch.cuda.is_available():
            return

        key = f"{operation}_{phase}"
        self.stats[key] = {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def report(self) -> str:
        """Generate memory usage report."""
        if not self.stats:
            return "No memory statistics collected"

        report = "\nMemory Usage Report:\n"
        report += "-" * 50 + "\n"
        for op, stats in self.stats.items():
            report += f"{op}:\n"
            for stat, value in stats.items():
                report += f"  {stat}: {value:.3f}\n"
        return report


# Global memory profiler
_memory_profiler = MemoryProfiler()

# ----------------------
# Helper Functions
# ----------------------


def _split_along_first_dim(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Split the input tensor along the first dimension.

    Args:
        input_: Input tensor to split
        group: Process group for communication

    Returns:
        Split tensor for current rank

    Raises:
        TensorShapeError: If tensor cannot be evenly split
    """
    try:
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
    except Exception as e:
        raise CommunicationError(f"Failed to get process group info: {e}")

    # Validate tensor
    validate_tensor_for_sequence_parallel(
        input_, min_seq_len=world_size, operation="Split along first dimension"
    )

    # Ensure the first dimension is divisible by world_size
    if input_.shape[0] % world_size != 0:
        raise TensorShapeError(
            f"First dimension ({input_.shape[0]}) must be "
            f"divisible by world_size ({world_size})"
        )

    # Calculate split size
    split_size = input_.shape[0] // world_size

    # Log operation if debug mode
    config = get_sequence_parallel_config()
    if config.debug_mode:
        logger.debug(
            f"Splitting tensor {input_.shape} along dim 0, "
            f"rank {rank} gets slice [{rank * split_size}:{(rank + 1) * split_size}]"
        )

    # Split tensor and return the chunk for this rank
    if config.optimize_memory:
        # Use view instead of split for memory efficiency when possible
        start_idx = rank * split_size
        end_idx = (rank + 1) * split_size
        output = input_[start_idx:end_idx].contiguous()
    else:
        output = torch.split(input_, split_size, dim=0)[rank].contiguous()

    return output


def _split_along_last_dim(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Split the input tensor along the last dimension.

    Args:
        input_: Input tensor to split
        group: Process group for communication

    Returns:
        Split tensor for current rank

    Raises:
        TensorShapeError: If tensor cannot be evenly split
    """
    try:
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
    except Exception as e:
        raise CommunicationError(f"Failed to get process group info: {e}")

    validate_tensor_for_sequence_parallel(
        input_, operation="Split along last dimension"
    )

    # Ensure the last dimension is divisible by world_size
    last_dim_size = input_.shape[-1]
    if last_dim_size % world_size != 0:
        raise TensorShapeError(
            f"Last dimension ({last_dim_size}) must be "
            f"divisible by world_size ({world_size})"
        )

    # Calculate split size
    split_size = last_dim_size // world_size

    config = get_sequence_parallel_config()

    # Log operation if debug mode
    if config.debug_mode:
        logger.debug(
            f"Splitting tensor {input_.shape} along dim -1, "
            f"rank {rank} gets slice [..., "
            f"{rank * split_size}:{(rank + 1) * split_size}]"
        )

    # Split tensor and return the chunk for this rank
    if config.optimize_memory:
        # Use slicing for memory efficiency
        start_idx = rank * split_size
        end_idx = (rank + 1) * split_size
        output = input_[..., start_idx:end_idx].contiguous()
    else:
        output = torch.split(input_, split_size, dim=-1)[rank].contiguous()

    return output


def _gather_along_first_dim(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Gather tensors along the first dimension and concatenate.

    Args:
        input_: Input tensor to gather
        group: Process group for communication

    Returns:
        Gathered tensor concatenated along first dimension

    Raises:
        CommunicationError: If all-gather fails
    """
    try:
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
    except Exception as e:
        raise CommunicationError(f"Failed to get process group info: {e}")

    # Skip if running on single GPU
    if world_size == 1:
        return input_

    # Validate input
    validate_tensor_for_sequence_parallel(
        input_, operation="Gather along first dimension"
    )

    config = get_sequence_parallel_config()

    # Profile memory if enabled
    if config.enable_memory_profiling:
        _memory_profiler.record("gather_first_dim", "before")

    try:
        # Size and dimension
        first_dim = 0

        # Log operation if debug mode
        if config.debug_mode:
            logger.debug(
                f"Gathering tensor {input_.shape} from {world_size} ranks along dim 0"
            )

        # Allocate output tensor list
        if config.optimize_memory:
            # Pre-allocate contiguous buffer for better memory layout
            output_shape = list(input_.shape)
            output_shape[first_dim] = input_.shape[first_dim] * world_size
            output_buffer = torch.empty(
                output_shape, dtype=input_.dtype, device=input_.device
            )

            # Create views into the buffer
            tensor_list = []
            chunk_size = input_.shape[first_dim]
            for i in range(world_size):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size
                tensor_list.append(output_buffer[start_idx:end_idx])

            tensor_list[rank] = input_

            # All-gather into pre-allocated buffer
            dist.all_gather(tensor_list, input_, group=group)
            output = output_buffer.contiguous()
        else:
            # Standard implementation
            tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
            tensor_list[rank] = input_

            # All-gather across the group
            dist.all_gather(tensor_list, input_, group=group)

            # Concatenate along first dimension
            output = torch.cat(tensor_list, dim=first_dim).contiguous()

        # Profile memory if enabled
        if config.enable_memory_profiling:
            _memory_profiler.record("gather_first_dim", "after")

        return output

    except Exception as e:
        raise CommunicationError(f"All-gather operation failed on rank {rank}: {e}")


def _gather_along_last_dim(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Gather tensors along the last dimension and concatenate.

    Args:
        input_: Input tensor to gather
        group: Process group for communication

    Returns:
        Gathered tensor concatenated along last dimension

    Raises:
        CommunicationError: If all-gather fails
    """
    try:
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
    except Exception as e:
        raise CommunicationError(f"Failed to get process group info: {e}")

    # Skip if running on single GPU
    if world_size == 1:
        return input_

    # Validate input
    validate_tensor_for_sequence_parallel(
        input_, operation="Gather along last dimension"
    )

    config = get_sequence_parallel_config()
    if config.enable_memory_profiling:
        _memory_profiler.record("gather_last_dim", "before")

    try:
        # Size and dimension
        last_dim = -1

        # Log operation if debug mode
        if config.debug_mode:
            logger.debug(
                f"Gathering tensor {input_.shape} from {world_size} ranks along dim -1"
            )

        # Allocate output tensor list
        if config.optimize_memory:
            # Pre-allocate contiguous buffer
            output_shape = list(input_.shape)
            output_shape[last_dim] = input_.shape[last_dim] * world_size
            output_buffer = torch.empty(
                output_shape, dtype=input_.dtype, device=input_.device
            )

            # Create views into the buffer
            tensor_list = []
            chunk_size = input_.shape[last_dim]
            for i in range(world_size):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size
                tensor_list.append(output_buffer[..., start_idx:end_idx])

            tensor_list[rank] = input_

            # All-gather into pre-allocated buffer
            dist.all_gather(tensor_list, input_, group=group)
            output = output_buffer.contiguous()
        else:
            # Standard implementation
            tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
            tensor_list[rank] = input_

            # All-gather across the group
            dist.all_gather(tensor_list, input_, group=group)

            # Concatenate along last dimension
            output = torch.cat(tensor_list, dim=last_dim).contiguous()

        # Profile memory if enabled
        if config.enable_memory_profiling:
            _memory_profiler.record("gather_last_dim", "after")

        return output

    except Exception as e:
        raise CommunicationError(f"All-gather operation failed on rank {rank}: {e}")


def _reduce_scatter_along_first_dim(
    input_: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    """Reduce-scatter along the first dimension.

    Args:
        input_: Input tensor to reduce-scatter
        group: Process group for communication

    Returns:
        Reduced and scattered tensor

    Raises:
        TensorShapeError: If tensor cannot be evenly split
        CommunicationError: If reduce-scatter fails
    """
    try:
        world_size = dist.get_world_size(group)
    except Exception as e:
        raise CommunicationError(f"Failed to get world size: {e}")

    # Skip if running on single GPU
    if world_size == 1:
        return input_

    # Validate input
    validate_tensor_for_sequence_parallel(
        input_, min_seq_len=world_size, operation="Reduce-scatter along first dimension"
    )

    # Ensure first dimension is divisible by world_size
    if input_.shape[0] % world_size != 0:
        raise TensorShapeError(
            f"First dimension ({input_.shape[0]}) must be "
            f"divisible by world_size ({world_size})"
        )

    config = get_sequence_parallel_config()

    # Profile memory if enabled
    if config.enable_memory_profiling:
        _memory_profiler.record("reduce_scatter_first_dim", "before")

    try:
        dim_size = input_.shape[0] // world_size
        output_shape = (dim_size,) + input_.shape[1:]

        # Log operation if debug mode
        if config.debug_mode:
            logger.debug(
                f"Reduce-scattering tensor {input_.shape} to {output_shape} "
                f"across {world_size} ranks"
            )

        # Allocate output tensor with proper memory alignment
        if config.optimize_memory:
            # Ensure output is contiguous and aligned for better performance
            output = torch.empty(
                output_shape,
                dtype=input_.dtype,
                device=input_.device,
                memory_format=torch.contiguous_format,
            )
        else:
            output = torch.empty(output_shape, dtype=input_.dtype, device=input_.device)

        # Use the newer API if available
        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(output, input_, group=group)
        else:
            # Fallback to older API
            dist._reduce_scatter_base(output, input_, group=group)  # type: ignore

        # Profile memory if enabled
        if config.enable_memory_profiling:
            _memory_profiler.record("reduce_scatter_first_dim", "after")

        return output

    except Exception as e:
        raise CommunicationError(f"Reduce-scatter operation failed: {e}")


# ----------------------
# Autograd Functions
# ----------------------


class _ScatterToSequenceParallelRegion(Function):
    """Split the input along the sequence dimension and keep the corresponding chunk.

    This autograd function implements the forward scatter and backward gather
    pattern for sequence parallelism, ensuring correct gradient flow.
    """

    @staticmethod
    def forward(
        ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        """Forward: split along sequence (first) dimension.

        Args:
            ctx: Autograd context
            input_: Input tensor [seq_len, batch, ...]
            group: Process group for communication

        Returns:
            Scattered tensor [seq_len/TP, batch, ...]
        """
        ctx.group = group

        # Validate input before operation
        validate_tensor_for_sequence_parallel(
            input_, expected_dims=3, operation="Scatter to sequence parallel"
        )

        config = get_sequence_parallel_config()
        if config.enable_communication_stats:
            import time

            start_time = time.perf_counter()
            output = _split_along_first_dim(input_, group)
            elapsed = time.perf_counter() - start_time
            logger.info(f"Scatter forward took {elapsed:.4f}s")
        else:
            output = _split_along_first_dim(input_, group)

        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None]:
        """Backward: all-gather along sequence dimension.

        Args:
            ctx: Autograd context with saved tensors
            grad_output: Gradient w.r.t. output [seq_len/TP, batch, ...]

        Returns:
            Gradient w.r.t. input [seq_len, batch, ...]
        """
        config = get_sequence_parallel_config()

        if config.enable_communication_stats:
            import time

            start_time = time.perf_counter()
            grad_input = _gather_along_first_dim(grad_output, ctx.group)
            elapsed = time.perf_counter() - start_time
            logger.info(f"Scatter backward took {elapsed:.4f}s")
        else:
            grad_input = _gather_along_first_dim(grad_output, ctx.group)

        return grad_input, None


class _GatherFromSequenceParallelRegion(Function):
    """Gather the input from sequence parallel region and concatenate.

    Implements the gather-scatter pattern with optional reduce-scatter
    optimization in the backward pass for gradient aggregation.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input_: torch.Tensor,
        group: dist.ProcessGroup,
        tensor_parallel_output_grad: bool = True,
    ) -> torch.Tensor:
        """Forward: all-gather along sequence (first) dimension.

        Args:
            ctx: Autograd context
            input_: Input tensor [seq_len/TP, batch, ...]
            group: Process group for communication
            tensor_parallel_output_grad: Use reduce-scatter in backward

        Returns:
            Gathered tensor [seq_len, batch, ...]
        """
        ctx.group = group
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad

        # Validate input
        validate_tensor_for_sequence_parallel(
            input_, expected_dims=3, operation="Gather from sequence parallel"
        )

        config = get_sequence_parallel_config()
        if config.enable_communication_stats:
            import time

            start_time = time.perf_counter()
            output = _gather_along_first_dim(input_, group)
            elapsed = time.perf_counter() - start_time
            logger.info(f"Gather forward took {elapsed:.4f}s")
        else:
            output = _gather_along_first_dim(input_, group)

        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None, None]:
        """Backward: reduce-scatter or split along sequence dimension.

        Args:
            ctx: Autograd context
            grad_output: Gradient w.r.t. output [seq_len, batch, ...]

        Returns:
            Gradient w.r.t. input [seq_len/TP, batch, ...]
        """
        config = get_sequence_parallel_config()

        if config.enable_communication_stats:
            import time

            start_time = time.perf_counter()

        if ctx.tensor_parallel_output_grad:
            # Use reduce-scatter for efficiency (sum gradients and scatter)
            grad_input = _reduce_scatter_along_first_dim(grad_output, ctx.group)
        else:
            # Simple split without reduction
            # (for cases where gradients shouldn't be summed)
            grad_input = _split_along_first_dim(grad_output, ctx.group)

        if config.enable_communication_stats:
            elapsed = time.perf_counter() - start_time  # type: ignore
            op_type = "reduce-scatter" if ctx.tensor_parallel_output_grad else "split"
            logger.info(f"Gather backward ({op_type}) took {elapsed:.4f}s")

        return grad_input, None, None


class _ReduceScatterToSequenceParallelRegion(Function):
    """Reduce-scatter to sequence parallel region.

    Performs reduction (sum) and scatter in forward pass,
    with corresponding gather in backward pass.
    """

    @staticmethod
    def forward(
        ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        """Forward: reduce-scatter along sequence (first) dimension.

        Args:
            ctx: Autograd context
            input_: Input tensor [seq_len, batch, ...]
            group: Process group for communication

        Returns:
            Reduced and scattered tensor [seq_len/TP, batch, ...]
        """
        ctx.group = group

        # Validate input
        validate_tensor_for_sequence_parallel(
            input_, expected_dims=3, operation="Reduce-scatter to sequence parallel"
        )

        config = get_sequence_parallel_config()
        if config.enable_communication_stats:
            import time

            start_time = time.perf_counter()
            output = _reduce_scatter_along_first_dim(input_, group)
            elapsed = time.perf_counter() - start_time
            logger.info(f"Reduce-scatter forward took {elapsed:.4f}s")
        else:
            output = _reduce_scatter_along_first_dim(input_, group)

        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None]:
        """Backward: all-gather along sequence dimension.

        Args:
            ctx: Autograd context
            grad_output: Gradient w.r.t. output [seq_len/TP, batch, ...]

        Returns:
            Gradient w.r.t. input [seq_len, batch, ...]
        """
        config = get_sequence_parallel_config()

        if config.enable_communication_stats:
            import time

            start_time = time.perf_counter()
            grad_input = _gather_along_first_dim(grad_output, ctx.group)
            elapsed = time.perf_counter() - start_time
            logger.info(f"Reduce-scatter backward took {elapsed:.4f}s")
        else:
            grad_input = _gather_along_first_dim(grad_output, ctx.group)

        return grad_input, None


class _AllToAllSequenceToHidden(Function):
    """All-to-all communication: sequence parallel to hidden parallel.

    Redistributes data from sequence-partitioned to hidden-partitioned format,
    enabling different parallelization strategies within a model.
    """

    @staticmethod
    def forward(
        ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        """Transform from [seq_len/TP, batch, hidden] to [seq_len, batch, hidden/TP].

        Args:
            ctx: Autograd context
            input_: Sequence-partitioned tensor [seq_len/TP, batch, hidden]
            group: Process group for communication

        Returns:
            Hidden-partitioned tensor [seq_len, batch, hidden/TP]
        """
        ctx.group = group

        try:
            world_size = dist.get_world_size(group)
        except Exception as e:
            raise CommunicationError(f"Failed to get world size: {e}")

        # Skip if single GPU
        if world_size == 1:
            return input_

        # Validate input shape
        if input_.ndim != 3:
            raise TensorShapeError(
                f"Expected 3D tensor for all-to-all, got {input_.ndim}D tensor"
            )

        seq_len, batch_size, hidden_size = input_.shape

        if hidden_size % world_size != 0:
            raise TensorShapeError(
                f"Hidden dimension ({hidden_size}) must be divisible by "
                f"world size ({world_size}) for all-to-all"
            )

        config = get_sequence_parallel_config()

        # Profile memory if enabled
        if config.enable_memory_profiling:
            _memory_profiler.record("all_to_all_seq_to_hidden", "before")

        try:
            # Log operation if debug mode
            if config.debug_mode:
                logger.debug(
                    f"All-to-all: seq parallel {input_.shape} -> "
                    f"hidden parallel [{seq_len * world_size}, {batch_size}, "
                    f"{hidden_size // world_size}]"
                )

            if config.optimize_memory:
                # Optimized reshaping with fewer intermediate tensors
                # Combine reshape and permute operations
                hidden_per_rank = hidden_size // world_size

                # Direct reshape and permute
                input_for_comm = (
                    input_.reshape(seq_len, batch_size, world_size, hidden_per_rank)
                    .permute(2, 0, 1, 3)
                    .reshape(world_size, -1)
                )

                # Pre-allocate output buffer
                output_buffer = torch.empty_like(input_for_comm)

                # Perform all-to-all
                dist.all_to_all_single(output_buffer, input_for_comm, group=group)

                # Reshape output directly to final shape
                output = (
                    output_buffer.reshape(
                        world_size, seq_len, batch_size, hidden_per_rank
                    )
                    .permute(1, 2, 0, 3)
                    .reshape(seq_len * world_size, batch_size, hidden_per_rank)
                )
            else:
                # Standard implementation with explicit steps
                # Split hidden dimension into chunks
                input_reshaped = input_.reshape(
                    seq_len, batch_size, world_size, hidden_size // world_size
                )
                input_transposed = input_reshaped.permute(2, 0, 1, 3).contiguous()
                input_flattened = input_transposed.reshape(world_size, -1)

                # Perform all-to-all
                output_flattened = torch.empty_like(input_flattened)
                dist.all_to_all_single(output_flattened, input_flattened, group=group)

                # Reshape back
                output_transposed = output_flattened.reshape(
                    world_size, seq_len, batch_size, hidden_size // world_size
                )
                output_reshaped = output_transposed.permute(1, 2, 0, 3).contiguous()
                output = output_reshaped.reshape(
                    seq_len * world_size, batch_size, hidden_size // world_size
                )

            # Profile memory if enabled
            if config.enable_memory_profiling:
                _memory_profiler.record("all_to_all_seq_to_hidden", "after")

            return output

        except Exception as e:
            raise CommunicationError(f"All-to-all (sequence to hidden) failed: {e}")

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None]:
        """Backward: reverse all-to-all (hidden to sequence parallel)."""
        # Use the inverse operation for gradient flow
        return _AllToAllHiddenToSequence.forward(ctx, grad_output, ctx.group), None


class _AllToAllHiddenToSequence(Function):
    """All-to-all communication: hidden parallel to sequence parallel.

    Redistributes data from hidden-partitioned to sequence-partitioned format,
    the inverse of _AllToAllSequenceToHidden.
    """

    @staticmethod
    def forward(
        ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        """Transform from [seq_len, batch, hidden/TP] to [seq_len/TP, batch, hidden].

        Args:
            ctx: Autograd context
            input_: Hidden-partitioned tensor [seq_len, batch, hidden/TP]
            group: Process group for communication

        Returns:
            Sequence-partitioned tensor [seq_len/TP, batch, hidden]
        """
        ctx.group = group

        try:
            world_size = dist.get_world_size(group)
        except Exception as e:
            raise CommunicationError(f"Failed to get world size: {e}")

        # Skip if single GPU
        if world_size == 1:
            return input_

        # Validate input shape
        if input_.ndim != 3:
            raise TensorShapeError(
                f"Expected 3D tensor for all-to-all, got {input_.ndim}D tensor"
            )

        total_seq_len, batch_size, hidden_per_rank = input_.shape

        if total_seq_len % world_size != 0:
            raise TensorShapeError(
                f"Sequence length ({total_seq_len}) must be divisible by "
                f"world size ({world_size}) for all-to-all"
            )

        seq_len_per_rank = total_seq_len // world_size

        config = get_sequence_parallel_config()

        # Profile memory if enabled
        if config.enable_memory_profiling:
            _memory_profiler.record("all_to_all_hidden_to_seq", "before")

        try:
            # Log operation if debug mode
            if config.debug_mode:
                logger.debug(
                    f"All-to-all: hidden parallel {input_.shape} -> "
                    f"seq parallel [{seq_len_per_rank}, {batch_size}, "
                    f"{hidden_per_rank * world_size}]"
                )

            if config.optimize_memory:
                # Optimized implementation with fewer intermediate tensors
                # Direct reshape and permute
                input_for_comm = (
                    input_.reshape(
                        world_size, seq_len_per_rank, batch_size, hidden_per_rank
                    )
                    .permute(0, 2, 1, 3)
                    .reshape(world_size, -1)
                )

                # Pre-allocate output buffer
                output_buffer = torch.empty_like(input_for_comm)

                # Perform all-to-all
                dist.all_to_all_single(output_buffer, input_for_comm, group=group)

                # Reshape output directly to final shape
                output = (
                    output_buffer.reshape(
                        world_size, batch_size, seq_len_per_rank, hidden_per_rank
                    )
                    .permute(2, 1, 3, 0)
                    .reshape(seq_len_per_rank, batch_size, hidden_per_rank * world_size)
                )
            else:
                # Standard implementation with explicit steps
                # Prepare for all-to-all
                input_reshaped = input_.reshape(
                    world_size, seq_len_per_rank, batch_size, hidden_per_rank
                )
                input_transposed = input_reshaped.permute(0, 2, 1, 3).contiguous()
                input_flattened = input_transposed.reshape(world_size, -1)

                # Perform all-to-all
                output_flattened = torch.empty_like(input_flattened)
                dist.all_to_all_single(output_flattened, input_flattened, group=group)

                # Reshape back
                output_transposed = output_flattened.reshape(
                    world_size, batch_size, seq_len_per_rank, hidden_per_rank
                )
                output_reshaped = output_transposed.permute(2, 1, 3, 0).contiguous()
                output = output_reshaped.reshape(
                    seq_len_per_rank, batch_size, hidden_per_rank * world_size
                )

            # Profile memory if enabled
            if config.enable_memory_profiling:
                _memory_profiler.record("all_to_all_hidden_to_seq", "after")

            return output

        except Exception as e:
            raise CommunicationError(f"All-to-all (hidden to sequence) failed: {e}")

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None]:
        """Backward: reverse all-to-all (sequence to hidden parallel)."""
        # Use the inverse operation for gradient flow
        return _AllToAllSequenceToHidden.forward(ctx, grad_output, ctx.group), None


# ----------------------
# Public API Functions
# ----------------------


def scatter_to_sequence_parallel_region(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> torch.Tensor:
    """
    Scatter input to sequence parallel region.

    This operation splits the input tensor along the sequence dimension
    (first dimension) and distributes chunks to different ranks in the
    tensor parallel group.

    Args:
        input_: Input tensor with shape [seq_len, batch, ...]
        group: Process group for communication (defaults to TP group)
        async_op: Whether to perform asynchronous communication (experimental)

    Returns:
        Tensor with shape [seq_len/TP, batch, ...] distributed across TP ranks

    Raises:
        RuntimeError: If parallel state is not initialized
        TensorShapeError: If input tensor has invalid shape
        CommunicationError: If scatter operation fails

    Example:
        >>> # With 2 TP ranks, input [512, 32, 768]
        >>> output = scatter_to_sequence_parallel_region(input_)
        >>> # Each rank gets [256, 32, 768]
    """
    # Check if parallel state is initialized
    if not is_initialized():
        raise RuntimeError(
            "Parallel state must be initialized before using "
            "sequence parallel operations"
        )

    if group is None:
        group = get_tensor_model_parallel_group()

    if group is None or dist.get_world_size(group) == 1:
        return input_

    # Validate input tensor
    if not isinstance(input_, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(input_)}")

    if input_.ndim < 2:
        raise TensorShapeError(
            f"Input tensor must have at least 2 dimensions for sequence parallel, "
            f"got {input_.ndim}D tensor"
        )

    # Log operation if debug mode
    config = get_sequence_parallel_config()
    if config.debug_mode:
        logger.debug(
            f"Scattering tensor {input_.shape} to sequence parallel region "
            f"with TP size {dist.get_world_size(group)}"
        )

    # Apply autograd function
    result = _ScatterToSequenceParallelRegion.apply(input_, group)

    # Ensure result is a tensor before marking
    assert isinstance(result, torch.Tensor)

    # Mark tensor as sequence parallel
    mark_tensor_as_sequence_parallel(result)

    return result


def gather_from_sequence_parallel_region(
    input_: torch.Tensor,
    tensor_parallel_output_grad: bool = True,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> torch.Tensor:
    """
    Gather input from sequence parallel region.

    This operation collects tensor chunks from all ranks in the tensor
    parallel group and concatenates them along the sequence dimension.

    Args:
        input_: Input tensor with shape [seq_len/TP, batch, ...]
        tensor_parallel_output_grad: If True, use reduce-scatter in backward pass
            for gradient aggregation. If False, use simple split.
        group: Process group for communication (defaults to TP group)
        async_op: Whether to perform asynchronous communication (experimental)

    Returns:
        Tensor with shape [seq_len, batch, ...] gathered from all TP ranks

    Raises:
        RuntimeError: If parallel state is not initialized
        TensorShapeError: If input tensor has invalid shape
        CommunicationError: If gather operation fails

    Example:
        >>> # With 2 TP ranks, each rank has [256, 32, 768]
        >>> output = gather_from_sequence_parallel_region(input_)
        >>> # Result is [512, 32, 768] with data from both ranks
    """
    # Check if parallel state is initialized
    if not is_initialized():
        raise RuntimeError(
            "Parallel state must be initialized before using "
            "sequence parallel operations"
        )

    if group is None:
        group = get_tensor_model_parallel_group()

    if group is None or dist.get_world_size(group) == 1:
        return input_

    # Validate input tensor
    if not isinstance(input_, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(input_)}")

    if input_.ndim < 2:
        raise TensorShapeError(
            f"Input tensor must have at least 2 dimensions for sequence parallel, "
            f"got {input_.ndim}D tensor"
        )

    # Log operation if debug mode
    config = get_sequence_parallel_config()
    if config.debug_mode:
        logger.debug(
            f"Gathering tensor {input_.shape} from sequence parallel region "
            f"with TP size {dist.get_world_size(group)}, "
            f"tensor_parallel_output_grad={tensor_parallel_output_grad}"
        )

    # Apply autograd function
    result = _GatherFromSequenceParallelRegion.apply(
        input_, group, tensor_parallel_output_grad
    )

    assert isinstance(result, torch.Tensor)
    return result


def reduce_scatter_to_sequence_parallel_region(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> torch.Tensor:
    """
    Reduce-scatter input to sequence parallel region.

    This operation performs an element-wise reduction (sum) across all ranks
    and then scatters the result along the sequence dimension. This is useful
    for gradient aggregation in sequence parallel training.

    Args:
        input_: Input tensor with shape [seq_len, batch, ...]
        group: Process group for communication (defaults to TP group)
        async_op: Whether to perform asynchronous communication (experimental)

    Returns:
        Tensor with shape [seq_len/TP, batch, ...] after reduce-scatter

    Raises:
        RuntimeError: If parallel state is not initialized
        TensorShapeError: If input tensor has invalid shape
        CommunicationError: If reduce-scatter operation fails

    Example:
        >>> # With 2 TP ranks, input [512, 32, 768] on each rank
        >>> output = reduce_scatter_to_sequence_parallel_region(input_)
        >>> # Each rank gets [256, 32, 768] with summed values
    """
    # Check if parallel state is initialized
    if not is_initialized():
        raise RuntimeError(
            "Parallel state must be initialized before using "
            "sequence parallel operations"
        )

    if group is None:
        group = get_tensor_model_parallel_group()

    if group is None or dist.get_world_size(group) == 1:
        return input_

    # Validate input tensor
    if not isinstance(input_, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(input_)}")

    if input_.ndim < 2:
        raise TensorShapeError(
            f"Input tensor must have at least 2 dimensions for sequence parallel, "
            f"got {input_.ndim}D tensor"
        )

    # Log operation if debug mode
    config = get_sequence_parallel_config()
    if config.debug_mode:
        logger.debug(
            f"Reduce-scattering tensor {input_.shape} to sequence parallel region "
            f"with TP size {dist.get_world_size(group)}"
        )

    # Apply autograd function
    result = _ReduceScatterToSequenceParallelRegion.apply(input_, group)

    # Ensure result is a tensor before marking
    assert isinstance(result, torch.Tensor)

    # Mark tensor as sequence parallel
    mark_tensor_as_sequence_parallel(result)

    return result


def all_to_all_sequence_to_hidden(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> torch.Tensor:
    """
    All-to-all communication from sequence parallel to hidden parallel.

    This operation redistributes data from being partitioned along the
    sequence dimension to being partitioned along the hidden dimension.
    This is useful when switching between different parallelization
    strategies within a model.

    Transforms tensor from [seq_len/TP, batch, hidden] to [seq_len, batch, hidden/TP].

    Args:
        input_: Input tensor in sequence parallel format
        group: Process group for communication (defaults to TP group)
        async_op: Whether to perform asynchronous communication (experimental)

    Returns:
        Tensor in hidden parallel format

    Raises:
        RuntimeError: If parallel state is not initialized
        TensorShapeError: If input tensor has invalid shape
        CommunicationError: If all-to-all operation fails

    Example:
        >>> # With 2 TP ranks, input [256, 32, 768] on each rank
        >>> output = all_to_all_sequence_to_hidden(input_)
        >>> # Each rank gets [512, 32, 384] with redistributed data
    """
    # Check if parallel state is initialized
    if not is_initialized():
        raise RuntimeError(
            "Parallel state must be initialized before using "
            "sequence parallel operations"
        )

    if group is None:
        group = get_tensor_model_parallel_group()

    if group is None or dist.get_world_size(group) == 1:
        return input_

    # Validate input tensor
    if not isinstance(input_, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(input_)}")

    if input_.ndim != 3:
        raise TensorShapeError(
            f"All-to-all requires 3D tensor [seq, batch, hidden], "
            f"got {input_.ndim}D tensor with shape {input_.shape}"
        )

    # Log operation if debug mode
    config = get_sequence_parallel_config()
    if config.debug_mode:
        logger.debug(
            f"All-to-all: sequence parallel {input_.shape} -> hidden parallel "
            f"with TP size {dist.get_world_size(group)}"
        )

    # Apply autograd function
    result = _AllToAllSequenceToHidden.apply(input_, group)

    assert isinstance(result, torch.Tensor)
    return result


def all_to_all_hidden_to_sequence(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> torch.Tensor:
    """
    All-to-all communication from hidden parallel to sequence parallel.

    This operation redistributes data from being partitioned along the
    hidden dimension to being partitioned along the sequence dimension.
    This is the inverse of all_to_all_sequence_to_hidden.

    Transforms tensor from [seq_len, batch, hidden/TP] to [seq_len/TP, batch, hidden].

    Args:
        input_: Input tensor in hidden parallel format
        group: Process group for communication (defaults to TP group)
        async_op: Whether to perform asynchronous communication (experimental)

    Returns:
        Tensor in sequence parallel format

    Raises:
        RuntimeError: If parallel state is not initialized
        TensorShapeError: If input tensor has invalid shape
        CommunicationError: If all-to-all operation fails

    Example:
        >>> # With 2 TP ranks, input [512, 32, 384] on each rank
        >>> output = all_to_all_hidden_to_sequence(input_)
        >>> # Each rank gets [256, 32, 768] with redistributed data
    """
    # Check if parallel state is initialized
    if not is_initialized():
        raise RuntimeError(
            "Parallel state must be initialized before using "
            "sequence parallel operations"
        )

    if group is None:
        group = get_tensor_model_parallel_group()

    if group is None or dist.get_world_size(group) == 1:
        return input_

    # Validate input tensor
    if not isinstance(input_, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(input_)}")

    if input_.ndim != 3:
        raise TensorShapeError(
            f"All-to-all requires 3D tensor [seq, batch, hidden], "
            f"got {input_.ndim}D tensor with shape {input_.shape}"
        )

    # Log operation if debug mode
    config = get_sequence_parallel_config()
    if config.debug_mode:
        logger.debug(
            f"All-to-all: hidden parallel {input_.shape} -> sequence parallel "
            f"with TP size {dist.get_world_size(group)}"
        )

    # Apply autograd function
    result = _AllToAllHiddenToSequence.apply(input_, group)

    # Ensure result is a tensor before marking
    assert isinstance(result, torch.Tensor)

    # Mark result as sequence parallel
    mark_tensor_as_sequence_parallel(result)

    return result


# ----------------------
# Utility Functions
# ----------------------


def mark_tensor_as_sequence_parallel(tensor: torch.Tensor) -> None:
    """Mark a tensor as being in sequence parallel format.

    This metadata helps track tensors that are distributed along
    the sequence dimension for debugging and validation.

    Args:
        tensor: Tensor to mark as sequence parallel
    """
    tensor._sequence_parallel = True  # type: ignore
    tensor._sequence_parallel_world_size = (  # type: ignore
        get_tensor_model_parallel_size()
    )


def is_sequence_parallel_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is marked as sequence parallel.

    Args:
        tensor: Tensor to check

    Returns:
        True if tensor is marked as sequence parallel, False otherwise
    """
    return getattr(tensor, "_sequence_parallel", False)


def get_sequence_parallel_world_size() -> int:
    """Get the world size for sequence parallelism (same as TP size).

    Returns:
        Number of ranks in the sequence parallel group
    """
    return get_tensor_model_parallel_size()


def get_sequence_parallel_rank() -> int:
    """Get the rank for sequence parallelism (same as TP rank).

    Returns:
        Current rank in the sequence parallel group
    """
    return get_tensor_model_parallel_rank()


def get_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the process group for sequence parallelism (same as TP group).

    Returns:
        Process group used for sequence parallel communication
    """
    return get_tensor_model_parallel_group()


# ----------------------
# Performance Utilities
# ----------------------


class SequenceParallelBenchmark:
    """Benchmark utilities for sequence parallel operations."""

    def __init__(self) -> None:
        self.timings: Dict[str, list] = {}
        self.memory_usage: Dict[str, list] = {}

    def benchmark_operation(
        self,
        operation: Callable,
        input_tensor: torch.Tensor,
        operation_name: str,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Dict[str, float]:
        """Benchmark a sequence parallel operation.

        Args:
            operation: Operation to benchmark
            input_tensor: Input tensor for the operation
            operation_name: Name for logging
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary with timing and memory statistics
        """
        import time

        # Warmup
        for _ in range(warmup_iterations):
            _ = operation(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Benchmark
        timings = []
        memory_before = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        for _ in range(num_iterations):
            start_time = time.perf_counter()

            _ = operation(input_tensor)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            timings.append(elapsed)

        memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = (memory_after - memory_before) / 1024**3  # Convert to GB

        # Calculate statistics
        avg_time = sum(timings) / len(timings)
        min_time = min(timings)
        max_time = max(timings)

        # Store results
        self.timings[operation_name] = timings
        self.memory_usage[operation_name] = [memory_used]

        return {
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "memory_gb": memory_used,
            "throughput_gb_s": (
                (input_tensor.numel() * input_tensor.element_size() / 1024**3)
                / avg_time
            ),
        }

    def generate_report(self) -> str:
        """Generate a performance report.

        Returns:
            Formatted performance report string
        """
        report = "\nSequence Parallel Performance Report\n"
        report += "=" * 50 + "\n"

        for op_name, timings in self.timings.items():
            avg_time = sum(timings) / len(timings) * 1000  # Convert to ms
            memory = self.memory_usage.get(op_name, [0])[0]

            report += f"\n{op_name}:\n"
            report += f"  Average Time: {avg_time:.3f} ms\n"
            report += f"  Memory Used: {memory:.3f} GB\n"

        return report


def validate_sequence_parallel_invariants(
    original_tensor: torch.Tensor,
    scattered_tensors: list,
    world_size: int,
) -> bool:
    """Validate sequence parallel invariants after scatter operation.

    Args:
        original_tensor: Original tensor before scatter
        scattered_tensors: List of scattered tensors from all ranks
        world_size: Number of ranks

    Returns:
        True if all invariants hold, False otherwise
    """
    # Check that concatenation recreates original
    reconstructed = torch.cat(scattered_tensors, dim=0)
    if not torch.allclose(original_tensor, reconstructed, rtol=1e-5, atol=1e-7):
        logger.error("Failed to reconstruct original tensor from scattered pieces")
        return False

    # Check that each piece has correct size
    expected_size = original_tensor.shape[0] // world_size
    for i, tensor in enumerate(scattered_tensors):
        if tensor.shape[0] != expected_size:
            logger.error(
                f"Rank {i} has incorrect sequence length: "
                f"{tensor.shape[0]} != {expected_size}"
            )
            return False

    # Check that sum is preserved (for reduce-scatter)
    original_sum = original_tensor.sum()
    scattered_sum = torch.stack([t.sum() for t in scattered_tensors]).sum()
    if not torch.allclose(original_sum, scattered_sum, rtol=1e-5, atol=1e-7):
        logger.error("Sum not preserved after scatter operation")
        return False

    return True
