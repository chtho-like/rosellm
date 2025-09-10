import logging
from typing import Optional

import torch
import torch.distributed as dist

from ..memory.global_memory_buffer import BufferType, allocate_tensor, release_tensor
from .async_linear import register_parameter_for_async_allreduce
from .parallel_state import get_global_memory_buffer as get_buffer_from_state

logger = logging.getLogger(__name__)

# References:
# [1] Shoeybi, M. et al. "Megatron-LM: Training Multi-Billion Parameter
#     Language Models Using Model Parallelism." arXiv:1909.08053 (2019)
# [2] Narayanan, D. et al. "Efficient Large-Scale Language Model Training
#     on GPU Clusters Using Megatron-LM." arXiv:2104.04473 (2021)
# [3] Lepikhin, D. et al. "GShard: Scaling Giant Models with Conditional
#     Computation and Automatic Sharding." arXiv:2006.16668 (2020)
# [4] DeepSpeed implementation:
#     https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/runtime/pipe
# [5] Korthikanti, V. et al. "Reducing Activation Recomputation in
#     Large Transformer Models." arXiv:2205.05198 (2022)


class TensorParallelism:
    """
    Tensor parallelism implementation for RoseLLM.
    Splits model parameters across devices for large models that don't fit
    on a single GPU.
    """

    def __init__(
        self,
        local_rank: int,
        world_size: int,
        tp_size: int,
        tp_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """
        Initialize the TensorParallelism manager.

        Args:
            local_rank: Local rank of the current process.
            world_size: Total number of processes.
            tp_size: Size of the tensor parallel group.
            tp_group: Optional existing tensor parallel process group.

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If distributed setup is invalid
        """
        # Validate inputs
        if not isinstance(local_rank, int) or local_rank < 0:
            raise ValueError(
                f"local_rank must be non-negative integer, got {local_rank}"
            )
        if not isinstance(world_size, int) or world_size <= 0:
            raise ValueError(f"world_size must be positive integer, got {world_size}")
        if not isinstance(tp_size, int) or tp_size <= 0:
            raise ValueError(f"tp_size must be positive integer, got {tp_size}")
        if local_rank >= world_size:
            raise ValueError(
                f"local_rank ({local_rank}) must be less than "
                f"world_size ({world_size})"
            )
        if world_size % tp_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"tp_size ({tp_size})"
            )
        if tp_size > world_size:
            raise ValueError(
                f"tp_size ({tp_size}) cannot be larger than "
                f"world_size ({world_size})"
            )

        self.local_rank = local_rank
        self.world_size = world_size
        self.tp_size = tp_size

        # Set up process groups if not provided
        if tp_group is None:
            self.tp_group = self._initialize_process_group()
        else:
            self.tp_group = tp_group

        # Get tensor parallel rank with error handling
        try:
            self.tp_rank = dist.get_rank(self.tp_group) if self.tp_group else 0
        except Exception as e:
            raise RuntimeError(f"Failed to get tensor parallel rank: {e}") from e

        logger.info(
            f"Initialized TensorParallelism: tp_size={tp_size}, "
            f"tp_rank={self.tp_rank}, world_size={world_size}"
        )

    def _initialize_process_group(self) -> dist.ProcessGroup:
        """Initialize the tensor parallel process group."""
        # Make sure global process group is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "Distributed is not initialized. Call dist.init_process_group first."
            )

        # Calculate the number of TP groups
        num_tp_groups = self.world_size // self.tp_size

        # Determine the TP group this rank belongs to
        tp_group_id = self.local_rank // self.tp_size

        # Create TP groups
        tp_groups = []
        for i in range(num_tp_groups):
            ranks_in_group = [i * self.tp_size + j for j in range(self.tp_size)]
            tp_groups.append(dist.new_group(ranks=ranks_in_group))

        # Return the TP group for this rank
        return tp_groups[tp_group_id]  # type: ignore[no-any-return]

    def split_tensor(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Split a tensor along the specified dimension for tensor parallelism.

        Args:
            tensor: The tensor to split.
            dim: The dimension to split on.

        Returns:
            The split tensor for this TP rank.

        Raises:
            ValueError: If tensor or dimension is invalid
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if tensor.numel() == 0:
            raise ValueError("Cannot split empty tensor")
        if not (-tensor.ndim <= dim < tensor.ndim):
            raise ValueError(
                f"dim {dim} out of range for tensor with {tensor.ndim} dimensions"
            )
        if tensor.size(dim) < self.tp_size:
            raise ValueError(
                f"Tensor dimension {dim} size ({tensor.size(dim)}) is smaller "
                f"than tp_size ({self.tp_size})"
            )

        # Normalize negative dimension
        if dim < 0:
            dim = tensor.ndim + dim

        # Calculate the split size
        split_size = tensor.size(dim) // self.tp_size

        # Handle uneven division with more efficient padding
        if tensor.size(dim) % self.tp_size != 0:
            logger.warning(
                f"Tensor size {tensor.size(dim)} not evenly divisible by "
                f"tp_size {self.tp_size}, padding will be applied"
            )
            # Pad tensor to make it evenly divisible
            pad_size = self.tp_size - (tensor.size(dim) % self.tp_size)

            # Create padding specification for torch.nn.functional.pad
            # pad format is (pad_left, pad_right) for the last dim,
            # then second-to-last, etc.
            pad_spec = [0] * (tensor.ndim * 2)
            # pad_right for the specified dimension
            pad_spec[-(dim + 1) * 2 - 1] = pad_size

            tensor = torch.nn.functional.pad(tensor, pad_spec, mode="constant", value=0)
            # Recalculate split size
            split_size = tensor.size(dim) // self.tp_size

        # Split the tensor efficiently
        try:
            chunks = torch.split(tensor, split_size, dim=dim)
            return chunks[self.tp_rank].contiguous()  # Ensure contiguous memory layout
        except Exception as e:
            raise RuntimeError(f"Failed to split tensor: {e}") from e

    def gather_tensor(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Gather a tensor that has been split across devices.

        Args:
            tensor: The local tensor.
            dim: The dimension to gather on.

        Returns:
            The gathered tensor.

        Raises:
            ValueError: If tensor or dimension is invalid
            RuntimeError: If gather operation fails
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if tensor.numel() == 0:
            raise ValueError("Cannot gather empty tensor")
        if not (-tensor.ndim <= dim < tensor.ndim):
            raise ValueError(
                f"dim {dim} out of range for tensor with {tensor.ndim} dimensions"
            )
        if self.tp_group is None:
            raise RuntimeError(
                "Cannot gather tensor: no tensor parallel group available"
            )

        # Normalize negative dimension
        if dim < 0:
            dim = tensor.ndim + dim

        try:
            # First, gather sizes to handle uneven splits
            local_size = torch.tensor(
                [tensor.size(dim)], device=tensor.device, dtype=torch.long
            )
            sizes = [torch.zeros_like(local_size) for _ in range(self.tp_size)]
            dist.all_gather(sizes, local_size, group=self.tp_group)

            # Convert sizes to integers
            sizes_int = [int(size.item()) for size in sizes]
            total_size = sum(sizes_int)

            if total_size == 0:
                raise ValueError("Total gathered size is zero")

            # All-gather tensors with optimized buffer management
            tensors = []
            buffer_manager = get_buffer_from_state()
            use_buffer_manager = buffer_manager is not None

            # Pre-allocate output tensors
            if use_buffer_manager:
                try:
                    for i, size in enumerate(sizes_int):
                        # Create shape dynamically
                        shape = list(tensor.shape)
                        shape[dim] = size
                        # Allocate from global buffer
                        buf_tensor = allocate_tensor(
                            tuple(shape),
                            dtype=tensor.dtype,
                            device=tensor.device,
                            buffer_type=BufferType.COMMUNICATION,
                            caller_info=f"TensorParallelism.gather_tensor.{i}",
                        )
                        tensors.append(buf_tensor)
                except Exception as e:
                    logger.warning(
                        f"Failed to allocate from buffer manager: {e}, "
                        "falling back to regular allocation"
                    )
                    use_buffer_manager = False
                    # Clean up any allocated tensors
                    for t in tensors:
                        try:
                            release_tensor(t)
                        except Exception:
                            pass
                    tensors.clear()

            if not use_buffer_manager:
                # Fallback to regular allocation
                for size in sizes_int:
                    shape = list(tensor.shape)
                    shape[dim] = size
                    tensors.append(
                        torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
                    )

            # Perform all-gather
            dist.all_gather(tensors, tensor, group=self.tp_group)

            # Concatenate along specified dimension
            result = torch.cat(tensors, dim=dim)

            # Release buffers if we used them
            if use_buffer_manager:
                for t in tensors:
                    try:
                        release_tensor(t)
                    except Exception as e:
                        logger.warning(f"Error releasing tensor buffer: {e}")

            return result

        except Exception as e:
            logger.error(f"Error during tensor gather: {e}")
            raise RuntimeError(f"Tensor gather failed: {e}") from e

    def parallelize_layer(
        self,
        layer: torch.nn.Module,
        tp_type: str,
        enable_async_allreduce: bool = False,
        layer_name: Optional[str] = None,
    ) -> torch.nn.Module:
        """
        Parallelize a layer according to its type.

        Args:
            layer: The layer to parallelize.
            tp_type: Type of tensor parallelism to apply
                    ('row', 'column', or 'attention').
            enable_async_allreduce: Enable async gradient allreduce for this layer.
            layer_name: Name of this layer for async allreduce prioritization.

        Returns:
            The parallelized layer.
        """
        if tp_type == "row" and isinstance(layer, torch.nn.Linear):
            return RowParallelLinear(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                tp_group=self.tp_group,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                layer=layer,
                enable_async_allreduce=enable_async_allreduce,
                layer_name=layer_name,
            )
        elif tp_type == "column" and isinstance(layer, torch.nn.Linear):
            return ColumnParallelLinear(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                tp_group=self.tp_group,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                layer=layer,
                enable_async_allreduce=enable_async_allreduce,
                layer_name=layer_name,
            )
        else:
            # Unsupported layer or type combo
            return layer


class ColumnParallelLinear(torch.nn.Module):
    """
    Linear layer with column parallelism.

    The linear layer is split along the output dimension, on each process
    only a subset of the columns is computed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_group: Optional[dist.ProcessGroup] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        layer: Optional[torch.nn.Linear] = None,
        enable_async_allreduce: bool = False,
        layer_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the column parallel linear layer.

        Args:
            in_features: Input feature size.
            out_features: Output feature size (will be divided by tp_size).
            bias: Whether to include bias.
            tp_group: Tensor parallel process group.
            tp_size: Size of tensor parallel group.
            tp_rank: Rank in tensor parallel group.
            layer: Existing layer to parallelize (optional).
            enable_async_allreduce: Enable async gradient allreduce for this layer.
            layer_name: Name of this layer for async allreduce prioritization.
        """
        super(ColumnParallelLinear, self).__init__()

        # Calculate local output size
        self.in_features = in_features
        self.out_features = out_features
        self.output_size_per_partition = out_features // tp_size

        # Store TP params
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # Async allreduce configuration
        self.enable_async_allreduce = enable_async_allreduce
        self.layer_name = layer_name

        # Create weights
        if layer is not None:
            # Split existing layer
            self.weight = torch.nn.Parameter(self._split_weight(layer.weight))
            if bias and layer.bias is not None:
                self.bias = torch.nn.Parameter(self._split_bias(layer.bias))
            else:
                self.register_parameter("bias", None)
        else:
            # Initialize new weights
            self.weight = torch.nn.Parameter(
                torch.empty(self.output_size_per_partition, in_features)
            )
            if bias:
                self.bias = torch.nn.Parameter(
                    torch.empty(self.output_size_per_partition)
                )
            else:
                self.register_parameter("bias", None)
            self._init_weights()

        # Register parameters for async allreduce if enabled
        if self.enable_async_allreduce:
            self._register_async_allreduce()

    def _init_weights(self) -> None:
        """Initialize the weights and bias."""
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _split_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Split the weight tensor along the output dimension."""
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(weight)}")
        if weight.size(0) < self.tp_size:
            raise ValueError(
                f"Weight output dimension {weight.size(0)} is smaller than "
                f"tp_size {self.tp_size}"
            )

        # Weight is (out_features, in_features)
        output_chunks = torch.split(weight, self.output_size_per_partition, dim=0)
        return output_chunks[self.tp_rank].contiguous()

    def _split_bias(self, bias: torch.Tensor) -> torch.Tensor:
        """Split the bias tensor along the output dimension."""
        if not isinstance(bias, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(bias)}")
        if bias.size(0) < self.tp_size:
            raise ValueError(
                f"Bias size {bias.size(0)} is smaller than tp_size {self.tp_size}"
            )

        bias_chunks = torch.split(bias, self.output_size_per_partition, dim=0)
        return bias_chunks[self.tp_rank].contiguous()

    def _register_async_allreduce(self) -> None:
        """Register parameters for async gradient allreduce."""
        try:
            weight_layer_name = (
                f"{self.layer_name}.weight" if self.layer_name else "weight"
            )
            register_parameter_for_async_allreduce(self.weight, weight_layer_name)
            if self.bias is not None:
                bias_layer_name = (
                    f"{self.layer_name}.bias" if self.layer_name else "bias"
                )
                register_parameter_for_async_allreduce(self.bias, bias_layer_name)
            logger.debug(
                f"Registered async allreduce for ColumnParallelLinear layer "
                f"{self.layer_name or 'unnamed'}"
            )
        except Exception as e:
            logger.error(
                f"Failed to register async allreduce for ColumnParallelLinear: {e}"
            )
            # Don't re-raise to avoid breaking model initialization

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear.

        Args:
            input_: Input tensor.

        Returns:
            Output tensor.

        Raises:
            ValueError: If input tensor is invalid
        """
        if not isinstance(input_, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(input_)}")
        if input_.size(-1) != self.in_features:
            raise ValueError(
                f"Input feature dimension mismatch: expected {self.in_features}, "
                f"got {input_.size(-1)}"
            )

        try:
            # Local linear operation
            output = torch.nn.functional.linear(input_, self.weight, self.bias)
            return output
        except Exception as e:
            logger.error(f"Error in ColumnParallelLinear forward: {e}")
            raise RuntimeError(f"ColumnParallelLinear forward failed: {e}") from e


class RowParallelLinear(torch.nn.Module):
    """
    Linear layer with row parallelism.

    The linear layer is split along the input dimension, rows of the weight
    matrix are partitioned, and a AllReduce is performed after the computation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_group: Optional[dist.ProcessGroup] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        layer: Optional[torch.nn.Linear] = None,
        skip_all_reduce: bool = False,
        enable_async_allreduce: bool = False,
        layer_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the row parallel linear layer.

        Args:
            in_features: Input feature size.
            out_features: Output feature size.
            bias: Whether to include bias.
            tp_group: Tensor parallel process group.
            tp_size: Size of tensor parallel group.
            tp_rank: Rank in tensor parallel group.
            layer: Existing layer to parallelize (optional).
            skip_all_reduce: Whether to skip the all-reduce step.
            enable_async_allreduce: Enable async gradient allreduce for this layer.
            layer_name: Name of this layer for async allreduce prioritization.
        """
        super(RowParallelLinear, self).__init__()

        # Calculate local input size
        self.in_features = in_features
        self.out_features = out_features
        self.input_size_per_partition = in_features // tp_size

        # Store TP params
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.skip_all_reduce = skip_all_reduce

        # Async allreduce configuration
        self.enable_async_allreduce = enable_async_allreduce
        self.layer_name = layer_name

        # Create weights
        if layer is not None:
            # Split existing layer
            self.weight = torch.nn.Parameter(self._split_weight(layer.weight))
            if bias and layer.bias is not None:
                self.bias = torch.nn.Parameter(layer.bias.clone())
            else:
                self.register_parameter("bias", None)
        else:
            # Initialize new weights
            self.weight = torch.nn.Parameter(
                torch.empty(out_features, self.input_size_per_partition)
            )
            if bias:
                self.bias = torch.nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter("bias", None)
            self._init_weights()

        # Register parameters for async allreduce if enabled
        if self.enable_async_allreduce:
            self._register_async_allreduce()

    def _init_weights(self) -> None:
        """Initialize the weights and bias."""
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _split_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Split the weight tensor along the input dimension."""
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(weight)}")
        if weight.size(1) < self.tp_size:
            raise ValueError(
                f"Weight input dimension {weight.size(1)} is smaller than "
                f"tp_size {self.tp_size}"
            )

        # Weight is (out_features, in_features)
        input_chunks = torch.split(weight, self.input_size_per_partition, dim=1)
        return input_chunks[self.tp_rank].contiguous()

    def _register_async_allreduce(self) -> None:
        """Register parameters for async gradient allreduce."""
        try:
            weight_layer_name = (
                f"{self.layer_name}.weight" if self.layer_name else "weight"
            )
            register_parameter_for_async_allreduce(self.weight, weight_layer_name)
            if self.bias is not None:
                bias_layer_name = (
                    f"{self.layer_name}.bias" if self.layer_name else "bias"
                )
                register_parameter_for_async_allreduce(self.bias, bias_layer_name)
            logger.debug(
                f"Registered async allreduce for RowParallelLinear layer "
                f"{self.layer_name or 'unnamed'}"
            )
        except Exception as e:
            logger.error(
                f"Failed to register async allreduce for RowParallelLinear: {e}"
            )
            # Don't re-raise to avoid breaking model initialization

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear.

        Args:
            input_: Input tensor.

        Returns:
            Output tensor.

        Raises:
            ValueError: If input tensor is invalid
            RuntimeError: If forward pass fails
        """
        if not isinstance(input_, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(input_)}")
        if input_.size(-1) != self.in_features:
            raise ValueError(
                f"Input feature dimension mismatch: expected {self.in_features}, "
                f"got {input_.size(-1)}"
            )

        try:
            # Partition input tensor along last dimension
            input_chunks = torch.split(input_, self.input_size_per_partition, dim=-1)
            if self.tp_rank >= len(input_chunks):
                raise RuntimeError(
                    f"tp_rank {self.tp_rank} exceeds number of input chunks "
                    f"{len(input_chunks)}"
                )

            input_parallel = input_chunks[self.tp_rank].contiguous()

            # Local linear operation
            output_parallel = torch.nn.functional.linear(input_parallel, self.weight)

            # All-reduce across processes
            if (
                not self.skip_all_reduce
                and self.tp_size > 1
                and self.tp_group is not None
            ):
                buffer_manager = get_buffer_from_state()
                use_buffer_manager = buffer_manager is not None

                # Use global buffer for all-reduce if available
                if use_buffer_manager:
                    try:
                        # Allocate buffer for all-reduce operation
                        comm_buffer = allocate_tensor(
                            tuple(output_parallel.shape),
                            dtype=output_parallel.dtype,
                            device=output_parallel.device,
                            buffer_type=BufferType.COMMUNICATION,
                            caller_info="RowParallelLinear.forward",
                        )
                        comm_buffer.copy_(output_parallel, non_blocking=True)
                        dist.all_reduce(comm_buffer, group=self.tp_group)
                        output_parallel.copy_(comm_buffer, non_blocking=True)
                        release_tensor(comm_buffer)
                    except Exception as e:
                        logger.warning(
                            f"Buffer manager allocation failed: {e}, "
                            "falling back to in-place allreduce"
                        )
                        dist.all_reduce(output_parallel, group=self.tp_group)
                else:
                    dist.all_reduce(output_parallel, group=self.tp_group)

            # Add bias
            if self.bias is not None:
                output_parallel = output_parallel + self.bias

            return output_parallel

        except Exception as e:
            logger.error(f"Error in RowParallelLinear forward: {e}")
            raise RuntimeError(f"RowParallelLinear forward failed: {e}") from e
