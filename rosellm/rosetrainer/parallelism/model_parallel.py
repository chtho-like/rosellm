from typing import Optional

import torch
import torch.distributed as dist

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
    ):
        """
        Initialize the TensorParallelism manager.

        Args:
            local_rank: Local rank of the current process.
            world_size: Total number of processes.
            tp_size: Size of the tensor parallel group.
            tp_group: Optional existing tensor parallel process group.
        """
        self.local_rank = local_rank
        self.world_size = world_size
        self.tp_size = tp_size

        # Ensure world_size is divisible by tp_size
        assert (
            world_size % tp_size == 0
        ), "World size must be divisible by tensor parallel size"

        # Set up process groups if not provided
        if tp_group is None:
            self.tp_group = self._initialize_process_group()
        else:
            self.tp_group = tp_group

        # Get tensor parallel rank
        self.tp_rank = dist.get_rank(self.tp_group)

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
        """
        # Calculate the split size
        split_size = tensor.size(dim) // self.tp_size

        # Handle uneven division
        if tensor.size(dim) % self.tp_size != 0:
            # Pad tensor to make it evenly divisible
            pad_size = self.tp_size - (tensor.size(dim) % self.tp_size)
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad_size
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=dim)
            # Recalculate split size
            split_size = tensor.size(dim) // self.tp_size

        # Split the tensor
        chunks = torch.split(tensor, split_size, dim=dim)
        return chunks[self.tp_rank]

    def gather_tensor(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Gather a tensor that has been split across devices.

        Args:
            tensor: The local tensor.
            dim: The dimension to gather on.

        Returns:
            The gathered tensor.
        """
        # First, gather sizes to handle uneven splits
        local_size = torch.tensor([tensor.size(dim)], device=tensor.device)
        sizes = [torch.zeros_like(local_size) for _ in range(self.tp_size)]
        dist.all_gather(sizes, local_size, group=self.tp_group)

        # Convert sizes to integers
        sizes_int = [int(size.item()) for size in sizes]

        # All-gather tensors
        tensors = []
        for size in sizes_int:
            # Create shape dynamically
            shape = list(tensor.shape)
            shape[dim] = size
            # Create tensor with correct shape
            tensors.append(torch.zeros(shape, dtype=tensor.dtype, device=tensor.device))

        dist.all_gather(tensors, tensor, group=self.tp_group)

        # Concatenate along specified dimension
        return torch.cat(tensors, dim=dim)

    def parallelize_layer(
        self, layer: torch.nn.Module, tp_type: str
    ) -> torch.nn.Module:
        """
        Parallelize a layer according to its type.

        Args:
            layer: The layer to parallelize.
            tp_type: Type of tensor parallelism to apply
                    ('row', 'column', or 'attention').

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
    ):
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

    def _init_weights(self) -> None:
        """Initialize the weights and bias."""
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _split_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Split the weight tensor along the output dimension."""
        # Weight is (out_features, in_features)
        output_chunks = torch.split(weight, self.output_size_per_partition, dim=0)
        return output_chunks[self.tp_rank]

    def _split_bias(self, bias: torch.Tensor) -> torch.Tensor:
        """Split the bias tensor along the output dimension."""
        bias_chunks = torch.split(bias, self.output_size_per_partition, dim=0)
        return bias_chunks[self.tp_rank]

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear.

        Args:
            input_: Input tensor.

        Returns:
            Output tensor.
        """
        # Local linear operation
        output = torch.nn.functional.linear(input_, self.weight, self.bias)
        return output


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
    ):
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

    def _init_weights(self) -> None:
        """Initialize the weights and bias."""
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _split_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Split the weight tensor along the input dimension."""
        # Weight is (out_features, in_features)
        input_chunks = torch.split(weight, self.input_size_per_partition, dim=1)
        return input_chunks[self.tp_rank]

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear.

        Args:
            input_: Input tensor.

        Returns:
            Output tensor.
        """
        # Partition input tensor along sequence dimension
        input_parallel = torch.split(input_, self.input_size_per_partition, dim=-1)[
            self.tp_rank
        ]

        # Local linear operation
        output_parallel = torch.nn.functional.linear(input_parallel, self.weight)

        # All-reduce across processes
        if not self.skip_all_reduce and self.tp_size > 1:
            dist.all_reduce(output_parallel, group=self.tp_group)

        # Add bias
        if self.bias is not None:
            output_parallel = output_parallel + self.bias

        return output_parallel
