"""
Integration Layer for Parameter Overlap with Model Parallelism

This module integrates the async parameter gathering with existing tensor and
pipeline parallelism implementations, providing seamless overlap of communication
with computation in distributed training.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..memory.parameter_overlap import (
    AsyncParameterGatherer,
    OverlapConfig,
    OverlapMode,
    OverlappedLinear,
    PipelineOverlapScheduler,
)

try:
    from .parallel_state import (
        get_pipeline_model_parallel_group,
        get_pipeline_model_parallel_rank,
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
    )
except ImportError:
    from rosellm.rosetrainer.parallelism.parallel_state import (
        get_pipeline_model_parallel_group,
        get_pipeline_model_parallel_rank,
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
    )

logger = logging.getLogger(__name__)


class OverlappedColumnParallelLinear(nn.Module):
    """
    Column parallel linear layer with overlapped parameter gathering.

    This layer splits the weight matrix column-wise across tensor parallel ranks
    and overlaps parameter gathering with computation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        gatherer: Optional[AsyncParameterGatherer] = None,
        skip_bias_add: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize column parallel linear layer with overlap.

        Args:
            in_features: Input features.
            out_features: Output features (total across all ranks).
            bias: Whether to use bias.
            gather_output: Whether to gather output across ranks.
            gatherer: Async parameter gatherer.
            skip_bias_add: Skip bias addition for fusion.
            device: Device for the layer.
            dtype: Data type for the layer.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.gatherer = gatherer

        # Get tensor parallel info
        tp_group = get_tensor_model_parallel_group()
        tp_size = dist.get_world_size(tp_group)

        # Calculate local output features
        assert out_features % tp_size == 0, "out_features must be divisible by tp_size"
        self.local_out_features = out_features // tp_size

        # Initialize weight and bias for this rank's column slice
        self.weight = nn.Parameter(
            torch.empty(
                self.local_out_features, in_features, device=device, dtype=dtype
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.local_out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # Setup gatherer if not provided
        if self.gatherer is None and device and device.type == "cuda":
            config = OverlapConfig(mode=OverlapMode.PIPELINE)
            self.gatherer = AsyncParameterGatherer(config, tp_group, device)

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain("relu"))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with overlapped communication.

        Args:
            input: Input tensor [batch_size, seq_len, in_features].

        Returns:
            Output tensor [batch_size, seq_len, local_out_features] or
            [batch_size, seq_len, out_features] if gather_output=True.
        """
        # Schedule async weight gathering if using overlap
        if self.gatherer and self.gatherer.config.mode != OverlapMode.NONE:
            # Prefetch weight to input device
            weight_future = self.gatherer.gather_async(
                param_id=f"col_weight_{id(self)}",
                tensor=self.weight,
                target_device=input.device,
                priority=2,
            )

            # Start any pending computation while waiting
            if input.requires_grad:
                input.retain_grad()

            # Get gathered weight
            weight = weight_future.result()
        else:
            weight = self.weight

        # Compute local matrix multiplication
        output = torch.matmul(input, weight.t())

        # Add bias if needed
        if self.bias is not None and not self.skip_bias_add:
            output = output + self.bias

        # Gather output across tensor parallel ranks if needed
        if self.gather_output:
            tp_group = get_tensor_model_parallel_group()

            # Schedule async all-gather
            if self.gatherer:
                # Use async all-gather with overlap
                gather_list = [
                    torch.empty_like(output)
                    for _ in range(dist.get_world_size(tp_group))
                ]
                gather_handle = dist.all_gather(
                    gather_list, output, group=tp_group, async_op=True
                )

                # Overlap with bias add if skipped earlier
                if self.bias is not None and self.skip_bias_add:
                    output = output + self.bias

                # Wait for gather to complete
                if gather_handle is not None:
                    gather_handle.wait()
                output = torch.cat(gather_list, dim=-1)
            else:
                # Standard synchronous all-gather
                gather_list = [
                    torch.empty_like(output)
                    for _ in range(dist.get_world_size(tp_group))
                ]
                dist.all_gather(gather_list, output, group=tp_group)
                output = torch.cat(gather_list, dim=-1)

        return output


class OverlappedRowParallelLinear(nn.Module):
    """
    Row parallel linear layer with overlapped parameter gathering.

    This layer splits the weight matrix row-wise across tensor parallel ranks
    and overlaps parameter gathering with computation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        gatherer: Optional[AsyncParameterGatherer] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize row parallel linear layer with overlap.

        Args:
            in_features: Input features (total across all ranks).
            out_features: Output features.
            bias: Whether to use bias.
            input_is_parallel: Whether input is already split across ranks.
            gatherer: Async parameter gatherer.
            device: Device for the layer.
            dtype: Data type for the layer.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.gatherer = gatherer

        # Get tensor parallel info
        tp_group = get_tensor_model_parallel_group()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = dist.get_world_size(tp_group)

        # Calculate local input features
        assert in_features % tp_size == 0, "in_features must be divisible by tp_size"
        self.local_in_features = in_features // tp_size

        # Initialize weight for this rank's row slice
        self.weight = nn.Parameter(
            torch.empty(
                out_features, self.local_in_features, device=device, dtype=dtype
            )
        )

        # Bias is only on rank 0 for row parallel
        if bias and tp_rank == 0:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # Setup gatherer if not provided
        if self.gatherer is None and device and device.type == "cuda":
            config = OverlapConfig(mode=OverlapMode.PIPELINE)
            self.gatherer = AsyncParameterGatherer(config, tp_group, device)

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain("relu"))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with overlapped communication.

        Args:
            input: Input tensor [batch_size, seq_len, in_features] or
                   [batch_size, seq_len, local_in_features] if input_is_parallel=True.

        Returns:
            Output tensor [batch_size, seq_len, out_features].
        """
        tp_group = get_tensor_model_parallel_group()

        # Split input if not already parallel
        if not self.input_is_parallel:
            # Split input across tensor parallel dimension
            tp_size = dist.get_world_size(tp_group)
            input_list = torch.chunk(input, tp_size, dim=-1)
            local_input = input_list[get_tensor_model_parallel_rank()]
        else:
            local_input = input

        # Schedule async weight gathering if using overlap
        if self.gatherer and self.gatherer.config.mode != OverlapMode.NONE:
            weight_future = self.gatherer.gather_async(
                param_id=f"row_weight_{id(self)}",
                tensor=self.weight,
                target_device=local_input.device,
                priority=2,
            )
            weight = weight_future.result()
        else:
            weight = self.weight

        # Compute local matrix multiplication
        local_output = torch.matmul(local_input, weight.t())

        # All-reduce output across tensor parallel ranks
        if self.gatherer:
            # Async all-reduce with overlap
            reduce_handle = dist.all_reduce(local_output, group=tp_group, async_op=True)

            # Overlap with other operations
            if self.bias is not None:
                # Prepare bias addition while reducing
                bias = self.bias

            # Wait for reduction
            if reduce_handle is not None:
                reduce_handle.wait()
            output = local_output

            # Add bias after reduction
            if self.bias is not None:
                output = output + bias
        else:
            # Standard synchronous all-reduce
            dist.all_reduce(local_output, group=tp_group)
            output = local_output

            if self.bias is not None:
                output = output + self.bias

        return output


class OverlappedPipelineEngine:
    """
    Pipeline parallel engine with overlapped gradient communication.

    This engine manages pipeline parallel training with overlapped
    gradient reduction and parameter prefetching.
    """

    def __init__(
        self,
        model: nn.Module,
        num_stages: int,
        num_microbatches: int,
        overlap_config: Optional[OverlapConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize overlapped pipeline engine.

        Args:
            model: Model for this pipeline stage.
            num_stages: Total number of pipeline stages.
            num_microbatches: Number of microbatches.
            overlap_config: Configuration for overlap optimization.
            device: Device for operations.
        """
        self.model = model
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Setup overlap components
        config = overlap_config or OverlapConfig(mode=OverlapMode.PIPELINE)
        pp_group = get_pipeline_model_parallel_group()
        self.gatherer = AsyncParameterGatherer(config, pp_group, self.device)
        self.scheduler = PipelineOverlapScheduler(
            num_stages, num_microbatches, self.gatherer, self.device
        )

        # Get pipeline rank
        self.stage_id = get_pipeline_model_parallel_rank()
        self.is_first_stage = self.stage_id == 0
        self.is_last_stage = self.stage_id == num_stages - 1

        # Communication streams
        self.send_stream = torch.cuda.Stream() if self.device.type == "cuda" else None
        self.recv_stream = torch.cuda.Stream() if self.device.type == "cuda" else None

    def forward_step(
        self,
        microbatch_id: int,
        input_tensor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward step with overlapped communication.

        Args:
            microbatch_id: Microbatch index.
            input_tensor: Input tensor from previous stage.

        Returns:
            Output tensor for next stage.
        """
        # Prefetch parameters for next microbatch
        if microbatch_id < self.num_microbatches - 1:
            params = [p.data for p in self.model.parameters() if p.requires_grad]
            self.scheduler.prefetch_next_parameters(self.stage_id, params)

        # Receive input from previous stage if not first
        if not self.is_first_stage and input_tensor is None:
            input_tensor = self._recv_tensor_async()

        # Forward pass through model
        output_tensor: torch.Tensor = self.model(input_tensor)

        # Send output to next stage if not last
        if not self.is_last_stage:
            self._send_tensor_async(output_tensor)

        return output_tensor

    def backward_step(
        self,
        microbatch_id: int,
        output_grad: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Backward step with overlapped gradient communication.

        Args:
            microbatch_id: Microbatch index.
            output_grad: Gradient from next stage.

        Returns:
            Input gradient for previous stage.
        """
        # Receive gradient from next stage if not last
        if not self.is_last_stage and output_grad is None:
            output_grad = self._recv_tensor_async()

        # Schedule async gradient reduction for previous microbatch
        if microbatch_id > 0:
            prev_grads = self._collect_gradients()
            self.scheduler.schedule_gradient_reduction(
                self.stage_id, microbatch_id - 1, prev_grads
            )

        # Backward pass
        input_grad = None
        if output_grad is not None:
            output_grad.backward(retain_graph=True)

            # Get input gradient if not first stage
            if not self.is_first_stage:
                # Find the input tensor and get its gradient
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        input_grad = param.grad.clone()
                        break

        # Send gradient to previous stage if not first
        if not self.is_first_stage and input_grad is not None:
            self._send_tensor_async(input_grad)

        return input_grad

    def _send_tensor_async(self, tensor: torch.Tensor) -> None:
        """Send tensor asynchronously to next/previous stage."""
        pp_group = get_pipeline_model_parallel_group()
        target_rank = (self.stage_id + 1) % self.num_stages

        if self.send_stream and self.device.type == "cuda":
            with torch.cuda.stream(self.send_stream):  # type: ignore
                dist.send(tensor.contiguous(), target_rank, group=pp_group)
        else:
            dist.send(tensor.contiguous(), target_rank, group=pp_group)

    def _recv_tensor_async(self) -> torch.Tensor:
        """Receive tensor asynchronously from previous/next stage."""
        pp_group = get_pipeline_model_parallel_group()
        source_rank = (self.stage_id - 1) % self.num_stages

        # Create buffer for receiving
        # This is a simplified version - in practice, you'd need to know the shape
        buffer = torch.empty((1, 512, 768), device=self.device)  # Example shape

        if self.recv_stream and self.device.type == "cuda":
            with torch.cuda.stream(self.recv_stream):  # type: ignore
                dist.recv(buffer, source_rank, group=pp_group)
        else:
            dist.recv(buffer, source_rank, group=pp_group)

        return buffer

    def _collect_gradients(self) -> torch.Tensor:
        """Collect all gradients from model parameters."""
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))

        if grads:
            return torch.cat(grads)
        else:
            return torch.zeros(1, device=self.device, dtype=torch.float32)

    def synchronize(self) -> None:
        """Synchronize all overlapped operations."""
        self.gatherer.synchronize()

        if self.send_stream:
            self.send_stream.synchronize()
        if self.recv_stream:
            self.recv_stream.synchronize()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about overlap efficiency."""
        return {
            "gatherer_stats": self.gatherer.get_stats(),
            "overlap_efficiency": self.scheduler.get_overlap_efficiency(),
            "stage_id": self.stage_id,
        }


def convert_to_overlapped_model(
    model: nn.Module,
    overlap_config: OverlapConfig,
    process_group: Optional[dist.ProcessGroup] = None,
) -> nn.Module:
    """
    Convert a model to use overlapped parameter gathering.

    This function replaces standard linear layers with overlapped versions
    that hide communication latency.

    Args:
        model: Model to convert.
        overlap_config: Configuration for overlap.
        process_group: Process group for communication.

    Returns:
        Converted model with overlapped layers.
    """
    # Find device from model parameters
    device = None
    for param in model.parameters():
        device = param.device
        break

    if device is None:
        # Fallback to default device if no parameters found
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gatherer = AsyncParameterGatherer(overlap_config, process_group, device)

    def replace_layer(module: nn.Module, name: str, child: nn.Module) -> None:
        """Replace a layer with its overlapped version."""
        if isinstance(child, nn.Linear):
            # Replace with overlapped linear
            overlapped = OverlappedLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                gatherer=gatherer,
                device=device,
                dtype=child.weight.dtype,
            )

            # Copy weights
            overlapped.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                overlapped.bias.data.copy_(child.bias.data)

            setattr(module, name, overlapped)

    # Recursively replace layers
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            replace_layer(model, name, child)
        else:
            convert_to_overlapped_model(child, overlap_config, process_group)

    return model
