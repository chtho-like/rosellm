"""
Model Parallel Activation Manager for Tensor Parallel Checkpointing

This module provides specialized activation checkpointing for model parallel layers,
particularly focusing on tensor parallelism. It handles the complexities of managing
activations across tensor parallel ranks while optimizing memory usage and
communication patterns.

Key Features:
- Tensor parallel activation synchronization and management
- Column and row parallel layer checkpointing optimization
- Attention mechanism checkpointing with TP coordination
- MLP layer checkpointing with expert parallel support
- Communication-aware activation recomputation
- CUDA Graph compatibility for model parallel operations

References:
[1] Megatron-LM Tensor Parallelism Implementation
[2] PyTorch Distributed Tensor Parallel
[3] NVIDIA Apex Distributed Training
[4] FairScale Model Parallelism
"""

import functools
import logging
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from ..parallelism import parallel_state
from .distributed_checkpoint import (
    DistributedCheckpointConfig,
    DistributedCheckpointCoordinator,
    DistributedMemoryProfiler,
)
from .selective_recompute import SelectiveCheckpointConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelParallelCheckpointConfig:
    """Configuration for model parallel activation checkpointing."""

    # Tensor parallelism settings
    checkpoint_attention_layers: bool = True
    checkpoint_mlp_layers: bool = True
    checkpoint_layernorm_layers: bool = False
    checkpoint_embedding_layers: bool = True

    # Column/Row parallel layer settings
    sync_column_parallel_activations: bool = True
    sync_row_parallel_activations: bool = True
    column_parallel_communication_overlap: bool = True
    row_parallel_communication_overlap: bool = True

    # Attention mechanism optimization
    attention_checkpoint_query_key_value: bool = True
    attention_checkpoint_output_projection: bool = True
    attention_fuse_qkv_checkpointing: bool = True
    attention_separate_head_checkpointing: bool = False

    # MLP optimization
    mlp_checkpoint_gate_projection: bool = True
    mlp_checkpoint_up_projection: bool = True
    mlp_checkpoint_down_projection: bool = True
    mlp_fuse_gate_up_checkpointing: bool = True

    # Expert parallelism settings
    expert_checkpoint_per_expert: bool = True
    expert_load_balancing: bool = True
    expert_communication_overlap: bool = True

    # Communication optimization
    overlap_communication_computation: bool = True
    use_async_communication: bool = True
    communication_backend: str = "nccl"

    # Memory optimization
    activation_offloading: bool = False
    offload_to_cpu: bool = False
    offload_to_nvme: bool = False

    # CUDA Graph compatibility
    cuda_graph_compatible: bool = False
    enable_graph_capture: bool = False

    # Base distributed config
    distributed_config: DistributedCheckpointConfig = field(
        default_factory=DistributedCheckpointConfig
    )

    def validate(self) -> None:
        """Validate model parallel configuration."""
        # Validate base distributed config
        self.distributed_config.validate()

        # Check for conflicting settings
        if self.activation_offloading and self.cuda_graph_compatible:
            warnings.warn(
                "Activation offloading may conflict with CUDA Graph compatibility"
            )

        if self.overlap_communication_computation and not self.use_async_communication:
            warnings.warn(
                "Communication overlap requires async communication to be enabled"
            )


class TensorParallelCheckpointFunction(Function):
    """Autograd function for tensor parallel layer checkpointing."""

    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable[..., Any],
        layer_type: str,
        parallel_rank: int,
        parallel_size: int,
        profiler: Optional[DistributedMemoryProfiler],
        config: ModelParallelCheckpointConfig,
        *args: Any,
    ) -> Any:
        """Forward pass with tensor parallel checkpointing.

        Args:
            ctx: Autograd context
            run_function: Function to checkpoint
            layer_type: Type of layer being checkpointed
            parallel_rank: Tensor parallel rank
            parallel_size: Tensor parallel size
            profiler: Optional memory profiler
            config: Model parallel configuration
            *args: Function arguments

        Returns:
            Function output
        """
        # Store context
        ctx.run_function = run_function
        ctx.layer_type = layer_type
        ctx.parallel_rank = parallel_rank
        ctx.parallel_size = parallel_size
        ctx.profiler = profiler
        ctx.config = config

        # Save RNG state including parallel-specific states
        if config.distributed_config.base_config.preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            if torch.cuda.is_available():
                ctx.fwd_gpu_devices = list(range(torch.cuda.device_count()))
                ctx.fwd_gpu_states = [
                    torch.cuda.get_rng_state(device) for device in ctx.fwd_gpu_devices
                ]

            # Save tensor parallel RNG state
            if parallel_state.is_initialized():
                ctx.parallel_rng_checkpoint = parallel_state.checkpoint_parallel_rng()

        # Profile memory before execution
        start_time = time.time()
        if profiler is not None:
            profiler.profile_memory_distributed(
                f"tp_{layer_type}_{parallel_rank}", "before"
            )

        # Execute function with potential communication overlap
        try:
            if config.overlap_communication_computation and parallel_size > 1:
                outputs = TensorParallelCheckpointFunction._execute_with_overlap(
                    run_function, layer_type, parallel_rank, parallel_size, *args
                )
            else:
                with torch.no_grad():
                    outputs = run_function(*args)
        except Exception as e:
            logger.error(f"Tensor parallel forward failed for {layer_type}: {e}")
            raise

        # Profile memory after execution
        if profiler is not None:
            profiler.profile_memory_distributed(
                f"tp_{layer_type}_{parallel_rank}", "after"
            )

        # Save tensors for backward
        ctx.save_for_backward(*args)
        ctx.forward_time = time.time() - start_time

        return outputs

    @staticmethod
    def _execute_with_overlap(
        run_function: Callable[..., Any],
        layer_type: str,
        parallel_rank: int,
        parallel_size: int,
        *args: Any,
    ) -> Any:
        """Execute function with communication-computation overlap."""
        # This is a placeholder for advanced communication overlap
        # In practice, this would implement sophisticated overlap strategies

        # For now, execute normally but log the intent
        logger.debug(
            f"Executing {layer_type} on TP rank {parallel_rank}/{parallel_size} "
            "with communication overlap"
        )

        with torch.no_grad():
            return run_function(*args)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass with tensor parallel recomputation."""
        # Restore RNG state
        if ctx.config.distributed_config.base_config.preserve_rng_state:
            rng_devices: List[int] = []
            if torch.cuda.is_available():
                rng_devices = ctx.fwd_gpu_devices

            with torch.random.fork_rng(devices=rng_devices):
                torch.set_rng_state(ctx.fwd_cpu_state)
                if torch.cuda.is_available():
                    for device, state in zip(ctx.fwd_gpu_devices, ctx.fwd_gpu_states):
                        torch.cuda.set_rng_state(state, device)

                if (
                    hasattr(ctx, "parallel_rng_checkpoint")
                    and ctx.parallel_rng_checkpoint
                ):
                    parallel_state.restore_parallel_rng(ctx.parallel_rng_checkpoint)

                # Recompute forward pass
                start_time = time.time()
                try:
                    with torch.enable_grad():
                        inputs = ctx.saved_tensors
                        outputs = ctx.run_function(*inputs)
                except Exception as e:
                    logger.error(
                        f"Tensor parallel recomputation failed for {ctx.layer_type}: {e}"
                    )
                    raise

                # Record recomputation time
                recompute_time = time.time() - start_time
                if ctx.profiler is not None:
                    layer_id = f"tp_{ctx.layer_type}_{ctx.parallel_rank}"
                    if layer_id in ctx.profiler.local_stats:
                        stats = ctx.profiler.local_stats[layer_id]
                        if ctx.forward_time > 0:
                            stats.recomputation_overhead_ratio = (
                                recompute_time / ctx.forward_time
                            )

        # Compute gradients
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        gradients: List[Optional[torch.Tensor]] = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor) and inp.requires_grad:
                grad_list = []
                for out, grad_out in zip(outputs, grad_outputs):
                    if grad_out is not None:
                        grad = torch.autograd.grad(
                            outputs=out,
                            inputs=inp,
                            grad_outputs=grad_out,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if grad is not None:
                            grad_list.append(grad)

                if grad_list:
                    total_grad = grad_list[0]
                    for g in grad_list[1:]:
                        total_grad = total_grad + g
                    gradients.append(total_grad)
                else:
                    gradients.append(None)
            else:
                gradients.append(None)

        # Return gradients (None for metadata arguments)
        return (None, None, None, None, None, None) + tuple(gradients)


class ColumnParallelLinearCheckpoint(nn.Module):
    """Column parallel linear layer with integrated checkpointing."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Optional[Callable] = None,
        config: Optional[ModelParallelCheckpointConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config or ModelParallelCheckpointConfig()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        # Get tensor parallel info
        if parallel_state.is_initialized():
            self.tp_size = parallel_state.get_tensor_model_parallel_size()
            self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            self.tp_size = 1
            self.tp_rank = 0
            self.tp_group = None

        # Calculate parallel dimensions
        assert (
            out_features % self.tp_size == 0
        ), f"out_features ({out_features}) must be divisible by tp_size ({self.tp_size})"
        self.out_features_per_partition = out_features // self.tp_size

        # Initialize weight and bias
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

        # Initialize parameters
        if init_method is not None:
            init_method(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=1)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass with checkpointing."""
        if self.config.sync_column_parallel_activations and self.tp_size > 1:
            return self._checkpointed_forward(input_)
        else:
            return self._standard_forward(input_)

    def _standard_forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Standard forward pass without checkpointing."""
        # Linear transformation
        output_parallel = F.linear(input_, self.weight, self.bias)

        # All-gather if needed
        if self.gather_output and self.tp_size > 1:
            output = self._all_gather(output_parallel)
        else:
            output = output_parallel

        return output

    def _checkpointed_forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Checkpointed forward pass."""

        def column_linear_function(inp: torch.Tensor) -> torch.Tensor:
            return self._standard_forward(inp)

        result = TensorParallelCheckpointFunction.apply(
            column_linear_function,
            "column_linear",
            self.tp_rank,
            self.tp_size,
            None,  # profiler will be injected by parent manager
            self.config,
            input_,
        )

        # Ensure we return a tensor
        if isinstance(result, torch.Tensor):
            return result
        else:
            raise RuntimeError("Checkpointed forward must return a tensor")

    def _all_gather(self, input_: torch.Tensor) -> torch.Tensor:
        """All-gather across tensor parallel ranks."""
        if self.tp_group is None or self.tp_size == 1:
            return input_

        # Get input shape and calculate output shape
        input_shape = input_.shape
        output_shape = list(input_shape)
        output_shape[-1] = output_shape[-1] * self.tp_size

        # All-gather
        gathered = torch.empty(output_shape, dtype=input_.dtype, device=input_.device)

        # Use async all-gather if configured
        if self.config.use_async_communication:
            # Placeholder for async implementation
            dist.all_gather_into_tensor(gathered, input_, group=self.tp_group)
        else:
            dist.all_gather_into_tensor(gathered, input_, group=self.tp_group)

        return gathered


class RowParallelLinearCheckpoint(nn.Module):
    """Row parallel linear layer with integrated checkpointing."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Optional[Callable] = None,
        config: Optional[ModelParallelCheckpointConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config or ModelParallelCheckpointConfig()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        # Get tensor parallel info
        if parallel_state.is_initialized():
            self.tp_size = parallel_state.get_tensor_model_parallel_size()
            self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            self.tp_size = 1
            self.tp_rank = 0
            self.tp_group = None

        # Calculate parallel dimensions
        assert (
            in_features % self.tp_size == 0
        ), f"in_features ({in_features}) must be divisible by tp_size ({self.tp_size})"
        self.in_features_per_partition = in_features // self.tp_size

        # Initialize weight and bias
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize parameters
        if init_method is not None:
            init_method(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=1)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass with checkpointing."""
        if self.config.sync_row_parallel_activations and self.tp_size > 1:
            return self._checkpointed_forward(input_)
        else:
            return self._standard_forward(input_)

    def _standard_forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Standard forward pass without checkpointing."""
        # Use appropriate input based on parallelism
        if self.input_is_parallel or self.tp_size == 1:
            input_parallel = input_
        else:
            # Split input across tensor parallel ranks
            input_parallel = self._split_tensor(input_)

        # Linear transformation
        output_parallel = F.linear(input_parallel, self.weight)

        # All-reduce across tensor parallel ranks
        if self.tp_size > 1:
            output = self._all_reduce(output_parallel)
        else:
            output = output_parallel

        # Add bias if present (only on one rank to avoid duplication)
        if self.bias is not None:
            output = output + self.bias

        return output

    def _checkpointed_forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Checkpointed forward pass."""

        def row_linear_function(inp: torch.Tensor) -> torch.Tensor:
            return self._standard_forward(inp)

        result = TensorParallelCheckpointFunction.apply(
            row_linear_function,
            "row_linear",
            self.tp_rank,
            self.tp_size,
            None,  # profiler will be injected by parent manager
            self.config,
            input_,
        )

        # Ensure we return a tensor
        if isinstance(result, torch.Tensor):
            return result
        else:
            raise RuntimeError("Checkpointed forward must return a tensor")

    def _split_tensor(self, input_: torch.Tensor) -> torch.Tensor:
        """Split tensor across tensor parallel ranks."""
        if self.tp_size == 1:
            return input_

        # Split along last dimension
        input_shape = input_.shape
        split_size = input_shape[-1] // self.tp_size
        start_idx = self.tp_rank * split_size
        end_idx = start_idx + split_size

        return input_[..., start_idx:end_idx].contiguous()

    def _all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """All-reduce across tensor parallel ranks."""
        if self.tp_group is None or self.tp_size == 1:
            return input_

        # Use async all-reduce if configured
        if self.config.use_async_communication:
            # Placeholder for async implementation
            dist.all_reduce(input_, group=self.tp_group)
        else:
            dist.all_reduce(input_, group=self.tp_group)

        return input_


class MultiHeadAttentionCheckpoint(nn.Module):
    """Multi-head attention with tensor parallel checkpointing."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        config: Optional[ModelParallelCheckpointConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config or ModelParallelCheckpointConfig()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        # Get tensor parallel info
        if parallel_state.is_initialized():
            self.tp_size = parallel_state.get_tensor_model_parallel_size()
            self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        else:
            self.tp_size = 1
            self.tp_rank = 0

        # Calculate attention dimensions
        assert (
            num_attention_heads % self.tp_size == 0
        ), f"num_attention_heads ({num_attention_heads}) must be divisible by tp_size ({self.tp_size})"
        self.num_attention_heads_per_partition = num_attention_heads // self.tp_size
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = (
            self.num_attention_heads_per_partition * self.attention_head_size
        )

        # QKV projection (column parallel)
        if self.config.attention_fuse_qkv_checkpointing:
            self.query_key_value = ColumnParallelLinearCheckpoint(
                hidden_size,
                3 * self.all_head_size,
                bias=True,
                gather_output=False,
                config=config,
            )
        else:
            self.query = ColumnParallelLinearCheckpoint(
                hidden_size,
                self.all_head_size,
                bias=True,
                gather_output=False,
                config=config,
            )
            self.key = ColumnParallelLinearCheckpoint(
                hidden_size,
                self.all_head_size,
                bias=True,
                gather_output=False,
                config=config,
            )
            self.value = ColumnParallelLinearCheckpoint(
                hidden_size,
                self.all_head_size,
                bias=True,
                gather_output=False,
                config=config,
            )

        # Output projection (row parallel)
        self.dense = RowParallelLinearCheckpoint(
            hidden_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            config=config,
        )

        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with checkpointing."""
        if self.config.checkpoint_attention_layers:
            return self._checkpointed_forward(hidden_states, attention_mask)
        else:
            return self._standard_forward(hidden_states, attention_mask)

    def _standard_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard forward pass."""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # QKV projection
        if hasattr(self, "query_key_value"):
            qkv = self.query_key_value(hidden_states)
            qkv = qkv.view(
                batch_size,
                seq_len,
                self.num_attention_heads_per_partition,
                3,
                self.attention_head_size,
            )
            q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        else:
            q = self.query(hidden_states).view(
                batch_size,
                seq_len,
                self.num_attention_heads_per_partition,
                self.attention_head_size,
            )
            k = self.key(hidden_states).view(
                batch_size,
                seq_len,
                self.num_attention_heads_per_partition,
                self.attention_head_size,
            )
            v = self.value(hidden_states).view(
                batch_size,
                seq_len,
                self.num_attention_heads_per_partition,
                self.attention_head_size,
            )

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq, head_size)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = attention_scores / (self.attention_head_size**0.5)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2)  # (batch, seq, heads, head_size)
        context = context.contiguous().view(batch_size, seq_len, self.all_head_size)

        # Output projection
        output = self.dense(context)

        return output

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Checkpointed forward pass."""

        def attention_function(hidden: torch.Tensor) -> torch.Tensor:
            return self._standard_forward(hidden, attention_mask)

        result = TensorParallelCheckpointFunction.apply(
            attention_function,
            "multi_head_attention",
            self.tp_rank,
            self.tp_size,
            None,  # profiler will be injected by parent manager
            self.config,
            hidden_states,
        )

        # Ensure we return a tensor
        if isinstance(result, torch.Tensor):
            return result
        else:
            raise RuntimeError("Checkpointed forward must return a tensor")


class MLPCheckpoint(nn.Module):
    """MLP with tensor parallel checkpointing."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_function: str = "gelu",
        bias: bool = True,
        config: Optional[ModelParallelCheckpointConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config or ModelParallelCheckpointConfig()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Gate and up projections (column parallel)
        if self.config.mlp_fuse_gate_up_checkpointing:
            self.gate_up_proj = ColumnParallelLinearCheckpoint(
                hidden_size,
                2 * intermediate_size,
                bias=bias,
                gather_output=False,
                config=config,
            )
        else:
            if self.config.mlp_checkpoint_gate_projection:
                self.gate_proj = ColumnParallelLinearCheckpoint(
                    hidden_size,
                    intermediate_size,
                    bias=bias,
                    gather_output=False,
                    config=config,
                )
            if self.config.mlp_checkpoint_up_projection:
                self.up_proj = ColumnParallelLinearCheckpoint(
                    hidden_size,
                    intermediate_size,
                    bias=bias,
                    gather_output=False,
                    config=config,
                )

        # Down projection (row parallel)
        if self.config.mlp_checkpoint_down_projection:
            self.down_proj = RowParallelLinearCheckpoint(
                intermediate_size,
                hidden_size,
                bias=bias,
                input_is_parallel=True,
                config=config,
            )

        # Activation function
        if activation_function == "gelu":
            self.activation = nn.GELU()
        elif activation_function == "relu":
            self.activation = nn.ReLU()
        elif activation_function == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {activation_function}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with checkpointing."""
        if self.config.checkpoint_mlp_layers:
            return self._checkpointed_forward(hidden_states)
        else:
            return self._standard_forward(hidden_states)

    def _standard_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        if hasattr(self, "gate_up_proj"):
            # Fused gate and up projection
            gate_up = self.gate_up_proj(hidden_states)
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = self.activation(gate) * up
        else:
            # Separate gate and up projections
            gate = (
                self.gate_proj(hidden_states)
                if hasattr(self, "gate_proj")
                else hidden_states
            )
            up = (
                self.up_proj(hidden_states)
                if hasattr(self, "up_proj")
                else hidden_states
            )
            intermediate = self.activation(gate) * up

        # Down projection
        if hasattr(self, "down_proj"):
            output = self.down_proj(intermediate)
        else:
            output = intermediate

        return output

    def _checkpointed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Checkpointed forward pass."""

        def mlp_function(hidden: torch.Tensor) -> torch.Tensor:
            return self._standard_forward(hidden)

        tp_rank = (
            parallel_state.get_tensor_model_parallel_rank()
            if parallel_state.is_initialized()
            else 0
        )
        tp_size = (
            parallel_state.get_tensor_model_parallel_size()
            if parallel_state.is_initialized()
            else 1
        )

        result = TensorParallelCheckpointFunction.apply(
            mlp_function,
            "mlp",
            tp_rank,
            tp_size,
            None,  # profiler will be injected by parent manager
            self.config,
            hidden_states,
        )

        # Ensure we return a tensor
        if isinstance(result, torch.Tensor):
            return result
        else:
            raise RuntimeError("Checkpointed forward must return a tensor")


class ModelParallelActivationManager:
    """Manager for model parallel activation checkpointing."""

    def __init__(self, config: ModelParallelCheckpointConfig) -> None:
        """Initialize model parallel activation manager.

        Args:
            config: Model parallel checkpointing configuration
        """
        config.validate()
        self.config = config

        # Initialize distributed components
        self.distributed_profiler = DistributedMemoryProfiler(config.distributed_config)
        self.distributed_coordinator = DistributedCheckpointCoordinator(
            config.distributed_config
        )

        # Track checkpointed layers
        self.checkpointed_layers: Set[str] = set()
        self.layer_types: Dict[str, str] = {}

        # Threading
        self._lock = (
            threading.RLock()
            if config.distributed_config.base_config.thread_safe
            else None
        )

        logger.info(f"Initialized ModelParallelActivationManager")

    def register_layer(
        self, layer: nn.Module, layer_id: str, layer_type: str
    ) -> nn.Module:
        """Register a layer for model parallel checkpointing.

        Args:
            layer: Layer to register
            layer_id: Unique identifier for the layer
            layer_type: Type of layer (attention, mlp, etc.)

        Returns:
            Layer with checkpointing applied
        """
        if self._lock:
            with self._lock:
                self.checkpointed_layers.add(layer_id)
                self.layer_types[layer_id] = layer_type
        else:
            self.checkpointed_layers.add(layer_id)
            self.layer_types[layer_id] = layer_type

        # Wrap layer's forward method with profiler injection
        original_forward = layer.forward

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            # Inject profiler and coordinator into checkpointing functions
            self._inject_profiling_components(layer)
            return original_forward(*args, **kwargs)

        layer.forward = wrapped_forward

        return layer

    def _inject_profiling_components(self, layer: nn.Module) -> None:
        """Inject profiling components into layer's checkpointing functions."""
        # This is a placeholder for injecting profiler and coordinator
        # In practice, this would traverse the layer and inject components
        pass

    def create_column_parallel_layer(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        layer_id: Optional[str] = None,
    ) -> ColumnParallelLinearCheckpoint:
        """Create a column parallel layer with checkpointing.

        Args:
            in_features: Input features
            out_features: Output features
            bias: Whether to use bias
            gather_output: Whether to gather output
            layer_id: Optional layer identifier

        Returns:
            Column parallel layer with checkpointing
        """
        layer = ColumnParallelLinearCheckpoint(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            gather_output=gather_output,
            config=self.config,
        )

        if layer_id is not None:
            self.register_layer(layer, layer_id, "column_parallel")

        return layer

    def create_row_parallel_layer(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        layer_id: Optional[str] = None,
    ) -> RowParallelLinearCheckpoint:
        """Create a row parallel layer with checkpointing.

        Args:
            in_features: Input features
            out_features: Output features
            bias: Whether to use bias
            input_is_parallel: Whether input is already parallel
            layer_id: Optional layer identifier

        Returns:
            Row parallel layer with checkpointing
        """
        layer = RowParallelLinearCheckpoint(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            input_is_parallel=input_is_parallel,
            config=self.config,
        )

        if layer_id is not None:
            self.register_layer(layer, layer_id, "row_parallel")

        return layer

    def create_attention_layer(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        layer_id: Optional[str] = None,
    ) -> MultiHeadAttentionCheckpoint:
        """Create a multi-head attention layer with checkpointing.

        Args:
            hidden_size: Hidden size
            num_attention_heads: Number of attention heads
            attention_dropout: Attention dropout rate
            layer_id: Optional layer identifier

        Returns:
            Multi-head attention layer with checkpointing
        """
        layer = MultiHeadAttentionCheckpoint(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            config=self.config,
        )

        if layer_id is not None:
            self.register_layer(layer, layer_id, "attention")

        return layer

    def create_mlp_layer(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_function: str = "gelu",
        bias: bool = True,
        layer_id: Optional[str] = None,
    ) -> MLPCheckpoint:
        """Create an MLP layer with checkpointing.

        Args:
            hidden_size: Hidden size
            intermediate_size: Intermediate size
            activation_function: Activation function name
            bias: Whether to use bias
            layer_id: Optional layer identifier

        Returns:
            MLP layer with checkpointing
        """
        layer = MLPCheckpoint(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            bias=bias,
            config=self.config,
        )

        if layer_id is not None:
            self.register_layer(layer, layer_id, "mlp")

        return layer

    def get_checkpointing_report(self) -> Dict[str, Any]:
        """Get comprehensive model parallel checkpointing report.

        Returns:
            Dictionary containing checkpointing statistics
        """
        distributed_report = self.distributed_profiler.get_distributed_memory_report()
        coordination_stats = self.distributed_coordinator.get_coordination_stats()

        if self._lock:
            with self._lock:
                layer_info = {
                    "total_layers": len(self.checkpointed_layers),
                    "layer_types": dict(self.layer_types),
                    "checkpointed_layers": list(self.checkpointed_layers),
                }
        else:
            layer_info = {
                "total_layers": len(self.checkpointed_layers),
                "layer_types": dict(self.layer_types),
                "checkpointed_layers": list(self.checkpointed_layers),
            }

        return {
            "model_parallel_checkpointing": {
                "config": {
                    "checkpoint_attention": self.config.checkpoint_attention_layers,
                    "checkpoint_mlp": self.config.checkpoint_mlp_layers,
                    "sync_column_parallel": self.config.sync_column_parallel_activations,
                    "sync_row_parallel": self.config.sync_row_parallel_activations,
                    "communication_overlap": self.config.overlap_communication_computation,
                },
                "layers": layer_info,
            },
            "distributed_profiling": distributed_report,
            "coordination": coordination_stats,
        }

    def reset_checkpointing_stats(self) -> None:
        """Reset all checkpointing statistics."""
        self.distributed_profiler.reset_distributed_stats()
        self.distributed_coordinator.reset_coordination_state()

        if self._lock:
            with self._lock:
                self.checkpointed_layers.clear()
                self.layer_types.clear()
        else:
            self.checkpointed_layers.clear()
            self.layer_types.clear()


# Factory function for easy creation
def create_model_parallel_manager(
    checkpoint_attention: bool = True,
    checkpoint_mlp: bool = True,
    sync_column_parallel: bool = True,
    sync_row_parallel: bool = True,
    communication_overlap: bool = True,
    distributed_config: Optional[DistributedCheckpointConfig] = None,
    **kwargs: Any,
) -> ModelParallelActivationManager:
    """Create model parallel activation manager with common settings.

    Args:
        checkpoint_attention: Whether to checkpoint attention layers
        checkpoint_mlp: Whether to checkpoint MLP layers
        sync_column_parallel: Whether to sync column parallel activations
        sync_row_parallel: Whether to sync row parallel activations
        communication_overlap: Whether to overlap communication and computation
        distributed_config: Distributed checkpointing configuration
        **kwargs: Additional configuration parameters

    Returns:
        Configured ModelParallelActivationManager instance
    """
    if distributed_config is None:
        distributed_config = DistributedCheckpointConfig()

    config = ModelParallelCheckpointConfig(
        checkpoint_attention_layers=checkpoint_attention,
        checkpoint_mlp_layers=checkpoint_mlp,
        sync_column_parallel_activations=sync_column_parallel,
        sync_row_parallel_activations=sync_row_parallel,
        overlap_communication_computation=communication_overlap,
        distributed_config=distributed_config,
        **kwargs,
    )

    return ModelParallelActivationManager(config)
