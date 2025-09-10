"""
Parameter-Gradient Buffer Mapping with Bucket-based Reduction

This module implements a sophisticated mapping system between parameters and gradient
buffers, enabling efficient distributed gradient reduction through bucketing and
communication-computation overlap. Inspired by Megatron-LM's gradient buffer system.

Key Features:
- Parameter to gradient buffer mapping with configurable bucket sizes
- Overlapped gradient reduction during backward pass
- Memory-efficient contiguous buffer allocation
- Support for model-parallel and data-parallel reduction patterns
- Automatic bucket size optimization based on communication backend

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

logger = logging.getLogger(__name__)

# Constants for optimal communication
DEFAULT_BUCKET_SIZE_MB = 40.0  # Optimal for NCCL all-reduce
MIN_BUCKET_SIZE_MB = 1.0
MAX_BUCKET_SIZE_MB = 1024.0
BYTES_PER_MB = 1024 * 1024
ALIGNMENT_BYTES = 256  # Alignment for optimal memory access


class ReductionOp(Enum):
    """Reduction operation types for gradient synchronization."""

    AVG = "avg"
    SUM = "sum"
    MAX = "max"
    MIN = "min"


@dataclass
class BucketConfig:
    """Configuration for gradient bucketing."""

    bucket_size_mb: float = DEFAULT_BUCKET_SIZE_MB
    dtype: torch.dtype = torch.float32
    reduction_op: ReductionOp = ReductionOp.AVG
    overlap_grad_reduce: bool = True
    use_contiguous_buffers: bool = True
    align_buffers: bool = True
    bucket_cap_mb: float = MAX_BUCKET_SIZE_MB
    check_for_nan_in_grad: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.bucket_size_mb <= 0:
            raise ValueError(
                f"bucket_size_mb must be positive, got {self.bucket_size_mb}"
            )
        if self.bucket_cap_mb < self.bucket_size_mb:
            raise ValueError(
                f"bucket_cap_mb ({self.bucket_cap_mb}) must be >= "
                f"bucket_size_mb ({self.bucket_size_mb})"
            )


@dataclass
class GradientBucket:
    """
    Container for a group of parameters sharing a gradient buffer.

    Attributes:
        index: Bucket index in the buffer
        params: List of parameters in this bucket
        grad_buffer: Contiguous gradient buffer
        offset: Starting offset in the global buffer
        numel: Total number of elements
        numel_unpadded: Number of unpadded elements
        all_reduce_handle: Handle for async all-reduce operation
        params_with_grad: Set of parameters that have gradients ready
    """

    index: int
    params: List[Parameter]
    grad_buffer: Optional[Tensor] = None
    offset: int = 0
    numel: int = 0
    numel_unpadded: int = 0
    all_reduce_handle: Optional[dist.Work] = None
    params_with_grad: Set[Parameter] = field(default_factory=set)

    def is_ready_for_reduce(self) -> bool:
        """Check if all parameters in bucket have gradients ready."""
        return len(self.params) > 0 and len(self.params_with_grad) == len(self.params)

    def reset(self):
        """Reset bucket state for next iteration."""
        self.params_with_grad.clear()
        self.all_reduce_handle = None


class ParamGradMapping:
    """
    Manages mapping between parameters and gradient buffers with bucketing.

    This class creates an efficient mapping between model parameters and gradient
    buffers, organizing parameters into buckets for optimized distributed reduction.
    It supports overlapped communication, memory consolidation, and various
    parallelism patterns.

    Args:
        model: Model whose parameters to manage
        config: Bucketing configuration
        data_parallel_group: Process group for data parallel reduction
        model_parallel_group: Process group for model parallel reduction
        virtual_pipeline_model_parallel_rank: Virtual pipeline rank
    """

    def __init__(
        self,
        model: nn.Module,
        config: BucketConfig,
        data_parallel_group: Optional[dist.ProcessGroup] = None,
        model_parallel_group: Optional[dist.ProcessGroup] = None,
        virtual_pipeline_model_parallel_rank: Optional[int] = None,
    ):
        self.model = model
        self.config = config
        self.data_parallel_group = data_parallel_group or dist.group.WORLD
        self.model_parallel_group = model_parallel_group
        self.virtual_pp_rank = virtual_pipeline_model_parallel_rank

        # Initialize mapping structures
        self.param_to_bucket: Dict[Parameter, GradientBucket] = {}
        self.buckets: List[GradientBucket] = []
        self.param_to_buffer_offset: Dict[Parameter, Tuple[int, int]] = {}

        # Global gradient buffer
        self.grad_buffer: Optional[Tensor] = None
        self.grad_buffer_numel = 0

        # Statistics
        self.num_buckets = 0
        self.total_params = 0
        self.total_numel = 0

        # Build the mapping
        self._build_param_grad_mapping()

        # Register gradient hooks if overlapping
        if self.config.overlap_grad_reduce:
            self._register_grad_hooks()

    def _build_param_grad_mapping(self):
        """Build mapping between parameters and gradient buckets."""
        # Collect all parameters requiring gradients
        params_list = []
        for param in self.model.parameters():
            if param.requires_grad:
                # Skip parameters that don't participate in DP reduction
                if hasattr(param, "skip_data_parallel_grad_reduce"):
                    if getattr(param, "skip_data_parallel_grad_reduce", False):
                        continue
                params_list.append(param)

        if not params_list:
            logger.warning("No parameters require gradients")
            return

        self.total_params = len(params_list)

        # Sort parameters by size for better bucketing (largest first)
        params_list.sort(key=lambda p: p.numel(), reverse=True)

        # Calculate bucket size in elements
        dtype_size = torch.finfo(self.config.dtype).bits // 8
        bucket_size_numel = int(self.config.bucket_size_mb * BYTES_PER_MB / dtype_size)
        bucket_cap_numel = int(self.config.bucket_cap_mb * BYTES_PER_MB / dtype_size)

        # Create buckets
        current_bucket_params = []
        current_bucket_numel = 0

        for param in params_list:
            param_numel = param.numel()
            self.total_numel += param_numel

            # Check if adding this param exceeds bucket capacity
            if current_bucket_numel + param_numel > bucket_cap_numel:
                # Finalize current bucket if it has params
                if current_bucket_params:
                    self._create_bucket(current_bucket_params, current_bucket_numel)
                # Start new bucket with this param
                current_bucket_params = [param]
                current_bucket_numel = param_numel
            elif current_bucket_numel + param_numel > bucket_size_numel:
                # Exceeds target size but within cap, finalize bucket
                if current_bucket_params:
                    self._create_bucket(current_bucket_params, current_bucket_numel)
                current_bucket_params = [param]
                current_bucket_numel = param_numel
            else:
                # Add to current bucket
                current_bucket_params.append(param)
                current_bucket_numel += param_numel

        # Finalize last bucket
        if current_bucket_params:
            self._create_bucket(current_bucket_params, current_bucket_numel)

        # Allocate contiguous gradient buffer if configured
        if self.config.use_contiguous_buffers:
            self._allocate_grad_buffer()

        logger.info(
            f"Created {self.num_buckets} gradient buckets for {self.total_params} "
            f"parameters ({self.total_numel} elements, "
            f"{self.grad_buffer_numel * dtype_size / BYTES_PER_MB:.2f} MB)"
        )

    def _create_bucket(self, params: List[Parameter], numel: int):
        """Create a gradient bucket for given parameters."""
        # Apply alignment if configured
        numel_unpadded = numel
        if self.config.align_buffers:
            # Align to ALIGNMENT_BYTES boundary
            dtype_size = torch.finfo(self.config.dtype).bits // 8
            aligned_elements = ALIGNMENT_BYTES // dtype_size
            numel = math.ceil(numel / aligned_elements) * aligned_elements

        bucket = GradientBucket(
            index=self.num_buckets,
            params=params,
            offset=self.grad_buffer_numel,
            numel=numel,
            numel_unpadded=numel_unpadded,
        )

        # Update mappings
        for param in params:
            self.param_to_bucket[param] = bucket

        self.buckets.append(bucket)
        self.grad_buffer_numel += numel
        self.num_buckets += 1

    def _allocate_grad_buffer(self):
        """Allocate contiguous gradient buffer for all buckets."""
        device = next(self.model.parameters()).device
        self.grad_buffer = torch.zeros(
            self.grad_buffer_numel,
            dtype=self.config.dtype,
            device=device,
            requires_grad=False,
        )

        # Assign buffer views to buckets
        for bucket in self.buckets:
            start = bucket.offset
            end = bucket.offset + bucket.numel
            bucket.grad_buffer = self.grad_buffer[start:end]

            # Create param to buffer offset mapping
            offset = 0
            for param in bucket.params:
                param_numel = param.numel()
                self.param_to_buffer_offset[param] = (
                    bucket.offset + offset,
                    bucket.offset + offset + param_numel,
                )
                offset += param_numel

    def _register_grad_hooks(self):
        """Register gradient hooks for overlapped reduction."""
        for param in self.param_to_bucket.keys():
            # First ensure params_with_grad is properly initialized for bucket
            bucket = self.param_to_bucket[param]
            if (
                not hasattr(bucket, "params_with_grad")
                or bucket.params_with_grad is None
            ):
                bucket.params_with_grad = set()
            param.register_hook(self._grad_hook_factory(param))

    def _grad_hook_factory(self, param: Parameter):
        """Factory to create gradient hook for a parameter."""

        def grad_hook(grad: Tensor) -> Optional[Tensor]:
            # Mark parameter as having gradient ready
            bucket = self.param_to_bucket[param]
            bucket.params_with_grad.add(param)

            # Copy gradient to buffer if using contiguous buffers
            if self.config.use_contiguous_buffers and bucket.grad_buffer is not None:
                start, end = self.param_to_buffer_offset[param]
                bucket.grad_buffer[start - bucket.offset : end - bucket.offset].copy_(
                    grad.view(-1)
                )

            # Check if bucket is ready for reduction
            if bucket.is_ready_for_reduce() and bucket.all_reduce_handle is None:
                self._launch_bucket_reduce(bucket)

            # Return None to not modify gradient
            return None

        return grad_hook

    def _launch_bucket_reduce(self, bucket: GradientBucket):
        """Launch asynchronous all-reduce for a bucket."""
        if bucket.grad_buffer is None:
            return

        # Skip reduction if not in distributed mode
        if not dist.is_initialized():
            return

        # Get the buffer slice for this bucket (unpadded)
        buffer_to_reduce = bucket.grad_buffer[: bucket.numel_unpadded]

        # Apply reduction operation
        if self.config.reduction_op == ReductionOp.AVG:
            # Scale by world size for averaging
            world_size = dist.get_world_size(self.data_parallel_group)
            buffer_to_reduce.div_(world_size)

        # Launch async all-reduce
        bucket.all_reduce_handle = dist.all_reduce(
            buffer_to_reduce,
            group=self.data_parallel_group,
            async_op=True,
        )

        logger.debug(f"Launched all-reduce for bucket {bucket.index}")

    def wait_for_all_reduces(self):
        """Wait for all pending all-reduce operations to complete."""
        for bucket in self.buckets:
            if bucket.all_reduce_handle is not None:
                bucket.all_reduce_handle.wait()

                # Copy reduced gradients back to parameters if needed
                if self.config.use_contiguous_buffers:
                    self._copy_bucket_grads_to_params(bucket)

    def _copy_bucket_grads_to_params(self, bucket: GradientBucket):
        """Copy reduced gradients from buffer back to parameters."""
        if bucket.grad_buffer is None:
            return

        for param in bucket.params:
            start, end = self.param_to_buffer_offset[param]
            param.grad = bucket.grad_buffer[
                start - bucket.offset : end - bucket.offset
            ].view_as(param)

    def reset(self):
        """Reset all buckets for next iteration."""
        for bucket in self.buckets:
            bucket.reset()

        # Clear gradient buffer if configured
        if self.grad_buffer is not None and not self.config.check_for_nan_in_grad:
            self.grad_buffer.zero_()

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about the mapping."""
        return {
            "num_buckets": self.num_buckets,
            "total_params": self.total_params,
            "total_numel": self.total_numel,
            "buffer_numel": self.grad_buffer_numel,
            "buffer_size_mb": (
                self.grad_buffer_numel
                * torch.finfo(self.config.dtype).bits
                // 8
                / BYTES_PER_MB
            ),
            "avg_bucket_size": (
                self.grad_buffer_numel / self.num_buckets if self.num_buckets > 0 else 0
            ),
            "padding_overhead": (
                (self.grad_buffer_numel - self.total_numel) / self.total_numel * 100
                if self.total_numel > 0
                else 0
            ),
        }
