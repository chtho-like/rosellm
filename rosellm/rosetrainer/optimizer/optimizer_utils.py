"""
Utility functions for distributed optimizer operations.

This module provides helper functions for gradient bucketing, parameter
partitioning, and communication optimization in distributed training.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter

logger = logging.getLogger(__name__)

# Constants
BYTES_PER_MB = 1024 * 1024
DEFAULT_BUCKET_SIZE_MB = 25.0


def create_parameter_buckets(
    params: List[Parameter],
    bucket_size_mb: float = DEFAULT_BUCKET_SIZE_MB,
    dtype: Optional[torch.dtype] = None,
) -> List[List[Parameter]]:
    """
    Create parameter buckets based on size constraints.

    Groups parameters into buckets where each bucket's total size
    is approximately bucket_size_mb. This optimizes communication
    by reducing the number of all-reduce operations.

    Args:
        params: List of model parameters to bucket
        bucket_size_mb: Target bucket size in megabytes
        dtype: Data type for size calculation (default: params[0].dtype)

    Returns:
        List of parameter buckets
    """
    if not params:
        return []

    # Determine dtype and element size
    if dtype is None:
        dtype = params[0].dtype if params else torch.float32
    element_size = torch.tensor([], dtype=dtype).element_size()

    # Validate inputs
    if bucket_size_mb <= 0:
        raise ValueError(f"bucket_size_mb must be positive, got {bucket_size_mb}")

    # Calculate target bucket size in elements
    bucket_size_bytes = int(bucket_size_mb * BYTES_PER_MB)
    bucket_size_elements = bucket_size_bytes // element_size

    # Create buckets
    buckets: List[List[Parameter]] = []
    current_bucket: List[Parameter] = []
    current_size = 0

    for param in params:
        if not param.requires_grad:
            continue

        param_size = param.numel()

        # Check if adding this param exceeds bucket size
        if current_size > 0 and current_size + param_size > bucket_size_elements:
            # Save current bucket and start new one
            if current_bucket:
                buckets.append(current_bucket)
            current_bucket = [param]
            current_size = param_size
        else:
            # Add to current bucket
            current_bucket.append(param)
            current_size += param_size

    # Add final bucket
    if current_bucket:
        buckets.append(current_bucket)

    logger.debug(
        f"Created {len(buckets)} parameter buckets with target size {bucket_size_mb}MB"
    )
    return buckets


def partition_parameters_round_robin(
    params: List[Parameter],
    world_size: int,
) -> Dict[int, List[Parameter]]:
    """
    Partition parameters across ranks using round-robin assignment.

    This ensures balanced distribution of parameters across all ranks
    for optimizer state sharding.

    Args:
        params: List of model parameters
        world_size: Number of ranks to partition across

    Returns:
        Dictionary mapping rank to assigned parameters
    """
    rank_to_params: Dict[int, List[Parameter]] = {i: [] for i in range(world_size)}

    for idx, param in enumerate(params):
        assigned_rank = idx % world_size
        rank_to_params[assigned_rank].append(param)

    # Log distribution
    for rank, rank_params in rank_to_params.items():
        total_elements = sum(p.numel() for p in rank_params)
        logger.debug(
            f"Rank {rank}: {len(rank_params)} params, {total_elements:,} elements"
        )

    return rank_to_params


def partition_parameters_by_size(
    params: List[Parameter],
    world_size: int,
) -> Dict[int, List[Parameter]]:
    """
    Partition parameters across ranks to balance memory usage.

    Uses a greedy algorithm to assign parameters to ranks such that
    each rank has approximately equal total parameter size.

    Args:
        params: List of model parameters
        world_size: Number of ranks to partition across

    Returns:
        Dictionary mapping rank to assigned parameters
    """
    # Sort parameters by size (largest first)
    sorted_params = sorted(params, key=lambda p: p.numel(), reverse=True)

    # Track total size per rank
    rank_sizes = [0] * world_size
    rank_to_params: Dict[int, List[Parameter]] = {i: [] for i in range(world_size)}

    # Greedily assign each parameter to rank with smallest current size
    for param in sorted_params:
        # Find rank with minimum size
        min_rank = min(range(world_size), key=lambda r: rank_sizes[r])

        # Assign parameter to this rank
        rank_to_params[min_rank].append(param)
        rank_sizes[min_rank] += param.numel()

    # Log distribution
    for rank in range(world_size):
        logger.debug(
            f"Rank {rank}: {len(rank_to_params[rank])} params, "
            f"{rank_sizes[rank]:,} elements"
        )

    return rank_to_params


def compute_bucket_assignment(
    params: List[Parameter],
    buckets: List[List[Parameter]],
) -> Tuple[Dict[Parameter, int], Dict[Parameter, Tuple[int, int]]]:
    """
    Compute parameter to bucket mapping and offsets.

    Args:
        params: List of all parameters
        buckets: List of parameter buckets

    Returns:
        Tuple of (param_to_bucket, param_to_offset) mappings
    """
    param_to_bucket: Dict[Parameter, int] = {}
    param_to_offset: Dict[Parameter, Tuple[int, int]] = {}

    for bucket_idx, bucket_params in enumerate(buckets):
        offset = 0
        for param in bucket_params:
            param_to_bucket[param] = bucket_idx
            param_size = param.numel()
            param_to_offset[param] = (offset, offset + param_size)
            offset += param_size

    return param_to_bucket, param_to_offset


def flatten_dense_tensors(tensors: List[Tensor]) -> Tensor:
    """
    Flatten and concatenate a list of dense tensors into a single buffer.

    Args:
        tensors: List of tensors to flatten

    Returns:
        Flattened tensor buffer
    """
    if not tensors:
        return torch.tensor([])

    # Calculate total size
    total_size = sum(t.numel() for t in tensors)

    # Create output buffer
    output = torch.empty(
        total_size,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )

    # Copy tensors into buffer
    offset = 0
    for tensor in tensors:
        tensor_size = tensor.numel()
        output[offset : offset + tensor_size] = tensor.flatten()
        offset += tensor_size

    return output


def unflatten_dense_tensors(
    flat_tensor: Tensor,
    tensors: List[Tensor],
) -> List[Tensor]:
    """
    Unflatten a buffer into a list of tensors with original shapes.

    Args:
        flat_tensor: Flattened tensor buffer
        tensors: List of tensors with target shapes

    Returns:
        List of unflattened tensors
    """
    outputs = []
    offset = 0

    for tensor in tensors:
        tensor_size = tensor.numel()
        unflat = flat_tensor[offset : offset + tensor_size].view_as(tensor)
        outputs.append(unflat)
        offset += tensor_size

    return outputs


def async_all_reduce_buckets(
    buckets: List[List[Parameter]],
    process_group: Optional[dist.ProcessGroup] = None,
) -> List[Optional[dist.Work]]:
    """
    Start asynchronous all-reduce operations for parameter buckets.

    Args:
        buckets: List of parameter buckets
        process_group: Process group for communication

    Returns:
        List of async work handles
    """
    if not dist.is_initialized():
        return []

    handles: List[Optional[dist.Work]] = []

    for bucket_params in buckets:
        # Skip if no gradients
        if not any(p.grad is not None for p in bucket_params):
            handles.append(None)
            continue

        # Flatten gradients
        grads = [p.grad for p in bucket_params if p.grad is not None]
        if not grads:
            handles.append(None)
            continue

        flat_grad = flatten_dense_tensors(grads)

        # Start async all-reduce
        handle = dist.all_reduce(flat_grad, group=process_group, async_op=True)
        handles.append(handle)

    return handles


def synchronize_bucket_gradients(
    buckets: List[List[Parameter]],
    handles: List[Optional[dist.Work]],
    world_size: int = 1,
) -> None:
    """
    Wait for bucket all-reduce operations and copy gradients back.

    Args:
        buckets: List of parameter buckets
        handles: List of async work handles
        world_size: Number of ranks for gradient averaging
    """
    for bucket_params, handle in zip(buckets, handles):
        if handle is None:
            continue

        # Wait for all-reduce to complete
        handle.wait()

        # Average gradients and copy back
        if world_size > 1:
            for param in bucket_params:
                if param.grad is not None:
                    param.grad.div_(world_size)


def estimate_memory_savings(
    num_params: int,
    param_dtype: torch.dtype,
    optimizer_state_size: int,
    world_size: int,
) -> Dict[str, float]:
    """
    Estimate memory savings from optimizer state partitioning.

    Args:
        num_params: Total number of parameters
        param_dtype: Parameter data type
        optimizer_state_size: Number of state tensors per parameter
        world_size: Number of ranks for partitioning

    Returns:
        Dictionary with memory estimates in MB
    """
    # Calculate sizes
    param_size = torch.tensor([], dtype=param_dtype).element_size()
    state_size = torch.tensor([], dtype=torch.float32).element_size()

    # Total memory without partitioning
    param_memory = num_params * param_size
    state_memory = num_params * optimizer_state_size * state_size
    total_memory = param_memory + state_memory

    # Memory with partitioning (each rank stores 1/world_size of states)
    partitioned_state_memory = state_memory / world_size
    partitioned_total = param_memory + partitioned_state_memory

    # Calculate savings
    savings = total_memory - partitioned_total
    savings_percent = (savings / total_memory) * 100

    return {
        "total_memory_mb": total_memory / (1024 * 1024),
        "partitioned_memory_mb": partitioned_total / (1024 * 1024),
        "savings_mb": savings / (1024 * 1024),
        "savings_percent": savings_percent,
    }


def get_optimizer_memory_usage(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """
    Calculate memory usage of optimizer states.

    Args:
        optimizer: Optimizer instance

    Returns:
        Dictionary with memory usage statistics in MB
    """
    total_param_memory = 0
    total_state_memory = 0
    num_params_with_state = 0

    for group in optimizer.param_groups:
        for param in group["params"]:
            # Parameter memory
            param_memory = param.numel() * param.element_size()
            total_param_memory += param_memory

            # State memory
            state = optimizer.state.get(param, {})
            if state:
                num_params_with_state += 1
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state_memory = value.numel() * value.element_size()
                        total_state_memory += state_memory

    return {
        "param_memory_mb": total_param_memory / BYTES_PER_MB,
        "state_memory_mb": total_state_memory / BYTES_PER_MB,
        "total_memory_mb": (total_param_memory + total_state_memory) / BYTES_PER_MB,
        "num_params_with_state": num_params_with_state,
    }


def validate_bucket_configuration(
    params: List[Parameter],
    bucket_size_mb: float,
) -> Dict[str, Any]:
    """
    Validate and analyze bucket configuration for parameters.

    Args:
        params: List of model parameters
        bucket_size_mb: Target bucket size in megabytes

    Returns:
        Dictionary with validation results and statistics
    """
    if not params:
        return {
            "valid": False,
            "error": "No parameters provided",
        }

    # Create buckets
    buckets = create_parameter_buckets(params, bucket_size_mb)

    # Analyze bucket statistics
    bucket_sizes = []
    total_elements = 0
    dtype = params[0].dtype if params else torch.float32
    element_size = torch.tensor([], dtype=dtype).element_size()

    for bucket in buckets:
        bucket_elements = sum(p.numel() for p in bucket)
        bucket_size = bucket_elements * element_size / BYTES_PER_MB  # MB
        bucket_sizes.append(bucket_size)
        total_elements += bucket_elements

    # Calculate statistics
    avg_bucket_size = sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0
    max_bucket_size = max(bucket_sizes) if bucket_sizes else 0
    min_bucket_size = min(bucket_sizes) if bucket_sizes else 0

    # Efficiency metrics
    efficiency = avg_bucket_size / bucket_size_mb if bucket_size_mb > 0 else 0
    size_variance = max_bucket_size - min_bucket_size

    return {
        "valid": True,
        "num_buckets": len(buckets),
        "num_params": len(params),
        "total_elements": total_elements,
        "total_size_mb": total_elements * element_size / BYTES_PER_MB,
        "avg_bucket_size_mb": avg_bucket_size,
        "max_bucket_size_mb": max_bucket_size,
        "min_bucket_size_mb": min_bucket_size,
        "efficiency": efficiency,
        "size_variance_mb": size_variance,
        "target_bucket_size_mb": bucket_size_mb,
    }
