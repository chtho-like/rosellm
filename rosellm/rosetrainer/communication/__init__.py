"""
Communication Module for RoseTrainer

Provides optimized communication patterns for distributed training:
- All-reduce operations
- Point-to-point communication
- Collective operations
- Communication overlap strategies
- Gradient bucketing for efficient communication
- Multi-bucket coordination and grouping
- Gradient bucket coalescing for optimized kernel launches
"""

from typing import Optional

import torch
import torch.distributed as dist

from .bucket_groups import (
    BucketGroup,
    BucketGroupConfig,
    BucketGroupManager,
    GroupStrategy,
    PriorityLevel,
)

# Import coalescing components
from .coalescing import (
    CoalescingConfig,
    CoalescingError,
    CoalescingManager,
    CoalescingMetrics,
)

# Import gradient bucketing components
from .gradient_buckets import (
    BucketCapacityError,
    BucketConfig,
    BucketFactory,
    BucketingError,
    BucketManager,
    BucketStateError,
    BucketStrategy,
    CommunicationBackend,
    CommunicationError,
    GradientBucket,
    GradientValidationError,
)


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    """Perform all-reduce operation on a tensor."""
    if dist.is_initialized():
        return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)
    return tensor


def broadcast(
    tensor: torch.Tensor,
    src: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    """Broadcast tensor from source rank."""
    if dist.is_initialized():
        return dist.broadcast(tensor, src=src, group=group, async_op=async_op)
    return tensor


__all__ = [
    # Basic communication operations
    "all_reduce",
    "broadcast",
    # Gradient bucketing
    "BucketConfig",
    "BucketFactory",
    "BucketManager",
    "BucketStrategy",
    "BucketingError",
    "BucketCapacityError",
    "BucketStateError",
    "CommunicationBackend",
    "CommunicationError",
    "GradientBucket",
    "GradientValidationError",
    # Bucket groups
    "BucketGroup",
    "BucketGroupConfig",
    "BucketGroupManager",
    "GroupStrategy",
    "PriorityLevel",
    # Coalescing
    "CoalescingConfig",
    "CoalescingError",
    "CoalescingManager",
    "CoalescingMetrics",
]
