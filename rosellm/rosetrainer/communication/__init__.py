"""
Communication Module for RoseTrainer

Provides optimized communication patterns for distributed training:
- All-reduce operations
- Point-to-point communication
- Collective operations
- Communication overlap strategies
"""

from typing import Optional

import torch
import torch.distributed as dist


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
    "all_reduce",
    "broadcast",
]
