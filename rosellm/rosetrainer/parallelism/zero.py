from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
from torch.optim import Optimizer

# References:
# [1] Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion
#     Parameter Models." arXiv:1910.02054 (2019)
# [2] Rajbhandari, S. et al. "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme
#     Scale Deep Learning." arXiv:2104.07857 (2021)
# [3] DeepSpeed ZeRO implementation:
#     https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/runtime/zero
# [4] FairScale ZeRO implementation:
#     https://github.com/facebookresearch/fairscale/blob/main/fairscale/optim/oss.py


class ZeROOptimizer:
    """
    Implementation of ZeRO (Zero Redundancy Optimizer) Stage 1.
    Partitions optimizer states across data parallel workers to reduce memory footprint.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        device: torch.device,
        local_rank: int = 0,
        world_size: int = 1,
        overlap_communication: bool = True,
        dp_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Initialize the ZeRO optimizer wrapper.

        Args:
            optimizer: Base optimizer (e.g., Adam, AdamW).
            device: Device to use for operations.
            local_rank: Local rank of the current process.
            world_size: Total number of processes.
            overlap_communication: Whether to overlap communication and computation.
            dp_group: Data parallel process group.
        """
        self.optimizer = optimizer
        self.device = device
        self.local_rank = local_rank
        self.world_size = world_size
        self.overlap_communication = overlap_communication
        self.dp_group = dp_group

        # Initialize ZeRO-specific variables
        self.partition_parameters()

        # Buffers for gradient partitioning
        self.buckets: Dict[int, List[torch.Tensor]] = {}
        self.grad_buffers: Dict[int, Optional[torch.Tensor]] = {}

        # Whether we should synchronize gradients at the next step
        self.require_grad_sync = True

    def partition_parameters(self) -> None:
        """Partition optimizer states across data parallel workers."""
        if self.world_size <= 1:
            return

        # Group parameters by data type
        self.parameter_groups: Dict[torch.dtype, List[torch.nn.Parameter]] = {}
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param.requires_grad:
                    dtype = param.dtype
                    if dtype not in self.parameter_groups:
                        self.parameter_groups[dtype] = []
                    self.parameter_groups[dtype].append(param)

        # Assign parameter partitions to different ranks
        self.param_to_rank: Dict[torch.nn.Parameter, int] = {}
        for dtype, params in self.parameter_groups.items():
            for i, param in enumerate(params):
                # Simple round-robin assignment
                rank = i % self.world_size
                self.param_to_rank[param] = rank

    def zero_grad(self) -> None:
        """Zero out gradients for parameters."""
        self.optimizer.zero_grad()

        # Reset buffers
        self.buckets = {}
        self.grad_buffers = {}

        # Set flag to sync gradients
        self.require_grad_sync = True

    def step(self) -> None:
        """Perform an optimization step."""
        # First synchronize gradients if required
        if self.require_grad_sync:
            self.synchronize_gradients()
            self.require_grad_sync = False

        # Update only parameters assigned to this rank
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if (
                    param.requires_grad
                    and self.param_to_rank.get(param, self.local_rank)
                    == self.local_rank
                ):
                    # Parameter is assigned to this rank, update it
                    if param.grad is not None:
                        param_group["optimizer"].step([param])
                else:
                    # Parameter is not assigned to this rank, skip it
                    param.grad = None

        # Synchronize updated parameters across all ranks
        self.synchronize_parameters()

    def synchronize_gradients(self) -> None:
        """Synchronize gradients across data parallel workers."""
        if self.world_size <= 1:
            return

        # Create buckets of parameters by dtype
        for dtype, params in self.parameter_groups.items():
            for param in params:
                if param.grad is not None:
                    # Get the assigned rank for this parameter
                    rank = self.param_to_rank.get(param, self.local_rank)

                    # Add to the correct bucket
                    if rank not in self.buckets:
                        self.buckets[rank] = []
                    self.buckets[rank].append(param.grad)

        # All-reduce gradients for each rank's bucket
        for rank, grads in self.buckets.items():
            # Flatten gradients for efficient communication
            flat_grads = []
            for grad in grads:
                flat_grads.append(grad.view(-1))

            # Concatenate tensors
            if flat_grads:
                buffer = torch.cat(flat_grads)
                self.grad_buffers[rank] = buffer

                # All-reduce if this is for the current rank
                if rank == self.local_rank:
                    dist.all_reduce(buffer, group=self.dp_group)

                    # Copy reduced gradients back to parameters
                    offset = 0
                    for i, grad in enumerate(grads):
                        numel = grad.numel()
                        grad.copy_(buffer[offset : offset + numel].view_as(grad))
                        offset += numel

    def synchronize_parameters(self) -> None:
        """Synchronize updated parameters across data parallel workers."""
        if self.world_size <= 1:
            return

        # Broadcast updated parameters from their respective ranks
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                rank = self.param_to_rank.get(param, self.local_rank)

                # Broadcast from the rank that owns this parameter
                dist.broadcast(param.data, rank, group=self.dp_group)

    def disable_grad_sync(self) -> None:
        """Disable gradient synchronization for the next step (used for gradient accumulation)."""
        self.require_grad_sync = False

    def enable_grad_sync(self) -> None:
        """Enable gradient synchronization for the next step."""
        self.require_grad_sync = True


class ZeROStateDict:
    """
    Helper for ZeRO optimizer state dict handling.
    Ensures correct saving and loading of partitioned optimizer states.
    """

    @staticmethod
    def get_optimizer_state(optimizer: ZeROOptimizer) -> Dict[str, Any]:
        """
        Get the state dictionary from a ZeRO optimizer.

        Args:
            optimizer: ZeRO optimizer instance.

        Returns:
            State dictionary with partitioned states.
        """
        state_dict = optimizer.optimizer.state_dict()

        # Add ZeRO-specific information
        state_dict["zero_param_to_rank"] = optimizer.param_to_rank
        state_dict["zero_local_rank"] = optimizer.local_rank
        state_dict["zero_world_size"] = optimizer.world_size

        return state_dict

    @staticmethod
    def load_optimizer_state(
        optimizer: ZeROOptimizer, state_dict: Dict[str, Any]
    ) -> None:
        """
        Load a state dictionary into a ZeRO optimizer.

        Args:
            optimizer: ZeRO optimizer instance.
            state_dict: State dictionary to load.
        """
        # Extract and remove ZeRO-specific info
        param_to_rank = state_dict.pop("zero_param_to_rank", {})
        local_rank = state_dict.pop("zero_local_rank", 0)
        world_size = state_dict.pop("zero_world_size", 1)

        # Check consistency
        if optimizer.world_size != world_size:
            print(
                f"Warning: Loading state with different world size: {world_size} vs {optimizer.world_size}"
            )

        # Only load optimizer states for parameters assigned to this rank
        filtered_state = optimizer.optimizer.state_dict()

        # Copy over param_groups
        filtered_state["param_groups"] = state_dict["param_groups"]

        # Filter states for the current rank
        filtered_state["state"] = {}
        for param_id, param_state in state_dict["state"].items():
            param = None

            # Find the parameter object from param_id
            for group in optimizer.optimizer.param_groups:
                for p in group["params"]:
                    if id(p) == param_id:
                        param = p
                        break

            if param is not None:
                # Check if this param belongs to the current rank
                if param_to_rank.get(id(param), local_rank) == optimizer.local_rank:
                    filtered_state["state"][param_id] = param_state

        # Load the filtered state
        optimizer.optimizer.load_state_dict(filtered_state)

        # Update partitioning info
        optimizer.partition_parameters()
