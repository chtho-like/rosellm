"""Parameter range mapping utilities for distributed optimizer.

This module provides efficient parameter partitioning across distributed ranks
for memory-efficient optimizer state management.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Constants
MIN_ELEMENTS_PER_RANK = 1
ALIGNMENT_SIZE = 8  # Align partitions to 8 elements for better memory access


@dataclass
class ParameterRange:
    """Represents a range of parameters assigned to a rank.

    Attributes:
        rank: The rank that owns this parameter range.
        start_idx: Starting parameter index (inclusive).
        end_idx: Ending parameter index (exclusive).
        param_start_offset: Starting offset within first parameter.
        param_end_offset: Ending offset within last parameter.
        total_elements: Total number of elements in this range.
        param_indices: List of parameter indices in this range.
    """

    rank: int
    start_idx: int
    end_idx: int
    param_start_offset: int
    param_end_offset: int
    total_elements: int
    param_indices: List[int]

    def contains_param(self, param_idx: int) -> bool:
        """Check if a parameter index is in this range."""
        return param_idx in self.param_indices

    def get_param_slice(
        self, param_idx: int, param_numel: int
    ) -> Optional[Tuple[int, int]]:
        """Get the slice of a parameter that belongs to this range.

        Args:
            param_idx: Index of the parameter.
            param_numel: Number of elements in the parameter.

        Returns:
            Tuple of (start, end) indices within the parameter, or None if not in range.
        """
        if param_idx not in self.param_indices:
            return None

        if param_idx == self.start_idx and param_idx == self.end_idx - 1:
            # Parameter spans entire range
            return (self.param_start_offset, self.param_end_offset)
        elif param_idx == self.start_idx:
            # First parameter in range
            return (self.param_start_offset, param_numel)
        elif param_idx == self.end_idx - 1:
            # Last parameter in range
            return (0, self.param_end_offset)
        else:
            # Middle parameter - take entire parameter
            return (0, param_numel)


class ParameterPartitioner:
    """Handles partitioning of parameters across ranks."""

    def __init__(self, world_size: int, rank: int):
        """Initialize parameter partitioner.

        Args:
            world_size: Total number of ranks.
            rank: Current rank.
        """
        self.world_size = world_size
        self.rank = rank
        self.param_ranges: List[ParameterRange] = []
        self.rank_to_range: Dict[int, ParameterRange] = {}

    def compute_partition_ranges(
        self, parameters: List[nn.Parameter], contiguous: bool = True
    ) -> List[ParameterRange]:
        """Compute parameter ranges for each rank.

        Args:
            parameters: List of model parameters to partition.
            contiguous: Whether to create contiguous partitions.

        Returns:
            List of ParameterRange objects, one per rank.

        Raises:
            ValueError: If parameters list is empty or contains invalid parameters.
        """
        if not parameters:
            # Empty parameters - create empty ranges
            self.param_ranges = []
            self.rank_to_range = {}
            return []

        # Validate parameters
        for i, param in enumerate(parameters):
            if not isinstance(param, nn.Parameter):
                raise ValueError(f"Element at index {i} is not a Parameter")
            if param.numel() == 0:
                raise ValueError(f"Parameter at index {i} has 0 elements")

        # Calculate total number of elements
        total_numel = sum(p.numel() for p in parameters)

        if total_numel < self.world_size * MIN_ELEMENTS_PER_RANK:
            # Too few elements to partition effectively
            # Assign all to rank 0
            if self.rank == 0:
                range_obj = ParameterRange(
                    rank=0,
                    start_idx=0,
                    end_idx=len(parameters),
                    param_start_offset=0,
                    param_end_offset=parameters[-1].numel() if parameters else 0,
                    total_elements=total_numel,
                    param_indices=list(range(len(parameters))),
                )
                self.param_ranges = [range_obj]
                self.rank_to_range = {0: range_obj}
                return self.param_ranges
            else:
                self.param_ranges = []
                self.rank_to_range = {}
                return []

        # Calculate elements per rank with alignment
        # base_numel_per_rank = total_numel // self.world_size  # Not used
        # remainder = total_numel % self.world_size  # Not used

        # Ensure minimum elements per rank and alignment
        numel_per_rank = max(
            math.ceil(total_numel / self.world_size), MIN_ELEMENTS_PER_RANK
        )

        # Align to ALIGNMENT_SIZE for better memory access
        if numel_per_rank % ALIGNMENT_SIZE != 0:
            numel_per_rank = ((numel_per_rank // ALIGNMENT_SIZE) + 1) * ALIGNMENT_SIZE

        ranges = []
        current_rank = 0
        range_start_idx = 0
        range_start_offset = 0
        range_elements = 0
        range_param_indices = []

        for param_idx, param in enumerate(parameters):
            param_numel = param.numel()
            param_offset = 0

            while param_offset < param_numel and current_rank < self.world_size:
                # Calculate how many elements to take from this parameter
                remaining_in_param = param_numel - param_offset
                remaining_for_rank = numel_per_rank - range_elements
                elements_to_take = min(remaining_in_param, remaining_for_rank)

                # Add parameter index if taking any elements from it
                if param_idx not in range_param_indices:
                    range_param_indices.append(param_idx)

                range_elements += elements_to_take
                param_offset += elements_to_take

                # Check if current rank's range is complete
                # Account for last rank getting remaining elements
                is_last_rank = current_rank == self.world_size - 1
                is_last_param = (
                    param_idx == len(parameters) - 1 and param_offset >= param_numel
                )

                if (not is_last_rank and range_elements >= numel_per_rank) or (
                    is_last_rank and is_last_param
                ):
                    # Create range for current rank
                    param_range = ParameterRange(
                        rank=current_rank,
                        start_idx=range_start_idx,
                        end_idx=(param_idx + 1 if param_offset > 0 else param_idx),
                        param_start_offset=range_start_offset,
                        param_end_offset=(
                            param_offset if param_offset > 0 else param_numel
                        ),
                        total_elements=range_elements,
                        param_indices=range_param_indices.copy(),
                    )
                    ranges.append(param_range)

                    # Move to next rank
                    current_rank += 1
                    if current_rank < self.world_size:
                        range_start_idx = (
                            param_idx if param_offset < param_numel else param_idx + 1
                        )
                        range_start_offset = (
                            param_offset if param_offset < param_numel else 0
                        )
                        range_elements = 0
                        range_param_indices = []

        # Store computed ranges
        self.param_ranges = ranges
        self.rank_to_range = {r.rank: r for r in ranges}

        return ranges

    def get_local_param_range(self) -> Optional[ParameterRange]:
        """Get the parameter range for the current rank."""
        return self.rank_to_range.get(self.rank)

    def get_param_owner(self, param_idx: int) -> Optional[int]:
        """Get the rank that owns a parameter.

        Args:
            param_idx: Index of the parameter.

        Returns:
            Rank that owns the parameter, or None if not found.
        """
        for param_range in self.param_ranges:
            if param_range.contains_param(param_idx):
                return param_range.rank
        return None

    def create_partition_buffer(
        self, parameters: List[nn.Parameter], dtype: Optional[torch.dtype] = None
    ) -> Tuple[Tensor, Dict[int, Tuple[int, int]]]:
        """Create a contiguous buffer for local parameter partition.

        Args:
            parameters: List of model parameters.
            dtype: Data type for the buffer (defaults to parameter dtype).

        Returns:
            Tuple of (buffer tensor, parameter offset mapping).
        """
        local_range = self.get_local_param_range()
        if local_range is None:
            return torch.empty(0), {}

        # Determine dtype
        if dtype is None:
            dtype = parameters[0].dtype if parameters else torch.float32

        # Create buffer
        buffer = torch.zeros(
            local_range.total_elements,
            dtype=dtype,
            device=parameters[0].device if parameters else "cpu",
        )

        # Create offset mapping
        param_offsets = {}
        current_offset = 0

        for param_idx in local_range.param_indices:
            if param_idx >= len(parameters):
                continue

            param = parameters[param_idx]
            param_slice = local_range.get_param_slice(param_idx, param.numel())
            if param_slice is None:
                continue

            start, end = param_slice
            slice_size = end - start

            # Copy parameter data to buffer
            param_flat = param.view(-1)
            buffer[current_offset : current_offset + slice_size] = param_flat[start:end]

            # Store offset mapping
            param_offsets[param_idx] = (current_offset, current_offset + slice_size)
            current_offset += slice_size

        return buffer, param_offsets

    def scatter_parameters(
        self,
        buffer: Tensor,
        parameters: List[nn.Parameter],
        param_offsets: Dict[int, Tuple[int, int]],
    ) -> None:
        """Scatter buffer data back to parameters.

        Args:
            buffer: Contiguous buffer containing parameter data.
            parameters: List of model parameters.
            param_offsets: Mapping of parameter indices to buffer offsets.
        """
        local_range = self.get_local_param_range()
        if local_range is None:
            return

        for param_idx, (buf_start, buf_end) in param_offsets.items():
            if param_idx >= len(parameters):
                continue

            param = parameters[param_idx]
            param_slice = local_range.get_param_slice(param_idx, param.numel())
            if param_slice is None:
                continue

            start, end = param_slice
            param_flat = param.view(-1)

            # Copy from buffer to parameter
            with torch.no_grad():
                param_flat[start:end] = buffer[buf_start:buf_end]
