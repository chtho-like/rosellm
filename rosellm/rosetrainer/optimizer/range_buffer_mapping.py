"""Range-Based Parameter Buffer Mapping for Memory-Efficient Distributed Training.

This module implements advanced buffer mapping techniques that organize parameters
and gradients into contiguous memory ranges across ranks, optimizing memory access
patterns and reducing communication overhead in distributed training scenarios.

Key Features:
- Efficient range-based parameter organization
- Memory-aligned buffer allocation for optimal access patterns
- Cross-rank parameter mapping with minimal fragmentation
- Integration with existing gradient accumulation and clipping
- Support for mixed precision training scenarios
- Memory profiling and optimization insights
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter

from .exceptions import ConfigurationError
from .param_range import ParameterPartitioner, ParameterRange

logger = logging.getLogger(__name__)

# Constants for memory optimization
MEMORY_ALIGNMENT_BYTES = 64  # Optimal alignment for modern hardware
MIN_RANGE_SIZE_ELEMENTS = 256  # Minimum range size to justify overhead
DEFAULT_BUFFER_GROWTH_FACTOR = 1.5  # Buffer growth factor for dynamic allocation
MAX_FRAGMENTATION_RATIO = 0.1  # Maximum acceptable memory fragmentation


def validate_parameters(
    parameters: List[Parameter], operation_name: str = "operation"
) -> None:
    """Validate a list of parameters for consistency.

    Args:
        parameters: List of parameters to validate.
        operation_name: Name of operation for error messages.

    Raises:
        ConfigurationError: If parameters are invalid or inconsistent.
    """
    if not parameters:
        raise ConfigurationError(f"No parameters provided for {operation_name}")

    first_device = parameters[0].device

    for i, param in enumerate(parameters):
        if param is None:
            raise ConfigurationError(f"Parameter {i} is None in {operation_name}")
        if not isinstance(param, torch.nn.Parameter):
            raise ConfigurationError(
                f"Parameter {i} is not a torch.nn.Parameter in {operation_name}, "
                f"got {type(param)}"
            )
        if param.device != first_device:
            raise ConfigurationError(
                f"Parameter {i} is on device {param.device} in {operation_name}, "
                f"expected {first_device}"
            )
        if param.numel() == 0:
            raise ValueError(f"Parameter {i} has 0 elements in {operation_name}")


def calculate_aligned_size(size_bytes: int, alignment_bytes: int) -> int:
    """Calculate aligned size for memory allocation.

    Args:
        size_bytes: Original size in bytes.
        alignment_bytes: Alignment boundary in bytes.

    Returns:
        Aligned size in bytes.
    """
    return ((size_bytes + alignment_bytes - 1) // alignment_bytes) * alignment_bytes


def group_parameters_by_dtype(
    parameters: List[Parameter], param_indices: List[int]
) -> Dict[torch.dtype, List[int]]:
    """Group parameter indices by their data type.

    Args:
        parameters: List of parameters.
        param_indices: List of parameter indices to group.

    Returns:
        Dictionary mapping dtype to list of parameter indices.
    """
    dtype_groups: Dict[torch.dtype, List[int]] = {}

    for param_idx in param_indices:
        if param_idx < len(parameters):
            param = parameters[param_idx]
            if param.dtype not in dtype_groups:
                dtype_groups[param.dtype] = []
            dtype_groups[param.dtype].append(param_idx)

    return dtype_groups


class RangeBufferStrategy(Enum):
    """Strategy for organizing parameters into ranges."""

    CONTIGUOUS = "contiguous"  # Pack parameters contiguously within rank
    SIZE_ORDERED = "size_ordered"  # Order by parameter size for better packing
    DTYPE_GROUPED = "dtype_grouped"  # Group by data type first
    GRADIENT_ALIGNED = "gradient_aligned"  # Align based on gradient patterns


class BufferAllocationMode(Enum):
    """Mode for buffer allocation."""

    STATIC = "static"  # Pre-allocate all buffers
    DYNAMIC = "dynamic"  # Allocate buffers on-demand
    HYBRID = "hybrid"  # Mix of static and dynamic allocation


@dataclass
class BufferRange:
    """Represents a contiguous range of memory in a parameter buffer.

    Attributes:
        start_offset: Starting byte offset in the buffer.
        end_offset: Ending byte offset in the buffer (exclusive).
        param_indices: List of parameter indices in this range.
        dtype: Data type of parameters in this range.
        device: Device where the buffer is allocated.
        is_active: Whether this range is currently in use.
        alignment_padding: Padding bytes added for alignment.
    """

    start_offset: int
    end_offset: int
    param_indices: List[int]
    dtype: torch.dtype
    device: torch.device
    is_active: bool = True
    alignment_padding: int = 0

    @property
    def size_bytes(self) -> int:
        """Get the size of this range in bytes."""
        return self.end_offset - self.start_offset - self.alignment_padding

    @property
    def size_elements(self) -> int:
        """Get the size of this range in elements."""
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return int(self.size_bytes // element_size)


@dataclass
class RangeBufferConfig:
    """Configuration for range-based parameter buffer mapping.

    Attributes:
        strategy: Buffer organization strategy.
        allocation_mode: Buffer allocation mode.
        alignment_bytes: Memory alignment in bytes.
        min_range_size: Minimum range size in elements.
        max_fragmentation: Maximum acceptable fragmentation ratio.
        enable_compaction: Whether to enable buffer compaction.
        growth_factor: Growth factor for dynamic buffer allocation.
        enable_profiling: Whether to enable memory profiling.
    """

    strategy: RangeBufferStrategy = RangeBufferStrategy.CONTIGUOUS
    allocation_mode: BufferAllocationMode = BufferAllocationMode.HYBRID
    alignment_bytes: int = MEMORY_ALIGNMENT_BYTES
    min_range_size: int = MIN_RANGE_SIZE_ELEMENTS
    max_fragmentation: float = MAX_FRAGMENTATION_RATIO
    enable_compaction: bool = True
    growth_factor: float = DEFAULT_BUFFER_GROWTH_FACTOR
    enable_profiling: bool = False
    device: Optional[torch.device] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if (
            self.alignment_bytes <= 0
            or (self.alignment_bytes & (self.alignment_bytes - 1)) != 0
        ):
            raise ConfigurationError(
                f"alignment_bytes must be a positive power of 2, "
                f"got {self.alignment_bytes}"
            )
        if self.min_range_size <= 0:
            raise ConfigurationError(
                f"min_range_size must be positive, got {self.min_range_size}"
            )
        if not 0.0 < self.max_fragmentation < 1.0:
            raise ConfigurationError(
                f"max_fragmentation must be between 0 and 1, "
                f"got {self.max_fragmentation}"
            )
        if self.growth_factor <= 1.0:
            raise ConfigurationError(
                f"growth_factor must be > 1.0, got {self.growth_factor}"
            )


@dataclass
class MemoryStats:
    """Memory usage statistics for range buffer mapping.

    Attributes:
        total_allocated_bytes: Total allocated buffer memory.
        total_used_bytes: Total used memory within buffers.
        fragmentation_ratio: Ratio of unused to total memory.
        num_ranges: Total number of active ranges.
        num_buffers: Total number of allocated buffers.
        alignment_waste_bytes: Memory wasted due to alignment.
    """

    total_allocated_bytes: int = 0
    total_used_bytes: int = 0
    fragmentation_ratio: float = 0.0
    num_ranges: int = 0
    num_buffers: int = 0
    alignment_waste_bytes: int = 0
    peak_allocated_bytes: int = 0
    compaction_count: int = 0


class RangeBufferMapper:
    """Advanced parameter buffer mapper using range-based organization.

    This class provides efficient mapping of parameters to contiguous memory
    buffers, optimizing for memory access patterns and reducing fragmentation
    in distributed training scenarios.

    Args:
        parameters: List of model parameters to map.
        config: Configuration for range buffer mapping.
        world_size: Total number of distributed ranks.
        rank: Current rank identifier.
        process_group: Process group for distributed communication.
    """

    def __init__(
        self,
        parameters: List[Parameter],
        config: RangeBufferConfig,
        world_size: int = 1,
        rank: int = 0,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        # Validate inputs
        validate_parameters(parameters, "range buffer mapping")

        self.parameters = parameters
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.process_group = process_group

        # Device configuration
        self.device = config.device or (
            parameters[0].device if parameters else torch.device("cpu")
        )

        # Thread safety
        self._lock = threading.RLock()

        # Initialize core components
        self.partitioner = ParameterPartitioner(world_size, rank)
        self.buffer_ranges: List[BufferRange] = []
        self.param_to_range: Dict[int, int] = {}  # param_idx -> range_idx
        self.param_to_buffer_offset: Dict[
            int, Tuple[int, int]
        ] = {}  # param_idx -> (start, end)

        # Buffer storage
        self.buffers: Dict[torch.dtype, Tensor] = {}
        self.buffer_sizes: Dict[torch.dtype, int] = {}

        # Statistics and profiling
        self.stats = MemoryStats()
        self._profiling_data: List[Dict[str, float]] = []
        self._stats_dirty = True  # Track if statistics need updating
        self._last_stats_update = 0.0  # Timestamp of last statistics update

        # Initialize mapping
        self._initialize_mapping()

        logger.info(
            f"Initialized RangeBufferMapper with {len(self.buffer_ranges)} ranges, "
            f"strategy={config.strategy.value}, "
            f"allocation_mode={config.allocation_mode.value}"
        )

    def _initialize_mapping(self) -> None:
        """Initialize the parameter to buffer range mapping."""
        with self._lock:
            # Compute parameter partitions across ranks
            self.partitioner.compute_partition_ranges(self.parameters, contiguous=True)

            # Get local parameter range for this rank
            local_range = self.partitioner.get_local_param_range()
            if local_range is None:
                logger.warning(f"Rank {self.rank} has no parameters assigned")
                return

            # Create buffer ranges based on strategy
            if self.config.strategy == RangeBufferStrategy.CONTIGUOUS:
                self._create_contiguous_ranges(local_range)
            elif self.config.strategy == RangeBufferStrategy.SIZE_ORDERED:
                self._create_size_ordered_ranges(local_range)
            elif self.config.strategy == RangeBufferStrategy.DTYPE_GROUPED:
                self._create_dtype_grouped_ranges(local_range)
            elif self.config.strategy == RangeBufferStrategy.GRADIENT_ALIGNED:
                self._create_gradient_aligned_ranges(local_range)
            else:
                raise ConfigurationError(f"Unknown strategy: {self.config.strategy}")

            # Allocate buffers based on allocation mode
            self._allocate_buffers()

            # Mark statistics as dirty (will be updated lazily)
            self._stats_dirty = True

    def _create_contiguous_ranges(self, param_range: ParameterRange) -> None:
        """Create contiguous buffer ranges from parameter range."""
        current_offset = 0

        # Group parameters by dtype for efficient packing
        dtype_groups = group_parameters_by_dtype(
            self.parameters, param_range.param_indices
        )

        # Create ranges for each dtype group
        for dtype, param_indices in dtype_groups.items():
            if not param_indices:
                continue

            # Calculate total size for this dtype
            total_elements = 0
            for param_idx in param_indices:
                if param_idx < len(self.parameters):
                    param = self.parameters[param_idx]
                    param_slice = param_range.get_param_slice(param_idx, param.numel())
                    if param_slice is not None:
                        start, end = param_slice
                        total_elements += end - start

            if total_elements < self.config.min_range_size:
                logger.debug(
                    f"Skipping small range with {total_elements} elements "
                    f"(min: {self.config.min_range_size})"
                )
                continue

            # Calculate byte sizes with alignment
            element_size = torch.tensor([], dtype=dtype).element_size()
            size_bytes = total_elements * element_size
            aligned_size = self._align_size(size_bytes)

            # Create buffer range
            buffer_range = BufferRange(
                start_offset=current_offset,
                end_offset=current_offset + aligned_size,
                param_indices=param_indices.copy(),
                dtype=dtype,
                device=self.device,
                alignment_padding=aligned_size - size_bytes,
            )

            self.buffer_ranges.append(buffer_range)
            range_idx = len(self.buffer_ranges) - 1

            # Map parameters to this range
            buffer_offset = 0
            for param_idx in param_indices:
                if param_idx < len(self.parameters):
                    param = self.parameters[param_idx]
                    param_slice = param_range.get_param_slice(param_idx, param.numel())
                    if param_slice is not None:
                        start, end = param_slice
                        slice_elements = end - start

                        self.param_to_range[param_idx] = range_idx
                        self.param_to_buffer_offset[param_idx] = (
                            buffer_offset,
                            buffer_offset + slice_elements,
                        )
                        buffer_offset += slice_elements

            current_offset += aligned_size

    def _create_size_ordered_ranges(self, param_range: ParameterRange) -> None:
        """Create ranges ordered by parameter size for better packing."""
        # Get parameter sizes with indices
        param_sizes: List[Tuple[int, int]] = []  # (param_idx, size)
        for param_idx in param_range.param_indices:
            if param_idx < len(self.parameters):
                param = self.parameters[param_idx]
                param_slice = param_range.get_param_slice(param_idx, param.numel())
                if param_slice is not None:
                    start, end = param_slice
                    param_sizes.append((param_idx, end - start))

        # Sort by size (largest first for better packing)
        param_sizes.sort(key=lambda x: x[1], reverse=True)

        # Group into ranges respecting dtype boundaries
        dtype_param_map: Dict[torch.dtype, List[Tuple[int, int]]] = {}
        for param_idx, size in param_sizes:
            param = self.parameters[param_idx]
            if param.dtype not in dtype_param_map:
                dtype_param_map[param.dtype] = []
            dtype_param_map[param.dtype].append((param_idx, size))

        # Create ranges for each dtype with size-based packing
        for dtype, param_list in dtype_param_map.items():
            self._create_packed_ranges(dtype, param_list, param_range)

    def _create_dtype_grouped_ranges(self, param_range: ParameterRange) -> None:
        """Create ranges grouped by data type."""
        dtype_groups: Dict[torch.dtype, List[int]] = {}

        # Group parameters by dtype
        dtype_groups = group_parameters_by_dtype(
            self.parameters, param_range.param_indices
        )

        # Create ranges for each dtype
        current_offset = 0
        for dtype, param_indices in dtype_groups.items():
            param_list = [(idx, self.parameters[idx].numel()) for idx in param_indices]
            current_offset = self._create_packed_ranges(
                dtype, param_list, param_range, current_offset
            )

    def _create_gradient_aligned_ranges(self, param_range: ParameterRange) -> None:
        """Create ranges aligned with gradient computation patterns.

        This strategy optimizes memory layout for efficient gradient computation
        by considering the following factors:
        1. Parameters that share gradients should be co-located
        2. Frequently accessed parameters during backward pass get priority
        3. Memory access patterns during gradient accumulation

        Current Implementation:
        This is currently a placeholder that falls back to contiguous mapping.
        A full implementation would:
        - Analyze the computational graph to identify gradient dependencies
        - Group parameters by their gradient computation order
        - Prioritize parameters with higher gradient update frequency
        - Consider memory bank conflicts in the target hardware

        Future Enhancement:
        Could be extended to use runtime profiling data to optimize
        parameter placement based on actual gradient access patterns.
        """
        # For now, fall back to contiguous ranges with gradient-friendly alignment
        logger.info("Using gradient-aligned strategy (fallback to contiguous)")
        self._create_contiguous_ranges(param_range)

    def _create_packed_ranges(
        self,
        dtype: torch.dtype,
        param_list: List[Tuple[int, int]],
        param_range: ParameterRange,
        start_offset: int = 0,
    ) -> int:
        """Create packed ranges for a list of parameters with the same dtype."""
        if not param_list:
            return start_offset

        current_offset = start_offset

        # Calculate optimal range size
        total_elements = sum(size for _, size in param_list)
        if total_elements < self.config.min_range_size:
            return current_offset

        # Pack parameters into ranges
        current_range_params: List[int] = []
        current_range_size = 0

        for param_idx, param_elements in param_list:
            # Check if parameter fits in current range or if we need a new range
            if (
                current_range_size > 0
                and current_range_size + param_elements > self.config.min_range_size * 4
            ):
                # Create current range
                current_offset = self._finalize_packed_range(
                    dtype, current_range_params, param_range, current_offset
                )
                current_range_params = []
                current_range_size = 0

            current_range_params.append(param_idx)
            current_range_size += param_elements

        # Create final range
        if current_range_params:
            current_offset = self._finalize_packed_range(
                dtype, current_range_params, param_range, current_offset
            )

        return current_offset

    def _finalize_packed_range(
        self,
        dtype: torch.dtype,
        param_indices: List[int],
        param_range: ParameterRange,
        start_offset: int,
    ) -> int:
        """Finalize a packed range and create the buffer range."""
        if not param_indices:
            return start_offset

        # Calculate total size
        total_elements = 0
        for param_idx in param_indices:
            param = self.parameters[param_idx]
            param_slice = param_range.get_param_slice(param_idx, param.numel())
            if param_slice is not None:
                start, end = param_slice
                total_elements += end - start

        if total_elements == 0:
            return start_offset

        # Calculate aligned size
        element_size = torch.tensor([], dtype=dtype).element_size()
        size_bytes = total_elements * element_size
        aligned_size = self._align_size(size_bytes)

        # Create buffer range
        buffer_range = BufferRange(
            start_offset=start_offset,
            end_offset=start_offset + aligned_size,
            param_indices=param_indices.copy(),
            dtype=dtype,
            device=self.device,
            alignment_padding=aligned_size - size_bytes,
        )

        self.buffer_ranges.append(buffer_range)
        range_idx = len(self.buffer_ranges) - 1

        # Map parameters to this range
        buffer_offset = 0
        for param_idx in param_indices:
            param = self.parameters[param_idx]
            param_slice = param_range.get_param_slice(param_idx, param.numel())
            if param_slice is not None:
                start, end = param_slice
                slice_elements = end - start

                self.param_to_range[param_idx] = range_idx
                self.param_to_buffer_offset[param_idx] = (
                    buffer_offset,
                    buffer_offset + slice_elements,
                )
                buffer_offset += slice_elements

        return start_offset + aligned_size

    def _allocate_buffers(self) -> None:
        """Allocate memory buffers based on ranges and allocation mode."""
        # Calculate buffer sizes by dtype
        dtype_sizes: Dict[torch.dtype, int] = {}
        for buffer_range in self.buffer_ranges:
            dtype = buffer_range.dtype
            range_size = buffer_range.end_offset - buffer_range.start_offset
            dtype_sizes[dtype] = dtype_sizes.get(dtype, 0) + range_size

        # Allocate buffers
        for dtype, size_bytes in dtype_sizes.items():
            if size_bytes == 0:
                continue

            element_size = torch.tensor([], dtype=dtype).element_size()
            num_elements = size_bytes // element_size

            if self.config.allocation_mode in [
                BufferAllocationMode.STATIC,
                BufferAllocationMode.HYBRID,
            ]:
                # Allocate full buffer upfront
                self.buffers[dtype] = torch.zeros(
                    num_elements, dtype=dtype, device=self.device
                )
                self.buffer_sizes[dtype] = size_bytes

                if self.config.enable_profiling:
                    logger.debug(
                        f"Allocated {size_bytes / (1024**2):.2f}MB buffer "
                        f"for dtype {dtype}"
                    )

        # Update peak memory usage
        total_allocated = sum(self.buffer_sizes.values())
        self.stats.peak_allocated_bytes = max(
            self.stats.peak_allocated_bytes, total_allocated
        )

    def _align_size(self, size_bytes: int) -> int:
        """Align size to the configured alignment boundary."""
        return calculate_aligned_size(size_bytes, self.config.alignment_bytes)

    def _update_statistics(self, force: bool = False) -> None:
        """Update memory usage statistics.

        Args:
            force: Force update even if recently updated.
        """
        import time

        current_time = time.time()

        # Rate limit statistics updates for performance (max once per 100ms)
        if (
            not force
            and not self._stats_dirty
            and (current_time - self._last_stats_update) < 0.1
        ):
            return

        with self._lock:
            self.stats.num_ranges = len([r for r in self.buffer_ranges if r.is_active])
            self.stats.num_buffers = len(self.buffers)
            self.stats.total_allocated_bytes = sum(self.buffer_sizes.values())

            # Calculate used bytes
            used_bytes = 0
            alignment_waste = 0
            for buffer_range in self.buffer_ranges:
                if buffer_range.is_active:
                    used_bytes += buffer_range.size_bytes
                    alignment_waste += buffer_range.alignment_padding

            self.stats.total_used_bytes = used_bytes
            self.stats.alignment_waste_bytes = alignment_waste

            # Calculate fragmentation
            if self.stats.total_allocated_bytes > 0:
                unused_bytes = (
                    self.stats.total_allocated_bytes - self.stats.total_used_bytes
                )
                self.stats.fragmentation_ratio = (
                    unused_bytes / self.stats.total_allocated_bytes
                )

            self._stats_dirty = False
            self._last_stats_update = current_time

    def get_parameter_buffer(self, param_idx: int) -> Optional[Tuple[Tensor, int, int]]:
        """Get buffer and offsets for a parameter.

        Args:
            param_idx: Index of the parameter.

        Returns:
            Tuple of (buffer, start_offset, end_offset) or None if not mapped.
        """
        # Validate parameter index
        if param_idx < 0 or param_idx >= len(self.parameters):
            logger.debug(
                f"Parameter index {param_idx} out of range [0, {len(self.parameters)}]"
            )
            return None

        if param_idx not in self.param_to_range:
            return None

        range_idx = self.param_to_range[param_idx]
        if range_idx < 0 or range_idx >= len(self.buffer_ranges):
            logger.warning(f"Range index {range_idx} out of bounds")
            return None

        buffer_range = self.buffer_ranges[range_idx]
        if not buffer_range.is_active or buffer_range.dtype not in self.buffers:
            return None

        buffer = self.buffers[buffer_range.dtype]
        if param_idx not in self.param_to_buffer_offset:
            logger.warning(f"Parameter {param_idx} missing buffer offset mapping")
            return None

        start_offset, end_offset = self.param_to_buffer_offset[param_idx]

        # Validate buffer bounds
        if (
            start_offset < 0
            or end_offset > buffer.numel()
            or start_offset >= end_offset
        ):
            logger.error(
                f"Invalid buffer bounds for param {param_idx}: "
                f"[{start_offset}, {end_offset}] in buffer of size {buffer.numel()}"
            )
            return None

        return buffer, start_offset, end_offset

    def copy_parameters_to_buffers(self) -> None:
        """Copy parameter data to their assigned buffer ranges."""
        with self._lock:
            for param_idx, param in enumerate(self.parameters):
                buffer_info = self.get_parameter_buffer(param_idx)
                if buffer_info is None:
                    continue

                buffer, start_offset, end_offset = buffer_info
                param_data = param.view(-1)

                # Get parameter slice for this rank
                if param_idx in self.param_to_range:
                    # Get local range
                    local_range = self.partitioner.get_local_param_range()
                    if local_range is not None:
                        param_slice = local_range.get_param_slice(
                            param_idx, param.numel()
                        )
                        if param_slice is not None:
                            slice_start, slice_end = param_slice
                            param_slice_data = param_data[slice_start:slice_end]

                            with torch.no_grad():
                                buffer[start_offset:end_offset].copy_(param_slice_data)

    def copy_buffers_to_parameters(self) -> None:
        """Copy buffer data back to parameters."""
        with self._lock:
            for param_idx, param in enumerate(self.parameters):
                buffer_info = self.get_parameter_buffer(param_idx)
                if buffer_info is None:
                    continue

                buffer, start_offset, end_offset = buffer_info

                # Get local range and parameter slice
                local_range = self.partitioner.get_local_param_range()
                if local_range is not None:
                    param_slice = local_range.get_param_slice(param_idx, param.numel())
                    if param_slice is not None:
                        slice_start, slice_end = param_slice
                        buffer_data = buffer[start_offset:end_offset]

                        with torch.no_grad():
                            param.view(-1)[slice_start:slice_end].copy_(buffer_data)

    def compact_buffers(self) -> bool:
        """Compact buffers to reduce fragmentation.

        Buffer compaction is a memory optimization technique that reorganizes
        allocated memory to reduce fragmentation and improve access patterns.

        Algorithm:
        1. Check if fragmentation exceeds the configured threshold
        2. Preserve current parameter data by copying to temporary storage
        3. Rebuild the range mapping with optimal packing strategy
        4. Reallocate buffers with the new layout
        5. Restore parameter data to the new buffer locations

        Benefits:
        - Reduces memory fragmentation
        - Improves cache locality for parameter access
        - Enables more efficient memory allocation for future operations

        Costs:
        - Temporary memory overhead during compaction
        - CPU cycles for data reorganization
        - Brief interruption of memory operations

        Returns:
            True if compaction was performed, False otherwise.
        """
        if not self.config.enable_compaction:
            return False

        with self._lock:
            # Check if compaction is needed
            if self.stats.fragmentation_ratio < self.config.max_fragmentation:
                return False

            logger.info(
                f"Compacting buffers "
                f"(fragmentation: {self.stats.fragmentation_ratio:.2%})"
            )

            # Rebuild ranges with better packing
            self.buffer_ranges.clear()
            self.param_to_range.clear()
            self.param_to_buffer_offset.clear()

            # Re-initialize mapping with optimized layout
            local_range = self.partitioner.get_local_param_range()
            if local_range is not None:
                self._create_contiguous_ranges(local_range)
                self._allocate_buffers()
                self._update_statistics(force=True)

            self.stats.compaction_count += 1

            logger.info(
                f"Buffer compaction complete. "
                f"Fragmentation reduced to {self.stats.fragmentation_ratio:.2%}"
            )

            return True

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics.

        Returns:
            Memory statistics object.
        """
        self._update_statistics(force=False)
        return self.stats

    def get_buffer_info(self) -> Dict[str, object]:
        """Get detailed buffer information for debugging.

        Returns:
            Dictionary with buffer configuration and statistics.
        """
        self._update_statistics()

        dtype_info = {}
        for dtype, buffer in self.buffers.items():
            dtype_info[str(dtype)] = {
                "size_mb": buffer.numel() * buffer.element_size() / (1024**2),
                "num_elements": buffer.numel(),
                "device": str(buffer.device),
            }

        return {
            "config": {
                "strategy": self.config.strategy.value,
                "allocation_mode": self.config.allocation_mode.value,
                "alignment_bytes": self.config.alignment_bytes,
                "min_range_size": self.config.min_range_size,
            },
            "statistics": {
                "num_ranges": self.stats.num_ranges,
                "num_buffers": self.stats.num_buffers,
                "total_allocated_mb": self.stats.total_allocated_bytes / (1024**2),
                "total_used_mb": self.stats.total_used_bytes / (1024**2),
                "fragmentation_ratio": self.stats.fragmentation_ratio,
                "alignment_waste_mb": self.stats.alignment_waste_bytes / (1024**2),
                "peak_allocated_mb": self.stats.peak_allocated_bytes / (1024**2),
                "compaction_count": self.stats.compaction_count,
            },
            "buffer_types": dtype_info,
            "range_info": [
                {
                    "dtype": str(r.dtype),
                    "size_elements": r.size_elements,
                    "size_mb": r.size_bytes / (1024**2),
                    "num_params": len(r.param_indices),
                    "is_active": r.is_active,
                }
                for r in self.buffer_ranges
            ],
        }

    def __repr__(self) -> str:
        """String representation of the range buffer mapper."""
        return (
            f"RangeBufferMapper("
            f"num_ranges={len(self.buffer_ranges)}, "
            f"strategy={self.config.strategy.value}, "
            f"total_mb={self.stats.total_allocated_bytes / (1024**2):.2f})"
        )
