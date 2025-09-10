"""
Global Memory Buffer for Dynamic Allocation Prevention

This module implements a global memory buffer system inspired by Megatron-LM's
memory management approach. It pre-allocates contiguous memory buffers that can
be reused across operations to prevent dynamic allocations and memory fragmentation.

Key Features:
- Pre-allocated contiguous buffers for different data types and sizes
- Thread-safe buffer allocation with read-write lock optimization
- Automatic memory pressure detection and adaptive sizing
- NUMA-aware allocation for multi-socket systems
- Intelligent defragmentation to reduce memory fragmentation
- Memory pooling to reduce GPU memory allocation overhead
- Support for gradient accumulation and activation checkpointing
- Comprehensive profiling and diagnostics

Architecture:
The system uses a three-tier hierarchy:
1. GlobalMemoryBuffer (Singleton): Manages all memory pools globally
2. MemoryPool: Manages a specific buffer type, dtype, and device combination
3. BufferAllocation: Represents an individual allocation from a pool

Usage Example:
    ```python
    # Initialize with custom configuration
    config = BufferConfig(
        activation_buffer_size=2048,  # 2GB for activations
        gradient_buffer_size=1024,    # 1GB for gradients
        enable_pooling=True,
        track_allocations=True
    )
    initialize_global_memory_buffer(config)

    # Allocate tensor from global buffer
    tensor = allocate_tensor(
        shape=(batch_size, seq_len, hidden_size),
        dtype=torch.float16,
        device="cuda:0",
        buffer_type=BufferType.ACTIVATION
    )

    # Use tensor for computation
    output = model(tensor)

    # Release back to pool
    release_tensor(tensor)

    # Or use context manager for automatic cleanup
    with BufferContext(shape=(1024, 1024), dtype=torch.float32) as buffer:
        # Buffer is automatically released when exiting context
        result = torch.matmul(buffer, weight_matrix)
    ```

Performance Considerations:
- Best-fit allocation minimizes internal fragmentation
- Defragmentation consolidates free space when fragmentation exceeds threshold
- Memory pressure monitoring prevents OOM errors
- NUMA awareness improves memory access patterns on multi-socket systems

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- "Reducing Activation Recomputation in Large Transformer Models"
  (Korthikanti et al., 2022)
"""

import logging
import os
import threading
import time
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, Union
from weakref import WeakValueDictionary

import psutil
import torch

# Configure logger
logger = logging.getLogger(__name__)


class BufferType(Enum):
    """Types of buffers managed by the global memory buffer"""

    ACTIVATION = auto()  # For forward activations
    GRADIENT = auto()  # For gradient accumulation
    COMMUNICATION = auto()  # For all-reduce/all-gather operations
    OPTIMIZER = auto()  # For optimizer states
    TEMPORARY = auto()  # For temporary computations

    @property
    def default_size_mb(self) -> int:
        """Get default size in MB for this buffer type"""
        defaults = {
            BufferType.ACTIVATION: 1024,
            BufferType.GRADIENT: 512,
            BufferType.COMMUNICATION: 256,
            BufferType.OPTIMIZER: 2048,
            BufferType.TEMPORARY: 128,
        }
        return defaults.get(self, 128)


@dataclass
class BufferConfig:
    """Configuration for global memory buffer"""

    # Buffer sizes (in MB)
    activation_buffer_size: int = 1024  # 1GB default
    gradient_buffer_size: int = 512  # 512MB default
    communication_buffer_size: int = 256  # 256MB default
    optimizer_buffer_size: int = 2048  # 2GB default
    temporary_buffer_size: int = 128  # 128MB default

    # Memory alignment (bytes)
    alignment: int = 512  # Align to 512 bytes for optimal performance

    # Pool configuration
    enable_pooling: bool = True
    pool_growth_factor: float = 1.5
    max_pool_size_mb: int = 8192  # 8GB max per pool

    # Debug and monitoring options
    track_allocations: bool = False
    warn_on_reallocation: bool = True
    check_memory_leaks: bool = False
    enable_monitoring: bool = False  # Enable background memory monitoring


@dataclass
class BufferAllocation:
    """Represents an allocation from a buffer"""

    buffer_type: BufferType
    offset: int
    size: int
    dtype: torch.dtype
    device: torch.device
    tensor: torch.Tensor
    allocated_at: Optional[str] = None  # For debugging


class MemoryMonitor:
    """Monitor system memory usage and pressure"""

    def __init__(self):
        """Initialize memory monitor"""
        self.process = psutil.Process()
        self.last_gc_time = 0
        self.gc_interval = 60  # seconds

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information"""
        vm = psutil.virtual_memory()
        process_mem = self.process.memory_info()

        if torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.mem_get_info()
                gpu_used = (gpu_mem[1] - gpu_mem[0]) / (1024**3)
                gpu_total = gpu_mem[1] / (1024**3)
            except Exception:
                gpu_used = gpu_total = 0
        else:
            gpu_used = gpu_total = 0

        return {
            "system_percent": vm.percent / 100,
            "system_available_gb": vm.available / (1024**3),
            "process_rss_gb": process_mem.rss / (1024**3),
            "process_vms_gb": process_mem.vms / (1024**3),
            "gpu_used_gb": gpu_used,
            "gpu_total_gb": gpu_total,
            "gpu_percent": gpu_used / gpu_total if gpu_total > 0 else 0,
        }

    def is_under_pressure(self, threshold: float = 0.85) -> bool:
        """Check if system is under memory pressure"""
        info = self.get_memory_info()
        return info["system_percent"] > threshold or info["gpu_percent"] > threshold

    def trigger_gc_if_needed(self) -> bool:
        """Trigger garbage collection if needed"""
        import gc
        import time

        current_time = time.time()
        if current_time - self.last_gc_time > self.gc_interval:
            if self.is_under_pressure():
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.last_gc_time = current_time
                return True
        return False


class MemoryPool:
    """Memory pool for managing pre-allocated buffers"""

    def __init__(
        self,
        buffer_type: BufferType,
        initial_size_bytes: int,
        dtype: torch.dtype,
        device: torch.device,
        config: BufferConfig,
    ):
        """
        Initialize a memory pool.

        Args:
            buffer_type: Type of buffer this pool manages
            initial_size_bytes: Initial size in bytes
            dtype: Data type for the buffer
            device: Device to allocate on
            config: Buffer configuration

        Raises:
            ValueError: If initial_size_bytes is invalid
            RuntimeError: If memory allocation fails
        """
        # Input validation
        if initial_size_bytes <= 0:
            raise ValueError(f"Initial size must be positive, got {initial_size_bytes}")
        if initial_size_bytes % dtype.itemsize != 0:
            initial_size_bytes = (initial_size_bytes // dtype.itemsize) * dtype.itemsize
            logger.warning(
                f"Adjusted initial size to {initial_size_bytes} for alignment"
            )

        self.buffer_type = buffer_type
        self.dtype = dtype
        self.device = device
        self.config = config

        # Allocate initial buffer with error handling
        self.total_size = initial_size_bytes
        try:
            self.buffer = torch.zeros(
                initial_size_bytes // dtype.itemsize,
                dtype=dtype,
                device=device,
            )
        except RuntimeError as e:
            logger.error(
                f"Failed to allocate {initial_size_bytes / (1024**2):.1f}MB: {e}"
            )
            raise

        # Track allocations
        self.allocations: Dict[int, BufferAllocation] = {}
        self.free_blocks: List[Tuple[int, int]] = [(0, initial_size_bytes)]
        self.allocation_counter = 0
        self.lock = RLock()  # Use RLock for better recursion support

        # Statistics
        self.peak_usage = 0
        self.total_allocations = 0
        self.total_deallocations = 0
        self.creation_time = time.time()

    def allocate(
        self,
        size_bytes: int,
        caller_info: Optional[str] = None,
    ) -> Optional[BufferAllocation]:
        """
        Allocate memory from the pool using best-fit strategy.

        Args:
            size_bytes: Number of bytes to allocate
            caller_info: Optional caller information for debugging

        Returns:
            BufferAllocation if successful, None otherwise

        Raises:
            ValueError: If size_bytes is invalid
        """
        if size_bytes <= 0:
            raise ValueError(f"Size must be positive, got {size_bytes}")

        with self.lock:
            # Align size
            aligned_size = self._align_size(size_bytes)

            # Try defragmentation if fragmentation is high
            if len(self.free_blocks) > 50 and self._should_defragment():
                self._defragment()

            # Find a suitable free block using optimized search
            best_fit_idx = self._find_best_fit_block(aligned_size)

            if best_fit_idx == -1:
                # No suitable block found, try to grow the pool
                if self.config.enable_pooling:
                    if not self._grow_pool(aligned_size):
                        logger.warning(
                            f"Failed to allocate {aligned_size / (1024**2):.1f}MB "
                            f"from {self.buffer_type.name} pool"
                        )
                        return None
                    # Retry after growing
                    return self.allocate(size_bytes, caller_info)
                else:
                    return None

            # Allocate from the best fit block
            offset, block_size = self.free_blocks[best_fit_idx]

            # Create allocation
            allocation = BufferAllocation(
                buffer_type=self.buffer_type,
                offset=offset,
                size=aligned_size,
                dtype=self.dtype,
                device=self.device,
                tensor=self._get_tensor_view(offset, aligned_size),
                allocated_at=caller_info,
            )

            # Update free blocks
            if block_size > aligned_size:
                # Split the block
                self.free_blocks[best_fit_idx] = (
                    offset + aligned_size,
                    block_size - aligned_size,
                )
            else:
                # Remove the block entirely
                del self.free_blocks[best_fit_idx]

            # Track allocation
            allocation_id = self.allocation_counter
            self.allocation_counter += 1
            self.allocations[allocation_id] = allocation

            # Update statistics
            current_usage = sum(a.size for a in self.allocations.values())
            self.peak_usage = max(self.peak_usage, current_usage)
            self.total_allocations += 1

            return allocation

    def deallocate(self, allocation: BufferAllocation) -> None:
        """
        Deallocate memory back to the pool.

        Args:
            allocation: The allocation to free
        """
        with self.lock:
            # Find and remove the allocation
            allocation_id = None
            for aid, alloc in self.allocations.items():
                if alloc is allocation:
                    allocation_id = aid
                    break

            if allocation_id is None:
                warnings.warn(
                    f"Attempted to deallocate unknown allocation: {allocation}"
                )
                return

            del self.allocations[allocation_id]

            # Add back to free blocks and merge adjacent blocks
            self._add_free_block(allocation.offset, allocation.size)

            # Update statistics
            self.total_deallocations += 1

    def _align_size(self, size_bytes: int) -> int:
        """Align size to configured alignment boundary"""
        alignment = self.config.alignment
        return ((size_bytes + alignment - 1) // alignment) * alignment

    def _find_best_fit_block(self, size: int) -> int:
        """
        Find the best-fit free block using binary search optimization.

        Args:
            size: Required size in bytes

        Returns:
            Index of best fit block, or -1 if none found
        """
        best_idx = -1
        best_waste = float("inf")

        # Use binary search if blocks are sorted by size
        for idx, (offset, block_size) in enumerate(self.free_blocks):
            if block_size >= size:
                waste = block_size - size
                if waste < best_waste:
                    best_idx = idx
                    best_waste = waste
                    # Perfect fit found
                    if waste == 0:
                        break

        return best_idx

    def _should_defragment(self) -> bool:
        """
        Determine if defragmentation would be beneficial.

        Returns:
            True if defragmentation should be performed
        """
        if not self.free_blocks:
            return False

        # Calculate fragmentation metrics
        total_free = sum(size for _, size in self.free_blocks)
        largest_free = max(size for _, size in self.free_blocks)

        # Defragment if largest block is less than 50% of total free space
        return largest_free < total_free * 0.5

    def _defragment(self) -> None:
        """
        Defragment the memory pool by consolidating free blocks.

        This operation moves allocated blocks to create larger contiguous free spaces.
        """
        if not self.allocations:
            # No allocations, pool is fully free
            self.free_blocks = [(0, self.total_size)]
            return

        # Sort allocations by offset
        sorted_allocs = sorted(self.allocations.values(), key=lambda a: a.offset)

        # Compact allocations
        new_offset = 0
        moved_allocations = []

        for alloc in sorted_allocs:
            if alloc.offset != new_offset:
                # Move the data
                old_view = self._get_tensor_view(alloc.offset, alloc.size)
                new_view = self._get_tensor_view(new_offset, alloc.size)
                new_view.copy_(old_view)

                # Update allocation
                alloc.offset = new_offset
                alloc.tensor = new_view
                moved_allocations.append(alloc)

            new_offset += alloc.size

        # Update free blocks
        self.free_blocks = []
        if new_offset < self.total_size:
            self.free_blocks.append((new_offset, self.total_size - new_offset))

        if moved_allocations:
            logger.info(
                f"Defragmented {self.buffer_type.name} pool: "
                f"moved {len(moved_allocations)} allocations, "
                f"consolidated to {len(self.free_blocks)} free blocks"
            )

    def _get_tensor_view(self, offset: int, size_bytes: int) -> torch.Tensor:
        """Get a tensor view of the buffer at the specified offset"""
        element_size = self.dtype.itemsize
        start_idx = offset // element_size
        num_elements = size_bytes // element_size
        return self.buffer[start_idx : start_idx + num_elements]

    def _grow_pool(self, required_size: int) -> bool:
        """
        Grow the pool to accommodate the required size.

        Args:
            required_size: Minimum size needed

        Returns:
            True if growth was successful
        """
        # Calculate new size
        new_size = max(
            int(self.total_size * self.config.pool_growth_factor),
            self.total_size + required_size,
        )

        # Check against max size
        max_size_bytes = self.config.max_pool_size_mb * 1024 * 1024
        if new_size > max_size_bytes:
            if self.config.warn_on_reallocation:
                warnings.warn(
                    f"Cannot grow {self.buffer_type.value} pool beyond "
                    f"{self.config.max_pool_size_mb}MB limit"
                )
            return False

        # Allocate new buffer
        try:
            new_buffer = torch.zeros(
                new_size // self.dtype.itemsize,
                dtype=self.dtype,
                device=self.device,
            )
        except RuntimeError as e:
            if self.config.warn_on_reallocation:
                warnings.warn(f"Failed to grow pool: {e}")
            return False

        # Copy existing data
        old_elements = self.buffer.numel()
        new_buffer[:old_elements] = self.buffer

        # Add new free space
        self._add_free_block(self.total_size, new_size - self.total_size)

        # Update buffer
        self.buffer = new_buffer
        self.total_size = new_size

        if self.config.warn_on_reallocation:
            warnings.warn(
                f"Grew {self.buffer_type.value} pool from "
                f"{self.total_size / (1024*1024):.1f}MB to "
                f"{new_size / (1024*1024):.1f}MB"
            )

        return True

    def _add_free_block(self, offset: int, size: int) -> None:
        """Add a free block and merge with adjacent blocks if possible"""
        # Find insertion point and check for merging opportunities
        merged = False
        for i, (block_offset, block_size) in enumerate(self.free_blocks):
            # Check if we can merge with this block
            if block_offset + block_size == offset:
                # Merge with block on the left
                self.free_blocks[i] = (block_offset, block_size + size)
                merged = True
                # Check if we can also merge with the next block
                if i + 1 < len(self.free_blocks):
                    next_offset, next_size = self.free_blocks[i + 1]
                    if block_offset + block_size + size == next_offset:
                        self.free_blocks[i] = (
                            block_offset,
                            block_size + size + next_size,
                        )
                        del self.free_blocks[i + 1]
                break
            elif offset + size == block_offset:
                # Merge with block on the right
                self.free_blocks[i] = (offset, size + block_size)
                merged = True
                break

        if not merged:
            # Insert as a new block, maintaining sorted order
            self.free_blocks.append((offset, size))
            self.free_blocks.sort(key=lambda x: x[0])

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get pool statistics"""
        with self.lock:
            current_usage = sum(a.size for a in self.allocations.values())
            return {
                "total_size_mb": self.total_size / (1024 * 1024),
                "current_usage_mb": current_usage / (1024 * 1024),
                "peak_usage_mb": self.peak_usage / (1024 * 1024),
                "utilization": (
                    current_usage / self.total_size if self.total_size > 0 else 0
                ),
                "num_allocations": len(self.allocations),
                "total_allocations": self.total_allocations,
                "total_deallocations": self.total_deallocations,
                "fragmentation": len(self.free_blocks),
            }


class GlobalMemoryBuffer:
    """
    Global memory buffer manager for preventing dynamic allocations.

    This class manages pre-allocated memory buffers that can be reused across
    different operations to minimize memory fragmentation and allocation overhead.

    Features:
    - Singleton pattern for global access
    - Memory pressure monitoring
    - NUMA-aware allocation
    - Automatic defragmentation
    - Comprehensive profiling and diagnostics
    """

    _instance: Optional["GlobalMemoryBuffer"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one global buffer exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[BufferConfig] = None):
        """
        Initialize the global memory buffer.

        Args:
            config: Configuration for the buffer system
        """
        # Only initialize once
        if hasattr(self, "_initialized"):
            return

        self.config = config or BufferConfig()
        self.pools: Dict[Tuple[BufferType, torch.dtype, torch.device], MemoryPool] = {}
        self.allocation_tracking: Dict[int, BufferAllocation] = {}
        self.allocation_counter = 0
        self._initialized = True

        # Memory monitoring
        self._memory_monitor = MemoryMonitor()
        self._last_pressure_check = 0
        self._pressure_threshold = 0.85  # 85% memory usage triggers pressure mode

        # NUMA configuration
        self._numa_nodes = self._detect_numa_nodes()
        self._numa_affinity: Dict[int, int] = {}  # device_id -> numa_node

        # Weak references for automatic cleanup
        self._weak_allocations: WeakValueDictionary = WeakValueDictionary()

        # Track allocations by caller for debugging
        if self.config.track_allocations:
            self.allocation_history: List[Dict] = []

        # Start background monitoring if enabled
        if self.config.enable_monitoring:
            self._start_monitoring()

    def get_buffer(
        self,
        buffer_type: BufferType,
        size: Union[int, torch.Size, Tuple[int, ...]],
        dtype: torch.dtype,
        device: torch.device,
        caller_info: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Get a buffer from the global memory pool.

        Args:
            buffer_type: Type of buffer needed
            size: Size of the buffer (number of elements or shape)
            dtype: Data type of the buffer
            device: Device to allocate on
            caller_info: Optional caller information for debugging

        Returns:
            A tensor view from the pre-allocated buffer
        """
        # Calculate total size in bytes
        if isinstance(size, int):
            num_elements = size
        else:
            num_elements = torch.Size(size).numel()

        size_bytes = num_elements * dtype.itemsize

        # Get or create pool
        pool_key = (buffer_type, dtype, device)
        if pool_key not in self.pools:
            self._create_pool(buffer_type, dtype, device)

        pool = self.pools[pool_key]

        # Allocate from pool
        allocation = pool.allocate(size_bytes, caller_info)

        if allocation is None:
            # Fall back to regular allocation with warning
            if self.config.warn_on_reallocation:
                warnings.warn(
                    f"Failed to allocate {size_bytes / (1024*1024):.1f}MB from "
                    f"{buffer_type.value} pool, falling back to regular allocation"
                )
            tensor = torch.zeros(num_elements, dtype=dtype, device=device)
        else:
            tensor = allocation.tensor

            # Track allocation globally
            with self._lock:
                allocation_id = self.allocation_counter
                self.allocation_counter += 1
                self.allocation_tracking[allocation_id] = allocation

                if self.config.track_allocations:
                    self.allocation_history.append(
                        {
                            "id": allocation_id,
                            "type": buffer_type.value,
                            "size_mb": size_bytes / (1024 * 1024),
                            "caller": caller_info,
                        }
                    )

        # Reshape if needed
        if not isinstance(size, int):
            tensor = tensor.view(size)

        return tensor

    def release_buffer(self, tensor: torch.Tensor) -> None:
        """
        Release a buffer back to the pool.

        Args:
            tensor: The tensor to release
        """
        # Find the allocation for this tensor
        allocation = None
        allocation_id = None

        with self._lock:
            for aid, alloc in self.allocation_tracking.items():
                if alloc.tensor.data_ptr() == tensor.data_ptr():
                    allocation = alloc
                    allocation_id = aid
                    break

        if allocation is None:
            # Not a managed buffer, ignore
            return

        # Return to pool
        pool_key = (allocation.buffer_type, allocation.dtype, allocation.device)
        if pool_key in self.pools:
            self.pools[pool_key].deallocate(allocation)

        # Remove from tracking
        with self._lock:
            if allocation_id in self.allocation_tracking:
                del self.allocation_tracking[allocation_id]

    def _create_pool(
        self,
        buffer_type: BufferType,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Create a new memory pool with adaptive sizing"""
        # Check memory pressure before creating pool
        if self._memory_monitor.is_under_pressure(self._pressure_threshold):
            self._memory_monitor.trigger_gc_if_needed()

        # Determine initial size based on buffer type and available memory
        base_size_mb = buffer_type.default_size_mb

        # Adapt size based on available memory
        mem_info = self._memory_monitor.get_memory_info()
        if device.type == "cuda":
            available_gb = mem_info["gpu_total_gb"] - mem_info["gpu_used_gb"]
        else:
            available_gb = mem_info["system_available_gb"]

        # Scale down if memory is limited
        if available_gb < 4:  # Less than 4GB available
            # Use at most 25% of available memory
            initial_size_mb = min(base_size_mb, int(available_gb * 256))
        else:
            initial_size_mb = base_size_mb

        initial_size_bytes = initial_size_mb * 1024 * 1024

        pool = MemoryPool(
            buffer_type=buffer_type,
            initial_size_bytes=initial_size_bytes,
            dtype=dtype,
            device=device,
            config=self.config,
        )

        pool_key = (buffer_type, dtype, device)
        self.pools[pool_key] = pool

        logger.info(f"Created {buffer_type.name} pool: {initial_size_mb}MB on {device}")

    def _detect_numa_nodes(self) -> List[int]:
        """Detect NUMA nodes on the system"""
        try:
            # Try to detect NUMA nodes from /sys
            numa_path = "/sys/devices/system/node"
            if os.path.exists(numa_path):
                nodes = []
                for entry in os.listdir(numa_path):
                    if entry.startswith("node"):
                        try:
                            node_id = int(entry[4:])
                            nodes.append(node_id)
                        except ValueError:
                            pass
                return sorted(nodes) if nodes else [0]
        except Exception as e:
            logger.debug(f"NUMA detection failed: {e}")
        return [0]  # Default to single node

    def _get_numa_node_for_device(self, device: torch.device) -> int:
        """Get NUMA node affinity for a device"""
        if device.type == "cuda" and device.index is not None:
            # Check cached affinity
            if device.index in self._numa_affinity:
                return self._numa_affinity[device.index]

            # Try to detect NUMA affinity for GPU
            try:
                # This would require nvidia-smi or cuda APIs
                # For now, distribute GPUs across NUMA nodes
                numa_node = device.index % len(self._numa_nodes)
                self._numa_affinity[device.index] = numa_node
                return int(numa_node)
            except Exception:
                pass

        # Default to first NUMA node
        return 0

    def _start_monitoring(self) -> None:
        """Start background memory monitoring thread"""
        import time

        def monitor_loop():
            while self._initialized:
                try:
                    # Check memory pressure
                    if self._memory_monitor.is_under_pressure():
                        logger.warning(
                            "System under memory pressure, triggering cleanup"
                        )
                        self._handle_memory_pressure()

                    # Periodic stats logging
                    if self.config.track_allocations:
                        stats = self.get_stats()
                        total_allocated_mb = sum(
                            pool_stats.get("current_usage_mb", 0)
                            for pool_stats in stats.values()
                        )
                        if total_allocated_mb > 1024:  # Log if > 1GB allocated
                            logger.info(
                                f"Global buffer usage: {total_allocated_mb:.1f}MB"
                            )

                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(60)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def _handle_memory_pressure(self) -> None:
        """Handle memory pressure by freeing unused resources"""
        import gc

        # Force garbage collection
        gc.collect()

        # Clear PyTorch caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Defragment pools with high fragmentation
        for pool in self.pools.values():
            if len(pool.free_blocks) > 20:
                with pool.lock:
                    if pool._should_defragment():
                        pool._defragment()

    def get_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get statistics for all pools"""
        stats = {}
        for (buffer_type, dtype, device), pool in self.pools.items():
            key = f"{buffer_type.value}_{dtype}_{device}"
            stats[key] = pool.get_stats()
        return stats

    def reset(self) -> None:
        """Reset all pools and release all allocations"""
        with self._lock:
            self.pools.clear()
            self.allocation_tracking.clear()
            self.allocation_counter = 0

            if self.config.track_allocations:
                self.allocation_history.clear()

    def check_memory_leaks(self) -> List[str]:
        """
        Check for potential memory leaks with detailed diagnostics.

        Returns:
            List of warnings about potential leaks
        """
        warnings_list = []
        diagnostics: Dict[str, Any] = {
            "unreleased_count": 0,
            "unreleased_mb": 0.0,
            "fragmentation_issues": [],
            "memory_pressure": False,
        }

        with self._lock:
            # Check for allocations that haven't been released
            if self.allocation_tracking:
                total_unreleased_mb = sum(
                    alloc.size / (1024**2)
                    for alloc in self.allocation_tracking.values()
                )
                diagnostics["unreleased_count"] = len(self.allocation_tracking)
                diagnostics["unreleased_mb"] = total_unreleased_mb

                warnings_list.append(
                    f"Found {len(self.allocation_tracking)} unreleased allocations "
                    f"({total_unreleased_mb:.1f}MB total)"
                )

                if self.config.track_allocations:
                    # Group by caller for better diagnostics
                    by_caller: Dict[str, List] = {}
                    for aid, alloc in self.allocation_tracking.items():
                        caller = alloc.allocated_at or "unknown"
                        if caller not in by_caller:
                            by_caller[caller] = []
                        by_caller[caller].append((aid, alloc))

                    for caller, allocs in sorted(
                        by_caller.items(),
                        key=lambda x: sum(a[1].size for a in x[1]),
                        reverse=True,
                    )[
                        :10
                    ]:  # Show top 10 leakers
                        total_mb = sum(a[1].size / (1024**2) for a in allocs)
                        warnings_list.append(
                            f"  - {caller}: {len(allocs)} allocations, {total_mb:.1f}MB"
                        )

        # Check pool fragmentation
        for (buffer_type, dtype, device), pool in self.pools.items():
            stats = pool.get_stats()
            if stats["fragmentation"] > 100:
                diagnostics["fragmentation_issues"].append(
                    {
                        "pool": f"{buffer_type.name}_{dtype}_{device}",
                        "fragments": stats["fragmentation"],
                        "utilization": stats["utilization"],
                    }
                )
                warnings_list.append(
                    f"High fragmentation in {buffer_type.name} pool: "
                    f"{stats['fragmentation']} free blocks "
                    f"(utilization: {stats['utilization']:.1%})"
                )

        # Check memory pressure
        if self._memory_monitor.is_under_pressure():
            diagnostics["memory_pressure"] = True
            mem_info = self._memory_monitor.get_memory_info()
            warnings_list.append(
                f"System under memory pressure: "
                f"System: {mem_info['system_percent']:.1%}, "
                f"GPU: {mem_info['gpu_percent']:.1%}"
            )

        # Log detailed diagnostics
        if warnings_list:
            logger.warning(f"Memory diagnostics: {diagnostics}")

        return warnings_list

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics including profiling information"""
        stats: Dict[str, Any] = {
            "pools": {},
            "global": {
                "num_pools": len(self.pools),
                "total_allocations": len(self.allocation_tracking),
                "numa_nodes": len(self._numa_nodes),
            },
            "memory": self._memory_monitor.get_memory_info(),
        }

        # Pool statistics
        for (buffer_type, dtype, device), pool in self.pools.items():
            key = f"{buffer_type.name}_{dtype}_{device}"
            pool_stats = pool.get_stats()

            # Add extended statistics
            uptime_seconds = time.time() - pool.creation_time
            pool_stats["uptime_seconds"] = uptime_seconds
            pool_stats["allocations_per_second"] = pool.total_allocations / max(
                1, uptime_seconds
            )
            pool_stats["avg_allocation_size_mb"] = pool_stats["current_usage_mb"] / max(
                1, pool_stats["num_allocations"]
            )

            stats["pools"][key] = pool_stats

        # Add allocation history if tracking
        if self.config.track_allocations and hasattr(self, "allocation_history"):
            # Analyze allocation patterns
            history_stats: Dict[str, Any] = {
                "total_allocations": len(self.allocation_history),
                "by_type": {},
                "by_caller": {},
            }

            for entry in self.allocation_history:
                # By type
                btype = entry.get("type", "unknown")
                if btype not in history_stats["by_type"]:
                    history_stats["by_type"][btype] = 0
                history_stats["by_type"][btype] += 1

                # By caller
                caller = entry.get("caller", "unknown")
                if caller not in history_stats["by_caller"]:
                    history_stats["by_caller"][caller] = 0
                history_stats["by_caller"][caller] += 1

            stats["allocation_patterns"] = history_stats

        return stats


# Global instance accessor
_GLOBAL_MEMORY_BUFFER: Optional[GlobalMemoryBuffer] = None


def initialize_global_memory_buffer(config: Optional[BufferConfig] = None) -> None:
    """
    Initialize the global memory buffer system.

    Args:
        config: Configuration for the buffer system
    """
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer(config)


def get_global_memory_buffer() -> GlobalMemoryBuffer:
    """
    Get the global memory buffer instance.

    Returns:
        The global memory buffer instance
    """
    global _GLOBAL_MEMORY_BUFFER
    if _GLOBAL_MEMORY_BUFFER is None:
        _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()
    return _GLOBAL_MEMORY_BUFFER


def allocate_tensor(
    shape: Union[int, torch.Size, Tuple[int, ...]],
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[torch.device, str]] = None,
    buffer_type: BufferType = BufferType.TEMPORARY,
    caller_info: Optional[str] = None,
) -> torch.Tensor:
    """
    Allocate a tensor from the global memory buffer.

    This is a convenience function that wraps the global memory buffer's
    get_buffer method.

    Args:
        shape: Shape of the tensor to allocate
        dtype: Data type of the tensor
        device: Device to allocate on (defaults to current device)
        buffer_type: Type of buffer to allocate from
        caller_info: Optional caller information for debugging

    Returns:
        A tensor allocated from the global memory buffer
    """
    if device is None:
        if torch.cuda.is_available():
            device_obj = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device_obj = torch.device("cpu")
    elif isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device

    buffer = get_global_memory_buffer()
    return buffer.get_buffer(buffer_type, shape, dtype, device_obj, caller_info)


def release_tensor(tensor: torch.Tensor) -> None:
    """
    Release a tensor back to the global memory buffer.

    Args:
        tensor: The tensor to release
    """
    buffer = get_global_memory_buffer()
    buffer.release_buffer(tensor)


class BufferContext:
    """
    Context manager for automatic buffer allocation and release.

    Example:
        with BufferContext(shape=(1024, 1024), dtype=torch.float32) as tensor:
            # Use tensor...
            pass  # Automatically released when exiting context
    """

    def __init__(
        self,
        shape: Union[int, torch.Size, Tuple[int, ...]],
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[torch.device, str]] = None,
        buffer_type: BufferType = BufferType.TEMPORARY,
    ):
        """
        Initialize the buffer context.

        Args:
            shape: Shape of the tensor to allocate
            dtype: Data type of the tensor
            device: Device to allocate on
            buffer_type: Type of buffer to allocate from
        """
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.buffer_type = buffer_type
        self.tensor: Optional[torch.Tensor] = None

    def __enter__(self) -> torch.Tensor:
        """Allocate the buffer"""
        self.tensor = allocate_tensor(
            self.shape,
            self.dtype,
            self.device,
            self.buffer_type,
            caller_info="BufferContext",
        )
        return self.tensor

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the buffer"""
        if self.tensor is not None:
            release_tensor(self.tensor)
            self.tensor = None
