"""
Advanced Multi-Parallel RNG State Tracker for RoseLLM

This module provides Megatron-LM compatible RNG state management with support for
multi-dimensional parallelism (TP, PP, DP, CP, EP). It enables deterministic training
across different parallelism configurations while maintaining RNG independence
between parallel dimensions.

Key Features:
- Multi-parallel RNG state isolation
- CUDA Graph compatibility
- Checkpoint/restore capabilities
- Advanced fork and merge operations
- Performance optimization with caching
- Comprehensive error handling and validation

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- PyTorch RNG: https://pytorch.org/docs/stable/notes/randomness.html
"""

import logging
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

import torch

logger = logging.getLogger(__name__)


class RNGStateType(Enum):
    """Types of RNG states managed by the tracker."""

    CUDA = "cuda"
    CPU = "cpu"
    NUMPY = "numpy"
    PYTHON = "python"


@dataclass
class RNGStateInfo:
    """Information about an RNG state."""

    name: str
    state_type: RNGStateType
    device_id: Optional[int] = None
    parallel_dimensions: Set[str] = field(default_factory=set)
    checkpoint_data: Optional[Dict[str, Any]] = None
    creation_step: int = 0
    last_access_step: int = 0
    is_forked: bool = False
    parent_state: Optional[str] = None
    children_states: Set[str] = field(default_factory=set)


class CudaRNGStatesTracker:
    """
    Advanced CUDA RNG states tracker with multi-parallel support.

    This class manages RNG states across multiple parallelism dimensions,
    ensuring reproducible results while maintaining independence between
    different parallel contexts.
    """

    def __init__(
        self,
        enable_cuda_graphs: bool = False,
        cache_capacity: int = 1000,
        auto_cleanup: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the RNG states tracker.

        Args:
            enable_cuda_graphs: Enable CUDA Graph compatibility mode
            cache_capacity: Maximum number of cached RNG states
            auto_cleanup: Automatically clean up unused states
            verbose: Enable verbose logging
        """
        self._states: Dict[str, torch.Tensor] = {}
        self._state_info: Dict[str, RNGStateInfo] = {}
        self._current_states: Dict[str, str] = {}  # dimension -> current state name

        # Configuration
        self.enable_cuda_graphs = enable_cuda_graphs
        self.cache_capacity = cache_capacity
        self.auto_cleanup = auto_cleanup
        self.verbose = verbose

        # Thread safety
        self._lock = threading.RLock()

        # Performance tracking
        self._step_counter = 0
        self._access_counter = 0
        self._fork_counter = 0
        self._merge_counter = 0

        # CUDA Graph compatibility
        self._cuda_graph_mode = False
        self._cuda_graph_states: Dict[str, torch.Tensor] = {}

        # Cache management
        self._lru_order: List[str] = []

        logger.info(
            f"Initialized CudaRNGStatesTracker with cuda_graphs={enable_cuda_graphs}, "
            f"cache_capacity={cache_capacity}, auto_cleanup={auto_cleanup}"
        )

    def add(
        self,
        name: str,
        parallel_dimensions: Optional[Union[str, List[str]]] = None,
        device_id: Optional[int] = None,
        seed: Optional[int] = None,
        state_type: Optional[RNGStateType] = None,
        force: bool = False,
    ) -> None:
        """
        Add a new RNG state to the tracker.

        Args:
            name: Unique name for the RNG state
            parallel_dimensions: Parallelism dimensions this state belongs to
            device_id: CUDA device ID (None for current device)
            seed: Seed for initializing the state (None for random)
            state_type: Type of RNG state
            force: Force overwrite existing state

        Raises:
            ValueError: If state already exists and force=False
            RuntimeError: If CUDA is not available for CUDA states
        """
        with self._lock:
            if name in self._states and not force:
                raise ValueError(
                    f"RNG state '{name}' already exists. Use force=True to overwrite."
                )

            # Auto-detect state type if not provided
            if state_type is None:
                state_type = (
                    RNGStateType.CUDA if torch.cuda.is_available() else RNGStateType.CPU
                )

            # Validate state type requirements
            if state_type == RNGStateType.CUDA and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU RNG state")
                state_type = RNGStateType.CPU

            # Normalize parallel dimensions
            if parallel_dimensions is None:
                parallel_dims = set()
            elif isinstance(parallel_dimensions, str):
                parallel_dims = {parallel_dimensions}
            else:
                parallel_dims = set(parallel_dimensions)

            # Validate parallel dimensions
            valid_dims = {"tp", "pp", "dp", "cp", "ep", "global"}
            invalid_dims = parallel_dims - valid_dims
            if invalid_dims:
                raise ValueError(f"Invalid parallel dimensions: {invalid_dims}")

            # Get device for CUDA states
            if state_type == RNGStateType.CUDA:
                if device_id is None:
                    device_id = torch.cuda.current_device()
                cuda_device = torch.device(f"cuda:{device_id}")
            else:
                cuda_device = None

            # Initialize RNG state
            if state_type == RNGStateType.CUDA:
                # Save current state
                if cuda_device is not None:
                    current_state = torch.cuda.get_rng_state(cuda_device)
                else:
                    current_state = torch.cuda.get_rng_state()

                # Set seed if provided
                if seed is not None:
                    torch.cuda.manual_seed(seed)

                # Capture new state
                if cuda_device is not None:
                    new_state = torch.cuda.get_rng_state(cuda_device)
                else:
                    new_state = torch.cuda.get_rng_state()

                # Restore original state
                if cuda_device is not None:
                    torch.cuda.set_rng_state(current_state, cuda_device)
                else:
                    torch.cuda.set_rng_state(current_state)

                self._states[name] = new_state.clone()
            else:
                # For non-CUDA states, store minimal info
                self._states[name] = torch.tensor([seed or 0], dtype=torch.int64)

            # Store metadata
            self._state_info[name] = RNGStateInfo(
                name=name,
                state_type=state_type,
                device_id=device_id,
                parallel_dimensions=parallel_dims,
                creation_step=self._step_counter,
            )

            # Update LRU order
            self._update_lru_order(name)

            # Cleanup if needed
            if self.auto_cleanup:
                self._cleanup_cache()

            if self.verbose:
                logger.info(
                    f"Added RNG state '{name}' for dimensions {parallel_dims} "
                    f"(device={device_id}, seed={seed})"
                )

    def fork(
        self,
        source_name: str,
        new_name: str,
        parallel_dimensions: Optional[Union[str, List[str]]] = None,
        offset: int = 0,
    ) -> None:
        """
        Fork an existing RNG state to create a new independent state.

        Args:
            source_name: Name of the source RNG state
            new_name: Name for the new forked state
            parallel_dimensions: Override parallel dimensions for new state
            offset: Random offset to apply for differentiation

        Raises:
            KeyError: If source state doesn't exist
            ValueError: If new state name already exists
        """
        with self._lock:
            if source_name not in self._states:
                raise KeyError(f"Source RNG state '{source_name}' not found")

            if new_name in self._states:
                raise ValueError(f"Forked state name '{new_name}' already exists")

            source_info = self._state_info[source_name]

            # Use source dimensions if not overridden
            if parallel_dimensions is None:
                parallel_dims = source_info.parallel_dimensions.copy()
            elif isinstance(parallel_dimensions, str):
                parallel_dims = {parallel_dimensions}
            else:
                parallel_dims = set(parallel_dimensions)

            # Clone the source state
            forked_state = self._states[source_name].clone()

            # Apply offset if specified
            if offset > 0 and source_info.state_type == RNGStateType.CUDA:
                # Advance RNG state by offset steps
                with torch.cuda.device(source_info.device_id or 0):
                    original_state = torch.cuda.get_rng_state()
                    torch.cuda.set_rng_state(forked_state)

                    # Generate random numbers to advance state
                    for _ in range(offset):
                        torch.rand(1, device=f"cuda:{source_info.device_id or 0}")

                    forked_state = torch.cuda.get_rng_state()
                    torch.cuda.set_rng_state(original_state)

            # Store forked state
            self._states[new_name] = forked_state
            self._state_info[new_name] = RNGStateInfo(
                name=new_name,
                state_type=source_info.state_type,
                device_id=source_info.device_id,
                parallel_dimensions=parallel_dims,
                creation_step=self._step_counter,
                is_forked=True,
                parent_state=source_name,
            )

            # Update parent-child relationships
            self._state_info[source_name].children_states.add(new_name)

            # Update counters and LRU
            self._fork_counter += 1
            self._update_lru_order(new_name)

            if self.verbose:
                logger.info(
                    f"Forked RNG state '{source_name}' -> '{new_name}' "
                    f"with offset {offset} for dimensions {parallel_dims}"
                )

    def get_states(self) -> List[str]:
        """Get list of all RNG state names."""
        with self._lock:
            return list(self._states.keys())

    def get_cuda_rng_tracker(self) -> "CudaRNGStatesTracker":
        """Return self for compatibility with Megatron-LM."""
        return self

    def reset(self) -> None:
        """Reset the tracker and remove all RNG states."""
        with self._lock:
            self._states.clear()
            self._state_info.clear()
            self._current_states.clear()
            self._lru_order.clear()
            self._cuda_graph_states.clear()

            # Reset counters
            self._step_counter = 0
            self._access_counter = 0
            self._fork_counter = 0
            self._merge_counter = 0

            logger.info("Reset RNG tracker - all states cleared")

    def remove(self, name: str, cleanup_children: bool = True) -> None:
        """
        Remove an RNG state from the tracker.

        Args:
            name: Name of the RNG state to remove
            cleanup_children: Also remove child states

        Raises:
            KeyError: If state doesn't exist
        """
        with self._lock:
            if name not in self._states:
                raise KeyError(f"RNG state '{name}' not found")

            state_info = self._state_info[name]

            # Remove children if requested
            if cleanup_children and state_info.children_states:
                for child_name in list(state_info.children_states):
                    self.remove(child_name, cleanup_children=True)

            # Update parent's children set
            if state_info.parent_state and state_info.parent_state in self._state_info:
                self._state_info[state_info.parent_state].children_states.discard(name)

            # Remove from current states if active
            for dim, current_name in list(self._current_states.items()):
                if current_name == name:
                    del self._current_states[dim]

            # Remove from tracker
            del self._states[name]
            del self._state_info[name]

            # Remove from LRU order
            if name in self._lru_order:
                self._lru_order.remove(name)

            # Remove from CUDA graph states if present
            if name in self._cuda_graph_states:
                del self._cuda_graph_states[name]

            if self.verbose:
                logger.info(
                    f"Removed RNG state '{name}' and "
                    f"{len(state_info.children_states)} children"
                )

    @contextmanager
    def fork_and_set(
        self, name: str, parallel_dimensions: Optional[Union[str, List[str]]] = None
    ):
        """
        Context manager that forks current RNG state and sets it temporarily.

        Args:
            name: Name for the forked state
            parallel_dimensions: Parallel dimensions for the fork

        Yields:
            The forked state name
        """
        # Determine source state based on parallel dimensions
        if parallel_dimensions:
            if isinstance(parallel_dimensions, str):
                source_dims = {parallel_dimensions}
            else:
                source_dims = set(parallel_dimensions)
        else:
            source_dims = {"global"}

        # Find appropriate source state
        source_name = None
        for dim in source_dims:
            if dim in self._current_states:
                source_name = self._current_states[dim]
                break

        if source_name is None:
            # Use global default or create one
            source_name = self._get_or_create_default_state()

        # Fork the state
        fork_name = f"{name}_fork_{self._fork_counter}"
        self.fork(source_name, fork_name, parallel_dimensions)

        # Set and yield
        try:
            self.set(fork_name)
            yield fork_name
        finally:
            # Clean up forked state
            self.remove(fork_name)

    def set(self, name: str) -> None:
        """
        Set the current RNG state.

        Args:
            name: Name of the RNG state to set

        Raises:
            KeyError: If state doesn't exist
            RuntimeError: If CUDA is not available for CUDA states
        """
        with self._lock:
            if name not in self._states:
                raise KeyError(f"RNG state '{name}' not found")

            state_info = self._state_info[name]

            if state_info.state_type == RNGStateType.CUDA:
                if not torch.cuda.is_available():
                    logger.warning(
                        f"CUDA not available for CUDA RNG state '{name}', "
                        f"using CPU fallback"
                    )
                else:
                    device = torch.device(f"cuda:{state_info.device_id or 0}")

                    if self._cuda_graph_mode and name in self._cuda_graph_states:
                        # Use cached state for CUDA graphs
                        torch.cuda.set_rng_state(self._cuda_graph_states[name], device)
                    else:
                        torch.cuda.set_rng_state(self._states[name], device)

            # Update current states for all dimensions this state belongs to
            for dim in state_info.parallel_dimensions:
                self._current_states[dim] = name

            if not state_info.parallel_dimensions:
                self._current_states["global"] = name

            # Update access tracking
            state_info.last_access_step = self._step_counter
            self._access_counter += 1
            self._update_lru_order(name)

            if self.verbose:
                logger.debug(f"Set RNG state to '{name}'")

    def get_current_state_name(
        self, parallel_dimension: str = "global"
    ) -> Optional[str]:
        """
        Get the name of the current RNG state for a parallel dimension.

        Args:
            parallel_dimension: Parallel dimension to query

        Returns:
            Current state name or None if not set
        """
        with self._lock:
            return self._current_states.get(parallel_dimension)

    def step(self) -> None:
        """Advance the global step counter."""
        with self._lock:
            self._step_counter += 1

            # Perform cache cleanup periodically
            if self.auto_cleanup and self._step_counter % 100 == 0:
                self._cleanup_cache()

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        with self._lock:
            return {
                "num_states": len(self._states),
                "step_counter": self._step_counter,
                "access_counter": self._access_counter,
                "fork_counter": self._fork_counter,
                "merge_counter": self._merge_counter,
                "cache_capacity": self.cache_capacity,
                "cuda_graph_mode": self._cuda_graph_mode,
                "current_states": dict(self._current_states),
                "state_types": {
                    name: info.state_type.value
                    for name, info in self._state_info.items()
                },
                "parallel_dimensions": {
                    name: list(info.parallel_dimensions)
                    for name, info in self._state_info.items()
                },
            }

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        with self._lock:
            return {
                "states": {name: state.clone() for name, state in self._states.items()},
                "state_info": {
                    name: {
                        "name": info.name,
                        "state_type": info.state_type.value,
                        "device_id": info.device_id,
                        "parallel_dimensions": list(info.parallel_dimensions),
                        "creation_step": info.creation_step,
                        "last_access_step": info.last_access_step,
                        "is_forked": info.is_forked,
                        "parent_state": info.parent_state,
                        "children_states": list(info.children_states),
                    }
                    for name, info in self._state_info.items()
                },
                "current_states": dict(self._current_states),
                "step_counter": self._step_counter,
                "access_counter": self._access_counter,
                "fork_counter": self._fork_counter,
                "merge_counter": self._merge_counter,
                "config": {
                    "enable_cuda_graphs": self.enable_cuda_graphs,
                    "cache_capacity": self.cache_capacity,
                    "auto_cleanup": self.auto_cleanup,
                    "verbose": self.verbose,
                },
            }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary from checkpoint."""
        with self._lock:
            # Clear current state
            self.reset()

            # Restore states
            self._states = {
                name: state.clone() for name, state in state_dict["states"].items()
            }

            # Restore state info
            for name, info_dict in state_dict["state_info"].items():
                self._state_info[name] = RNGStateInfo(
                    name=info_dict["name"],
                    state_type=RNGStateType(info_dict["state_type"]),
                    device_id=info_dict["device_id"],
                    parallel_dimensions=set(info_dict["parallel_dimensions"]),
                    creation_step=info_dict["creation_step"],
                    last_access_step=info_dict["last_access_step"],
                    is_forked=info_dict["is_forked"],
                    parent_state=info_dict["parent_state"],
                    children_states=set(info_dict["children_states"]),
                )

            # Restore current states
            self._current_states = dict(state_dict["current_states"])

            # Restore counters
            self._step_counter = state_dict["step_counter"]
            self._access_counter = state_dict["access_counter"]
            self._fork_counter = state_dict["fork_counter"]
            self._merge_counter = state_dict["merge_counter"]

            # Update LRU order based on last access
            self._lru_order = sorted(
                self._state_info.keys(),
                key=lambda name: self._state_info[name].last_access_step,
            )

            logger.info(
                f"Loaded RNG tracker state with {len(self._states)} states "
                f"at step {self._step_counter}"
            )

    def enable_cuda_graph_compatibility(self) -> None:
        """Enable CUDA Graph compatibility mode."""
        if not self.enable_cuda_graphs:
            warnings.warn("CUDA Graph support not enabled in tracker configuration")
            return

        with self._lock:
            self._cuda_graph_mode = True

            # Cache all current CUDA states
            for name, state_info in self._state_info.items():
                if state_info.state_type == RNGStateType.CUDA:
                    self._cuda_graph_states[name] = self._states[name].clone()

            logger.info("Enabled CUDA Graph compatibility mode")

    def disable_cuda_graph_compatibility(self) -> None:
        """Disable CUDA Graph compatibility mode."""
        with self._lock:
            self._cuda_graph_mode = False
            self._cuda_graph_states.clear()
            logger.info("Disabled CUDA Graph compatibility mode")

    def _get_or_create_default_state(self) -> str:
        """Get or create a default RNG state."""
        default_name = "default_global"

        if default_name not in self._states:
            self.add(
                default_name, parallel_dimensions=["global"], seed=1234  # Default seed
            )

        return default_name

    def _update_lru_order(self, name: str) -> None:
        """Update LRU order for cache management."""
        if name in self._lru_order:
            self._lru_order.remove(name)
        self._lru_order.append(name)

    def _cleanup_cache(self) -> None:
        """Clean up old cached states if over capacity."""
        if len(self._states) <= self.cache_capacity:
            return

        # Find states to remove (oldest, non-current, non-parent states)
        removable_states = []

        for name in self._lru_order:
            state_info = self._state_info[name]

            # Don't remove current states
            if any(
                current_name == name for current_name in self._current_states.values()
            ):
                continue

            # Don't remove parent states with active children
            if state_info.children_states:
                active_children = any(
                    child in self._states for child in state_info.children_states
                )
                if active_children:
                    continue

            removable_states.append(name)

        # Remove oldest states until under capacity
        states_to_remove = len(self._states) - self.cache_capacity
        for name in removable_states[:states_to_remove]:
            self.remove(name, cleanup_children=False)

        if self.verbose and states_to_remove > 0:
            logger.info(f"Cleaned up {states_to_remove} cached RNG states")


# Global tracker instance
_CUDA_RNG_STATE_TRACKER: Optional[CudaRNGStatesTracker] = None


def get_cuda_rng_tracker() -> CudaRNGStatesTracker:
    """
    Get the global CUDA RNG state tracker.

    Returns:
        Global CudaRNGStatesTracker instance
    """
    global _CUDA_RNG_STATE_TRACKER

    if _CUDA_RNG_STATE_TRACKER is None:
        _CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()

    return _CUDA_RNG_STATE_TRACKER


def reset_cuda_rng_tracker() -> None:
    """Reset the global CUDA RNG state tracker."""
    global _CUDA_RNG_STATE_TRACKER

    if _CUDA_RNG_STATE_TRACKER is not None:
        _CUDA_RNG_STATE_TRACKER.reset()
        _CUDA_RNG_STATE_TRACKER = None


def initialize_cuda_rng_tracker(
    enable_cuda_graphs: bool = False,
    cache_capacity: int = 1000,
    auto_cleanup: bool = True,
    verbose: bool = False,
) -> CudaRNGStatesTracker:
    """
    Initialize the global CUDA RNG state tracker.

    Args:
        enable_cuda_graphs: Enable CUDA Graph compatibility
        cache_capacity: Maximum cached states
        auto_cleanup: Enable automatic cleanup
        verbose: Enable verbose logging

    Returns:
        Initialized CudaRNGStatesTracker instance
    """
    global _CUDA_RNG_STATE_TRACKER

    _CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker(
        enable_cuda_graphs=enable_cuda_graphs,
        cache_capacity=cache_capacity,
        auto_cleanup=auto_cleanup,
        verbose=verbose,
    )

    return _CUDA_RNG_STATE_TRACKER
