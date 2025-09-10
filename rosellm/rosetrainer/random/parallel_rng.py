"""
Parallel-Aware Random Number Generation for Multi-Dimensional Parallelism

This module provides Megatron-LM compatible random number generation utilities
that work correctly across multiple parallelism dimensions (TP, PP, DP, CP, EP).
It ensures deterministic training while maintaining proper RNG independence
between different parallel contexts.

Key Features:
- Multi-parallel seed management
- Deterministic cross-rank synchronization
- Advanced RNG state forking and isolation
- Checkpoint/restore capabilities
- Performance optimization for large-scale training

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- PyTorch Distributed RNG: https://pytorch.org/docs/stable/notes/randomness.html
"""

import hashlib
import logging
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ..parallelism import parallel_state
from .rng_tracker import get_cuda_rng_tracker

logger = logging.getLogger(__name__)


# Constants for better maintainability
_DEFAULT_TENSOR_PARALLEL_OFFSET = 0
_DEFAULT_PIPELINE_PARALLEL_OFFSET = 100000
_DEFAULT_DATA_PARALLEL_OFFSET = 200000
_DEFAULT_CONTEXT_PARALLEL_OFFSET = 300000
_DEFAULT_EXPERT_PARALLEL_OFFSET = 400000


class RNGSeedError(RuntimeError):
    """Base exception for RNG seed operations."""

    pass


class ParallelStateNotInitializedError(RNGSeedError):
    """Raised when parallel state is not initialized."""

    pass


def model_parallel_cuda_manual_seed(
    seed: int,
    tensor_parallel_seed_offset: int = _DEFAULT_TENSOR_PARALLEL_OFFSET,
    pipeline_parallel_seed_offset: int = _DEFAULT_PIPELINE_PARALLEL_OFFSET,
    data_parallel_seed_offset: int = _DEFAULT_DATA_PARALLEL_OFFSET,
    context_parallel_seed_offset: int = _DEFAULT_CONTEXT_PARALLEL_OFFSET,
    expert_parallel_seed_offset: int = _DEFAULT_EXPERT_PARALLEL_OFFSET,
    enable_deterministic: bool = True,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Set manual seed for multi-parallel CUDA RNG with proper dimension isolation.

    This function creates independent RNG states for different parallelism dimensions
    while maintaining deterministic behavior across runs. Each parallel dimension
    gets its own RNG state with a deterministic but distinct seed.

    Args:
        seed: Base seed for RNG initialization
        tensor_parallel_seed_offset: Offset for tensor parallel dimension
        pipeline_parallel_seed_offset: Offset for pipeline parallel dimension
        data_parallel_seed_offset: Offset for data parallel dimension
        context_parallel_seed_offset: Offset for context parallel dimension
        expert_parallel_seed_offset: Offset for expert parallel dimension
        enable_deterministic: Enable deterministic operations
        verbose: Enable verbose logging

    Returns:
        Dictionary mapping dimension names to their computed seeds

    Raises:
        RuntimeError: If parallel state is not initialized
    """
    # Input validation
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got {seed}")

    # Validate offsets to prevent seed collisions
    offsets = [
        tensor_parallel_seed_offset,
        pipeline_parallel_seed_offset,
        data_parallel_seed_offset,
        context_parallel_seed_offset,
        expert_parallel_seed_offset,
    ]
    if len(set(offsets)) != len(offsets):
        raise ValueError("Seed offsets must be unique to prevent RNG state collisions")

    if not parallel_state.is_initialized():
        raise ParallelStateNotInitializedError(
            "Parallel state must be initialized before setting RNG seeds. "
            "Call parallel_state.initialize_model_parallel() first."
        )

    # Get parallel ranks and sizes
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_size = parallel_state.get_context_parallel_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    ep_size = parallel_state.get_expert_model_parallel_size()

    # Calculate dimension-specific seeds
    seeds = {}

    # Global seed (same across all ranks)
    seeds["global"] = seed

    # Tensor parallel seed (different per TP rank, same across other dimensions)
    seeds["tensor_parallel"] = seed + tensor_parallel_seed_offset + tp_rank

    # Pipeline parallel seed (different per PP rank, same across other dimensions)
    seeds["pipeline_parallel"] = seed + pipeline_parallel_seed_offset + pp_rank

    # Data parallel seed (different per DP rank, same across other dimensions)
    seeds["data_parallel"] = seed + data_parallel_seed_offset + dp_rank

    # Context parallel seed (different per CP rank, same across other dimensions)
    seeds["context_parallel"] = seed + context_parallel_seed_offset + cp_rank

    # Expert parallel seed (different per EP rank, same across other dimensions)
    seeds["expert_parallel"] = seed + expert_parallel_seed_offset + ep_rank

    # Combined seeds for common use cases
    seeds["model_parallel"] = _combine_seeds(
        [seeds["tensor_parallel"], seeds["pipeline_parallel"]]
    )

    seeds["data_and_context"] = _combine_seeds(
        [seeds["data_parallel"], seeds["context_parallel"]]
    )

    # Comprehensive seed combining all dimensions
    seeds["all_parallel"] = _combine_seeds(
        [
            seeds["tensor_parallel"],
            seeds["pipeline_parallel"],
            seeds["data_parallel"],
            seeds["context_parallel"],
            seeds["expert_parallel"],
        ]
    )

    # Get RNG tracker
    rng_tracker = get_cuda_rng_tracker()

    # Create RNG states for each dimension
    dimension_configs = [
        ("global", seeds["global"], ["global"]),
        ("tensor_parallel", seeds["tensor_parallel"], ["tp"]),
        ("pipeline_parallel", seeds["pipeline_parallel"], ["pp"]),
        ("data_parallel", seeds["data_parallel"], ["dp"]),
        ("context_parallel", seeds["context_parallel"], ["cp"]),
        ("expert_parallel", seeds["expert_parallel"], ["ep"]),
        ("model_parallel", seeds["model_parallel"], ["tp", "pp"]),
        ("data_and_context", seeds["data_and_context"], ["dp", "cp"]),
        ("all_parallel", seeds["all_parallel"], ["tp", "pp", "dp", "cp", "ep"]),
    ]

    for state_name, state_seed, parallel_dims in dimension_configs:
        # Only create states for active dimensions
        if _is_dimension_active(parallel_dims):
            try:
                rng_tracker.add(
                    name=state_name,
                    parallel_dimensions=parallel_dims,
                    seed=state_seed,
                    # state_type will auto-detect CUDA/CPU
                    force=True,  # Allow overwriting existing states
                )
            except RuntimeError as e:
                if "CUDA" in str(e):
                    # Fallback to CPU RNG if CUDA not available
                    logger.warning(
                        f"CUDA not available, using CPU RNG for {state_name}"
                    )
                    torch.manual_seed(state_seed)
                    random.seed(state_seed)
                    np.random.seed(state_seed % (2**32))
                else:
                    raise

    # Set global Python/NumPy seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))  # NumPy requires 32-bit seed

    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Enable deterministic operations if requested
    if enable_deterministic:
        _enable_deterministic_operations()

    # Set default RNG state to global
    try:
        rng_tracker.set("global")
    except KeyError:
        # If global state doesn't exist, try to create it
        logger.warning(
            "Global RNG state not found, this may indicate CUDA fallback occurred"
        )
        # The tracker should have handled fallbacks in the loop above

    if verbose:
        logger.info(f"Initialized multi-parallel RNG with base seed {seed}")
        logger.info(f"Computed seeds: {seeds}")
        logger.info(
            f"Parallel configuration: TP={tp_size}, PP={pp_size}, DP={dp_size}, "
            f"CP={cp_size}, EP={ep_size}"
        )
        logger.info(
            f"Rank configuration: TP={tp_rank}, PP={pp_rank}, DP={dp_rank}, "
            f"CP={cp_rank}, EP={ep_rank}"
        )

    return seeds


def get_parallel_rng_state(
    parallel_dimension: str = "global", include_tracker_state: bool = True
) -> Dict[str, Any]:
    """
    Get the current RNG state for a specific parallel dimension.

    Args:
        parallel_dimension: Parallel dimension to query
        include_tracker_state: Include full tracker state in result

    Returns:
        Dictionary containing RNG state information

    Raises:
        ValueError: If parallel dimension is invalid
    """
    valid_dimensions = {
        "global",
        "tensor_parallel",
        "pipeline_parallel",
        "data_parallel",
        "context_parallel",
        "expert_parallel",
        "model_parallel",
        "data_and_context",
        "all_parallel",
    }

    if parallel_dimension not in valid_dimensions:
        raise ValueError(f"Invalid parallel dimension: {parallel_dimension}")

    rng_tracker = get_cuda_rng_tracker()

    result: Dict[str, Any] = {
        "parallel_dimension": parallel_dimension,
        "current_state_name": rng_tracker.get_current_state_name(parallel_dimension),
        "torch_seed": torch.initial_seed(),
        "numpy_state": np.random.get_state(),
        "python_state": random.getstate(),
    }

    # Add CUDA state if available
    if torch.cuda.is_available():
        result["cuda_state"] = torch.cuda.get_rng_state()
        result["cuda_device"] = torch.cuda.current_device()

    # Add tracker state if requested
    if include_tracker_state:
        result["tracker_statistics"] = rng_tracker.get_statistics()

    return result


def set_parallel_rng_state(
    parallel_dimension: str, state_name: Optional[str] = None
) -> None:
    """
    Set the RNG state for a specific parallel dimension.

    Args:
        parallel_dimension: Parallel dimension to set
        state_name: Specific state name to set (None for dimension default)

    Raises:
        ValueError: If parallel dimension is invalid
        KeyError: If state name doesn't exist
    """
    rng_tracker = get_cuda_rng_tracker()

    if state_name is None:
        # Use dimension as state name
        state_name = parallel_dimension

    try:
        rng_tracker.set(state_name)
    except KeyError:
        # Try to create default state if it doesn't exist
        logger.warning(f"RNG state '{state_name}' not found, using global state")
        try:
            rng_tracker.set("global")
        except KeyError:
            # Create global state if it doesn't exist
            rng_tracker.add("global", parallel_dimensions=["global"], seed=1234)
            rng_tracker.set("global")


def fork_parallel_rng_state(
    source_dimension: str = "global",
    new_name: Optional[str] = None,
    target_dimensions: Optional[Union[str, List[str]]] = None,
    offset: int = 0,
) -> str:
    """
    Fork an RNG state for use in a different parallel context.

    Args:
        source_dimension: Source parallel dimension to fork from
        new_name: Name for the forked state (auto-generated if None)
        target_dimensions: Target parallel dimensions for the fork
        offset: Random offset to apply for differentiation

    Returns:
        Name of the forked state

    Raises:
        KeyError: If source dimension state doesn't exist
    """
    rng_tracker = get_cuda_rng_tracker()

    # Auto-generate name if not provided
    if new_name is None:
        new_name = f"fork_{source_dimension}_{rng_tracker._fork_counter}"

    # Use target dimensions or inherit from source
    if target_dimensions is None:
        target_dimensions = [source_dimension]

    # Fork the state
    rng_tracker.fork(
        source_name=source_dimension,
        new_name=new_name,
        parallel_dimensions=target_dimensions,
        offset=offset,
    )

    return new_name


def checkpoint_parallel_rng_state() -> Dict[str, Any]:
    """
    Create a checkpoint of all parallel RNG states.

    Returns:
        Dictionary containing all RNG state information for checkpointing
    """
    rng_tracker = get_cuda_rng_tracker()

    checkpoint = {
        "tracker_state": rng_tracker.state_dict(),
        "torch_state": {
            "manual_seed": torch.initial_seed(),
        },
        "numpy_state": np.random.get_state(),
        "python_state": random.getstate(),
    }

    # Add CUDA states if available
    if torch.cuda.is_available():
        cuda_states: Dict[str, torch.Tensor] = {}
        for device_id in range(torch.cuda.device_count()):
            cuda_states[str(device_id)] = torch.cuda.get_rng_state(device_id)
        checkpoint["cuda_states"] = cuda_states

    return checkpoint


def restore_parallel_rng_state(checkpoint: Dict[str, Any]) -> None:
    """
    Restore parallel RNG states from a checkpoint.

    Args:
        checkpoint: Checkpoint dictionary from checkpoint_parallel_rng_state

    Raises:
        KeyError: If checkpoint format is invalid
    """
    rng_tracker = get_cuda_rng_tracker()

    # Restore tracker state
    if "tracker_state" in checkpoint:
        rng_tracker.load_state_dict(checkpoint["tracker_state"])

    # Restore NumPy state
    if "numpy_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_state"])

    # Restore Python random state
    if "python_state" in checkpoint:
        random.setstate(checkpoint["python_state"])

    # Restore CUDA states
    if "cuda_states" in checkpoint and torch.cuda.is_available():
        for device_id, state in checkpoint["cuda_states"].items():
            torch.cuda.set_rng_state(state, device=int(device_id))


def synchronize_parallel_rng_states(
    dimensions: Optional[List[str]] = None, source_rank: int = 0
) -> None:
    """
    Synchronize RNG states across ranks for specified dimensions.

    This function ensures all ranks have the same RNG state for the specified
    parallel dimensions, which is useful for ensuring deterministic behavior
    in certain contexts.

    Args:
        dimensions: Parallel dimensions to synchronize (None for all)
        source_rank: Rank to broadcast state from

    Raises:
        RuntimeError: If distributed is not initialized
    """
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("Distributed not initialized")

        if dimensions is None:
            dimensions = [
                "global",
                "tensor_parallel",
                "pipeline_parallel",
                "data_parallel",
                "context_parallel",
                "expert_parallel",
            ]

        rng_tracker = get_cuda_rng_tracker()

        for dimension in dimensions:
            current_name = rng_tracker.get_current_state_name(dimension)
            if current_name and current_name in rng_tracker._states:
                # Broadcast RNG state from source rank
                state_tensor = rng_tracker._states[current_name]
                dist.broadcast(state_tensor, src=source_rank)

                # Update local state
                rng_tracker._states[current_name] = state_tensor.clone()

        logger.info(f"Synchronized RNG states for dimensions: {dimensions}")

    except ImportError:
        logger.warning("torch.distributed not available, skipping RNG synchronization")


def get_rng_state_summary() -> Dict[str, Any]:
    """
    Get a summary of current RNG state configuration.

    Returns:
        Dictionary containing RNG state summary
    """
    rng_tracker = get_cuda_rng_tracker()

    summary = {
        "tracker_stats": rng_tracker.get_statistics(),
        "torch_initial_seed": torch.initial_seed(),
        "cuda_available": torch.cuda.is_available(),
        "deterministic_enabled": torch.are_deterministic_algorithms_enabled(),
    }

    if torch.cuda.is_available():
        summary["cuda_device_count"] = torch.cuda.device_count()
        summary["current_cuda_device"] = torch.cuda.current_device()

    # Add parallel state information if available
    if parallel_state.is_initialized():
        summary["parallel_config"] = {
            "tensor_parallel_size": parallel_state.get_tensor_model_parallel_size(),
            "pipeline_parallel_size": parallel_state.get_pipeline_model_parallel_size(),
            "data_parallel_size": parallel_state.get_data_parallel_size(),
            "context_parallel_size": parallel_state.get_context_parallel_size(),
            "expert_parallel_size": parallel_state.get_expert_model_parallel_size(),
        }

        summary["parallel_ranks"] = {
            "tensor_parallel_rank": parallel_state.get_tensor_model_parallel_rank(),
            "pipeline_parallel_rank": parallel_state.get_pipeline_model_parallel_rank(),
            "data_parallel_rank": parallel_state.get_data_parallel_rank(),
            "context_parallel_rank": parallel_state.get_context_parallel_rank(),
            "expert_parallel_rank": parallel_state.get_expert_model_parallel_rank(),
        }

    return summary


def _combine_seeds(seeds: List[int]) -> int:
    """
    Combine multiple seeds into a single deterministic seed.

    Args:
        seeds: List of seeds to combine

    Returns:
        Combined seed as integer
    """
    # Create deterministic hash of all seeds
    combined_str = "_".join(str(s) for s in sorted(seeds))
    hash_obj = hashlib.md5(combined_str.encode())

    # Convert to integer (use first 8 bytes for 64-bit int)
    hash_bytes = hash_obj.digest()[:8]
    combined_seed = int.from_bytes(hash_bytes, byteorder="big", signed=False)

    # Ensure positive 32-bit integer for compatibility
    return combined_seed % (2**31)


def _is_dimension_active(parallel_dims: List[str]) -> bool:
    """
    Check if any of the specified parallel dimensions are active.

    Args:
        parallel_dims: List of parallel dimension names

    Returns:
        True if any dimension is active (size > 1)
    """
    if not parallel_state.is_initialized():
        return "global" in parallel_dims

    size_map = {
        "tp": parallel_state.get_tensor_model_parallel_size(),
        "pp": parallel_state.get_pipeline_model_parallel_size(),
        "dp": parallel_state.get_data_parallel_size(),
        "cp": parallel_state.get_context_parallel_size(),
        "ep": parallel_state.get_expert_model_parallel_size(),
        "global": 1,  # Always active
    }

    return any(size_map.get(dim, 1) > 1 or dim == "global" for dim in parallel_dims)


class RNGConfigurationFactory:
    """Factory for creating RNG configurations for different use cases."""

    @staticmethod
    def create_default_config(
        seed: int, enable_deterministic: bool = True
    ) -> Dict[str, Any]:
        """Create default RNG configuration."""
        return {
            "seed": seed,
            "tensor_parallel_seed_offset": _DEFAULT_TENSOR_PARALLEL_OFFSET,
            "pipeline_parallel_seed_offset": _DEFAULT_PIPELINE_PARALLEL_OFFSET,
            "data_parallel_seed_offset": _DEFAULT_DATA_PARALLEL_OFFSET,
            "context_parallel_seed_offset": _DEFAULT_CONTEXT_PARALLEL_OFFSET,
            "expert_parallel_seed_offset": _DEFAULT_EXPERT_PARALLEL_OFFSET,
            "enable_deterministic": enable_deterministic,
            "verbose": False,
        }

    @staticmethod
    def create_training_config(
        seed: int, enable_deterministic: bool = True
    ) -> Dict[str, Any]:
        """Create RNG configuration optimized for training reproducibility."""
        return {
            "seed": seed,
            "tensor_parallel_seed_offset": 1000,
            "pipeline_parallel_seed_offset": 2000,
            "data_parallel_seed_offset": 3000,
            "context_parallel_seed_offset": 4000,
            "expert_parallel_seed_offset": 5000,
            "enable_deterministic": enable_deterministic,
            "verbose": True,
        }

    @staticmethod
    def create_inference_config(seed: int) -> Dict[str, Any]:
        """Create RNG configuration optimized for inference."""
        return {
            "seed": seed,
            "tensor_parallel_seed_offset": 100,
            "pipeline_parallel_seed_offset": 200,
            "data_parallel_seed_offset": 300,
            "context_parallel_seed_offset": 400,
            "expert_parallel_seed_offset": 500,
            "enable_deterministic": False,  # Less strict for inference
            "verbose": False,
        }


def setup_model_parallel_rng(config_type: str = "default", **kwargs) -> Dict[str, int]:
    """
    Convenient function to set up model parallel RNG with predefined configurations.

    Args:
        config_type: Type of configuration ("default", "training", "inference")
        **kwargs: Additional arguments to override configuration

    Returns:
        Dictionary mapping dimension names to their computed seeds

    Raises:
        ValueError: If config_type is invalid
    """
    factory = RNGConfigurationFactory()

    config_methods: Dict[str, Any] = {
        "default": factory.create_default_config,
        "training": factory.create_training_config,
        "inference": factory.create_inference_config,
    }

    if config_type not in config_methods:
        raise ValueError(
            f"Invalid config_type: {config_type}. "
            f"Must be one of: {list(config_methods.keys())}"
        )

    # Get base config and override with kwargs
    config_func = config_methods[config_type]
    # Extract seed if provided, otherwise use default
    seed = kwargs.pop("seed", 42)
    # Call the config function with seed
    if config_type == "inference":
        config = config_func(seed)
    else:
        enable_deterministic = kwargs.pop("enable_deterministic", True)
        config = config_func(seed, enable_deterministic)
    # Update with any remaining kwargs
    config.update(kwargs)

    return model_parallel_cuda_manual_seed(**config)


def _enable_deterministic_operations() -> None:
    """Enable deterministic operations for reproducible training."""
    try:
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True)

        # Set environment variables for deterministic behavior
        import os

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Disable benchmarking for deterministic behavior
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        logger.info("Enabled deterministic operations for reproducible training")

    except Exception as e:
        logger.warning(f"Failed to enable some deterministic operations: {e}")


# Context managers for temporary RNG state changes
class parallel_rng_context:
    """Context manager for temporarily changing parallel RNG state."""

    def __init__(
        self,
        parallel_dimension: str,
        state_name: Optional[str] = None,
        fork_if_needed: bool = True,
    ):
        """
        Initialize the context manager.

        Args:
            parallel_dimension: Parallel dimension to use
            state_name: Specific state name (None for dimension default)
            fork_if_needed: Fork current state if target doesn't exist
        """
        self.parallel_dimension = parallel_dimension
        self.state_name = state_name or parallel_dimension
        self.fork_if_needed = fork_if_needed
        self.previous_state = None
        self.forked_state = None

    def __enter__(self):
        """Enter the context and set the RNG state."""
        rng_tracker = get_cuda_rng_tracker()

        # Save current state
        self.previous_state = rng_tracker.get_current_state_name(
            self.parallel_dimension
        )

        # Try to set the target state
        try:
            rng_tracker.set(self.state_name)
        except KeyError:
            if self.fork_if_needed and self.previous_state:
                # Fork current state for temporary use
                self.forked_state = f"temp_{self.state_name}_{id(self)}"
                rng_tracker.fork(
                    source_name=self.previous_state,
                    new_name=self.forked_state,
                    parallel_dimensions=[self.parallel_dimension],
                )
                rng_tracker.set(self.forked_state)
            else:
                # Fall back to global state
                rng_tracker.set("global")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore previous RNG state."""
        rng_tracker = get_cuda_rng_tracker()

        # Clean up forked state if created
        if self.forked_state:
            rng_tracker.remove(self.forked_state)

        # Restore previous state
        if self.previous_state:
            rng_tracker.set(self.previous_state)
