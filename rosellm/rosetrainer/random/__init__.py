"""
Advanced Multi-Parallel RNG State Management for RoseLLM

This package provides comprehensive random number generation utilities for
multi-dimensional parallel training. It ensures deterministic and reproducible
training across different parallelism configurations while maintaining proper
RNG independence between parallel dimensions.

Main Features:
- Multi-dimensional parallel RNG state management (TP, PP, DP, CP, EP)
- Megatron-LM compatible interface and behavior
- Advanced state forking and isolation
- Checkpoint/restore capabilities
- CUDA Graph compatibility
- Performance optimization with caching

Key Classes:
- CudaRNGStatesTracker: Core RNG state management
- RNGStateInfo: Metadata for RNG states
- parallel_rng_context: Context manager for temporary state changes

Key Functions:
- model_parallel_cuda_manual_seed: Initialize multi-parallel RNG states
- fork_parallel_rng_state: Create independent RNG forks
- checkpoint_parallel_rng_state: Save RNG states for checkpointing
- restore_parallel_rng_state: Restore RNG states from checkpoint

Usage Example:
    ```python
    from rosellm.rosetrainer.random import (
        model_parallel_cuda_manual_seed,
        parallel_rng_context,
        get_cuda_rng_tracker
    )

    # Initialize multi-parallel RNG
    seeds = model_parallel_cuda_manual_seed(1234, verbose=True)

    # Use specific parallel dimension RNG
    with parallel_rng_context("tensor_parallel"):
        # Operations use tensor-parallel RNG state
        tensor = torch.randn(1024, 1024)

    # Get tracker for advanced operations
    tracker = get_cuda_rng_tracker()
    tracker.fork("global", "custom_state", offset=100)
    ```

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- PyTorch RNG: https://pytorch.org/docs/stable/notes/randomness.html
"""

from .parallel_rng import (
    checkpoint_parallel_rng_state,
    fork_parallel_rng_state,
    get_parallel_rng_state,
    get_rng_state_summary,
    model_parallel_cuda_manual_seed,
    parallel_rng_context,
    restore_parallel_rng_state,
    set_parallel_rng_state,
    synchronize_parallel_rng_states,
)
from .rng_tracker import (
    CudaRNGStatesTracker,
    RNGStateInfo,
    RNGStateType,
    get_cuda_rng_tracker,
    initialize_cuda_rng_tracker,
    reset_cuda_rng_tracker,
)

__all__ = [
    # Core tracker classes and functions
    "CudaRNGStatesTracker",
    "RNGStateInfo",
    "RNGStateType",
    "get_cuda_rng_tracker",
    "reset_cuda_rng_tracker",
    "initialize_cuda_rng_tracker",
    # Parallel RNG management functions
    "model_parallel_cuda_manual_seed",
    "get_parallel_rng_state",
    "set_parallel_rng_state",
    "fork_parallel_rng_state",
    "checkpoint_parallel_rng_state",
    "restore_parallel_rng_state",
    "synchronize_parallel_rng_states",
    "get_rng_state_summary",
    # Context managers
    "parallel_rng_context",
]

# Version info
__version__ = "1.0.0"
__author__ = "RoseLLM Team"
__description__ = "Advanced Multi-Parallel RNG State Management"
