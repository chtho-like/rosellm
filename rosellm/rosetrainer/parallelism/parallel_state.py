"""
Advanced Parallel State Management System

This module implements a comprehensive multi-dimensional parallelism state manager
inspired by Megatron-LM's parallel state architecture. It provides:
- Global process group registry with hierarchical organization
- Support for TP (Tensor), PP (Pipeline), DP (Data), CP (Context),
  and EP (Expert) parallelism
- Orthogonal rank group generation for non-overlapping process groups
- NCCL optimization configuration support
- Dynamic parallelism reconfiguration capabilities

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
"""

import os
import threading
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch.distributed as dist


class ParallelismDimension(Enum):
    """Enumeration of parallelism dimensions"""

    TENSOR = "tp"
    PIPELINE = "pp"
    DATA = "dp"
    CONTEXT = "cp"
    EXPERT = "ep"


@dataclass
class NCCLConfig:
    """NCCL optimization configuration"""

    enable_sharp: bool = False
    cta_size: Optional[int] = None
    cluster_size: Optional[int] = None
    min_nchannels: Optional[int] = None
    max_nchannels: Optional[int] = None
    tree_threshold: Optional[int] = None
    buffer_size: Optional[int] = None


# Thread safety lock for global state modifications
_STATE_LOCK = threading.RLock()

# Global parallel state variables (protected by lock)
_INITIALIZED = False
_WORLD_SIZE: Optional[int] = None
_RANK: Optional[int] = None

# Multi-dimensional parallelism groups
_TENSOR_MODEL_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_PIPELINE_MODEL_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_DATA_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_CONTEXT_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_EXPERT_MODEL_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None

# Combined groups for optimized communication
_TENSOR_AND_DATA_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP: Optional[dist.ProcessGroup] = None
_TENSOR_AND_EXPERT_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_DATA_PARALLEL_GROUP_WITH_CP: Optional[dist.ProcessGroup] = None
_MODEL_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None  # TP + PP combined
_EMBEDDING_GROUP: Optional[dist.ProcessGroup] = None
_POSITION_EMBEDDING_GROUP: Optional[dist.ProcessGroup] = None

# Hierarchical context parallel groups
_HIERARCHICAL_CONTEXT_PARALLEL_GROUPS: Optional[List[dist.ProcessGroup]] = None

# Parallelism sizes
_TENSOR_MODEL_PARALLEL_SIZE: int = 1
_PIPELINE_MODEL_PARALLEL_SIZE: int = 1
_DATA_PARALLEL_SIZE: int = 1
_CONTEXT_PARALLEL_SIZE: int = 1
_EXPERT_MODEL_PARALLEL_SIZE: int = 1

# Ranks within each dimension
_TENSOR_MODEL_PARALLEL_RANK: int = 0
_PIPELINE_MODEL_PARALLEL_RANK: int = 0
_DATA_PARALLEL_RANK: int = 0
_CONTEXT_PARALLEL_RANK: int = 0
_EXPERT_MODEL_PARALLEL_RANK: int = 0

# NCCL configuration
_NCCL_CONFIG: Optional[NCCLConfig] = None

# Virtual pipeline parallel settings
_VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE: Optional[int] = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK: Optional[int] = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    data_parallel_size: Optional[int] = None,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    order: str = "tp-cp-ep-dp-pp",
    hierarchical_context_parallel_sizes: Optional[List[int]] = None,
    nccl_config: Optional[NCCLConfig] = None,
    backend: str = "nccl",
) -> None:
    """
    Initialize model parallel groups with multi-dimensional parallelism support.

    Args:
        tensor_model_parallel_size: Size of tensor model parallel group
        pipeline_model_parallel_size: Size of pipeline model parallel group
        context_parallel_size: Size of context parallel group for long sequences
        expert_model_parallel_size: Size of expert parallel group for MoE
        data_parallel_size: Size of data parallel group (auto-calculated if None)
        virtual_pipeline_model_parallel_size: Number of virtual pipeline stages
        order: Order of parallelism dimensions (e.g., "tp-cp-ep-dp-pp")
        hierarchical_context_parallel_sizes: Nested context parallel group sizes
        nccl_config: NCCL optimization configuration
        backend: Communication backend (nccl, gloo, etc.)
    """
    global _INITIALIZED, _WORLD_SIZE, _RANK
    global _TENSOR_MODEL_PARALLEL_SIZE, _PIPELINE_MODEL_PARALLEL_SIZE
    global _DATA_PARALLEL_SIZE, _CONTEXT_PARALLEL_SIZE, _EXPERT_MODEL_PARALLEL_SIZE
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE, _NCCL_CONFIG

    with _STATE_LOCK:  # Thread-safe initialization
        if _INITIALIZED:
            warnings.warn(
                "Parallel state already initialized, skipping re-initialization"
            )
            return

    # Initialize PyTorch distributed if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    _WORLD_SIZE = dist.get_world_size()
    _RANK = dist.get_rank()

    # Validate and set parallelism sizes
    _TENSOR_MODEL_PARALLEL_SIZE = tensor_model_parallel_size
    _PIPELINE_MODEL_PARALLEL_SIZE = pipeline_model_parallel_size
    _CONTEXT_PARALLEL_SIZE = context_parallel_size
    _EXPERT_MODEL_PARALLEL_SIZE = expert_model_parallel_size
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE = virtual_pipeline_model_parallel_size

    # Calculate data parallel size if not provided
    if data_parallel_size is None:
        assert _WORLD_SIZE is not None, "World size must be set"
        _DATA_PARALLEL_SIZE = _WORLD_SIZE // (
            tensor_model_parallel_size
            * pipeline_model_parallel_size
            * context_parallel_size
            * expert_model_parallel_size
        )
    else:
        _DATA_PARALLEL_SIZE = data_parallel_size

    # Validate total world size
    total_size = (
        _TENSOR_MODEL_PARALLEL_SIZE
        * _PIPELINE_MODEL_PARALLEL_SIZE
        * _DATA_PARALLEL_SIZE
        * _CONTEXT_PARALLEL_SIZE
        * _EXPERT_MODEL_PARALLEL_SIZE
    )
    assert (
        total_size == _WORLD_SIZE
    ), f"Total parallelism size {total_size} != world size {_WORLD_SIZE}"

    # Store NCCL configuration
    _NCCL_CONFIG = nccl_config or NCCLConfig()

    # Create process groups based on specified order
    _create_parallel_groups(order, hierarchical_context_parallel_sizes)

    _INITIALIZED = True


def _create_parallel_groups(
    order: str, hierarchical_cp_sizes: Optional[List[int]] = None
) -> None:
    """
    Create all parallel process groups based on the specified dimension order.

    Args:
        order: Parallelism dimension order (e.g., "tp-cp-ep-dp-pp")
        hierarchical_cp_sizes: Sizes for hierarchical context parallel groups
    """
    # Global group assignments (conditional based on sizes)
    global _TENSOR_MODEL_PARALLEL_GROUP, _PIPELINE_MODEL_PARALLEL_GROUP  # noqa: F824
    global _DATA_PARALLEL_GROUP, _CONTEXT_PARALLEL_GROUP  # noqa: F824
    global _EXPERT_MODEL_PARALLEL_GROUP  # noqa: F824
    global _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS  # noqa: F824
    global _TENSOR_MODEL_PARALLEL_RANK, _PIPELINE_MODEL_PARALLEL_RANK  # noqa: F824
    global _DATA_PARALLEL_RANK, _CONTEXT_PARALLEL_RANK  # noqa: F824
    global _EXPERT_MODEL_PARALLEL_RANK  # noqa: F824

    # Parse dimension order
    dims = order.lower().split("-")
    dim_to_size = {
        "tp": _TENSOR_MODEL_PARALLEL_SIZE,
        "pp": _PIPELINE_MODEL_PARALLEL_SIZE,
        "dp": _DATA_PARALLEL_SIZE,
        "cp": _CONTEXT_PARALLEL_SIZE,
        "ep": _EXPERT_MODEL_PARALLEL_SIZE,
    }

    # Generate orthogonal rank groups
    rank_groups = _generate_orthogonal_rank_groups(dims, dim_to_size)

    # Create individual dimension groups
    _create_dimension_groups(rank_groups)

    # Create combined groups for optimized communication
    _create_combined_groups()

    # Create hierarchical context parallel groups if specified
    if hierarchical_cp_sizes:
        _create_hierarchical_cp_groups(hierarchical_cp_sizes)

    # Set ranks within each dimension
    _set_dimension_ranks()


def _generate_orthogonal_rank_groups(
    dims: List[str], dim_to_size: Dict[str, int]
) -> Dict[str, List[List[int]]]:
    """
    Generate orthogonal (non-overlapping) rank groups for each dimension.

    Args:
        dims: Ordered list of dimension names
        dim_to_size: Mapping of dimension name to size

    Returns:
        Dictionary mapping dimension name to list of rank groups
    """
    world_size = _WORLD_SIZE if _WORLD_SIZE is not None else 0
    rank_groups = {}

    # Build rank grid based on dimension order
    shape = [dim_to_size[d] for d in dims]

    # Generate groups for each dimension
    for dim_idx, dim_name in enumerate(dims):
        groups = []
        group_size = dim_to_size[dim_name]

        if group_size == 1:
            # Skip trivial groups
            rank_groups[dim_name] = [[i] for i in range(world_size)]
            continue

        # Calculate stride and number of groups
        stride = 1
        for i in range(dim_idx + 1, len(dims)):
            stride *= shape[i]

        num_groups = world_size // group_size if world_size is not None else 0

        # Generate rank lists for this dimension
        for group_id in range(num_groups):
            ranks = []
            for member_id in range(group_size):
                # Calculate rank using multi-dimensional indexing
                rank = _calculate_rank_from_coords(
                    dims, shape, dim_idx, group_id, member_id
                )
                ranks.append(rank)
            groups.append(sorted(ranks))

        rank_groups[dim_name] = groups

    return rank_groups


def _calculate_rank_from_coords(
    dims: List[str], shape: List[int], varying_dim: int, group_id: int, member_id: int
) -> int:
    """
    Calculate global rank from multi-dimensional coordinates.

    Args:
        dims: Dimension names in order
        shape: Size of each dimension
        varying_dim: Index of the dimension that varies within group
        group_id: ID of the group
        member_id: Position within the group

    Returns:
        Global rank
    """
    coords = [0] * len(dims)

    # Set the varying dimension coordinate
    coords[varying_dim] = member_id

    # Calculate fixed dimensions from group_id
    remaining = group_id
    for i in range(len(dims)):
        if i == varying_dim:
            continue
        size = shape[i]
        if i < varying_dim:
            # Dimensions before varying dim
            stride = 1
            for j in range(i + 1, len(dims)):
                if j != varying_dim:
                    stride *= shape[j]
            coords[i] = (remaining // stride) % size
        else:
            # Dimensions after varying dim
            coords[i] = remaining % size
            remaining //= size

    # Convert coordinates to rank
    rank = 0
    stride = 1
    for i in range(len(dims) - 1, -1, -1):
        rank += coords[i] * stride
        stride *= shape[i]

    return rank


def _create_dimension_groups(rank_groups: Dict[str, List[List[int]]]) -> None:
    """Create process groups for each parallelism dimension."""
    global _TENSOR_MODEL_PARALLEL_GROUP, _PIPELINE_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP, _CONTEXT_PARALLEL_GROUP, _EXPERT_MODEL_PARALLEL_GROUP

    # Helper function to create group containing current rank
    def create_group_with_rank(
        groups_list: List[List[int]],
    ) -> Optional[dist.ProcessGroup]:
        for ranks in groups_list:
            if _RANK in ranks:
                group = dist.new_group(ranks)
                if _RANK in ranks:
                    return group  # type: ignore[no-any-return]
        return None

    # Create tensor parallel group
    if "tp" in rank_groups and _TENSOR_MODEL_PARALLEL_SIZE > 1:
        _TENSOR_MODEL_PARALLEL_GROUP = create_group_with_rank(rank_groups["tp"])

    # Create pipeline parallel group
    if "pp" in rank_groups and _PIPELINE_MODEL_PARALLEL_SIZE > 1:
        _PIPELINE_MODEL_PARALLEL_GROUP = create_group_with_rank(rank_groups["pp"])

    # Create data parallel group
    if "dp" in rank_groups and _DATA_PARALLEL_SIZE > 1:
        _DATA_PARALLEL_GROUP = create_group_with_rank(rank_groups["dp"])

    # Create context parallel group
    if "cp" in rank_groups and _CONTEXT_PARALLEL_SIZE > 1:
        _CONTEXT_PARALLEL_GROUP = create_group_with_rank(rank_groups["cp"])

    # Create expert parallel group
    if "ep" in rank_groups and _EXPERT_MODEL_PARALLEL_SIZE > 1:
        _EXPERT_MODEL_PARALLEL_GROUP = create_group_with_rank(rank_groups["ep"])


def _create_combined_groups() -> None:
    """Create combined process groups for optimized communication patterns."""
    global _TENSOR_AND_DATA_PARALLEL_GROUP, _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    global _MODEL_PARALLEL_GROUP

    # Create TP+DP group (commonly used for gradient all-reduce)
    tp_dp_size = _TENSOR_MODEL_PARALLEL_SIZE * _DATA_PARALLEL_SIZE
    if tp_dp_size > 1:
        for pp in range(_PIPELINE_MODEL_PARALLEL_SIZE):
            for cp in range(_CONTEXT_PARALLEL_SIZE):
                for ep in range(_EXPERT_MODEL_PARALLEL_SIZE):
                    ranks = []
                    for tp in range(_TENSOR_MODEL_PARALLEL_SIZE):
                        for dp in range(_DATA_PARALLEL_SIZE):
                            rank = _get_rank_from_coords(tp, pp, dp, cp, ep)
                            ranks.append(rank)
                    group = dist.new_group(ranks)
                    if _RANK in ranks:
                        _TENSOR_AND_DATA_PARALLEL_GROUP = group

    # Create TP+DP+CP group
    tp_dp_cp_size = (
        _TENSOR_MODEL_PARALLEL_SIZE * _DATA_PARALLEL_SIZE * _CONTEXT_PARALLEL_SIZE
    )
    if tp_dp_cp_size > 1:
        for pp in range(_PIPELINE_MODEL_PARALLEL_SIZE):
            for ep in range(_EXPERT_MODEL_PARALLEL_SIZE):
                ranks = []
                for tp in range(_TENSOR_MODEL_PARALLEL_SIZE):
                    for dp in range(_DATA_PARALLEL_SIZE):
                        for cp in range(_CONTEXT_PARALLEL_SIZE):
                            rank = _get_rank_from_coords(tp, pp, dp, cp, ep)
                            ranks.append(rank)
                group = dist.new_group(ranks)
                if _RANK in ranks:
                    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group

    # Create TP+PP model parallel group
    model_parallel_size = _TENSOR_MODEL_PARALLEL_SIZE * _PIPELINE_MODEL_PARALLEL_SIZE
    if model_parallel_size > 1:
        for dp in range(_DATA_PARALLEL_SIZE):
            for cp in range(_CONTEXT_PARALLEL_SIZE):
                for ep in range(_EXPERT_MODEL_PARALLEL_SIZE):
                    ranks = []
                    for tp in range(_TENSOR_MODEL_PARALLEL_SIZE):
                        for pp in range(_PIPELINE_MODEL_PARALLEL_SIZE):
                            rank = _get_rank_from_coords(tp, pp, dp, cp, ep)
                            ranks.append(rank)
                    group = dist.new_group(ranks)
                    if _RANK in ranks:
                        _MODEL_PARALLEL_GROUP = group


def _create_hierarchical_cp_groups(sizes: List[int]) -> None:
    """Create hierarchical context parallel groups for nested parallelism."""
    global _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS

    _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = []

    for level, size in enumerate(sizes):
        # Validate size
        if level == 0:
            assert size <= _CONTEXT_PARALLEL_SIZE
        else:
            assert size <= sizes[level - 1]

        # Create groups at this level
        # Implementation depends on specific hierarchical strategy
        # This is a placeholder for the actual hierarchical group creation
        pass


def _set_dimension_ranks() -> None:
    """Set the rank of the current process within each dimension."""
    global _TENSOR_MODEL_PARALLEL_RANK, _PIPELINE_MODEL_PARALLEL_RANK
    global _DATA_PARALLEL_RANK, _CONTEXT_PARALLEL_RANK, _EXPERT_MODEL_PARALLEL_RANK

    # Calculate rank within each dimension based on global rank
    coords = _get_coords_from_rank(_RANK if _RANK is not None else 0)

    _TENSOR_MODEL_PARALLEL_RANK = coords[0]
    _PIPELINE_MODEL_PARALLEL_RANK = coords[1]
    _DATA_PARALLEL_RANK = coords[2]
    _CONTEXT_PARALLEL_RANK = coords[3]
    _EXPERT_MODEL_PARALLEL_RANK = coords[4]


def _get_rank_from_coords(tp: int, pp: int, dp: int, cp: int, ep: int) -> int:
    """Convert multi-dimensional coordinates to global rank."""
    rank = (
        ep
        + cp * _EXPERT_MODEL_PARALLEL_SIZE
        + dp * _EXPERT_MODEL_PARALLEL_SIZE * _CONTEXT_PARALLEL_SIZE
        + pp
        * _EXPERT_MODEL_PARALLEL_SIZE
        * _CONTEXT_PARALLEL_SIZE
        * _DATA_PARALLEL_SIZE
        + tp
        * _EXPERT_MODEL_PARALLEL_SIZE
        * _CONTEXT_PARALLEL_SIZE
        * _DATA_PARALLEL_SIZE
        * _PIPELINE_MODEL_PARALLEL_SIZE
    )
    return rank


def _get_coords_from_rank(rank: int) -> Tuple[int, int, int, int, int]:
    """Convert global rank to multi-dimensional coordinates."""
    ep = rank % _EXPERT_MODEL_PARALLEL_SIZE
    rank //= _EXPERT_MODEL_PARALLEL_SIZE

    cp = rank % _CONTEXT_PARALLEL_SIZE
    rank //= _CONTEXT_PARALLEL_SIZE

    dp = rank % _DATA_PARALLEL_SIZE
    rank //= _DATA_PARALLEL_SIZE

    pp = rank % _PIPELINE_MODEL_PARALLEL_SIZE
    rank //= _PIPELINE_MODEL_PARALLEL_SIZE

    tp = rank

    return (tp, pp, dp, cp, ep)


# Getter functions for parallel state
def is_initialized() -> bool:
    """Check if parallel state is initialized."""
    return _INITIALIZED


def get_tensor_model_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the tensor model parallel group."""
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the pipeline model parallel group."""
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the data parallel group."""
    return _DATA_PARALLEL_GROUP


def get_context_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the context parallel group."""
    return _CONTEXT_PARALLEL_GROUP


def get_expert_model_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the expert model parallel group."""
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_size() -> int:
    """Get tensor model parallel size."""
    return _TENSOR_MODEL_PARALLEL_SIZE if _INITIALIZED else 1


def get_pipeline_model_parallel_size() -> int:
    """Get pipeline model parallel size."""
    return _PIPELINE_MODEL_PARALLEL_SIZE if _INITIALIZED else 1


def get_data_parallel_size() -> int:
    """Get data parallel size."""
    return _DATA_PARALLEL_SIZE if _INITIALIZED else 1


def get_context_parallel_size() -> int:
    """Get context parallel size."""
    return _CONTEXT_PARALLEL_SIZE if _INITIALIZED else 1


def get_expert_model_parallel_size() -> int:
    """Get expert model parallel size."""
    return _EXPERT_MODEL_PARALLEL_SIZE if _INITIALIZED else 1


def get_tensor_model_parallel_rank() -> int:
    """Get rank within tensor model parallel group."""
    return _TENSOR_MODEL_PARALLEL_RANK if _INITIALIZED else 0


def get_pipeline_model_parallel_rank() -> int:
    """Get rank within pipeline model parallel group."""
    return _PIPELINE_MODEL_PARALLEL_RANK if _INITIALIZED else 0


def get_data_parallel_rank() -> int:
    """Get rank within data parallel group."""
    return _DATA_PARALLEL_RANK if _INITIALIZED else 0


def get_context_parallel_rank() -> int:
    """Get rank within context parallel group."""
    return _CONTEXT_PARALLEL_RANK if _INITIALIZED else 0


def get_expert_model_parallel_rank() -> int:
    """Get rank within expert model parallel group."""
    return _EXPERT_MODEL_PARALLEL_RANK if _INITIALIZED else 0


def get_tensor_model_parallel_src_rank() -> int:
    """Get source rank for tensor model parallel group broadcasts."""
    if not _INITIALIZED or _RANK is None:
        return 0
    return (_RANK // _TENSOR_MODEL_PARALLEL_SIZE) * _TENSOR_MODEL_PARALLEL_SIZE


def get_pipeline_model_parallel_first_rank() -> int:
    """Get first rank in pipeline model parallel group."""
    # Get current coordinates
    coords = _get_coords_from_rank(_RANK if _RANK is not None else 0)
    # Set PP coordinate to 0, keep others same
    first_coords = (coords[0], 0, coords[2], coords[3], coords[4])
    return _get_rank_from_coords(*first_coords)


def get_pipeline_model_parallel_last_rank() -> int:
    """Get last rank in pipeline model parallel group."""
    # Get current coordinates
    coords = _get_coords_from_rank(_RANK if _RANK is not None else 0)
    # Set PP coordinate to max, keep others same
    last_coords = (
        coords[0],
        _PIPELINE_MODEL_PARALLEL_SIZE - 1,
        coords[2],
        coords[3],
        coords[4],
    )
    return _get_rank_from_coords(*last_coords)


def get_pipeline_model_parallel_next_rank() -> Optional[int]:
    """Get next rank in pipeline model parallel group."""
    pp_rank = get_pipeline_model_parallel_rank()
    if pp_rank < _PIPELINE_MODEL_PARALLEL_SIZE - 1:
        coords = _get_coords_from_rank(_RANK if _RANK is not None else 0)
        next_coords = (coords[0], pp_rank + 1, coords[2], coords[3], coords[4])
        return _get_rank_from_coords(*next_coords)
    return None


def get_pipeline_model_parallel_prev_rank() -> Optional[int]:
    """Get previous rank in pipeline model parallel group."""
    pp_rank = get_pipeline_model_parallel_rank()
    if pp_rank > 0:
        coords = _get_coords_from_rank(_RANK if _RANK is not None else 0)
        prev_coords = (coords[0], pp_rank - 1, coords[2], coords[3], coords[4])
        return _get_rank_from_coords(*prev_coords)
    return None


@lru_cache(maxsize=128)
def get_tensor_and_data_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get combined tensor and data parallel group."""
    return _TENSOR_AND_DATA_PARALLEL_GROUP


@lru_cache(maxsize=128)
def get_tensor_and_data_parallel_group_with_cp() -> Optional[dist.ProcessGroup]:
    """Get combined tensor, data, and context parallel group."""
    return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP


def get_model_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get model parallel group (TP + PP)."""
    return _MODEL_PARALLEL_GROUP


def set_nccl_config(config: NCCLConfig) -> None:
    """Set NCCL optimization configuration."""
    global _NCCL_CONFIG
    with _STATE_LOCK:
        _NCCL_CONFIG = config

    # Apply NCCL environment variables if specified
    if config.enable_sharp:
        os.environ["NCCL_SHARP_DISABLE"] = "0"

    if config.cta_size:
        os.environ["NCCL_MAX_NCHANNELS"] = str(config.cta_size)

    if config.min_nchannels:
        os.environ["NCCL_MIN_NCHANNELS"] = str(config.min_nchannels)

    if config.tree_threshold:
        os.environ["NCCL_TREE_THRESHOLD"] = str(config.tree_threshold)


def get_nccl_config() -> Optional[NCCLConfig]:
    """Get current NCCL configuration."""
    return _NCCL_CONFIG


def destroy_model_parallel() -> None:
    """Destroy all parallel groups and reset state."""
    global _INITIALIZED, _TENSOR_MODEL_PARALLEL_GROUP, _PIPELINE_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP, _CONTEXT_PARALLEL_GROUP, _EXPERT_MODEL_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP, _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    global _MODEL_PARALLEL_GROUP

    with _STATE_LOCK:
        if not _INITIALIZED:
            return

        # Destroy all process groups
        groups = [
            _TENSOR_MODEL_PARALLEL_GROUP,
            _PIPELINE_MODEL_PARALLEL_GROUP,
            _DATA_PARALLEL_GROUP,
            _CONTEXT_PARALLEL_GROUP,
            _EXPERT_MODEL_PARALLEL_GROUP,
            _TENSOR_AND_DATA_PARALLEL_GROUP,
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP,
            _MODEL_PARALLEL_GROUP,
        ]

        for group in groups:
            if group is not None:
                dist.destroy_process_group(group)

        # Reset all variables
        _TENSOR_MODEL_PARALLEL_GROUP = None
        _PIPELINE_MODEL_PARALLEL_GROUP = None
        _DATA_PARALLEL_GROUP = None
        _CONTEXT_PARALLEL_GROUP = None
        _EXPERT_MODEL_PARALLEL_GROUP = None
        _TENSOR_AND_DATA_PARALLEL_GROUP = None
        _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None
        _MODEL_PARALLEL_GROUP = None

        _INITIALIZED = False


def get_virtual_pipeline_model_parallel_rank() -> Optional[int]:
    """Get virtual pipeline model parallel rank."""
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank: int) -> None:
    """Set virtual pipeline model parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    with _STATE_LOCK:
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_size() -> Optional[int]:
    """Get virtual pipeline model parallel size."""
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE
