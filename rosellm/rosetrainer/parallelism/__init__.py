"""
Parallelism Module for RoseTrainer

This module provides various parallelism strategies for distributed training:
- Data Parallelism (DP)
- Tensor/Model Parallelism (TP)
- Pipeline Parallelism (PP)
- Context Parallelism (CP)
- Expert Parallelism (EP)
- ZeRO Optimizer
- Advanced Parallel State Management
- Parameter Gathering Overlap with Computation
"""

from .data_parallel import DataParallelTrainer
from .microbatch_calculator import (
    AdaptiveMicrobatchCalculator,
    ConstantNumMicrobatches,
    MicrobatchCalculatorBase,
    RampupBatchSizeNumMicrobatches,
    calculate_optimal_microbatch_size,
    destroy_microbatch_calculator,
    get_micro_batch_size,
    get_microbatch_calculator,
    get_microbatch_schedule,
    get_num_microbatches,
    initialize_microbatch_calculator,
    update_microbatch_calculator,
)
from .model_parallel import ColumnParallelLinear, RowParallelLinear, TensorParallelism
from .overlap_integration import (
    OverlappedColumnParallelLinear,
    OverlappedPipelineEngine,
    OverlappedRowParallelLinear,
    convert_to_overlapped_model,
)
from .parallel_state import (
    NCCLConfig,
    ParallelismDimension,
    destroy_model_parallel,
    disable_sequence_parallel,
    enable_sequence_parallel,
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_size,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_size,
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_size,
    get_model_parallel_group,
    get_nccl_config,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_size,
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_tensor_and_data_parallel_group,
    get_tensor_and_data_parallel_group_with_cp,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
    get_virtual_pipeline_model_parallel_rank,
    get_virtual_pipeline_model_parallel_size,
    initialize_model_parallel,
    is_initialized,
    is_sequence_parallel_enabled,
    set_nccl_config,
    set_virtual_pipeline_model_parallel_rank,
)
from .pipeline_parallel import MicrobatchInfo, PipelineParallel, PipelineStage
from .sequence_parallel import (
    all_to_all_hidden_to_sequence,
    all_to_all_sequence_to_hidden,
    gather_from_sequence_parallel_region,
    is_sequence_parallel_tensor,
    mark_tensor_as_sequence_parallel,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from .zero import ZeROOptimizer

__all__ = [
    # Data Parallel
    "DataParallelTrainer",
    # Model Parallel
    "TensorParallelism",
    "ColumnParallelLinear",
    "RowParallelLinear",
    # Overlap Integration
    "OverlappedColumnParallelLinear",
    "OverlappedRowParallelLinear",
    "OverlappedPipelineEngine",
    "convert_to_overlapped_model",
    # Pipeline Parallel
    "PipelineParallel",
    "PipelineStage",
    "MicrobatchInfo",
    # Microbatch Calculator
    "MicrobatchCalculatorBase",
    "ConstantNumMicrobatches",
    "RampupBatchSizeNumMicrobatches",
    "AdaptiveMicrobatchCalculator",
    "initialize_microbatch_calculator",
    "get_microbatch_calculator",
    "get_num_microbatches",
    "get_micro_batch_size",
    "update_microbatch_calculator",
    "destroy_microbatch_calculator",
    "calculate_optimal_microbatch_size",
    "get_microbatch_schedule",
    # Sequence Parallel
    "scatter_to_sequence_parallel_region",
    "gather_from_sequence_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
    "all_to_all_sequence_to_hidden",
    "all_to_all_hidden_to_sequence",
    "mark_tensor_as_sequence_parallel",
    "is_sequence_parallel_tensor",
    "is_sequence_parallel_enabled",
    "get_sequence_parallel_group",
    "get_sequence_parallel_world_size",
    "get_sequence_parallel_rank",
    "enable_sequence_parallel",
    "disable_sequence_parallel",
    # ZeRO
    "ZeROOptimizer",
    # Parallel State Management
    "initialize_model_parallel",
    "is_initialized",
    "destroy_model_parallel",
    "get_tensor_model_parallel_group",
    "get_pipeline_model_parallel_group",
    "get_data_parallel_group",
    "get_context_parallel_group",
    "get_expert_model_parallel_group",
    "get_tensor_model_parallel_size",
    "get_pipeline_model_parallel_size",
    "get_data_parallel_size",
    "get_context_parallel_size",
    "get_expert_model_parallel_size",
    "get_tensor_model_parallel_rank",
    "get_pipeline_model_parallel_rank",
    "get_data_parallel_rank",
    "get_context_parallel_rank",
    "get_expert_model_parallel_rank",
    "get_model_parallel_group",
    "get_tensor_and_data_parallel_group",
    "get_tensor_and_data_parallel_group_with_cp",
    "get_virtual_pipeline_model_parallel_rank",
    "get_virtual_pipeline_model_parallel_size",
    "set_virtual_pipeline_model_parallel_rank",
    "NCCLConfig",
    "ParallelismDimension",
    "set_nccl_config",
    "get_nccl_config",
]
