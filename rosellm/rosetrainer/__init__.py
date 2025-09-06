"""
RoseTrainer - Distributed Training Framework for Large Language Models

This module provides a comprehensive distributed training framework with support for:
- Multi-dimensional parallelism (TP, PP, DP, CP, EP)
- Memory optimization techniques
- Advanced communication patterns
"""

from .engine import RoseTrainer
from .parallelism import ColumnParallelLinear  # Parallelism components
from .parallelism import (
    DataParallelTrainer,
    MicrobatchInfo,
    NCCLConfig,
    ParallelismDimension,
    PipelineParallel,
    PipelineStage,
    RowParallelLinear,
    TensorParallelism,
    ZeROOptimizer,
    destroy_model_parallel,
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
    get_tensor_and_data_parallel_group,
    get_tensor_and_data_parallel_group_with_cp,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
    initialize_model_parallel,
    is_initialized,
    set_nccl_config,
)

__all__ = [
    "RoseTrainer",
    # Data Parallel
    "DataParallelTrainer",
    # Model Parallel
    "TensorParallelism",
    "ColumnParallelLinear",
    "RowParallelLinear",
    # Pipeline Parallel
    "PipelineParallel",
    "PipelineStage",
    "MicrobatchInfo",
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
    "NCCLConfig",
    "ParallelismDimension",
    "set_nccl_config",
    "get_nccl_config",
]
