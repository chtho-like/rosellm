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
"""

from .data_parallel import DataParallelTrainer
from .model_parallel import (ColumnParallelLinear, RowParallelLinear,
                             TensorParallelism)
from .parallel_state import (NCCLConfig, ParallelismDimension,
                             destroy_model_parallel,
                             get_context_parallel_group,
                             get_context_parallel_rank,
                             get_context_parallel_size,
                             get_data_parallel_group, get_data_parallel_rank,
                             get_data_parallel_size,
                             get_expert_model_parallel_group,
                             get_expert_model_parallel_rank,
                             get_expert_model_parallel_size,
                             get_model_parallel_group, get_nccl_config,
                             get_pipeline_model_parallel_group,
                             get_pipeline_model_parallel_rank,
                             get_pipeline_model_parallel_size,
                             get_tensor_and_data_parallel_group,
                             get_tensor_and_data_parallel_group_with_cp,
                             get_tensor_model_parallel_group,
                             get_tensor_model_parallel_rank,
                             get_tensor_model_parallel_size,
                             get_virtual_pipeline_model_parallel_rank,
                             get_virtual_pipeline_model_parallel_size,
                             initialize_model_parallel, is_initialized,
                             set_nccl_config,
                             set_virtual_pipeline_model_parallel_rank)
from .pipeline_parallel import MicrobatchInfo, PipelineParallel, PipelineStage
from .zero import ZeROOptimizer

__all__ = [
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
    "get_virtual_pipeline_model_parallel_rank",
    "get_virtual_pipeline_model_parallel_size",
    "set_virtual_pipeline_model_parallel_rank",
    "NCCLConfig",
    "ParallelismDimension",
    "set_nccl_config",
    "get_nccl_config",
]
