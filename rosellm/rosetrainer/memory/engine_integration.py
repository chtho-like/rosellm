"""
Engine Integration for Distributed Activation Checkpointing

This module provides integration hooks for the RoseTrainer engine to support
advanced distributed activation checkpointing capabilities. It extends the
training engine with memory-efficient distributed training features.

Key Features:
- Seamless integration with existing RoseTrainer workflow
- Automatic distributed checkpointing setup based on parallelism configuration
- Memory optimization coordination during training steps
- Profiling and monitoring integration
- Dynamic memory management across training iterations

Usage Example:
    from rosellm.rosetrainer.memory import integrate_distributed_checkpointing

    trainer = RoseTrainer(model, optimizer, config)
    trainer = integrate_distributed_checkpointing(trainer)

    # Training loop continues as normal, but with distributed checkpointing
    for batch in dataloader:
        loss = trainer.training_step(batch)
        trainer.backward_step(loss)
        trainer.optimizer_step()
"""

import logging
from typing import Any, Dict, Optional, Union

import torch.nn as nn

from ..parallelism import parallel_state
from .distributed_memory_optimizer import create_distributed_memory_optimizer

logger = logging.getLogger(__name__)


def integrate_distributed_checkpointing(
    trainer: Any,
    strategy: str = "hierarchical",
    enable_memory_balancing: bool = True,
    enable_model_parallel: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """Integrate distributed checkpointing with an existing RoseTrainer instance.

    Args:
        trainer: RoseTrainer instance to enhance
        strategy: Distributed checkpointing strategy
        enable_memory_balancing: Whether to enable cross-rank memory balancing
        enable_model_parallel: Whether to enable model parallel optimizations
        config_overrides: Optional configuration overrides

    Returns:
        Enhanced trainer with distributed checkpointing capabilities
    """
    if not hasattr(trainer, "model") or not hasattr(trainer, "optimizer"):
        raise ValueError("Trainer must have 'model' and 'optimizer' attributes")

    # Create distributed memory optimizer
    distributed_optimizer = create_distributed_memory_optimizer(
        model=trainer.model,
        optimizer=trainer.optimizer,
        enable_memory_balancing=enable_memory_balancing,
        enable_cpu_offloading=False,  # Keep conservative defaults
        enable_mixed_precision=False,
        enable_parameter_overlap=enable_model_parallel,
        distributed_checkpoint_strategy=strategy,
        **(config_overrides or {}),
    )

    # Integrate with model
    trainer.model = distributed_optimizer.integrate_with_model()

    # Store optimizer reference
    trainer._distributed_memory_optimizer = distributed_optimizer

    # Enhance training step method
    original_training_step = getattr(trainer, "training_step", None)
    if original_training_step:

        def enhanced_training_step(*args, **kwargs):
            # Perform memory optimization before training step
            step = getattr(trainer, "_current_step", 0)
            optimization_stats = distributed_optimizer.optimize_step(step)

            # Store optimization stats
            trainer._last_optimization_stats = optimization_stats

            # Call original training step
            return original_training_step(*args, **kwargs)

        trainer.training_step = enhanced_training_step

    # Enhance backward step method
    original_backward_step = getattr(trainer, "backward_step", None)
    if original_backward_step:

        def enhanced_backward_step(*args, **kwargs):
            result = original_backward_step(*args, **kwargs)

            # Update step counter
            current_step = getattr(trainer, "_current_step", 0) + 1
            trainer._current_step = current_step

            return result

        trainer.backward_step = enhanced_backward_step

    # Add method to get distributed profiling report
    def get_distributed_profiling_report():
        """Get comprehensive distributed profiling report."""
        return distributed_optimizer.get_optimization_report()

    trainer.get_distributed_profiling_report = get_distributed_profiling_report

    # Add method to reset distributed profiling
    def reset_distributed_profiling():
        """Reset all distributed profiling statistics."""
        distributed_optimizer.reset_optimization_stats()

    trainer.reset_distributed_profiling = reset_distributed_profiling

    # Initialize step counter
    trainer._current_step = 0
    trainer._last_optimization_stats = {}

    logger.info(
        f"Successfully integrated distributed checkpointing with strategy: {strategy}"
    )

    return trainer


def create_distributed_trainer_config(
    base_config: Dict[str, Any],
    distributed_strategy: str = "hierarchical",
    enable_advanced_checkpointing: bool = True,
    memory_optimization_level: str = "balanced",  # "conservative", "balanced",
    # "aggressive"
) -> Dict[str, Any]:
    """Create training configuration optimized for distributed checkpointing.

    Args:
        base_config: Base training configuration
        distributed_strategy: Distributed checkpointing strategy
        enable_advanced_checkpointing: Whether to enable advanced checkpointing features
        memory_optimization_level: Level of memory optimization

    Returns:
        Enhanced configuration dictionary
    """
    config = base_config.copy()

    # Add distributed checkpointing configuration
    distributed_config = {
        "distributed_checkpointing": {
            "strategy": distributed_strategy,
            "enable_load_balancing": memory_optimization_level
            in ["balanced", "aggressive"],
            "coordinate_across_tp": True,
            "coordinate_across_cp": True,
            "coordinate_across_ep": True,
            "enable_cross_rank_profiling": enable_advanced_checkpointing,
            "verbose_distributed": False,
        }
    }

    # Memory optimization settings based on level
    if memory_optimization_level == "conservative":
        memory_config = {
            "enable_memory_balancing": False,
            "coordinate_cpu_offloading": False,
            "coordinate_precision_scaling": False,
            "enable_parameter_overlap": False,
        }
    elif memory_optimization_level == "balanced":
        memory_config = {
            "enable_memory_balancing": True,
            "coordinate_cpu_offloading": False,
            "coordinate_precision_scaling": False,
            "enable_parameter_overlap": True,
        }
    else:  # aggressive
        memory_config = {
            "enable_memory_balancing": True,
            "coordinate_cpu_offloading": True,
            "coordinate_precision_scaling": True,
            "enable_parameter_overlap": True,
            "enable_distributed_profiling": True,
        }

    distributed_config["memory_optimization"] = dict(memory_config)

    # Model parallel settings
    if enable_advanced_checkpointing:
        model_parallel_config = {
            "checkpoint_attention_layers": True,
            "checkpoint_mlp_layers": True,
            "sync_column_parallel_activations": True,
            "sync_row_parallel_activations": True,
            "overlap_communication_computation": True,
        }
        distributed_config["model_parallel"] = dict(model_parallel_config)

    config.update(distributed_config)

    return config


def get_optimal_distributed_strategy() -> str:
    """Determine optimal distributed checkpointing strategy based on current
    parallelism setup.

    Returns:
        Recommended distributed checkpointing strategy name
    """
    if not parallel_state.is_initialized():
        return "coordinated"  # Simple coordination for single rank

    # Get parallelism dimensions
    tp_size = parallel_state.get_tensor_model_parallel_size()
    pp_size = parallel_state.get_pipeline_model_parallel_size()
    dp_size = parallel_state.get_data_parallel_size()
    cp_size = parallel_state.get_context_parallel_size()
    ep_size = parallel_state.get_expert_model_parallel_size()

    # Determine strategy based on active parallelism dimensions
    active_dimensions = sum(
        [
            tp_size > 1,
            pp_size > 1,
            dp_size > 1,
            cp_size > 1,
            ep_size > 1,
        ]
    )

    if ep_size > 1:
        # Expert parallelism present - use expert-aware strategy
        return "expert_aware"
    elif pp_size > 1:
        # Pipeline parallelism present - use pipeline-aware strategy
        return "pipeline_aware"
    elif active_dimensions >= 3:
        # Multiple dimensions active - use adaptive strategy
        return "adaptive"
    elif active_dimensions >= 2:
        # Two dimensions active - use hierarchical strategy
        return "hierarchical"
    elif tp_size > 1 or cp_size > 1:
        # Only tensor or context parallelism - use coordinated strategy
        return "coordinated"
    else:
        # Only data parallelism or single rank - use load balanced
        return "load_balanced"


def estimate_memory_savings(
    model: nn.Module,
    strategy: str = "hierarchical",
    sample_input_shape: tuple = (2, 512, 4096),
) -> Dict[str, Union[float, str]]:
    """Estimate potential memory savings from distributed checkpointing.

    Args:
        model: Model to analyze
        strategy: Distributed checkpointing strategy
        sample_input_shape: Sample input shape for analysis

    Returns:
        Dictionary with estimated memory savings statistics
    """
    # This is a simplified estimation - in practice would be more sophisticated
    total_params = sum(p.numel() for p in model.parameters())
    param_memory_gb = total_params * 4 / (1024**3)  # Assume FP32

    # Estimate activation memory based on model depth
    num_layers = len(list(model.modules()))
    activation_memory_gb = num_layers * 0.1  # Rough estimate

    # Strategy-based savings estimation
    strategy_savings = {
        "coordinated": 0.15,  # 15% savings
        "load_balanced": 0.20,  # 20% savings
        "hierarchical": 0.25,  # 25% savings
        "expert_aware": 0.30,  # 30% savings
        "pipeline_aware": 0.28,  # 28% savings
        "adaptive": 0.35,  # 35% savings
    }

    savings_ratio = strategy_savings.get(strategy, 0.20)

    estimated_savings_gb = activation_memory_gb * savings_ratio

    return {
        "total_parameter_memory_gb": param_memory_gb,
        "estimated_activation_memory_gb": activation_memory_gb,
        "estimated_savings_gb": estimated_savings_gb,
        "savings_percentage": savings_ratio * 100,
        "strategy": strategy,
    }


# Convenience function for automatic integration
def auto_integrate_distributed_checkpointing(trainer: Any) -> Any:
    """Automatically integrate distributed checkpointing with optimal settings.

    Args:
        trainer: RoseTrainer instance to enhance

    Returns:
        Enhanced trainer with automatically configured distributed checkpointing
    """
    # Determine optimal strategy
    strategy = get_optimal_distributed_strategy()

    # Estimate memory savings
    if hasattr(trainer, "model"):
        savings_estimate = estimate_memory_savings(trainer.model, strategy)
        logger.info(
            f"Estimated memory savings: "
            f"{savings_estimate['estimated_savings_gb']:.2f} GB "
            f"({savings_estimate['savings_percentage']:.1f}%) with {strategy} strategy"
        )

    # Integrate with optimal settings
    return integrate_distributed_checkpointing(
        trainer=trainer,
        strategy=strategy,
        enable_memory_balancing=True,
        enable_model_parallel=parallel_state.is_initialized(),
    )
