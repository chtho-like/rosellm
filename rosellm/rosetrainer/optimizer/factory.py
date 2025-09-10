"""Factory pattern for creating distributed optimizers.

This module provides a factory interface for creating distributed optimizers
with various configurations and strategies.
"""

from typing import Any, Dict, Iterator, List, Optional, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from .config import DistributedOptimizerConfig, PartitioningStrategy
from .distributed_optimizer import DistributedOptimizer
from .multi_tensor_adam import MultiTensorAdam, MultiTensorAdamConfig


class OptimizerFactory:
    """Factory for creating distributed optimizers with various configurations."""

    # Registry of optimizer presets
    PRESETS: Dict[str, DistributedOptimizerConfig] = {
        "memory_efficient": DistributedOptimizerConfig(
            partition_parameters=True,
            partition_gradients=True,
            partition_optimizer_states=True,
            contiguous_gradients=True,
            partitioning_strategy=PartitioningStrategy.BALANCED,
        ),
        "speed_optimized": DistributedOptimizerConfig(
            partition_parameters=False,
            partition_gradients=True,
            partition_optimizer_states=False,
            contiguous_gradients=True,
            overlap_grad_reduce=True,
        ),
        "mixed_precision": DistributedOptimizerConfig(
            partition_parameters=True,
            partition_gradients=True,
            partition_optimizer_states=True,
            mixed_precision=True,
            memory_efficient_fp16=True,
            check_gradients=True,
        ),
        "cpu_offload": DistributedOptimizerConfig(
            partition_parameters=True,
            partition_gradients=True,
            partition_optimizer_states=True,
            cpu_offload=True,
            contiguous_gradients=True,
        ),
        "baseline": DistributedOptimizerConfig(
            partition_parameters=False,
            partition_gradients=False,
            partition_optimizer_states=False,
            contiguous_gradients=False,
        ),
        "multi_tensor_performance": DistributedOptimizerConfig(
            partition_parameters=False,
            partition_gradients=True,
            partition_optimizer_states=False,
            contiguous_gradients=True,
            overlap_grad_reduce=True,
            mixed_precision=False,
        ),
        "multi_tensor_mixed_precision": DistributedOptimizerConfig(
            partition_parameters=True,
            partition_gradients=True,
            partition_optimizer_states=True,
            mixed_precision=True,
            memory_efficient_fp16=True,
            check_gradients=True,
            contiguous_gradients=True,
        ),
    }

    @classmethod
    def create(
        cls,
        params: Union[Iterator[nn.Parameter], List[Dict[str, Any]]],
        optimizer_class: Type[Optimizer],
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        preset: Optional[str] = None,
        config: Optional[DistributedOptimizerConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> DistributedOptimizer:
        """Create a distributed optimizer.

        Args:
            params: Model parameters or parameter groups.
            optimizer_class: Base optimizer class (e.g., torch.optim.Adam).
            optimizer_kwargs: Arguments for base optimizer.
            preset: Name of configuration preset to use.
            config: Custom configuration (overrides preset).
            process_group: Process group for communication.

        Returns:
            Configured distributed optimizer instance.

        Raises:
            ValueError: If invalid preset name is provided.

        Example:
            >>> optimizer = OptimizerFactory.create(
            ...     model.parameters(),
            ...     torch.optim.AdamW,
            ...     {"lr": 1e-4, "weight_decay": 0.01},
            ...     preset="memory_efficient"
            ... )
        """
        # Determine configuration
        if config is None:
            if preset is not None:
                if preset not in cls.PRESETS:
                    raise ValueError(
                        f"Unknown preset '{preset}'. Available presets: "
                        f"{list(cls.PRESETS.keys())}"
                    )
                config = cls.PRESETS[preset]
            else:
                # Use default configuration
                config = DistributedOptimizerConfig()

        # Default optimizer kwargs
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        # Create and return optimizer
        return DistributedOptimizer(
            params=params,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            config=config,
            process_group=process_group,
        )

    @classmethod
    def create_from_model(
        cls,
        model: nn.Module,
        optimizer_name: str = "AdamW",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        preset: Optional[str] = None,
        config: Optional[DistributedOptimizerConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> DistributedOptimizer:
        """Create a distributed optimizer from a model.

        Args:
            model: Model to optimize.
            optimizer_name: Name of optimizer (Adam, AdamW, SGD, etc.).
            lr: Learning rate.
            weight_decay: Weight decay for regularization.
            preset: Configuration preset name.
            config: Custom configuration.
            process_group: Process group for communication.

        Returns:
            Configured distributed optimizer.

        Example:
            >>> optimizer = OptimizerFactory.create_from_model(
            ...     model,
            ...     optimizer_name="AdamW",
            ...     lr=1e-4,
            ...     preset="mixed_precision"
            ... )
        """
        # Map optimizer names to classes
        optimizer_map = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "MultiTensorAdam": MultiTensorAdam,
            "MultiTensorAdamW": MultiTensorAdam,  # Same class, different config
            "SGD": torch.optim.SGD,
            "RMSprop": torch.optim.RMSprop,
            "Adagrad": torch.optim.Adagrad,
        }

        if optimizer_name not in optimizer_map:
            raise ValueError(
                f"Unknown optimizer '{optimizer_name}'. "
                f"Available: {list(optimizer_map.keys())}"
            )

        optimizer_class = optimizer_map[optimizer_name]

        # Build optimizer kwargs based on optimizer type
        optimizer_kwargs: Dict[str, Any] = {"lr": lr}

        if optimizer_name in ["AdamW", "Adam"]:
            optimizer_kwargs["weight_decay"] = weight_decay
            optimizer_kwargs["betas"] = (0.9, 0.999)
            optimizer_kwargs["eps"] = 1e-8
        elif optimizer_name in ["MultiTensorAdam", "MultiTensorAdamW"]:
            # Import WeightDecayMode for proper type conversion
            from .multi_tensor_adam import WeightDecayMode

            # Create configuration for multi-tensor Adam
            optimizer_kwargs["config"] = MultiTensorAdamConfig(
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay_mode=(
                    WeightDecayMode.DECOUPLED
                    if optimizer_name == "MultiTensorAdamW"
                    else WeightDecayMode.L2_REGULARIZATION
                ),
                use_mixed_precision=bool(config and config.mixed_precision),
                enable_multi_tensor=True,
            )
        elif optimizer_name == "SGD":
            optimizer_kwargs["weight_decay"] = weight_decay
            optimizer_kwargs["momentum"] = 0.9
        elif optimizer_name == "RMSprop":
            optimizer_kwargs["weight_decay"] = weight_decay
            optimizer_kwargs["alpha"] = 0.99
            optimizer_kwargs["eps"] = 1e-8

        return cls.create(
            params=model.parameters(),
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            preset=preset,
            config=config,
            process_group=process_group,
        )

    @classmethod
    def register_preset(
        cls,
        name: str,
        config: DistributedOptimizerConfig,
        override: bool = False,
    ) -> None:
        """Register a new configuration preset.

        Args:
            name: Name for the preset.
            config: Configuration to register.
            override: Whether to override existing preset.

        Raises:
            ValueError: If preset exists and override is False.
        """
        if name in cls.PRESETS and not override:
            raise ValueError(
                f"Preset '{name}' already exists. Set override=True to replace."
            )
        cls.PRESETS[name] = config

    @classmethod
    def get_preset_names(cls) -> List[str]:
        """Get list of available preset names.

        Returns:
            List of preset names.
        """
        return list(cls.PRESETS.keys())

    @classmethod
    def create_multi_tensor_optimizer(
        cls,
        params: Union[Iterator[nn.Parameter], List[Dict[str, Any]]],
        optimizer_type: str = "MultiTensorAdamW",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        mixed_precision: bool = False,
        max_grad_norm: Optional[float] = None,
        preset: Optional[str] = None,
        config: Optional[DistributedOptimizerConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        **optimizer_kwargs: Any,
    ) -> DistributedOptimizer:
        """Create a multi-tensor optimizer with advanced features.

        Args:
            params: Model parameters or parameter groups.
            optimizer_type: Type of multi-tensor optimizer.
            lr: Learning rate.
            weight_decay: Weight decay coefficient.
            mixed_precision: Enable mixed precision training.
            max_grad_norm: Maximum gradient norm for clipping.
            preset: Configuration preset name.
            config: Custom configuration.
            process_group: Process group for communication.
            **optimizer_kwargs: Additional optimizer arguments.

        Returns:
            Configured multi-tensor optimizer.

        Example:
            >>> optimizer = OptimizerFactory.create_multi_tensor_optimizer(
            ...     model.parameters(),
            ...     optimizer_type="MultiTensorAdamW",
            ...     lr=1e-4,
            ...     mixed_precision=True,
            ...     max_grad_norm=1.0,
            ...     preset="multi_tensor_mixed_precision"
            ... )
        """
        # Choose appropriate preset if not specified
        if preset is None:
            if mixed_precision:
                preset = "multi_tensor_mixed_precision"
            else:
                preset = "multi_tensor_performance"

        # Create multi-tensor Adam configuration
        from .multi_tensor_adam import WeightDecayMode

        mt_config_kwargs = {
            "lr": lr,
            "weight_decay": weight_decay,
            "weight_decay_mode": (
                WeightDecayMode.DECOUPLED
                if "AdamW" in optimizer_type
                else WeightDecayMode.L2_REGULARIZATION
            ),
            "use_mixed_precision": mixed_precision,
            "max_grad_norm": max_grad_norm,
            "enable_multi_tensor": True,
            "enable_profiling": False,
            **optimizer_kwargs,
        }

        mt_config = MultiTensorAdamConfig(**mt_config_kwargs)

        # Create optimizer kwargs
        opt_kwargs = {"config": mt_config}

        return cls.create(
            params=params,
            optimizer_class=MultiTensorAdam,
            optimizer_kwargs=opt_kwargs,
            preset=preset,
            config=config,
            process_group=process_group,
        )

    @classmethod
    def describe_preset(cls, name: str) -> Dict[str, Any]:
        """Get description of a preset configuration.

        Args:
            name: Preset name.

        Returns:
            Dictionary describing the preset configuration.

        Raises:
            ValueError: If preset doesn't exist.
        """
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown preset '{name}'")

        config = cls.PRESETS[name]
        return config.to_dict()
