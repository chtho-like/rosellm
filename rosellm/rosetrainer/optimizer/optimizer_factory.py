"""Factory for creating various optimizer configurations including ChainedOptimizer.

This module extends the optimizer creation capabilities with support for:
- ChainedOptimizer for multi-optimizer setups
- Parameter group separation based on various criteria
- MoE-specific optimizer configurations
- Fine-tuning optimizers with layer-wise learning rates
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Type, Union, cast

import torch
import torch.nn as nn
from torch.optim import SGD, Adagrad, Adam, AdamW, Optimizer, RMSprop

from .chained_optimizer import ChainedOptimizer
from .distributed_optimizer import DistributedOptimizer
from .multi_tensor_adam import MultiTensorAdam, MultiTensorAdamConfig, WeightDecayMode

logger = logging.getLogger(__name__)


class OptimFactory:
    """Extended factory for creating optimizers with advanced configurations."""

    # Registry of available optimizer classes
    OPTIMIZER_REGISTRY: Dict[str, Type[Optimizer]] = {
        "adam": Adam,
        "adamw": AdamW,
        "sgd": SGD,
        "rmsprop": RMSprop,
        "adagrad": Adagrad,
        "multi_tensor_adam": MultiTensorAdam,
        "distributed": DistributedOptimizer,
        "chained": ChainedOptimizer,
    }

    @classmethod
    def create_optimizer(
        cls,
        params: Union[Iterator[nn.Parameter], List[Dict[str, Any]]],
        optimizer_type: str = "adamw",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        **kwargs: Any,
    ) -> Optimizer:
        """Create a single optimizer instance.

        Args:
            params: Model parameters or parameter groups.
            optimizer_type: Type of optimizer to create.
            lr: Learning rate.
            weight_decay: Weight decay coefficient.
            **kwargs: Additional optimizer-specific arguments.

        Returns:
            Configured optimizer instance.
        """
        optimizer_type = optimizer_type.lower()

        if optimizer_type not in cls.OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unknown optimizer type '{optimizer_type}'. "
                f"Available: {list(cls.OPTIMIZER_REGISTRY.keys())}"
            )

        optimizer_class = cls.OPTIMIZER_REGISTRY[optimizer_type]

        # Build optimizer arguments based on type
        opt_kwargs: Dict[str, Any] = {"lr": lr}

        if optimizer_type in ["adam", "adamw"]:
            opt_kwargs.update(
                {
                    "weight_decay": weight_decay,
                    "betas": kwargs.get("betas", (0.9, 0.999)),
                    "eps": kwargs.get("eps", 1e-8),
                    "amsgrad": kwargs.get("amsgrad", False),
                }
            )
        elif optimizer_type == "sgd":
            opt_kwargs.update(
                {
                    "weight_decay": weight_decay,
                    "momentum": kwargs.get("momentum", 0.9),
                    "dampening": kwargs.get("dampening", 0),
                    "nesterov": kwargs.get("nesterov", False),
                }
            )
        elif optimizer_type == "rmsprop":
            opt_kwargs.update(
                {
                    "weight_decay": weight_decay,
                    "alpha": kwargs.get("alpha", 0.99),
                    "eps": kwargs.get("eps", 1e-8),
                    "momentum": kwargs.get("momentum", 0),
                }
            )
        elif optimizer_type == "adagrad":
            opt_kwargs.update(
                {
                    "weight_decay": weight_decay,
                    "lr_decay": kwargs.get("lr_decay", 0),
                    "eps": kwargs.get("eps", 1e-10),
                }
            )
        elif optimizer_type == "multi_tensor_adam":
            # Create configuration for multi-tensor Adam
            config = MultiTensorAdamConfig(
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get("betas", (0.9, 0.999)),
                eps=kwargs.get("eps", 1e-8),
                weight_decay_mode=kwargs.get(
                    "weight_decay_mode", WeightDecayMode.DECOUPLED
                ),
                use_mixed_precision=kwargs.get("use_mixed_precision", False),
                max_grad_norm=kwargs.get("max_grad_norm"),
                enable_multi_tensor=kwargs.get("enable_multi_tensor", True),
            )
            opt_kwargs = {"config": config}
        else:
            # Pass through any additional kwargs
            opt_kwargs.update(kwargs)

        return cast(Optimizer, optimizer_class(params, **opt_kwargs))

    @classmethod
    def create_chained_optimizer(
        cls,
        param_groups: List[Dict[str, Any]],
        optimizer_configs: List[Dict[str, Any]],
        enable_metrics: bool = False,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> ChainedOptimizer:
        """Create a ChainedOptimizer with multiple sub-optimizers.

        Args:
            param_groups: List of parameter group specifications.
                Each dict should have 'params' and 'optimizer_idx'.
            optimizer_configs: List of optimizer configurations.
                Each dict should specify optimizer type and hyperparameters.
            enable_metrics: Whether to enable metrics collection.
            grad_scaler: Optional gradient scaler for mixed precision.

        Returns:
            Configured ChainedOptimizer instance.

        Example:
            >>> param_groups = [
            ...     {'params': dense_params, 'optimizer_idx': 0},
            ...     {'params': expert_params, 'optimizer_idx': 1},
            ... ]
            >>> optimizer_configs = [
            ...     {'type': 'adamw', 'lr': 1e-3},
            ...     {'type': 'adam', 'lr': 1e-4},
            ... ]
            >>> optimizer = OptimFactory.create_chained_optimizer(
            ...     param_groups, optimizer_configs
            ... )
        """
        # Group parameters by optimizer index
        params_by_optimizer: Dict[int, List[Dict[str, Any]]] = {}

        for group in param_groups:
            opt_idx = group.get("optimizer_idx", 0)
            if opt_idx not in params_by_optimizer:
                params_by_optimizer[opt_idx] = []

            # Create parameter group for this optimizer
            param_group = {
                "params": group["params"],
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
            }
            # Remove None values
            param_group = {k: v for k, v in param_group.items() if v is not None}
            params_by_optimizer[opt_idx].append(param_group)

        # Create optimizers
        optimizers = []
        for idx, config in enumerate(optimizer_configs):
            if idx not in params_by_optimizer:
                logger.warning(f"No parameters assigned to optimizer {idx}, skipping")
                continue

            opt_type = config.pop("type", "adamw")
            optimizer = cls.create_optimizer(
                params_by_optimizer[idx], optimizer_type=opt_type, **config
            )
            optimizers.append(optimizer)

        return ChainedOptimizer(
            optimizers,
            enable_metrics=enable_metrics,
            grad_scaler=grad_scaler,
        )

    @classmethod
    def create_moe_optimizer(
        cls,
        model: nn.Module,
        base_lr: float = 1e-3,
        expert_lr_multiplier: float = 0.1,
        weight_decay: float = 0.01,
        optimizer_type: str = "adamw",
        enable_metrics: bool = False,
    ) -> ChainedOptimizer:
        """Create optimizers for MoE models with separate expert handling.

        Args:
            model: MoE model with expert and dense parameters.
            base_lr: Base learning rate for dense parameters.
            expert_lr_multiplier: Multiplier for expert learning rate.
            weight_decay: Weight decay coefficient.
            optimizer_type: Type of optimizer to use.
            enable_metrics: Whether to enable metrics collection.

        Returns:
            ChainedOptimizer with separate optimizers for experts and dense.
        """
        # Separate parameters
        expert_params = []
        dense_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Check various ways parameters might be marked as expert
            is_expert = (
                (hasattr(param, "is_expert") and getattr(param, "is_expert", False))
                or (
                    hasattr(param, "allreduce")
                    and not getattr(param, "allreduce", True)
                )
                or ("expert" in name.lower())
            )

            # Determine weight decay
            no_decay = "bias" in name or "norm" in name or "embedding" in name
            wd = 0.0 if no_decay else weight_decay

            param_spec = {
                "params": [param],
                "weight_decay": wd,
                "name": name,
            }

            if is_expert:
                expert_params.append(param_spec)
            else:
                dense_params.append(param_spec)

        logger.info(
            f"MoE optimizer: {len(dense_params)} dense groups, "
            f"{len(expert_params)} expert groups"
        )

        # Create parameter groups
        param_groups = []
        if dense_params:
            param_groups.extend(
                [{**group, "optimizer_idx": 0} for group in dense_params]
            )
        if expert_params:
            param_groups.extend(
                [{**group, "optimizer_idx": 1} for group in expert_params]
            )

        # Create optimizer configs
        optimizer_configs = []
        if dense_params:
            optimizer_configs.append(
                {
                    "type": optimizer_type,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                }
            )
        if expert_params:
            optimizer_configs.append(
                {
                    "type": optimizer_type,
                    "lr": base_lr * expert_lr_multiplier,
                    "weight_decay": weight_decay,
                }
            )

        return cls.create_chained_optimizer(
            param_groups,
            optimizer_configs,
            enable_metrics=enable_metrics,
        )

    @classmethod
    def create_layer_wise_optimizer(
        cls,
        model: nn.Module,
        base_lr: float = 1e-4,
        lr_decay: float = 0.9,
        weight_decay: float = 0.01,
        optimizer_type: str = "adamw",
        num_layers: Optional[int] = None,
    ) -> Optimizer:
        """Create optimizer with layer-wise learning rate decay.

        Args:
            model: Model to optimize.
            base_lr: Base learning rate for the last layer.
            lr_decay: Learning rate decay factor per layer.
            weight_decay: Weight decay coefficient.
            optimizer_type: Type of optimizer to use.
            num_layers: Number of layers (auto-detected if None).

        Returns:
            Optimizer with layer-wise learning rates.
        """
        # Auto-detect number of layers if not provided
        if num_layers is None:
            # Count transformer layers or similar structures
            num_layers = 0
            for name, _ in model.named_modules():
                if "layer" in name or "block" in name:
                    try:
                        layer_idx = int(name.split(".")[-1])
                        num_layers = max(num_layers, layer_idx + 1)
                    except (ValueError, IndexError):
                        pass

            if num_layers == 0:
                # Fallback: count by depth
                num_layers = 12  # Common default

        # Create parameter groups with layer-wise LR
        param_groups = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine layer index
            layer_idx = num_layers - 1  # Default to last layer
            for i in range(num_layers):
                if f"layer.{i}" in name or f"block.{i}" in name:
                    layer_idx = i
                    break

            # Calculate learning rate with decay
            lr = base_lr * (lr_decay ** (num_layers - 1 - layer_idx))

            # Determine weight decay
            no_decay = "bias" in name or "norm" in name or "embedding" in name
            wd = 0.0 if no_decay else weight_decay

            param_groups.append(
                {
                    "params": [param],
                    "lr": lr,
                    "weight_decay": wd,
                    "name": name,
                    "layer_idx": layer_idx,
                }
            )

        logger.info(
            f"Layer-wise optimizer: {len(param_groups)} parameter groups "
            f"across {num_layers} layers"
        )

        return cls.create_optimizer(
            param_groups,
            optimizer_type=optimizer_type,
            lr=base_lr,  # Will be overridden by group-specific LR
            weight_decay=weight_decay,
        )

    @classmethod
    def create_parameter_efficient_optimizer(
        cls,
        model: nn.Module,
        trainable_pattern: Optional[str] = None,
        base_lr: float = 1e-4,
        lora_lr_multiplier: float = 10.0,
        weight_decay: float = 0.01,
        optimizer_type: str = "adamw",
    ) -> Optimizer:
        """Create optimizer for parameter-efficient fine-tuning.

        Args:
            model: Model with some frozen and some trainable parameters.
            trainable_pattern: Pattern to match trainable parameter names.
            base_lr: Base learning rate for regular parameters.
            lora_lr_multiplier: LR multiplier for LoRA or adapter parameters.
            weight_decay: Weight decay coefficient.
            optimizer_type: Type of optimizer to use.

        Returns:
            Optimizer configured for parameter-efficient training.
        """
        # Separate parameters by type
        lora_params = []
        regular_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this is a LoRA/adapter parameter
            is_lora = (
                "lora" in name.lower()
                or "adapter" in name.lower()
                or (trainable_pattern and trainable_pattern in name)
            )

            # No weight decay for LoRA parameters
            wd = 0.0 if is_lora else weight_decay

            param_spec = {
                "params": [param],
                "weight_decay": wd,
                "name": name,
            }

            if is_lora:
                param_spec["lr"] = base_lr * lora_lr_multiplier
                lora_params.append(param_spec)
            else:
                param_spec["lr"] = base_lr
                regular_params.append(param_spec)

        # Combine parameter groups
        param_groups = regular_params + lora_params

        logger.info(
            f"Parameter-efficient optimizer: "
            f"{len(regular_params)} regular params, "
            f"{len(lora_params)} LoRA/adapter params"
        )

        return cls.create_optimizer(
            param_groups,
            optimizer_type=optimizer_type,
            lr=base_lr,
            weight_decay=weight_decay,
        )
