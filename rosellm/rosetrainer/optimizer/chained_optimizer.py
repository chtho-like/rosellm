"""ChainedOptimizer for managing multiple optimizers with different parameter groups.

This module provides ChainedOptimizer which enables:
- Managing multiple sub-optimizers for different parameter groups
- Different learning rates and hyperparameters for different parameters
- Unified interface for step, zero_grad, and state_dict operations
- Memory-efficient optimization for large models
- Support for MoE models with expert and dense parameters
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .exceptions import OptimizerError
from .metrics import OptimizerMetrics

logger = logging.getLogger(__name__)


class ChainedOptimizer(Optimizer):
    """Optimizer that chains multiple sub-optimizers for different parameter groups.

    This optimizer allows using different optimization algorithms and hyperparameters
    for different sets of model parameters. It's particularly useful for:
    - MoE models with expert and dense parameters
    - Fine-tuning where different layers need different learning rates
    - Complex models with heterogeneous parameter types

    Args:
        optimizers: List of optimizer instances to chain together.
        enable_metrics: Whether to collect optimization metrics.
        grad_scaler: Optional gradient scaler for mixed precision training.

    Example:
        >>> # Create separate optimizers for different parameter groups
        >>> dense_optimizer = torch.optim.Adam(dense_params, lr=1e-3)
        >>> expert_optimizer = torch.optim.Adam(expert_params, lr=1e-4)
        >>>
        >>> # Chain them together
        >>> optimizer = ChainedOptimizer([dense_optimizer, expert_optimizer])
        >>>
        >>> # Use as a regular optimizer
        >>> loss.backward()
        >>> optimizer.step()
        >>> optimizer.zero_grad()
    """

    def __init__(
        self,
        optimizers: List[Optimizer],
        enable_metrics: bool = False,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
        """Initialize ChainedOptimizer with multiple sub-optimizers."""
        if not optimizers:
            raise ValueError("ChainedOptimizer requires at least one optimizer")

        # Set flag to avoid issues during parent initialization
        self._initializing = True

        self.chained_optimizers = optimizers
        self.enable_metrics = enable_metrics
        self.grad_scaler = grad_scaler

        # Initialize metrics tracking
        self.metrics = OptimizerMetrics() if enable_metrics else None

        # Combine all parameter groups from sub-optimizers
        param_groups_list: List[Dict[str, Any]] = []
        self._param_to_optimizer: Dict[nn.Parameter, Tuple[Optimizer, int]] = {}
        self._optimizer_group_offsets: List[int] = [0]

        for opt_idx, optimizer in enumerate(self.chained_optimizers):
            for group_idx, param_group in enumerate(optimizer.param_groups):
                # Create a shallow copy to avoid modifying the original
                new_group = dict(param_group)
                new_group["_optimizer_idx"] = opt_idx
                new_group["_local_group_idx"] = group_idx
                param_groups_list.append(new_group)

                # Map parameters to their optimizer and local group index
                params = param_group.get("params", [])
                if not isinstance(params, list):
                    params = list(params)

                for param in params:
                    if param in self._param_to_optimizer:
                        raise ValueError(
                            f"Parameter appears in multiple optimizers. "
                            f"Each parameter should belong to exactly one optimizer."
                        )
                    self._param_to_optimizer[param] = (optimizer, group_idx)

            self._optimizer_group_offsets.append(len(param_groups_list))

        # Initialize base optimizer state (required by PyTorch)
        defaults = {
            "lr": 0.0,  # Dummy value, actual LR comes from sub-optimizers
            "chained": True,
        }

        # PyTorch Optimizer requires at least one param group
        # We'll initialize with a dummy group and then replace it
        if param_groups_list:
            # Use first group for initialization, then replace all
            first_group = param_groups_list[0].copy()
            super().__init__([first_group], defaults)
            # Replace with our actual param groups
            self.param_groups = param_groups_list
        else:
            # No parameters at all - create dummy
            dummy_param = torch.zeros(1, requires_grad=True)
            super().__init__([{"params": [dummy_param]}], defaults)
            self.param_groups = []

        # Initialization done
        self._initializing = False

        # Track step count
        self._step_count = 0

        logger.info(
            f"Initialized ChainedOptimizer with {len(self.chained_optimizers)} "
            f"sub-optimizers and {len(self.param_groups)} parameter groups"
        )

    @property
    def chained_optimizers(self) -> List[Optimizer]:
        """Get list of chained optimizers."""
        return self._chained_optimizers

    @chained_optimizers.setter
    def chained_optimizers(self, optimizers: List[Optimizer]) -> None:
        """Set chained optimizers with validation."""
        if not isinstance(optimizers, list):
            raise TypeError("Optimizers must be a list")
        if not all(isinstance(opt, Optimizer) for opt in optimizers):
            raise TypeError("All items must be Optimizer instances")
        self._chained_optimizers = optimizers

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform optimization step on all sub-optimizers.

        Args:
            closure: Optional closure that reevaluates the model and returns loss.

        Returns:
            Loss value if closure is provided, None otherwise.
        """
        loss = None

        # Handle gradient scaling for mixed precision
        if self.grad_scaler is not None:
            # Unscale gradients for all optimizers
            for optimizer in self.chained_optimizers:
                self.grad_scaler.unscale_(optimizer)

        # Track metrics if enabled
        if self.enable_metrics and self.metrics is not None:
            import time

            start_time = time.perf_counter()

        # Step through each optimizer
        for opt_idx, optimizer in enumerate(self.chained_optimizers):
            # Skip if optimizer has no parameters
            if not any(
                group["_optimizer_idx"] == opt_idx for group in self.param_groups
            ):
                continue

            try:
                if closure is not None and opt_idx == 0:
                    # Only evaluate closure once for the first optimizer
                    loss = optimizer.step(closure)
                elif self.grad_scaler is not None:
                    # Use gradient scaler's step method
                    self.grad_scaler.step(optimizer)
                else:
                    optimizer.step()
            except Exception as e:
                raise OptimizerError(f"Error in optimizer {opt_idx}: {str(e)}") from e

        # Update gradient scaler if used
        if self.grad_scaler is not None:
            self.grad_scaler.update()

        # Update metrics
        if self.enable_metrics and self.metrics is not None:
            end_time = time.perf_counter()
            self.metrics.total_step_time = end_time - start_time
            self.metrics.parameter_update_time = end_time - start_time

        self._step_count += 1

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients for all parameters in all sub-optimizers.

        Args:
            set_to_none: Whether to set gradients to None instead of zero.
        """
        for optimizer in self.chained_optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict containing all sub-optimizer states.

        Returns:
            Dictionary containing:
            - 'state': Combined state from all optimizers
            - 'param_groups': Combined parameter groups
            - 'optimizer_states': Individual optimizer state dicts
            - 'step_count': Number of optimization steps
        """
        # Collect individual optimizer states
        optimizer_states = []
        for opt_idx, optimizer in enumerate(self.chained_optimizers):
            opt_state = optimizer.state_dict()
            optimizer_states.append(opt_state)

        # Build combined state mapping
        combined_state = {}
        param_id_mapping: Dict[int, Dict[Any, str]] = {}

        for opt_idx, opt_state_dict in enumerate(optimizer_states):
            opt_state = opt_state_dict.get("state", {})

            for param_id, param_state in opt_state.items():
                # Create unique ID for parameter across optimizers
                global_param_id = f"opt_{opt_idx}_param_{param_id}"
                combined_state[global_param_id] = param_state

                # Track mapping for loading
                if opt_idx not in param_id_mapping:
                    param_id_mapping[opt_idx] = {}
                param_id_mapping[opt_idx][param_id] = global_param_id

        state_dict = {
            "state": combined_state,
            "param_groups": self.param_groups,
            "optimizer_states": optimizer_states,
            "param_id_mapping": param_id_mapping,
            "step_count": self._step_count,
            "num_optimizers": len(self.chained_optimizers),
        }

        # Add metrics if enabled
        if self.enable_metrics and self.metrics is not None:
            state_dict["metrics"] = self.metrics.to_dict()

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state from state dict.

        Args:
            state_dict: State dictionary from state_dict() method.
        """
        if "optimizer_states" not in state_dict:
            raise ValueError("Invalid state_dict: missing 'optimizer_states'")

        optimizer_states = state_dict["optimizer_states"]

        # Validate number of optimizers
        if len(optimizer_states) != len(self.chained_optimizers):
            raise ValueError(
                f"State dict has {len(optimizer_states)} optimizers but "
                f"ChainedOptimizer has {len(self.chained_optimizers)}"
            )

        # Load individual optimizer states
        for opt_idx, (optimizer, opt_state) in enumerate(
            zip(self.chained_optimizers, optimizer_states)
        ):
            try:
                optimizer.load_state_dict(opt_state)
            except Exception as e:
                raise OptimizerError(
                    f"Error loading state for optimizer {opt_idx}: {str(e)}"
                ) from e

        # Update parameter groups if provided
        if "param_groups" in state_dict:
            self.param_groups = state_dict["param_groups"]

        # Restore step count
        self._step_count = state_dict.get("step_count", 0)

        # Restore metrics if available
        if self.enable_metrics and self.metrics is not None:
            if "metrics" in state_dict:
                # Metrics are read-only summaries, just log them
                logger.info(f"Loaded metrics: {state_dict['metrics']}")

        logger.info(
            f"Loaded ChainedOptimizer state with {len(self.chained_optimizers)} "
            f"optimizers at step {self._step_count}"
        )

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a parameter group to the appropriate sub-optimizer.

        Args:
            param_group: Parameter group dictionary with optimizer index.

        Raises:
            ValueError: If optimizer index is invalid or missing.
        """
        # During initialization, skip adding param groups as they're
        # already in sub-optimizers
        if getattr(self, "_initializing", False):
            return

        if "_optimizer_idx" not in param_group:
            raise ValueError(
                "Parameter group must specify '_optimizer_idx' to indicate "
                "which sub-optimizer it belongs to"
            )

        opt_idx = param_group["_optimizer_idx"]
        if opt_idx >= len(self.chained_optimizers):
            raise ValueError(
                f"Invalid optimizer index {opt_idx}. "
                f"ChainedOptimizer has {len(self.chained_optimizers)} optimizers"
            )

        # Remove internal fields before adding to sub-optimizer
        clean_group = {k: v for k, v in param_group.items() if not k.startswith("_")}

        # Add to sub-optimizer
        self.chained_optimizers[opt_idx].add_param_group(clean_group)

        # Update our parameter groups
        local_group_idx = len(self.chained_optimizers[opt_idx].param_groups) - 1
        param_group["_local_group_idx"] = local_group_idx
        self.param_groups.append(param_group)

        # Update parameter mappings
        params = param_group.get("params", [])
        if not isinstance(params, list):
            params = list(params)

        for param in params:
            if param in self._param_to_optimizer:
                raise ValueError(
                    f"Parameter already exists in optimizer. "
                    f"Each parameter should belong to exactly one optimizer."
                )
            self._param_to_optimizer[param] = (
                self.chained_optimizers[opt_idx],
                local_group_idx,
            )

    def get_lr(self) -> List[float]:
        """Get learning rates from all parameter groups.

        Returns:
            List of learning rates for each parameter group.
        """
        lrs = []
        for optimizer in self.chained_optimizers:
            for group in optimizer.param_groups:
                lrs.append(group.get("lr", 0.0))
        return lrs

    def set_lr(self, lr: Union[float, List[float]]) -> None:
        """Set learning rates for all or specific parameter groups.

        Args:
            lr: Single learning rate for all groups or list of learning rates.
        """
        if isinstance(lr, (int, float)):
            # Set same LR for all groups
            for optimizer in self.chained_optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = float(lr)
        else:
            # Set individual LRs
            if len(lr) != len(self.param_groups):
                raise ValueError(
                    f"Expected {len(self.param_groups)} learning rates, "
                    f"got {len(lr)}"
                )

            idx = 0
            for optimizer in self.chained_optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = float(lr[idx])
                    idx += 1

    def get_optimizer_for_param(
        self, param: nn.Parameter
    ) -> Optional[Tuple[Optimizer, int]]:
        """Get the optimizer and group index for a specific parameter.

        Args:
            param: Parameter to look up.

        Returns:
            Tuple of (optimizer, group_index) or None if not found.
        """
        return self._param_to_optimizer.get(param)

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get optimization metrics if enabled.

        Returns:
            Metrics dictionary or None if metrics are disabled.
        """
        if self.enable_metrics and self.metrics is not None:
            metrics_dict = self.metrics.to_dict()
            metrics_dict["total_steps"] = self._step_count
            return metrics_dict
        return None

    def __repr__(self) -> str:
        """String representation of ChainedOptimizer."""
        optimizer_info = []
        for i, opt in enumerate(self.chained_optimizers):
            num_params = sum(
                len(g["params"]) if isinstance(g["params"], list) else 1
                for g in opt.param_groups
            )
            optimizer_info.append(
                f"  [{i}] {opt.__class__.__name__}: {num_params} parameters"
            )

        info_str = "\n".join(optimizer_info)
        return (
            f"ChainedOptimizer(\n"
            f"  num_optimizers={len(self.chained_optimizers)},\n"
            f"  total_param_groups={len(self.param_groups)},\n"
            f"  optimizers=[\n{info_str}\n  ]\n"
            f")"
        )
