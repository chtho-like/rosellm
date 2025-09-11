"""ChainedOptimizer for managing multiple optimizers with different parameter groups.

This module provides ChainedOptimizer which enables:
- Managing multiple sub-optimizers for different parameter groups
- Different learning rates and hyperparameters for different parameters
- Unified interface for step, zero_grad, and state_dict operations
- Memory-efficient optimization for large models
- Support for MoE models with expert and dense parameters
- Per-optimizer gradient clipping (norm and value)
- Thread-safe operations for concurrent training
- Parameter freezing/unfreezing for fine-tuning
- Comprehensive optimizer statistics and metrics

Key Features:
    - **Multi-Optimizer Management**: Chain multiple optimizers with different
      algorithms (Adam, SGD, etc.) for different parameter groups
    - **Gradient Clipping**: Support for both norm and value clipping,
      configurable per optimizer or globally
    - **Thread Safety**: Optional thread-safe mode for multi-threaded training
    - **Mixed Precision**: Full support for automatic mixed precision training
    - **State Management**: Efficient state dict serialization and loading
    - **Metrics Collection**: Optional performance metrics tracking

Performance Optimizations:
    - Lazy parameter group validation
    - Cached optimizer parameter mappings
    - Efficient gradient clipping application
    - Minimal overhead when features are disabled
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, overload

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
        thread_safe: bool = False,
        grad_clip_norm: Optional[Union[float, List[Optional[float]]]] = None,
        grad_clip_value: Optional[Union[float, List[Optional[float]]]] = None,
    ):
        """Initialize ChainedOptimizer with multiple sub-optimizers.

        Args:
            optimizers: List of optimizer instances to chain together.
            enable_metrics: Whether to collect optimization metrics.
            grad_scaler: Optional gradient scaler for mixed precision training.
            thread_safe: Whether to enable thread-safe operations.
            grad_clip_norm: Max norm for gradient clipping (per optimizer or global).
            grad_clip_value: Max value for gradient clipping (per optimizer or global).
        """
        if not optimizers:
            raise ValueError("ChainedOptimizer requires at least one optimizer")

        # Validate optimizer types
        for i, opt in enumerate(optimizers):
            if not isinstance(opt, Optimizer):
                raise TypeError(
                    f"Item at index {i} is not an Optimizer instance, "
                    f"got {type(opt).__name__}"
                )

        # Set flag to avoid issues during parent initialization
        self._initializing = True

        # Thread safety
        self._thread_safe = thread_safe
        self._lock = threading.RLock() if thread_safe else None

        self.chained_optimizers = optimizers
        self.enable_metrics = enable_metrics
        self.grad_scaler = grad_scaler

        # Gradient clipping configuration
        self._setup_gradient_clipping(grad_clip_norm, grad_clip_value)

        # Initialize metrics tracking
        self.metrics = OptimizerMetrics() if enable_metrics else None

        # Combine all parameter groups from sub-optimizers
        param_groups_list: List[Dict[str, Any]] = []
        self._param_to_optimizer: Dict[nn.Parameter, Tuple[Optimizer, int]] = {}
        self._optimizer_group_offsets: List[int] = [0]
        self._seen_params: Set[int] = set()  # Track parameter IDs for validation

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
                    param_id = id(param)
                    if param_id in self._seen_params:
                        raise ValueError(
                            f"Parameter appears in multiple optimizers. "
                            f"Each parameter should belong to exactly one optimizer."
                        )
                    self._seen_params.add(param_id)
                    self._param_to_optimizer[param] = (optimizer, group_idx)

                    # Validate parameter requires gradients
                    if not param.requires_grad:
                        logger.warning(
                            f"Parameter in optimizer does not require gradients. "
                            f"This may lead to unexpected behavior."
                        )

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

        # Performance optimization: cache frequently accessed data
        self._param_cache: Optional[Dict[int, List[nn.Parameter]]] = None
        self._param_count_cache: Optional[int] = None

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

    def _setup_gradient_clipping(
        self,
        grad_clip_norm: Optional[Union[float, List[Optional[float]]]],
        grad_clip_value: Optional[Union[float, List[Optional[float]]]],
    ) -> None:
        """Setup gradient clipping configuration."""
        self.grad_clip_norm: Optional[List[Optional[float]]] = None
        self.grad_clip_value: Optional[List[Optional[float]]] = None

        if grad_clip_norm is not None:
            if isinstance(grad_clip_norm, (int, float)):
                # Same norm for all optimizers
                norm_val: Optional[float] = float(grad_clip_norm)
                self.grad_clip_norm = [
                    norm_val for _ in range(len(self.chained_optimizers))
                ]
            else:
                # Per-optimizer norms
                if len(grad_clip_norm) != len(self.chained_optimizers):
                    raise ValueError(
                        f"grad_clip_norm list length ({len(grad_clip_norm)}) "
                        f"must match number of optimizers "
                        f"({len(self.chained_optimizers)})"
                    )
                self.grad_clip_norm = [
                    float(v) if v is not None else None for v in grad_clip_norm
                ]

        if grad_clip_value is not None:
            if isinstance(grad_clip_value, (int, float)):
                # Same value for all optimizers
                val: Optional[float] = float(grad_clip_value)
                self.grad_clip_value = [
                    val for _ in range(len(self.chained_optimizers))
                ]
            else:
                # Per-optimizer values
                if len(grad_clip_value) != len(self.chained_optimizers):
                    raise ValueError(
                        f"grad_clip_value list length "
                        f"({len(grad_clip_value)}) must match number of "
                        f"optimizers ({len(self.chained_optimizers)})"
                    )
                self.grad_clip_value = [
                    float(v) if v is not None else None for v in grad_clip_value
                ]

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform optimization step on all sub-optimizers.

        Args:
            closure: Optional closure that reevaluates the model and returns loss.

        Returns:
            Loss value if closure is provided, None otherwise.
        """
        if self._thread_safe and self._lock:
            with self._lock:
                return self._step_impl(closure)
        return self._step_impl(closure)

    def _step_impl(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """Internal implementation of step method."""
        loss = None

        # Track metrics if enabled
        start_time = time.perf_counter() if self.enable_metrics else 0.0

        # Handle gradient scaling for mixed precision
        if self.grad_scaler is not None:
            # Unscale gradients for all optimizers before stepping
            for optimizer in self.chained_optimizers:
                self.grad_scaler.unscale_(optimizer)

        # Apply gradient clipping if configured
        self._apply_gradient_clipping()

        # Step through each optimizer
        if self.grad_scaler is not None:
            # Step with gradient scaler
            for opt_idx, optimizer in enumerate(self.chained_optimizers):
                # Skip if optimizer has no parameters
                if not self._optimizer_has_params(opt_idx):
                    continue

                try:
                    if closure is not None and opt_idx == 0:
                        # Only evaluate closure once for the first optimizer
                        loss = optimizer.step(closure)
                    else:
                        # Use gradient scaler's step method
                        self.grad_scaler.step(optimizer)
                except Exception as e:
                    raise OptimizerError(
                        f"Error in optimizer {opt_idx} "
                        f"({optimizer.__class__.__name__}): "
                        f"{str(e)}"
                    ) from e

            # Update gradient scaler after all optimizers have stepped
            self.grad_scaler.update()
        else:
            # Normal stepping without gradient scaler
            for opt_idx, optimizer in enumerate(self.chained_optimizers):
                # Skip if optimizer has no parameters
                if not self._optimizer_has_params(opt_idx):
                    continue

                try:
                    if closure is not None and opt_idx == 0:
                        # Only evaluate closure once for the first optimizer
                        loss = optimizer.step(closure)
                    else:
                        optimizer.step()
                except Exception as e:
                    raise OptimizerError(
                        f"Error in optimizer {opt_idx} "
                        f"({optimizer.__class__.__name__}): "
                        f"{str(e)}"
                    ) from e

        # Update metrics
        if self.enable_metrics and self.metrics is not None:
            end_time = time.perf_counter()
            self.metrics.total_step_time = end_time - start_time
            self.metrics.parameter_update_time = end_time - start_time

        self._step_count += 1
        return loss

    def _apply_gradient_clipping(self) -> None:
        """Apply gradient clipping to each optimizer's parameters."""
        if self.grad_clip_norm is None and self.grad_clip_value is None:
            return

        for opt_idx, optimizer in enumerate(self.chained_optimizers):
            # Use cached parameters if available
            params = self._get_optimizer_params(opt_idx)

            if not params:
                continue

            # Apply norm clipping
            if self.grad_clip_norm is not None and opt_idx < len(self.grad_clip_norm):
                clip_norm = self.grad_clip_norm[opt_idx]
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, clip_norm)

            # Apply value clipping
            if self.grad_clip_value is not None and opt_idx < len(self.grad_clip_value):
                clip_value = self.grad_clip_value[opt_idx]
                if clip_value is not None:
                    torch.nn.utils.clip_grad_value_(params, clip_value)

    def _get_optimizer_params(self, opt_idx: int) -> List[nn.Parameter]:
        """Get cached parameters for an optimizer.

        Args:
            opt_idx: Optimizer index.

        Returns:
            List of parameters for the optimizer.
        """
        # Initialize cache if needed
        if self._param_cache is None:
            self._param_cache = {}

        # Return cached value if available
        if opt_idx in self._param_cache:
            return self._param_cache[opt_idx]

        # Build parameter list and cache it
        params = []
        optimizer = self.chained_optimizers[opt_idx]
        for group in optimizer.param_groups:
            params.extend(group["params"])

        self._param_cache[opt_idx] = params
        return params

    def _invalidate_cache(self) -> None:
        """Invalidate cached data when optimizer configuration changes."""
        self._param_cache = None
        self._param_count_cache = None

    def _optimizer_has_params(self, opt_idx: int) -> bool:
        """Check if optimizer has any parameter groups."""
        return any(
            group.get("_optimizer_idx") == opt_idx for group in self.param_groups
        )

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients for all parameters in all sub-optimizers.

        Args:
            set_to_none: Whether to set gradients to None instead of zero.
        """
        if self._thread_safe and self._lock:
            with self._lock:
                for optimizer in self.chained_optimizers:
                    optimizer.zero_grad(set_to_none=set_to_none)
        else:
            for optimizer in self.chained_optimizers:
                optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict containing all sub-optimizer states.

        Returns:
            Dictionary containing:
            - 'optimizer_states': Individual optimizer state dicts
            - 'param_groups': Combined parameter groups
            - 'step_count': Number of optimization steps
            - 'version': State dict version for compatibility
        """
        if self._thread_safe and self._lock:
            with self._lock:
                return self._state_dict_impl()
        return self._state_dict_impl()

    def _state_dict_impl(self) -> Dict[str, Any]:
        """Internal implementation of state_dict."""
        # Collect individual optimizer states (more memory efficient)
        optimizer_states = [
            optimizer.state_dict() for optimizer in self.chained_optimizers
        ]

        state_dict = {
            "optimizer_states": optimizer_states,
            "param_groups": self.param_groups.copy(),  # Shallow copy for safety
            "step_count": self._step_count,
            "num_optimizers": len(self.chained_optimizers),
            "version": 2,  # Version for future compatibility
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
        if self._thread_safe and self._lock:
            with self._lock:
                self._load_state_dict_impl(state_dict)
        else:
            self._load_state_dict_impl(state_dict)

    def _load_state_dict_impl(self, state_dict: Dict[str, Any]) -> None:
        """Internal implementation of load_state_dict."""
        if "optimizer_states" not in state_dict:
            raise ValueError("Invalid state_dict: missing 'optimizer_states'")

        optimizer_states = state_dict["optimizer_states"]

        # Check version for backward compatibility
        version = state_dict.get("version", 1)
        if version > 2:
            logger.warning(
                f"Loading state dict from newer version {version}, "
                f"current version is 2. Some features may not work correctly."
            )

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
                    f"Error loading state for optimizer {opt_idx} "
                    f"({optimizer.__class__.__name__}): {str(e)}"
                ) from e

        # Update parameter groups if provided
        if "param_groups" in state_dict:
            self.param_groups = state_dict["param_groups"].copy()

        # Restore step count
        self._step_count = state_dict.get("step_count", 0)

        # Restore metrics if available
        if self.enable_metrics and self.metrics is not None:
            if "metrics" in state_dict:
                # Metrics are read-only summaries, just log them
                logger.debug(f"Loaded metrics: {state_dict['metrics']}")

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

        # Invalidate caches since configuration changed
        self._invalidate_cache()

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

    def freeze_optimizer(self, optimizer_idx: int) -> None:
        """Freeze all parameters managed by a specific optimizer.

        Args:
            optimizer_idx: Index of optimizer whose parameters to freeze.
        """
        if optimizer_idx >= len(self.chained_optimizers):
            raise ValueError(f"Invalid optimizer index {optimizer_idx}")

        optimizer = self.chained_optimizers[optimizer_idx]
        for group in optimizer.param_groups:
            for param in group["params"]:
                param.requires_grad = False

        logger.info(f"Froze all parameters in optimizer {optimizer_idx}")

    def unfreeze_optimizer(self, optimizer_idx: int) -> None:
        """Unfreeze all parameters managed by a specific optimizer.

        Args:
            optimizer_idx: Index of optimizer whose parameters to unfreeze.
        """
        if optimizer_idx >= len(self.chained_optimizers):
            raise ValueError(f"Invalid optimizer index {optimizer_idx}")

        optimizer = self.chained_optimizers[optimizer_idx]
        for group in optimizer.param_groups:
            for param in group["params"]:
                param.requires_grad = True

        logger.info(f"Unfroze all parameters in optimizer {optimizer_idx}")

    def get_optimizer_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for each optimizer.

        Returns:
            Dictionary mapping optimizer index to statistics.
        """
        stats = {}
        for opt_idx, optimizer in enumerate(self.chained_optimizers):
            opt_stats: Dict[str, Any] = {
                "type": optimizer.__class__.__name__,
                "num_param_groups": len(optimizer.param_groups),
                "total_params": 0,
                "trainable_params": 0,
                "learning_rates": [],
            }

            for group in optimizer.param_groups:
                params = group["params"]
                opt_stats["learning_rates"].append(group.get("lr", 0.0))
                for param in params:
                    opt_stats["total_params"] += param.numel()
                    if param.requires_grad:
                        opt_stats["trainable_params"] += param.numel()

            stats[opt_idx] = opt_stats

        return stats

    def get_total_param_count(self) -> int:
        """Get total number of parameters across all optimizers.

        Returns:
            Total parameter count.
        """
        if self._param_count_cache is not None:
            return self._param_count_cache

        total = 0
        for optimizer in self.chained_optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    total += param.numel()

        self._param_count_cache = total
        return total

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get optimization metrics if enabled.

        Returns:
            Metrics dictionary or None if metrics are disabled.
        """
        if self.enable_metrics and self.metrics is not None:
            if self._thread_safe and self._lock:
                with self._lock:
                    metrics_dict = self.metrics.to_dict()
                    metrics_dict["total_steps"] = self._step_count
                    return metrics_dict
            else:
                metrics_dict = self.metrics.to_dict()
                metrics_dict["total_steps"] = self._step_count
                return metrics_dict
        return None

    def __getstate__(self) -> Dict[str, Any]:
        """Get state for pickling."""
        state = self.__dict__.copy()
        # Remove the lock as it can't be pickled
        if "_lock" in state:
            del state["_lock"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set state for unpickling."""
        self.__dict__.update(state)
        # Recreate the lock if needed
        if self._thread_safe:
            self._lock = threading.RLock()
        else:
            self._lock = None

    def __repr__(self) -> str:
        """String representation of ChainedOptimizer."""
        try:
            optimizer_info = []
            for i, opt in enumerate(self.chained_optimizers):
                try:
                    num_params = sum(
                        (
                            len(g.get("params", []))
                            if isinstance(g.get("params", []), (list, tuple))
                            else sum(1 for _ in g.get("params", []))
                        )
                        for g in opt.param_groups
                    )
                    optimizer_info.append(
                        f"  [{i}] {opt.__class__.__name__}: {num_params} parameters"
                    )
                except Exception:
                    # Fallback if param counting fails
                    optimizer_info.append(
                        f"  [{i}] {opt.__class__.__name__}: <unknown> parameters"
                    )

            info_str = "\n".join(optimizer_info)
            return (
                f"ChainedOptimizer(\n"
                f"  num_optimizers={len(self.chained_optimizers)},\n"
                f"  total_param_groups={len(self.param_groups)},\n"
                f"  thread_safe={self._thread_safe},\n"
                f"  step_count={self._step_count},\n"
                f"  optimizers=[\n{info_str}\n  ]\n"
                f")"
            )
        except Exception:
            # Ultimate fallback
            return f"ChainedOptimizer(num_optimizers={len(self.chained_optimizers)})"
