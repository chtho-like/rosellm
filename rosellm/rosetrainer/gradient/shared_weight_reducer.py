"""Shared weight gradient reduction for tied embeddings.

This module implements gradient synchronization for shared weights between
input embeddings and output layers, following Megatron-LM's pattern.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from rosellm.rosetrainer.parallelism import parallel_state

try:
    from .reduction_strategies import ReductionStrategyBase, create_reduction_strategy
except ImportError:
    # Fallback if strategies module is not available
    ReductionStrategyBase = None  # type: ignore
    create_reduction_strategy = None  # type: ignore

try:
    from .security_utils import (
        GradientSecurityValidator,
        InputValidator,
        create_security_validator,
    )
except ImportError:
    # Fallback if security module is not available
    GradientSecurityValidator = None  # type: ignore
    InputValidator = None  # type: ignore
    create_security_validator = None  # type: ignore

logger = logging.getLogger(__name__)


class ReductionStrategy(Enum):
    """Strategy for gradient reduction."""

    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    HIERARCHICAL = "hierarchical"


@dataclass
class ReductionMetrics:
    """Metrics for gradient reduction operations."""

    total_bytes_reduced: int = 0
    reduction_time_ms: float = 0.0
    num_parameters_reduced: int = 0
    gradient_norm_before: float = 0.0
    gradient_norm_after: float = 0.0
    overflow_detected: bool = False


class SharedWeightGradientReducer:
    """Manages gradient reduction for shared weights across pipeline stages.

    This class handles the synchronization of gradients for parameters that are
    shared between different pipeline stages, particularly input embeddings and
    output projection weights.

    Key features:
    - All-reduce gradients across embedding group for tied weights
    - Support for position embeddings shared across encoder/decoder
    - Efficient coalesced communication for multiple shared parameters
    - Compatibility with mixed precision training

    Args:
        config: Configuration object with gradient settings
    """

    def __init__(self, config):
        self.config = config
        self.timers = getattr(config, "timers", None)

        # Cache process groups to avoid repeated lookups
        self._pp_group: Optional[torch.distributed.ProcessGroup] = None
        self._embd_group: Optional[torch.distributed.ProcessGroup] = None
        self._pos_embd_group: Optional[torch.distributed.ProcessGroup] = None

        # Cache for process group membership checks
        self._group_membership_cache: Dict[int, Set[int]] = {}

        # Performance tracking
        self._reduction_metrics: ReductionMetrics = ReductionMetrics()

        # Configuration
        self.max_gradient_norm: float = getattr(config, "max_gradient_norm", 1e10)
        self.check_for_nan: bool = getattr(config, "check_for_nan", True)
        self.reduction_strategy = ReductionStrategy.ALL_REDUCE

        # Safety checks
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.max_gradient_norm <= 0:
            raise ValueError(
                f"max_gradient_norm must be positive, got {self.max_gradient_norm}"
            )

        if hasattr(self.config, "reduction_strategy"):
            strategy = getattr(self.config, "reduction_strategy")
            if strategy not in [s.value for s in ReductionStrategy]:
                raise ValueError(f"Invalid reduction strategy: {strategy}")

    def _get_current_rank_safe(self) -> int:
        """Get current rank safely, handling uninitialized distributed.

        Returns:
            Current rank or 0 if not distributed.
        """
        if not dist.is_initialized():
            return 0
        return int(dist.get_rank())

    def _unwrap_model(self, model: nn.Module, max_depth: int = 3) -> nn.Module:
        """Recursively unwrap DDP and other wrappers.

        Args:
            model: Potentially wrapped model
            max_depth: Maximum unwrapping depth

        Returns:
            Unwrapped model
        """
        depth = 0
        while depth < max_depth:
            if hasattr(model, "module"):
                unwrapped = getattr(model, "module")
                if isinstance(unwrapped, nn.Module):
                    model = unwrapped
                    depth += 1
                else:
                    break
            else:
                break
        return model

    def _is_in_position_embedding_group(self) -> bool:
        """Check if current rank is in position embedding group.

        Returns:
            True if in position embedding group, False otherwise.
        """
        if self.pos_embd_group is None:
            return False

        # Check cache first
        group_id = id(self.pos_embd_group)
        if group_id in self._group_membership_cache:
            current_rank = self._get_current_rank_safe()
            return current_rank in self._group_membership_cache[group_id]

        # Compute and cache membership
        try:
            if not dist.is_initialized():
                return False

            current_rank = dist.get_rank()
            group_ranks = set(dist.get_process_group_ranks(self.pos_embd_group))
            self._group_membership_cache[group_id] = group_ranks
            return current_rank in group_ranks
        except Exception as e:
            logger.warning(f"Failed to check position embedding group membership: {e}")
            return False

    @property
    def pp_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Get pipeline parallel process group."""
        if self._pp_group is None:
            self._pp_group = parallel_state.get_pipeline_model_parallel_group()
        return self._pp_group

    @property
    def embd_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Get embedding process group."""
        if self._embd_group is None:
            # Use pipeline parallel group for embedding synchronization
            # In a full Megatron-LM implementation, this would be a dedicated group
            # that connects first and last pipeline stages
            self._embd_group = parallel_state.get_pipeline_model_parallel_group()
        return self._embd_group

    @property
    def pos_embd_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Get position embedding process group."""
        if self._pos_embd_group is None:
            # Use pipeline parallel group for position embedding synchronization
            # This would typically be a specialized group in full implementations
            self._pos_embd_group = parallel_state.get_pipeline_model_parallel_group()
        return self._pos_embd_group

    def _get_main_grad_attr(self, param: nn.Parameter) -> str:
        """Get the attribute name for the main gradient.

        Args:
            param: Parameter to check

        Returns:
            'main_grad' if using mixed precision, otherwise 'grad'
        """
        if hasattr(param, "main_grad"):
            return "main_grad"
        return "grad"

    def _is_first_stage(self) -> bool:
        """Check if current rank is first pipeline stage."""
        # Check if pipeline parallel is initialized
        if not parallel_state.is_initialized():
            return True

        # Use the correct function name from parallel_state
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        return pp_rank == 0

    def _is_last_stage(self) -> bool:
        """Check if current rank is last pipeline stage."""
        # Check if pipeline parallel is initialized
        if not parallel_state.is_initialized():
            return True

        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_size = parallel_state.get_pipeline_model_parallel_size()
        return pp_rank == pp_size - 1

    def _get_process_group_size(
        self, group: Optional[torch.distributed.ProcessGroup]
    ) -> int:
        """Get size of process group, handling None case.

        Args:
            group: Process group or None

        Returns:
            Size of group, or 1 if group is None
        """
        if group is None:
            return 1

        try:
            if not dist.is_initialized():
                return 1
            return int(dist.get_world_size(group=group))
        except Exception as e:
            logger.warning(f"Failed to get process group size: {e}")
            return 1

    def _is_in_embedding_group(self) -> bool:
        """Check if current rank is in the embedding group.

        Returns:
            True if current rank is in embedding group, False otherwise.

        Note:
            Uses caching to avoid repeated expensive group membership checks.
        """
        if self.embd_group is None:
            return False

        # Check cache first
        group_id = id(self.embd_group)
        if group_id in self._group_membership_cache:
            current_rank = self._get_current_rank_safe()
            return current_rank in self._group_membership_cache[group_id]

        # Compute and cache membership
        try:
            if not dist.is_initialized():
                return False

            current_rank = dist.get_rank()
            group_ranks = set(dist.get_process_group_ranks(self.embd_group))
            self._group_membership_cache[group_id] = group_ranks
            return current_rank in group_ranks
        except Exception as e:
            logger.warning(f"Failed to check embedding group membership: {e}")
            return False

    def _get_world_size_safe(self) -> int:
        """Get world size safely.

        Returns:
            World size or 1 if not distributed.
        """
        if not dist.is_initialized():
            return 1
        return int(dist.get_world_size())

    def allreduce_word_embedding_grads(
        self,
        model: List[nn.Module],
        get_embedding_weight: Optional[
            Callable[[nn.Module], Optional[nn.Parameter]]
        ] = None,
    ) -> None:
        """All-reduce word embedding gradients across first and last pipeline stages.

        This ensures that word_embeddings parameters stay in sync when they are
        shared between the input and output layers.

        Args:
            model: List of model chunks (including virtual pipeline chunks)
            get_embedding_weight: Function to extract embedding weight from model.
                                If None, uses default extraction logic.
        """
        # Skip if embedding group is not initialized or too small
        if self._get_process_group_size(self.embd_group) <= 1:
            return

        # Skip if not in embedding group
        if not self._is_in_embedding_group():
            return

        # Start timer if available
        if self.timers is not None:
            self.timers("embedding-grads-all-reduce", log_level=1).start()

        try:
            self._allreduce_embedding_grad(
                model=model,
                weight_getter=get_embedding_weight
                or self._default_get_word_embedding_weight,
                embd_group=self.embd_group,
                skip_if_none=True,
            )
        finally:
            if self.timers is not None:
                self.timers("embedding-grads-all-reduce").stop()

    def allreduce_position_embedding_grads(
        self,
        model: List[nn.Module],
        get_position_weight: Optional[
            Callable[[nn.Module], Optional[nn.Parameter]]
        ] = None,
    ) -> None:
        """All-reduce position embedding gradients across encoder and decoder stages.

        This ensures position embeddings parameters stay in sync across stages.

        Args:
            model: List of model chunks
            get_position_weight: Function to extract position embedding weight
        """
        # Skip if position embedding group is not initialized or too small
        if self._get_process_group_size(self.pos_embd_group) <= 1:
            return

        # Skip if not in position embedding group
        if not self._is_in_position_embedding_group():
            return

        # Start timer if available
        if self.timers is not None:
            self.timers("position-embedding-grads-all-reduce", log_level=1).start()

        try:
            self._allreduce_embedding_grad(
                model=model,
                weight_getter=get_position_weight
                or self._default_get_position_embedding_weight,
                embd_group=self.pos_embd_group,
                skip_if_none=False,
            )
        finally:
            if self.timers is not None:
                self.timers("position-embedding-grads-all-reduce").stop()

    def _allreduce_embedding_grad(
        self,
        model: List[nn.Module],
        weight_getter: Callable[[nn.Module], Optional[nn.Parameter]],
        embd_group: Optional[torch.distributed.ProcessGroup],
        skip_if_none: bool = True,
    ) -> None:
        """Unified helper to all-reduce embedding parameters across pipeline stages.

        Args:
            model: List of model chunks (PP/VPP)
            weight_getter: Function that takes the pre-process model chunk and
                          returns the parameter to be reduced
            embd_group: The process group over which to reduce
            skip_if_none: If True, quietly returns when parameter or gradient is None

        Raises:
            ValueError: If weight is None and skip_if_none is False
            RuntimeError: If gradient reduction fails
        """
        try:
            # Determine which model chunk to use based on pipeline stage
            if self._is_first_stage():
                model_module = model[0]
            elif self._is_last_stage():
                model_module = model[-1]
            else:
                # For intermediate stages with shared embeddings (encoder-decoder)
                model_module = model[0]

            # Robust DDP unwrapping with multiple levels
            model_module = self._unwrap_model(model_module)

            # Get the weight parameter using the provided getter
            weight = weight_getter(model_module)
            if weight is None:
                if skip_if_none:
                    return
                raise ValueError("Expected weight parameter but got None")

            # Validate parameter
            if not isinstance(weight, nn.Parameter):
                logger.warning(f"Weight is not a Parameter: {type(weight)}")
                return

            # Get gradient attribute and value
            grad_attr = self._get_main_grad_attr(weight)
            grad = getattr(weight, grad_attr, None)

            # Skip if gradient is None (e.g., frozen embedding)
            if grad is None:
                if skip_if_none:
                    return
                raise ValueError(f"No gradient found for weight (attr: {grad_attr})")

            # Validate gradient before reduction
            if self.check_for_nan and torch.isnan(grad).any():
                logger.error("NaN detected in gradient before reduction")
                self._reduction_metrics.overflow_detected = True
                return

            # Check gradient norm
            grad_norm_before = grad.norm().item()
            if grad_norm_before > self.max_gradient_norm:
                logger.warning(f"Large gradient norm detected: {grad_norm_before:.2f}")
                # Optionally clip gradient
                grad = grad * (self.max_gradient_norm / grad_norm_before)

            # Perform all-reduce on the gradient if group exists
            if embd_group is not None:
                import time

                start_time = time.perf_counter()

                dist.all_reduce(grad, group=embd_group)

                # Track metrics
                self._reduction_metrics.reduction_time_ms += (
                    time.perf_counter() - start_time
                ) * 1000
                self._reduction_metrics.total_bytes_reduced += (
                    grad.numel() * grad.element_size()
                )
                self._reduction_metrics.num_parameters_reduced += 1
                self._reduction_metrics.gradient_norm_before = grad_norm_before
                self._reduction_metrics.gradient_norm_after = grad.norm().item()

            # Re-assign gradient (important for autograd graph)
            setattr(weight, grad_attr, grad)

        except Exception as e:
            logger.error(f"Failed to all-reduce embedding gradient: {e}")
            if not skip_if_none:
                raise RuntimeError(f"Gradient reduction failed: {e}") from e

    def _default_get_word_embedding_weight(
        self, model_module: nn.Module
    ) -> Optional[nn.Parameter]:
        """Default method to get word embedding weight.

        Args:
            model_module: Model module to extract from

        Returns:
            Word embedding weight parameter or None
        """
        # Check for shared_embedding_or_output_weight method (Megatron-LM pattern)
        if hasattr(model_module, "shared_embedding_or_output_weight"):
            method = getattr(model_module, "shared_embedding_or_output_weight")
            if callable(method):
                weight = method()
                if isinstance(weight, nn.Parameter):
                    return weight
            return None

        # Check for explicit embedding attribute
        if hasattr(model_module, "word_embeddings"):
            word_embeds = getattr(model_module, "word_embeddings")
            if isinstance(word_embeds, nn.Module) and hasattr(word_embeds, "weight"):
                weight = getattr(word_embeds, "weight")
                if isinstance(weight, nn.Parameter):
                    return weight

        # Check for embedding layer
        if hasattr(model_module, "embedding"):
            embedding = getattr(model_module, "embedding")
            if isinstance(embedding, nn.Module):
                if hasattr(embedding, "word_embeddings"):
                    word_embeds = getattr(embedding, "word_embeddings")
                    if isinstance(word_embeds, nn.Module) and hasattr(
                        word_embeds, "weight"
                    ):
                        weight = getattr(word_embeds, "weight")
                        if isinstance(weight, nn.Parameter):
                            return weight
                if hasattr(embedding, "weight"):
                    weight = getattr(embedding, "weight")
                    if isinstance(weight, nn.Parameter):
                        return weight

        return None

    def _default_get_position_embedding_weight(
        self, model_module: nn.Module
    ) -> Optional[nn.Parameter]:
        """Default method to get position embedding weight.

        Args:
            model_module: Model module to extract from

        Returns:
            Position embedding weight parameter or None
        """
        # Check for position_embeddings attribute
        if hasattr(model_module, "position_embeddings"):
            pos_embeds = getattr(model_module, "position_embeddings")
            if isinstance(pos_embeds, nn.Module) and hasattr(pos_embeds, "weight"):
                weight = getattr(pos_embeds, "weight")
                if isinstance(weight, nn.Parameter):
                    return weight

        # Check within embedding layer
        if hasattr(model_module, "embedding"):
            embedding = getattr(model_module, "embedding")
            if isinstance(embedding, nn.Module) and hasattr(
                embedding, "position_embeddings"
            ):
                pos_embeds = getattr(embedding, "position_embeddings")
                if isinstance(pos_embeds, nn.Module) and hasattr(pos_embeds, "weight"):
                    weight = getattr(pos_embeds, "weight")
                    if isinstance(weight, nn.Parameter):
                        return weight

        return None

    def allreduce_shared_params(
        self,
        model: List[nn.Module],
        shared_params: List[Tuple[str, nn.Parameter]],
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        """All-reduce arbitrary shared parameters.

        This is a general-purpose method for reducing gradients of any shared
        parameters across a process group.

        Args:
            model: List of model chunks
            shared_params: List of (name, parameter) tuples to reduce
            reduce_group: Process group for reduction (defaults to embedding group)

        Raises:
            RuntimeError: If gradient reduction fails
        """
        if not shared_params:
            return

        if reduce_group is None:
            reduce_group = self.embd_group

        if self._get_process_group_size(reduce_group) <= 1:
            return

        try:
            # Collect gradients to reduce
            grads_to_reduce = []
            params_list = []
            total_elements = 0

            for name, param in shared_params:
                if not isinstance(param, nn.Parameter):
                    logger.warning(f"Skipping non-parameter {name}: {type(param)}")
                    continue

                if not param.requires_grad:
                    continue

                grad_attr = self._get_main_grad_attr(param)
                grad = getattr(param, grad_attr, None)

                if grad is not None:
                    # Validate gradient
                    if self.check_for_nan and torch.isnan(grad).any():
                        logger.error(f"NaN detected in gradient for {name}")
                        self._reduction_metrics.overflow_detected = True
                        continue

                    grads_to_reduce.append(grad)
                    params_list.append((param, grad_attr, name))
                    total_elements += grad.numel()

            if not grads_to_reduce:
                return

            # Log reduction info
            logger.debug(
                f"Reducing {len(grads_to_reduce)} gradients, {total_elements} elements"
            )

            # Coalesce gradients for efficient communication
            import time

            start_time = time.perf_counter()

            coalesced = _flatten_dense_tensors(grads_to_reduce)
            dist.all_reduce(coalesced, group=reduce_group)

            # Unflatten and reassign gradients
            unflattened_grads = _unflatten_dense_tensors(coalesced, grads_to_reduce)
            for (param, grad_attr, name), grad in zip(params_list, unflattened_grads):
                # Final validation
                if self.check_for_nan and torch.isnan(grad).any():
                    logger.error(f"NaN detected after reduction for {name}")
                    self._reduction_metrics.overflow_detected = True
                    # Skip assignment to prevent propagation
                    continue

                setattr(param, grad_attr, grad)

            # Update metrics
            self._reduction_metrics.reduction_time_ms += (
                time.perf_counter() - start_time
            ) * 1000
            self._reduction_metrics.total_bytes_reduced += (
                coalesced.numel() * coalesced.element_size()
            )
            self._reduction_metrics.num_parameters_reduced += len(grads_to_reduce)

        except Exception as e:
            logger.error(f"Failed to all-reduce shared parameters: {e}")
            raise RuntimeError(f"Shared parameter reduction failed: {e}") from e

    def get_reduction_metrics(self) -> ReductionMetrics:
        """Get current reduction metrics.

        Returns:
            Current metrics for gradient reduction operations.
        """
        return self._reduction_metrics

    def reset_metrics(self) -> None:
        """Reset reduction metrics."""
        self._reduction_metrics = ReductionMetrics()


@dataclass
class SharedWeightConfig:
    """Configuration for shared weight gradient reduction.

    This configuration class provides fine-grained control over how shared
    weight gradients are synchronized across distributed training processes.

    Attributes:
        share_embeddings_and_output_weights: Whether input/output weights are tied
        share_position_embeddings: Whether position embeddings are shared
        embedding_reduce_group_size: Size of embedding reduction group
        position_embedding_reduce_group_size: Size of position embedding group
        max_gradient_norm: Maximum allowed gradient norm before clipping
        check_for_nan: Whether to check for NaN/Inf in gradients
        reduction_strategy: Strategy for gradient reduction
        enable_metrics: Whether to collect detailed metrics
        coalesce_gradients: Whether to coalesce small gradients
        hierarchical_reduction: Use hierarchical reduction for large groups
        async_reduction: Enable asynchronous gradient reduction
        gradient_predivide_factor: Factor to predivide gradients by
        gradient_postdivide_factor: Factor to postdivide gradients by
    """

    # Core configuration
    share_embeddings_and_output_weights: bool = False
    share_position_embeddings: bool = False
    embedding_reduce_group_size: Optional[int] = None
    position_embedding_reduce_group_size: Optional[int] = None

    # Safety and validation
    max_gradient_norm: float = 1e10
    check_for_nan: bool = True
    check_for_inf: bool = True
    skip_on_error: bool = True

    # Performance optimization
    reduction_strategy: str = "all_reduce"
    enable_metrics: bool = True
    coalesce_gradients: bool = True
    hierarchical_reduction: bool = False
    async_reduction: bool = False

    # Gradient scaling
    gradient_predivide_factor: float = 1.0
    gradient_postdivide_factor: float = 1.0

    # Advanced options
    use_fp16_compression: bool = False
    gradient_accumulation_steps: int = 1
    bucket_size_mb: int = 25

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.max_gradient_norm <= 0:
            raise ValueError(
                f"max_gradient_norm must be positive, got {self.max_gradient_norm}"
            )

        if self.reduction_strategy not in [
            "all_reduce",
            "reduce_scatter",
            "hierarchical",
        ]:
            raise ValueError(f"Invalid reduction strategy: {self.reduction_strategy}")

        if self.gradient_predivide_factor <= 0 or self.gradient_postdivide_factor <= 0:
            raise ValueError("Gradient scaling factors must be positive")

        if self.bucket_size_mb <= 0:
            raise ValueError(
                f"bucket_size_mb must be positive, got {self.bucket_size_mb}"
            )

        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, "
                f"got {self.gradient_accumulation_steps}"
            )
