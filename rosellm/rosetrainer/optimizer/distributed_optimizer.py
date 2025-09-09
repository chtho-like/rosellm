"""Distributed optimizer with parameter partitioning for memory efficiency."""

import logging
import threading
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Union, cast, overload

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from ..gradient.decoupled_grad import DecoupledGradientConfig, DecoupledGradientManager
from ..utils.gradient_utils import (
    GradientClipConfig,
    apply_gradient_clipping,
    check_gradient_finite,
)
from ..utils.multi_tensor_ops import MultiTensorOperator, multi_tensor_scale
from .config import DistributedOptimizerConfig
from .param_range import ParameterPartitioner, ParameterRange

logger = logging.getLogger(__name__)

# Constants for better maintainability
MAX_OVERFLOW_RETRIES = 10
OVERFLOW_WARNING_THRESHOLD = 5
MEMORY_REPORT_INTERVAL = 100
GRADIENT_SYNC_TIMEOUT = 30.0
DEFAULT_LOSS_SCALE_WINDOW = 2000
MIN_LOSS_SCALE = 1.0
MAX_LOSS_SCALE = 2**24


class OptimizerState(Enum):
    """State of the distributed optimizer."""

    INITIALIZED = "initialized"
    READY = "ready"
    OVERFLOW = "overflow"
    ERROR = "error"


class DistributedOptimizer(Optimizer):
    """Distributed optimizer with parameter and state partitioning.

    This optimizer partitions parameters, gradients, and optimizer states
    across data parallel ranks to reduce memory usage. It supports:
    - Parameter partitioning across DP ranks
    - Gradient accumulation and reduction
    - Mixed precision training with FP32 main parameters
    - CPU offloading of optimizer states
    - Integration with gradient utilities
    """

    def __init__(
        self,
        params: Union[Iterator[nn.Parameter], List[Dict[str, Any]]],
        optimizer_class: type,
        optimizer_kwargs: Dict[str, Any],
        config: DistributedOptimizerConfig,
        process_group: Optional[dist.ProcessGroup] = None,
        decoupled_grad_config: Optional[DecoupledGradientConfig] = None,
        model: Optional[nn.Module] = None,
    ):
        """Initialize distributed optimizer.

        Args:
            params: Model parameters or parameter groups.
            optimizer_class: Base optimizer class (e.g., torch.optim.Adam).
            optimizer_kwargs: Arguments for base optimizer.
            config: Configuration for distributed optimizer.
            process_group: Process group for communication.
            decoupled_grad_config: Configuration for decoupled gradient storage.
            model: Model for decoupled gradient management.
        """
        self.config = config
        self.process_group = process_group or dist.group.WORLD
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)

        # Initialize decoupled gradient manager if configured
        self.decoupled_grad_manager: Optional[DecoupledGradientManager] = None
        self.decoupled_grad_config = decoupled_grad_config
        self._using_decoupled_grads = False

        if (
            decoupled_grad_config
            and decoupled_grad_config.enabled
            and model is not None
        ):
            try:
                self.decoupled_grad_manager = DecoupledGradientManager(
                    model, decoupled_grad_config
                )
                self._using_decoupled_grads = True
                logger.info(
                    f"Initialized DistributedOptimizer with decoupled gradient storage"
                )
            except RuntimeError as e:
                logger.warning(
                    f"Failed to initialize decoupled gradient manager: {e}. "
                    f"Falling back to standard gradient storage."
                )
                self.decoupled_grad_manager = None
                self._using_decoupled_grads = False

        # Convert params to parameter groups
        if not isinstance(params, list):
            params = [{"params": list(params)}]
        elif len(params) > 0 and not isinstance(params[0], dict):
            params = [{"params": params}]

        # Store original parameter groups
        self.param_groups = params

        # Initialize base optimizer with placeholder
        defaults = optimizer_kwargs.copy()
        super().__init__(params, defaults)

        # Setup parameter partitioning
        self._setup_partitioning()

        # Create base optimizer for local parameters
        self.base_optimizer = self._create_base_optimizer(
            optimizer_class, optimizer_kwargs
        )

        # Setup gradient handling
        self._setup_gradient_handling()

        # Setup mixed precision if enabled
        if config.mixed_precision:
            self._setup_mixed_precision()

        # Initialize communication buffers
        self._init_communication_buffers()

        # Statistics and state management
        self.step_count = 0
        self.overflow_count = 0
        self.consecutive_overflows = 0
        self.optimizer_state = OptimizerState.INITIALIZED

        # Thread safety for distributed operations
        self._lock = threading.RLock()
        self._comm_lock = threading.Lock()

        # Initialize multi-tensor operator for optimized operations
        self.multi_tensor_operator = MultiTensorOperator(
            device=self.local_params[0].device if self.local_params else None,
            enable_benchmarking=config.verbose,
        )

    def _setup_partitioning(self) -> None:
        """Setup parameter partitioning across ranks."""
        self.partitioners: List[ParameterPartitioner] = []
        self.param_to_range: Dict[nn.Parameter, ParameterRange] = {}
        self.local_params: List[nn.Parameter] = []
        self.local_param_groups: List[Dict[str, Any]] = []

        for group_idx, group in enumerate(self.param_groups):
            params = list(group["params"])

            # Create partitioner for this group
            partitioner = ParameterPartitioner(self.world_size, self.rank)
            partitioner.compute_partition_ranges(
                params, contiguous=self.config.contiguous_gradients
            )
            self.partitioners.append(partitioner)

            # Get local range for this rank
            local_range = partitioner.get_local_param_range()
            if local_range is None:
                continue

            # Collect local parameters
            local_group_params = []
            for param_idx in local_range.param_indices:
                if param_idx < len(params):
                    param = params[param_idx]
                    self.param_to_range[param] = local_range
                    local_group_params.append(param)

            if local_group_params:
                # Create local parameter group
                local_group = group.copy()
                local_group["params"] = local_group_params
                self.local_param_groups.append(local_group)
                self.local_params.extend(local_group_params)

        if self.config.verbose:
            logger.info(
                f"Rank {self.rank}: Partitioned {len(self.local_params)} parameters "
                f"from {sum(len(g['params']) for g in self.param_groups)} total"
            )

    def _create_base_optimizer(
        self, optimizer_class: type, optimizer_kwargs: Dict[str, Any]
    ) -> Any:
        """Create base optimizer for local parameters.

        Args:
            optimizer_class: Base optimizer class.
            optimizer_kwargs: Arguments for base optimizer.

        Returns:
            Initialized base optimizer.
        """
        if not self.local_param_groups:
            # No local parameters, create dummy optimizer
            dummy_param = torch.zeros(1, requires_grad=True)
            return optimizer_class([dummy_param], **optimizer_kwargs)

        # Create optimizer with local parameters only
        if self.config.partition_parameters:
            # Only optimize local partition
            return optimizer_class(self.local_param_groups, **optimizer_kwargs)
        else:
            # Optimize all parameters (no partitioning)
            return optimizer_class(self.param_groups, **optimizer_kwargs)

    def _setup_gradient_handling(self) -> None:
        """Setup gradient accumulation and reduction."""
        # Initialize gradient clip config if needed
        self.grad_clip_config = None
        if self.config.grad_clip_value is not None:
            self.grad_clip_config = GradientClipConfig(
                clip_type="norm",
                max_norm=self.config.grad_clip_value,
                norm_type=2.0,
                use_multitensor=self.config.use_multi_tensor_apply,
                model_parallel_reduce=False,  # We handle this separately
            )

    def _setup_mixed_precision(self) -> None:
        """Setup mixed precision training with FP32 main parameters."""
        # Create FP32 main parameters
        self.fp32_params: Dict[nn.Parameter, Tensor] = {}

        for param in self.local_params:
            if param.dtype != torch.float32:
                # Create FP32 copy
                self.fp32_params[param] = param.detach().float().clone()
                self.fp32_params[param].requires_grad = param.requires_grad
            else:
                # Already FP32
                self.fp32_params[param] = param

        # Initialize loss scale for mixed precision
        self.loss_scale = self.config.grad_scaler_config["init_scale"]
        self.loss_scale_growth_factor = self.config.grad_scaler_config["growth_factor"]
        self.loss_scale_backoff_factor = self.config.grad_scaler_config[
            "backoff_factor"
        ]
        self.loss_scale_growth_interval = self.config.grad_scaler_config[
            "growth_interval"
        ]
        self.loss_scale_growth_counter = 0

    def _init_communication_buffers(self) -> None:
        """Initialize buffers for communication."""
        # Calculate buffer sizes
        bytes_per_mb = 1024 * 1024
        self.reduce_bucket_size = self.config.reduce_bucket_size_mb * bytes_per_mb
        self.allgather_bucket_size = self.config.allgather_bucket_size_mb * bytes_per_mb

        # Create gradient reduction buffers
        if self.config.contiguous_gradients:
            total_numel = sum(p.numel() for p in self.local_params)
            if total_numel > 0:
                self.grad_buffer = torch.zeros(
                    total_numel,
                    dtype=self.config.dtype,
                    device=self.local_params[0].device if self.local_params else "cpu",
                )
            else:
                self.grad_buffer = torch.empty(0, dtype=self.config.dtype)
        else:
            self.grad_buffer = torch.empty(0, dtype=self.config.dtype)

        # Create allgather buffers for parameter synchronization
        if self.config.partition_parameters:
            self.allgather_buffers: List[Tensor] = []
            for group in self.param_groups:
                params = list(group["params"])
                if params:
                    total_numel = sum(p.numel() for p in params)
                    buffer = torch.zeros(
                        total_numel, dtype=params[0].dtype, device=params[0].device
                    )
                    self.allgather_buffers.append(buffer)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients of all parameters.

        Args:
            set_to_none: Whether to set gradients to None instead of zero.
        """
        # Zero decoupled gradients if enabled
        if self._using_decoupled_grads and self.decoupled_grad_manager is not None:
            # Don't use set_to_none for decoupled gradients as we want to keep buffers
            self.decoupled_grad_manager.zero_gradients(set_to_none=False)

        # Zero local gradients
        for param in self.local_params:
            if set_to_none:
                param.grad = None
            else:
                if param.grad is not None:
                    param.grad.zero_()

        # Zero base optimizer gradients
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _reduce_gradients(self) -> None:
        """Reduce gradients across data parallel ranks."""
        if self.world_size == 1:
            return

        # Sync decoupled gradients to parameters if needed
        if self._using_decoupled_grads and self.decoupled_grad_manager is not None:
            # Use non-cloning sync for efficiency during reduction
            self.decoupled_grad_manager.sync_gradients_to_params(clone=False)

        # Reduce gradients
        if self.config.contiguous_gradients and self.grad_buffer is not None:
            # Pack gradients into contiguous buffer
            offset = 0
            for param in self.local_params:
                grad = self._get_gradient(param)
                if grad is not None:
                    numel = param.numel()
                    self.grad_buffer[offset : offset + numel] = grad.view(-1)
                    offset += numel

            # All-reduce buffer
            dist.all_reduce(
                self.grad_buffer, op=dist.ReduceOp.SUM, group=self.process_group
            )

            # Unpack gradients
            offset = 0
            for param in self.local_params:
                if self._has_gradient(param):
                    numel = param.numel()
                    grad_view = self.grad_buffer[offset : offset + numel].view_as(param)
                    self._set_gradient(param, grad_view)
                    offset += numel
        else:
            # Reduce individual gradients
            for param in self.local_params:
                grad = self._get_gradient(param)
                if grad is not None:
                    dist.all_reduce(
                        grad, op=dist.ReduceOp.SUM, group=self.process_group
                    )
                    self._set_gradient(param, grad)

        # Scale gradients by world size using multi-tensor operations
        scale_factor = 1.0 / self.world_size
        scale_factor *= self.config.gradient_postdivide_factor

        if self._using_decoupled_grads and self.decoupled_grad_manager is not None:
            self.decoupled_grad_manager.scale_gradients(scale_factor)
            # Also scale any param.grad that might exist
            grads_to_scale = [p.grad for p in self.local_params if p.grad is not None]
            if grads_to_scale:
                multi_tensor_scale(
                    grads_to_scale, scale_factor, self.multi_tensor_operator
                )
        else:
            # Use multi-tensor scaling for efficiency
            grads_to_scale = [p.grad for p in self.local_params if p.grad is not None]
            if grads_to_scale:
                multi_tensor_scale(
                    grads_to_scale, scale_factor, self.multi_tensor_operator
                )

    def _get_gradient(self, param: nn.Parameter) -> Optional[Tensor]:
        """Get gradient for a parameter, supporting decoupled storage.

        Args:
            param: Parameter to get gradient for.

        Returns:
            Gradient tensor or None.
        """
        if self._using_decoupled_grads and self.decoupled_grad_manager is not None:
            grad = self.decoupled_grad_manager.get_gradient(param)
            # Fall back to param.grad if decoupled gradient not available
            return grad if grad is not None else param.grad
        return param.grad

    def _set_gradient(self, param: nn.Parameter, grad: Tensor) -> None:
        """Set gradient for a parameter, supporting decoupled storage.

        Args:
            param: Parameter to set gradient for.
            grad: Gradient tensor.
        """
        if self._using_decoupled_grads and self.decoupled_grad_manager is not None:
            try:
                self.decoupled_grad_manager.set_gradient(param, grad)
            except (ValueError, RuntimeError) as e:
                # Fall back to standard gradient storage on error
                logger.debug(f"Failed to set decoupled gradient: {e}")
                param.grad = grad
        else:
            param.grad = grad

    def _has_gradient(self, param: nn.Parameter) -> bool:
        """Check if parameter has a gradient.

        Args:
            param: Parameter to check.

        Returns:
            True if parameter has gradient.
        """
        if self._using_decoupled_grads and self.decoupled_grad_manager is not None:
            has_decoupled = self.decoupled_grad_manager.get_gradient(param) is not None
            # Check both decoupled and standard storage
            return has_decoupled or param.grad is not None
        return param.grad is not None

    def _check_gradients(self, check_scaled: bool = False) -> bool:
        """Check gradients for NaN/Inf values.

        Args:
            check_scaled: If True, check scaled gradients (for mixed precision).

        Returns:
            True if gradients are valid, False otherwise.
        """
        if not self.config.check_gradients:
            return True

        with self._lock:
            # For mixed precision, check the appropriate gradients
            params_to_check = self.local_params
            if check_scaled and self.config.mixed_precision:
                # Check FP32 gradients if they exist
                params_to_check = [
                    p for p in self.local_params if p in self.fp32_params
                ]
                if params_to_check:
                    # Temporarily assign FP32 grads for checking
                    temp_grads = []
                    for p in params_to_check:
                        temp_grads.append(p.grad)
                        if self.fp32_params[p].grad is not None:
                            p.grad = self.fp32_params[p].grad

            # Use gradient utils to check for finite gradients
            all_finite, stats = check_gradient_finite(
                params_to_check, raise_on_nonfinite=False
            )

            # Restore original gradients if we swapped them
            if check_scaled and self.config.mixed_precision and params_to_check:
                for p, orig_grad in zip(params_to_check, temp_grads):
                    p.grad = orig_grad

            if not all_finite:
                self.consecutive_overflows += 1
                if (
                    self.config.verbose
                    and self.consecutive_overflows % OVERFLOW_WARNING_THRESHOLD == 0
                ):
                    logger.warning(
                        f"Found non-finite gradients: {stats['nan_parameters']} NaN, "
                        f"{stats['inf_parameters']} Inf parameters. "
                        f"Consecutive overflows: {self.consecutive_overflows}"
                    )
            else:
                self.consecutive_overflows = 0

            return all_finite

    def _clip_gradients(self) -> float:
        """Clip gradients by global norm.

        Returns:
            Total gradient norm before clipping.
        """
        if self.grad_clip_config is None:
            return 0.0

        # Apply gradient clipping - cast to List[Tensor] for type compatibility
        params_as_tensors = cast(List[torch.Tensor], self.local_params)
        stats = apply_gradient_clipping(params_as_tensors, self.grad_clip_config)
        return stats.get("grad_norm", 0.0)

    def _allgather_parameters(self) -> None:
        """Allgather parameters from all ranks after optimization.

        Optimized implementation with better memory access patterns.
        """
        if not self.config.partition_parameters or self.world_size == 1:
            return

        with self._comm_lock:
            for group_idx, (group, partitioner) in enumerate(
                zip(self.param_groups, self.partitioners)
            ):
                params = list(group["params"])
                if not params:
                    continue

                # Get buffer for this group
                buffer = self.allgather_buffers[group_idx]

                # Optimized packing using contiguous memory access
                local_range = partitioner.get_local_param_range()
                if local_range is not None:
                    # Pre-allocate send buffer slice
                    send_offset = self._compute_rank_offset(partitioner, self.rank)
                    send_size = local_range.total_elements

                    if send_offset >= 0 and send_size > 0:
                        send_buffer = buffer[send_offset : send_offset + send_size]

                        # Pack parameters contiguously
                        pack_offset = 0
                        for param_idx in local_range.param_indices:
                            if param_idx < len(params):
                                param = params[param_idx]
                                param_slice = local_range.get_param_slice(
                                    param_idx, param.numel()
                                )
                                if param_slice is not None:
                                    start, end = param_slice
                                    slice_size = end - start
                                    with torch.no_grad():
                                        send_buffer[
                                            pack_offset : pack_offset + slice_size
                                        ].copy_(param.view(-1)[start:end])
                                    pack_offset += slice_size

                # Allgather with timeout handling
                try:
                    work = dist.all_gather_into_tensor(
                        buffer, buffer, group=self.process_group, async_op=True
                    )
                    if work is not None:
                        work.wait(timeout=timedelta(seconds=GRADIENT_SYNC_TIMEOUT))
                except RuntimeError as e:
                    logger.error(f"Allgather failed: {e}")
                    self.optimizer_state = OptimizerState.ERROR
                    raise

                # Unpack parameters from buffer with validation
                self._unpack_allgathered_params(buffer, params)

    def _compute_rank_offset(self, partitioner: ParameterPartitioner, rank: int) -> int:
        """Compute buffer offset for a given rank.

        Args:
            partitioner: Parameter partitioner.
            rank: Rank to compute offset for.

        Returns:
            Buffer offset for the rank, or -1 if rank has no data.
        """
        offset = 0
        for r in range(rank):
            rank_range = partitioner.rank_to_range.get(r)
            if rank_range is not None:
                offset += rank_range.total_elements

        # Check if target rank has data
        if partitioner.rank_to_range.get(rank) is None:
            return -1

        return offset

    def _unpack_allgathered_params(
        self, buffer: Tensor, params: List[nn.Parameter]
    ) -> None:
        """Unpack parameters from allgathered buffer.

        Args:
            buffer: Allgathered buffer containing all parameters.
            params: List of parameters to unpack into.
        """
        offset = 0
        for param in params:
            numel = param.numel()
            if offset + numel > buffer.numel():
                raise RuntimeError(
                    f"Buffer underflow: expected {offset + numel} elements, "
                    f"but buffer has only {buffer.numel()}"
                )
            with torch.no_grad():
                param.data = buffer[offset : offset + numel].view_as(param)
            offset += numel

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform optimization step.

        Args:
            closure: Closure that reevaluates the model and returns the loss.

        Returns:
            Loss value if closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Check for valid gradients
        if not self._check_gradients():
            self.overflow_count += 1
            if self.config.verbose:
                logger.warning(
                    f"Skipping step due to invalid gradients "
                    f"(overflow count: {self.overflow_count})"
                )
            return loss

        # Reduce gradients across ranks
        self._reduce_gradients()

        # Clip gradients
        grad_norm = self._clip_gradients()

        # Handle mixed precision with proper overflow detection
        if self.config.mixed_precision:
            # First unscale gradients to FP32 for accurate overflow detection
            with self._lock:
                for param in self.local_params:
                    grad = self._get_gradient(param)
                    if param in self.fp32_params and grad is not None:
                        # Unscale gradient and convert to FP32
                        self.fp32_params[param].grad = grad.float() / self.loss_scale

                # Now check for overflow in unscaled FP32 gradients
                had_overflow = not self._check_gradients(check_scaled=True)

                if not had_overflow:
                    # No overflow, proceed with optimization
                    self._step_with_fp32_params()

                    # Update loss scale for successful step
                    self._update_loss_scale_success()
                else:
                    # Had overflow, handle it
                    self._handle_gradient_overflow()

                    # Check if we've exceeded max retries
                    if self.consecutive_overflows >= MAX_OVERFLOW_RETRIES:
                        logger.error(
                            f"Exceeded max overflow retries "
                            f"({MAX_OVERFLOW_RETRIES}). Training may be unstable."
                        )
                        self.optimizer_state = OptimizerState.ERROR

                    return loss
        else:
            # Regular step with error handling
            try:
                self.base_optimizer.step()
                self.optimizer_state = OptimizerState.READY
            except RuntimeError as e:
                logger.error(f"Optimizer step failed: {e}")
                self.optimizer_state = OptimizerState.ERROR
                raise

        # Allgather parameters if partitioned
        self._allgather_parameters()

        self.step_count += 1

        if self.config.verbose and self.step_count % MEMORY_REPORT_INTERVAL == 0:
            logger.info(
                f"Step {self.step_count}: grad_norm={grad_norm:.4f}, "
                f"overflow_count={self.overflow_count}"
            )

        return loss

    def _step_with_fp32_params(self) -> None:
        """Perform optimizer step with FP32 parameters."""
        # Step with FP32 parameters
        self.base_optimizer.step()

        # Copy updated FP32 params back to original dtype
        with torch.no_grad():
            for param in self.local_params:
                if param in self.fp32_params:
                    param.data.copy_(self.fp32_params[param].data.to(param.dtype))

    def _update_loss_scale_success(self) -> None:
        """Update loss scale after successful step."""
        self.loss_scale_growth_counter += 1
        if self.loss_scale_growth_counter >= self.loss_scale_growth_interval:
            new_scale = min(
                self.loss_scale * self.loss_scale_growth_factor, MAX_LOSS_SCALE
            )
            if new_scale != self.loss_scale:
                self.loss_scale = new_scale
                if self.config.verbose:
                    logger.debug(f"Increased loss scale to {self.loss_scale}")
            self.loss_scale_growth_counter = 0

    def _handle_gradient_overflow(self) -> None:
        """Handle gradient overflow by adjusting loss scale."""
        old_scale = self.loss_scale
        self.loss_scale = max(
            self.loss_scale * self.loss_scale_backoff_factor, MIN_LOSS_SCALE
        )
        self.loss_scale_growth_counter = 0
        self.overflow_count += 1

        if self.config.verbose:
            logger.warning(
                f"Gradient overflow #{self.overflow_count} detected, "
                f"reducing loss scale from {old_scale} to {self.loss_scale}"
            )

        # Clear gradients to prevent accumulation of invalid values
        self.zero_grad(set_to_none=True)

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dictionary.

        Returns:
            State dictionary containing optimizer state.
        """
        state_dict = {
            "state": self.base_optimizer.state_dict()["state"],
            "param_groups": self.param_groups,
            "step_count": self.step_count,
            "overflow_count": self.overflow_count,
        }

        if self.config.mixed_precision:
            state_dict["loss_scale"] = self.loss_scale
            state_dict["loss_scale_growth_counter"] = self.loss_scale_growth_counter

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state dictionary.

        Args:
            state_dict: State dictionary to load.
        """
        self.base_optimizer.load_state_dict(
            {"state": state_dict["state"], "param_groups": self.local_param_groups}
        )

        self.step_count = state_dict.get("step_count", 0)
        self.overflow_count = state_dict.get("overflow_count", 0)

        if self.config.mixed_precision:
            if "loss_scale" in state_dict:
                self.loss_scale = state_dict["loss_scale"]
            if "loss_scale_growth_counter" in state_dict:
                self.loss_scale_growth_counter = state_dict["loss_scale_growth_counter"]

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics.

        Returns:
            Dictionary with memory usage in MB.
        """
        memory_stats = {}

        # Parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in self.local_params)
        memory_stats["parameters_mb"] = param_memory / (1024 * 1024)

        # Gradient memory
        grad_memory = sum(
            p.grad.numel() * p.grad.element_size()
            for p in self.local_params
            if p.grad is not None
        )
        memory_stats["gradients_mb"] = grad_memory / (1024 * 1024)

        # Optimizer state memory
        state_memory = 0
        for state in self.base_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state_memory += v.numel() * v.element_size()
        memory_stats["optimizer_states_mb"] = state_memory / (1024 * 1024)

        # Total memory
        memory_stats["total_mb"] = sum(memory_stats.values())

        return memory_stats
