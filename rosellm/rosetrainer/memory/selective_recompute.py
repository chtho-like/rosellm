"""
Selective Activation Recomputation Module for RoseTrainer

This module provides intelligent activation checkpointing that decides which
activations to save based on memory usage, computation cost, and activation patterns.
It extends PyTorch's gradient checkpointing with cost-aware heuristics.

Key Features:
- Cost model-based selection of checkpoint layers
- Memory profiling and adaptive thresholds
- Integration with existing checkpointing infrastructure
- Support for custom selection strategies

References:
[1] Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016)
[2] Jain et al., "Checkmate: Breaking the Memory Wall with Optimal Tensor "
    "Rematerialization" (2020)
[3] Kirisame et al., "Dynamic Tensor Rematerialization" (2021)
"""

import enum
import functools
import logging
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    cast,
)

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint as torch_checkpoint

logger = logging.getLogger(__name__)


class SelectionStrategy(enum.Enum):
    """Strategies for selecting which layers to checkpoint.

    Available strategies:
        UNIFORM: Checkpoint every N layers at regular intervals
        MEMORY_BASED: Checkpoint layers exceeding memory threshold
        COMPUTATION_BASED: Checkpoint computationally expensive layers
        HYBRID: Combine memory and computation factors with scoring
        MANUAL: Use explicitly specified layer lists
        ADAPTIVE: Dynamically adjust selection based on runtime profiling
    """

    UNIFORM = "uniform"  # Checkpoint every N layers
    MEMORY_BASED = "memory_based"  # Checkpoint based on memory usage
    COMPUTATION_BASED = "computation_based"  # Checkpoint based on computation cost
    HYBRID = "hybrid"  # Combine memory and computation factors
    MANUAL = "manual"  # User-specified layers
    ADAPTIVE = "adaptive"  # Dynamic selection based on runtime profiling


@dataclass
class LayerProfile:
    """Profile information for a single layer.

    Tracks performance metrics and usage statistics for intelligent
    checkpoint selection decisions.

    Attributes:
        layer_id: Unique identifier for the layer
        memory_usage: Peak memory usage in megabytes
        computation_time: Forward pass time in seconds (EMA)
        recompute_time: Recomputation time in seconds (EMA)
        activation_size: Size of activations in bytes
        parameter_count: Number of parameters in the layer
        flops: Estimated FLOPs for the layer
        checkpoint_count: Number of times layer was checkpointed
        skip_count: Number of times checkpointing was skipped
    """

    layer_id: str
    memory_usage: float = 0.0  # MB
    computation_time: float = 0.0  # seconds
    recompute_time: float = 0.0  # seconds
    activation_size: int = 0  # bytes
    parameter_count: int = 0
    flops: int = 0
    checkpoint_count: int = 0  # Number of times checkpointed
    skip_count: int = 0  # Number of times skipped


@dataclass
class SelectiveCheckpointConfig:
    """Configuration for selective activation checkpointing."""

    # Selection strategy
    strategy: SelectionStrategy = SelectionStrategy.HYBRID

    # Memory thresholds
    memory_threshold_mb: float = 1024.0  # Checkpoint if activation > threshold
    total_memory_budget_mb: Optional[float] = None  # Total memory budget

    # Computation thresholds
    computation_threshold_ms: float = 10.0  # Checkpoint if compute time > threshold
    recompute_factor: float = 1.5  # Only checkpoint if recompute < factor * forward

    # Uniform strategy parameters
    checkpoint_interval: int = 3  # Checkpoint every N layers

    # Manual strategy parameters
    layers_to_checkpoint: List[Union[int, str]] = field(default_factory=list)
    layers_to_skip: List[Union[int, str]] = field(default_factory=list)

    # Adaptive strategy parameters
    profile_warmup_steps: int = 10  # Steps before enabling adaptive selection
    profile_update_interval: int = 100  # Update selection every N steps
    adaptive_threshold_percentile: float = 75.0  # Checkpoint top percentile layers

    # Profiling parameters
    ema_decay_factor: float = 0.9  # Exponential moving average decay
    max_profile_history: int = 10000  # Maximum profiles to keep in memory

    # General settings
    use_reentrant: bool = True
    preserve_rng_state: bool = True
    profile_enabled: bool = False
    verbose: bool = False
    thread_safe: bool = True  # Enable thread-safe operations

    # Performance optimizations
    chunk_size: Optional[int] = None  # Group layers into chunks
    nested_checkpointing: bool = False  # Enable nested checkpointing

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        errors = []

        if self.memory_threshold_mb <= 0:
            errors.append("memory_threshold_mb must be positive")
        if self.computation_threshold_ms < 0:
            errors.append("computation_threshold_ms must be non-negative")
        if self.recompute_factor <= 0:
            errors.append("recompute_factor must be positive")
        if self.checkpoint_interval <= 0:
            errors.append("checkpoint_interval must be positive")
        if not 0 <= self.adaptive_threshold_percentile <= 100:
            errors.append("adaptive_threshold_percentile must be between 0 and 100")
        if not 0 < self.ema_decay_factor < 1:
            errors.append("ema_decay_factor must be between 0 and 1 (exclusive)")
        if self.max_profile_history <= 0:
            errors.append("max_profile_history must be positive")
        if self.profile_warmup_steps < 0:
            errors.append("profile_warmup_steps must be non-negative")
        if self.profile_update_interval <= 0:
            errors.append("profile_update_interval must be positive")
        if self.chunk_size is not None and self.chunk_size <= 0:
            errors.append("chunk_size must be positive if specified")
        if self.total_memory_budget_mb is not None and self.total_memory_budget_mb <= 0:
            errors.append("total_memory_budget_mb must be positive if specified")

        if errors:
            raise ValueError(
                f"Configuration validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )


class SelectiveCheckpointFunction(Function):
    """Custom autograd function for selective checkpointing with profiling."""

    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable[..., Any],
        preserve_rng_state: bool,
        layer_id: str,
        profiler: Optional["LayerProfiler"],
        *args: Any,
    ) -> Any:
        """Forward pass with optional profiling.

        Args:
            ctx: Context for saving tensors
            run_function: Function to run and potentially checkpoint
            preserve_rng_state: Whether to preserve RNG state
            layer_id: Identifier for the layer
            profiler: Optional profiler for recording statistics
            *args: Arguments to the function

        Returns:
            Output of the function

        Raises:
            RuntimeError: If forward execution fails
        """
        # Validate inputs
        if run_function is None:
            raise ValueError("run_function cannot be None")
        if not layer_id:
            raise ValueError("layer_id cannot be empty")

        # Store metadata in context
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.layer_id = layer_id
        ctx.profiler = profiler

        # Save RNG state if needed
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            if torch.cuda.is_available():
                ctx.fwd_gpu_devices = list(range(torch.cuda.device_count()))
                ctx.fwd_gpu_states = [
                    torch.cuda.get_rng_state(device) for device in ctx.fwd_gpu_devices
                ]

        # Profile forward pass
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Run forward function
        try:
            with torch.no_grad():
                outputs = run_function(*args)
        except Exception as e:
            logger.error(f"Forward execution failed for layer {layer_id}: {e}")
            raise RuntimeError(f"Forward execution failed for layer {layer_id}") from e

        # Record profiling data
        if profiler is not None:
            elapsed_time = time.time() - start_time
            memory_used = (
                torch.cuda.memory_allocated() - start_memory
                if torch.cuda.is_available()
                else 0
            )
            profiler.record_forward(layer_id, elapsed_time, memory_used)

        # Save tensors for backward (only non-tensor inputs)
        ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward pass with recomputation and profiling.

        Args:
            ctx: Context with saved tensors and metadata
            *grad_outputs: Gradients from the next layer

        Returns:
            Tuple of gradients for each input
        """
        # Restore RNG state if needed
        if ctx.preserve_rng_state:
            rng_devices: List[int] = []
            if torch.cuda.is_available():
                rng_devices = ctx.fwd_gpu_devices

            with torch.random.fork_rng(devices=rng_devices):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.fwd_cpu_state)
                    if torch.cuda.is_available():
                        for device, state in zip(
                            ctx.fwd_gpu_devices, ctx.fwd_gpu_states
                        ):
                            torch.cuda.set_rng_state(state, device)

                # Profile recomputation
                start_time = time.time()

                # Recompute forward pass
                try:
                    with torch.enable_grad():
                        inputs = ctx.saved_tensors
                        outputs = ctx.run_function(*inputs)
                except Exception as e:
                    logger.error(f"Recomputation failed for layer {ctx.layer_id}: {e}")
                    raise RuntimeError(
                        f"Recomputation failed for layer {ctx.layer_id}"
                    ) from e

                # Record recomputation time
                if ctx.profiler is not None:
                    elapsed_time = time.time() - start_time
                    ctx.profiler.record_recompute(ctx.layer_id, elapsed_time)

        # Ensure outputs is a tuple
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Ensure we have gradients for all outputs
        if len(grad_outputs) != len(outputs):
            raise RuntimeError(
                f"Gradient count mismatch: expected {len(outputs)}, "
                f"got {len(grad_outputs)}"
            )

        # Compute gradients for inputs that require them
        gradients: List[Optional[torch.Tensor]] = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor) and inp.requires_grad:
                # Compute gradient for this input
                grad_list = []
                for out, grad_out in zip(outputs, grad_outputs):
                    if grad_out is not None:
                        grad = torch.autograd.grad(
                            outputs=out,
                            inputs=inp,
                            grad_outputs=grad_out,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if grad is not None:
                            grad_list.append(grad)

                if grad_list:
                    # Sum all gradients for this input
                    total_grad = grad_list[0]
                    for g in grad_list[1:]:
                        total_grad = total_grad + g
                    gradients.append(total_grad)
                else:
                    gradients.append(None)
            else:
                gradients.append(None)

        # Return gradients (None for metadata arguments)
        return (None, None, None, None) + tuple(gradients)


class SelectionStrategyProtocol(Protocol):
    """Protocol for selection strategy implementations.

    Defines the interface that all selection strategies must implement
    to work with the SelectiveRecomputeManager.
    """

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Determine if a layer should be checkpointed.

        Args:
            layer_id: Identifier for the layer
            profile: Optional profiling data for the layer

        Returns:
            True if the layer should be checkpointed
        """
        ...

    def update_selection(
        self, profiles: Dict[str, LayerProfile], step: int
    ) -> Set[str]:
        """Update layer selection based on profiles.

        Args:
            profiles: Dictionary of layer profiles
            step: Current training step

        Returns:
            Set of layer IDs to checkpoint
        """
        ...


class LayerProfiler:
    """Thread-safe profiler for tracking layer statistics during training."""

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        self.config = config
        self.profiles: Dict[str, LayerProfile] = {}
        self.step_count: int = 0
        self._lock = threading.RLock() if config.thread_safe else None
        self._profile_history: Deque[str] = deque(maxlen=config.max_profile_history)

    def record_forward(self, layer_id: str, time_sec: float, memory_bytes: int) -> None:
        """Record forward pass statistics with thread safety."""
        if self._lock:
            with self._lock:
                self._record_forward_impl(layer_id, time_sec, memory_bytes)
        else:
            self._record_forward_impl(layer_id, time_sec, memory_bytes)

    def _record_forward_impl(
        self, layer_id: str, time_sec: float, memory_bytes: int
    ) -> None:
        """Internal implementation of forward recording."""
        if layer_id not in self.profiles:
            self.profiles[layer_id] = LayerProfile(layer_id)
            self._profile_history.append(layer_id)
            self._cleanup_old_profiles()

        profile = self.profiles[layer_id]
        decay = self.config.ema_decay_factor
        profile.computation_time = profile.computation_time * decay + time_sec * (
            1 - decay
        )
        profile.memory_usage = memory_bytes / (1024 * 1024)  # Convert to MB
        profile.activation_size = memory_bytes

    def _cleanup_old_profiles(self) -> None:
        """Clean up old profiles to prevent memory growth."""
        if len(self.profiles) <= self.config.max_profile_history:
            return

        # Calculate how many profiles to remove
        num_to_remove = len(self.profiles) - self.config.max_profile_history
        if num_to_remove <= 0:
            return

        # Sort profiles by importance (checkpoint count, then by age)
        profiles_to_consider = []
        # Consider more than needed
        for layer_id in list(self._profile_history)[: num_to_remove * 2]:
            if layer_id in self.profiles:
                profile = self.profiles[layer_id]
                # Score: higher is more important to keep
                score = profile.checkpoint_count * 1000 + profile.skip_count
                profiles_to_consider.append((layer_id, score))

        # Sort by score (ascending, so least important first)
        profiles_to_consider.sort(key=lambda x: x[1])

        # Remove least important profiles
        removed = 0
        for layer_id, _ in profiles_to_consider:
            if removed >= num_to_remove:
                break
            if layer_id in self.profiles:
                del self.profiles[layer_id]
                try:
                    self._profile_history.remove(layer_id)
                except ValueError:
                    pass  # Already removed
                removed += 1

    def record_recompute(self, layer_id: str, time_sec: float) -> None:
        """Record recomputation statistics with thread safety."""
        if self._lock:
            with self._lock:
                self._record_recompute_impl(layer_id, time_sec)
        else:
            self._record_recompute_impl(layer_id, time_sec)

    def _record_recompute_impl(self, layer_id: str, time_sec: float) -> None:
        """Internal implementation of recompute recording."""
        if layer_id in self.profiles:
            profile = self.profiles[layer_id]
            decay = self.config.ema_decay_factor
            profile.recompute_time = profile.recompute_time * decay + time_sec * (
                1 - decay
            )

    def record_checkpoint_decision(self, layer_id: str, checkpointed: bool) -> None:
        """Record whether a layer was checkpointed with thread safety."""
        if self._lock:
            with self._lock:
                self._record_decision_impl(layer_id, checkpointed)
        else:
            self._record_decision_impl(layer_id, checkpointed)

    def _record_decision_impl(self, layer_id: str, checkpointed: bool) -> None:
        """Internal implementation of decision recording."""
        if layer_id not in self.profiles:
            self.profiles[layer_id] = LayerProfile(layer_id)
            self._profile_history.append(layer_id)

        profile = self.profiles[layer_id]
        if checkpointed:
            profile.checkpoint_count += 1
        else:
            profile.skip_count += 1

    def get_profile(self, layer_id: str) -> Optional[LayerProfile]:
        """Get profile for a specific layer with thread safety."""
        if self._lock:
            with self._lock:
                return self.profiles.get(layer_id)
        return self.profiles.get(layer_id)

    def get_top_memory_layers(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get layers with highest memory usage.

        Args:
            top_k: Number of top layers to return

        Returns:
            List of (layer_id, memory_usage_mb) tuples
        """
        if self._lock:
            with self._lock:
                return self._get_top_memory_layers_impl(top_k)
        return self._get_top_memory_layers_impl(top_k)

    def _get_top_memory_layers_impl(self, top_k: int) -> List[Tuple[str, float]]:
        """Internal implementation of get_top_memory_layers."""
        if not self.profiles:
            return []

        # Use heap for efficient top-k selection
        import heapq

        top_layers = heapq.nlargest(
            top_k, self.profiles.items(), key=lambda x: x[1].memory_usage
        )
        return [(lid, prof.memory_usage) for lid, prof in top_layers]

    def get_top_computation_layers(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get layers with highest computation time.

        Args:
            top_k: Number of top layers to return

        Returns:
            List of (layer_id, computation_time_ms) tuples
        """
        if self._lock:
            with self._lock:
                return self._get_top_computation_layers_impl(top_k)
        return self._get_top_computation_layers_impl(top_k)

    def _get_top_computation_layers_impl(self, top_k: int) -> List[Tuple[str, float]]:
        """Internal implementation of get_top_computation_layers."""
        if not self.profiles:
            return []

        # Use heap for efficient top-k selection
        import heapq

        top_layers = heapq.nlargest(
            top_k, self.profiles.items(), key=lambda x: x[1].computation_time
        )
        return [(lid, prof.computation_time * 1000) for lid, prof in top_layers]

    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary with thread safety."""
        if self._lock:
            with self._lock:
                return self._get_summary_impl()
        return self._get_summary_impl()

    def _get_summary_impl(self) -> Dict[str, Any]:
        """Internal implementation of summary generation."""
        total_memory = sum(p.memory_usage for p in self.profiles.values())
        total_compute = sum(p.computation_time for p in self.profiles.values())
        total_recompute = sum(p.recompute_time for p in self.profiles.values())

        checkpointed_layers = [
            lid for lid, p in self.profiles.items() if p.checkpoint_count > 0
        ]

        return {
            "total_layers": len(self.profiles),
            "checkpointed_layers": len(checkpointed_layers),
            "total_memory_mb": total_memory,
            "total_compute_time_sec": total_compute,
            "total_recompute_time_sec": total_recompute,
            "memory_saved_mb": sum(
                p.memory_usage
                for lid, p in self.profiles.items()
                if lid in checkpointed_layers
            ),
            "recompute_overhead_ratio": (
                total_recompute / total_compute if total_compute > 0 else 0
            ),
        }

    def clear(self) -> None:
        """Clear all profiling data."""
        if self._lock:
            with self._lock:
                self.profiles.clear()
                self._profile_history.clear()
                self.step_count = 0
        else:
            self.profiles.clear()
            self._profile_history.clear()
            self.step_count = 0


class BaseSelectionStrategy:
    """Base class for selection strategies.

    Provides common functionality for all selection strategies and defines
    the interface that subclasses must implement.

    Args:
        config: Configuration for selective checkpointing
    """

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        """Initialize the selection strategy.

        Args:
            config: Configuration for selective checkpointing
        """
        self.config = config

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Determine if a layer should be checkpointed.

        Args:
            layer_id: Identifier for the layer
            profile: Optional profiling data for the layer

        Returns:
            True if the layer should be checkpointed

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def update_selection(
        self, profiles: Dict[str, LayerProfile], step: int
    ) -> Set[str]:
        """Update layer selection based on profiles.

        Args:
            profiles: Dictionary of layer profiles
            step: Current training step

        Returns:
            Set of layer IDs to checkpoint (empty by default)
        """
        return set()


class UniformStrategy(BaseSelectionStrategy):
    """Uniform checkpointing strategy with optimized layer ID parsing."""

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        super().__init__(config)
        self._cache: Dict[str, bool] = {}

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Checkpoint every N layers with caching."""
        # Check cache first
        if layer_id in self._cache:
            return self._cache[layer_id]

        # Extract layer number efficiently
        result = False
        parts = layer_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            layer_num = int(parts[1])
            result = layer_num % self.config.checkpoint_interval == 0

        # Cache result
        self._cache[layer_id] = result
        return result


class ManualStrategy(BaseSelectionStrategy):
    """Manual checkpointing strategy."""

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        super().__init__(config)
        self.checkpoint_set = set(config.layers_to_checkpoint)
        self.skip_set = set(config.layers_to_skip)

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Use explicit layer lists."""
        if layer_id in self.skip_set:
            return False
        return layer_id in self.checkpoint_set if self.checkpoint_set else False


class MemoryBasedStrategy(BaseSelectionStrategy):
    """Memory-based checkpointing strategy."""

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Checkpoint layers exceeding memory threshold."""
        if profile is None:
            return False
        return profile.memory_usage > self.config.memory_threshold_mb


class ComputationBasedStrategy(BaseSelectionStrategy):
    """Computation-based checkpointing strategy."""

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Checkpoint computationally expensive layers."""
        if profile is None:
            return False

        compute_ms = profile.computation_time * 1000
        if compute_ms <= self.config.computation_threshold_ms:
            return False

        # Check recompute factor
        if profile.recompute_time > 0 and profile.computation_time > 0:
            factor = profile.recompute_time / profile.computation_time
            return factor < self.config.recompute_factor
        return True


class AdaptiveStrategy(BaseSelectionStrategy):
    """Adaptive checkpointing strategy."""

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        super().__init__(config)
        self.selected_layers: Set[str] = set()
        self._lock = threading.RLock() if config.thread_safe else None

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Use dynamically selected layers."""
        if self._lock:
            with self._lock:
                return layer_id in self.selected_layers
        return layer_id in self.selected_layers

    def update_selection(
        self, profiles: Dict[str, LayerProfile], step: int
    ) -> Set[str]:
        """Update selection based on profiling data."""
        if step < self.config.profile_warmup_steps:
            # During warmup, checkpoint all layers
            return set(profiles.keys())

        if step % self.config.profile_update_interval != 0:
            return self.selected_layers

        # Select top percentile by memory usage
        sorted_profiles = sorted(
            profiles.values(), key=lambda p: p.memory_usage, reverse=True
        )
        cutoff_idx = int(
            len(sorted_profiles) * self.config.adaptive_threshold_percentile / 100
        )
        new_selection = {p.layer_id for p in sorted_profiles[:cutoff_idx]}

        if self._lock:
            with self._lock:
                self.selected_layers = new_selection
        else:
            self.selected_layers = new_selection

        return self.selected_layers


class HybridStrategy(BaseSelectionStrategy):
    """Hybrid checkpointing strategy combining multiple factors."""

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        super().__init__(config)
        self.selected_layers: Set[str] = set()
        self._lock = threading.RLock() if config.thread_safe else None

    def should_checkpoint(
        self, layer_id: str, profile: Optional[LayerProfile] = None
    ) -> bool:
        """Use combined selection criteria."""
        if self._lock:
            with self._lock:
                return layer_id in self.selected_layers
        return layer_id in self.selected_layers

    def update_selection(
        self, profiles: Dict[str, LayerProfile], step: int
    ) -> Set[str]:
        """Update selection based on combined factors."""
        if step < self.config.profile_warmup_steps:
            return set(profiles.keys())

        if step % self.config.profile_update_interval != 0:
            return self.selected_layers

        # Score layers based on multiple factors
        scored_layers = []
        for profile in profiles.values():
            memory_score = profile.memory_usage / max(
                self.config.memory_threshold_mb, 1.0
            )
            compute_score = (
                profile.computation_time
                * 1000
                / max(self.config.computation_threshold_ms, 1.0)
            )

            # Penalize expensive recomputation
            recompute_penalty = 1.0
            if profile.recompute_time > 0 and profile.computation_time > 0:
                factor = profile.recompute_time / profile.computation_time
                if factor > self.config.recompute_factor:
                    recompute_penalty = 0.1

            total_score = (memory_score + compute_score) * recompute_penalty
            scored_layers.append((profile.layer_id, total_score))

        # Select top scoring layers
        scored_layers.sort(key=lambda x: x[1], reverse=True)
        cutoff_idx = int(
            len(scored_layers) * self.config.adaptive_threshold_percentile / 100
        )
        new_selection = {lid for lid, _ in scored_layers[:cutoff_idx]}

        if self._lock:
            with self._lock:
                self.selected_layers = new_selection
        else:
            self.selected_layers = new_selection

        return self.selected_layers


class SelectiveRecomputeManager:
    """Manager for selective activation recomputation across a model."""

    def __init__(self, config: SelectiveCheckpointConfig) -> None:
        """Initialize the selective recompute manager.

        Args:
            config: Configuration for selective checkpointing

        Raises:
            ValueError: If configuration is invalid
        """
        config.validate()
        self.config = config
        self.profiler = LayerProfiler(config) if config.profile_enabled else None
        self.step_count = 0
        self._layer_counter = 0
        self._layer_map: Dict[str, weakref.ref[nn.Module]] = {}
        self._lock = threading.RLock() if config.thread_safe else None

        # Initialize strategy
        self.strategy = self._create_strategy(config)

        if config.verbose:
            logger.info(
                f"Initialized SelectiveRecomputeManager with "
                f"strategy: {config.strategy.value}"
            )

    def _create_strategy(
        self, config: SelectiveCheckpointConfig
    ) -> BaseSelectionStrategy:
        """Create the appropriate strategy based on configuration."""
        strategy_map = {
            SelectionStrategy.UNIFORM: UniformStrategy,
            SelectionStrategy.MANUAL: ManualStrategy,
            SelectionStrategy.MEMORY_BASED: MemoryBasedStrategy,
            SelectionStrategy.COMPUTATION_BASED: ComputationBasedStrategy,
            SelectionStrategy.ADAPTIVE: AdaptiveStrategy,
            SelectionStrategy.HYBRID: HybridStrategy,
        }

        strategy_class = strategy_map.get(config.strategy)
        if strategy_class is None:
            raise ValueError(f"Unknown strategy: {config.strategy}")

        return strategy_class(config)

    def should_checkpoint_layer(self, layer_id: str) -> bool:
        """Determine if a layer should be checkpointed based on the strategy.

        Args:
            layer_id: Identifier for the layer

        Returns:
            True if the layer should be checkpointed
        """
        # Get profile if available
        profile = None
        if self.profiler is not None:
            profile = self.profiler.get_profile(layer_id)

        # Special handling for warmup phase
        if (
            self.config.strategy
            in [SelectionStrategy.ADAPTIVE, SelectionStrategy.HYBRID]
            and self.step_count < self.config.profile_warmup_steps
        ):
            return True

        return self.strategy.should_checkpoint(layer_id, profile)

    def update_selection(self) -> None:
        """Update the selection of layers to checkpoint based on profiling data."""
        if self.profiler is None:
            return

        # Let strategy handle selection update
        profiles = self.profiler.profiles
        if profiles:
            self.strategy.update_selection(profiles, self.step_count)

            if (
                self.config.verbose
                and self.step_count % self.config.profile_update_interval == 0
            ):
                if hasattr(self.strategy, "selected_layers"):
                    selected = cast(Any, self.strategy).selected_layers
                    logger.info(
                        f"Updated checkpoint selection: "
                        f"{len(selected)}/{len(profiles)} layers"
                    )

    def checkpoint_layer(
        self,
        layer: Union[nn.Module, Callable[..., Any]],
        *args: Any,
        layer_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Checkpoint a layer with selective recomputation.

        Args:
            layer: The layer module or callable to potentially checkpoint
            *args: Positional arguments to the layer
            layer_id: Optional identifier for the layer
            **kwargs: Keyword arguments to the layer

        Returns:
            Output of the layer (checkpointed or not based on selection)

        Raises:
            RuntimeError: If layer execution fails
        """
        # Validate inputs
        if layer is None:
            raise ValueError("Layer cannot be None")

        # Generate layer ID if needed
        if layer_id is None:
            if self._lock:
                with self._lock:
                    layer_id = f"layer_{self._layer_counter}"
                    self._layer_counter += 1
            else:
                layer_id = f"layer_{self._layer_counter}"
                self._layer_counter += 1

        # Update step count and selection
        if self._lock:
            with self._lock:
                self.step_count += 1
        else:
            self.step_count += 1

        self.update_selection()

        # Decide whether to checkpoint
        should_checkpoint = self.should_checkpoint_layer(layer_id)

        # Record decision
        if self.profiler is not None:
            self.profiler.record_checkpoint_decision(layer_id, should_checkpoint)

        try:
            if should_checkpoint:
                return self._checkpoint_with_profiling(layer, layer_id, *args)
            else:
                return self._forward_with_profiling(layer, layer_id, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing layer {layer_id}: {e}")
            raise RuntimeError(f"Layer execution failed for {layer_id}") from e

    def _checkpoint_with_profiling(
        self, layer: Callable[..., Any], layer_id: str, *args: Any
    ) -> Any:
        """Execute layer with checkpointing and optional profiling."""
        if self.config.profile_enabled and self.profiler is not None:
            return SelectiveCheckpointFunction.apply(
                layer,
                self.config.preserve_rng_state,
                layer_id,
                self.profiler,
                *args,
            )
        else:
            # Use standard PyTorch checkpointing
            return torch_checkpoint(
                layer,
                *args,
                use_reentrant=self.config.use_reentrant,
                preserve_rng_state=self.config.preserve_rng_state,
            )

    def _forward_with_profiling(
        self, layer: Callable[..., Any], layer_id: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute layer without checkpointing, with optional profiling."""
        if self.profiler is not None:
            start_time = time.time()
            start_memory = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )

            output = layer(*args, **kwargs)

            elapsed_time = time.time() - start_time
            memory_used = (
                torch.cuda.memory_allocated() - start_memory
                if torch.cuda.is_available()
                else 0
            )
            self.profiler.record_forward(layer_id, elapsed_time, memory_used)

            return output
        else:
            return layer(*args, **kwargs)

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap a model with selective checkpointing.

        Args:
            model: The model to wrap

        Returns:
            Model with selective checkpointing applied

        Raises:
            ValueError: If model is None
        """
        if model is None:
            raise ValueError("Model cannot be None")

        wrapped_count = 0

        # Find all eligible layers
        def find_layers(module: nn.Module, prefix: str = "") -> None:
            nonlocal wrapped_count
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                # Check if this is a layer we want to potentially checkpoint
                if isinstance(
                    child, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)
                ):
                    # Store weak reference to avoid circular references
                    self._layer_map[full_name] = weakref.ref(child)

                    # Wrap the forward method
                    original_forward = child.forward
                    setattr(
                        child,
                        "forward",
                        functools.partial(
                            self._wrapped_forward, original_forward, full_name
                        ),
                    )
                    wrapped_count += 1

                    if self.config.verbose:
                        logger.debug(f"Wrapped layer: {full_name}")

                # Recursively process children
                find_layers(child, full_name)

        find_layers(model)

        if self.config.verbose:
            logger.info(f"Wrapped {wrapped_count} layers with selective checkpointing")

        return model

    def _wrapped_forward(
        self,
        original_forward: Callable[..., Any],
        layer_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Wrapped forward function for selective checkpointing.

        Args:
            original_forward: The original forward function
            layer_id: Identifier for the layer
            *args: Positional arguments to forward
            **kwargs: Keyword arguments to forward

        Returns:
            Output of the forward function
        """
        return self.checkpoint_layer(
            original_forward, *args, layer_id=layer_id, **kwargs
        )

    def get_profiling_report(self) -> Dict[str, Any]:
        """Get profiling report if profiling is enabled.

        Returns:
            Dictionary containing profiling statistics
        """
        if self.profiler is None:
            return {"error": "Profiling not enabled"}

        summary = self.profiler.get_summary()
        summary["top_memory_layers"] = self.profiler.get_top_memory_layers(5)
        summary["top_computation_layers"] = self.profiler.get_top_computation_layers(5)

        # Get selected layers from strategy if available
        selected_layers: List[str] = []
        if hasattr(self.strategy, "selected_layers"):
            strategy_with_selection = cast(Any, self.strategy)
            if (
                hasattr(strategy_with_selection, "_lock")
                and strategy_with_selection._lock
            ):
                with strategy_with_selection._lock:
                    selected_layers = list(strategy_with_selection.selected_layers)
            else:
                selected_layers = list(strategy_with_selection.selected_layers)

        summary["selected_layers"] = selected_layers
        summary["selection_strategy"] = self.config.strategy.value

        return summary

    def reset_profiling(self) -> None:
        """Reset profiling statistics."""
        if self.profiler is not None:
            self.profiler.clear()

        if self._lock:
            with self._lock:
                self.step_count = 0
        else:
            self.step_count = 0

        # Reset strategy state if applicable
        if hasattr(self.strategy, "selected_layers"):
            strategy_with_selection = cast(Any, self.strategy)
            if (
                hasattr(strategy_with_selection, "_lock")
                and strategy_with_selection._lock
            ):
                with strategy_with_selection._lock:
                    strategy_with_selection.selected_layers.clear()
            else:
                strategy_with_selection.selected_layers.clear()

        if self.config.verbose:
            logger.info("Reset profiling statistics")


def selective_checkpoint(
    func: Callable[..., Any],
    *args: Any,
    config: Optional[SelectiveCheckpointConfig] = None,
    layer_id: Optional[str] = None,
    manager: Optional[SelectiveRecomputeManager] = None,
    **kwargs: Any,
) -> Any:
    """Convenience function for selective checkpointing.

    Args:
        func: Function to potentially checkpoint
        *args: Positional arguments to the function
        config: Optional configuration (creates new manager if provided)
        layer_id: Optional layer identifier
        manager: Optional existing manager to use
        **kwargs: Keyword arguments to the function

    Returns:
        Output of the function

    Raises:
        ValueError: If func is None
    """
    if func is None:
        raise ValueError("Function cannot be None")

    if manager is None:
        if config is None:
            config = SelectiveCheckpointConfig()
        manager = SelectiveRecomputeManager(config)

    return manager.checkpoint_layer(func, *args, layer_id=layer_id, **kwargs)


def create_selective_checkpoint_wrapper(
    config: SelectiveCheckpointConfig,
) -> Callable[..., Any]:
    """Create a reusable selective checkpoint wrapper.

    Args:
        config: Configuration for selective checkpointing

    Returns:
        Wrapper function that can be used for checkpointing

    Raises:
        ValueError: If configuration is invalid
    """
    manager = SelectiveRecomputeManager(config)

    def wrapper(
        func: Callable[..., Any],
        *args: Any,
        layer_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Wrapper function for selective checkpointing.

        Args:
            func: Function to checkpoint
            *args: Positional arguments
            layer_id: Optional layer identifier
            **kwargs: Keyword arguments

        Returns:
            Output of the function
        """
        if func is None:
            raise ValueError("Function cannot be None")
        return manager.checkpoint_layer(func, *args, layer_id=layer_id, **kwargs)

    # Attach manager for external access
    setattr(wrapper, "manager", manager)
    return wrapper
