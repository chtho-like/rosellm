import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

logger = logging.getLogger(__name__)

# References:
# [1] Chen, T. et al. "Training Deep Nets with Sublinear Memory Cost."
#     arXiv:1604.06174 (2016)
# [2] Bulatov, Y. "Fitting larger networks into memory." (2018)
#     http://openai.com/blog/block-sparse-gpu-kernels/
# [3] PyTorch Checkpoint implementation:
#     https://pytorch.org/docs/stable/checkpoint.html
# [4] Megatron-LM implementation:
#     https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/random.py


class MemoryProfiler:
    """Memory profiling utilities for activation checkpointing."""

    def __init__(self):
        self.memory_stats = {
            "before_checkpoint": {},
            "after_checkpoint": {},
            "memory_saved": {},
            "recompute_time": {},
        }

    def profile_memory(self, tag: str, phase: str = "before") -> Dict[str, float]:
        """Profile memory usage at a specific point."""
        stats = {}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            stats["allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            stats["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3

        if phase == "before":
            self.memory_stats["before_checkpoint"][tag] = stats
        elif phase == "after":
            self.memory_stats["after_checkpoint"][tag] = stats
            # Calculate memory saved
            if tag in self.memory_stats["before_checkpoint"]:
                before = self.memory_stats["before_checkpoint"][tag]
                saved = before.get("allocated_gb", 0) - stats.get("allocated_gb", 0)
                self.memory_stats["memory_saved"][tag] = saved
                logger.info(f"Checkpointing {tag}: Saved {saved:.3f} GB memory")

        return stats

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory profiling report."""
        return {
            "total_memory_saved_gb": sum(self.memory_stats["memory_saved"].values()),
            "average_recompute_time_ms": (
                sum(self.memory_stats["recompute_time"].values())
                / max(len(self.memory_stats["recompute_time"]), 1)
                * 1000
            ),
            "detailed_stats": self.memory_stats.copy(),
        }

    def reset(self) -> None:
        """Reset all memory profiling statistics."""
        self.memory_stats = {
            "before_checkpoint": {},
            "after_checkpoint": {},
            "memory_saved": {},
            "recompute_time": {},
        }
        logger.info("Memory profiling statistics reset")

    def cleanup(self, max_entries: int = 1000) -> None:
        """Clean up old entries to prevent memory leak.

        Args:
            max_entries: Maximum number of entries to keep per category
        """
        for category in self.memory_stats:
            if isinstance(self.memory_stats[category], dict):
                if len(self.memory_stats[category]) > max_entries:
                    # Keep only the most recent entries
                    items = list(self.memory_stats[category].items())
                    self.memory_stats[category] = dict(items[-max_entries:])
                    logger.debug(
                        f"Cleaned up {category}, kept {max_entries} most recent entries"
                    )


class ActivationCheckpointing:
    """
    Utility class for managing activation checkpointing in large models.
    This implementation helps reduce memory usage during training by
    recomputing forward activations during the backward pass.
    """

    def __init__(self):
        self.profiler = MemoryProfiler()
        self.profiling_enabled = False

    def enable_profiling(self, enabled: bool = True):
        """Enable or disable memory profiling."""
        self.profiling_enabled = enabled
        if enabled:
            logger.info("Memory profiling enabled for activation checkpointing")

    def apply_to_transformer_layers(
        self,
        model: nn.Module,
        use_reentrant: bool = True,
        layer_attr: str = "transformer.h",
        chunks: Optional[int] = None,
        selection: Optional[List[int]] = None,
        profile: bool = False,
    ) -> nn.Module:
        """
        Apply activation checkpointing to transformer layers.

        Args:
            model: PyTorch model.
            use_reentrant: Whether to use the reentrant version of checkpointing.
            layer_attr: The attribute name to access transformer layers.
            chunks: Split layers into this many chunks for memory efficiency.
            selection: List of layer indices to apply checkpointing to.

        Returns:
            Model with activation checkpointing applied.
        """
        # Ensure model has the specified layer attribute
        if not hasattr(model, layer_attr.split(".")[0]):
            print(
                f"Warning: Model doesn't have attribute {layer_attr}, skipping checkpointing."
            )
            return model

        # Get layers
        current = model
        for attr in layer_attr.split("."):
            if not hasattr(current, attr):
                print(
                    f"Warning: Model doesn't have attribute {attr}, skipping checkpointing."
                )
                return model
            current = getattr(current, attr)

        # Now current should point to the list/ModuleList of transformer layers
        if not isinstance(current, nn.ModuleList) and not isinstance(current, list):
            print(
                f"Warning: {layer_attr} is not a ModuleList or list, skipping checkpointing."
            )
            return model

        layers = current
        num_layers = len(layers)

        # Determine which layers to checkpoint
        if selection is None:
            # Checkpoint all layers
            layers_to_checkpoint = list(range(num_layers))
        else:
            # Checkpoint only the specified layers
            layers_to_checkpoint = [i for i in selection if 0 <= i < num_layers]

        # Apply checkpointing to selected layers
        for i in layers_to_checkpoint:
            layer = layers[i]

            # Replace the forward method with a checkpointed version
            original_forward = layer.forward

            # Create a checkpointed forward function with profiling
            def checkpointed_forward(
                module, original_forward, layer_idx, *args, **kwargs
            ):
                # Profile memory before if enabled
                if self.profiling_enabled or profile:
                    tag = f"layer_{layer_idx}"
                    self.profiler.profile_memory(tag, "before")

                # Custom checkpointed forward function
                def custom_forward(*inputs):
                    # Track recompute time if profiling
                    if self.profiling_enabled or profile:
                        start_time = time.time()

                    # Handle both positional and keyword arguments
                    if kwargs:
                        result = original_forward(*inputs, **kwargs)
                    else:
                        result = original_forward(*inputs)

                    if self.profiling_enabled or profile:
                        self.profiler.memory_stats["recompute_time"][
                            f"layer_{layer_idx}"
                        ] = (time.time() - start_time)

                    return result

                if use_reentrant:
                    result = checkpoint.checkpoint(custom_forward, *args)
                else:
                    result = checkpoint.checkpoint(
                        custom_forward, *args, use_reentrant=False
                    )

                # Profile memory after if enabled
                if self.profiling_enabled or profile:
                    self.profiler.profile_memory(tag, "after")

                return result

            # Create a bound method for this specific layer's forward
            layer.forward = functools.partial(
                checkpointed_forward, layer, original_forward, i
            )

            logger.info(f"Applied checkpointing to layer {i}")

        return model

    def apply_to_modules(
        self,
        model: nn.Module,
        module_types: List[Type[nn.Module]],
        use_reentrant: bool = True,
        nested: bool = True,
        profile: bool = False,
    ) -> nn.Module:
        """
        Apply activation checkpointing to all modules of specified types.

        Args:
            model: PyTorch model.
            module_types: List of module types to apply checkpointing to.
            use_reentrant: Whether to use the reentrant version of checkpointing.
            nested: Whether to check for nested modules.

        Returns:
            Model with activation checkpointing applied.
        """
        modules_to_checkpoint = []

        # Function to recursively find modules of specified types
        def find_modules(module, prefix=""):
            for name, child in module.named_children():
                if any(isinstance(child, module_type) for module_type in module_types):
                    modules_to_checkpoint.append(
                        (f"{prefix}.{name}" if prefix else name, child)
                    )

                if nested:
                    find_modules(child, f"{prefix}.{name}" if prefix else name)

        # Find modules to checkpoint
        find_modules(model)

        # Apply checkpointing to found modules
        for name, module in modules_to_checkpoint:
            original_forward = module.forward

            # Create a checkpointed forward function with profiling
            def checkpointed_forward(
                module, original_forward, module_name, *args, **kwargs
            ):
                # Profile memory before if enabled
                if self.profiling_enabled or profile:
                    self.profiler.profile_memory(module_name, "before")

                def custom_forward(*inputs):
                    if self.profiling_enabled or profile:
                        start_time = time.time()

                    if kwargs:
                        result = original_forward(*inputs, **kwargs)
                    else:
                        result = original_forward(*inputs)

                    if self.profiling_enabled or profile:
                        self.profiler.memory_stats["recompute_time"][module_name] = (
                            time.time() - start_time
                        )

                    return result

                if use_reentrant:
                    result = checkpoint.checkpoint(custom_forward, *args)
                else:
                    result = checkpoint.checkpoint(
                        custom_forward, *args, use_reentrant=False
                    )

                # Profile memory after if enabled
                if self.profiling_enabled or profile:
                    self.profiler.profile_memory(module_name, "after")

                return result

            # Create a bound method for this specific module's forward
            module.forward = functools.partial(
                checkpointed_forward, module, original_forward, name
            )

            logger.info(f"Applied checkpointing to module {name}")

        return model

    def apply_to_custom_function(
        self,
        func: Callable,
        *args,
        use_reentrant: bool = True,
        preserve_rng_state: bool = True,
        profile: bool = False,
        profile_tag: str = "custom_function",
    ):
        """
        Apply activation checkpointing to a custom function.

        Args:
            func: The function to apply checkpointing to.
            *args: Arguments to pass to the function.
            use_reentrant: Whether to use the reentrant version of checkpointing.
            preserve_rng_state: Whether to preserve the RNG state.

        Returns:
            Output of the checkpointed function.
        """
        # Profile memory before if enabled
        if self.profiling_enabled or profile:
            self.profiler.profile_memory(profile_tag, "before")

        # Wrap function with timing if profiling
        if self.profiling_enabled or profile:
            original_func = func

            def timed_func(*args):
                start_time = time.time()
                result = original_func(*args)
                self.profiler.memory_stats["recompute_time"][profile_tag] = (
                    time.time() - start_time
                )
                return result

            func = timed_func

        if use_reentrant:
            result = checkpoint.checkpoint(
                func, *args, preserve_rng_state=preserve_rng_state
            )
        else:
            result = checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=preserve_rng_state
            )

        # Profile memory after if enabled
        if self.profiling_enabled or profile:
            self.profiler.profile_memory(profile_tag, "after")

        return result

    def checkpoint_sequential(
        self,
        functions: List[Callable],
        segments: int,
        input_,
        use_reentrant: bool = True,
        profile: bool = False,
    ):
        """
        Apply checkpointing to a sequence of functions.

        Args:
            functions: List of functions to checkpoint.
            segments: Number of segments to split the sequence into.
            input_: Input tensor.
            use_reentrant: Whether to use the reentrant version of checkpointing.

        Returns:
            Output of the checkpointed sequence.
        """
        # Profile memory before if enabled
        if self.profiling_enabled or profile:
            self.profiler.profile_memory("sequential", "before")

        if use_reentrant:
            result = checkpoint.checkpoint_sequential(functions, segments, input_)
        else:
            result = checkpoint.checkpoint_sequential(
                functions, segments, input_, use_reentrant=False
            )

        # Profile memory after if enabled
        if self.profiling_enabled or profile:
            self.profiler.profile_memory("sequential", "after")

        return result

    def get_profiling_report(self) -> Dict[str, Any]:
        """Get memory profiling report."""
        return self.profiler.get_memory_report()
