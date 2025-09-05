from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

# References:
# [1] Chen, T. et al. "Training Deep Nets with Sublinear Memory Cost."
#     arXiv:1604.06174 (2016)
# [2] Bulatov, Y. "Fitting larger networks into memory." (2018)
#     http://openai.com/blog/block-sparse-gpu-kernels/
# [3] PyTorch Checkpoint implementation:
#     https://pytorch.org/docs/stable/checkpoint.html
# [4] Megatron-LM implementation:
#     https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/random.py


class ActivationCheckpointing:
    """
    Utility class for managing activation checkpointing in large models.
    This implementation helps reduce memory usage during training by
    recomputing forward activations during the backward pass.
    """

    @staticmethod
    def apply_to_transformer_layers(
        model: nn.Module,
        use_reentrant: bool = True,
        layer_attr: str = "transformer.h",
        chunks: Optional[int] = None,
        selection: Optional[List[int]] = None,
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

            # Create a checkpointed forward function
            def checkpointed_forward(module, original_forward, *args, **kwargs):
                # Custom checkpointed forward function
                def custom_forward(*inputs):
                    # Handle both positional and keyword arguments
                    if kwargs:
                        return original_forward(*inputs, **kwargs)
                    else:
                        return original_forward(*inputs)

                if use_reentrant:
                    return checkpoint.checkpoint(custom_forward, *args)
                else:
                    return checkpoint.checkpoint(
                        custom_forward, *args, use_reentrant=False
                    )

            # Create a bound method for this specific layer's forward
            layer.forward = lambda *args, **kwargs: checkpointed_forward(
                layer, original_forward, *args, **kwargs
            )

            print(f"Applied checkpointing to layer {i}")

        return model

    @staticmethod
    def apply_to_modules(
        model: nn.Module,
        module_types: List[Type[nn.Module]],
        use_reentrant: bool = True,
        nested: bool = True,
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

            # Create a checkpointed forward function
            def checkpointed_forward(module, original_forward, *args, **kwargs):
                def custom_forward(*inputs):
                    if kwargs:
                        return original_forward(*inputs, **kwargs)
                    else:
                        return original_forward(*inputs)

                if use_reentrant:
                    return checkpoint.checkpoint(custom_forward, *args)
                else:
                    return checkpoint.checkpoint(
                        custom_forward, *args, use_reentrant=False
                    )

            # Create a bound method for this specific module's forward
            module.forward = lambda *args, **kwargs: checkpointed_forward(
                module, original_forward, *args, **kwargs
            )

            print(f"Applied checkpointing to module {name}")

        return model

    @staticmethod
    def apply_to_custom_function(
        func: Callable,
        *args,
        use_reentrant: bool = True,
        preserve_rng_state: bool = True,
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
        if use_reentrant:
            return checkpoint.checkpoint(
                func, *args, preserve_rng_state=preserve_rng_state
            )
        else:
            return checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=preserve_rng_state
            )

    @staticmethod
    def checkpoint_sequential(
        functions: List[Callable],
        segments: int,
        input_,
        use_reentrant: bool = True,
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
        if use_reentrant:
            return checkpoint.checkpoint_sequential(functions, segments, input_)
        else:
            return checkpoint.checkpoint_sequential(
                functions, segments, input_, use_reentrant=False
            )
