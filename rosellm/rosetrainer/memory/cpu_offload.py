from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn

# References:
# [1] Rajbhandari, S. et al. "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme
#     Scale Deep Learning." arXiv:2104.07857 (2021)
# [2] Ren, J. et al. "ZeRO-Offload: Democratizing Billion-Scale Model Training."
#     arXiv:2101.06840 (2021)
# [3] DeepSpeed CPU offload implementation:
#     https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/runtime/zero
# [4] FairScale CPU offload implementation:
#     https://github.com/facebookresearch/fairscale/blob/main/fairscale/optim/offload


class CPUOffloadOptimizer:
    """
    Wrapper around optimizer that offloads optimizer states to CPU
    to reduce GPU memory usage during training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        pin_memory: bool = False,
        offload_params: bool = False,
    ):
        """
        Initialize the CPU offload optimizer.

        Args:
            optimizer: Base optimizer (e.g., Adam, AdamW).
            pin_memory: Whether to pin offloaded memory for faster transfer.
            offload_params: Whether to also offload model parameters to CPU.
        """
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self.offload_params = offload_params

        # Store original param groups for reference
        self.param_groups = self.optimizer.param_groups

        # Maps from parameters to their device before offloading
        self.param_to_device: Dict[nn.Parameter, torch.device] = {}

        # Initialize CPU/GPU parameter buffers for each parameter
        self._init_param_buffers()

        # Offload optimizer states
        self._offload_optimizer_states()

        # Offload parameters if requested
        if self.offload_params:
            self._offload_parameters()

    def _init_param_buffers(self) -> None:
        """Initialize parameter buffers on CPU and GPU."""
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param.requires_grad:
                    # Store the original device
                    self.param_to_device[param] = param.device

    def _offload_optimizer_states(self) -> None:
        """Move optimizer states to CPU."""
        # Get optimizer state dict
        state_dict = self.optimizer.state_dict()

        # For each parameter with state, move state to CPU
        for param_id, param_state in state_dict["state"].items():
            for key, value in param_state.items():
                if isinstance(value, torch.Tensor):
                    # Move tensor to CPU (with pinned memory if requested)
                    if self.pin_memory:
                        param_state[key] = value.cpu().pin_memory()
                    else:
                        param_state[key] = value.cpu()

        # Load the modified state dict back
        self.optimizer.load_state_dict(state_dict)

    def _offload_parameters(self) -> None:
        """Move model parameters to CPU."""
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param.requires_grad:
                    # Store the parameter's original device and data
                    self.param_to_device[param] = param.device

                    # Create CPU copy of the parameter
                    if self.pin_memory:
                        param.data = param.data.cpu().pin_memory()
                    else:
                        param.data = param.data.cpu()

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Zero out parameter gradients.

        Args:
            set_to_none: If True, set gradients to None instead of zeros.
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> Optional[torch.Tensor]:
        """
        Perform optimization step with CPU offloading.

        Args:
            closure: Closure for re-evaluating the model and returning the loss.

        Returns:
            Optional loss value if closure is provided.
        """
        # Move all parameter gradients to the same device as parameters
        self._sync_gradients_to_parameters()

        # If parameters were offloaded, move them to the original device
        if self.offload_params:
            self._move_params_to_original_device()

        # Move optimizer states back to the original device
        self._move_optimizer_states_to_device()

        # Perform optimization step
        loss = self.optimizer.step(closure)

        # Offload optimizer states back to CPU
        self._offload_optimizer_states()

        # Offload parameters back to CPU if requested
        if self.offload_params:
            self._offload_parameters()

        return loss

    def _sync_gradients_to_parameters(self) -> None:
        """Ensure gradients are on the same device as their parameters."""
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None and param.grad.device != param.device:
                    param.grad = param.grad.to(param.device)

    def _move_params_to_original_device(self) -> None:
        """Move parameters back to their original devices."""
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if (
                    param.requires_grad
                    and param.device.type == "cpu"
                    and self.param_to_device[param].type != "cpu"
                ):
                    # Move parameter back to original device
                    param.data = param.data.to(self.param_to_device[param])

    def _move_optimizer_states_to_device(self) -> None:
        """Move optimizer states to the same device as their parameters."""
        state_dict = self.optimizer.state_dict()

        # Map from parameter ID to parameter object
        param_id_to_param = {}
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                param_id_to_param[id(param)] = param

        # For each parameter, move its state to the parameter's device
        for param_id, param_state in state_dict["state"].items():
            # Check if this is a tensor parameter ID in our map
            if param_id in param_id_to_param:
                param = param_id_to_param[param_id]
                # Move each state tensor to the parameter's device
                for key, value in param_state.items():
                    if isinstance(value, torch.Tensor) and value.device != param.device:
                        param_state[key] = value.to(param.device)

        # Load the modified state dict back
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self) -> Dict[str, Any]:
        """
        Get optimizer state dictionary.

        Returns:
            Optimizer state dictionary with CPU offloading metadata.
        """
        state_dict = self.optimizer.state_dict()
        state_dict["param_to_device"] = {
            id(param): device for param, device in self.param_to_device.items()
        }
        state_dict["pin_memory"] = self.pin_memory
        state_dict["offload_params"] = self.offload_params
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load optimizer state dictionary.

        Args:
            state_dict: State dictionary to load.
        """
        # Extract and remove wrapper-specific keys
        param_id_to_device = state_dict.pop("param_to_device", {})
        self.pin_memory = state_dict.pop("pin_memory", self.pin_memory)
        self.offload_params = state_dict.pop("offload_params", self.offload_params)

        # Load state dict into optimizer
        self.optimizer.load_state_dict(state_dict)

        # Rebuild param_to_device map
        self.param_to_device = {}

        # Map from parameter ID to parameter object
        param_id_to_param = {}
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                param_id_to_param[id(param)] = param

        # Restore param_to_device mapping
        for param_id, device in param_id_to_device.items():
            if param_id in param_id_to_param:
                self.param_to_device[param_id_to_param[param_id]] = device

        # Re-offload states and parameters
        self._offload_optimizer_states()
        if self.offload_params:
            self._offload_parameters()


class ParameterOffloader:
    """
    Helper class to offload model parameters to CPU on demand.
    Useful for large models that don't fit entirely on GPU.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        pin_memory: bool = False,
    ):
        """
        Initialize the parameter offloader.

        Args:
            model: The model to manage parameter offloading for.
            device: The device to use when parameters are needed.
            pin_memory: Whether to pin offloaded memory for faster transfer.
        """
        self.model = model
        self.device = device
        self.pin_memory = pin_memory

        # Store original parameters and their devices
        self.param_to_device = {}
        for name, param in model.named_parameters():
            self.param_to_device[name] = param.device

    def offload_all_parameters(self) -> None:
        """Offload all model parameters to CPU."""
        for name, param in self.model.named_parameters():
            if param.device.type != "cpu":
                # Save original device
                self.param_to_device[name] = param.device

                # Move to CPU
                if self.pin_memory:
                    param.data = param.data.cpu().pin_memory()
                else:
                    param.data = param.data.cpu()

                # Free gradients
                if param.grad is not None:
                    param.grad.data = param.grad.data.cpu()
                    param.grad = None

    def offload_module_parameters(self, module_names: List[str]) -> None:
        """
        Offload parameters of specific modules to CPU.

        Args:
            module_names: List of module names to offload parameters for.
        """
        for name, param in self.model.named_parameters():
            # Check if this parameter belongs to any of the modules to offload
            if any(name.startswith(module_name) for module_name in module_names):
                if param.device.type != "cpu":
                    # Save original device
                    self.param_to_device[name] = param.device

                    # Move to CPU
                    if self.pin_memory:
                        param.data = param.data.cpu().pin_memory()
                    else:
                        param.data = param.data.cpu()

                    # Free gradients
                    if param.grad is not None:
                        param.grad.data = param.grad.data.cpu()
                        param.grad = None

    def load_module_parameters(self, module_names: List[str]) -> None:
        """
        Load parameters of specific modules back to their original devices.

        Args:
            module_names: List of module names to load parameters for.
        """
        for name, param in self.model.named_parameters():
            # Check if this parameter belongs to any of the modules to load
            if any(name.startswith(module_name) for module_name in module_names):
                if (
                    param.device.type == "cpu"
                    and self.param_to_device.get(name, "cpu") != "cpu"
                ):
                    # Move back to original device
                    param.data = param.data.to(self.param_to_device[name])

    def load_all_parameters(self) -> None:
        """Load all model parameters back to their original devices."""
        for name, param in self.model.named_parameters():
            if (
                param.device.type == "cpu"
                and self.param_to_device.get(name, "cpu") != "cpu"
            ):
                # Move back to original device
                param.data = param.data.to(self.param_to_device[name])
