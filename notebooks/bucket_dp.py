import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn


class BucketManager:
    def __init__(self, parameters, process_group, bucket_size, grad_type):
        self.bucket_size = bucket_size
        self.grad_type = grad_type
        self.process_group = process_group
        self.parameters = list(parameters)
        self.ready_params = set()

    def mark_param_as_ready(self, param):
        """Mark a parameter as ready for gradient synchronization."""
        self.ready_params.add(param)

    def wait(self):
        """Wait for gradient synchronization to complete."""
        # Implementation would handle the actual synchronization
        self.ready_params.clear()

    def reset(self):
        """Reset the bucket manager state."""
        self.ready_params.clear()


class DataParallelBucket(nn.Module):
    """
    Data Parallelism with gradient grouped into buckets to reduce the communication overhead.
    """

    def __init__(
        self, module, process_group=None, bucket_cap_mb=25, grad_type=torch.float32
    ):
        """
        Initialize the DataParallelBucket module.

        Args:
            module (nn.Module): The model to be parallelized.
            process_group: The process group for gradient synchronization, which can be either
                           a data parallel group or a context parallel group.
            bucket_cap_mb (int, optional): The maximum size of each gradient synchronization bucket in megabytes.
                                           Defaults to 25 MB.
            grad_type (torch.dtype, optional): The data type of gradients, defaulting to float32.
        """
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True  # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        grad_size = 2 if grad_type == torch.bfloat16 else 4  # float32 gradient: 4 bytes
        bucket_size = (
            bucket_cap_mb * 1024 * 1024 // grad_size
        )  # number of gradients in one bucket
        self.bucket_manager = BucketManager(
            module.parameters(),
            process_group,
            bucket_size,
            grad_type,
        )
        # Create a buffer for each parameter to store accumulated gradients
        self.grad_buffers = {}
        for param in module.parameters():
            if param.requires_grad:
                self.grad_buffers[param] = torch.zeros_like(param.data)

        self.register_backward_hook()
        self._post_backward_callback_set = (
            False  # whether the callback for wait gradient synchronization is set
        )

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)

    def register_backward_hook(self):
        """
        Registers a backward hook to manually accumulate and synchronize gradients.

        This hook serves two main purposes:
        1. PyTorch does not natively support gradient accumulation with mixed precision.
        2. After gradient accumulation, it flags parameters as ready for synchronization.

        The gradient accumulation functions are stored to prevent them from going out of scope.

        References:
        - https://github.com/NVIDIA/Megatron-LM/issues/690
        - https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html
        - https://arxiv.org/abs/2006.15704 (page 5)
        """
        self.grad_accs = []
        self.hook_handles = []  # Store hooks to prevent garbage collection
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                hook = grad_acc_fn.register_hook(
                    self._make_param_hook(param, self.bucket_manager)
                )
                self.hook_handles.append(hook)
                self.grad_accs.append(grad_acc_fn)

    def _make_param_hook(
        self, param: torch.nn.Parameter, bucket_manager: BucketManager
    ):
        """
        Creates the a hook for each parameter to handle gradient accumulation and synchronization.
        """

        def param_hook(*unused):
            """
            The hook called after the gradient is ready. It performs the following:
            1. Accumulates the gradient into a separate gradient buffer.
            2. Adds a post-backward callback to wait for gradient synchronization completion.
            3. Marks the parameter as ready for synchronization.
            """
            if param.requires_grad:
                assert param.grad is not None
                # Accumulate gradients in our buffer dictionary
                self.grad_buffers[param].add_(param.grad.data)
                param.grad = None

                # skip the gradient synchronization (gradient accumulation/PP micro batches)
                if self.require_backward_grad_sync:
                    # Add a callback to wait for gradient synchronization. Ensures the callback is added only once.
                    # Callback is executed after the backward pass. It should be added per backward pass.
                    if not self._post_backward_callback_set:
                        # Use a simpler approach with a dummy tensor that requires grad
                        dummy = torch.ones(1, requires_grad=True)

                        # Create a hook on the dummy tensor
                        def _final_callback_hook(*args):
                            self._post_backward()
                            return None

                        dummy.register_hook(_final_callback_hook)
                        # Ensure dummy tensor is part of the graph for this iteration
                        with torch.no_grad():
                            dummy.backward()
                        self._post_backward_callback_set = True

                    # mark the parameter as ready for gradient synchronization.
                    bucket_manager.mark_param_as_ready(param)

        return param_hook

    @contextlib.contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True

    def _post_backward(self):
        """
        A post-backward callback that waits for gradient synchronization to finish, then copies
        the synchronized gradients back to the parameters' grad attribute.

        This method is called after the backward pass and before the optimizer step.
        """
        self.bucket_manager.wait()
        self._post_backward_callback_set = False
        # copy to params.grad so we can use the optimizer to update the parameters
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = self.grad_buffers[p].to(p.dtype)

    def reset(self):
        """
        Reset the bucket manager and zero out gradients in the model.
        """
        self.bucket_manager.reset()
        # Clear gradient buffers
        for param, buffer in self.grad_buffers.items():
            buffer.zero_()
