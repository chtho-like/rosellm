from typing import Callable, Dict, Optional, cast

import torch
import torch.distributed as dist

# References:
# [1] PyTorch DistributedDataParallel:
#     https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel
# [2] Goyal, P. et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour."
#     arXiv:1706.02677 (2017)
# [3] Li, S. et al. "PyTorch Distributed: Experiences on Accelerating
#     Data Parallel Training." arXiv:2006.15704 (2020)
# [4] Ott, M. et al. "fairseq: A Fast, Extensible Toolkit for Sequence Modeling."
#     NAACL-HLT (2019)


class DataParallelTrainer:
    """
    Data parallelism implementation for RoseLLM.
    Handles distributed data parallel training with efficient gradient
    averaging and communication.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        local_rank: int,
        world_size: int,
        gradient_accumulation_steps: int = 1,
        fp16: bool = False,
        grad_clip: Optional[float] = None,
    ):
        """
        Initialize the DataParallelTrainer.

        Args:
            model: The PyTorch model to parallelize.
            device: The device to place the model on.
            local_rank: Local rank of the current process.
            world_size: Total number of processes.
            gradient_accumulation_steps: Number of steps to accumulate gradients.
            fp16: Whether to use FP16 precision.
            grad_clip: Value to clip gradients to.
        """
        self.model = model
        self.device = device
        self.local_rank = local_rank
        self.world_size = world_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.grad_clip = grad_clip

        # Initialize step counter for gradient accumulation
        self.accumulation_step = 0

        # Configure distributed model
        self._setup_distributed_model()

    def _setup_distributed_model(self) -> None:
        """Set up the model for distributed data parallel training."""
        # Move model to device
        self.model.to(self.device)

        # Set up DDP
        if self.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                broadcast_buffers=False,  # Don't sync buffers for better performance
                find_unused_parameters=False,  # Set to True if needed
            )

    def forward_backward(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Perform forward and backward passes with gradient accumulation.

        Args:
            batch: Dictionary containing the input tensors.
            optimizer: The optimizer to use.
            loss_fn: Optional custom loss function.

        Returns:
            The loss tensor.
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # First accumulation step: zero gradients
        if self.accumulation_step == 0:
            optimizer.zero_grad()

        # Forward pass
        outputs = self.model(**batch)

        # Calculate loss
        if loss_fn is not None:
            loss = loss_fn(outputs, batch)
        else:
            # Default behavior assuming model returns loss
            loss = outputs.loss

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps

        # Backward pass
        scaled_loss.backward()

        # Increment accumulation step
        self.accumulation_step += 1

        # If accumulation is complete, update parameters
        if self.accumulation_step == self.gradient_accumulation_steps:
            # Apply gradient clipping if configured
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Update parameters
            optimizer.step()

            # Reset accumulation step
            self.accumulation_step = 0

        # Return the unscaled loss for reporting
        return loss

    def all_reduce_gradients(self) -> None:
        """
        Manually all-reduce gradients across processes.
        This is an alternative to using DDP if finer control is needed.
        """
        if self.world_size <= 1:
            return

        # Get all parameter gradients
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                # All-reduce across processes
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                # Average by dividing by world size
                param.grad.data = param.grad.data / self.world_size

    def get_model(self) -> torch.nn.Module:
        """
        Get the underlying model (unwrapping from DDP if necessary).

        Returns:
            The PyTorch model.
        """
        if hasattr(self.model, "module"):
            return cast(torch.nn.Module, self.model.module)
        return cast(torch.nn.Module, self.model)

    def sync_model_parameters(self) -> None:
        """Ensure model parameters are synchronized across all processes."""
        if self.world_size <= 1:
            return

        # All-reduce on parameters to ensure uniformity
        with torch.no_grad():
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
