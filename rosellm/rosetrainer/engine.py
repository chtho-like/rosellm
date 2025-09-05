import os
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist

# References:
# [1] Shoeybi, M. et al. "Megatron-LM: Training Multi-Billion Parameter Language Models
#     Using Model Parallelism." arXiv:1909.08053 (2019)
# [2] PyTorch DistributedDataParallel: https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel
# [3] Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion
#     Parameter Models." arXiv:1910.02054 (2019)


class RoseTrainer:
    """
    Main engine for RoseLLM's distributed training framework.
    Handles initialization of distributed training, model parallelism,
    and optimization strategies.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        """
        Initialize the RoseTrainer.

        Args:
            model: The PyTorch model to train.
            optimizer: The optimizer to use for training.
            config: Configuration dictionary containing training settings.
            local_rank: Local rank for distributed training.
            world_size: Total number of processes for distributed training.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config

        # Initialize distributed training
        self.local_rank = (
            local_rank
            if local_rank is not None
            else int(os.environ.get("LOCAL_RANK", 0))
        )
        self.world_size = (
            world_size
            if world_size is not None
            else int(os.environ.get("WORLD_SIZE", 1))
        )
        self.distributed = self.world_size > 1

        self._initialize_distributed()

        # Default device
        self.device = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Move model to device
        self.model.to(self.device)

        # Configure distributed model if needed
        if self.distributed:
            self._setup_distributed_model()

    def _initialize_distributed(self) -> None:
        """Initialize the distributed training environment."""
        if not self.distributed:
            return

        if not dist.is_initialized():
            # Initialize the process group
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=self.local_rank,
                world_size=self.world_size,
            )

        # Set the device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

    def _setup_distributed_model(self) -> None:
        """Set up the model for distributed training using DDP."""
        # Basic DDP setup
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Dictionary containing the input tensors.

        Returns:
            Dictionary containing loss and metrics.
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Apply gradient clipping if configured
        if self.config.get("max_grad_norm", None):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["max_grad_norm"]
            )

        # Update parameters
        self.optimizer.step()

        # Reduce loss across devices if distributed
        loss_item = loss.item()
        if self.distributed:
            # Create a tensor with the scalar loss value
            loss_tensor = torch.tensor([loss_item], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_item = (loss_tensor / self.world_size).item()

        return {"loss": loss_item}

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save a checkpoint of the model and optimizer.

        Args:
            filepath: Path where to save the checkpoint.
        """
        # Get model without DDP wrapper
        if hasattr(self.model, "module"):
            model_to_save = cast(torch.nn.Module, self.model.module)
        else:
            model_to_save = cast(torch.nn.Module, self.model)

        checkpoint = {
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }

        # Only save on rank 0 to avoid file conflicts
        if not self.distributed or self.local_rank == 0:
            torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load a checkpoint into model and optimizer.

        Args:
            filepath: Path to the checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Get model without DDP wrapper
        if hasattr(self.model, "module"):
            model_to_load = cast(torch.nn.Module, self.model.module)
        else:
            model_to_load = cast(torch.nn.Module, self.model)

        model_to_load.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Update config with loaded config
        self.config.update(checkpoint.get("config", {}))
