import hashlib
import logging
import os
import time
from typing import Any, Dict, Optional, Union, cast

import psutil
import torch
import torch.distributed as dist

from .config import TrainingConfig, validate_config
from .scheduler import OptimizerParamScheduler

# References:
# [1] Shoeybi, M. et al. "Megatron-LM: Training Multi-Billion Parameter Language Models
#     Using Model Parallelism." arXiv:1909.08053 (2019)
# [2] PyTorch DistributedDataParallel:
#     https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel
# [3] Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion
#     Parameter Models." arXiv:1910.02054 (2019)


logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


class DistributedInitializationError(Exception):
    """Raised when distributed initialization fails."""

    pass


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
        config: Union[Dict[str, Any], TrainingConfig],
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

        # Validate and convert configuration
        self.config = validate_config(config)
        self.config_dict = self.config.to_dict()  # For backward compatibility

        # Memory tracking
        self.memory_stats = {
            "peak_memory_gb": 0.0,
            "current_memory_gb": 0.0,
            "cpu_memory_gb": 0.0,
        }

        # Performance tracking
        self.performance_stats = {
            "samples_per_second": 0.0,
            "tokens_per_second": 0.0,
            "time_per_step": 0.0,
            "total_samples": 0,
            "total_steps": 0,
        }
        self._step_start_time: Optional[float] = None

        # Gradient clipping configuration from validated config
        self.grad_clip_type = self.config.gradient.clip_type
        self.grad_clip_value = self.config.gradient.clip_value

        # Initialize scheduler if enabled
        self.scheduler = None
        if self.config.scheduler.enabled:
            self.scheduler = OptimizerParamScheduler(
                optimizer=self.optimizer,
                init_lr=self.config.scheduler.init_lr,
                max_lr=self.config.scheduler.max_lr,
                min_lr=self.config.scheduler.min_lr,
                lr_warmup_steps=self.config.scheduler.lr_warmup_steps,
                lr_decay_steps=self.config.scheduler.lr_decay_steps,
                lr_decay_style=self.config.scheduler.lr_decay_style,
                start_wd=self.config.scheduler.start_wd,
                end_wd=self.config.scheduler.end_wd,
                wd_incr_steps=self.config.scheduler.wd_incr_steps,
                wd_incr_style=self.config.scheduler.wd_incr_style,
                wsd_decay_steps=self.config.scheduler.wsd_decay_steps,
                lr_wsd_decay_style=self.config.scheduler.lr_wsd_decay_style,
            )
            logger.info(
                f"Initialized learning rate scheduler with "
                f"{self.config.scheduler.lr_decay_style} decay"
            )

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

    def _update_performance_stats(self, batch_size: int) -> None:
        """Update performance tracking statistics."""
        if self._step_start_time is not None:
            step_time = time.time() - self._step_start_time
            self.performance_stats["time_per_step"] = step_time
            self.performance_stats["samples_per_second"] = batch_size / step_time
            self.performance_stats["total_samples"] += batch_size
            self.performance_stats["total_steps"] += 1

            # Calculate running average
            alpha = 0.1  # Exponential moving average factor
            if self.performance_stats["total_steps"] > 1:
                self.performance_stats["samples_per_second"] = (
                    alpha * (batch_size / step_time)
                    + (1 - alpha) * self.performance_stats["samples_per_second"]
                )

    def _initialize_distributed(self) -> None:
        """Initialize the distributed training environment with error handling."""
        if not self.distributed:
            return

        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            max_retries = 3
            retry_delay = 5  # seconds

            for attempt in range(max_retries):
                try:
                    # Initialize the process group
                    dist.init_process_group(
                        backend=backend,
                        rank=self.local_rank,
                        world_size=self.world_size,
                    )
                    logger.info(
                        f"Successfully initialized distributed training "
                        f"with backend={backend}"
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Failed to initialize distributed "
                            f"(attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        logger.warning(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        error_msg = (
                            f"Failed to initialize distributed training "
                            f"after {max_retries} attempts: {e}"
                        )
                        logger.error(error_msg)
                        raise DistributedInitializationError(error_msg) from e

        # Set the device
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(self.local_rank)
            except Exception as e:
                logger.error(f"Failed to set CUDA device {self.local_rank}: {e}")
                raise DistributedInitializationError(
                    f"CUDA device initialization failed: {e}"
                ) from e

    def _setup_distributed_model(self) -> None:
        """Set up the model for distributed training using DDP with error handling."""
        try:
            # Basic DDP setup with additional options
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=self.config_dict.get(
                    "find_unused_parameters", False
                ),
                gradient_as_bucket_view=self.config_dict.get(
                    "gradient_as_bucket_view", True
                ),
            )
            logger.info("Successfully initialized DistributedDataParallel")
        except Exception as e:
            logger.error(f"Failed to setup distributed model: {e}")
            raise DistributedInitializationError(
                f"DDP initialization failed: {e}"
            ) from e

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # Track step time for performance metrics
        self._step_start_time = time.time()
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
        if self.grad_clip_value is not None and self.grad_clip_type != "none":
            if self.grad_clip_type == "norm":
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_value
                )
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    logger.warning(f"Gradient norm is {total_norm}, skipping update")
                    self.optimizer.zero_grad()
                    return {"loss": float("nan"), "grad_norm": float(total_norm)}
            elif self.grad_clip_type == "value":
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self.grad_clip_value
                )

        # Update parameters
        self.optimizer.step()

        # Step the scheduler if enabled
        if self.scheduler is not None:
            self.scheduler.step()

        # Reduce loss across devices if distributed
        loss_item = loss.item()
        if self.distributed:
            # Create a tensor with the scalar loss value
            loss_tensor = torch.tensor([loss_item], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_item = (loss_tensor / self.world_size).item()

        # Update statistics
        self._update_memory_stats()
        batch_size = batch.get(
            "input_ids", batch.get("inputs", torch.tensor([]))
        ).shape[0]
        self._update_performance_stats(batch_size)

        metrics = {"loss": loss_item}

        # Add gradient norm to metrics if gradient clipping is enabled
        if self.grad_clip_value is not None and self.grad_clip_type == "norm":
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float("inf")
            )
            metrics["grad_norm"] = float(grad_norm)

        # Add memory stats to metrics if tracking is enabled
        if self.config.track_memory:
            metrics.update(
                {
                    "peak_memory_gb": self.memory_stats["peak_memory_gb"],
                    "current_memory_gb": self.memory_stats["current_memory_gb"],
                }
            )

        # Add performance stats if tracking is enabled
        if self.config.track_throughput:
            metrics.update(
                {
                    "samples_per_second": self.performance_stats["samples_per_second"],
                    "time_per_step": self.performance_stats["time_per_step"],
                }
            )

        # Add current learning rate to metrics if scheduler is enabled
        if self.scheduler is not None:
            metrics["learning_rate"] = self.scheduler.get_lr()

        return metrics

    def save_checkpoint(self, filepath: str, compute_checksum: bool = True) -> None:
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
            "config": self.config.model_dump(mode="json"),  # Serialize properly
            "performance_stats": self.performance_stats,
        }

        # Save scheduler state if enabled
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        # Add checksum for validation
        if compute_checksum:
            checkpoint_str = str(checkpoint)
            checkpoint_bytes = checkpoint_str.encode("utf-8")
            checksum = hashlib.sha256(checkpoint_bytes).hexdigest()
            checkpoint["checksum"] = checksum  # type: ignore[assignment]

        # Only save on rank 0 to avoid file conflicts
        if not self.distributed or self.local_rank == 0:
            torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str, validate_checksum: bool = True) -> None:
        """
        Load a checkpoint into model and optimizer.

        Args:
            filepath: Path to the checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Validate checksum if present
        if validate_checksum and "checksum" in checkpoint:
            stored_checksum = checkpoint.pop("checksum")
            checkpoint_bytes = str(checkpoint).encode("utf-8")
            computed_checksum = hashlib.sha256(checkpoint_bytes).hexdigest()
            if stored_checksum != computed_checksum:
                logger.warning("Checkpoint checksum mismatch! File may be corrupted.")
            checkpoint["checksum"] = stored_checksum  # Restore for completeness

        # Get model without DDP wrapper
        if hasattr(self.model, "module"):
            model_to_load = cast(torch.nn.Module, self.model.module)
        else:
            model_to_load = cast(torch.nn.Module, self.model)

        model_to_load.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scheduler state if present and scheduler is enabled
        if "scheduler" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info("Loaded scheduler state from checkpoint")

        # Update config with loaded config
        if "config" in checkpoint:
            self.config = TrainingConfig.from_dict(checkpoint["config"])
            self.config_dict = self.config.to_dict()

        # Restore performance stats if available
        if "performance_stats" in checkpoint:
            self.performance_stats.update(checkpoint["performance_stats"])

    def _update_memory_stats(self) -> None:
        """Update memory usage statistics."""
        if torch.cuda.is_available():
            # GPU memory
            current_memory = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            self.memory_stats["current_memory_gb"] = current_memory
            self.memory_stats["peak_memory_gb"] = max(
                peak_memory, self.memory_stats["peak_memory_gb"]
            )

        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**3  # Convert to GB
        self.memory_stats["cpu_memory_gb"] = cpu_memory

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        self._update_memory_stats()
        return self.memory_stats.copy()

    def cleanup(self) -> None:
        """Clean up distributed training resources."""
        if self.distributed and dist.is_initialized():
            try:
                # Synchronize all processes before cleanup
                # Note: timeout parameter may not be available in all PyTorch versions
                # We use a simple barrier without timeout for compatibility
                dist.barrier()
                dist.destroy_process_group()
                logger.info("Successfully destroyed distributed process group")
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
                # Force cleanup even if barrier fails
                try:
                    dist.destroy_process_group()
                except:
                    pass

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "performance": self.performance_stats.copy(),
            "memory": self.memory_stats.copy(),
            "config": {
                "batch_size": self.config.batch_size,
                "precision": self.config.precision.value,
                "gradient_accumulation": self.config.gradient.accumulation_steps,
            },
            "hardware": {
                "gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "distributed": self.distributed,
                "world_size": self.world_size,
            },
        }

        # Add scheduler info if enabled
        if self.scheduler is not None:
            report["scheduler"] = {
                "current_lr": self.scheduler.get_lr(),
                "current_wd": self.scheduler.get_wd(),
                "num_steps": self.scheduler.num_steps,
                "decay_style": self.config.scheduler.lr_decay_style,
            }

        return report

    def get_current_lr(self) -> float:
        """Get current learning rate from scheduler or optimizer.

        Returns:
            Current learning rate
        """
        if self.scheduler is not None:
            return self.scheduler.get_lr()
        else:
            # Return the first param group's lr from optimizer
            return float(self.optimizer.param_groups[0]["lr"])
