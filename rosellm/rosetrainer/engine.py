import hashlib
import logging
import os
import time
from typing import Any, Dict, Optional, Union, cast

import psutil
import torch
import torch.distributed as dist

from .communication import (
    BucketConfig,
    BucketGroupConfig,
    BucketGroupManager,
    BucketManager,
    BucketStrategy,
    CommunicationBackend,
    GroupStrategy,
)
from .config import TrainingConfig, validate_config
from .gradient import (
    AdvancedGradientFinalizer,
    GradientDataTypeManager,
    GradientFinalizationConfig,
    create_gradient_data_type_manager,
)
from .mixed_precision import MixedPrecisionManager, create_mixed_precision_manager
from .parallelism import parallel_state
from .scheduler import OptimizerParamScheduler
from .utils.gradient_utils import (
    GradientClipConfig,
    apply_gradient_clipping,
    check_gradient_finite,
    get_gradient_stats,
    sync_gradients,
)
from .utils.timers import Timers, set_timers

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
    ) -> None:
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

        # Initialize timers
        self.timers = Timers(self.config.timers)
        set_timers(self.timers)  # Set as global timers for compatibility

        # Create gradient clipping configuration from validated config
        self.gradient_clip_config = GradientClipConfig(
            clip_type=self.config.gradient.clip_type,
            max_norm=self.config.gradient.clip_value or 1.0,
            norm_type=self.config.gradient.norm_type,
            error_if_nonfinite=self.config.gradient.error_if_nonfinite,
            model_parallel_reduce=self.config.gradient.model_parallel_reduce,
            use_multitensor=self.config.gradient.use_multitensor,
        )

        # Gradient tracking
        self.gradient_stats_step = 0

        # Initialize mixed precision manager if enabled
        self.mixed_precision_manager = None
        if self.config.mixed_precision.enabled:
            self.mixed_precision_manager = create_mixed_precision_manager(
                precision=self.config.mixed_precision.precision_type,
                use_dynamic_scaling=self.config.mixed_precision.use_dynamic_scaling,
                initial_scale=self.config.mixed_precision.initial_scale,
                device=None,  # Will auto-detect device later
            )

            # Override with detailed configuration if needed
            from .mixed_precision.dynamic_scaler import DynamicScalerConfig
            from .mixed_precision.mixed_precision import MixedPrecisionConfig

            detailed_mp_config = MixedPrecisionConfig(
                precision=self.config.mixed_precision.precision_type,
                use_dynamic_scaling=self.config.mixed_precision.use_dynamic_scaling,
                scaler_config=(
                    DynamicScalerConfig(
                        initial_scale=self.config.mixed_precision.initial_scale,
                        min_scale=self.config.mixed_precision.min_scale,
                        max_scale=self.config.mixed_precision.max_scale,
                        growth_factor=self.config.mixed_precision.growth_factor,
                        backoff_factor=self.config.mixed_precision.backoff_factor,
                        growth_interval=self.config.mixed_precision.growth_interval,
                        hysteresis=self.config.mixed_precision.hysteresis,
                        use_multi_tensor=self.config.mixed_precision.use_multi_tensor,
                        chunk_size=self.config.mixed_precision.chunk_size,
                        enable_inf_nan_check=(
                            self.config.mixed_precision.enable_inf_nan_check
                        ),
                        check_frequency=self.config.mixed_precision.check_frequency,
                        skip_first_n_steps=(
                            self.config.mixed_precision.skip_first_n_steps
                        ),
                        use_fused_kernels=self.config.mixed_precision.use_fused_kernels,
                        cache_inv_scale=self.config.mixed_precision.cache_inv_scale,
                        log_scale_changes=self.config.mixed_precision.log_scale_changes,
                        detailed_overflow_info=(
                            self.config.mixed_precision.detailed_overflow_info
                        ),
                        track_overflow_history=(
                            self.config.mixed_precision.track_overflow_history
                        ),
                    )
                    if self.config.mixed_precision.use_dynamic_scaling
                    else None
                ),
                autocast_enabled=self.config.mixed_precision.autocast_enabled,
                log_overflow_info=self.config.mixed_precision.log_scale_changes,
                track_scale_history=True,
            )

            self.mixed_precision_manager = MixedPrecisionManager(
                config=detailed_mp_config,
                device=None,  # Will be set after device initialization
            )

            logger.info(
                f"Initialized mixed precision manager: "
                f"{self.config.mixed_precision.precision_type}, "
                f"dynamic_scaling={self.config.mixed_precision.use_dynamic_scaling}"
            )

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
                "Initialized learning rate scheduler with "
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

        # Initialize RNG state management
        self._initialize_rng_state_management()

        # Default device
        self.device = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Move model to device
        self.model.to(self.device)

        # Update mixed precision manager device after device is set
        if self.mixed_precision_manager is not None:
            self.mixed_precision_manager.device = self.device

        # Configure distributed model if needed
        if self.distributed:
            self._setup_distributed_model()

        # Initialize gradient bucketing if enabled
        self.bucket_manager: Optional[BucketManager] = None
        self.bucket_group_manager: Optional[BucketGroupManager] = None

        if self.config.bucketing.enabled and self.distributed:
            self._initialize_gradient_bucketing()

        # Initialize advanced gradient finalization if enabled
        self.advanced_gradient_finalizer: Optional[AdvancedGradientFinalizer] = None
        self.gradient_data_type_manager: Optional[GradientDataTypeManager] = None

        if self.config.gradient.enable_advanced_finalization:
            self._initialize_advanced_gradient_finalization()

    def _initialize_gradient_bucketing(self) -> None:
        """Initialize gradient bucketing components."""
        try:
            # Convert config enums to bucketing module enums
            bucket_strategy_map = {
                "size_based": BucketStrategy.SIZE_BASED,
                "layer_based": BucketStrategy.LAYER_BASED,
                "mixed": BucketStrategy.MIXED,
                "custom": BucketStrategy.CUSTOM,
            }

            backend_map = {
                "nccl": CommunicationBackend.NCCL,
                "gloo": CommunicationBackend.GLOO,
                "auto": CommunicationBackend.AUTO,
            }

            # Create bucket configuration
            bucket_config = BucketConfig(
                strategy=bucket_strategy_map[self.config.bucketing.strategy],
                max_bucket_size_mb=self.config.bucketing.max_bucket_size_mb,
                min_bucket_size_mb=self.config.bucketing.min_bucket_size_mb,
                backend=backend_map[self.config.bucketing.backend],
                overlap_communication=self.config.bucketing.overlap_communication,
                compress_gradients=self.config.bucketing.compress_gradients,
                dynamic_bucketing=self.config.bucketing.dynamic_bucketing,
                gradient_predivision=self.config.bucketing.gradient_predivision,
                communication_timeout_ms=self.config.bucketing.communication_timeout_ms,
                bucket_cap_mb=self.config.bucketing.bucket_cap_mb,
            )

            # Initialize bucket manager
            self.bucket_manager = BucketManager(
                config=bucket_config,
                device=self.device,
                dtype=torch.float32,  # Use FP32 for gradient communication
            )

            # Initialize bucket group manager if groups are enabled
            if self.config.bucketing.enable_groups:
                group_strategy_map = {
                    "parallel": GroupStrategy.PARALLEL,
                    "sequential": GroupStrategy.SEQUENTIAL,
                    "hierarchical": GroupStrategy.HIERARCHICAL,
                    "adaptive": GroupStrategy.ADAPTIVE,
                }

                group_config = BucketGroupConfig(
                    group_strategy=group_strategy_map[
                        self.config.bucketing.group_strategy
                    ],
                    max_groups=self.config.bucketing.max_groups,
                    overlap_groups=self.config.bucketing.overlap_communication,
                )

                self.bucket_group_manager = BucketGroupManager(
                    config=group_config,
                    bucket_manager=self.bucket_manager,
                )

            logger.info(
                f"Initialized gradient bucketing with strategy: "
                f"{self.config.bucketing.strategy}, max_size: "
                f"{self.config.bucketing.max_bucket_size_mb}MB, "
                f"groups: {self.config.bucketing.enable_groups}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize gradient bucketing: {e}")
            self.bucket_manager = None
            self.bucket_group_manager = None
            raise

    def _initialize_advanced_gradient_finalization(self) -> None:
        """Initialize advanced gradient finalization components."""
        try:
            # Create gradient data type manager
            self.gradient_data_type_manager = create_gradient_data_type_manager(
                master_precision=self.config.gradient.master_precision,
                communication_precision=self.config.gradient.communication_precision,
                enable_compression=self.config.gradient.enable_gradient_compression,
                compression_threshold_mb=self.config.gradient.compression_threshold_mb,
                preserve_master_precision=True,
            )

            # Create gradient finalization config
            grad_finalization_config = GradientFinalizationConfig(
                sync_strategy="hierarchical",
                reduction_op="mean",
                dimension_order="hierarchical",
                bucket_size_mb=25.0,
                overlap_grad_sync=True,
                sync_grad_before_clip=True,
                use_contiguous_buffers=True,
                check_gradient_norm=True,
                fp16_compression=self.config.gradient.enable_gradient_compression,
                enable_gradient_stats=self.config.gradient.track_gradient_stats,
                verbose=self.config.gradient.finalization_verbose,
            )

            # Create advanced gradient finalizer
            self.advanced_gradient_finalizer = AdvancedGradientFinalizer(
                model=self.model,
                config=grad_finalization_config,
                data_type_manager=self.gradient_data_type_manager,
                enable_advanced_sync=True,
                verbose=self.config.gradient.finalization_verbose,
            )

            logger.info(
                f"Initialized advanced gradient finalization with "
                f"master_precision={self.config.gradient.master_precision}, "
                f"compression={self.config.gradient.enable_gradient_compression}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize advanced gradient finalization: {e}")
            self.advanced_gradient_finalizer = None
            self.gradient_data_type_manager = None
            raise

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
                        "Successfully initialized distributed training "
                        f"with backend={backend}"
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "Failed to initialize distributed "
                            f"(attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        logger.warning(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        error_msg = (
                            "Failed to initialize distributed training "
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

    def _initialize_rng_state_management(self) -> None:
        """Initialize RNG state management for deterministic training."""
        try:
            # Initialize RNG with configuration from training config
            if hasattr(self.config, "random") and self.config.random.enabled:
                parallel_state.initialize_parallel_rng(
                    seed=self.config.random.seed,
                    enable_cuda_graphs=self.config.random.enable_cuda_graphs,
                    cache_capacity=self.config.random.cache_capacity,
                    auto_cleanup=self.config.random.auto_cleanup,
                    enable_deterministic=self.config.random.enable_deterministic,
                    verbose=self.config.random.verbose,
                )
                logger.info(
                    f"Initialized parallel RNG state management with seed "
                    f"{self.config.random.seed}"
                )
            else:
                # Use default RNG initialization with global seed
                parallel_state.initialize_parallel_rng(
                    seed=self.config.seed, enable_deterministic=True, verbose=False
                )
                logger.info(
                    f"Initialized default parallel RNG state management with seed "
                    f"{self.config.seed}"
                )
        except Exception as e:
            logger.warning(f"Failed to initialize RNG state management: {e}")
            logger.warning("Continuing without advanced RNG state management")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Perform a single training step.

        Args:
            batch: Dictionary containing the input tensors.

        Returns:
            Dictionary containing loss and metrics.
        """
        # Track step time for performance metrics
        self._step_start_time = time.time()

        # Start overall training step timer
        self.timers.start("training-step")
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass with mixed precision if enabled
        self.timers.start("forward-compute")
        if self.mixed_precision_manager is not None:
            with self.mixed_precision_manager.autocast_context():
                outputs = self.model(**batch)
        else:
            outputs = self.model(**batch)
        self.timers.stop("forward-compute")

        # Handle different output formats
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        elif isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        else:
            raise ValueError(
                f"Model output must have 'loss' attribute or key, got {type(outputs)}"
            )

        # Backward pass with mixed precision scaling if enabled
        self.timers.start("backward-compute")
        if self.mixed_precision_manager is not None:
            self.mixed_precision_manager.backward_step(loss)
        else:
            loss.backward()
        self.timers.stop("backward-compute")

        # Handle mixed precision overflow detection and gradient processing
        overflow_detected = False
        clip_stats = {"grad_norm": 0.0, "clipped": False}

        if self.mixed_precision_manager is not None:
            # Use mixed precision manager for overflow detection and handling
            overflow_detected = self.mixed_precision_manager.check_overflow_and_step(
                self.model, self.optimizer
            )

            if overflow_detected:
                # Mixed precision detected overflow - skip this step
                logger.debug(
                    "Mixed precision overflow detected, skipping optimizer step"
                )
                return {
                    "loss": loss.item(),
                    "grad_norm": float("nan"),
                    "grad_overflow": True,
                    "optimizer_skipped": True,
                }

        # Gradient processing for non-mixed-precision or no overflow cases
        try:
            # Use advanced gradient finalization if enabled
            if self.advanced_gradient_finalizer is not None:
                finalization_stats = (
                    self.advanced_gradient_finalizer.finalize_gradients(
                        clip_gradients=True,
                        check_finite=(self.mixed_precision_manager is None),
                        normalize_gradients=self.config.gradient.normalize_gradients,
                        collect_stats=self.config.gradient.track_gradient_stats,
                        custom_sync_order=self.config.gradient.advanced_sync_order,
                    )
                )

                # Extract clip stats from finalization results
                clip_stats = {
                    "grad_norm": finalization_stats.get("gradient_norm", 0.0),
                    "clipped": finalization_stats.get("clipped", False),
                }

                # Handle non-finite gradients from advanced finalization
                if not finalization_stats.get("finite", True):
                    if self.config.gradient.error_if_nonfinite:
                        self.optimizer.zero_grad()
                        return {
                            "loss": float("nan"),
                            "grad_norm": float("nan"),
                            "finite_gradients": False,
                            "advanced_finalization": finalization_stats,
                        }

                # Log finalization stats if verbose
                if self.config.gradient.finalization_verbose:
                    logger.debug(
                        f"Advanced gradient finalization stats: {finalization_stats}"
                    )

            else:
                # Fallback to traditional gradient processing
                # Check for non-finite gradients (fallback or additional check)
                if self.mixed_precision_manager is None:
                    is_finite, finite_stats = check_gradient_finite(
                        self.model, raise_on_nonfinite=False
                    )

                    if not is_finite:
                        logger.warning(
                            "Non-finite gradients detected: "
                            f"{finite_stats['nan_parameters']} NaN, "
                            f"{finite_stats['inf_parameters']} Inf parameters"
                        )
                        if self.config.gradient.error_if_nonfinite:
                            self.optimizer.zero_grad()
                            # Create return dict with consistent types
                            error_metrics = {
                                "loss": float("nan"),
                                "grad_norm": float("nan"),
                                "finite_gradients": False,
                            }
                            # Add gradient stats separately to avoid type mixing
                            error_metrics.update(
                                {
                                    f"gradient_stat_{k}": (
                                        float(v) if isinstance(v, (int, float)) else v
                                    )
                                    for k, v in finite_stats.items()
                                }
                            )
                            return error_metrics

                # Apply advanced gradient clipping
                self.timers.start("gradient-clip")
                clip_stats = apply_gradient_clipping(
                    self.model, self.gradient_clip_config
                )
                self.timers.stop("gradient-clip")

                # Add gradient statistics to metrics if enabled
                if self.config.gradient.track_gradient_stats:
                    if (
                        self.gradient_stats_step
                        % self.config.gradient.gradient_stats_interval
                        == 0
                    ):
                        grad_stats = get_gradient_stats(
                            self.model,
                            include_histograms=(
                                self.config.gradient.include_gradient_histograms
                            ),
                        )
                        logger.info(f"Gradient stats: {grad_stats}")
                    self.gradient_stats_step += 1

        except Exception as e:
            logger.error(f"Advanced gradient processing failed: {e}")
            # Fallback to simple gradient clipping
            if (
                self.config.gradient.clip_value is not None
                and self.config.gradient.clip_type != "none"
            ):
                if self.config.gradient.clip_type == "norm":
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient.clip_value
                    )
                    clip_stats = {
                        "grad_norm": float(total_norm),
                        "clipped": total_norm > self.config.gradient.clip_value,
                    }
                elif self.config.gradient.clip_type == "value":
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), self.config.gradient.clip_value
                    )
                    clip_stats = {"grad_norm": 0.0, "clipped": True}
                else:
                    clip_stats = {"grad_norm": 0.0, "clipped": False}
            else:
                clip_stats = {"grad_norm": 0.0, "clipped": False}

        # Synchronize gradients if distributed and configured
        if self.distributed and (
            self.config.gradient.sync_on_accumulation
            or (self.gradient_stats_step % self.config.gradient.accumulation_steps == 0)
        ):
            self.timers.start("gradient-sync")
            if self.bucket_manager is not None:
                # Use gradient bucketing for synchronization
                self._synchronize_gradients_with_bucketing()
            else:
                # Use traditional gradient synchronization
                sync_gradients(self.model)
            self.timers.stop("gradient-sync")

        # Update parameters with mixed precision handling
        self.timers.start("optimizer-step")
        if self.mixed_precision_manager is not None and not overflow_detected:
            # Mixed precision optimizer step (already handled overflow)
            success = self.mixed_precision_manager.optimizer_step(
                self.optimizer,
                self.model,
                unscale_gradients=False,  # Already handled in check_overflow_and_step
                clip_gradients=False,  # Handle separately for consistency
            )
            if not success:
                logger.debug("Mixed precision optimizer step skipped due to overflow")
                return {
                    "loss": loss.item(),
                    "grad_norm": float("nan"),
                    "grad_overflow": True,
                    "optimizer_skipped": True,
                }
        elif not overflow_detected:
            # Standard optimizer step
            self.optimizer.step()
        self.timers.stop("optimizer-step")

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

        # Add gradient clipping statistics to metrics
        if "clip_stats" in locals():
            metrics["grad_norm"] = clip_stats.get("grad_norm", 0.0)
            metrics["grad_clipped"] = clip_stats.get("clipped", False)
            if "scale_factor" in clip_stats:
                metrics["grad_scale_factor"] = clip_stats["scale_factor"]

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

        # Add mixed precision statistics if enabled
        if self.mixed_precision_manager is not None:
            mp_stats = self.mixed_precision_manager.get_statistics()
            metrics.update(
                {
                    "mixed_precision_success_rate": mp_stats.get("success_rate", 1.0),
                    "mixed_precision_overflow_count": mp_stats.get("overflow_count", 0),
                    "mixed_precision_total_steps": mp_stats.get("total_steps", 0),
                }
            )

            # Add current loss scale if available
            if "scaler_info" in mp_stats:
                scaler_info = mp_stats["scaler_info"]
                if "current_scale" in scaler_info:
                    metrics["loss_scale"] = scaler_info["current_scale"]
            elif "current_scale" in mp_stats:
                metrics["loss_scale"] = mp_stats["current_scale"]

        # Stop overall training step timer
        self.timers.stop("training-step")

        # Log timers if configured
        if self.config.timers.should_log(int(self.performance_stats["total_steps"])):
            self.timers.log(step=int(self.performance_stats["total_steps"]))

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

        # Save mixed precision state if enabled
        if self.mixed_precision_manager is not None:
            checkpoint["mixed_precision"] = self.mixed_precision_manager.state_dict()

        # Save RNG state if available
        rng_checkpoint = parallel_state.checkpoint_parallel_rng()
        if rng_checkpoint is not None:
            checkpoint["rng_state"] = rng_checkpoint

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

        # Load mixed precision state if present and manager is enabled
        if "mixed_precision" in checkpoint and self.mixed_precision_manager is not None:
            self.mixed_precision_manager.load_state_dict(checkpoint["mixed_precision"])
            logger.info("Loaded mixed precision state from checkpoint")

        # Load RNG state if present
        if "rng_state" in checkpoint:
            try:
                parallel_state.restore_parallel_rng(checkpoint["rng_state"])
                logger.info("Loaded RNG state from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load RNG state from checkpoint: {e}")

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
        # Clean up advanced gradient finalization resources
        if self.advanced_gradient_finalizer is not None:
            try:
                self.advanced_gradient_finalizer.cleanup()
                logger.info("Cleaned up advanced gradient finalizer")
            except Exception as e:
                logger.warning(f"Failed to cleanup advanced gradient finalizer: {e}")

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

    def _synchronize_gradients_with_bucketing(self) -> Dict[str, Any]:
        """
        Synchronize gradients using the gradient bucketing system.

        Returns:
            Dictionary containing synchronization statistics
        """
        if self.bucket_manager is None:
            raise RuntimeError("Bucket manager not initialized")

        sync_stats = {}

        try:
            # Reset buckets for new synchronization round
            self.bucket_manager.reset()

            # Assign gradients to buckets
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    bucket_id = self.bucket_manager.assign_gradient(
                        param_name=name,
                        gradient=param.grad,
                    )
                    logger.debug(f"Assigned {name} to bucket {bucket_id}")

            # Use bucket group manager if available,
            # otherwise use bucket manager directly
            if self.bucket_group_manager is not None:
                # Assign buckets to groups
                group_assignment_stats = (
                    self.bucket_group_manager.assign_buckets_to_groups()
                )
                sync_stats.update(group_assignment_stats)

                # Synchronize bucket groups
                group_sync_stats = self.bucket_group_manager.synchronize_groups()
                sync_stats.update(group_sync_stats)
            else:
                # Direct bucket synchronization
                bucket_sync_stats = self.bucket_manager.synchronize_buckets(
                    overlap=self.config.bucketing.overlap_communication
                )
                sync_stats.update(bucket_sync_stats)

            # Get synchronized gradients and update model parameters
            updated_gradients = self.bucket_manager.get_bucket_assignments()

            for name, param in self.model.named_parameters():
                if name in updated_gradients:
                    param.grad = updated_gradients[name]

            # Add bucketing statistics
            bucket_stats = self.bucket_manager.get_statistics()
            sync_stats["bucket_stats"] = bucket_stats

            if self.bucket_group_manager is not None:
                group_stats = self.bucket_group_manager.get_statistics()
                sync_stats["group_stats"] = group_stats

            logger.debug(f"Gradient bucketing synchronization completed: {sync_stats}")

        except Exception as e:
            logger.error(f"Gradient bucketing synchronization failed: {e}")
            # Fallback to traditional synchronization
            sync_gradients(self.model)
            sync_stats["fallback_used"] = True
            sync_stats["error"] = str(e)

        return sync_stats

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
