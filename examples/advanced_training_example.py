"""Advanced example demonstrating all the improvements made to RoseTrainer."""

import os
import time
from contextlib import contextmanager

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from rosellm.rosetrainer.config import PrecisionType, TrainingConfig
from rosellm.rosetrainer.constants import DEFAULT_GROWTH_INTERVAL, DEFAULT_INIT_SCALE

# Import RoseTrainer components with new improvements
from rosellm.rosetrainer.engine import RoseTrainer
from rosellm.rosetrainer.memory.activation_checkpoint import ActivationCheckpointing
from rosellm.rosetrainer.memory.mixed_precision import MixedPrecisionManager
from rosellm.rosetrainer.parallelism import (
    destroy_model_parallel,
    initialize_model_parallel,
)


@contextmanager
def timer(name):
    """Simple timer context manager."""
    start = time.time()
    yield
    print(f"{name} took {time.time() - start:.2f} seconds")


def main():
    """Demonstrate advanced features with improved RoseTrainer."""

    # 1. Create validated configuration using Pydantic schema
    print("=" * 60)
    print("1. Creating validated configuration with Pydantic")
    print("=" * 60)

    config = TrainingConfig(
        # Basic parameters
        batch_size=4,
        num_epochs=1,
        seed=42,
        # Precision configuration (NEW)
        precision=PrecisionType.MIXED,  # Auto-selects best precision
        # Optimizer configuration (IMPROVED)
        optimizer={
            "name": "adamw",
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
        },
        # Gradient configuration (IMPROVED)
        gradient={
            "clip_type": "norm",
            "clip_value": 1.0,
            "accumulation_steps": 2,
        },
        # Memory optimization (NEW)
        memory={
            "activation_checkpointing": True,
            "cpu_offload": False,
            "memory_efficient_attention": True,
        },
        # Parallelism configuration
        parallelism={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": None,  # Auto-calculated
        },
        # Performance tracking (NEW)
        track_memory=True,
        track_throughput=True,
        # Checkpointing
        checkpoint_interval=100,
        checkpoint_dir="./checkpoints_advanced",
    )

    print("✓ Configuration validated successfully")
    print(f"  Precision: {config.precision.value}")
    print(f"  Learning rate: {config.optimizer.learning_rate}")
    print(
        f"  Gradient clipping: {config.gradient.clip_type} @ "
        f"{config.gradient.clip_value}"
    )

    # 2. Initialize parallel state with thread safety (IMPROVED)
    print("\n" + "=" * 60)
    print("2. Initializing parallel state with thread safety")
    print("=" * 60)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # Thread-safe initialization (NEW)
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=world_size,
        )
        print(f"✓ Initialized parallel state (world_size={world_size})")

    # 3. Set up device and load model
    print("\n" + "=" * 60)
    print("3. Loading model with memory profiling")
    print("=" * 60)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    with timer("Model loading"):
        model_name = "EleutherAI/pythia-70m"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.to(device)

    print(f"✓ Model loaded: {model_name}")

    # 4. Apply memory optimizations with profiling (IMPROVED)
    print("\n" + "=" * 60)
    print("4. Applying memory optimizations with profiling")
    print("=" * 60)

    # Initialize activation checkpointing with memory profiling (NEW)
    checkpoint_manager = ActivationCheckpointing()
    checkpoint_manager.enable_profiling(True)  # Enable memory profiling

    if config.memory.activation_checkpointing:
        model = checkpoint_manager.apply_to_transformer_layers(
            model,
            layer_attr="gpt_neox.layers",
            use_reentrant=False,
            profile=True,  # Profile memory savings
        )
        print("✓ Activation checkpointing applied with profiling")

    # 5. Set up mixed precision training (NEW)
    print("\n" + "=" * 60)
    print("5. Configuring mixed precision training")
    print("=" * 60)

    precision_manager = MixedPrecisionManager(
        precision=config.precision,
        init_scale=DEFAULT_INIT_SCALE,
        growth_interval=DEFAULT_GROWTH_INTERVAL,
    )

    if precision_manager.enabled:
        model = precision_manager.convert_model(model)
        print(f"✓ Mixed precision enabled: {precision_manager.autocast_dtype}")

    # 6. Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
    )

    # 7. Initialize RoseTrainer with all improvements
    print("\n" + "=" * 60)
    print("6. Initializing RoseTrainer with enhanced features")
    print("=" * 60)

    trainer = RoseTrainer(
        model=model,
        optimizer=optimizer,
        config=config,  # Now accepts TrainingConfig directly
        local_rank=local_rank,
        world_size=world_size,
    )

    print("✓ RoseTrainer initialized with:")
    print("  - Error recovery mechanisms")
    print("  - Performance tracking")
    print("  - Memory monitoring")
    print("  - Checksum validation")

    # 8. Training loop with performance tracking
    print("\n" + "=" * 60)
    print("7. Training with performance monitoring")
    print("=" * 60)

    for step in range(5):
        # Create dummy batch
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (config.batch_size, 32), device=device
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(config.batch_size, 32, device=device),
            "labels": input_ids.clone(),
        }

        # Use mixed precision context (NEW)
        with precision_manager.autocast_context():
            metrics = trainer.train_step(batch)

        if local_rank == 0:
            print(f"Step {step + 1}:")
            print(f"  Loss: {metrics['loss']:.4f}")

            # Performance metrics (NEW)
            if "samples_per_second" in metrics:
                print(f"  Throughput: {metrics['samples_per_second']:.1f} samples/sec")

            # Memory metrics (NEW)
            if "current_memory_gb" in metrics:
                print(f"  Memory: {metrics['current_memory_gb']:.2f} GB")

    # 9. Get comprehensive reports (NEW)
    print("\n" + "=" * 60)
    print("8. Performance and Memory Reports")
    print("=" * 60)

    # Get memory profiling report
    if config.memory.activation_checkpointing:
        memory_report = checkpoint_manager.get_profiling_report()
        print(
            "Memory saved by checkpointing: "
            f"{memory_report['total_memory_saved_gb']:.3f} GB"
        )
        print(
            "Average recompute time: "
            f"{memory_report['average_recompute_time_ms']:.1f} ms"
        )

    # Get performance report (NEW)
    perf_report = trainer.get_performance_report()
    print("\nPerformance Summary:")
    print(f"  Total samples processed: {perf_report['performance']['total_samples']}")
    print(
        "  Average throughput: "
        f"{perf_report['performance']['samples_per_second']:.1f} samples/sec"
    )
    print(f"  Peak memory usage: {perf_report['memory']['peak_memory_gb']:.2f} GB")

    # 10. Save checkpoint with validation (IMPROVED)
    print("\n" + "=" * 60)
    print("9. Saving checkpoint with checksum validation")
    print("=" * 60)

    if local_rank == 0:
        checkpoint_path = "advanced_model.pt"
        trainer.save_checkpoint(checkpoint_path, compute_checksum=True)
        print("✓ Checkpoint saved with checksum validation")

        # Test loading with validation
        trainer.load_checkpoint(checkpoint_path, validate_checksum=True)
        print("✓ Checkpoint loaded and validated successfully")

    # 11. Graceful cleanup with timeout (NEW)
    print("\n" + "=" * 60)
    print("10. Graceful cleanup with timeout")
    print("=" * 60)

    trainer.cleanup()  # Graceful shutdown

    if world_size > 1:
        destroy_model_parallel()  # Thread-safe cleanup

    # Clean up memory profiler to prevent leaks (NEW)
    checkpoint_manager.profiler.cleanup(max_entries=100)

    print("✓ Training completed successfully with all improvements!")

    # Summary of improvements
    print("\n" + "=" * 60)
    print("SUMMARY OF IMPROVEMENTS DEMONSTRATED")
    print("=" * 60)
    print("✅ Thread-safe parallel state management")
    print("✅ Pydantic configuration validation")
    print("✅ Memory profiling with cleanup")
    print("✅ Comprehensive precision support (FP16/BF16/FP8)")
    print("✅ Performance metrics tracking")
    print("✅ Checkpoint validation with checksums")
    print("✅ Graceful shutdown with timeout")
    print("✅ Error recovery mechanisms")
    print("✅ Memory leak prevention")


if __name__ == "__main__":
    main()
