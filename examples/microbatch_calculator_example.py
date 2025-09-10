#!/usr/bin/env python3
"""End-to-end example demonstrating microbatch calculator usage.

This example shows how to use the microbatch calculator for efficient
distributed training with pipeline parallelism, including:
- Basic constant microbatch configuration
- Rampup/warmup for gradual batch size increase
- Adaptive memory-aware adjustment
- Integration with training loops
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from rosellm.rosetrainer.parallelism.microbatch_calculator import (
    AdaptiveMicrobatchCalculator,
    RampupBatchSizeNumMicrobatches,
    calculate_optimal_microbatch_size,
    destroy_microbatch_calculator,
    get_micro_batch_size,
    get_microbatch_schedule,
    get_num_microbatches,
    initialize_microbatch_calculator,
    update_microbatch_calculator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """Simple model for demonstration."""

    def __init__(self, hidden_size: int = 768, num_layers: int = 12):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.activation = nn.GELU()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


def initialize_distributed():
    """Initialize distributed training if running with multiple processes."""
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # Initialize process group
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=rank,
        )

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}")
        return rank, world_size, local_rank
    else:
        logger.info("Running in single-process mode")
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def example_constant_microbatch():
    """Example using constant microbatch calculator."""
    logger.info("=" * 60)
    logger.info("Example 1: Constant Microbatch Calculator")
    logger.info("=" * 60)

    # Initialize calculator
    calc = initialize_microbatch_calculator(
        global_batch_size=128,
        micro_batch_size=16,
        data_parallel_size=2,
        calculator_type="constant",
    )

    logger.info(f"Initialized constant calculator:")
    logger.info(f"  Global batch size: {calc.global_batch_size}")
    logger.info(f"  Micro batch size: {get_micro_batch_size()}")
    logger.info(f"  Number of microbatches: {get_num_microbatches()}")
    logger.info(f"  Data parallel size: {calc.data_parallel_size}")

    # Simulate training loop
    total_samples = 0
    for iteration in range(5):
        num_microbatches = get_num_microbatches()
        micro_batch_size = get_micro_batch_size()

        logger.info(f"\nIteration {iteration + 1}:")
        logger.info(
            f"  Processing {num_microbatches} microbatches of size {micro_batch_size}"
        )

        # Process microbatches
        for mb_idx in range(num_microbatches):
            if calc.is_first_microbatch(mb_idx):
                logger.debug("  Starting gradient accumulation")

            # Simulate processing
            total_samples += micro_batch_size

            if calc.is_last_microbatch(mb_idx):
                logger.debug("  Completing gradient accumulation, updating weights")

        # Update calculator (no-op for constant)
        update_microbatch_calculator(total_samples, consistency_check=False)

    logger.info(f"\nTotal samples processed: {total_samples}")

    # Clean up
    destroy_microbatch_calculator()


def example_rampup_microbatch():
    """Example using rampup microbatch calculator."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Rampup Microbatch Calculator")
    logger.info("=" * 60)

    # Generate rampup schedule
    rampup_schedule = get_microbatch_schedule(
        start_batch_size=32,
        target_batch_size=256,
        warmup_steps=5,
        schedule_type="linear",
    )

    logger.info(f"Generated rampup schedule: {rampup_schedule}")

    # Initialize calculator with rampup
    calc = initialize_microbatch_calculator(
        global_batch_size=256,
        micro_batch_size=16,
        data_parallel_size=2,
        rampup_batch_size=rampup_schedule,
        calculator_type="rampup",
    )

    logger.info(f"Initialized rampup calculator:")
    logger.info(f"  Target global batch size: {calc.global_batch_size}")
    assert isinstance(calc, RampupBatchSizeNumMicrobatches)
    logger.info(f"  Starting batch size: {calc.current_global_batch_size}")
    logger.info(f"  Micro batch size: {get_micro_batch_size()}")

    # Simulate training with rampup
    total_samples = 0
    for iteration in range(10):
        assert isinstance(calc, RampupBatchSizeNumMicrobatches)
        current_batch_size = calc.get_current_global_batch_size()
        num_microbatches = get_num_microbatches()
        micro_batch_size = get_micro_batch_size()

        logger.info(f"\nIteration {iteration + 1}:")
        logger.info(f"  Current global batch size: {current_batch_size}")
        logger.info(
            f"  Processing {num_microbatches} microbatches of size {micro_batch_size}"
        )

        # Process batch
        batch_samples = num_microbatches * micro_batch_size * calc.data_parallel_size
        total_samples += batch_samples

        # Update calculator
        update_microbatch_calculator(total_samples, consistency_check=False)

        if isinstance(calc, RampupBatchSizeNumMicrobatches) and not calc.ramping_up:
            logger.info("  Rampup complete!")

    logger.info(f"\nTotal samples processed: {total_samples}")
    assert isinstance(calc, RampupBatchSizeNumMicrobatches)
    logger.info(f"Final batch size: {calc.get_current_global_batch_size()}")

    # Clean up
    destroy_microbatch_calculator()


def example_adaptive_microbatch():
    """Example using adaptive microbatch calculator."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Adaptive Microbatch Calculator")
    logger.info("=" * 60)

    # Calculate optimal microbatch size
    optimal_size = calculate_optimal_microbatch_size(
        model_size_gb=1.0,
        available_memory_gb=8.0,
        sequence_length=512,
        hidden_size=768,
        num_layers=12,
        pipeline_parallel_size=1,
        activation_checkpoint=False,
    )

    logger.info(f"Calculated optimal microbatch size: {optimal_size}")

    # Initialize adaptive calculator
    calc = initialize_microbatch_calculator(
        global_batch_size=128,
        micro_batch_size=optimal_size,
        data_parallel_size=2,
        calculator_type="adaptive",
    )

    logger.info(f"Initialized adaptive calculator:")
    logger.info(f"  Global batch size: {calc.global_batch_size}")
    logger.info(f"  Initial micro batch size: {get_micro_batch_size()}")
    assert isinstance(calc, AdaptiveMicrobatchCalculator)
    logger.info(f"  Min micro batch size: {calc.min_micro_batch_size}")
    logger.info(f"  Max micro batch size: {calc.max_micro_batch_size}")
    logger.info(f"  Memory threshold: {calc.memory_threshold}")

    # Simulate training with memory monitoring
    total_samples = 0
    for iteration in range(10):
        micro_batch_size = get_micro_batch_size()
        num_microbatches = get_num_microbatches()

        logger.info(f"\nIteration {iteration + 1}:")
        logger.info(f"  Current micro batch size: {micro_batch_size}")
        logger.info(f"  Number of microbatches: {num_microbatches}")

        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"  GPU memory: {memory_used:.2f}GB allocated, "
                f"{memory_reserved:.2f}GB reserved"
            )

        # Process batch
        batch_samples = num_microbatches * micro_batch_size * calc.data_parallel_size
        total_samples += batch_samples

        # Update calculator (may adjust based on memory)
        update_microbatch_calculator(total_samples, consistency_check=False)

    logger.info(f"\nTotal samples processed: {total_samples}")
    assert isinstance(calc, AdaptiveMicrobatchCalculator)
    logger.info(f"Adjustment history: {len(calc.adjustment_history)} entries")

    # Clean up
    destroy_microbatch_calculator()


def example_pipeline_integration():
    """Example showing integration with pipeline parallelism."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Pipeline Parallelism Integration")
    logger.info("=" * 60)

    # Configuration for pipeline parallel training
    pipeline_stages = 4
    global_batch_size = 256
    micro_batch_size = 32
    data_parallel_size = 2

    # Initialize calculator
    initialize_microbatch_calculator(
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
        calculator_type="constant",
    )

    num_microbatches = get_num_microbatches()

    logger.info(f"Pipeline configuration:")
    logger.info(f"  Pipeline stages: {pipeline_stages}")
    logger.info(f"  Global batch size: {global_batch_size}")
    logger.info(f"  Micro batch size: {micro_batch_size}")
    logger.info(f"  Number of microbatches: {num_microbatches}")
    logger.info(f"  Data parallel size: {data_parallel_size}")

    # Calculate pipeline metrics
    # Pipeline bubble: (p-1) * (m/p) where p=stages, m=microbatches
    bubble_size = (pipeline_stages - 1) * (num_microbatches / pipeline_stages)
    efficiency = 1 - (bubble_size / num_microbatches)

    logger.info(f"\nPipeline metrics:")
    logger.info(f"  Pipeline bubble: {bubble_size:.1f} microbatches")
    logger.info(f"  Pipeline efficiency: {efficiency:.1%}")
    logger.info(f"  Gradient accumulation steps: {num_microbatches}")

    # Simulate pipeline schedule (simplified 1F1B)
    logger.info(f"\nSimulated 1F1B pipeline schedule:")

    # Warm-up phase
    for stage in range(pipeline_stages):
        logger.info(f"  Stage {stage}: Warm-up with {stage + 1} microbatch(es)")

    # Steady state (1F1B)
    steady_state_iters = num_microbatches - pipeline_stages + 1
    if steady_state_iters > 0:
        logger.info(
            f"  All stages: Steady-state 1F1B for {steady_state_iters} iterations"
        )

    # Cool-down phase
    for stage in range(pipeline_stages - 1, 0, -1):
        logger.info(
            f"  Stage {pipeline_stages - stage - 1}: Cool-down with "
            f"{stage} microbatch(es)"
        )

    # Clean up
    destroy_microbatch_calculator()


def example_real_training_loop():
    """Example with a real training loop simulation."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Real Training Loop Simulation")
    logger.info("=" * 60)

    # Initialize distributed if available
    rank, world_size, local_rank = initialize_distributed()

    # Determine data parallel size
    data_parallel_size = world_size

    # Create model
    model = SimpleModel(hidden_size=768, num_layers=12)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Generate warmup schedule
    warmup_schedule = get_microbatch_schedule(
        start_batch_size=64,
        target_batch_size=256,
        warmup_steps=100,
        schedule_type="cosine",
    )[
        :10
    ]  # Use first 10 for demo

    # Initialize calculator
    calc = initialize_microbatch_calculator(
        global_batch_size=256,
        micro_batch_size=32,
        data_parallel_size=data_parallel_size,
        rampup_batch_size=warmup_schedule,
        calculator_type="rampup",
    )

    logger.info(f"Training configuration:")
    logger.info(
        f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters"
    )
    logger.info(f"  Device: {device}")
    logger.info(f"  Data parallel size: {data_parallel_size}")
    assert isinstance(calc, RampupBatchSizeNumMicrobatches)
    logger.info(f"  Initial batch size: {calc.current_global_batch_size}")
    logger.info(f"  Target batch size: {calc.global_batch_size}")

    # Training loop
    total_samples = 0
    num_iterations = 20

    for iteration in range(num_iterations):
        # Get current microbatch configuration
        num_microbatches = get_num_microbatches()
        micro_batch_size = get_micro_batch_size()
        assert isinstance(calc, RampupBatchSizeNumMicrobatches)
        current_global_batch = calc.get_current_global_batch_size()

        if rank == 0:
            logger.info(f"\nIteration {iteration + 1}/{num_iterations}:")
            logger.info(f"  Global batch size: {current_global_batch}")
            logger.info(f"  Microbatches: {num_microbatches} x {micro_batch_size}")

        # Zero gradients
        optimizer.zero_grad()

        total_loss = 0.0

        # Process microbatches with gradient accumulation
        for mb_idx in range(num_microbatches):
            # Create dummy input
            batch = torch.randn(
                micro_batch_size, 512, 768, device=device, requires_grad=True
            )

            # Forward pass
            output = model(batch)

            # Compute loss (dummy loss for demo)
            loss = output.mean()

            # Scale loss by number of microbatches
            scaled_loss = loss / num_microbatches

            # Backward pass
            scaled_loss.backward()

            total_loss += loss.item()

            if calc.is_last_microbatch(mb_idx):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Optimizer step
                optimizer.step()

                if rank == 0:
                    logger.info(f"    Loss: {total_loss/num_microbatches:.6f}")

        # Update samples counter
        batch_samples = num_microbatches * micro_batch_size * data_parallel_size
        total_samples += batch_samples

        # Update microbatch calculator
        update_microbatch_calculator(
            total_samples, consistency_check=dist.is_initialized()
        )

        # Check if rampup complete
        if (
            isinstance(calc, RampupBatchSizeNumMicrobatches)
            and calc.ramping_up is False
            and rank == 0
        ):
            logger.info("  Warmup complete! Reached target batch size.")

    if rank == 0:
        logger.info(f"\nTraining complete:")
        logger.info(f"  Total iterations: {num_iterations}")
        logger.info(f"  Total samples: {total_samples}")
        assert isinstance(calc, RampupBatchSizeNumMicrobatches)
        logger.info(f"  Final batch size: {calc.get_current_global_batch_size()}")

    # Clean up
    destroy_microbatch_calculator()
    cleanup_distributed()


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Microbatch Calculator Examples")
    parser.add_argument(
        "--example",
        type=str,
        choices=["constant", "rampup", "adaptive", "pipeline", "training", "all"],
        default="all",
        help="Which example to run",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    examples = {
        "constant": example_constant_microbatch,
        "rampup": example_rampup_microbatch,
        "adaptive": example_adaptive_microbatch,
        "pipeline": example_pipeline_integration,
        "training": example_real_training_loop,
    }

    if args.example == "all":
        # Run all examples except real training
        for name, func in examples.items():
            if name != "training":  # Skip training in 'all' mode
                func()
    else:
        examples[args.example]()

    logger.info("\n" + "=" * 60)
    logger.info("Examples completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
