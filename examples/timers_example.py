#!/usr/bin/env python3
"""Example demonstrating comprehensive timer usage in RoseLLM.

This example shows how to use the timers system for performance profiling
in distributed training scenarios.
"""

import time

import torch
import torch.distributed as dist
import torch.nn as nn

from rosellm.rosetrainer.utils import (
    TimerAggregation,
    TimerConfig,
    TimerLogLevel,
    Timers,
    get_timers,
    set_timers,
)


class SimpleModel(nn.Module):
    """Simple model for demonstration."""

    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def basic_timer_usage():
    """Demonstrate basic timer usage."""
    print("\n" + "=" * 60)
    print("Basic Timer Usage")
    print("=" * 60)

    # Create timer configuration
    config = TimerConfig(
        enabled=True,
        log_level=TimerLogLevel.INTERVAL,
        log_interval=5,
        synchronize_cuda=torch.cuda.is_available(),
        precision=3,
    )

    # Create timers instance
    timers = Timers(config)

    # Use timers with context manager
    print("\nRunning timed operations...")

    for step in range(10):
        # Time overall step
        with timers("step")():
            # Time data loading
            with timers("data-loader")():
                time.sleep(0.01)  # Simulate data loading

            # Time computation
            with timers("compute")():
                time.sleep(0.02)  # Simulate computation

            # Time communication
            with timers("communication")():
                time.sleep(0.005)  # Simulate communication

        # Log timers at intervals
        if config.should_log(step):
            print(f"\nStep {step} timers:")
            timers.log(step=step)

    # Print final summary
    print("\nFinal Timer Summary:")
    print(timers.summary())


def training_loop_with_timers():
    """Demonstrate timers in a training loop."""
    print("\n" + "=" * 60)
    print("Training Loop with Timers")
    print("=" * 60)

    # Configure timers for training
    config = TimerConfig(
        enabled=True,
        log_level=TimerLogLevel.INTERVAL,
        log_interval=10,
        synchronize_cuda=torch.cuda.is_available(),
        track_memory=torch.cuda.is_available(),
        warmup_steps=5,
        timer_categories={
            "forward": ["forward-pass", "loss-computation"],
            "backward": ["backward-pass", "gradient-sync"],
            "optimizer": ["optimizer-step", "gradient-clip"],
            "data": ["data-loading", "data-transfer"],
        },
    )

    timers = Timers(config)
    set_timers(timers)  # Set as global timers

    # Create model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining on device: {device}")
    print("Running 20 training steps with timing...")

    for step in range(20):
        # Get global timers
        timers = get_timers()

        with timers("training-step")():
            # Data loading
            with timers("data-loading")():
                batch_size = 32
                inputs = torch.randn(batch_size, 512).to(device)
                targets = torch.randint(0, 10, (batch_size,)).to(device)

            # Forward pass
            with timers("forward-pass")():
                outputs = model(inputs)

            # Loss computation
            with timers("loss-computation")():
                loss = criterion(outputs, targets)

            # Backward pass
            with timers("backward-pass")():
                optimizer.zero_grad()
                loss.backward()

            # Gradient clipping
            with timers("gradient-clip")():
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            with timers("optimizer-step")():
                optimizer.step()

        # Log at intervals
        if step > 0 and step % 10 == 0:
            print(f"\nStep {step} - Loss: {loss.item():.4f}")
            timers.log(step=step)

    # Final summary with categories
    print("\n" + "=" * 60)
    print("Training Complete - Final Timer Summary")
    print(timers.summary())


def distributed_timer_example():
    """Demonstrate distributed timer aggregation."""
    print("\n" + "=" * 60)
    print("Distributed Timer Aggregation (Simulated)")
    print("=" * 60)

    # Simulate distributed environment
    if not dist.is_initialized():
        print("Note: Running in single-process mode (distributed not initialized)")
        print("In a real distributed setting, timers would aggregate across ranks")

    # Configure timers for distributed training
    config = TimerConfig(
        enabled=True,
        log_level=TimerLogLevel.INTERVAL,
        log_interval=5,
        aggregation_method=TimerAggregation.MEAN,
        use_barrier=dist.is_initialized(),  # Use barrier only if distributed
        synchronize_cuda=torch.cuda.is_available(),
    )

    timers = Timers(config)

    # Simulate different timing on different "ranks"
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    print(f"\nSimulating rank {rank} of {world_size}")

    for step in range(10):
        # Simulate rank-dependent timing variance
        rank_delay = rank * 0.001 if dist.is_initialized() else 0

        with timers("all-reduce")():
            time.sleep(0.01 + rank_delay)

        with timers("broadcast")():
            time.sleep(0.005 + rank_delay)

        if step % 5 == 0 and step > 0:
            # In distributed mode, this would aggregate across ranks
            stats = timers._aggregate_stats()

            print(f"\nStep {step} - Aggregated Stats:")
            for name, timer_stats in stats.items():
                print(
                    f"  {name}: mean={timer_stats['mean']:.4f}s, "
                    f"count={timer_stats['count']}"
                )


def advanced_timer_features():
    """Demonstrate advanced timer features."""
    print("\n" + "=" * 60)
    print("Advanced Timer Features")
    print("=" * 60)

    # Configure with advanced features
    config = TimerConfig(
        enabled=True,
        log_level=TimerLogLevel.VERBOSE,
        track_memory=torch.cuda.is_available(),
        precision=4,
        enabled_timers=["important", "critical"],  # Only these timers will run
        output_file="timer_output.txt",
    )

    timers = Timers(config)

    print("\nDemonstrating selective timer enabling...")

    # Only enabled timers will actually time
    with timers("important")():
        print("  Running important operation...")
        time.sleep(0.1)

    with timers("not-enabled")():
        print("  This timer is disabled (no-op)")
        time.sleep(0.1)

    with timers("critical")():
        print("  Running critical operation...")
        time.sleep(0.05)

    # Check stats
    stats = timers.get_all_stats()

    print("\nTimer Statistics:")
    for name, timer_stats in stats.items():
        if timer_stats["count"] > 0:
            print(
                f"  {name}: {timer_stats['total']:.4f}s "
                f"({timer_stats['count']} calls)"
            )

    # Write to file
    if config.output_file:
        timers.write_summary(config.output_file)
        print(f"\nTimer summary written to {config.output_file}")


def memory_profiling_example():
    """Demonstrate memory profiling with timers."""
    if not torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Memory Profiling (Skipped - CUDA not available)")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("Memory Profiling with Timers")
    print("=" * 60)

    config = TimerConfig(
        enabled=True,
        track_memory=True,
        synchronize_cuda=True,
    )

    timers = Timers(config)
    device = torch.device("cuda")

    print("\nAllocating tensors with memory tracking...")

    # Time and track memory for different operations
    with timers("small-allocation")():
        small_tensor = torch.randn(100, 100, device=device)

    with timers("large-allocation")():
        large_tensor = torch.randn(1000, 1000, device=device)

    with timers("computation")():
        result = torch.matmul(large_tensor, large_tensor.T)

    # Get memory statistics
    stats = timers.get_all_stats()

    print("\nMemory Usage Statistics:")
    for name, timer_stats in stats.items():
        if "memory_used_mb" in timer_stats:
            print(f"  {name}:")
            print(f"    Time: {timer_stats['total']:.4f}s")
            print(f"    Memory Used: {timer_stats['memory_used_mb']:.2f} MB")
            print(f"    Peak Memory: {timer_stats['peak_memory_mb']:.2f} MB")

    # Clean up
    del small_tensor, large_tensor, result
    torch.cuda.empty_cache()


def main():
    """Run all timer examples."""
    print("=" * 60)
    print("RoseLLM Timer System Examples")
    print("=" * 60)

    # Run examples
    basic_timer_usage()
    training_loop_with_timers()
    distributed_timer_example()
    advanced_timer_features()
    memory_profiling_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
