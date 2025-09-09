#!/usr/bin/env python3
"""Example demonstrating the Decoupled Gradient Storage feature."""

from unittest.mock import patch

import torch
import torch.nn as nn

from rosellm.rosetrainer.gradient import (
    DecoupledGradientConfig,
    DecoupledGradientManager,
    StorageMode,
)
from rosellm.rosetrainer.optimizer import (
    DistributedOptimizer,
    DistributedOptimizerConfig,
)


class SimpleModel(nn.Module):
    """Simple model for demonstration."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


def main():
    """Demonstrate decoupled gradient storage."""

    # Create model
    model = SimpleModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Configure decoupled gradient storage
    grad_config = DecoupledGradientConfig(
        enabled=True,
        storage_mode=StorageMode.CONTIGUOUS,  # Use new storage_mode enum
        device="cpu",  # Can be "cuda" if available
        thread_safe=True,  # Enable thread safety
        enable_profiling=True,  # Enable memory profiling
        debug_mode=False,  # Set to True for detailed logging
        memory_efficient_hooks=True,  # Use optimized hooks
        validate_dtypes=True,  # Validate gradient types
    )

    # Create gradient manager with different grouping strategies
    print("\n=== Testing Parameter Grouping Strategies ===")

    strategies = ["single_group", "by_requires_grad", "by_layer", "by_size"]

    for strategy in strategies:
        manager = DecoupledGradientManager(
            model, grad_config, param_grouping_strategy=strategy
        )

        stats = manager.get_memory_usage()
        print(f"\nStrategy: {strategy}")
        print(f"  Buffers: {stats['num_buffers']}")
        print(f"  Memory: {stats['total_allocated_mb']:.2f} MB")
        print(
            f"  Parameters: {stats['num_grad_parameters']} grad, "
            f"{stats['num_no_grad_parameters']} no-grad"
        )

        manager.release()

    # Use with DistributedOptimizer
    print("\n=== Using with DistributedOptimizer ===")

    # Create fresh manager
    manager = DecoupledGradientManager(
        model, grad_config, param_grouping_strategy="by_layer"
    )

    # Configure optimizer
    opt_config = DistributedOptimizerConfig(
        partition_parameters=False,
        contiguous_gradients=True,
        grad_clip_value=1.0,
    )

    # Mock distributed environment for demo
    with patch("torch.distributed.is_initialized") as mock_init, patch(
        "torch.distributed.get_world_size"
    ) as mock_world_size, patch("torch.distributed.get_rank") as mock_rank:
        mock_init.return_value = True
        mock_world_size.return_value = 1
        mock_rank.return_value = 0

        # Create optimizer with decoupled gradients
        optimizer = DistributedOptimizer(
            model.parameters(),
            torch.optim.Adam,
            {"lr": 0.001},
            opt_config,
            decoupled_grad_config=grad_config,
            model=model,
        )

        # Simulate training step
        print("\nSimulating training step...")

        # Forward pass
        batch_size = 32
        x = torch.randn(batch_size, 784)
        y_true = torch.randint(0, 10, (batch_size,))

        y_pred = model(x)
        loss = nn.CrossEntropyLoss()(y_pred, y_true)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradient storage
        print(f"Loss: {loss.item():.4f}")

        # Get memory usage after backward
        stats = manager.get_memory_usage()
        print(f"\nMemory Usage After Backward:")
        print(f"  Total Updates: {stats['total_gradient_updates']}")
        print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
        print(f"  Allocated Memory: {stats['total_allocated_mb']:.2f} MB")

        # Optimizer step
        optimizer.step()

        print(f"\nOptimizer step completed successfully!")
        print(f"Step count: {optimizer.step_count}")

    # Demonstrate gradient operations
    print("\n=== Gradient Operations ===")

    # Scale gradients
    manager.scale_gradients(0.5)
    print("Scaled gradients by 0.5")

    # Sync operations
    manager.sync_gradients_from_params(clear_param_grads=True)
    print("Synced gradients from parameters")

    manager.sync_gradients_to_params(clone=False)
    print("Synced gradients to parameters")

    # Final statistics
    final_stats = manager.get_memory_usage()
    print(f"\n=== Final Statistics ===")
    print(f"Total Operations: {final_stats['total_operations']}")
    print(f"Gradient Sync Count: {final_stats['gradient_sync_count']}")
    print(f"Last Sync Time: {final_stats.get('last_sync_time_ms', 'N/A')} ms")

    # Clean up
    manager.release()
    print("\nCleaned up gradient manager")


if __name__ == "__main__":
    main()
