#!/usr/bin/env python3
"""
Multi-Parallel RNG State Management Example

This example demonstrates the advanced RNG state management capabilities
of RoseLLM's training framework. It shows how to use multi-dimensional
parallel RNG states for deterministic training across different
parallelism configurations.

Key Features Demonstrated:
- Multi-parallel RNG initialization
- State forking and isolation
- Context-based RNG switching
- Checkpoint/restore operations
- Performance monitoring
- CUDA Graph compatibility

Usage:
    # Basic usage
    python examples/rng_management_example.py

    # With different parallel configurations
    python examples/rng_management_example.py --tp-size 2 --pp-size 2

    # With CUDA Graphs enabled
    python examples/rng_management_example.py --enable-cuda-graphs

    # Distributed training simulation
    torchrun --nproc_per_node=4 examples/rng_management_example.py
"""

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import RoseLLM RNG management
from rosellm.rosetrainer.random import (
    checkpoint_parallel_rng_state,
    fork_parallel_rng_state,
    get_cuda_rng_tracker,
    get_rng_state_summary,
    initialize_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
    parallel_rng_context,
    restore_parallel_rng_state,
)


# Mock parallel state for demonstration
class MockParallelState:
    """Mock parallel state for demonstration purposes."""

    def __init__(
        self,
        tp_size: int = 1,
        pp_size: int = 1,
        dp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
    ):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.cp_size = cp_size
        self.ep_size = ep_size

        # Compute ranks based on world size and sizes
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        if world_size > 1:
            total_parallel_size = tp_size * pp_size * dp_size * cp_size * ep_size
            assert total_parallel_size == world_size, (
                f"Parallel sizes ({total_parallel_size}) don't match world size "
                f"({world_size})"
            )

        # Simple rank decomposition (for demo purposes)
        self.tp_rank = rank % tp_size
        self.pp_rank = (rank // tp_size) % pp_size
        self.dp_rank = (rank // (tp_size * pp_size)) % dp_size
        self.cp_rank = 0  # Simplified
        self.ep_rank = 0  # Simplified


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=level
    )


class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer for demonstration."""

    def __init__(self, hidden_size: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Multi-head attention
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

        # Feed-forward network
        self.ffn1 = nn.Linear(hidden_size, hidden_size * 4)
        self.ffn2 = nn.Linear(hidden_size * 4, hidden_size)

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_input = self.norm1(x)
        q = self.query(attn_input)
        k = self.key(attn_input)
        v = self.value(attn_input)

        # Simplified attention (for demo)
        attn_output = F.scaled_dot_product_attention(
            q.view(-1, self.num_heads, self.hidden_size // self.num_heads),
            k.view(-1, self.num_heads, self.hidden_size // self.num_heads),
            v.view(-1, self.num_heads, self.hidden_size // self.num_heads),
        )
        attn_output = attn_output.view(-1, self.hidden_size)
        attn_output = self.output(attn_output)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection
        ffn_input = self.norm2(x)
        ffn_output = self.ffn2(F.relu(self.ffn1(ffn_input)))
        x = x + self.dropout(ffn_output)

        return x


def demonstrate_basic_rng_setup(args: argparse.Namespace) -> None:
    """Demonstrate basic RNG setup and initialization."""
    print("=" * 60)
    print("DEMONSTRATION 1: Basic RNG Setup")
    print("=" * 60)

    # Initialize RNG tracker
    tracker = initialize_cuda_rng_tracker(
        enable_cuda_graphs=args.enable_cuda_graphs,
        cache_capacity=args.cache_capacity,
        auto_cleanup=True,
        verbose=args.verbose,
    )

    print(f"Initialized RNG tracker with CUDA graphs: {args.enable_cuda_graphs}")

    # Mock parallel state initialization
    mock_state = MockParallelState(
        tp_size=args.tp_size, pp_size=args.pp_size, dp_size=args.dp_size
    )

    # Patch parallel state functions for demo
    import rosellm.rosetrainer.parallelism.parallel_state as ps

    ps._INITIALIZED = True
    ps._TENSOR_MODEL_PARALLEL_SIZE = mock_state.tp_size
    ps._PIPELINE_MODEL_PARALLEL_SIZE = mock_state.pp_size
    ps._DATA_PARALLEL_SIZE = mock_state.dp_size
    ps._CONTEXT_PARALLEL_SIZE = mock_state.cp_size
    ps._EXPERT_MODEL_PARALLEL_SIZE = mock_state.ep_size
    ps._TENSOR_MODEL_PARALLEL_RANK = mock_state.tp_rank
    ps._PIPELINE_MODEL_PARALLEL_RANK = mock_state.pp_rank
    ps._DATA_PARALLEL_RANK = mock_state.dp_rank
    ps._CONTEXT_PARALLEL_RANK = mock_state.cp_rank
    ps._EXPERT_MODEL_PARALLEL_RANK = mock_state.ep_rank

    # Initialize parallel RNG seeds
    seeds = model_parallel_cuda_manual_seed(
        seed=args.seed, enable_deterministic=args.deterministic, verbose=args.verbose
    )

    print(f"Initialized parallel RNG with base seed: {args.seed}")
    print("Computed seeds for different parallel dimensions:")
    for dimension, seed_value in seeds.items():
        print(f"  {dimension}: {seed_value}")

    # Show RNG tracker statistics
    stats = tracker.get_statistics()
    print(f"\nRNG Tracker Statistics:")
    print(f"  Number of states: {stats['num_states']}")
    print(f"  Current states: {stats['current_states']}")
    print(f"  State types: {stats['state_types']}")


def demonstrate_state_forking(args: argparse.Namespace) -> None:
    """Demonstrate RNG state forking and isolation."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 2: RNG State Forking and Isolation")
    print("=" * 60)

    tracker = get_cuda_rng_tracker()

    # Fork states for different use cases
    attention_fork = fork_parallel_rng_state(
        source_dimension="tensor_parallel",
        new_name="attention_dropout",
        target_dimensions=["tp"],
        offset=100,
    )

    ffn_fork = fork_parallel_rng_state(
        source_dimension="tensor_parallel",
        new_name="ffn_dropout",
        target_dimensions=["tp"],
        offset=200,
    )

    print(f"Created attention dropout fork: {attention_fork}")
    print(f"Created FFN dropout fork: {ffn_fork}")

    # Demonstrate isolated random number generation
    print("\nGenerating random numbers with different RNG states:")

    with parallel_rng_context("attention_dropout"):
        attention_randoms = [torch.rand(1).item() for _ in range(5)]
        print(f"  Attention dropout randoms: {attention_randoms}")

    with parallel_rng_context("ffn_dropout"):
        ffn_randoms = [torch.rand(1).item() for _ in range(5)]
        print(f"  FFN dropout randoms: {ffn_randoms}")

    # Show that the same states produce the same numbers
    print("\nRepeating with same states (should be identical):")

    with parallel_rng_context("attention_dropout"):
        tracker.set("attention_dropout")  # Reset state
        attention_randoms2 = [torch.rand(1).item() for _ in range(5)]
        print(f"  Attention dropout randoms: {attention_randoms2}")

    with parallel_rng_context("ffn_dropout"):
        tracker.set("ffn_dropout")  # Reset state
        ffn_randoms2 = [torch.rand(1).item() for _ in range(5)]
        print(f"  FFN dropout randoms: {ffn_randoms2}")

    # Verify reproducibility
    attention_match = np.allclose(attention_randoms, attention_randoms2, rtol=1e-10)
    ffn_match = np.allclose(ffn_randoms, ffn_randoms2, rtol=1e-10)

    print(f"\nReproducibility check:")
    print(f"  Attention state reproducible: {attention_match}")
    print(f"  FFN state reproducible: {ffn_match}")


def demonstrate_model_training(args: argparse.Namespace) -> None:
    """Demonstrate RNG usage in actual model training."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 3: RNG in Model Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create simple model
    model = SimpleTransformerLayer(hidden_size=256, num_heads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Generate sample data
    batch_size = 16
    seq_length = 32
    hidden_size = 256

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop with different RNG contexts
    print("\nTraining with different RNG states for dropout:")

    losses = []
    for step in range(args.train_steps):
        # Use different RNG states for different components

        # Generate input data with data parallel RNG
        with parallel_rng_context("data_parallel"):
            inputs = torch.randn(batch_size, seq_length, hidden_size, device=device)
            targets = torch.randn(batch_size, seq_length, hidden_size, device=device)

        optimizer.zero_grad()

        # Forward pass with tensor parallel RNG for model operations
        with parallel_rng_context("tensor_parallel"):
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 10 == 0 or step == args.train_steps - 1:
            print(f"  Step {step}: Loss = {loss.item():.6f}")

    print(f"\nTraining completed. Final loss: {losses[-1]:.6f}")


def demonstrate_checkpointing(args: argparse.Namespace) -> None:
    """Demonstrate RNG state checkpointing and restoration."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 4: RNG State Checkpointing")
    print("=" * 60)

    tracker = get_cuda_rng_tracker()

    # Modify some RNG states
    with parallel_rng_context("tensor_parallel"):
        pre_checkpoint_randoms = [torch.rand(1).item() for _ in range(3)]
        print(f"Random numbers before checkpoint: {pre_checkpoint_randoms}")

    # Create checkpoint
    print("Creating RNG checkpoint...")
    checkpoint = checkpoint_parallel_rng_state()

    # Generate more random numbers (this advances the state)
    with parallel_rng_context("tensor_parallel"):
        post_checkpoint_randoms = [torch.rand(1).item() for _ in range(3)]
        print(f"Random numbers after checkpoint: {post_checkpoint_randoms}")

    # Add some new states
    fork_parallel_rng_state(
        source_dimension="global", new_name="temporary_state", offset=999
    )

    print(f"States before restore: {tracker.get_states()}")

    # Restore from checkpoint
    print("Restoring from RNG checkpoint...")
    restore_parallel_rng_state(checkpoint)

    print(f"States after restore: {tracker.get_states()}")

    # Verify restoration by generating the same sequence
    with parallel_rng_context("tensor_parallel"):
        restored_randoms = [torch.rand(1).item() for _ in range(3)]
        print(f"Random numbers after restore: {restored_randoms}")

    # Check if we got the expected sequence
    expected_match = np.allclose(restored_randoms, post_checkpoint_randoms, rtol=1e-10)
    print(f"Restored sequence matches post-checkpoint: {expected_match}")


def demonstrate_performance_monitoring(args: argparse.Namespace) -> None:
    """Demonstrate performance monitoring and optimization."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 5: Performance Monitoring")
    print("=" * 60)

    tracker = get_cuda_rng_tracker()

    # Benchmark different RNG operations
    operations = []

    # Benchmark state creation
    start_time = time.time()
    for i in range(100):
        tracker.add(f"perf_state_{i}", seed=i, force=True)
    create_time = time.time() - start_time
    operations.append(("State creation (100 states)", create_time))

    # Benchmark state switching
    start_time = time.time()
    for i in range(100):
        tracker.set(f"perf_state_{i}")
    switch_time = time.time() - start_time
    operations.append(("State switching (100 switches)", switch_time))

    # Benchmark forking
    start_time = time.time()
    for i in range(50):
        tracker.fork(f"perf_state_{i}", f"fork_state_{i}", offset=i)
    fork_time = time.time() - start_time
    operations.append(("State forking (50 forks)", fork_time))

    # Benchmark checkpointing
    start_time = time.time()
    for i in range(10):
        _ = checkpoint_parallel_rng_state()  # Use result or mark as unused
    checkpoint_time = time.time() - start_time
    operations.append(("Checkpointing (10 checkpoints)", checkpoint_time))

    print("Performance benchmark results:")
    for operation, duration in operations:
        print(f"  {operation}: {duration:.4f} seconds")

    # Show tracker statistics
    stats = tracker.get_statistics()
    print(f"\nFinal RNG Tracker Statistics:")
    print(f"  Number of states: {stats['num_states']}")
    print(f"  Step counter: {stats['step_counter']}")
    print(f"  Access counter: {stats['access_counter']}")
    print(f"  Fork counter: {stats['fork_counter']}")
    print(f"  Cache capacity: {stats['cache_capacity']}")
    print(f"  CUDA graph mode: {stats['cuda_graph_mode']}")

    # Show overall RNG summary
    print(f"\nOverall RNG State Summary:")
    summary = get_rng_state_summary()
    print(f"  CUDA available: {summary['cuda_available']}")
    print(f"  Deterministic enabled: {summary['deterministic_enabled']}")
    print(f"  Torch initial seed: {summary['torch_initial_seed']}")

    if "cuda_device_count" in summary:
        print(f"  CUDA device count: {summary['cuda_device_count']}")
        print(f"  Current CUDA device: {summary['current_cuda_device']}")


def main():
    """Main example function."""
    parser = argparse.ArgumentParser(
        description="Multi-Parallel RNG State Management Example"
    )

    # RNG configuration
    parser.add_argument(
        "--seed", type=int, default=1234, help="Base seed for RNG initialization"
    )
    parser.add_argument(
        "--enable-cuda-graphs",
        action="store_true",
        help="Enable CUDA Graph compatibility",
    )
    parser.add_argument(
        "--cache-capacity", type=int, default=1000, help="RNG state cache capacity"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Enable deterministic algorithms",
    )

    # Parallel configuration
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size")

    # Training configuration
    parser.add_argument(
        "--train-steps", type=int, default=50, help="Number of training steps"
    )

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    print("RoseLLM Multi-Parallel RNG State Management Example")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Base seed: {args.seed}")
    print(f"  CUDA graphs: {args.enable_cuda_graphs}")
    print(f"  Cache capacity: {args.cache_capacity}")
    print(f"  Deterministic: {args.deterministic}")
    print(f"  Parallel config: TP={args.tp_size}, PP={args.pp_size}, DP={args.dp_size}")
    print(f"  Training steps: {args.train_steps}")
    print(f"  Verbose: {args.verbose}")

    try:
        # Run demonstrations
        demonstrate_basic_rng_setup(args)
        demonstrate_state_forking(args)
        demonstrate_model_training(args)
        demonstrate_checkpointing(args)
        demonstrate_performance_monitoring(args)

        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
