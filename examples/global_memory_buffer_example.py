#!/usr/bin/env python3
"""
End-to-End Example: Global Memory Buffer for Dynamic Allocation Prevention

This example demonstrates how to use the GlobalMemoryBuffer system to prevent
dynamic memory allocations during training, reducing memory fragmentation and
improving performance.

Key features demonstrated:
1. Basic buffer allocation and release
2. Integration with tensor parallel operations
3. Memory pooling for different operation types
4. Performance comparison with regular allocation
5. Memory leak detection and monitoring

Usage:
    # Single GPU or CPU
    python global_memory_buffer_example.py

    # Multi-GPU with tensor parallelism
    torchrun --nproc_per_node=2 global_memory_buffer_example.py --distributed

    # With custom configuration
    python global_memory_buffer_example.py --activation-buffer-mb 2048 --enable-stats
"""

import argparse
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from rosellm.rosetrainer.memory.global_memory_buffer import (
    BufferConfig,
    BufferContext,
    BufferType,
    GlobalMemoryBuffer,
    allocate_tensor,
    get_global_memory_buffer,
    initialize_global_memory_buffer,
    release_tensor,
)
from rosellm.rosetrainer.parallelism import parallel_state
from rosellm.rosetrainer.parallelism.model_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)

# Type hints imported as needed


def print_rank0(message: str) -> None:
    """Print only on rank 0"""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message)


class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer for demonstration"""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        tp_size: int = 1,
        use_global_buffer: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_global_buffer = use_global_buffer

        if tp_size > 1:
            # Use tensor parallel layers
            tp_group = parallel_state.get_tensor_model_parallel_group()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()

            self.fc1 = ColumnParallelLinear(
                hidden_size,
                ffn_hidden_size,
                bias=True,
                tp_group=tp_group,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
            self.fc2 = RowParallelLinear(
                ffn_hidden_size,
                hidden_size,
                bias=True,
                tp_group=tp_group,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
        else:
            # Regular layers
            self.fc1 = nn.Linear(hidden_size, ffn_hidden_size)
            self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)

        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional global buffer usage"""
        residual = x

        # Layer norm
        x = self.layernorm(x)

        if self.use_global_buffer:
            # Use global buffer for intermediate activations
            with BufferContext(
                shape=x.shape,
                dtype=x.dtype,
                device=x.device,
                buffer_type=BufferType.ACTIVATION,
            ) as hidden:
                # First linear layer
                hidden.copy_(self.fc1(x))

                # Activation
                hidden = self.activation(hidden)

                # Second linear layer
                output = self.fc2(hidden)
        else:
            # Regular computation without global buffers
            hidden = self.fc1(x)
            hidden = self.activation(hidden)
            output = self.fc2(hidden)

        # Residual connection
        return output + residual


def demonstrate_basic_usage():
    """Demonstrate basic buffer allocation and release"""
    print_rank0("\n=== Basic Buffer Usage ===")

    # Initialize with custom configuration
    config = BufferConfig(
        activation_buffer_size=100,  # 100MB for activations
        gradient_buffer_size=50,  # 50MB for gradients
        communication_buffer_size=25,  # 25MB for communication
        track_allocations=True,  # Enable tracking for debugging
    )
    initialize_global_memory_buffer(config)

    # Get the global buffer instance
    buffer = get_global_memory_buffer()

    # Allocate tensors from different pools
    print_rank0("Allocating tensors from different buffer pools...")

    # 1. Activation buffer
    activation = allocate_tensor(
        shape=(1024, 1024),
        dtype=torch.float32,
        buffer_type=BufferType.ACTIVATION,
        caller_info="forward_pass",
    )
    print_rank0(
        f"  Activation tensor: shape={activation.shape}, dtype={activation.dtype}"
    )

    # 2. Gradient buffer
    gradient = allocate_tensor(
        shape=(512, 512),
        dtype=torch.float32,
        buffer_type=BufferType.GRADIENT,
        caller_info="backward_pass",
    )
    print_rank0(f"  Gradient tensor: shape={gradient.shape}, dtype={gradient.dtype}")

    # 3. Communication buffer (for all-reduce, etc.)
    comm_buffer = allocate_tensor(
        shape=(256, 256),
        dtype=torch.float32,
        buffer_type=BufferType.COMMUNICATION,
        caller_info="all_reduce_op",
    )
    print_rank0(
        f"  Communication tensor: shape={comm_buffer.shape}, dtype={comm_buffer.dtype}"
    )

    # Use the tensors
    activation.normal_()
    gradient.zero_()
    comm_buffer.fill_(1.0)

    # Release tensors back to pools
    print_rank0("\nReleasing tensors back to pools...")
    release_tensor(activation)
    release_tensor(gradient)
    release_tensor(comm_buffer)

    # Check statistics
    stats = buffer.get_stats()
    print_rank0("\nBuffer statistics after release:")
    for pool_name, pool_stats in stats.items():
        print_rank0(f"  {pool_name}:")
        print_rank0(f"    Total size: {pool_stats['total_size_mb']:.1f} MB")
        print_rank0(f"    Current usage: {pool_stats['current_usage_mb']:.1f} MB")
        print_rank0(f"    Utilization: {pool_stats['utilization']:.1%}")


def demonstrate_context_manager():
    """Demonstrate using context manager for automatic release"""
    print_rank0("\n=== Context Manager Usage ===")

    # Context manager automatically allocates and releases
    print_rank0("Using context manager for automatic buffer management...")

    with BufferContext(
        shape=(512, 512, 16),
        dtype=torch.float16,
        buffer_type=BufferType.ACTIVATION,
    ) as tensor:
        print_rank0(f"  Allocated tensor: shape={tensor.shape}, dtype={tensor.dtype}")

        # Use the tensor
        tensor.normal_()
        result = tensor.mean()
        print_rank0(f"  Computed mean: {result.item():.4f}")

    # Tensor is automatically released here
    print_rank0("  Tensor automatically released after context exit")

    # Nested contexts for multiple buffers
    print_rank0("\nUsing nested contexts...")
    with BufferContext(
        (256, 256), torch.float32, buffer_type=BufferType.ACTIVATION
    ) as act:
        with BufferContext(
            (256, 256), torch.float32, buffer_type=BufferType.GRADIENT
        ) as grad:
            # Simulate forward and backward
            act.normal_()
            grad.copy_(act * 0.1)  # Fake gradient
            print_rank0(f"  Activation norm: {act.norm().item():.4f}")
            print_rank0(f"  Gradient norm: {grad.norm().item():.4f}")


def demonstrate_performance_comparison():
    """Compare performance with and without global buffers"""
    print_rank0("\n=== Performance Comparison ===")

    num_iterations = 100
    tensor_size = (1024, 1024)

    # Warm up
    for _ in range(10):
        t = allocate_tensor(
            tensor_size, torch.float32, buffer_type=BufferType.TEMPORARY
        )
        release_tensor(t)

    # Benchmark with global buffer
    print_rank0(f"Running {num_iterations} allocations with global buffer...")
    start_time = time.time()
    for _ in range(num_iterations):
        tensor = allocate_tensor(
            tensor_size,
            torch.float32,
            buffer_type=BufferType.TEMPORARY,
        )
        tensor.fill_(1.0)  # Some computation
        release_tensor(tensor)
    buffer_time = time.time() - start_time

    # Benchmark with regular allocation
    print_rank0(f"Running {num_iterations} regular allocations...")
    start_time = time.time()
    for _ in range(num_iterations):
        tensor = torch.zeros(tensor_size, dtype=torch.float32)
        tensor.fill_(1.0)  # Some computation
        # Regular tensors are freed by garbage collector
    regular_time = time.time() - start_time

    print_rank0(f"\nResults:")
    print_rank0(
        f"  Global buffer time: {buffer_time:.3f}s ({buffer_time/num_iterations*1000:.2f}ms per alloc)"
    )
    print_rank0(
        f"  Regular alloc time: {regular_time:.3f}s ({regular_time/num_iterations*1000:.2f}ms per alloc)"
    )
    print_rank0(f"  Speedup: {regular_time/buffer_time:.2f}x")

    # Note: Global buffer may be slightly slower per allocation but reduces fragmentation
    print_rank0(
        "\nNote: Global buffer reduces memory fragmentation and improves long-term stability"
    )


def demonstrate_memory_patterns():
    """Demonstrate different memory usage patterns"""
    print_rank0("\n=== Memory Usage Patterns ===")

    get_global_memory_buffer()  # Initialize if needed

    # Pattern 1: Gradient accumulation
    print_rank0("\n1. Gradient Accumulation Pattern:")
    accumulated_grad = None
    num_micro_batches = 4

    for i in range(num_micro_batches):
        with BufferContext(
            (512, 512), torch.float32, buffer_type=BufferType.GRADIENT
        ) as grad:
            # Simulate gradient computation
            grad.normal_(std=0.01)

            if accumulated_grad is None:
                accumulated_grad = grad.clone()
            else:
                accumulated_grad.add_(grad)

        print_rank0(
            f"  Micro-batch {i+1}: accumulated grad norm = {accumulated_grad.norm().item():.4f}"
        )

    # Pattern 2: Activation checkpointing
    print_rank0("\n2. Activation Checkpointing Pattern:")
    checkpointed_acts = []
    num_layers = 6

    for layer in range(num_layers):
        with BufferContext(
            (256, 256), torch.float32, buffer_type=BufferType.ACTIVATION
        ) as act:
            # Simulate layer computation
            act.normal_()

            # Checkpoint every other layer
            if layer % 2 == 0:
                checkpointed_acts.append(act.clone())
                print_rank0(
                    f"  Layer {layer}: checkpointed (norm={act.norm().item():.4f})"
                )
            else:
                print_rank0(f"  Layer {layer}: not checkpointed")

    # Pattern 3: Pipeline parallel communication
    print_rank0("\n3. Pipeline Communication Pattern:")
    for stage in range(3):
        with BufferContext(
            (128, 128), torch.float32, buffer_type=BufferType.COMMUNICATION
        ) as comm:
            # Simulate pipeline communication
            comm.fill_(stage + 1.0)
            print_rank0(
                f"  Stage {stage}: sending tensor with value {comm[0, 0].item()}"
            )


def demonstrate_distributed_usage(args):
    """Demonstrate usage with distributed training"""
    print_rank0("\n=== Distributed Training with Global Buffers ===")

    # Initialize distributed
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Initialize parallel state with buffer configuration
    buffer_config = BufferConfig(
        activation_buffer_size=args.activation_buffer_mb,
        gradient_buffer_size=args.gradient_buffer_mb,
        communication_buffer_size=args.communication_buffer_mb,
        enable_pooling=True,
        track_allocations=args.enable_stats,
    )

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=min(world_size, args.tp_size),
        buffer_config=buffer_config,
    )

    print_rank0(f"Initialized distributed training:")
    print_rank0(f"  World size: {world_size}")
    print_rank0(f"  Tensor parallel size: {min(world_size, args.tp_size)}")
    print_rank0(f"  Device: {device}")

    # Create model with tensor parallelism
    model = SimpleTransformerLayer(
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.hidden_size * 4,
        tp_size=min(world_size, args.tp_size),
        use_global_buffer=True,
    ).to(device)

    # Run forward pass
    batch_size = 32
    seq_len = 128

    print_rank0(f"\nRunning forward pass with global buffers...")
    input_tensor = torch.randn(batch_size, seq_len, args.hidden_size, device=device)

    # Forward with timing
    start_time = time.time()
    output = model(input_tensor)

    # Synchronize for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    forward_time = time.time() - start_time

    print_rank0(f"  Forward pass completed in {forward_time*1000:.2f}ms")
    print_rank0(f"  Output shape: {output.shape}")
    print_rank0(f"  Output norm: {output.norm().item():.4f}")

    # Check buffer statistics
    if args.enable_stats:
        buffer = parallel_state.get_global_memory_buffer()
        if buffer and rank == 0:
            stats = buffer.get_stats()
            print("\nGlobal buffer statistics:")
            for pool_name, pool_stats in stats.items():
                if (
                    pool_stats["current_usage_mb"] > 0
                    or pool_stats["total_allocations"] > 0
                ):
                    print(f"  {pool_name}:")
                    print(f"    Peak usage: {pool_stats['peak_usage_mb']:.1f} MB")
                    print(f"    Total allocations: {pool_stats['total_allocations']}")
                    print(f"    Fragmentation: {pool_stats['fragmentation']} blocks")

    # Clean up
    dist.destroy_process_group()


def demonstrate_memory_monitoring():
    """Demonstrate memory leak detection and monitoring"""
    print_rank0("\n=== Memory Monitoring and Leak Detection ===")

    # Initialize with leak detection enabled
    config = BufferConfig(
        track_allocations=True,
        check_memory_leaks=True,
        warn_on_reallocation=True,
    )
    buffer = GlobalMemoryBuffer(config)

    # Intentionally create a "leak" (unreleased allocation)
    print_rank0("Creating intentional memory leak for demonstration...")
    leaked_tensor = allocate_tensor(
        (256, 256),
        torch.float32,
        buffer_type=BufferType.TEMPORARY,
        caller_info="intentional_leak_demo",
    )

    # Allocate and properly release another tensor
    with BufferContext((128, 128), torch.float32) as proper_tensor:
        proper_tensor.fill_(1.0)

    # Check for leaks
    print_rank0("\nChecking for memory leaks...")
    warnings = buffer.check_memory_leaks()

    if warnings:
        print_rank0("Detected issues:")
        for warning in warnings:
            print_rank0(f"  - {warning}")
    else:
        print_rank0("  No issues detected")

    # Clean up the leak
    print_rank0("\nCleaning up leaked allocation...")
    release_tensor(leaked_tensor)

    # Check again
    warnings = buffer.check_memory_leaks()
    print_rank0("After cleanup:")
    if warnings:
        for warning in warnings:
            print_rank0(f"  - {warning}")
    else:
        print_rank0("  All allocations properly released")


def main():
    parser = argparse.ArgumentParser(description="Global Memory Buffer Example")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run distributed example with tensor parallelism",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=2,
        help="Tensor parallel size (default: 2)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=768,
        help="Hidden size for transformer layer (default: 768)",
    )
    parser.add_argument(
        "--activation-buffer-mb",
        type=int,
        default=512,
        help="Activation buffer size in MB (default: 512)",
    )
    parser.add_argument(
        "--gradient-buffer-mb",
        type=int,
        default=256,
        help="Gradient buffer size in MB (default: 256)",
    )
    parser.add_argument(
        "--communication-buffer-mb",
        type=int,
        default=128,
        help="Communication buffer size in MB (default: 128)",
    )
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        help="Enable detailed statistics tracking",
    )
    parser.add_argument(
        "--skip-basic",
        action="store_true",
        help="Skip basic demonstrations",
    )

    args = parser.parse_args()

    if args.distributed:
        # Run distributed example
        demonstrate_distributed_usage(args)
    else:
        # Run single-process examples
        if not args.skip_basic:
            demonstrate_basic_usage()
            demonstrate_context_manager()
            demonstrate_performance_comparison()
            demonstrate_memory_patterns()
            demonstrate_memory_monitoring()
        else:
            # Just run performance comparison
            config = BufferConfig(
                activation_buffer_size=args.activation_buffer_mb,
                gradient_buffer_size=args.gradient_buffer_mb,
                communication_buffer_size=args.communication_buffer_mb,
            )
            initialize_global_memory_buffer(config)
            demonstrate_performance_comparison()

    print_rank0("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
