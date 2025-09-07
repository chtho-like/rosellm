#!/usr/bin/env python3
"""
Example demonstrating the DistributedOptimizer with gradient bucketing.

This example shows how to use the DistributedOptimizer for efficient
distributed training with memory optimization and communication overlap.

Usage:
    # Single GPU/CPU
    python distributed_optimizer_example.py

    # Multi-GPU (2 GPUs)
    torchrun --nproc_per_node=2 distributed_optimizer_example.py

    # Multi-process CPU simulation
    torchrun --nproc_per_node=4 distributed_optimizer_example.py --cpu
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from rosellm.rosetrainer.optimizer import (
    DistributedOptimizer,
    PartitioningStrategyFactory,
    estimate_memory_savings,
    get_optimizer_memory_usage,
    validate_bucket_configuration,
)


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # MLP
        x = self.norm2(x + self.mlp(x))
        return x


class DemoModel(nn.Module):
    """Demo model with multiple transformer blocks"""

    def __init__(self, vocab_size=10000, dim=512, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)  # (batch, seq) -> (seq, batch)

        for layer in self.layers:
            x = layer(x)

        x = x.transpose(0, 1)  # (seq, batch) -> (batch, seq)
        return self.output(x)


def setup_distributed():
    """Setup distributed training environment"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Initialize process group
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        return local_rank, world_size, device
    else:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Distributed Optimizer Example")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument(
        "--bucket-size-mb", type=float, default=25.0, help="Bucket size in MB"
    )
    parser.add_argument(
        "--grad-accumulation", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--partitioning",
        type=str,
        default="round_robin",
        choices=PartitioningStrategyFactory.list_strategies(),
        help="Parameter partitioning strategy",
    )
    parser.add_argument(
        "--enable-metrics", action="store_true", help="Enable performance metrics"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of training steps"
    )
    args = parser.parse_args()

    # Force CPU if requested
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Setup distributed
    local_rank, world_size, device = setup_distributed()

    # Print info on rank 0
    if local_rank == 0:
        print(f"Running on {world_size} process(es), device: {device}")
        print(
            f"Model config: vocab_size={args.vocab_size}, dim={args.dim}, "
            f"layers={args.num_layers}"
        )
        print(
            f"Training config: batch_size={args.batch_size}, "
            f"seq_length={args.seq_length}"
        )
        print(
            f"Optimizer config: bucket_size={args.bucket_size_mb}MB, "
            f"grad_accumulation={args.grad_accumulation}"
        )
        print(f"Partitioning strategy: {args.partitioning}")
        print(
            f"Performance metrics: {'enabled' if args.enable_metrics else 'disabled'}"
        )
        print("-" * 50)

    # Create model
    model = DemoModel(args.vocab_size, args.dim, args.num_layers).to(device)

    # Wrap in DDP if distributed
    if world_size > 1:
        model = DDP(model)

    # Create base optimizer
    base_optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Create distributed optimizer with gradient bucketing
    optimizer = DistributedOptimizer(
        base_optimizer,
        models=model,
        bucket_size_mb=args.bucket_size_mb,
        overlap_grad_reduce=True,
        partition_optimizer_states=(world_size > 1),
        partitioning_strategy=args.partitioning,
        gradient_accumulation_steps=args.grad_accumulation,
        clip_grad_norm=1.0,
        enable_metrics=args.enable_metrics,
    )

    # Validate bucket configuration
    if local_rank == 0:
        params = list(model.parameters())
        validation = validate_bucket_configuration(params, args.bucket_size_mb)
        print(f"Bucket configuration:")
        print(f"  - Number of buckets: {validation['num_buckets']}")
        print(f"  - Average bucket size: {validation['avg_bucket_size_mb']:.2f} MB")
        print(f"  - Efficiency: {validation['efficiency']:.2%}")

        # Estimate memory savings
        if world_size > 1:
            total_params = sum(p.numel() for p in params)
            savings = estimate_memory_savings(
                num_params=total_params,
                param_dtype=torch.float32,
                optimizer_state_size=2,  # AdamW has 2 state tensors
                world_size=world_size,
            )
            print(f"\nMemory savings from state partitioning:")
            print(f"  - Total memory: {savings['total_memory_mb']:.2f} MB")
            print(f"  - Partitioned memory: {savings['partitioned_memory_mb']:.2f} MB")
            print(
                f"  - Savings: {savings['savings_mb']:.2f} MB "
                f"({savings['savings_percent']:.1f}%)"
            )
        print("-" * 50)

    # Training loop
    criterion = nn.CrossEntropyLoss()

    for step in range(args.steps):
        start_time = time.time()

        # Generate random data
        input_ids = torch.randint(
            0, args.vocab_size, (args.batch_size, args.seq_length)
        ).to(device)
        labels = torch.randint(
            0, args.vocab_size, (args.batch_size, args.seq_length)
        ).to(device)

        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, args.vocab_size), labels.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Timing
        step_time = time.time() - start_time

        # Print progress on rank 0
        if local_rank == 0:
            stats = optimizer.get_statistics()
            output = (
                f"Step {step + 1}/{args.steps}: loss={loss.item():.4f}, "
                f"time={step_time:.3f}s, "
                f"grad_norm={stats.get('total_norm', 0):.3f}"
            )

            if args.enable_metrics and "performance" in stats:
                perf = stats["performance"]
                output += (
                    f", reduction_time={perf['timing']['gradient_reduction_ms']:.1f}ms"
                    f", efficiency={perf['efficiency']['communication']:.1%}"
                )
            else:
                output += f", bucket_reductions={stats.get('num_bucket_reductions', 0)}"

            print(output)

    # Final statistics
    if local_rank == 0:
        print("-" * 50)
        print("Training completed!")

        # Get optimizer memory usage
        memory_usage = get_optimizer_memory_usage(optimizer.base_optimizer)
        print(f"\nOptimizer memory usage:")
        print(f"  - Parameter memory: {memory_usage['param_memory_mb']:.2f} MB")
        print(f"  - State memory: {memory_usage['state_memory_mb']:.2f} MB")
        print(f"  - Total: {memory_usage['total_memory_mb']:.2f} MB")

        # Get final statistics
        final_stats = optimizer.get_statistics()
        if "gradient_buffer" in final_stats:
            buffer_info = final_stats["gradient_buffer"]
            print(f"\nGradient buffer statistics:")
            print(f"  - Number of buckets: {buffer_info['num_buckets']}")
            print(
                f"  - Total buffer size: {buffer_info['total_buffer_size_mb']:.2f} MB"
            )
            print(f"  - Parameters per bucket: {buffer_info['num_params_per_bucket']}")

        if args.enable_metrics and "performance_avg" in final_stats:
            avg_metrics = final_stats["performance_avg"]
            print(f"\nAverage performance metrics:")
            for key, value in avg_metrics.items():
                if isinstance(value, float):
                    if "time" in key or "ms" in key:
                        print(f"  - {key}: {value:.2f}")
                    else:
                        print(f"  - {key}: {value:.4f}")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
