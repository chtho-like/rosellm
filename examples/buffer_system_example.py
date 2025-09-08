#!/usr/bin/env python3
"""
Parameter and Gradient Buffer System Integration Example

This example demonstrates how to use the ParamAndGradBuffer system for:
1. Efficient memory management with contiguous buffers
2. Gradient bucketing for optimized distributed training
3. Integration with mixed precision training
4. Multi-GPU distributed training with overlapped communication

Run with:
    # Single GPU
    python examples/buffer_system_example.py

    # Multi-GPU (2 GPUs)
    torchrun --nproc_per_node=2 examples/buffer_system_example.py

    # CPU simulation with 4 processes
    CUDA_VISIBLE_DEVICES="" torchrun --nproc_per_node=4 \
        examples/buffer_system_example.py
"""

import argparse
import logging
import os
import time
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from rosellm.rosetrainer.memory import BucketConfig, BufferManager
from rosellm.rosetrainer.parallelism import (
    get_data_parallel_group,
    initialize_model_parallel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""

    def __init__(
        self, hidden_size: int = 768, num_heads: int = 12, dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class SimpleTransformer(nn.Module):
    """Simple transformer model for demonstration."""

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        # Embeddings
        x = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)

        # Transformer blocks
        x = x.transpose(0, 1)  # (seq_len, batch, hidden)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1)  # (batch, seq_len, hidden)

        # Output
        x = self.ln_final(x)
        logits = self.head(x)

        return logits


def setup_distributed():
    """Initialize distributed training environment."""
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

        # Initialize model parallel (using data parallel only for this example)
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=world_size,
        )

        return local_rank, world_size, device
    else:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def benchmark_buffer_system(
    model: nn.Module,
    buffer_manager: BufferManager,
    device: torch.device,
    batch_size: int = 8,
    seq_len: int = 128,
    num_iterations: int = 10,
) -> Dict[str, float]:
    """Benchmark the buffer system performance."""

    logger.info("Starting buffer system benchmark...")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Timing statistics
    forward_times = []
    backward_times = []
    comm_times = []
    update_times = []

    for iteration in range(num_iterations):
        # Generate random input
        input_ids = torch.randint(0, 50000, (batch_size, seq_len), device=device)
        labels = torch.randint(0, 50000, (batch_size, seq_len), device=device)

        # Forward pass
        start_time = time.time()

        # Forward pass
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        forward_time = time.time() - start_time
        forward_times.append(forward_time)

        # Backward pass
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        backward_times.append(backward_time)

        # All-reduce gradients using buffer system
        start_time = time.time()
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Use async all-reduce for overlapping
            buffer_manager.all_reduce_gradients(async_op=True)

            # Simulate computation overlap
            time.sleep(0.001)  # Simulate other computation

            # Wait for communication to complete
            buffer_manager.wait_for_all_reduce()
        comm_time = time.time() - start_time
        comm_times.append(comm_time)

        # Optimizer step
        start_time = time.time()

        # Note: In production, you would handle mixed precision scaling here
        # For simplicity, we skip the unscaling step in this example

        # Gradient clipping using buffer system
        total_norm = buffer_manager.clip_gradients(max_norm=1.0)

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        update_time = time.time() - start_time
        update_times.append(update_time)

        # Log iteration stats
        if iteration % 2 == 0:
            logger.info(
                f"Iteration {iteration}: "
                f"loss={loss.item():.4f}, "
                f"grad_norm={total_norm:.4f}, "
                f"forward={forward_time:.4f}s, "
                f"backward={backward_time:.4f}s, "
                f"comm={comm_time:.4f}s, "
                f"update={update_time:.4f}s"
            )

    # Calculate statistics
    stats = {
        "avg_forward_time": sum(forward_times) / len(forward_times),
        "avg_backward_time": sum(backward_times) / len(backward_times),
        "avg_comm_time": sum(comm_times) / len(comm_times),
        "avg_update_time": sum(update_times) / len(update_times),
        "total_time": sum(forward_times + backward_times + comm_times + update_times),
    }

    # Memory usage
    memory_stats = buffer_manager.get_memory_usage()
    # Safely extract numeric values from memory stats
    for key in ["param_buffer_mb", "grad_buffer_mb", "bucket_memory_mb", "total_mb"]:
        value = memory_stats.get(key, 0.0)
        # Ensure it's a numeric type
        if isinstance(value, (int, float)):
            stats[key] = value
        else:
            stats[key] = 0.0

    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Buffer System Example")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument(
        "--num-iterations", type=int, default=10, help="Number of iterations"
    )
    parser.add_argument(
        "--bucket-size-mb", type=float, default=25.0, help="Bucket size in MB"
    )
    parser.add_argument(
        "--use-mixed-precision", action="store_true", help="Use mixed precision"
    )
    parser.add_argument(
        "--overlap-comm", action="store_true", help="Overlap communication"
    )
    args = parser.parse_args()

    # Setup distributed training
    local_rank, world_size, device = setup_distributed()

    logger.info(f"Rank {local_rank}/{world_size} using device: {device}")

    # Create model
    model = SimpleTransformer(
        vocab_size=50000,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=12,
        max_seq_len=512,
    )
    model.to(device)

    # Create buffer configuration
    bucket_config = BucketConfig(
        bucket_size_mb=args.bucket_size_mb,
        alignment=128,
        overlap_comm=args.overlap_comm,
    )

    # Create buffer manager
    dp_group = get_data_parallel_group() if dist.is_initialized() else None
    buffer_manager = BufferManager(
        model=model,
        data_parallel_group=dp_group,
        bucket_config=bucket_config,
        create_per_dtype_buffers=args.use_mixed_precision,
        overlap_comm=args.overlap_comm,
    )

    # Log buffer statistics
    logger.info(f"Buffer Manager Statistics:")
    logger.info(f"  Total parameters: {buffer_manager.total_params:,}")
    logger.info(f"  Total memory: {buffer_manager.total_memory_mb:.2f} MB")
    logger.info(f"  Number of buffers: {len(buffer_manager.buffers)}")

    for name, buffer in buffer_manager.buffers.items():
        logger.info(f"  Buffer '{name}':")
        logger.info(f"    Parameters: {buffer.stats['num_params']:,}")
        logger.info(f"    Elements: {buffer.numel:,}")
        logger.info(f"    Buckets: {buffer.stats['num_buckets']}")
        logger.info(f"    Memory: {buffer.stats['buffer_memory_mb']:.2f} MB")

    # Run benchmark
    stats = benchmark_buffer_system(
        model=model,
        buffer_manager=buffer_manager,
        device=device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_iterations=args.num_iterations,
    )

    # Print results
    if local_rank == 0:
        logger.info("\n" + "=" * 60)
        logger.info("Benchmark Results:")
        logger.info("=" * 60)
        logger.info(f"Average forward time: {stats['avg_forward_time']:.4f}s")
        logger.info(f"Average backward time: {stats['avg_backward_time']:.4f}s")
        logger.info(f"Average communication time: {stats['avg_comm_time']:.4f}s")
        logger.info(f"Average update time: {stats['avg_update_time']:.4f}s")
        logger.info(f"Total time: {stats['total_time']:.4f}s")
        logger.info(
            f"Throughput: {args.num_iterations / stats['total_time']:.2f} iter/s"
        )
        logger.info("\nMemory Usage:")
        logger.info(f"  Parameter buffers: {stats.get('param_buffer_mb', 0):.2f} MB")
        logger.info(f"  Gradient buffers: {stats.get('grad_buffer_mb', 0):.2f} MB")
        logger.info(f"  Bucket memory: {stats.get('bucket_memory_mb', 0):.2f} MB")
        logger.info(f"  Total memory: {stats.get('total_mb', 0):.2f} MB")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
