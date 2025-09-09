#!/usr/bin/env python3
"""
End-to-end example of gradient bucketing with communication overlap.

This example demonstrates how to use RoseLLM's gradient bucketing feature
to optimize distributed training with efficient gradient communication.

Features demonstrated:
- Multiple bucketing strategies
- Communication/computation overlap
- Integration with distributed training
- Performance monitoring

To run this example:
    # Single GPU (no bucketing needed, but works)
    python gradient_bucketing_example.py

    # Multi-GPU with bucketing
    torchrun --nproc_per_node=2 gradient_bucketing_example.py

    # Multi-node training
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        gradient_bucketing_example.py
"""

import argparse
import logging
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# RoseLLM imports
from rosellm.rosetrainer.gradient import (
    BucketingStrategy,
    GradientBucketConfig,
    GradientBucketManager,
    create_gradient_buckets,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class DemoTransformer(nn.Module):
    """Demo transformer model for testing gradient bucketing."""

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.positional_embedding(positions)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        x = self.norm(x)
        return self.output(x)


def setup_distributed() -> Tuple[int, int]:
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Initialize process group
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}")
        return rank, world_size
    else:
        logger.info("Running in single-process mode")
        return 0, 1


def create_model_and_optimizer(
    args: argparse.Namespace, device: torch.device
) -> Tuple[nn.Module, optim.Optimizer]:
    """Create model and optimizer."""
    model = DemoTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
    ).to(device)

    # Wrap in DDP if distributed
    if dist.is_initialized() and dist.get_world_size() > 1:
        model = DDP(model)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    return model, optimizer


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: torch.Tensor,
    labels: torch.Tensor,
    bucket_manager: Optional[GradientBucketManager] = None,
    use_bucketing: bool = True,
) -> float:
    """Single training step with optional gradient bucketing."""
    optimizer.zero_grad()

    # Forward pass
    output = model(batch)
    loss = nn.functional.cross_entropy(
        output.view(-1, output.size(-1)), labels.view(-1)
    )

    # Backward pass
    loss.backward()

    # Gradient synchronization with bucketing
    if use_bucketing and bucket_manager is not None:
        bucket_manager.synchronize_gradients()

    # Optimizer step
    optimizer.step()

    # Reset bucket manager for next iteration
    if use_bucketing and bucket_manager is not None:
        bucket_manager.reset()

    return float(loss.item())


def benchmark_bucketing_strategies(
    model: nn.Module, args: argparse.Namespace, device: torch.device
) -> Dict[str, float]:
    """Benchmark different bucketing strategies."""
    results = {}

    strategies = [
        BucketingStrategy.SIZE_BASED,
        BucketingStrategy.TYPE_BASED,
        BucketingStrategy.LAYER_BASED,
        BucketingStrategy.HYBRID,
    ]

    batch = torch.randint(
        0, args.vocab_size, (args.batch_size, args.max_seq_len), device=device
    )
    labels = torch.randint(
        0, args.vocab_size, (args.batch_size, args.max_seq_len), device=device
    )

    for strategy in strategies:
        logger.info(f"Testing strategy: {strategy}")

        # Create bucket manager with strategy
        config = GradientBucketConfig(
            bucket_size_mb=args.bucket_size_mb,
            bucketing_strategy=strategy,
            overlap_communication=args.overlap_communication,
        )

        # Get underlying model if using DDP
        if isinstance(model, DDP):
            base_model = model.module
        else:
            base_model = model
        bucket_manager = create_gradient_buckets(base_model, config)

        # Warmup
        for _ in range(3):
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            train_step(model, optimizer, batch, labels, bucket_manager)

        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        num_steps = 10
        for _ in range(num_steps):
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            train_step(model, optimizer, batch, labels, bucket_manager)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start_time

        avg_time = elapsed / num_steps
        results[strategy.value] = avg_time

        # Log statistics
        stats = bucket_manager.get_statistics()
        logger.info(f"  Strategy: {strategy}")
        logger.info(f"  Buckets: {stats['num_buckets']}")
        logger.info(f"  Avg bucket size: {stats['avg_bucket_size']:.0f} elements")
        logger.info(f"  Time per step: {avg_time:.4f}s")

    return results


def main():
    """Main training loop with gradient bucketing."""
    parser = argparse.ArgumentParser(description="Gradient Bucketing Example")

    # Model arguments
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument(
        "--hidden-size", type=int, default=512, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=128, help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument(
        "--num-steps", type=int, default=100, help="Number of training steps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")

    # Bucketing arguments
    parser.add_argument(
        "--bucket-size-mb", type=float, default=50, help="Target bucket size in MB"
    )
    parser.add_argument(
        "--bucketing-strategy",
        type=str,
        default=BucketingStrategy.SIZE_BASED,
        choices=[s.value for s in BucketingStrategy],
        help="Bucketing strategy to use",
    )
    parser.add_argument(
        "--overlap-communication",
        action="store_true",
        help="Enable communication/computation overlap",
    )
    parser.add_argument(
        "--benchmark-strategies",
        action="store_true",
        help="Benchmark all bucketing strategies",
    )
    parser.add_argument(
        "--no-bucketing",
        action="store_true",
        help="Disable gradient bucketing (baseline)",
    )

    args = parser.parse_args()

    # Setup distributed training
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")
    logger.info(f"World size: {world_size}")

    # Create model and optimizer
    model, optimizer = create_model_and_optimizer(args, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Benchmark different strategies if requested
    if args.benchmark_strategies:
        logger.info("=" * 50)
        logger.info("Benchmarking bucketing strategies...")
        logger.info("=" * 50)

        results = benchmark_bucketing_strategies(model, args, device)

        logger.info("\nBenchmark Results:")
        logger.info("-" * 30)
        for strategy, time_per_step in results.items():
            logger.info(f"{strategy}: {time_per_step:.4f}s per step")

        best_strategy = min(results, key=lambda k: results[k])
        logger.info(f"\nBest strategy: {best_strategy}")
        return

    # Create gradient bucket manager
    bucket_manager = None
    if not args.no_bucketing and world_size > 1:
        config = GradientBucketConfig(
            bucket_size_mb=args.bucket_size_mb,
            bucketing_strategy=args.bucketing_strategy,
            overlap_communication=args.overlap_communication,
            use_distributed_optimizer=False,  # Enable for ZeRO-style opt
            dtype_bucketing=True,
        )

        # Get underlying model if using DDP
        if isinstance(model, DDP):
            base_model = model.module
        else:
            base_model = model
        bucket_manager = create_gradient_buckets(base_model, config)

        # Log bucketing statistics
        stats = bucket_manager.get_statistics()
        logger.info(f"\nGradient Bucketing Configuration:")
        logger.info(f"  Strategy: {args.bucketing_strategy}")
        logger.info(f"  Number of buckets: {stats['num_buckets']}")
        logger.info(f"  Total parameters: {stats['total_parameters']}")
        logger.info(f"  Average bucket size: {stats['avg_bucket_size']:.0f} elements")
        logger.info(f"  Overlap enabled: {stats['overlap_enabled']}")

        if "dtype_distribution" in stats:
            logger.info(f"  Dtype distribution:")
            for dtype, count in stats["dtype_distribution"].items():
                logger.info(f"    {dtype}: {count:,} elements")

    # Training loop
    logger.info("\nStarting training...")
    logger.info("=" * 50)

    loss_history = []
    step_times = []

    for step in range(args.num_steps):
        # Generate random batch (in real training, use DataLoader)
        batch = torch.randint(
            0, args.vocab_size, (args.batch_size, args.max_seq_len), device=device
        )
        labels = torch.randint(
            0, args.vocab_size, (args.batch_size, args.max_seq_len), device=device
        )

        # Time the training step
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        # Training step with bucketing
        loss = train_step(
            model,
            optimizer,
            batch,
            labels,
            bucket_manager,
            use_bucketing=(not args.no_bucketing),
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        step_time = time.perf_counter() - start_time

        loss_history.append(loss)
        step_times.append(step_time)

        # Log progress
        if (step + 1) % 10 == 0:
            avg_loss = sum(loss_history[-10:]) / min(10, len(loss_history))
            avg_time = sum(step_times[-10:]) / min(10, len(step_times))

            logger.info(
                f"Step {step + 1}/{args.num_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {avg_time:.4f}s/step | "
                f"Throughput: {args.batch_size / avg_time:.1f} samples/s"
            )

    # Final statistics
    logger.info("\n" + "=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Average loss: {sum(loss_history) / len(loss_history):.4f}")
    logger.info(f"Average time per step: {sum(step_times) / len(step_times):.4f}s")

    if bucket_manager is not None:
        final_stats = bucket_manager.get_statistics()
        logger.info(f"\nFinal Bucketing Statistics:")
        logger.info(f"  Total buckets used: {final_stats['num_buckets']}")
        logger.info(f"  Parameters bucketed: {final_stats['total_parameters']}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
