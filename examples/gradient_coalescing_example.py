#!/usr/bin/env python3
"""
Gradient Bucket Coalescing Example

This example demonstrates how to use gradient bucket coalescing to optimize
distributed training communication. The implementation follows Megatron-LM's
approach while integrating seamlessly with RoseLLM's training infrastructure.

Key Features Demonstrated:
    - Basic coalescing setup and configuration
    - Integration with distributed training
    - Performance comparison with/without coalescing
    - Adaptive sizing based on performance
    - Monitoring and metrics collection

To run this example:
    # Single GPU (simulated coalescing)
    python gradient_coalescing_example.py

    # Multi-GPU with coalescing
    torchrun --nproc_per_node=2 gradient_coalescing_example.py --enable-coalescing

    # Multi-GPU without coalescing (for comparison)
    torchrun --nproc_per_node=2 gradient_coalescing_example.py --no-coalescing
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# RoseLLM imports
from rosellm.rosetrainer.communication.coalescing import CoalescingConfig
from rosellm.rosetrainer.optimizer.coalesced_gradient_buffer import (
    CoalescedGradientBuffer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training example."""

    # Model configuration
    hidden_size: int = 1024
    num_layers: int = 12
    vocab_size: int = 50000
    seq_length: int = 512

    # Training configuration
    batch_size: int = 8
    num_steps: int = 100
    learning_rate: float = 1e-4

    # Coalescing configuration
    enable_coalescing: bool = True
    bucket_size_mb: float = 25.0
    max_coalesce_size_mb: float = 100.0
    adaptive_sizing: bool = True
    profile_communication: bool = True


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feedforward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class SimpleTransformer(nn.Module):
    """Simple transformer model for testing coalescing."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(config.seq_length, config.hidden_size)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config.hidden_size) for _ in range(config.num_layers)]
        )

        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        # Embeddings
        x = self.embedding(input_ids)
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(pos)

        # Transformer blocks
        x = x.transpose(0, 1)  # (seq_len, batch, hidden)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(0, 1)  # (batch, seq_len, hidden)

        # Output
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


def setup_distributed():
    """Initialize distributed training environment."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        return local_rank, world_size
    else:
        # Single process mode
        return 0, 1


def create_model_and_optimizer(config: TrainingConfig, device: torch.device):
    """Create model and optimizer."""
    model = SimpleTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    return model, optimizer


def train_step_with_coalescing(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_buffer: CoalescedGradientBuffer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Training step with coalesced gradient synchronization."""
    # Forward pass
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Synchronize gradients with coalescing
    grad_buffer.synchronize_gradients()

    # Optimizer step
    optimizer.step()

    return float(loss.item())


def train_step_standard(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Standard training step without coalescing."""
    # Forward pass
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Optimizer step (DDP handles gradient sync)
    optimizer.step()

    return float(loss.item())


def benchmark_training(config: TrainingConfig):
    """Benchmark training with and without coalescing."""
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running on rank {local_rank}/{world_size}, device: {device}")

    # Create model and optimizer
    model, optimizer = create_model_and_optimizer(config, device)

    # Setup for distributed training
    if world_size > 1:
        if config.enable_coalescing:
            # Use coalesced gradient buffer
            coalescing_config = CoalescingConfig(
                enable_coalescing=True,
                max_coalesce_size_mb=config.max_coalesce_size_mb,
                adaptive_sizing=config.adaptive_sizing,
                profile_communication=config.profile_communication,
            )

            grad_buffer = CoalescedGradientBuffer(
                params=list(model.parameters()),
                enable_coalescing=True,
                coalescing_config=coalescing_config,
                bucket_size_mb=config.bucket_size_mb,
                process_group=dist.group.WORLD,
            )

            logger.info(
                f"Created coalesced gradient buffer with "
                f"{len(grad_buffer.coalescing_groups)} coalescing groups"
            )
        else:
            # Use standard DDP
            model = DDP(model, device_ids=[local_rank])
            grad_buffer = None
            logger.info("Using standard DDP without coalescing")
    else:
        grad_buffer = None
        logger.info("Running in single GPU mode")

    # Training loop
    total_time = 0.0
    losses = []

    for step in range(config.num_steps):
        # Generate random data
        input_ids = torch.randint(
            0, config.vocab_size, (config.batch_size, config.seq_length), device=device
        )
        labels = torch.randint(
            0, config.vocab_size, (config.batch_size, config.seq_length), device=device
        )

        # Training step
        step_start = time.perf_counter()

        if config.enable_coalescing and grad_buffer is not None:
            loss = train_step_with_coalescing(
                model, optimizer, grad_buffer, input_ids, labels
            )
        else:
            loss = train_step_standard(
                model if world_size == 1 else model, optimizer, input_ids, labels
            )

        step_time = time.perf_counter() - step_start
        total_time += step_time
        losses.append(loss)

        # Log progress
        if (step + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            logger.info(
                f"Step {step + 1}/{config.num_steps}, "
                f"Loss: {avg_loss:.4f}, "
                f"Step time: {step_time*1000:.2f}ms"
            )

        # Adaptive optimization
        if (
            config.enable_coalescing
            and grad_buffer is not None
            and (step + 1) % 20 == 0
        ):
            grad_buffer.optimize_coalescing_groups()

    # Print statistics
    avg_step_time = total_time / config.num_steps
    logger.info(f"\n{'='*50}")
    logger.info(f"Training Summary (Rank {local_rank}):")
    logger.info(f"{'='*50}")
    logger.info(f"Total steps: {config.num_steps}")
    logger.info(f"Average step time: {avg_step_time*1000:.2f}ms")
    logger.info(f"Total training time: {total_time:.2f}s")

    if config.enable_coalescing and grad_buffer is not None:
        stats = grad_buffer.get_coalescing_stats()
        logger.info(f"\nCoalescing Statistics:")
        logger.info(f"  Total coalesced ops: {stats['total_coalesced_ops']}")
        logger.info(f"  Average ops per coalesce: {stats['avg_ops_per_coalesce']:.1f}")
        logger.info(f"  Peak coalesce size: {stats['peak_coalesce_size_mb']:.1f}MB")
        logger.info(f"  Number of coalescing groups: {stats['num_coalescing_groups']}")
        logger.info(f"  Number of fallbacks: {stats['num_fallbacks']}")

        # Log detailed metrics
        if grad_buffer.coalescing_manager:
            grad_buffer.coalescing_manager.log_metrics()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gradient Coalescing Example")

    # Model arguments
    parser.add_argument(
        "--hidden-size", type=int, default=1024, help="Hidden size of the model"
    )
    parser.add_argument(
        "--num-layers", type=int, default=12, help="Number of transformer layers"
    )
    parser.add_argument("--vocab-size", type=int, default=50000, help="Vocabulary size")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument(
        "--num-steps", type=int, default=100, help="Number of training steps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )

    # Coalescing arguments
    parser.add_argument(
        "--enable-coalescing",
        action="store_true",
        default=True,
        help="Enable gradient coalescing",
    )
    parser.add_argument(
        "--no-coalescing", action="store_true", help="Disable gradient coalescing"
    )
    parser.add_argument(
        "--bucket-size-mb", type=float, default=25.0, help="Bucket size in MB"
    )
    parser.add_argument(
        "--max-coalesce-size-mb",
        type=float,
        default=100.0,
        help="Maximum coalesce size in MB",
    )
    parser.add_argument(
        "--adaptive-sizing",
        action="store_true",
        default=True,
        help="Enable adaptive sizing",
    )
    parser.add_argument(
        "--profile-communication",
        action="store_true",
        default=True,
        help="Profile communication",
    )

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        enable_coalescing=not args.no_coalescing,
        bucket_size_mb=args.bucket_size_mb,
        max_coalesce_size_mb=args.max_coalesce_size_mb,
        adaptive_sizing=args.adaptive_sizing,
        profile_communication=args.profile_communication,
    )

    # Run benchmark
    benchmark_training(config)


if __name__ == "__main__":
    main()
