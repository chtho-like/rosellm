#!/usr/bin/env python3
"""
Contiguous Parameter-Gradient Buffer Example

This example demonstrates how to use the contiguous parameter-gradient buffer system
in RoseLLM for efficient distributed training. The contiguous buffer system provides:
- Unified memory management for parameters and gradients
- Automatic gradient accumulation with hooks
- Efficient bucketing for distributed communication
- Integration with async gradient all-reduce

Usage:
    # Single GPU/CPU (for testing)
    python contiguous_buffer_example.py

    # Multi-GPU distributed training
    torchrun --nproc_per_node=4 contiguous_buffer_example.py --distributed

    # With different configurations
    python contiguous_buffer_example.py --bucket-size 50 --use-hooks
    python contiguous_buffer_example.py --distributed --async-allreduce
"""

import argparse
import logging
import os
import time
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from rosellm.rosetrainer.memory import ContiguousBucketConfig, ContiguousParamGradBuffer
from rosellm.rosetrainer.parallelism import DataParallelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """A simple transformer block for demonstration."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)

        return x


class LanguageModel(nn.Module):
    """Simple language model for testing contiguous buffers."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position ids
        position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = self.embedding_dropout(token_embeds + position_embeds)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


def create_synthetic_dataset(
    vocab_size: int,
    seq_len: int,
    num_samples: int,
    batch_size: int,
) -> DataLoader:
    """Create a synthetic dataset for training."""
    # Generate random sequences
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))

    # Shift tokens for next-token prediction
    targets = torch.roll(input_ids, -1, dims=1)
    targets[:, -1] = 0  # Padding token

    dataset = TensorDataset(input_ids, targets)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return dataloader


def setup_distributed() -> Dict:
    """Setup distributed training environment."""
    if not dist.is_initialized():
        # Initialize process group
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return {
        "local_rank": local_rank,
        "world_size": world_size,
        "rank": rank,
        "device": device,
    }


def demonstrate_contiguous_buffers(
    model: nn.Module,
    device: torch.device,
    bucket_config: ContiguousBucketConfig,
    world_size: int = 1,
) -> None:
    """Demonstrate contiguous buffer features."""

    logger.info("=" * 60)
    logger.info("Demonstrating Contiguous Parameter-Gradient Buffers")
    logger.info("=" * 60)

    # Create buffer manager
    process_group = dist.group.WORLD if world_size > 1 else None
    buffer_mgr = ContiguousParamGradBuffer(
        model=model,
        bucket_config=bucket_config,
        data_parallel_group=process_group,
    )

    # Display buffer statistics
    stats = buffer_mgr.get_memory_usage()
    logger.info(f"\nBuffer Statistics:")
    logger.info(f"  Total parameters: {stats['total_params']:,}")
    logger.info(f"  Total buckets: {stats['total_buckets']}")
    logger.info(f"  Parameter memory: {stats['total_param_memory_mb']:.2f} MB")
    logger.info(f"  Gradient memory: {stats['total_grad_memory_mb']:.2f} MB")
    logger.info(f"  Total memory: {stats['total_memory_mb']:.2f} MB")

    # Display bucket details
    logger.info(f"\nBucket Details:")
    bucket_stats_dict = stats.get("bucket_stats", {})
    if isinstance(bucket_stats_dict, dict):
        for key, bucket_list in bucket_stats_dict.items():
            logger.info(f"  {key}:")
            for i, bucket_stats in enumerate(bucket_list):
                logger.info(f"    Bucket {i}:")
                logger.info(
                    f"      Parameter fill: {bucket_stats['param_fill_ratio']:.1%}"
                )
                logger.info(
                    f"      Gradient fill: {bucket_stats['grad_fill_ratio']:.1%}"
                )
                logger.info(f"      Memory: {bucket_stats['total_memory_mb']:.2f} MB")

    # Demonstrate gradient operations
    logger.info(f"\nGradient Operations:")

    # Perform a forward-backward pass
    batch_size = 8
    seq_len = 128
    # Get vocab size from the linear layer
    vocab_size: int = 10000  # Default
    if hasattr(model, "lm_head"):
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "out_features"):
            vocab_size = int(lm_head.out_features)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.roll(input_ids, -1, dims=1)

    # Forward pass
    logits = model(input_ids)
    loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), targets.view(-1))

    # Zero gradients
    buffer_mgr.zero_gradients()
    logger.info("  ✓ Zeroed gradients")

    # Backward pass (gradients accumulated via hooks)
    loss.backward()
    logger.info("  ✓ Performed backward pass (gradients accumulated in buffers)")

    # Sync gradients to parameters
    buffer_mgr.sync_gradients_to_params()
    logger.info("  ✓ Synchronized gradients to parameters")

    # Calculate gradient norm
    total_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm_sq += (param.grad * param.grad).sum().item()
    total_norm = total_norm_sq**0.5
    logger.info(f"  ✓ Gradient norm: {total_norm:.4f}")

    # Demonstrate gradient clipping
    max_norm = 1.0
    clipped_norm = buffer_mgr.clip_gradients(max_norm)
    logger.info(
        f"  ✓ Clipped gradients (original norm: {clipped_norm:.4f}, max: {max_norm})"
    )

    # Demonstrate all-reduce (if distributed)
    if world_size > 1:
        logger.info(f"\nDistributed Operations:")

        # Perform all-reduce
        handles = buffer_mgr.all_reduce_gradients(async_op=True)
        logger.info(
            f"  ✓ Started async all-reduce ({len(handles) if handles else 0} handles)"
        )

        # Simulate other computation
        time.sleep(0.1)

        # Complete all-reduce
        buffer_mgr.finish_all_reduce()
        logger.info("  ✓ Completed all-reduce and synchronized gradients")

    # Cleanup
    buffer_mgr.restore_params()
    logger.info("\n✓ Restored parameters to independent memory")


def train_with_contiguous_buffers(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    trainer: DataParallelTrainer,
    num_epochs: int,
    learning_rate: float,
    log_interval: int = 10,
    use_async_allreduce: bool = False,
) -> Dict:
    """Train model using contiguous buffers."""

    logger.info("\n" + "=" * 60)
    logger.info("Training with Contiguous Buffers")
    logger.info("=" * 60)

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Training metrics
    total_loss = 0.0
    total_steps = 0
    epoch_times = []

    model.train()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            # Prepare batch
            batch = {
                "input_ids": input_ids,
                "labels": targets,
            }

            # Custom loss function
            def loss_fn(logits, batch_dict):
                vocab_size = logits.size(-1)
                return nn.CrossEntropyLoss()(
                    logits.view(-1, vocab_size), batch_dict["labels"].view(-1)
                )

            # Forward-backward pass
            loss = trainer.forward_backward(
                batch, optimizer, loss_fn, async_allreduce=use_async_allreduce
            )

            # Complete async all-reduce if needed
            if use_async_allreduce and trainer.buffer_manager is not None:
                # Simulate other computation
                pass

                # Complete all-reduce and optimizer step
                trainer.finish_async_allreduce(optimizer)

            # Update metrics
            total_loss += loss.item()
            epoch_loss += loss.item()
            total_steps += 1
            num_batches += 1

            # Log progress
            if batch_idx % log_interval == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{num_epochs}, "
                    f"Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        # Epoch statistics
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_loss = epoch_loss / num_batches

        logger.info(
            f"Epoch {epoch+1} completed in {epoch_time:.2f}s, "
            f"Average Loss: {avg_epoch_loss:.4f}"
        )

        # Log buffer statistics
        if trainer.buffer_manager is not None:
            stats = trainer.get_buffer_statistics()
            if stats:
                logger.info(f"  Buffer memory usage: {stats['total_memory_mb']:.2f} MB")

    return {
        "avg_loss": total_loss / total_steps,
        "total_steps": total_steps,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Contiguous Parameter-Gradient Buffer Example"
    )

    # Model configuration
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training configuration
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num-samples", type=int, default=500, help="Number of samples"
    )
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")

    # Buffer configuration
    parser.add_argument(
        "--bucket-size", type=float, default=25.0, help="Bucket size in MB"
    )
    parser.add_argument("--alignment", type=int, default=128, help="Memory alignment")
    parser.add_argument("--use-hooks", action="store_true", help="Use gradient hooks")
    parser.add_argument("--auto-clip", action="store_true", help="Auto clip gradients")
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Max gradient norm"
    )

    # Distributed configuration
    parser.add_argument("--distributed", action="store_true", help="Enable distributed")
    parser.add_argument(
        "--async-allreduce", action="store_true", help="Use async allreduce"
    )

    # Demo options
    parser.add_argument(
        "--demo-only", action="store_true", help="Only run demonstration"
    )

    args = parser.parse_args()

    # Setup device and distributed training
    if args.distributed:
        dist_info = setup_distributed()
        device = dist_info["device"]
        world_size = dist_info["world_size"]
        rank = dist_info["rank"]
        local_rank = dist_info["local_rank"]

        logger.info(
            f"Distributed setup: rank={rank}, world_size={world_size}, device={device}"
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
        local_rank = 0

    logger.info(f"Using device: {device}")

    # Create model
    model = LanguageModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    )
    model = model.to(device)

    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model created: {total_params:,} total parameters "
        f"({trainable_params:,} trainable)"
    )

    # Configure bucket settings
    bucket_config = ContiguousBucketConfig(
        bucket_size_mb=args.bucket_size,
        alignment=args.alignment,
        overlap_comm=True,
        dtype_buckets=True,
        device_buckets=True,
        use_gradient_hooks=args.use_hooks,
        auto_clip_gradients=args.auto_clip,
        max_gradient_norm=args.max_grad_norm,
    )

    # Run demonstration
    if args.demo_only:
        demonstrate_contiguous_buffers(model, device, bucket_config, world_size)
    else:
        # Create dataset
        dataloader = create_synthetic_dataset(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )

        # Create trainer with contiguous buffers
        trainer = DataParallelTrainer(
            model=model,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
            use_contiguous_buffers=True,
            bucket_config=bucket_config,
            grad_clip=args.max_grad_norm if not args.auto_clip else None,
        )

        # Train model
        start_time = time.time()
        training_stats = train_with_contiguous_buffers(
            model=model,
            dataloader=dataloader,
            device=device,
            trainer=trainer,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            log_interval=args.log_interval,
            use_async_allreduce=args.async_allreduce,
        )
        total_time = time.time() - start_time

        # Log results
        logger.info("\n" + "=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        logger.info(f"Total training time: {total_time:.2f}s")
        logger.info(f"Average loss: {training_stats['avg_loss']:.4f}")
        logger.info(f"Average epoch time: {training_stats['avg_epoch_time']:.2f}s")
        logger.info(f"Total steps: {training_stats['total_steps']}")

        # Final buffer statistics
        if trainer.buffer_manager is not None:
            stats = trainer.get_buffer_statistics()
            if stats:
                logger.info(f"\nFinal Buffer Statistics:")
                logger.info(f"  Total parameters: {stats['total_params']:,}")
                logger.info(f"  Total buckets: {stats['total_buckets']}")
                logger.info(f"  Total memory: {stats['total_memory_mb']:.2f} MB")

        # Cleanup
        trainer.cleanup()

    # Cleanup distributed
    if args.distributed:
        dist.destroy_process_group()

    logger.info("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()
