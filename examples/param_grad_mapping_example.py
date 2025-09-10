#!/usr/bin/env python3
"""
Example demonstrating parameter-gradient mapping with bucket-based reduction.

This example shows how to use the ParamGradMapping feature for efficient
gradient synchronization in distributed training scenarios.
"""

import argparse
import logging
import time
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from rosellm.rosetrainer.optimizer import (
    ParameterType,
    ParamGradMapping,
    ParamGradMappingBuilder,
    ReductionStrategy,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleTransformer(nn.Module):
    """Simple transformer model for demonstration."""

    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(512, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Output
        self.ln_final = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_embeds = self.position_embedding(position_ids)

        x = self.dropout(token_embeds + position_embeds.unsqueeze(0))

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits


def create_param_grad_mapping(
    model: nn.Module, args: argparse.Namespace
) -> ParamGradMapping:
    """Create parameter-gradient mapping with specified configuration."""

    logger.info("Creating parameter-gradient mapping...")

    # Build mapping with configuration
    builder = ParamGradMappingBuilder().with_parameters(list(model.parameters()))

    # Configure bucket size
    builder = builder.with_bucket_size(args.bucket_size_mb)

    # Configure reduction strategy
    strategy_map = {
        "immediate": ReductionStrategy.IMMEDIATE,
        "delayed": ReductionStrategy.DELAYED,
        "overlapped": ReductionStrategy.OVERLAPPED,
        "hierarchical": ReductionStrategy.HIERARCHICAL,
    }
    builder = builder.with_reduction_strategy(
        strategy_map.get(args.reduction_strategy, ReductionStrategy.OVERLAPPED)
    )

    # Configure gradient accumulation
    if args.gradient_accumulation_steps > 1:
        builder = builder.with_gradient_accumulation(args.gradient_accumulation_steps)

    # Configure gradient clipping
    if args.gradient_clip > 0:
        builder = builder.with_gradient_clipping(args.gradient_clip)

    # Configure type-specific buckets
    if args.type_specific_buckets:
        type_sizes = {
            ParameterType.EMBEDDING: args.embedding_bucket_size,
            ParameterType.WEIGHT: args.weight_bucket_size,
            ParameterType.BIAS: args.bias_bucket_size,
            ParameterType.NORM: args.norm_bucket_size,
        }
        builder = builder.with_type_specific_buckets(type_sizes)

    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.use_fp16 else torch.float32
    builder = builder.with_device(device).with_dtype(dtype)

    # Build the mapping
    mapping = builder.build()

    # Log configuration
    stats = mapping.get_statistics()
    logger.info(f"Created mapping with {stats['total_parameters']} parameters")
    logger.info(f"Total parameter elements: {stats['total_parameter_elements']:,}")
    logger.info(
        f"Gradient parameter elements: {stats['gradient_parameter_elements']:,}"
    )

    if "bucket_statistics" in stats and stats["bucket_statistics"]:
        bucket_stats = stats["bucket_statistics"]
        logger.info(f"Number of buckets: {bucket_stats.get('num_buckets', 0)}")

    return mapping


def train_step(
    model: nn.Module,
    mapping: ParamGradMapping,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    step: int,
) -> Dict[str, Any]:
    """Perform a single training step."""

    start_time = time.time()

    # Forward pass
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    # Backward pass
    loss.backward()

    # Accumulate gradients with mapping
    mapping.accumulate_gradients()

    # Check if we should reduce gradients
    should_update = mapping.should_reduce_gradients()

    sync_time = 0.0
    if should_update:
        # Synchronize gradients across ranks
        sync_start = time.time()
        sync_stats = mapping.synchronize_gradients()
        sync_time = time.time() - sync_start

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Log synchronization stats
        if step % 10 == 0:
            logger.info(
                f"Step {step}: Gradient sync completed in {sync_time:.3f}s, "
                f"stats: {sync_stats}"
            )

    total_time = time.time() - start_time

    return {
        "loss": loss.item(),
        "total_time": total_time,
        "sync_time": sync_time,
        "updated": should_update,
    }


def main():
    parser = argparse.ArgumentParser(description="Parameter-gradient mapping example")

    # Model configuration
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden size")
    parser.add_argument(
        "--num-layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )

    # Training configuration
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")
    parser.add_argument(
        "--num-steps", type=int, default=100, help="Number of training steps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )

    # ParamGradMapping configuration
    parser.add_argument(
        "--bucket-size-mb", type=float, default=25.0, help="Bucket size in MB"
    )
    parser.add_argument(
        "--reduction-strategy",
        type=str,
        default="overlapped",
        choices=["immediate", "delayed", "overlapped", "hierarchical"],
        help="Gradient reduction strategy",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (0 to disable)",
    )
    parser.add_argument(
        "--type-specific-buckets",
        action="store_true",
        help="Use type-specific bucket sizes",
    )
    parser.add_argument(
        "--embedding-bucket-size",
        type=float,
        default=50.0,
        help="Bucket size for embeddings (MB)",
    )
    parser.add_argument(
        "--weight-bucket-size",
        type=float,
        default=25.0,
        help="Bucket size for weights (MB)",
    )
    parser.add_argument(
        "--bias-bucket-size",
        type=float,
        default=5.0,
        help="Bucket size for biases (MB)",
    )
    parser.add_argument(
        "--norm-bucket-size",
        type=float,
        default=10.0,
        help="Bucket size for normalization parameters (MB)",
    )
    parser.add_argument(
        "--use-fp16", action="store_true", help="Use FP16 mixed precision"
    )

    # Distributed configuration
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )
    parser.add_argument(
        "--local-rank", type=int, default=0, help="Local rank for distributed training"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    # Initialize distributed if enabled
    if args.distributed and torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        logger.info(f"Initialized distributed training, rank {dist.get_rank()}")

    # Create model
    logger.info("Creating model...")
    model = SimpleTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    # Wrap with DDP if distributed
    if args.distributed and torch.cuda.is_available():
        model = DDP(model, device_ids=[args.local_rank])

    # Create parameter-gradient mapping
    base_model: nn.Module
    if hasattr(model, "module"):
        base_model = model.module  # type: ignore
    else:
        base_model = model  # type: ignore

    mapping = create_param_grad_mapping(base_model, args)

    # Create optimizer
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=args.learning_rate)

    # Training loop
    logger.info("Starting training...")
    total_loss = 0.0
    total_sync_time = 0.0
    update_count = 0

    for step in range(args.num_steps):
        # Generate random data
        input_ids = torch.randint(
            0, args.vocab_size, (args.batch_size, args.seq_length), device=device
        )
        labels = torch.randint(
            0, args.vocab_size, (args.batch_size, args.seq_length), device=device
        )

        # Train step
        step_stats = train_step(model, mapping, optimizer, input_ids, labels, step)

        # Accumulate statistics
        total_loss += step_stats["loss"]
        total_sync_time += step_stats["sync_time"]
        if step_stats["updated"]:
            update_count += 1

        # Log progress
        if (step + 1) % 10 == 0:
            avg_loss = total_loss / (step + 1)
            logger.info(
                f"Step {step + 1}/{args.num_steps}: "
                f"Loss = {avg_loss:.4f}, "
                f"Updates = {update_count}, "
                f"Sync time = {total_sync_time:.3f}s"
            )

    # Final statistics
    logger.info("\n" + "=" * 50)
    logger.info("Training completed!")

    final_stats = mapping.get_statistics()
    logger.info("\nFinal ParamGradMapping Statistics:")
    logger.info(f"  Total reductions: {final_stats['total_reductions']}")
    logger.info(
        f"  Total communication time: {final_stats['total_communication_time']:.3f}s"
    )
    logger.info(
        f"  Average communication time: {final_stats['avg_communication_time']:.3f}s"
    )

    if "bucket_statistics" in final_stats and final_stats["bucket_statistics"]:
        bucket_stats = final_stats["bucket_statistics"]
        logger.info(f"\nBucket Statistics:")
        logger.info(f"  Strategy: {bucket_stats.get('strategy', 'unknown')}")
        logger.info(f"  Number of buckets: {bucket_stats.get('num_buckets', 0)}")
        logger.info(f"  Total gradients: {bucket_stats.get('total_gradients', 0)}")
        logger.info(f"  Total size (MB): {bucket_stats.get('total_size_mb', 0):.2f}")
        logger.info(
            f"  Average bucket size (MB): {bucket_stats.get('avg_bucket_size_mb', 0):.2f}"
        )

    # Cleanup
    if args.distributed and torch.cuda.is_available():
        dist.destroy_process_group()

    logger.info("\nExample completed successfully!")


if __name__ == "__main__":
    main()
    main()
