#!/usr/bin/env python3
"""
Example demonstrating gradient finalization with distributed optimizer.

This example shows how to use the GradientFinalizer with DistributedOptimizer
for efficient gradient synchronization across multiple parallelism dimensions.

Usage:
    # Single GPU
    python gradient_finalization_example.py

    # Multi-GPU with data parallelism
    torchrun --nproc_per_node=2 gradient_finalization_example.py

    # Multi-dimensional parallelism (TP=2, DP=2)
    torchrun --nproc_per_node=4 gradient_finalization_example.py --tp-size 2 --dp-size 2
"""

import argparse
import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from rosellm.rosetrainer import (
    DistributedOptimizer,
    DistributedOptimizerConfig,
    GradientFinalizationConfig,
    GradientFinalizer,
    initialize_model_parallel,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)

        # MLP
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)

        return x


class SimpleLanguageModel(nn.Module):
    """Simple language model for demonstration."""

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()
        # Ensure hidden_size is divisible by num_heads
        assert (
            hidden_size % num_heads == 0
        ), f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(1024, hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        x = self.embedding(input_ids) + self.position_embedding(position_ids)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits


def setup_distributed() -> Tuple[int, int]:
    """Setup distributed training environment.

    Returns:
        Tuple of (rank, world_size).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        rank = 0
        world_size = 1

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    return rank, world_size


def main():
    parser = argparse.ArgumentParser(description="Gradient Finalization Example")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument(
        "--num-heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--num-steps", type=int, default=10, help="Number of training steps"
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps"
    )

    # Parallelism configuration
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--dp-size", type=int, default=-1, help="Data parallel size")
    parser.add_argument("--cp-size", type=int, default=1, help="Context parallel size")
    parser.add_argument("--ep-size", type=int, default=1, help="Expert parallel size")

    # Gradient finalization configuration
    parser.add_argument(
        "--sync-strategy",
        type=str,
        default="bucketed",
        choices=["simple", "bucketed", "hierarchical"],
        help="Gradient synchronization strategy",
    )
    parser.add_argument(
        "--bucket-size-mb", type=float, default=25.0, help="Bucket size in MB"
    )
    parser.add_argument(
        "--fp16-compression", action="store_true", help="Use FP16 compression"
    )
    parser.add_argument(
        "--overlap-grad-sync", action="store_true", help="Overlap gradient sync"
    )
    parser.add_argument(
        "--enable-stats", action="store_true", help="Enable gradient statistics"
    )

    args = parser.parse_args()

    # Setup distributed
    rank, world_size = setup_distributed()

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Calculate data parallel size if not specified
    if args.dp_size == -1:
        args.dp_size = world_size // (
            args.tp_size * args.pp_size * args.cp_size * args.ep_size
        )

    # Initialize model parallel
    if world_size > 1:
        initialize_model_parallel(
            tensor_model_parallel_size=args.tp_size,
            pipeline_model_parallel_size=args.pp_size,
            data_parallel_size=args.dp_size,
            context_parallel_size=args.cp_size,
            expert_model_parallel_size=args.ep_size,
        )

    # Create model
    model = SimpleLanguageModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    logger.info(
        f"Rank {rank}/{world_size}: Created model with "
        f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters"
    )

    # Configure distributed optimizer
    opt_config = DistributedOptimizerConfig(
        partition_parameters=world_size > 1,
        contiguous_gradients=True,
        reduce_bucket_size_mb=args.bucket_size_mb,
        overlap_grad_reduce=args.overlap_grad_sync,
        grad_clip_value=1.0,
        mixed_precision=False,  # Can enable for FP16 training
        verbose=rank == 0,
    )

    # Create optimizer - use distributed optimizer only if distributed is initialized
    if dist.is_initialized() and world_size > 1:
        distributed_optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={"lr": args.lr, "weight_decay": 0.01},
            config=opt_config,
            process_group=dist.group.WORLD,
        )
        optimizer = distributed_optimizer
    else:
        # Use regular optimizer for single GPU
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
        )
        distributed_optimizer = None  # No distributed optimizer for single GPU

    # Configure gradient finalization
    grad_config = GradientFinalizationConfig(
        sync_strategy=args.sync_strategy,
        reduction_op="mean",
        dimension_order=(
            "hierarchical" if args.sync_strategy == "hierarchical" else "tp-pp-dp-cp-ep"
        ),
        bucket_size_mb=args.bucket_size_mb,
        overlap_grad_sync=args.overlap_grad_sync,
        sync_grad_before_clip=True,
        use_contiguous_buffers=True,
        fp16_compression=args.fp16_compression,
        enable_gradient_stats=args.enable_stats,
        gradient_norm_type=2.0,
        verbose=rank == 0,
    )

    # Create gradient finalizer
    grad_finalizer = GradientFinalizer(
        model=model,
        config=grad_config,
        distributed_optimizer=distributed_optimizer,
    )

    logger.info(
        f"Rank {rank}: Initialized gradient finalizer with "
        f"{args.sync_strategy} strategy"
    )

    # Training loop
    for step in range(args.num_steps):
        # Generate random data
        input_ids = torch.randint(
            0, args.vocab_size, (args.batch_size, args.seq_length)
        ).to(device)
        labels = torch.randint(
            0, args.vocab_size, (args.batch_size, args.seq_length)
        ).to(device)

        # Forward pass
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), labels.view(-1))

        # Scale loss for gradient accumulation
        if args.grad_accum_steps > 1:
            loss = loss / args.grad_accum_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (step + 1) % args.grad_accum_steps == 0:
            # Finalize gradients with synchronization
            finalize_stats = grad_finalizer.finalize_gradients(
                clip_gradients=True,
                check_finite=True,
                collect_stats=args.enable_stats,
            )

            # Log statistics
            if rank == 0:
                logger.info(
                    f"Step {step + 1}: loss={loss.item() * args.grad_accum_steps:.4f}, "
                    f"grad_norm={finalize_stats['gradient_norm']:.4f}, "
                    f"sync_time="
                    f"{finalize_stats.get('sync_stats', {}).get('sync_time', 0):.3f}s"
                )

                if args.enable_stats and "gradient_stats" in finalize_stats:
                    grad_stats = finalize_stats["gradient_stats"]
                    logger.info(
                        f"  Gradient stats: mean={grad_stats['grad_mean']:.6f}, "
                        f"std={grad_stats['grad_std']:.6f}, "
                        f"min={grad_stats['grad_min']:.6f}, "
                        f"max={grad_stats['grad_max']:.6f}"
                    )

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

    # Print final statistics summary
    if rank == 0 and args.enable_stats:
        summary = grad_finalizer.get_statistics_summary()
        logger.info("\nGradient Finalization Summary:")
        logger.info(f"  Total finalizations: {summary['total_finalizations']}")
        logger.info(f"  Avg finalize time: {summary['avg_finalization_time']:.3f}s")
        logger.info(f"  Average gradient norm: {summary['avg_gradient_norm']:.4f}")
        logger.info(f"  Max gradient norm: {summary['max_gradient_norm']:.4f}")
        logger.info(f"  Number of clips: {summary['num_clipped']}")
        logger.info(f"  Number of non-finite: {summary['num_non_finite']}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    logger.info(f"Rank {rank}: Training completed successfully!")


if __name__ == "__main__":
    main()
