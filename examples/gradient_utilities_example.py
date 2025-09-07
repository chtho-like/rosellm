#!/usr/bin/env python3
"""
Example demonstrating advanced gradient utilities in RoseLLM.

This example shows how to use the gradient utilities for:
- Multi-tensor gradient norm calculation
- Gradient clipping with various strategies
- Gradient accumulation with DDP
- Mixed precision training with custom scaler
- Gradient statistics monitoring
"""

import argparse
import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from rosellm.rosetrainer.utils import (
    CustomGradientScaler,
    GradientClipConfig,
    apply_gradient_clipping,
    calculate_gradient_norm_multitensor,
    check_gradient_finite,
    get_gradient_stats,
    gradient_accumulation_context,
    sync_gradients,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""

    def __init__(self, d_model: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DemoModel(nn.Module):
    """Demo model with transformer blocks."""

    def __init__(self, vocab_size: int = 10000, d_model: int = 512, n_layers: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, d_model))

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model) for _ in range(n_layers)]
        )

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)

        # Embeddings with positional encoding
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer blocks
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        # Output projection
        return self.output(x)


def setup_distributed() -> Tuple[int, int]:
    """Set up distributed training if available."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        return local_rank, world_size
    else:
        return 0, 1


def demonstrate_gradient_utilities(
    use_mixed_precision: bool = True,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    monitor_stats: bool = True,
):
    """Demonstrate various gradient utilities."""

    # Setup
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running on device: {device}, World size: {world_size}")

    # Model setup
    model = DemoModel().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Gradient scaler for mixed precision
    scaler = CustomGradientScaler(enabled=use_mixed_precision and device.type == "cuda")

    # Gradient clipping configuration
    clip_config = GradientClipConfig(
        clip_type="norm",
        max_norm=max_grad_norm,
        norm_type=2.0,
        error_if_nonfinite=True,
        model_parallel_reduce=False,
        use_multitensor=True,
    )

    # Training loop
    batch_size = 8
    seq_len = 128
    num_steps = 20

    for step in range(num_steps):
        # Reset gradients at the start of accumulation
        if step % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        # Generate random data
        input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
        target = torch.randint(0, 10000, (batch_size, seq_len), device=device)

        # Use gradient accumulation context
        with gradient_accumulation_context(
            model, gradient_accumulation_steps, sync_on_last_step=True
        ) as is_last_accumulation:
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(
                enabled=use_mixed_precision and device.type == "cuda"
            ):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), target.reshape(-1)
                )
                loss = loss / gradient_accumulation_steps

            # Backward pass
            if use_mixed_precision and device.type == "cuda":
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
            else:
                loss.backward()

            # Only perform optimizer step on last accumulation
            if is_last_accumulation:
                # Unscale gradients if using mixed precision
                if use_mixed_precision and device.type == "cuda":
                    scaler.unscale_(optimizer)

                # Check for non-finite gradients
                is_finite, finite_stats = check_gradient_finite(
                    model, raise_on_nonfinite=False
                )

                if not is_finite:
                    logger.warning(
                        f"Step {step}: Non-finite gradients detected - "
                        f"NaN: {finite_stats['nan_parameters']}, "
                        f"Inf: {finite_stats['inf_parameters']}"
                    )
                    optimizer.zero_grad()
                    continue

                # Calculate gradient norm before clipping
                grad_norm_before = calculate_gradient_norm_multitensor(
                    model,
                    norm_type=2.0,
                    use_multitensor=True,
                    model_parallel_reduce=False,
                )

                # Apply gradient clipping
                clip_stats = apply_gradient_clipping(model, clip_config)

                # Synchronize gradients across data parallel ranks
                if world_size > 1:
                    sync_gradients(model)

                # Monitor gradient statistics
                if monitor_stats and step % 5 == 0:
                    grad_stats = get_gradient_stats(
                        model,
                        include_histograms=False,
                        compute_percentiles=True,
                    )

                    logger.info(
                        f"Step {step}: Loss={loss.item():.4f}, "
                        f"Grad norm before={grad_norm_before:.4f}, "
                        f"Grad norm after={clip_stats['grad_norm']:.4f}, "
                        f"Clipped={clip_stats['clipped']}, "
                        f"Mean grad={grad_stats['grad_mean']:.6f}, "
                        f"Std grad={grad_stats['grad_std']:.6f}"
                    )

                    if "percentiles" in grad_stats:
                        logger.info(
                            f"  Gradient percentiles: "
                            f"p50={grad_stats['percentiles']['p50']:.6f}, "
                            f"p90={grad_stats['percentiles']['p90']:.6f}, "
                            f"p99={grad_stats['percentiles']['p99']:.6f}"
                        )

                # Optimizer step
                if use_mixed_precision and device.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # Log scaler state periodically
                if use_mixed_precision and device.type == "cuda" and step % 10 == 0:
                    logger.info(
                        f"Scaler state: scale={scaler.get_scale():.1f}, "
                        f"growth_tracker={scaler.get_growth_tracker()}"
                    )

    logger.info("Training demonstration complete!")

    # Clean up distributed
    if world_size > 1:
        dist.destroy_process_group()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gradient Utilities Example")
    parser.add_argument(
        "--mixed-precision", action="store_true", help="Use mixed precision training"
    )
    parser.add_argument(
        "--accumulation-steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--monitor-stats", action="store_true", help="Monitor gradient statistics"
    )

    args = parser.parse_args()

    demonstrate_gradient_utilities(
        use_mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        monitor_stats=args.monitor_stats,
    )


if __name__ == "__main__":
    main()
