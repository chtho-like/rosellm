#!/usr/bin/env python3
"""
Example script demonstrating custom gradient scaler usage with FP16 training.

This example shows how to use the custom gradient scalers for mixed precision
training with automatic loss scaling and overflow handling.
"""

import argparse
import logging
import os
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rosellm.rosetrainer.mixed_precision import AbstractGradScaler
from rosellm.rosetrainer.mixed_precision import (
    EnhancedDynamicGradScaler as DynamicGradScaler,
)
from rosellm.rosetrainer.mixed_precision import GradScalerConfig
from rosellm.rosetrainer.mixed_precision.gradient_scaler import check_for_inf_and_nan

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""

    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class SimpleLanguageModel(nn.Module):
    """Simple language model for demonstration."""

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        num_layers: int = 6,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model) for _ in range(num_layers)]
        )

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embedding and positional encoding
        x = self.embedding(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        logits = self.output_proj(x)
        return logits


def create_dummy_data(
    batch_size: int = 32,
    seq_len: int = 128,
    vocab_size: int = 10000,
    num_batches: int = 100,
) -> DataLoader:
    """Create dummy data for demonstration."""
    # Generate random token sequences
    input_ids = torch.randint(0, vocab_size, (batch_size * num_batches, seq_len))
    # Shift targets by one position for language modeling
    targets = torch.randint(0, vocab_size, (batch_size * num_batches, seq_len))

    dataset = TensorDataset(input_ids, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Union[AbstractGradScaler, DynamicGradScaler],
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, bool]:
    """
    Perform a single training step with gradient scaling.

    Returns:
        Tuple of (loss_value, gradient_overflow_occurred)
    """
    input_ids, targets = batch
    input_ids = input_ids.to(device)
    targets = targets.to(device)

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass with automatic mixed precision
    if use_amp:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
    else:
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    # Scale loss and backward pass
    scaled_loss = scaler.scale_loss(loss)
    scaled_loss.backward()

    # Check for gradient overflow
    found_inf = check_for_inf_and_nan(model, scaler)

    if not found_inf:
        # Unscale gradients before clipping
        scaler.unscale_gradients(model)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

    return loss.item(), found_inf


def main():
    parser = argparse.ArgumentParser(description="Gradient Scaler Example")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument(
        "--scaler-type",
        type=str,
        default="dynamic",
        choices=["constant", "dynamic", "none"],
    )
    parser.add_argument("--initial-scale", type=float, default=2**16)
    parser.add_argument("--growth-interval", type=int, default=500)
    parser.add_argument(
        "--use-amp", action="store_true", help="Use automatic mixed precision"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create model
    model = SimpleLanguageModel(
        vocab_size=10000,
        d_model=256,  # Smaller for example
        num_layers=4,
        max_seq_len=128,
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Configure and create gradient scaler
    scaler_config = GradScalerConfig(
        scaler_type=args.scaler_type,
        initial_scale=args.initial_scale,
        growth_interval=args.growth_interval,
        growth_factor=2.0,
        backoff_factor=0.5,
        hysteresis=2,
    )

    scaler = scaler_config.create_scaler(device=str(device))
    if scaler is None:
        # Create a dynamic scaler if none was specified
        scaler = DynamicGradScaler(initial_scale=1.0, device=device)

    logger.info(
        f"Created {args.scaler_type} gradient scaler with "
        f"initial scale: {scaler.scale.item():.1f}"
    )

    # Create dummy data
    dataloader = create_dummy_data(
        batch_size=args.batch_size, seq_len=128, vocab_size=10000, num_batches=100
    )

    # Training loop
    step = 0
    overflow_count = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_overflows = 0

        for batch_idx, batch in enumerate(dataloader):
            loss, found_inf = train_step(
                model, batch, optimizer, scaler, device, use_amp=args.use_amp
            )

            epoch_loss += loss
            if found_inf:
                epoch_overflows += 1
                overflow_count += 1

            step += 1

            # Log progress
            if step % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs}, "
                    f"Step {step}, "
                    f"Loss: {loss:.4f}, "
                    f"Scale: {scaler.scale.item():.1f}, "
                    f"Overflows: {overflow_count}"
                )

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        logger.info(
            f"Epoch {epoch+1} Summary - "
            f"Avg Loss: {avg_loss:.4f}, "
            f"Overflows: {epoch_overflows}, "
            f"Final Scale: {scaler.scale.item():.1f}"
        )

    # Save model and scaler state
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "overflow_count": overflow_count,
    }

    checkpoint_path = "gradient_scaler_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Demonstrate loading
    logger.info("\n--- Demonstrating checkpoint loading ---")

    # Create new model and scaler
    new_model = SimpleLanguageModel(
        vocab_size=10000, d_model=256, num_layers=4, max_seq_len=128
    ).to(device)

    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=args.lr)
    new_scaler = scaler_config.create_scaler(device=str(device))
    if new_scaler is None:
        new_scaler = DynamicGradScaler(initial_scale=1.0, device=device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_model.load_state_dict(checkpoint["model_state_dict"])
    new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    new_scaler.load_state_dict(checkpoint["scaler_state_dict"])

    logger.info(
        f"Loaded checkpoint - "
        f"Step: {checkpoint['step']}, "
        f"Scale: {new_scaler.scale.item():.1f}, "
        f"Total Overflows: {checkpoint['overflow_count']}"
    )

    # Clean up
    os.remove(checkpoint_path)
    logger.info(f"Cleaned up checkpoint file")


if __name__ == "__main__":
    main()
