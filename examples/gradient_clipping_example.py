#!/usr/bin/env python3
"""
Example demonstrating gradient clipping utilities in RoseTrainer.

This example shows:
1. Basic L2 norm clipping
2. Value clipping for gradient explosion prevention
3. Adaptive clipping based on statistics
4. Distributed training with proper norm reduction
5. Integration with training loops
6. Monitoring and statistics collection
"""

import argparse
import os
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from rosellm.rosetrainer.gradient import (
    ClipType,
    GradientClipper,
    clip_grad_norm,
    clip_grad_value,
)


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""

    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        vocab_size: int = 10000,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embedding and positional encoding
        x = self.embedding(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer layers
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        # Output projection
        logits = self.output_proj(x)
        return logits


def setup_distributed():
    """Setup distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Initialize process group
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())

        return rank, world_size
    return 0, 1


def demonstrate_norm_clipping(model: nn.Module, args: argparse.Namespace):
    """Demonstrate L2 norm gradient clipping."""
    print("\n" + "=" * 60)
    print("L2 NORM GRADIENT CLIPPING")
    print("=" * 60)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create gradient clipper
    clipper = GradientClipper(
        max_norm=args.max_norm,
        clip_type=ClipType.NORM,
        log_stats=True,
    )

    # Training loop
    for step in range(args.steps):
        # Generate random data
        batch_size = args.batch_size
        seq_len = 128
        input_ids = torch.randint(0, 10000, (batch_size, seq_len))
        labels = torch.randint(0, 10000, (batch_size, seq_len))

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        # Forward pass
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients and get statistics
        stats = clipper.clip_gradients(model)

        # Optimizer step
        optimizer.step()

        # Print statistics
        print(f"\nStep {step + 1}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradient norm (before clip): {stats['total_norm']:.4f}")
        print(f"  Clipped: {stats['clipped']}")
        if stats["clipped"]:
            print(f"  Clip coefficient: {stats['clip_coef']:.4f}")

    print("\nAlternative API (PyTorch-compatible):")

    # Demonstrate PyTorch-compatible API
    for step in range(2):
        # Generate data
        input_ids = torch.randint(0, 10000, (batch_size, seq_len))
        labels = torch.randint(0, 10000, (batch_size, seq_len))

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        # Forward and backward
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()

        # Use PyTorch-compatible function
        total_norm = clip_grad_norm(list(model.parameters()), max_norm=args.max_norm)

        optimizer.step()

        print(f"  Step {step + 1}: Norm = {total_norm:.4f}")


def demonstrate_value_clipping(model: nn.Module, args: argparse.Namespace):
    """Demonstrate value-based gradient clipping."""
    print("\n" + "=" * 60)
    print("VALUE-BASED GRADIENT CLIPPING")
    print("=" * 60)

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Create gradient clipper
    clipper = GradientClipper(
        max_value=args.max_value,
        clip_type=ClipType.VALUE,
        log_stats=True,
    )

    # Training loop
    for step in range(args.steps):
        # Generate random data with potential gradient explosion
        batch_size = args.batch_size
        seq_len = 128
        input_ids = torch.randint(0, 10000, (batch_size, seq_len))
        labels = torch.randint(0, 10000, (batch_size, seq_len))

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        # Forward pass
        logits = model(input_ids)

        # Scale loss to create large gradients
        loss = (
            F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            * 100
        )  # Amplify gradients

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients by value
        stats = clipper.clip_gradients(model)

        # Optimizer step
        optimizer.step()

        # Print statistics
        print(f"\nStep {step + 1}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Max gradient value: {stats['max_grad']:.4f}")
        print(f"  Parameters clipped: {stats['num_clipped']}")

    print("\nAlternative API:")

    # Demonstrate alternative API
    for step in range(2):
        input_ids = torch.randint(0, 10000, (batch_size, seq_len))
        labels = torch.randint(0, 10000, (batch_size, seq_len))

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        logits = model(input_ids)
        loss = (
            F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            * 100
        )

        optimizer.zero_grad()
        loss.backward()

        # Use value clipping function
        num_clipped = clip_grad_value(
            list(model.parameters()), clip_value=args.max_value
        )

        optimizer.step()

        print(f"  Step {step + 1}: Clipped {num_clipped} parameters")


def demonstrate_adaptive_clipping(model: nn.Module, args: argparse.Namespace):
    """Demonstrate adaptive gradient clipping."""
    print("\n" + "=" * 60)
    print("ADAPTIVE GRADIENT CLIPPING")
    print("=" * 60)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create adaptive gradient clipper
    clipper = GradientClipper(
        max_norm=10.0,  # High default threshold
        clip_type=ClipType.ADAPTIVE,
        log_stats=True,
    )

    # Training loop with varying gradient magnitudes
    for step in range(args.steps):
        # Generate data with varying difficulty
        batch_size = args.batch_size
        seq_len = 128

        # Alternate between easy and hard examples
        if step % 2 == 0:
            # Easy examples (small loss)
            input_ids = torch.randint(0, 100, (batch_size, seq_len))
            labels = input_ids.clone()  # Perfect prediction task
        else:
            # Hard examples (large loss)
            input_ids = torch.randint(0, 10000, (batch_size, seq_len))
            labels = torch.randint(0, 10000, (batch_size, seq_len))

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        # Forward pass
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Adaptive clipping
        stats = clipper.clip_gradients(model)

        # Optimizer step
        optimizer.step()

        # Print statistics
        print(f"\nStep {step + 1} ({'Easy' if step % 2 == 0 else 'Hard'}):")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Mean gradient norm: {stats['mean_norm']:.4f}")
        print(f"  Std gradient norm: {stats['std_norm']:.4f}")
        print(f"  Adaptive threshold: {stats['adaptive_threshold']:.4f}")
        print(f"  Total norm: {stats['total_norm']:.4f}")
        print(f"  Clipped: {stats['clipped']}")


def demonstrate_distributed_clipping(model: nn.Module, args: argparse.Namespace):
    """Demonstrate gradient clipping in distributed training."""
    print("\n" + "=" * 60)
    print("DISTRIBUTED GRADIENT CLIPPING")
    print("=" * 60)

    rank, world_size = setup_distributed()

    if world_size == 1:
        print("Running in single-process mode (no distributed training)")
    else:
        print(f"Running on rank {rank} of {world_size} processes")

    # Wrap model in DDP if distributed
    if world_size > 1:
        model = DDP(model)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create gradient clipper with distributed support
    clipper = GradientClipper(
        max_norm=args.max_norm,
        clip_type=ClipType.NORM,
        log_stats=(rank == 0),  # Only log on rank 0
    )

    # Training loop
    for step in range(args.steps):
        # Generate data (different for each rank)
        batch_size = args.batch_size
        seq_len = 128

        # Use rank as seed for different data per process
        torch.manual_seed(step * world_size + rank)
        input_ids = torch.randint(0, 10000, (batch_size, seq_len))
        labels = torch.randint(0, 10000, (batch_size, seq_len))

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        # Forward pass
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Get base model for clipping
        base_model = model.module if isinstance(model, DDP) else model

        # Clip gradients (norms will be synchronized across ranks)
        stats = clipper.clip_gradients(base_model)

        # Optimizer step
        optimizer.step()

        # Print statistics (only on rank 0)
        if rank == 0:
            print(f"\nStep {step + 1}:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Gradient norm: {stats['total_norm']:.4f}")
            print(f"  Clipped: {stats['clipped']}")

    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()


def demonstrate_gradient_monitoring(model: nn.Module, args: argparse.Namespace):
    """Demonstrate gradient monitoring and statistics collection."""
    print("\n" + "=" * 60)
    print("GRADIENT MONITORING AND STATISTICS")
    print("=" * 60)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create gradient clipper with monitoring
    clipper = GradientClipper(
        max_norm=args.max_norm,
        clip_type=ClipType.NORM,
        check_for_nan_in_grad=True,  # Enable NaN checking
        log_stats=False,  # We'll handle logging manually
    )

    # Collect statistics over time
    history: Dict[str, list] = {
        "step": [],
        "loss": [],
        "grad_norm": [],
        "clipped": [],
        "clip_coef": [],
    }

    # Training loop
    for step in range(args.steps):
        # Generate data
        batch_size = args.batch_size
        seq_len = 128
        input_ids = torch.randint(0, 10000, (batch_size, seq_len))
        labels = torch.randint(0, 10000, (batch_size, seq_len))

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        # Forward pass
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip and collect statistics
        try:
            stats = clipper.clip_gradients(model)

            # Record statistics
            history["step"].append(step + 1)
            history["loss"].append(loss.item())
            history["grad_norm"].append(stats["total_norm"])
            history["clipped"].append(stats["clipped"])
            history["clip_coef"].append(stats.get("clip_coef", 1.0))

        except ValueError as e:
            print(f"  Warning at step {step + 1}: {e}")
            print("  Skipping this step due to NaN gradients")
            continue

        # Optimizer step
        optimizer.step()

    # Print summary statistics
    print("\nTraining Summary:")
    print(f"  Total steps: {len(history['step'])}")
    print(f"  Average loss: {sum(history['loss'])/len(history['loss']):.4f}")
    print(
        f"  Average gradient norm: {sum(history['grad_norm'])/len(history['grad_norm']):.4f}"
    )
    print(f"  Max gradient norm: {max(history['grad_norm']):.4f}")
    print(f"  Min gradient norm: {min(history['grad_norm']):.4f}")
    print(f"  Steps clipped: {sum(history['clipped'])}/{len(history['clipped'])}")

    if any(history["clipped"]):
        clipped_coefs = [
            c for c, clipped in zip(history["clip_coef"], history["clipped"]) if clipped
        ]
        print(
            f"  Average clip coefficient (when clipped): {sum(clipped_coefs)/len(clipped_coefs):.4f}"
        )

    # Show gradient norm progression
    print("\nGradient Norm Progression:")
    for i in range(0, len(history["step"]), max(1, len(history["step"]) // 5)):
        step = history["step"][i]
        norm = history["grad_norm"][i]
        clipped = "CLIPPED" if history["clipped"][i] else ""
        print(f"  Step {step:3d}: {norm:8.4f} {clipped}")


def main():
    """Main function to run gradient clipping examples."""
    parser = argparse.ArgumentParser(description="Gradient Clipping Examples")
    parser.add_argument(
        "--demo",
        type=str,
        default="all",
        choices=["all", "norm", "value", "adaptive", "distributed", "monitoring"],
        help="Which demonstration to run",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--max-norm", type=float, default=1.0, help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=0.5,
        help="Maximum gradient value for value clipping",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Model size to use",
    )

    args = parser.parse_args()

    # Create model based on size
    model_configs = {
        "small": {"d_model": 256, "n_layers": 2, "n_heads": 4},
        "medium": {"d_model": 512, "n_layers": 4, "n_heads": 8},
        "large": {"d_model": 768, "n_layers": 6, "n_heads": 12},
    }

    config = model_configs[args.model_size]
    model = SimpleTransformer(
        vocab_size=10000,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
    )

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    print(
        f"Created {args.model_size} model with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters"
    )
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Run demonstrations
    if args.demo == "all":
        demonstrate_norm_clipping(model, args)
        demonstrate_value_clipping(model, args)
        demonstrate_adaptive_clipping(model, args)
        demonstrate_gradient_monitoring(model, args)
        if torch.cuda.is_available():
            demonstrate_distributed_clipping(model, args)
    elif args.demo == "norm":
        demonstrate_norm_clipping(model, args)
    elif args.demo == "value":
        demonstrate_value_clipping(model, args)
    elif args.demo == "adaptive":
        demonstrate_adaptive_clipping(model, args)
    elif args.demo == "distributed":
        demonstrate_distributed_clipping(model, args)
    elif args.demo == "monitoring":
        demonstrate_gradient_monitoring(model, args)

    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
