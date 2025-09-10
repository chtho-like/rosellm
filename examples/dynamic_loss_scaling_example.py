#!/usr/bin/env python3
"""
Dynamic Loss Scaling Example for Stable FP16 Training

This example demonstrates how to use the DynamicGradientScaler for stable
mixed precision training with automatic overflow detection and recovery.

Key Features Demonstrated:
- Basic FP16 training setup with dynamic loss scaling
- Integration with gradient clipping for stable training
- Overflow detection and automatic scale adjustment
- Performance monitoring and debugging
- Advanced scaling strategies

Usage:
    python examples/dynamic_loss_scaling_example.py --help
    python examples/dynamic_loss_scaling_example.py --strategy adaptive --clip-norm 1.0
"""

import argparse
import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import RoseLLM dynamic gradient scaler
from rosellm.rosetrainer.utils import (
    DynamicGradientScaler,
    GradientClipConfig,
    ScalingStrategy,
    create_gradient_scaler,
    create_integrated_training_step,
)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class ExampleModel(nn.Module):
    """Example model for testing dynamic loss scaling."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 256, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model, nhead=8, dim_feedforward=512, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)


def create_synthetic_dataset(batch_size: int = 32, num_batches: int = 20) -> DataLoader:
    """Create synthetic dataset for training."""
    seq_len, vocab_size = 64, 1000
    input_ids = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len))
    labels = torch.roll(input_ids, -1, dims=1)

    dataset = TensorDataset(input_ids, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_with_basic_scaling(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    """Basic training example with dynamic loss scaling."""
    print("\n=== Basic Dynamic Loss Scaling Example ===")

    scaler = create_gradient_scaler(
        init_scale=2**16,
        scaling_strategy="exponential",
    )

    model.train()
    stats = {"successful_steps": 0, "skipped_steps": 0}

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        # Check for overflow
        found_overflow = False
        for param in model.parameters():
            if param.grad is not None and (
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            ):
                found_overflow = True
                break

        if not found_overflow:
            scaler.unscale_(optimizer)
            optimizer.step()
            stats["successful_steps"] += 1
        else:
            stats["skipped_steps"] += 1
            print(f"Skipped step {batch_idx} due to overflow")

        scaler.update()
        optimizer.zero_grad()

        if batch_idx % 5 == 0:
            print(
                f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                f"Scale: {scaler.get_scale():.0f}"
            )

    print(
        f"Training completed. Successful: {stats['successful_steps']}, "
        f"Skipped: {stats['skipped_steps']}"
    )
    return stats


def train_with_integrated_step(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    """Training with integrated step function."""
    print("\n=== Integrated Training Step Example ===")

    scaler = DynamicGradientScaler(
        init_scale=2**15,
        scaling_strategy=ScalingStrategy.ADAPTIVE,
        use_multi_tensor=True,
        device=device,
    )

    clip_config = GradientClipConfig(clip_type="norm", max_norm=1.0)

    training_step = create_integrated_training_step(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        clip_config=clip_config,
        accumulation_steps=2,
    )

    model.train()
    stats = {"total_steps": 0, "successful_steps": 0}

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        step_stats = training_step(loss)

        stats["total_steps"] += 1
        if step_stats["stepped"]:
            stats["successful_steps"] += 1

        if batch_idx % 5 == 0:
            print(
                f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                f"Scale: {step_stats.get('scale', 1.0):.0f}, "
                f"Stepped: {step_stats['stepped']}"
            )

    perf_stats = scaler.get_performance_stats()
    print(f"Performance: {perf_stats}")
    return stats


def demonstrate_overflow_recovery(device: torch.device) -> None:
    """Demonstrate automatic overflow recovery."""
    print("\n=== Overflow Recovery Demonstration ===")

    model = nn.Linear(10, 1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scaler = DynamicGradientScaler(
        init_scale=1000.0, hysteresis=1, growth_interval=5, device=device
    )

    print(f"Initial scale: {scaler.get_scale()}")

    # Normal steps
    for step in range(8):
        x, y = torch.randn(5, 10, device=device), torch.randn(5, 1, device=device)
        loss = F.mse_loss(model(x), y)
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        result = scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % 2 == 0:
            print(f"Step {step}: Scale={scaler.get_scale():.0f}")

    # Inject overflow
    print("Injecting overflow...")
    for param in model.parameters():
        param.grad = torch.full_like(param, float("inf"))

    result = scaler.step(optimizer)
    scaler.update()
    print(
        f"After overflow: Scale={scaler.get_scale():.0f}, Stepped={result is not None}"
    )

    # Recovery
    for step in range(10):
        x, y = torch.randn(5, 10, device=device), torch.randn(5, 1, device=device)
        loss = F.mse_loss(model(x), y)
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % 2 == 0:
            print(f"Recovery {step}: Scale={scaler.get_scale():.0f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Dynamic Loss Scaling Examples")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    logger.info(f"Using device: {device}")

    model = ExampleModel().to(device)
    if device.type == "cuda":
        model = model.half()

    dataloader = create_synthetic_dataset(batch_size=args.batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run examples
    try:
        train_with_basic_scaling(model, dataloader, optimizer, device)
        train_with_integrated_step(model, dataloader, optimizer, device)
        demonstrate_overflow_recovery(device)
        print("\n=== All examples completed successfully! ===")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
