#!/usr/bin/env python3
"""
ChainedOptimizer Example: Multi-Optimizer Training for MoE Models

This example demonstrates:
1. Creating separate optimizers for expert and dense parameters
2. Using ChainedOptimizer to manage multiple optimizers
3. Training with different learning rates for different parameter groups
4. State dict save/load for checkpointing
5. Integration with distributed training
"""

import argparse
import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from rosellm.rosetrainer.optimizer import Adam, ChainedOptimizer
from rosellm.rosetrainer.parallelism import initialize_model_parallel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpertLayer(nn.Module):
    """Simple expert layer for demonstration."""

    def __init__(self, hidden_size: int, num_experts: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(hidden_size, num_experts)

        # Mark expert parameters
        for expert in self.experts:
            for param in expert.parameters():
                param.allreduce = False  # Mark as expert parallel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple top-1 gating
        gate_scores = self.gate(x)
        expert_idx = gate_scores.argmax(dim=-1)

        # Route to experts (simplified)
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (expert_idx == i).unsqueeze(-1)
            if mask.any():
                expert_out = self.experts[i](x)
                output = output + mask * expert_out

        return output


class MoEModel(nn.Module):
    """Simple MoE model with expert and dense layers."""

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_experts: int = 8,
        moe_frequency: int = 2,  # MoE every 2 layers
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i % moe_frequency == 0:
                # MoE layer
                self.layers.append(ExpertLayer(hidden_size, num_experts))
            else:
                # Dense layer
                self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x) + x  # Residual connection

        return self.output(x)


def create_chained_optimizer_for_moe(
    model: nn.Module,
    base_lr: float = 1e-3,
    expert_lr_multiplier: float = 0.1,
    weight_decay: float = 0.01,
) -> ChainedOptimizer:
    """
    Create ChainedOptimizer with separate optimizers for expert and dense parameters.

    Args:
        model: Model with expert and dense parameters
        base_lr: Learning rate for dense parameters
        expert_lr_multiplier: Multiplier for expert learning rate
        weight_decay: Weight decay coefficient

    Returns:
        ChainedOptimizer managing both optimizers
    """

    # Separate parameters
    expert_params = []
    dense_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter is marked as expert parallel
        is_expert = not getattr(param, "allreduce", True)

        if is_expert:
            expert_params.append(
                {
                    "params": param,
                    "name": name,
                    "is_expert_parallel": True,
                }
            )
        else:
            # Apply different weight decay for embeddings
            wd = 0.0 if "embedding" in name or "bias" in name else weight_decay
            dense_params.append(
                {
                    "params": param,
                    "name": name,
                    "weight_decay": wd,
                    "is_expert_parallel": False,
                }
            )

    optimizers = []

    # Dense optimizer (for non-expert parameters)
    if dense_params:
        dense_optimizer = Adam(
            dense_params,
            lr=base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
        optimizers.append(dense_optimizer)
        logger.info(f"Created dense optimizer with {len(dense_params)} param groups")

    # Expert optimizer (with different learning rate)
    if expert_params:
        expert_optimizer = Adam(
            expert_params,
            lr=base_lr * expert_lr_multiplier,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
        optimizers.append(expert_optimizer)
        logger.info(f"Created expert optimizer with {len(expert_params)} param groups")

    # Create ChainedOptimizer
    chained_optimizer = ChainedOptimizer(optimizers)

    return chained_optimizer


def train_step(
    model: nn.Module,
    optimizer: ChainedOptimizer,
    data: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Single training step."""

    # Forward pass
    output = model(data)
    loss = nn.functional.cross_entropy(
        output.view(-1, output.size(-1)), target.view(-1)
    )

    # Backward pass
    loss.backward()

    # Optimizer step (handles multiple optimizers)
    optimizer.step()
    optimizer.zero_grad()

    return float(loss.item())


def save_checkpoint(
    model: nn.Module,
    optimizer: ChainedOptimizer,
    epoch: int,
    loss: float,
    path: str,
):
    """Save model and optimizer checkpoint."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    # Only rank 0 saves
    if not dist.is_initialized() or dist.get_rank() == 0:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: ChainedOptimizer,
    path: str,
) -> Tuple[int, float]:
    """Load model and optimizer checkpoint."""

    if os.path.exists(path):
        checkpoint = torch.load(path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        logger.info(f"Loaded checkpoint from {path} (epoch {epoch}, loss {loss:.4f})")
        return epoch, loss

    return 0, float("inf")


def main():
    parser = argparse.ArgumentParser(description="ChainedOptimizer Example")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--expert-lr-mult", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    args = parser.parse_args()

    # Initialize distributed if requested
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # Initialize model parallel groups
        initialize_model_parallel(
            expert_parallel_size=args.expert_parallel_size,
        )

    # Create model
    model = MoEModel(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
    )

    if args.distributed:
        model = model.cuda()
        model = DDP(model)

    # Create ChainedOptimizer
    optimizer = create_chained_optimizer_for_moe(
        model,
        base_lr=args.base_lr,
        expert_lr_multiplier=args.expert_lr_mult,
    )

    # Load checkpoint if exists
    checkpoint_path = os.path.join(args.checkpoint_dir, "moe_checkpoint.pt")
    start_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_path)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0
        num_steps = 100  # Simplified for example

        for step in range(num_steps):
            # Generate synthetic data
            data = torch.randint(0, 50257, (args.batch_size, args.seq_length))
            target = torch.randint(0, 50257, (args.batch_size, args.seq_length))

            if args.distributed:
                data = data.cuda()
                target = target.cuda()

            # Training step
            loss = train_step(model, optimizer, data, target)
            epoch_loss += loss

            if step % 10 == 0:
                logger.info(f"Epoch {epoch}, Step {step}/{num_steps}, Loss: {loss:.4f}")

        # Calculate average loss
        avg_loss = epoch_loss / num_steps
        logger.info(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")

        # Save checkpoint if improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)

    logger.info("Training completed!")

    # Demonstrate state dict handling
    logger.info("\nDemonstrating state dict operations:")

    # Get state dict from ChainedOptimizer
    state = optimizer.state_dict()
    logger.info(f"State dict keys: {list(state.keys())[:5]}...")  # Show first 5 keys

    # Create new optimizer and load state
    new_optimizer = create_chained_optimizer_for_moe(
        model,
        base_lr=args.base_lr,
        expert_lr_multiplier=args.expert_lr_mult,
    )
    new_optimizer.load_state_dict(state)
    logger.info("Successfully loaded state dict into new optimizer")

    # Verify param groups
    logger.info(f"\nOptimizer structure:")
    logger.info(f"Number of chained optimizers: {len(optimizer.chained_optimizers)}")
    for i, opt in enumerate(optimizer.chained_optimizers):
        num_params = sum(len(g["params"]) for g in opt.param_groups)
        logger.info(f"Optimizer {i}: {num_params} parameters")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
