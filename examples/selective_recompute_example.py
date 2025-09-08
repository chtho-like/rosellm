#!/usr/bin/env python3
"""
Example demonstrating selective activation recomputation in RoseLLM.

This example shows how to use different selection strategies to intelligently
choose which layers to checkpoint for optimal memory-compute tradeoffs.

Run with:
    python examples/selective_recompute_example.py

For distributed training:
    torchrun --nproc_per_node=2 examples/selective_recompute_example.py
"""

import argparse
import logging
import time
from typing import Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rosellm.rosetrainer.memory import (
    ActivationCheckpointing,
    SelectionStrategy,
    SelectiveCheckpointConfig,
    SelectiveRecomputeManager,
    create_selective_checkpoint_wrapper,
    selective_checkpoint,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Model Definition
# ============================================================================


class TransformerBlock(nn.Module):
    """A transformer block with self-attention and FFN."""

    def __init__(self, d_model: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # FFN
        ffn_out = self.linear2(self.activation(self.linear1(x)))
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)

        return x


class LanguageModel(nn.Module):
    """A simple language model for demonstration."""

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        num_layers: int = 12,
        nhead: int = 8,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, nhead) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)

        # Embedding + positional encoding
        x = self.embedding(x) * (self.d_model**0.5)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        x = self.norm(x)
        x = self.output(x)

        return x


# ============================================================================
# Training Functions
# ============================================================================


def train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Single training step."""
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = model(inputs)
    loss = F.cross_entropy(
        outputs.reshape(-1, outputs.size(-1)),
        targets.reshape(-1),
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    return float(loss.item())


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model performance."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1),
            )

            total_loss += float(loss.item())
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


# ============================================================================
# Selective Checkpointing Examples
# ============================================================================


def example_uniform_strategy(model: nn.Module) -> nn.Module:
    """Example: Uniform strategy - checkpoint every N layers."""
    logger.info("=" * 60)
    logger.info("Example 1: Uniform Selection Strategy")
    logger.info("Checkpointing every 3rd layer")
    logger.info("=" * 60)

    config = SelectiveCheckpointConfig(
        strategy=SelectionStrategy.UNIFORM,
        checkpoint_interval=3,
        verbose=True,
    )

    # Using ActivationCheckpointing with selective config
    checkpoint_manager = ActivationCheckpointing(selective_config=config)
    model = checkpoint_manager.apply_selective_checkpointing(model)

    return model


def example_memory_based_strategy(model: nn.Module) -> nn.Module:
    """Example: Memory-based strategy - checkpoint high-memory layers."""
    logger.info("=" * 60)
    logger.info("Example 2: Memory-Based Selection Strategy")
    logger.info("Checkpointing layers using >2MB memory")
    logger.info("=" * 60)

    config = SelectiveCheckpointConfig(
        strategy=SelectionStrategy.MEMORY_BASED,
        memory_threshold_mb=2.0,
        profile_enabled=True,
        verbose=True,
    )

    manager = SelectiveRecomputeManager(config)

    # Wrap model layers with selective checkpointing
    assert hasattr(model, "layers")
    layers = cast(nn.ModuleList, model.layers)
    for i, layer in enumerate(layers):
        original_forward = layer.forward
        layer.forward = (
            lambda x, orig=original_forward, idx=i: manager.checkpoint_layer(
                orig, x, layer_id=f"transformer_layer_{idx}"
            )
        )

    return model


def example_computation_based_strategy(model: nn.Module) -> nn.Module:
    """Example: Computation-based strategy - checkpoint expensive layers."""
    logger.info("=" * 60)
    logger.info("Example 3: Computation-Based Selection Strategy")
    logger.info("Checkpointing layers with >5ms computation time")
    logger.info("=" * 60)

    config = SelectiveCheckpointConfig(
        strategy=SelectionStrategy.COMPUTATION_BASED,
        computation_threshold_ms=5.0,
        recompute_factor=1.5,
        profile_enabled=True,
        verbose=True,
    )

    manager = SelectiveRecomputeManager(config)
    return manager.wrap_model(model)


def example_adaptive_strategy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> nn.Module:
    """Example: Adaptive strategy - dynamically select layers."""
    logger.info("=" * 60)
    logger.info("Example 4: Adaptive Selection Strategy")
    logger.info("Dynamically selecting top 50% memory-intensive layers")
    logger.info("=" * 60)

    config = SelectiveCheckpointConfig(
        strategy=SelectionStrategy.ADAPTIVE,
        profile_warmup_steps=5,
        profile_update_interval=10,
        adaptive_threshold_percentile=50.0,
        profile_enabled=True,
        verbose=True,
    )

    manager = SelectiveRecomputeManager(config)

    # Wrap model
    assert hasattr(model, "layers")
    layers = cast(nn.ModuleList, model.layers)
    for i, layer in enumerate(layers):
        original_forward = layer.forward
        layer.forward = (
            lambda x, orig=original_forward, idx=i: manager.checkpoint_layer(
                orig, x, layer_id=f"layer_{idx}"
            )
        )

    # Run warmup steps to gather profiling data
    logger.info("Running warmup steps for profiling...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step, batch in enumerate(dataloader):
        if step >= 10:
            break
        train_step(model, batch, optimizer, device)

    # Get profiling report
    report = manager.get_profiling_report()
    logger.info("Profiling Report:")
    logger.info(f"  Total layers: {report['total_layers']}")
    logger.info(f"  Checkpointed layers: {report['checkpointed_layers']}")
    logger.info(f"  Memory saved: {report.get('memory_saved_mb', 0):.2f} MB")

    return model


def example_manual_strategy(model: nn.Module) -> nn.Module:
    """Example: Manual strategy - explicitly specify layers."""
    logger.info("=" * 60)
    logger.info("Example 5: Manual Selection Strategy")
    logger.info("Checkpointing layers 0, 3, 6, 9")
    logger.info("=" * 60)

    config = SelectiveCheckpointConfig(
        strategy=SelectionStrategy.MANUAL,
        layers_to_checkpoint=[f"layer_{i}" for i in [0, 3, 6, 9]],
        verbose=True,
    )

    manager = SelectiveRecomputeManager(config)

    # Apply to specific layers
    assert hasattr(model, "layers")
    layers = cast(nn.ModuleList, model.layers)
    for i, layer in enumerate(layers):
        original_forward = layer.forward
        layer.forward = (
            lambda x, orig=original_forward, idx=i: manager.checkpoint_layer(
                orig, x, layer_id=f"layer_{idx}"
            )
        )

    return model


def example_hybrid_strategy(model: nn.Module) -> nn.Module:
    """Example: Hybrid strategy - combine memory and computation factors."""
    logger.info("=" * 60)
    logger.info("Example 6: Hybrid Selection Strategy")
    logger.info("Combining memory and computation costs")
    logger.info("=" * 60)

    config = SelectiveCheckpointConfig(
        strategy=SelectionStrategy.HYBRID,
        memory_threshold_mb=1.0,
        computation_threshold_ms=3.0,
        recompute_factor=2.0,
        adaptive_threshold_percentile=60.0,
        profile_enabled=True,
        verbose=True,
    )

    # Create a checkpoint wrapper for reuse
    checkpoint_wrapper = create_selective_checkpoint_wrapper(config)

    # Apply to model layers
    assert hasattr(model, "layers")
    layers = cast(nn.ModuleList, model.layers)
    for i, layer in enumerate(layers):
        original_forward = layer.forward
        layer.forward = lambda x, orig=original_forward, idx=i: checkpoint_wrapper(
            orig, x, layer_id=f"layer_{idx}"
        )

    return model


def example_function_checkpointing() -> None:
    """Example: Checkpointing individual functions."""
    logger.info("=" * 60)
    logger.info("Example 7: Function-Level Checkpointing")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define a compute-intensive function
    def expensive_computation(x: torch.Tensor) -> torch.Tensor:
        # Simulate expensive computation
        y = x
        for _ in range(10):
            y = torch.matmul(y, y.transpose(-1, -2))
            y = F.softmax(y, dim=-1)
        return y

    # Create input
    x = torch.randn(8, 64, 64, requires_grad=True, device=device)

    # Without checkpointing
    logger.info("Running without checkpointing...")
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    output1 = expensive_computation(x)
    loss1 = output1.sum()
    loss1.backward()

    time_without = time.time() - start_time
    if torch.cuda.is_available():
        mem_without = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        mem_without = 0

    logger.info(f"  Time: {time_without:.3f}s, Memory: {mem_without:.1f}MB")

    # With selective checkpointing
    logger.info("Running with selective checkpointing...")
    x.grad = None  # Reset gradient

    config = SelectiveCheckpointConfig(
        strategy=SelectionStrategy.MANUAL,
        layers_to_checkpoint=["expensive_func"],
    )

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    output2 = selective_checkpoint(
        expensive_computation,
        x,
        config=config,
        layer_id="expensive_func",
    )
    loss2 = output2.sum()
    loss2.backward()

    time_with = time.time() - start_time
    if torch.cuda.is_available():
        mem_with = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        mem_with = 0

    logger.info(f"  Time: {time_with:.3f}s, Memory: {mem_with:.1f}MB")

    if torch.cuda.is_available():
        logger.info(f"  Memory saved: {mem_without - mem_with:.1f}MB")
        logger.info(
            f"  Time overhead: {(time_with - time_without) / time_without * 100:.1f}%"
        )


# ============================================================================
# Main Function
# ============================================================================


def main():
    """Main function demonstrating selective recomputation."""
    parser = argparse.ArgumentParser(description="Selective Recomputation Example")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[
            "uniform",
            "memory",
            "computation",
            "adaptive",
            "manual",
            "hybrid",
            "all",
        ],
        default="all",
        help="Selection strategy to demonstrate",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of training steps",
    )

    args = parser.parse_args()

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Model configuration based on size
    model_configs = {
        "small": {"d_model": 256, "num_layers": 6, "nhead": 4},
        "medium": {"d_model": 512, "num_layers": 12, "nhead": 8},
        "large": {"d_model": 1024, "num_layers": 24, "nhead": 16},
    }

    config = model_configs[args.model_size]

    # Create synthetic dataset
    logger.info("Creating synthetic dataset...")
    vocab_size = 10000
    num_samples = 100

    input_ids = torch.randint(0, vocab_size, (num_samples, args.seq_len))
    target_ids = torch.randint(0, vocab_size, (num_samples, args.seq_len))

    dataset = TensorDataset(input_ids, target_ids)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Run examples based on strategy
    if args.strategy == "all":
        # Demonstrate function checkpointing
        example_function_checkpointing()

        # Demonstrate each strategy
        strategies = [
            "uniform",
            "memory",
            "computation",
            "adaptive",
            "manual",
            "hybrid",
        ]
    else:
        strategies = [args.strategy]

    for strategy in strategies:
        # Create fresh model for each strategy
        logger.info(f"\nCreating {args.model_size} model...")
        model = LanguageModel(
            vocab_size=vocab_size,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            nhead=config["nhead"],
            max_seq_len=args.seq_len,
        )
        model.to(device)

        # Apply selective checkpointing based on strategy
        if strategy == "uniform":
            model = example_uniform_strategy(model)
        elif strategy == "memory":
            model = example_memory_based_strategy(model)
        elif strategy == "computation":
            model = example_computation_based_strategy(model)
        elif strategy == "adaptive":
            model = example_adaptive_strategy(model, dataloader, device)
        elif strategy == "manual":
            model = example_manual_strategy(model)
        elif strategy == "hybrid":
            model = example_hybrid_strategy(model)

        # Training loop
        logger.info(f"\nTraining with {strategy} strategy...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        for step, batch in enumerate(dataloader):
            if step >= args.num_steps:
                break

            loss = train_step(model, batch, optimizer, device)

            if step % 5 == 0:
                logger.info(f"  Step {step}: Loss = {loss:.4f}")

        training_time = time.time() - start_time

        # Report memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(f"\nTraining completed:")
            logger.info(f"  Time: {training_time:.2f}s")
            logger.info(f"  Peak memory: {peak_memory:.2f}GB")
        else:
            logger.info(f"\nTraining completed in {training_time:.2f}s")

        # Evaluate model
        eval_loss = evaluate_model(model, dataloader, device)
        logger.info(f"  Evaluation loss: {eval_loss:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("Selective recomputation examples completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
