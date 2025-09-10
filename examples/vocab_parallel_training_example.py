#!/usr/bin/env python3
"""
End-to-End Vocabulary Parallel Training Example

This example demonstrates how to use vocabulary parallel cross-entropy loss
for training large language models with tensor parallelism. It shows:

1. Setting up tensor parallel groups
2. Creating a model with vocabulary parallelism
3. Using vocab parallel cross-entropy for efficient training
4. Comparing memory usage and accuracy with standard approach

Usage:
    # Single GPU (TP=1, baseline)
    python vocab_parallel_training_example.py

    # Multi-GPU with tensor parallelism
    torchrun --nproc_per_node=2 vocab_parallel_training_example.py --tp-size 2

    # With label smoothing
    python vocab_parallel_training_example.py --label-smoothing 0.1
"""

import argparse
import os
import time
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from rosellm.rosetrainer.parallelism import (
    destroy_model_parallel,
    get_data_parallel_group,
    get_data_parallel_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
    initialize_model_parallel,
)
from rosellm.rosetrainer.tensor_parallel.vocab_parallel_cross_entropy import (
    VocabParallelCrossEntropyLoss,
    vocab_parallel_cross_entropy,
)


class VocabParallelEmbedding(nn.Module):
    """Vocabulary parallel embedding layer."""

    def __init__(self, vocab_size: int, hidden_size: int, tp_size: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tp_size = tp_size

        # Calculate partition size
        assert (
            vocab_size % tp_size == 0
        ), f"Vocab size {vocab_size} must be divisible by TP size {tp_size}"
        self.vocab_size_per_partition = vocab_size // tp_size

        # Get TP rank to determine vocab range
        self.tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        self.vocab_start_index = self.tp_rank * self.vocab_size_per_partition
        self.vocab_end_index = (self.tp_rank + 1) * self.vocab_size_per_partition

        # Create embedding for this partition only
        self.embedding = nn.Embedding(self.vocab_size_per_partition, hidden_size)

        # Initialize weights
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with vocabulary parallel embedding."""
        # Create mask for tokens in this partition
        partition_mask = (input_ids >= self.vocab_start_index) & (
            input_ids < self.vocab_end_index
        )

        # Adjust indices to partition-local
        masked_input = input_ids.clone()
        masked_input[partition_mask] -= self.vocab_start_index
        masked_input[~partition_mask] = 0  # Use index 0 for out-of-partition tokens

        # Get embeddings
        embeddings = self.embedding(masked_input)

        # Zero out embeddings for out-of-partition tokens
        embeddings = embeddings * partition_mask.unsqueeze(-1).float()

        # All-reduce across TP group to combine embeddings
        if self.tp_size > 1:
            dist.all_reduce(embeddings, group=get_tensor_model_parallel_group())

        return embeddings  # type: ignore[no-any-return]


class VocabParallelOutput(nn.Module):
    """Vocabulary parallel output projection layer."""

    def __init__(self, hidden_size: int, vocab_size: int, tp_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tp_size = tp_size

        # Calculate partition size
        assert (
            vocab_size % tp_size == 0
        ), f"Vocab size {vocab_size} must be divisible by TP size {tp_size}"
        self.vocab_size_per_partition = vocab_size // tp_size

        # Create output projection for this partition only
        self.output_projection = nn.Linear(
            hidden_size, self.vocab_size_per_partition, bias=False
        )

        # Initialize weights
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with vocabulary parallel output."""
        # Project to vocab partition
        logits = self.output_projection(hidden_states)
        return logits  # type: ignore[no-any-return]


class SimpleTransformerLM(nn.Module):
    """Simple transformer language model with vocabulary parallelism."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        tp_size: int = 1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tp_size = tp_size

        # Vocabulary parallel embedding
        self.embedding = VocabParallelEmbedding(vocab_size, hidden_size, tp_size)

        # Transformer layers (simplified - could also be parallelized)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    batch_first=False,  # Use sequence-first format
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(hidden_size)

        # Vocabulary parallel output
        self.output = VocabParallelOutput(hidden_size, vocab_size, tp_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Get embeddings
        hidden_states = self.embedding(input_ids)

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Final layer norm
        hidden_states = self.ln_final(hidden_states)

        # Get logits (vocab parallel)
        logits = self.output(hidden_states)

        return logits  # type: ignore[no-any-return]


def create_synthetic_data(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic training data."""
    input_ids = torch.randint(0, vocab_size, (seq_length, batch_size), device=device)
    labels = torch.randint(0, vocab_size, (seq_length, batch_size), device=device)
    return input_ids, labels


def train_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    log_interval: int = 10,
    step: int = 0,
) -> float:
    """Single training step."""
    # Forward pass
    logits = model(input_ids)
    loss = loss_fn(logits, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step
    optimizer.step()

    # Logging
    if step % log_interval == 0:
        loss_value = loss.item()
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"Step {step}: Loss = {loss_value:.4f}")
        return loss_value  # type: ignore[no-any-return]

    return float(loss.item())


def compare_memory_usage(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    device: torch.device,
) -> dict:
    """Compare memory usage between vocab parallel and standard approach."""
    results = {}

    # Measure memory with vocab parallel
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    input_ids, labels = create_synthetic_data(
        batch_size, seq_length, vocab_size, device
    )
    logits = model(input_ids)
    loss = vocab_parallel_cross_entropy(logits, labels)
    loss.backward()

    torch.cuda.synchronize(device)
    vocab_parallel_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    results["vocab_parallel_mb"] = vocab_parallel_memory

    # For comparison, calculate theoretical standard memory
    # Standard would need full vocab_size logits: [seq_len, batch, vocab_size]
    standard_logits_memory = (
        seq_length * batch_size * vocab_size * 4
    ) / 1024**2  # float32 in MB
    results["standard_logits_mb"] = standard_logits_memory

    # Memory savings
    tp_size = get_tensor_model_parallel_size()
    results["memory_reduction_factor"] = tp_size
    results["saved_memory_mb"] = standard_logits_memory * (1 - 1 / tp_size)

    return results


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Vocabulary Parallel Training Example")
    parser.add_argument("--vocab-size", type=int, default=50256, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size")
    parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument(
        "--num-steps", type=int, default=100, help="Number of training steps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="Label smoothing factor"
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Initialize distributed if running with torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize model parallelism
    initialize_model_parallel(tensor_model_parallel_size=args.tp_size)

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_size()

    if tp_rank == 0:
        print(f"\n{'='*60}")
        print(f"Vocabulary Parallel Training Example")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Vocab Size: {args.vocab_size}")
        print(f"  Hidden Size: {args.hidden_size}")
        print(f"  Num Layers: {args.num_layers}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Sequence Length: {args.seq_length}")
        print(f"  Tensor Parallel Size: {tp_size}")
        print(f"  Label Smoothing: {args.label_smoothing}")
        print(f"  Device: {device}")
        print(f"{'='*60}\n")

    # Adjust vocab size to be divisible by TP size
    if args.vocab_size % tp_size != 0:
        new_vocab_size = ((args.vocab_size + tp_size - 1) // tp_size) * tp_size
        if tp_rank == 0:
            print(
                f"Adjusting vocab size from {args.vocab_size} to {new_vocab_size} for TP={tp_size}"
            )
        args.vocab_size = new_vocab_size

    # Create model
    model = SimpleTransformerLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        tp_size=tp_size,
    ).to(device)

    # Wrap with DDP if using data parallelism
    if get_data_parallel_size() > 1:
        model = DDP(model, process_group=get_data_parallel_group())

    # Create loss function
    loss_fn = VocabParallelCrossEntropyLoss(
        label_smoothing=args.label_smoothing, reduction="mean"
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Memory usage comparison
    if torch.cuda.is_available() and tp_rank == 0:
        print("\nMemory Usage Analysis:")
        print("-" * 40)
        memory_stats = compare_memory_usage(
            model, args.batch_size, args.seq_length, args.vocab_size, device
        )
        print(f"  Vocab Parallel Memory: {memory_stats['vocab_parallel_mb']:.2f} MB")
        print(
            f"  Standard Logits Memory (theoretical): {memory_stats['standard_logits_mb']:.2f} MB"
        )
        print(f"  Memory Reduction Factor: {memory_stats['memory_reduction_factor']}x")
        print(f"  Memory Saved: {memory_stats['saved_memory_mb']:.2f} MB")
        print("-" * 40)

    # Training loop
    if tp_rank == 0:
        print("\nStarting training...")
        print("-" * 40)

    start_time = time.time()
    losses = []

    for step in range(args.num_steps):
        # Generate synthetic data
        input_ids, labels = create_synthetic_data(
            args.batch_size, args.seq_length, args.vocab_size, device
        )

        # Training step
        loss = train_step(
            model, input_ids, labels, loss_fn, optimizer, log_interval=10, step=step
        )
        losses.append(loss)

    end_time = time.time()
    training_time = end_time - start_time

    # Print summary
    if tp_rank == 0:
        print("-" * 40)
        print(f"\nTraining Summary:")
        print(f"  Total Time: {training_time:.2f} seconds")
        print(f"  Steps/Second: {args.num_steps / training_time:.2f}")
        print(f"  Average Loss: {sum(losses) / len(losses):.4f}")
        print(f"  Final Loss: {losses[-1]:.4f}")

        # Demonstrate gradient flow
        print(f"\nGradient Flow Check:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
                if grad_norm == 0:
                    print(f"    WARNING: Zero gradient detected!")

        print(f"\n{'='*60}")
        print(f"Training completed successfully!")
        print(f"{'='*60}\n")

    # Cleanup
    destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
