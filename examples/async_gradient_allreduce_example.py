#!/usr/bin/env python3
"""
Async Gradient Allreduce Example

This example demonstrates how to use the async gradient allreduce feature
in RoseLLM for distributed training. The async gradient allreduce overlaps
gradient computation with communication to improve training throughput.

Usage:
    # Single GPU/CPU (for testing)
    python async_gradient_allreduce_example.py
    # Multi-GPU distributed training
    torchrun --nproc_per_node=4 async_gradient_allreduce_example.py --distributed

    # With different strategies
    python async_gradient_allreduce_example.py --strategy bucketed
    python async_gradient_allreduce_example.py --strategy immediate
    python async_gradient_allreduce_example.py --strategy priority
"""

import argparse
import logging
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from rosellm.rosetrainer.parallelism import (
    AsyncAllreduceConfig,
    AsyncAllreduceStrategy,
    ColumnParallelLinear,
    RowParallelLinear,
    async_allreduce_step,
    destroy_async_allreduce,
    initialize_async_allreduce,
    initialize_model_parallel,
    register_parameter_for_async_allreduce,
    sync_async_allreduce,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleTransformerBlock(nn.Module):
    """A simple transformer block for demonstration."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Multi-head attention components
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_output = nn.Linear(hidden_size, hidden_size)

        # Feed-forward network
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)

        # Layer normalization
        self.attention_layernorm = nn.LayerNorm(hidden_size)
        self.output_layernorm = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        batch_size, seq_len, hidden_size = x.size()

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape for multi-head attention
        q = q.view(
            batch_size, seq_len, self.num_heads, hidden_size // self.num_heads
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_len, self.num_heads, hidden_size // self.num_heads
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_len, self.num_heads, hidden_size // self.num_heads
        ).transpose(1, 2)

        # Attention scores
        scores = (
            torch.matmul(q, k.transpose(-2, -1))
            / (hidden_size // self.num_heads) ** 0.5
        )
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        # Reshape back
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, hidden_size)
        )

        # Attention output projection
        attention_output = self.attention_output(attention_output)
        attention_output = self.dropout(attention_output)

        # Add & norm
        x = self.attention_layernorm(x + attention_output)

        # Feed-forward
        intermediate_output = torch.relu(self.intermediate(x))
        output = self.output(intermediate_output)
        output = self.dropout(output)

        # Add & norm
        output = self.output_layernorm(x + output)

        return output


class ParallelTransformerBlock(nn.Module):
    """A transformer block using tensor parallel linear layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int = 8,
        tp_size: int = 2,
        tp_rank: int = 0,
        tp_group: Optional[dist.ProcessGroup] = None,
        enable_async_allreduce: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Multi-head attention with column parallelism
        self.query = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            enable_async_allreduce=enable_async_allreduce,
            layer_name="attention.query",
        )
        self.key = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            enable_async_allreduce=enable_async_allreduce,
            layer_name="attention.key",
        )
        self.value = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            enable_async_allreduce=enable_async_allreduce,
            layer_name="attention.value",
        )

        # Attention output with row parallelism
        self.attention_output = RowParallelLinear(
            hidden_size,
            hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            enable_async_allreduce=enable_async_allreduce,
            layer_name="attention.output",
        )

        # Feed-forward network with tensor parallelism
        self.intermediate = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            enable_async_allreduce=enable_async_allreduce,
            layer_name="ffn.intermediate",
        )
        self.output = RowParallelLinear(
            intermediate_size,
            hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            enable_async_allreduce=enable_async_allreduce,
            layer_name="ffn.output",
        )

        # Layer normalization (not parallelized)
        self.attention_layernorm = nn.LayerNorm(hidden_size)
        self.output_layernorm = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified forward pass for demonstration
        # In a real implementation, you'd need to handle the tensor parallel attention properly

        # Self-attention (simplified)
        q = self.query(x)  # Column parallel
        k = self.key(x)  # Column parallel
        v = self.value(x)  # Column parallel

        # For simplicity, we'll use a basic attention mechanism
        # In practice, you'd implement proper multi-head attention with tensor parallelism
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.hidden_size**0.5
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        attention_output = self.attention_output(attention_output)  # Row parallel
        attention_output = self.dropout(attention_output)

        # Add & norm
        x = self.attention_layernorm(x + attention_output)

        # Feed-forward
        intermediate_output = torch.relu(self.intermediate(x))  # Column parallel
        output = self.output(intermediate_output)  # Row parallel
        output = self.dropout(output)

        # Add & norm
        output = self.output_layernorm(x + output)

        return output


class SimpleModel(nn.Module):
    """Simple model for training with async gradient allreduce."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int = 8,
        max_seq_len: int = 512,
        use_parallel: bool = False,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group: Optional[dist.ProcessGroup] = None,
        enable_async_allreduce: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_parallel = use_parallel

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if use_parallel:
                layer = ParallelTransformerBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    tp_group=tp_group,
                    enable_async_allreduce=enable_async_allreduce,
                )
            else:
                layer = SimpleTransformerBlock(
                    hidden_size, intermediate_size, num_heads
                )
            self.layers.append(layer)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        # Layer normalization
        self.final_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)

        # Position embeddings
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        position_embeds = self.position_embedding(positions)

        # Combined embeddings
        x = token_embeds + position_embeds

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        x = self.final_layernorm(x)

        # Output projection
        logits = self.output_layer(x)

        return logits


def create_synthetic_dataset(
    vocab_size: int,
    seq_len: int,
    num_samples: int,
    batch_size: int,
) -> DataLoader:
    """Create a synthetic dataset for training."""
    # Generate random input sequences
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))

    # For language modeling, targets are input_ids shifted by 1
    targets = torch.roll(input_ids, -1, dims=1)
    targets[:, -1] = 0  # Set last token to 0 (padding)

    dataset = TensorDataset(input_ids, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def setup_distributed() -> Dict[str, Any]:
    """Setup distributed training environment."""
    if not dist.is_initialized():
        # Initialize distributed process group
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

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


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    use_async_allreduce: bool = True,
    log_interval: int = 10,
) -> Dict[str, Any]:
    """Train the model with optional async gradient allreduce."""

    model.train()
    total_loss = 0.0
    total_steps = 0
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids)

            # Compute loss (flatten for cross entropy)
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

            # Backward pass
            loss.backward()

            # Async allreduce step (if using async)
            if use_async_allreduce:
                async_allreduce_step()

            # Optimizer step
            optimizer.step()

            # Update statistics
            total_loss += loss.item()
            epoch_loss += loss.item()
            total_steps += 1
            num_batches += 1

            # Log progress
            if batch_idx % log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        avg_epoch_loss = epoch_loss / num_batches
        logger.info(
            f"Epoch {epoch+1} completed in {epoch_time:.2f}s, "
            f"Average Loss: {avg_epoch_loss:.4f}"
        )

    # Final synchronization
    if use_async_allreduce:
        sync_async_allreduce()

    avg_loss = total_loss / total_steps
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    return {
        "avg_loss": avg_loss,
        "total_steps": total_steps,
        "avg_epoch_time": avg_epoch_time,
        "epoch_times": epoch_times,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Async Gradient Allreduce Example")

    # Model configuration
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden dimension")
    parser.add_argument(
        "--intermediate-size", type=int, default=2048, help="Intermediate dimension"
    )
    parser.add_argument(
        "--num-layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")

    # Training configuration
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of training samples"
    )

    # Distributed configuration
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )
    parser.add_argument(
        "--use-parallel-layers", action="store_true", help="Use tensor parallel layers"
    )
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size")

    # Async allreduce configuration
    parser.add_argument(
        "--disable-async", action="store_true", help="Disable async gradient allreduce"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="bucketed",
        choices=["immediate", "bucketed", "layerwise", "priority"],
        help="Async allreduce strategy",
    )
    parser.add_argument(
        "--bucket-size", type=int, default=25 * 1024 * 1024, help="Bucket size in bytes"
    )
    parser.add_argument(
        "--max-buckets", type=int, default=4, help="Maximum number of buckets"
    )
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps")
    parser.add_argument(
        "--log-comm-stats", action="store_true", help="Log communication statistics"
    )

    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")

    args = parser.parse_args()

    # Setup device and distributed training
    if args.distributed:
        dist_info = setup_distributed()
        device = dist_info["device"]
        world_size = dist_info["world_size"]
        rank = dist_info["rank"]

        logger.info(
            f"Initialized distributed training: rank={rank}, world_size={world_size}"
        )

        # Initialize model parallel if using parallel layers
        if args.use_parallel_layers:
            initialize_model_parallel(
                tensor_model_parallel_size=args.tp_size,
                pipeline_model_parallel_size=1,
                data_parallel_size=world_size // args.tp_size,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0

    logger.info(f"Using device: {device}")

    # Configure async gradient allreduce
    use_async_allreduce = not args.disable_async and world_size > 1

    if use_async_allreduce:
        # Map strategy string to enum
        strategy_map = {
            "immediate": AsyncAllreduceStrategy.IMMEDIATE,
            "bucketed": AsyncAllreduceStrategy.BUCKETED,
            "layerwise": AsyncAllreduceStrategy.LAYERWISE,
            "priority": AsyncAllreduceStrategy.PRIORITY_BASED,
        }

        async_config = AsyncAllreduceConfig(
            enabled=True,
            strategy=strategy_map[args.strategy],
            bucket_size=args.bucket_size,
            max_buckets=args.max_buckets,
            warmup_steps=args.warmup_steps,
            log_communication_stats=args.log_comm_stats,
            # Set priority layers for priority strategy
            priority_layers=["attention.query", "attention.output"]
            if args.strategy == "priority"
            else None,
        )

        # Initialize async allreduce manager
        async_manager = initialize_async_allreduce(async_config)
        logger.info(
            f"Initialized async gradient allreduce with strategy: {args.strategy}"
        )
    else:
        logger.info("Async gradient allreduce disabled")

    # Create model
    if args.use_parallel_layers and args.distributed:
        from rosellm.rosetrainer.parallelism import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_rank,
        )

        tp_group = get_tensor_model_parallel_group()
        tp_rank = get_tensor_model_parallel_rank()

        if tp_group is None:
            raise RuntimeError("Failed to get tensor model parallel group")

        model = SimpleModel(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_seq_len=args.seq_len,
            use_parallel=True,
            tp_size=args.tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            enable_async_allreduce=use_async_allreduce,
        )
    else:
        model = SimpleModel(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_seq_len=args.seq_len,
            use_parallel=False,
        )

        # Register parameters for async allreduce if enabled
        if use_async_allreduce:
            for name, param in model.named_parameters():
                register_parameter_for_async_allreduce(param, name)

    model = model.to(device)

    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)"
    )

    # Create dataset and dataloader
    dataloader = create_synthetic_dataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )

    logger.info(f"Starting training for {args.num_epochs} epochs...")

    # Train model
    start_time = time.time()
    training_stats = train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        use_async_allreduce=use_async_allreduce,
        log_interval=args.log_interval,
    )
    total_training_time = time.time() - start_time

    # Log final results
    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info(f"Total training time: {total_training_time:.2f}s")
    logger.info(f"Average loss: {training_stats['avg_loss']:.4f}")
    logger.info(f"Average epoch time: {training_stats['avg_epoch_time']:.2f}s")
    logger.info(f"Total steps: {training_stats['total_steps']}")

    # Log async allreduce statistics if enabled
    if use_async_allreduce and async_manager is not None:
        async_stats = async_manager.get_statistics()
        logger.info("Async Allreduce Statistics:")
        logger.info(f"  Strategy: {async_stats['config'].strategy.value}")
        logger.info(f"  Step count: {async_stats['step_count']}")
        logger.info(f"  World size: {async_stats['world_size']}")
        logger.info(f"  Buckets used: {async_stats['num_buckets']}")

        if "avg_comm_time" in async_stats:
            logger.info(
                f"  Avg communication time: {async_stats['avg_comm_time']:.4f}s"
            )
            logger.info(
                f"  Min communication time: {async_stats['min_comm_time']:.4f}s"
            )
            logger.info(
                f"  Max communication time: {async_stats['max_comm_time']:.4f}s"
            )

    # Cleanup
    if use_async_allreduce:
        destroy_async_allreduce()

    if args.distributed:
        dist.destroy_process_group()

    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
