"""Example demonstrating distributed optimizer with parameter partitioning.

This example shows how to use the DistributedOptimizer to reduce memory usage
by partitioning parameters, gradients, and optimizer states across data parallel ranks.

To run this example:
    # Single GPU (no distribution)
    python examples/distributed_optimizer_example.py

    # Multiple GPUs (distributed)
    torchrun --nproc_per_node=2 examples/distributed_optimizer_example.py

    # CPU simulation of distributed training
    CUDA_VISIBLE_DEVICES="" torchrun --nproc_per_node=4 \
        examples/distributed_optimizer_example.py
"""

import logging
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from rosellm.rosetrainer.optimizer import (
    DistributedOptimizer,
    MemoryProfiler,
    OptimizerFactory,
)
from rosellm.rosetrainer.parallelism import (
    get_data_parallel_group,
    initialize_model_parallel,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

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
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, hidden_size))
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embedding and positional encoding
        x = self.embedding(input_ids)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        return self.output_proj(x)


def setup_distributed() -> Tuple[int, int, Optional[dist.ProcessGroup]]:
    """Setup distributed training environment."""
    if "LOCAL_RANK" in os.environ:
        # Distributed training
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )

        # Initialize model parallel (data parallel only in this example)
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=world_size,
        )

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        process_group = get_data_parallel_group()

        return local_rank, world_size, process_group
    else:
        # Single process training
        return 0, 1, None


def demonstrate_factory_pattern(model, process_group):
    """Demonstrate using the factory pattern for optimizer creation."""
    # Method 1: Using presets
    optimizer_preset = OptimizerFactory.create_from_model(
        model,
        optimizer_name="AdamW",
        lr=1e-4,
        preset="memory_efficient",
        process_group=process_group,
    )

    # Method 2: Custom configuration (example, not used in this demo)
    # custom_config = DistributedOptimizerConfig(
    #     partition_parameters=True,
    #     mixed_precision=True,
    #     grad_clip_value=1.0,
    # )
    # optimizer_custom = OptimizerFactory.create(
    #     model.parameters(),
    #     torch.optim.AdamW,
    #     {"lr": 1e-4, "weight_decay": 0.01},
    #     config=custom_config,
    #     process_group=process_group,
    # )

    return optimizer_preset


def train_step(
    model: nn.Module,
    optimizer: DistributedOptimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    step: int,
) -> float:
    """Perform a single training step."""
    # Forward pass
    logits = model(input_ids)

    # Compute loss
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_ids.view(-1))

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Log memory usage periodically
    if step % 10 == 0:
        memory_stats = optimizer.get_memory_usage()
        logger.info(
            f"Step {step} - Loss: {loss.item():.4f}, "
            f"Memory: {memory_stats['total_mb']:.2f} MB "
            f"(params: {memory_stats['parameters_mb']:.2f}, "
            f"grads: {memory_stats['gradients_mb']:.2f}, "
            f"states: {memory_stats['optimizer_states_mb']:.2f})"
        )

    return float(loss.item())


def main():
    """Main training loop."""
    # Setup distributed training
    local_rank, world_size, process_group = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    logger.info(f"Rank {local_rank}/{world_size} using device: {device}")

    # Create model
    model = SimpleTransformer(
        vocab_size=10000, hidden_size=512, num_layers=6, num_heads=8
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {total_params:,} parameters")

    # Wrap model in DDP if distributed
    if world_size > 1:
        model = DDP(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None
        )

    # Initialize memory profiler
    memory_profiler = MemoryProfiler()
    memory_profiler.set_baseline()

    # Create optimizer using factory pattern
    preset_name = (
        "memory_efficient"  # Can use: baseline, speed_optimized, mixed_precision, etc.
    )
    logger.info(f"Using optimizer preset: {preset_name}")

    optimizer = OptimizerFactory.create_from_model(
        model,
        optimizer_name="AdamW",
        lr=1e-4,
        weight_decay=0.01,
        preset=preset_name,
        process_group=process_group,
    )

    # Get configuration details
    config = optimizer.config

    # Log configuration and memory analysis
    if local_rank == 0:
        logger.info(f"Optimizer configuration:")
        config_dict = OptimizerFactory.describe_preset(preset_name)
        for key, value in list(config_dict.items())[:10]:  # Show first 10 settings
            logger.info(f"  - {key}: {value}")

        # Memory analysis
        model_memory = memory_profiler.analyze_model_memory(model)
        optimizer_memory = memory_profiler.estimate_optimizer_memory(
            total_params, "AdamW", config.dtype
        )

        logger.info(f"Memory breakdown:")
        logger.info(f"  - Model parameters: {model_memory['parameters_mb']:.2f} MB")
        logger.info(f"  - Model gradients: {model_memory['gradients_mb']:.2f} MB")
        logger.info(f"  - Optimizer states: {optimizer_memory:.2f} MB")
        logger.info(
            f"  - Total estimated: {model_memory['total_mb'] + optimizer_memory:.2f} MB"
        )

    # Training loop
    num_steps = 50
    batch_size = 8
    seq_len = 128

    logger.info(f"Starting training for {num_steps} steps...")

    for step in range(num_steps):
        # Generate random data (in real training, use DataLoader)
        input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
        target_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)

        # Train step
        train_step(model, optimizer, input_ids, target_ids, step)

        # Memory profiling every 20 steps
        if step % 20 == 0 and local_rank == 0:
            memory_profiler.record_snapshot()
            if step == 20:  # First real measurement
                logger.info(memory_profiler.get_memory_summary())

    logger.info("Training completed!")

    # Final reports
    if local_rank == 0:
        # Optimizer memory report
        final_memory = optimizer.get_memory_usage()
        logger.info(f"Final optimizer memory usage:")
        for key, value in final_memory.items():
            logger.info(f"  - {key}: {value:.2f}")

        # System memory report
        logger.info("\nSystem memory analysis:")
        logger.info(memory_profiler.get_memory_summary())

        # Optimization recommendations
        recommendations = memory_profiler.optimize_memory()
        if recommendations:
            logger.info("\nMemory optimization recommendations:")
            for rec_type, rec_text in recommendations.items():
                logger.info(f"  [{rec_type}]: {rec_text}")

        # Optimizer statistics
        logger.info(f"\nOptimizer statistics:")
        logger.info(f"  - Total steps: {optimizer.step_count}")
        logger.info(f"  - Overflow count: {optimizer.overflow_count}")
        logger.info(f"  - State: {optimizer.optimizer_state.value}")

        # Cleanup
        memory_profiler.cleanup()

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
