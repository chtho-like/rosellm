#!/usr/bin/env python3
r"""
Advanced Sequence Parallel Training Example for RoseLLM

This example demonstrates production-ready sequence parallelism implementation
following Megatron-LM design patterns. Sequence parallelism distributes activations
along the sequence dimension to reduce memory footprint while maintaining
computational efficiency.

Key Features Demonstrated:
- Automatic sequence dimension distribution across tensor parallel ranks
- Memory-efficient LayerNorm and attention implementations
- Integration with mixed precision training
- Performance benchmarking and memory profiling
- Configuration management for optimization
- Error handling and validation

Usage Examples:
    # Single GPU (no parallelism, for debugging)
    python sequence_parallel_example.py

    # 2 GPUs with sequence parallelism enabled
    torchrun --nproc_per_node=2 sequence_parallel_example.py --enable-sp

    # 4 GPUs with TP=2, DP=2, and sequence parallelism
    torchrun --nproc_per_node=4 sequence_parallel_example.py --tp-size 2 --enable-sp

    # CPU testing with 4 processes (for development)
    CUDA_VISIBLE_DEVICES="" torchrun --nproc_per_node=4 sequence_parallel_example.py

    # With performance profiling and debug mode
    torchrun --nproc_per_node=2 sequence_parallel_example.py \
        --enable-sp --profile --debug

Performance Tips:
- Use --enable-sp only with tensor parallelism (TP > 1)
- Larger sequence lengths benefit more from sequence parallelism
- Monitor memory reduction with --profile flag
- Use mixed precision (--mixed-precision) for additional memory savings
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from rosellm.rosetrainer.parallelism import (
    gather_from_sequence_parallel_region,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
    initialize_model_parallel,
    is_sequence_parallel_enabled,
    scatter_to_sequence_parallel_region,
)
from rosellm.rosetrainer.parallelism.sequence_parallel import (
    SequenceParallelBenchmark,
    SequenceParallelConfig,
    set_sequence_parallel_config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    loss: float
    throughput_samples_per_sec: float
    memory_allocated_gb: float
    memory_reserved_gb: float
    grad_norm: Optional[float] = None


class Timer:
    """Context manager for timing operations."""

    def __init__(
        self, name: str = "Operation", logger: Optional[logging.Logger] = None
    ):
        self.name = name
        self.logger = logger
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            if self.logger:
                self.logger.debug(f"{self.name} took {self.elapsed:.4f} seconds")


class SequenceParallelLayerNorm(nn.Module):
    """
    LayerNorm optimized for sequence-parallel tensors.

    This implementation operates directly on distributed tensors without
    requiring gather operations, maintaining memory efficiency.

    Input: [seq_len/TP, batch, hidden] (distributed)
    Output: [seq_len/TP, batch, hidden] (distributed)
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional mixed precision support.

        Args:
            x: Input tensor [seq_len/TP, batch, hidden]

        Returns:
            Normalized tensor with same shape as input
        """
        # Use Welford's algorithm for numerical stability
        orig_dtype = x.dtype
        x = x.float()  # Ensure float32 for stability

        # Compute mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)

        # Apply affine transformation if enabled
        if self.elementwise_affine:
            x = x * self.weight + self.bias

        return x.to(orig_dtype)


class SequenceParallelAttention(nn.Module):
    """
    Multi-head attention with sequence parallel support.

    Demonstrates how attention layers interact with sequence parallelism.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # QKV projection (operates on gathered sequence)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sequence_parallel: bool = True) -> torch.Tensor:
        """
        Forward pass with optional sequence parallel.

        Args:
            x: Input tensor [seq_len/TP, batch, hidden] if SP,
                else [seq_len, batch, hidden]
            sequence_parallel: Whether input is sequence parallel
        """
        # Gather sequence for QKV computation if needed
        if sequence_parallel and is_sequence_parallel_enabled():
            # Gather from sequence parallel region for attention computation
            x_gathered = gather_from_sequence_parallel_region(x)
        else:
            x_gathered = x

        batch_size, seq_len, _ = x_gathered.shape

        # Compute QKV
        qkv = self.qkv_proj(x_gathered)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.out_proj(attn_output)

        # Scatter back to sequence parallel if needed
        if sequence_parallel and is_sequence_parallel_enabled():
            output = scatter_to_sequence_parallel_region(output)

        return cast(torch.Tensor, output)


class SequenceParallelTransformerBlock(nn.Module):
    """
    Transformer block with sequence parallel support.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1
    ):
        super().__init__()

        # Layer normalization (operates on SP tensors)
        self.ln1 = SequenceParallelLayerNorm(hidden_size)
        self.ln2 = SequenceParallelLayerNorm(hidden_size)

        # Attention layer
        self.attention = SequenceParallelAttention(hidden_size, num_heads, dropout)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sequence_parallel: bool = True) -> torch.Tensor:
        """Forward pass with sequence parallel support."""
        # Attention block with residual
        attn_out = self.attention(self.ln1(x), sequence_parallel)
        x = x + self.dropout(attn_out)

        # MLP block with residual
        # Note: MLP operates on sequence parallel tensors directly
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out

        return x


class SimpleSequenceParallelModel(nn.Module):
    """
    Simple transformer model demonstrating sequence parallelism.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList(
            [
                SequenceParallelTransformerBlock(
                    hidden_size, num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = SequenceParallelLayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic sequence parallel distribution.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Generate position IDs
        position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        # Embeddings
        x = self.embedding(input_ids)
        x = x + self.pos_embedding(position_ids)
        x = self.dropout(x)

        # Scatter to sequence parallel region if enabled
        if is_sequence_parallel_enabled():
            x = scatter_to_sequence_parallel_region(x)
            sequence_parallel = True
        else:
            sequence_parallel = False

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, sequence_parallel=sequence_parallel)

        # Final layer norm
        x = self.ln_final(x)

        # Gather for output projection if using SP
        if sequence_parallel:
            x = gather_from_sequence_parallel_region(x)

        # Language model head
        logits = self.lm_head(x)

        return cast(torch.Tensor, logits)


def get_memory_stats() -> Dict[str, float]:
    """Get current GPU memory statistics.

    Returns:
        Dictionary with memory statistics in GB
    """
    if torch.cuda.is_available():
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
    return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}


def print_memory_stats(rank: int, stage: str, detailed: bool = False):
    """Print GPU memory statistics.

    Args:
        rank: Process rank
        stage: Description of current stage
        detailed: Whether to print detailed statistics
    """
    stats = get_memory_stats()

    if detailed:
        logger.info(
            f"[Rank {rank}] {stage} - "
            f"Allocated: {stats['allocated_gb']:.3f} GB, "
            f"Reserved: {stats['reserved_gb']:.3f} GB, "
            f"Max Allocated: {stats['max_allocated_gb']:.3f} GB"
        )
    else:
        logger.info(
            f"[Rank {rank}] {stage} - "
            f"Memory: {stats['allocated_gb']:.2f}/{stats['reserved_gb']:.2f} GB"
        )


def train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    rank: int,
    mixed_precision: bool = False,
    gradient_clip: float = 1.0,
) -> TrainingMetrics:
    """Single training step with mixed precision and gradient clipping support.

    Args:
        model: Model to train
        batch: Tuple of (input_ids, labels)
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        rank: Process rank
        mixed_precision: Whether to use mixed precision
        gradient_clip: Maximum gradient norm

    Returns:
        TrainingMetrics with loss and performance statistics
    """
    input_ids, labels = batch
    batch_size = input_ids.shape[0]

    # Time the forward and backward passes
    start_time = time.perf_counter()

    # Forward pass with optional mixed precision
    if mixed_precision and torch.cuda.is_available():
        with autocast():
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
    else:
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    # Backward pass
    if mixed_precision and scaler is not None:
        scaler.scale(loss).backward()

        # Unscale gradients for clipping
        scaler.unscale_(optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        # Optimizer step with scaling
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        # Optimizer step
        optimizer.step()

    optimizer.zero_grad()

    # Calculate throughput
    elapsed = time.perf_counter() - start_time
    throughput = batch_size / elapsed

    # Get memory stats
    memory_stats = get_memory_stats()

    return TrainingMetrics(
        loss=loss.item(),
        throughput_samples_per_sec=throughput,
        memory_allocated_gb=memory_stats["allocated_gb"],
        memory_reserved_gb=memory_stats["reserved_gb"],
        grad_norm=grad_norm.item()
        if isinstance(grad_norm, torch.Tensor)
        else grad_norm,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Sequence Parallel Training Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Parallelism configuration
    parallel_group = parser.add_argument_group("Parallelism Configuration")
    parallel_group.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallel size"
    )
    parallel_group.add_argument(
        "--pp-size", type=int, default=1, help="Pipeline parallel size"
    )
    parallel_group.add_argument(
        "--enable-sp",
        action="store_true",
        help="Enable sequence parallelism (requires TP>1)",
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--vocab-size", type=int, default=50257, help="Vocabulary size"
    )
    model_group.add_argument(
        "--hidden-size", type=int, default=768, help="Hidden dimension"
    )
    model_group.add_argument(
        "--num-layers", type=int, default=4, help="Number of transformer layers"
    )
    model_group.add_argument(
        "--num-heads", type=int, default=12, help="Number of attention heads"
    )
    model_group.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    model_group.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training configuration
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--batch-size", type=int, default=2, help="Batch size per rank"
    )
    training_group.add_argument(
        "--num-steps", type=int, default=10, help="Number of training steps"
    )
    training_group.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    training_group.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay"
    )
    training_group.add_argument(
        "--gradient-clip", type=float, default=1.0, help="Gradient clipping"
    )
    training_group.add_argument(
        "--mixed-precision", action="store_true", help="Enable mixed precision training"
    )

    # Performance and debugging
    perf_group = parser.add_argument_group("Performance and Debugging")
    perf_group.add_argument(
        "--profile", action="store_true", help="Enable performance profiling"
    )
    perf_group.add_argument(
        "--debug", action="store_true", help="Enable debug mode with detailed logging"
    )
    perf_group.add_argument(
        "--log-interval", type=int, default=5, help="Steps between logging"
    )
    perf_group.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )

    args = parser.parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    # Get distributed info from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    # Initialize distributed if multi-process
    if world_size > 1:
        dist.init_process_group(backend=backend)
    elif not dist.is_initialized():
        # For single GPU, initialize process group for convenience
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend=backend)

    # Configure sequence parallelism settings
    if args.enable_sp:
        sp_config = SequenceParallelConfig(
            enable_memory_profiling=args.profile,
            enable_communication_stats=args.profile,
            debug_mode=args.debug,
            optimize_memory=True,
            communication_overlap=False,  # Can be enabled for advanced users
            gradient_accumulation_fusion=False,
        )
        set_sequence_parallel_config(sp_config)
        logger.info(f"Sequence parallel config: {sp_config}")

    # Initialize model parallelism
    # Auto-calculate data parallel size
    dp_size = world_size // (args.tp_size * args.pp_size)

    # Enable sequence parallel only if TP > 1 and requested
    sequence_parallel_enabled = args.enable_sp and args.tp_size > 1

    if args.enable_sp and args.tp_size == 1:
        logger.warning(
            "Sequence parallelism requires tensor_model_parallel_size > 1. "
            "Continuing without sequence parallelism."
        )
        sequence_parallel_enabled = False

    initialize_model_parallel(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        data_parallel_size=dp_size,
        sequence_parallel_enabled=sequence_parallel_enabled,
    )

    # Print parallel configuration
    if local_rank == 0:
        print(f"Parallel Configuration:")
        print(f"  World Size: {world_size}")
        print(f"  Tensor Parallel: {args.tp_size}")
        print(f"  Pipeline Parallel: {args.pp_size}")
        print(f"  Data Parallel: {dp_size}")
        print(f"  Sequence Parallel: {sequence_parallel_enabled}")
        print(f"  Device: {device}")
        print(f"  Backend: {backend}")

    # Print rank information
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_size()
    sp_enabled = is_sequence_parallel_enabled()

    print(f"[Rank {local_rank}] TP Rank: {tp_rank}/{tp_size}, SP Enabled: {sp_enabled}")

    # Create model
    model = SimpleSequenceParallelModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
        dropout=0.1,
    ).to(device)

    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),  # GPT-3 settings
        eps=1e-8,
    )

    # Create gradient scaler for mixed precision
    scaler = (
        GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    )

    # Initialize benchmark if requested
    benchmark = SequenceParallelBenchmark() if args.benchmark else None

    # Print initial memory stats
    print_memory_stats(local_rank, "After model creation", detailed=args.profile)

    # Training loop
    for step in range(args.num_steps):
        # Generate random data
        input_ids = torch.randint(
            0, args.vocab_size, (args.batch_size, args.seq_len), device=device
        )
        labels = torch.randint(
            0, args.vocab_size, (args.batch_size, args.seq_len), device=device
        )

        # Adjust sequence length for SP if enabled
        if sp_enabled and tp_size > 1:
            # When using sequence parallel, each rank sees seq_len/TP
            expected_seq_len = args.seq_len // tp_size
            if local_rank == 0 and step == 0:
                print(
                    "Sequence parallel active: each rank processes "
                    f"{expected_seq_len} tokens"
                )

        # Train step with metrics
        metrics = train_step(
            model,
            (input_ids, labels),
            optimizer,
            scaler,
            local_rank,
            mixed_precision=args.mixed_precision,
            gradient_clip=args.gradient_clip,
        )

        # Benchmark sequence parallel operations if requested
        if args.benchmark and benchmark is not None and step == 0:
            # Benchmark scatter operation
            test_tensor = torch.randn(
                args.seq_len, args.batch_size, args.hidden_size, device=device
            )
            if sp_enabled:
                scatter_stats = benchmark.benchmark_operation(
                    scatter_to_sequence_parallel_region,
                    test_tensor,
                    "scatter_to_sp",
                    num_iterations=20,
                    warmup_iterations=5,
                )
                logger.info(f"Scatter benchmark: {scatter_stats}")

        # Print progress
        if step % args.log_interval == 0:
            grad_norm_str = f"{metrics.grad_norm:.3f}" if metrics.grad_norm else "N/A"
            logger.info(
                f"[Rank {local_rank}] Step {step}/{args.num_steps} - "
                f"Loss: {metrics.loss:.4f}, "
                f"Throughput: {metrics.throughput_samples_per_sec:.1f} samples/s, "
                f"Grad Norm: {grad_norm_str}, "
                f"Memory: {metrics.memory_allocated_gb:.2f} GB"
            )

            if step == 0 and args.profile:
                print_memory_stats(local_rank, "After first step", detailed=True)

    # Final memory stats and benchmark report
    if args.profile:
        print_memory_stats(local_rank, "After training", detailed=True)

        # Print memory reduction if using sequence parallelism
        if sp_enabled and local_rank == 0:
            memory_reduction = (1.0 - 1.0 / tp_size) * 100
            logger.info(
                f"\nSequence Parallelism Memory Savings:\n"
                f"  Theoretical activation memory reduction: {memory_reduction:.1f}%\n"
                f"  Effective TP size for activations: {tp_size}\n"
            )

    # Print benchmark report if available
    if benchmark is not None and local_rank == 0:
        report = benchmark.generate_report()
        logger.info(report)

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

    if local_rank == 0:
        logger.info("\n" + "=" * 50)
        logger.info("Training completed successfully!")

        if sp_enabled:
            logger.info(
                "Sequence parallelism was active - activation memory was "
                f"distributed across {tp_size} tensor parallel ranks."
            )
            logger.info(
                "Each rank processed sequence chunks of size "
                f"{args.seq_len // tp_size} tokens."
            )

        logger.info("=" * 50)


if __name__ == "__main__":
    main()
