#!/usr/bin/env python3
"""
Parameter Gathering Overlap Example

This example demonstrates how to use the parameter gathering overlap feature
in RoseLLM to hide communication latency in distributed training.

Usage:
    # Single GPU example
    python examples/parameter_overlap_example.py

    # Multi-GPU example (requires 2+ GPUs)
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/parameter_overlap_example.py

    # CPU multi-process example
    CUDA_VISIBLE_DEVICES="" torchrun --nproc_per_node=4 examples/parameter_overlap_example.py
"""

import argparse
import os
import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from rosellm.rosetrainer.memory.parameter_overlap import (
    AsyncParameterGatherer,
    OverlapConfig,
    OverlapMode,
)
from rosellm.rosetrainer.parallelism.overlap_integration import (
    OverlappedColumnParallelLinear,
    OverlappedRowParallelLinear,
    convert_to_overlapped_model,
)


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""

    def __init__(
        self, hidden_dim: int, ff_dim: int, device: Optional[torch.device] = None
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True, device=device
        )
        self.norm1 = nn.LayerNorm(hidden_dim, device=device)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim, device=device),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim, device=device),
        )
        self.norm2 = nn.LayerNorm(hidden_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class SimpleTransformer(nn.Module):
    """Simple transformer model for demonstration."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, device=device)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim, device=device)

        self.layers = nn.ModuleList(
            [
                SimpleTransformerBlock(hidden_dim, hidden_dim * 4, device)
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(hidden_dim, vocab_size, device=device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_emb = self.embedding(input_ids)
        pos_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        pos_emb = self.pos_embedding(pos_ids)
        x = token_emb + pos_emb

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        return self.output_proj(x)


def setup_distributed() -> Tuple[int, int, torch.device]:
    """Setup distributed training environment."""
    if "LOCAL_RANK" in os.environ:
        # Distributed setup
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Initialize process group
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        # Single process
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return local_rank, world_size, device


def create_synthetic_data(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic training data."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids, labels


def benchmark_model(
    model: nn.Module,
    data_loader: list,
    optimizer: optim.Optimizer,
    num_iterations: int,
    device: torch.device,
    description: str,
) -> float:
    """Benchmark model training performance."""
    model.train()

    # Warmup
    for i, (input_ids, labels) in enumerate(data_loader[:2]):
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()

    for i in range(num_iterations):
        input_ids, labels = data_loader[i % len(data_loader)]

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)), labels.view(-1)
        )
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time

    print(
        f"{description}: {total_time:.4f}s ({total_time/num_iterations:.4f}s per iteration)"
    )
    return total_time


def demonstrate_basic_overlap(local_rank: int, device: torch.device) -> None:
    """Demonstrate basic parameter overlap functionality."""
    print(f"Rank {local_rank}: Demonstrating basic parameter overlap...")

    # Create simple model
    model = nn.Sequential(
        nn.Linear(512, 1024, device=device),
        nn.ReLU(),
        nn.Linear(1024, 512, device=device),
        nn.ReLU(),
        nn.Linear(512, 256, device=device),
    )

    # Convert to overlapped model
    config = OverlapConfig(
        mode=OverlapMode.PIPELINE,
        num_streams=4,
        cache_size_mb=50,
        enable_profiling=True,
    )

    overlapped_model = convert_to_overlapped_model(model, config)

    # Test forward pass
    input_tensor = torch.randn(32, 512, device=device)

    with torch.no_grad():
        standard_output = model(input_tensor)
        overlapped_output = overlapped_model(input_tensor)

    # Verify outputs are close (they should be identical with no overlap)
    if config.mode == OverlapMode.NONE:
        torch.testing.assert_close(
            standard_output, overlapped_output, rtol=1e-5, atol=1e-5
        )
        print(f"Rank {local_rank}: Output verification passed!")
    else:
        print(f"Rank {local_rank}: Overlapped model forward pass completed")

    # Get statistics
    for module in overlapped_model.modules():
        gatherer = getattr(module, "gatherer", None)
        if gatherer is not None:
            stats = gatherer.get_stats()
            if local_rank == 0:
                print(f"Gatherer stats: {stats}")
            gatherer.shutdown()
            break

    print(f"Rank {local_rank}: Basic overlap demonstration completed")


def demonstrate_tensor_parallel_overlap(
    local_rank: int, world_size: int, device: torch.device
) -> None:
    """Demonstrate overlap with tensor parallelism."""
    if world_size < 2:
        print("Tensor parallel demonstration requires at least 2 processes")
        return

    print(f"Rank {local_rank}: Demonstrating tensor parallel overlap...")

    # Create overlap config
    config = OverlapConfig(
        mode=OverlapMode.AGGRESSIVE,
        num_streams=2,
        cache_size_mb=20,
    )

    # Create async gatherer (mock distributed for demo)
    gatherer = AsyncParameterGatherer(config, device=device)

    # Create overlapped column parallel layer
    col_layer = OverlappedColumnParallelLinear(
        in_features=256,
        out_features=512,  # Will be split across ranks
        bias=True,
        gather_output=True,
        gatherer=gatherer,
        device=device,
    )

    # Create overlapped row parallel layer
    row_layer = OverlappedRowParallelLinear(
        in_features=512,
        out_features=256,
        bias=True,
        input_is_parallel=False,
        gatherer=gatherer,
        device=device,
    )

    # Test forward passes
    input_tensor = torch.randn(16, 256, device=device)

    try:
        # This will fail in single-process mode due to mock distributed setup
        # but demonstrates the API usage
        col_output = col_layer(input_tensor)
        row_output = row_layer(col_output)

        print(f"Rank {local_rank}: Tensor parallel layers executed successfully")
        print(f"Rank {local_rank}: Input shape: {input_tensor.shape}")
        print(f"Rank {local_rank}: Column output shape: {col_output.shape}")
        print(f"Rank {local_rank}: Row output shape: {row_output.shape}")

    except Exception as e:
        print(
            f"Rank {local_rank}: Tensor parallel test failed (expected in single-process): {e}"
        )

    # Clean up
    gatherer.shutdown()
    print(f"Rank {local_rank}: Tensor parallel demonstration completed")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Parameter Overlap Example")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Vocabulary size")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num-iterations", type=int, default=10, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )

    args = parser.parse_args()

    # Setup distributed environment
    local_rank, world_size, device = setup_distributed()

    if local_rank == 0:
        print(f"Running on {world_size} process(es) with device: {device}")
        print(
            f"Configuration: vocab_size={args.vocab_size}, hidden_dim={args.hidden_dim}, "
            f"num_layers={args.num_layers}, seq_len={args.seq_len}"
        )

    # Demonstrate basic overlap
    demonstrate_basic_overlap(local_rank, device)

    # Demonstrate tensor parallel overlap (if multi-process)
    if world_size > 1:
        demonstrate_tensor_parallel_overlap(local_rank, world_size, device)

    # Performance benchmark
    if args.benchmark:
        if local_rank == 0:
            print("\n=== Performance Benchmark ===")

        # Create models
        standard_model = SimpleTransformer(
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            max_seq_len=args.seq_len,
            device=device,
        )

        overlapped_model = SimpleTransformer(
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            max_seq_len=args.seq_len,
            device=device,
        )

        # Convert overlapped model
        config = OverlapConfig(
            mode=OverlapMode.AGGRESSIVE,
            num_streams=4,
            cache_size_mb=100,
            enable_profiling=True,
        )
        overlapped_model = convert_to_overlapped_model(overlapped_model, config)

        # Copy weights for fair comparison
        with torch.no_grad():
            for std_param, ovl_param in zip(
                standard_model.parameters(), overlapped_model.parameters()
            ):
                if std_param.shape == ovl_param.shape:
                    ovl_param.copy_(std_param)

        # Setup DDP if multi-process
        if world_size > 1:
            standard_model = DDP(
                standard_model,
                device_ids=[local_rank] if device.type == "cuda" else None,
            )
            overlapped_model = DDP(
                overlapped_model,
                device_ids=[local_rank] if device.type == "cuda" else None,
            )

        # Create optimizers
        std_optimizer = optim.Adam(standard_model.parameters(), lr=1e-4)
        ovl_optimizer = optim.Adam(overlapped_model.parameters(), lr=1e-4)

        # Create synthetic data
        data_loader = []
        for _ in range(args.num_iterations * 2):  # Extra data for variety
            input_ids, labels = create_synthetic_data(
                args.batch_size, args.seq_len, args.vocab_size, device
            )
            data_loader.append((input_ids, labels))

        # Benchmark
        std_time = benchmark_model(
            standard_model,
            data_loader,
            std_optimizer,
            args.num_iterations,
            device,
            f"Rank {local_rank} Standard Model",
        )

        ovl_time = benchmark_model(
            overlapped_model,
            data_loader,
            ovl_optimizer,
            args.num_iterations,
            device,
            f"Rank {local_rank} Overlapped Model",
        )

        if local_rank == 0:
            speedup = std_time / ovl_time
            print(f"\nSpeedup: {speedup:.2f}x")

            # Print overlap statistics
            for module in overlapped_model.modules():
                gatherer = getattr(module, "gatherer", None)
                if gatherer is not None:
                    stats = gatherer.get_stats()
                    print(f"Final gatherer stats: {stats}")
                    break

        # Clean up gatherers
        for module in overlapped_model.modules():
            gatherer = getattr(module, "gatherer", None)
            if gatherer is not None:
                gatherer.shutdown()

    # Clean up distributed
    if world_size > 1:
        dist.destroy_process_group()

    if local_rank == 0:
        print("\nParameter overlap example completed successfully!")


if __name__ == "__main__":
    main()
