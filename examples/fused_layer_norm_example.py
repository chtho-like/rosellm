"""
Fused Layer Normalization Example

This example demonstrates the usage and performance benefits of fused layer
normalization compared to standard PyTorch LayerNorm. It includes:
- Basic usage demonstration
- Performance benchmarking
- Accuracy validation
- Memory usage comparison
- Integration with distributed training
"""

import argparse
import time
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from rosellm.rosetrainer.fusions import FusedLayerNorm, LayerNormConfig


class TransformerBlock(nn.Module):
    """Simple transformer block to demonstrate layer norm usage."""

    def __init__(self, hidden_size: int, use_fused: bool = True):
        super().__init__()
        self.hidden_size = hidden_size

        # Choose layer norm implementation
        if use_fused:
            config = LayerNormConfig(
                hidden_size=hidden_size,
                eps=1e-5,
                persist_layer_norm=True,
                zero_centered_gamma=False,
                memory_efficient=True,
            )
            self.ln1 = FusedLayerNorm(config)
            self.ln2 = FusedLayerNorm(config)
        else:
            self.ln1 = nn.LayerNorm(hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)

        # Simple MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # Simple attention (placeholder)
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x

        return x


def benchmark_layer_norm(
    hidden_size: int,
    batch_size: int,
    seq_length: int,
    num_iterations: int = 100,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, float]:
    """Benchmark fused vs standard layer norm performance.

    Args:
        hidden_size: Hidden dimension size
        batch_size: Batch size
        seq_length: Sequence length
        num_iterations: Number of iterations for timing
        device: Device to run on

    Returns:
        Dictionary with timing results
    """
    # Create inputs
    input_tensor = torch.randn(
        batch_size, seq_length, hidden_size, device=device, requires_grad=True
    )
    grad_output = torch.randn_like(input_tensor)

    # Standard LayerNorm
    standard_ln = nn.LayerNorm(hidden_size).to(device)

    # Warmup
    for _ in range(10):
        out = standard_ln(input_tensor)
        out.backward(grad_output)

    # Benchmark standard
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    for _ in range(num_iterations):
        out = standard_ln(input_tensor)
        out.backward(grad_output)
    torch.cuda.synchronize() if device.type == "cuda" else None
    standard_time = time.time() - start

    # Fused LayerNorm
    config = LayerNormConfig(
        hidden_size=hidden_size,
        eps=1e-5,
        persist_layer_norm=True,
        zero_centered_gamma=False,
        memory_efficient=True,
        device=device,
    )
    fused_ln = FusedLayerNorm(config)

    # Warmup
    for _ in range(10):
        out = fused_ln(input_tensor)
        out.backward(grad_output)

    # Benchmark fused
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    for _ in range(num_iterations):
        out = fused_ln(input_tensor)
        out.backward(grad_output)
    torch.cuda.synchronize() if device.type == "cuda" else None
    fused_time = time.time() - start

    return {
        "standard_time": standard_time,
        "fused_time": fused_time,
        "speedup": standard_time / fused_time,
    }


def validate_accuracy(
    hidden_size: int = 1024,
    batch_size: int = 2,
    seq_length: int = 128,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[float, float]:
    """Validate accuracy of fused layer norm against PyTorch.

    Args:
        hidden_size: Hidden dimension size
        batch_size: Batch size
        seq_length: Sequence length
        device: Device to run on

    Returns:
        Tuple of (forward_error, backward_error)
    """
    # Create identical inputs
    torch.manual_seed(42)
    input_tensor = torch.randn(
        batch_size, seq_length, hidden_size, device=device, requires_grad=True
    )
    input_copy = input_tensor.clone().detach().requires_grad_(True)
    grad_output = torch.randn_like(input_tensor)

    # Standard LayerNorm
    standard_ln = nn.LayerNorm(hidden_size, eps=1e-5).to(device)

    # Fused LayerNorm with same parameters
    config = LayerNormConfig(
        hidden_size=hidden_size,
        eps=1e-5,
        persist_layer_norm=False,  # Use standard fused for accuracy
        zero_centered_gamma=False,
        device=device,
    )
    fused_ln = FusedLayerNorm(config)

    # Copy weights
    with torch.no_grad():
        fused_ln.weight.copy_(standard_ln.weight)
        fused_ln.bias.copy_(standard_ln.bias)

    # Forward pass
    standard_out = standard_ln(input_tensor)
    fused_out = fused_ln(input_copy)

    # Check forward accuracy
    forward_error = (standard_out - fused_out).abs().max().item()

    # Backward pass
    standard_out.backward(grad_output)
    fused_out.backward(grad_output)

    # Check gradient accuracy
    if input_tensor.grad is None or input_copy.grad is None:
        raise RuntimeError("Gradients not computed")
    grad_error = (input_tensor.grad - input_copy.grad).abs().max().item()

    return forward_error, grad_error


def distributed_example(rank: int, world_size: int, hidden_size: int = 2048):
    """Example of using fused layer norm in distributed training.

    Args:
        rank: Process rank
        world_size: Total number of processes
        hidden_size: Hidden dimension size
    """
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Create model with fused layer norm
    model = TransformerBlock(hidden_size, use_fused=True).cuda(rank)
    model = DDP(model, device_ids=[rank])

    # Training step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step in range(10):
        # Create batch
        batch = torch.randn(2, 128, hidden_size).cuda(rank)

        # Forward and backward
        output = model(batch)
        loss = output.mean()
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        if rank == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")

    dist.destroy_process_group()


def main():
    """Main function to run examples and benchmarks."""
    parser = argparse.ArgumentParser(description="Fused Layer Norm Example")
    parser.add_argument(
        "--hidden-size", type=int, default=2048, help="Hidden dimension size"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate accuracy against PyTorch"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Fused Layer Normalization Example")
    print("=" * 80)

    # Basic usage example
    print("\n1. Basic Usage Example:")
    print("-" * 40)

    # Create layer norm
    config = LayerNormConfig(
        hidden_size=args.hidden_size,
        eps=1e-5,
        persist_layer_norm=True,
        zero_centered_gamma=True,  # Better numerical stability
        memory_efficient=True,
    )
    layer_norm = FusedLayerNorm(config).to(device)

    # Create input
    x = torch.randn(args.batch_size, args.seq_length, args.hidden_size).to(device)

    # Forward pass
    output = layer_norm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Layer norm config: {layer_norm.extra_repr()}")

    # Accuracy validation
    if args.validate:
        print("\n2. Accuracy Validation:")
        print("-" * 40)

        forward_err, backward_err = validate_accuracy(
            args.hidden_size, args.batch_size, args.seq_length, device
        )

        print(f"Forward pass max error: {forward_err:.2e}")
        print(f"Backward pass max error: {backward_err:.2e}")

        if forward_err < 1e-5 and backward_err < 1e-5:
            print("✓ Accuracy validation PASSED")
        else:
            print("✗ Accuracy validation FAILED")

    # Performance benchmark
    if args.benchmark:
        print("\n3. Performance Benchmark:")
        print("-" * 40)

        # Test different hidden sizes
        test_sizes = [1024, 2048, 4096, 8192]

        print(
            f"{'Hidden Size':<15} {'Standard (ms)':<15} "
            f"{'Fused (ms)':<15} {'Speedup':<10}"
        )
        print("-" * 55)

        for hidden_size in test_sizes:
            if device.type == "cuda" or hidden_size <= 2048:  # Limit CPU tests
                results = benchmark_layer_norm(
                    hidden_size,
                    args.batch_size,
                    args.seq_length,
                    num_iterations=100,
                    device=device,
                )

                standard_ms = results["standard_time"] * 1000 / 100
                fused_ms = results["fused_time"] * 1000 / 100

                print(
                    f"{hidden_size:<15} {standard_ms:<15.3f} {fused_ms:<15.3f} "
                    f"{results['speedup']:<10.2f}x"
                )

    # Model comparison
    print("\n4. Model Comparison:")
    print("-" * 40)

    # Create models
    standard_model = TransformerBlock(args.hidden_size, use_fused=False).to(device)
    fused_model = TransformerBlock(args.hidden_size, use_fused=True).to(device)

    # Count parameters
    standard_params = sum(p.numel() for p in standard_model.parameters())
    fused_params = sum(p.numel() for p in fused_model.parameters())

    print(f"Standard model parameters: {standard_params:,}")
    print(f"Fused model parameters: {fused_params:,}")
    print(f"Parameter difference: {abs(standard_params - fused_params)}")

    # Memory usage comparison
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

        # Standard model forward/backward
        x = torch.randn(args.batch_size, args.seq_length, args.hidden_size).to(device)
        y = standard_model(x)
        y.mean().backward()
        standard_memory = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()

        # Fused model forward/backward
        x = torch.randn(args.batch_size, args.seq_length, args.hidden_size).to(device)
        y = fused_model(x)
        y.mean().backward()
        fused_memory = torch.cuda.max_memory_allocated() / 1024**2

        print(f"\nMemory Usage:")
        print(f"Standard model: {standard_memory:.2f} MB")
        print(f"Fused model: {fused_memory:.2f} MB")
        print(
            f"Memory saved: {standard_memory - fused_memory:.2f} MB "
            f"({(1 - fused_memory/standard_memory) * 100:.1f}%)"
        )

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
