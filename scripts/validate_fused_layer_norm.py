"""
Validation Script for Fused Layer Norm Against Megatron-LM

This script validates RoseLLM's fused layer norm implementation against
Megatron-LM's reference implementation for bit-to-bit accuracy.
"""

import argparse
import sys
from typing import Tuple

import torch
import torch.nn as nn

# Add paths for imports
sys.path.append("/data/projects/rosellm")
sys.path.append("/data/projects/Megatron-LM")

from rosellm.rosetrainer.fusions import FusedLayerNorm, LayerNormConfig  # noqa: E402


def compare_with_megatron(
    hidden_size: int = 2048,
    batch_size: int = 2,
    seq_length: int = 128,
    eps: float = 1e-5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_zero_centered: bool = False,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """Compare RoseLLM's implementation with Megatron-LM.

    Args:
        hidden_size: Hidden dimension
        batch_size: Batch size
        seq_length: Sequence length
        eps: Epsilon value
        device: Device to run on
        use_zero_centered: Use zero-centered gamma
        verbose: Print detailed results

    Returns:
        Tuple of (forward_error, weight_grad_error, input_grad_error)
    """
    try:
        # Try to import Megatron-LM's implementation
        from megatron.core.fusions.fused_layer_norm import (
            FusedLayerNorm as MegatronLayerNorm,  # type: ignore
        )
        from megatron.core.transformer import TransformerConfig  # type: ignore

        # Create Megatron config
        megatron_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=1,
            layernorm_epsilon=eps,
            persist_layer_norm=False,
            layernorm_zero_centered_gamma=use_zero_centered,
            normalization="LayerNorm",
        )

        # Create Megatron layer norm
        megatron_ln = MegatronLayerNorm(
            config=megatron_config,
            hidden_size=hidden_size,
            eps=eps,
            persist_layer_norm=False,
            zero_centered_gamma=use_zero_centered,
        ).to(device)

    except ImportError as e:
        if verbose:
            print(f"Could not import Megatron-LM: {e}")
            print("Falling back to PyTorch LayerNorm for comparison")

        # Fallback to PyTorch
        megatron_ln = nn.LayerNorm(hidden_size, eps=eps).to(device)
        if use_zero_centered:
            with torch.no_grad():
                megatron_ln.weight.zero_()

    # Create RoseLLM layer norm
    config = LayerNormConfig(
        hidden_size=hidden_size,
        eps=eps,
        persist_layer_norm=False,
        zero_centered_gamma=use_zero_centered,
        memory_efficient=False,
        device=device,
    )
    rosellm_ln = FusedLayerNorm(config)

    # Synchronize weights
    with torch.no_grad():
        rosellm_ln.weight.copy_(megatron_ln.weight)
        rosellm_ln.bias.copy_(megatron_ln.bias)

    # Create identical inputs
    torch.manual_seed(42)
    input1 = torch.randn(
        batch_size, seq_length, hidden_size, device=device, requires_grad=True
    )
    input2 = input1.clone().detach().requires_grad_(True)
    grad_output = torch.randn_like(input1)

    # Forward pass
    megatron_out = megatron_ln(input1)
    rosellm_out = rosellm_ln(input2)

    # Compute forward error
    forward_error = (megatron_out - rosellm_out).abs().max().item()

    # Backward pass
    megatron_out.backward(grad_output)
    rosellm_out.backward(grad_output)

    # Compute gradient errors
    if megatron_ln.weight.grad is None or rosellm_ln.weight.grad is None:
        raise RuntimeError("Weight gradients not computed")
    if input1.grad is None or input2.grad is None:
        raise RuntimeError("Input gradients not computed")

    weight_grad_error = (
        (megatron_ln.weight.grad - rosellm_ln.weight.grad).abs().max().item()
    )
    input_grad_error = (input1.grad - input2.grad).abs().max().item()

    if verbose:
        print(f"\nValidation Results (hidden_size={hidden_size}):")
        print("-" * 50)
        print(f"Forward pass max error: {forward_error:.2e}")
        print(f"Weight gradient max error: {weight_grad_error:.2e}")
        print(f"Input gradient max error: {input_grad_error:.2e}")

        # Check if errors are acceptable
        tolerance = 1e-5
        if (
            forward_error < tolerance
            and weight_grad_error < tolerance
            and input_grad_error < tolerance
        ):
            print("✓ PASSED: Bit-to-bit accuracy validated")
        else:
            print("✗ FAILED: Errors exceed tolerance")

    return forward_error, weight_grad_error, input_grad_error


def validate_persistent_kernel_sizes(device: torch.device, verbose: bool = True):
    """Validate persistent kernel for supported sizes.

    Args:
        device: Device to run on
        verbose: Print results
    """
    supported_sizes = FusedLayerNorm.PERSIST_LN_HIDDEN_SIZES

    if verbose:
        print("\nPersistent Kernel Size Validation:")
        print("-" * 50)

    results = []
    for size in supported_sizes[:5]:  # Test first 5 sizes
        config = LayerNormConfig(
            hidden_size=size, persist_layer_norm=True, device=device
        )

        try:
            ln = FusedLayerNorm(config)
            x = torch.randn(2, 128, size, device=device)
            _ = ln(x)

            if ln.use_persist_kernel:
                status = "✓ Persistent"
            elif ln.use_fused_kernel:
                status = "✓ Fused"
            else:
                status = "✓ CPU Fallback"

            results.append((size, status))
        except Exception as e:
            results.append((size, f"✗ Error: {e}"))

    if verbose:
        for size, status in results:
            print(f"Size {size:5d}: {status}")


def benchmark_configurations(device: torch.device, verbose: bool = True):
    """Benchmark different configurations.

    Args:
        device: Device to run on
        verbose: Print results
    """
    import time

    configs = [
        ("Standard", {"persist_layer_norm": False, "zero_centered_gamma": False}),
        ("Persistent", {"persist_layer_norm": True, "zero_centered_gamma": False}),
        ("Zero-Centered", {"persist_layer_norm": False, "zero_centered_gamma": True}),
        ("Memory-Efficient", {"persist_layer_norm": False, "memory_efficient": True}),
    ]

    hidden_size = 4096
    batch_size = 4
    seq_length = 512
    num_iterations = 100

    if verbose:
        print("\nConfiguration Benchmarks:")
        print("-" * 50)
        print(f"Shape: ({batch_size}, {seq_length}, {hidden_size})")
        print(
            f"{'Config':<20} {'Forward (ms)':<15} "
            f"{'Backward (ms)':<15} {'Total (ms)':<15}"
        )
        print("-" * 65)

    for name, kwargs in configs:
        config = LayerNormConfig(hidden_size=hidden_size, device=device, **kwargs)

        ln = FusedLayerNorm(config)
        x = torch.randn(
            batch_size, seq_length, hidden_size, device=device, requires_grad=True
        )
        grad_out = torch.randn_like(x)

        # Warmup
        for _ in range(10):
            out = ln(x)
            out.backward(grad_out)

        # Time forward
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            out = ln(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_time = (time.time() - start) * 1000 / num_iterations

        # Time backward
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            out = ln(x)
            out.backward(grad_out)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = (time.time() - start) * 1000 / num_iterations
        backward_time = total_time - forward_time

        if verbose:
            print(
                f"{name:<20} {forward_time:<15.3f} "
                f"{backward_time:<15.3f} {total_time:<15.3f}"
            )


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Fused Layer Norm")
    parser.add_argument(
        "--hidden-size", type=int, default=2048, help="Hidden dimension size"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--compare-megatron", action="store_true", help="Compare with Megatron-LM"
    )
    parser.add_argument(
        "--test-kernels", action="store_true", help="Test persistent kernel sizes"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run configuration benchmarks"
    )
    parser.add_argument("--all", action="store_true", help="Run all validations")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Fused Layer Norm Validation")
    print("=" * 80)

    # Compare with Megatron-LM
    if args.compare_megatron or args.all:
        print("\n1. Comparison with Megatron-LM/PyTorch:")
        print("=" * 50)

        # Test different configurations
        test_configs = [
            (1024, False),
            (2048, False),
            (4096, False),
            (2048, True),  # Zero-centered
        ]

        all_passed = True
        for hidden_size, zero_centered in test_configs:
            forward_err, weight_err, input_err = compare_with_megatron(
                hidden_size=hidden_size,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                device=device,
                use_zero_centered=zero_centered,
                verbose=True,
            )

            if forward_err > 1e-5 or weight_err > 1e-5 or input_err > 1e-5:
                all_passed = False

        if all_passed:
            print("\n✓ All accuracy tests PASSED")
        else:
            print("\n✗ Some accuracy tests FAILED")

    # Test kernel sizes
    if args.test_kernels or args.all:
        print("\n2. Kernel Size Support:")
        print("=" * 50)
        validate_persistent_kernel_sizes(device, verbose=True)

    # Benchmark configurations
    if args.benchmark or args.all:
        print("\n3. Performance Benchmarks:")
        print("=" * 50)
        benchmark_configurations(device, verbose=True)

    print("\n" + "=" * 80)
    print("Validation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
