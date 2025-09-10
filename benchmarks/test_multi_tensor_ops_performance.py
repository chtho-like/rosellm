#!/usr/bin/env python3
"""
Performance benchmarks for multi-tensor gradient operations.

This script benchmarks the performance of multi-tensor operations across
different backends and compares them with standard PyTorch operations.

Usage:
    python benchmarks/test_multi_tensor_ops_performance.py [options]
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from rosellm.rosetrainer.utils.multi_tensor_ops import Backend, MultiTensorOperator


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""

    device: str = "cuda"
    num_warmup: int = 10
    num_iterations: int = 100
    tensor_sizes: Optional[Dict[str, Tuple[int, int]]] = None

    def __post_init__(self):
        if self.tensor_sizes is None:
            self.tensor_sizes = {
                "small": (10, 1000),  # 10 tensors of 1K elements
                "medium": (100, 10000),  # 100 tensors of 10K elements
                "large": (1000, 100000),  # 1000 tensors of 100K elements
            }


class BenchmarkSuite:
    """Suite of performance benchmarks for multi-tensor operations."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark suite."""
        self.config = config
        self.device = torch.device(config.device)

        # Check device availability
        if config.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")

        # Results storage
        self.results: Dict[str, Any] = {
            "config": {
                "device": str(self.device),
                "num_warmup": config.num_warmup,
                "num_iterations": config.num_iterations,
            },
            "benchmarks": {},
        }

    def create_test_tensors(self, size_key: str) -> List[torch.Tensor]:
        """Create test tensors for benchmarking."""
        if self.config.tensor_sizes is None:
            raise ValueError("tensor_sizes not configured")
        num_tensors, tensor_size = self.config.tensor_sizes[size_key]

        # Create tensors with varying scales for realistic scenarios
        tensors = []
        for i in range(num_tensors):
            scale = 10 ** ((i % 5) - 2)  # Scales from 0.01 to 100
            tensor = (
                torch.randn(tensor_size, device=self.device, dtype=torch.float32)
                * scale
            )
            tensors.append(tensor)

        return tensors

    def create_test_model(self, size_key: str) -> nn.Module:
        """Create test model for gradient operations."""
        if self.config.tensor_sizes is None:
            raise ValueError("tensor_sizes not configured")
        num_tensors, tensor_size = self.config.tensor_sizes[size_key]

        # Create a simple model with multiple layers
        layers = []
        input_size = int(np.sqrt(tensor_size))
        hidden_size = input_size

        for _ in range(min(num_tensors, 10)):  # Limit layers for large configs
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        model = nn.Sequential(*layers).to(self.device)

        # Add dummy gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 10.0

        return model

    def time_operation(self, func, *args, **kwargs) -> Tuple[float, Any]:
        """Time a single operation with CUDA synchronization if needed."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = func(*args, **kwargs)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        return elapsed, result

    def benchmark_norm_calculation(self, size_key: str) -> Dict[str, Any]:
        """Benchmark norm calculation across backends."""
        print(f"\nBenchmarking norm calculation ({size_key})...")
        tensors = self.create_test_tensors(size_key)

        results = {}

        # Test each backend
        for backend in [Backend.PYTORCH, Backend.APEX, Backend.TRANSFORMER_ENGINE]:
            try:
                operator = MultiTensorOperator(
                    preferred_backend=backend,
                    device=self.device,
                    enable_benchmarking=True,
                )

                # Skip if backend not available
                if operator.backend.name != backend:
                    continue

                # Warmup
                for _ in range(self.config.num_warmup):
                    operator.calculate_norm(tensors, norm_type=2.0)

                # Benchmark
                times = []
                for _ in range(self.config.num_iterations):
                    result = self.time_operation(
                        operator.calculate_norm, tensors, norm_type=2.0
                    )
                    elapsed = result[0]
                    norm = result[1]
                    times.append(elapsed)

                # Calculate statistics
                results[backend.value] = {
                    "mean_time": np.mean(times),
                    "std_time": np.std(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "total_time": np.sum(times),
                    "norm_value": float(norm),
                }

                print(
                    f"  {backend.value}: {np.mean(times)*1000:.3f}ms "
                    f"± {np.std(times)*1000:.3f}ms"
                )

            except Exception as e:
                print(f"  {backend.value}: Failed - {e}")

        # Compare with standard PyTorch
        times = []
        for _ in range(self.config.num_iterations):
            result = self.time_operation(
                lambda: torch.sqrt(sum(t.pow(2).sum() for t in tensors))
            )
            elapsed = result[0]
            norm = result[1]
            times.append(elapsed)

        results["pytorch_standard"] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "total_time": np.sum(times),
            "norm_value": float(norm),
        }

        print(
            f"  pytorch_standard: {np.mean(times)*1000:.3f}ms "
            f"± {np.std(times)*1000:.3f}ms"
        )

        return results

    def benchmark_gradient_clipping(self, size_key: str) -> Dict[str, Any]:
        """Benchmark gradient clipping operations."""
        print(f"\nBenchmarking gradient clipping ({size_key})...")
        model = self.create_test_model(size_key)
        params = list(model.parameters())

        results = {}

        # Test multi-tensor clipping
        operator = MultiTensorOperator(device=self.device, enable_benchmarking=True)

        # Warmup
        for _ in range(self.config.num_warmup):
            # Reset gradients
            for p in params:
                if p.grad is not None:
                    p.grad = torch.randn_like(p.grad) * 10.0

            operator.clip_grad_norm(params, max_norm=1.0)

        # Benchmark multi-tensor
        times = []
        for _ in range(self.config.num_iterations):
            # Reset gradients
            for p in params:
                if p.grad is not None:
                    p.grad = torch.randn_like(p.grad) * 10.0

            result = self.time_operation(operator.clip_grad_norm, params, max_norm=1.0)
            elapsed = result[0]
            times.append(elapsed)

        results["multi_tensor"] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
        }

        print(
            f"  multi_tensor: {np.mean(times)*1000:.3f}ms ± {np.std(times)*1000:.3f}ms"
        )

        # Compare with standard PyTorch clipping
        times = []
        for _ in range(self.config.num_iterations):
            # Reset gradients
            for p in params:
                if p.grad is not None:
                    p.grad = torch.randn_like(p.grad) * 10.0

            result = self.time_operation(
                torch.nn.utils.clip_grad_norm_, params, max_norm=1.0
            )
            elapsed = result[0]
            times.append(elapsed)

        results["pytorch_standard"] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
        }

        print(
            f"  pytorch_standard: {np.mean(times)*1000:.3f}ms "
            f"± {np.std(times)*1000:.3f}ms"
        )

        return results

    def benchmark_tensor_scaling(self, size_key: str) -> Dict[str, Any]:
        """Benchmark tensor scaling operations."""
        print(f"\nBenchmarking tensor scaling ({size_key})...")
        tensors = self.create_test_tensors(size_key)

        results = {}

        # Test multi-tensor scaling
        operator = MultiTensorOperator(device=self.device, enable_benchmarking=True)

        # Warmup
        for _ in range(self.config.num_warmup):
            test_tensors = [t.clone() for t in tensors]
            operator.scale_tensors(test_tensors, 0.5, in_place=True)

        # Benchmark
        times = []
        for _ in range(self.config.num_iterations):
            test_tensors = [t.clone() for t in tensors]
            result = self.time_operation(
                operator.scale_tensors, test_tensors, 0.5, in_place=True
            )
            elapsed = result[0]
            times.append(elapsed)

        results["multi_tensor"] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
        }

        print(
            f"  multi_tensor: {np.mean(times)*1000:.3f}ms ± {np.std(times)*1000:.3f}ms"
        )

        # Compare with standard PyTorch
        times = []
        for _ in range(self.config.num_iterations):
            test_tensors = [t.clone() for t in tensors]

            def scale_standard():
                for t in test_tensors:
                    t.mul_(0.5)

            result = self.time_operation(scale_standard)
            elapsed = result[0]
            times.append(elapsed)

        results["pytorch_standard"] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
        }

        print(
            f"  pytorch_standard: {np.mean(times)*1000:.3f}ms "
            f"± {np.std(times)*1000:.3f}ms"
        )

        return results

    def benchmark_memory_usage(self, size_key: str) -> Dict[str, Any]:
        """Benchmark memory usage of operations."""
        if self.device.type != "cuda":
            return {"skipped": "Memory benchmarking requires CUDA"}

        print(f"\nBenchmarking memory usage ({size_key})...")
        tensors = self.create_test_tensors(size_key)

        results = {}

        # Test multi-tensor operations
        operator = MultiTensorOperator(device=self.device)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Perform operations
        for _ in range(10):
            _ = operator.calculate_norm(tensors, norm_type=2.0)
            operator.scale_tensors(tensors, 0.99, in_place=True)

        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()

        results["multi_tensor"] = {
            "peak_memory_mb": peak_memory / (1024 * 1024),
        }

        print(f"  multi_tensor peak memory: {peak_memory / (1024 * 1024):.2f} MB")

        # Compare with standard operations
        torch.cuda.reset_peak_memory_stats()

        for _ in range(10):
            _ = torch.sqrt(sum(t.pow(2).sum() for t in tensors))
            for t in tensors:
                t.mul_(0.99)

        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()

        results["pytorch_standard"] = {
            "peak_memory_mb": peak_memory / (1024 * 1024),
        }

        print(f"  pytorch_standard peak memory: {peak_memory / (1024 * 1024):.2f} MB")

        return results

    def run_all_benchmarks(self) -> None:
        """Run all benchmarks for all sizes."""
        print("=" * 60)
        print("Multi-Tensor Operations Performance Benchmarks")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Warmup iterations: {self.config.num_warmup}")
        print(f"Benchmark iterations: {self.config.num_iterations}")

        for size_key in ["small", "medium", "large"]:
            if self.config.tensor_sizes is None:
                raise ValueError("tensor_sizes not configured")
            num_tensors, tensor_size = self.config.tensor_sizes[size_key]

            print("\n" + "=" * 60)
            print(f"Size: {size_key} ({num_tensors} tensors × {tensor_size} elements)")
            print("=" * 60)

            # Run benchmarks
            self.results["benchmarks"][size_key] = {
                "config": {
                    "num_tensors": num_tensors,
                    "tensor_size": tensor_size,
                },
                "norm_calculation": self.benchmark_norm_calculation(size_key),
                "gradient_clipping": self.benchmark_gradient_clipping(size_key),
                "tensor_scaling": self.benchmark_tensor_scaling(size_key),
            }

            if self.device.type == "cuda":
                self.results["benchmarks"][size_key][
                    "memory_usage"
                ] = self.benchmark_memory_usage(size_key)

        # Print summary
        self.print_summary()

        # Save results
        self.save_results()

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for size_key in ["small", "medium", "large"]:
            if size_key not in self.results["benchmarks"]:
                continue

            print(f"\n{size_key.upper()}:")

            size_results = self.results["benchmarks"][size_key]

            # Calculate speedups
            for op in ["norm_calculation", "gradient_clipping", "tensor_scaling"]:
                if op not in size_results:
                    continue

                op_results = size_results[op]
                if "multi_tensor" in op_results and "pytorch_standard" in op_results:
                    mt_time = op_results.get("multi_tensor", {}).get("mean_time", 0)
                    std_time = op_results["pytorch_standard"]["mean_time"]

                    if mt_time > 0:
                        speedup = std_time / mt_time
                        print(f"  {op}: {speedup:.2f}x speedup")

    def save_results(self) -> None:
        """Save benchmark results to file."""
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"multi_tensor_ops_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark multi-tensor gradient operations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run benchmarks on",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="all",
        choices=["small", "medium", "large", "all"],
        help="Size of benchmarks to run",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )

    args = parser.parse_args()

    # Create config
    config = BenchmarkConfig(
        device=args.device,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
    )

    # Adjust sizes if specific size requested
    if args.size != "all":
        original_sizes = config.tensor_sizes
        config.tensor_sizes = {args.size: original_sizes[args.size]}

    # Run benchmarks
    suite = BenchmarkSuite(config)
    suite.run_all_benchmarks()


if __name__ == "__main__":
    main()
