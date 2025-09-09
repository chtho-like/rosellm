#!/usr/bin/env python3
"""Performance benchmarks and utilities for RoPE implementation.

This module provides tools to benchmark and profile RoPE operations
across different configurations and hardware setups.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Union

import torch

from rosellm.rosetrainer.embeddings.rope import (
    FusedRoPE,
    RoPEConfig,
    RoPEInterpolationType,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_optimized,
    rotate_half,
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    config_name: str
    avg_time_ms: float
    std_time_ms: float
    throughput_tokens_per_sec: float
    memory_mb: float
    device: str


class RoPEBenchmark:
    """Benchmark suite for RoPE implementations."""

    def __init__(
        self,
        batch_size: int = 32,
        seq_length: int = 2048,
        num_heads: int = 32,
        head_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
    ):
        """Initialize benchmark suite.

        Args:
            batch_size: Batch size for benchmarking
            seq_length: Sequence length for benchmarking
            num_heads: Number of attention heads
            head_dim: Dimension per head
            device: Device to run benchmarks on
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
        """
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

        # Create test tensors
        self.q = torch.randn(
            batch_size,
            seq_length,
            num_heads,
            head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.k = torch.randn(
            batch_size,
            seq_length,
            num_heads,
            head_dim,
            device=self.device,
            dtype=torch.float32,
        )

    def benchmark_config(
        self,
        config: RoPEConfig,
        config_name: str,
    ) -> BenchmarkResult:
        """Benchmark a specific RoPE configuration.

        Args:
            config: RoPE configuration to benchmark
            config_name: Name for this configuration

        Returns:
            Benchmark results
        """
        # Create RoPE instance
        rope: Union[FusedRoPE, RotaryEmbedding]
        if config.use_fused:
            rope = FusedRoPE(config).to(self.device)
        else:
            rope = RotaryEmbedding(config).to(self.device)

        # Warmup
        for _ in range(self.warmup_iterations):
            _ = rope(self.q, self.k)

        # Synchronize if using CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Measure memory before
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            mem_before = 0

        # Benchmark
        times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            _ = rope(self.q, self.k)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        # Measure memory after
        if self.device.type == "cuda":
            mem_after = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            memory_used = mem_after - mem_before
        else:
            memory_used = 0

        # Calculate statistics
        times_tensor = torch.tensor(times)
        avg_time = times_tensor.mean().item()
        std_time = times_tensor.std().item()

        # Calculate throughput
        total_tokens = self.batch_size * self.seq_length
        throughput = total_tokens / (avg_time / 1000)  # tokens/sec

        return BenchmarkResult(
            config_name=config_name,
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            throughput_tokens_per_sec=throughput,
            memory_mb=memory_used,
            device=str(self.device),
        )

    def benchmark_all_configs(self) -> List[BenchmarkResult]:
        """Benchmark all standard RoPE configurations.

        Returns:
            List of benchmark results
        """
        configs = [
            (
                "Standard RoPE",
                RoPEConfig(
                    dim=self.head_dim,
                    max_position_embeddings=self.seq_length,
                    use_fused=False,
                ),
            ),
            (
                "Optimized RoPE",
                RoPEConfig(
                    dim=self.head_dim,
                    max_position_embeddings=self.seq_length,
                    use_fused=True,
                ),
            ),
            (
                "Partial RoPE (50%)",
                RoPEConfig(
                    dim=self.head_dim,
                    max_position_embeddings=self.seq_length,
                    partial_rotary_factor=0.5,
                    use_fused=True,
                ),
            ),
            (
                "Linear Interpolation 2x",
                RoPEConfig(
                    dim=self.head_dim,
                    max_position_embeddings=self.seq_length // 2,
                    interpolation_type=RoPEInterpolationType.LINEAR,
                    scaling_factor=2.0,
                    use_fused=True,
                ),
            ),
            (
                "NTK Interpolation 2x",
                RoPEConfig(
                    dim=self.head_dim,
                    max_position_embeddings=self.seq_length // 2,
                    interpolation_type=RoPEInterpolationType.NTK,
                    scaling_factor=2.0,
                    use_fused=True,
                ),
            ),
            (
                "YaRN Interpolation 2x",
                RoPEConfig(
                    dim=self.head_dim,
                    max_position_embeddings=self.seq_length // 2,
                    interpolation_type=RoPEInterpolationType.YaRN,
                    scaling_factor=2.0,
                    yarn_beta_fast=32.0,
                    yarn_beta_slow=1.0,
                    use_fused=True,
                ),
            ),
        ]

        results = []
        for name, config in configs:
            print(f"Benchmarking {name}...")
            result = self.benchmark_config(config, name)
            results.append(result)
            print(f"  Avg time: {result.avg_time_ms:.3f} ± {result.std_time_ms:.3f} ms")
            print(f"  Throughput: {result.throughput_tokens_per_sec:.0f} tokens/sec")
            print(f"  Memory: {result.memory_mb:.2f} MB")

        return results

    def compare_operations(self) -> Dict[str, BenchmarkResult]:
        """Compare individual RoPE operations.

        Returns:
            Dictionary of operation benchmarks
        """
        results = {}

        # Test rotate_half
        print("Benchmarking rotate_half...")
        times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            _ = rotate_half(self.q)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = torch.tensor(times).mean().item()
        results["rotate_half"] = BenchmarkResult(
            config_name="rotate_half",
            avg_time_ms=avg_time,
            std_time_ms=torch.tensor(times).std().item(),
            throughput_tokens_per_sec=0,
            memory_mb=0,
            device=str(self.device),
        )

        # Test apply_rotary_pos_emb
        cos = torch.ones(self.seq_length, self.head_dim, device=self.device)
        sin = torch.ones(self.seq_length, self.head_dim, device=self.device)

        print("Benchmarking apply_rotary_pos_emb...")
        times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            _ = apply_rotary_pos_emb(self.q, cos, sin)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = torch.tensor(times).mean().item()
        results["apply_rotary_pos_emb"] = BenchmarkResult(
            config_name="apply_rotary_pos_emb",
            avg_time_ms=avg_time,
            std_time_ms=torch.tensor(times).std().item(),
            throughput_tokens_per_sec=0,
            memory_mb=0,
            device=str(self.device),
        )

        # Test apply_rotary_pos_emb_optimized
        print("Benchmarking apply_rotary_pos_emb_optimized...")
        times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            _ = apply_rotary_pos_emb_optimized(self.q, cos, sin)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = torch.tensor(times).mean().item()
        results["apply_rotary_pos_emb_optimized"] = BenchmarkResult(
            config_name="apply_rotary_pos_emb_optimized",
            avg_time_ms=avg_time,
            std_time_ms=torch.tensor(times).std().item(),
            throughput_tokens_per_sec=0,
            memory_mb=0,
            device=str(self.device),
        )

        return results


def print_benchmark_summary(results: List[BenchmarkResult]):
    """Print a formatted summary of benchmark results.

    Args:
        results: List of benchmark results
    """
    print("\n" + "=" * 80)
    print("RoPE Benchmark Summary")
    print("=" * 80)
    print(
        f"{'Configuration':<30} {'Time (ms)':<15} "
        f"{'Throughput':<20} {'Memory (MB)':<10}"
    )
    print("-" * 80)

    baseline_time = results[0].avg_time_ms if results else 1.0

    for result in results:
        speedup = baseline_time / result.avg_time_ms
        print(
            f"{result.config_name:<30} "
            f"{result.avg_time_ms:>7.3f} ± {result.std_time_ms:<5.3f} "
            f"{result.throughput_tokens_per_sec:>12.0f} tok/s "
            f"{result.memory_mb:>8.2f} "
            f"({speedup:.2f}x)"
        )

    print("=" * 80)


if __name__ == "__main__":
    # Run benchmarks
    benchmark = RoPEBenchmark(
        batch_size=8,
        seq_length=512,
        num_heads=32,
        head_dim=128,
    )

    print("Running RoPE benchmarks...")
    print(f"Device: {benchmark.device}")
    print(f"Batch size: {benchmark.batch_size}")
    print(f"Sequence length: {benchmark.seq_length}")
    print(f"Num heads: {benchmark.num_heads}")
    print(f"Head dim: {benchmark.head_dim}")
    print()

    # Benchmark all configurations
    results = benchmark.benchmark_all_configs()
    print_benchmark_summary(results)

    # Compare individual operations
    print("\nComparing individual operations...")
    op_results = benchmark.compare_operations()
    for op_name, result in op_results.items():
        print(f"{op_name}: {result.avg_time_ms:.3f} ± {result.std_time_ms:.3f} ms")
