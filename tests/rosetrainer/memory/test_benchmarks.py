"""
Performance Benchmarks for Distributed Activation Checkpointing

This test suite provides comprehensive benchmarks for distributed activation
checkpointing performance across different scenarios, model sizes, and
parallelism configurations. It helps validate that the implementation
meets performance requirements and scales appropriately.

Benchmark Categories:
1. Checkpointing overhead benchmarks
2. Memory usage scaling tests
3. Communication coordination benchmarks
4. Multi-rank performance tests
5. Large model scaling benchmarks
6. Strategy comparison benchmarks
7. Real-world scenario simulations
"""

import gc
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import psutil
import pytest
import torch
import torch.nn as nn

from rosellm.rosetrainer.memory.distributed_checkpoint import (
    DistributedActivationCheckpointing,
    DistributedCheckpointConfig,
    DistributedCheckpointStrategy,
    create_distributed_checkpointing,
)
from rosellm.rosetrainer.memory.distributed_memory_optimizer import (
    DistributedMemoryConfig,
    DistributedMemoryOptimizer,
)
from rosellm.rosetrainer.memory.distributed_strategies import (
    CommunicationPattern,
    create_distributed_strategy,
    estimate_communication_cost,
)
from rosellm.rosetrainer.memory.model_parallel_checkpoint import (
    ModelParallelActivationManager,
    ModelParallelCheckpointConfig,
)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    execution_time_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    throughput_items_per_sec: Optional[float] = None
    memory_efficiency: Optional[float] = None
    additional_metrics: Optional[Dict[str, float]] = None


class BenchmarkSuite:
    """Base class for benchmark suites."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def run_benchmark(self, func, name: str, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark and collect metrics."""
        # Clear cache and collect garbage
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        # Measure initial memory
        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )
        initial_cpu_memory = psutil.Process().memory_info().rss

        # Run benchmark
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        # Measure final memory
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        peak_memory = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        final_cpu_memory = psutil.Process().memory_info().rss

        # Calculate metrics
        execution_time_ms = (end_time - start_time) * 1000
        gpu_memory_usage_mb = (final_memory - initial_memory) / (1024**2)
        cpu_memory_usage_mb = (final_cpu_memory - initial_cpu_memory) / (1024**2)
        peak_memory_mb = peak_memory / (1024**2)

        # Create result
        benchmark_result = BenchmarkResult(
            name=name,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=max(gpu_memory_usage_mb, cpu_memory_usage_mb),
            peak_memory_mb=peak_memory_mb,
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results."""
        if not self.results:
            return {"error": "No benchmark results available"}

        execution_times = [r.execution_time_ms for r in self.results]
        memory_usage = [r.memory_usage_mb for r in self.results]

        return {
            "total_benchmarks": len(self.results),
            "execution_time_stats": {
                "mean_ms": statistics.mean(execution_times),
                "median_ms": statistics.median(execution_times),
                "min_ms": min(execution_times),
                "max_ms": max(execution_times),
                "std_ms": (
                    statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                ),
            },
            "memory_stats": {
                "mean_mb": statistics.mean(memory_usage),
                "median_mb": statistics.median(memory_usage),
                "min_mb": min(memory_usage),
                "max_mb": max(memory_usage),
                "std_mb": (
                    statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0
                ),
            },
            "individual_results": [
                {
                    "name": r.name,
                    "time_ms": r.execution_time_ms,
                    "memory_mb": r.memory_usage_mb,
                }
                for r in self.results
            ],
        }


class CheckpointingOverheadBenchmarks(BenchmarkSuite):
    """Benchmarks for checkpointing overhead analysis."""

    def test_basic_checkpointing_overhead(self):
        """Benchmark basic checkpointing overhead."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        # Test different layer sizes
        layer_sizes = [256, 512, 1024, 2048]
        batch_size = 32

        for size in layer_sizes:
            layer = nn.Linear(size, size)
            input_tensor = torch.randn(batch_size, size)

            # Benchmark without checkpointing
            def no_checkpoint():
                return layer(input_tensor)

            no_cp_result = self.run_benchmark(
                no_checkpoint, f"no_checkpoint_size_{size}"
            )

            # Benchmark with checkpointing
            def with_checkpoint():
                return checkpointing.checkpoint_layer_distributed(
                    layer, input_tensor, layer_id=f"benchmark_layer_{size}"
                )

            cp_result = self.run_benchmark(
                with_checkpoint, f"with_checkpoint_size_{size}"
            )

            # Calculate overhead
            overhead_ratio = (
                cp_result.execution_time_ms / no_cp_result.execution_time_ms
            )
            print(f"Size {size}: Overhead ratio = {overhead_ratio:.2f}x")

            # Overhead should be reasonable (less than 10x for these sizes)
            assert (
                overhead_ratio < 10.0
            ), f"Overhead too high for size {size}: {overhead_ratio}x"

    def test_strategy_comparison_overhead(self):
        """Compare overhead of different checkpointing strategies."""
        strategies = [
            DistributedCheckpointStrategy.COORDINATED,
            DistributedCheckpointStrategy.LOAD_BALANCED,
            DistributedCheckpointStrategy.HIERARCHICAL,
            DistributedCheckpointStrategy.ADAPTIVE,
        ]

        layer = nn.Linear(1024, 1024)
        input_tensor = torch.randn(16, 1024)

        strategy_results = {}

        for strategy in strategies:
            config = DistributedCheckpointConfig(strategy=strategy)
            checkpointing = DistributedActivationCheckpointing(config)

            def strategy_checkpoint():
                return checkpointing.checkpoint_layer_distributed(
                    layer, input_tensor, layer_id=f"strategy_test_{strategy.value}"
                )

            result = self.run_benchmark(
                strategy_checkpoint, f"strategy_{strategy.value}"
            )

            strategy_results[strategy.value] = result.execution_time_ms

        # Print strategy comparison
        print("Strategy performance comparison:")
        for strategy_name, time_ms in strategy_results.items():
            print(f"  {strategy_name}: {time_ms:.2f}ms")

        # All strategies should complete in reasonable time
        for time_ms in strategy_results.values():
            assert time_ms < 1000.0, f"Strategy too slow: {time_ms}ms"


class MemoryScalingBenchmarks(BenchmarkSuite):
    """Benchmarks for memory scaling analysis."""

    def test_memory_scaling_with_model_size(self):
        """Test memory scaling with increasing model size."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        # Different model configurations
        model_configs = [
            (512, 512, 8),  # Small
            (1024, 1024, 16),  # Medium
            (2048, 2048, 32),  # Large
            (4096, 4096, 64),  # Very Large
        ]

        scaling_results = []

        for hidden_size, ff_size, batch_size in model_configs:
            # Create model layers
            attention_layer = nn.MultiheadAttention(
                hidden_size, num_heads=8, batch_first=True
            )
            ff_layer = nn.Sequential(
                nn.Linear(hidden_size, ff_size),
                nn.ReLU(),
                nn.Linear(ff_size, hidden_size),
            )

            input_tensor = torch.randn(batch_size, 128, hidden_size)  # seq_len=128

            def model_forward():
                # Attention
                attn_out = checkpointing.checkpoint_layer_distributed(
                    lambda x: attention_layer(x, x, x)[0],
                    input_tensor,
                    layer_id=f"attention_{hidden_size}",
                )

                # Feed-forward
                ff_out = checkpointing.checkpoint_layer_distributed(
                    ff_layer, attn_out, layer_id=f"ff_{hidden_size}"
                )

                return ff_out

            result = self.run_benchmark(model_forward, f"model_size_{hidden_size}")

            scaling_results.append(
                {
                    "hidden_size": hidden_size,
                    "batch_size": batch_size,
                    "time_ms": result.execution_time_ms,
                    "memory_mb": result.memory_usage_mb,
                    "peak_memory_mb": result.peak_memory_mb,
                }
            )

        # Print scaling results
        print("Memory scaling results:")
        for result in scaling_results:
            print(
                f"  Size {result['hidden_size']}: "
                f"{result['time_ms']:.2f}ms, "
                f"{result['memory_mb']:.2f}MB, "
                f"Peak: {result['peak_memory_mb']:.2f}MB"
            )

        # Memory should scale reasonably with model size
        memory_values = [r["memory_mb"] for r in scaling_results]
        for i in range(1, len(memory_values)):
            ratio = memory_values[i] / memory_values[i - 1]
            # Should not grow too aggressively (less than 10x per size increase)
            assert ratio < 10.0, f"Memory scaling too aggressive: {ratio}x"

    def test_sequence_length_scaling(self):
        """Test memory scaling with increasing sequence length."""
        config = DistributedCheckpointConfig()
        checkpointing = DistributedActivationCheckpointing(config)

        layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, batch_first=True
        )

        sequence_lengths = [128, 256, 512, 1024]
        batch_size = 8

        seq_results = []

        for seq_len in sequence_lengths:
            input_tensor = torch.randn(batch_size, seq_len, 512)

            def transformer_forward():
                return checkpointing.checkpoint_layer_distributed(
                    layer, input_tensor, layer_id=f"transformer_seq_{seq_len}"
                )

            result = self.run_benchmark(transformer_forward, f"seq_len_{seq_len}")

            seq_results.append(
                {
                    "seq_len": seq_len,
                    "time_ms": result.execution_time_ms,
                    "memory_mb": result.memory_usage_mb,
                }
            )

        # Print sequence scaling results
        print("Sequence length scaling results:")
        for result in seq_results:
            print(
                f"  Seq len {result['seq_len']}: "
                f"{result['time_ms']:.2f}ms, "
                f"{result['memory_mb']:.2f}MB"
            )


class CommunicationBenchmarks(BenchmarkSuite):
    """Benchmarks for communication coordination."""

    def test_communication_cost_estimation(self):
        """Test communication cost estimation accuracy."""
        patterns = [
            CommunicationPattern.ALL_REDUCE,
            CommunicationPattern.ALL_GATHER,
            CommunicationPattern.BROADCAST,
            CommunicationPattern.REDUCE_SCATTER,
        ]

        data_sizes = [1024, 4096, 16384, 65536]  # bytes
        num_participants = 4

        for pattern in patterns:
            pattern_results = []

            for data_size in data_sizes:
                # Estimate communication cost
                def estimate_cost():
                    return estimate_communication_cost(
                        layer_size_bytes=data_size,
                        pattern=pattern,
                        num_participants=num_participants,
                    )

                result = self.run_benchmark(
                    estimate_cost, f"comm_cost_{pattern.value}_{data_size}"
                )

                pattern_results.append(
                    {
                        "data_size": data_size,
                        "estimated_time_s": estimate_cost(),
                        "compute_time_ms": result.execution_time_ms,
                    }
                )

            print(f"Communication pattern {pattern.value}:")
            for result in pattern_results:
                print(
                    f"  Size {result['data_size']}: "
                    f"{result['estimated_time_s']*1000:.3f}ms estimated, "
                    f"{result['compute_time_ms']:.3f}ms to compute"
                )

    @patch("rosellm.rosetrainer.parallelism.parallel_state.is_initialized")
    @patch("torch.distributed.is_initialized")
    def test_coordination_overhead(self, mock_dist_init, mock_parallel_init):
        """Test coordination overhead in multi-rank scenarios."""
        mock_parallel_init.return_value = True
        mock_dist_init.return_value = True

        config = DistributedCheckpointConfig()

        with patch("torch.distributed.get_world_size", return_value=4), patch(
            "torch.distributed.get_rank", return_value=0
        ):

            checkpointing = DistributedActivationCheckpointing(config)

            layer = nn.Linear(512, 512)
            input_tensor = torch.randn(8, 512)

            def coordinated_checkpoint():
                return checkpointing.checkpoint_layer_distributed(
                    layer, input_tensor, layer_id="coordination_test"
                )

            result = self.run_benchmark(
                coordinated_checkpoint, "coordinated_multi_rank"
            )

            print(f"Coordination overhead: {result.execution_time_ms:.2f}ms")

            # Coordination should not add excessive overhead
            assert result.execution_time_ms < 500.0


class ModelParallelBenchmarks(BenchmarkSuite):
    """Benchmarks for model parallel checkpointing."""

    def test_model_parallel_manager_overhead(self):
        """Test overhead of model parallel activation manager."""
        config = ModelParallelCheckpointConfig()
        manager = ModelParallelActivationManager(config)

        # Test different layer types
        layers_to_test = [
            (
                "column_parallel",
                lambda: manager.create_column_parallel_layer(512, 1024),
            ),
            ("row_parallel", lambda: manager.create_row_parallel_layer(1024, 512)),
            ("attention", lambda: manager.create_attention_layer(512, 8)),
            ("mlp", lambda: manager.create_mlp_layer(512, 2048)),
        ]

        for layer_name, layer_factory in layers_to_test:

            def create_and_use_layer():
                layer = layer_factory()

                # Create appropriate input
                if layer_name == "attention":
                    input_tensor = torch.randn(4, 64, 512)  # batch, seq, hidden
                elif layer_name == "column_parallel":
                    input_tensor = torch.randn(4, 512)  # input_size=512
                elif layer_name == "row_parallel":
                    input_tensor = torch.randn(4, 1024)  # input_size=1024
                elif layer_name == "mlp":
                    input_tensor = torch.randn(4, 512)  # input_size=512
                else:
                    input_tensor = torch.randn(4, 512)  # default

                return layer(input_tensor)

            result = self.run_benchmark(create_and_use_layer, f"mp_{layer_name}")

            print(
                f"{layer_name}: {result.execution_time_ms:.2f}ms, {result.memory_usage_mb:.2f}MB"
            )

    def test_tensor_parallel_scaling(self):
        """Test scaling with different tensor parallel sizes."""
        # Simulate different TP sizes
        tp_sizes = [1, 2, 4, 8]

        for tp_size in tp_sizes:
            config = ModelParallelCheckpointConfig()

            # Mock tensor parallel size
            with patch(
                "rosellm.rosetrainer.parallelism.parallel_state.get_tensor_model_parallel_size",
                return_value=tp_size,
            ):
                manager = ModelParallelActivationManager(config)

                def tp_layer_test():
                    # Create layers that would be affected by TP
                    layer = manager.create_attention_layer(512, 8)
                    input_tensor = torch.randn(4, 64, 512)
                    return layer(input_tensor)

                result = self.run_benchmark(tp_layer_test, f"tp_size_{tp_size}")

                print(f"TP size {tp_size}: {result.execution_time_ms:.2f}ms")


class DistributedMemoryOptimizerBenchmarks(BenchmarkSuite):
    """Benchmarks for distributed memory optimizer."""

    def test_memory_optimizer_integration_overhead(self):
        """Test overhead of memory optimizer integration."""
        # Create test model
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        optimizer = torch.optim.Adam(model.parameters())

        # Test without memory optimization
        def baseline_forward():
            input_tensor = torch.randn(16, 256)
            # Check if model uses half precision and convert input accordingly
            first_param = next(model.parameters())
            if first_param.dtype == torch.float16:
                input_tensor = input_tensor.half()
            return model(input_tensor)

        baseline_result = self.run_benchmark(baseline_forward, "baseline_model")

        # Test with memory optimization
        config = DistributedMemoryConfig()
        memory_optimizer = DistributedMemoryOptimizer(model, optimizer, config)
        optimized_model = memory_optimizer.integrate_with_model()

        def optimized_forward():
            input_tensor = torch.randn(16, 256)
            # Check if model uses half precision and convert input accordingly
            first_param = next(optimized_model.parameters())
            if first_param.dtype == torch.float16:
                input_tensor = input_tensor.half()
            return optimized_model(input_tensor)

        optimized_result = self.run_benchmark(optimized_forward, "optimized_model")

        # Compare results
        overhead_ratio = (
            optimized_result.execution_time_ms / baseline_result.execution_time_ms
        )

        print(f"Memory optimizer overhead: {overhead_ratio:.2f}x")
        print(f"Baseline: {baseline_result.execution_time_ms:.2f}ms")
        print(f"Optimized: {optimized_result.execution_time_ms:.2f}ms")

        # Overhead should be reasonable
        assert overhead_ratio < 5.0, f"Overhead too high: {overhead_ratio}x"

    def test_optimization_step_performance(self):
        """Test performance of optimization steps."""
        model = nn.Linear(1024, 1024)
        optimizer = torch.optim.AdamW(model.parameters())

        config = DistributedMemoryConfig()
        memory_optimizer = DistributedMemoryOptimizer(model, optimizer, config)

        # Test optimization step performance
        step_times = []

        for step in range(10):

            def optimization_step():
                return memory_optimizer.optimize_step(step)

            result = self.run_benchmark(optimization_step, f"opt_step_{step}")
            step_times.append(result.execution_time_ms)

        avg_step_time = statistics.mean(step_times)
        print(f"Average optimization step time: {avg_step_time:.2f}ms")

        # Steps should complete quickly
        assert avg_step_time < 100.0, f"Optimization steps too slow: {avg_step_time}ms"


class LargeModelBenchmarks(BenchmarkSuite):
    """Benchmarks for large model scenarios."""

    @pytest.mark.slow
    def test_large_transformer_benchmark(self):
        """Benchmark with large transformer-like model."""
        from rosellm.rosetrainer.memory.distributed_transformer import (
            create_distributed_transformer,
        )

        # Create large transformer
        model = create_distributed_transformer(
            vocab_size=32000,
            d_model=1024,
            nhead=16,
            num_layers=6,  # Reduced for testing
            dim_feedforward=4096,
            max_length=2048,
        )

        # Test input
        input_ids = torch.randint(0, 32000, (4, 512))  # batch=4, seq_len=512

        def large_model_forward():
            return model.encode(input_ids)

        result = self.run_benchmark(large_model_forward, "large_transformer")

        print(
            f"Large transformer: {result.execution_time_ms:.2f}ms, "
            f"{result.memory_usage_mb:.2f}MB, "
            f"Peak: {result.peak_memory_mb:.2f}MB"
        )

        # Should complete in reasonable time
        assert result.execution_time_ms < 10000.0  # 10 seconds max

    @pytest.mark.slow
    def test_memory_efficiency_large_scale(self):
        """Test memory efficiency at large scale."""
        config = DistributedCheckpointConfig(
            strategy=DistributedCheckpointStrategy.ADAPTIVE,
            enable_load_balancing=True,
        )
        checkpointing = DistributedActivationCheckpointing(config)

        # Create multiple large layers
        layers = [nn.Linear(2048, 2048) for _ in range(20)]
        input_tensor = torch.randn(32, 2048)

        def large_scale_forward():
            x = input_tensor
            for i, layer in enumerate(layers):
                x = checkpointing.checkpoint_layer_distributed(
                    layer, x, layer_id=f"large_layer_{i}"
                )
            return x

        result = self.run_benchmark(large_scale_forward, "large_scale_model")

        print(
            f"Large scale model: {result.execution_time_ms:.2f}ms, "
            f"Memory: {result.memory_usage_mb:.2f}MB"
        )

        # Get profiling report
        report = checkpointing.get_distributed_profiling_report()

        print("Profiling summary:")
        if "distributed_checkpointing" in report:
            dc_report = report["distributed_checkpointing"]
            print(f"  Steps: {dc_report.get('step_count', 'N/A')}")
            if "memory_profiling" in dc_report:
                mem_report = dc_report["memory_profiling"]
                if "global_memory_stats" in mem_report:
                    print(f"  Memory efficiency: {mem_report['global_memory_stats']}")


# Pytest fixtures for benchmarks
@pytest.fixture(scope="class")
def benchmark_suite():
    """Fixture providing benchmark suite."""
    return BenchmarkSuite()


# Benchmark test collection
class TestBenchmarkSuite:
    """Main benchmark test collection."""

    def test_run_checkpointing_overhead_benchmarks(self):
        """Run checkpointing overhead benchmarks."""
        benchmarks = CheckpointingOverheadBenchmarks()
        benchmarks.test_basic_checkpointing_overhead()
        benchmarks.test_strategy_comparison_overhead()

        summary = benchmarks.get_summary()
        print("Checkpointing overhead benchmark summary:")
        print(
            f"  Mean execution time: {summary['execution_time_stats']['mean_ms']:.2f}ms"
        )
        print(f"  Mean memory usage: {summary['memory_stats']['mean_mb']:.2f}MB")

    def test_run_memory_scaling_benchmarks(self):
        """Run memory scaling benchmarks."""
        benchmarks = MemoryScalingBenchmarks()
        benchmarks.test_memory_scaling_with_model_size()
        benchmarks.test_sequence_length_scaling()

        summary = benchmarks.get_summary()
        print("Memory scaling benchmark summary:")
        print(f"  Total benchmarks: {summary['total_benchmarks']}")
        print(f"  Max memory usage: {summary['memory_stats']['max_mb']:.2f}MB")

    def test_run_communication_benchmarks(self):
        """Run communication benchmarks."""
        benchmarks = CommunicationBenchmarks()
        benchmarks.test_communication_cost_estimation()
        benchmarks.test_coordination_overhead()

        summary = benchmarks.get_summary()
        print("Communication benchmark summary:")
        print(
            f"  Mean coordination time: {summary['execution_time_stats']['mean_ms']:.2f}ms"
        )

    def test_run_model_parallel_benchmarks(self):
        """Run model parallel benchmarks."""
        benchmarks = ModelParallelBenchmarks()
        benchmarks.test_model_parallel_manager_overhead()
        benchmarks.test_tensor_parallel_scaling()

        summary = benchmarks.get_summary()
        print("Model parallel benchmark summary:")
        print(f"  Components tested: {summary['total_benchmarks']}")

    def test_run_memory_optimizer_benchmarks(self):
        """Run distributed memory optimizer benchmarks."""
        benchmarks = DistributedMemoryOptimizerBenchmarks()
        benchmarks.test_memory_optimizer_integration_overhead()
        benchmarks.test_optimization_step_performance()

        summary = benchmarks.get_summary()
        print("Memory optimizer benchmark summary:")
        print(f"  Average overhead: {summary['execution_time_stats']['mean_ms']:.2f}ms")

    @pytest.mark.slow
    def test_run_large_model_benchmarks(self):
        """Run large model benchmarks (slow)."""
        benchmarks = LargeModelBenchmarks()
        benchmarks.test_large_transformer_benchmark()
        benchmarks.test_memory_efficiency_large_scale()

        summary = benchmarks.get_summary()
        print("Large model benchmark summary:")
        print(f"  Peak memory usage: {summary['memory_stats']['max_mb']:.2f}MB")
        print(
            f"  Max execution time: {summary['execution_time_stats']['max_ms']:.2f}ms"
        )


def run_comprehensive_benchmarks():
    """Run all benchmarks and generate comprehensive report."""
    print("=" * 80)
    print("COMPREHENSIVE DISTRIBUTED CHECKPOINTING BENCHMARKS")
    print("=" * 80)

    # Initialize benchmark categories
    benchmark_categories = [
        ("Checkpointing Overhead", CheckpointingOverheadBenchmarks()),
        ("Memory Scaling", MemoryScalingBenchmarks()),
        ("Communication", CommunicationBenchmarks()),
        ("Model Parallel", ModelParallelBenchmarks()),
        ("Memory Optimizer", DistributedMemoryOptimizerBenchmarks()),
    ]

    # Run benchmarks
    all_results = {}

    for category_name, benchmark_suite in benchmark_categories:
        print(f"\nRunning {category_name} benchmarks...")

        # Run category-specific benchmarks
        if hasattr(benchmark_suite, "test_basic_checkpointing_overhead"):
            benchmark_suite.test_basic_checkpointing_overhead()
        if hasattr(benchmark_suite, "test_memory_scaling_with_model_size"):
            benchmark_suite.test_memory_scaling_with_model_size()
        if hasattr(benchmark_suite, "test_communication_cost_estimation"):
            benchmark_suite.test_communication_cost_estimation()
        if hasattr(benchmark_suite, "test_model_parallel_manager_overhead"):
            benchmark_suite.test_model_parallel_manager_overhead()
        if hasattr(benchmark_suite, "test_memory_optimizer_integration_overhead"):
            benchmark_suite.test_memory_optimizer_integration_overhead()

        summary = benchmark_suite.get_summary()
        all_results[category_name] = summary

    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 80)

    for category_name, results in all_results.items():
        print(f"\n{category_name}:")
        if "error" not in results:
            print(f"  Benchmarks run: {results['total_benchmarks']}")
            print(
                f"  Mean execution time: {results['execution_time_stats']['mean_ms']:.2f}ms"
            )
            print(f"  Mean memory usage: {results['memory_stats']['mean_mb']:.2f}MB")

            # Best and worst performing
            individual = results.get("individual_results", [])
            if individual:
                fastest = min(individual, key=lambda x: x["time_ms"])
                slowest = max(individual, key=lambda x: x["time_ms"])
                print(f"  Fastest: {fastest['name']} ({fastest['time_ms']:.2f}ms)")
                print(f"  Slowest: {slowest['name']} ({slowest['time_ms']:.2f}ms)")
        else:
            print(f"  Error: {results['error']}")

    print("\n" + "=" * 80)
    print("Benchmark run completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Run comprehensive benchmarks
    run_comprehensive_benchmarks()
