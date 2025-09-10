"""Range-Based Parameter Buffer Mapping Memory Efficiency Example.

This example demonstrates the memory efficiency gains achieved through
range-based parameter buffer mapping in distributed training scenarios.

The example compares standard parameter management with range-based
optimization across different model sizes and distributed configurations.
"""

import argparse
import logging
import time
from datetime import timedelta
from typing import Dict, List, Optional, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.multiprocessing.spawn import spawn

from rosellm.rosetrainer.optimizer.config import DistributedOptimizerConfig
from rosellm.rosetrainer.optimizer.distributed_optimizer import DistributedOptimizer
from rosellm.rosetrainer.optimizer.range_buffer_mapping import (
    RangeBufferConfig,
    RangeBufferStrategy,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BenchmarkModel(nn.Module):
    """Configurable model for memory efficiency benchmarking."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 10,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, dtype=dtype))
            layers.append(nn.BatchNorm1d(hidden_dim, dtype=dtype))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim, dtype=dtype))

        self.network = nn.Sequential(*layers)

        # Store model info
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.total_params = sum(p.numel() for p in self.parameters())

        logger.info(f"Created model with {self.total_params:,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def setup_distributed_environment(rank: int, world_size: int, backend: str = "gloo"):
    """Setup distributed training environment."""
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(12355 + rank)  # Unique port per process

    try:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=30),
        )
        logger.info(f"Rank {rank}/{world_size} initialized with backend {backend}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize distributed environment: {e}")
        return False


def cleanup_distributed_environment():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


class MemoryBenchmarkRunner:
    """Memory efficiency benchmark runner."""

    def __init__(self, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.results: List[Dict] = []

    def benchmark_configuration(
        self,
        model: BenchmarkModel,
        config_name: str,
        optimizer_config: DistributedOptimizerConfig,
        range_buffer_config: Optional[RangeBufferConfig] = None,
        num_steps: int = 10,
        batch_size: int = 32,
    ) -> Dict:
        """Benchmark a specific configuration."""
        logger.info(f"Rank {self.rank}: Benchmarking configuration '{config_name}'")

        # Create optimizer
        optimizer = DistributedOptimizer(
            params=model.parameters(),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.001, "weight_decay": 1e-4},
            config=optimizer_config,
            range_buffer_config=range_buffer_config,
        )

        # Warm-up phase
        self._run_training_steps(model, optimizer, num_steps=2, batch_size=batch_size)

        # Measure memory before main benchmark
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0

        # Main benchmark
        start_time = time.time()
        self._run_training_steps(model, optimizer, num_steps, batch_size)
        elapsed_time = time.time() - start_time

        # Measure memory after benchmark
        if torch.cuda.is_available() and self.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated()
            final_memory = torch.cuda.memory_allocated()
            memory_usage = {
                "initial_mb": initial_memory / (1024**2),
                "peak_mb": peak_memory / (1024**2),
                "final_mb": final_memory / (1024**2),
            }
        else:
            memory_usage = {"initial_mb": 0, "peak_mb": 0, "final_mb": 0}

        # Get optimizer memory statistics
        optimizer_memory = optimizer.get_memory_usage()

        result = {
            "config_name": config_name,
            "rank": self.rank,
            "world_size": self.world_size,
            "elapsed_time": elapsed_time,
            "time_per_step": elapsed_time / num_steps,
            "device_memory": memory_usage,
            "optimizer_memory": optimizer_memory,
            "model_params": model.total_params if hasattr(model, "total_params") else 0,
        }

        self.results.append(result)
        logger.info(f"Rank {self.rank}: Configuration '{config_name}' completed")

        return result

    def _run_training_steps(
        self, model: BenchmarkModel, optimizer, num_steps: int, batch_size: int
    ):
        """Run training steps for benchmarking."""
        model.train()

        for step in range(num_steps):
            # Generate batch data
            torch.manual_seed(
                step * self.world_size + self.rank
            )  # Deterministic per rank
            x = torch.randn(batch_size, int(model.input_dim), device=self.device)
            y_true = torch.randint(
                0, int(model.output_dim), (batch_size,), device=self.device
            )

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x)
            loss = nn.functional.cross_entropy(y_pred, y_true)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Synchronize for timing accuracy
            if dist.is_initialized():
                dist.barrier()

    def print_results_summary(self):
        """Print summary of benchmark results."""
        if self.rank == 0:  # Only print from rank 0
            logger.info("=" * 80)
            logger.info("MEMORY EFFICIENCY BENCHMARK RESULTS")
            logger.info("=" * 80)

            for result in self.results:
                logger.info(f"\nConfiguration: {result['config_name']}")
                logger.info(f"  Time per step: {result['time_per_step']:.4f}s")

                if result["device_memory"]["peak_mb"] > 0:
                    logger.info(
                        f"  Peak GPU memory: "
                        f"{result['device_memory']['peak_mb']:.2f} MB"
                    )

                opt_mem = result["optimizer_memory"]
                logger.info(
                    f"  Parameter memory: {opt_mem.get('parameters_mb', 0):.2f} MB"
                )
                logger.info(
                    f"  Gradient memory: {opt_mem.get('gradients_mb', 0):.2f} MB"
                )
                logger.info(
                    f"  Optimizer states: "
                    f"{opt_mem.get('optimizer_states_mb', 0):.2f} MB"
                )

                if "range_buffer_allocated_mb" in opt_mem:
                    logger.info(
                        f"  Range buffer memory: "
                        f"{opt_mem['range_buffer_allocated_mb']:.2f} MB"
                    )
                    logger.info(
                        f"  Buffer fragmentation: "
                        f"{opt_mem.get('range_buffer_fragmentation', 0):.2%}"
                    )
                    logger.info(
                        f"  Gradient buckets: {opt_mem.get('num_gradient_buckets', 0)}"
                    )

                logger.info(
                    f"  Total optimizer memory: {opt_mem.get('total_mb', 0):.2f} MB"
                )

    def compare_configurations(self):
        """Compare different configurations and highlight improvements."""
        if self.rank != 0 or len(self.results) < 2:
            return

        logger.info("\n" + "=" * 80)
        logger.info("CONFIGURATION COMPARISON")
        logger.info("=" * 80)

        baseline = next(
            (r for r in self.results if "standard" in r["config_name"].lower()), None
        )
        if not baseline:
            baseline = self.results[0]

        for result in self.results:
            if result == baseline:
                continue

            config_name = result["config_name"]

            # Time comparison
            time_improvement = (
                baseline["time_per_step"] - result["time_per_step"]
            ) / baseline["time_per_step"]

            # Memory comparison
            baseline_total_mem = baseline["optimizer_memory"].get("total_mb", 0)
            result_total_mem = result["optimizer_memory"].get("total_mb", 0)

            if baseline_total_mem > 0:
                memory_improvement = (
                    baseline_total_mem - result_total_mem
                ) / baseline_total_mem
            else:
                memory_improvement = 0

            # Fragmentation comparison (lower is better)
            # baseline_frag = baseline["optimizer_memory"].get(
            #     "range_buffer_fragmentation", 0
            # )
            result_frag = result["optimizer_memory"].get(
                "range_buffer_fragmentation", 0
            )

            logger.info(f"\n{config_name} vs Baseline:")
            logger.info(f"  Time improvement: {time_improvement:+.2%}")
            logger.info(f"  Memory improvement: {memory_improvement:+.2%}")

            if result_frag > 0:
                logger.info(f"  Fragmentation: {result_frag:.2%}")


def run_memory_efficiency_benchmark(
    rank: int,
    world_size: int,
    model_size: str,
    backend: str,
    num_steps: int,
    batch_size: int,
):
    """Run memory efficiency benchmark on a specific rank."""
    # Setup distributed environment
    if not setup_distributed_environment(rank, world_size, backend):
        return

    try:
        # Determine device
        device = torch.device(
            "cuda" if torch.cuda.is_available() and backend == "nccl" else "cpu"
        )
        logger.info(f"Rank {rank}: Using device {device}")

        # Create model based on size
        model_configs = {
            "small": {
                "input_dim": 128,
                "hidden_dims": [256, 512, 256],
                "output_dim": 10,
            },
            "medium": {
                "input_dim": 256,
                "hidden_dims": [512, 1024, 512, 256],
                "output_dim": 10,
            },
            "large": {
                "input_dim": 512,
                "hidden_dims": [1024, 2048, 1024, 512, 256],
                "output_dim": 10,
            },
            "xlarge": {
                "input_dim": 1024,
                "hidden_dims": [2048, 4096, 2048, 1024, 512],
                "output_dim": 10,
            },
        }

        config = model_configs[model_size]
        model = BenchmarkModel(
            input_dim=cast(int, config["input_dim"]),
            hidden_dims=cast(List[int], config["hidden_dims"]),
            output_dim=cast(int, config["output_dim"]),
            dropout=0.1,
        )
        model.to(device)

        # Create benchmark runner
        benchmark_runner = MemoryBenchmarkRunner(rank, world_size, device)

        # Configuration 1: Standard distributed optimizer
        standard_config = DistributedOptimizerConfig(
            partition_parameters=True,
            contiguous_gradients=True,
            reduce_bucket_size_mb=25,
        )

        benchmark_runner.benchmark_configuration(
            model=model,
            config_name="Standard Distributed",
            optimizer_config=standard_config,
            num_steps=num_steps,
            batch_size=batch_size,
        )

        # Configuration 2: Range buffer with contiguous strategy
        range_config_contiguous = RangeBufferConfig(
            strategy=RangeBufferStrategy.CONTIGUOUS,
            device=device,
            enable_profiling=True,
        )

        benchmark_runner.benchmark_configuration(
            model=model,
            config_name="Range Buffer (Contiguous)",
            optimizer_config=standard_config,
            range_buffer_config=range_config_contiguous,
            num_steps=num_steps,
            batch_size=batch_size,
        )

        # Configuration 3: Range buffer with size-ordered strategy
        range_config_size = RangeBufferConfig(
            strategy=RangeBufferStrategy.SIZE_ORDERED,
            device=device,
            enable_profiling=True,
        )

        benchmark_runner.benchmark_configuration(
            model=model,
            config_name="Range Buffer (Size Ordered)",
            optimizer_config=standard_config,
            range_buffer_config=range_config_size,
            num_steps=num_steps,
            batch_size=batch_size,
        )

        # Configuration 4: Range buffer with dtype grouping
        range_config_dtype = RangeBufferConfig(
            strategy=RangeBufferStrategy.DTYPE_GROUPED,
            device=device,
            enable_profiling=True,
        )

        benchmark_runner.benchmark_configuration(
            model=model,
            config_name="Range Buffer (Dtype Grouped)",
            optimizer_config=standard_config,
            range_buffer_config=range_config_dtype,
            num_steps=num_steps,
            batch_size=batch_size,
        )

        # Configuration 5: Range buffer with optimized settings
        optimized_config = DistributedOptimizerConfig(
            partition_parameters=True,
            contiguous_gradients=True,
            reduce_bucket_size_mb=50,  # Larger buckets
        )

        range_config_optimized = RangeBufferConfig(
            strategy=RangeBufferStrategy.CONTIGUOUS,
            alignment_bytes=128,  # Better alignment
            min_range_size=512,  # Larger minimum range
            max_fragmentation=0.05,  # Lower fragmentation threshold
            enable_compaction=True,
            device=device,
            enable_profiling=True,
        )

        benchmark_runner.benchmark_configuration(
            model=model,
            config_name="Range Buffer (Optimized)",
            optimizer_config=optimized_config,
            range_buffer_config=range_config_optimized,
            num_steps=num_steps,
            batch_size=batch_size,
        )

        # Synchronize all ranks before printing results
        if dist.is_initialized():
            dist.barrier()

        # Print results and comparison
        benchmark_runner.print_results_summary()
        benchmark_runner.compare_configurations()

    except Exception as e:
        logger.error(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        cleanup_distributed_environment()


def main():
    """Main function to run the memory efficiency demonstration."""
    parser = argparse.ArgumentParser(
        description="Range Buffer Memory Efficiency Benchmark"
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of distributed ranks (default: 2)",
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large", "xlarge"],
        default="medium",
        help="Model size for benchmarking (default: medium)",
    )
    parser.add_argument(
        "--backend",
        choices=["gloo", "nccl"],
        default="gloo",
        help="Distributed backend (default: gloo)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of training steps per configuration (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )

    args = parser.parse_args()

    logger.info("Starting Range Buffer Memory Efficiency Benchmark")
    logger.info(f"Configuration: {args}")

    # Check if distributed is available
    if not torch.distributed.is_available():
        logger.error("Distributed training not available")
        return

    # Run benchmark across multiple processes
    try:
        spawn(
            run_memory_efficiency_benchmark,
            args=(
                args.world_size,
                args.model_size,
                args.backend,
                args.num_steps,
                args.batch_size,
            ),
            nprocs=args.world_size,
            join=True,
        )

        logger.info("Memory efficiency benchmark completed successfully!")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
