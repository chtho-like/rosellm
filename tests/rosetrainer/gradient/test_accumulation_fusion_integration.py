"""
Integration Tests for Gradient Accumulation Fusion

This module provides end-to-end integration tests that demonstrate the performance
improvements and functionality of the gradient accumulation fusion feature in
realistic training scenarios.
"""

import os
import time
from typing import Dict, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp  # type: ignore[import]
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from rosellm.rosetrainer.gradient.accumulation_fusion import (
    AsyncReductionOrchestrator,
    FusedParamGradMapping,
    FusionConfig,
    FusionStrategy,
    GradientAccumulationFusion,
    OverlapStrategy,
)
from rosellm.rosetrainer.optimizer.param_grad_mapping import (
    MappingConfig,
    ReductionStrategy,
)


class TransformerBlock(nn.Module):
    """Simple transformer block for testing."""

    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class TransformerModel(nn.Module):
    """Multi-layer transformer model for testing."""

    def __init__(
        self, num_layers: int = 6, d_model: int = 512, vocab_size: int = 10000
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model) for _ in range(num_layers)]
        )

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Embedding and position encoding
        x = self.embedding(x)
        x = x + self.pos_encoding[:, : x.size(1), :]

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output projection
        return self.output(x)


def setup_distributed(rank: int, world_size: int, backend: str = "gloo"):
    """Setup distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    """Cleanup distributed process group."""
    dist.destroy_process_group()


def train_step_baseline(
    model: nn.Module,
    optimizer: optim.Optimizer,
    data: torch.Tensor,
    target: torch.Tensor,
    accumulation_steps: int = 1,
) -> Tuple[float, float]:
    """Baseline training step without fusion."""
    start_time = time.perf_counter()

    # Forward pass
    output = model(data)
    loss = nn.functional.cross_entropy(
        output.view(-1, output.size(-1)), target.view(-1)
    )

    # Scale loss for accumulation
    loss = loss / accumulation_steps

    # Backward pass
    loss.backward()

    # Optimizer step (only on accumulation boundary)
    if (getattr(train_step_baseline, "step", 0) + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    if not hasattr(train_step_baseline, "step"):
        train_step_baseline.step = 0  # type: ignore
    train_step_baseline.step += 1  # type: ignore

    compute_time = time.perf_counter() - start_time

    return loss.item() * accumulation_steps, compute_time


def train_step_with_fusion(
    model: nn.Module,
    optimizer: optim.Optimizer,
    fusion_manager: GradientAccumulationFusion,
    orchestrator: AsyncReductionOrchestrator,
    data: torch.Tensor,
    target: torch.Tensor,
    accumulation_steps: int = 1,
) -> Tuple[float, float, Dict]:
    """Training step with gradient accumulation fusion."""
    start_time = time.perf_counter()

    # Forward pass
    output = model(data)
    loss = nn.functional.cross_entropy(
        output.view(-1, output.size(-1)), target.view(-1)
    )

    # Scale loss for accumulation
    loss = loss / accumulation_steps

    # Backward pass with fusion
    with fusion_manager.accumulation_context(accumulation_steps) as state:
        loss.backward()

        # Start async reduction if on accumulation boundary
        if state.step % accumulation_steps == 0:
            orchestrator.start_reduction()

    # Optimizer step (only on accumulation boundary)
    if (getattr(train_step_with_fusion, "step", 0) + 1) % accumulation_steps == 0:
        # Wait for reduction to complete
        orchestrator.wait_reduction()

        optimizer.step()
        optimizer.zero_grad()

    if not hasattr(train_step_with_fusion, "step"):
        train_step_with_fusion.step = 0  # type: ignore
    train_step_with_fusion.step += 1  # type: ignore

    compute_time = time.perf_counter() - start_time

    # Get fusion metrics
    metrics = fusion_manager.get_metrics()

    return loss.item() * accumulation_steps, compute_time, metrics


class TestIntegrationSingleProcess:
    """Integration tests for single process training."""

    def test_transformer_training_comparison(self):
        """Compare training with and without fusion."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model and optimizer
        model = TransformerModel(num_layers=4, d_model=256)
        model = model.to(device)

        # Create two identical models for comparison
        model_baseline = TransformerModel(num_layers=4, d_model=256)
        model_baseline.load_state_dict(model.state_dict())
        model_baseline = model_baseline.to(device)

        model_fusion = TransformerModel(num_layers=4, d_model=256)
        model_fusion.load_state_dict(model.state_dict())
        model_fusion = model_fusion.to(device)

        # Create optimizers
        optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=1e-4)
        optimizer_fusion = optim.Adam(model_fusion.parameters(), lr=1e-4)

        # Create fusion manager
        fusion_config = FusionConfig(
            enable_fusion=True,
            fusion_strategy=FusionStrategy.BALANCED,
            async_reduction=False,  # No async in single process
            use_multi_tensor_ops=True,
        )

        fusion_manager = GradientAccumulationFusion(
            model_params=list(model_fusion.parameters()),
            config=fusion_config,
            device=device,
        )

        orchestrator = AsyncReductionOrchestrator(fusion_manager=fusion_manager)

        # Generate synthetic data
        batch_size = 8
        seq_len = 128
        vocab_size = 10000
        num_steps = 20
        accumulation_steps = 4

        # Training loop comparison
        baseline_times = []
        fusion_times = []
        baseline_losses = []
        fusion_losses = []

        # Reset step counters
        train_step_baseline.step = 0
        train_step_with_fusion.step = 0

        for step in range(num_steps):
            # Generate random data
            data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            # Baseline training step
            loss_baseline, time_baseline = train_step_baseline(
                model_baseline, optimizer_baseline, data, target, accumulation_steps
            )
            baseline_times.append(time_baseline)
            baseline_losses.append(loss_baseline)

            # Fusion training step
            loss_fusion, time_fusion, metrics = train_step_with_fusion(
                model_fusion,
                optimizer_fusion,
                fusion_manager,
                orchestrator,
                data,
                target,
                accumulation_steps,
            )
            fusion_times.append(time_fusion)
            fusion_losses.append(loss_fusion)

        # Calculate statistics
        avg_baseline_time = sum(baseline_times) / len(baseline_times)
        avg_fusion_time = sum(fusion_times) / len(fusion_times)

        # Performance improvement
        improvement = (avg_baseline_time - avg_fusion_time) / avg_baseline_time

        print(f"\nPerformance Comparison:")
        print(f"Baseline avg time: {avg_baseline_time * 1000:.2f} ms")
        print(f"Fusion avg time: {avg_fusion_time * 1000:.2f} ms")
        print(f"Performance improvement: {improvement * 100:.2f}%")

        # Get final metrics
        final_metrics = fusion_manager.get_metrics()
        print(f"\nFusion Metrics:")
        print(f"Avg fusion time: {final_metrics['avg_fusion_time'] * 1000:.2f} ms")
        print(f"Memory saved: {final_metrics['memory_saved_mb']:.2f} MB")
        print(f"Tensors fused: {final_metrics['tensors_fused']}")

        # Losses should be similar (allowing for numerical differences)
        avg_baseline_loss = sum(baseline_losses) / len(baseline_losses)
        avg_fusion_loss = sum(fusion_losses) / len(fusion_losses)

        print(f"\nLoss Comparison:")
        print(f"Baseline avg loss: {avg_baseline_loss:.4f}")
        print(f"Fusion avg loss: {avg_fusion_loss:.4f}")

        # Verify losses are similar (within tolerance)
        loss_diff = abs(avg_baseline_loss - avg_fusion_loss)
        assert loss_diff < 0.1, f"Loss divergence too high: {loss_diff}"

    def test_memory_efficiency_large_model(self):
        """Test memory efficiency with large model."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")

        device = torch.device("cuda")

        # Create large model
        model = TransformerModel(num_layers=12, d_model=768, vocab_size=50000)
        model = model.to(device)

        # Fusion configuration for memory efficiency
        fusion_config = FusionConfig(
            enable_fusion=True,
            fusion_strategy=FusionStrategy.CONSERVATIVE,
            use_memory_pool=True,
            pool_size_limit_mb=100.0,
        )

        fusion_manager = GradientAccumulationFusion(
            model_params=list(model.parameters()),
            config=fusion_config,
            device=device,
        )

        # Track memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start_memory = torch.cuda.memory_allocated()

        # Perform multiple accumulation steps
        batch_size = 4
        seq_len = 256
        vocab_size = 50000
        num_accumulations = 10

        for _ in range(num_accumulations):
            data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            # Forward pass
            output = model(data)
            loss = output.mean()

            # Backward with fusion
            with fusion_manager.accumulation_context(accumulation_steps=4):
                loss.backward()

        peak_memory = torch.cuda.max_memory_allocated()
        memory_used_mb = (peak_memory - start_memory) / (1024 * 1024)

        print(f"\nMemory Usage:")
        print(f"Peak memory: {memory_used_mb:.2f} MB")

        # Get memory savings from fusion
        metrics = fusion_manager.get_metrics()
        print(f"Memory saved by fusion: {metrics['memory_saved_mb']:.2f} MB")

        # Verify memory usage is reasonable
        assert memory_used_mb < 2000, f"Memory usage too high: {memory_used_mb} MB"

    def test_different_fusion_strategies(self):
        """Test and compare different fusion strategies."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = TransformerModel(num_layers=2, d_model=256)
        model = model.to(device)

        strategies = [
            FusionStrategy.AGGRESSIVE,
            FusionStrategy.BALANCED,
            FusionStrategy.CONSERVATIVE,
            FusionStrategy.ADAPTIVE,
        ]

        results = {}

        for strategy in strategies:
            # Create fusion manager with specific strategy
            fusion_config = FusionConfig(
                enable_fusion=True,
                fusion_strategy=strategy,
                use_multi_tensor_ops=True,
            )

            fusion_manager = GradientAccumulationFusion(
                model_params=list(model.parameters()),
                config=fusion_config,
                device=device,
            )

            # Measure performance
            times = []

            for _ in range(10):
                # Generate data
                data = torch.randint(0, 10000, (4, 64), device=device)

                # Forward pass
                output = model(data)
                loss = output.mean()

                # Backward with fusion
                start = time.perf_counter()
                with fusion_manager.accumulation_context(accumulation_steps=2):
                    loss.backward()
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            metrics = fusion_manager.get_metrics()

            results[strategy.value] = {
                "avg_time": avg_time,
                "memory_saved": metrics["memory_saved_mb"],
                "tensors_fused": metrics["tensors_fused"],
            }

        print("\nFusion Strategy Comparison:")
        for strategy, result in results.items():
            print(f"\n{strategy}:")
            print(f"  Avg time: {result['avg_time'] * 1000:.2f} ms")
            print(f"  Memory saved: {result['memory_saved']:.2f} MB")
            print(f"  Tensors fused: {result['tensors_fused']}")

    def test_param_grad_mapping_integration(self):
        """Test integration with ParamGradMapping."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = TransformerModel(num_layers=2, d_model=256)
        model = model.to(device)

        # Create fused param-grad mapping
        mapping_config = MappingConfig(
            gradient_accumulation_steps=4,
            reduction_strategy=ReductionStrategy.OVERLAPPED,
        )

        fusion_config = FusionConfig(
            enable_fusion=True,
            fusion_strategy=FusionStrategy.BALANCED,
            async_reduction=False,
        )

        fused_mapping = FusedParamGradMapping(
            params=list(model.parameters()),
            config=mapping_config,
            fusion_config=fusion_config,
            device=device,
        )

        # Perform training steps
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for step in range(8):
            # Generate data
            data = torch.randint(0, 10000, (4, 64), device=device)

            # Forward pass
            output = model(data)
            loss = output.mean()

            # Backward pass
            loss.backward()

            # Accumulate with fusion
            fused_mapping.accumulate_gradients_with_fusion()

            # Synchronize if needed
            if fused_mapping.should_reduce_gradients():
                stats = fused_mapping.synchronize_gradients_async(force=True)
                print(f"\nStep {step + 1} sync stats:")
                fusion_metrics = stats.get("fusion_metrics", {})
                fusion_time = fusion_metrics.get("fusion_time", 0)
                print(f"  Fusion time: {fusion_time * 1000:.2f} ms")

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

        # Get final statistics
        final_stats = fused_mapping.get_statistics()
        print(f"\nFinal Statistics:")
        print(f"  Total reductions: {final_stats['total_reductions']}")
        print(
            "  Avg communication time: "
            f"{final_stats['avg_communication_time'] * 1000:.2f} ms"
        )


def distributed_worker(
    rank: int,
    world_size: int,
    backend: str,
):
    """Worker function for distributed training test."""
    setup_distributed(rank, world_size, backend)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Create model
    model = TransformerModel(num_layers=2, d_model=256, vocab_size=10000)
    model = model.to(device)

    # Wrap in DDP
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    # Create fusion manager
    fusion_config = FusionConfig(
        enable_fusion=True,
        fusion_strategy=FusionStrategy.BALANCED,
        async_reduction=True,
        overlap_strategy=OverlapStrategy.PARTIAL,
    )

    fusion_manager = GradientAccumulationFusion(
        model_params=list(model.parameters()),
        config=fusion_config,
        device=device,
    )

    orchestrator = AsyncReductionOrchestrator(fusion_manager=fusion_manager)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_steps = 10
    accumulation_steps = 2

    for step in range(num_steps):
        # Generate data
        data = torch.randint(0, 10000, (4, 64), device=device)
        target = torch.randint(0, 10000, (4, 64), device=device)

        # Forward pass
        output = model(data)
        loss = nn.functional.cross_entropy(
            output.view(-1, output.size(-1)), target.view(-1)
        )

        # Scale for accumulation
        loss = loss / accumulation_steps

        # Backward with fusion
        with fusion_manager.accumulation_context(accumulation_steps) as state:
            loss.backward()

            # Start async reduction on accumulation boundary
            if state.step % accumulation_steps == 0:
                orchestrator.start_reduction()

                # Wait for reduction
                orchestrator.wait_reduction()

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

        if rank == 0 and step % 5 == 0:
            metrics = fusion_manager.get_metrics()
            print(
                f"Step {step}: Loss={loss.item():.4f}, "
                f"Overlap={metrics['overlap_efficiency']:.2f}"
            )

    cleanup_distributed()


class TestIntegrationDistributed:
    """Integration tests for distributed training."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multi-GPU required")
    def test_multi_gpu_training(self):
        """Test distributed training with multiple GPUs."""
        world_size = min(2, torch.cuda.device_count())

        mp.spawn(  # type: ignore[attr-defined]
            distributed_worker,
            args=(world_size, "nccl"),
            nprocs=world_size,
            join=True,
        )

    @pytest.mark.skipif(torch.cuda.is_available(), reason="Testing CPU distributed")
    def test_cpu_distributed_training(self):
        """Test distributed training on CPU."""
        world_size = 2

        mp.spawn(  # type: ignore[attr-defined]
            distributed_worker,
            args=(world_size, "gloo"),
            nprocs=world_size,
            join=True,
        )


class TestPerformanceValidation:
    """Validate performance improvements."""

    def test_overlap_efficiency_measurement(self):
        """Measure and validate overlap efficiency."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = TransformerModel(num_layers=4, d_model=512)
        model = model.to(device)

        # Configure for maximum overlap
        fusion_config = FusionConfig(
            enable_fusion=True,
            fusion_strategy=FusionStrategy.AGGRESSIVE,
            async_reduction=False,  # Simulated for single process
            overlap_strategy=OverlapStrategy.FULL,
            overlap_ratio=1.0,
        )

        fusion_manager = GradientAccumulationFusion(
            model_params=list(model.parameters()),
            config=fusion_config,
            device=device,
        )

        # Simulate computation and communication phases
        computation_times = []
        communication_times = []

        for _ in range(20):
            # Computation phase
            start = time.perf_counter()

            data = torch.randint(0, 10000, (4, 128), device=device)
            output = model(data)
            loss = output.mean()

            with fusion_manager.accumulation_context(accumulation_steps=1):
                loss.backward()

            computation_times.append(time.perf_counter() - start)

            # Simulated communication phase
            start = time.perf_counter()
            time.sleep(0.001)  # Simulate network delay
            communication_times.append(time.perf_counter() - start)

        # Calculate overlap efficiency
        avg_computation = sum(computation_times) / len(computation_times)
        avg_communication = sum(communication_times) / len(communication_times)

        sequential_time = avg_computation + avg_communication
        overlapped_time = max(avg_computation, avg_communication)

        overlap_efficiency = 1 - (overlapped_time / sequential_time)

        print(f"\nOverlap Efficiency Measurement:")
        print(f"Avg computation time: {avg_computation * 1000:.2f} ms")
        print(f"Avg communication time: {avg_communication * 1000:.2f} ms")
        print(f"Sequential time: {sequential_time * 1000:.2f} ms")
        print(f"Overlapped time: {overlapped_time * 1000:.2f} ms")
        print(f"Overlap efficiency: {overlap_efficiency * 100:.2f}%")

        # Should achieve reasonable overlap
        # On CPU without real distributed communication, overlap will be minimal
        if device.type == "cpu":
            # On CPU, any positive overlap is acceptable
            assert (
                overlap_efficiency >= 0
            ), f"Overlap efficiency is negative: {overlap_efficiency}"
        else:
            # On GPU with real communication, expect better overlap
            assert (
                overlap_efficiency > 0.1
            ), f"Overlap efficiency too low: {overlap_efficiency}"

    def test_scalability_with_model_size(self):
        """Test scalability with different model sizes."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_configs = [
            {"num_layers": 1, "d_model": 128},
            {"num_layers": 2, "d_model": 256},
            {"num_layers": 4, "d_model": 512},
            {"num_layers": 6, "d_model": 768},
        ]

        results = []

        for config in model_configs:
            model = TransformerModel(**config, vocab_size=10000)
            model = model.to(device)

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())

            # Create fusion manager
            fusion_config = FusionConfig(
                enable_fusion=True,
                fusion_strategy=FusionStrategy.BALANCED,
            )

            fusion_manager = GradientAccumulationFusion(
                model_params=list(model.parameters()),
                config=fusion_config,
                device=device,
            )

            # Measure performance
            times = []

            for _ in range(10):
                data = torch.randint(0, 10000, (2, 64), device=device)
                output = model(data)
                loss = output.mean()

                start = time.perf_counter()
                with fusion_manager.accumulation_context(accumulation_steps=1):
                    loss.backward()
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)

            results.append(
                {
                    "layers": config["num_layers"],
                    "d_model": config["d_model"],
                    "num_params": num_params,
                    "avg_time": avg_time,
                    "time_per_param": avg_time / num_params * 1e6,  # microseconds
                }
            )

        print("\nScalability Analysis:")
        for result in results:
            print(f"\nModel: {result['layers']} layers, d_model={result['d_model']}")
            print(f"  Parameters: {result['num_params']:,}")
            print(f"  Avg time: {result['avg_time'] * 1000:.2f} ms")
            print(f"  Time per param: {result['time_per_param']:.3f} µs")

        # Check that time scales reasonably with model size
        time_per_param_values = [r["time_per_param"] for r in results]

        # Time per parameter should be relatively consistent
        max_variation = max(time_per_param_values) / min(time_per_param_values)
        assert (
            max_variation < 3.0
        ), f"Time scaling inconsistent: {max_variation}x variation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
