#!/usr/bin/env python3
"""
Gradient Accumulation Fusion Example

This example demonstrates how to use the gradient accumulation fusion feature
for efficient distributed training with overlapped communication.

The example shows:
1. Basic usage with a simple model
2. Configuration options for different strategies
3. Performance comparison with and without fusion
4. Integration with existing training loops
5. Advanced features like adaptive optimization

Run this example:
    # Single GPU
    python gradient_accumulation_fusion_example.py

    # Multi-GPU (2 GPUs)
    torchrun --nproc_per_node=2 gradient_accumulation_fusion_example.py --distributed

    # CPU distributed simulation
    torchrun --nproc_per_node=4 gradient_accumulation_fusion_example.py --distributed --cpu
"""

import argparse
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

# Import gradient accumulation fusion components
from rosellm.rosetrainer.gradient.accumulation_fusion import (
    AsyncReductionOrchestrator,
    FusionConfig,
    FusionStrategy,
    GradientAccumulationFusion,
    OverlapStrategy,
)


class SimpleTransformer(nn.Module):
    """Simple transformer model for demonstration."""

    def __init__(
        self, vocab_size: int = 10000, d_model: int = 512, num_layers: int = 6
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model) * 0.01)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=2048,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        x = x + self.pos_embedding[:, : x.size(1), :]

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output
        x = self.norm(x)
        return self.output(x)


def setup_distributed():
    """Setup distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Initialize process group
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

        return rank, world_size

    return 0, 1


def create_fusion_manager(
    model: nn.Module,
    strategy: str = "balanced",
    device: Optional[torch.device] = None,
) -> Tuple[GradientAccumulationFusion, AsyncReductionOrchestrator]:
    """
    Create fusion manager and orchestrator.

    Args:
        model: PyTorch model
        strategy: Fusion strategy name
        device: Device for operations

    Returns:
        Tuple of (fusion_manager, orchestrator)
    """
    # Map strategy names to enums
    strategy_map = {
        "aggressive": FusionStrategy.AGGRESSIVE,
        "balanced": FusionStrategy.BALANCED,
        "conservative": FusionStrategy.CONSERVATIVE,
        "adaptive": FusionStrategy.ADAPTIVE,
    }

    # Create fusion configuration
    fusion_config = FusionConfig(
        enable_fusion=True,
        fusion_strategy=strategy_map.get(strategy, FusionStrategy.BALANCED),
        fusion_buffer_size_mb=100.0,
        async_reduction=dist.is_initialized(),
        overlap_strategy=OverlapStrategy.PARTIAL,
        overlap_ratio=0.8,
        use_memory_pool=True,
        use_multi_tensor_ops=True,
        adaptive_optimization=strategy == "adaptive",
    )

    # Create fusion manager
    fusion_manager = GradientAccumulationFusion(
        model_params=list(model.parameters()),
        config=fusion_config,
        device=device,
    )

    # Create async orchestrator
    orchestrator = AsyncReductionOrchestrator(
        fusion_manager=fusion_manager,
        process_group=None,  # Uses default process group
    )

    return fusion_manager, orchestrator


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    data: torch.Tensor,
    target: torch.Tensor,
    fusion_manager: Optional[GradientAccumulationFusion] = None,
    orchestrator: Optional[AsyncReductionOrchestrator] = None,
    accumulation_steps: int = 1,
    step_num: int = 0,
) -> Dict:
    """
    Single training step with optional fusion.

    Args:
        model: Model to train
        optimizer: Optimizer
        data: Input data
        target: Target labels
        fusion_manager: Optional fusion manager
        orchestrator: Optional async orchestrator
        accumulation_steps: Number of accumulation steps
        step_num: Current step number

    Returns:
        Dictionary with training metrics
    """
    start_time = time.perf_counter()

    # Forward pass
    output = model(data)
    loss = nn.functional.cross_entropy(
        output.view(-1, output.size(-1)), target.view(-1)
    )

    # Scale loss for accumulation
    loss = loss / accumulation_steps

    # Backward pass
    if fusion_manager is not None:
        # Use fusion context
        with fusion_manager.accumulation_context(accumulation_steps) as state:
            loss.backward()

            # Start async reduction on accumulation boundary
            if state.step % accumulation_steps == 0 and orchestrator is not None:
                orchestrator.start_reduction()
    else:
        # Standard backward
        loss.backward()

    # Optimizer step on accumulation boundary
    if (step_num + 1) % accumulation_steps == 0:
        # Wait for reduction if using async
        if orchestrator is not None and fusion_manager is not None:
            if fusion_manager.accumulation_state.step % accumulation_steps == 0:
                orchestrator.wait_reduction()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

    # Calculate metrics
    compute_time = time.perf_counter() - start_time

    metrics = {
        "loss": loss.item() * accumulation_steps,
        "compute_time": compute_time,
        "perplexity": torch.exp(loss * accumulation_steps).item(),
    }

    # Add fusion metrics if available
    if fusion_manager is not None:
        fusion_metrics = fusion_manager.get_metrics()
        metrics.update(
            {
                "fusion_time": fusion_metrics["fusion_time"],
                "reduction_time": fusion_metrics["reduction_time"],
                "overlap_efficiency": fusion_metrics["overlap_efficiency"],
                "memory_saved_mb": fusion_metrics["memory_saved_mb"],
            }
        )

    return metrics


def compare_training_methods(
    model: nn.Module,
    device: torch.device,
    num_steps: int = 50,
    batch_size: int = 8,
    seq_len: int = 128,
    accumulation_steps: int = 4,
):
    """
    Compare training with and without fusion.

    Args:
        model: Model to train
        device: Device for training
        num_steps: Number of training steps
        batch_size: Batch size
        seq_len: Sequence length
        accumulation_steps: Gradient accumulation steps
    """
    vocab_size = 10000

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: With vs Without Gradient Accumulation Fusion")
    print("=" * 80)

    # Create two identical models
    model_baseline = SimpleTransformer(vocab_size=vocab_size, d_model=256, num_layers=4)
    model_baseline.load_state_dict(model.state_dict())
    model_baseline = model_baseline.to(device)

    model_fusion = SimpleTransformer(vocab_size=vocab_size, d_model=256, num_layers=4)
    model_fusion.load_state_dict(model.state_dict())
    model_fusion = model_fusion.to(device)

    # Create optimizers
    optimizer_baseline = optim.AdamW(model_baseline.parameters(), lr=1e-4)
    optimizer_fusion = optim.AdamW(model_fusion.parameters(), lr=1e-4)

    # Create fusion manager
    fusion_manager, orchestrator = create_fusion_manager(
        model_fusion, "balanced", device
    )

    # Training metrics
    baseline_metrics = []
    fusion_metrics = []

    print(
        f"\nTraining for {num_steps} steps with accumulation_steps={accumulation_steps}"
    )
    print("-" * 60)

    for step in range(num_steps):
        # Generate synthetic data
        data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Baseline training
        metrics_b = train_step(
            model_baseline,
            optimizer_baseline,
            data,
            target,
            accumulation_steps=accumulation_steps,
            step_num=step,
        )
        baseline_metrics.append(metrics_b)

        # Fusion training
        metrics_f = train_step(
            model_fusion,
            optimizer_fusion,
            data,
            target,
            fusion_manager,
            orchestrator,
            accumulation_steps=accumulation_steps,
            step_num=step,
        )
        fusion_metrics.append(metrics_f)

        # Print progress
        if (step + 1) % 10 == 0:
            avg_baseline_time = (
                sum(m["compute_time"] for m in baseline_metrics[-10:]) / 10
            )
            avg_fusion_time = sum(m["compute_time"] for m in fusion_metrics[-10:]) / 10
            improvement = (
                (avg_baseline_time - avg_fusion_time) / avg_baseline_time * 100
            )

            print(
                f"Step {step + 1:3d}: "
                f"Baseline={avg_baseline_time * 1000:6.2f}ms, "
                f"Fusion={avg_fusion_time * 1000:6.2f}ms, "
                f"Speedup={improvement:5.1f}%"
            )

    # Final statistics
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    # Calculate averages
    avg_baseline_time = sum(m["compute_time"] for m in baseline_metrics) / len(
        baseline_metrics
    )
    avg_fusion_time = sum(m["compute_time"] for m in fusion_metrics) / len(
        fusion_metrics
    )

    # Performance improvement
    time_improvement = (avg_baseline_time - avg_fusion_time) / avg_baseline_time * 100

    print(f"\nTiming Results:")
    print(f"  Baseline average: {avg_baseline_time * 1000:.2f} ms/step")
    print(f"  Fusion average:   {avg_fusion_time * 1000:.2f} ms/step")
    print(f"  Performance improvement: {time_improvement:.1f}%")

    print(f"\nLoss Convergence:")
    print(f"  Baseline final loss: {baseline_metrics[-1]['loss']:.4f}")
    print(f"  Fusion final loss:   {fusion_metrics[-1]['loss']:.4f}")
    print(
        f"  Loss difference:     {abs(baseline_metrics[-1]['loss'] - fusion_metrics[-1]['loss']):.4f}"
    )

    # Fusion-specific metrics
    if fusion_metrics[-1].get("overlap_efficiency") is not None:
        print(f"\nFusion Metrics:")
        print(
            f"  Average fusion time:     {sum(m.get('fusion_time', 0) for m in fusion_metrics) / len(fusion_metrics) * 1000:.2f} ms"
        )
        print(
            f"  Average reduction time:  {sum(m.get('reduction_time', 0) for m in fusion_metrics) / len(fusion_metrics) * 1000:.2f} ms"
        )
        print(
            f"  Overlap efficiency:      {fusion_metrics[-1]['overlap_efficiency'] * 100:.1f}%"
        )
        print(
            f"  Memory saved:           {fusion_metrics[-1]['memory_saved_mb']:.2f} MB"
        )


def demonstrate_fusion_strategies(model: nn.Module, device: torch.device):
    """
    Demonstrate different fusion strategies.

    Args:
        model: Model for testing
        device: Device for operations
    """
    print("\n" + "=" * 80)
    print("FUSION STRATEGY COMPARISON")
    print("=" * 80)

    strategies = ["aggressive", "balanced", "conservative", "adaptive"]
    vocab_size = 10000
    batch_size = 4
    seq_len = 64
    num_steps = 20

    for strategy in strategies:
        print(f"\n{strategy.upper()} Strategy:")
        print("-" * 40)

        # Create model copy
        test_model = SimpleTransformer(vocab_size=vocab_size, d_model=256, num_layers=2)
        test_model = test_model.to(device)

        # Create optimizer
        optimizer = optim.AdamW(test_model.parameters(), lr=1e-4)

        # Create fusion manager with specific strategy
        fusion_manager, orchestrator = create_fusion_manager(
            test_model, strategy, device
        )

        # Run training steps
        times = []
        memory_saved = []

        for step in range(num_steps):
            data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            metrics = train_step(
                test_model,
                optimizer,
                data,
                target,
                fusion_manager,
                orchestrator,
                accumulation_steps=2,
                step_num=step,
            )

            times.append(metrics["compute_time"])
            if "memory_saved_mb" in metrics:
                memory_saved.append(metrics["memory_saved_mb"])

        # Print statistics
        avg_time = sum(times) / len(times)
        avg_memory = sum(memory_saved) / len(memory_saved) if memory_saved else 0

        print(f"  Average time: {avg_time * 1000:.2f} ms")
        print(f"  Memory saved: {avg_memory:.2f} MB")

        # Get final fusion metrics
        final_metrics = fusion_manager.get_metrics()
        print(f"  Tensors fused: {final_metrics['tensors_fused']}")
        print(f"  Buffer utilization: {final_metrics['buffer_utilization']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Gradient Accumulation Fusion Example")
    parser.add_argument(
        "--distributed", action="store_true", help="Run in distributed mode"
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--num-steps", type=int, default=50, help="Number of training steps"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument(
        "--accumulation-steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="balanced",
        choices=["aggressive", "balanced", "conservative", "adaptive"],
        help="Fusion strategy to use",
    )
    parser.add_argument("--compare", action="store_true", help="Compare with baseline")
    parser.add_argument(
        "--demo-strategies", action="store_true", help="Demonstrate all strategies"
    )

    args = parser.parse_args()

    # Setup device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup distributed if requested
    rank, world_size = 0, 1
    if args.distributed:
        rank, world_size = setup_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")

    print(f"\nRunning on device: {device}")
    if world_size > 1:
        print(f"Distributed training: rank {rank}/{world_size}")

    # Create model
    model = SimpleTransformer(vocab_size=10000, d_model=256, num_layers=4)
    model = model.to(device)

    # Wrap in DDP if distributed
    if world_size > 1:
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[rank])
        else:
            model = DDP(model)

    # Run demonstrations
    if args.demo_strategies:
        demonstrate_fusion_strategies(model, device)
    elif args.compare:
        compare_training_methods(
            model,
            device,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            accumulation_steps=args.accumulation_steps,
        )
    else:
        # Single strategy training
        print(f"\nTraining with {args.strategy.upper()} fusion strategy")
        print("=" * 60)

        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        # Create fusion manager
        fusion_manager, orchestrator = create_fusion_manager(
            model, args.strategy, device
        )

        # Training loop
        for step in range(args.num_steps):
            # Generate data
            data = torch.randint(
                0, 10000, (args.batch_size, args.seq_len), device=device
            )
            target = torch.randint(
                0, 10000, (args.batch_size, args.seq_len), device=device
            )

            # Training step
            metrics = train_step(
                model,
                optimizer,
                data,
                target,
                fusion_manager,
                orchestrator,
                accumulation_steps=args.accumulation_steps,
                step_num=step,
            )

            # Print progress
            if (step + 1) % 10 == 0:
                print(
                    f"Step {step + 1:3d}: "
                    f"Loss={metrics['loss']:.4f}, "
                    f"Time={metrics['compute_time'] * 1000:.2f}ms, "
                    f"Overlap={metrics.get('overlap_efficiency', 0) * 100:.1f}%"
                )

        # Final metrics
        print("\nFinal Metrics:")
        final_metrics = fusion_manager.get_metrics()
        for key, value in final_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, list):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")

    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
