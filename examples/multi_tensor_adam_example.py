#!/usr/bin/env python3
"""
Multi-Tensor Adam Optimizer Example with Comprehensive Benchmarking.

This example demonstrates the Multi-Tensor Adam optimizer with:
1. Basic usage and configuration
2. Performance comparison against standard optimizers
3. Mixed precision training
4. Distributed training setup
5. Memory efficiency analysis
6. Real-world model training scenarios

Usage:
    # Basic training
    python multi_tensor_adam_example.py --mode basic

    # Performance benchmarking
    python multi_tensor_adam_example.py --mode benchmark --iterations 100

    # Mixed precision training
    python multi_tensor_adam_example.py --mode mixed_precision --use_amp

    # Distributed training (run with torchrun)
    torchrun --nproc_per_node=2 multi_tensor_adam_example.py --mode distributed

    # Memory efficiency analysis
    python multi_tensor_adam_example.py --mode memory_analysis --model_size large
"""

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# Import RoseLLM components
from rosellm.rosetrainer.optimizer import (
    MultiTensorAdam,
    MultiTensorAdamConfig,
    WeightDecayMode,
)


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""

    optimizer_name: str
    backend: str
    total_time: float
    avg_step_time: float
    throughput: float  # samples/second
    memory_usage: float  # MB
    accuracy: float
    convergence_steps: int
    final_loss: float

    def to_dict(self) -> Dict:
        return asdict(self)


class VisionTransformer(nn.Module):
    """Simple Vision Transformer for benchmarking."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        n_classes: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, d_model, kernel_size=patch_size, stride=patch_size
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # B, d_model, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)  # B, n_patches, d_model

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Classification
        x = self.ln(x[:, 0])  # Use class token
        x = self.head(x)

        return x


class CNNModel(nn.Module):
    """Convolutional neural network for comparison."""

    def __init__(self, n_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Third block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Fourth block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_synthetic_dataset(
    num_samples: int = 1000,
    image_size: int = 224,
    n_classes: int = 1000,
    device: torch.device = torch.device("cpu"),
) -> DataLoader:
    """Create synthetic dataset for benchmarking."""

    # Generate random images and labels
    images = torch.randn(num_samples, 3, image_size, image_size, device=device)
    labels = torch.randint(0, n_classes, (num_samples,), device=device)

    dataset = TensorDataset(images, labels)

    # Use reasonable batch size based on available memory
    batch_size = 32 if device.type == "cpu" else 16
    if torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if memory_gb > 16:
            batch_size = 32
        elif memory_gb > 8:
            batch_size = 16
        else:
            batch_size = 8

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


@contextmanager
def memory_profiler(device: torch.device):
    """Context manager for memory profiling."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        yield

        peak_memory = torch.cuda.max_memory_allocated()
        final_memory = torch.cuda.memory_allocated()

        print(f"Memory usage:")
        print(f"  Initial: {initial_memory / 1e6:.1f} MB")
        print(f"  Peak: {peak_memory / 1e6:.1f} MB")
        print(f"  Final: {final_memory / 1e6:.1f} MB")
        print(f"  Increase: {(final_memory - initial_memory) / 1e6:.1f} MB")
    else:
        yield


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    use_amp: bool = False,
    target_loss: float = 1.0,
) -> Tuple[float, int, List[float]]:
    """Train model and return final loss, convergence steps, and loss history."""

    model.train()
    losses = []
    convergence_step = -1

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    total_steps = 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()

            if use_amp and scaler:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = F.cross_entropy(output, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = F.cross_entropy(output, target)

                if isinstance(optimizer, MultiTensorAdam):
                    # Multi-tensor Adam with mixed precision
                    optimizer.backward(loss)
                else:
                    loss.backward()

                optimizer.step()

            losses.append(loss.item())
            total_steps += 1

            # Check for convergence
            if convergence_step == -1 and loss.item() < target_loss:
                convergence_step = total_steps

            # Break after reasonable number of steps for benchmarking
            if total_steps >= 100:
                break

        if total_steps >= 100:
            break

    return losses[-1], convergence_step, losses


def benchmark_optimizer(
    optimizer_name: str,
    model_factory,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict,
    use_amp: bool = False,
    iterations: int = 50,
) -> BenchmarkResults:
    """Benchmark a specific optimizer configuration."""

    print(f"\nBenchmarking {optimizer_name}...")

    # Create model
    model = model_factory().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create optimizer
    optimizer: Union[MultiTensorAdam, AdamW]
    if optimizer_name == "MultiTensorAdam":
        mt_config = MultiTensorAdamConfig(**config)
        optimizer = MultiTensorAdam(model.parameters(), mt_config)
        backend_name = optimizer.get_backend_info()["backend"]
    elif optimizer_name == "MultiTensorAdamW":
        mt_config = MultiTensorAdamConfig(**config)
        optimizer = MultiTensorAdam(model.parameters(), mt_config)
        backend_name = optimizer.get_backend_info()["backend"]
    elif optimizer_name == "PyTorchAdamW":
        optimizer = AdamW(model.parameters(), **config)
        backend_name = "pytorch"
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Training setup
    limited_dataloader = []
    for i, batch in enumerate(dataloader):
        limited_dataloader.append(batch)
        if i >= iterations - 1:
            break

    # Warm up
    for i in range(min(5, len(limited_dataloader))):
        data, target = limited_dataloader[i]
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        if use_amp:
            with torch.cuda.amp.autocast():
                loss.backward()
        else:
            if isinstance(optimizer, MultiTensorAdam):
                optimizer.backward(loss)
            else:
                loss.backward()

        optimizer.step()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    initial_memory = torch.cuda.memory_allocated() if device.type == "cuda" else 0

    losses = []
    for i in range(len(limited_dataloader)):
        data, target = limited_dataloader[i]

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = F.cross_entropy(output, target)
            loss.backward()
        else:
            output = model(data)
            loss = F.cross_entropy(output, target)

            if isinstance(optimizer, MultiTensorAdam):
                optimizer.backward(loss)
            else:
                loss.backward()

        optimizer.step()
        losses.append(loss.item())

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    final_memory = torch.cuda.memory_allocated() if device.type == "cuda" else 0

    # Calculate metrics
    total_time = end_time - start_time
    avg_step_time = total_time / len(limited_dataloader)

    # Calculate throughput (samples per second)
    total_samples = sum(len(batch[1]) for batch in limited_dataloader)
    throughput = total_samples / total_time

    memory_usage = (final_memory - initial_memory) / 1e6  # MB

    # Simple accuracy metric (loss reduction)
    accuracy = max(0, (losses[0] - losses[-1]) / losses[0])

    convergence_steps: int = len(losses)
    for idx, loss_val in enumerate(losses):
        if loss_val < losses[0] * 0.8:  # 20% reduction
            convergence_steps = idx + 1
            break

    return BenchmarkResults(
        optimizer_name=optimizer_name,
        backend=backend_name,
        total_time=total_time,
        avg_step_time=avg_step_time,
        throughput=throughput,
        memory_usage=memory_usage,
        accuracy=accuracy,
        convergence_steps=convergence_steps,
        final_loss=losses[-1],
    )


def run_basic_example(args):
    """Run basic Multi-Tensor Adam example."""

    print("=== Basic Multi-Tensor Adam Example ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model and data
    model = CNNModel(n_classes=100).to(device)
    dataloader = create_synthetic_dataset(500, 224, 100, device)

    # Create Multi-Tensor Adam optimizer
    config = MultiTensorAdamConfig(
        lr=1e-3,
        weight_decay=0.01,
        weight_decay_mode=WeightDecayMode.DECOUPLED,
        use_mixed_precision=device.type == "cuda",
        max_grad_norm=1.0,
        enable_profiling=True,
    )

    optimizer = MultiTensorAdam(model.parameters(), config)

    print(f"Optimizer backend: {optimizer.get_backend_info()['backend']}")
    print(f"Mixed precision: {config.use_mixed_precision}")

    # Train for a few steps
    print("\nTraining...")
    final_loss, convergence, losses = train_model(
        model, optimizer, dataloader, device, epochs=1, use_amp=False
    )

    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss reduction: {(losses[0] - final_loss) / losses[0] * 100:.1f}%")

    # Show performance metrics
    metrics = optimizer.get_metrics()
    print(f"\nOptimizer metrics:")
    print(f"  Steps: {metrics.step}")
    print(f"  Total time: {metrics.total_time:.3f}s")
    print(f"  Avg step time: {metrics.total_time / max(metrics.step, 1):.4f}s")
    print(f"  Backend: {metrics.backend_used}")

    # Show detailed performance stats
    perf_stats = optimizer.get_performance_stats()
    if perf_stats:
        print(f"\nDetailed performance stats:")
        for operation, stats in perf_stats.items():
            print(f"  {operation}: {stats}")


def run_benchmark_comparison(args):
    """Run comprehensive benchmark comparison."""

    print("=== Multi-Tensor Adam Benchmark Comparison ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configurations
    model_configs = {
        "small": lambda: CNNModel(n_classes=100),
        "medium": lambda: VisionTransformer(
            d_model=512, n_heads=8, n_layers=6, n_classes=100
        ),
        "large": lambda: (
            VisionTransformer(d_model=768, n_heads=12, n_layers=12, n_classes=1000)
            if device.type == "cuda"
            else CNNModel(n_classes=1000)
        ),
    }

    model_factory = model_configs[args.model_size]

    # Create dataset
    n_classes = 100 if args.model_size == "small" else 1000
    dataloader = create_synthetic_dataset(1000, 224, n_classes, device)

    # Optimizer configurations
    base_config = {"lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.999), "eps": 1e-8}

    optimizers_to_test = [
        ("PyTorchAdamW", base_config),
        (
            "MultiTensorAdam",
            {
                **base_config,
                "weight_decay_mode": WeightDecayMode.L2_REGULARIZATION,
                "enable_multi_tensor": True,
                "use_mixed_precision": args.use_amp and device.type == "cuda",
            },
        ),
        (
            "MultiTensorAdamW",
            {
                **base_config,
                "weight_decay_mode": WeightDecayMode.DECOUPLED,
                "enable_multi_tensor": True,
                "use_mixed_precision": args.use_amp and device.type == "cuda",
            },
        ),
    ]

    # Run benchmarks
    results = []

    for optimizer_name, config in optimizers_to_test:
        try:
            result = benchmark_optimizer(
                optimizer_name,
                model_factory,
                dataloader,
                device,
                config,
                args.use_amp,
                args.iterations,
            )
            results.append(result)

            print(f"Results for {optimizer_name}:")
            print(f"  Backend: {result.backend}")
            print(f"  Total time: {result.total_time:.3f}s")
            print(f"  Avg step time: {result.avg_step_time:.4f}s")
            print(f"  Throughput: {result.throughput:.1f} samples/s")
            print(f"  Memory usage: {result.memory_usage:.1f} MB")
            print(f"  Final loss: {result.final_loss:.4f}")
            print(f"  Convergence steps: {result.convergence_steps}")

        except Exception as e:
            print(f"Error benchmarking {optimizer_name}: {e}")

    # Compare results
    if len(results) > 1:
        print("\n=== Comparison ===")

        baseline = next(r for r in results if r.optimizer_name == "PyTorchAdamW")

        for result in results:
            if result.optimizer_name != "PyTorchAdamW":
                speedup = baseline.total_time / result.total_time
                throughput_ratio = result.throughput / baseline.throughput
                memory_ratio = result.memory_usage / max(baseline.memory_usage, 0.1)

                print(f"\n{result.optimizer_name} vs PyTorchAdamW:")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Throughput ratio: {throughput_ratio:.2f}x")
                print(f"  Memory ratio: {memory_ratio:.2f}x")
                print(f"  Backend: {result.backend}")

    # Save results
    if args.save_results:
        results_data = [result.to_dict() for result in results]
        with open("benchmark_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to benchmark_results.json")


def run_mixed_precision_example(args):
    """Run mixed precision training example."""

    print("=== Mixed Precision Training Example ===")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping mixed precision example")
        return

    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Create larger model for mixed precision benefits
    model = VisionTransformer(d_model=768, n_heads=12, n_layers=6).to(device)
    dataloader = create_synthetic_dataset(500, 224, 1000, device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compare FP32 vs mixed precision
    configs = [
        (
            "FP32",
            MultiTensorAdamConfig(
                lr=1e-3,
                weight_decay=0.01,
                use_mixed_precision=False,
                enable_profiling=True,
            ),
        ),
        (
            "Mixed Precision",
            MultiTensorAdamConfig(
                lr=1e-3,
                weight_decay=0.01,
                use_mixed_precision=True,
                dynamic_loss_scale=True,
                loss_scale=2**16,
                enable_profiling=True,
            ),
        ),
    ]

    for name, config in configs:
        print(f"\n--- {name} Training ---")

        # Create fresh model
        test_model = VisionTransformer(d_model=768, n_heads=12, n_layers=6).to(device)
        optimizer = MultiTensorAdam(test_model.parameters(), config)

        with memory_profiler(device):
            start_time = time.perf_counter()

            final_loss, convergence, losses = train_model(
                test_model,
                optimizer,
                dataloader,
                device,
                epochs=1,
                use_amp=False,  # Multi-tensor handles mixed precision internally
            )

            end_time = time.perf_counter()

        print(f"Time: {end_time - start_time:.3f}s")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Backend: {optimizer.get_backend_info()['backend']}")

        if config.use_mixed_precision:
            print(f"Final loss scale: {optimizer.loss_scale}")
            print(f"Overflow count: {optimizer.overflow_count}")


def run_distributed_example(args):
    """Run distributed training example."""

    print("=== Distributed Training Example ===")

    # Initialize distributed training
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )
        except:
            print("Failed to initialize distributed training")
            return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None

    print(f"Rank {dist.get_rank()}/{world_size}, device: {device}")

    # Create model
    model = CNNModel(n_classes=100).to(device)

    # Wrap with DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if torch.cuda.is_available() else None
    )

    # Create Multi-Tensor Adam optimizer
    config = MultiTensorAdamConfig(
        lr=1e-3,
        weight_decay=0.01,
        use_mixed_precision=torch.cuda.is_available(),
        enable_multi_tensor=True,
    )

    optimizer = MultiTensorAdam(model.parameters(), config)

    # Create distributed data
    dataloader = create_synthetic_dataset(200, 224, 100, device)

    print(f"Training on rank {dist.get_rank()}...")

    # Train
    final_loss, convergence, losses = train_model(
        model, optimizer, dataloader, device, epochs=1
    )

    if dist.get_rank() == 0:
        print(f"Final loss: {final_loss:.4f}")
        print(f"Backend: {optimizer.get_backend_info()['backend']}")

    dist.destroy_process_group()


def run_memory_analysis(args):
    """Run memory efficiency analysis."""

    print("=== Memory Efficiency Analysis ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        print("Memory analysis requires CUDA")
        return

    print(f"Using device: {device}")
    print(
        f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    # Test different model sizes
    model_sizes = {
        "small": lambda: CNNModel(n_classes=100),
        "medium": lambda: VisionTransformer(d_model=512, n_heads=8, n_layers=6),
        "large": lambda: VisionTransformer(d_model=768, n_heads=12, n_layers=12),
    }

    selected_size = args.model_size
    if selected_size not in model_sizes:
        selected_size = "medium"

    model_factory = model_sizes[selected_size]

    # Test different optimizer configurations
    configs = [
        ("Standard", {"enable_multi_tensor": False, "use_mixed_precision": False}),
        ("Multi-Tensor", {"enable_multi_tensor": True, "use_mixed_precision": False}),
        ("Mixed Precision", {"enable_multi_tensor": True, "use_mixed_precision": True}),
    ]

    for name, extra_config in configs:
        print(f"\n--- {name} Configuration ---")

        # Build config parameters properly
        config_kwargs: Dict[str, Any] = {
            "lr": 1e-3,
            "weight_decay": 0.01,
        }
        config_kwargs.update(extra_config)
        config = MultiTensorAdamConfig(**config_kwargs)

        model = model_factory().to(device)
        optimizer = MultiTensorAdam(model.parameters(), config)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create smaller dataset for memory testing
        dataloader = create_synthetic_dataset(100, 224, 100, device)

        with memory_profiler(device):
            try:
                final_loss, _, losses = train_model(
                    model, optimizer, dataloader, device, epochs=1
                )
                print(f"Training successful, final loss: {final_loss:.4f}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("Out of memory!")
                else:
                    print(f"Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Tensor Adam Optimizer Example")

    parser.add_argument(
        "--mode",
        choices=[
            "basic",
            "benchmark",
            "mixed_precision",
            "distributed",
            "memory_analysis",
        ],
        default="basic",
        help="Example mode to run",
    )

    parser.add_argument(
        "--model_size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size for benchmarking",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations for benchmarking",
    )

    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision"
    )

    parser.add_argument(
        "--save_results", action="store_true", help="Save benchmark results to file"
    )

    args = parser.parse_args()

    # Set up
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Run selected mode
    if args.mode == "basic":
        run_basic_example(args)
    elif args.mode == "benchmark":
        run_benchmark_comparison(args)
    elif args.mode == "mixed_precision":
        run_mixed_precision_example(args)
    elif args.mode == "distributed":
        run_distributed_example(args)
    elif args.mode == "memory_analysis":
        run_memory_analysis(args)

    print("\nExample completed!")


if __name__ == "__main__":
    import os

    main()
