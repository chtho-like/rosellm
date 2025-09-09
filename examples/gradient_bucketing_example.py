#!/usr/bin/env python3
r"""
Gradient Communication Bucketing Example for RoseTrainer

This example demonstrates how to use the gradient bucketing functionality
to optimize distributed training communication. It shows different bucketing
strategies and their impact on communication efficiency.

Usage:
    # Single GPU (for testing bucketing logic)
    python gradient_bucketing_example.py --single-gpu

    # Multi-GPU distributed training
    torchrun --nproc_per_node=2 gradient_bucketing_example.py --distributed

    # CPU-only distributed training (for testing)
    CUDA_VISIBLE_DEVICES="" torchrun --nproc_per_node=4 \
        gradient_bucketing_example.py --cpu-only
"""

import argparse
import logging
import os
import time
from typing import Any, Dict, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import RoseTrainer components
from rosellm.rosetrainer import RoseTrainer
from rosellm.rosetrainer.config import TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TransformerModel(nn.Module):
    """
    Simple transformer model for demonstrating gradient bucketing.

    This model has different layer types (embedding, attention, feedforward,
    normalization, output) to showcase layer-based bucketing strategies.
    """

    def __init__(
        self, vocab_size: int = 1000, hidden_size: int = 512, num_layers: int = 4
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers
        # Create individual layers for type safety
        self.attn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(num_layers):
            # Attention layers
            attn_layer = nn.ModuleDict(
                {
                    "query": nn.Linear(hidden_size, hidden_size),
                    "key": nn.Linear(hidden_size, hidden_size),
                    "value": nn.Linear(hidden_size, hidden_size),
                    "output": nn.Linear(hidden_size, hidden_size),
                }
            )
            self.attn_layers.append(attn_layer)

            # MLP and normalization
            layer_components = nn.ModuleDict(
                {
                    "mlp_fc1": nn.Linear(hidden_size, hidden_size * 4),
                    "mlp_fc2": nn.Linear(hidden_size * 4, hidden_size),
                    "norm1": nn.LayerNorm(hidden_size),
                    "norm2": nn.LayerNorm(hidden_size),
                }
            )
            self.norm_layers.append(layer_components)

        self.output_head = nn.Linear(hidden_size, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len = input_ids.shape

        # Embedding
        x = self.embedding(input_ids)

        # Transformer layers (simplified - no actual attention)
        for i in range(len(self.attn_layers)):
            attn_layer = self.attn_layers[i]
            layer_components = self.norm_layers[i]

            # Type cast to ModuleDict for proper indexing
            attn_dict = cast(nn.ModuleDict, attn_layer)
            components_dict = cast(nn.ModuleDict, layer_components)

            # Simplified attention
            q = attn_dict["query"](x)
            k = attn_dict["key"](x)
            v = attn_dict["value"](x)

            # Simple attention (not real scaled dot-product)
            attn_out = attn_dict["output"](q + k + v)
            x = components_dict["norm1"](x + attn_out)

            # MLP
            mlp_out = components_dict["mlp_fc2"](
                torch.relu(components_dict["mlp_fc1"](x))
            )
            x = components_dict["norm2"](x + mlp_out)

        # Output projection
        logits = self.output_head(x)

        return logits


class ModelWithLoss(nn.Module):
    """Wrapper that adds loss computation for compatibility with RoseTrainer."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with loss computation."""
        logits = self.model(input_ids)

        # Reshape for cross-entropy loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        labels_flat = labels.view(batch_size * seq_len)

        loss = self.loss_fn(logits_flat, labels_flat)

        return {"loss": loss, "logits": logits}


def create_synthetic_dataset(
    num_samples: int = 1000, seq_length: int = 64, vocab_size: int = 1000
):
    """Create synthetic dataset for training."""
    # Generate random sequences
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))

    # Labels are shifted input_ids (language modeling)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = 0  # Pad last position

    return TensorDataset(input_ids, labels)


def run_bucketing_comparison(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    strategies: list,
    num_steps: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different bucketing strategies and measure their performance.

    Args:
        model: Model to train
        dataloader: Training data
        device: Device to use
        strategies: List of bucketing configurations to test
        num_steps: Number of training steps to run

    Returns:
        Dictionary mapping strategy names to performance metrics
    """
    results = {}

    for strategy_name, config in strategies:
        logger.info(f"\n=== Testing strategy: {strategy_name} ===")

        # Create fresh model copy
        base_model = TransformerModel(vocab_size=1000, hidden_size=512, num_layers=4)
        model_copy = ModelWithLoss(base_model)
        model_copy.to(device)

        # Create optimizer
        optimizer = optim.AdamW(model_copy.parameters(), lr=1e-4)

        # Create trainer with specific bucketing config
        trainer = RoseTrainer(
            model=model_copy,
            optimizer=optimizer,
            config=config,
        )

        # Measure training time
        start_time = time.time()
        step_times = []
        bucketing_stats = []

        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            step_start = time.time()

            # Move batch to device
            batch_dict = {
                "input_ids": batch[0].to(device),
                "labels": batch[1].to(device),
            }

            # Training step
            metrics = trainer.train_step(batch_dict)

            step_time = time.time() - step_start
            step_times.append(step_time)

            # Collect bucketing statistics if available
            if (
                hasattr(trainer, "bucket_manager")
                and trainer.bucket_manager is not None
            ):
                bucket_stats = trainer.bucket_manager.get_statistics()
                bucketing_stats.append(bucket_stats)

            if step % 5 == 0:
                logger.info(
                    f"Step {step}: loss={metrics['loss']:.4f}, "
                    f"time={step_time:.4f}s"
                )

        total_time = time.time() - start_time
        avg_step_time = sum(step_times) / len(step_times)

        # Collect results
        result = {
            "total_time": total_time,
            "avg_step_time": avg_step_time,
            "step_times": step_times,
            "final_loss": metrics["loss"],
        }

        # Add bucketing-specific metrics
        if bucketing_stats:
            final_stats = bucketing_stats[-1]
            result.update(
                {
                    "num_buckets": final_stats.get("num_buckets", 0),
                    "total_size_mb": final_stats.get("total_size_mb", 0),
                    "avg_communication_time": final_stats.get(
                        "avg_communication_time", 0
                    ),
                    "bucketing_efficiency": final_stats.get("avg_bucket_size_mb", 0),
                }
            )

        results[strategy_name] = result

        # Cleanup
        trainer.cleanup()
        del trainer, model_copy, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logger.info(f"Strategy {strategy_name} completed in {total_time:.2f}s")

    return results


def demonstrate_advanced_features(device: torch.device):
    """Demonstrate advanced bucketing features like dynamic optimization."""
    logger.info("\n=== Advanced Features Demonstration ===")

    # Create model and data
    model = TransformerModel(vocab_size=1000, hidden_size=256, num_layers=2)
    model = ModelWithLoss(model)
    model.to(device)

    dataset = create_synthetic_dataset(100, seq_length=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Configuration with advanced features
    config = TrainingConfig(
        batch_size=4,
        max_steps=5,
        num_epochs=None,  # Must be None when using max_steps
        warmup_steps=0,
        seed=42,
        checkpoint_interval=10,
        log_interval=1,
        eval_interval=5,
    )

    trainer = RoseTrainer(model=model, optimizer=optimizer, config=config)

    if trainer.bucket_manager is not None:
        logger.info("=== Initial Bucket Configuration ===")
        initial_stats = trainer.bucket_manager.get_statistics()
        logger.info(f"Strategy: {initial_stats['strategy']}")
        logger.info(f"Initial buckets: {initial_stats['num_buckets']}")

        # Run some training steps
        for step, batch in enumerate(dataloader):
            if step >= 5:
                break

            batch_dict = {
                "input_ids": batch[0].to(device),
                "labels": batch[1].to(device),
            }

            _ = trainer.train_step(batch_dict)

            # Show bucket statistics evolution
            current_stats = trainer.bucket_manager.get_statistics()
            logger.info(
                f"Step {step}: buckets={current_stats['num_buckets']}, "
                f"size={current_stats['total_size_mb']:.2f}MB"
            )

            # Demonstrate bucket optimization
            if step == 2:
                logger.info("=== Running Bucket Optimization ===")
                trainer.bucket_manager.optimize_buckets()

        # Show group statistics if available
        if trainer.bucket_group_manager is not None:
            group_stats = trainer.bucket_group_manager.get_statistics()
            logger.info("=== Group Statistics ===")
            logger.info(f"Active groups: {group_stats['groups']['active']}")
            logger.info(f"Group strategy: {group_stats['config']['strategy']}")

    trainer.cleanup()


def demonstrate_memory_efficiency(device: torch.device):
    """Demonstrate memory efficiency of bucketing system."""
    logger.info("\n=== Memory Efficiency Demonstration ===")

    # Create larger model to show memory benefits
    model = TransformerModel(vocab_size=2000, hidden_size=768, num_layers=6)
    model = ModelWithLoss(model)
    model.to(device)

    # Count model parameters and estimate memory
    total_params = sum(p.numel() for p in model.parameters())
    param_memory_mb = total_params * 4 / (1024 * 1024)  # Assuming FP32

    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Estimated parameter memory: {param_memory_mb:.2f}MB")

    # Create gradient tensors
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Create fake gradients
            gradients[name] = torch.randn_like(param)

    gradient_memory_mb = sum(g.numel() * 4 for g in gradients.values()) / (1024 * 1024)
    logger.info(f"Gradient memory: {gradient_memory_mb:.2f}MB")

    # Test different bucket sizes and their memory efficiency
    bucket_sizes = [1.0, 5.0, 10.0, 25.0]  # MB

    for bucket_size in bucket_sizes:
        from rosellm.rosetrainer.communication import BucketConfig, BucketManager
        from rosellm.rosetrainer.communication.gradient_buckets import BucketStrategy

        config = BucketConfig(
            strategy=BucketStrategy.SIZE_BASED,
            max_bucket_size_mb=bucket_size,
        )

        manager = BucketManager(config, device)

        # Assign all gradients
        for param_name, gradient in gradients.items():
            manager.assign_gradient(param_name, gradient)

        stats = manager.get_statistics()
        logger.info(
            f"Bucket size {bucket_size}MB: {stats['num_buckets']} buckets, "
            f"efficiency: {stats['avg_bucket_size_mb']:.2f}MB avg"
        )


def main():
    parser = argparse.ArgumentParser(description="Gradient Bucketing Example")
    parser.add_argument("--single-gpu", action="store_true", help="Run on single GPU")
    parser.add_argument(
        "--distributed", action="store_true", help="Run distributed training"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Force CPU-only execution"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of training steps"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size for testing",
    )

    args = parser.parse_args()

    # Setup device
    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info("Using CPU")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        logger.info(f"Using GPU {local_rank}")

    # Model size configurations
    model_configs = {
        "small": {"vocab_size": 500, "hidden_size": 256, "num_layers": 2},
        "medium": {"vocab_size": 1000, "hidden_size": 512, "num_layers": 4},
        "large": {"vocab_size": 2000, "hidden_size": 768, "num_layers": 6},
    }

    model_config = model_configs[args.model_size]
    logger.info(f"Using {args.model_size} model: {model_config}")

    # Create model and data
    model = TransformerModel(**model_config)
    model = ModelWithLoss(model)
    model.to(device)

    dataset = create_synthetic_dataset(
        1000, seq_length=64, vocab_size=model_config["vocab_size"]
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Define bucketing strategies to compare
    strategies = {
        "no_bucketing": TrainingConfig(
            batch_size=args.batch_size,
            max_steps=args.steps,
            num_epochs=None,
            warmup_steps=0,
            seed=42,
            checkpoint_interval=50,
            log_interval=10,
            eval_interval=25,
        ),
        "size_based": TrainingConfig(
            batch_size=args.batch_size,
            max_steps=args.steps,
            num_epochs=None,
            warmup_steps=0,
            seed=42,
            checkpoint_interval=50,
            log_interval=10,
            eval_interval=25,
        ),
        "layer_based": TrainingConfig(
            batch_size=args.batch_size,
            max_steps=args.steps,
            num_epochs=None,
            warmup_steps=0,
            seed=42,
            checkpoint_interval=50,
            log_interval=10,
            eval_interval=25,
        ),
        "mixed_strategy": TrainingConfig(
            batch_size=args.batch_size,
            max_steps=args.steps,
            num_epochs=None,
            warmup_steps=0,
            seed=42,
            checkpoint_interval=50,
            log_interval=10,
            eval_interval=25,
        ),
        "small_buckets": TrainingConfig(
            batch_size=args.batch_size,
            max_steps=args.steps,
            num_epochs=None,
            warmup_steps=0,
            seed=42,
            checkpoint_interval=50,
            log_interval=10,
            eval_interval=25,
        ),
    }

    # Only test distributed features if actually distributed
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1:
        # Remove distributed-only strategies for single process
        strategies = {k: v for k, v in strategies.items() if k != "no_bucketing"}
        logger.info("Single process mode - testing bucketing logic only")

    # Run performance comparison
    logger.info("Starting bucketing strategy comparison...")
    results = run_bucketing_comparison(
        model, dataloader, device, list(strategies.items()), args.steps
    )

    # Display results
    logger.info("\n=== Performance Comparison Results ===")
    for strategy_name, result in results.items():
        logger.info(f"\n{strategy_name.upper()}:")
        logger.info(f"  Total time: {result['total_time']:.3f}s")
        logger.info(f"  Avg step time: {result['avg_step_time']:.4f}s")
        logger.info(f"  Final loss: {result['final_loss']:.4f}")

        if "num_buckets" in result:
            logger.info(f"  Buckets: {result['num_buckets']}")
            logger.info(f"  Total size: {result['total_size_mb']:.2f}MB")
            logger.info(
                f"  Bucketing efficiency: "
                f"{result.get('bucketing_efficiency', 0):.2f}MB avg"
            )

    # Find best strategy
    if len(results) > 1:
        best_strategy = min(results.keys(), key=lambda k: results[k]["avg_step_time"])
        logger.info(f"\nBest performing strategy: {best_strategy}")
        logger.info(
            f"Speed improvement: "
            f"{results[best_strategy]['avg_step_time']:.4f}s per step"
        )

    # Run advanced features demo if not in minimal mode
    if not args.single_gpu:
        demonstrate_advanced_features(device)
        demonstrate_memory_efficiency(device)

    logger.info("\n=== Recommendations ===")
    logger.info("1. For models with many small parameters: use 'layer_based' strategy")
    logger.info("2. For models with varied parameter sizes: use 'mixed' strategy")
    logger.info(
        "3. For maximum communication overlap: enable groups with 'parallel' strategy"
    )
    logger.info(
        "4. For memory-constrained environments: use smaller bucket sizes (1-5MB)"
    )
    logger.info("5. Enable dynamic bucketing for adaptive optimization")

    logger.info("\nGradient bucketing example completed successfully!")


if __name__ == "__main__":
    main()
