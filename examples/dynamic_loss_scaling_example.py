#!/usr/bin/env python3
"""
Dynamic Loss Scaling Example for RoseLLM

This example demonstrates how to use the advanced dynamic loss scaling
features in RoseLLM for stable and efficient mixed precision training.

Key Features Demonstrated:
- Dynamic gradient scaler with APEX integration
- Mixed precision manager with autocast
- Overflow detection and handling
- Performance monitoring and statistics
- Checkpointing with mixed precision state
- Comparison with different scaling strategies

Usage:
    python examples/dynamic_loss_scaling_example.py
    # With specific configuration
    python examples/dynamic_loss_scaling_example.py --model-size large \
        --precision fp16 --steps 1000
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# RoseLLM imports
from rosellm.rosetrainer.mixed_precision import (
    DynamicScalerConfig,
    MixedPrecisionConfig,
    MixedPrecisionManager,
    PrecisionType,
    get_recommended_config,
    is_apex_available,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        # Self attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SimpleLanguageModel(nn.Module):
    """Simple language model for demonstration."""

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_length, d_model)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.compute_loss: bool = False  # Will be set to True when needed

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)

        # Embeddings
        token_embeddings = self.embedding(input_ids)
        position_embeddings = self.pos_encoding(position_ids)
        x = token_embeddings + position_embeddings
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)

        x = self.layer_norm(x)
        logits = self.output_projection(x)

        # Compute loss if labels are provided (for demo purposes)
        if hasattr(self, "compute_loss") and input_ids.size(1) > 1:
            # Shift for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            return {"logits": logits, "loss": loss}

        return {"logits": logits}


def create_synthetic_dataset(
    vocab_size: int = 10000,
    seq_length: int = 128,
    num_samples: int = 1000,
    device: torch.device = torch.device("cpu"),
) -> DataLoader:
    """Create synthetic dataset for demonstration."""
    # Generate random sequences
    data = torch.randint(0, vocab_size, (num_samples, seq_length), device=device)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def compare_scaling_strategies(
    model: nn.Module, dataloader: DataLoader, device: torch.device, num_steps: int = 100
) -> Dict[str, Dict]:
    """Compare different scaling strategies."""
    logger.info("Comparing scaling strategies...")

    strategies = {
        "no_scaling": None,
        "constant_scaling": DynamicScalerConfig(
            initial_scale=2**16,
            growth_interval=1000000,  # Very large interval = effectively constant
        ),
        "conservative_dynamic": get_recommended_config("medium", "fp16", "stable"),
        "balanced_dynamic": get_recommended_config("medium", "fp16", "balanced"),
        "aggressive_dynamic": get_recommended_config("medium", "fp16", "aggressive"),
    }

    results = {}

    for name, scaler_config in strategies.items():
        logger.info(f"Testing {name} strategy...")

        # Create fresh model copy
        model_copy = SimpleLanguageModel().to(device)
        model_copy.load_state_dict(model.state_dict())
        model_copy.compute_loss = True  # Enable loss computation

        optimizer = torch.optim.AdamW(model_copy.parameters(), lr=1e-4)

        # Create mixed precision manager
        if name == "no_scaling":
            mp_config = MixedPrecisionConfig(
                precision=PrecisionType.FP32,
                use_dynamic_scaling=False,
            )
        else:
            mp_config = MixedPrecisionConfig(
                precision=PrecisionType.FP16
                if device.type == "cuda"
                else PrecisionType.FP32,
                use_dynamic_scaling=(scaler_config is not None),
                scaler_config=scaler_config,
            )

        mp_manager = MixedPrecisionManager(mp_config, device)

        # Training loop
        losses = []
        times = []

        start_time = time.time()

        for step, (batch,) in enumerate(dataloader):
            if step >= num_steps:
                break

            step_start = time.time()

            optimizer.zero_grad()

            with mp_manager.autocast_context():
                outputs = model_copy(batch[0])
                loss = outputs["loss"]

            mp_manager.backward_step(loss)
            success = mp_manager.optimizer_step(optimizer, model_copy)

            if success:
                losses.append(loss.item())

            times.append(time.time() - step_start)

        total_time = time.time() - start_time

        # Collect statistics
        mp_stats = mp_manager.get_statistics()

        results[name] = {
            "avg_loss": sum(losses) / len(losses) if losses else float("inf"),
            "success_rate": mp_stats.get("success_rate", 0.0),
            "overflow_count": mp_stats.get("overflow_count", 0),
            "avg_step_time": sum(times) / len(times),
            "total_time": total_time,
            "final_scale": mp_stats.get("current_scale", "N/A"),
        }

        logger.info(
            f"{name}: Loss={results[name]['avg_loss']:.4f}, "
            f"Success={results[name]['success_rate']:.2%}, "
            f"Time={results[name]['avg_step_time']*1000:.1f}ms/step"
        )

    return results


def demonstrate_advanced_features(
    model: nn.Module, dataloader: DataLoader, device: torch.device, output_dir: Path
):
    """Demonstrate advanced features of dynamic loss scaling."""
    logger.info("Demonstrating advanced features...")

    # Create advanced configuration
    scaler_config = DynamicScalerConfig(
        initial_scale=2**15,  # 32K
        min_scale=1.0,
        max_scale=2**20,  # 1M
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=500,
        hysteresis=2,
        use_multi_tensor=True,
        enable_inf_nan_check=True,
        log_scale_changes=True,
        detailed_overflow_info=True,
        track_overflow_history=50,
    )

    mp_config = MixedPrecisionConfig(
        precision=PrecisionType.FP16 if device.type == "cuda" else PrecisionType.FP32,
        use_dynamic_scaling=True,
        scaler_config=scaler_config,
        autocast_enabled=True,
        log_overflow_info=True,
        track_scale_history=True,
    )

    mp_manager = MixedPrecisionManager(mp_config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Enable loss computation
    setattr(model, "compute_loss", True)

    # Training with monitoring
    scale_history = []
    loss_history = []
    overflow_events = []

    logger.info("Starting training with advanced monitoring...")

    for step, (batch,) in enumerate(dataloader):
        if step >= 500:  # Limit for demo
            break

        optimizer.zero_grad()

        with mp_manager.autocast_context():
            # batch is a tuple from DataLoader, unpack it
            input_ids = batch[0]
            outputs = model(input_ids)
            loss = outputs["loss"]

        mp_manager.backward_step(loss)
        success = mp_manager.optimizer_step(optimizer, model)

        if success:
            loss_history.append(loss.item())

        # Collect statistics
        stats = mp_manager.get_statistics()
        if "scaler_info" in stats:
            current_scale = stats["scaler_info"].get("current_scale", 0)
            scale_history.append(current_scale)

        # Log overflow events
        if not success:
            overflow_events.append(step)
            logger.warning(f"Overflow at step {step}")

        # Periodic logging
        if step % 100 == 0 and loss_history:
            recent_loss = sum(loss_history[-10:]) / min(10, len(loss_history))
            current_scale = scale_history[-1] if scale_history else "N/A"
            logger.info(f"Step {step}: Loss={recent_loss:.4f}, Scale={current_scale}")

    # Final statistics
    final_stats = mp_manager.get_statistics()
    logger.info(f"Final statistics: {final_stats}")

    # Demonstrate checkpointing
    checkpoint_path = output_dir / "mixed_precision_checkpoint.pt"
    logger.info(f"Saving checkpoint with mixed precision state to {checkpoint_path}")

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "mixed_precision": mp_manager.state_dict(),
        "step": step,
        "loss_history": loss_history[-100:],  # Last 100 losses
    }
    torch.save(checkpoint, checkpoint_path)

    # Demonstrate loading
    logger.info("Demonstrating checkpoint loading...")
    new_model = SimpleLanguageModel().to(device)
    new_optimizer = torch.optim.AdamW(new_model.parameters())
    new_mp_manager = MixedPrecisionManager(mp_config, device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_model.load_state_dict(checkpoint["model"])
    new_optimizer.load_state_dict(checkpoint["optimizer"])
    new_mp_manager.load_state_dict(checkpoint["mixed_precision"])

    logger.info("Checkpoint loading successful!")

    return {
        "scale_history": scale_history,
        "loss_history": loss_history,
        "overflow_events": overflow_events,
        "final_stats": final_stats,
    }


def benchmark_performance(
    device: torch.device,
    model_sizes: List[str] = ["small", "medium"],
    precisions: List[str] = ["fp32", "fp16"],
    batch_sizes: List[int] = [8, 16],
):
    """Benchmark performance across different configurations."""
    logger.info("Running performance benchmarks...")

    results = {}

    for model_size in model_sizes:
        for precision in precisions:
            for batch_size in batch_sizes:
                if precision == "fp16" and device.type == "cpu":
                    continue  # Skip FP16 on CPU

                key = f"{model_size}_{precision}_bs{batch_size}"
                logger.info(f"Benchmarking {key}...")

                # Create model based on size
                if model_size == "small":
                    model = SimpleLanguageModel(
                        vocab_size=5000,
                        d_model=256,
                        nhead=4,
                        num_layers=4,
                        dim_feedforward=1024,
                    ).to(device)
                else:  # medium
                    model = SimpleLanguageModel(
                        vocab_size=10000,
                        d_model=512,
                        nhead=8,
                        num_layers=6,
                        dim_feedforward=2048,
                    ).to(device)

                # Create dataset
                dataloader = create_synthetic_dataset(
                    vocab_size=model.embedding.num_embeddings,
                    seq_length=128,
                    num_samples=batch_size * 50,
                    device=device,
                )

                # Configure mixed precision
                mp_config = MixedPrecisionConfig(
                    precision=PrecisionType(precision),
                    use_dynamic_scaling=(precision != "fp32"),
                    scaler_config=get_recommended_config(
                        model_size, precision, "balanced"
                    ),
                )
                mp_manager = MixedPrecisionManager(mp_config, device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

                # Benchmark
                model.compute_loss = True
                times = []

                for step, (batch,) in enumerate(dataloader):
                    if step >= 20:  # Limited for benchmark
                        break

                    start_time = time.time()

                    optimizer.zero_grad()
                    with mp_manager.autocast_context():
                        outputs = model(batch[0])
                        loss = outputs["loss"]

                    mp_manager.backward_step(loss)
                    mp_manager.optimizer_step(optimizer, model)

                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    times.append(time.time() - start_time)

                avg_time = sum(times[2:]) / len(times[2:])  # Skip first 2 for warmup
                results[key] = {
                    "avg_step_time_ms": avg_time * 1000,
                    "throughput_samples_per_sec": batch_size / avg_time,
                }

                logger.info(
                    f"{key}: {avg_time*1000:.1f}ms/step, "
                    f"{batch_size/avg_time:.1f} samples/sec"
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="Dynamic Loss Scaling Example")
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Model size",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="Training precision",
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Number of training steps"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--output-dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Compare different scaling strategies",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )

    args = parser.parse_args()

    # Setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")
    logger.info(f"APEX available: {is_apex_available()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create model based on size
    model_configs = {
        "small": {"vocab_size": 5000, "d_model": 256, "nhead": 4, "num_layers": 4},
        "medium": {"vocab_size": 10000, "d_model": 512, "nhead": 8, "num_layers": 6},
        "large": {"vocab_size": 15000, "d_model": 768, "nhead": 12, "num_layers": 8},
    }

    model_config = model_configs[args.model_size]
    model = SimpleLanguageModel(**model_config).to(device)

    logger.info(
        f"Created {args.model_size} model with "
        f"{sum(p.numel() for p in model.parameters())} parameters"
    )

    # Create dataset
    dataloader = create_synthetic_dataset(
        vocab_size=model_config["vocab_size"],
        num_samples=args.batch_size * (args.steps + 50),
        device=device,
    )

    # Run different demonstrations
    if args.compare_strategies:
        comparison_results = compare_scaling_strategies(
            model, dataloader, device, args.steps // 5
        )
        logger.info("Strategy comparison results:")
        for strategy, results in comparison_results.items():
            logger.info(f"  {strategy}: {results}")

    if args.benchmark:
        benchmark_results = benchmark_performance(device)
        logger.info("Benchmark results:")
        for config, results in benchmark_results.items():
            logger.info(f"  {config}: {results}")

    # Main demonstration
    logger.info("Running main demonstration with advanced features...")
    demo_results = demonstrate_advanced_features(model, dataloader, device, output_dir)

    # Summary
    logger.info("=== SUMMARY ===")
    logger.info(f"Device: {device}")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Precision: {args.precision}")
    logger.info(f"Final loss: {demo_results['loss_history'][-1]:.4f}")
    logger.info(f"Overflow events: {len(demo_results['overflow_events'])}")
    logger.info(f"Final statistics: {demo_results['final_stats']}")

    if demo_results["scale_history"]:
        min_scale = min(demo_results["scale_history"])
        max_scale = max(demo_results["scale_history"])
        final_scale = demo_results["scale_history"][-1]
        logger.info(
            f"Scale range: {min_scale:.1f} - {max_scale:.1f} (final: {final_scale:.1f})"
        )

    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
