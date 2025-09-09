#!/usr/bin/env python3
r"""
Advanced Gradient Finalization Example for RoseLLM

This example demonstrates the advanced gradient finalization capabilities
of RoseLLM, including:
- Multi-precision gradient data type management
- Multi-dimensional parallelism aware operations
- Advanced gradient synchronization strategies
- Integration with RoseTrainer

Usage:
    # Single GPU/CPU
    python advanced_gradient_finalization_example.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=2 advanced_gradient_finalization_example.py --distributed

    # Advanced configuration
    python advanced_gradient_finalization_example.py \
        --master-precision fp16 --enable-compression --verbose
"""

import argparse
import logging
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from rosellm.rosetrainer import RoseTrainer
    from rosellm.rosetrainer.config import TrainingConfig
    from rosellm.rosetrainer.gradient import (
        AdvancedGradientFinalizer,
        GradientFinalizationConfig,
        create_gradient_data_type_manager,
    )

    ROSELLM_AVAILABLE = True
except ImportError as e:
    logger.error(f"RoseLLM not available: {e}")
    ROSELLM_AVAILABLE = False


class SimpleTransformerModel(nn.Module):
    """Simple transformer model for demonstration."""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        max_seq_length: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        batch_size, seq_length = input_ids.shape

        # Create position ids
        position_ids = (
            torch.arange(seq_length, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        embeddings = self.dropout(token_embeds + position_embeds)

        # Transformer
        transformer_output = self.transformer(embeddings)

        # Output projection
        logits = self.output_projection(transformer_output)

        # Compute loss if labels provided
        if labels is not None:
            # Shift labels for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}


def create_dummy_dataset(
    num_samples: int = 1000,
    seq_length: int = 128,
    vocab_size: int = 1000,
    batch_size: int = 8,
):
    """Create a dummy dataset for demonstration."""
    # Generate random sequences
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    labels = torch.randint(0, vocab_size, (num_samples, seq_length))

    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def demonstrate_standalone_gradient_finalization():
    """Demonstrate standalone advanced gradient finalization."""
    logger.info("=== Standalone Gradient Finalization Demo ===")

    # Create a simple model
    model = SimpleTransformerModel(vocab_size=100, d_model=128, num_layers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create dummy data
    batch_size = 4
    seq_length = 32
    vocab_size = 100

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

    # Forward pass to create gradients
    outputs = model(input_ids, labels)
    loss = outputs["loss"]
    loss.backward()

    logger.info(f"Initial loss: {loss.item():.4f}")

    # Create gradient data type manager
    data_type_manager = create_gradient_data_type_manager(
        master_precision="fp32",
        communication_precision="fp16",
        enable_compression=True,
    )

    # Create gradient finalization config
    config = GradientFinalizationConfig(
        sync_strategy="hierarchical",
        reduction_op="mean",
        enable_gradient_stats=True,
        verbose=True,
    )

    # Create advanced gradient finalizer
    finalizer = AdvancedGradientFinalizer(
        model=model,
        config=config,
        data_type_manager=data_type_manager,
        enable_advanced_sync=True,
        verbose=True,
    )

    # Finalize gradients
    logger.info("Finalizing gradients with advanced features...")
    finalization_stats = finalizer.finalize_gradients(
        clip_gradients=True,
        check_finite=True,
        normalize_gradients=True,
        collect_stats=True,
    )

    logger.info("Gradient finalization completed!")
    logger.info(f"Gradient norm: {finalization_stats['gradient_norm']:.4f}")
    logger.info(f"Finalization time: {finalization_stats['finalization_time']:.3f}s")
    logger.info(f"Success: {finalization_stats.get('success', False)}")

    # Get performance metrics
    perf_metrics = finalizer.get_performance_metrics()
    logger.info(f"Performance metrics: {perf_metrics}")

    # Get parallelism info
    parallel_info = finalizer.get_parallelism_info()
    logger.info(f"Parallelism info: {parallel_info}")

    # Clean up
    finalizer.cleanup()
    logger.info("Standalone demo completed successfully!")


def demonstrate_integrated_training():
    """Demonstrate integrated training with RoseTrainer."""
    logger.info("=== Integrated Training Demo ===")

    if not ROSELLM_AVAILABLE:
        logger.error("RoseLLM not available for integrated training demo")
        return

    # Create model and optimizer
    model = SimpleTransformerModel(vocab_size=500, d_model=256, num_layers=3)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Create training configuration with advanced gradient finalization
    config = TrainingConfig(
        batch_size=8,
        num_epochs=1,
        max_steps=10,  # Limit for demo
        warmup_steps=0,
        seed=42,
        checkpoint_interval=50,
        log_interval=1,
        eval_interval=10,
    )

    # Update gradient configuration
    config.gradient.enable_advanced_finalization = True
    config.gradient.master_precision = "fp32"
    config.gradient.communication_precision = "fp16"
    config.gradient.enable_gradient_compression = True
    config.gradient.compression_threshold_mb = 1.0
    config.gradient.normalize_gradients = True
    config.gradient.track_gradient_stats = True
    config.gradient.finalization_verbose = True
    config.gradient.clip_value = 1.0

    # Disable mixed precision for demo simplicity
    config.mixed_precision.enabled = False

    logger.info("Configuration created with advanced gradient finalization enabled")

    # Create trainer
    trainer = RoseTrainer(
        model=model,
        optimizer=optimizer,
        config=config,
    )

    # Create dummy dataset
    dataloader = create_dummy_dataset(num_samples=100, batch_size=8)

    logger.info("Starting training with advanced gradient finalization...")

    # Training loop
    total_steps = 0
    total_loss = 0.0

    num_epochs = config.num_epochs or 1
    for epoch in range(num_epochs):
        for step, (input_ids, labels) in enumerate(dataloader):
            if step >= 10:  # Limit for demo
                break

            # Prepare batch
            batch = {"input_ids": input_ids, "labels": labels}

            # Training step with advanced gradient finalization
            metrics = trainer.train_step(batch)

            total_loss += metrics["loss"]
            total_steps += 1

            # Log metrics
            if step % 2 == 0:
                logger.info(f"Step {step}: {metrics}")

                # Get current learning rate
                current_lr = trainer.get_current_lr()
                logger.info(f"Current LR: {current_lr:.6f}")

                # Get memory stats
                memory_stats = trainer.get_memory_stats()
                logger.info(f"Memory: {memory_stats}")

    # Get final performance report
    performance_report = trainer.get_performance_report()
    logger.info(f"Final performance report: {performance_report}")

    # Clean up
    trainer.cleanup()

    avg_loss = total_loss / total_steps if total_steps > 0 else 0
    logger.info(f"Training completed! Average loss: {avg_loss:.4f}")


def demonstrate_gradient_data_types():
    """Demonstrate gradient data type management."""
    logger.info("=== Gradient Data Type Management Demo ===")

    # Create a simple model
    model = nn.Linear(100, 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create dummy data and compute gradients
    x = torch.randn(10, 100, device=device)
    y = torch.randn(10, 50, device=device)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    logger.info("Original gradient dtypes:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            logger.info(f"  {name}: {param.grad.dtype}")

    # Test different precision configurations
    configs = [
        ("fp32_master", "fp32", None, False),
        ("fp16_comm", "fp32", "fp16", False),
        ("compressed", "fp32", "fp16", True),
        ("bf16_master", "bf16", "fp16", True),
    ]

    for name, master, comm, compress in configs:
        logger.info(f"\nTesting configuration: {name}")

        # Create data type manager
        try:
            dtm = create_gradient_data_type_manager(
                master_precision=master,
                communication_precision=comm,
                enable_compression=compress,
                compression_threshold_mb=0.1,  # Low threshold for demo
            )

            # Convert to master precision
            master_grads = dtm.convert_gradients_to_master(model)
            logger.info(
                f"  Converted {len(master_grads)} gradients to master precision"
            )

            # Convert for communication
            comm_grads, metadata = dtm.convert_gradients_for_communication(master_grads)
            logger.info(
                f"  Communication conversion - "
                f"compression ratio: {metadata['compression_ratio']:.3f}"
            )

            # Restore from communication
            dtm.restore_gradients_from_communication(model, comm_grads, metadata)
            logger.info(f"  Restored gradients from communication")

            # Get statistics
            stats = dtm.get_statistics()
            logger.info(f"  Statistics: {stats}")

            # Cleanup
            dtm.cleanup()

        except Exception as e:
            logger.error(f"  Configuration {name} failed: {e}")

    logger.info("Gradient data type management demo completed!")


def main():
    """Main function to run the advanced gradient finalization examples."""
    parser = argparse.ArgumentParser(
        description="Advanced Gradient Finalization Example for RoseLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--demo",
        choices=["standalone", "integrated", "datatypes", "all"],
        default="all",
        help="Which demo to run",
    )

    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training (requires torchrun)",
    )

    parser.add_argument(
        "--master-precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Master precision for gradients",
    )

    parser.add_argument(
        "--enable-compression",
        action="store_true",
        help="Enable gradient compression",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger("rosellm").setLevel(logging.DEBUG)
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    logger.info("Starting Advanced Gradient Finalization Example")
    logger.info(f"Arguments: {args}")

    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")

    # Check distributed environment
    if args.distributed:
        if "LOCAL_RANK" in os.environ:
            logger.info(
                f"Distributed training: LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
                f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}"
            )
        else:
            logger.warning("Distributed flag set but no LOCAL_RANK found")

    # Run selected demos
    demos_to_run = []
    if args.demo == "all":
        demos_to_run = ["datatypes", "standalone", "integrated"]
    else:
        demos_to_run = [args.demo]

    results = {}

    for demo_name in demos_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running demo: {demo_name}")
        logger.info(f"{'='*60}")

        start_time = time.time()
        success = True
        error_msg = None

        try:
            if demo_name == "standalone":
                demonstrate_standalone_gradient_finalization()
            elif demo_name == "integrated":
                demonstrate_integrated_training()
            elif demo_name == "datatypes":
                demonstrate_gradient_data_types()
            else:
                raise ValueError(f"Unknown demo: {demo_name}")

        except Exception as e:
            logger.error(f"Demo {demo_name} failed: {e}")
            success = False
            error_msg = str(e)

        elapsed_time = time.time() - start_time
        results[demo_name] = {
            "success": success,
            "time": elapsed_time,
            "error": error_msg,
        }

        logger.info(
            f"Demo {demo_name} completed in {elapsed_time:.2f}s - "
            f"{'SUCCESS' if success else 'FAILED'}"
        )

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")

    for demo_name, result in results.items():
        status = "✓ PASSED" if result["success"] else "✗ FAILED"
        logger.info(f"{demo_name:20} | {status:8} | {result['time']:6.2f}s")
        if not result["success"] and result["error"]:
            logger.info(f"                     | Error: {result['error']}")

    total_success = sum(1 for r in results.values() if r["success"])
    total_demos = len(results)

    logger.info(f"\nOverall: {total_success}/{total_demos} demos passed")

    if total_success == total_demos:
        logger.info("🎉 All demos completed successfully!")
        return 0
    else:
        logger.error("❌ Some demos failed!")
        return 1


if __name__ == "__main__":
    exit(main())
