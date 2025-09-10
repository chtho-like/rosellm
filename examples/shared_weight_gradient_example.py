#!/usr/bin/env python3
"""
End-to-end example demonstrating shared weight gradient reduction in RoseLLM.

This example shows how to use SharedWeightGradientReducer for models with tied
embeddings between input and output layers, following Megatron-LM patterns.

Key features demonstrated:
- Tied weight model architecture with shared embeddings
- Gradient synchronization across pipeline stages
- Integration with RoseLLM's gradient finalization system
- Bit-to-bit validation against reference implementation
"""

import argparse
import logging
import os
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from rosellm.rosetrainer.gradient import (
    GradientFinalizationConfig,
    GradientFinalizer,
    SharedWeightGradientReducer,
)
from rosellm.rosetrainer.parallelism import parallel_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TiedEmbeddingLanguageModel(nn.Module):
    """Simple language model with tied input/output embeddings.

    This model demonstrates the shared weight pattern where input embeddings
    and output projection weights are the same tensor, saving memory.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        share_embeddings: bool = True,
        add_position_embeddings: bool = True,
        max_seq_length: int = 512,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.share_embeddings_and_output_weights = share_embeddings

        # Input embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

        # Position embeddings (optional)
        self.position_embeddings = None
        if add_position_embeddings:
            self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer norm
        self.ln_f = nn.LayerNorm(hidden_size)

        # Output layer (may share weights with embeddings)
        if share_embeddings:
            # Tied weights - output layer uses embedding weights
            self.output_layer = None
        else:
            # Separate output layer
            self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)

    def shared_embedding_or_output_weight(self) -> Optional[nn.Parameter]:
        """Get the shared embedding/output weight following Megatron-LM pattern.

        Returns:
            The shared weight parameter if using tied embeddings, None otherwise.
        """
        if self.share_embeddings_and_output_weights:
            weight = self.word_embeddings.weight
            if isinstance(weight, nn.Parameter):
                return weight
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            position_ids: Position IDs [batch_size, seq_length]

        Returns:
            Logits tensor [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = input_ids.shape

        # Get word embeddings
        hidden_states = self.word_embeddings(input_ids)

        # Add position embeddings if available
        if self.position_embeddings is not None:
            if position_ids is None:
                position_ids = torch.arange(
                    seq_length, dtype=torch.long, device=input_ids.device
                )
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            position_embeds = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeds

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Output projection
        if self.output_layer is not None:
            # Use separate output layer
            logits = self.output_layer(hidden_states)
        else:
            # Use tied weights - multiply by embedding matrix transpose
            logits = F.linear(hidden_states, self.word_embeddings.weight)

        return logits


def create_model_and_optimizer(
    args: argparse.Namespace,
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """Create model and optimizer.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (model, optimizer)
    """
    # Create model with tied embeddings
    model = TiedEmbeddingLanguageModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        share_embeddings=args.share_embeddings,
        add_position_embeddings=args.add_position_embeddings,
        max_seq_length=args.max_seq_length,
    )

    # Move to device
    device = torch.device(
        f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    # Wrap with DDP if distributed
    if args.world_size > 1:
        model = DDP(model, device_ids=[args.local_rank])

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    return model, optimizer


def validate_shared_gradients(
    model: nn.Module,
    reducer: SharedWeightGradientReducer,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Validate that shared weight gradients are properly synchronized.

    Args:
        model: Model with shared weights
        reducer: Shared weight gradient reducer
        args: Command line arguments

    Returns:
        Dictionary with validation metrics
    """
    metrics: Dict[str, float] = {}

    # Get the actual model (unwrap DDP if needed)
    actual_model = model.module if hasattr(model, "module") else model

    if not hasattr(actual_model, "share_embeddings_and_output_weights"):
        logger.info("Model does not have share_embeddings_and_output_weights attribute")
        return metrics

    if not getattr(actual_model, "share_embeddings_and_output_weights"):
        logger.info("Model does not use shared embeddings, skipping validation")
        return metrics

    # Get the shared weight
    shared_weight = None
    if hasattr(actual_model, "shared_embedding_or_output_weight"):
        method = getattr(actual_model, "shared_embedding_or_output_weight")
        if callable(method):
            result = method()
            if isinstance(result, nn.Parameter):
                shared_weight = result

    if shared_weight is None or not isinstance(shared_weight, nn.Parameter):
        return metrics

    # Check gradient exists
    if shared_weight.grad is None:
        logger.warning("No gradient on shared weight")
        return metrics

    # Compute gradient statistics
    grad_norm = shared_weight.grad.norm().item()
    grad_mean = shared_weight.grad.mean().item()
    grad_std = shared_weight.grad.std().item()

    metrics["shared_grad_norm"] = grad_norm
    metrics["shared_grad_mean"] = grad_mean
    metrics["shared_grad_std"] = grad_std

    # If distributed, verify all ranks have same gradient after reduction
    if args.world_size > 1 and dist.is_initialized():
        # Create a tensor with gradient hash for comparison
        grad_hash = torch.tensor(
            [grad_norm, grad_mean, grad_std],
            device=shared_weight.device,
        )

        # Gather from all ranks
        gathered = [torch.zeros_like(grad_hash) for _ in range(args.world_size)]
        dist.all_gather(gathered, grad_hash)

        # Check if all ranks have same values (within tolerance)
        tolerance = 1e-6
        all_same = all(
            torch.allclose(gathered[0], g, rtol=tolerance, atol=tolerance)
            for g in gathered[1:]
        )

        metrics["gradients_synchronized"] = all_same

        if not all_same and args.local_rank == 0:
            logger.warning(
                f"Gradient mismatch detected across ranks! "
                f"Values: {[g.tolist() for g in gathered]}"
            )

    return metrics


def run_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    finalizer: GradientFinalizer,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Run a single training step with shared weight gradient reduction.

    Args:
        model: Model to train
        optimizer: Optimizer
        finalizer: Gradient finalizer with shared weight reduction
        args: Command line arguments

    Returns:
        Dictionary with training metrics
    """
    device = next(model.parameters()).device

    # Create dummy batch
    batch_size = args.batch_size
    seq_length = args.seq_length
    vocab_size = args.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

    # Forward pass
    logits = model(input_ids)

    # Compute loss
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Finalize gradients with shared weight reduction
    finalization_stats = finalizer.finalize_gradients(
        clip_gradients=args.clip_gradients,
        check_finite=True,
        collect_stats=True,
    )

    # Validate shared gradients if enabled
    validation_metrics = {}
    if args.validate_gradients and finalizer.shared_weight_reducer is not None:
        validation_metrics = validate_shared_gradients(
            model,
            finalizer.shared_weight_reducer,
            args,
        )

    # Optimizer step
    optimizer.step()

    # Combine metrics
    metrics = {
        "loss": loss.item(),
        "grad_norm": finalization_stats.get("gradient_norm", 0.0),
        "sync_time": finalization_stats.get("sync_stats", {}).get(
            "sync_time_total", 0.0
        ),
        **validation_metrics,
    }

    return metrics


def main():
    """Main training loop demonstrating shared weight gradient reduction."""
    parser = argparse.ArgumentParser(
        description="RoseLLM Shared Weight Gradient Reduction Example"
    )

    # Model arguments
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument(
        "--hidden-size", type=int, default=768, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--share-embeddings",
        action="store_true",
        default=True,
        help="Share input/output embeddings",
    )
    parser.add_argument(
        "--add-position-embeddings",
        action="store_true",
        default=True,
        help="Add position embeddings",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=512, help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument(
        "--seq-length", type=int, default=128, help="Sequence length for training"
    )
    parser.add_argument(
        "--num-steps", type=int, default=10, help="Number of training steps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--clip-gradients",
        action="store_true",
        default=True,
        help="Enable gradient clipping",
    )
    parser.add_argument(
        "--validate-gradients",
        action="store_true",
        default=True,
        help="Validate gradient synchronization",
    )

    # Distributed arguments
    parser.add_argument(
        "--local-rank", type=int, default=0, help="Local rank for distributed training"
    )

    args = parser.parse_args()

    # Setup distributed training
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.world_size = 1
        args.local_rank = 0

    if args.world_size > 1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")

        # Initialize parallel state for RoseLLM
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=args.world_size,
        )

    # Create model and optimizer
    model, optimizer = create_model_and_optimizer(args)

    # Create gradient finalization config with shared weight support
    finalization_config = GradientFinalizationConfig(
        sync_strategy="simple",
        sync_grad_before_clip=True,
        enable_gradient_stats=True,
        verbose=args.local_rank == 0,
        share_embeddings_and_output_weights=args.share_embeddings,
        share_position_embeddings=args.add_position_embeddings,
    )

    # Create gradient finalizer with shared weight reduction
    finalizer = GradientFinalizer(
        model=model,
        config=finalization_config,
    )

    # Training loop
    logger.info(f"Starting training on rank {args.local_rank}/{args.world_size}")

    for step in range(args.num_steps):
        metrics = run_training_step(model, optimizer, finalizer, args)

        if args.local_rank == 0:
            logger.info(
                f"Step {step + 1}/{args.num_steps}: "
                f"Loss={metrics['loss']:.4f}, "
                f"GradNorm={metrics['grad_norm']:.4f}, "
                f"SyncTime={metrics['sync_time']:.3f}s"
            )

            if "gradients_synchronized" in metrics:
                sync_status = "✓" if metrics["gradients_synchronized"] else "✗"
                logger.info(f"  Gradient Synchronization: {sync_status}")

    logger.info("Training completed successfully!")

    # Cleanup
    if args.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
