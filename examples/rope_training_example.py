#!/usr/bin/env python3
"""Example demonstrating RoPE (Rotary Position Embeddings) with RoseLLM.

This example shows how to:
1. Configure and use RoPE in a transformer model
2. Compare different interpolation methods for context extension
3. Integrate RoPE with RoseTrainer for distributed training
4. Benchmark performance of different RoPE configurations
"""

import argparse
import math
import os
import time
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from rosellm.rosetrainer import RoseTrainer
from rosellm.rosetrainer.config import PositionEmbeddingConfig, TrainingConfig
from rosellm.rosetrainer.embeddings import (
    FusedRoPE,
    RoPEConfig,
    RoPEInterpolationType,
    RotaryEmbedding,
)


class SyntheticTextDataset(Dataset):
    """Synthetic dataset for testing RoPE."""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random token sequences
        input_ids = torch.randint(1, self.vocab_size, (self.seq_length,))
        # Use shifted input as labels for language modeling
        labels = torch.cat([input_ids[1:], torch.tensor([0])])
        return {"input_ids": input_ids, "labels": labels}


class MultiHeadAttentionWithRoPE(nn.Module):
    """Multi-head attention with RoPE support."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rope_config: Optional[RoPEConfig] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert (
            self.head_dim * num_heads == hidden_size
        ), "hidden_size must be divisible by num_heads"

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Initialize RoPE if config provided
        self.rope: Optional[Union[FusedRoPE, RotaryEmbedding]] = None
        if rope_config is not None:
            # Ensure RoPE dim matches head_dim
            if rope_config.dim != self.head_dim:
                rope_config.dim = self.head_dim

            if rope_config.use_fused:
                self.rope = FusedRoPE(rope_config)
            else:
                self.rope = RotaryEmbedding(rope_config)
        else:
            self.rope = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE if configured
        if self.rope is not None:
            q, k = self.rope(q, k, position_ids=position_ids)

        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Final projection
        output = self.out_proj(attn_output)

        return output


class TransformerBlockWithRoPE(nn.Module):
    """Transformer block with RoPE support."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rope_config: Optional[RoPEConfig] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Multi-head attention with RoPE
        self.attention = MultiHeadAttentionWithRoPE(
            hidden_size, num_heads, rope_config, dropout
        )

        # Feed-forward network
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SimpleTransformerWithRoPE(nn.Module):
    """Simple transformer model with RoPE for demonstration."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_length: int,
        rope_config: Optional[RoPEConfig] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlockWithRoPE(
                    hidden_size, num_heads, rope_config, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        # Output layer
        self.ln_final = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie input and output embeddings
        self.lm_head.weight = self.token_embeddings.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        hidden_states = self.token_embeddings(input_ids)
        hidden_states = self.dropout(hidden_states)

        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device) * float("-inf"),
                diagonal=1,
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        # Pass through transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, position_ids)

        # Final layer norm
        hidden_states = self.ln_final(hidden_states)

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        return logits


def create_rope_config(args) -> Optional[RoPEConfig]:
    """Create RoPE configuration from arguments."""
    if not args.use_rope:
        return None

    interpolation_type = RoPEInterpolationType.NONE
    if args.rope_interpolation:
        interpolation_type = RoPEInterpolationType(args.rope_interpolation)

    return RoPEConfig(
        dim=args.hidden_size // args.num_heads,  # head_dim
        max_position_embeddings=args.max_seq_length,
        base=args.rope_base,
        interpolation_type=interpolation_type,
        scaling_factor=args.rope_scaling_factor,
        partial_rotary_factor=args.rope_partial_factor,
        use_fused=args.use_fused_rope,
    )


def train_step(
    model: nn.Module,
    batch: dict,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Tensor, dict]:
    """Single training step."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Forward pass
    logits = model(input_ids)

    # Compute loss
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

    # Compute perplexity
    perplexity = torch.exp(loss).item()

    metrics = {
        "loss": loss.item(),
        "perplexity": perplexity,
    }

    return loss, metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)

    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
    }


def benchmark_rope_performance(args):
    """Benchmark different RoPE configurations."""
    print("\n" + "=" * 60)
    print("Benchmarking RoPE Performance")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    seq_length = args.max_seq_length

    # Test configurations
    configs = [
        ("No RoPE", None),
        (
            "Standard RoPE",
            RoPEConfig(
                dim=args.hidden_size // args.num_heads,
                max_position_embeddings=seq_length,
            ),
        ),
        (
            "Fused RoPE",
            RoPEConfig(
                dim=args.hidden_size // args.num_heads,
                max_position_embeddings=seq_length,
                use_fused=True,
            ),
        ),
        (
            "Partial RoPE (50%)",
            RoPEConfig(
                dim=args.hidden_size // args.num_heads,
                max_position_embeddings=seq_length,
                partial_rotary_factor=0.5,
            ),
        ),
        (
            "Linear Interpolation (2x)",
            RoPEConfig(
                dim=args.hidden_size // args.num_heads,
                max_position_embeddings=seq_length,
                interpolation_type=RoPEInterpolationType.LINEAR,
                scaling_factor=2.0,
            ),
        ),
        (
            "NTK Interpolation (2x)",
            RoPEConfig(
                dim=args.hidden_size // args.num_heads,
                max_position_embeddings=seq_length,
                interpolation_type=RoPEInterpolationType.NTK,
                scaling_factor=2.0,
            ),
        ),
    ]

    results = []

    for name, rope_config in configs:
        # Create model
        model = SimpleTransformerWithRoPE(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_seq_length=seq_length,
            rope_config=rope_config,
        ).to(device)

        # Create dummy input
        input_ids = torch.randint(
            1, args.vocab_size, (batch_size, seq_length), device=device
        )

        # Warmup
        for _ in range(10):
            _ = model(input_ids)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()
        num_iterations = 100

        for _ in range(num_iterations):
            _ = model(input_ids)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_iterations * 1000  # ms

        # Calculate throughput
        total_tokens = batch_size * seq_length
        throughput = total_tokens / (avg_time / 1000)  # tokens/sec

        results.append(
            {
                "config": name,
                "avg_time_ms": avg_time,
                "throughput": throughput,
            }
        )

        print(f"\n{name}:")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  Throughput: {throughput:.0f} tokens/sec")

    # Compare results
    print("\n" + "-" * 60)
    print("Performance Summary:")
    print("-" * 60)

    baseline = results[0]["avg_time_ms"]
    for result in results:
        speedup = baseline / result["avg_time_ms"]
        print(
            f"{result['config']:25s}: {result['avg_time_ms']:6.2f} ms "
            f"(speedup: {speedup:.2f}x)"
        )


def main():
    parser = argparse.ArgumentParser(description="RoPE training example with RoseLLM")

    # Model configuration
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size")
    parser.add_argument(
        "--num-layers", type=int, default=12, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=512, help="Maximum sequence length"
    )

    # RoPE configuration
    parser.add_argument("--use-rope", action="store_true", help="Use RoPE embeddings")
    parser.add_argument(
        "--rope-base", type=float, default=10000.0, help="RoPE base frequency"
    )
    parser.add_argument(
        "--rope-interpolation",
        type=str,
        default="none",
        choices=["none", "linear", "ntk", "dynamic_ntk", "yarn"],
        help="RoPE interpolation method",
    )
    parser.add_argument(
        "--rope-scaling-factor",
        type=float,
        default=1.0,
        help="RoPE scaling factor for context extension",
    )
    parser.add_argument(
        "--rope-partial-factor",
        type=float,
        default=1.0,
        help="Fraction of dimensions to apply RoPE to",
    )
    parser.add_argument(
        "--use-fused-rope", action="store_true", help="Use fused RoPE operations"
    )

    # Training configuration
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of training samples"
    )

    # Other options
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Run benchmark if requested
    if args.benchmark:
        benchmark_rope_performance(args)
        return

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create RoPE configuration
    rope_config = create_rope_config(args)
    if rope_config:
        print(f"\nRoPE Configuration:")
        print(f"  Dimension: {rope_config.dim}")
        print(f"  Base: {rope_config.base}")
        print(f"  Interpolation: {rope_config.interpolation_type.value}")
        print(f"  Scaling factor: {rope_config.scaling_factor}")
        print(f"  Partial factor: {rope_config.partial_rotary_factor}")
        print(f"  Use fused: {rope_config.use_fused}")

    # Create model
    print(f"\nCreating model...")
    model = SimpleTransformerWithRoPE(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_length=args.max_seq_length,
        rope_config=rope_config,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create datasets
    print(f"\nCreating datasets...")
    train_dataset = SyntheticTextDataset(
        args.num_samples, args.max_seq_length, args.vocab_size
    )
    eval_dataset = SyntheticTextDataset(
        args.num_samples // 10, args.max_seq_length, args.vocab_size
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training with RoseTrainer (if available)
    try:
        # Create training config
        from rosellm.rosetrainer.config import OptimizerConfig

        optimizer_config = OptimizerConfig(
            name="adamw",
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        config = TrainingConfig(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            max_steps=None,
            warmup_steps=100,
            seed=args.seed,
            checkpoint_interval=1000,
            log_interval=10,
            eval_interval=100,
            optimizer=optimizer_config,
        )

        # Setup position embedding config if using RoPE
        if rope_config:
            config.position_embedding = PositionEmbeddingConfig(
                embedding_type="rotary",
                max_position_embeddings=args.max_seq_length,
                hidden_size=None,  # Not needed for RoPE
                rope_dim=rope_config.dim,
                rope_base=rope_config.base,
                rope_scaling=None,  # Using individual params instead
                rope_interpolation_type=rope_config.interpolation_type.value,
                rope_scaling_factor=rope_config.scaling_factor,
                rope_partial_factor=rope_config.partial_rotary_factor,
                rope_use_fused=rope_config.use_fused,
                alibi_num_heads=None,  # Not using ALiBi
                learned_dropout=0.0,  # Not using learned embeddings
            )

        # Initialize trainer
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        trainer = RoseTrainer(
            model=model,
            optimizer=optimizer,
            config=config,
            local_rank=local_rank,
            world_size=world_size,
        )

        print("\nTraining with RoseTrainer...")
        print("=" * 60)

    except Exception as e:
        print(f"\nNote: RoseTrainer not available ({e}), using simple training loop")
        trainer = None

    # Simple training loop (fallback)
    if trainer is None:
        print("\nTraining with simple loop...")
        print("=" * 60)

        for epoch in range(args.num_epochs):
            model.train()
            total_loss = 0.0
            total_samples = 0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                loss, metrics = train_step(model, batch, criterion, device)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{args.num_epochs} "
                        f"[{batch_idx}/{len(train_loader)}] "
                        f"Loss: {metrics['loss']:.4f} "
                        f"Perplexity: {metrics['perplexity']:.2f}"
                    )

            # Evaluate
            eval_metrics = evaluate_model(model, eval_loader, criterion, device)
            print(f"\nEpoch {epoch+1} Evaluation:")
            print(f"  Loss: {eval_metrics['eval_loss']:.4f}")
            print(f"  Perplexity: {eval_metrics['eval_perplexity']:.2f}")
            print("-" * 60)

    print("\nTraining completed!")

    # Test different sequence lengths
    if rope_config and rope_config.interpolation_type != RoPEInterpolationType.NONE:
        print("\nTesting context extension...")
        print("=" * 60)

        test_lengths = [
            args.max_seq_length,
            args.max_seq_length * 2,
            args.max_seq_length * 4,
        ]

        model.eval()
        with torch.no_grad():
            for test_len in test_lengths:
                try:
                    input_ids = torch.randint(
                        1, args.vocab_size, (1, test_len), device=device
                    )
                    _ = model(input_ids)
                    print(f"Successfully processed sequence length: {test_len}")
                except Exception as e:
                    print(f"Failed at sequence length {test_len}: {e}")


if __name__ == "__main__":
    main()
