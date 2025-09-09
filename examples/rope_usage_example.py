#!/usr/bin/env python3
"""
Example demonstrating Rotary Position Embeddings (RoPE) usage in RoseLLM.

This example shows:
1. How to configure RoPE for a transformer model
2. Training with RoPE-enabled attention
3. Testing extrapolation to longer sequences
4. Comparing with traditional position embeddings

Usage:
    # Single GPU
    python rope_usage_example.py

    # Multi-GPU (2 GPUs)
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 rope_usage_example.py

    # CPU testing
    CUDA_VISIBLE_DEVICES="" python rope_usage_example.py --device cpu
"""

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Future imports after RoPE implementation
# from rosellm.rosetrainer.embeddings import RotaryEmbedding, apply_rotary_pos_emb
# from rosellm.rosetrainer import RoseTrainer
# from rosellm.rosetrainer.config import TrainerConfig, PositionEmbeddingConfig


@dataclass
class RoPEConfig:
    """Configuration for Rotary Position Embeddings."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    max_position_embeddings: int = 2048
    rotary_percent: float = 1.0
    rotary_interleaved: bool = False
    rotary_base: int = 10000
    rope_scaling: bool = False
    rope_scaling_factor: float = 8.0
    seq_len_interpolation_factor: Optional[float] = None


class MockRotaryEmbedding(nn.Module):
    """Mock RoPE implementation for demonstration purposes."""

    def __init__(self, config: RoPEConfig):
        super().__init__()
        self.config = config

        # Compute rotary dimension
        dim = config.hidden_size // config.num_attention_heads
        if config.rotary_percent < 1.0:
            dim = int(dim * config.rotary_percent)

        # Initialize inverse frequencies
        inv_freq = 1.0 / (config.rotary_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Apply RoPE scaling if enabled
        if config.rope_scaling:
            self._apply_rope_scaling()

    def _apply_rope_scaling(self):
        """Apply LLaMA 3.x style RoPE scaling."""
        factor = self.config.rope_scaling_factor
        # Simplified scaling implementation
        self.inv_freq = self.inv_freq / factor

    def forward(
        self, seq_len: int, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate cos and sin for rotary embeddings."""
        device = self.inv_freq.device  # type: ignore[attr-defined]
        seq = torch.arange(seq_len, device=device) + offset

        if self.config.seq_len_interpolation_factor is not None:
            seq = seq / self.config.seq_len_interpolation_factor

        freqs = torch.outer(seq.float(), self.inv_freq)  # type: ignore[arg-type]
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        return cos, sin


def apply_mock_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K."""

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RoPEAttention(nn.Module):
    """Multi-head attention with RoPE support."""

    def __init__(self, config: RoPEConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # QKV projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # RoPE
        self.rotary_pos_emb = MockRotaryEmbedding(config)

        # Scaling factor
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_pos_emb(seq_len)
        q, k = apply_mock_rotary_pos_emb(q, k, cos, sin)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.config.hidden_size)
        output = self.out_proj(attn_output)

        return output


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block with RoPE attention."""

    def __init__(self, config: RoPEConfig):
        super().__init__()
        self.attention = RoPEAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.ln1(x)
        x = self.attention(x)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x

        return x


def test_rope_extrapolation(model: nn.Module, config: RoPEConfig):
    """Test RoPE's ability to extrapolate to longer sequences."""
    print("\n=== Testing RoPE Extrapolation ===")

    device = next(model.parameters()).device
    model.eval()

    # Test different sequence lengths
    test_lengths = [
        config.max_position_embeddings // 2,  # Half of training length
        config.max_position_embeddings,  # Training length
        config.max_position_embeddings * 2,  # 2x training length
        config.max_position_embeddings * 4,  # 4x training length
    ]

    with torch.no_grad():
        for seq_len in test_lengths:
            # Create dummy input
            batch_size = 2
            input_tensor = torch.randn(
                batch_size, seq_len, config.hidden_size, device=device
            )

            # Forward pass
            start_time = time.time()
            output = model(input_tensor)
            elapsed = time.time() - start_time

            # Check output validity
            is_valid = not torch.isnan(output).any() and not torch.isinf(output).any()

            print(
                f"Sequence Length: {seq_len:5d} | "
                f"Valid: {'✓' if is_valid else '✗'} | "
                f"Time: {elapsed:.3f}s | "
                f"Extrapolation Ratio: {seq_len/config.max_position_embeddings:.1f}x"
            )


def benchmark_rope_vs_learned(config: RoPEConfig):
    """Benchmark RoPE against traditional learned embeddings."""
    print("\n=== Performance Comparison ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    seq_len = 512
    num_iterations = 100

    # Create models
    rope_model = SimpleTransformerBlock(config).to(device)

    # Warmup
    dummy_input = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
    for _ in range(10):
        _ = rope_model(dummy_input)

    # Benchmark forward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    for _ in range(num_iterations):
        _ = rope_model(dummy_input)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    rope_time = time.time() - start_time

    print(
        f"RoPE Attention - Forward Pass: "
        f"{rope_time/num_iterations*1000:.2f}ms per iteration"
    )

    # Memory usage
    if torch.cuda.is_available():
        print(f"GPU Memory Used: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")


def train_with_rope(config: RoPEConfig, num_steps: int = 100):
    """Demonstrate training with RoPE-enabled model."""
    print("\n=== Training with RoPE ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = SimpleTransformerBlock(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Training loop
    batch_size = 4
    seq_len = 256

    losses = []
    for step in range(num_steps):
        # Generate dummy data
        input_data = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
        target = torch.randn(batch_size, seq_len, config.hidden_size, device=device)

        # Forward pass
        output = model(input_data)
        loss = F.mse_loss(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 20 == 0:
            avg_loss = sum(losses[-20:]) / 20
            print(f"Step {step+1:3d} | Loss: {avg_loss:.6f}")

    print(f"Final Loss: {losses[-1]:.6f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="RoPE Usage Example")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--rope-scaling", action="store_true")
    parser.add_argument("--num-train-steps", type=int, default=100)
    args = parser.parse_args()

    # Configure RoPE
    config = RoPEConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        max_position_embeddings=args.max_seq_len,
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=args.rope_scaling,
        rope_scaling_factor=8.0,
    )

    print("=" * 60)
    print("RoPE (Rotary Position Embeddings) Usage Example")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Attention Heads: {config.num_attention_heads}")
    print(f"  Max Sequence Length: {config.max_position_embeddings}")
    print(f"  RoPE Base: {config.rotary_base}")
    print(f"  RoPE Scaling: {config.rope_scaling}")
    if config.rope_scaling:
        print(f"  Scaling Factor: {config.rope_scaling_factor}")

    # Train model with RoPE
    model = train_with_rope(config, args.num_train_steps)

    # Test extrapolation
    test_rope_extrapolation(model, config)

    # Benchmark performance
    benchmark_rope_vs_learned(config)

    print("\n" + "=" * 60)
    print("RoPE demonstration completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
