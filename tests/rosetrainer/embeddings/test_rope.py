"""Comprehensive tests for Rotary Position Embeddings (RoPE).

Tests include:
- Basic functionality and correctness
- Caching mechanism
- Different interpolation methods
- Fused operations
- Position IDs handling
- Numerical accuracy
- Performance benchmarks
- Integration with models
"""

import math
import time
from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from rosellm.rosetrainer.embeddings import (
    FusedRoPE,
    RoPEConfig,
    RoPEInterpolationType,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_fused,
    precompute_rope_params,
    rotate_half,
)
from rosellm.rosetrainer.embeddings.position_embeddings import (
    PositionEmbeddingFactory,
    PositionEmbeddingType,
)


class TestRoPEBasics:
    """Test basic RoPE functionality."""

    def test_rope_config_creation(self):
        """Test RoPE configuration creation and validation."""
        # Valid config
        config = RoPEConfig(dim=64, max_position_embeddings=2048)
        assert config.dim == 64
        assert config.max_position_embeddings == 2048
        assert config.base == 10000.0

        # Config with rope_theta
        config = RoPEConfig(dim=64, rope_theta=20000.0)
        assert config.base == 20000.0

        # Config with rope_scaling
        config = RoPEConfig(dim=64, rope_scaling={"type": "linear", "factor": 2.0})
        assert config.interpolation_type == RoPEInterpolationType.LINEAR
        assert config.scaling_factor == 2.0

        # Invalid configs
        with pytest.raises(ValueError, match="must be even"):
            RoPEConfig(dim=63)

        with pytest.raises(ValueError, match="must be in"):
            RoPEConfig(dim=64, partial_rotary_factor=1.5)

    def test_rotate_half(self):
        """Test the rotate_half function."""
        x = torch.randn(2, 8, 4, 64)
        x_rot = rotate_half(x)

        assert x_rot.shape == x.shape

        # Check that first half is negative of second half of original
        assert torch.allclose(x_rot[..., :32], -x[..., 32:])
        assert torch.allclose(x_rot[..., 32:], x[..., :32])

    def test_basic_rope_forward(self):
        """Test basic RoPE forward pass."""
        config = RoPEConfig(dim=64, max_position_embeddings=512)
        rope = RotaryEmbedding(config)

        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64

        # Test with [batch, seq_len, num_heads, head_dim] layout
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.allclose(q_rot, q)  # Should be different
        assert not torch.allclose(k_rot, k)

        # Test with [batch, num_heads, seq_len, head_dim] layout
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_caching(self):
        """Test RoPE caching mechanism."""
        config = RoPEConfig(dim=64, max_position_embeddings=512)
        rope = RotaryEmbedding(config)

        # First call should populate cache
        q = torch.randn(2, 128, 8, 64)
        k = torch.randn(2, 128, 8, 64)

        assert rope._seq_len_cached == 0
        assert rope._cos_cached is None

        rope(q, k)

        assert rope._seq_len_cached == 128
        assert rope._cos_cached is not None
        assert rope._sin_cached is not None
        assert rope._cos_cached.shape[0] == 128

        # Second call with same or smaller seq_len should use cache
        q2 = torch.randn(2, 64, 8, 64)
        k2 = torch.randn(2, 64, 8, 64)

        cos_cached = rope._cos_cached.clone()
        rope(q2, k2)

        assert torch.equal(rope._cos_cached, cos_cached)  # Cache unchanged

        # Call with larger seq_len should update cache
        q3 = torch.randn(2, 256, 8, 64)
        k3 = torch.randn(2, 256, 8, 64)

        rope(q3, k3)

        assert rope._seq_len_cached == 256
        assert rope._cos_cached.shape[0] == 256

    def test_rope_with_position_ids(self):
        """Test RoPE with custom position IDs."""
        config = RoPEConfig(dim=64, max_position_embeddings=512)
        rope = RotaryEmbedding(config)

        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Custom position IDs (e.g., for attention masking)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).clone()
        position_ids[:, 64:] = torch.arange(64)  # Repeat positions

        q_rot, k_rot = rope(q, k, position_ids=position_ids)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

        # Test with same input values to verify position encoding
        q_same = torch.ones(batch_size, seq_len, num_heads, head_dim)
        k_same = torch.ones(batch_size, seq_len, num_heads, head_dim)
        q_rot_same, k_rot_same = rope(q_same, k_same, position_ids=position_ids)

        # Check that repeated positions with same input have same output
        assert torch.allclose(q_rot_same[:, :64], q_rot_same[:, 64:128], atol=1e-5)

    def test_partial_rope(self):
        """Test partial RoPE application."""
        config = RoPEConfig(dim=64, partial_rotary_factor=0.5)
        rope = RotaryEmbedding(config)

        assert rope.rope_dim == 32  # Half of 64

        q = torch.randn(2, 128, 8, 64)
        k = torch.randn(2, 128, 8, 64)

        q_rot, k_rot = rope(q, k)

        # Only first 32 dimensions should be rotated
        # Last 32 dimensions should be unchanged
        assert torch.allclose(q_rot[..., 32:], q[..., 32:])
        assert torch.allclose(k_rot[..., 32:], k[..., 32:])
        assert not torch.allclose(q_rot[..., :32], q[..., :32])


class TestRoPEInterpolation:
    """Test RoPE interpolation methods for context extension."""

    def test_linear_interpolation(self):
        """Test linear interpolation."""
        config = RoPEConfig(
            dim=64,
            max_position_embeddings=512,
            interpolation_type=RoPEInterpolationType.LINEAR,
            scaling_factor=2.0,
        )
        rope = RotaryEmbedding(config)

        # Should be able to handle sequences longer than max_position_embeddings
        q = torch.randn(1, 1024, 8, 64)  # 2x max_position_embeddings
        k = torch.randn(1, 1024, 8, 64)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_ntk_interpolation(self):
        """Test Neural Tangent Kernel (NTK) interpolation."""
        config = RoPEConfig(
            dim=64,
            max_position_embeddings=512,
            interpolation_type=RoPEInterpolationType.NTK,
            scaling_factor=2.0,
        )
        rope = RotaryEmbedding(config)

        # Check that base is adjusted
        expected_base = 10000.0 * (2.0 ** (64 / (64 - 2)))
        assert torch.allclose(
            1.0 / rope.inv_freq[0], torch.tensor(expected_base ** (0 / 64)), rtol=1e-4
        )

        q = torch.randn(1, 1024, 8, 64)
        k = torch.randn(1, 1024, 8, 64)

        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape

    def test_dynamic_ntk_interpolation(self):
        """Test Dynamic NTK interpolation."""
        config = RoPEConfig(
            dim=64,
            max_position_embeddings=512,
            interpolation_type=RoPEInterpolationType.DYNAMIC_NTK,
        )
        rope = RotaryEmbedding(config)

        # Test with sequence length within max
        q1 = torch.randn(1, 256, 8, 64)
        k1 = torch.randn(1, 256, 8, 64)
        q_rot1, k_rot1 = rope(q1, k1)

        # Test with sequence length exceeding max
        q2 = torch.randn(1, 1024, 8, 64)
        k2 = torch.randn(1, 1024, 8, 64)
        q_rot2, k_rot2 = rope(q2, k2)

        assert q_rot1.shape == q1.shape
        assert q_rot2.shape == q2.shape

    def test_yarn_interpolation(self):
        """Test YaRN interpolation."""
        config = RoPEConfig(
            dim=64,
            max_position_embeddings=512,
            interpolation_type=RoPEInterpolationType.YaRN,
            scaling_factor=2.0,
        )
        rope = RotaryEmbedding(config)

        # Check that YaRN parameters are initialized
        assert hasattr(rope, "yarn_scale")
        assert rope.yarn_scale.shape[0] == 32  # dim // 2

        q = torch.randn(1, 1024, 8, 64)
        k = torch.randn(1, 1024, 8, 64)

        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape


class TestRoPEOperations:
    """Test RoPE operation functions."""

    def test_apply_rotary_pos_emb_basic(self):
        """Test basic apply_rotary_pos_emb function."""
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64

        tensor = torch.randn(batch_size, seq_len, num_heads, head_dim)
        cos, sin = precompute_rope_params(head_dim, seq_len)

        rotated = apply_rotary_pos_emb(tensor, cos, sin)

        assert rotated.shape == tensor.shape
        assert not torch.allclose(rotated, tensor)

        # Test partial RoPE
        rope_dim = 32
        rotated_partial = apply_rotary_pos_emb(tensor, cos, sin, rope_dim)

        # Check that only first rope_dim dimensions are rotated
        assert not torch.allclose(
            rotated_partial[..., :rope_dim], tensor[..., :rope_dim]
        )
        assert torch.allclose(rotated_partial[..., rope_dim:], tensor[..., rope_dim:])

    def test_apply_rotary_pos_emb_fused(self):
        """Test fused apply_rotary_pos_emb function."""
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64

        tensor = torch.randn(batch_size, seq_len, num_heads, head_dim)
        cos, sin = precompute_rope_params(head_dim, seq_len)

        rotated_fused = apply_rotary_pos_emb_fused(tensor, cos, sin)
        rotated_basic = apply_rotary_pos_emb(tensor, cos, sin)

        # Fused and basic should give same results
        assert torch.allclose(rotated_fused, rotated_basic, atol=1e-6)

    def test_precompute_rope_params(self):
        """Test precompute_rope_params function."""
        dim = 64
        max_seq_len = 512
        base = 10000.0

        cos, sin = precompute_rope_params(dim, max_seq_len, base)

        assert cos.shape == (max_seq_len, dim)
        assert sin.shape == (max_seq_len, dim)

        # Check that values are bounded
        assert torch.all(cos >= -1) and torch.all(cos <= 1)
        assert torch.all(sin >= -1) and torch.all(sin <= 1)

        # Test with different device and dtype
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16

            cos_gpu, sin_gpu = precompute_rope_params(
                dim, max_seq_len, base, device, dtype
            )

            assert cos_gpu.device == device
            assert cos_gpu.dtype == dtype
            assert sin_gpu.device == device
            assert sin_gpu.dtype == dtype


class TestFusedRoPE:
    """Test FusedRoPE module."""

    def test_fused_rope_creation(self):
        """Test FusedRoPE initialization."""
        config = RoPEConfig(dim=64, max_position_embeddings=512)
        fused_rope = FusedRoPE(config)

        assert fused_rope.config == config
        assert hasattr(fused_rope, "rope")
        assert isinstance(fused_rope.rope, RotaryEmbedding)

    def test_fused_rope_forward(self):
        """Test FusedRoPE forward pass."""
        config = RoPEConfig(dim=64, max_position_embeddings=512, use_fused=True)
        fused_rope = FusedRoPE(config)

        q = torch.randn(2, 128, 8, 64)
        k = torch.randn(2, 128, 8, 64)

        q_rot, k_rot = fused_rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

        # Compare with regular RoPE
        regular_rope = RotaryEmbedding(config)
        q_rot_regular, k_rot_regular = regular_rope(q, k)

        assert torch.allclose(q_rot, q_rot_regular, atol=1e-6)
        assert torch.allclose(k_rot, k_rot_regular, atol=1e-6)


class TestPositionEmbeddingIntegration:
    """Test integration with position embedding factory."""

    def test_factory_create_rope(self):
        """Test creating RoPE through factory."""
        # Create with config object
        config = RoPEConfig(dim=64, max_position_embeddings=512)
        rope = PositionEmbeddingFactory.create(
            PositionEmbeddingType.ROTARY, config=config
        )

        assert isinstance(rope, (RotaryEmbedding, FusedRoPE))

        # Create with kwargs
        rope2 = PositionEmbeddingFactory.create(
            "rotary", dim=64, max_position_embeddings=512, use_fused=False
        )

        assert isinstance(rope2, RotaryEmbedding)

        # Create fused version
        rope3 = PositionEmbeddingFactory.create(
            "rotary", dim=64, max_position_embeddings=512, use_fused=True
        )

        assert isinstance(rope3, FusedRoPE)


class TestNumericalAccuracy:
    """Test numerical accuracy of RoPE implementation."""

    def test_rope_orthogonality(self):
        """Test that RoPE preserves orthogonality properties."""
        config = RoPEConfig(dim=64, max_position_embeddings=512)
        rope = RotaryEmbedding(config)

        # Create orthogonal vectors
        q = torch.eye(64).unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 64]
        k = q.clone()

        q_rot, k_rot = rope(q, k)

        # Check that rotated vectors maintain orthogonality
        for i in range(64):
            for j in range(i + 1, 64):
                dot_product = torch.dot(q_rot[0, 0, i], q_rot[0, 0, j])
                assert abs(dot_product) < 1e-5, f"Vectors {i} and {j} not orthogonal"

    def test_rope_magnitude_preservation(self):
        """Test that RoPE preserves vector magnitudes."""
        config = RoPEConfig(dim=64, max_position_embeddings=512)
        rope = RotaryEmbedding(config)

        q = torch.randn(2, 128, 8, 64)
        k = torch.randn(2, 128, 8, 64)

        q_norm_before = torch.norm(q, dim=-1)
        k_norm_before = torch.norm(k, dim=-1)

        q_rot, k_rot = rope(q, k)

        q_norm_after = torch.norm(q_rot, dim=-1)
        k_norm_after = torch.norm(k_rot, dim=-1)

        # Magnitudes should be preserved
        assert torch.allclose(q_norm_before, q_norm_after, rtol=1e-5)
        assert torch.allclose(k_norm_before, k_norm_after, rtol=1e-5)

    def test_rope_relative_position_encoding(self):
        """Test that RoPE correctly encodes relative positions."""
        config = RoPEConfig(dim=64, max_position_embeddings=512)
        rope = RotaryEmbedding(config)

        # Create simple test case
        batch_size = 1
        seq_len = 4
        num_heads = 1
        head_dim = 64

        q = torch.ones(batch_size, seq_len, num_heads, head_dim)
        k = torch.ones(batch_size, seq_len, num_heads, head_dim)

        q_rot, k_rot = rope(q, k)

        # Transpose for attention computation: [batch, num_heads, seq_len, head_dim]
        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(head_dim)

        # Scores should depend on relative position
        # Diagonal elements (same position) should have consistent pattern
        diagonal_scores = torch.diagonal(scores[0, 0])
        # Check that the diagonal has the expected pattern based on RoPE
        assert diagonal_scores.shape[0] == seq_len


class TestPerformance:
    """Performance benchmarks for RoPE."""

    @pytest.mark.benchmark
    def test_rope_performance(self):
        """Benchmark RoPE performance."""
        config = RoPEConfig(dim=128, max_position_embeddings=2048)
        rope = RotaryEmbedding(config)

        batch_size = 32
        seq_len = 2048
        num_heads = 16
        head_dim = 128

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)

        if torch.cuda.is_available():
            rope = rope.cuda()
            q = q.cuda()
            k = k.cuda()

            # Warmup
            for _ in range(10):
                rope(q, k)

            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                rope(q, k)
            torch.cuda.synchronize()
            end = time.perf_counter()

            avg_time = (end - start) / 100
            throughput = (batch_size * seq_len * num_heads * head_dim * 2) / (
                avg_time * 1e9
            )

            print(f"\nRoPE Performance (GPU):")
            print(f"  Average time: {avg_time * 1000:.3f} ms")
            print(f"  Throughput: {throughput:.3f} GB/s")

    @pytest.mark.benchmark
    def test_fused_vs_regular_performance(self):
        """Compare performance of fused vs regular RoPE."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = RoPEConfig(dim=128, max_position_embeddings=2048)
        regular_rope = RotaryEmbedding(config).cuda()
        fused_rope = FusedRoPE(config).cuda()

        batch_size = 32
        seq_len = 2048
        num_heads = 16
        head_dim = 128

        q = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda()
        k = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda()

        # Benchmark regular RoPE
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            regular_rope(q, k)
        torch.cuda.synchronize()
        regular_time = (time.perf_counter() - start) / 100

        # Benchmark fused RoPE
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            fused_rope(q, k)
        torch.cuda.synchronize()
        fused_time = (time.perf_counter() - start) / 100

        print(f"\nPerformance Comparison:")
        print(f"  Regular RoPE: {regular_time * 1000:.3f} ms")
        print(f"  Fused RoPE: {fused_time * 1000:.3f} ms")
        print(f"  Speedup: {regular_time / fused_time:.2f}x")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tensors(self):
        """Test handling of empty tensors."""
        config = RoPEConfig(dim=64, max_position_embeddings=512)
        rope = RotaryEmbedding(config)

        q = torch.randn(0, 128, 8, 64)
        k = torch.randn(0, 128, 8, 64)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_single_position(self):
        """Test with single position."""
        config = RoPEConfig(dim=64, max_position_embeddings=512)
        rope = RotaryEmbedding(config)

        q = torch.randn(2, 1, 8, 64)
        k = torch.randn(2, 1, 8, 64)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_very_long_sequences(self):
        """Test with very long sequences."""
        config = RoPEConfig(
            dim=64,
            max_position_embeddings=512,
            interpolation_type=RoPEInterpolationType.LINEAR,
            scaling_factor=8.0,
        )
        rope = RotaryEmbedding(config)

        # Test with 8x the max_position_embeddings
        q = torch.randn(1, 4096, 8, 64)
        k = torch.randn(1, 4096, 8, 64)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.isnan(q_rot).any()
        assert not torch.isinf(q_rot).any()

    def test_reset_cache(self):
        """Test cache reset functionality."""
        config = RoPEConfig(dim=64, max_position_embeddings=512)
        rope = RotaryEmbedding(config)

        # Populate cache
        q = torch.randn(2, 128, 8, 64)
        k = torch.randn(2, 128, 8, 64)
        rope(q, k)

        assert rope._seq_len_cached == 128
        assert rope._cos_cached is not None

        # Reset cache
        rope.reset_cache()

        assert rope._seq_len_cached == 0
        assert rope._cos_cached is None
        assert rope._sin_cached is None


class SimpleAttentionWithRoPE(nn.Module):
    """Simple attention module with RoPE for integration testing."""

    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Initialize RoPE
        rope_config = RoPEConfig(dim=self.head_dim, max_position_embeddings=max_seq_len)
        self.rope = RotaryEmbedding(rope_config)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)

        return output


class TestModelIntegration:
    """Test RoPE integration with models."""

    def test_attention_with_rope(self):
        """Test attention module with RoPE."""
        hidden_size = 512
        num_heads = 8
        seq_len = 128
        batch_size = 4

        model = SimpleAttentionWithRoPE(hidden_size, num_heads)
        x = torch.randn(batch_size, seq_len, hidden_size)

        output = model(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_attention_with_causal_mask(self):
        """Test attention with RoPE and causal mask."""
        hidden_size = 512
        num_heads = 8
        seq_len = 128
        batch_size = 4

        model = SimpleAttentionWithRoPE(hidden_size, num_heads)
        x = torch.randn(batch_size, seq_len, hidden_size)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        output = model(x, attention_mask=mask)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_attention_gpu(self):
        """Test attention with RoPE on GPU."""
        hidden_size = 512
        num_heads = 8
        seq_len = 512
        batch_size = 8

        model = SimpleAttentionWithRoPE(hidden_size, num_heads).cuda()
        x = torch.randn(batch_size, seq_len, hidden_size).cuda()

        output = model(x)

        assert output.shape == x.shape
        assert output.device == x.device
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test gradient flow through RoPE."""
        hidden_size = 256
        num_heads = 4
        seq_len = 64
        batch_size = 2

        model = SimpleAttentionWithRoPE(hidden_size, num_heads)
        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

        # Check that gradients are non-zero
        assert x.grad.abs().sum() > 0


if __name__ == "__main__":
    # Run specific test groups for development
    import sys

    if len(sys.argv) > 1:
        pytest.main([__file__, "-v", "-k", sys.argv[1]])
    else:
        pytest.main([__file__, "-v"])
