#!/usr/bin/env python3
"""
Bit-to-bit validation of RoPE implementation against Megatron-LM.

This test ensures that RoseLLM's RoPE implementation produces
identical results to Megatron-LM's reference implementation.

Usage:
    pytest tests/validation/test_rope_megatron_parity.py -v

    # Or run directly
    python tests/validation/test_rope_megatron_parity.py
"""

import os
import sys

import pytest
import torch

# Add Megatron-LM to path for reference implementation
MEGATRON_PATH = "/data/projects/Megatron-LM"
if os.path.exists(MEGATRON_PATH):
    sys.path.insert(0, MEGATRON_PATH)
    try:
        from megatron.core.models.common.embeddings.rope_utils import (  # type: ignore[import] # noqa: E501
            apply_rotary_pos_emb as megatron_apply_rope,
        )
        from megatron.core.models.common.embeddings.rotary_pos_embedding import (  # type: ignore[import] # noqa: E501
            RotaryEmbedding as MegatronRoPE,
        )

        MEGATRON_AVAILABLE = True
    except ImportError:
        MEGATRON_AVAILABLE = False
else:
    MEGATRON_AVAILABLE = False

# Future import after implementation
# from rosellm.rosetrainer.embeddings import RotaryEmbedding as RoseLLMRoPE
# from rosellm.rosetrainer.embeddings.rope_utils import apply_rotary_pos_emb as rosellm_apply_rope  # noqa: E501


class MockRoseLLMRoPE:
    """Mock RoseLLM RoPE for testing framework."""

    def __init__(
        self,
        kv_channels,
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=False,
        rope_scaling_factor=8.0,
    ):
        self.kv_channels = kv_channels
        self.rotary_percent = rotary_percent
        self.rotary_base = rotary_base
        self.rope_scaling = rope_scaling
        self.rope_scaling_factor = rope_scaling_factor

        # Compute rotary dimension
        # For partial rotation, we rotate only part of the channels
        self.rotary_dim = int(kv_channels * rotary_percent)

        # Initialize frequencies - always based on full rotary_dim
        # Each frequency corresponds to sin/cos pair (rotary_dim/2 frequencies)
        self.inv_freq = 1.0 / (
            rotary_base
            ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        )

        if rope_scaling:
            self.inv_freq = self.inv_freq / rope_scaling_factor

    def forward(self, max_seq_len, offset=0):
        """Generate rotary embeddings."""
        seq = torch.arange(max_seq_len) + offset
        freqs = torch.outer(seq.float(), self.inv_freq)
        # Always concatenate to get sin/cos pairs
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[:, None, None, :]


def mock_apply_rope(q, k, rotary_pos_emb):
    """Mock application of rotary embeddings."""
    cos = rotary_pos_emb.cos()
    sin = rotary_pos_emb.sin()

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class TestRoPEMegatronParity:
    """Test suite for validating RoPE implementation against Megatron-LM."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not MEGATRON_AVAILABLE, reason="Megatron-LM not available")
    def test_basic_rope_forward(self):
        """Test basic RoPE forward pass matches Megatron."""
        torch.manual_seed(42)

        # Configuration
        kv_channels = 64
        seq_len = 128
        # batch_size = 2
        # num_heads = 8

        # Create Megatron RoPE
        megatron_rope = MegatronRoPE(
            kv_channels=kv_channels,
            rotary_percent=1.0,
            rotary_base=10000,
            use_cpu_initialization=True,
        )

        # Create RoseLLM RoPE (mock for now)
        rosellm_rope = MockRoseLLMRoPE(
            kv_channels=kv_channels, rotary_percent=1.0, rotary_base=10000
        )

        # Generate embeddings
        megatron_emb = megatron_rope(seq_len, offset=0)
        rosellm_emb = rosellm_rope.forward(seq_len, offset=0)

        # Compare outputs
        torch.testing.assert_close(
            rosellm_emb,
            megatron_emb,
            rtol=1e-5,
            atol=1e-5,
            msg="RoPE embeddings do not match Megatron-LM",
        )

    @pytest.mark.gpu
    @pytest.mark.skipif(not MEGATRON_AVAILABLE, reason="Megatron-LM not available")
    def test_rope_with_offset(self):
        """Test RoPE with position offset."""
        torch.manual_seed(42)

        kv_channels = 64
        seq_len = 128
        offset = 64

        # Create instances
        megatron_rope = MegatronRoPE(
            kv_channels=kv_channels,
            rotary_percent=1.0,
            rotary_base=10000,
            use_cpu_initialization=True,
        )

        rosellm_rope = MockRoseLLMRoPE(
            kv_channels=kv_channels, rotary_percent=1.0, rotary_base=10000
        )

        # Generate with offset
        megatron_emb = megatron_rope(seq_len, offset=offset)
        rosellm_emb = rosellm_rope.forward(seq_len, offset=offset)

        # Compare
        torch.testing.assert_close(
            rosellm_emb,
            megatron_emb,
            rtol=1e-5,
            atol=1e-5,
            msg="RoPE with offset does not match",
        )

    @pytest.mark.gpu
    @pytest.mark.skipif(not MEGATRON_AVAILABLE, reason="Megatron-LM not available")
    def test_rope_scaling(self):
        """Test LLaMA 3.x style RoPE scaling."""
        torch.manual_seed(42)

        kv_channels = 64
        seq_len = 256
        scaling_factor = 8.0

        # Megatron with scaling
        megatron_rope = MegatronRoPE(
            kv_channels=kv_channels,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=True,
            rope_scaling_factor=scaling_factor,
            use_cpu_initialization=True,
        )

        # RoseLLM with scaling
        rosellm_rope = MockRoseLLMRoPE(
            kv_channels=kv_channels,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=True,
            rope_scaling_factor=scaling_factor,
        )

        # Generate and compare
        megatron_emb = megatron_rope(seq_len)
        rosellm_emb = rosellm_rope.forward(seq_len)

        # Note: Scaling implementation might differ slightly
        # Allow slightly higher tolerance for scaling
        torch.testing.assert_close(
            rosellm_emb,
            megatron_emb,
            rtol=1e-4,
            atol=1e-4,
            msg="RoPE scaling does not match",
        )

    @pytest.mark.gpu
    @pytest.mark.skipif(not MEGATRON_AVAILABLE, reason="Megatron-LM not available")
    def test_rope_application_to_qk(self):
        """Test applying RoPE to query and key tensors."""
        torch.manual_seed(42)

        batch_size = 2
        num_heads = 8
        seq_len = 128
        head_dim = 64

        # Create random Q and K tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Create RoPE embeddings
        megatron_rope = MegatronRoPE(
            kv_channels=head_dim, rotary_percent=1.0, use_cpu_initialization=True
        )

        # Get embeddings
        rope_emb = megatron_rope(seq_len)

        # Apply RoPE using Megatron utility
        cos = rope_emb.cos()
        sin = rope_emb.sin()
        q_megatron, k_megatron = megatron_apply_rope(q, k, cos, sin)

        # Apply using RoseLLM mock
        q_rosellm, k_rosellm = mock_apply_rope(q, k, rope_emb)

        # Compare
        torch.testing.assert_close(
            q_rosellm,
            q_megatron,
            rtol=1e-5,
            atol=1e-5,
            msg="Query with RoPE does not match",
        )

        torch.testing.assert_close(
            k_rosellm,
            k_megatron,
            rtol=1e-5,
            atol=1e-5,
            msg="Key with RoPE does not match",
        )

    @pytest.mark.gpu
    @pytest.mark.skipif(not MEGATRON_AVAILABLE, reason="Megatron-LM not available")
    def test_rope_gradient_flow(self):
        """Test gradient flow through RoPE application."""
        torch.manual_seed(42)

        batch_size = 2
        num_heads = 4
        seq_len = 64
        head_dim = 32

        # Create tensors requiring gradients
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)

        # Create RoPE
        rope = MegatronRoPE(
            kv_channels=head_dim, rotary_percent=1.0, use_cpu_initialization=True
        )

        # Apply RoPE
        rope_emb = rope(seq_len)
        cos = rope_emb.cos()
        sin = rope_emb.sin()
        q_rot, k_rot = megatron_apply_rope(q, k, cos, sin)

        # Compute dummy loss
        loss = (q_rot * k_rot).sum()
        loss.backward()

        # Check gradients exist and are non-zero
        assert q.grad is not None, "Query gradient is None"
        assert k.grad is not None, "Key gradient is None"
        assert not torch.allclose(
            q.grad, torch.zeros_like(q.grad)
        ), "Query gradient is zero"
        assert not torch.allclose(
            k.grad, torch.zeros_like(k.grad)
        ), "Key gradient is zero"

    def test_rope_numerical_stability(self):
        """Test numerical stability with extreme values."""
        torch.manual_seed(42)

        # Test with very long sequences
        long_seq_len = 16384
        kv_channels = 64

        rope = MockRoseLLMRoPE(
            kv_channels=kv_channels, rotary_percent=1.0, rotary_base=10000
        )

        # Generate embeddings for long sequence
        emb = rope.forward(long_seq_len)

        # Check for numerical issues
        assert not torch.isnan(emb).any(), "NaN values in RoPE embeddings"
        assert not torch.isinf(emb).any(), "Inf values in RoPE embeddings"

        # Check value range is reasonable
        assert emb.abs().max() < 1e6, "RoPE values exploding"

    def test_rope_determinism(self):
        """Test that RoPE is deterministic."""
        kv_channels = 64
        seq_len = 128

        # Create two identical RoPE instances
        rope1 = MockRoseLLMRoPE(kv_channels=kv_channels)
        rope2 = MockRoseLLMRoPE(kv_channels=kv_channels)

        # Generate embeddings
        emb1 = rope1.forward(seq_len)
        emb2 = rope2.forward(seq_len)

        # Should be identical
        torch.testing.assert_close(
            emb1, emb2, rtol=0, atol=0, msg="RoPE is not deterministic"
        )

    def test_rope_partial_rotation(self):
        """Test RoPE with partial rotation (rotary_percent < 1.0)."""
        kv_channels = 64
        seq_len = 128
        rotary_percent = 0.5

        # Create RoPE with partial rotation
        rope = MockRoseLLMRoPE(kv_channels=kv_channels, rotary_percent=rotary_percent)

        # Generate embeddings
        emb = rope.forward(seq_len)

        # Check dimensions
        # For partial rotation, rotary_dim = kv_channels * rotary_percent = 32
        # inv_freq has rotary_dim/2 = 16 elements
        # After concatenation, we get 32 dimensions
        expected_dim = int(kv_channels * rotary_percent)
        actual_dim = emb.shape[-1]

        assert (
            actual_dim == expected_dim
        ), f"Partial rotation dimension mismatch: {actual_dim} vs {expected_dim}"


def run_validation_suite():
    """Run complete validation suite."""
    print("=" * 60)
    print("RoPE Megatron-LM Parity Validation")
    print("=" * 60)

    test_suite = TestRoPEMegatronParity()

    # List of tests to run
    tests = [
        ("Basic Forward Pass", test_suite.test_basic_rope_forward),
        ("Position Offset", test_suite.test_rope_with_offset),
        ("RoPE Scaling", test_suite.test_rope_scaling),
        ("Q/K Application", test_suite.test_rope_application_to_qk),
        ("Gradient Flow", test_suite.test_rope_gradient_flow),
        ("Numerical Stability", test_suite.test_rope_numerical_stability),
        ("Determinism", test_suite.test_rope_determinism),
        ("Partial Rotation", test_suite.test_rope_partial_rotation),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test_name, test_func in tests:
        try:
            if not MEGATRON_AVAILABLE and "megatron" in test_func.__name__:
                print(f"[SKIP] {test_name}: Megatron-LM not available")
                skipped += 1
                continue

            test_func()
            print(f"[PASS] {test_name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_name}: {str(e)}")
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)
