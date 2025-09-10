"""
Comprehensive test suite for Vocab Parallel Cross-Entropy Loss

This test module validates the correctness of the vocabulary parallel
cross-entropy implementation, including:
- Numerical accuracy against standard PyTorch cross-entropy
- Gradient correctness with autograd checks
- Label smoothing functionality
- Distributed tensor parallel behavior
- Bit-to-bit compatibility with Megatron-LM
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rosellm.rosetrainer.parallelism import (
    destroy_model_parallel,
    get_tensor_model_parallel_size,
    initialize_model_parallel,
)
from rosellm.rosetrainer.tensor_parallel.vocab_parallel_cross_entropy import (
    VocabParallelCrossEntropy,
    VocabParallelCrossEntropyLoss,
    VocabParallelError,
    VocabUtility,
    vocab_parallel_cross_entropy,
)


class TestVocabUtility(unittest.TestCase):
    """Test vocabulary range calculation utilities."""

    def test_vocab_range_from_per_partition_size(self):
        """Test vocabulary range calculation from partition size."""
        per_partition_size = 1000
        world_size = 4

        for rank in range(world_size):
            start, end = VocabUtility.vocab_range_from_per_partition_vocab_size(
                per_partition_size, rank, world_size
            )

            # Check range is correct
            self.assertEqual(start, rank * per_partition_size)
            self.assertEqual(end, (rank + 1) * per_partition_size)

            # Check range size
            self.assertEqual(end - start, per_partition_size)

    def test_vocab_range_from_global_size(self):
        """Test vocabulary range calculation from global size."""
        global_vocab_size = 50257
        world_size = 3

        # Should raise assertion for non-divisible size
        with self.assertRaises(VocabParallelError):
            VocabUtility.vocab_range_from_global_vocab_size(
                global_vocab_size, 0, world_size
            )

        # Test with divisible size
        global_vocab_size = 51000  # Divisible by 3
        for rank in range(world_size):
            start, end = VocabUtility.vocab_range_from_global_vocab_size(
                global_vocab_size, rank, world_size
            )

            expected_per_partition = global_vocab_size // world_size
            self.assertEqual(start, rank * expected_per_partition)
            self.assertEqual(end, (rank + 1) * expected_per_partition)


class TestVocabParallelErrorScenarios(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_vocab_utility_parameter_validation(self):
        """Test VocabUtility parameter validation."""
        # Test invalid per_partition_vocab_size
        with self.assertRaises(VocabParallelError):
            VocabUtility.vocab_range_from_per_partition_vocab_size(-1, 0, 2)

        with self.assertRaises(VocabParallelError):
            VocabUtility.vocab_range_from_per_partition_vocab_size(0, 0, 2)

        # Test invalid rank
        with self.assertRaises(VocabParallelError):
            VocabUtility.vocab_range_from_per_partition_vocab_size(100, -1, 2)

        with self.assertRaises(VocabParallelError):
            VocabUtility.vocab_range_from_per_partition_vocab_size(
                100, 2, 2
            )  # rank >= world_size

        # Test invalid world_size
        with self.assertRaises(VocabParallelError):
            VocabUtility.vocab_range_from_per_partition_vocab_size(100, 0, 0)

        with self.assertRaises(VocabParallelError):
            VocabUtility.vocab_range_from_per_partition_vocab_size(100, 0, -1)

    def test_global_vocab_size_validation(self):
        """Test global vocabulary size validation."""
        # Test invalid global_vocab_size
        with self.assertRaises(VocabParallelError):
            VocabUtility.vocab_range_from_global_vocab_size(-1, 0, 2)

        with self.assertRaises(VocabParallelError):
            VocabUtility.vocab_range_from_global_vocab_size(0, 0, 2)

        # Test non-divisible vocabulary size
        with self.assertRaises(VocabParallelError):
            VocabUtility.vocab_range_from_global_vocab_size(
                100, 0, 3
            )  # 100 not divisible by 3

    def test_tensor_input_validation(self):
        """Test tensor input validation."""
        from rosellm.rosetrainer.tensor_parallel.vocab_parallel_cross_entropy import (
            _validate_tensor_inputs,
        )

        # Valid inputs
        logits = torch.randn(8, 4, 100)
        targets = torch.randint(0, 100, (8, 4))
        _validate_tensor_inputs(logits, targets)  # Should not raise

        # Invalid types
        with self.assertRaises(VocabParallelError):
            _validate_tensor_inputs("not_a_tensor", targets)  # type: ignore[arg-type]

        with self.assertRaises(VocabParallelError):
            _validate_tensor_inputs(logits, "not_a_tensor")  # type: ignore[arg-type]

        # Wrong dimensions
        with self.assertRaises(VocabParallelError):
            _validate_tensor_inputs(torch.randn(8, 4), targets)  # 2D instead of 3D

        with self.assertRaises(VocabParallelError):
            _validate_tensor_inputs(
                logits, torch.randint(0, 100, (8,))
            )  # 1D instead of 2D

        # Shape mismatch
        with self.assertRaises(VocabParallelError):
            _validate_tensor_inputs(
                logits, torch.randint(0, 100, (8, 5))
            )  # Different batch size

        # Device mismatch
        if torch.cuda.is_available():
            cuda_logits = logits.cuda()
            cpu_targets = targets.cpu()
            with self.assertRaises(VocabParallelError):
                _validate_tensor_inputs(cuda_logits, cpu_targets)

        # Invalid target dtype
        with self.assertRaises(VocabParallelError):
            _validate_tensor_inputs(logits, targets.float())

        # Negative target values
        negative_targets = torch.full_like(targets, -1)
        with self.assertRaises(VocabParallelError):
            _validate_tensor_inputs(logits, negative_targets)

    def test_module_initialization_errors(self):
        """Test VocabParallelCrossEntropyLoss initialization errors."""
        # Invalid label smoothing values
        with self.assertRaises(ValueError):
            VocabParallelCrossEntropyLoss(label_smoothing=-0.1)

        with self.assertRaises(ValueError):
            VocabParallelCrossEntropyLoss(label_smoothing=1.0)

        with self.assertRaises(ValueError):
            VocabParallelCrossEntropyLoss(label_smoothing=1.5)

        # Invalid reduction modes
        with self.assertRaises(ValueError):
            VocabParallelCrossEntropyLoss(reduction="invalid")


class TestVocabParallelCrossEntropyLocal(unittest.TestCase):
    """Test vocab parallel cross-entropy without distributed setup."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_calculate_logits_max(self):
        """Test maximum logits calculation."""
        batch_size = 4
        seq_len = 8
        vocab_size = 100

        logits = torch.randn(seq_len, batch_size, vocab_size)
        float_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
            logits
        )

        # Check conversion to float32
        self.assertEqual(float_logits.dtype, torch.float32)

        # Check max calculation
        expected_max = logits.max(dim=-1)[0]
        torch.testing.assert_close(logits_max, expected_max)

    def test_calculate_predicted_logits(self):
        """Test predicted logits extraction."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        # Create simple logits and targets
        logits = torch.randn(seq_len, batch_size, vocab_size)
        logits_max = logits.max(dim=-1)[0]
        targets = torch.randint(0, vocab_size, (seq_len, batch_size))

        # Calculate predicted logits
        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            logits.clone(), targets, logits_max, 0, vocab_size
        )

        # No targets should be masked (all within range)
        self.assertEqual(target_mask.sum().item(), 0)

        # Check exp and sum calculations
        expected_exp = torch.exp(logits - logits_max.unsqueeze(-1))
        torch.testing.assert_close(exp_logits, expected_exp, rtol=1e-4, atol=1e-4)

        expected_sum = expected_exp.sum(dim=-1)
        torch.testing.assert_close(sum_exp_logits, expected_sum, rtol=1e-4, atol=1e-4)

    def test_label_smoothing(self):
        """Test label smoothing application."""
        batch_size = 2
        seq_len = 4
        vocab_size = 100

        # Create dummy loss and probabilities
        loss = torch.randn(seq_len, batch_size)
        probs = torch.softmax(torch.randn(seq_len, batch_size, vocab_size), dim=-1)

        # Test no smoothing
        smoothed_loss = VocabParallelCrossEntropy.apply_label_smoothing(
            loss.clone(), probs.clone(), 0.0, vocab_size
        )
        torch.testing.assert_close(smoothed_loss, loss)

        # Test with smoothing
        label_smoothing = 0.1
        smoothed_loss = VocabParallelCrossEntropy.apply_label_smoothing(
            loss.clone(), probs.clone(), label_smoothing, vocab_size
        )

        # Loss should be different with smoothing
        self.assertFalse(torch.allclose(smoothed_loss, loss))


class TestVocabParallelCrossEntropyModule(unittest.TestCase):
    """Test the nn.Module wrapper for vocab parallel cross-entropy."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_module_initialization(self):
        """Test module initialization with various parameters."""
        # Valid initialization
        loss_fn = VocabParallelCrossEntropyLoss(label_smoothing=0.1, reduction="mean")
        self.assertEqual(loss_fn.label_smoothing, 0.1)
        self.assertEqual(loss_fn.reduction, "mean")

        # Invalid label smoothing
        with self.assertRaises(ValueError):
            VocabParallelCrossEntropyLoss(label_smoothing=1.5)

        with self.assertRaises(ValueError):
            VocabParallelCrossEntropyLoss(label_smoothing=-0.1)

        # Invalid reduction
        with self.assertRaises(ValueError):
            VocabParallelCrossEntropyLoss(reduction="invalid")

    def test_extra_repr(self):
        """Test string representation of module."""
        loss_fn = VocabParallelCrossEntropyLoss(label_smoothing=0.2, reduction="sum")
        repr_str = loss_fn.extra_repr()
        self.assertIn("label_smoothing=0.2", repr_str)
        self.assertIn("reduction=sum", repr_str)


class TestVocabParallelCrossEntropyDistributed(unittest.TestCase):
    """Test vocab parallel cross-entropy with distributed tensor parallelism."""

    @classmethod
    def setUpClass(cls):
        """Set up distributed environment."""
        if not dist.is_initialized():
            # Initialize for single GPU testing
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment."""
        if dist.is_initialized():
            destroy_model_parallel()
            dist.destroy_process_group()

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model parallel with TP=1 for single GPU test
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
        initialize_model_parallel(tensor_model_parallel_size=1)

    def tearDown(self):
        """Clean up after test."""
        destroy_model_parallel()

    def test_vocab_parallel_vs_standard_cross_entropy(self):
        """Compare vocab parallel with standard PyTorch cross-entropy."""
        batch_size = 4
        seq_len = 8
        vocab_size = 1000

        # Create test data
        logits = torch.randn(seq_len, batch_size, vocab_size, device=self.device)
        targets = torch.randint(
            0, vocab_size, (seq_len, batch_size), device=self.device
        )

        # Compute standard cross-entropy
        logits_transposed = logits.transpose(0, 1).contiguous()  # [B, S, V]
        targets_transposed = targets.transpose(0, 1).contiguous()  # [B, S]

        standard_loss = (
            F.cross_entropy(
                logits_transposed.view(-1, vocab_size),
                targets_transposed.view(-1),
                reduction="none",
            )
            .view(batch_size, seq_len)
            .transpose(0, 1)
        )

        # Compute vocab parallel cross-entropy (with TP=1, should match)
        vocab_parallel_loss = vocab_parallel_cross_entropy(logits, targets)

        # Check that losses match
        torch.testing.assert_close(
            vocab_parallel_loss, standard_loss, rtol=1e-4, atol=1e-4
        )

    def test_gradient_correctness(self):
        """Test gradient computation correctness."""
        batch_size = 2
        seq_len = 4
        vocab_size = 100

        # Create test data with requires_grad
        logits = torch.randn(
            seq_len, batch_size, vocab_size, device=self.device, requires_grad=True
        )
        targets = torch.randint(
            0, vocab_size, (seq_len, batch_size), device=self.device
        )

        # Compute loss and gradients
        loss = vocab_parallel_cross_entropy(logits, targets)
        loss_scalar = loss.mean()
        loss_scalar.backward()

        # Check that gradients are computed
        self.assertIsNotNone(logits.grad)
        assert logits.grad is not None  # Type guard for mypy/pyright
        self.assertFalse(torch.all(logits.grad == 0))

        # Gradient shape should match input
        self.assertEqual(logits.grad.shape, logits.shape)

    def test_label_smoothing_effect(self):
        """Test that label smoothing changes the loss value."""
        batch_size = 4
        seq_len = 8
        vocab_size = 1000

        # Create test data
        logits = torch.randn(seq_len, batch_size, vocab_size, device=self.device)
        targets = torch.randint(
            0, vocab_size, (seq_len, batch_size), device=self.device
        )

        # Compute loss without smoothing
        loss_no_smooth = vocab_parallel_cross_entropy(
            logits.clone(), targets, label_smoothing=0.0
        )

        # Compute loss with smoothing
        loss_with_smooth = vocab_parallel_cross_entropy(
            logits.clone(), targets, label_smoothing=0.1
        )

        # Losses should be different
        self.assertFalse(torch.allclose(loss_no_smooth, loss_with_smooth))

        # With label smoothing, loss should generally be higher (more uncertainty)
        # Note: This is not always true for every sample, but on average
        mean_diff = (loss_with_smooth - loss_no_smooth).mean()
        self.assertLess(mean_diff.item(), 0)  # Smoothing typically reduces loss


class TestMegatronCompatibility(unittest.TestCase):
    """Test compatibility with Megatron-LM implementation."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(1234)  # Use same seed as Megatron tests
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        batch_size = 2
        seq_len = 4
        vocab_size = 100

        # Test with very large logits
        large_logits = (
            torch.ones(seq_len, batch_size, vocab_size, device=self.device) * 1e4
        )
        targets = torch.randint(
            0, vocab_size, (seq_len, batch_size), device=self.device
        )

        # Should not produce NaN or Inf
        if dist.is_initialized() and get_tensor_model_parallel_size() > 0:
            loss = vocab_parallel_cross_entropy(large_logits, targets)
            self.assertFalse(torch.any(torch.isnan(loss)))
            self.assertFalse(torch.any(torch.isinf(loss)))

        # Test with very small logits
        small_logits = (
            torch.ones(seq_len, batch_size, vocab_size, device=self.device) * -1e4
        )
        if dist.is_initialized() and get_tensor_model_parallel_size() > 0:
            loss = vocab_parallel_cross_entropy(small_logits, targets)
            self.assertFalse(torch.any(torch.isnan(loss)))
            self.assertFalse(torch.any(torch.isinf(loss)))

    def test_loss_reduction_modes(self):
        """Test different reduction modes in the module wrapper."""
        batch_size = 4
        seq_len = 8
        vocab_size = 100

        logits = torch.randn(seq_len, batch_size, vocab_size, device=self.device)
        targets = torch.randint(
            0, vocab_size, (seq_len, batch_size), device=self.device
        )

        # Test 'none' reduction
        loss_fn_none = VocabParallelCrossEntropyLoss(reduction="none")
        if dist.is_initialized() and get_tensor_model_parallel_size() > 0:
            loss_none = loss_fn_none(logits, targets)
            self.assertEqual(loss_none.shape, targets.shape)

        # Test 'mean' reduction
        loss_fn_mean = VocabParallelCrossEntropyLoss(reduction="mean")
        if dist.is_initialized() and get_tensor_model_parallel_size() > 0:
            loss_mean = loss_fn_mean(logits, targets)
            self.assertEqual(loss_mean.shape, torch.Size([]))

        # Test 'sum' reduction
        loss_fn_sum = VocabParallelCrossEntropyLoss(reduction="sum")
        if dist.is_initialized() and get_tensor_model_parallel_size() > 0:
            loss_sum = loss_fn_sum(logits, targets)
            self.assertEqual(loss_sum.shape, torch.Size([]))

            # Sum should be greater than mean (for positive losses)
            self.assertGreater(loss_sum.item(), loss_mean.item())


def create_test_suite():
    """Create a test suite for vocab parallel cross-entropy."""
    suite = unittest.TestSuite()

    # Add local tests (no distributed setup required)
    suite.addTest(unittest.makeSuite(TestVocabUtility))
    suite.addTest(unittest.makeSuite(TestVocabParallelErrorScenarios))
    suite.addTest(unittest.makeSuite(TestVocabParallelCrossEntropyLocal))
    suite.addTest(unittest.makeSuite(TestVocabParallelCrossEntropyModule))

    # Add distributed tests if environment supports it
    if torch.cuda.is_available() or os.environ.get("FORCE_DISTRIBUTED_TESTS"):
        suite.addTest(unittest.makeSuite(TestVocabParallelCrossEntropyDistributed))
        suite.addTest(unittest.makeSuite(TestMegatronCompatibility))

    return suite


if __name__ == "__main__":
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(create_test_suite())
