"""
Comprehensive tests for advanced gradient finalization functionality.

This module tests the advanced gradient finalization features including:
- GradientDataTypeManager for multi-precision conversions
- AdvancedGradientFinalizer with multi-dimensional parallelism
- Integration with RoseTrainer
- Error handling and edge cases
"""

import logging
import unittest
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

# Import the modules under test
try:
    from rosellm.rosetrainer.config import TrainingConfig
    from rosellm.rosetrainer.engine import RoseTrainer
    from rosellm.rosetrainer.gradient import (
        AdvancedGradientFinalizer,
        GradientDataType,
        GradientDataTypeManager,
        GradientFinalizationConfig,
        create_gradient_data_type_manager,
        finalize_gradients_advanced,
    )

    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"Required imports not available: {e}", allow_module_level=True)

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """Simple model for testing."""

    def __init__(
        self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, y=None):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        if y is not None:
            loss = self.loss_fn(out, y)
            return {"loss": loss, "output": out}
        return {"output": out}


class TestGradientDataType(unittest.TestCase):
    """Test GradientDataType enum."""

    def test_gradient_data_type_values(self):
        """Test that gradient data type enum has expected values."""
        self.assertEqual(GradientDataType.FP32.value, "fp32")
        self.assertEqual(GradientDataType.FP16.value, "fp16")
        self.assertEqual(GradientDataType.BF16.value, "bf16")
        self.assertEqual(GradientDataType.FP8.value, "fp8")


class TestGradientDataTypeManager(unittest.TestCase):
    """Test GradientDataTypeManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleTestModel().to(self.device)

        # Create dummy gradients
        x = torch.randn(5, 10, device=self.device)
        y = torch.randn(5, 5, device=self.device)
        outputs = self.model(x, y)
        outputs["loss"].backward()

    def test_initialization_default(self):
        """Test default initialization."""
        manager = GradientDataTypeManager()

        self.assertEqual(manager.master_dtype, GradientDataType.FP32)
        self.assertEqual(manager.compute_dtype, GradientDataType.FP32)
        self.assertEqual(manager.communication_dtype, GradientDataType.FP32)
        self.assertFalse(manager.enable_compression)

    def test_initialization_custom(self):
        """Test custom initialization."""
        manager = GradientDataTypeManager(
            master_dtype=GradientDataType.FP32,
            compute_dtype=GradientDataType.FP16,
            communication_dtype=GradientDataType.FP16,
            enable_compression=True,
        )

        self.assertEqual(manager.master_dtype, GradientDataType.FP32)
        self.assertEqual(manager.compute_dtype, GradientDataType.FP16)
        self.assertEqual(manager.communication_dtype, GradientDataType.FP16)
        self.assertTrue(manager.enable_compression)

    def test_get_torch_dtype(self):
        """Test torch dtype conversion."""
        manager = GradientDataTypeManager()

        self.assertEqual(manager.get_torch_dtype(GradientDataType.FP32), torch.float32)
        self.assertEqual(manager.get_torch_dtype(GradientDataType.FP16), torch.float16)
        self.assertEqual(manager.get_torch_dtype(GradientDataType.BF16), torch.bfloat16)

    def test_get_torch_dtype_invalid(self):
        """Test invalid torch dtype conversion."""
        manager = GradientDataTypeManager()

        with self.assertRaises(ValueError):
            manager.get_torch_dtype("invalid_type")  # type: ignore

    def test_convert_gradients_to_master(self):
        """Test gradient conversion to master precision."""
        manager = GradientDataTypeManager(master_dtype=GradientDataType.FP32)

        # Convert gradients
        converted_grads = manager.convert_gradients_to_master(self.model)

        # Verify conversion
        self.assertIsInstance(converted_grads, dict)
        self.assertGreater(len(converted_grads), 0)

        # Check that all gradients are in master precision
        for name, grad in converted_grads.items():
            self.assertEqual(grad.dtype, torch.float32)

    def test_convert_gradients_for_communication(self):
        """Test gradient conversion for communication."""
        manager = GradientDataTypeManager(
            master_dtype=GradientDataType.FP32,
            communication_dtype=GradientDataType.FP16,
            enable_compression=True,
        )

        # First convert to master
        master_grads = manager.convert_gradients_to_master(self.model)

        # Then convert for communication
        comm_grads, metadata = manager.convert_gradients_for_communication(master_grads)

        # Verify conversion
        self.assertIsInstance(comm_grads, dict)
        self.assertIsInstance(metadata, dict)
        self.assertIn("compression_ratio", metadata)
        self.assertIn("compressed_params", metadata)

    def test_restore_gradients_from_communication(self):
        """Test gradient restoration from communication."""
        manager = GradientDataTypeManager(
            master_dtype=GradientDataType.FP32,
            communication_dtype=GradientDataType.FP16,
        )

        # Convert gradients through the pipeline
        master_grads = manager.convert_gradients_to_master(self.model)
        comm_grads, metadata = manager.convert_gradients_for_communication(master_grads)

        # Restore gradients
        manager.restore_gradients_from_communication(self.model, comm_grads, metadata)

        # Verify restoration - gradients should be back to master precision
        for param in self.model.parameters():
            if param.grad is not None:
                self.assertEqual(param.grad.dtype, torch.float32)

    def test_statistics(self):
        """Test statistics collection."""
        manager = GradientDataTypeManager()

        # Perform some operations
        manager.convert_gradients_to_master(self.model)

        stats = manager.get_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn("conversion_stats", stats)
        self.assertIn("master_dtype", stats)
        self.assertIn("compute_dtype", stats)
        self.assertIn("communication_dtype", stats)

    def test_reset_statistics(self):
        """Test statistics reset."""
        manager = GradientDataTypeManager()

        # Perform operations to generate stats
        manager.convert_gradients_to_master(self.model)

        # Reset statistics
        manager.reset_statistics()

        stats = manager.get_statistics()
        self.assertEqual(stats["conversion_stats"]["total_conversions"], 0)

    def test_cleanup(self):
        """Test cleanup functionality."""
        manager = GradientDataTypeManager()
        manager.convert_gradients_to_master(self.model, store_originals=True)

        # Should have stored gradients
        self.assertGreater(len(manager._master_gradients), 0)

        # Cleanup
        manager.cleanup()

        # Should be empty now
        self.assertEqual(len(manager._master_gradients), 0)


class TestAdvancedGradientFinalizer(unittest.TestCase):
    """Test AdvancedGradientFinalizer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleTestModel().to(self.device)

        # Create dummy gradients
        x = torch.randn(5, 10, device=self.device)
        y = torch.randn(5, 5, device=self.device)
        outputs = self.model(x, y)
        outputs["loss"].backward()

    def test_initialization_default(self):
        """Test default initialization."""
        finalizer = AdvancedGradientFinalizer(self.model)

        self.assertIsInstance(finalizer.config, GradientFinalizationConfig)
        self.assertIsInstance(finalizer.data_type_manager, GradientDataTypeManager)
        self.assertTrue(finalizer.enable_advanced_sync)

        # Clean up
        finalizer.cleanup()

    def test_initialization_custom(self):
        """Test custom initialization."""
        config = GradientFinalizationConfig(verbose=True)
        dtm = create_gradient_data_type_manager("fp16", "fp16", True)

        finalizer = AdvancedGradientFinalizer(
            model=self.model,
            config=config,
            data_type_manager=dtm,
            enable_advanced_sync=False,
            verbose=True,
        )

        self.assertEqual(finalizer.config, config)
        self.assertEqual(finalizer.data_type_manager, dtm)
        self.assertFalse(finalizer.enable_advanced_sync)
        self.assertTrue(finalizer.verbose)

        # Clean up
        finalizer.cleanup()

    @patch("rosellm.rosetrainer.gradient.finalization.parallel_state")
    def test_finalize_gradients_basic(self, mock_parallel_state):
        """Test basic gradient finalization."""
        # Mock parallel state to avoid distributed requirements
        mock_parallel_state.is_initialized.return_value = False

        finalizer = AdvancedGradientFinalizer(self.model, verbose=True)

        stats = finalizer.finalize_gradients(
            clip_gradients=True,
            check_finite=True,
            normalize_gradients=False,
            collect_stats=True,
        )

        # Verify stats structure
        self.assertIsInstance(stats, dict)
        self.assertIn("step", stats)
        self.assertIn("finalization_time", stats)
        self.assertIn("finite", stats)
        self.assertIn("success", stats)

        # Should be successful
        self.assertTrue(stats.get("success", False))

        # Clean up
        finalizer.cleanup()

    @patch("rosellm.rosetrainer.gradient.finalization.parallel_state")
    def test_finalize_gradients_with_normalization(self, mock_parallel_state):
        """Test gradient finalization with normalization."""
        # Mock parallel state to avoid distributed requirements
        mock_parallel_state.is_initialized.return_value = False

        finalizer = AdvancedGradientFinalizer(self.model)

        stats = finalizer.finalize_gradients(
            clip_gradients=False,
            check_finite=False,
            normalize_gradients=True,
            collect_stats=False,
        )

        self.assertIn("normalized", stats)
        self.assertTrue(stats.get("success", False))

        # Clean up
        finalizer.cleanup()

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        finalizer = AdvancedGradientFinalizer(self.model)

        # Run some operations
        with patch(
            "rosellm.rosetrainer.gradient.finalization.parallel_state"
        ) as mock_ps:
            mock_ps.is_initialized.return_value = False
            finalizer.finalize_gradients()

        metrics = finalizer.get_performance_metrics()

        self.assertIsInstance(metrics, dict)
        self.assertIn("finalization_count", metrics)
        self.assertIn("avg_finalization_time", metrics)
        self.assertIn("data_type_stats", metrics)

        # Clean up
        finalizer.cleanup()

    def test_parallelism_info(self):
        """Test parallelism information."""
        finalizer = AdvancedGradientFinalizer(self.model)

        info = finalizer.get_parallelism_info()

        self.assertIsInstance(info, dict)
        self.assertIn("initialized", info)
        self.assertIn("groups", info)
        self.assertIn("ranks", info)
        self.assertIn("sizes", info)
        self.assertIn("config", info)

        # Clean up
        finalizer.cleanup()

    def test_reset_statistics(self):
        """Test statistics reset."""
        finalizer = AdvancedGradientFinalizer(self.model)

        # Run operations to generate stats
        with patch(
            "rosellm.rosetrainer.gradient.finalization.parallel_state"
        ) as mock_ps:
            mock_ps.is_initialized.return_value = False
            finalizer.finalize_gradients()

        self.assertEqual(finalizer._finalization_count, 1)

        # Reset
        finalizer.reset_statistics()

        self.assertEqual(finalizer._finalization_count, 0)

        # Clean up
        finalizer.cleanup()


class TestConvenienceAPIs(unittest.TestCase):
    """Test convenience APIs."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleTestModel().to(self.device)

        # Create dummy gradients
        x = torch.randn(5, 10, device=self.device)
        y = torch.randn(5, 5, device=self.device)
        outputs = self.model(x, y)
        outputs["loss"].backward()

    def test_create_gradient_data_type_manager(self):
        """Test factory function for gradient data type manager."""
        manager = create_gradient_data_type_manager(
            master_precision="fp32",
            communication_precision="fp16",
            enable_compression=True,
        )

        self.assertIsInstance(manager, GradientDataTypeManager)
        self.assertEqual(manager.master_dtype, GradientDataType.FP32)
        self.assertEqual(manager.communication_dtype, GradientDataType.FP16)
        self.assertTrue(manager.enable_compression)

    @patch("rosellm.rosetrainer.gradient.finalization.parallel_state")
    def test_finalize_gradients_advanced_function(self, mock_parallel_state):
        """Test standalone finalize_gradients_advanced function."""
        # Mock parallel state to avoid distributed requirements
        mock_parallel_state.is_initialized.return_value = False

        stats = finalize_gradients_advanced(
            model=self.model,
            clip_gradients=True,
            check_finite=True,
            normalize_gradients=False,
            collect_stats=True,
            verbose=True,
        )

        self.assertIsInstance(stats, dict)
        self.assertIn("success", stats)
        self.assertTrue(stats.get("success", False))


class TestIntegration(unittest.TestCase):
    """Test integration with RoseTrainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleTestModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    @patch("rosellm.rosetrainer.engine.dist")
    def test_rosetrainer_with_advanced_finalization(self, mock_dist):
        """Test RoseTrainer with advanced gradient finalization enabled."""
        # Mock distributed environment
        mock_dist.is_initialized.return_value = False

        # Create config with advanced finalization
        config = TrainingConfig(
            batch_size=4,
            num_epochs=None,
            max_steps=10,
            warmup_steps=0,
            seed=42,
            checkpoint_interval=50,
            log_interval=10,
            eval_interval=100,
        )

        # Update gradient configuration
        config.gradient.enable_advanced_finalization = True
        config.gradient.master_precision = "fp32"
        config.gradient.communication_precision = "fp16"
        config.gradient.enable_gradient_compression = True
        config.gradient.finalization_verbose = True

        # Create trainer
        trainer = RoseTrainer(
            model=self.model,
            optimizer=self.optimizer,
            config=config,
        )

        # Verify advanced finalization is initialized
        self.assertIsNotNone(trainer.advanced_gradient_finalizer)
        self.assertIsNotNone(trainer.gradient_data_type_manager)

        # Test training step
        x = torch.randn(4, 10, device=self.device)
        y = torch.randn(4, 5, device=self.device)
        batch = {"input_ids": x, "labels": y}  # Mimic expected format

        # This should use advanced gradient finalization
        with patch.object(trainer.model, "forward") as mock_forward:
            # Mock the model forward to return expected format
            mock_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
            mock_forward.return_value = {"loss": mock_loss}

            metrics = trainer.train_step(batch)

        self.assertIsInstance(metrics, dict)
        self.assertIn("loss", metrics)

        # Clean up
        trainer.cleanup()


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleTestModel().to(self.device)

    def test_gradient_data_type_manager_invalid_dtype(self):
        """Test invalid data type handling."""
        with self.assertRaises(ValueError):
            _manager = GradientDataTypeManager(  # noqa: F841
                master_dtype="invalid_dtype"  # type: ignore
            )

    def test_advanced_finalizer_with_invalid_model(self):
        """Test advanced finalizer with model without gradients."""
        # Model with no gradients
        clean_model = SimpleTestModel().to(self.device)

        finalizer = AdvancedGradientFinalizer(clean_model)

        # Should handle gracefully
        with patch(
            "rosellm.rosetrainer.gradient.finalization.parallel_state"
        ) as mock_ps:
            mock_ps.is_initialized.return_value = False
            stats = finalizer.finalize_gradients()

        self.assertIsInstance(stats, dict)
        # Should still succeed even with no gradients
        self.assertTrue(stats.get("success", False))

        # Clean up
        finalizer.cleanup()

    def test_data_type_manager_empty_gradients(self):
        """Test data type manager with empty gradients."""
        clean_model = SimpleTestModel().to(self.device)
        manager = GradientDataTypeManager()

        # Should handle empty gradients gracefully
        converted_grads = manager.convert_gradients_to_master(clean_model)
        self.assertIsInstance(converted_grads, dict)
        self.assertEqual(len(converted_grads), 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU tests require CUDA")
class TestGPUSpecific(unittest.TestCase):
    """Test GPU-specific functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda")
        self.model = SimpleTestModel().to(self.device)

        # Create dummy gradients
        x = torch.randn(5, 10, device=self.device)
        y = torch.randn(5, 5, device=self.device)
        outputs = self.model(x, y)
        outputs["loss"].backward()

    def test_gpu_precision_conversion(self):
        """Test precision conversion on GPU."""
        manager = GradientDataTypeManager(
            master_dtype=GradientDataType.FP32,
            communication_dtype=GradientDataType.FP16,
        )

        # Convert gradients
        master_grads = manager.convert_gradients_to_master(self.model)
        comm_grads, metadata = manager.convert_gradients_for_communication(master_grads)

        # Verify GPU tensors
        for grad in comm_grads.values():
            self.assertTrue(grad.is_cuda)
            self.assertEqual(grad.dtype, torch.float16)

    def test_advanced_finalizer_gpu(self):
        """Test advanced finalizer on GPU."""
        finalizer = AdvancedGradientFinalizer(self.model, verbose=True)

        with patch(
            "rosellm.rosetrainer.gradient.finalization.parallel_state"
        ) as mock_ps:
            mock_ps.is_initialized.return_value = False
            stats = finalizer.finalize_gradients()

        self.assertTrue(stats.get("success", False))

        # Clean up
        finalizer.cleanup()


class TestBF16Support(unittest.TestCase):
    """Test BF16 support if available."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleTestModel().to(self.device)

        # Create dummy gradients
        x = torch.randn(5, 10, device=self.device)
        y = torch.randn(5, 5, device=self.device)
        outputs = self.model(x, y)
        outputs["loss"].backward()

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        reason="BF16 tests require CUDA and BF16 support",
    )
    def test_bf16_conversion(self):
        """Test BF16 gradient conversion."""
        manager = GradientDataTypeManager(
            master_dtype=GradientDataType.BF16,
            communication_dtype=GradientDataType.BF16,
        )

        # Convert gradients
        master_grads = manager.convert_gradients_to_master(self.model)

        # Verify BF16 conversion
        for grad in master_grads.values():
            self.assertEqual(grad.dtype, torch.bfloat16)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
