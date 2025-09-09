"""Unit tests for gradient finalization and synchronization module."""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.nn as nn

from rosellm.rosetrainer.gradient import GradientFinalizationConfig, GradientFinalizer
from rosellm.rosetrainer.gradient.strategies import (
    BucketedGradientSync,
    HierarchicalGradientSync,
    SimpleGradientSync,
)


class SimpleTestModel(nn.Module):
    """Simple test model."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        if x.dim() == 2 and x.size(0) > 1:
            x = self.bn(x)
        x = self.ln(x)
        x = self.fc2(x)
        return x


class TestGradientFinalizationConfig(unittest.TestCase):
    """Test gradient finalization configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = GradientFinalizationConfig()
        self.assertEqual(config.sync_strategy, "bucketed")
        self.assertEqual(config.reduction_op, "mean")
        self.assertEqual(config.dimension_order, "hierarchical")
        self.assertEqual(config.bucket_size_mb, 25.0)
        self.assertTrue(config.use_contiguous_buffers)

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid sync strategy
        with self.assertRaises(ValueError):
            GradientFinalizationConfig(sync_strategy="invalid")

        # Invalid reduction op
        with self.assertRaises(ValueError):
            GradientFinalizationConfig(reduction_op="invalid")

        # Invalid dimension order
        with self.assertRaises(ValueError):
            GradientFinalizationConfig(dimension_order="invalid")

        # Invalid bucket size
        with self.assertRaises(ValueError):
            GradientFinalizationConfig(bucket_size_mb=-1)

        # Invalid bucket cap
        with self.assertRaises(ValueError):
            GradientFinalizationConfig(bucket_size_mb=100, bucket_cap_mb=50)

    def test_hierarchical_levels_validation(self):
        """Test hierarchical levels validation."""
        # Valid hierarchical levels
        config = GradientFinalizationConfig(
            dimension_order="hierarchical",
            hierarchical_levels=[["tp"], ["pp"], ["dp", "cp", "ep"]],
        )
        self.assertEqual(len(config.hierarchical_levels), 3)

        # Invalid dimension in hierarchical levels
        with self.assertRaises(ValueError):
            GradientFinalizationConfig(
                dimension_order="hierarchical",
                hierarchical_levels=[["invalid_dim"]],
            )

        # Duplicate dimension
        with self.assertRaises(ValueError):
            GradientFinalizationConfig(
                dimension_order="hierarchical",
                hierarchical_levels=[["tp"], ["tp", "pp"]],
            )

    def test_custom_dimension_order(self):
        """Test custom dimension order."""
        # Valid custom order
        config = GradientFinalizationConfig(
            dimension_order="custom",
            custom_dimension_order=["dp", "tp", "pp"],
        )
        self.assertEqual(config.custom_dimension_order, ["dp", "tp", "pp"])

        # Missing custom order
        with self.assertRaises(ValueError):
            GradientFinalizationConfig(dimension_order="custom")

        # Invalid dimension in custom order
        with self.assertRaises(ValueError):
            GradientFinalizationConfig(
                dimension_order="custom",
                custom_dimension_order=["invalid"],
            )

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = GradientFinalizationConfig(
            sync_strategy="hierarchical",
            bucket_size_mb=50.0,
            fp16_compression=True,
        )

        # To dict
        config_dict = config.to_dict()
        self.assertEqual(config_dict["sync_strategy"], "hierarchical")
        self.assertEqual(config_dict["bucket_size_mb"], 50.0)
        self.assertTrue(config_dict["fp16_compression"])

        # From dict
        config2 = GradientFinalizationConfig.from_dict(config_dict)
        self.assertEqual(config2.sync_strategy, config.sync_strategy)
        self.assertEqual(config2.bucket_size_mb, config.bucket_size_mb)
        self.assertEqual(config2.fp16_compression, config.fp16_compression)


class TestGradientSyncStrategies(unittest.TestCase):
    """Test gradient synchronization strategies."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleTestModel()
        self.config = GradientFinalizationConfig()

        # Create dummy gradients
        for param in self.model.parameters():
            param.grad = torch.randn_like(param)

        # Mock process groups
        self.process_groups = {
            "tp": MagicMock(spec=dist.ProcessGroup),
            "pp": MagicMock(spec=dist.ProcessGroup),
            "dp": MagicMock(spec=dist.ProcessGroup),
            "cp": None,
            "ep": None,
        }

    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.all_reduce")
    def test_simple_sync_strategy(self, mock_all_reduce, mock_world_size):
        """Test simple gradient sync strategy."""
        mock_world_size.return_value = 4
        mock_all_reduce.return_value = None

        strategy = SimpleGradientSync(self.config)
        stats = strategy.sync_gradients(self.model, self.process_groups)

        self.assertIn("sync_time", stats)
        self.assertIn("num_params_synced", stats)
        self.assertIn("total_gradient_norm", stats)
        self.assertGreater(stats["num_params_synced"], 0)

        # Verify all_reduce was called
        self.assertTrue(mock_all_reduce.called)

    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.all_reduce")
    def test_bucketed_sync_strategy(self, mock_all_reduce, mock_world_size):
        """Test bucketed gradient sync strategy."""
        mock_world_size.return_value = 4
        mock_all_reduce.return_value = None

        config = GradientFinalizationConfig(
            sync_strategy="bucketed",
            bucket_size_mb=0.001,  # Small bucket for testing
        )
        strategy = BucketedGradientSync(config)
        stats = strategy.sync_gradients(self.model, self.process_groups)

        self.assertIn("num_buckets", stats)
        self.assertGreater(stats["num_buckets"], 0)
        self.assertTrue(mock_all_reduce.called)

    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.all_reduce")
    def test_hierarchical_sync_strategy(self, mock_all_reduce, mock_world_size):
        """Test hierarchical gradient sync strategy."""
        mock_world_size.return_value = 4
        mock_all_reduce.return_value = None

        config = GradientFinalizationConfig(
            sync_strategy="hierarchical",
            hierarchical_levels=[["tp"], ["dp"]],
        )
        strategy = HierarchicalGradientSync(config)
        stats = strategy.sync_gradients(self.model, self.process_groups)

        self.assertIn("num_levels", stats)
        self.assertIn("level_times", stats)
        self.assertEqual(stats["num_levels"], 2)
        self.assertEqual(len(stats["level_times"]), 2)

    def test_gradient_scaling(self):
        """Test gradient scaling."""
        strategy = SimpleGradientSync(self.config)

        # Store original gradients
        original_grads = []
        params = list(self.model.parameters())
        for param in params:
            if param.grad is not None:
                original_grads.append(param.grad.clone())

        # Scale gradients
        scale_factor = 0.5
        strategy._scale_gradients(params, scale_factor)

        # Check scaling
        for param, orig_grad in zip(params, original_grads):
            if param.grad is not None:
                torch.testing.assert_close(param.grad, orig_grad * scale_factor)


class TestGradientFinalizer(unittest.TestCase):
    """Test main gradient finalizer."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleTestModel()
        self.config = GradientFinalizationConfig(verbose=False)

        # Create dummy gradients
        for param in self.model.parameters():
            param.grad = torch.randn_like(param)

    @patch("rosellm.rosetrainer.parallelism.parallel_state.is_initialized")
    @patch("torch.distributed.is_initialized")
    def test_finalizer_initialization(self, mock_dist_init, mock_parallel_init):
        """Test gradient finalizer initialization."""
        mock_dist_init.return_value = False
        mock_parallel_init.return_value = False

        finalizer = GradientFinalizer(self.model, self.config)

        self.assertIsNotNone(finalizer.sync_strategy)
        self.assertEqual(finalizer.finalization_count, 0)
        self.assertIsInstance(finalizer.process_groups, dict)
        self.assertIsInstance(finalizer.world_sizes, dict)
        self.assertIsInstance(finalizer.ranks, dict)

    @patch("rosellm.rosetrainer.parallelism.parallel_state.is_initialized")
    @patch("torch.distributed.is_initialized")
    def test_finalize_gradients(self, mock_dist_init, mock_parallel_init):
        """Test gradient finalization."""
        mock_dist_init.return_value = False
        mock_parallel_init.return_value = False

        finalizer = GradientFinalizer(self.model, self.config)

        # Mock sync strategy
        finalizer.sync_strategy.sync_gradients = MagicMock(
            return_value={"total_gradient_norm": 1.0}
        )

        stats = finalizer.finalize_gradients()

        self.assertIn("finalization_time", stats)
        self.assertIn("sync_stats", stats)
        self.assertIn("gradient_norm", stats)
        self.assertIn("finite", stats)
        self.assertIn("step", stats)
        self.assertEqual(stats["step"], 0)
        self.assertTrue(stats["finite"])

        # Check finalization count incremented
        self.assertEqual(finalizer.finalization_count, 1)

    @patch("rosellm.rosetrainer.parallelism.parallel_state.is_initialized")
    @patch("torch.distributed.is_initialized")
    def test_non_finite_gradient_handling(self, mock_dist_init, mock_parallel_init):
        """Test handling of non-finite gradients."""
        mock_dist_init.return_value = False
        mock_parallel_init.return_value = False

        # Add NaN gradient
        if self.model.fc1.weight.grad is not None:
            self.model.fc1.weight.grad[0, 0] = float("nan")

        finalizer = GradientFinalizer(self.model, self.config)
        stats = finalizer.finalize_gradients(check_finite=True)

        self.assertFalse(stats["finite"])
        self.assertIn("finite_stats", stats)
        self.assertGreater(stats["finite_stats"]["nan_parameters"], 0)

    @patch("rosellm.rosetrainer.parallelism.parallel_state.is_initialized")
    @patch("torch.distributed.is_initialized")
    def test_gradient_statistics(self, mock_dist_init, mock_parallel_init):
        """Test gradient statistics collection."""
        mock_dist_init.return_value = False
        mock_parallel_init.return_value = False

        config = GradientFinalizationConfig(enable_gradient_stats=True)
        finalizer = GradientFinalizer(self.model, config)

        # Mock sync strategy
        finalizer.sync_strategy.sync_gradients = MagicMock(
            return_value={"total_gradient_norm": 1.0}
        )

        stats = finalizer.finalize_gradients(collect_stats=True)

        self.assertIn("gradient_stats", stats)
        grad_stats = stats["gradient_stats"]
        self.assertIn("grad_mean", grad_stats)
        self.assertIn("grad_std", grad_stats)
        self.assertIn("grad_min", grad_stats)
        self.assertIn("grad_max", grad_stats)

    @patch("rosellm.rosetrainer.parallelism.parallel_state.is_initialized")
    @patch("torch.distributed.is_initialized")
    def test_statistics_summary(self, mock_dist_init, mock_parallel_init):
        """Test statistics summary."""
        mock_dist_init.return_value = False
        mock_parallel_init.return_value = False

        config = GradientFinalizationConfig(enable_gradient_stats=True)
        finalizer = GradientFinalizer(self.model, config)

        # Mock sync strategy
        finalizer.sync_strategy.sync_gradients = MagicMock(
            return_value={"total_gradient_norm": 1.0}
        )

        # Run multiple finalizations
        for i in range(5):
            finalizer.finalize_gradients()

        summary = finalizer.get_statistics_summary()

        self.assertEqual(summary["total_finalizations"], 5)
        self.assertIn("avg_finalization_time", summary)
        self.assertIn("avg_gradient_norm", summary)
        self.assertIn("max_gradient_norm", summary)
        self.assertIn("min_gradient_norm", summary)

    @patch("rosellm.rosetrainer.parallelism.parallel_state.is_initialized")
    @patch("torch.distributed.is_initialized")
    def test_state_dict(self, mock_dist_init, mock_parallel_init):
        """Test state dict save/load."""
        mock_dist_init.return_value = False
        mock_parallel_init.return_value = False

        finalizer = GradientFinalizer(self.model, self.config)
        finalizer.finalization_count = 10

        # Save state
        state_dict = finalizer.state_dict()
        self.assertEqual(state_dict["finalization_count"], 10)
        self.assertIn("config", state_dict)
        self.assertIn("statistics_summary", state_dict)

        # Load state
        new_finalizer = GradientFinalizer(self.model, self.config)
        new_finalizer.load_state_dict(state_dict)
        self.assertEqual(new_finalizer.finalization_count, 10)

    @patch("rosellm.rosetrainer.parallelism.parallel_state.is_initialized")
    @patch("torch.distributed.is_initialized")
    def test_gradient_buffer_initialization(self, mock_dist_init, mock_parallel_init):
        """Test gradient buffer initialization."""
        mock_dist_init.return_value = False
        mock_parallel_init.return_value = False

        config = GradientFinalizationConfig(
            use_contiguous_buffers=True,
            fp16_compression=True,
        )
        finalizer = GradientFinalizer(self.model, config)

        # Check that gradient buffers are created with dtype keys
        self.assertTrue(len(finalizer.gradient_buffers) > 0)
        # Find the main buffer for the model's dtype
        model_dtype = next(self.model.parameters()).dtype
        buffer_key = f"main_{model_dtype}"
        self.assertIn(buffer_key, finalizer.gradient_buffers)
        main_buffer = finalizer.gradient_buffers[buffer_key]
        self.assertIsInstance(main_buffer, torch.Tensor)

        # Check FP16 buffer created for FP32 model
        if next(self.model.parameters()).dtype == torch.float32:
            fp16_key = f"fp16_{torch.float32}"
            self.assertIn(fp16_key, finalizer.gradient_buffers)


class TestIntegrationWithDistributedOptimizer(unittest.TestCase):
    """Test integration with distributed optimizer."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleTestModel()
        self.config = GradientFinalizationConfig()

    @patch("rosellm.rosetrainer.parallelism.parallel_state.is_initialized")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    def test_optimizer_integration(
        self, mock_rank, mock_world_size, mock_dist_init, mock_parallel_init
    ):
        """Test integration with distributed optimizer."""
        mock_dist_init.return_value = True
        mock_parallel_init.return_value = False
        mock_world_size.return_value = 4
        mock_rank.return_value = 0

        # Create mock distributed optimizer
        from rosellm.rosetrainer.optimizer import (
            DistributedOptimizer,
            DistributedOptimizerConfig,
        )

        opt_config = DistributedOptimizerConfig()

        # Mock the distributed optimizer
        with patch.object(DistributedOptimizer, "__init__", return_value=None):
            optimizer = DistributedOptimizer(
                self.model.parameters(),
                torch.optim.Adam,
                {"lr": 0.001},
                opt_config,
            )

            # Set required attributes
            from rosellm.rosetrainer.utils import GradientClipConfig

            optimizer.grad_clip_config = GradientClipConfig(
                clip_type="norm", max_norm=1.0, norm_type=2.0
            )
            optimizer.local_params = list(self.model.parameters())

            finalizer = GradientFinalizer(
                self.model, self.config, distributed_optimizer=optimizer
            )

            # Create gradients
            for param in self.model.parameters():
                param.grad = torch.randn_like(param)

            # Mock sync strategy
            finalizer.sync_strategy.sync_gradients = MagicMock(
                return_value={"total_gradient_norm": 1.0}
            )

            stats = finalizer.finalize_gradients(clip_gradients=True)

            self.assertIn("clip_stats", stats)


if __name__ == "__main__":
    unittest.main()
