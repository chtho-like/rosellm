"""
Comprehensive tests for selective activation recomputation module.

Tests various selection strategies, profiling capabilities, and integration
with PyTorch models to ensure correct functionality and memory savings.
"""

import logging
import unittest
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rosellm.rosetrainer.memory.selective_recompute import (
    LayerProfiler,
    SelectionStrategy,
    SelectiveCheckpointConfig,
    SelectiveCheckpointFunction,
    SelectiveRecomputeManager,
    create_selective_checkpoint_wrapper,
    selective_checkpoint,
)

logger = logging.getLogger(__name__)


class SimpleLayer(nn.Module):
    """Simple layer for testing."""

    def __init__(self, size: int = 256) -> None:
        super().__init__()
        self.linear = nn.Linear(size, size)
        self.norm = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(self.linear(x)))


class MultiLayerModel(nn.Module):
    """Multi-layer model for testing selective checkpointing."""

    def __init__(self, num_layers: int = 8, size: int = 256) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SimpleLayer(size) for _ in range(num_layers)])
        self.embedding = nn.Linear(10, size)
        self.output = nn.Linear(size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        return self.output(x)


class TransformerBlock(nn.Module):
    """Transformer block for testing."""

    def __init__(self, d_model: int = 256, nhead: int = 8) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)

        # FFN with residual
        x = x + self.ffn(x)
        x = self.norm2(x)

        return x


class TransformerModel(nn.Module):
    """Transformer model for testing."""

    def __init__(self, num_layers: int = 6, d_model: int = 256) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=1024)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TestSelectiveCheckpointConfig(unittest.TestCase):
    """Tests for SelectiveCheckpointConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SelectiveCheckpointConfig()
        self.assertEqual(config.strategy, SelectionStrategy.HYBRID)
        self.assertEqual(config.memory_threshold_mb, 1024.0)
        self.assertEqual(config.computation_threshold_ms, 10.0)
        self.assertEqual(config.checkpoint_interval, 3)
        self.assertTrue(config.use_reentrant)

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        # Valid config should not raise
        config = SelectiveCheckpointConfig()
        config.validate()

        # Invalid memory threshold
        config = SelectiveCheckpointConfig(memory_threshold_mb=-1)
        with self.assertRaises(ValueError):
            config.validate()

        # Invalid computation threshold
        config = SelectiveCheckpointConfig(computation_threshold_ms=-1)
        with self.assertRaises(ValueError):
            config.validate()

        # Invalid recompute factor
        config = SelectiveCheckpointConfig(recompute_factor=0)
        with self.assertRaises(ValueError):
            config.validate()

        # Invalid checkpoint interval
        config = SelectiveCheckpointConfig(checkpoint_interval=0)
        with self.assertRaises(ValueError):
            config.validate()

        # Invalid adaptive threshold percentile
        config = SelectiveCheckpointConfig(adaptive_threshold_percentile=101)
        with self.assertRaises(ValueError):
            config.validate()


class TestLayerProfiler(unittest.TestCase):
    """Tests for LayerProfiler."""

    def test_record_forward(self) -> None:
        """Test recording forward pass statistics."""
        config = SelectiveCheckpointConfig()
        profiler = LayerProfiler(config)

        # Record forward pass
        profiler.record_forward("layer_0", 0.01, 1024 * 1024)  # 1MB

        profile = profiler.get_profile("layer_0")
        self.assertIsNotNone(profile)
        assert profile is not None  # Type guard
        self.assertAlmostEqual(profile.memory_usage, 1.0, places=2)
        # Note: computation_time uses EMA with default decay of 0.9
        # First value: 0 * 0.9 + 0.01 * 0.1 = 0.001
        self.assertAlmostEqual(profile.computation_time, 0.001, places=4)

    def test_record_recompute(self) -> None:
        """Test recording recomputation statistics."""
        config = SelectiveCheckpointConfig()
        profiler = LayerProfiler(config)

        # Need to record forward first
        profiler.record_forward("layer_0", 0.01, 1024 * 1024)
        profiler.record_recompute("layer_0", 0.005)

        profile = profiler.get_profile("layer_0")
        self.assertIsNotNone(profile)
        assert profile is not None  # Type guard
        # Note: recompute_time uses EMA with default decay of 0.9
        # First value: 0 * 0.9 + 0.005 * 0.1 = 0.0005
        self.assertAlmostEqual(profile.recompute_time, 0.0005, places=4)

    def test_checkpoint_decision_tracking(self) -> None:
        """Test tracking checkpoint decisions."""
        config = SelectiveCheckpointConfig()
        profiler = LayerProfiler(config)

        # Record multiple decisions
        for i in range(10):
            profiler.record_checkpoint_decision("layer_0", i % 2 == 0)

        profile = profiler.get_profile("layer_0")
        self.assertIsNotNone(profile)
        assert profile is not None  # Type guard
        self.assertEqual(profile.checkpoint_count, 5)
        self.assertEqual(profile.skip_count, 5)

    def test_top_layers_retrieval(self) -> None:
        """Test retrieving top memory/computation layers."""
        config = SelectiveCheckpointConfig()
        profiler = LayerProfiler(config)

        # Record different layers
        profiler.record_forward("layer_0", 0.01, 1024 * 1024)  # 1MB
        profiler.record_forward("layer_1", 0.02, 2 * 1024 * 1024)  # 2MB
        profiler.record_forward("layer_2", 0.005, 512 * 1024)  # 0.5MB

        # Get top memory layers
        top_memory = profiler.get_top_memory_layers(2)
        self.assertEqual(len(top_memory), 2)
        self.assertEqual(top_memory[0][0], "layer_1")
        self.assertEqual(top_memory[1][0], "layer_0")

        # Get top computation layers
        top_compute = profiler.get_top_computation_layers(2)
        self.assertEqual(len(top_compute), 2)
        self.assertEqual(top_compute[0][0], "layer_1")
        self.assertEqual(top_compute[1][0], "layer_0")

    def test_profiling_summary(self) -> None:
        """Test getting profiling summary."""
        config = SelectiveCheckpointConfig()
        profiler = LayerProfiler(config)

        # Record some data
        profiler.record_forward("layer_0", 0.01, 1024 * 1024)
        profiler.record_forward("layer_1", 0.02, 2 * 1024 * 1024)
        profiler.record_recompute("layer_0", 0.005)
        profiler.record_checkpoint_decision("layer_0", True)
        profiler.record_checkpoint_decision("layer_1", False)

        summary = profiler.get_summary()

        self.assertEqual(summary["total_layers"], 2)
        self.assertEqual(summary["checkpointed_layers"], 1)
        self.assertAlmostEqual(summary["total_memory_mb"], 3.0, places=1)
        self.assertGreater(summary["total_compute_time_sec"], 0)
        self.assertGreaterEqual(summary["recompute_overhead_ratio"], 0)


class TestSelectiveRecomputeManager(unittest.TestCase):
    """Tests for SelectiveRecomputeManager."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_uniform_strategy(self) -> None:
        """Test uniform selection strategy."""
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.UNIFORM,
            checkpoint_interval=2,
        )
        manager = SelectiveRecomputeManager(config)

        # Test layer selection (every 2nd layer starting from 0)
        self.assertTrue(manager.should_checkpoint_layer("layer_0"))  # 0 % 2 = 0
        self.assertFalse(manager.should_checkpoint_layer("layer_1"))  # 1 % 2 = 1
        self.assertTrue(manager.should_checkpoint_layer("layer_2"))  # 2 % 2 = 0
        self.assertFalse(manager.should_checkpoint_layer("layer_3"))  # 3 % 2 = 1
        self.assertTrue(manager.should_checkpoint_layer("layer_4"))  # 4 % 2 = 0

    def test_manual_strategy(self) -> None:
        """Test manual selection strategy."""
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.MANUAL,
            layers_to_checkpoint=["layer_0", "layer_2", "layer_4"],
            layers_to_skip=["layer_2"],  # Should override checkpoint list
        )
        manager = SelectiveRecomputeManager(config)

        self.assertTrue(manager.should_checkpoint_layer("layer_0"))
        self.assertFalse(manager.should_checkpoint_layer("layer_1"))
        self.assertFalse(manager.should_checkpoint_layer("layer_2"))  # Skipped
        self.assertFalse(manager.should_checkpoint_layer("layer_3"))
        self.assertTrue(manager.should_checkpoint_layer("layer_4"))

    def test_memory_based_strategy(self) -> None:
        """Test memory-based selection strategy."""
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.MEMORY_BASED,
            memory_threshold_mb=1.5,
            profile_enabled=True,
        )
        manager = SelectiveRecomputeManager(config)

        # Without profiling data, should return False
        self.assertFalse(manager.should_checkpoint_layer("layer_0"))

        # Add profiling data
        if manager.profiler is not None:
            manager.profiler.record_forward("layer_0", 0.01, 1024 * 1024)  # 1MB
            manager.profiler.record_forward("layer_1", 0.01, 2 * 1024 * 1024)  # 2MB

            self.assertFalse(manager.should_checkpoint_layer("layer_0"))  # < threshold
            self.assertTrue(manager.should_checkpoint_layer("layer_1"))  # > threshold

    def test_computation_based_strategy(self) -> None:
        """Test computation-based selection strategy."""
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.COMPUTATION_BASED,
            computation_threshold_ms=15,
            recompute_factor=2.0,
            profile_enabled=True,
        )
        manager = SelectiveRecomputeManager(config)

        if manager.profiler is not None:
            # Add layers with different computation times
            # Note: EMA with decay 0.9 scales first value by (1 - 0.9) = 0.1
            manager.profiler.record_forward(
                "layer_0", 0.1, 1024 * 1024
            )  # 100ms -> 10ms after EMA (0 * 0.9 + 0.1 * 0.1)
            manager.profiler.record_forward(
                "layer_1", 0.2, 1024 * 1024
            )  # 200ms -> 20ms after EMA (0 * 0.9 + 0.2 * 0.1)
            manager.profiler.record_recompute(
                "layer_1", 0.35
            )  # 350ms -> 35ms after EMA (0 * 0.9 + 0.35 * 0.1)

            self.assertFalse(
                manager.should_checkpoint_layer("layer_0")
            )  # 10ms < 15ms threshold
            self.assertTrue(
                manager.should_checkpoint_layer("layer_1")
            )  # 20ms > 15ms threshold, factor = 1.75 < 2.0

            # Add layer with bad recompute factor
            manager.profiler.record_forward(
                "layer_2", 0.2, 1024 * 1024
            )  # 200ms -> 20ms after EMA (0 * 0.9 + 0.2 * 0.1)
            manager.profiler.record_recompute(
                "layer_2", 0.5
            )  # 500ms -> 50ms after EMA (0 * 0.9 + 0.5 * 0.1), factor = 50/20 = 2.5

            self.assertFalse(
                manager.should_checkpoint_layer("layer_2")
            )  # Bad factor > 2.0

    def test_checkpoint_layer_function(self) -> None:
        """Test checkpointing a layer function."""
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.MANUAL,
            layers_to_checkpoint=["test_layer"],
        )
        manager = SelectiveRecomputeManager(config)

        # Create a simple function
        def simple_func(x: torch.Tensor) -> torch.Tensor:
            return x * 2 + 1

        # Test with checkpointing
        x = torch.randn(4, 8, requires_grad=True, device=self.device)
        result = manager.checkpoint_layer(simple_func, x, layer_id="test_layer")

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, x.shape)

        # Can do backward
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_model_wrapping(self) -> None:
        """Test wrapping a model with selective checkpointing."""
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.UNIFORM,
            checkpoint_interval=2,
        )
        manager = SelectiveRecomputeManager(config)

        # Create and wrap model
        model = TransformerModel(num_layers=4, d_model=128)
        model.to(self.device)
        wrapped_model = manager.wrap_model(model)

        # Test forward pass
        x = torch.randn(10, 8, 128, device=self.device)
        output = wrapped_model(x)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, x.shape)

    def test_adaptive_strategy_warmup(self) -> None:
        """Test adaptive strategy during warmup phase."""
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.ADAPTIVE,
            profile_warmup_steps=5,
            profile_update_interval=2,
            profile_enabled=True,
        )
        manager = SelectiveRecomputeManager(config)

        # During warmup, should checkpoint everything
        for i in range(5):
            self.assertTrue(manager.should_checkpoint_layer(f"layer_{i}"))
            manager.step_count = i + 1

        # After warmup, selection should be based on profiles
        manager.step_count = 6

        # Add some profiling data
        if manager.profiler is not None:
            for i in range(10):
                manager.profiler.record_forward(
                    f"layer_{i}", 0.001 * i, i * 1024 * 1024
                )

        # Update selection
        manager.update_selection()

        # Should have selected top percentile layers
        # Check if strategy has selected layers
        if hasattr(manager.strategy, "selected_layers"):
            selected = manager.strategy.selected_layers  # type: ignore[attr-defined]
            self.assertGreater(len(selected), 0)
            self.assertLess(len(selected), 10)

    def test_profiling_report(self) -> None:
        """Test getting profiling report."""
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.HYBRID,
            profile_enabled=True,
        )
        manager = SelectiveRecomputeManager(config)

        # Add some profiling data
        if manager.profiler is not None:
            manager.profiler.record_forward("layer_0", 0.01, 1024 * 1024)
            manager.profiler.record_checkpoint_decision("layer_0", True)

        report = manager.get_profiling_report()

        self.assertIn("total_layers", report)
        self.assertIn("selection_strategy", report)
        self.assertEqual(report["selection_strategy"], "hybrid")

    def test_reset_profiling(self) -> None:
        """Test resetting profiling statistics."""
        config = SelectiveCheckpointConfig(profile_enabled=True)
        manager = SelectiveRecomputeManager(config)

        # Add data
        if manager.profiler is not None:
            manager.profiler.record_forward("layer_0", 0.01, 1024 * 1024)

        # Add some selection state if strategy supports it
        if hasattr(manager.strategy, "selected_layers"):
            manager.strategy.selected_layers.add(  # type: ignore[attr-defined]
                "layer_0"
            )
        manager.step_count = 10

        # Reset
        manager.reset_profiling()

        self.assertEqual(manager.step_count, 0)
        # Check that strategy state was cleared
        if hasattr(manager.strategy, "selected_layers"):
            self.assertEqual(
                len(manager.strategy.selected_layers), 0  # type: ignore[attr-defined]
            )
        if manager.profiler is not None:
            self.assertEqual(len(manager.profiler.profiles), 0)


class TestSelectiveCheckpointFunction(unittest.TestCase):
    """Tests for SelectiveCheckpointFunction."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward_backward(self) -> None:
        """Test forward and backward passes."""

        # Create a simple function
        def func(x: torch.Tensor) -> torch.Tensor:
            return x.pow(2).sum(dim=-1)

        # Create input
        x = torch.randn(4, 8, requires_grad=True, device=self.device)

        # Apply checkpoint function
        output = SelectiveCheckpointFunction.apply(func, True, "test_layer", None, x)

        # Check output
        expected = func(x)
        self.assertIsNotNone(output)
        assert output is not None  # Type guard
        self.assertTrue(torch.allclose(output, expected))

        # Test backward
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_with_profiler(self) -> None:
        """Test with profiler enabled."""
        config = SelectiveCheckpointConfig()
        profiler = LayerProfiler(config)

        def func(x: torch.Tensor) -> torch.Tensor:
            return F.relu(x)

        x = torch.randn(4, 8, requires_grad=True, device=self.device)

        # Forward pass with profiler
        output = SelectiveCheckpointFunction.apply(
            func, True, "test_layer", profiler, x
        )

        # Check profiling data was recorded
        profile = profiler.get_profile("test_layer")
        self.assertIsNotNone(profile)
        assert profile is not None  # Type guard
        self.assertGreater(profile.computation_time, 0)

        # Backward pass should record recompute time
        self.assertIsNotNone(output)
        assert output is not None  # Type guard
        loss = output.sum()
        loss.backward()

        profile = profiler.get_profile("test_layer")
        self.assertIsNotNone(profile)
        assert profile is not None  # Type guard
        self.assertGreater(profile.recompute_time, 0)


class TestUtilityFunctions(unittest.TestCase):
    """Tests for utility functions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_selective_checkpoint_function(self) -> None:
        """Test selective_checkpoint convenience function."""

        def func(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        x = torch.randn(4, 8, requires_grad=True, device=self.device)

        # Test with default config
        result = selective_checkpoint(func, x)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, x.shape)

        # Test with custom config
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.MANUAL,
            layers_to_checkpoint=["test"],
        )
        result = selective_checkpoint(func, x, config=config, layer_id="test")
        self.assertIsNotNone(result)

        # Can do backward
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_create_checkpoint_wrapper(self) -> None:
        """Test creating a reusable checkpoint wrapper."""
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.UNIFORM,
            checkpoint_interval=1,
        )
        wrapper = create_selective_checkpoint_wrapper(config)

        def func(x: torch.Tensor) -> torch.Tensor:
            return x.pow(2)

        x = torch.randn(4, 8, requires_grad=True, device=self.device)

        # Use wrapper
        result = wrapper(func, x, layer_id="layer_0")
        self.assertIsNotNone(result)

        # Check manager is attached
        self.assertTrue(hasattr(wrapper, "manager"))
        self.assertIsInstance(
            wrapper.manager, SelectiveRecomputeManager  # type: ignore[attr-defined]
        )


class TestIntegration(unittest.TestCase):
    """Integration tests with real models."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_end_to_end_training_step(self) -> None:
        """Test a complete training step with selective checkpointing."""
        # Create model
        model = MultiLayerModel(num_layers=4, size=64)
        model.to(self.device)

        # Configure selective checkpointing
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.UNIFORM,
            checkpoint_interval=2,
            profile_enabled=True,
        )
        manager = SelectiveRecomputeManager(config)

        # Wrap model layers manually for this test
        for i, layer in enumerate(model.layers):
            original_forward = layer.forward
            layer.forward = (
                lambda x, orig=original_forward, idx=i: manager.checkpoint_layer(
                    orig, x, layer_id=f"layer_{idx}"
                )
            )

        # Training step
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for step in range(3):
            x = torch.randn(8, 16, 10, device=self.device)
            target = torch.randn(8, 16, 10, device=self.device)

            # Forward
            output = model(x)
            loss = F.mse_loss(output, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check profiling report
        report = manager.get_profiling_report()
        self.assertIn("total_layers", report)
        self.assertGreater(report["total_layers"], 0)

    def test_memory_savings(self) -> None:
        """Test that selective checkpointing can manage memory usage."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")

        # Create a larger model
        model = MultiLayerModel(num_layers=8, size=512)
        model.to(self.device)

        x = torch.randn(4, 32, 10, device=self.device)

        # Measure memory without checkpointing
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        output1 = model(x)
        loss1 = output1.sum()
        loss1.backward()

        mem_without = torch.cuda.max_memory_allocated()

        # Clear gradients and reset
        model.zero_grad()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Apply selective checkpointing - checkpoint every 2nd layer
        # This provides a balance between memory savings and overhead
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.UNIFORM,
            checkpoint_interval=2,  # Checkpoint every 2nd layer
        )
        manager = SelectiveRecomputeManager(config)

        # Wrap model layers
        for i, layer in enumerate(model.layers):
            original_forward = layer.forward
            layer.forward = (
                lambda x, orig=original_forward, idx=i: manager.checkpoint_layer(
                    orig, x, layer_id=f"layer_{idx}"
                )
            )

        # Measure memory with checkpointing
        output2 = model(x)
        loss2 = output2.sum()
        loss2.backward()

        mem_with = torch.cuda.max_memory_allocated()

        # Log memory usage for debugging
        mem_saved_pct = (mem_without - mem_with) / mem_without * 100
        logger.info(f"Memory without checkpointing: {mem_without / 1024**2:.2f} MB")
        logger.info(f"Memory with checkpointing: {mem_with / 1024**2:.2f} MB")
        logger.info(f"Memory saved: {mem_saved_pct:.1f}%")

        # Checkpointing should at least not significantly increase memory
        # Allow up to 10% increase due to overhead
        self.assertLessEqual(mem_with, mem_without * 1.1)

        # Outputs should be similar (allowing for numerical differences)
        self.assertTrue(torch.allclose(output1, output2, rtol=1e-4, atol=1e-6))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_model(self) -> None:
        """Test with a model that has no eligible layers."""
        model = nn.Linear(10, 10)
        config = SelectiveCheckpointConfig()
        manager = SelectiveRecomputeManager(config)

        # Should handle gracefully
        wrapped = manager.wrap_model(model)
        self.assertIsNotNone(wrapped)

    def test_invalid_config(self) -> None:
        """Test configuration validation."""
        # Test multiple invalid parameters
        config = SelectiveCheckpointConfig(
            memory_threshold_mb=-1,
            computation_threshold_ms=-1,
            recompute_factor=0,
            checkpoint_interval=0,
            adaptive_threshold_percentile=101,
            ema_decay_factor=1.5,
            max_profile_history=-1,
        )

        with self.assertRaises(ValueError) as cm:
            config.validate()

        error_msg = str(cm.exception)
        self.assertIn("memory_threshold_mb must be positive", error_msg)
        self.assertIn("computation_threshold_ms must be non-negative", error_msg)
        self.assertIn("recompute_factor must be positive", error_msg)

    def test_concurrent_access(self) -> None:
        """Test thread safety with concurrent access."""
        import threading

        config = SelectiveCheckpointConfig(
            thread_safe=True,
            profile_enabled=True,
        )
        manager = SelectiveRecomputeManager(config)

        def worker(thread_id: int) -> None:
            for i in range(10):
                layer_id = f"thread_{thread_id}_layer_{i}"
                result = manager.should_checkpoint_layer(layer_id)
                self.assertIsInstance(result, bool)

                if manager.profiler:
                    manager.profiler.record_forward(layer_id, 0.001, 1024)

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should complete without deadlocks or errors
        self.assertTrue(True)

    def test_none_inputs(self) -> None:
        """Test handling of None inputs in checkpoint function."""

        def func(x: Optional[torch.Tensor]) -> torch.Tensor:
            if x is None:
                return torch.zeros(1)
            return x * 2

        config = SelectiveCheckpointConfig()
        manager = SelectiveRecomputeManager(config)

        # Should handle None gracefully
        result = manager.checkpoint_layer(func, None)
        self.assertIsNotNone(result)

    def test_multiple_outputs(self) -> None:
        """Test functions with multiple outputs."""

        def func(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return x * 2, x * 3

        x = torch.randn(4, 8, requires_grad=True)
        config = SelectiveCheckpointConfig(
            strategy=SelectionStrategy.MANUAL,
            layers_to_checkpoint=["multi_output"],
        )
        manager = SelectiveRecomputeManager(config)

        # Test that multiple outputs work correctly
        result = manager.checkpoint_layer(func, x, layer_id="multi_output")

        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        # Verify outputs are correct
        expected = func(x)
        self.assertTrue(torch.allclose(result[0], expected[0]))
        self.assertTrue(torch.allclose(result[1], expected[1]))

        # Test backward pass
        loss = result[0].sum() + result[1].sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_nested_checkpointing(self) -> None:
        """Test nested checkpointing configuration."""
        config = SelectiveCheckpointConfig(
            nested_checkpointing=True,
        )
        manager = SelectiveRecomputeManager(config)

        # Create nested functions
        def inner(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        def outer(x: torch.Tensor) -> torch.Tensor:
            y = manager.checkpoint_layer(inner, x, layer_id="inner")
            return y + 1

        x = torch.randn(4, 8, requires_grad=True)
        result = manager.checkpoint_layer(outer, x, layer_id="outer")

        self.assertIsNotNone(result)

        # Can do backward
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
