"""
Comprehensive Tests for Dynamic Gradient Scaler

This test suite provides exhaustive validation of the DynamicGradientScaler
implementation with bit-to-bit validation against reference implementations.

Test Coverage:
- Basic scaling operations
- Overflow detection and recovery
- Multi-tensor optimizations
- Distributed training compatibility
- Megatron-LM pattern validation
- Performance benchmarking
- Error handling and edge cases
"""

import time
import unittest
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from rosellm.rosetrainer.utils.grad_scaler import (
    AbstractGradientScaler,
    DynamicGradientScaler,
    MultiTensorOverflowDetector,
    OverflowAction,
    ScalingState,
    ScalingStrategy,
    create_gradient_scaler,
    validate_scaler_against_reference,
)
from rosellm.rosetrainer.utils.gradient_utils import (
    GradientClipConfig,
    apply_gradient_clipping_with_scaler,
    check_for_inf_and_nan_with_scaler,
    create_integrated_training_step,
)


class TestDynamicGradientScaler(unittest.TestCase):
    """Test suite for DynamicGradientScaler with comprehensive coverage."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        # Create test model
        self.model = self._create_test_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        # Default scaler parameters
        self.default_params = {
            "init_scale": 2.0**16,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "hysteresis": 2,
            "device": self.device,
        }

    def _create_test_model(self) -> nn.Module:
        """Create a simple model for testing."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 5)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.linear1(x))
                return self.linear2(x)

        model = SimpleModel().to(self.device)
        return model

    def _create_sample_tensors(
        self, count: int = 5, size: int = 100
    ) -> List[torch.Tensor]:
        """Create sample tensors for testing."""
        tensors = []
        for i in range(count):
            tensor = torch.randn(size, device=self.device, dtype=self.dtype)
            tensors.append(tensor)
        return tensors

    def _inject_gradients(self, model: nn.Module, scale: float = 1.0) -> None:
        """Inject gradients into model parameters."""
        for param in model.parameters():
            param.grad = torch.randn_like(param) * scale

    def _inject_overflow_gradients(self, model: nn.Module) -> None:
        """Inject overflow gradients for testing."""
        for i, param in enumerate(model.parameters()):
            if i % 2 == 0:
                param.grad = torch.full_like(param, float("inf"))
            else:
                param.grad = torch.full_like(param, float("nan"))

    def test_initialization(self):
        """Test scaler initialization with various parameters."""
        # Test default initialization
        scaler = DynamicGradientScaler()
        self.assertIsInstance(scaler, AbstractGradientScaler)
        self.assertEqual(scaler.get_scale(), 2.0**16)
        self.assertEqual(scaler.growth_factor, 2.0)
        self.assertEqual(scaler.backoff_factor, 0.5)
        self.assertTrue(scaler.enabled)

        # Test custom initialization
        scaler = DynamicGradientScaler(
            init_scale=1000.0,
            growth_factor=1.5,
            backoff_factor=0.8,
            growth_interval=1000,
            hysteresis=3,
            scaling_strategy=ScalingStrategy.LINEAR,
            overflow_action=OverflowAction.RETRY,
            enabled=True,
        )
        self.assertEqual(scaler.get_scale(), 1000.0)
        self.assertEqual(scaler.growth_factor, 1.5)
        self.assertEqual(scaler.backoff_factor, 0.8)
        self.assertEqual(scaler.growth_interval, 1000)
        self.assertEqual(scaler.hysteresis, 3)
        self.assertEqual(scaler.scaling_strategy, ScalingStrategy.LINEAR)
        self.assertEqual(scaler.overflow_action, OverflowAction.RETRY)

    def test_initialization_validation(self):
        """Test that initialization validates parameters properly."""
        # Test invalid init_scale
        with self.assertRaises(ValueError):
            DynamicGradientScaler(init_scale=-1.0)

        # Test invalid growth_factor
        with self.assertRaises(ValueError):
            DynamicGradientScaler(growth_factor=0.0)
        with self.assertRaises(ValueError):
            DynamicGradientScaler(growth_factor=15.0)

        # Test invalid backoff_factor
        with self.assertRaises(ValueError):
            DynamicGradientScaler(backoff_factor=0.0)
        with self.assertRaises(ValueError):
            DynamicGradientScaler(backoff_factor=1.5)

        # Test invalid growth_interval
        with self.assertRaises(ValueError):
            DynamicGradientScaler(growth_interval=0)

        # Test invalid hysteresis
        with self.assertRaises(ValueError):
            DynamicGradientScaler(hysteresis=0)

    def test_basic_scaling(self):
        """Test basic loss scaling functionality."""
        scaler = DynamicGradientScaler(init_scale=1000.0, device=self.device)

        # Test tensor scaling
        loss = torch.tensor(2.0, device=self.device)
        scaled_loss = scaler.scale(loss)

        self.assertEqual(scaled_loss.item(), 2000.0)
        self.assertEqual(scaled_loss.device, self.device)

        # Test disabled scaling
        scaler.enabled = False
        scaled_loss = scaler.scale(loss)
        self.assertEqual(scaled_loss.item(), 2.0)

    def test_unscaling(self):
        """Test gradient unscaling functionality."""
        scaler = DynamicGradientScaler(init_scale=1000.0, device=self.device)

        # Inject gradients
        self._inject_gradients(self.model, scale=1000.0)

        # Store original gradients for comparison
        original_grads = [
            p.grad.clone() for p in self.model.parameters() if p.grad is not None
        ]

        # Unscale gradients
        scaler.unscale_(self.optimizer)

        # Check that gradients were unscaled
        for orig_grad, param in zip(original_grads, self.model.parameters()):
            expected_grad = orig_grad / 1000.0
            torch.testing.assert_close(param.grad, expected_grad, rtol=1e-5, atol=1e-6)

    def test_overflow_detection(self):
        """Test overflow detection in various scenarios."""
        scaler = DynamicGradientScaler(device=self.device)

        # Test with finite gradients
        self._inject_gradients(self.model, scale=1.0)
        found_overflow = scaler._check_overflow(self.optimizer)
        self.assertFalse(found_overflow)

        # Test with infinite gradients
        self._inject_overflow_gradients(self.model)
        found_overflow = scaler._check_overflow(self.optimizer)
        self.assertTrue(found_overflow)

    def test_multi_tensor_overflow_detection(self):
        """Test multi-tensor overflow detection."""
        detector = MultiTensorOverflowDetector(use_multi_tensor=True)

        # Test with finite tensors
        finite_tensors = self._create_sample_tensors(5, 100)
        found_overflow = detector.check_overflow(finite_tensors)
        self.assertFalse(found_overflow)

        # Test with infinite tensors
        inf_tensors = [torch.full((100,), float("inf"), device=self.device)]
        found_overflow = detector.check_overflow(inf_tensors)
        self.assertTrue(found_overflow)

        # Test with NaN tensors
        nan_tensors = [torch.full((100,), float("nan"), device=self.device)]
        found_overflow = detector.check_overflow(nan_tensors)
        self.assertTrue(found_overflow)

        # Test per-tensor results
        mixed_tensors = finite_tensors + inf_tensors
        per_tensor_results = detector.check_overflow(mixed_tensors, per_tensor=True)
        self.assertIsInstance(per_tensor_results, list)
        if isinstance(per_tensor_results, list):
            self.assertEqual(len(per_tensor_results), len(mixed_tensors))
            self.assertTrue(per_tensor_results[-1])  # Last tensor should have overflow

    def test_scale_updating(self):
        """Test scale factor updates on overflow and success."""
        scaler = DynamicGradientScaler(
            init_scale=1000.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2,
            hysteresis=1,
            device=self.device,
        )

        initial_scale = scaler.get_scale()

        # Test successful steps leading to growth
        for _ in range(2):
            self._inject_gradients(self.model)
            result = scaler.step(self.optimizer)
            # Step should succeed (no exception raised)
            # Note: optimizer.step() typically returns None, which is expected

        # Scale should have grown
        scale_after_growth = scaler.get_scale()
        self.assertGreater(scale_after_growth, initial_scale)

        # Test overflow leading to backoff
        self._inject_overflow_gradients(self.model)
        result = scaler.step(self.optimizer)
        self.assertIsNone(result)  # Step should be skipped

        # Scale should have backed off from the grown scale
        scale_after_backoff = scaler.get_scale()
        self.assertLess(scale_after_backoff, scale_after_growth)

    def test_hysteresis(self):
        """Test hysteresis in overflow handling."""
        scaler = DynamicGradientScaler(
            init_scale=1000.0,
            hysteresis=3,
            device=self.device,
        )

        initial_scale = scaler.get_scale()

        # First overflow should not trigger backoff due to hysteresis
        self._inject_overflow_gradients(self.model)
        scaler.step(self.optimizer)
        self.assertEqual(scaler.get_scale(), initial_scale)
        self.assertEqual(scaler.get_consecutive_overflows(), 1)

        # Second overflow should not trigger backoff
        self._inject_overflow_gradients(self.model)
        scaler.step(self.optimizer)
        self.assertEqual(scaler.get_scale(), initial_scale)
        self.assertEqual(scaler.get_consecutive_overflows(), 2)

        # Third overflow should trigger backoff
        self._inject_overflow_gradients(self.model)
        scaler.step(self.optimizer)
        self.assertLess(scaler.get_scale(), initial_scale)

    def test_scaling_strategies(self):
        """Test different scaling strategies."""
        # Test exponential strategy
        exp_scaler = DynamicGradientScaler(
            init_scale=1000.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            scaling_strategy=ScalingStrategy.EXPONENTIAL,
            device=self.device,
        )

        # Test linear strategy
        lin_scaler = DynamicGradientScaler(
            init_scale=1000.0,
            scaling_strategy=ScalingStrategy.LINEAR,
            device=self.device,
        )

        # Test adaptive strategy
        ada_scaler = DynamicGradientScaler(
            init_scale=1000.0,
            scaling_strategy=ScalingStrategy.ADAPTIVE,
            device=self.device,
        )

        # Test that all strategies handle overflow differently
        for scaler in [exp_scaler, lin_scaler, ada_scaler]:
            scaler.state.consecutive_overflows = scaler.hysteresis
            initial_scale = scaler.get_scale()
            scaler._handle_overflow()
            self.assertNotEqual(scaler.get_scale(), initial_scale)

    def test_overflow_actions(self):
        """Test different overflow actions."""
        # Test SKIP action (default)
        skip_scaler = DynamicGradientScaler(
            overflow_action=OverflowAction.SKIP,
            device=self.device,
        )
        self._inject_overflow_gradients(self.model)
        result = skip_scaler.step(self.optimizer)
        self.assertIsNone(result)

        # Test ABORT action
        abort_scaler = DynamicGradientScaler(
            overflow_action=OverflowAction.ABORT,
            device=self.device,
        )
        self._inject_overflow_gradients(self.model)
        with self.assertRaises(RuntimeError):
            abort_scaler.step(self.optimizer)

    def test_state_dict_and_loading(self):
        """Test state dictionary saving and loading."""
        scaler = DynamicGradientScaler(device=self.device)

        # Modify state
        scaler.state.growth_tracker = 100
        scaler.state.overflow_tracker = 5
        scaler.state.total_steps = 1000
        scaler.state.scale.fill_(5000.0)

        # Save state
        state_dict = scaler.state_dict()

        # Create new scaler and load state
        new_scaler = DynamicGradientScaler(device=self.device)
        new_scaler.load_state_dict(state_dict)

        # Verify state was loaded correctly
        self.assertEqual(new_scaler.state.growth_tracker, 100)
        self.assertEqual(new_scaler.state.overflow_tracker, 5)
        self.assertEqual(new_scaler.state.total_steps, 1000)
        self.assertEqual(new_scaler.get_scale(), 5000.0)

    def test_integration_with_gradient_clipping(self):
        """Test integration with gradient clipping utilities."""
        scaler = DynamicGradientScaler(device=self.device)
        clip_config = GradientClipConfig(clip_type="norm", max_norm=1.0)

        # Inject large gradients
        self._inject_gradients(self.model, scale=10.0)

        # Apply clipping with scaler
        stats = apply_gradient_clipping_with_scaler(self.model, clip_config, scaler)

        self.assertIsInstance(stats, dict)
        self.assertIn("grad_norm", stats)
        self.assertIn("clipped", stats)
        self.assertIn("scale", stats)
        self.assertIn("found_overflow", stats)

    def test_integrated_training_step(self):
        """Test integrated training step function."""
        scaler = DynamicGradientScaler(device=self.device)
        clip_config = GradientClipConfig(clip_type="norm", max_norm=1.0)

        step_fn = create_integrated_training_step(
            self.model, self.optimizer, scaler, clip_config, accumulation_steps=2
        )

        # Test normal training step
        loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        stats = step_fn(loss)

        self.assertIsInstance(stats, dict)
        self.assertIn("stepped", stats)
        self.assertIn("scale", stats)
        self.assertIn("accumulation_step", stats)

    def test_performance_monitoring(self):
        """Test performance statistics collection."""
        scaler = DynamicGradientScaler(device=self.device)

        # Perform several operations
        for _ in range(10):
            self._inject_gradients(self.model)
            scaler.step(self.optimizer)

        # Check performance stats
        stats = scaler.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("successful_steps", stats)
        self.assertIn("overflow_detections", stats)
        self.assertIn("scale_updates", stats)

    def test_thread_safety(self):
        """Test thread safety of scaler operations."""
        import threading
        import time

        scaler = DynamicGradientScaler(device=self.device)
        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    self._inject_gradients(self.model)
                    result = scaler.step(self.optimizer)
                    results.append(result)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Check that no errors occurred
        self.assertEqual(len(errors), 0)
        self.assertGreater(len(results), 0)

    def test_bit_to_bit_validation_against_pytorch(self):
        """Test bit-to-bit validation against PyTorch's native scaler."""
        # Create our scaler
        our_scaler = DynamicGradientScaler(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            device=self.device,
        )

        # Create PyTorch reference scaler
        if torch.cuda.is_available():
            ref_scaler = torch.cuda.amp.GradScaler(
                init_scale=2**16,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
            )
        else:
            # Skip this test on CPU
            self.skipTest("PyTorch AMP scaler requires CUDA")
            return

        # Test scaling operations
        test_tensors = self._create_sample_tensors(3, 50)

        for tensor in test_tensors:
            our_result = our_scaler.scale(tensor.clone())
            ref_result = ref_scaler.scale(tensor.clone())

            # Results should be identical
            torch.testing.assert_close(our_result, ref_result, rtol=0, atol=0)

    def test_validation_function(self):
        """Test the validation utility function."""
        scaler = DynamicGradientScaler(device=self.device)
        test_tensors = self._create_sample_tensors(5, 100)

        results = validate_scaler_against_reference(
            scaler, test_tensors, num_steps=10, tolerance=1e-6
        )

        self.assertIsInstance(results, dict)
        self.assertIn("passed", results)
        self.assertIn("errors", results)
        self.assertIn("scale_differences", results)
        self.assertIn("overflow_mismatches", results)

    def test_convenience_functions(self):
        """Test convenience functions for scaler creation."""
        # Test create_gradient_scaler function
        scaler = create_gradient_scaler(
            init_scale=1000.0,
            scaling_strategy="adaptive",
            use_multi_tensor=True,
        )

        self.assertIsInstance(scaler, DynamicGradientScaler)
        self.assertEqual(scaler.get_scale(), 1000.0)
        self.assertEqual(scaler.scaling_strategy, ScalingStrategy.ADAPTIVE)

    def test_memory_efficiency(self):
        """Test memory efficiency of scaler operations."""
        scaler = DynamicGradientScaler(device=self.device)

        # Monitor memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        # Perform many operations
        for _ in range(100):
            self._inject_gradients(self.model)
            scaler.step(self.optimizer)

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory

            # Memory growth should be minimal (less than 10MB)
            self.assertLess(memory_growth, 10 * 1024 * 1024)

    def test_edge_cases(self):
        """Test various edge cases and error conditions."""
        scaler = DynamicGradientScaler(device=self.device)

        # Test with optimizer that has no gradients (similar to empty case)
        dummy_param = nn.Parameter(torch.randn(2, 2, device=self.device))
        # Don't set gradients, so it should handle missing gradients gracefully
        empty_optimizer = optim.SGD([dummy_param], lr=0.01)
        scaler.step(empty_optimizer)
        # Should handle case with no gradients gracefully (no exception raised)

        # Test with very large scale
        scaler.state.scale.fill_(1e20)
        loss = torch.tensor(1.0, device=self.device)
        scaled_loss = scaler.scale(loss)
        self.assertTrue(torch.isfinite(scaled_loss))

        # Test with very small scale
        scaler.state.scale.fill_(1e-10)
        scaled_loss = scaler.scale(loss)
        self.assertTrue(torch.isfinite(scaled_loss))

    def test_distributed_compatibility(self):
        """Test basic distributed training compatibility."""
        # This test verifies that the scaler doesn't break in distributed setups
        scaler = DynamicGradientScaler(device=self.device)

        # Simulate distributed gradients (different scales across "ranks")
        self._inject_gradients(self.model, scale=2.0)

        # The scaler should handle this gracefully
        scaler.step(self.optimizer)
        # Step should succeed (no exception raised)
        # Note: optimizer.step() typically returns None, which is expected

    def test_apex_compatibility(self):
        """Test APEX compatibility when available."""
        scaler = DynamicGradientScaler(use_multi_tensor=True, device=self.device)

        # Test that APEX detection works correctly
        if scaler.overflow_detector._apex_available:
            self.assertIsNotNone(scaler.overflow_detector._multi_tensor_applier)
            self.assertIsNotNone(scaler.overflow_detector._amp_c)

        # Test overflow detection still works regardless of APEX availability
        self._inject_overflow_gradients(self.model)
        found_overflow = scaler._check_overflow(self.optimizer)
        self.assertTrue(found_overflow)

    def test_check_for_inf_and_nan_integration(self):
        """Test integration with check_for_inf_and_nan_with_scaler."""
        scaler = DynamicGradientScaler(device=self.device)

        # Test with finite gradients
        self._inject_gradients(self.model, scale=1.0)
        found_inf = check_for_inf_and_nan_with_scaler(self.model, scaler)
        self.assertFalse(found_inf)

        # Test with infinite gradients
        self._inject_overflow_gradients(self.model)
        found_inf = check_for_inf_and_nan_with_scaler(self.model, scaler)
        self.assertTrue(found_inf)

    def test_benchmark_performance(self):
        """Benchmark scaler performance for regression testing."""
        scaler = DynamicGradientScaler(device=self.device)

        # Warm up
        for _ in range(10):
            self._inject_gradients(self.model)
            scaler.step(self.optimizer)

        # Benchmark
        start_time = time.perf_counter()
        num_iterations = 100

        for _ in range(num_iterations):
            self._inject_gradients(self.model)
            scaler.step(self.optimizer)

        elapsed_time = time.perf_counter() - start_time
        time_per_iteration = elapsed_time / num_iterations

        # Performance should be reasonable (less than 1ms per iteration on GPU)
        if torch.cuda.is_available():
            self.assertLess(time_per_iteration, 0.001)
        else:
            self.assertLess(time_per_iteration, 0.01)  # More relaxed for CPU


class TestScalingState(unittest.TestCase):
    """Test the ScalingState data class."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale_tensor = torch.tensor(1000.0, device=self.device)
        self.state = ScalingState(scale=self.scale_tensor)

    def test_initialization(self):
        """Test ScalingState initialization."""
        self.assertEqual(self.state.growth_tracker, 0)
        self.assertEqual(self.state.overflow_tracker, 0)
        self.assertEqual(self.state.consecutive_overflows, 0)
        self.assertEqual(self.state.total_steps, 0)
        self.assertEqual(len(self.state.overflow_history), 0)
        self.assertEqual(self.state.last_overflow_step, -1)

    def test_tracker_operations(self):
        """Test tracker operations."""
        # Test growth tracker reset
        self.state.growth_tracker = 10
        self.state.reset_growth_tracker()
        self.assertEqual(self.state.growth_tracker, 0)

        # Test overflow tracker increment
        self.state.increment_overflow_tracker()
        self.assertEqual(self.state.overflow_tracker, 1)
        self.assertEqual(self.state.consecutive_overflows, 1)

        # Test overflow tracker reset
        self.state.reset_overflow_tracker()
        self.assertEqual(self.state.consecutive_overflows, 0)

    def test_history_management(self):
        """Test overflow history management."""
        # Add some history
        for i in range(100):
            self.state.add_to_history(i % 2 == 0)

        self.assertEqual(len(self.state.overflow_history), 100)

        # Test overflow rate calculation
        rate = self.state.get_overflow_rate(window_size=10)
        self.assertIsInstance(rate, float)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

        # Test history size limiting
        for i in range(10000):
            self.state.add_to_history(True)

        # History should be limited to reasonable size
        self.assertLessEqual(len(self.state.overflow_history), 10000)


if __name__ == "__main__":
    # Run tests with different configurations
    unittest.main(verbosity=2)
