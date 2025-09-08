"""
Comprehensive tests for Dynamic Loss Scaling implementation

This test suite covers:
- DynamicScalerConfig validation and creation
- DynamicGradScaler core functionality
- Multi-tensor overflow detection
- APEX integration (when available)
- Integration with MixedPrecisionManager
- Performance and accuracy validation
- Edge cases and error handling
"""

import warnings

import pytest
import torch
import torch.nn as nn

from rosellm.rosetrainer.mixed_precision.dynamic_scaler import (
    DynamicGradScaler,
    DynamicScalerConfig,
    MultiTensorOverflowDetector,
    create_dynamic_scaler,
    get_recommended_config,
    is_apex_available,
)
from rosellm.rosetrainer.mixed_precision.mixed_precision import (
    MixedPrecisionConfig,
    MixedPrecisionManager,
    PrecisionType,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(
        self, input_size: int = 10, hidden_size: int = 20, output_size: int = 1
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear2(x)


def create_test_tensors(device: torch.device, dtype: torch.dtype = torch.float32):
    """Create test tensors with different characteristics."""
    return [
        torch.randn(100, 50, device=device, dtype=dtype),  # Normal tensor
        torch.randn(200, device=device, dtype=dtype),  # 1D tensor
        torch.randn(10, 10, 10, device=device, dtype=dtype),  # 3D tensor
        torch.zeros(50, device=device, dtype=dtype),  # Zeros
        torch.ones(25, 25, device=device, dtype=dtype),  # Ones
    ]


def create_overflow_tensors(device: torch.device, dtype: torch.dtype = torch.float32):
    """Create tensors with overflow values."""
    tensors = create_test_tensors(device, dtype)
    # Add overflow values
    tensors[0][0, 0] = float("inf")  # Positive infinity
    tensors[1][0] = float("-inf")  # Negative infinity
    tensors[2][0, 0, 0] = float("nan")  # NaN
    return tensors


class TestDynamicScalerConfig:
    """Test suite for DynamicScalerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DynamicScalerConfig()

        assert config.initial_scale == 2**16
        assert config.min_scale == 1.0
        assert config.max_scale == 2**24
        assert config.growth_factor == 2.0
        assert config.backoff_factor == 0.5
        assert config.growth_interval == 2000
        assert config.hysteresis == 2
        assert config.use_multi_tensor is True
        assert config.enable_inf_nan_check is True
        assert config.log_scale_changes is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DynamicScalerConfig(
            initial_scale=1024,
            min_scale=0.5,
            growth_factor=1.5,
            backoff_factor=0.25,
            growth_interval=1000,
            hysteresis=3,
            use_multi_tensor=False,
            detailed_overflow_info=True,
        )

        assert config.initial_scale == 1024
        assert config.min_scale == 0.5
        assert config.growth_factor == 1.5
        assert config.backoff_factor == 0.25
        assert config.growth_interval == 1000
        assert config.hysteresis == 3
        assert config.use_multi_tensor is False
        assert config.detailed_overflow_info is True

    def test_invalid_scale_bounds(self):
        """Test validation of scale bounds."""
        with pytest.raises(ValueError, match="Scale bounds must satisfy"):
            DynamicScalerConfig(min_scale=100, initial_scale=50)

        with pytest.raises(ValueError, match="Scale bounds must satisfy"):
            DynamicScalerConfig(initial_scale=1000, max_scale=500)

    def test_invalid_factors(self):
        """Test validation of growth and backoff factors."""
        with pytest.raises(ValueError, match="growth_factor must be in"):
            DynamicScalerConfig(growth_factor=0.5)

        with pytest.raises(ValueError, match="backoff_factor must be in"):
            DynamicScalerConfig(backoff_factor=1.5)

    def test_invalid_intervals(self):
        """Test validation of timing parameters."""
        with pytest.raises(ValueError, match="growth_interval must be in"):
            DynamicScalerConfig(growth_interval=0)

        with pytest.raises(ValueError, match="hysteresis must be in"):
            DynamicScalerConfig(hysteresis=0)


class TestMultiTensorOverflowDetector:
    """Test suite for MultiTensorOverflowDetector."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_no_overflow_detection(self, device):
        """Test detection with normal tensors."""
        detector = MultiTensorOverflowDetector()
        tensors = create_test_tensors(device)

        has_overflow, info = detector.detect_overflow(tensors)

        assert has_overflow is False
        assert info["total_tensors"] == len(tensors)
        assert info["total_elements"] > 0

    def test_overflow_detection(self, device):
        """Test detection with overflow tensors."""
        detector = MultiTensorOverflowDetector()
        tensors = create_overflow_tensors(device)

        has_overflow, info = detector.detect_overflow(tensors)

        assert has_overflow is True
        assert info["total_tensors"] == len(tensors)
        assert info["total_elements"] > 0

    def test_empty_tensor_list(self, device):
        """Test detection with empty tensor list."""
        detector = MultiTensorOverflowDetector()

        has_overflow, info = detector.detect_overflow([])

        assert has_overflow is False
        assert info["total_tensors"] == 0
        assert info["total_elements"] == 0

    def test_none_tensors(self, device):
        """Test detection with None values in tensor list."""
        detector = MultiTensorOverflowDetector()
        tensors = [None, torch.randn(10, device=device), None]

        has_overflow, info = detector.detect_overflow(tensors)

        assert has_overflow is False
        assert info["total_tensors"] == 1  # Only non-None tensor counted

    def test_large_tensor_chunking(self, device):
        """Test chunking with large tensors."""
        detector = MultiTensorOverflowDetector(chunk_size=1000)
        # Create large tensor that will be chunked
        large_tensor = torch.randn(5000, device=device)

        has_overflow, info = detector.detect_overflow([large_tensor])

        assert has_overflow is False
        assert info["total_elements"] == 5000

    def test_mixed_dtypes(self, device):
        """Test detection with mixed data types."""
        detector = MultiTensorOverflowDetector()
        tensors = [
            torch.randn(10, device=device, dtype=torch.float32),
            (
                torch.randn(10, device=device, dtype=torch.float16)
                if device.type == "cuda"
                else torch.randn(10, device=device, dtype=torch.float32)
            ),
        ]

        has_overflow, info = detector.detect_overflow(tensors)

        assert has_overflow is False
        assert info["total_tensors"] == len(tensors)


class TestDynamicGradScaler:
    """Test suite for DynamicGradScaler."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def model(self, device):
        model = SimpleModel()
        model.to(device)
        return model

    @pytest.fixture
    def scaler(self, device):
        config = DynamicScalerConfig(
            initial_scale=1024,
            growth_interval=10,  # Short interval for testing
            hysteresis=2,
        )
        return DynamicGradScaler(config=config, device=device)

    def test_initialization(self, device):
        """Test scaler initialization."""
        config = DynamicScalerConfig(initial_scale=2048)
        scaler = DynamicGradScaler(config=config, device=device)

        assert scaler.device == device
        assert scaler.scale.item() == 2048
        assert scaler.config.initial_scale == 2048

    def test_scale_properties(self, scaler):
        """Test scale and inverse scale properties."""
        initial_scale = scaler.scale.item()
        inv_scale = scaler.inv_scale

        assert abs(initial_scale * inv_scale.item() - 1.0) < 1e-6

        # Test caching
        inv_scale2 = scaler.inv_scale
        assert torch.equal(inv_scale, inv_scale2)

    def test_loss_scaling(self, scaler):
        """Test loss scaling functionality."""
        loss = torch.tensor(2.5, device=scaler.device)
        scaled_loss = scaler.scale_loss(loss)

        expected = loss * scaler.scale
        assert torch.allclose(scaled_loss, expected)

    def test_gradient_unscaling(self, scaler, model, device):
        """Test gradient unscaling."""
        # Create fake gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 1000  # Large gradients

        # Store original gradients
        original_grads = [p.grad.clone() for p in model.parameters()]

        # Unscale gradients
        scaler.unscale_gradients(model)

        # Check gradients were unscaled
        inv_scale = scaler.inv_scale.item()
        for orig_grad, param in zip(original_grads, model.parameters()):
            expected_grad = orig_grad * inv_scale
            assert torch.allclose(param.grad, expected_grad, atol=1e-6)

    def test_no_overflow_update(self, scaler):
        """Test scale update with no overflow."""
        initial_scale = scaler.scale.item()

        # Simulate many successful steps
        for _ in range(15):  # More than growth_interval
            scaler.update_scale(found_overflow=False)

        # Scale should have grown
        assert scaler.scale.item() > initial_scale

    def test_overflow_update(self, scaler):
        """Test scale update with overflow."""
        initial_scale = scaler.scale.item()

        # Simulate overflow
        scaler.update_scale(found_overflow=True)
        scaler.update_scale(found_overflow=True)  # Trigger hysteresis

        # Scale should have decreased
        assert scaler.scale.item() < initial_scale

    def test_overflow_detection_integration(self, scaler, model, device):
        """Test integrated overflow detection and update."""
        # Create gradients with overflow
        for param in model.parameters():
            param.grad = torch.full_like(param, float("inf"))

        has_overflow = scaler.check_overflow_and_update(model, force_check=True)

        assert has_overflow is True

    def test_min_scale_bound(self, device):
        """Test minimum scale boundary."""
        config = DynamicScalerConfig(
            initial_scale=4.0,
            min_scale=2.0,
            backoff_factor=0.25,
            hysteresis=1,
        )
        scaler = DynamicGradScaler(config=config, device=device)

        # Force scale to minimum
        for _ in range(5):
            scaler.update_scale(found_overflow=True)

        assert scaler.scale.item() == 2.0

    def test_max_scale_bound(self, device):
        """Test maximum scale boundary."""
        config = DynamicScalerConfig(
            initial_scale=512,
            max_scale=1024,
            growth_factor=3.0,
            growth_interval=5,
        )
        scaler = DynamicGradScaler(config=config, device=device)

        # Force scale to maximum
        for _ in range(20):
            scaler.update_scale(found_overflow=False)

        assert scaler.scale.item() == 1024

    def test_state_dict_save_load(self, scaler, device):
        """Test state dict save and load functionality."""
        # Modify scaler state
        scaler._growth_tracker = 5
        scaler._step_count = 100
        original_scale = scaler.scale.item()

        # Save state
        state_dict = scaler.state_dict()

        # Create new scaler and load state
        config = DynamicScalerConfig()
        new_scaler = DynamicGradScaler(config=config, device=device)
        new_scaler.load_state_dict(state_dict)

        assert new_scaler.scale.item() == original_scale
        assert new_scaler._growth_tracker == 5
        assert new_scaler._step_count == 100

    def test_scale_info(self, scaler):
        """Test scale information retrieval."""
        info = scaler.get_scale_info()

        required_keys = [
            "current_scale",
            "step_count",
            "growth_tracker",
            "hysteresis_tracker",
            "total_overflows",
            "config",
        ]

        for key in required_keys:
            assert key in info

        assert info["current_scale"] == scaler.scale.item()
        assert isinstance(info["config"], dict)


class TestMixedPrecisionIntegration:
    """Test integration between dynamic scaler and mixed precision manager."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def model(self, device):
        model = SimpleModel()
        model.to(device)
        return model

    def test_mixed_precision_manager_creation(self, device):
        """Test mixed precision manager with dynamic scaler."""
        config = MixedPrecisionConfig(
            precision=PrecisionType.FP16,
            use_dynamic_scaling=True,
            scaler_config=DynamicScalerConfig(initial_scale=1024),
        )
        manager = MixedPrecisionManager(config, device)

        assert manager.config.precision == PrecisionType.FP16
        assert manager.config.use_dynamic_scaling is True
        assert manager.scaler is not None

    def test_autocast_context(self, device):
        """Test autocast context creation."""
        config = MixedPrecisionConfig(precision=PrecisionType.FP16)
        manager = MixedPrecisionManager(config, device)

        with manager.autocast_context():
            # Should not raise error
            x = torch.randn(10, device=device)
            y = x * 2.0
            assert y.shape == x.shape

    def test_loss_scaling_and_backward(self, device, model):
        """Test loss scaling and backward pass."""
        if device.type == "cpu":
            pytest.skip("Mixed precision primarily for GPU")

        config = MixedPrecisionConfig(
            precision=PrecisionType.FP16,
            use_dynamic_scaling=True,
        )
        manager = MixedPrecisionManager(config, device)
        # optimizer = torch.optim.Adam(model.parameters())

        # Forward pass
        x = torch.randn(5, 10, device=device)
        with manager.autocast_context():
            output = model(x)
            loss = output.sum()

        # Backward pass with scaling
        manager.backward_step(loss)

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None

    def test_optimizer_step_integration(self, device, model):
        """Test complete optimizer step with mixed precision."""
        config = MixedPrecisionConfig(
            precision=(
                PrecisionType.FP16 if device.type == "cuda" else PrecisionType.FP32
            ),
            use_dynamic_scaling=True,
        )
        manager = MixedPrecisionManager(config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training step simulation
        x = torch.randn(5, 10, device=device)
        target = torch.randn(5, 1, device=device)

        optimizer.zero_grad()

        with manager.autocast_context():
            output = model(x)
            loss = nn.MSELoss()(output, target)

        manager.backward_step(loss)
        success = manager.optimizer_step(optimizer, model)

        assert success is True

    def test_statistics_tracking(self, device, model):
        """Test statistics tracking in mixed precision manager."""
        config = MixedPrecisionConfig(
            use_dynamic_scaling=True,
            track_scale_history=True,
        )
        manager = MixedPrecisionManager(config, device)

        # Simulate some steps
        for _ in range(5):
            manager.check_overflow_and_step(model)

        stats = manager.get_statistics()

        assert "total_steps" in stats
        assert "successful_steps" in stats
        assert "overflow_count" in stats
        assert "success_rate" in stats
        assert stats["total_steps"] == 5

    def test_checkpoint_save_load(self, device):
        """Test checkpointing of mixed precision manager."""
        config = MixedPrecisionConfig(use_dynamic_scaling=True)
        manager = MixedPrecisionManager(config, device)

        # Modify state
        manager._total_steps = 100
        manager._overflow_count = 5

        # Save and load
        state_dict = manager.state_dict()
        new_manager = MixedPrecisionManager(config, device)
        new_manager.load_state_dict(state_dict)

        assert new_manager._total_steps == 100
        assert new_manager._overflow_count == 5


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_dynamic_scaler(self):
        """Test factory function for creating dynamic scaler."""
        scaler = create_dynamic_scaler(
            initial_scale=2048,
            growth_interval=1000,
            use_multi_tensor=False,
        )

        assert scaler.config.initial_scale == 2048
        assert scaler.config.growth_interval == 1000
        assert scaler.config.use_multi_tensor is False

    def test_is_apex_available(self):
        """Test APEX availability detection."""
        result = is_apex_available()
        assert isinstance(result, bool)

    def test_recommended_config_small_model(self):
        """Test recommended configuration for small models."""
        config = get_recommended_config(
            model_size="small", precision="fp16", stability_preference="stable"
        )

        assert config.initial_scale == 2**14  # 16K for small models
        assert config.growth_interval == 2000  # Doubled for stable
        assert config.hysteresis == 4  # Conservative

    def test_recommended_config_large_model(self):
        """Test recommended configuration for large models."""
        config = get_recommended_config(
            model_size="xlarge", precision="bf16", stability_preference="aggressive"
        )

        assert config.initial_scale == 2**20  # 1M for xlarge models
        assert config.growth_factor == 2.5  # BF16 adjustment
        assert config.hysteresis == 1  # Aggressive


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_zero_gradients(self, device):
        """Test handling of zero gradients."""
        config = DynamicScalerConfig()
        scaler = DynamicGradScaler(config=config, device=device)

        model = SimpleModel().to(device)
        # Don't call backward, so gradients are None

        has_overflow = scaler.check_overflow_and_update(model)
        assert has_overflow is False

    def test_very_large_scale(self, device):
        """Test behavior with very large scales."""
        config = DynamicScalerConfig(
            initial_scale=1e10,  # Very large
            max_scale=1e15,
        )
        scaler = DynamicGradScaler(config=config, device=device)

        loss = torch.tensor(1e-10, device=device)  # Very small loss
        scaled_loss = scaler.scale_loss(loss)

        # Should not overflow to infinity
        assert torch.isfinite(scaled_loss).all()

    def test_device_mismatch_handling(self):
        """Test handling of device mismatches."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create scaler on CUDA
        scaler = DynamicGradScaler(device=torch.device("cuda:0"))

        # Create model on CPU
        model = SimpleModel()

        # This should handle device mismatch gracefully without crashing
        result = scaler.check_overflow_and_update(model, force_check=True)
        assert isinstance(result, bool)

    def test_nan_in_config(self):
        """Test handling of NaN values in configuration."""
        with pytest.raises(ValueError):
            DynamicScalerConfig(initial_scale=float("nan"))

    def test_config_extreme_values(self):
        """Test configuration with extreme values."""
        # Very small values
        config = DynamicScalerConfig(
            initial_scale=1e-9,
            min_scale=1e-10,
            growth_interval=1,
            hysteresis=1,
        )
        scaler = DynamicGradScaler(config=config)
        assert (
            abs(scaler.scale.item() - 1e-9) < 1e-12
        )  # Allow for floating point precision

        # Very large growth interval
        config = DynamicScalerConfig(growth_interval=999999)
        assert config.growth_interval == 999999


class TestPerformanceBenchmarks:
    """Performance benchmarks for the dynamic scaler."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.benchmark
    def test_overflow_detection_performance(self, device):
        """Benchmark overflow detection performance."""
        detector = MultiTensorOverflowDetector()

        # Create many tensors of varying sizes
        tensors = []
        for i in range(100):
            size = (i + 1) * 10
            tensors.append(torch.randn(size, device=device))

        # Benchmark detection
        import time

        start_time = time.time()

        for _ in range(10):
            has_overflow, info = detector.detect_overflow(tensors)

        elapsed = time.time() - start_time
        print(
            f"Overflow detection: {elapsed:.4f}s for 10 iterations with "
            f"{len(tensors)} tensors"
        )

        assert elapsed < 1.0  # Should complete in reasonable time

    @pytest.mark.benchmark
    def test_gradient_unscaling_performance(self, device):
        """Benchmark gradient unscaling performance."""
        scaler = DynamicGradScaler(device=device)

        # Create large model
        model = nn.Sequential(*[nn.Linear(1000, 1000) for _ in range(10)])
        model.to(device)

        # Create gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        # Benchmark unscaling
        import time

        start_time = time.time()

        for _ in range(10):
            scaler.unscale_gradients(model)

        elapsed = time.time() - start_time
        print(f"Gradient unscaling: {elapsed:.4f}s for 10 iterations with large model")

        assert elapsed < 2.0  # Should complete in reasonable time


# Test fixtures and setup
@pytest.fixture(scope="session")
def suppress_warnings():
    """Suppress warnings during testing."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=DeprecationWarning)


# Integration test with actual training loop
class TestRealWorldIntegration:
    """Test integration in realistic training scenarios."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_full_training_loop(self, device):
        """Test complete training loop with dynamic scaling."""
        # Setup
        model = SimpleModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        config = MixedPrecisionConfig(
            precision=(
                PrecisionType.FP16 if device.type == "cuda" else PrecisionType.FP32
            ),
            use_dynamic_scaling=True,
            scaler_config=DynamicScalerConfig(
                initial_scale=1024,
                growth_interval=10,
            ),
        )
        mp_manager = MixedPrecisionManager(config, device)

        # Training loop
        losses = []

        for step in range(50):
            optimizer.zero_grad()

            # Generate synthetic data
            x = torch.randn(8, 10, device=device)
            target = torch.randn(8, 1, device=device)

            # Forward pass with autocast
            with mp_manager.autocast_context():
                output = model(x)
                loss = nn.MSELoss()(output, target)

            # Backward pass with scaling
            mp_manager.backward_step(loss)

            # Optimizer step with overflow handling
            success = mp_manager.optimizer_step(optimizer, model)

            if success:
                losses.append(loss.item())

            # Check that training progresses
            if step > 0 and step % 10 == 0:
                recent_loss = sum(losses[-10:]) / len(losses[-10:])
                print(f"Step {step}, Recent avg loss: {recent_loss:.6f}")

        # Verify training progressed
        assert len(losses) > 40  # Most steps should succeed

        # Get final statistics
        stats = mp_manager.get_statistics()
        print(f"Final statistics: {stats}")

        assert stats["total_steps"] == 50
        assert stats["success_rate"] > 0.8  # Most steps should succeed


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
