"""
Unit tests for OptimizerParamScheduler.

Tests learning rate scheduling, weight decay scheduling, state persistence,
and various decay styles with comprehensive edge case coverage.
"""

import math

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from rosellm.rosetrainer.scheduler import (
    InvalidSchedulerStateError,
    OptimizerParamScheduler,
    create_scheduler,
)


class TestOptimizerParamScheduler:
    """Test suite for OptimizerParamScheduler."""

    @pytest.fixture
    def simple_model_and_optimizer(self):
        """Create a simple model and optimizer for testing."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters(), lr=1e-3)
        return model, optimizer

    def test_initialization(self, simple_model_and_optimizer):
        """Test scheduler initialization with default parameters."""
        model, optimizer = simple_model_and_optimizer

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
        )

        assert scheduler.init_lr == 1e-5
        assert scheduler.max_lr == 1e-3
        assert scheduler.min_lr == 1e-5
        assert scheduler.lr_warmup_steps == 100
        assert scheduler.lr_decay_steps == 1000
        assert scheduler.num_steps == 0

    def test_linear_warmup(self, simple_model_and_optimizer):
        """Test linear warmup phase."""
        model, optimizer = simple_model_and_optimizer

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=0.0,
            max_lr=1e-3,
            min_lr=0.0,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            lr_decay_style="constant",
        )

        # Test warmup progression
        for step in range(1, 101):
            scheduler.step()
            expected_lr = (1e-3 / 100) * step
            actual_lr = scheduler.get_lr()
            assert (
                abs(actual_lr - expected_lr) < 1e-9
            ), f"Step {step}: {actual_lr} != {expected_lr}"

        # After warmup, should be at max_lr
        assert abs(scheduler.get_lr() - 1e-3) < 1e-9

    def test_linear_decay(self, simple_model_and_optimizer):
        """Test linear decay style."""
        model, optimizer = simple_model_and_optimizer

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=1e-3,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=0,
            lr_decay_steps=100,
            lr_decay_style="linear",
        )

        # Initial LR should be max_lr
        assert abs(scheduler.get_lr() - 1e-3) < 1e-9

        # Test linear decay
        scheduler.step(50)
        expected_lr = 1e-3 - 0.5 * (1e-3 - 1e-5)
        assert abs(scheduler.get_lr() - expected_lr) < 1e-9

        # At end of decay
        scheduler.step(50)
        assert abs(scheduler.get_lr() - 1e-5) < 1e-9

        # After decay, should stay at min_lr
        scheduler.step(100)
        assert abs(scheduler.get_lr() - 1e-5) < 1e-9

    def test_cosine_decay(self, simple_model_and_optimizer):
        """Test cosine decay style."""
        model, optimizer = simple_model_and_optimizer

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=1e-3,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=0,
            lr_decay_steps=100,
            lr_decay_style="cosine",
        )

        # Test cosine decay at specific points
        scheduler.step(25)  # 1/4 through decay
        decay_ratio = 0.25
        expected_lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (
            math.cos(math.pi * decay_ratio) + 1.0
        )
        assert abs(scheduler.get_lr() - expected_lr) < 1e-9

        scheduler.num_steps = 50  # Reset to halfway
        decay_ratio = 0.5
        expected_lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (
            math.cos(math.pi * decay_ratio) + 1.0
        )
        assert abs(scheduler.get_lr() - expected_lr) < 1e-9

    def test_inverse_square_root_decay(self, simple_model_and_optimizer):
        """Test inverse square root decay style."""
        model, optimizer = simple_model_and_optimizer

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            lr_decay_style="inverse-square-root",
        )

        # After warmup
        scheduler.step(100)

        # Test inverse square root decay
        scheduler.step(100)  # Step to 200
        expected_lr = 1e-3 * (100**0.5) / (200**0.5)
        expected_lr = max(1e-5, expected_lr)
        assert abs(scheduler.get_lr() - expected_lr) < 1e-9

    def test_constant_decay(self, simple_model_and_optimizer):
        """Test constant learning rate (no decay)."""
        model, optimizer = simple_model_and_optimizer

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=1e-3,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=0,
            lr_decay_steps=1000,
            lr_decay_style="constant",
        )

        # Should stay constant at max_lr
        for step in [10, 100, 500, 1000, 2000]:
            scheduler.num_steps = step
            assert abs(scheduler.get_lr() - 1e-3) < 1e-9

    def test_wsd_decay(self, simple_model_and_optimizer):
        """Test Warmup-Stable-Decay (WSD) style."""
        model, optimizer = simple_model_and_optimizer

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=0.0,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            lr_decay_style="WSD",
            wsd_decay_steps=200,
            lr_wsd_decay_style="linear",
        )

        # Warmup phase (0-100)
        scheduler.step(50)
        assert scheduler.get_lr() < 1e-3  # Still warming up

        # Stable phase (100-800)
        scheduler.num_steps = 500
        assert abs(scheduler.get_lr() - 1e-3) < 1e-9  # Should be at max_lr

        # Decay phase (800-1000)
        scheduler.num_steps = 900
        assert scheduler.get_lr() < 1e-3  # Should be decaying
        assert scheduler.get_lr() > 1e-5  # But not at minimum yet

        # End of decay
        scheduler.num_steps = 1000
        assert abs(scheduler.get_lr() - 1e-5) < 1e-9

    def test_weight_decay_scheduling(self, simple_model_and_optimizer):
        """Test weight decay scheduling."""
        model, optimizer = simple_model_and_optimizer

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            start_wd=0.01,
            end_wd=0.1,
            wd_incr_steps=100,
            wd_incr_style="linear",
        )

        # Initial weight decay
        assert abs(scheduler.get_wd() - 0.01) < 1e-9

        # Halfway through increment
        scheduler.step(50)
        expected_wd = 0.01 + 0.5 * (0.1 - 0.01)
        assert abs(scheduler.get_wd() - expected_wd) < 1e-9

        # End of increment
        scheduler.step(50)
        assert abs(scheduler.get_wd() - 0.1) < 1e-9

        # After increment period
        scheduler.step(100)
        assert abs(scheduler.get_wd() - 0.1) < 1e-9

    def test_per_param_group_lr(self, simple_model_and_optimizer):
        """Test per-parameter group learning rates."""
        model, optimizer = simple_model_and_optimizer

        # Add custom max_lr to param group
        optimizer.param_groups[0]["max_lr"] = 5e-3
        optimizer.param_groups[0]["min_lr"] = 5e-5

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=0,
            lr_decay_steps=100,
            lr_decay_style="linear",
        )

        # Should use param group's max_lr
        param_group = optimizer.param_groups[0]
        lr = scheduler.get_lr(param_group)
        assert abs(lr - 5e-3) < 1e-9

        # After full decay, should use param group's min_lr
        scheduler.step(100)
        lr = scheduler.get_lr(param_group)
        assert abs(lr - 5e-5) < 1e-9

    def test_lr_multiplier(self, simple_model_and_optimizer):
        """Test learning rate multiplier functionality."""
        model, optimizer = simple_model_and_optimizer

        # Add lr_mult to param group
        optimizer.param_groups[0]["lr_mult"] = 0.1

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            max_lr=1e-3,
            min_lr=1e-3,  # Set min_lr = max_lr to avoid decay
            lr_warmup_steps=0,
            lr_decay_steps=100,
            lr_decay_style="constant",  # Use constant to avoid decay
        )

        scheduler.step()

        # Check that lr_mult is applied
        actual_lr = optimizer.param_groups[0]["lr"]
        expected_lr = 1e-3 * 0.1
        assert abs(actual_lr - expected_lr) < 1e-9

    def test_state_dict_save_load(self, simple_model_and_optimizer):
        """Test saving and loading scheduler state."""
        model, optimizer = simple_model_and_optimizer

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            lr_decay_style="cosine",
        )

        # Step to a specific state
        scheduler.step(250)

        # Save state
        state_dict = scheduler.state_dict()
        saved_lr = scheduler.get_lr()
        saved_wd = scheduler.get_wd()

        # Create new scheduler
        new_optimizer = Adam(model.parameters(), lr=1e-3)
        new_scheduler = OptimizerParamScheduler(
            optimizer=new_optimizer,
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            lr_decay_style="cosine",
        )

        # Load state
        new_scheduler.load_state_dict(state_dict)

        # Verify state is restored
        assert new_scheduler.num_steps == 250
        assert abs(new_scheduler.get_lr() - saved_lr) < 1e-9
        assert abs(new_scheduler.get_wd() - saved_wd) < 1e-9

    def test_get_last_lr(self, simple_model_and_optimizer):
        """Test get_last_lr method."""
        model, optimizer = simple_model_and_optimizer

        # Create a new parameter and add it as a separate param group
        extra_param = nn.Parameter(torch.randn(5, 5))
        optimizer.add_param_group({"params": [extra_param]})

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            max_lr=1e-3,
            min_lr=1e-3,  # Set min_lr = max_lr to avoid decay
            lr_warmup_steps=0,
            lr_decay_steps=100,
            lr_decay_style="constant",  # Use constant to avoid decay
        )

        scheduler.step()

        last_lrs = scheduler.get_last_lr()
        assert len(last_lrs) == 2  # Two param groups
        assert all(abs(lr - 1e-3) < 1e-9 for lr in last_lrs)

    def test_invalid_configurations(self, simple_model_and_optimizer):
        """Test that invalid configurations raise appropriate errors."""
        model, optimizer = simple_model_and_optimizer

        # min_lr > max_lr
        with pytest.raises(ValueError, match="max_lr.*must be >= min_lr"):
            OptimizerParamScheduler(
                optimizer=optimizer,
                max_lr=1e-5,
                min_lr=1e-3,
            )

        # init_lr > max_lr
        with pytest.raises(ValueError, match="init_lr.*must be <= max_lr"):
            OptimizerParamScheduler(
                optimizer=optimizer,
                init_lr=1e-2,
                max_lr=1e-3,
            )

        # warmup_steps >= decay_steps
        with pytest.raises(ValueError, match="lr_warmup_steps.*must be less than"):
            OptimizerParamScheduler(
                optimizer=optimizer,
                lr_warmup_steps=100,
                lr_decay_steps=100,
            )

        # WSD without wsd_decay_steps
        with pytest.raises(ValueError, match="wsd_decay_steps is required"):
            OptimizerParamScheduler(
                optimizer=optimizer,
                lr_decay_style="WSD",
            )

    def test_integration_with_training_loop(self, simple_model_and_optimizer):
        """Test scheduler in a mock training loop."""
        model, optimizer = simple_model_and_optimizer

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=10,
            lr_decay_steps=50,
            lr_decay_style="cosine",
        )

        # Mock training loop
        losses = []
        lrs = []

        for step in range(60):
            # Mock forward pass
            x = torch.randn(32, 10)
            y = model(x)
            loss = y.mean()

            # Mock backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Record metrics
            losses.append(loss.item())
            lrs.append(scheduler.get_lr())

        # Verify LR schedule was applied
        assert lrs[0] < lrs[10]  # Warmup occurred
        assert lrs[30] > lrs[50]  # Decay occurred
        assert abs(lrs[-1] - 1e-5) < 1e-9  # Reached min_lr


class TestSchedulerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_warmup_steps(self):
        """Test scheduler with no warmup."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=1e-3,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=0,
            lr_decay_steps=100,
        )

        # Should start at max_lr immediately
        assert abs(scheduler.get_lr() - 1e-3) < 1e-9

    def test_single_step_decay(self):
        """Test scheduler with single-step decay."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=0,
            lr_decay_steps=1,
            lr_decay_style="linear",
        )

        # After one step, should be at min_lr
        scheduler.step()
        assert abs(scheduler.get_lr() - 1e-5) < 1e-9

    def test_large_step_increment(self):
        """Test stepping by large increments."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
        )

        # Step by 500 at once
        scheduler.step(500)
        assert scheduler.num_steps == 500

        # Verify LR is calculated correctly
        lr = scheduler.get_lr()
        assert lr < 1e-3  # Should be decaying
        assert lr > 1e-5  # But not at minimum

    def test_negative_step_increment(self):
        """Test that negative step increment raises error."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        scheduler = OptimizerParamScheduler(optimizer=optimizer)

        with pytest.raises(ValueError, match="Step increment must be non-negative"):
            scheduler.step(-1)

    def test_cache_clearing(self):
        """Test that caches are cleared appropriately."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=10,
            lr_decay_steps=1000,
        )

        # Step to a non-zero position first
        scheduler.step(5)

        # Access LR to populate cache
        scheduler.get_lr()
        assert len(scheduler._lr_cache) > 0

        # Step with increment > 0 should clear cache before recalculating
        old_num_steps = scheduler.num_steps
        scheduler.step(1)
        assert scheduler.num_steps == old_num_steps + 1
        # Cache gets cleared and repopulated during step

        # Verify cache behavior - after step(0), cache should remain cleared
        # since increment=0 doesn't trigger cache clear but also doesn't recalc
        scheduler.get_lr()
        cache_size_before = len(scheduler._lr_cache)
        scheduler.step(0)  # No increment
        # Cache should still have same size since no clear happened
        assert len(scheduler._lr_cache) == cache_size_before

        # Test that positive increment clears cache
        scheduler.step(1)
        # Now accessing LR will repopulate with new step value
        scheduler.get_lr()
        assert scheduler.num_steps == old_num_steps + 2

    def test_invalid_weight_decay_config(self):
        """Test invalid weight decay configurations."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        # Negative start_wd
        with pytest.raises(ValueError, match="start_wd must be non-negative"):
            OptimizerParamScheduler(
                optimizer=optimizer,
                start_wd=-0.1,
            )

        # end_wd < start_wd
        with pytest.raises(ValueError, match="end_wd.*must be >= start_wd"):
            OptimizerParamScheduler(
                optimizer=optimizer,
                start_wd=0.1,
                end_wd=0.01,
            )

    def test_wsd_validation(self):
        """Test WSD-specific validation."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        # WSD without lr_wsd_decay_style
        with pytest.raises(ValueError, match="lr_wsd_decay_style is required"):
            OptimizerParamScheduler(
                optimizer=optimizer,
                lr_decay_style="WSD",
                wsd_decay_steps=100,
            )

        # wsd_decay_steps > lr_decay_steps
        with pytest.raises(ValueError, match="wsd_decay_steps.*must be <="):
            OptimizerParamScheduler(
                optimizer=optimizer,
                lr_decay_style="WSD",
                wsd_decay_steps=2000,
                lr_decay_steps=1000,
                lr_wsd_decay_style="linear",
            )


class TestSchedulerFactory:
    """Test the create_scheduler factory function."""

    def test_default_preset(self):
        """Test default scheduler preset."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        config = {
            "max_lr": 1e-3,
            "min_lr": 1e-5,
            "lr_decay_steps": 1000,
        }

        scheduler = create_scheduler(optimizer, config, "default")
        assert scheduler.lr_decay_style == "linear"
        assert scheduler.max_lr == 1e-3

    def test_warmup_cosine_preset(self):
        """Test warmup_cosine preset."""
        model = nn.Linear(10, 10)
        optimizer = SGD(model.parameters(), lr=0.1)

        config = {
            "max_lr": 0.1,
            "min_lr": 1e-4,
            "lr_warmup_steps": 100,
            "lr_decay_steps": 1000,
        }

        scheduler = create_scheduler(optimizer, config, "warmup_cosine")
        assert scheduler.lr_decay_style == "cosine"
        assert scheduler.init_lr == 0.0

    def test_inverse_sqrt_preset(self):
        """Test inverse square root preset."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        config = {
            "max_lr": 1e-3,
            "min_lr": 1e-6,
            "lr_warmup_steps": 100,
            "lr_decay_steps": 10000,
        }

        scheduler = create_scheduler(optimizer, config, "inverse_sqrt")
        assert scheduler.lr_decay_style == "inverse-square-root"
        assert scheduler.init_lr == 0.0

    def test_wsd_preset(self):
        """Test warmup_stable_decay preset."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        config = {
            "max_lr": 1e-3,
            "min_lr": 1e-5,
            "lr_warmup_steps": 100,
            "lr_decay_steps": 1000,
            "wsd_decay_steps": 200,
        }

        scheduler = create_scheduler(optimizer, config, "warmup_stable_decay")
        assert scheduler.lr_decay_style == "WSD"
        assert scheduler.wsd_decay_steps == 200
        assert scheduler.lr_wsd_decay_style == "cosine"

    def test_invalid_preset(self):
        """Test that invalid preset raises error."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        config = {"max_lr": 1e-3}

        with pytest.raises(ValueError, match="Unknown scheduler_type"):
            create_scheduler(optimizer, config, "invalid_preset")

    def test_config_override(self):
        """Test that explicit config values override preset."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        config = {
            "max_lr": 1e-3,
            "lr_decay_style": "linear",  # Override cosine preset
            "lr_decay_steps": 2000,
        }

        scheduler = create_scheduler(optimizer, config, "warmup_cosine")
        # Should use linear from config, not cosine from preset
        assert scheduler.lr_decay_style == "linear"
        assert scheduler.lr_decay_steps == 2000


class TestSchedulerStateErrors:
    """Test error handling for invalid scheduler states."""

    def test_missing_num_steps_in_state(self):
        """Test loading state dict without num_steps."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())
        scheduler = OptimizerParamScheduler(optimizer=optimizer)

        invalid_state = {"max_lr": 1e-3}  # Missing num_steps

        with pytest.raises(
            InvalidSchedulerStateError, match="missing required 'num_steps'"
        ):
            scheduler.load_state_dict(invalid_state)

    def test_critical_config_mismatch(self):
        """Test critical config mismatch in state loading."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            lr_decay_style="linear",
            lr_decay_steps=1000,
        )

        state = scheduler.state_dict()
        state["lr_decay_style"] = "cosine"  # Critical mismatch

        with pytest.raises(
            InvalidSchedulerStateError, match="Critical scheduler config mismatch"
        ):
            scheduler.load_state_dict(state)

    def test_non_critical_config_warning(self, caplog):
        """Test that non-critical mismatches only warn."""
        model = nn.Linear(10, 10)
        optimizer = Adam(model.parameters())

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            max_lr=1e-3,
            min_lr=1e-5,
        )

        state = scheduler.state_dict()
        state["max_lr"] = 2e-3  # Non-critical mismatch

        scheduler.load_state_dict(state)

        # Check that warning was logged
        assert "Non-critical scheduler config differences" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
