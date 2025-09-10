"""
Tests for Energy Monitoring Configuration System.

This test suite covers:
- Configuration validation and defaults
- Environment variable handling
- Configuration serialization/deserialization
- Factory methods for different use cases
"""

import os
from unittest.mock import patch

import pytest

from rosellm.rosetrainer.monitoring.config import (
    DistributedConfig,
    EnergyMonitoringConfig,
    EnergyMonitoringMode,
    FallbackStrategy,
    GPUTrackerConfig,
    IntegrationConfig,
)


class TestGPUTrackerConfig:
    """Test GPUTrackerConfig validation and functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GPUTrackerConfig()

        assert config.devices is None
        assert config.sampling_interval == 1.0
        assert config.enable_detailed_metrics is True
        assert config.fallback_power_estimate == 250.0
        assert config.fallback_strategy == FallbackStrategy.ESTIMATE
        assert config.max_consecutive_errors == 10
        assert config.error_recovery_delay == 5.0
        assert config.measurement_buffer_size == 1000
        assert config.enable_background_monitoring is True
        assert config.thread_priority is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GPUTrackerConfig(
            devices=[0, 1],
            sampling_interval=0.5,
            fallback_power_estimate=300.0,
            fallback_strategy=FallbackStrategy.ZERO,
        )

        assert config.devices == [0, 1]
        assert config.sampling_interval == 0.5
        assert config.fallback_power_estimate == 300.0
        assert config.fallback_strategy == FallbackStrategy.ZERO

    def test_validation_positive_sampling_interval(self):
        """Test validation of sampling interval."""
        with pytest.raises(ValueError, match="sampling_interval must be positive"):
            GPUTrackerConfig(sampling_interval=0.0)

        with pytest.raises(ValueError, match="sampling_interval must be positive"):
            GPUTrackerConfig(sampling_interval=-1.0)

    def test_validation_negative_fallback_power(self):
        """Test validation of fallback power estimate."""
        with pytest.raises(
            ValueError, match="fallback_power_estimate cannot be negative"
        ):
            GPUTrackerConfig(fallback_power_estimate=-10.0)

    def test_validation_max_errors(self):
        """Test validation of max consecutive errors."""
        with pytest.raises(ValueError, match="max_consecutive_errors must be positive"):
            GPUTrackerConfig(max_consecutive_errors=0)

        with pytest.raises(ValueError, match="max_consecutive_errors must be positive"):
            GPUTrackerConfig(max_consecutive_errors=-1)

    def test_validation_recovery_delay(self):
        """Test validation of error recovery delay."""
        with pytest.raises(ValueError, match="error_recovery_delay cannot be negative"):
            GPUTrackerConfig(error_recovery_delay=-1.0)

    def test_validation_buffer_size(self):
        """Test validation of measurement buffer size."""
        with pytest.raises(
            ValueError, match="measurement_buffer_size must be positive"
        ):
            GPUTrackerConfig(measurement_buffer_size=0)

    def test_validation_device_list(self):
        """Test validation of device list."""
        # Valid device lists
        GPUTrackerConfig(devices=[0, 1, 2])
        GPUTrackerConfig(devices=[])

        # Invalid device lists
        with pytest.raises(
            ValueError, match="devices must be a list of non-negative integers"
        ):
            GPUTrackerConfig(devices=[-1, 0])

        with pytest.raises(
            ValueError, match="devices must be a list of non-negative integers"
        ):
            GPUTrackerConfig(devices=["0", "1"])  # type: ignore


class TestDistributedConfig:
    """Test DistributedConfig validation and functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DistributedConfig()

        assert config.aggregation_interval == 5.0
        assert config.enable_hierarchical_reporting is True
        assert config.max_history_size == 1000
        assert config.communication_backend == "nccl"
        assert config.timeout_seconds == 30.0
        assert config.enable_fault_tolerance is True
        assert config.max_failed_aggregations == 5
        assert config.use_async_aggregation is False
        assert config.compression_enabled is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DistributedConfig(
            aggregation_interval=2.0,
            communication_backend="gloo",
            enable_hierarchical_reporting=False,
        )

        assert config.aggregation_interval == 2.0
        assert config.communication_backend == "gloo"
        assert config.enable_hierarchical_reporting is False

    def test_validation_aggregation_interval(self):
        """Test validation of aggregation interval."""
        with pytest.raises(ValueError, match="aggregation_interval must be positive"):
            DistributedConfig(aggregation_interval=0.0)

    def test_validation_history_size(self):
        """Test validation of history size."""
        with pytest.raises(ValueError, match="max_history_size must be positive"):
            DistributedConfig(max_history_size=0)

    def test_validation_timeout(self):
        """Test validation of timeout."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            DistributedConfig(timeout_seconds=0.0)

    def test_validation_failed_aggregations(self):
        """Test validation of max failed aggregations."""
        with pytest.raises(
            ValueError, match="max_failed_aggregations must be positive"
        ):
            DistributedConfig(max_failed_aggregations=0)

    def test_validation_unknown_backend(self):
        """Test warning for unknown communication backend."""
        with pytest.warns(UserWarning, match="Unknown communication backend"):
            DistributedConfig(communication_backend="unknown")


class TestIntegrationConfig:
    """Test IntegrationConfig validation and functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IntegrationConfig()

        assert config.integrate_with_trainer is True
        assert config.log_interval == 100
        assert config.save_measurements is True
        assert config.include_in_checkpoints is False
        assert config.checkpoint_energy_summary is True
        assert config.log_level == "INFO"
        assert config.enable_tensorboard is False
        assert config.enable_wandb is False
        assert config.output_format == "json"
        assert config.output_directory is None
        assert config.pause_during_checkpointing is True
        assert config.pause_during_evaluation is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = IntegrationConfig(
            log_interval=50,
            log_level="DEBUG",
            output_format="csv",
            output_directory="/tmp/energy",
        )

        assert config.log_interval == 50
        assert config.log_level == "DEBUG"
        assert config.output_format == "csv"
        assert config.output_directory == "/tmp/energy"

    def test_validation_log_interval(self):
        """Test validation of log interval."""
        with pytest.raises(ValueError, match="log_interval must be positive"):
            IntegrationConfig(log_interval=0)

    def test_validation_log_level(self):
        """Test validation of log level."""
        with pytest.raises(ValueError, match="log_level must be one of"):
            IntegrationConfig(log_level="INVALID")

    def test_validation_output_format(self):
        """Test validation of output format."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            IntegrationConfig(output_format="invalid")

    def test_validation_output_directory(self):
        """Test validation of output directory."""
        with pytest.raises(
            ValueError, match="output_directory must be a string or None"
        ):
            IntegrationConfig(output_directory=123)  # type: ignore


class TestEnergyMonitoringConfig:
    """Test main EnergyMonitoringConfig functionality."""

    def test_default_config(self):
        """Test default configuration."""
        config = EnergyMonitoringConfig()

        assert config.mode == EnergyMonitoringMode.DISTRIBUTED
        assert config.enabled is True
        assert config.auto_start is True
        assert config.auto_stop is True
        assert config.respect_env_vars is True
        assert isinstance(config.gpu_tracker, GPUTrackerConfig)
        assert isinstance(config.distributed, DistributedConfig)
        assert isinstance(config.integration, IntegrationConfig)

    def test_custom_config(self):
        """Test custom configuration."""
        gpu_config = GPUTrackerConfig(sampling_interval=0.5)
        dist_config = DistributedConfig(aggregation_interval=2.0)
        int_config = IntegrationConfig(log_interval=50)

        config = EnergyMonitoringConfig(
            mode=EnergyMonitoringMode.LOCAL_ONLY,
            enabled=False,
            gpu_tracker=gpu_config,
            distributed=dist_config,
            integration=int_config,
        )

        assert config.mode == EnergyMonitoringMode.LOCAL_ONLY
        assert config.enabled is False
        assert config.gpu_tracker.sampling_interval == 0.5
        assert config.distributed.aggregation_interval == 2.0
        assert config.integration.log_interval == 50

    def test_validation_cross_config(self):
        """Test cross-configuration validation warnings."""
        with pytest.warns(
            UserWarning, match="Energy monitoring is enabled but mode is DISABLED"
        ):
            EnergyMonitoringConfig(mode=EnergyMonitoringMode.DISABLED, enabled=True)

    @patch("torch.distributed.is_available", return_value=False)
    def test_validation_distributed_unavailable(self, mock_is_available):
        """Test warning when distributed mode requested but unavailable."""
        with pytest.warns(
            UserWarning,
            match="Distributed mode requested but torch.distributed not available",
        ):
            EnergyMonitoringConfig(mode=EnergyMonitoringMode.DISTRIBUTED)


class TestEnvironmentVariables:
    """Test environment variable handling."""

    def test_env_override_enabled(self):
        """Test environment override for enabled flag."""
        with patch.dict(os.environ, {"ROSE_ENERGY_ENABLED": "false"}):
            config = EnergyMonitoringConfig()
            assert config.enabled is False

        with patch.dict(os.environ, {"ROSE_ENERGY_ENABLED": "true"}):
            config = EnergyMonitoringConfig()
            assert config.enabled is True

        with patch.dict(os.environ, {"ROSE_ENERGY_ENABLED": "1"}):
            config = EnergyMonitoringConfig()
            assert config.enabled is True

    def test_env_override_mode(self):
        """Test environment override for monitoring mode."""
        with patch.dict(os.environ, {"ROSE_ENERGY_MODE": "local"}):
            config = EnergyMonitoringConfig()
            assert config.mode == EnergyMonitoringMode.LOCAL_ONLY

        with patch.dict(os.environ, {"ROSE_ENERGY_MODE": "disabled"}):
            config = EnergyMonitoringConfig()
            assert config.mode == EnergyMonitoringMode.DISABLED

    def test_env_override_sampling_interval(self):
        """Test environment override for sampling interval."""
        with patch.dict(os.environ, {"ROSE_ENERGY_SAMPLING_INTERVAL": "0.5"}):
            config = EnergyMonitoringConfig()
            assert config.gpu_tracker.sampling_interval == 0.5

    def test_env_override_fallback_power(self):
        """Test environment override for fallback power."""
        with patch.dict(os.environ, {"ROSE_ENERGY_FALLBACK_POWER": "300.0"}):
            config = EnergyMonitoringConfig()
            assert config.gpu_tracker.fallback_power_estimate == 300.0

    def test_env_override_devices(self):
        """Test environment override for devices."""
        with patch.dict(os.environ, {"ROSE_ENERGY_DEVICES": "0,1,2"}):
            config = EnergyMonitoringConfig()
            assert config.gpu_tracker.devices == [0, 1, 2]

        with patch.dict(os.environ, {"ROSE_ENERGY_DEVICES": ""}):
            config = EnergyMonitoringConfig()
            # Should handle empty string gracefully

    def test_env_override_aggregation_interval(self):
        """Test environment override for aggregation interval."""
        with patch.dict(os.environ, {"ROSE_ENERGY_AGGREGATION_INTERVAL": "10.0"}):
            config = EnergyMonitoringConfig()
            assert config.distributed.aggregation_interval == 10.0

    def test_env_override_log_interval(self):
        """Test environment override for log interval."""
        with patch.dict(os.environ, {"ROSE_ENERGY_LOG_INTERVAL": "200"}):
            config = EnergyMonitoringConfig()
            assert config.integration.log_interval == 200

    def test_env_override_output_dir(self):
        """Test environment override for output directory."""
        with patch.dict(os.environ, {"ROSE_ENERGY_OUTPUT_DIR": "/tmp/energy"}):
            config = EnergyMonitoringConfig()
            assert config.integration.output_directory == "/tmp/energy"

    def test_env_override_log_level(self):
        """Test environment override for log level."""
        with patch.dict(os.environ, {"ROSE_ENERGY_LOG_LEVEL": "debug"}):
            config = EnergyMonitoringConfig()
            assert config.integration.log_level == "DEBUG"

    def test_env_override_invalid_values(self):
        """Test handling of invalid environment values."""
        with patch.dict(os.environ, {"ROSE_ENERGY_SAMPLING_INTERVAL": "invalid"}):
            with pytest.warns(
                UserWarning, match="Invalid ROSE_ENERGY_SAMPLING_INTERVAL value"
            ):
                config = EnergyMonitoringConfig()
                # Should use default value
                assert config.gpu_tracker.sampling_interval == 1.0

    def test_respect_env_vars_disabled(self):
        """Test disabling environment variable overrides."""
        with patch.dict(os.environ, {"ROSE_ENERGY_ENABLED": "false"}):
            config = EnergyMonitoringConfig(respect_env_vars=False)
            # Should use default value, not environment
            assert config.enabled is True


class TestConfigFactoryMethods:
    """Test configuration factory methods."""

    def test_create_default(self):
        """Test creating default configuration."""
        config = EnergyMonitoringConfig.create_default()

        assert config.mode == EnergyMonitoringMode.DISTRIBUTED
        assert config.enabled is True

    def test_create_for_mode_disabled(self):
        """Test creating configuration for disabled mode."""
        config = EnergyMonitoringConfig.create_for_mode(EnergyMonitoringMode.DISABLED)

        assert config.mode == EnergyMonitoringMode.DISABLED
        assert config.enabled is False

    def test_create_for_mode_local(self):
        """Test creating configuration for local mode."""
        config = EnergyMonitoringConfig.create_for_mode(EnergyMonitoringMode.LOCAL_ONLY)

        assert config.mode == EnergyMonitoringMode.LOCAL_ONLY
        assert config.distributed.enable_hierarchical_reporting is False
        assert config.distributed.aggregation_interval == 10.0

    def test_create_for_mode_hierarchical(self):
        """Test creating configuration for hierarchical mode."""
        config = EnergyMonitoringConfig.create_for_mode(
            EnergyMonitoringMode.HIERARCHICAL
        )

        assert config.mode == EnergyMonitoringMode.HIERARCHICAL
        assert config.distributed.enable_hierarchical_reporting is True
        assert config.distributed.aggregation_interval == 2.0
        assert config.gpu_tracker.enable_detailed_metrics is True

    def test_create_production(self):
        """Test creating production configuration."""
        config = EnergyMonitoringConfig.create_production()

        # Should have conservative settings
        assert config.gpu_tracker.sampling_interval == 2.0
        assert config.gpu_tracker.enable_detailed_metrics is False
        assert config.gpu_tracker.measurement_buffer_size == 500
        assert config.distributed.aggregation_interval == 10.0
        assert config.integration.log_interval == 500
        assert config.integration.save_measurements is False

    def test_create_debug(self):
        """Test creating debug configuration."""
        config = EnergyMonitoringConfig.create_debug()

        # Should have detailed settings
        assert config.gpu_tracker.sampling_interval == 0.5
        assert config.gpu_tracker.enable_detailed_metrics is True
        assert config.distributed.aggregation_interval == 1.0
        assert config.integration.log_level == "DEBUG"
        assert config.integration.log_interval == 10
        assert config.integration.save_measurements is True


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = EnergyMonitoringConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "mode" in config_dict
        assert "enabled" in config_dict
        assert "gpu_tracker" in config_dict
        assert "distributed" in config_dict
        assert "integration" in config_dict

        # Check nested structure
        assert "sampling_interval" in config_dict["gpu_tracker"]
        assert "aggregation_interval" in config_dict["distributed"]
        assert "log_interval" in config_dict["integration"]

    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "mode": "local",
            "enabled": False,
            "gpu_tracker": {
                "sampling_interval": 0.5,
                "fallback_power_estimate": 300.0,
                "fallback_strategy": "zero",
            },
            "distributed": {"aggregation_interval": 2.0},
            "integration": {"log_interval": 50, "log_level": "DEBUG"},
        }

        config = EnergyMonitoringConfig.from_dict(config_dict)

        assert config.mode == EnergyMonitoringMode.LOCAL_ONLY
        assert config.enabled is False
        assert config.gpu_tracker.sampling_interval == 0.5
        assert config.gpu_tracker.fallback_power_estimate == 300.0
        assert config.gpu_tracker.fallback_strategy == FallbackStrategy.ZERO
        assert config.distributed.aggregation_interval == 2.0
        assert config.integration.log_interval == 50
        assert config.integration.log_level == "DEBUG"

    def test_roundtrip_serialization(self):
        """Test roundtrip serialization (to_dict -> from_dict)."""
        original_config = EnergyMonitoringConfig()
        original_config.mode = EnergyMonitoringMode.HIERARCHICAL
        original_config.gpu_tracker.sampling_interval = 0.5
        original_config.distributed.aggregation_interval = 2.0
        original_config.integration.log_interval = 50

        # Convert to dict and back
        config_dict = original_config.to_dict()
        restored_config = EnergyMonitoringConfig.from_dict(config_dict)

        # Should be equivalent
        assert restored_config.mode == original_config.mode
        assert (
            restored_config.gpu_tracker.sampling_interval
            == original_config.gpu_tracker.sampling_interval
        )
        assert (
            restored_config.distributed.aggregation_interval
            == original_config.distributed.aggregation_interval
        )
        assert (
            restored_config.integration.log_interval
            == original_config.integration.log_interval
        )

    def test_clone(self):
        """Test configuration cloning."""
        original_config = EnergyMonitoringConfig()
        original_config.gpu_tracker.sampling_interval = 0.5

        cloned_config = original_config.clone()

        # Should be separate objects
        assert cloned_config is not original_config
        assert cloned_config.gpu_tracker is not original_config.gpu_tracker

        # But with same values
        assert cloned_config.gpu_tracker.sampling_interval == 0.5

        # Modifying one shouldn't affect the other
        cloned_config.gpu_tracker.sampling_interval = 1.0
        assert original_config.gpu_tracker.sampling_interval == 0.5

    def test_string_representation(self):
        """Test string representation of configuration."""
        config = EnergyMonitoringConfig()

        config_str = str(config)
        assert "EnergyMonitoringConfig" in config_str
        assert "distributed" in config_str
        assert "enabled=True" in config_str

        config_repr = repr(config)
        assert isinstance(config_repr, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
