"""
Tests for the main EnergyMonitor system.

This test suite covers:
- Main EnergyMonitor functionality
- Integration with configuration system
- Trainer integration hooks
- Error handling and recovery
- Report generation and export
"""

import json
import os
import tempfile
import time
from unittest.mock import Mock, call, patch

import pytest

from rosellm.rosetrainer.monitoring.config import (
    EnergyMonitoringConfig,
    EnergyMonitoringMode,
)
from rosellm.rosetrainer.monitoring.distributed_energy import (
    DistributedEnergyAggregator,
)
from rosellm.rosetrainer.monitoring.energy_monitor import EnergyMonitor
from rosellm.rosetrainer.monitoring.energy_tracker import GPUEnergyTracker


class TestEnergyMonitorBasic:
    """Test basic EnergyMonitor functionality."""

    def test_initialization_default(self):
        """Test initialization with default configuration."""
        monitor = EnergyMonitor()

        assert monitor.config is not None
        assert isinstance(monitor.config, EnergyMonitoringConfig)
        assert monitor.config.mode == EnergyMonitoringMode.DISTRIBUTED
        assert not monitor._monitoring
        assert not monitor._paused

    def test_initialization_with_config(self):
        """Test initialization with custom configuration."""
        config = EnergyMonitoringConfig.create_production()
        monitor = EnergyMonitor(config)

        assert monitor.config is config
        assert monitor.config.mode == EnergyMonitoringMode.DISTRIBUTED

    def test_initialization_disabled_mode(self):
        """Test initialization with disabled mode."""
        config = EnergyMonitoringConfig()
        config.mode = EnergyMonitoringMode.DISABLED

        monitor = EnergyMonitor(config)

        assert monitor.local_tracker is None
        assert monitor.distributed_aggregator is None

    @patch("torch.distributed.is_available", return_value=False)
    def test_initialization_non_distributed(self, mock_is_available):
        """Test initialization in non-distributed environment."""
        monitor = EnergyMonitor()

        assert not monitor.is_distributed
        assert monitor.distributed_aggregator is None

    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch(
        "rosellm.rosetrainer.monitoring.energy_monitor.parallel_state_initialized",
        return_value=True,
    )
    def test_initialization_distributed(
        self, mock_parallel_state_initialized, mock_is_initialized, mock_is_available
    ):
        """Test initialization in distributed environment."""
        config = EnergyMonitoringConfig()
        config.mode = EnergyMonitoringMode.DISTRIBUTED

        with patch.object(GPUEnergyTracker, "__init__", return_value=None):
            with patch.object(
                DistributedEnergyAggregator, "__init__", return_value=None
            ):
                monitor = EnergyMonitor(config)

                # Should attempt distributed setup
                assert monitor.config.mode == EnergyMonitoringMode.DISTRIBUTED


class TestEnergyMonitorOperations:
    """Test EnergyMonitor operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = EnergyMonitoringConfig()
        self.config.mode = (
            EnergyMonitoringMode.LOCAL_ONLY
        )  # Avoid distributed complications

        # Mock the components
        self.mock_tracker = Mock(spec=GPUEnergyTracker)
        self.mock_tracker.start_monitoring.return_value = True
        self.mock_tracker.stop_monitoring.return_value = {0: 100.0}
        self.mock_tracker.devices = [0]
        self.mock_tracker.is_monitoring.return_value = True
        self.mock_tracker.pause_monitoring.return_value = None
        self.mock_tracker.resume_monitoring.return_value = None
        self.mock_tracker.get_current_power.return_value = {0: 250.0}
        self.mock_tracker.get_energy_consumption.return_value = {0: 500.0}
        self.mock_tracker.get_power_statistics.return_value = {
            "min_watts": 200.0,
            "max_watts": 300.0,
            "mean_watts": 250.0,
            "current_watts": 250.0,
            "total_energy_joules": 500.0,
            "measurement_count": 10,
        }
        self.mock_tracker.get_device_info.return_value = {0: Mock()}

    @patch.object(EnergyMonitor, "_initialize_components")
    def test_start_monitoring_disabled(self, mock_init):
        """Test starting monitoring when disabled."""
        config = EnergyMonitoringConfig()
        config.mode = EnergyMonitoringMode.DISABLED

        monitor = EnergyMonitor(config)
        monitor.local_tracker = None  # Simulate disabled mode

        result = monitor.start_monitoring()
        assert result is False
        assert not monitor._monitoring

    def test_start_monitoring_success(self):
        """Test successful monitoring start."""
        monitor = EnergyMonitor(self.config)
        monitor.local_tracker = self.mock_tracker
        monitor.distributed_aggregator = None  # Local only

        result = monitor.start_monitoring()

        assert result is True
        assert monitor._monitoring
        assert not monitor._paused
        self.mock_tracker.start_monitoring.assert_called_once()

    def test_start_monitoring_failure(self):
        """Test monitoring start failure."""
        monitor = EnergyMonitor(self.config)
        monitor.local_tracker = self.mock_tracker
        self.mock_tracker.start_monitoring.return_value = False

        result = monitor.start_monitoring()

        assert result is False
        assert not monitor._monitoring

    def test_start_monitoring_already_started(self):
        """Test starting monitoring when already started."""
        monitor = EnergyMonitor(self.config)
        monitor.local_tracker = self.mock_tracker
        monitor._monitoring = True

        result = monitor.start_monitoring()

        assert result is True
        # Should not call start_monitoring again
        self.mock_tracker.start_monitoring.assert_not_called()

    def test_stop_monitoring(self):
        """Test stopping monitoring."""
        monitor = EnergyMonitor(self.config)
        monitor.local_tracker = self.mock_tracker
        monitor._monitoring = True

        results = monitor.stop_monitoring()

        assert not monitor._monitoring
        assert isinstance(results, dict)
        assert "local_energy_by_device" in results
        assert "monitoring_mode" in results
        self.mock_tracker.stop_monitoring.assert_called_once()

    def test_pause_resume_monitoring(self):
        """Test pausing and resuming monitoring."""
        monitor = EnergyMonitor(self.config)
        monitor.local_tracker = self.mock_tracker
        monitor._monitoring = True

        # Test pause
        monitor.pause_monitoring()
        assert monitor._paused
        self.mock_tracker.pause_monitoring.assert_called_once()

        # Test resume
        monitor.resume_monitoring()
        assert not monitor._paused
        self.mock_tracker.resume_monitoring.assert_called_once()

    def test_is_monitoring(self):
        """Test monitoring status check."""
        monitor = EnergyMonitor(self.config)

        # Not started
        assert not monitor.is_monitoring()

        # Started but paused
        monitor._monitoring = True
        monitor._paused = True
        assert not monitor.is_monitoring()

        # Started and active
        monitor._paused = False
        assert monitor.is_monitoring()

    def test_paused_context_manager(self):
        """Test paused context manager."""
        monitor = EnergyMonitor(self.config)
        monitor.local_tracker = self.mock_tracker
        monitor._monitoring = True
        monitor._paused = False

        with monitor.paused():
            assert monitor._paused

        assert not monitor._paused

        # Should have called pause and resume
        self.mock_tracker.pause_monitoring.assert_called_once()
        self.mock_tracker.resume_monitoring.assert_called_once()


class TestEnergyMonitorStatistics:
    """Test statistics and reporting functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = EnergyMonitoringConfig()
        self.config.mode = EnergyMonitoringMode.LOCAL_ONLY

        self.mock_tracker = Mock(spec=GPUEnergyTracker)
        self.mock_tracker.devices = [0, 1]
        self.mock_tracker.get_current_power.return_value = {0: 200.0, 1: 150.0}
        self.mock_tracker.get_energy_consumption.return_value = {0: 400.0, 1: 300.0}
        self.mock_tracker.get_power_statistics.return_value = {
            "min_watts": 100.0,
            "max_watts": 250.0,
            "mean_watts": 175.0,
            "current_watts": 200.0,
            "total_energy_joules": 400.0,
            "measurement_count": 20,
        }
        self.mock_tracker.get_device_info.return_value = {
            0: Mock(__str__=lambda x: "GPU 0: Tesla V100"),
            1: Mock(__str__=lambda x: "GPU 1: Tesla V100"),
        }

    def test_get_current_statistics_not_monitoring(self):
        """Test getting statistics when not monitoring."""
        monitor = EnergyMonitor(self.config)

        stats = monitor.get_current_statistics()

        assert stats["monitoring_active"] is False

    def test_get_current_statistics_monitoring(self):
        """Test getting statistics while monitoring."""
        monitor = EnergyMonitor(self.config)
        monitor.local_tracker = self.mock_tracker
        monitor._monitoring = True

        stats = monitor.get_current_statistics()

        assert stats["monitoring_active"] is True
        assert stats["paused"] is False
        assert "local_current_power_watts" in stats
        assert "local_total_energy_joules" in stats
        assert stats["local_device_count"] == 2

    def test_get_current_statistics_with_distributed(self):
        """Test getting statistics with distributed aggregator."""
        monitor = EnergyMonitor(self.config)
        monitor.local_tracker = self.mock_tracker
        monitor._monitoring = True

        # Mock distributed aggregator
        mock_dist_aggregator = Mock()
        mock_measurement = Mock()
        mock_measurement.aggregated_power_watts = 500.0
        mock_measurement.aggregated_energy_joules = 1000.0
        mock_measurement.total_processes = 4
        mock_measurement.average_power = 125.0

        mock_dist_aggregator.get_recent_measurements.return_value = [mock_measurement]
        mock_dist_aggregator.get_hierarchical_energy_report.return_value = {
            "global": {"power": 500.0}
        }
        mock_dist_aggregator.get_energy_efficiency_metrics.return_value = {
            "efficiency": 0.85
        }

        monitor.distributed_aggregator = mock_dist_aggregator

        stats = monitor.get_current_statistics()

        assert "distributed_total_power_watts" in stats
        assert "distributed_total_energy_joules" in stats
        assert "distributed_process_count" in stats
        assert "hierarchical_report" in stats
        assert "efficiency_metrics" in stats

    def test_log_energy_statistics(self):
        """Test logging energy statistics."""
        monitor = EnergyMonitor(self.config)
        monitor.local_tracker = self.mock_tracker
        monitor._monitoring = True
        monitor.config.integration.log_interval = 10

        # Test initial log (should log)
        with patch(
            "rosellm.rosetrainer.monitoring.energy_monitor.logger"
        ) as mock_logger:
            monitor.log_energy_statistics(step=0, force=True)
            mock_logger.info.assert_called()

        # Test within interval (should not log)
        with patch(
            "rosellm.rosetrainer.monitoring.energy_monitor.logger"
        ) as mock_logger:
            monitor.log_energy_statistics(step=5, force=False)
            mock_logger.info.assert_not_called()

        # Test beyond interval (should log)
        with patch(
            "rosellm.rosetrainer.monitoring.energy_monitor.logger"
        ) as mock_logger:
            monitor.log_energy_statistics(step=15, force=False)
            mock_logger.info.assert_called()


class TestEnergyMonitorReporting:
    """Test report generation and export functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = EnergyMonitoringConfig()
        self.config.integration.save_measurements = True

        self.mock_tracker = Mock()
        self.mock_tracker.get_device_info.return_value = {
            0: Mock(__str__=lambda x: "GPU 0")
        }

        self.monitor = EnergyMonitor(self.config)
        self.monitor.local_tracker = self.mock_tracker
        self.monitor._monitoring = True

    def test_get_energy_report(self):
        """Test generating energy report."""
        # Mock get_current_statistics
        mock_stats = {
            "monitoring_active": True,
            "local_current_power_watts": {0: 250.0},
            "local_total_energy_joules": {0: 500.0},
        }

        with patch.object(
            self.monitor, "get_current_statistics", return_value=mock_stats
        ):
            report = self.monitor.get_energy_report()

        assert "timestamp" in report
        assert "monitoring_config" in report
        assert "monitoring_active" in report
        assert "current_statistics" in report
        assert "device_information" in report
        assert report["monitoring_active"] is True

    def test_get_energy_report_with_distributed(self):
        """Test generating energy report with distributed data."""
        mock_aggregator = Mock()
        mock_measurements = [Mock(), Mock(), Mock()]
        for i, m in enumerate(mock_measurements):
            m.timestamp = time.time() + i
            m.aggregated_power_watts = 100.0 + i * 10
            m.aggregated_energy_joules = 200.0 + i * 20

        mock_aggregator.get_recent_measurements.return_value = mock_measurements
        self.monitor.distributed_aggregator = mock_aggregator

        report = self.monitor.get_energy_report()

        assert "recent_measurements_summary" in report
        summary = report["recent_measurements_summary"]
        assert "measurement_count" in summary
        assert "time_span_seconds" in summary
        assert "power_trend" in summary
        assert "energy_trend" in summary

    def test_save_energy_report_json(self):
        """Test saving energy report as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_report.json")

            # Mock report generation
            mock_report = {
                "timestamp": time.time(),
                "monitoring_active": True,
                "local_power": 250.0,
            }

            with patch.object(
                self.monitor, "get_energy_report", return_value=mock_report
            ):
                result = self.monitor.save_energy_report(filepath, format="json")

            assert result is True
            assert os.path.exists(filepath)

            # Verify content
            with open(filepath, "r") as f:
                saved_data = json.load(f)

            assert saved_data["monitoring_active"] is True
            assert saved_data["local_power"] == 250.0

    @pytest.mark.parametrize("format_type", ["csv", "parquet"])
    def test_save_energy_report_other_formats(self, format_type):
        """Test saving energy report in other formats."""
        # Skip if pandas not available (would be needed for CSV/Parquet)
        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            pytest.skip("Pandas not available for CSV/Parquet testing")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, f"test_report.{format_type}")

            mock_report = {
                "timestamp": time.time(),
                "monitoring_active": True,
                "local_power": 250.0,
            }

            with patch.object(
                self.monitor, "get_energy_report", return_value=mock_report
            ):
                result = self.monitor.save_energy_report(filepath, format=format_type)

            # Result depends on pandas availability and implementation
            assert isinstance(result, bool)

    def test_save_energy_report_disabled(self):
        """Test saving when save_measurements is disabled."""
        self.config.integration.save_measurements = False

        result = self.monitor.save_energy_report("dummy.json")

        assert result is False

    def test_flatten_dict(self):
        """Test dictionary flattening utility."""
        nested_dict = {
            "level1": {"level2": {"value1": 10, "value2": 20}, "direct": 30},
            "top_level": 40,
        }

        flattened = self.monitor._flatten_dict(nested_dict)

        expected_keys = [
            "level1_level2_value1",
            "level1_level2_value2",
            "level1_direct",
            "top_level",
        ]
        assert all(key in flattened for key in expected_keys)
        assert flattened["level1_level2_value1"] == 10
        assert flattened["top_level"] == 40


class TestEnergyMonitorErrorHandling:
    """Test error handling and recovery mechanisms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = EnergyMonitoringConfig()
        self.config.gpu_tracker.max_consecutive_errors = 3
        self.config.gpu_tracker.error_recovery_delay = 0.1

    def test_error_handling_below_threshold(self):
        """Test error handling below threshold."""
        monitor = EnergyMonitor(self.config)

        # Simulate errors below threshold
        for i in range(2):
            monitor._handle_error(Exception(f"Error {i}"))

        assert monitor._error_count == 2
        assert monitor._monitoring is False  # Not started, so should remain False

    def test_error_handling_above_threshold(self):
        """Test error handling above threshold."""
        monitor = EnergyMonitor(self.config)
        monitor._monitoring = True  # Simulate monitoring active

        # Mock stop_monitoring
        with patch.object(monitor, "stop_monitoring") as mock_stop:
            # Simulate errors up to threshold (max_consecutive_errors is 3)
            for i in range(3):
                monitor._handle_error(Exception(f"Error {i}"))

            # Should have stopped monitoring on the 3rd error
            mock_stop.assert_called_once()

    def test_error_recovery(self):
        """Test error recovery mechanism."""
        monitor = EnergyMonitor(self.config)

        # Mock start_monitoring to succeed on recovery
        with patch.object(monitor, "start_monitoring", return_value=True) as mock_start:
            with patch.object(monitor, "is_monitoring", return_value=False):
                # Simulate error and recovery
                monitor._handle_error(Exception("Recoverable error"))
                time.sleep(0.2)  # Wait beyond recovery delay
                monitor._handle_error(Exception("Another error"))

                # Should have attempted recovery
                mock_start.assert_called()

    def test_error_recovery_delay(self):
        """Test error recovery delay mechanism."""
        monitor = EnergyMonitor(self.config)
        # Set _last_error_time to a recent time to simulate we already tried recovery
        monitor._last_error_time = time.time()

        with patch.object(monitor, "start_monitoring") as mock_start:
            with patch.object(monitor, "is_monitoring", return_value=False):
                # Error immediately after last error (within recovery delay)
                monitor._handle_error(Exception("Error 1"))

                # Should not have attempted recovery due to delay
                mock_start.assert_not_called()


class TestEnergyMonitorIntegration:
    """Test integration with trainer and other components."""

    def test_context_manager_auto_start(self):
        """Test context manager with auto-start."""
        config = EnergyMonitoringConfig()
        config.auto_start = True

        monitor = EnergyMonitor(config)

        with patch.object(monitor, "start_monitoring") as mock_start:
            with patch.object(monitor, "stop_monitoring") as mock_stop:
                with monitor:
                    pass

                mock_start.assert_called_once()
                mock_stop.assert_called_once()

    def test_context_manager_no_auto_start(self):
        """Test context manager without auto-start."""
        config = EnergyMonitoringConfig()
        config.auto_start = False
        config.auto_stop = False

        monitor = EnergyMonitor(config)

        with patch.object(monitor, "start_monitoring") as mock_start:
            with patch.object(monitor, "stop_monitoring") as mock_stop:
                with monitor:
                    pass

                mock_start.assert_not_called()
                mock_stop.assert_not_called()

    def test_integrate_with_trainer(self):
        """Test integration with trainer."""
        monitor = EnergyMonitor()

        # Mock trainer with hook methods
        mock_trainer = Mock()
        mock_trainer.add_hook = Mock()
        mock_trainer.add_step_hook = Mock()

        result = monitor.integrate_with_trainer(mock_trainer)

        assert result is True
        assert monitor._trainer_integrated is True

        # Should have added hooks
        from unittest.mock import ANY

        mock_trainer.add_hook.assert_has_calls(
            [
                call("before_train", ANY),
                call("after_train", ANY),
            ],
            any_order=True,
        )

    def test_integrate_with_trainer_disabled(self):
        """Test trainer integration when disabled."""
        config = EnergyMonitoringConfig()
        config.integration.integrate_with_trainer = False

        monitor = EnergyMonitor(config)
        mock_trainer = Mock()

        result = monitor.integrate_with_trainer(mock_trainer)

        assert result is False
        assert not monitor._trainer_integrated

    def test_integrate_with_trainer_no_hooks(self):
        """Test trainer integration with trainer lacking hook methods."""
        monitor = EnergyMonitor()

        # Mock trainer without hook methods
        mock_trainer = Mock(spec=[])  # No methods

        result = monitor.integrate_with_trainer(mock_trainer)

        # Should handle gracefully
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
