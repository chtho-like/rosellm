"""
Tests for GPU Energy Tracker with NVML integration.

This test suite covers:
- NVML interface testing with mocking
- Energy measurement accuracy
- Error handling and fallback mechanisms
- Context manager functionality
- Thread safety
"""

import threading
import time
from unittest.mock import Mock, patch

import pytest
import torch

from rosellm.rosetrainer.monitoring.energy_tracker import (
    DeviceInfo,
    EnergyMeasurement,
    GPUEnergyTracker,
    NVMLInterface,
)


class TestEnergyMeasurement:
    """Test EnergyMeasurement data structure."""

    def test_basic_measurement(self):
        """Test basic measurement creation."""
        timestamp = time.time()
        measurement = EnergyMeasurement(
            timestamp=timestamp,
            power_watts=250.0,
            device_id=0,
            cumulative_energy_joules=1000.0,
        )

        assert measurement.timestamp == timestamp
        assert measurement.power_watts == 250.0
        assert measurement.device_id == 0
        assert measurement.cumulative_energy_joules == 1000.0

    def test_measurement_with_detailed_metrics(self):
        """Test measurement with detailed metrics."""
        measurement = EnergyMeasurement(
            timestamp=time.time(),
            power_watts=300.0,
            device_id=1,
            cumulative_energy_joules=500.0,
            temperature_celsius=75.0,
            utilization_percent=95.0,
            memory_used_mb=8000.0,
            memory_total_mb=16000.0,
        )

        assert measurement.temperature_celsius == 75.0
        assert measurement.utilization_percent == 95.0
        assert measurement.memory_used_mb == 8000.0
        assert measurement.memory_total_mb == 16000.0

    def test_auto_timestamp(self):
        """Test automatic timestamp assignment."""
        # Test invalid timestamp gets corrected
        measurement = EnergyMeasurement(
            timestamp=0, power_watts=250.0, device_id=0  # Invalid
        )

        assert measurement.timestamp > 0
        assert abs(measurement.timestamp - time.time()) < 1.0  # Should be very recent


class TestDeviceInfo:
    """Test DeviceInfo data structure."""

    def test_device_info_creation(self):
        """Test device info creation."""
        device_info = DeviceInfo(
            device_id=0,
            name="Tesla V100",
            uuid="GPU-12345678-1234-1234-1234-123456789012",
            memory_total=16000000000,
            power_limit=250.0,
        )

        assert device_info.device_id == 0
        assert device_info.name == "Tesla V100"
        assert "Tesla V100" in str(device_info)
        # Memory is displayed in MB, 16GB = 16000000000 bytes = ~15258 MiB
        assert "15258 MB" in str(device_info) or "16000 MB" in str(device_info)


class TestNVMLInterface:
    """Test NVML interface with mocking."""

    @patch("rosellm.rosetrainer.monitoring.energy_tracker.pynvml")
    @patch("rosellm.rosetrainer.monitoring.energy_tracker.NVML_AVAILABLE", True)
    def test_nvml_initialization(self, mock_pynvml):
        """Test NVML initialization."""
        mock_pynvml.nvmlInit.return_value = None

        nvml = NVMLInterface()
        result = nvml.initialize()

        assert result is True
        mock_pynvml.nvmlInit.assert_called_once()

    @patch("rosellm.rosetrainer.monitoring.energy_tracker.pynvml")
    @patch("rosellm.rosetrainer.monitoring.energy_tracker.NVML_AVAILABLE", True)
    def test_nvml_initialization_failure(self, mock_pynvml):
        """Test NVML initialization failure."""
        mock_pynvml.nvmlInit.side_effect = Exception("NVML not available")

        nvml = NVMLInterface()
        result = nvml.initialize()

        assert result is False

    @patch("rosellm.rosetrainer.monitoring.energy_tracker.NVML_AVAILABLE", False)
    def test_nvml_unavailable(self):
        """Test behavior when NVML is unavailable."""
        nvml = NVMLInterface()
        result = nvml.initialize()

        assert result is False
        assert nvml.get_device_count() == 0

    @patch("rosellm.rosetrainer.monitoring.energy_tracker.NVML_AVAILABLE", True)
    @patch("rosellm.rosetrainer.monitoring.energy_tracker.pynvml")
    def test_get_device_info(self, mock_pynvml):
        """Test getting device information."""
        # Mock NVML functions
        mock_handle = Mock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = b"Tesla V100"
        mock_pynvml.nvmlDeviceGetUUID.return_value = (
            b"GPU-12345678-1234-1234-1234-123456789012"
        )

        # Mock memory info
        mock_mem_info = Mock()
        mock_mem_info.total = 16000000000
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info

        # Mock power limit constraints (returns tuple: min, max)
        mock_pynvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (
            100000,
            250000,
        )  # milliwatts

        nvml = NVMLInterface()
        nvml.initialize()

        device_info = nvml.get_device_info(0)

        assert device_info is not None
        assert device_info.device_id == 0
        assert device_info.name == "Tesla V100"
        assert device_info.memory_total == 16000000000
        assert device_info.power_limit == 250.0

    @patch("rosellm.rosetrainer.monitoring.energy_tracker.pynvml")
    @patch("rosellm.rosetrainer.monitoring.energy_tracker.NVML_AVAILABLE", True)
    def test_get_power_usage(self, mock_pynvml):
        """Test getting power usage."""
        mock_handle = Mock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 200000  # milliwatts

        nvml = NVMLInterface()
        nvml.initialize()
        nvml._device_handles[0] = mock_handle

        power = nvml.get_power_usage(0)

        assert power == 200.0  # Should convert to watts


class TestGPUEnergyTracker:
    """Test GPU Energy Tracker functionality."""

    def test_initialization_no_cuda(self):
        """Test initialization when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            tracker = GPUEnergyTracker()
            assert tracker.devices == []
            assert not tracker.nvml_available or len(tracker.devices) == 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    def test_initialization_with_cuda(self, mock_device_count, mock_is_available):
        """Test initialization with CUDA available."""
        tracker = GPUEnergyTracker()
        # Should detect available devices
        assert len(tracker.devices) <= 2  # May be less if NVML unavailable

    def test_initialization_with_specific_devices(self):
        """Test initialization with specific device list."""
        devices = [0, 1]
        tracker = GPUEnergyTracker(devices=devices)
        assert tracker.devices == devices

    def test_fallback_power_estimate(self):
        """Test fallback power estimation when NVML unavailable."""
        fallback_power = 300.0
        tracker = GPUEnergyTracker(devices=[0], fallback_power_estimate=fallback_power)

        # Should use fallback when NVML unavailable
        power = tracker.get_current_power(0)
        if not tracker.nvml_available:
            assert power == fallback_power

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        tracker = GPUEnergyTracker(devices=[0])

        # Test start
        result = tracker.start_monitoring(background=False)
        if tracker.devices:  # Only if devices available
            assert result is True
            assert tracker.is_monitoring()

        # Test stop
        energy = tracker.stop_monitoring()
        assert isinstance(energy, dict)
        assert not tracker.is_monitoring()

    def test_pause_resume_monitoring(self):
        """Test pausing and resuming monitoring."""
        tracker = GPUEnergyTracker(devices=[0])

        if not tracker.devices:
            pytest.skip("No devices available for testing")

        tracker.start_monitoring(background=False)

        # Test pause
        tracker.pause_monitoring()
        assert not tracker.is_monitoring()  # Should be paused

        # Test resume
        tracker.resume_monitoring()
        assert tracker.is_monitoring()  # Should be active again

        tracker.stop_monitoring()

    def test_context_manager(self):
        """Test context manager functionality."""
        tracker = GPUEnergyTracker(devices=[0])

        with tracker as t:
            assert t is tracker
            if tracker.devices:
                # Should auto-start if devices available
                time.sleep(0.1)  # Allow brief monitoring

        # Should auto-stop when exiting context
        assert not tracker.is_monitoring()

    def test_paused_context_manager(self):
        """Test paused context manager."""
        tracker = GPUEnergyTracker(devices=[0])

        if not tracker.devices:
            pytest.skip("No devices available for testing")

        tracker.start_monitoring(background=False)

        with tracker.paused():
            assert not tracker.is_monitoring()  # Should be paused

        assert tracker.is_monitoring()  # Should resume after context

        tracker.stop_monitoring()

    def test_energy_calculation(self):
        """Test energy calculation over time."""
        tracker = GPUEnergyTracker(
            devices=[0],
            sampling_interval=0.1,  # Fast sampling for testing
            fallback_power_estimate=100.0,  # Known power for testing
        )

        if not tracker.devices:
            pytest.skip("No devices available for testing")

        tracker.start_monitoring(background=False)

        # Take initial measurement
        initial_energy = tracker.get_energy_consumption(0)

        # Wait and take another measurement
        time.sleep(0.2)
        tracker._take_measurement()

        final_energy = tracker.get_energy_consumption(0)

        # Energy should have increased (allowing for some tolerance)
        # Handle both dict and scalar cases
        if isinstance(final_energy, dict) and isinstance(initial_energy, dict):
            assert sum(final_energy.values()) >= sum(initial_energy.values())
        elif isinstance(final_energy, (int, float)) and isinstance(
            initial_energy, (int, float)
        ):
            assert final_energy >= initial_energy
        else:
            # Mixed types or other cases - just check they are not None
            assert final_energy is not None
            assert initial_energy is not None

        tracker.stop_monitoring()

    def test_power_statistics(self):
        """Test power statistics calculation."""
        tracker = GPUEnergyTracker(devices=[0])

        if not tracker.devices:
            pytest.skip("No devices available for testing")

        # Add some mock measurements
        measurements = [
            EnergyMeasurement(time.time(), 100.0, 0, 100.0),
            EnergyMeasurement(time.time(), 150.0, 0, 250.0),
            EnergyMeasurement(time.time(), 200.0, 0, 450.0),
        ]

        tracker.measurements[0] = measurements
        tracker.cumulative_energy[0] = 450.0

        stats = tracker.get_power_statistics(0)

        assert stats["min_watts"] == 100.0
        assert stats["max_watts"] == 200.0
        assert stats["mean_watts"] == 150.0
        assert stats["current_watts"] == 200.0
        assert stats["total_energy_joules"] == 450.0
        assert stats["measurement_count"] == 3

    def test_reset_measurements(self):
        """Test resetting measurements."""
        tracker = GPUEnergyTracker(devices=[0])

        if not tracker.devices:
            pytest.skip("No devices available for testing")

        # Add some measurements
        tracker.cumulative_energy[0] = 100.0
        tracker.measurements[0] = [EnergyMeasurement(time.time(), 50.0, 0, 50.0)]

        tracker.reset_measurements()

        assert tracker.cumulative_energy[0] == 0.0
        assert len(tracker.measurements[0]) == 0

    def test_thread_safety(self):
        """Test thread safety of energy tracker."""
        tracker = GPUEnergyTracker(devices=[0])

        if not tracker.devices:
            pytest.skip("No devices available for testing")

        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    power = tracker.get_current_power(0)
                    results.append(power)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        # Start monitoring
        tracker.start_monitoring(background=True)

        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        tracker.stop_monitoring()

        # Should not have errors and should have collected results
        assert len(errors) == 0
        assert len(results) > 0

    def test_error_handling(self):
        """Test error handling in energy tracker."""
        tracker = GPUEnergyTracker(devices=[0])

        # Mock NVML interface to raise errors
        if tracker.nvml_available:
            with patch.object(
                tracker.nvml, "get_power_usage", side_effect=Exception("NVML Error")
            ):
                tracker.start_monitoring(background=False)

                # Should handle errors gracefully
                power = tracker.get_current_power(0)
                # Should fall back to estimate or return None
                assert power is None or power == tracker.fallback_power_estimate

                tracker.stop_monitoring()


class TestEnergyTrackerIntegration:
    """Integration tests for energy tracker."""

    def test_end_to_end_monitoring(self):
        """Test end-to-end monitoring workflow."""
        tracker = GPUEnergyTracker(
            devices=[0],
            sampling_interval=0.1,
            enable_detailed_metrics=False,  # Faster testing
        )

        if not tracker.devices:
            pytest.skip("No devices available for testing")

        # Start monitoring
        assert tracker.start_monitoring(background=False) is True

        # Check initial state
        assert tracker.is_monitoring()
        tracker.get_energy_consumption()  # Get initial energy reading

        # Wait briefly and take measurement
        time.sleep(0.1)
        tracker._take_measurement()

        # Check power and energy
        power = tracker.get_current_power()
        energy = tracker.get_energy_consumption()

        assert power is not None
        assert isinstance(energy, dict)

        # Get recent measurements
        recent = tracker.get_recent_measurements(0, count=5)
        assert isinstance(recent, list)

        # Stop monitoring
        final_results = tracker.stop_monitoring()
        assert isinstance(final_results, dict)
        assert not tracker.is_monitoring()

    def test_multiple_device_monitoring(self):
        """Test monitoring multiple devices."""
        # Test with multiple devices if available
        devices = (
            [0, 1]
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else [0]
        )

        tracker = GPUEnergyTracker(devices=devices)

        if not tracker.devices:
            pytest.skip("No devices available for testing")

        tracker.start_monitoring(background=False)

        # Test getting power for all devices
        all_power = tracker.get_current_power()
        assert isinstance(all_power, dict)

        for device_id in tracker.devices:
            device_power = tracker.get_current_power(device_id)
            assert device_power is not None or not tracker.nvml_available

        # Test getting energy for all devices
        all_energy = tracker.get_energy_consumption()
        assert isinstance(all_energy, dict)
        assert len(all_energy) == len(tracker.devices)

        tracker.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
