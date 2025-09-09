"""
NVML-based GPU Energy Tracking with Graceful Degradation

This module provides GPU energy monitoring using NVIDIA Management Library (NVML).
It includes comprehensive error handling and fallback mechanisms to ensure robustness
in production environments.

Key Features:
- NVML-based power consumption tracking
- Graceful degradation when NVML is unavailable
- Context manager support for pause/resume
- Thread-safe operation
- Comprehensive error handling
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class EnergyMeasurement:
    """Energy measurement data structure."""

    timestamp: float
    power_watts: float
    device_id: int
    cumulative_energy_joules: float = 0.0
    temperature_celsius: Optional[float] = None
    utilization_percent: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None

    def __post_init__(self):
        """Ensure timestamp is valid."""
        if self.timestamp <= 0:
            self.timestamp = time.time()


@dataclass
class DeviceInfo:
    """GPU device information."""

    device_id: int
    name: str
    uuid: str
    memory_total: int
    power_limit: float
    nvml_handle: Optional[object] = None

    def __str__(self) -> str:
        return (
            f"GPU {self.device_id}: {self.name} ({self.memory_total // (1024**2)} MB)"
        )


class NVMLInterface:
    """Thread-safe NVML interface with error handling."""

    def __init__(self) -> None:
        self._initialized = False
        self._lock = threading.Lock()
        self._device_handles: Dict[int, object] = {}
        self._device_info: Dict[int, DeviceInfo] = {}

    def initialize(self) -> bool:
        """Initialize NVML interface."""
        with self._lock:
            if self._initialized:
                return True

            if not NVML_AVAILABLE or pynvml is None:
                logger.warning(
                    "NVML not available - energy tracking will use fallback mode"
                )
                return False

            try:
                pynvml.nvmlInit()
                self._initialized = True
                logger.info("NVML initialized successfully")
                return True
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                return False

    def shutdown(self):
        """Shutdown NVML interface."""
        with self._lock:
            if self._initialized and pynvml is not None:
                try:
                    pynvml.nvmlShutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down NVML: {e}")
                finally:
                    self._initialized = False
                    self._device_handles.clear()
                    self._device_info.clear()

    def get_device_count(self) -> int:
        """Get number of GPU devices."""
        if not self._initialized or pynvml is None:
            return 0

        try:
            count = pynvml.nvmlDeviceGetCount()
            return int(count)
        except Exception as e:
            logger.warning(f"Failed to get device count: {e}")
            return 0

    def get_device_info(self, device_id: int) -> Optional[DeviceInfo]:
        """Get device information."""
        if device_id in self._device_info:
            return self._device_info[device_id]

        if not self._initialized or pynvml is None:
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            uuid = pynvml.nvmlDeviceGetUUID(handle).decode("utf-8")

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = int(mem_info.total)

            # Get power limit
            try:
                power_limit = (
                    pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1]
                    / 1000.0
                )
            except:
                # Try alternative power limit method with type guard
                try:
                    power_limit = (
                        pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000.0
                    )
                except:
                    # Fall back to a reasonable default
                    power_limit = 250.0

            device_info = DeviceInfo(
                device_id=device_id,
                name=name,
                uuid=uuid,
                memory_total=memory_total,
                power_limit=power_limit,
                nvml_handle=handle,
            )

            self._device_handles[device_id] = handle
            self._device_info[device_id] = device_info
            return device_info

        except Exception as e:
            logger.warning(f"Failed to get info for device {device_id}: {e}")
            return None

    def get_power_usage(self, device_id: int) -> Optional[float]:
        """Get current power usage in watts."""
        if device_id not in self._device_handles:
            if not self.get_device_info(device_id):
                return None

        if pynvml is None:
            return None

        try:
            handle = self._device_handles[device_id]
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            return float(power_mw) / 1000.0  # Convert to watts
        except Exception as e:
            logger.debug(f"Failed to get power usage for device {device_id}: {e}")
            return None

    def get_temperature(self, device_id: int) -> Optional[float]:
        """Get GPU temperature in Celsius."""
        if device_id not in self._device_handles:
            if not self.get_device_info(device_id):
                return None

        if pynvml is None:
            return None

        try:
            handle = self._device_handles[device_id]
            return float(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except Exception as e:
            logger.debug(f"Failed to get temperature for device {device_id}: {e}")
            return None

    def get_utilization(self, device_id: int) -> Optional[float]:
        """Get GPU utilization percentage."""
        if device_id not in self._device_handles:
            if not self.get_device_info(device_id):
                return None

        if pynvml is None:
            return None

        try:
            handle = self._device_handles[device_id]
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception as e:
            logger.debug(f"Failed to get utilization for device {device_id}: {e}")
            return None

    def get_memory_info(self, device_id: int) -> Optional[Tuple[float, float]]:
        """Get memory usage (used_mb, total_mb)."""
        if device_id not in self._device_handles:
            if not self.get_device_info(device_id):
                return None

        if pynvml is None:
            return None

        try:
            handle = self._device_handles[device_id]
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = float(mem_info.used) / (1024**2)
            total_mb = float(mem_info.total) / (1024**2)
            return used_mb, total_mb
        except Exception as e:
            logger.debug(f"Failed to get memory info for device {device_id}: {e}")
            return None


class GPUEnergyTracker:
    """
    GPU Energy Tracker with NVML integration and fallback mechanisms.

    This class provides robust GPU energy monitoring with graceful degradation
    when hardware monitoring is unavailable.
    """

    def __init__(
        self,
        devices: Optional[List[int]] = None,
        sampling_interval: float = 1.0,
        enable_detailed_metrics: bool = True,
        fallback_power_estimate: float = 250.0,
    ) -> None:
        """
        Initialize GPU Energy Tracker.

        Args:
            devices: List of GPU device IDs to monitor. If None, auto-detect.
            sampling_interval: Sampling interval in seconds
            enable_detailed_metrics: Collect detailed metrics (temp, util, memory)
            fallback_power_estimate: Power estimate to use when NVML unavailable (watts)
        """
        self.sampling_interval = sampling_interval
        self.enable_detailed_metrics = enable_detailed_metrics
        self.fallback_power_estimate = fallback_power_estimate

        # Initialize NVML interface
        self.nvml = NVMLInterface()
        self.nvml_available = self.nvml.initialize()

        # Device management
        self.devices = self._initialize_devices(devices)
        self.device_info: Dict[int, DeviceInfo] = {}

        # Energy tracking state
        self.measurements: Dict[int, List[EnergyMeasurement]] = {
            device: [] for device in self.devices
        }
        self.cumulative_energy: Dict[int, float] = {
            device: 0.0 for device in self.devices
        }
        self.last_measurement_time: Dict[int, float] = {}

        # Thread safety
        self._lock = threading.RLock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._paused = False

        # Initialize device info
        self._collect_device_info()

        logger.info(f"GPUEnergyTracker initialized for devices: {self.devices}")
        if not self.nvml_available:
            logger.warning(
                f"NVML unavailable - using fallback power estimate: "
                f"{fallback_power_estimate}W"
            )

    def _initialize_devices(self, devices: Optional[List[int]]) -> List[int]:
        """Initialize device list."""
        if devices is not None:
            return list(devices)

        # Auto-detect devices
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        else:
            logger.warning("CUDA not available - no devices to monitor")
            return []

    def _collect_device_info(self):
        """Collect information about monitored devices."""
        for device_id in self.devices:
            if self.nvml_available:
                info = self.nvml.get_device_info(device_id)
                if info:
                    self.device_info[device_id] = info
                else:
                    # Create fallback device info
                    self.device_info[device_id] = DeviceInfo(
                        device_id=device_id,
                        name=f"GPU_{device_id}",
                        uuid=f"fallback_{device_id}",
                        memory_total=0,
                        power_limit=self.fallback_power_estimate,
                    )
            else:
                # Create fallback device info
                self.device_info[device_id] = DeviceInfo(
                    device_id=device_id,
                    name=f"GPU_{device_id}",
                    uuid=f"fallback_{device_id}",
                    memory_total=0,
                    power_limit=self.fallback_power_estimate,
                )

    def get_device_info(
        self, device_id: Optional[int] = None
    ) -> Union[DeviceInfo, Dict[int, DeviceInfo], None]:
        """Get device information."""
        if device_id is not None:
            return self.device_info.get(device_id)
        return self.device_info.copy()

    def start_monitoring(self, background: bool = True) -> bool:
        """
        Start energy monitoring.

        Args:
            background: Whether to run monitoring in background thread

        Returns:
            True if monitoring started successfully
        """
        with self._lock:
            if self._monitoring:
                logger.warning("Monitoring already started")
                return True

            if not self.devices:
                logger.error("No devices to monitor")
                return False

            self._monitoring = True
            self._paused = False

            if background:
                self._monitor_thread = threading.Thread(
                    target=self._monitoring_loop, name="GPUEnergyMonitor", daemon=True
                )
                self._monitor_thread.start()
                logger.info("Background energy monitoring started")
            else:
                logger.info("Manual energy monitoring mode enabled")

            return True

    def stop_monitoring(self) -> Dict[int, float]:
        """
        Stop energy monitoring and return total energy consumption.

        Returns:
            Dictionary mapping device_id to total energy (joules)
        """
        with self._lock:
            if not self._monitoring:
                return self.cumulative_energy.copy()

            self._monitoring = False

            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
                if self._monitor_thread.is_alive():
                    logger.warning("Monitor thread did not terminate cleanly")

            # Take final measurement
            self._take_measurement()

            logger.info(
                f"Energy monitoring stopped. Total energy: {self.cumulative_energy}"
            )
            return self.cumulative_energy.copy()

    def pause_monitoring(self):
        """Pause energy monitoring without stopping."""
        with self._lock:
            if self._monitoring and not self._paused:
                self._paused = True
                # Take measurement at pause point
                self._take_measurement()
                logger.debug("Energy monitoring paused")

    def resume_monitoring(self):
        """Resume energy monitoring."""
        with self._lock:
            if self._monitoring and self._paused:
                self._paused = False
                # Reset timing for accurate energy calculation
                current_time = time.time()
                for device_id in self.devices:
                    self.last_measurement_time[device_id] = current_time
                logger.debug("Energy monitoring resumed")

    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._monitoring and not self._paused

    @contextmanager
    def paused(self):
        """Context manager for temporarily pausing monitoring."""
        was_monitoring = self.is_monitoring()
        if was_monitoring:
            self.pause_monitoring()
        try:
            yield
        finally:
            if was_monitoring:
                self.resume_monitoring()

    def _monitoring_loop(self):
        """Background monitoring loop."""
        logger.debug("Energy monitoring loop started")

        while self._monitoring:
            try:
                if not self._paused:
                    self._take_measurement()

                time.sleep(self.sampling_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                # Continue monitoring despite errors
                time.sleep(self.sampling_interval)

        logger.debug("Energy monitoring loop stopped")

    def _take_measurement(self):
        """Take energy measurement from all devices."""
        current_time = time.time()

        for device_id in self.devices:
            try:
                measurement = self._measure_device(device_id, current_time)
                if measurement:
                    self.measurements[device_id].append(measurement)
                    self._update_cumulative_energy(device_id, measurement, current_time)
            except Exception as e:
                logger.debug(f"Failed to measure device {device_id}: {e}")

    def _measure_device(
        self, device_id: int, timestamp: float
    ) -> Optional[EnergyMeasurement]:
        """Measure energy consumption for a single device."""
        # Get power consumption
        if self.nvml_available:
            power = self.nvml.get_power_usage(device_id)
            if power is None:
                power = self.fallback_power_estimate
        else:
            power = self.fallback_power_estimate

        # Create measurement
        measurement = EnergyMeasurement(
            timestamp=timestamp,
            power_watts=power,
            device_id=device_id,
            cumulative_energy_joules=self.cumulative_energy[device_id],
        )

        # Add detailed metrics if enabled
        if self.enable_detailed_metrics and self.nvml_available:
            measurement.temperature_celsius = self.nvml.get_temperature(device_id)
            measurement.utilization_percent = self.nvml.get_utilization(device_id)

            mem_info = self.nvml.get_memory_info(device_id)
            if mem_info:
                measurement.memory_used_mb, measurement.memory_total_mb = mem_info

        return measurement

    def _update_cumulative_energy(
        self, device_id: int, measurement: EnergyMeasurement, current_time: float
    ):
        """Update cumulative energy consumption."""
        last_time = self.last_measurement_time.get(device_id)

        if last_time is not None:
            time_delta = current_time - last_time
            # Energy = Power * Time (J = W * s)
            energy_delta = measurement.power_watts * time_delta
            self.cumulative_energy[device_id] += energy_delta
            measurement.cumulative_energy_joules = self.cumulative_energy[device_id]

        self.last_measurement_time[device_id] = current_time

    def get_current_power(
        self, device_id: Optional[int] = None
    ) -> Union[float, Dict[int, float], None]:
        """
        Get current power consumption.

        Args:
            device_id: Specific device ID, or None for all devices

        Returns:
            Power consumption in watts
        """
        if device_id is not None:
            if device_id not in self.devices:
                return None

            if self.nvml_available:
                power = self.nvml.get_power_usage(device_id)
                return power if power is not None else self.fallback_power_estimate
            else:
                return self.fallback_power_estimate
        else:
            # Return all devices
            result: Dict[int, float] = {}
            for dev_id in self.devices:
                device_power: Union[
                    float, Dict[int, float], None
                ] = self.get_current_power(dev_id)
                if device_power is not None and isinstance(device_power, (int, float)):
                    result[dev_id] = float(device_power)
            return result

    def get_energy_consumption(
        self, device_id: Optional[int] = None
    ) -> Union[float, Dict[int, float]]:
        """
        Get cumulative energy consumption.

        Args:
            device_id: Specific device ID, or None for all devices

        Returns:
            Energy consumption in joules
        """
        if device_id is not None:
            return self.cumulative_energy.get(device_id, 0.0)
        else:
            return self.cumulative_energy.copy()

    def get_recent_measurements(
        self, device_id: int, count: int = 10
    ) -> List[EnergyMeasurement]:
        """Get recent energy measurements for a device."""
        if device_id not in self.measurements:
            return []

        measurements = self.measurements[device_id]
        return (
            measurements[-count:] if len(measurements) >= count else measurements.copy()
        )

    def get_power_statistics(self, device_id: int) -> Dict[str, float]:
        """Get power consumption statistics for a device."""
        measurements = self.measurements.get(device_id, [])
        if not measurements:
            return {}

        powers = [m.power_watts for m in measurements]

        return {
            "min_watts": min(powers),
            "max_watts": max(powers),
            "mean_watts": sum(powers) / len(powers),
            "current_watts": powers[-1] if powers else 0.0,
            "total_energy_joules": self.cumulative_energy.get(device_id, 0.0),
            "measurement_count": len(powers),
        }

    def reset_measurements(self):
        """Reset all measurements and cumulative energy."""
        with self._lock:
            for device_id in self.devices:
                self.measurements[device_id].clear()
                self.cumulative_energy[device_id] = 0.0
                if device_id in self.last_measurement_time:
                    del self.last_measurement_time[device_id]

            logger.info("Energy measurements reset")

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
        self.nvml.shutdown()

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, "_monitoring") and self._monitoring:
                self.stop_monitoring()
            if hasattr(self, "nvml"):
                self.nvml.shutdown()
        except Exception:
            pass  # Ignore cleanup errors
