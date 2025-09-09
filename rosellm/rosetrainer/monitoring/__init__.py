"""
Energy Monitoring System for RoseLLM

This module provides comprehensive energy monitoring capabilities for distributed
training workloads, including GPU power tracking, distributed aggregation,
and integration with RoseLLM's parallelism infrastructure.

Key Components:
- EnergyMonitor: Main interface for energy monitoring
- GPUEnergyTracker: NVML-based GPU energy tracking with fallback
- DistributedEnergyAggregator: Distributed energy aggregation across processes
- EnergyMonitoringConfig: Configuration system with validation

Features:
- NVML-based power monitoring with graceful degradation
- Distributed energy aggregation across TP, PP, DP, CP, EP dimensions
- Context manager support for pause/resume operations
- Comprehensive error handling and fault tolerance
- Integration with RoseTrainer and existing RoseLLM components
- Production-ready monitoring with configurable sampling rates
- Hierarchical energy reporting by parallelism dimension
- Export capabilities (JSON, CSV, Parquet)

Usage:
    ```python
    from rosellm.rosetrainer.monitoring import EnergyMonitor, EnergyMonitoringConfig

    # Basic usage
    with EnergyMonitor() as monitor:
        # Training code here
        pass

    # Custom configuration
    config = EnergyMonitoringConfig.create_production()
    monitor = EnergyMonitor(config)
    monitor.start_monitoring()

    # Get statistics
    stats = monitor.get_current_statistics()

    # Generate report
    report = monitor.get_energy_report()
    monitor.save_energy_report("energy_report.json")

    monitor.stop_monitoring()
    ```

Environment Variables:
    ROSE_ENERGY_ENABLED: Enable/disable energy monitoring (true/false)
    ROSE_ENERGY_MODE: Monitoring mode (disabled/local/distributed/hierarchical)
    ROSE_ENERGY_SAMPLING_INTERVAL: Sampling interval in seconds
    ROSE_ENERGY_DEVICES: Comma-separated list of GPU device IDs
    ROSE_ENERGY_FALLBACK_POWER: Fallback power estimate in watts
    ROSE_ENERGY_LOG_INTERVAL: Logging interval in training steps
    ROSE_ENERGY_OUTPUT_DIR: Directory for energy monitoring outputs
"""

from .config import (
    DistributedConfig,
    EnergyMonitoringConfig,
    EnergyMonitoringMode,
    FallbackStrategy,
    GPUTrackerConfig,
    IntegrationConfig,
)
from .distributed_energy import (
    DistributedEnergyAggregator,
    DistributedEnergyMeasurement,
    ParallelismInfo,
    ParallelismType,
)
from .energy_monitor import EnergyMonitor
from .energy_tracker import (
    DeviceInfo,
    EnergyMeasurement,
    GPUEnergyTracker,
    NVMLInterface,
)

__all__ = [
    # Main interface
    "EnergyMonitor",
    # Configuration
    "EnergyMonitoringConfig",
    "EnergyMonitoringMode",
    "FallbackStrategy",
    "GPUTrackerConfig",
    "DistributedConfig",
    "IntegrationConfig",
    # Local GPU tracking
    "GPUEnergyTracker",
    "EnergyMeasurement",
    "DeviceInfo",
    "NVMLInterface",
    # Distributed aggregation
    "DistributedEnergyAggregator",
    "DistributedEnergyMeasurement",
    "ParallelismType",
    "ParallelismInfo",
]

# Version information
__version__ = "1.0.0"
__author__ = "RoseLLM Team"
__email__ = "rosellm@example.com"

# Module metadata
__description__ = "Energy monitoring system for distributed training workloads"
__license__ = "MIT"
__url__ = "https://github.com/rosellm/rosellm"


def get_version():
    """Get the version of the monitoring module."""
    return __version__


def create_default_monitor():
    """Create a default energy monitor instance."""
    return EnergyMonitor()


def create_production_monitor():
    """Create a production-optimized energy monitor."""
    config = EnergyMonitoringConfig.create_production()
    return EnergyMonitor(config)


def create_debug_monitor():
    """Create a debug energy monitor with verbose logging."""
    config = EnergyMonitoringConfig.create_debug()
    return EnergyMonitor(config)
