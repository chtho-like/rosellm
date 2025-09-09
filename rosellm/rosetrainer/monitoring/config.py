"""
Energy Monitoring Configuration System

This module provides comprehensive configuration management for the energy monitoring
system, including validation, defaults, and integration with RoseLLM's training
configuration framework.

Key Features:
- Structured configuration with validation
- Environment variable support
- Integration with RoseLLM training configs
- Production-ready defaults
- Extensible configuration schema
"""

import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch

logger = __import__("logging").getLogger(__name__)


class EnergyMonitoringMode(Enum):
    """Energy monitoring operation modes."""

    DISABLED = "disabled"
    LOCAL_ONLY = "local"
    DISTRIBUTED = "distributed"
    HIERARCHICAL = "hierarchical"


class FallbackStrategy(Enum):
    """Strategies for handling NVML unavailability."""

    ESTIMATE = "estimate"
    ZERO = "zero"
    DISABLE = "disable"


@dataclass
class GPUTrackerConfig:
    """Configuration for GPU energy tracking."""

    # Device selection
    devices: Optional[List[int]] = None
    """List of GPU device IDs to monitor. If None, auto-detect all available GPUs."""

    # Sampling configuration
    sampling_interval: float = 1.0
    """Sampling interval in seconds for energy measurements."""

    enable_detailed_metrics: bool = True
    """Whether to collect detailed metrics (temperature, utilization, memory)."""

    # Fallback configuration
    fallback_power_estimate: float = 250.0
    """Power estimate in watts when NVML is unavailable."""

    fallback_strategy: FallbackStrategy = FallbackStrategy.ESTIMATE
    """Strategy to use when NVML is unavailable."""

    # Error handling
    max_consecutive_errors: int = 10
    """Maximum consecutive errors before disabling monitoring."""

    error_recovery_delay: float = 5.0
    """Delay in seconds before retrying after errors."""

    # Performance tuning
    measurement_buffer_size: int = 1000
    """Maximum number of measurements to keep in memory per device."""

    enable_background_monitoring: bool = True
    """Whether to run monitoring in background thread."""

    thread_priority: Optional[int] = None
    """Thread priority for monitoring thread (OS-specific)."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self):
        """Validate configuration parameters."""
        if self.sampling_interval <= 0:
            raise ValueError("sampling_interval must be positive")

        if self.fallback_power_estimate < 0:
            raise ValueError("fallback_power_estimate cannot be negative")

        if self.max_consecutive_errors <= 0:
            raise ValueError("max_consecutive_errors must be positive")

        if self.error_recovery_delay < 0:
            raise ValueError("error_recovery_delay cannot be negative")

        if self.measurement_buffer_size <= 0:
            raise ValueError("measurement_buffer_size must be positive")

        if self.devices is not None:
            if not all(isinstance(d, int) and d >= 0 for d in self.devices):
                raise ValueError("devices must be a list of non-negative integers")


@dataclass
class DistributedConfig:
    """Configuration for distributed energy monitoring."""

    # Aggregation settings
    aggregation_interval: float = 5.0
    """Interval in seconds for distributed energy aggregation."""

    enable_hierarchical_reporting: bool = True
    """Whether to enable hierarchical energy reporting by parallelism dimension."""

    max_history_size: int = 1000
    """Maximum number of distributed measurements to keep in history."""

    # Communication settings
    communication_backend: str = "nccl"
    """Communication backend for distributed operations (nccl, gloo)."""

    timeout_seconds: float = 30.0
    """Timeout for distributed operations in seconds."""

    # Fault tolerance
    enable_fault_tolerance: bool = True
    """Whether to enable fault tolerance for distributed operations."""

    max_failed_aggregations: int = 5
    """Maximum consecutive failed aggregations before disabling distributed mode."""

    # Optimization
    use_async_aggregation: bool = False
    """Whether to use asynchronous aggregation (experimental)."""

    compression_enabled: bool = False
    """Whether to enable compression for distributed data (experimental)."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self):
        """Validate configuration parameters."""
        if self.aggregation_interval <= 0:
            raise ValueError("aggregation_interval must be positive")

        if self.max_history_size <= 0:
            raise ValueError("max_history_size must be positive")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if self.max_failed_aggregations <= 0:
            raise ValueError("max_failed_aggregations must be positive")

        if self.communication_backend not in ["nccl", "gloo", "mpi"]:
            warnings.warn(
                f"Unknown communication backend: {self.communication_backend}"
            )


@dataclass
class IntegrationConfig:
    """Configuration for integration with RoseLLM components."""

    # Trainer integration
    integrate_with_trainer: bool = True
    """Whether to integrate energy monitoring with RoseTrainer."""

    log_interval: int = 100
    """Interval (in training steps) for logging energy statistics."""

    save_measurements: bool = True
    """Whether to save energy measurements to files."""

    # Checkpoint integration
    include_in_checkpoints: bool = False
    """Whether to include energy statistics in training checkpoints."""

    checkpoint_energy_summary: bool = True
    """Whether to include energy summary in checkpoint metadata."""

    # Logging and reporting
    log_level: str = "INFO"
    """Logging level for energy monitoring (DEBUG, INFO, WARNING, ERROR)."""

    enable_tensorboard: bool = False
    """Whether to log energy metrics to TensorBoard."""

    enable_wandb: bool = False
    """Whether to log energy metrics to Weights & Biases."""

    # Output configuration
    output_format: str = "json"
    """Output format for energy reports (json, csv, parquet)."""

    output_directory: Optional[str] = None
    """Directory to save energy monitoring outputs. If None, use current directory."""

    # Performance impact
    pause_during_checkpointing: bool = True
    """Whether to pause energy monitoring during checkpoint save/load."""

    pause_during_evaluation: bool = False
    """Whether to pause energy monitoring during evaluation phases."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self):
        """Validate configuration parameters."""
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")

        valid_output_formats = ["json", "csv", "parquet"]
        if self.output_format not in valid_output_formats:
            raise ValueError(f"output_format must be one of {valid_output_formats}")

        if self.output_directory is not None and not isinstance(
            self.output_directory, str
        ):
            raise ValueError("output_directory must be a string or None")


@dataclass
class EnergyMonitoringConfig:
    """
    Comprehensive energy monitoring configuration.

    This is the main configuration class that combines all aspects of energy monitoring
    configuration for RoseLLM training.
    """

    # Main operation mode
    mode: EnergyMonitoringMode = EnergyMonitoringMode.DISTRIBUTED
    """Energy monitoring operation mode."""

    # Sub-configurations
    gpu_tracker: GPUTrackerConfig = field(default_factory=GPUTrackerConfig)
    """GPU energy tracker configuration."""

    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    """Distributed energy monitoring configuration."""

    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    """Integration configuration with RoseLLM components."""

    # Global settings
    enabled: bool = True
    """Whether energy monitoring is enabled globally."""

    auto_start: bool = True
    """Whether to automatically start monitoring when trainer starts."""

    auto_stop: bool = True
    """Whether to automatically stop monitoring when trainer stops."""

    # Environment variable overrides
    respect_env_vars: bool = True
    """Whether to respect environment variable overrides."""

    def __post_init__(self):
        """Post-initialization processing."""
        if self.respect_env_vars:
            self.apply_env_overrides()
        self.validate()

    def validate(self):
        """Validate the entire configuration."""
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")

        if not isinstance(self.mode, EnergyMonitoringMode):
            raise ValueError("mode must be an EnergyMonitoringMode")

        # Validate sub-configurations
        self.gpu_tracker.validate()
        self.distributed.validate()
        self.integration.validate()

        # Cross-validation
        if self.mode == EnergyMonitoringMode.DISABLED and self.enabled:
            warnings.warn("Energy monitoring is enabled but mode is DISABLED")

        if self.mode in [
            EnergyMonitoringMode.DISTRIBUTED,
            EnergyMonitoringMode.HIERARCHICAL,
        ]:
            if not torch.distributed.is_available():
                warnings.warn(
                    "Distributed mode requested but torch.distributed not available"
                )

    def apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Global settings
        if "ROSE_ENERGY_ENABLED" in os.environ:
            self.enabled = os.environ["ROSE_ENERGY_ENABLED"].lower() in [
                "true",
                "1",
                "yes",
            ]

        if "ROSE_ENERGY_MODE" in os.environ:
            mode_str = os.environ["ROSE_ENERGY_MODE"].lower()
            for mode in EnergyMonitoringMode:
                if mode.value == mode_str:
                    self.mode = mode
                    break

        if "ROSE_ENERGY_AUTO_START" in os.environ:
            self.auto_start = os.environ["ROSE_ENERGY_AUTO_START"].lower() in [
                "true",
                "1",
                "yes",
            ]

        # GPU tracker overrides
        if "ROSE_ENERGY_SAMPLING_INTERVAL" in os.environ:
            try:
                self.gpu_tracker.sampling_interval = float(
                    os.environ["ROSE_ENERGY_SAMPLING_INTERVAL"]
                )
            except ValueError:
                warnings.warn("Invalid ROSE_ENERGY_SAMPLING_INTERVAL value")

        if "ROSE_ENERGY_FALLBACK_POWER" in os.environ:
            try:
                self.gpu_tracker.fallback_power_estimate = float(
                    os.environ["ROSE_ENERGY_FALLBACK_POWER"]
                )
            except ValueError:
                warnings.warn("Invalid ROSE_ENERGY_FALLBACK_POWER value")

        if "ROSE_ENERGY_DEVICES" in os.environ:
            try:
                device_str = os.environ["ROSE_ENERGY_DEVICES"]
                if device_str.strip():
                    self.gpu_tracker.devices = [
                        int(d.strip()) for d in device_str.split(",")
                    ]
            except ValueError:
                warnings.warn("Invalid ROSE_ENERGY_DEVICES value")

        # Distributed overrides
        if "ROSE_ENERGY_AGGREGATION_INTERVAL" in os.environ:
            try:
                self.distributed.aggregation_interval = float(
                    os.environ["ROSE_ENERGY_AGGREGATION_INTERVAL"]
                )
            except ValueError:
                warnings.warn("Invalid ROSE_ENERGY_AGGREGATION_INTERVAL value")

        # Integration overrides
        if "ROSE_ENERGY_LOG_INTERVAL" in os.environ:
            try:
                self.integration.log_interval = int(
                    os.environ["ROSE_ENERGY_LOG_INTERVAL"]
                )
            except ValueError:
                warnings.warn("Invalid ROSE_ENERGY_LOG_INTERVAL value")

        if "ROSE_ENERGY_OUTPUT_DIR" in os.environ:
            self.integration.output_directory = os.environ["ROSE_ENERGY_OUTPUT_DIR"]

        if "ROSE_ENERGY_LOG_LEVEL" in os.environ:
            self.integration.log_level = os.environ["ROSE_ENERGY_LOG_LEVEL"].upper()

    @classmethod
    def create_default(cls) -> "EnergyMonitoringConfig":
        """Create a default configuration."""
        return cls()

    @classmethod
    def create_for_mode(cls, mode: EnergyMonitoringMode) -> "EnergyMonitoringConfig":
        """Create configuration optimized for a specific mode."""
        config = cls()
        config.mode = mode

        if mode == EnergyMonitoringMode.DISABLED:
            config.enabled = False
        elif mode == EnergyMonitoringMode.LOCAL_ONLY:
            config.distributed.enable_hierarchical_reporting = False
            config.distributed.aggregation_interval = 10.0  # Less frequent
        elif mode == EnergyMonitoringMode.HIERARCHICAL:
            config.distributed.enable_hierarchical_reporting = True
            config.distributed.aggregation_interval = 2.0  # More frequent
            config.gpu_tracker.enable_detailed_metrics = True

        return config

    @classmethod
    def create_production(cls) -> "EnergyMonitoringConfig":
        """Create production-optimized configuration."""
        config = cls()

        # Conservative settings for production
        config.gpu_tracker.sampling_interval = 2.0
        config.gpu_tracker.enable_detailed_metrics = False
        config.gpu_tracker.measurement_buffer_size = 500

        config.distributed.aggregation_interval = 10.0
        config.distributed.max_history_size = 500
        config.distributed.enable_fault_tolerance = True

        config.integration.log_interval = 500
        config.integration.pause_during_checkpointing = True
        config.integration.save_measurements = False  # Avoid I/O overhead

        return config

    @classmethod
    def create_debug(cls) -> "EnergyMonitoringConfig":
        """Create debug configuration with verbose logging."""
        config = cls()

        # Detailed settings for debugging
        config.gpu_tracker.sampling_interval = 0.5
        config.gpu_tracker.enable_detailed_metrics = True
        config.gpu_tracker.max_consecutive_errors = 3

        config.distributed.aggregation_interval = 1.0
        config.distributed.enable_fault_tolerance = True

        config.integration.log_level = "DEBUG"
        config.integration.log_interval = 10
        config.integration.save_measurements = True

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "mode": self.mode.value,
            "enabled": self.enabled,
            "auto_start": self.auto_start,
            "auto_stop": self.auto_stop,
            "respect_env_vars": self.respect_env_vars,
            "gpu_tracker": {
                "devices": self.gpu_tracker.devices,
                "sampling_interval": self.gpu_tracker.sampling_interval,
                "enable_detailed_metrics": self.gpu_tracker.enable_detailed_metrics,
                "fallback_power_estimate": self.gpu_tracker.fallback_power_estimate,
                "fallback_strategy": self.gpu_tracker.fallback_strategy.value,
                "max_consecutive_errors": self.gpu_tracker.max_consecutive_errors,
                "error_recovery_delay": self.gpu_tracker.error_recovery_delay,
                "measurement_buffer_size": self.gpu_tracker.measurement_buffer_size,
                "enable_background_monitoring": (
                    self.gpu_tracker.enable_background_monitoring
                ),
            },
            "distributed": {
                "aggregation_interval": self.distributed.aggregation_interval,
                "enable_hierarchical_reporting": (
                    self.distributed.enable_hierarchical_reporting
                ),
                "max_history_size": self.distributed.max_history_size,
                "communication_backend": self.distributed.communication_backend,
                "timeout_seconds": self.distributed.timeout_seconds,
                "enable_fault_tolerance": self.distributed.enable_fault_tolerance,
                "max_failed_aggregations": self.distributed.max_failed_aggregations,
            },
            "integration": {
                "integrate_with_trainer": self.integration.integrate_with_trainer,
                "log_interval": self.integration.log_interval,
                "save_measurements": self.integration.save_measurements,
                "include_in_checkpoints": self.integration.include_in_checkpoints,
                "log_level": self.integration.log_level,
                "output_format": self.integration.output_format,
                "output_directory": self.integration.output_directory,
                "pause_during_checkpointing": (
                    self.integration.pause_during_checkpointing
                ),
                "pause_during_evaluation": self.integration.pause_during_evaluation,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EnergyMonitoringConfig":
        """Create configuration from dictionary."""
        # Extract main fields
        mode = EnergyMonitoringMode(config_dict.get("mode", "distributed"))
        enabled = config_dict.get("enabled", True)
        auto_start = config_dict.get("auto_start", True)
        auto_stop = config_dict.get("auto_stop", True)
        respect_env_vars = config_dict.get("respect_env_vars", True)

        # Create sub-configurations
        gpu_config = GPUTrackerConfig()
        if "gpu_tracker" in config_dict:
            gpu_dict = config_dict["gpu_tracker"]
            for key, value in gpu_dict.items():
                if key == "fallback_strategy":
                    value = FallbackStrategy(value)
                if hasattr(gpu_config, key):
                    setattr(gpu_config, key, value)

        distributed_config = DistributedConfig()
        if "distributed" in config_dict:
            dist_dict = config_dict["distributed"]
            for key, value in dist_dict.items():
                if hasattr(distributed_config, key):
                    setattr(distributed_config, key, value)

        integration_config = IntegrationConfig()
        if "integration" in config_dict:
            int_dict = config_dict["integration"]
            for key, value in int_dict.items():
                if hasattr(integration_config, key):
                    setattr(integration_config, key, value)

        # Create main configuration
        config = cls(
            mode=mode,
            enabled=enabled,
            auto_start=auto_start,
            auto_stop=auto_stop,
            respect_env_vars=respect_env_vars,
            gpu_tracker=gpu_config,
            distributed=distributed_config,
            integration=integration_config,
        )

        return config

    def clone(self) -> "EnergyMonitoringConfig":
        """Create a deep copy of the configuration."""
        return self.from_dict(self.to_dict())

    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"EnergyMonitoringConfig(mode={self.mode.value}, enabled={self.enabled}, "
            f"sampling={self.gpu_tracker.sampling_interval}s, "
            f"aggregation={self.distributed.aggregation_interval}s)"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return str(self.to_dict())
