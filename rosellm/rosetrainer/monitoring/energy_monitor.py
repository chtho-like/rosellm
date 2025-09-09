"""
Comprehensive Energy Monitoring System for RoseLLM

This module provides the main EnergyMonitor class that integrates all components
of the energy monitoring system, including local tracking, distributed aggregation,
and trainer integration.

Key Features:
- Unified energy monitoring interface
- Automatic integration with RoseLLM parallelism
- Context manager support
- Comprehensive error handling and recovery
- Production-ready monitoring capabilities
"""

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..parallelism.parallel_state import is_initialized as parallel_state_initialized
from .config import EnergyMonitoringConfig, EnergyMonitoringMode
from .distributed_energy import DistributedEnergyAggregator
from .energy_tracker import GPUEnergyTracker

logger = logging.getLogger(__name__)


class EnergyMonitor:
    """
    Main Energy Monitoring System for RoseLLM.

    This class provides a unified interface for energy monitoring across all
    parallelism dimensions and integrates seamlessly with RoseLLM's training
    infrastructure.
    """

    def __init__(self, config: Optional[EnergyMonitoringConfig] = None):
        """
        Initialize Energy Monitor.

        Args:
            config: Energy monitoring configuration. If None, uses default.
        """
        if config is None:
            config = EnergyMonitoringConfig.create_default()

        self.config = config
        self.is_distributed = self._check_distributed_setup()

        # Initialize components based on configuration
        self._initialize_components()

        # State management
        self._monitoring = False
        self._paused = False
        self._error_count = 0
        self._last_error_time = 0.0
        self._lock = threading.RLock()

        # Integration state
        self._trainer_integrated = False
        self._step_count = 0
        self._last_log_step = 0

        logger.info(
            f"EnergyMonitor initialized (mode: {self.config.mode.value}, "
            f"distributed: {self.is_distributed})"
        )

    def _check_distributed_setup(self) -> bool:
        """Check if we're in a distributed environment."""
        return (
            self.config.mode
            in [EnergyMonitoringMode.DISTRIBUTED, EnergyMonitoringMode.HIERARCHICAL]
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and parallel_state_initialized()
        )

    def _initialize_components(self):
        """Initialize monitoring components based on configuration."""
        self.local_tracker: Optional[GPUEnergyTracker] = None
        self.distributed_aggregator: Optional[DistributedEnergyAggregator] = None

        if self.config.mode == EnergyMonitoringMode.DISABLED:
            logger.info("Energy monitoring disabled by configuration")
            return

        # Initialize local GPU tracker
        try:
            self.local_tracker = GPUEnergyTracker(
                devices=self.config.gpu_tracker.devices,
                sampling_interval=self.config.gpu_tracker.sampling_interval,
                enable_detailed_metrics=self.config.gpu_tracker.enable_detailed_metrics,
                fallback_power_estimate=self.config.gpu_tracker.fallback_power_estimate,
            )
            logger.info("Local GPU energy tracker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize local GPU tracker: {e}")
            if self.config.gpu_tracker.fallback_strategy.value == "disable":
                raise

        # Initialize distributed aggregator if needed
        if self.is_distributed and self.local_tracker:
            try:
                self.distributed_aggregator = DistributedEnergyAggregator(
                    local_tracker=self.local_tracker,
                    aggregation_interval=self.config.distributed.aggregation_interval,
                    enable_hierarchical_reporting=(
                        self.config.distributed.enable_hierarchical_reporting
                    ),
                    max_history_size=self.config.distributed.max_history_size,
                )
                logger.info("Distributed energy aggregator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize distributed aggregator: {e}")
                # Fall back to local-only mode
                self.is_distributed = False

    def start_monitoring(self) -> bool:
        """
        Start energy monitoring.

        Returns:
            True if monitoring started successfully
        """
        with self._lock:
            if self._monitoring:
                logger.warning("Energy monitoring already started")
                return True

            if (
                self.config.mode == EnergyMonitoringMode.DISABLED
                or not self.config.enabled
            ):
                logger.info("Energy monitoring is disabled")
                return False

            success = False

            try:
                # Start local tracker
                if self.local_tracker:
                    if not self.local_tracker.start_monitoring(
                        background=self.config.gpu_tracker.enable_background_monitoring
                    ):
                        logger.error("Failed to start local energy tracker")
                        return False
                    success = True

                # Start distributed aggregator
                if self.distributed_aggregator:
                    if not self.distributed_aggregator.start_monitoring():
                        logger.error("Failed to start distributed energy aggregator")
                        if self.local_tracker:
                            self.local_tracker.stop_monitoring()
                        return False
                    success = True

                if success:
                    self._monitoring = True
                    self._paused = False
                    self._error_count = 0
                    logger.info("Energy monitoring started successfully")

                return success

            except Exception as e:
                logger.error(f"Failed to start energy monitoring: {e}")
                self._handle_error(e)
                return False

    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop energy monitoring and return comprehensive statistics.

        Returns:
            Dictionary containing energy statistics and metadata
        """
        with self._lock:
            if not self._monitoring:
                return {}

            self._monitoring = False
            results: Dict[str, Any] = {}

            try:
                # Stop distributed aggregator first to get final measurements
                if self.distributed_aggregator:
                    dist_results = self.distributed_aggregator.stop_monitoring()
                    results.update(dist_results)

                # Stop local tracker
                if self.local_tracker:
                    local_results = self.local_tracker.stop_monitoring()
                    results["local_energy_by_device"] = local_results

                # Add monitoring metadata
                results.update(
                    {
                        "monitoring_mode": self.config.mode.value,
                        "is_distributed": self.is_distributed,
                        "error_count": self._error_count,
                        "step_count": self._step_count,
                    }
                )

                logger.info("Energy monitoring stopped successfully")
                return results

            except Exception as e:
                logger.error(f"Error stopping energy monitoring: {e}")
                return results

    def pause_monitoring(self):
        """Pause energy monitoring."""
        with self._lock:
            if not self._monitoring or self._paused:
                return

            try:
                if self.local_tracker:
                    self.local_tracker.pause_monitoring()

                # Note: distributed aggregator pause is handled automatically
                # since it relies on local measurements

                self._paused = True
                logger.debug("Energy monitoring paused")

            except Exception as e:
                logger.error(f"Error pausing energy monitoring: {e}")
                self._handle_error(e)

    def resume_monitoring(self):
        """Resume energy monitoring."""
        with self._lock:
            if not self._monitoring or not self._paused:
                return

            try:
                if self.local_tracker:
                    self.local_tracker.resume_monitoring()

                self._paused = False
                logger.debug("Energy monitoring resumed")

            except Exception as e:
                logger.error(f"Error resuming energy monitoring: {e}")
                self._handle_error(e)

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

    def get_current_statistics(self) -> Dict[str, Any]:
        """
        Get current energy monitoring statistics.

        Returns:
            Dictionary containing current energy statistics
        """
        if not self._monitoring:
            return {"monitoring_active": False}

        stats: Dict[str, Any] = {"monitoring_active": True, "paused": self._paused}

        try:
            # Local statistics
            if self.local_tracker:
                local_power = self.local_tracker.get_current_power()
                local_energy = self.local_tracker.get_energy_consumption()

                stats.update(
                    {
                        "local_current_power_watts": local_power,
                        "local_total_energy_joules": local_energy,
                        "local_device_count": len(self.local_tracker.devices),
                    }
                )

                # Per-device statistics
                for device_id in self.local_tracker.devices:
                    device_stats = self.local_tracker.get_power_statistics(device_id)
                    if device_stats:
                        stats[f"device_{device_id}_stats"] = device_stats

            # Distributed statistics
            if self.distributed_aggregator:
                # Get recent distributed measurement
                recent_measurements = (
                    self.distributed_aggregator.get_recent_measurements(count=1)
                )
                if recent_measurements:
                    latest = recent_measurements[-1]
                    stats.update(
                        {
                            "distributed_total_power_watts": (
                                latest.aggregated_power_watts
                            ),
                            "distributed_total_energy_joules": (
                                latest.aggregated_energy_joules
                            ),
                            "distributed_process_count": latest.total_processes,
                            "distributed_average_power_watts": latest.average_power,
                        }
                    )

                # Hierarchical report if enabled
                if self.config.distributed.enable_hierarchical_reporting:
                    hierarchical_stats = (
                        self.distributed_aggregator.get_hierarchical_energy_report()
                    )
                    if hierarchical_stats:
                        stats["hierarchical_report"] = hierarchical_stats

                # Efficiency metrics
                efficiency_metrics = (
                    self.distributed_aggregator.get_energy_efficiency_metrics()
                )
                if efficiency_metrics:
                    stats["efficiency_metrics"] = efficiency_metrics

            return stats

        except Exception as e:
            logger.error(f"Error getting current statistics: {e}")
            self._handle_error(e)
            return stats

    def log_energy_statistics(self, step: Optional[int] = None, force: bool = False):
        """
        Log energy statistics if logging interval has been reached.

        Args:
            step: Current training step
            force: Force logging regardless of interval
        """
        if not self._monitoring:
            return

        if step is not None:
            self._step_count = step

        # Check if we should log
        if not force:
            if (
                self._step_count - self._last_log_step
            ) < self.config.integration.log_interval:
                return

        try:
            stats = self.get_current_statistics()

            if stats.get("monitoring_active", False):
                # Format statistics for logging
                log_parts = [f"Step {self._step_count}"]

                if "local_current_power_watts" in stats:
                    power = stats["local_current_power_watts"]
                    if isinstance(power, dict):
                        total_power = sum(power.values())
                        log_parts.append(f"Local Power: {total_power:.1f}W")
                    else:
                        log_parts.append(f"Local Power: {power:.1f}W")

                if "local_total_energy_joules" in stats:
                    energy = stats["local_total_energy_joules"]
                    if isinstance(energy, dict):
                        total_energy = sum(energy.values())
                        log_parts.append(f"Local Energy: {total_energy:.1f}J")
                    else:
                        log_parts.append(f"Local Energy: {energy:.1f}J")

                if "distributed_total_power_watts" in stats:
                    dist_power = stats["distributed_total_power_watts"]
                    log_parts.append(f"Distributed Power: {dist_power:.1f}W")

                if "distributed_total_energy_joules" in stats:
                    dist_energy = stats["distributed_total_energy_joules"]
                    log_parts.append(f"Distributed Energy: {dist_energy:.1f}J")

                if "distributed_process_count" in stats:
                    process_count = stats["distributed_process_count"]
                    log_parts.append(f"Processes: {process_count}")

                logger.info("Energy Stats - " + " | ".join(log_parts))

                self._last_log_step = self._step_count

        except Exception as e:
            logger.error(f"Error logging energy statistics: {e}")
            self._handle_error(e)

    def get_energy_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive energy report.

        Returns:
            Detailed energy monitoring report
        """
        report = {
            "timestamp": time.time(),
            "monitoring_config": self.config.to_dict(),
            "monitoring_active": self._monitoring,
            "paused": self._paused,
            "error_count": self._error_count,
            "step_count": self._step_count,
        }

        if not self._monitoring:
            return report

        try:
            # Current statistics
            current_stats = self.get_current_statistics()
            report["current_statistics"] = current_stats

            # Device information
            if self.local_tracker:
                device_info = self.local_tracker.get_device_info()
                if isinstance(device_info, dict):
                    report["device_information"] = {
                        str(dev_id): str(info) for dev_id, info in device_info.items()
                    }
                else:
                    report["device_information"] = str(device_info)

            # Historical data summary
            if self.distributed_aggregator:
                recent_measurements = (
                    self.distributed_aggregator.get_recent_measurements(count=10)
                )
                if recent_measurements:
                    report["recent_measurements_summary"] = {
                        "measurement_count": len(recent_measurements),
                        "time_span_seconds": recent_measurements[-1].timestamp
                        - recent_measurements[0].timestamp,
                        "power_trend": [
                            m.aggregated_power_watts for m in recent_measurements[-5:]
                        ],
                        "energy_trend": [
                            m.aggregated_energy_joules for m in recent_measurements[-5:]
                        ],
                    }

            return report

        except Exception as e:
            logger.error(f"Error generating energy report: {e}")
            report["error"] = str(e)
            return report

    def save_energy_report(
        self, filepath: Optional[str] = None, format: str = "json"
    ) -> bool:
        """
        Save energy report to file.

        Args:
            filepath: File path to save to. If None, auto-generate.
            format: Output format (json, csv, parquet)

        Returns:
            True if saved successfully
        """
        if not self.config.integration.save_measurements:
            logger.debug("Save measurements disabled in configuration")
            return False

        try:
            # Generate report
            report = self.get_energy_report()

            # Determine output path
            if filepath is None:
                output_dir = self.config.integration.output_directory or "."
                os.makedirs(output_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(
                    output_dir, f"energy_report_{timestamp}.{format}"
                )

            # Save based on format
            if format == "json":
                import json

                with open(filepath, "w") as f:
                    json.dump(report, f, indent=2)
            elif format == "csv":
                # Flatten report for CSV
                import pandas as pd

                flat_data = self._flatten_dict(report)
                df = pd.DataFrame([flat_data])
                df.to_csv(filepath, index=False)
            elif format == "parquet":
                import pandas as pd

                flat_data = self._flatten_dict(report)
                df = pd.DataFrame([flat_data])
                df.to_parquet(filepath, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return False

            logger.info(f"Energy report saved to: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save energy report: {e}")
            return False

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "_",
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        """Flatten nested dictionary for CSV/parquet export with depth limit."""
        if max_depth <= 0:
            # Prevent infinite recursion and excessive memory usage
            return {parent_key: str(d)}

        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict) and max_depth > 1:
                items.extend(
                    self._flatten_dict(
                        v, new_key, sep=sep, max_depth=max_depth - 1
                    ).items()
                )
            elif (
                isinstance(v, (list, tuple)) and len(v) < 100
            ):  # Limit list size to prevent memory issues
                items.append((new_key, str(v)))
            elif isinstance(v, (int, float, str, bool, type(None))):
                items.append((new_key, v))
            else:
                # Convert complex objects to string representation
                items.append((new_key, str(v)))
        return dict(items)

    def _handle_error(self, error: Exception):
        """Handle errors with recovery logic."""
        self._error_count += 1
        current_time = time.time()

        if self._error_count >= self.config.gpu_tracker.max_consecutive_errors:
            logger.error(
                f"Too many consecutive errors ({self._error_count}), "
                "disabling monitoring"
            )
            try:
                self.stop_monitoring()
            except Exception:
                pass
            return

        # Implement error recovery delay
        if (
            current_time - self._last_error_time
        ) < self.config.gpu_tracker.error_recovery_delay:
            return

        self._last_error_time = current_time

        # Attempt recovery
        try:
            if not self.is_monitoring():
                logger.info(
                    f"Attempting to restart monitoring after error "
                    f"(attempt {self._error_count})"
                )
                if self.start_monitoring():
                    self._error_count = 0  # Reset error count on successful recovery
                    logger.info("Energy monitoring recovered successfully")
        except Exception as recovery_error:
            logger.error(f"Failed to recover from error: {recovery_error}")

    def integrate_with_trainer(self, trainer) -> bool:
        """
        Integrate energy monitoring with RoseTrainer.

        Args:
            trainer: RoseTrainer instance

        Returns:
            True if integration successful
        """
        if not self.config.integration.integrate_with_trainer:
            return False

        try:
            # Set up trainer hooks for automatic start/stop
            if hasattr(trainer, "add_hook"):
                trainer.add_hook("before_train", lambda: self.start_monitoring())
                trainer.add_hook("after_train", lambda: self.stop_monitoring())

                if self.config.integration.pause_during_checkpointing:
                    trainer.add_hook(
                        "before_checkpoint", lambda: self.pause_monitoring()
                    )
                    trainer.add_hook(
                        "after_checkpoint", lambda: self.resume_monitoring()
                    )

                if self.config.integration.pause_during_evaluation:
                    trainer.add_hook("before_eval", lambda: self.pause_monitoring())
                    trainer.add_hook("after_eval", lambda: self.resume_monitoring())

            # Set up step-based logging
            if hasattr(trainer, "add_step_hook"):

                def log_step(step):
                    self.log_energy_statistics(step=step)

                trainer.add_step_hook(log_step)

            self._trainer_integrated = True
            logger.info("Energy monitoring integrated with trainer")
            return True

        except Exception as e:
            logger.error(f"Failed to integrate with trainer: {e}")
            return False

    def __enter__(self):
        """Context manager entry."""
        if self.config.auto_start:
            self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.config.auto_stop:
            self.stop_monitoring()

    def cleanup_resources(self) -> None:
        """Explicit resource cleanup method."""
        try:
            # Stop monitoring if active
            if hasattr(self, "_monitoring") and self._monitoring:
                self.stop_monitoring()

            # Clean up components
            if hasattr(self, "local_tracker") and self.local_tracker:
                try:
                    # Try to call cleanup method if it exists
                    cleanup_method = getattr(
                        self.local_tracker, "cleanup_resources", None
                    ) or getattr(self.local_tracker, "cleanup", None)
                    if cleanup_method and callable(cleanup_method):
                        cleanup_method()
                except Exception as e:
                    logger.debug(f"Local tracker cleanup failed: {e}")

            if hasattr(self, "distributed_aggregator") and self.distributed_aggregator:
                try:
                    # Try to call cleanup method if it exists
                    cleanup_method = getattr(
                        self.distributed_aggregator, "cleanup_resources", None
                    ) or getattr(self.distributed_aggregator, "cleanup", None)
                    if cleanup_method and callable(cleanup_method):
                        cleanup_method()
                except Exception as e:
                    logger.debug(f"Distributed aggregator cleanup failed: {e}")

            # Clear references
            self.local_tracker = None
            self.distributed_aggregator = None

            logger.debug("Energy monitor resources cleaned up successfully")

        except Exception as e:
            logger.warning(f"Error during energy monitor cleanup: {e}")

    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.cleanup_resources()
        except Exception:
            pass  # Ignore cleanup errors to avoid issues during garbage collection
