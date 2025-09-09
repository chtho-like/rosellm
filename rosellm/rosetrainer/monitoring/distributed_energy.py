"""
Distributed Energy Aggregation System

This module provides distributed energy monitoring and aggregation across all
parallelism dimensions in RoseLLM. It integrates with the parallel state management
system to provide comprehensive energy tracking in distributed training scenarios.

Key Features:
- Energy aggregation across TP, PP, DP, CP, and EP dimensions
- Automatic process group detection and management
- Hierarchical energy reporting
- Integration with existing RoseLLM parallelism infrastructure
- Fault-tolerant distributed operations
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist

from ..parallelism.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_size,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_size,
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_size,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)
from ..parallelism.parallel_state import is_initialized as parallel_state_initialized
from .energy_tracker import EnergyMeasurement, GPUEnergyTracker

logger = logging.getLogger(__name__)


class ParallelismType(Enum):
    """Types of parallelism for energy aggregation."""

    DATA_PARALLEL = "dp"
    TENSOR_PARALLEL = "tp"
    PIPELINE_PARALLEL = "pp"
    CONTEXT_PARALLEL = "cp"
    EXPERT_PARALLEL = "ep"
    GLOBAL = "global"


@dataclass
class ParallelismInfo:
    """Information about a parallelism dimension."""

    parallel_type: ParallelismType
    rank: int
    size: int
    group: Optional[dist.ProcessGroup] = None

    def __str__(self) -> str:
        return f"{self.parallel_type.value}={self.size}(rank={self.rank})"


@dataclass
class DistributedEnergyMeasurement:
    """Distributed energy measurement across processes."""

    timestamp: float
    measurements_by_rank: Dict[int, EnergyMeasurement] = field(default_factory=dict)
    aggregated_power_watts: float = 0.0
    aggregated_energy_joules: float = 0.0
    parallelism_info: Dict[ParallelismType, ParallelismInfo] = field(
        default_factory=dict
    )

    @property
    def total_processes(self) -> int:
        """Total number of processes in this measurement."""
        return len(self.measurements_by_rank)

    @property
    def average_power(self) -> float:
        """Average power across all processes."""
        if not self.measurements_by_rank:
            return 0.0
        return sum(m.power_watts for m in self.measurements_by_rank.values()) / len(
            self.measurements_by_rank
        )


class DistributedEnergyAggregator:
    """
    Distributed Energy Aggregation System for RoseLLM.

    This class handles energy measurement aggregation across all parallelism dimensions,
    providing comprehensive energy monitoring for distributed training workloads.
    """

    def __init__(
        self,
        local_tracker: Optional[GPUEnergyTracker] = None,
        aggregation_interval: float = 5.0,
        enable_hierarchical_reporting: bool = True,
        max_history_size: int = 1000,
    ):
        """
        Initialize Distributed Energy Aggregator.

        Args:
            local_tracker: Local GPU energy tracker. If None, creates one automatically.
            aggregation_interval: Interval for distributed aggregation (seconds)
            enable_hierarchical_reporting: Enable hierarchical energy reporting
            max_history_size: Maximum number of historical measurements to keep
        """
        self.aggregation_interval = aggregation_interval
        self.enable_hierarchical_reporting = enable_hierarchical_reporting
        self.max_history_size = max_history_size

        # Initialize local tracker
        if local_tracker is None:
            self.local_tracker = GPUEnergyTracker()
        else:
            self.local_tracker = local_tracker

        # Distributed state
        self.is_distributed = self._check_distributed_state()
        self.parallelism_info: Dict[ParallelismType, ParallelismInfo] = {}
        self._collect_parallelism_info()

        # Energy aggregation state
        self.distributed_measurements: List[DistributedEnergyMeasurement] = []
        self.last_aggregation_time = 0.0
        self._monitoring = False

        # Rank information
        self.global_rank = dist.get_rank() if self.is_distributed else 0
        self.global_size = dist.get_world_size() if self.is_distributed else 1

        logger.info(
            f"DistributedEnergyAggregator initialized "
            f"(rank {self.global_rank}/{self.global_size})"
        )
        if self.parallelism_info:
            parallelism_str = ", ".join(
                str(info) for info in self.parallelism_info.values()
            )
            logger.info(f"Parallelism dimensions: {parallelism_str}")

    def _check_distributed_state(self) -> bool:
        """Check if we're in a distributed environment."""
        return (
            dist.is_available()
            and dist.is_initialized()
            and parallel_state_initialized()
        )

    def _collect_parallelism_info(self):
        """Collect information about all parallelism dimensions."""
        if not self.is_distributed:
            return

        try:
            # Data Parallel
            dp_size = get_data_parallel_size()
            if dp_size > 1:
                self.parallelism_info[ParallelismType.DATA_PARALLEL] = ParallelismInfo(
                    parallel_type=ParallelismType.DATA_PARALLEL,
                    rank=get_data_parallel_rank(),
                    size=dp_size,
                    group=get_data_parallel_group(),
                )

            # Tensor Parallel
            tp_size = get_tensor_model_parallel_size()
            if tp_size > 1:
                self.parallelism_info[
                    ParallelismType.TENSOR_PARALLEL
                ] = ParallelismInfo(
                    parallel_type=ParallelismType.TENSOR_PARALLEL,
                    rank=get_tensor_model_parallel_rank(),
                    size=tp_size,
                    group=get_tensor_model_parallel_group(),
                )

            # Pipeline Parallel
            pp_size = get_pipeline_model_parallel_size()
            if pp_size > 1:
                self.parallelism_info[
                    ParallelismType.PIPELINE_PARALLEL
                ] = ParallelismInfo(
                    parallel_type=ParallelismType.PIPELINE_PARALLEL,
                    rank=get_pipeline_model_parallel_rank(),
                    size=pp_size,
                    group=get_pipeline_model_parallel_group(),
                )

            # Context Parallel
            try:
                cp_size = get_context_parallel_size()
                if cp_size > 1:
                    self.parallelism_info[
                        ParallelismType.CONTEXT_PARALLEL
                    ] = ParallelismInfo(
                        parallel_type=ParallelismType.CONTEXT_PARALLEL,
                        rank=get_context_parallel_rank(),
                        size=cp_size,
                        group=get_context_parallel_group(),
                    )
            except Exception:
                pass  # Context parallelism may not be available

            # Expert Parallel
            try:
                ep_size = get_expert_model_parallel_size()
                if ep_size > 1:
                    self.parallelism_info[
                        ParallelismType.EXPERT_PARALLEL
                    ] = ParallelismInfo(
                        parallel_type=ParallelismType.EXPERT_PARALLEL,
                        rank=get_expert_model_parallel_rank(),
                        size=ep_size,
                        group=get_expert_model_parallel_group(),
                    )
            except Exception:
                pass  # Expert parallelism may not be available

        except Exception as e:
            logger.warning(f"Failed to collect some parallelism info: {e}")

    def start_monitoring(self, background: bool = True) -> bool:
        """
        Start distributed energy monitoring.

        Args:
            background: Whether to run in background thread

        Returns:
            True if monitoring started successfully
        """
        if self._monitoring:
            logger.warning("Distributed energy monitoring already started")
            return True

        # Start local tracker
        if not self.local_tracker.start_monitoring(background=False):
            logger.error("Failed to start local energy tracker")
            return False

        self._monitoring = True
        logger.info(f"Distributed energy monitoring started (rank {self.global_rank})")
        return True

    def stop_monitoring(self) -> Dict[str, Union[float, Dict]]:
        """
        Stop monitoring and return comprehensive energy statistics.

        Returns:
            Dictionary containing local and distributed energy statistics
        """
        if not self._monitoring:
            return {}

        self._monitoring = False

        # Stop local tracker and get final measurement
        local_energy = self.local_tracker.stop_monitoring()

        # Perform final aggregation
        final_measurement = self.aggregate_energy_measurements()

        # Compile results
        results = {
            "local_energy_joules": local_energy,
            "local_total_energy_joules": sum(local_energy.values()),
            "global_rank": self.global_rank,
            "global_size": self.global_size,
            "parallelism_info": {
                ptype.value: str(info) for ptype, info in self.parallelism_info.items()
            },
        }

        if final_measurement:
            results.update(
                {
                    "distributed_total_power_watts": (
                        final_measurement.aggregated_power_watts
                    ),
                    "distributed_total_energy_joules": (
                        final_measurement.aggregated_energy_joules
                    ),
                    "average_power_per_process": final_measurement.average_power,
                    "total_processes": final_measurement.total_processes,
                }
            )

        logger.info(f"Distributed energy monitoring stopped (rank {self.global_rank})")
        return results

    def aggregate_energy_measurements(self) -> Optional[DistributedEnergyMeasurement]:
        """
        Aggregate energy measurements across all processes.

        Returns:
            Aggregated measurement or None if aggregation failed
        """
        if not self.is_distributed:
            return self._create_local_measurement()

        try:
            current_time = time.time()

            # Get local measurements
            local_measurements = {}
            for device_id in self.local_tracker.devices:
                recent = self.local_tracker.get_recent_measurements(device_id, count=1)
                if recent:
                    local_measurements[device_id] = recent[-1]

            # Create local measurement summary
            if local_measurements:
                # Use first device for timing (all should be close)
                first_measurement = next(iter(local_measurements.values()))
                local_summary = EnergyMeasurement(
                    timestamp=first_measurement.timestamp,
                    power_watts=sum(m.power_watts for m in local_measurements.values()),
                    device_id=self.global_rank,  # Use rank as identifier
                    cumulative_energy_joules=sum(
                        m.cumulative_energy_joules for m in local_measurements.values()
                    ),
                    temperature_celsius=None,  # Not meaningful to aggregate
                    utilization_percent=None,  # Not meaningful to aggregate
                    memory_used_mb=None,
                    memory_total_mb=None,
                )
            else:
                # Create fallback measurement
                local_summary = EnergyMeasurement(
                    timestamp=current_time,
                    power_watts=0.0,
                    device_id=self.global_rank,
                    cumulative_energy_joules=0.0,
                )

            # Perform distributed aggregation
            aggregated_measurement = self._perform_distributed_aggregation(
                local_summary
            )

            # Store measurement
            if aggregated_measurement:
                self.distributed_measurements.append(aggregated_measurement)

                # Maintain history size limit
                if len(self.distributed_measurements) > self.max_history_size:
                    self.distributed_measurements = self.distributed_measurements[
                        -self.max_history_size :
                    ]

                self.last_aggregation_time = current_time

            return aggregated_measurement

        except Exception as e:
            logger.error(f"Failed to aggregate energy measurements: {e}")
            return None

    def _create_local_measurement(self) -> Optional[DistributedEnergyMeasurement]:
        """Create measurement for non-distributed case."""
        try:
            current_time = time.time()

            # Get local measurements
            local_measurements = {}
            total_power = 0.0
            total_energy = 0.0

            for device_id in self.local_tracker.devices:
                recent = self.local_tracker.get_recent_measurements(device_id, count=1)
                if recent:
                    measurement = recent[-1]
                    local_measurements[device_id] = measurement
                    total_power += measurement.power_watts
                    total_energy += measurement.cumulative_energy_joules

            if not local_measurements:
                return None

            # Create distributed measurement
            distributed_measurement = DistributedEnergyMeasurement(
                timestamp=current_time,
                aggregated_power_watts=total_power,
                aggregated_energy_joules=total_energy,
                parallelism_info=self.parallelism_info.copy(),
            )

            # Add local measurements
            for device_id, measurement in local_measurements.items():
                distributed_measurement.measurements_by_rank[device_id] = measurement

            return distributed_measurement

        except Exception as e:
            logger.error(f"Failed to create local measurement: {e}")
            return None

    def _perform_distributed_aggregation(
        self, local_measurement: EnergyMeasurement
    ) -> Optional[DistributedEnergyMeasurement]:
        """
        Perform distributed aggregation using all-gather.

        Args:
            local_measurement: Local energy measurement

        Returns:
            Aggregated measurement across all processes
        """
        try:
            # Prepare data for all-gather
            local_data = torch.tensor(
                [
                    local_measurement.timestamp,
                    local_measurement.power_watts,
                    local_measurement.cumulative_energy_joules,
                    float(self.global_rank),
                ],
                dtype=torch.float32,
            )

            # All-gather across all processes
            gathered_data = [
                torch.zeros_like(local_data) for _ in range(self.global_size)
            ]
            dist.all_gather(gathered_data, local_data)

            # Process gathered data
            distributed_measurement = DistributedEnergyMeasurement(
                timestamp=local_measurement.timestamp,
                parallelism_info=self.parallelism_info.copy(),
            )

            total_power = 0.0
            total_energy = 0.0

            for rank_data in gathered_data:
                rank_data_list = rank_data.tolist()
                rank_timestamp = rank_data_list[0]
                rank_power = rank_data_list[1]
                rank_energy = rank_data_list[2]
                rank_id = int(rank_data_list[3])

                # Create measurement for this rank
                rank_measurement = EnergyMeasurement(
                    timestamp=rank_timestamp,
                    power_watts=rank_power,
                    device_id=rank_id,
                    cumulative_energy_joules=rank_energy,
                )

                distributed_measurement.measurements_by_rank[rank_id] = rank_measurement
                total_power += rank_power
                total_energy += rank_energy

            distributed_measurement.aggregated_power_watts = total_power
            distributed_measurement.aggregated_energy_joules = total_energy

            return distributed_measurement

        except Exception as e:
            logger.error(f"Failed to perform distributed aggregation: {e}")
            return None

    def get_hierarchical_energy_report(self) -> Dict[str, Dict]:
        """
        Get hierarchical energy report organized by parallelism dimensions.

        Returns:
            Nested dictionary with energy statistics by parallelism type
        """
        if not self.enable_hierarchical_reporting or not self.distributed_measurements:
            return {}

        try:
            latest_measurement = self.distributed_measurements[-1]
            report = {}

            # Global statistics
            report["global"] = {
                "total_power_watts": latest_measurement.aggregated_power_watts,
                "total_energy_joules": latest_measurement.aggregated_energy_joules,
                "average_power_per_process": latest_measurement.average_power,
                "total_processes": latest_measurement.total_processes,
                "measurement_timestamp": latest_measurement.timestamp,
            }

            # Per-parallelism dimension statistics
            for ptype, pinfo in self.parallelism_info.items():
                dimension_report = self._calculate_dimension_statistics(
                    latest_measurement, ptype, pinfo
                )
                if dimension_report:
                    report[ptype.value] = dimension_report

            return report

        except Exception as e:
            logger.error(f"Failed to generate hierarchical energy report: {e}")
            return {}

    def _calculate_dimension_statistics(
        self,
        measurement: DistributedEnergyMeasurement,
        ptype: ParallelismType,
        pinfo: ParallelismInfo,
    ) -> Optional[Dict]:
        """Calculate energy statistics for a specific parallelism dimension."""
        try:
            # For now, provide basic statistics
            # More sophisticated hierarchical analysis could be added here

            dimension_stats = {
                "parallelism_type": ptype.value,
                "size": pinfo.size,
                "local_rank": pinfo.rank,
                "power_contribution_watts": 0.0,
                "energy_contribution_joules": 0.0,
            }

            # Calculate contribution from processes in this dimension
            # Simplified approach - more sophisticated grouping possible
            if self.global_rank in measurement.measurements_by_rank:
                local_measurement = measurement.measurements_by_rank[self.global_rank]
                dimension_stats[
                    "power_contribution_watts"
                ] = local_measurement.power_watts
                dimension_stats[
                    "energy_contribution_joules"
                ] = local_measurement.cumulative_energy_joules

            return dimension_stats

        except Exception as e:
            logger.debug(f"Failed to calculate dimension statistics for {ptype}: {e}")
            return None

    def get_recent_measurements(
        self, count: int = 10
    ) -> List[DistributedEnergyMeasurement]:
        """Get recent distributed energy measurements."""
        if count <= 0:
            return []

        measurements = self.distributed_measurements
        return (
            measurements[-count:] if len(measurements) >= count else measurements.copy()
        )

    def should_aggregate(self) -> bool:
        """Check if it's time to perform aggregation."""
        current_time = time.time()
        return (current_time - self.last_aggregation_time) >= self.aggregation_interval

    @contextmanager
    def monitoring_context(self):
        """Context manager for energy monitoring."""
        self.start_monitoring()
        try:
            yield self
        finally:
            self.stop_monitoring()

    def get_energy_efficiency_metrics(self) -> Dict[str, float]:
        """
        Calculate energy efficiency metrics.

        Returns:
            Dictionary containing efficiency metrics
        """
        if not self.distributed_measurements:
            return {}

        try:
            # Calculate metrics over recent measurements
            recent_measurements = self.get_recent_measurements(
                count=min(10, len(self.distributed_measurements))
            )

            if len(recent_measurements) < 2:
                return {}

            # Time span
            time_span = (
                recent_measurements[-1].timestamp - recent_measurements[0].timestamp
            )
            if time_span <= 0:
                return {}

            # Power trends
            powers = [m.aggregated_power_watts for m in recent_measurements]
            avg_power = sum(powers) / len(powers)
            max_power = max(powers)
            min_power = min(powers)

            # Energy efficiency (simplified)
            total_energy = recent_measurements[-1].aggregated_energy_joules
            energy_per_second = total_energy / time_span if time_span > 0 else 0

            return {
                "average_power_watts": avg_power,
                "peak_power_watts": max_power,
                "min_power_watts": min_power,
                "power_variance": sum((p - avg_power) ** 2 for p in powers)
                / len(powers),
                "energy_per_second_joules": energy_per_second,
                "total_energy_joules": total_energy,
                "measurement_duration_seconds": time_span,
                "processes_count": recent_measurements[-1].total_processes,
            }

        except Exception as e:
            logger.error(f"Failed to calculate efficiency metrics: {e}")
            return {}

    def reset_measurements(self):
        """Reset all distributed measurements."""
        self.distributed_measurements.clear()
        self.last_aggregation_time = 0.0
        self.local_tracker.reset_measurements()
        logger.info(f"Distributed energy measurements reset (rank {self.global_rank})")

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
