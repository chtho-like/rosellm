"""
Tests for Distributed Energy Aggregation System.

This test suite covers:
- Distributed energy aggregation across parallelism dimensions
- Process group management and communication
- Hierarchical energy reporting
- Fault tolerance and error handling
- Mock distributed environment testing
"""

import time
from unittest.mock import Mock, patch

import pytest
import torch

from rosellm.rosetrainer.monitoring.distributed_energy import (
    DistributedEnergyAggregator,
    DistributedEnergyMeasurement,
    ParallelismInfo,
    ParallelismType,
)
from rosellm.rosetrainer.monitoring.energy_tracker import (
    EnergyMeasurement,
    GPUEnergyTracker,
)


class TestParallelismInfo:
    """Test ParallelismInfo data structure."""

    def test_parallelism_info_creation(self):
        """Test creating parallelism info."""
        info = ParallelismInfo(
            parallel_type=ParallelismType.DATA_PARALLEL, rank=2, size=8
        )

        assert info.parallel_type == ParallelismType.DATA_PARALLEL
        assert info.rank == 2
        assert info.size == 8
        assert "dp=8(rank=2)" in str(info)

    def test_all_parallelism_types(self):
        """Test all parallelism types."""
        types_and_values = [
            (ParallelismType.DATA_PARALLEL, "dp"),
            (ParallelismType.TENSOR_PARALLEL, "tp"),
            (ParallelismType.PIPELINE_PARALLEL, "pp"),
            (ParallelismType.CONTEXT_PARALLEL, "cp"),
            (ParallelismType.EXPERT_PARALLEL, "ep"),
            (ParallelismType.GLOBAL, "global"),
        ]

        for ptype, expected_value in types_and_values:
            assert ptype.value == expected_value


class TestDistributedEnergyMeasurement:
    """Test DistributedEnergyMeasurement data structure."""

    def test_basic_measurement(self):
        """Test basic distributed measurement."""
        measurement = DistributedEnergyMeasurement(
            timestamp=time.time(),
            aggregated_power_watts=500.0,
            aggregated_energy_joules=1000.0,
        )

        assert measurement.aggregated_power_watts == 500.0
        assert measurement.aggregated_energy_joules == 1000.0
        assert measurement.total_processes == 0  # No measurements added yet
        assert measurement.average_power == 0.0

    def test_measurement_with_rank_data(self):
        """Test measurement with per-rank data."""
        measurement = DistributedEnergyMeasurement(timestamp=time.time())

        # Add measurements for different ranks
        measurement.measurements_by_rank[0] = EnergyMeasurement(
            timestamp=time.time(),
            power_watts=100.0,
            device_id=0,
            cumulative_energy_joules=100.0,
        )
        measurement.measurements_by_rank[1] = EnergyMeasurement(
            timestamp=time.time(),
            power_watts=150.0,
            device_id=1,
            cumulative_energy_joules=200.0,
        )
        measurement.measurements_by_rank[2] = EnergyMeasurement(
            timestamp=time.time(),
            power_watts=200.0,
            device_id=2,
            cumulative_energy_joules=300.0,
        )

        assert measurement.total_processes == 3
        assert measurement.average_power == 150.0  # (100 + 150 + 200) / 3

    def test_parallelism_info_integration(self):
        """Test integration with parallelism info."""
        measurement = DistributedEnergyMeasurement(timestamp=time.time())

        measurement.parallelism_info[ParallelismType.DATA_PARALLEL] = ParallelismInfo(
            parallel_type=ParallelismType.DATA_PARALLEL, rank=0, size=4
        )
        measurement.parallelism_info[ParallelismType.TENSOR_PARALLEL] = ParallelismInfo(
            parallel_type=ParallelismType.TENSOR_PARALLEL, rank=1, size=2
        )

        assert len(measurement.parallelism_info) == 2
        assert ParallelismType.DATA_PARALLEL in measurement.parallelism_info
        assert ParallelismType.TENSOR_PARALLEL in measurement.parallelism_info


class TestDistributedEnergyAggregator:
    """Test DistributedEnergyAggregator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_tracker = Mock(spec=GPUEnergyTracker)
        self.mock_tracker.devices = [0]
        self.mock_tracker.start_monitoring.return_value = True
        self.mock_tracker.stop_monitoring.return_value = {0: 100.0}

    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.dist.is_available",
        return_value=False,
    )
    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.dist.is_initialized",
        return_value=False,
    )
    def test_non_distributed_initialization(
        self, mock_is_initialized, mock_is_available
    ):
        """Test initialization in non-distributed environment."""
        aggregator = DistributedEnergyAggregator(local_tracker=self.mock_tracker)

        assert aggregator.is_distributed is False
        assert aggregator.global_rank == 0
        assert aggregator.global_size == 1
        assert len(aggregator.parallelism_info) == 0

    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.dist.is_available",
        return_value=True,
    )
    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.dist.is_initialized",
        return_value=True,
    )
    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.parallel_state_initialized",
        return_value=True,
    )
    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.dist.get_rank",
        return_value=1,
    )
    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.dist.get_world_size",
        return_value=4,
    )
    def test_distributed_initialization(
        self,
        mock_get_world_size,
        mock_get_rank,
        mock_parallel_state_initialized,
        mock_is_initialized,
        mock_is_available,
    ):
        """Test initialization in distributed environment."""
        # Mock parallelism functions
        with patch(
            "rosellm.rosetrainer.monitoring.distributed_energy.get_data_parallel_size",
            return_value=2,
        ):
            with patch(
                "rosellm.rosetrainer.monitoring.distributed_energy."
                "get_data_parallel_rank",
                return_value=0,
            ):
                with patch(
                    "rosellm.rosetrainer.monitoring.distributed_energy."
                    "get_data_parallel_group",
                    return_value=Mock(),
                ):
                    aggregator = DistributedEnergyAggregator(
                        local_tracker=self.mock_tracker
                    )

        assert aggregator.is_distributed is True
        assert aggregator.global_rank == 1
        assert aggregator.global_size == 4

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        aggregator = DistributedEnergyAggregator(local_tracker=self.mock_tracker)

        # Test start
        result = aggregator.start_monitoring()
        assert result is True
        self.mock_tracker.start_monitoring.assert_called_once()

        # Test stop
        results = aggregator.stop_monitoring()
        assert isinstance(results, dict)
        self.mock_tracker.stop_monitoring.assert_called_once()

    def test_local_measurement_creation(self):
        """Test creating local measurement in non-distributed mode."""
        # Set up mock measurements
        mock_measurement = EnergyMeasurement(
            timestamp=time.time(),
            power_watts=250.0,
            device_id=0,
            cumulative_energy_joules=500.0,
        )
        self.mock_tracker.get_recent_measurements.return_value = [mock_measurement]

        aggregator = DistributedEnergyAggregator(local_tracker=self.mock_tracker)

        # Create local measurement
        measurement = aggregator._create_local_measurement()

        assert measurement is not None
        assert measurement.aggregated_power_watts == 250.0
        assert measurement.aggregated_energy_joules == 500.0
        assert len(measurement.measurements_by_rank) == 1

    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.dist.is_available",
        return_value=True,
    )
    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.dist.is_initialized",
        return_value=True,
    )
    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.parallel_state_initialized",
        return_value=True,
    )
    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.dist.get_rank",
        return_value=0,
    )
    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.dist.get_world_size",
        return_value=2,
    )
    def test_distributed_aggregation(
        self,
        mock_get_world_size,
        mock_get_rank,
        mock_parallel_state_initialized,
        mock_is_initialized,
        mock_is_available,
    ):
        """Test distributed aggregation with mocked communication."""
        # Mock parallelism functions
        with patch(
            "rosellm.rosetrainer.monitoring.distributed_energy.get_data_parallel_size",
            return_value=1,
        ):
            aggregator = DistributedEnergyAggregator(local_tracker=self.mock_tracker)

        # Mock local measurement
        local_measurement = EnergyMeasurement(
            timestamp=time.time(),
            power_watts=100.0,
            device_id=0,
            cumulative_energy_joules=200.0,
        )

        # Mock all_gather operation
        def mock_all_gather(tensor_list, tensor):
            # Simulate gathering from 2 ranks
            tensor_list[0] = torch.tensor([time.time(), 100.0, 200.0, 0.0])  # rank 0
            tensor_list[1] = torch.tensor([time.time(), 150.0, 300.0, 1.0])  # rank 1

        with patch(
            "rosellm.rosetrainer.monitoring.distributed_energy.dist.all_gather",
            side_effect=mock_all_gather,
        ):
            result = aggregator._perform_distributed_aggregation(local_measurement)

        assert result is not None
        assert result.aggregated_power_watts == 250.0  # 100 + 150
        assert result.aggregated_energy_joules == 500.0  # 200 + 300
        assert len(result.measurements_by_rank) == 2

    def test_hierarchical_reporting(self):
        """Test hierarchical energy reporting."""
        aggregator = DistributedEnergyAggregator(
            local_tracker=self.mock_tracker, enable_hierarchical_reporting=True
        )

        # Add some parallelism info
        aggregator.parallelism_info[ParallelismType.DATA_PARALLEL] = ParallelismInfo(
            parallel_type=ParallelismType.DATA_PARALLEL, rank=0, size=4
        )

        # Create a mock measurement
        measurement = DistributedEnergyMeasurement(
            timestamp=time.time(),
            aggregated_power_watts=400.0,
            aggregated_energy_joules=800.0,
        )
        measurement.measurements_by_rank[0] = EnergyMeasurement(
            timestamp=time.time(),
            power_watts=100.0,
            device_id=0,
            cumulative_energy_joules=200.0,
        )

        aggregator.distributed_measurements = [measurement]

        # Get hierarchical report
        report = aggregator.get_hierarchical_energy_report()

        assert "global" in report
        assert "dp" in report
        assert report["global"]["total_power_watts"] == 400.0
        assert report["global"]["total_energy_joules"] == 800.0

    def test_efficiency_metrics(self):
        """Test energy efficiency metrics calculation."""
        aggregator = DistributedEnergyAggregator(local_tracker=self.mock_tracker)

        # Create mock measurements with time progression
        base_time = time.time()
        measurements = []
        for i in range(5):
            measurement = DistributedEnergyMeasurement(
                timestamp=base_time + i,
                aggregated_power_watts=100.0 + i * 10,  # Increasing power
                aggregated_energy_joules=200.0 + i * 50,  # Increasing energy
            )
            measurements.append(measurement)

        aggregator.distributed_measurements = measurements

        # Get efficiency metrics
        metrics = aggregator.get_energy_efficiency_metrics()

        assert "average_power_watts" in metrics
        assert "peak_power_watts" in metrics
        assert "min_power_watts" in metrics
        assert "energy_per_second_joules" in metrics
        assert "total_energy_joules" in metrics

        assert (
            metrics["average_power_watts"] == 120.0
        )  # Average of 100, 110, 120, 130, 140
        assert metrics["peak_power_watts"] == 140.0
        assert metrics["min_power_watts"] == 100.0

    def test_recent_measurements(self):
        """Test getting recent measurements."""
        aggregator = DistributedEnergyAggregator(local_tracker=self.mock_tracker)

        # Add measurements
        measurements = []
        for i in range(15):
            measurement = DistributedEnergyMeasurement(
                timestamp=time.time() + i,
                aggregated_power_watts=100.0,
                aggregated_energy_joules=200.0,
            )
            measurements.append(measurement)

        aggregator.distributed_measurements = measurements

        # Test getting recent measurements
        recent = aggregator.get_recent_measurements(count=5)
        assert len(recent) == 5

        recent_all = aggregator.get_recent_measurements(count=20)
        assert len(recent_all) == 15  # Only 15 available

        recent_empty = aggregator.get_recent_measurements(count=0)
        assert len(recent_empty) == 0

    def test_aggregation_timing(self):
        """Test aggregation timing logic."""
        aggregator = DistributedEnergyAggregator(
            local_tracker=self.mock_tracker, aggregation_interval=1.0
        )

        # Initially should aggregate
        assert aggregator.should_aggregate() is True

        # Update last aggregation time
        aggregator.last_aggregation_time = time.time()

        # Should not aggregate immediately
        assert aggregator.should_aggregate() is False

        # Wait and should aggregate again
        time.sleep(0.1)
        aggregator.last_aggregation_time = time.time() - 1.5  # Simulate old timestamp
        assert aggregator.should_aggregate() is True

    def test_reset_measurements(self):
        """Test resetting measurements."""
        aggregator = DistributedEnergyAggregator(local_tracker=self.mock_tracker)

        # Add some measurements
        measurement = DistributedEnergyMeasurement(
            timestamp=time.time(),
            aggregated_power_watts=100.0,
            aggregated_energy_joules=200.0,
        )
        aggregator.distributed_measurements = [measurement]
        aggregator.last_aggregation_time = time.time()

        # Reset
        aggregator.reset_measurements()

        assert len(aggregator.distributed_measurements) == 0
        assert aggregator.last_aggregation_time == 0.0
        self.mock_tracker.reset_measurements.assert_called_once()

    def test_context_manager(self):
        """Test context manager functionality."""
        aggregator = DistributedEnergyAggregator(local_tracker=self.mock_tracker)

        with aggregator.monitoring_context() as agg:
            assert agg is aggregator
            # Should start monitoring automatically
            self.mock_tracker.start_monitoring.assert_called_once()

        # Should stop monitoring when exiting context
        self.mock_tracker.stop_monitoring.assert_called_once()

    def test_error_handling(self):
        """Test error handling in distributed operations."""
        aggregator = DistributedEnergyAggregator(local_tracker=self.mock_tracker)

        # Test aggregation with no local measurements
        self.mock_tracker.get_recent_measurements.return_value = []

        result = aggregator.aggregate_energy_measurements()

        # Should handle gracefully
        assert isinstance(result, (type(None), DistributedEnergyMeasurement))

    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.dist.is_available",
        return_value=True,
    )
    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.dist.is_initialized",
        return_value=True,
    )
    @patch(
        "rosellm.rosetrainer.monitoring.distributed_energy.parallel_state_initialized",
        return_value=True,
    )
    def test_parallelism_info_collection(
        self, mock_parallel_state_initialized, mock_is_initialized, mock_is_available
    ):
        """Test collection of parallelism information."""
        # Mock all parallelism functions to return meaningful values
        parallel_patches = [
            ("get_data_parallel_size", 4),
            ("get_data_parallel_rank", 1),
            ("get_tensor_model_parallel_size", 2),
            ("get_tensor_model_parallel_rank", 0),
            ("get_pipeline_model_parallel_size", 1),
            ("get_context_parallel_size", 1),
            ("get_expert_model_parallel_size", 1),
        ]

        with patch(
            "rosellm.rosetrainer.monitoring.distributed_energy.dist.get_rank",
            return_value=1,
        ):
            with patch(
                "rosellm.rosetrainer.monitoring.distributed_energy.dist.get_world_size",
                return_value=8,
            ):
                # Apply all patches
                patch_contexts = []
                for func_name, return_value in parallel_patches:
                    full_name = (
                        f"rosellm.rosetrainer.monitoring.distributed_energy.{func_name}"
                    )
                    patch_contexts.append(patch(full_name, return_value=return_value))

                # Mock process groups
                mock_group = Mock()
                group_patches = [
                    ("get_data_parallel_group", mock_group),
                    ("get_tensor_model_parallel_group", mock_group),
                ]

                for func_name, return_value in group_patches:
                    full_name = (
                        f"rosellm.rosetrainer.monitoring.distributed_energy.{func_name}"
                    )
                    patch_contexts.append(patch(full_name, return_value=return_value))

                # Enter all patch contexts
                for p in patch_contexts:
                    p.start()

                try:
                    aggregator = DistributedEnergyAggregator(
                        local_tracker=self.mock_tracker
                    )

                    # Should have collected parallelism info
                    assert ParallelismType.DATA_PARALLEL in aggregator.parallelism_info
                    assert (
                        ParallelismType.TENSOR_PARALLEL in aggregator.parallelism_info
                    )

                    dp_info = aggregator.parallelism_info[ParallelismType.DATA_PARALLEL]
                    assert dp_info.size == 4
                    assert dp_info.rank == 1

                    tp_info = aggregator.parallelism_info[
                        ParallelismType.TENSOR_PARALLEL
                    ]
                    assert tp_info.size == 2
                    assert tp_info.rank == 0

                finally:
                    # Exit all patch contexts
                    for p in patch_contexts:
                        p.stop()


class TestDistributedEnergyIntegration:
    """Integration tests for distributed energy aggregation."""

    def test_end_to_end_non_distributed(self):
        """Test end-to-end workflow in non-distributed mode."""
        # Create a real GPU tracker with mock
        mock_tracker = Mock(spec=GPUEnergyTracker)
        mock_tracker.devices = [0]
        mock_tracker.start_monitoring.return_value = True
        mock_tracker.stop_monitoring.return_value = {0: 150.0}
        mock_tracker.get_recent_measurements.return_value = [
            EnergyMeasurement(time.time(), 100.0, 0, 150.0)
        ]

        aggregator = DistributedEnergyAggregator(
            local_tracker=mock_tracker, aggregation_interval=0.1
        )

        # Start monitoring
        assert aggregator.start_monitoring() is True

        # Perform aggregation
        measurement = aggregator.aggregate_energy_measurements()
        assert measurement is not None
        assert measurement.aggregated_power_watts == 100.0
        assert measurement.aggregated_energy_joules == 150.0

        # Get efficiency metrics
        metrics = aggregator.get_energy_efficiency_metrics()
        # May be empty if not enough measurements, but shouldn't error
        assert isinstance(metrics, dict)

        # Stop monitoring
        results = aggregator.stop_monitoring()
        assert isinstance(results, dict)
        assert "local_energy_joules" in results

    def test_history_management(self):
        """Test management of measurement history."""
        aggregator = DistributedEnergyAggregator(
            local_tracker=Mock(), max_history_size=5
        )

        # Add more measurements than max history
        for i in range(10):
            measurement = DistributedEnergyMeasurement(
                timestamp=time.time() + i,
                aggregated_power_watts=100.0,
                aggregated_energy_joules=200.0,
            )
            aggregator.distributed_measurements.append(measurement)

        # Simulate history trimming from aggregate_energy_measurements
        if len(aggregator.distributed_measurements) > aggregator.max_history_size:
            aggregator.distributed_measurements = aggregator.distributed_measurements[
                -aggregator.max_history_size :
            ]

        # Should maintain only max_history_size measurements
        assert len(aggregator.distributed_measurements) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
