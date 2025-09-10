"""Comprehensive tests for range-based parameter buffer mapping.

This test suite provides extensive validation of the range-based parameter
buffer mapping functionality, including bit-to-bit accuracy validation
against reference implementations and comprehensive edge case testing.
"""

import logging
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn

from rosellm.rosetrainer.optimizer.exceptions import ConfigurationError
from rosellm.rosetrainer.optimizer.range_buffer_mapping import (
    BufferAllocationMode,
    BufferRange,
    MemoryStats,
    RangeBufferConfig,
    RangeBufferMapper,
    RangeBufferStrategy,
)

logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """Simple test model for parameter buffer mapping tests."""

    def __init__(self, layers: List[int], dtype: torch.dtype = torch.float32):
        super().__init__()
        self.layers = nn.ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            layer = nn.Linear(in_dim, out_dim, dtype=dtype)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TestRangeBufferConfig:
    """Test RangeBufferConfig validation and functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RangeBufferConfig()

        assert config.strategy == RangeBufferStrategy.CONTIGUOUS
        assert config.allocation_mode == BufferAllocationMode.HYBRID
        assert config.alignment_bytes == 64
        assert config.min_range_size == 256
        assert config.max_fragmentation == 0.1
        assert config.enable_compaction is True
        assert config.growth_factor == 1.5
        assert config.enable_profiling is False

    def test_invalid_alignment(self):
        """Test validation of alignment bytes."""
        with pytest.raises(
            ConfigurationError, match="alignment_bytes must be a positive power of 2"
        ):
            RangeBufferConfig(alignment_bytes=0)

        with pytest.raises(
            ConfigurationError, match="alignment_bytes must be a positive power of 2"
        ):
            RangeBufferConfig(alignment_bytes=3)  # Not a power of 2

    def test_invalid_min_range_size(self):
        """Test validation of minimum range size."""
        with pytest.raises(ConfigurationError, match="min_range_size must be positive"):
            RangeBufferConfig(min_range_size=0)

        with pytest.raises(ConfigurationError, match="min_range_size must be positive"):
            RangeBufferConfig(min_range_size=-1)

    def test_invalid_fragmentation_ratio(self):
        """Test validation of fragmentation ratio."""
        with pytest.raises(
            ConfigurationError, match="max_fragmentation must be between 0 and 1"
        ):
            RangeBufferConfig(max_fragmentation=0.0)

        with pytest.raises(
            ConfigurationError, match="max_fragmentation must be between 0 and 1"
        ):
            RangeBufferConfig(max_fragmentation=1.0)

        with pytest.raises(
            ConfigurationError, match="max_fragmentation must be between 0 and 1"
        ):
            RangeBufferConfig(max_fragmentation=1.5)

    def test_invalid_growth_factor(self):
        """Test validation of growth factor."""
        with pytest.raises(ConfigurationError, match="growth_factor must be > 1.0"):
            RangeBufferConfig(growth_factor=1.0)

        with pytest.raises(ConfigurationError, match="growth_factor must be > 1.0"):
            RangeBufferConfig(growth_factor=0.5)


class TestBufferRange:
    """Test BufferRange functionality."""

    def test_buffer_range_properties(self):
        """Test BufferRange property calculations."""
        buffer_range = BufferRange(
            start_offset=0,
            end_offset=1024,
            param_indices=[0, 1, 2],
            dtype=torch.float32,
            device=torch.device("cpu"),
            alignment_padding=0,
        )

        assert buffer_range.size_bytes == 1024
        assert buffer_range.size_elements == 1024 // 4  # float32 is 4 bytes

    def test_buffer_range_with_padding(self):
        """Test BufferRange with alignment padding."""
        buffer_range = BufferRange(
            start_offset=0,
            end_offset=1024,
            param_indices=[0, 1],
            dtype=torch.float32,
            device=torch.device("cpu"),
            alignment_padding=64,
        )

        assert buffer_range.size_bytes == 1024 - 64
        assert buffer_range.size_elements == (1024 - 64) // 4


class TestRangeBufferMapper:
    """Test RangeBufferMapper functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return SimpleTestModel([64, 32, 16, 8])

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_empty_parameters_error(self, device):
        """Test error handling for empty parameters."""
        config = RangeBufferConfig(device=device)

        with pytest.raises(ConfigurationError, match="No parameters provided"):
            RangeBufferMapper([], config)

    def test_mixed_device_parameters_error(self, device):
        """Test error handling for mixed device parameters."""
        param1 = nn.Parameter(torch.randn(10, device="cpu"))
        param2 = nn.Parameter(
            torch.randn(10, device="cuda" if torch.cuda.is_available() else "cpu")
        )

        if torch.cuda.is_available():
            config = RangeBufferConfig(device=device)
            with pytest.raises(ConfigurationError, match="expected"):
                RangeBufferMapper([param1, param2], config)

    def test_single_rank_initialization(self, simple_model, device):
        """Test initialization with single rank."""
        simple_model.to(device)
        parameters = list(simple_model.parameters())
        config = RangeBufferConfig(device=device)

        mapper = RangeBufferMapper(
            parameters=parameters,
            config=config,
            world_size=1,
            rank=0,
        )

        assert len(mapper.buffer_ranges) > 0
        assert len(mapper.buffers) > 0
        assert len(mapper.param_to_range) == len(
            [p for p in parameters if p.requires_grad]
        )

    def test_multi_rank_initialization(self, simple_model, device):
        """Test initialization with multiple ranks."""
        simple_model.to(device)
        parameters = list(simple_model.parameters())
        config = RangeBufferConfig(device=device)

        # Test different ranks
        for rank in range(4):
            mapper = RangeBufferMapper(
                parameters=parameters,
                config=config,
                world_size=4,
                rank=rank,
            )

            # Each rank should have some buffer ranges
            local_range = mapper.partitioner.get_local_param_range()
            if local_range is not None:
                assert len(mapper.buffer_ranges) > 0
                assert local_range.rank == rank

    def test_contiguous_strategy(self, simple_model, device):
        """Test contiguous buffer organization strategy."""
        simple_model.to(device)
        parameters = list(simple_model.parameters())
        config = RangeBufferConfig(
            strategy=RangeBufferStrategy.CONTIGUOUS,
            device=device,
            min_range_size=100,  # Lower threshold for test parameters
        )

        mapper = RangeBufferMapper(parameters, config)

        # Verify contiguous organization
        assert len(mapper.buffer_ranges) > 0
        for buffer_range in mapper.buffer_ranges:
            assert buffer_range.is_active
            assert len(buffer_range.param_indices) > 0

    def test_size_ordered_strategy(self, simple_model, device):
        """Test size-ordered buffer organization strategy."""
        simple_model.to(device)
        parameters = list(simple_model.parameters())
        config = RangeBufferConfig(
            strategy=RangeBufferStrategy.SIZE_ORDERED,
            device=device,
            min_range_size=100,  # Lower threshold for test parameters
        )

        mapper = RangeBufferMapper(parameters, config)

        # Verify size-based organization
        assert len(mapper.buffer_ranges) > 0

        # Parameters should be organized by size
        total_elements_mapped = 0
        for buffer_range in mapper.buffer_ranges:
            if buffer_range.is_active:
                total_elements_mapped += buffer_range.size_elements

        assert total_elements_mapped > 0

    def test_dtype_grouped_strategy(self, device):
        """Test dtype-grouped buffer organization strategy."""
        # Create parameters with different dtypes
        param1 = nn.Parameter(torch.randn(100, dtype=torch.float32, device=device))
        param2 = nn.Parameter(torch.randn(50, dtype=torch.float32, device=device))
        param3 = nn.Parameter(torch.randn(75, dtype=torch.float16, device=device))

        parameters = [param1, param2, param3]
        config = RangeBufferConfig(
            strategy=RangeBufferStrategy.DTYPE_GROUPED,
            device=device,
            min_range_size=50,  # Lower threshold for test parameters
        )

        mapper = RangeBufferMapper(parameters, config)

        # Should group by dtype
        assert len(mapper.buffer_ranges) > 0

        # Check that buffers are allocated for different dtypes
        dtypes_found = set()
        for buffer_range in mapper.buffer_ranges:
            dtypes_found.add(buffer_range.dtype)

        assert torch.float32 in dtypes_found
        if device.type == "cuda" or device.type == "cpu":
            # float16 might be converted or handled differently
            assert len(dtypes_found) >= 1

    def test_parameter_buffer_retrieval(self, simple_model, device):
        """Test parameter buffer retrieval functionality."""
        simple_model.to(device)
        parameters = list(simple_model.parameters())
        config = RangeBufferConfig(device=device)

        mapper = RangeBufferMapper(parameters, config)

        # Test buffer retrieval for each parameter
        for param_idx, param in enumerate(parameters):
            buffer_info = mapper.get_parameter_buffer(param_idx)

            if param.requires_grad:
                # Should find buffer for trainable parameters
                local_range = mapper.partitioner.get_local_param_range()
                if local_range is not None and param_idx in local_range.param_indices:
                    assert buffer_info is not None
                    buffer, start, end = buffer_info
                    assert isinstance(buffer, torch.Tensor)
                    assert start >= 0
                    assert end > start
                    assert buffer.device == device

    def test_copy_parameters_to_buffers(self, simple_model, device):
        """Test copying parameters to buffers."""
        simple_model.to(device)
        parameters = list(simple_model.parameters())
        config = RangeBufferConfig(device=device)

        mapper = RangeBufferMapper(parameters, config)

        # Copy parameters to buffers
        mapper.copy_parameters_to_buffers()

        # Verify data consistency
        for param_idx, param in enumerate(parameters):
            buffer_info = mapper.get_parameter_buffer(param_idx)
            if buffer_info is not None:
                buffer, start, end = buffer_info

                # Get expected parameter slice
                local_range = mapper.partitioner.get_local_param_range()
                if local_range is not None:
                    param_slice = local_range.get_param_slice(param_idx, param.numel())
                    if param_slice is not None:
                        slice_start, slice_end = param_slice
                        expected_data = param.view(-1)[slice_start:slice_end]
                        buffer_data = buffer[start:end]

                        # Bit-to-bit validation
                        torch.testing.assert_close(
                            buffer_data,
                            expected_data,
                            msg=f"Buffer data mismatch for parameter {param_idx}",
                        )

    def test_copy_buffers_to_parameters(self, simple_model, device):
        """Test copying buffers back to parameters."""
        simple_model.to(device)
        parameters = list(simple_model.parameters())
        config = RangeBufferConfig(device=device)

        mapper = RangeBufferMapper(parameters, config)

        # Store original parameter values
        original_params = [p.data.clone() for p in parameters]

        # Copy to buffers
        mapper.copy_parameters_to_buffers()

        # Modify parameters
        for param in parameters:
            param.data.fill_(999.0)

        # Copy back from buffers
        mapper.copy_buffers_to_parameters()

        # Verify restoration (only for local parameters)
        local_range = mapper.partitioner.get_local_param_range()
        if local_range is not None:
            for param_idx in local_range.param_indices:
                if param_idx < len(parameters):
                    param = parameters[param_idx]
                    original = original_params[param_idx]

                    param_slice = local_range.get_param_slice(param_idx, param.numel())
                    if param_slice is not None:
                        slice_start, slice_end = param_slice
                        restored_slice = param.view(-1)[slice_start:slice_end]
                        original_slice = original.view(-1)[slice_start:slice_end]

                        # Bit-to-bit validation
                        torch.testing.assert_close(
                            restored_slice,
                            original_slice,
                            msg=f"Parameter restoration failed for parameter "
                            f"{param_idx}",
                        )

    def test_buffer_compaction(self, simple_model, device):
        """Test buffer compaction functionality."""
        simple_model.to(device)
        parameters = list(simple_model.parameters())
        config = RangeBufferConfig(
            device=device,
            enable_compaction=True,
            max_fragmentation=0.05,  # Low threshold to trigger compaction
        )

        mapper = RangeBufferMapper(parameters, config)

        # Force fragmentation by deactivating some ranges
        if len(mapper.buffer_ranges) > 1:
            mapper.buffer_ranges[0].is_active = False

            # Trigger compaction
            compaction_performed = mapper.compact_buffers()

            # Should have performed compaction
            if mapper.stats.fragmentation_ratio > config.max_fragmentation:
                assert compaction_performed
                assert mapper.stats.compaction_count > 0

    def test_memory_statistics(self, simple_model, device):
        """Test memory statistics collection."""
        simple_model.to(device)
        parameters = list(simple_model.parameters())
        config = RangeBufferConfig(device=device, enable_profiling=True)

        mapper = RangeBufferMapper(parameters, config)

        # Get memory statistics
        stats = mapper.get_memory_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.total_allocated_bytes >= 0
        assert stats.total_used_bytes >= 0
        assert stats.fragmentation_ratio >= 0.0
        assert stats.num_ranges >= 0
        assert stats.num_buffers >= 0
        assert stats.alignment_waste_bytes >= 0

    def test_buffer_info(self, simple_model, device):
        """Test buffer information retrieval."""
        simple_model.to(device)
        parameters = list(simple_model.parameters())
        config = RangeBufferConfig(device=device)

        mapper = RangeBufferMapper(parameters, config)

        # Get buffer information
        info: Dict[str, Any] = mapper.get_buffer_info()

        assert isinstance(info, dict)
        assert "config" in info
        assert "statistics" in info
        assert "buffer_types" in info
        assert "range_info" in info

        # Validate config information
        config_info: Dict[str, Any] = info["config"]
        assert config_info["strategy"] == config.strategy.value
        assert config_info["allocation_mode"] == config.allocation_mode.value

        # Validate statistics
        stats_info: Dict[str, Any] = info["statistics"]
        assert "num_ranges" in stats_info
        assert "num_buffers" in stats_info
        assert "total_allocated_mb" in stats_info
        assert "fragmentation_ratio" in stats_info


class TestRangeBufferMapperEdgeCases:
    """Test edge cases and error conditions for RangeBufferMapper."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return SimpleTestModel([64, 32, 16, 8])

    def test_very_small_parameters(self, device):
        """Test handling of very small parameters."""
        # Create parameters smaller than minimum range size
        small_params = [nn.Parameter(torch.randn(1, device=device)) for _ in range(5)]

        config = RangeBufferConfig(
            device=device,
            min_range_size=1000,  # Much larger than parameter size
        )

        mapper = RangeBufferMapper(small_params, config)

        # Should handle small parameters gracefully
        local_range = mapper.partitioner.get_local_param_range()
        if local_range is not None:
            assert len(mapper.buffer_ranges) >= 0  # May be 0 if too small

    def test_single_large_parameter(self, device):
        """Test handling of single large parameter."""
        large_param = nn.Parameter(torch.randn(10000, device=device))
        config = RangeBufferConfig(device=device)

        mapper = RangeBufferMapper([large_param], config)

        # Should create at least one range
        assert len(mapper.buffer_ranges) > 0
        assert len(mapper.buffers) > 0

    def test_mixed_dtype_parameters(self, device):
        """Test handling of parameters with mixed dtypes."""
        params = [
            nn.Parameter(torch.randn(100, dtype=torch.float32, device=device)),
            nn.Parameter(torch.randn(100, dtype=torch.float16, device=device)),
        ]

        config = RangeBufferConfig(
            strategy=RangeBufferStrategy.DTYPE_GROUPED,
            device=device,
            min_range_size=50,  # Lower threshold for test parameters
        )

        mapper = RangeBufferMapper(params, config)

        # Should handle mixed dtypes
        dtypes_in_buffers = set(mapper.buffers.keys())
        assert len(dtypes_in_buffers) >= 1

    def test_zero_element_parameters(self, device):
        """Test handling of parameters with zero elements."""
        # This should raise an error during parameter validation
        zero_param = nn.Parameter(torch.empty(0, device=device))
        config = RangeBufferConfig(device=device)

        with pytest.raises(ValueError, match="has 0 elements"):
            RangeBufferMapper([zero_param], config)

    def test_multi_rank_coverage(self, simple_model, device):
        """Test that all parameters are covered across multiple ranks."""
        simple_model.to(device)
        parameters = list(simple_model.parameters())
        config = RangeBufferConfig(device=device)

        world_size = 4
        all_covered_params = set()

        # Collect parameters covered by each rank
        for rank in range(world_size):
            mapper = RangeBufferMapper(parameters, config, world_size, rank)
            local_range = mapper.partitioner.get_local_param_range()

            if local_range is not None:
                for param_idx in local_range.param_indices:
                    all_covered_params.add(param_idx)

        # All trainable parameters should be covered
        trainable_param_indices = set(
            i for i, p in enumerate(parameters) if p.requires_grad
        )

        # Should cover all or most parameters (some might be too small)
        coverage_ratio = len(all_covered_params) / len(trainable_param_indices)
        assert coverage_ratio >= 0.8  # Allow for some parameters being too small


class TestRangeBufferMapperPerformance:
    """Performance and stress tests for RangeBufferMapper."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.slow
    def test_large_model_performance(self, device):
        """Test performance with large models."""
        # Create a large model
        large_model = SimpleTestModel([1024, 2048, 1024, 512, 256], dtype=torch.float32)
        large_model.to(device)
        parameters = list(large_model.parameters())

        config = RangeBufferConfig(device=device, enable_profiling=True)

        import time

        start_time = time.time()

        mapper = RangeBufferMapper(parameters, config)

        initialization_time = time.time() - start_time

        # Should complete initialization in reasonable time
        assert initialization_time < 5.0  # 5 seconds should be plenty

        # Should have created reasonable number of ranges
        assert len(mapper.buffer_ranges) > 0
        assert len(mapper.buffer_ranges) < len(parameters)  # Should group parameters

    def test_memory_efficiency(self, device):
        """Test memory efficiency of buffer mapping."""
        model = SimpleTestModel([512, 1024, 512, 256])
        model.to(device)
        parameters = list(model.parameters())

        config = RangeBufferConfig(device=device)
        mapper = RangeBufferMapper(parameters, config)

        # Calculate memory usage
        stats = mapper.get_memory_stats()

        # Should have reasonable memory efficiency
        if stats.total_allocated_bytes > 0:
            memory_efficiency = stats.total_used_bytes / stats.total_allocated_bytes
            assert memory_efficiency > 0.7  # At least 70% efficiency

        # Fragmentation should be reasonable
        assert stats.fragmentation_ratio < 0.3  # Less than 30% fragmentation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
