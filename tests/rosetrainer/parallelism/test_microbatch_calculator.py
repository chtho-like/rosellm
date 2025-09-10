"""Comprehensive tests for microbatch calculator module.

Tests all calculator types, edge cases, and integration scenarios following
Megatron-LM patterns for distributed training with pipeline parallelism.
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from rosellm.rosetrainer.parallelism.microbatch_calculator import (
    AdaptiveMicrobatchCalculator,
    AdjustmentRecord,
    CalculatorType,
    ConstantNumMicrobatches,
    InvalidConfigurationError,
    RampupBatchSizeNumMicrobatches,
    calculate_optimal_microbatch_size,
    destroy_microbatch_calculator,
    get_micro_batch_size,
    get_microbatch_calculator,
    get_microbatch_schedule,
    get_num_microbatches,
    initialize_microbatch_calculator,
    update_microbatch_calculator,
)


class TestConstantNumMicrobatches:
    """Test constant microbatch calculator."""

    def test_initialization(self):
        """Test basic initialization."""
        calc = ConstantNumMicrobatches(
            global_batch_size=64, micro_batch_size=8, data_parallel_size=2
        )

        assert calc.global_batch_size == 64
        assert calc.micro_batch_size == 8
        assert calc.data_parallel_size == 2
        assert calc.global_batch_size_per_gpu == 32
        assert calc.num_microbatches == 4

    def test_get_methods(self):
        """Test getter methods."""
        calc = ConstantNumMicrobatches(
            global_batch_size=128, micro_batch_size=16, data_parallel_size=4
        )

        assert calc.get_num_microbatches() == 2
        assert calc.get_micro_batch_size() == 16
        assert calc.get_current_global_batch_size() == 128

    def test_microbatch_indexing(self):
        """Test first and last microbatch detection."""
        calc = ConstantNumMicrobatches(
            global_batch_size=32, micro_batch_size=8, data_parallel_size=1
        )

        assert calc.is_first_microbatch(0) is True
        assert calc.is_first_microbatch(1) is False
        assert calc.is_last_microbatch(3) is True
        assert calc.is_last_microbatch(2) is False

    def test_update_noop(self):
        """Test that update is a no-op for constant calculator."""
        calc = ConstantNumMicrobatches(
            global_batch_size=64, micro_batch_size=8, data_parallel_size=2
        )

        initial_num = calc.get_num_microbatches()
        calc.update(consumed_samples=1000)
        assert calc.get_num_microbatches() == initial_num

    def test_rampup_warning(self):
        """Test warning when rampup_batch_size provided."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConstantNumMicrobatches(
                global_batch_size=64,
                micro_batch_size=8,
                data_parallel_size=2,
                rampup_batch_size=[16, 32, 64],
            )
            assert len(w) == 1
            assert "ignored" in str(w[0].message)

    def test_invalid_initialization(self):
        """Test invalid initialization parameters."""
        # Global batch size not divisible by data parallel size
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ConstantNumMicrobatches(
                global_batch_size=63, micro_batch_size=8, data_parallel_size=2
            )
        assert "divisible" in str(exc_info.value)

        # Micro batch size doesn't divide evenly
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ConstantNumMicrobatches(
                global_batch_size=64, micro_batch_size=7, data_parallel_size=2
            )
        assert "divisible" in str(exc_info.value)

        # Zero or negative values
        with pytest.raises(InvalidConfigurationError) as exc_info:
            ConstantNumMicrobatches(
                global_batch_size=0, micro_batch_size=8, data_parallel_size=2
            )
        assert "positive" in str(exc_info.value)

        # Non-integer values
        with pytest.raises(InvalidConfigurationError):
            ConstantNumMicrobatches(
                global_batch_size=64.5,  # type: ignore
                micro_batch_size=8,
                data_parallel_size=2,
            )


class TestRampupBatchSizeNumMicrobatches:
    """Test rampup batch size calculator."""

    def test_initialization(self):
        """Test basic initialization with rampup."""
        calc = RampupBatchSizeNumMicrobatches(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            rampup_batch_size=[16, 32, 48, 64],
        )

        assert calc.global_batch_size == 64
        assert calc.micro_batch_size == 8
        assert calc.rampup_batch_size == [16, 32, 48, 64]
        assert calc.current_global_batch_size == 16  # Starts at first rampup value
        assert calc.ramping_up is True

    def test_rampup_progression(self):
        """Test batch size progression during rampup."""
        calc = RampupBatchSizeNumMicrobatches(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            rampup_batch_size=[16, 32, 48, 64],
        )

        # Initial state
        assert calc.get_current_global_batch_size() == 16
        assert calc.get_num_microbatches() == 1  # 16/2/8 = 1

        # After consuming 16 samples
        calc.update(consumed_samples=16)
        assert calc.get_current_global_batch_size() == 16

        # After consuming 32 samples
        calc.update(consumed_samples=32)
        assert calc.get_current_global_batch_size() == 32
        assert calc.get_num_microbatches() == 2  # 32/2/8 = 2

        # After consuming 48 samples
        calc.update(consumed_samples=48)
        assert calc.get_current_global_batch_size() == 48
        assert calc.get_num_microbatches() == 3  # 48/2/8 = 3

        # After consuming 64 samples - should reach target
        calc.update(consumed_samples=64)
        assert calc.get_current_global_batch_size() == 64
        assert calc.get_num_microbatches() == 4  # 64/2/8 = 4
        assert calc.ramping_up is False

    def test_custom_start_batch_size(self):
        """Test custom starting batch size."""
        calc = RampupBatchSizeNumMicrobatches(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            rampup_batch_size=[32, 48, 64],
            start_global_batch_size=16,  # Changed from 8 to 16 to be divisible
        )

        assert calc.current_global_batch_size == 16
        assert calc.get_num_microbatches() == 1  # 16/2/8 = 1

    def test_invalid_rampup_schedule(self):
        """Test invalid rampup schedules."""
        # Empty rampup schedule
        with pytest.raises(InvalidConfigurationError) as exc_info:
            RampupBatchSizeNumMicrobatches(
                global_batch_size=64,
                micro_batch_size=8,
                data_parallel_size=2,
                rampup_batch_size=[],
            )
        assert "non-empty" in str(exc_info.value)

        # Rampup values not divisible by data parallel size
        with pytest.raises(InvalidConfigurationError) as exc_info:
            RampupBatchSizeNumMicrobatches(
                global_batch_size=64,
                micro_batch_size=8,
                data_parallel_size=2,
                rampup_batch_size=[15, 30, 64],  # 15 not divisible by 2
            )
        assert "divisible" in str(exc_info.value)

        # Rampup values not divisible by micro batch size
        with pytest.raises(InvalidConfigurationError) as exc_info:
            RampupBatchSizeNumMicrobatches(
                global_batch_size=64,
                micro_batch_size=8,
                data_parallel_size=2,
                rampup_batch_size=[20, 40, 64],  # 20/2=10, not divisible by 8
            )
        assert "divisible" in str(exc_info.value)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.all_reduce")
    def test_consistency_check(self, mock_all_reduce, mock_world_size, mock_is_init):
        """Test distributed consistency checking."""
        mock_is_init.return_value = True
        mock_world_size.return_value = 4

        calc = RampupBatchSizeNumMicrobatches(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            rampup_batch_size=[32, 64],
        )

        # Mock all_reduce to simulate correct behavior
        def all_reduce_side_effect(tensor, op):
            tensor.mul_(mock_world_size.return_value)
            return tensor

        mock_all_reduce.side_effect = all_reduce_side_effect

        # Update with consistency check
        calc.update(consumed_samples=32, consistency_check=True)

        # Verify all_reduce was called
        assert mock_all_reduce.called


class TestAdaptiveMicrobatchCalculator:
    """Test adaptive microbatch calculator."""

    def test_initialization(self):
        """Test basic initialization."""
        calc = AdaptiveMicrobatchCalculator(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            min_micro_batch_size=4,
            max_micro_batch_size=16,
            memory_threshold=0.8,
        )

        assert calc.global_batch_size == 64
        assert calc.micro_batch_size == 8
        assert calc.current_micro_batch_size == 8
        assert calc.min_micro_batch_size == 4
        assert calc.max_micro_batch_size == 16
        assert calc.memory_threshold == 0.8

    def test_default_max_micro_batch_size(self):
        """Test default max micro batch size."""
        calc = AdaptiveMicrobatchCalculator(
            global_batch_size=64, micro_batch_size=8, data_parallel_size=2
        )

        assert calc.max_micro_batch_size == 16  # 2x initial

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.current_device")
    def test_memory_based_adjustment(
        self, mock_current_device, mock_props, mock_reserved, mock_allocated, mock_cuda
    ):
        """Test memory-based microbatch size adjustment."""
        mock_cuda.return_value = True
        mock_current_device.return_value = 0  # Mock current device as 0
        mock_allocated.return_value = 4 * 1024**3  # 4 GB
        mock_reserved.return_value = 8 * 1024**3  # 8 GB

        # Mock device properties
        mock_device = MagicMock()
        mock_device.total_memory = 10 * 1024**3  # 10 GB total
        mock_props.return_value = mock_device

        calc = AdaptiveMicrobatchCalculator(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            min_micro_batch_size=4,
            max_micro_batch_size=16,
            memory_threshold=0.7,  # 70% threshold
        )

        # Initial size
        assert calc.get_micro_batch_size() == 8

        # Update multiple times to trigger adjustment check
        for i in range(10):
            calc.update(consumed_samples=calc.global_batch_size * (i + 1))

        # Should have reduced due to high memory usage (80%)
        assert calc.get_micro_batch_size() == 4

    @patch("torch.cuda.is_available")
    def test_cpu_memory_usage(self, mock_cuda):
        """Test memory usage on CPU returns 0."""
        mock_cuda.return_value = False

        calc = AdaptiveMicrobatchCalculator(
            global_batch_size=64, micro_batch_size=8, data_parallel_size=2
        )

        memory_usage = calc._get_memory_usage()
        assert memory_usage == 0.0

    def test_adjustment_history(self):
        """Test that adjustment history is tracked."""
        calc = AdaptiveMicrobatchCalculator(
            global_batch_size=64, micro_batch_size=8, data_parallel_size=2
        )

        calc.update(consumed_samples=100)
        calc.update(consumed_samples=200)

        assert len(calc.adjustment_history) == 2
        assert isinstance(calc.adjustment_history[0], AdjustmentRecord)
        assert calc.adjustment_history[0].consumed_samples == 100
        assert calc.adjustment_history[1].consumed_samples == 200

    def test_invalid_parameters(self):
        """Test invalid initialization parameters."""
        # Invalid memory threshold
        with pytest.raises(InvalidConfigurationError) as exc_info:
            AdaptiveMicrobatchCalculator(
                global_batch_size=64,
                micro_batch_size=8,
                data_parallel_size=2,
                memory_threshold=1.5,  # > 1
            )
        assert "memory_threshold" in str(exc_info.value)

        # Min > Max
        with pytest.raises(InvalidConfigurationError) as exc_info:
            AdaptiveMicrobatchCalculator(
                global_batch_size=64,
                micro_batch_size=8,
                data_parallel_size=2,
                min_micro_batch_size=16,
                max_micro_batch_size=8,
            )
        assert "max_micro_batch_size" in str(exc_info.value)


class TestGlobalStateManagement:
    """Test global state management functions."""

    def teardown_method(self):
        """Clean up global state after each test."""
        destroy_microbatch_calculator()

    def test_initialize_constant(self):
        """Test initializing constant calculator globally."""
        initialize_microbatch_calculator(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            calculator_type="constant",
        )

        # Test with enum
        destroy_microbatch_calculator()
        calc2 = initialize_microbatch_calculator(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            calculator_type=CalculatorType.CONSTANT,
        )

        assert calc2 is not None
        assert isinstance(calc2, ConstantNumMicrobatches)
        assert get_microbatch_calculator() is calc2

    def test_initialize_rampup(self):
        """Test initializing rampup calculator globally."""
        calc = initialize_microbatch_calculator(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            rampup_batch_size=[32, 64],
            calculator_type="rampup",
        )

        assert calc is not None
        assert isinstance(calc, RampupBatchSizeNumMicrobatches)
        assert get_microbatch_calculator() is calc

    def test_initialize_adaptive(self):
        """Test initializing adaptive calculator globally."""
        calc = initialize_microbatch_calculator(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            calculator_type="adaptive",
        )

        assert calc is not None
        assert isinstance(calc, AdaptiveMicrobatchCalculator)
        assert get_microbatch_calculator() is calc

    def test_global_getters(self):
        """Test global getter functions."""
        initialize_microbatch_calculator(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            calculator_type="constant",
        )

        assert get_num_microbatches() == 4
        assert get_micro_batch_size() == 8

    def test_global_update(self):
        """Test global update function."""
        initialize_microbatch_calculator(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            rampup_batch_size=[32, 64],
            calculator_type="rampup",
        )

        # Initial state
        assert get_num_microbatches() == 2  # 32/2/8

        # Update
        update_microbatch_calculator(consumed_samples=32)
        assert get_num_microbatches() == 2  # Still 32/2/8 until we consume more

        # Update with more samples to trigger next rampup
        update_microbatch_calculator(consumed_samples=64)
        assert get_num_microbatches() == 4  # Now 64/2/8

    def test_uninitialized_access(self):
        """Test accessing uninitialized calculator."""
        with pytest.raises(RuntimeError):
            get_num_microbatches()

        with pytest.raises(RuntimeError):
            get_micro_batch_size()

    def test_destroy_calculator(self):
        """Test destroying global calculator."""
        initialize_microbatch_calculator(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
            calculator_type="constant",
        )

        assert get_microbatch_calculator() is not None

        destroy_microbatch_calculator()
        assert get_microbatch_calculator() is None

    def test_invalid_calculator_type(self):
        """Test invalid calculator type."""
        with pytest.raises(ValueError):
            initialize_microbatch_calculator(
                global_batch_size=64,
                micro_batch_size=8,
                data_parallel_size=2,
                calculator_type="invalid",
            )

    def test_rampup_without_schedule(self):
        """Test rampup calculator without schedule."""
        with pytest.raises(InvalidConfigurationError) as exc_info:
            initialize_microbatch_calculator(
                global_batch_size=64,
                micro_batch_size=8,
                data_parallel_size=2,
                calculator_type="rampup",
                # Missing rampup_batch_size
            )
        assert "required" in str(exc_info.value)


class TestUtilityFunctions:
    """Test utility functions for batch size calculation."""

    def test_calculate_with_different_optimizers(self):
        """Test optimal size calculation with different optimizers."""
        size_adam = calculate_optimal_microbatch_size(
            model_size_gb=1.0,
            available_memory_gb=8.0,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
            pipeline_parallel_size=1,
            activation_checkpoint=False,
            optimizer_type="adam",
        )

        size_sgd = calculate_optimal_microbatch_size(
            model_size_gb=1.0,
            available_memory_gb=8.0,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
            pipeline_parallel_size=1,
            activation_checkpoint=False,
            optimizer_type="sgd",
        )

        # SGD uses less memory (no variance), should allow larger batch
        assert size_sgd >= size_adam

    def test_calculate_with_different_precisions(self):
        """Test optimal size calculation with different precisions."""
        size_fp16 = calculate_optimal_microbatch_size(
            model_size_gb=1.0,
            available_memory_gb=8.0,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
            pipeline_parallel_size=1,
            activation_checkpoint=False,
            precision="fp16",
        )

        size_fp32 = calculate_optimal_microbatch_size(
            model_size_gb=1.0,
            available_memory_gb=8.0,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
            pipeline_parallel_size=1,
            activation_checkpoint=False,
            precision="fp32",
        )

        # FP32 uses more memory, should have smaller batch
        assert size_fp16 >= size_fp32

    def test_calculate_invalid_inputs(self):
        """Test calculation with invalid inputs."""
        with pytest.raises(ValueError):
            calculate_optimal_microbatch_size(
                model_size_gb=-1.0,  # Negative
                available_memory_gb=8.0,
                sequence_length=512,
                hidden_size=768,
                num_layers=12,
            )

        with pytest.raises(ValueError):
            calculate_optimal_microbatch_size(
                model_size_gb=1.0,
                available_memory_gb=8.0,
                sequence_length=0,  # Zero
                hidden_size=768,
                num_layers=12,
            )

    def test_calculate_optimal_microbatch_size_basic(self):
        """Test basic optimal microbatch size calculation."""
        size = calculate_optimal_microbatch_size(
            model_size_gb=1.0,
            available_memory_gb=8.0,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
            pipeline_parallel_size=1,
            activation_checkpoint=False,
        )

        assert size >= 1
        assert size & (size - 1) == 0  # Power of 2

    def test_calculate_with_activation_checkpoint(self):
        """Test calculation with activation checkpointing."""
        size_no_checkpoint = calculate_optimal_microbatch_size(
            model_size_gb=1.0,
            available_memory_gb=8.0,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
            pipeline_parallel_size=1,
            activation_checkpoint=False,
        )

        size_with_checkpoint = calculate_optimal_microbatch_size(
            model_size_gb=1.0,
            available_memory_gb=8.0,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
            pipeline_parallel_size=1,
            activation_checkpoint=True,
        )

        # With checkpointing should allow larger batch size
        assert size_with_checkpoint >= size_no_checkpoint

    def test_calculate_with_pipeline_parallel(self):
        """Test calculation with pipeline parallelism."""
        size_pp1 = calculate_optimal_microbatch_size(
            model_size_gb=4.0,
            available_memory_gb=8.0,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
            pipeline_parallel_size=1,
            activation_checkpoint=False,
        )

        size_pp4 = calculate_optimal_microbatch_size(
            model_size_gb=4.0,
            available_memory_gb=8.0,
            sequence_length=512,
            hidden_size=768,
            num_layers=12,
            pipeline_parallel_size=4,
            activation_checkpoint=False,
        )

        # More pipeline stages means less model per GPU, more room for batch
        assert size_pp4 >= size_pp1

    def test_calculate_insufficient_memory(self):
        """Test calculation with insufficient memory."""
        size = calculate_optimal_microbatch_size(
            model_size_gb=10.0,
            available_memory_gb=8.0,
            sequence_length=2048,
            hidden_size=4096,
            num_layers=32,
            pipeline_parallel_size=1,
            activation_checkpoint=False,
        )

        assert size == 1  # Minimum size

    def test_linear_schedule(self):
        """Test linear rampup schedule generation."""
        schedule = get_microbatch_schedule(
            start_batch_size=16,
            target_batch_size=64,
            warmup_steps=4,
            schedule_type="linear",
        )

        assert len(schedule) >= 1
        assert schedule[0] >= 16
        assert schedule[-1] <= 64
        # Check monotonic increase
        for i in range(1, len(schedule)):
            assert schedule[i] >= schedule[i - 1]

    def test_exponential_schedule(self):
        """Test exponential rampup schedule generation."""
        schedule = get_microbatch_schedule(
            start_batch_size=16,
            target_batch_size=256,
            warmup_steps=4,
            schedule_type="exponential",
        )

        assert len(schedule) >= 1
        assert schedule[0] >= 16
        assert schedule[-1] <= 256
        # Check monotonic increase
        for i in range(1, len(schedule)):
            assert schedule[i] >= schedule[i - 1]

    def test_cosine_schedule(self):
        """Test cosine rampup schedule generation."""
        schedule = get_microbatch_schedule(
            start_batch_size=16,
            target_batch_size=64,
            warmup_steps=8,
            schedule_type="cosine",
        )

        assert len(schedule) >= 1
        assert schedule[0] >= 16
        assert schedule[-1] <= 64
        # Check monotonic increase
        for i in range(1, len(schedule)):
            assert schedule[i] >= schedule[i - 1]

    def test_zero_warmup_steps(self):
        """Test schedule with zero warmup steps."""
        schedule = get_microbatch_schedule(
            start_batch_size=16,
            target_batch_size=64,
            warmup_steps=0,
            schedule_type="linear",
        )

        assert schedule == [64]

    def test_invalid_schedule_type(self):
        """Test invalid schedule type."""
        with pytest.raises(ValueError) as exc_info:
            get_microbatch_schedule(
                start_batch_size=16,
                target_batch_size=64,
                warmup_steps=4,
                schedule_type="invalid",
            )
        assert "Unknown schedule type" in str(exc_info.value)

    def test_polynomial_schedule(self):
        """Test polynomial rampup schedule generation."""
        schedule = get_microbatch_schedule(
            start_batch_size=16,
            target_batch_size=256,
            warmup_steps=5,
            schedule_type="polynomial",
        )

        assert len(schedule) >= 1
        assert schedule[0] >= 16
        assert schedule[-1] <= 256
        # Check monotonic increase
        for i in range(1, len(schedule)):
            assert schedule[i] >= schedule[i - 1]

    def test_schedule_with_divisibility(self):
        """Test schedule with divisibility constraint."""
        schedule = get_microbatch_schedule(
            start_batch_size=17,
            target_batch_size=97,
            warmup_steps=5,
            schedule_type="linear",
            ensure_divisible_by=8,
        )

        # All values should be divisible by 8
        for bs in schedule:
            assert bs % 8 == 0

    def test_invalid_batch_sizes(self):
        """Test invalid batch size parameters."""
        with pytest.raises(ValueError):
            get_microbatch_schedule(
                start_batch_size=-16,
                target_batch_size=64,
                warmup_steps=4,
                schedule_type="linear",
            )

        with pytest.raises(ValueError):
            get_microbatch_schedule(
                start_batch_size=64,
                target_batch_size=16,  # Start > target
                warmup_steps=4,
                schedule_type="linear",
            )


class TestThreadSafety:
    """Test thread safety of global state management."""

    def teardown_method(self):
        """Clean up global state after each test."""
        destroy_microbatch_calculator()

    def test_concurrent_initialization(self):
        """Test concurrent initialization attempts."""
        import threading

        results = []
        errors = []

        def init_calculator(idx):
            try:
                calc = initialize_microbatch_calculator(
                    global_batch_size=64 * (idx + 1),
                    micro_batch_size=8,
                    data_parallel_size=2,
                    calculator_type="constant",
                )
                results.append((idx, calc))
            except Exception as e:
                errors.append((idx, e))

        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=init_calculator, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have one successful initialization (last one)
        assert len(errors) == 0
        assert get_microbatch_calculator() is not None

    def test_concurrent_access(self):
        """Test concurrent access to global calculator."""
        import threading

        initialize_microbatch_calculator(
            global_batch_size=256,
            micro_batch_size=32,
            data_parallel_size=4,
            calculator_type="constant",
        )

        results = []

        def access_calculator(idx):
            for _ in range(100):
                num_mb = get_num_microbatches()
                mb_size = get_micro_batch_size()
                calc = get_microbatch_calculator()
                results.append((idx, num_mb, mb_size, calc is not None))

        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=access_calculator, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # All accesses should return consistent values
        for _, num_mb, mb_size, has_calc in results:
            assert num_mb == 2  # 256/4/32
            assert mb_size == 32
            assert has_calc


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""

    def teardown_method(self):
        """Clean up global state after each test."""
        destroy_microbatch_calculator()

    def test_training_loop_simulation(self):
        """Simulate a training loop with microbatch calculator."""
        # Initialize calculator
        calc = initialize_microbatch_calculator(
            global_batch_size=256,
            micro_batch_size=32,
            data_parallel_size=4,
            rampup_batch_size=[128, 256],  # Use only values divisible by 32*4=128
            calculator_type="rampup",
        )

        total_samples = 0
        iterations = 20

        for iteration in range(iterations):
            # Get current configuration
            num_microbatches = get_num_microbatches()
            micro_batch_size = get_micro_batch_size()

            # Simulate processing microbatches
            for mb_idx in range(num_microbatches):
                # Check if first/last
                calc.is_first_microbatch(mb_idx)
                calc.is_last_microbatch(mb_idx)

                # Process microbatch (simulated)
                samples_in_mb = micro_batch_size
                total_samples += samples_in_mb

            # Update calculator
            update_microbatch_calculator(total_samples, consistency_check=False)

            # Verify batch size increases during rampup
            if iteration < 2:  # Changed to 2 since we have 2 rampup steps now
                assert isinstance(calc, RampupBatchSizeNumMicrobatches)
                assert calc.ramping_up or calc.current_global_batch_size == 256

        # After sufficient iterations, should reach target
        assert isinstance(calc, RampupBatchSizeNumMicrobatches)
        assert calc.current_global_batch_size == 256

    def test_memory_constrained_training(self):
        """Test training with memory constraints."""
        # Calculate optimal batch size for constrained memory
        optimal_mb_size = calculate_optimal_microbatch_size(
            model_size_gb=2.0,
            available_memory_gb=4.0,
            sequence_length=1024,
            hidden_size=1024,
            num_layers=24,
            pipeline_parallel_size=2,
            activation_checkpoint=True,
        )

        # Initialize adaptive calculator
        # Ensure micro_batch_size is reasonable
        micro_batch_size = min(optimal_mb_size, 16)  # Cap at 16 for test
        calc = initialize_microbatch_calculator(
            global_batch_size=64,
            micro_batch_size=micro_batch_size,
            data_parallel_size=2,
            calculator_type="adaptive",
        )

        # Simulate training with memory pressure
        for step in range(10):
            get_num_microbatches()
            mb_size = get_micro_batch_size()

            # Process batch
            consumed = (step + 1) * calc.global_batch_size
            update_microbatch_calculator(consumed, consistency_check=False)

            # Verify microbatch size stays within bounds
            assert mb_size >= 1
            assert mb_size <= micro_batch_size * 2

    def test_memory_history_limit(self):
        """Test that adjustment history doesn't grow unbounded."""
        calc = AdaptiveMicrobatchCalculator(
            global_batch_size=64,
            micro_batch_size=8,
            data_parallel_size=2,
        )

        # Update many times
        for i in range(2000):
            calc.update(consumed_samples=i * 64, consistency_check=False)

        # History should be capped
        assert len(calc.adjustment_history) <= 1000  # MAX_HISTORY_SIZE

    def test_multi_node_consistency(self):
        """Test consistency across multiple nodes (mocked)."""
        with patch("torch.distributed.is_initialized", return_value=True), patch(
            "torch.distributed.get_world_size", return_value=8
        ), patch("torch.distributed.all_reduce") as mock_all_reduce:
            # Initialize on "multiple nodes"
            initialize_microbatch_calculator(
                global_batch_size=512,
                micro_batch_size=32,
                data_parallel_size=8,
                calculator_type="constant",
            )

            # Simulate all_reduce behavior
            def all_reduce_effect(tensor, op):
                tensor.mul_(8)  # Simulate sum across 8 ranks
                return tensor

            mock_all_reduce.side_effect = all_reduce_effect

            # Update with consistency check
            update_microbatch_calculator(1000, consistency_check=True)

            # Verify configuration is valid for multi-node
            assert get_num_microbatches() == 2  # 512/8/32
            assert get_micro_batch_size() == 32


class TestIntegrationWithPipeline:
    """Test integration scenarios with pipeline parallelism."""

    def test_pipeline_bubble_calculation(self):
        """Test calculating pipeline bubble with microbatches."""
        calc = ConstantNumMicrobatches(
            global_batch_size=128, micro_batch_size=8, data_parallel_size=2
        )

        num_microbatches = calc.get_num_microbatches()
        pipeline_stages = 4

        # Pipeline bubble formula: (p-1) * m/p
        # Where p = pipeline stages, m = microbatches
        bubble_microbatches = (pipeline_stages - 1) * (
            num_microbatches / pipeline_stages
        )

        assert bubble_microbatches > 0
        assert bubble_microbatches < num_microbatches

    def test_gradient_accumulation_steps(self):
        """Test gradient accumulation calculation."""
        calc = ConstantNumMicrobatches(
            global_batch_size=256, micro_batch_size=32, data_parallel_size=4
        )

        # Gradient accumulation steps = num_microbatches
        grad_accum_steps = calc.get_num_microbatches()

        # Verify total batch size
        total_batch = grad_accum_steps * calc.micro_batch_size * calc.data_parallel_size
        assert total_batch == calc.global_batch_size

    def test_schedule_for_warmup(self):
        """Test creating schedule for model warmup."""
        # Generate warmup schedule
        schedule = get_microbatch_schedule(
            start_batch_size=32,
            target_batch_size=512,
            warmup_steps=100,
            schedule_type="cosine",
        )

        # Use schedule with calculator
        # Filter schedule to ensure divisibility
        valid_schedule = [bs for bs in schedule if bs >= 128 and bs % (32 * 4) == 0]
        if not valid_schedule:
            valid_schedule = [128, 256, 512]

        calc = RampupBatchSizeNumMicrobatches(
            global_batch_size=512,
            micro_batch_size=32,
            data_parallel_size=4,
            rampup_batch_size=valid_schedule,
        )

        # Verify smooth rampup
        assert calc.current_global_batch_size <= 512
        samples = 0
        for batch_size in schedule[:10]:  # Check first 10 steps
            samples += batch_size
            calc.update(samples)
            assert calc.current_global_batch_size <= 512
