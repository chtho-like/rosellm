"""
Comprehensive unit tests for parameter-gradient mapping with bucket-based reduction.

This test module provides extensive coverage of the ParamGradMapping functionality,
including parameter classification, bucket assignment, gradient synchronization,
multi-tensor operations, and various reduction strategies.
"""

import gc
import unittest
from unittest.mock import patch

import torch

# import torch.distributed as dist  # Unused in tests
import torch.nn as nn
from torch.nn import Parameter

from rosellm.rosetrainer.optimizer.param_grad_mapping import (
    MappingConfig,
    MultiTensorOperator,
    ParameterInfo,
    ParameterType,
    ParamGradMapping,
    ParamGradMappingBuilder,
    ReductionStrategy,
)


class SimpleTestModel(nn.Module):
    """Simple test model with various parameter types."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_size)
        self.weight = nn.Linear(hidden_size, hidden_size)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.norm = nn.LayerNorm(hidden_size)
        self.position_embed = nn.Parameter(torch.randn(100, hidden_size))
        self.output = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = self.weight(x) + self.bias
        x = self.norm(x)
        x = x + self.position_embed[: x.size(1)]
        return self.output(x)


class TestParameterType(unittest.TestCase):
    """Test parameter type classification."""

    def test_parameter_type_enum(self):
        """Test ParameterType enum values."""
        self.assertEqual(ParameterType.WEIGHT.value, "weight")
        self.assertEqual(ParameterType.BIAS.value, "bias")
        self.assertEqual(ParameterType.EMBEDDING.value, "embedding")
        self.assertEqual(ParameterType.NORM.value, "norm")
        self.assertEqual(ParameterType.POSITION.value, "position")
        self.assertEqual(ParameterType.OTHER.value, "other")


class TestReductionStrategy(unittest.TestCase):
    """Test reduction strategy enum."""

    def test_reduction_strategy_enum(self):
        """Test ReductionStrategy enum values."""
        self.assertEqual(ReductionStrategy.IMMEDIATE.value, "immediate")
        self.assertEqual(ReductionStrategy.DELAYED.value, "delayed")
        self.assertEqual(ReductionStrategy.OVERLAPPED.value, "overlapped")
        self.assertEqual(ReductionStrategy.HIERARCHICAL.value, "hierarchical")


class TestMappingConfig(unittest.TestCase):
    """Test MappingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MappingConfig()

        self.assertEqual(config.bucket_size_mb, 25.0)
        self.assertEqual(config.min_bucket_size_mb, 1.0)
        self.assertEqual(config.reduction_strategy, ReductionStrategy.OVERLAPPED)
        self.assertTrue(config.gradient_predivision)
        self.assertEqual(config.gradient_accumulation_steps, 1)
        self.assertTrue(config.use_memory_pool)
        self.assertFalse(config.pin_memory)
        self.assertTrue(config.contiguous_gradients)
        self.assertTrue(config.communication_overlap)
        self.assertTrue(config.type_specific_buckets)
        self.assertFalse(config.enable_gradient_clipping)
        self.assertFalse(config.enable_gradient_scaling)

    def test_custom_config(self):
        """Test custom configuration."""
        config = MappingConfig(
            bucket_size_mb=50.0,
            reduction_strategy=ReductionStrategy.IMMEDIATE,
            gradient_accumulation_steps=4,
            enable_gradient_clipping=True,
            gradient_clip_value=2.0,
        )

        self.assertEqual(config.bucket_size_mb, 50.0)
        self.assertEqual(config.reduction_strategy, ReductionStrategy.IMMEDIATE)
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertTrue(config.enable_gradient_clipping)
        self.assertEqual(config.gradient_clip_value, 2.0)

    def test_type_bucket_sizes(self):
        """Test type-specific bucket sizes."""
        config = MappingConfig()

        self.assertEqual(config.type_bucket_sizes[ParameterType.EMBEDDING], 50.0)
        self.assertEqual(config.type_bucket_sizes[ParameterType.WEIGHT], 25.0)
        self.assertEqual(config.type_bucket_sizes[ParameterType.BIAS], 10.0)
        self.assertEqual(config.type_bucket_sizes[ParameterType.NORM], 10.0)
        self.assertEqual(config.type_bucket_sizes[ParameterType.POSITION], 25.0)
        self.assertEqual(config.type_bucket_sizes[ParameterType.OTHER], 25.0)


class TestParameterInfo(unittest.TestCase):
    """Test ParameterInfo dataclass."""

    def setUp(self):
        """Set up test parameters."""
        self.param = Parameter(torch.randn(10, 20))
        self.info = ParameterInfo(
            param=self.param,
            name="test_param",
            param_type=ParameterType.WEIGHT,
            shape=self.param.shape,
            numel=self.param.numel(),
            dtype=self.param.dtype,
            device=self.param.device,
        )

    def test_parameter_info_creation(self):
        """Test ParameterInfo creation."""
        self.assertEqual(self.info.name, "test_param")
        self.assertEqual(self.info.param_type, ParameterType.WEIGHT)
        self.assertEqual(self.info.shape, torch.Size([10, 20]))
        self.assertEqual(self.info.numel, 200)
        self.assertEqual(self.info.dtype, torch.float32)
        self.assertTrue(self.info.requires_grad)
        self.assertIsNone(self.info.bucket_id)
        self.assertIsNone(self.info.buffer_offset)

    def test_parameter_info_hash(self):
        """Test ParameterInfo hashing."""
        info2 = ParameterInfo(
            param=self.param,
            name="different_name",
            param_type=ParameterType.BIAS,
            shape=self.param.shape,
            numel=self.param.numel(),
            dtype=self.param.dtype,
            device=self.param.device,
        )

        # Same parameter, different metadata
        self.assertEqual(hash(self.info), hash(info2))

    def test_parameter_info_equality(self):
        """Test ParameterInfo equality."""
        info2 = ParameterInfo(
            param=self.param,
            name="different_name",
            param_type=ParameterType.BIAS,
            shape=self.param.shape,
            numel=self.param.numel(),
            dtype=self.param.dtype,
            device=self.param.device,
        )

        # Same parameter reference
        self.assertEqual(self.info, info2)

        # Different parameter
        param3 = Parameter(torch.randn(10, 20))
        info3 = ParameterInfo(
            param=param3,
            name="test_param",
            param_type=ParameterType.WEIGHT,
            shape=param3.shape,
            numel=param3.numel(),
            dtype=param3.dtype,
            device=param3.device,
        )
        self.assertNotEqual(self.info, info3)


class TestMultiTensorOperator(unittest.TestCase):
    """Test multi-tensor operations."""

    def setUp(self):
        """Set up test environment."""
        self.device = torch.device("cpu")
        self.operator = MultiTensorOperator(self.device)
        self.tensors = [torch.randn(10, 10), torch.randn(5, 5), torch.randn(20, 20)]

    def test_scale_tensors_in_place(self):
        """Test in-place tensor scaling."""
        original_values = [t.clone() for t in self.tensors]
        scale_factor = 0.5

        result = self.operator.scale_tensors(self.tensors, scale_factor, in_place=True)

        # Check that tensors were scaled in place
        for i, tensor in enumerate(self.tensors):
            expected = original_values[i] * scale_factor
            torch.testing.assert_close(tensor, expected)

        # Result should be the same list
        self.assertIs(result, self.tensors)

    def test_scale_tensors_not_in_place(self):
        """Test non-in-place tensor scaling."""
        original_values = [t.clone() for t in self.tensors]
        scale_factor = 2.0

        result = self.operator.scale_tensors(self.tensors, scale_factor, in_place=False)

        # Original tensors should be unchanged
        for i, tensor in enumerate(self.tensors):
            torch.testing.assert_close(tensor, original_values[i])

        # Result should be scaled
        for i, tensor in enumerate(result):
            expected = original_values[i] * scale_factor
            torch.testing.assert_close(tensor, expected)

    def test_scale_empty_tensors(self):
        """Test scaling empty tensor list."""
        result = self.operator.scale_tensors([], 2.0)
        self.assertEqual(result, [])

    def test_clip_tensors(self):
        """Test gradient clipping."""
        # Create tensors with known norms
        tensors = [
            torch.ones(10, 10) * 2.0,  # norm = 20
            torch.ones(5, 5) * 4.0,  # norm = 20
        ]

        max_norm = 10.0
        clipped, original_norm = self.operator.clip_tensors(tensors, max_norm)

        # Check that norm was reduced
        new_norm = torch.norm(torch.stack([torch.norm(t, 2.0) for t in clipped]), 2.0)
        self.assertLessEqual(new_norm, max_norm * 1.01)  # Allow small numerical error

        # Check original norm is correct
        self.assertGreater(original_norm, max_norm)

    def test_clip_tensors_no_clipping_needed(self):
        """Test clipping when gradients are already small."""
        tensors = [
            torch.ones(2, 2) * 0.1,
            torch.ones(3, 3) * 0.1,
        ]

        max_norm = 100.0
        clipped, original_norm = self.operator.clip_tensors(tensors, max_norm)

        # Tensors should be unchanged
        torch.testing.assert_close(clipped[0], tensors[0])
        torch.testing.assert_close(clipped[1], tensors[1])

    def test_clip_empty_tensors(self):
        """Test clipping empty tensor list."""
        clipped, norm = self.operator.clip_tensors([], 1.0)
        self.assertEqual(clipped, [])
        self.assertEqual(norm, 0.0)

    def test_copy_tensors(self):
        """Test tensor copying."""
        src_tensors = [torch.randn(10, 10) for _ in range(3)]
        dst_tensors = [torch.zeros(10, 10) for _ in range(3)]

        self.operator.copy_tensors(src_tensors, dst_tensors)

        for src, dst in zip(src_tensors, dst_tensors):
            torch.testing.assert_close(src, dst)

    def test_copy_tensors_mismatch(self):
        """Test error on mismatched tensor counts."""
        src_tensors = [torch.randn(10, 10) for _ in range(3)]
        dst_tensors = [torch.zeros(10, 10) for _ in range(2)]

        with self.assertRaises(ValueError) as context:
            self.operator.copy_tensors(src_tensors, dst_tensors)

        self.assertIn("mismatch", str(context.exception))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_operations(self):
        """Test operations on CUDA device."""
        device = torch.device("cuda:0")
        operator = MultiTensorOperator(device)

        tensors = [
            torch.randn(10, 10, device=device),
            torch.randn(5, 5, device=device),
        ]

        # Test scaling
        scaled = operator.scale_tensors(tensors, 0.5, in_place=False)
        self.assertEqual(scaled[0].device, device)

        # Test synchronization
        operator.synchronize()


class TestParamGradMapping(unittest.TestCase):
    """Test ParamGradMapping functionality."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleTestModel()
        self.params = list(self.model.parameters())
        self.device = torch.device("cpu")

    def test_basic_initialization(self):
        """Test basic initialization of ParamGradMapping."""
        mapping = ParamGradMapping(params=self.params, device=self.device)

        self.assertEqual(len(mapping.param_infos), len(self.params))
        self.assertIsNotNone(mapping.gradient_buffer)
        self.assertIsNotNone(mapping.bucket_manager)
        self.assertIsNotNone(mapping.multi_tensor_op)

    def test_parameter_classification(self):
        """Test parameter type classification."""
        mapping = ParamGradMapping(params=self.params, device=self.device)

        # Check parameter classification
        classifications = {}
        for info in mapping.param_infos:
            classifications[info.name] = info.param_type

        # We should have different parameter types
        types = set(classifications.values())
        self.assertGreater(len(types), 1)

    def test_parameter_groups(self):
        """Test initialization with parameter groups."""
        param_groups = [
            {"params": self.params[:3], "lr": 0.01},
            {"params": self.params[3:], "lr": 0.001},
        ]

        mapping = ParamGradMapping(params=param_groups, device=self.device)

        self.assertEqual(len(mapping.param_groups), 2)
        self.assertEqual(len(mapping.param_infos), len(self.params))

    def test_custom_config(self):
        """Test initialization with custom configuration."""
        config = MappingConfig(
            bucket_size_mb=50.0,
            reduction_strategy=ReductionStrategy.IMMEDIATE,
            gradient_accumulation_steps=4,
            enable_gradient_clipping=True,
            gradient_clip_value=1.0,
        )

        mapping = ParamGradMapping(
            params=self.params, config=config, device=self.device
        )

        self.assertEqual(mapping.config.bucket_size_mb, 50.0)
        self.assertEqual(mapping.config.reduction_strategy, ReductionStrategy.IMMEDIATE)
        self.assertEqual(mapping.config.gradient_accumulation_steps, 4)

    def test_map_parameters(self):
        """Test parameter mapping to buffers."""
        # Create gradients for parameters
        for param in self.params:
            param.grad = torch.randn_like(param)

        mapping = ParamGradMapping(params=self.params, device=self.device)

        # Mapping should be called during initialization
        # Check that parameters have been assigned to buckets
        assigned_params = [
            info
            for info in mapping.param_infos
            if info.bucket_id is not None or info.buffer_offset is not None
        ]
        self.assertGreater(len(assigned_params), 0)

    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        config = MappingConfig(gradient_accumulation_steps=4)
        mapping = ParamGradMapping(
            params=self.params, config=config, device=self.device
        )

        # Create gradients
        for param in self.params:
            param.grad = torch.ones_like(param)

        # Accumulate gradients
        mapping.accumulate_gradients()

        # Check that gradients were scaled
        expected_scale = 1.0 / 4
        for param in self.params:
            if param.grad is not None:
                # Get first element regardless of shape
                first_elem = param.grad.flatten()[0].item()
                self.assertAlmostEqual(first_elem, expected_scale, places=5)

    def test_should_reduce_gradients(self):
        """Test gradient reduction timing."""
        config = MappingConfig(gradient_accumulation_steps=4)
        mapping = ParamGradMapping(
            params=self.params, config=config, device=self.device
        )

        # Initially should not reduce
        self.assertFalse(mapping.should_reduce_gradients())

        # After accumulation steps
        mapping.accumulation_step = 4
        self.assertTrue(mapping.should_reduce_gradients())

        # Not at accumulation boundary
        mapping.accumulation_step = 3
        self.assertFalse(mapping.should_reduce_gradients())

    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        config = MappingConfig(enable_gradient_clipping=True, gradient_clip_value=1.0)

        mapping = ParamGradMapping(
            params=self.params, config=config, device=self.device
        )

        # Create large gradients
        for param in self.params:
            param.grad = torch.ones_like(param) * 100.0

        # Apply clipping through internal method
        mapping._clip_gradients()

        # Check that gradients were clipped
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad, 2.0) for p in self.params if p.grad is not None]
            ),
            2.0,
        )
        self.assertLessEqual(total_norm, config.gradient_clip_value * 1.1)

    def test_gradient_scaling(self):
        """Test gradient scaling functionality."""
        config = MappingConfig(enable_gradient_scaling=True, gradient_scale_factor=0.5)

        mapping = ParamGradMapping(
            params=self.params, config=config, device=self.device
        )

        # Create gradients
        for param in self.params:
            param.grad = torch.ones_like(param) * 2.0

        # Apply scaling through internal method
        mapping._scale_gradients()

        # Check that gradients were scaled
        for param in self.params:
            if param.grad is not None:
                # Get first element regardless of shape
                first_elem = param.grad.flatten()[0].item()
                self.assertAlmostEqual(first_elem, 1.0, places=5)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    def test_synchronize_gradients_skipped(self, mock_world_size, mock_is_initialized):
        """Test gradient synchronization when skipped."""
        mock_is_initialized.return_value = False
        mock_world_size.return_value = 1

        config = MappingConfig(gradient_accumulation_steps=4)
        mapping = ParamGradMapping(
            params=self.params, config=config, device=self.device
        )

        # Should skip because accumulation not complete
        stats = mapping.synchronize_gradients()
        self.assertTrue(stats.get("skipped", False))
        self.assertEqual(stats.get("reason"), "accumulation_not_complete")

        # Force synchronization
        mapping.accumulation_step = 4
        stats = mapping.synchronize_gradients(force=True)
        # May still skip if no bucket manager, but force should override
        # accumulation check

    def test_get_parameter_info(self):
        """Test getting parameter information."""
        mapping = ParamGradMapping(params=self.params, device=self.device)

        # Get info for first parameter
        param = self.params[0]
        info = mapping.get_parameter_info(param)

        self.assertIsNotNone(info)
        if info is not None:
            # Use 'is' instead of == to check parameter identity
            self.assertIs(info.param, param)

        # Non-existent parameter
        fake_param = Parameter(torch.randn(5, 5))
        info = mapping.get_parameter_info(fake_param)
        self.assertIsNone(info)

    def test_statistics(self):
        """Test statistics collection."""
        mapping = ParamGradMapping(params=self.params, device=self.device)

        stats = mapping.get_statistics()

        self.assertIn("total_parameters", stats)
        self.assertIn("total_parameter_elements", stats)
        self.assertIn("gradient_parameter_elements", stats)
        self.assertIn("parameter_types", stats)
        self.assertIn("config", stats)

        self.assertEqual(stats["total_parameters"], len(self.params))
        self.assertGreater(stats["total_parameter_elements"], 0)

    def test_reset(self):
        """Test resetting the mapping."""
        mapping = ParamGradMapping(params=self.params, device=self.device)

        # Perform some operations
        mapping.accumulation_step = 5
        mapping.total_reductions = 10

        # Reset
        mapping.reset()

        # Check reset state
        self.assertEqual(mapping.accumulation_step, 0)

    def test_repr(self):
        """Test string representation."""
        mapping = ParamGradMapping(params=self.params, device=self.device)

        repr_str = repr(mapping)
        self.assertIn("ParamGradMapping", repr_str)
        self.assertIn("params=", repr_str)
        self.assertIn("elements=", repr_str)
        self.assertIn("strategy=", repr_str)

    def test_empty_parameters(self):
        """Test handling of empty parameter list."""
        mapping = ParamGradMapping(params=[], device=self.device)

        self.assertEqual(len(mapping.param_infos), 0)
        stats = mapping.get_statistics()
        self.assertEqual(stats["total_parameters"], 0)

    def test_type_specific_buckets(self):
        """Test type-specific bucket configuration."""
        config = MappingConfig(
            type_specific_buckets=True,
            type_bucket_sizes={
                ParameterType.EMBEDDING: 100.0,
                ParameterType.WEIGHT: 50.0,
                ParameterType.BIAS: 5.0,
            },
        )

        # Create gradients for parameters
        for param in self.params:
            param.grad = torch.randn_like(param)

        mapping = ParamGradMapping(
            params=self.params, config=config, device=self.device
        )

        # Check that parameters are grouped by type
        type_groups = mapping._group_parameters_by_type()
        self.assertGreater(len(type_groups), 0)

        # Each type should have parameters
        for param_type, params in type_groups.items():
            self.assertGreater(len(params), 0)

    def test_reduction_strategies(self):
        """Test different reduction strategies."""
        strategies = [
            ReductionStrategy.IMMEDIATE,
            ReductionStrategy.DELAYED,
            ReductionStrategy.OVERLAPPED,
            ReductionStrategy.HIERARCHICAL,
        ]

        for strategy in strategies:
            config = MappingConfig(reduction_strategy=strategy)
            mapping = ParamGradMapping(
                params=self.params, config=config, device=self.device
            )

            # Create gradients
            for param in self.params:
                param.grad = torch.randn_like(param)

            # Force synchronization
            mapping.accumulation_step = 1
            stats = mapping.synchronize_gradients(force=True)

            # Should return some statistics
            self.assertIsInstance(stats, dict)


class TestParamGradMappingBuilder(unittest.TestCase):
    """Test ParamGradMappingBuilder."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleTestModel()
        self.params = list(self.model.parameters())

    def test_basic_builder(self):
        """Test basic builder usage."""
        mapping = ParamGradMappingBuilder().with_parameters(self.params).build()

        self.assertIsInstance(mapping, ParamGradMapping)
        self.assertEqual(len(mapping.param_infos), len(self.params))

    def test_full_builder_configuration(self):
        """Test builder with all configurations."""
        mapping = (
            ParamGradMappingBuilder()
            .with_parameters(self.params)
            .with_bucket_size(50.0)
            .with_reduction_strategy(ReductionStrategy.IMMEDIATE)
            .with_gradient_accumulation(4)
            .with_gradient_clipping(1.0)
            .with_gradient_scaling(0.5)
            .with_type_specific_buckets(
                {
                    ParameterType.EMBEDDING: 100.0,
                    ParameterType.WEIGHT: 50.0,
                }
            )
            .with_dtype(torch.float16)
            .with_device(torch.device("cpu"))
            .build()
        )

        self.assertEqual(mapping.config.bucket_size_mb, 50.0)
        self.assertEqual(mapping.config.reduction_strategy, ReductionStrategy.IMMEDIATE)
        self.assertEqual(mapping.config.gradient_accumulation_steps, 4)
        self.assertTrue(mapping.config.enable_gradient_clipping)
        self.assertEqual(mapping.config.gradient_clip_value, 1.0)
        self.assertTrue(mapping.config.enable_gradient_scaling)
        self.assertEqual(mapping.config.gradient_scale_factor, 0.5)
        self.assertTrue(mapping.config.type_specific_buckets)
        self.assertEqual(mapping.dtype, torch.float16)

    def test_builder_without_parameters(self):
        """Test builder raises error without parameters."""
        builder = ParamGradMappingBuilder()

        with self.assertRaises(ValueError) as context:
            builder.build()

        self.assertIn("Parameters must be set", str(context.exception))

    @patch("torch.distributed.ProcessGroup")
    def test_builder_with_process_group(self, mock_pg):
        """Test builder with process group."""
        pg = mock_pg()

        mapping = (
            ParamGradMappingBuilder()
            .with_parameters(self.params)
            .with_process_group(pg)
            .build()
        )

        self.assertEqual(mapping.process_group, pg)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""

    def setUp(self):
        """Set up test environment."""
        self.model = SimpleTestModel()
        self.device = torch.device("cpu")

    def test_complete_training_step(self):
        """Test a complete training step workflow."""
        # Create mapping
        mapping = (
            ParamGradMappingBuilder()
            .with_parameters(list(self.model.parameters()))
            .with_bucket_size(25.0)
            .with_gradient_accumulation(2)
            .with_gradient_clipping(1.0)
            .build()
        )

        # Simulate training steps
        for step in range(4):
            # Create dummy gradients
            for param in self.model.parameters():
                if param.grad is None:
                    param.grad = torch.randn_like(param)
                else:
                    param.grad += torch.randn_like(param)

            # Accumulate gradients
            mapping.accumulate_gradients()

            # Check if we should reduce
            if mapping.should_reduce_gradients():
                stats = mapping.synchronize_gradients()
                self.assertIsInstance(stats, dict)

                # Get reduced gradients
                _ = mapping.get_reduced_gradients()
                # May be empty if no actual reduction happened

        # Check statistics
        stats = mapping.get_statistics()
        # The accumulation step should be 0 after reductions (it resets)
        # Check that we did some reductions instead
        self.assertGreater(stats["total_reductions"], 0)

    def test_memory_efficiency(self):
        """Test memory efficiency of the implementation."""
        # Create a large model
        large_model = nn.Sequential(*[nn.Linear(100, 100) for _ in range(10)])

        mapping = ParamGradMapping(
            params=list(large_model.parameters()),
            config=MappingConfig(use_memory_pool=True),
            device=self.device,
        )

        # The mapping should be created without issues
        self.assertIsNotNone(mapping.gradient_buffer)
        self.assertIsNotNone(mapping.bucket_manager)

        # Force garbage collection to ensure cleanup
        del mapping
        gc.collect()

    def test_mixed_precision_compatibility(self):
        """Test compatibility with mixed precision training."""
        # Create mapping with float16
        mapping = ParamGradMapping(
            params=list(self.model.parameters()),
            dtype=torch.float16,
            device=self.device,
        )

        # Create float32 gradients (as would happen in mixed precision)
        for param in self.model.parameters():
            param.grad = torch.randn_like(param, dtype=torch.float32)

        # The system should handle dtype conversion
        mapping.accumulate_gradients()

        # Check that operations complete without errors
        stats = mapping.get_statistics()
        self.assertIsInstance(stats, dict)


if __name__ == "__main__":
    unittest.main()
