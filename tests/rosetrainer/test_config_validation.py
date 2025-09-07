#!/usr/bin/env python3
"""
Test suite for configuration validation in config.py.

Tests the comprehensive validation of TrainingConfig including:
- Recursive sub-config validation
- Factory functions
- Legacy config conversion
- Error handling
"""

import unittest

import pytest
import torch
from pydantic import ValidationError

from rosellm.rosetrainer.config import (
    GradientClipType,
    GradientConfig,
    MemoryConfig,
    OptimizerConfig,
    ParallelismConfig,
    PrecisionType,
    TrainingConfig,
    _default_gradient,
    _default_memory,
    _default_optimizer,
    _default_parallelism,
    validate_config,
)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for sub-configurations."""

    def test_default_optimizer_factory(self):
        """Test that default optimizer factory creates valid config."""
        config = _default_optimizer()
        self.assertIsInstance(config, OptimizerConfig)
        self.assertEqual(config.name, "adamw")
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.weight_decay, 0.01)
        self.assertEqual(config.eps, 1e-8)

    def test_default_gradient_factory(self):
        """Test that default gradient factory creates valid config."""
        config = _default_gradient()
        self.assertIsInstance(config, GradientConfig)
        self.assertEqual(config.clip_type, GradientClipType.NORM)
        self.assertEqual(config.clip_value, 1.0)
        self.assertEqual(config.accumulation_steps, 1)

    def test_default_memory_factory(self):
        """Test that default memory factory creates valid config."""
        config = _default_memory()
        self.assertIsInstance(config, MemoryConfig)
        self.assertFalse(config.activation_checkpointing)
        self.assertFalse(config.cpu_offload)
        self.assertEqual(config.zero_optimization_stage, 0)
        self.assertEqual(config.gradient_checkpointing_ratio, 0.0)

    def test_default_parallelism_factory(self):
        """Test that default parallelism factory creates valid config."""
        config = _default_parallelism()
        self.assertIsInstance(config, ParallelismConfig)
        self.assertEqual(config.tensor_parallel_size, 1)
        self.assertEqual(config.pipeline_parallel_size, 1)
        self.assertIsNone(config.data_parallel_size)
        self.assertEqual(config.context_parallel_size, 1)
        self.assertEqual(config.expert_parallel_size, 1)


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig validation and creation."""

    def test_default_training_config(self):
        """Test creating TrainingConfig with defaults."""
        config = TrainingConfig()  # type: ignore[call-arg]
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.num_epochs, 3)
        self.assertIsNone(config.max_steps)
        self.assertEqual(config.precision, PrecisionType.FP32)
        self.assertIsInstance(config.optimizer, OptimizerConfig)
        self.assertIsInstance(config.gradient, GradientConfig)
        self.assertIsInstance(config.memory, MemoryConfig)
        self.assertIsInstance(config.parallelism, ParallelismConfig)

    def test_custom_training_config(self):
        """Test creating TrainingConfig with custom values."""
        config = TrainingConfig(  # type: ignore[call-arg]
            batch_size=64,
            num_epochs=10,
            precision=PrecisionType.FP32,  # Use FP32 for CPU tests
        )
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.num_epochs, 10)
        self.assertEqual(config.precision, PrecisionType.FP32)

    @pytest.mark.gpu
    @unittest.skipIf(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        "BF16 not supported on current hardware",
    )
    def test_custom_training_config_bf16(self):
        """Test creating TrainingConfig with BF16 precision."""
        config = TrainingConfig(  # type: ignore[call-arg]
            batch_size=64,
            num_epochs=10,
            precision=PrecisionType.BF16,
        )
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.num_epochs, 10)
        self.assertEqual(config.precision, PrecisionType.BF16)

    def test_validation_batch_size(self):
        """Test batch size validation."""
        with self.assertRaises(ValidationError):
            TrainingConfig(batch_size=0)  # type: ignore[call-arg]  # Must be >= 1

        with self.assertRaises(ValidationError):
            TrainingConfig(batch_size=-1)  # type: ignore[call-arg]

    def test_validation_epochs(self):
        """Test epoch validation."""
        with self.assertRaises(ValidationError):
            TrainingConfig(num_epochs=0)  # type: ignore[call-arg]  # Must be >= 1

    def test_max_steps_epochs_conflict(self):
        """Test that max_steps and num_epochs conflict is handled."""
        # Both specified should raise error
        with self.assertRaises(ValidationError):
            # Both specified
            TrainingConfig(max_steps=1000, num_epochs=3)  # type: ignore[call-arg]

        # One or the other should work
        # Only max_steps
        config1 = TrainingConfig(max_steps=1000)  # type: ignore[call-arg]
        self.assertEqual(config1.max_steps, 1000)

        # Only num_epochs
        config2 = TrainingConfig(num_epochs=5)  # type: ignore[call-arg]
        self.assertEqual(config2.num_epochs, 5)


class TestConfigConversion(unittest.TestCase):
    """Test configuration conversion and legacy support."""

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "batch_size": 48,
            "num_epochs": 5,
            "seed": 123,
        }
        config = TrainingConfig.from_dict(config_dict)
        self.assertEqual(config.batch_size, 48)
        self.assertEqual(config.num_epochs, 5)
        self.assertEqual(config.seed, 123)

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TrainingConfig(batch_size=24)  # type: ignore[call-arg]
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["batch_size"], 24)
        # Check flattening for backward compatibility
        self.assertIn("learning_rate", config_dict)
        self.assertIn("weight_decay", config_dict)

    def test_legacy_config_conversion(self):
        """Test conversion of legacy config format."""
        legacy_config = {
            "batch_size": 32,
            "learning_rate": 2e-5,  # Old flat structure
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,  # Old name
        }
        config = TrainingConfig.from_dict(legacy_config)
        self.assertEqual(config.optimizer.learning_rate, 2e-5)
        self.assertEqual(config.optimizer.weight_decay, 0.1)
        self.assertEqual(config.gradient.clip_value, 1.0)
        self.assertEqual(config.gradient.clip_type, GradientClipType.NORM)


class TestValidateConfig(unittest.TestCase):
    """Test the validate_config function."""

    def test_validate_dict(self):
        """Test validating a dictionary config."""
        config_dict = {"batch_size": 16}
        validated = validate_config(config_dict)
        self.assertIsInstance(validated, TrainingConfig)
        self.assertEqual(validated.batch_size, 16)

    def test_validate_training_config(self):
        """Test validating an existing TrainingConfig."""
        config = TrainingConfig(batch_size=8)  # type: ignore[call-arg]
        validated = validate_config(config)
        self.assertIsInstance(validated, TrainingConfig)
        self.assertEqual(validated.batch_size, 8)

    def test_validate_invalid_type(self):
        """Test validation with invalid type."""
        with self.assertRaises(TypeError) as cm:
            validate_config("not a config")  # type: ignore[arg-type]
        self.assertIn("Config must be dict or TrainingConfig", str(cm.exception))

    def test_recursive_validation(self):
        """Test that sub-configs are recursively validated."""
        config = TrainingConfig()  # type: ignore[call-arg]
        # Modify a sub-config to test re-validation
        original_lr = config.optimizer.learning_rate
        validated = validate_config(config)
        # Sub-configs should be re-validated
        self.assertEqual(validated.optimizer.learning_rate, original_lr)

    def test_validation_error_handling(self):
        """Test that validation errors are properly raised."""
        bad_config = {
            "batch_size": -1,  # Invalid
        }
        with self.assertRaises(ValidationError):
            validate_config(bad_config)


class TestSubConfigurations(unittest.TestCase):
    """Test individual sub-configuration classes."""

    def test_optimizer_config_validation(self):
        """Test OptimizerConfig validation."""
        # Valid config
        config = OptimizerConfig(  # type: ignore[call-arg]
            name="adam",
            learning_rate=1e-3,
            weight_decay=0.0,
            eps=1e-7,
        )
        self.assertEqual(config.name, "adam")

        # Invalid learning rate
        with self.assertRaises(ValidationError):
            OptimizerConfig(learning_rate=-1)  # type: ignore[call-arg]

        # Invalid weight decay
        with self.assertRaises(ValidationError):
            OptimizerConfig(weight_decay=2.0)  # type: ignore[call-arg]  # Must be <= 1

    def test_gradient_config_validation(self):
        """Test GradientConfig validation."""
        # Valid config with clipping
        config = GradientConfig(  # type: ignore[call-arg]
            clip_type=GradientClipType.VALUE,
            clip_value=5.0,
            accumulation_steps=2,
        )
        self.assertEqual(config.clip_value, 5.0)

        # Missing clip_value when needed
        with self.assertRaises(ValidationError):
            GradientConfig(  # type: ignore[call-arg]
                clip_type=GradientClipType.NORM,
                clip_value=None,  # Required when clip_type != NONE
            )

        # Invalid accumulation steps
        with self.assertRaises(ValidationError):
            # Must be >= 1
            GradientConfig(accumulation_steps=0)  # type: ignore[call-arg]

    def test_memory_config_validation(self):
        """Test MemoryConfig validation."""
        config = MemoryConfig(  # type: ignore[call-arg]
            activation_checkpointing=True,
            cpu_offload=True,
            zero_optimization_stage=2,
            gradient_checkpointing_ratio=0.5,
        )
        self.assertEqual(config.zero_optimization_stage, 2)

        # Invalid gradient checkpointing ratio
        with self.assertRaises(ValidationError):
            # Must be <= 1
            MemoryConfig(gradient_checkpointing_ratio=1.5)  # type: ignore[call-arg]

    def test_parallelism_config_validation(self):
        """Test ParallelismConfig validation."""
        config = ParallelismConfig(  # type: ignore[call-arg]
            tensor_parallel_size=2,
            pipeline_parallel_size=4,
            data_parallel_size=8,
        )
        self.assertEqual(config.tensor_parallel_size, 2)

        # Power of 2 validation
        with self.assertRaises(ValidationError):
            # Not power of 2
            ParallelismConfig(tensor_parallel_size=3)  # type: ignore[call-arg]

        # Valid power of 2
        config2 = ParallelismConfig(  # type: ignore[call-arg]
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=16,  # Power of 2
        )
        self.assertEqual(config2.data_parallel_size, 16)


class TestPrecisionType(unittest.TestCase):
    """Test PrecisionType enum and validation."""

    def test_precision_values(self):
        """Test all precision type values."""
        self.assertEqual(PrecisionType.FP32.value, "fp32")
        self.assertEqual(PrecisionType.FP16.value, "fp16")
        self.assertEqual(PrecisionType.BF16.value, "bf16")
        self.assertEqual(PrecisionType.FP8.value, "fp8")
        self.assertEqual(PrecisionType.MIXED.value, "mixed")

    def test_precision_in_config(self):
        """Test precision type in config."""
        config = TrainingConfig(precision=PrecisionType.FP16)  # type: ignore[call-arg]
        self.assertEqual(config.precision, PrecisionType.FP16)

    @pytest.mark.gpu
    @unittest.skipIf(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        "BF16 not supported on current hardware",
    )
    def test_precision_bf16_from_dict(self):
        """Test BF16 precision from dictionary config."""
        config_dict = {"precision": "bf16"}
        config = TrainingConfig.from_dict(config_dict)
        self.assertEqual(config.precision, PrecisionType.BF16)


class TestBetaValidation(unittest.TestCase):
    """Test beta parameter validation in OptimizerConfig."""

    def test_valid_betas(self):
        """Test valid beta values."""
        config = OptimizerConfig(betas=(0.9, 0.999))  # type: ignore[call-arg]
        self.assertEqual(config.betas, (0.9, 0.999))

        config2 = OptimizerConfig(betas=(0.0, 0.99))  # type: ignore[call-arg]
        self.assertEqual(config2.betas, (0.0, 0.99))

    def test_invalid_betas(self):
        """Test invalid beta values."""
        with self.assertRaises(ValidationError):
            # beta1 must be < 1
            OptimizerConfig(betas=(1.0, 0.999))  # type: ignore[call-arg]

        with self.assertRaises(ValidationError):
            # beta2 must be < 1
            OptimizerConfig(betas=(0.9, 1.0))  # type: ignore[call-arg]

        with self.assertRaises(ValidationError):
            # beta1 must be >= 0
            OptimizerConfig(betas=(-0.1, 0.999))  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
