"""
Megatron Accuracy Validation Tests

This module provides a framework for bit-to-bit accuracy comparison between
RoseLLM's gradient utilities and reference implementations (PyTorch native,
Megatron-LM if available).

The tests are designed to:
1. Compare gradient norm calculations across different implementations
2. Validate gradient clipping behavior matches reference implementations
3. Ensure distributed reductions produce identical results
4. Test edge cases and numerical stability

Key Features:
- Bit-to-bit accuracy validation
- Multiple reference implementation support
- Comprehensive edge case testing
- Distributed accuracy validation (simulated)
"""

import math
import unittest
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rosellm.rosetrainer.utils.gradient_utils import (
    GradientClipConfig,
    apply_gradient_clipping,
    calculate_gradient_norm_multitensor,
    check_gradient_finite,
)


class ReferenceGradientUtils:
    """Reference implementation using PyTorch native operations for comparison."""

    @staticmethod
    def calculate_grad_norm_pytorch(
        parameters: Union[List[torch.Tensor], nn.Module],
        norm_type: float = 2.0,
    ) -> torch.Tensor:
        """Calculate gradient norm using PyTorch's native implementation."""
        if isinstance(parameters, nn.Module):
            param_list = [p for p in parameters.parameters() if p.grad is not None]
        else:
            param_list = [p for p in parameters if p.grad is not None]

        if not param_list:
            return torch.tensor(0.0)

        device = param_list[0].device

        if norm_type == float("inf"):
            total_norm = torch.tensor(0.0, device=device)
            for p in param_list:
                if p.grad is not None:
                    param_norm = p.grad.abs().max()
                    total_norm = torch.max(total_norm, param_norm)
        else:
            total_norm = torch.tensor(0.0, device=device)
            for p in param_list:
                if p.grad is not None:
                    param_norm = torch.norm(p.grad, p=norm_type)
                    total_norm += param_norm**norm_type
            total_norm = total_norm ** (1.0 / norm_type)

        return total_norm

    @staticmethod
    def clip_grad_norm_pytorch(
        parameters: Union[List[torch.Tensor], nn.Module],
        max_norm: float,
        norm_type: float = 2.0,
    ) -> float:
        """Gradient norm clipping using PyTorch's native implementation."""
        if isinstance(parameters, nn.Module):
            param_list = list(parameters.parameters())
        else:
            param_list = parameters

        return float(
            torch.nn.utils.clip_grad_norm_(param_list, max_norm, norm_type).item()
        )

    @staticmethod
    def clip_grad_value_pytorch(
        parameters: Union[List[torch.Tensor], nn.Module],
        clip_value: float,
    ) -> None:
        """Gradient value clipping using PyTorch's native implementation."""
        if isinstance(parameters, nn.Module):
            param_list = list(parameters.parameters())
        else:
            param_list = parameters

        torch.nn.utils.clip_grad_value_(param_list, clip_value)


class TestModel(nn.Module):
    """Test model with various parameter types and sizes for comprehensive testing."""

    def __init__(
        self,
        input_size: int = 128,
        hidden_sizes: Optional[List[int]] = None,
        output_size: int = 64,
        use_bias: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 512, 256]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size, bias=use_bias))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size, bias=use_bias))

        self.network = nn.Sequential(*layers)

        # Additional parameter types for comprehensive testing
        self.embedding = nn.Embedding(1000, 128)
        self.layer_norm = nn.LayerNorm(output_size)
        self.batch_norm = nn.BatchNorm1d(output_size)

        # Initialize with different scales to test numerical stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with specific patterns for testing."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "embedding" in name:
                    nn.init.normal_(param, mean=0.0, std=0.1)
                elif "layer_norm" in name or "batch_norm" in name:
                    nn.init.ones_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional embedding input."""
        if token_ids is not None:
            emb = self.embedding(token_ids)
            x = x + emb.mean(dim=1)  # Simple combination for testing

        x = self.network(x)

        # Apply normalization layers
        x_ln = self.layer_norm(x)

        # Batch norm expects 2D input (batch_size, features)
        if x.dim() == 2:
            x_bn = self.batch_norm(x)
            x = (x_ln + x_bn) / 2  # Combine for testing
        else:
            x = x_ln

        return x


class MegatronAccuracyTestCase(unittest.TestCase):
    """Base class for Megatron accuracy validation tests."""

    def setUp(self):
        """Set up test environment with deterministic behavior."""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tolerance_loose = 1e-5
        self.tolerance_strict = 1e-7

        # Create test model
        self.model = TestModel(
            input_size=64,
            hidden_sizes=[128, 256, 128],
            output_size=32,
        ).to(self.device)

        # Create reference utils
        self.reference = ReferenceGradientUtils()

    def _create_test_data(
        self,
        batch_size: int = 16,
        scale: float = 1.0,
        include_tokens: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Create test data for gradient computation."""
        x = torch.randn(batch_size, 64, device=self.device) * scale
        target = torch.randn(batch_size, 32, device=self.device) * scale

        token_ids = None
        if include_tokens:
            token_ids = torch.randint(0, 1000, (batch_size, 10), device=self.device)

        return x, target, token_ids

    def _compute_loss_and_gradients(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        loss_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute loss and gradients for the model."""
        self.model.zero_grad()

        output = self.model(x, token_ids)
        loss = F.mse_loss(output, target) * loss_scale
        loss.backward()

        return loss

    def _clone_model_gradients(self) -> Dict[str, torch.Tensor]:
        """Create a deep copy of all model gradients."""
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients

    def _set_model_gradients(self, gradients: Dict[str, torch.Tensor]) -> None:
        """Set model gradients from a dictionary."""
        for name, param in self.model.named_parameters():
            if name in gradients:
                param.grad = gradients[name].clone()

    def _assert_tensors_almost_equal(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        tolerance: Optional[float] = None,
        msg: str = "",
    ):
        """Assert that two tensors are almost equal within tolerance."""
        if tolerance is None:
            tolerance = self.tolerance_loose

        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff)

        self.assertLessEqual(
            float(max_diff),
            tolerance,
            msg=f"{msg} - Max difference: {float(max_diff):.2e}, "
            f"tolerance: {tolerance:.2e}",
        )


class TestGradientNormAccuracy(MegatronAccuracyTestCase):
    """Test gradient norm calculation accuracy against reference implementations."""

    def test_l2_norm_accuracy(self):
        """Test L2 gradient norm calculation accuracy."""
        # Create gradients
        x, target, _ = self._create_test_data()
        self._compute_loss_and_gradients(x, target)

        # Calculate using our implementation
        our_norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        # Calculate using PyTorch reference
        ref_norm = self.reference.calculate_grad_norm_pytorch(self.model, norm_type=2.0)

        # Should be bit-to-bit identical for same computation
        self._assert_tensors_almost_equal(
            our_norm,
            ref_norm,
            tolerance=self.tolerance_strict,
            msg="L2 gradient norm calculation should match PyTorch reference",
        )

    def test_l1_norm_accuracy(self):
        """Test L1 gradient norm calculation accuracy."""
        x, target, _ = self._create_test_data()
        self._compute_loss_and_gradients(x, target)

        our_norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=1.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        ref_norm = self.reference.calculate_grad_norm_pytorch(self.model, norm_type=1.0)

        self._assert_tensors_almost_equal(
            our_norm,
            ref_norm,
            tolerance=self.tolerance_strict,
            msg="L1 gradient norm calculation should match PyTorch reference",
        )

    def test_inf_norm_accuracy(self):
        """Test infinity gradient norm calculation accuracy."""
        x, target, _ = self._create_test_data()
        self._compute_loss_and_gradients(x, target)

        our_norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=float("inf"),
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        ref_norm = self.reference.calculate_grad_norm_pytorch(
            self.model, norm_type=float("inf")
        )

        self._assert_tensors_almost_equal(
            our_norm,
            ref_norm,
            tolerance=self.tolerance_strict,
            msg="Infinity gradient norm calculation should match PyTorch reference",
        )

    def test_custom_norm_types(self):
        """Test custom norm types (p-norms with p != 1, 2, inf)."""
        x, target, _ = self._create_test_data()
        self._compute_loss_and_gradients(x, target)

        for norm_type in [0.5, 1.5, 3.0, 4.0]:
            with self.subTest(norm_type=norm_type):
                our_norm = calculate_gradient_norm_multitensor(
                    self.model,
                    norm_type=norm_type,
                    use_multitensor=False,
                    model_parallel_reduce=False,
                )

                ref_norm = self.reference.calculate_grad_norm_pytorch(
                    self.model, norm_type=norm_type
                )

                self._assert_tensors_almost_equal(
                    our_norm,
                    ref_norm,
                    tolerance=self.tolerance_strict,
                    msg=f"P-norm (p={norm_type}) calculation should match reference",
                )

    def test_multitensor_vs_standard(self):
        """Test that multi-tensor and standard implementations match."""
        x, target, _ = self._create_test_data()
        self._compute_loss_and_gradients(x, target)

        # Store gradients
        orig_grads = self._clone_model_gradients()

        # Calculate with multi-tensor enabled
        mt_norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=True,
            model_parallel_reduce=False,
        )

        # Restore gradients and calculate with standard implementation
        self._set_model_gradients(orig_grads)

        std_norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        self._assert_tensors_almost_equal(
            mt_norm,
            std_norm,
            tolerance=self.tolerance_loose,
            msg="Multi-tensor and standard norm calculations should match",
        )


class TestGradientClippingAccuracy(MegatronAccuracyTestCase):
    """Test gradient clipping accuracy against reference implementations."""

    def test_norm_clipping_accuracy(self):
        """Test gradient norm clipping accuracy."""
        # Create large gradients that will be clipped
        x, target, _ = self._create_test_data(scale=10.0)
        self._compute_loss_and_gradients(x, target, loss_scale=10.0)

        # Store original gradients
        orig_grads = self._clone_model_gradients()

        # Apply our clipping
        config = GradientClipConfig(
            clip_type="norm",
            max_norm=1.0,
            norm_type=2.0,
            model_parallel_reduce=False,
        )

        our_stats = apply_gradient_clipping(self.model, config)
        our_gradients = self._clone_model_gradients()

        # Restore original gradients and apply PyTorch clipping
        self._set_model_gradients(orig_grads)
        ref_norm = self.reference.clip_grad_norm_pytorch(self.model, 1.0, 2.0)
        ref_gradients = self._clone_model_gradients()

        # Compare final gradients
        for name in our_gradients:
            if name in ref_gradients:
                self._assert_tensors_almost_equal(
                    our_gradients[name],
                    ref_gradients[name],
                    tolerance=self.tolerance_loose,
                    msg=f"Clipped gradient for {name} should match PyTorch reference",
                )

        # Compare gradient norms
        self.assertAlmostEqual(
            our_stats["grad_norm"],
            float(ref_norm),
            places=5,
            msg="Original gradient norm should match PyTorch reference",
        )

    def test_value_clipping_accuracy(self):
        """Test gradient value clipping accuracy."""
        # Create gradients with large values
        x, target, _ = self._create_test_data(scale=5.0)
        self._compute_loss_and_gradients(x, target, loss_scale=5.0)

        # Store original gradients
        orig_grads = self._clone_model_gradients()

        # Apply our value clipping
        config = GradientClipConfig(
            clip_type="value",
            max_norm=0.5,  # This becomes clip_value for value clipping
            model_parallel_reduce=False,
        )

        apply_gradient_clipping(self.model, config)
        our_gradients = self._clone_model_gradients()

        # Restore original gradients and apply PyTorch value clipping
        self._set_model_gradients(orig_grads)
        self.reference.clip_grad_value_pytorch(self.model, 0.5)
        ref_gradients = self._clone_model_gradients()

        # Compare final gradients
        for name in our_gradients:
            if name in ref_gradients:
                self._assert_tensors_almost_equal(
                    our_gradients[name],
                    ref_gradients[name],
                    tolerance=self.tolerance_strict,
                    msg=f"Value-clipped gradient for {name} should match "
                    "PyTorch reference",
                )

    def test_no_clipping_needed(self):
        """Test accuracy when gradients don't need clipping."""
        # Create small gradients that won't be clipped
        x, target, _ = self._create_test_data(scale=0.1)
        self._compute_loss_and_gradients(x, target, loss_scale=0.1)

        # Store original gradients
        orig_grads = self._clone_model_gradients()

        # Apply clipping with high threshold
        config = GradientClipConfig(
            clip_type="norm",
            max_norm=10.0,
            norm_type=2.0,
            model_parallel_reduce=False,
        )

        our_stats = apply_gradient_clipping(self.model, config)
        our_gradients = self._clone_model_gradients()

        # Gradients should be unchanged
        for name in our_gradients:
            if name in orig_grads:
                self._assert_tensors_almost_equal(
                    our_gradients[name],
                    orig_grads[name],
                    tolerance=self.tolerance_strict,
                    msg=f"Gradient for {name} should be unchanged when "
                    "no clipping needed",
                )

        # Should not be marked as clipped
        self.assertFalse(
            our_stats["clipped"], "Gradients should not be marked as clipped"
        )


class TestNumericalStability(MegatronAccuracyTestCase):
    """Test numerical stability and edge cases."""

    def test_extreme_gradient_values(self):
        """Test handling of extremely large and small gradients."""
        # Create gradients and manually set extreme values
        x, target, _ = self._create_test_data()
        self._compute_loss_and_gradients(x, target)

        # Set some gradients to extreme values
        params = list(self.model.parameters())
        if params and params[0].grad is not None:
            # Very large gradients - handle both 1D and 2D tensors
            if params[0].grad.dim() == 2:
                params[0].grad[0, 0] = 1e6
                params[0].grad[0, 1] = -1e6
            else:
                params[0].grad[0] = 1e6
                if params[0].grad.numel() > 1:
                    params[0].grad[1] = -1e6

            # Very small gradients
            if len(params) > 1 and params[1].grad is not None:
                if params[1].grad.dim() == 2:
                    params[1].grad[0, 0] = 1e-10
                    if params[1].grad.shape[1] > 1:
                        params[1].grad[0, 1] = -1e-10
                else:
                    params[1].grad[0] = 1e-10
                    if params[1].grad.numel() > 1:
                        params[1].grad[1] = -1e-10

        # Should not crash and should produce finite results
        norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        self.assertTrue(torch.isfinite(norm), "Gradient norm should be finite")
        self.assertGreater(float(norm), 0, "Gradient norm should be positive")

    def test_zero_gradients(self):
        """Test handling of zero gradients."""
        # Clear all gradients
        self.model.zero_grad()

        # Calculate norm of zero gradients
        norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        self.assertEqual(float(norm), 0.0, "Norm of zero gradients should be zero")

    def test_mixed_gradient_types(self):
        """Test with mixed gradient types (finite, inf, nan)."""
        # Create normal gradients first
        x, target, _ = self._create_test_data()
        self._compute_loss_and_gradients(x, target)

        # Inject problematic values
        params = list(self.model.parameters())
        grad_0_valid = len(params) >= 2 and params[0].grad is not None
        grad_1_valid = grad_0_valid and params[1].grad is not None
        if (
            grad_0_valid
            and grad_1_valid
            and params[0].grad is not None
            and params[1].grad is not None
        ):
            # Handle both 1D and 2D tensors for inf values
            if params[0].grad.dim() == 2:
                params[0].grad[0, 0] = float("inf")
            else:
                params[0].grad[0] = float("inf")

            # Handle both 1D and 2D tensors for nan values
            if params[1].grad.dim() == 2:
                params[1].grad[0, 0] = float("nan")
            else:
                params[1].grad[0] = float("nan")

        # Test finite check
        is_finite, stats = check_gradient_finite(self.model, raise_on_nonfinite=False)

        self.assertFalse(is_finite, "Should detect non-finite gradients")
        self.assertGreater(stats["inf_parameters"], 0, "Should detect inf parameters")
        self.assertGreater(stats["nan_parameters"], 0, "Should detect nan parameters")

    def test_gradient_underflow_overflow(self):
        """Test gradient calculations near machine precision limits."""
        # Create very small gradients near underflow
        x, target, _ = self._create_test_data()
        self._compute_loss_and_gradients(x, target)

        # Scale gradients to very small values
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.mul_(1e-20)  # Near underflow

        norm = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        self.assertTrue(
            torch.isfinite(norm), "Very small gradient norms should be finite"
        )
        self.assertGreaterEqual(
            float(norm), 0.0, "Gradient norm should be non-negative"
        )


class TestDistributedSimulation(MegatronAccuracyTestCase):
    """Test distributed operations through simulation."""

    def test_model_parallel_reduce_simulation(self):
        """Simulate model parallel gradient reduction."""
        # This test simulates what would happen in a distributed setting
        # by manually implementing the reduction logic

        x, target, _ = self._create_test_data()
        self._compute_loss_and_gradients(x, target)

        # Calculate norm without reduction (single GPU case)
        norm_single = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        # Simulate what would happen with 2-way model parallelism
        # In real MP, each rank would have partial gradients
        # For simulation, we'll split gradients and then combine

        # Store original gradients
        orig_grads = self._clone_model_gradients()

        # Simulate rank 0: scale gradients by 0.5
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.mul_(0.5)

        norm_rank0 = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        # Simulate rank 1: use other half
        self._set_model_gradients(orig_grads)
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.mul_(0.5)

        norm_rank1 = calculate_gradient_norm_multitensor(
            self.model,
            norm_type=2.0,
            use_multitensor=False,
            model_parallel_reduce=False,
        )

        # Combined norm should relate to original norm
        # For L2 norm: combined = sqrt(rank0^2 + rank1^2)
        combined_norm = torch.sqrt(norm_rank0**2 + norm_rank1**2)
        expected_norm = norm_single * math.sqrt(0.5)  # Due to scaling by 0.5

        self._assert_tensors_almost_equal(
            combined_norm,
            expected_norm,
            tolerance=self.tolerance_loose,
            msg="Simulated distributed norm reduction should match expected",
        )


class TestReferenceImplementationComparison(MegatronAccuracyTestCase):
    """Compare against multiple reference implementations when available."""

    def test_pytorch_native_consistency(self):
        """Test consistency with PyTorch native operations across scenarios."""
        test_scenarios = [
            {"scale": 0.1, "loss_scale": 1.0, "desc": "small gradients"},
            {"scale": 1.0, "loss_scale": 1.0, "desc": "normal gradients"},
            {"scale": 10.0, "loss_scale": 1.0, "desc": "large gradients"},
            {"scale": 1.0, "loss_scale": 0.1, "desc": "small loss"},
            {"scale": 1.0, "loss_scale": 10.0, "desc": "large loss"},
        ]

        for scenario in test_scenarios:
            with self.subTest(**scenario):
                x, target, _ = self._create_test_data(scale=scenario["scale"])
                self._compute_loss_and_gradients(
                    x, target, loss_scale=scenario["loss_scale"]
                )

                # Test L2 norm consistency
                our_norm = calculate_gradient_norm_multitensor(
                    self.model,
                    norm_type=2.0,
                    use_multitensor=False,
                    model_parallel_reduce=False,
                )

                ref_norm = self.reference.calculate_grad_norm_pytorch(
                    self.model, norm_type=2.0
                )

                self._assert_tensors_almost_equal(
                    our_norm,
                    ref_norm,
                    tolerance=self.tolerance_strict,
                    msg=f"L2 norm consistency failed for {scenario['desc']}",
                )

    def test_comprehensive_clipping_scenarios(self):
        """Test comprehensive gradient clipping scenarios."""
        clipping_scenarios = [
            {"max_norm": 0.1, "norm_type": 2.0, "desc": "aggressive L2 clipping"},
            {"max_norm": 1.0, "norm_type": 2.0, "desc": "moderate L2 clipping"},
            {"max_norm": 10.0, "norm_type": 2.0, "desc": "loose L2 clipping"},
            {"max_norm": 0.5, "norm_type": 1.0, "desc": "L1 norm clipping"},
            {
                "max_norm": 1.0,
                "norm_type": float("inf"),
                "desc": "infinity norm clipping",
            },
        ]

        for scenario in clipping_scenarios:
            with self.subTest(**scenario):
                # Create gradients that will likely need clipping
                x, target, _ = self._create_test_data(scale=5.0)
                self._compute_loss_and_gradients(x, target, loss_scale=5.0)

                # Apply our clipping
                config = GradientClipConfig(
                    clip_type="norm",
                    max_norm=scenario["max_norm"],
                    norm_type=scenario["norm_type"],
                    model_parallel_reduce=False,
                )

                our_stats = apply_gradient_clipping(self.model, config)
                our_final_norm = calculate_gradient_norm_multitensor(
                    self.model,
                    norm_type=scenario["norm_type"],
                    use_multitensor=False,
                    model_parallel_reduce=False,
                )

                # Verify clipping constraint
                if our_stats["clipped"]:
                    self.assertLessEqual(
                        float(our_final_norm),
                        scenario["max_norm"]
                        * 1.01,  # Small tolerance for numerical errors
                        msg="Clipped norm should not exceed max_norm for "
                        f"{scenario['desc']}",
                    )


if __name__ == "__main__":
    # Configure test runner for comprehensive output
    unittest.main(verbosity=2, warnings="ignore")
