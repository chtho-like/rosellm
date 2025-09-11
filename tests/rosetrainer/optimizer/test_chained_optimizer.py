"""Comprehensive tests for ChainedOptimizer functionality.

Tests include:
- Basic functionality (step, zero_grad, state_dict)
- Multi-optimizer management
- Parameter group handling
- MoE model optimization
- State dict save/load
- Integration with distributed training
- Error handling and edge cases
"""

import unittest

import torch
import torch.nn as nn

from rosellm.rosetrainer.optimizer import (
    SGD,
    Adam,
    AdamW,
    ChainedOptimizer,
    OptimFactory,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(
        self, input_size: int = 10, hidden_size: int = 20, output_size: int = 10
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        return output  # type: ignore[no-any-return]


class MoELayer(nn.Module):
    """Mock MoE layer for testing."""

    def __init__(self, hidden_size: int = 20, num_experts: int = 4):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(hidden_size, num_experts)

        # Mark expert parameters
        for expert in self.experts:
            for param in expert.parameters():
                setattr(param, "allreduce", False)  # Mark as expert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores = self.gate(x)
        expert_idx = gate_scores.argmax(dim=-1)

        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i).unsqueeze(-1)
            if mask.any():
                output = output + mask * expert(x)
        return output


class MoEModel(nn.Module):
    """Mock MoE model for testing."""

    def __init__(
        self, input_size: int = 10, hidden_size: int = 20, output_size: int = 10
    ):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.moe_layer = MoELayer(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.moe_layer(x)
        output = self.output(x)
        return output  # type: ignore[no-any-return]


class TestChainedOptimizer(unittest.TestCase):
    """Test suite for ChainedOptimizer."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_single_optimizer(self):
        """Test ChainedOptimizer with a single optimizer."""
        model = SimpleModel()
        optimizer = Adam(model.parameters(), lr=1e-3)
        chained = ChainedOptimizer([optimizer])

        # Check initialization
        self.assertEqual(len(chained.chained_optimizers), 1)
        self.assertEqual(len(chained.param_groups), 1)

        # Test step
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.mean()
        loss.backward()

        chained.step()
        chained.zero_grad()

        # Verify gradients were cleared (zero_grad sets to zero, not None by default)
        for param in model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.all(param.grad == 0))

    def test_multiple_optimizers(self):
        """Test ChainedOptimizer with multiple optimizers."""
        model = SimpleModel()

        # Split parameters
        fc1_params = list(model.fc1.parameters())
        fc2_params = list(model.fc2.parameters())

        # Create separate optimizers
        opt1 = Adam(fc1_params, lr=1e-3)
        opt2 = SGD(fc2_params, lr=1e-2, momentum=0.9)

        chained = ChainedOptimizer([opt1, opt2])

        # Check initialization
        self.assertEqual(len(chained.chained_optimizers), 2)
        self.assertEqual(len(chained.param_groups), 2)

        # Test optimization
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.mean()
        loss.backward()

        # Store initial parameters
        initial_params = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

        # Optimize
        chained.step()

        # Verify parameters were updated
        for name, param in model.named_parameters():
            self.assertFalse(
                torch.allclose(param, initial_params[name]),
                f"Parameter {name} was not updated",
            )

        chained.zero_grad()

    def test_moe_optimizer(self):
        """Test ChainedOptimizer with MoE model."""
        model = MoEModel()

        # Create MoE optimizer using factory
        optimizer = OptimFactory.create_moe_optimizer(
            model,
            base_lr=1e-3,
            expert_lr_multiplier=0.1,
            weight_decay=0.01,
            optimizer_type="adam",
        )

        self.assertIsInstance(optimizer, ChainedOptimizer)

        # Test optimization
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.mean()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Check learning rates
        lrs = optimizer.get_lr()
        self.assertTrue(len(lrs) > 0)

    def test_state_dict_save_load(self):
        """Test state dict save and load functionality."""
        model = SimpleModel()

        fc1_params = list(model.fc1.parameters())
        fc2_params = list(model.fc2.parameters())

        opt1 = Adam(fc1_params, lr=1e-3)
        opt2 = AdamW(fc2_params, lr=1e-2)

        chained = ChainedOptimizer([opt1, opt2])

        # Run a few optimization steps
        for _ in range(3):
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.mean()
            loss.backward()
            chained.step()
            chained.zero_grad()

        # Save state dict
        state_dict = chained.state_dict()

        # Create new optimizer and load state
        new_opt1 = Adam(fc1_params, lr=1e-3)
        new_opt2 = AdamW(fc2_params, lr=1e-2)
        new_chained = ChainedOptimizer([new_opt1, new_opt2])

        new_chained.load_state_dict(state_dict)

        # Verify state was loaded correctly
        self.assertEqual(new_chained._step_count, chained._step_count)

        # Compare optimizer states
        for i, (opt, new_opt) in enumerate(
            zip(chained.chained_optimizers, new_chained.chained_optimizers)
        ):
            opt_state = opt.state_dict()
            new_opt_state = new_opt.state_dict()

            # Check parameter groups match
            self.assertEqual(
                len(opt_state["param_groups"]),
                len(new_opt_state["param_groups"]),
                f"Optimizer {i} param groups mismatch",
            )

    def test_add_param_group(self):
        """Test adding parameter groups dynamically."""
        model1 = SimpleModel(10, 20, 10)
        model2 = SimpleModel(10, 30, 10)

        opt1 = Adam(model1.parameters(), lr=1e-3)
        chained = ChainedOptimizer([opt1])

        initial_groups = len(chained.param_groups)

        # Add new parameter group
        new_group = {
            "params": list(model2.parameters()),
            "_optimizer_idx": 0,
            "lr": 1e-4,
        }

        chained.add_param_group(new_group)

        # Verify group was added
        self.assertEqual(len(chained.param_groups), initial_groups + 1)

    def test_learning_rate_adjustment(self):
        """Test learning rate getter and setter."""
        model = SimpleModel()

        fc1_params = list(model.fc1.parameters())
        fc2_params = list(model.fc2.parameters())

        opt1 = Adam(fc1_params, lr=1e-3)
        opt2 = SGD(fc2_params, lr=1e-2)

        chained = ChainedOptimizer([opt1, opt2])

        # Get initial learning rates
        initial_lrs = chained.get_lr()
        self.assertEqual(len(initial_lrs), 2)
        self.assertAlmostEqual(initial_lrs[0], 1e-3)
        self.assertAlmostEqual(initial_lrs[1], 1e-2)

        # Set new learning rates
        new_lrs = [5e-4, 5e-3]
        chained.set_lr(new_lrs)

        updated_lrs = chained.get_lr()
        self.assertAlmostEqual(updated_lrs[0], 5e-4)
        self.assertAlmostEqual(updated_lrs[1], 5e-3)

        # Set single learning rate for all
        chained.set_lr(1e-4)
        uniform_lrs = chained.get_lr()
        for lr in uniform_lrs:
            self.assertAlmostEqual(lr, 1e-4)

    def test_parameter_lookup(self):
        """Test parameter to optimizer mapping."""
        model = SimpleModel()

        fc1_params = list(model.fc1.parameters())
        fc2_params = list(model.fc2.parameters())

        opt1 = Adam(fc1_params, lr=1e-3)
        opt2 = SGD(fc2_params, lr=1e-2)

        chained = ChainedOptimizer([opt1, opt2])

        # Check fc1 parameters map to opt1
        for param in fc1_params:
            result = chained.get_optimizer_for_param(param)
            self.assertIsNotNone(result)
            assert result is not None  # Type hint for pyright
            opt, idx = result
            self.assertEqual(opt, opt1)
            self.assertEqual(idx, 0)

        # Check fc2 parameters map to opt2
        for param in fc2_params:
            result = chained.get_optimizer_for_param(param)
            self.assertIsNotNone(result)
            assert result is not None  # Type hint for pyright
            opt, idx = result
            self.assertEqual(opt, opt2)
            self.assertEqual(idx, 0)

    def test_mixed_precision_integration(self):
        """Test ChainedOptimizer with mixed precision training."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        model = SimpleModel().cuda()

        fc1_params = list(model.fc1.parameters())
        fc2_params = list(model.fc2.parameters())

        opt1 = Adam(fc1_params, lr=1e-3)
        opt2 = SGD(fc2_params, lr=1e-2)

        grad_scaler = torch.cuda.amp.GradScaler()
        chained = ChainedOptimizer([opt1, opt2], grad_scaler=grad_scaler)

        # Mixed precision training step
        with torch.cuda.amp.autocast():
            x = torch.randn(4, 10).cuda()
            y = model(x)
            loss = y.mean()

        grad_scaler.scale(loss).backward()
        chained.step()
        chained.zero_grad()

    def test_metrics_collection(self):
        """Test metrics collection functionality."""
        model = SimpleModel()
        optimizer = Adam(model.parameters(), lr=1e-3)
        chained = ChainedOptimizer([optimizer], enable_metrics=True)

        # Run optimization steps
        for _ in range(5):
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.mean()
            loss.backward()
            chained.step()
            chained.zero_grad()

        # Get metrics
        metrics = chained.get_metrics()
        self.assertIsNotNone(metrics)
        assert metrics is not None  # Type hint for pyright
        self.assertIn("total_steps", metrics)
        self.assertEqual(metrics["total_steps"], 5)

    def test_error_handling(self):
        """Test error handling in ChainedOptimizer."""
        # Test empty optimizer list
        with self.assertRaises(ValueError):
            ChainedOptimizer([])

        # Test duplicate parameters
        model = SimpleModel()
        params = list(model.parameters())

        opt1 = Adam(params, lr=1e-3)
        opt2 = SGD(params, lr=1e-2)  # Same parameters

        with self.assertRaises(ValueError):
            ChainedOptimizer([opt1, opt2])

        # Test invalid optimizer index in add_param_group
        opt = Adam(model.parameters(), lr=1e-3)
        chained = ChainedOptimizer([opt])

        with self.assertRaises(ValueError):
            chained.add_param_group(
                {
                    "params": [torch.randn(10, 10)],
                    "_optimizer_idx": 999,  # Invalid index
                }
            )

    def test_factory_methods(self):
        """Test OptimFactory creation methods."""
        model = MoEModel()

        # Test MoE optimizer creation
        moe_opt = OptimFactory.create_moe_optimizer(
            model, base_lr=1e-3, expert_lr_multiplier=0.1, optimizer_type="adamw"
        )
        self.assertIsInstance(moe_opt, ChainedOptimizer)

        # Test layer-wise optimizer creation
        layer_opt = OptimFactory.create_layer_wise_optimizer(
            model, base_lr=1e-4, lr_decay=0.9, num_layers=3
        )
        self.assertIsInstance(layer_opt, torch.optim.Optimizer)

        # Test parameter-efficient optimizer
        pe_opt = OptimFactory.create_parameter_efficient_optimizer(
            model, base_lr=1e-4, lora_lr_multiplier=10.0
        )
        self.assertIsInstance(pe_opt, torch.optim.Optimizer)

    def test_chained_optimizer_repr(self):
        """Test string representation of ChainedOptimizer."""
        model = SimpleModel()

        fc1_params = list(model.fc1.parameters())
        fc2_params = list(model.fc2.parameters())

        opt1 = Adam(fc1_params, lr=1e-3)
        opt2 = SGD(fc2_params, lr=1e-2)

        chained = ChainedOptimizer([opt1, opt2])

        repr_str = repr(chained)
        self.assertIn("ChainedOptimizer", repr_str)
        self.assertIn("Adam", repr_str)
        self.assertIn("SGD", repr_str)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_optimization(self):
        """Test ChainedOptimizer with CUDA tensors."""
        model = SimpleModel().cuda()

        fc1_params = list(model.fc1.parameters())
        fc2_params = list(model.fc2.parameters())

        opt1 = Adam(fc1_params, lr=1e-3)
        opt2 = AdamW(fc2_params, lr=1e-2)

        chained = ChainedOptimizer([opt1, opt2])

        # Run optimization on CUDA
        x = torch.randn(4, 10).cuda()
        y = model(x)
        loss = y.mean()
        loss.backward()

        # Store initial parameters
        initial_params = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }

        chained.step()

        # Verify parameters were updated
        for name, param in model.named_parameters():
            self.assertFalse(
                torch.allclose(param, initial_params[name]),
                f"Parameter {name} was not updated on CUDA",
            )

        chained.zero_grad()

    def test_closure_support(self):
        """Test ChainedOptimizer with closure function."""
        model = SimpleModel()
        optimizer = Adam(model.parameters(), lr=1e-3)
        chained = ChainedOptimizer([optimizer])

        def closure():
            chained.zero_grad()
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.mean()
            loss.backward()
            return loss.item()

        loss_value = chained.step(closure)
        self.assertIsNotNone(loss_value)
        self.assertIsInstance(loss_value, float)

    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        model = SimpleModel()

        fc1_params = list(model.fc1.parameters())
        fc2_params = list(model.fc2.parameters())

        opt1 = Adam(fc1_params, lr=1e-3)
        opt2 = SGD(fc2_params, lr=1e-2)

        # Test with global gradient clipping
        chained = ChainedOptimizer(
            [opt1, opt2],
            grad_clip_norm=1.0,
            grad_clip_value=0.5,
        )

        x = torch.randn(4, 10)
        y = model(x)
        loss = y.mean()
        loss.backward()

        # Amplify gradients to ensure clipping is triggered
        for param in model.parameters():
            if param.grad is not None:
                param.grad *= 100

        chained.step()
        chained.zero_grad()

        # Test with per-optimizer gradient clipping
        chained2 = ChainedOptimizer(
            [opt1, opt2],
            grad_clip_norm=[0.5, 1.0],
            grad_clip_value=[None, 0.1],
        )

        x = torch.randn(4, 10)
        y = model(x)
        loss = y.mean()
        loss.backward()

        chained2.step()
        chained2.zero_grad()

    def test_freeze_unfreeze_optimizer(self):
        """Test freezing and unfreezing optimizer parameters."""
        model = SimpleModel()

        fc1_params = list(model.fc1.parameters())
        fc2_params = list(model.fc2.parameters())

        opt1 = Adam(fc1_params, lr=1e-3)
        opt2 = SGD(fc2_params, lr=1e-2)

        chained = ChainedOptimizer([opt1, opt2])

        # Freeze first optimizer's parameters
        chained.freeze_optimizer(0)

        for param in fc1_params:
            self.assertFalse(param.requires_grad)
        for param in fc2_params:
            self.assertTrue(param.requires_grad)

        # Unfreeze first optimizer's parameters
        chained.unfreeze_optimizer(0)

        for param in fc1_params:
            self.assertTrue(param.requires_grad)

    def test_optimizer_statistics(self):
        """Test getting optimizer statistics."""
        model = SimpleModel()

        fc1_params = list(model.fc1.parameters())
        fc2_params = list(model.fc2.parameters())

        opt1 = Adam(fc1_params, lr=1e-3)
        opt2 = AdamW(fc2_params, lr=1e-2)

        chained = ChainedOptimizer([opt1, opt2])

        stats = chained.get_optimizer_statistics()

        self.assertEqual(len(stats), 2)
        self.assertIn("type", stats[0])
        self.assertIn("total_params", stats[0])
        self.assertIn("trainable_params", stats[0])
        self.assertIn("learning_rates", stats[0])

        self.assertEqual(stats[0]["type"], "Adam")
        self.assertEqual(stats[1]["type"], "AdamW")
        self.assertGreater(stats[0]["total_params"], 0)
        self.assertGreater(stats[1]["total_params"], 0)

    def test_thread_safety(self):
        """Test thread-safe operations."""
        model = SimpleModel()
        optimizer = Adam(model.parameters(), lr=1e-3)

        # Create thread-safe chained optimizer
        chained = ChainedOptimizer([optimizer], thread_safe=True)

        # This should not raise any errors
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.mean()
        loss.backward()

        chained.step()
        chained.zero_grad()

        # Test state dict operations with thread safety
        state_dict = chained.state_dict()
        chained.load_state_dict(state_dict)

    def test_pickle_support(self):
        """Test pickling and unpickling support."""
        import pickle

        model = SimpleModel()
        optimizer = Adam(model.parameters(), lr=1e-3)
        chained = ChainedOptimizer([optimizer], thread_safe=True)

        # Run a few steps
        for _ in range(3):
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.mean()
            loss.backward()
            chained.step()
            chained.zero_grad()

        # Pickle and unpickle
        pickled = pickle.dumps(chained)
        unpickled = pickle.loads(pickled)

        # Verify unpickled optimizer works
        self.assertEqual(unpickled._step_count, chained._step_count)
        self.assertEqual(
            len(unpickled.chained_optimizers), len(chained.chained_optimizers)
        )


class TestOptimFactory(unittest.TestCase):
    """Test suite for OptimFactory."""

    def test_create_optimizer(self):
        """Test basic optimizer creation."""
        model = SimpleModel()

        # Test Adam creation
        adam = OptimFactory.create_optimizer(
            model.parameters(), optimizer_type="adam", lr=1e-3
        )
        self.assertIsInstance(adam, Adam)

        # Test SGD creation
        sgd = OptimFactory.create_optimizer(
            model.parameters(), optimizer_type="sgd", lr=1e-2, momentum=0.9
        )
        self.assertIsInstance(sgd, SGD)

    def test_create_chained_optimizer(self):
        """Test ChainedOptimizer creation through factory."""
        model = SimpleModel()

        fc1_params = list(model.fc1.parameters())
        fc2_params = list(model.fc2.parameters())

        param_groups = [
            {"params": fc1_params, "optimizer_idx": 0},
            {"params": fc2_params, "optimizer_idx": 1},
        ]

        optimizer_configs = [
            {"type": "adam", "lr": 1e-3},
            {"type": "sgd", "lr": 1e-2, "momentum": 0.9},
        ]

        chained = OptimFactory.create_chained_optimizer(param_groups, optimizer_configs)

        self.assertIsInstance(chained, ChainedOptimizer)
        self.assertEqual(len(chained.chained_optimizers), 2)

    def test_invalid_optimizer_type(self):
        """Test error handling for invalid optimizer type."""
        model = SimpleModel()

        with self.assertRaises(ValueError):
            OptimFactory.create_optimizer(
                model.parameters(), optimizer_type="invalid_optimizer"
            )


if __name__ == "__main__":
    unittest.main()
