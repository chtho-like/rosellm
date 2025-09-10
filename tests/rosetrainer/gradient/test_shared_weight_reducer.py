"""Unit tests for SharedWeightGradientReducer."""

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from rosellm.rosetrainer.gradient.shared_weight_reducer import (
    SharedWeightConfig,
    SharedWeightGradientReducer,
)


class MockModelWithSharedWeights(nn.Module):
    """Mock model with shared embeddings for testing."""

    def __init__(self, vocab_size=100, hidden_size=32, share_weights=True):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)
        self.share_embeddings_and_output_weights = share_weights

        if not share_weights:
            self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)

    def shared_embedding_or_output_weight(self):
        """Return shared embedding weight following Megatron-LM pattern."""
        if self.share_embeddings_and_output_weights:
            return self.word_embeddings.weight
        return None

    def forward(self, x):
        return x


class TestSharedWeightGradientReducer:
    """Test cases for SharedWeightGradientReducer."""

    def test_initialization(self):
        """Test reducer initialization."""
        config = MagicMock()
        reducer = SharedWeightGradientReducer(config)

        assert reducer.config == config
        assert reducer._pp_group is None
        assert reducer._embd_group is None
        assert reducer._pos_embd_group is None

    def test_get_main_grad_attr(self):
        """Test gradient attribute detection."""
        config = MagicMock()
        reducer = SharedWeightGradientReducer(config)

        # Test with regular gradient
        param = nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn(10, 10)
        assert reducer._get_main_grad_attr(param) == "grad"

        # Test with main_grad (mixed precision)
        setattr(param, "main_grad", torch.randn(10, 10))
        assert reducer._get_main_grad_attr(param) == "main_grad"

    def test_default_get_word_embedding_weight(self):
        """Test default word embedding weight extraction."""
        config = MagicMock()
        reducer = SharedWeightGradientReducer(config)

        # Test with model having shared_embedding_or_output_weight method
        model = MockModelWithSharedWeights(share_weights=True)
        weight = reducer._default_get_word_embedding_weight(model)
        assert weight is model.word_embeddings.weight

        # Test with model without shared weights
        model_no_share = MockModelWithSharedWeights(share_weights=False)
        weight = reducer._default_get_word_embedding_weight(model_no_share)
        assert weight is None

    def test_default_get_position_embedding_weight(self):
        """Test default position embedding weight extraction."""
        config = MagicMock()
        reducer = SharedWeightGradientReducer(config)

        model = MockModelWithSharedWeights()
        weight = reducer._default_get_position_embedding_weight(model)
        assert weight is model.position_embeddings.weight

    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_process_group_ranks")
    @patch("torch.distributed.all_reduce")
    def test_allreduce_embedding_grad(
        self, mock_all_reduce, mock_get_ranks, mock_get_rank, mock_get_world_size
    ):
        """Test embedding gradient all-reduce."""
        # Setup mocks
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0
        mock_get_ranks.return_value = [0, 1]

        config = MagicMock()
        reducer = SharedWeightGradientReducer(config)

        # Mock pipeline stage detection
        with patch.object(reducer, "_is_first_stage", return_value=True):
            model = MockModelWithSharedWeights()
            model.word_embeddings.weight.grad = torch.randn(100, 32)

            # Create mock process group
            mock_group = MagicMock()

            # Test all-reduce
            def get_weight(m: nn.Module) -> Optional[nn.Parameter]:
                if hasattr(m, "word_embeddings"):
                    word_embeds = getattr(m, "word_embeddings")
                    if isinstance(word_embeds, nn.Module) and hasattr(
                        word_embeds, "weight"
                    ):
                        weight = getattr(word_embeds, "weight")
                        if isinstance(weight, nn.Parameter):
                            return weight
                return None

            reducer._allreduce_embedding_grad(
                model=[model],
                weight_getter=get_weight,
                embd_group=mock_group,
                skip_if_none=True,
            )

            # Verify all_reduce was called
            mock_all_reduce.assert_called_once()
            args = mock_all_reduce.call_args[0]
            assert args[0] is model.word_embeddings.weight.grad

    def test_allreduce_word_embedding_grads_no_group(self):
        """Test word embedding gradient reduction with no process group."""
        config = MagicMock()
        reducer = SharedWeightGradientReducer(config)

        model = MockModelWithSharedWeights()
        model.word_embeddings.weight.grad = torch.randn(100, 32)

        # Should return early if group size is 1
        with patch.object(reducer, "_get_process_group_size", return_value=1):
            reducer.allreduce_word_embedding_grads([model])
            # No error should occur

    @patch("torch.distributed.all_reduce")
    def test_allreduce_shared_params(self, mock_all_reduce):
        """Test all-reduce for arbitrary shared parameters."""
        config = MagicMock()
        reducer = SharedWeightGradientReducer(config)

        # Create shared parameters with gradients
        param1 = nn.Parameter(torch.randn(10, 10))
        param1.grad = torch.randn(10, 10)
        param2 = nn.Parameter(torch.randn(20, 20))
        param2.grad = torch.randn(20, 20)

        shared_params = [
            ("param1", param1),
            ("param2", param2),
        ]

        # Mock process group
        mock_group = MagicMock()

        with patch.object(reducer, "_get_process_group_size", return_value=2):
            reducer.allreduce_shared_params(
                model=[MagicMock()],
                shared_params=shared_params,
                reduce_group=mock_group,
            )

            # Verify all_reduce was called
            mock_all_reduce.assert_called_once()

    def test_shared_weight_config(self):
        """Test SharedWeightConfig initialization."""
        config = SharedWeightConfig(
            share_embeddings_and_output_weights=True,
            share_position_embeddings=True,
            embedding_reduce_group_size=2,
            position_embedding_reduce_group_size=2,
        )

        assert config.share_embeddings_and_output_weights is True
        assert config.share_position_embeddings is True
        assert config.embedding_reduce_group_size == 2
        assert config.position_embedding_reduce_group_size == 2

    def test_process_group_caching(self):
        """Test that process groups are cached properly."""
        config = MagicMock()
        reducer = SharedWeightGradientReducer(config)

        # Create mock process groups
        mock_pp_group = MagicMock()

        with patch(
            "rosellm.rosetrainer.parallelism.parallel_state.get_pipeline_model_parallel_group",  # noqa: E501
            return_value=mock_pp_group,
        ):
            # First access should create and cache
            group1 = reducer.pp_group
            # Second access should return cached
            group2 = reducer.pp_group

            assert group1 is group2
            assert group1 is mock_pp_group

    def test_gradient_reduction_with_mixed_precision(self):
        """Test gradient reduction with mixed precision training."""
        config = MagicMock()
        reducer = SharedWeightGradientReducer(config)

        model = MockModelWithSharedWeights()

        # Simulate mixed precision with main_grad
        embedding_weight = model.word_embeddings.weight
        assert isinstance(embedding_weight, nn.Parameter)
        setattr(embedding_weight, "main_grad", torch.randn(100, 32))

        # Should use main_grad instead of grad
        grad_attr = reducer._get_main_grad_attr(embedding_weight)
        assert grad_attr == "main_grad"

        grad = getattr(embedding_weight, grad_attr)
        assert grad is getattr(embedding_weight, "main_grad")


class TestIntegrationWithGradientFinalizer:
    """Integration tests with GradientFinalizer."""

    @patch("rosellm.rosetrainer.gradient.finalizer.parallel_state")
    def test_gradient_finalizer_integration(self, mock_parallel_state):
        """Test integration with GradientFinalizer."""
        from rosellm.rosetrainer.gradient import (
            GradientFinalizationConfig,
            GradientFinalizer,
        )

        # Setup mock parallel state
        mock_parallel_state.is_initialized.return_value = False

        # Create config with shared weight support
        config = GradientFinalizationConfig(
            sync_strategy="simple",
            share_embeddings_and_output_weights=True,
        )

        model = MockModelWithSharedWeights()

        # Create finalizer
        finalizer = GradientFinalizer(model, config)

        # Verify shared weight reducer was created
        assert finalizer.shared_weight_reducer is not None
        assert isinstance(finalizer.shared_weight_reducer, SharedWeightGradientReducer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
