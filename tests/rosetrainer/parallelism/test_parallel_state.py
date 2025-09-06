"""
Tests for Advanced Parallel State Management System
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from rosellm.rosetrainer.parallelism import parallel_state
from rosellm.rosetrainer.parallelism.parallel_state import (
    NCCLConfig, ParallelismDimension)


class TestParallelState:
    """Test suite for parallel state management"""

    def setup_method(self):
        """Reset parallel state before each test"""
        # Reset global state
        parallel_state._INITIALIZED = False
        parallel_state._WORLD_SIZE = None
        parallel_state._RANK = None
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP = None
        parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = None
        parallel_state._DATA_PARALLEL_GROUP = None
        parallel_state._CONTEXT_PARALLEL_GROUP = None
        parallel_state._EXPERT_MODEL_PARALLEL_GROUP = None

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.new_group")
    def test_basic_initialization(
        self,
        mock_new_group,
        mock_get_rank,
        mock_get_world_size,
        mock_init_process_group,
        mock_is_initialized,
    ):
        """Test basic parallel state initialization"""
        # Setup mocks
        mock_is_initialized.return_value = False
        mock_get_world_size.return_value = 8
        mock_get_rank.return_value = 0
        mock_new_group.return_value = MagicMock()

        # Initialize with TP=2, PP=2, DP=2
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            data_parallel_size=2,
        )

        # Verify initialization
        assert parallel_state.is_initialized()
        assert parallel_state.get_tensor_model_parallel_size() == 2
        assert parallel_state.get_pipeline_model_parallel_size() == 2
        assert parallel_state.get_data_parallel_size() == 2
        assert parallel_state.get_context_parallel_size() == 1
        assert parallel_state.get_expert_model_parallel_size() == 1

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    def test_auto_data_parallel_size(
        self, mock_get_rank, mock_get_world_size, mock_is_initialized
    ):
        """Test automatic data parallel size calculation"""
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 16
        mock_get_rank.return_value = 0

        with patch("torch.distributed.new_group"):
            # Initialize with TP=2, PP=2, auto DP
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=2, pipeline_model_parallel_size=2
            )

            # Should calculate DP = 16 / (2 * 2) = 4
            assert parallel_state.get_data_parallel_size() == 4

    def test_rank_coordinate_conversion(self):
        """Test rank to coordinate and coordinate to rank conversions"""
        # Setup dimensions
        parallel_state._TENSOR_MODEL_PARALLEL_SIZE = 2
        parallel_state._PIPELINE_MODEL_PARALLEL_SIZE = 2
        parallel_state._DATA_PARALLEL_SIZE = 2
        parallel_state._CONTEXT_PARALLEL_SIZE = 1
        parallel_state._EXPERT_MODEL_PARALLEL_SIZE = 1

        # Test conversions for various ranks
        test_cases = [
            (0, (0, 0, 0, 0, 0)),
            (1, (0, 0, 1, 0, 0)),
            (2, (0, 1, 0, 0, 0)),
            (3, (0, 1, 1, 0, 0)),
            (4, (1, 0, 0, 0, 0)),
            (5, (1, 0, 1, 0, 0)),
            (6, (1, 1, 0, 0, 0)),
            (7, (1, 1, 1, 0, 0)),
        ]

        for rank, expected_coords in test_cases:
            coords = parallel_state._get_coords_from_rank(rank)
            assert coords == expected_coords

            # Test inverse conversion
            calculated_rank = parallel_state._get_rank_from_coords(*coords)
            assert calculated_rank == rank

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.new_group")
    def test_context_parallelism(
        self, mock_new_group, mock_get_rank, mock_get_world_size, mock_is_initialized
    ):
        """Test context parallelism initialization"""
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 8
        mock_get_rank.return_value = 0
        mock_new_group.return_value = MagicMock()

        # Initialize with CP=2
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2, context_parallel_size=2, data_parallel_size=2
        )

        assert parallel_state.get_context_parallel_size() == 2

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.new_group")
    def test_expert_parallelism(
        self, mock_new_group, mock_get_rank, mock_get_world_size, mock_is_initialized
    ):
        """Test expert parallelism initialization for MoE"""
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 8
        mock_get_rank.return_value = 0
        mock_new_group.return_value = MagicMock()

        # Initialize with EP=2
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            data_parallel_size=2,
        )

        assert parallel_state.get_expert_model_parallel_size() == 2

    def test_nccl_config(self):
        """Test NCCL configuration management"""
        config = NCCLConfig(
            enable_sharp=True, cta_size=8, min_nchannels=4, tree_threshold=1000
        )

        parallel_state.set_nccl_config(config)
        retrieved_config = parallel_state.get_nccl_config()

        assert retrieved_config is not None
        assert retrieved_config.enable_sharp == True
        assert retrieved_config.cta_size == 8
        assert retrieved_config.min_nchannels == 4
        assert retrieved_config.tree_threshold == 1000

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.new_group")
    def test_pipeline_parallel_ranks(
        self, mock_new_group, mock_get_rank, mock_get_world_size, mock_is_initialized
    ):
        """Test pipeline parallel rank calculations"""
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 8
        mock_new_group.return_value = MagicMock()

        # Test for different ranks
        # With default order "tp-cp-ep-dp-pp" and TP=1, CP=1, EP=1, DP=2, PP=4:
        # Rank mapping: rank = dp + pp * DP_SIZE
        # So: rank 0: dp=0, pp=0; rank 1: dp=1, pp=0; rank 2: dp=0, pp=1; etc.
        test_cases = [
            (0, 0, 0, 2, None),  # rank 0: pp=0, dp=0 (first stage, no prev)
            (2, 1, 0, 4, 0),  # rank 2: pp=1, dp=0 (second stage, prev=rank 0)
            (4, 2, 0, 6, 2),  # rank 4: pp=2, dp=0 (third stage, prev=rank 2)
            (6, 3, 0, None, 4),  # rank 6: pp=3, dp=0 (last stage, no next, prev=rank 4)
        ]

        for rank, expected_pp_rank, first_rank, next_rank, prev_rank in test_cases:
            mock_get_rank.return_value = rank

            # Reset state
            self.setup_method()

            # Initialize with PP=4, DP=2
            parallel_state.initialize_model_parallel(
                pipeline_model_parallel_size=4, data_parallel_size=2
            )

            assert parallel_state.get_pipeline_model_parallel_rank() == expected_pp_rank
            assert parallel_state.get_pipeline_model_parallel_first_rank() == first_rank
            assert parallel_state.get_pipeline_model_parallel_next_rank() == next_rank
            assert parallel_state.get_pipeline_model_parallel_prev_rank() == prev_rank

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.new_group")
    def test_virtual_pipeline_parallel(
        self, mock_new_group, mock_get_rank, mock_get_world_size, mock_is_initialized
    ):
        """Test virtual pipeline parallelism support"""
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 8
        mock_get_rank.return_value = 0
        mock_new_group.return_value = MagicMock()

        # Initialize with virtual pipeline stages
        parallel_state.initialize_model_parallel(
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=4,
            data_parallel_size=4,
        )

        assert parallel_state.get_virtual_pipeline_model_parallel_size() == 4

        # Test setting virtual rank
        parallel_state.set_virtual_pipeline_model_parallel_rank(2)
        assert parallel_state.get_virtual_pipeline_model_parallel_rank() == 2

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.new_group")
    @patch("torch.distributed.destroy_process_group")
    def test_destroy_parallel_state(
        self,
        mock_destroy_group,
        mock_new_group,
        mock_get_rank,
        mock_get_world_size,
        mock_is_initialized,
    ):
        """Test destroying parallel state"""
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 8
        mock_get_rank.return_value = 0
        mock_group = MagicMock()
        mock_new_group.return_value = mock_group

        # Initialize
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            data_parallel_size=2,
        )

        assert parallel_state.is_initialized()

        # Destroy
        parallel_state.destroy_model_parallel()

        assert not parallel_state.is_initialized()
        assert parallel_state.get_tensor_model_parallel_group() is None
        assert parallel_state.get_pipeline_model_parallel_group() is None
        assert parallel_state.get_data_parallel_group() is None

    def test_parallelism_dimension_enum(self):
        """Test ParallelismDimension enum"""
        assert ParallelismDimension.TENSOR.value == "tp"
        assert ParallelismDimension.PIPELINE.value == "pp"
        assert ParallelismDimension.DATA.value == "dp"
        assert ParallelismDimension.CONTEXT.value == "cp"
        assert ParallelismDimension.EXPERT.value == "ep"

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.new_group")
    def test_custom_parallelism_order(
        self, mock_new_group, mock_get_rank, mock_get_world_size, mock_is_initialized
    ):
        """Test custom parallelism dimension ordering"""
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 16
        mock_get_rank.return_value = 0
        mock_new_group.return_value = MagicMock()

        # Initialize with custom order
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            data_parallel_size=2,
            context_parallel_size=2,
            order="dp-cp-tp-pp-ep",  # Different from default
        )

        assert parallel_state.is_initialized()
        # The ordering affects how ranks are mapped to groups
        # but the sizes should remain the same
        assert parallel_state.get_tensor_model_parallel_size() == 2
        assert parallel_state.get_pipeline_model_parallel_size() == 2
        assert parallel_state.get_data_parallel_size() == 2
        assert parallel_state.get_context_parallel_size() == 2
