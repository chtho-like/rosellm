"""
Comprehensive tests for gradient bucketing functionality.

Tests cover the core bucketing system, bucket groups, and integration scenarios
with various strategies and configurations.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from rosellm.rosetrainer.communication.bucket_groups import (
    BucketGroup,
    BucketGroupConfig,
    BucketGroupManager,
    GroupStrategy,
    PriorityLevel,
)
from rosellm.rosetrainer.communication.gradient_buckets import (
    BucketConfig,
    BucketManager,
    BucketStrategy,
    CommunicationBackend,
    GradientBucket,
)


class TestGradientBucket:
    """Test the GradientBucket class functionality."""

    @pytest.fixture
    def device(self):
        """Get test device (CPU for CI compatibility)."""
        return torch.device("cpu")

    @pytest.fixture
    def bucket(self, device):
        """Create a test gradient bucket."""
        return GradientBucket(
            bucket_id=0,
            max_size_bytes=1024 * 1024,  # 1MB
            device=device,
            dtype=torch.float32,
        )

    @pytest.fixture
    def sample_gradients(self, device):
        """Create sample gradient tensors."""
        return [
            torch.randn(10, 20, device=device),  # 800 bytes
            torch.randn(100, device=device),  # 400 bytes
            torch.randn(5, 5, device=device),  # 100 bytes
        ]

    def test_bucket_initialization(self, device):
        """Test bucket initialization with various parameters."""
        bucket = GradientBucket(
            bucket_id=42,
            max_size_bytes=2048,
            device=device,
            dtype=torch.float16,
        )

        assert bucket.bucket_id == 42
        assert bucket.max_size_bytes == 2048
        assert bucket.device == device
        assert bucket.dtype == torch.float16
        assert len(bucket.gradients) == 0
        assert bucket.current_size_bytes == 0
        assert not bucket.is_ready

    def test_can_add_gradient(self, bucket, sample_gradients):
        """Test gradient capacity checking."""
        grad = sample_gradients[0]
        assert bucket.can_add_gradient(grad)

        # Add gradient and check remaining capacity
        bucket.add_gradient(grad, "test_param", "test_layer")

        # Should still have capacity for smaller gradients
        assert bucket.can_add_gradient(sample_gradients[1])
        assert bucket.can_add_gradient(sample_gradients[2])

        # Test with very large gradient that exceeds capacity
        large_grad = torch.randn(10000, 10000)  # ~400MB
        assert not bucket.can_add_gradient(large_grad)

    def test_add_gradient_success(self, bucket, sample_gradients, device):
        """Test successful gradient addition."""
        grad = sample_gradients[0]
        param_name = "layer.weight"
        layer_type = "linear"

        result = bucket.add_gradient(grad, param_name, layer_type)

        assert result is True
        assert len(bucket.gradients) == 1
        assert len(bucket.gradient_metadata) == 1
        assert bucket.current_size_bytes > 0
        assert not bucket.is_ready  # Not ready until flattened

        metadata = bucket.gradient_metadata[0]
        assert metadata["param_name"] == param_name
        assert metadata["layer_type"] == layer_type
        assert metadata["shape"] == grad.shape
        assert metadata["size_bytes"] == grad.numel() * grad.element_size()

    def test_add_gradient_capacity_exceeded(self, device):
        """Test gradient addition when capacity is exceeded."""
        from rosellm.rosetrainer.communication.gradient_buckets import (
            BucketCapacityError,
        )

        # Create a small bucket for testing capacity limits
        small_bucket = GradientBucket(
            bucket_id=99,
            max_size_bytes=2048,  # Only 2KB capacity
            device=device,
            dtype=torch.float32,
        )

        # Fill bucket with large gradient (512 * 4 bytes = 2KB, exactly at capacity)
        large_grad = torch.randn(512, device=device)
        result1 = small_bucket.add_gradient(large_grad, "large_param", "large_layer")
        assert result1 is True

        # Try to add another gradient (should raise capacity error)
        small_grad = torch.randn(10, device=device)  # Even small gradient won't fit
        with pytest.raises(
            BucketCapacityError, match="Cannot add gradient.*Available space"
        ):
            small_bucket.add_gradient(small_grad, "small_param", "small_layer")

        assert len(small_bucket.gradients) == 1  # Only first gradient added

    def test_flatten_gradients(self, bucket, sample_gradients):
        """Test gradient flattening functionality."""
        # Add multiple gradients
        for i, grad in enumerate(sample_gradients):
            bucket.add_gradient(grad, f"param_{i}", f"layer_{i}")

        flattened = bucket.flatten_gradients()

        assert bucket.is_ready
        assert bucket.flattened_gradient is not None
        assert torch.is_tensor(flattened)

        # Check total size matches sum of individual gradients
        expected_size = sum(grad.numel() for grad in sample_gradients)
        assert flattened.numel() == expected_size

    def test_unflatten_gradients(self, bucket, sample_gradients):
        """Test gradient unflattening functionality."""
        # Add gradients and flatten
        original_shapes = []
        for i, grad in enumerate(sample_gradients):
            bucket.add_gradient(grad, f"param_{i}", f"layer_{i}")
            original_shapes.append(grad.shape)

        bucket.flatten_gradients()

        # Modify flattened gradient to test round-trip
        bucket.flattened_gradient.fill_(42.0)

        unflattened = bucket.unflatten_gradients()

        assert len(unflattened) == len(sample_gradients)
        for i, (unflat_grad, orig_shape) in enumerate(
            zip(unflattened, original_shapes)
        ):
            assert unflat_grad.shape == orig_shape
            assert torch.all(unflat_grad == 42.0)  # Check modification preserved

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.get_world_size")
    def test_start_communication_mock(
        self,
        mock_get_world_size,
        mock_all_reduce,
        mock_is_initialized,
        bucket,
        sample_gradients,
    ):
        """Test communication start with mocked distributed operations."""
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 2
        mock_work = MagicMock()
        mock_all_reduce.return_value = mock_work

        # Add and flatten gradients
        for i, grad in enumerate(sample_gradients):
            bucket.add_gradient(grad, f"param_{i}", f"layer_{i}")

        handle = bucket.start_communication(predivide=True)

        assert handle is mock_work
        assert bucket.communication_handle is mock_work
        mock_all_reduce.assert_called_once()

    def test_get_statistics(self, bucket, sample_gradients):
        """Test statistics collection."""
        # Add gradients
        for i, grad in enumerate(sample_gradients):
            bucket.add_gradient(grad, f"param_{i}", f"layer_{i}")

        # Simulate some communication times
        bucket.communication_times = [0.1, 0.15, 0.12]

        stats = bucket.get_statistics()

        assert stats["bucket_id"] == bucket.bucket_id
        assert stats["num_gradients"] == len(sample_gradients)
        assert stats["current_size_mb"] > 0
        assert 0 <= stats["utilization"] <= 1
        assert stats["avg_communication_time"] == pytest.approx(0.123, rel=1e-2)
        assert stats["total_communications"] == 3

    def test_clear_bucket(self, bucket, sample_gradients):
        """Test bucket clearing functionality."""
        # Add gradients and flatten
        for i, grad in enumerate(sample_gradients):
            bucket.add_gradient(grad, f"param_{i}", f"layer_{i}")
        bucket.flatten_gradients()

        # Clear bucket
        bucket.clear()

        assert len(bucket.gradients) == 0
        assert len(bucket.gradient_metadata) == 0
        assert bucket.flattened_gradient is None
        assert bucket.current_size_bytes == 0
        assert not bucket.is_ready
        assert bucket.communication_handle is None


class TestBucketConfig:
    """Test bucket configuration validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BucketConfig()

        assert config.strategy == BucketStrategy.SIZE_BASED
        assert config.max_bucket_size_mb == 25.0
        assert config.min_bucket_size_mb == 1.0
        assert config.backend == CommunicationBackend.AUTO
        assert config.overlap_communication is True
        assert config.gradient_predivision is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BucketConfig(
            strategy=BucketStrategy.LAYER_BASED,
            max_bucket_size_mb=50.0,
            compress_gradients=True,
            dynamic_bucketing=True,
        )

        assert config.strategy == BucketStrategy.LAYER_BASED
        assert config.max_bucket_size_mb == 50.0
        assert config.compress_gradients is True
        assert config.dynamic_bucketing is True


class TestBucketManager:
    """Test the BucketManager class functionality."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def config(self):
        """Create test bucket configuration."""
        return BucketConfig(
            strategy=BucketStrategy.SIZE_BASED,
            max_bucket_size_mb=10.0,  # Larger buckets for testing
            min_bucket_size_mb=0.1,
        )

    @pytest.fixture
    def manager(self, config, device):
        """Create test bucket manager."""
        return BucketManager(config, device, torch.float32)

    @pytest.fixture
    def sample_gradients(self, device):
        """Create sample gradients with different sizes."""
        return {
            "small_param": torch.randn(10, 10, device=device),  # Small
            "medium_param": torch.randn(100, 50, device=device),  # Medium
            "large_param": torch.randn(200, 200, device=device),  # Large
            "tiny_param": torch.randn(5, device=device),  # Tiny
        }

    def test_manager_initialization(self, config, device):
        """Test bucket manager initialization."""
        manager = BucketManager(config, device, torch.float16)

        assert manager.config == config
        assert manager.device == device
        assert manager.dtype == torch.float16
        assert len(manager.buckets) == 0
        assert len(manager.bucket_assignments) == 0
        assert manager.next_bucket_id == 0

    def test_size_based_strategy_initialization(self, device):
        """Test size-based strategy initialization."""
        config = BucketConfig(strategy=BucketStrategy.SIZE_BASED)
        manager = BucketManager(config, device)

        assert len(manager._size_buckets) > 0
        # Size buckets should be in ascending order
        for i in range(len(manager._size_buckets) - 1):
            assert manager._size_buckets[i][1] <= manager._size_buckets[i + 1][0]

    def test_layer_based_strategy_initialization(self, device):
        """Test layer-based strategy initialization."""
        config = BucketConfig(strategy=BucketStrategy.LAYER_BASED)
        manager = BucketManager(config, device)

        # Should have layer groups defined
        assert len(manager.config.layer_groups) > 0
        assert "embedding" in manager.config.layer_groups
        assert "attention" in manager.config.layer_groups

    def test_parameter_classification(self, manager):
        """Test parameter classification for bucketing."""
        test_cases = [
            ("model.embedding.weight", "embedding"),
            ("transformer.attn.query.weight", "attention"),
            ("mlp.fc1.weight", "feedforward"),
            ("layer_norm.weight", "normalization"),
            ("output_head.weight", "output"),
            ("random.param.weight", "other"),
        ]

        dummy_grad = torch.randn(10, 10)
        for param_name, expected_type in test_cases:
            classified_type = manager._classify_parameter(param_name, dummy_grad)
            assert classified_type == expected_type

    def test_assign_gradient_new_bucket(self, manager, sample_gradients):
        """Test gradient assignment creating new buckets."""
        param_name = "test_param"
        gradient = sample_gradients["small_param"]

        bucket_id = manager.assign_gradient(param_name, gradient)

        assert bucket_id == 0  # First bucket should have ID 0
        assert param_name in manager.bucket_assignments
        assert manager.bucket_assignments[param_name] == bucket_id
        assert len(manager.buckets) == 1
        assert len(manager.buckets[0].gradients) == 1

    def test_assign_gradient_existing_bucket(self, manager, sample_gradients):
        """Test gradient assignment to existing bucket."""
        # First assignment
        param1 = "param1"
        grad1 = sample_gradients["small_param"]
        bucket_id1 = manager.assign_gradient(param1, grad1)

        # Second assignment that should fit in same bucket
        param2 = "param2"
        grad2 = sample_gradients["tiny_param"]
        bucket_id2 = manager.assign_gradient(param2, grad2)

        # Should use existing bucket if there's capacity
        if bucket_id1 == bucket_id2:
            assert len(manager.buckets[bucket_id1].gradients) == 2
        else:
            # Or create new bucket if no capacity
            assert len(manager.buckets) == 2

    def test_assign_gradient_overflow_new_bucket(self, device):
        """Test gradient assignment when existing buckets are full."""
        # Create a manager with small bucket size to force overflow
        config = BucketConfig(
            strategy=BucketStrategy.SIZE_BASED,
            max_bucket_size_mb=0.1,  # Very small buckets (100KB)
            min_bucket_size_mb=0.01,
        )
        manager = BucketManager(config, device, torch.float32)

        # Fill first bucket with medium gradient
        param1 = "medium_param1"
        grad1 = torch.randn(50, 50, device=device)  # ~10KB gradient
        bucket_id1 = manager.assign_gradient(param1, grad1)

        # Try to add multiple more gradients to force new buckets
        bucket_ids = [bucket_id1]
        for i in range(10):  # Add multiple gradients
            param_name = f"param_{i}"
            grad = torch.randn(50, 50, device=device)  # Same size gradients
            bucket_id = manager.assign_gradient(param_name, grad)
            bucket_ids.append(bucket_id)

        # Should have created multiple buckets due to size constraints
        unique_bucket_ids = set(bucket_ids)
        assert (
            len(unique_bucket_ids) >= 2
        ), f"Expected multiple buckets, got {len(unique_bucket_ids)}"
        assert len(manager.buckets) >= 2

    @patch("torch.distributed.is_initialized")
    def test_synchronize_buckets_no_buckets(self, mock_is_initialized, manager):
        """Test synchronization with no buckets."""
        mock_is_initialized.return_value = True

        stats = manager.synchronize_buckets()

        assert stats["num_buckets"] == 0
        assert stats["total_time"] == 0.0

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.get_world_size")
    def test_synchronize_buckets_with_gradients(
        self,
        mock_get_world_size,
        mock_all_reduce,
        mock_is_initialized,
        manager,
        sample_gradients,
    ):
        """Test bucket synchronization with gradients."""
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 2
        mock_work = MagicMock()
        mock_work.wait = MagicMock()
        mock_all_reduce.return_value = mock_work

        # Add gradients to buckets
        for param_name, gradient in sample_gradients.items():
            manager.assign_gradient(param_name, gradient)

        stats = manager.synchronize_buckets()

        assert stats["num_buckets"] > 0
        assert "total_time" in stats
        assert "communication_time" in stats
        assert stats["total_time"] >= 0

    def test_get_bucket_assignments(self, manager, sample_gradients):
        """Test getting bucket assignments after synchronization."""
        # Add gradients
        for param_name, gradient in sample_gradients.items():
            manager.assign_gradient(param_name, gradient)

        # Simulate synchronization by flattening buckets
        for bucket in manager.buckets:
            if bucket.gradients:
                bucket.flatten_gradients()
                # Modify gradient to test assignment
                bucket.flattened_gradient.fill_(99.0)

        assignments = manager.get_bucket_assignments()

        assert len(assignments) == len(sample_gradients)
        for param_name in sample_gradients.keys():
            assert param_name in assignments
            # Check that gradient was modified
            assert torch.all(assignments[param_name] == 99.0)

    def test_get_statistics(self, manager, sample_gradients):
        """Test statistics collection from manager."""
        # Add gradients
        for param_name, gradient in sample_gradients.items():
            manager.assign_gradient(param_name, gradient)

        # Simulate some communications
        manager.total_communications = 5
        manager.total_communication_time = 1.0

        stats = manager.get_statistics()

        assert stats["strategy"] == BucketStrategy.SIZE_BASED.value
        assert stats["num_buckets"] == len(manager.buckets)
        assert stats["total_gradients"] == len(sample_gradients)
        assert stats["total_size_mb"] > 0
        assert stats["total_communications"] == 5
        assert stats["avg_communication_time"] == 0.2
        assert "bucket_details" in stats

    def test_reset_manager(self, manager, sample_gradients):
        """Test manager reset functionality."""
        # Add gradients
        for param_name, gradient in sample_gradients.items():
            manager.assign_gradient(param_name, gradient)

        # Reset manager
        manager.reset()

        # Buckets should be cleared but preserved
        assert len(manager.buckets) > 0  # Buckets still exist
        for bucket in manager.buckets:
            assert len(bucket.gradients) == 0  # But are empty
        assert len(manager.bucket_assignments) == 0  # Assignments cleared


class TestBucketGroup:
    """Test the BucketGroup class functionality."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def group(self):
        """Create test bucket group."""
        return BucketGroup(
            group_id=1,
            priority=PriorityLevel.HIGH,
            max_buckets=5,
        )

    @pytest.fixture
    def sample_bucket(self, device):
        """Create sample bucket."""
        return GradientBucket(
            bucket_id=0,
            max_size_bytes=1024,
            device=device,
        )

    def test_group_initialization(self):
        """Test bucket group initialization."""
        group = BucketGroup(
            group_id=42,
            priority=PriorityLevel.CRITICAL,
            max_buckets=10,
        )

        assert group.group_id == 42
        assert group.priority == PriorityLevel.CRITICAL
        assert group.max_buckets == 10
        assert len(group.buckets) == 0
        assert not group.is_communicating

    def test_can_add_bucket(self, group, sample_bucket):
        """Test bucket capacity checking."""
        assert group.can_add_bucket(sample_bucket)

        # Add buckets up to capacity
        for i in range(group.max_buckets):
            bucket = GradientBucket(
                bucket_id=i, max_size_bytes=1024, device=torch.device("cpu")
            )
            group.add_bucket(bucket)

        # Should not be able to add more
        extra_bucket = GradientBucket(
            bucket_id=99, max_size_bytes=1024, device=torch.device("cpu")
        )
        assert not group.can_add_bucket(extra_bucket)

    def test_add_bucket_success(self, group, sample_bucket):
        """Test successful bucket addition."""
        result = group.add_bucket(sample_bucket)

        assert result is True
        assert len(group.buckets) == 1
        assert sample_bucket.bucket_id in group.bucket_ids
        assert group.buckets[0] is sample_bucket

    def test_add_bucket_duplicate(self, group, sample_bucket):
        """Test adding duplicate bucket."""
        group.add_bucket(sample_bucket)

        # Try to add same bucket again
        result = group.add_bucket(sample_bucket)

        assert result is False
        assert len(group.buckets) == 1

    def test_remove_bucket(self, group, sample_bucket):
        """Test bucket removal."""
        group.add_bucket(sample_bucket)

        result = group.remove_bucket(sample_bucket.bucket_id)

        assert result is True
        assert len(group.buckets) == 0
        assert sample_bucket.bucket_id not in group.bucket_ids

    def test_remove_nonexistent_bucket(self, group):
        """Test removing non-existent bucket."""
        result = group.remove_bucket(999)
        assert result is False

    def test_get_total_size(self, group, device):
        """Test total size calculation."""
        # Create buckets with known sizes - larger to accommodate test gradients
        bucket1 = GradientBucket(bucket_id=1, max_size_bytes=2048, device=device)
        bucket2 = GradientBucket(bucket_id=2, max_size_bytes=2048, device=device)

        # Add gradients to buckets
        grad1 = torch.randn(10, 10)
        grad2 = torch.randn(20, 20)
        bucket1.add_gradient(grad1, "param1", "layer1")
        bucket2.add_gradient(grad2, "param2", "layer2")

        # Add buckets to group
        group.add_bucket(bucket1)
        group.add_bucket(bucket2)

        total_size = group.get_total_size()
        expected_size = bucket1.current_size_bytes + bucket2.current_size_bytes

        assert total_size == expected_size
        assert total_size > 0

    def test_get_statistics(self, group, device):
        """Test statistics collection from group."""
        # Add bucket with gradient
        bucket = GradientBucket(
            bucket_id=1, max_size_bytes=1024 * 1024, device=device
        )  # 1MB bucket
        grad = torch.randn(50, 20, device=device)
        success = bucket.add_gradient(grad, "param", "layer")
        assert success, "Failed to add gradient to bucket"

        success = group.add_bucket(bucket)
        assert success, "Failed to add bucket to group"

        # Verify bucket has the gradient
        assert (
            len(bucket.gradients) == 1
        ), f"Bucket should have 1 gradient, has {len(bucket.gradients)}"

        # Simulate some communications
        group.successful_communications = 3
        group.communication_times = [0.1, 0.12, 0.11]

        stats = group.get_statistics()

        assert stats["group_id"] == group.group_id
        assert stats["priority"] == group.priority.name
        assert stats["num_buckets"] == 1
        assert (
            stats["total_gradients"] == 1
        ), f"Expected 1 gradient, got {stats['total_gradients']}"
        assert stats["total_size_mb"] > 0
        assert 0 <= stats["utilization"] <= 1
        assert stats["successful_communications"] == 3
        assert stats["avg_communication_time"] == pytest.approx(0.11, rel=1e-2)


class TestBucketGroupManager:
    """Test the BucketGroupManager class functionality."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def bucket_manager(self, device):
        """Create test bucket manager."""
        config = BucketConfig()
        return BucketManager(config, device)

    @pytest.fixture
    def group_config(self):
        """Create test group configuration."""
        return BucketGroupConfig(
            group_strategy=GroupStrategy.ADAPTIVE,
            max_groups=4,
        )

    @pytest.fixture
    def group_manager(self, group_config, bucket_manager):
        """Create test group manager."""
        return BucketGroupManager(group_config, bucket_manager)

    def test_manager_initialization(self, group_config, bucket_manager):
        """Test group manager initialization."""
        manager = BucketGroupManager(group_config, bucket_manager)

        assert manager.config == group_config
        assert manager.bucket_manager == bucket_manager
        assert len(manager.groups) == 0
        assert manager.next_group_id == 0

    def test_create_group(self, group_manager):
        """Test group creation."""
        group = group_manager.create_group(PriorityLevel.HIGH)

        assert group.group_id == 0
        assert group.priority == PriorityLevel.HIGH
        assert len(group_manager.groups) == 1
        assert group in group_manager.group_by_priority[PriorityLevel.HIGH]

    def test_assign_buckets_parallel_strategy(self, group_manager, device):
        """Test parallel strategy bucket assignment."""
        # Create some test buckets
        buckets = []
        for i in range(6):
            bucket = GradientBucket(bucket_id=i, max_size_bytes=1024, device=device)
            grad = torch.randn(10, 10)
            bucket.add_gradient(grad, f"param_{i}", "layer")
            buckets.append(bucket)
            group_manager.bucket_manager.buckets.append(bucket)

        # Create groups first
        for _ in range(3):
            group_manager.create_group()

        # Test parallel assignment
        group_manager.config.group_strategy = GroupStrategy.PARALLEL
        stats = group_manager.assign_buckets_to_groups()

        assert stats["strategy"] == "PARALLEL"
        assert stats["total_buckets"] == 6
        assert stats["num_groups_used"] > 0

    def test_assign_buckets_hierarchical_strategy(self, group_manager, device):
        """Test hierarchical strategy bucket assignment."""
        # Create buckets with different priorities
        bucket_sizes = [100, 200, 50, 300, 150]  # Different sizes for priority testing
        buckets = []

        for i, size in enumerate(bucket_sizes):
            bucket = GradientBucket(bucket_id=i, max_size_bytes=2048, device=device)
            grad = torch.randn(size, device=device)
            bucket.add_gradient(grad, f"param_{i}", "layer")
            buckets.append(bucket)
            group_manager.bucket_manager.buckets.append(bucket)

        # Test hierarchical assignment
        group_manager.config.group_strategy = GroupStrategy.HIERARCHICAL
        stats = group_manager.assign_buckets_to_groups()

        assert stats["strategy"] == "HIERARCHICAL"
        assert stats["total_buckets"] == 5

        # Should have groups with different priorities
        priority_groups = {
            p: len(groups)
            for p, groups in group_manager.group_by_priority.items()
            if groups
        }
        assert len(priority_groups) > 0

    @patch("torch.distributed.is_initialized")
    def test_synchronize_groups_no_active(self, mock_is_initialized, group_manager):
        """Test synchronization with no active groups."""
        mock_is_initialized.return_value = True

        stats = group_manager.synchronize_groups()

        assert "message" in stats
        assert stats["message"] == "No active groups to synchronize"

    def test_get_statistics(self, group_manager, device):
        """Test statistics collection from group manager."""
        # Create some groups and buckets
        group1 = group_manager.create_group(PriorityLevel.HIGH)
        group_manager.create_group(PriorityLevel.NORMAL)

        bucket = GradientBucket(bucket_id=1, max_size_bytes=2048, device=device)
        grad = torch.randn(20, 20)
        bucket.add_gradient(grad, "param", "layer")
        group1.add_bucket(bucket)

        # Simulate some communications
        group_manager.total_group_communications = 2
        group_manager.total_group_time = 0.5

        stats = group_manager.get_statistics()

        assert stats["config"]["strategy"] == GroupStrategy.ADAPTIVE.name
        assert stats["groups"]["total"] == 2
        assert stats["groups"]["active"] == 1  # Only group1 has buckets
        assert stats["performance"]["total_communications"] == 2
        assert stats["performance"]["avg_time_per_communication"] == 0.25


class TestIntegrationScenarios:
    """Test integration scenarios combining bucketing components."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def create_model_gradients(self, device):
        """Create gradients simulating a model."""

        def _create():
            gradients = {}
            # Embedding layer
            gradients["embedding.weight"] = torch.randn(1000, 512, device=device)

            # Transformer layers
            for layer in range(2):
                gradients[f"transformer.{layer}.attn.query.weight"] = torch.randn(
                    512, 512, device=device
                )
                gradients[f"transformer.{layer}.attn.key.weight"] = torch.randn(
                    512, 512, device=device
                )
                gradients[f"transformer.{layer}.attn.value.weight"] = torch.randn(
                    512, 512, device=device
                )
                gradients[f"transformer.{layer}.attn.output.weight"] = torch.randn(
                    512, 512, device=device
                )
                gradients[f"transformer.{layer}.mlp.fc1.weight"] = torch.randn(
                    512, 2048, device=device
                )
                gradients[f"transformer.{layer}.mlp.fc2.weight"] = torch.randn(
                    2048, 512, device=device
                )
                gradients[f"transformer.{layer}.norm1.weight"] = torch.randn(
                    512, device=device
                )
                gradients[f"transformer.{layer}.norm2.weight"] = torch.randn(
                    512, device=device
                )

            # Output head
            gradients["output.weight"] = torch.randn(1000, 512, device=device)

            return gradients

        return _create

    def test_end_to_end_size_based_bucketing(self, device, create_model_gradients):
        """Test complete size-based bucketing workflow."""
        config = BucketConfig(
            strategy=BucketStrategy.SIZE_BASED,
            max_bucket_size_mb=10.0,  # 10MB buckets
        )
        manager = BucketManager(config, device)
        gradients = create_model_gradients()

        # Assign all gradients
        bucket_assignments = {}
        for param_name, gradient in gradients.items():
            bucket_id = manager.assign_gradient(param_name, gradient)
            bucket_assignments[param_name] = bucket_id

        # Verify assignments
        assert len(manager.buckets) > 0
        assert len(bucket_assignments) == len(gradients)

        # Verify bucketing strategy worked
        total_gradient_size = sum(
            grad.numel() * grad.element_size() for grad in gradients.values()
        )
        total_bucket_size = sum(bucket.current_size_bytes for bucket in manager.buckets)
        assert total_bucket_size == total_gradient_size

        # Test statistics
        stats = manager.get_statistics()
        assert stats["num_buckets"] > 1  # Should create multiple buckets
        assert stats["total_gradients"] == len(gradients)

    def test_end_to_end_layer_based_bucketing(self, device, create_model_gradients):
        """Test complete layer-based bucketing workflow."""
        config = BucketConfig(
            strategy=BucketStrategy.LAYER_BASED,
            max_bucket_size_mb=10.0,  # Large buckets to accommodate layer grouping
        )
        manager = BucketManager(config, device)
        gradients = create_model_gradients()

        # Assign all gradients
        for param_name, gradient in gradients.items():
            manager.assign_gradient(param_name, gradient)

        # Verify layer-based grouping
        stats = manager.get_statistics()
        assert stats["num_buckets"] > 0

        # Check that similar layers are grouped together by examining bucket contents
        layer_types_in_buckets = {}
        for bucket in manager.buckets:
            layer_types = set()
            for metadata in bucket.gradient_metadata:
                # Create a dummy gradient for classification
                dummy_grad = torch.randn(1)
                layer_type = manager._classify_parameter(
                    metadata["param_name"], dummy_grad
                )
                layer_types.add(layer_type)
            layer_types_in_buckets[bucket.bucket_id] = layer_types

        # At least some buckets should have consistent layer types
        consistent_buckets = sum(
            1 for types in layer_types_in_buckets.values() if len(types) == 1
        )
        assert consistent_buckets > 0

    def test_end_to_end_with_groups(self, device, create_model_gradients):
        """Test complete workflow with bucket groups."""
        # Create bucket manager
        bucket_config = BucketConfig(
            strategy=BucketStrategy.MIXED,
            max_bucket_size_mb=10.0,  # Larger buckets for testing
        )
        bucket_manager = BucketManager(bucket_config, device)

        # Create group manager
        group_config = BucketGroupConfig(
            group_strategy=GroupStrategy.HIERARCHICAL,
            max_groups=3,
        )
        group_manager = BucketGroupManager(group_config, bucket_manager)

        gradients = create_model_gradients()

        # Assign gradients to buckets
        for param_name, gradient in gradients.items():
            bucket_manager.assign_gradient(param_name, gradient)

        # Assign buckets to groups
        assignment_stats = group_manager.assign_buckets_to_groups()
        assert assignment_stats["num_groups_used"] > 0
        assert assignment_stats["total_buckets"] > 0

        # Verify group assignments
        total_buckets_in_groups = sum(
            len(group.buckets) for group in group_manager.groups
        )
        assert total_buckets_in_groups == len(bucket_manager.buckets)

        # Test statistics
        group_stats = group_manager.get_statistics()
        assert group_stats["groups"]["active"] > 0

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.get_world_size")
    def test_communication_simulation(
        self,
        mock_get_world_size,
        mock_all_reduce,
        mock_is_initialized,
        device,
        create_model_gradients,
    ):
        """Test communication simulation with mocked distributed ops."""
        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 2
        mock_work = MagicMock()
        mock_work.wait = MagicMock()
        mock_all_reduce.return_value = mock_work

        # Setup complete bucketing system
        bucket_config = BucketConfig(max_bucket_size_mb=10.0)
        bucket_manager = BucketManager(bucket_config, device)

        group_config = BucketGroupConfig(overlap_groups=True)
        group_manager = BucketGroupManager(group_config, bucket_manager)

        gradients = create_model_gradients()

        # Assign gradients and synchronize
        for param_name, gradient in gradients.items():
            bucket_manager.assign_gradient(param_name, gradient)

        group_manager.assign_buckets_to_groups()
        sync_stats = group_manager.synchronize_groups()

        # Verify communication was attempted
        assert mock_all_reduce.call_count > 0
        assert "total_time" in sync_stats
        assert sync_stats["groups_synchronized"] > 0

    def test_memory_efficiency(self, device, create_model_gradients):
        """Test that bucketing doesn't significantly increase memory usage."""
        gradients = create_model_gradients()

        # Calculate original memory usage
        original_memory = sum(
            grad.numel() * grad.element_size() for grad in gradients.values()
        )

        # Create bucketing system
        config = BucketConfig(max_bucket_size_mb=5.0)
        manager = BucketManager(config, device)

        # Assign all gradients
        for param_name, gradient in gradients.items():
            manager.assign_gradient(param_name, gradient)

        # Calculate bucketed memory usage
        bucketed_memory = sum(bucket.current_size_bytes for bucket in manager.buckets)

        # Memory usage should be identical (no duplication)
        assert bucketed_memory == original_memory

        # Flatten all buckets and check flattened memory
        flattened_memory = 0
        for bucket in manager.buckets:
            if bucket.gradients:
                bucket.flatten_gradients()
                if bucket.flattened_gradient is not None:
                    flattened_memory += (
                        bucket.flattened_gradient.numel()
                        * bucket.flattened_gradient.element_size()
                    )

        # Flattened memory should equal original (no extra overhead)
        assert flattened_memory == original_memory

    def test_performance_benchmarking(self, device, create_model_gradients):
        """Test performance characteristics of bucketing system."""
        import time

        gradients = create_model_gradients()
        config = BucketConfig(strategy=BucketStrategy.SIZE_BASED)
        manager = BucketManager(config, device)

        # Time gradient assignment
        start_time = time.time()
        for param_name, gradient in gradients.items():
            manager.assign_gradient(param_name, gradient)
        assignment_time = time.time() - start_time

        # Time bucket flattening
        start_time = time.time()
        for bucket in manager.buckets:
            if bucket.gradients:
                bucket.flatten_gradients()
        flattening_time = time.time() - start_time

        # Time statistics collection
        start_time = time.time()
        stats = manager.get_statistics()
        stats_time = time.time() - start_time

        # Verify reasonable performance (these are very loose bounds)
        assert assignment_time < 1.0  # Should be fast
        assert flattening_time < 1.0  # Should be fast
        assert stats_time < 0.1  # Should be very fast

        # Log performance for manual inspection
        print(f"Assignment time: {assignment_time:.4f}s")
        print(f"Flattening time: {flattening_time:.4f}s")
        print(f"Stats time: {stats_time:.4f}s")
        print(f"Total buckets: {stats['num_buckets']}")
        print(f"Total size: {stats['total_size_mb']:.2f}MB")


class TestTensorMemoryPool:
    """Test the TensorMemoryPool functionality."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def pool(self, device):
        """Create test memory pool."""
        from rosellm.rosetrainer.communication.gradient_buckets import TensorMemoryPool

        return TensorMemoryPool(device, torch.float32)

    def test_pool_get_tensor_new(self, pool):
        """Test getting a tensor from empty pool."""
        tensor = pool.get_tensor(100)

        assert tensor.numel() == 100
        assert tensor.dtype == torch.float32
        assert tensor.device == pool.device

    def test_pool_return_and_reuse(self, pool):
        """Test returning tensor to pool and reusing."""
        # Get initial tensor
        tensor1 = pool.get_tensor(50)
        tensor1.fill_(42.0)

        # Return to pool
        pool.return_tensor(tensor1)

        # Get another tensor of same size
        tensor2 = pool.get_tensor(50)

        # Should be the same tensor, now zeroed
        assert tensor2 is tensor1
        assert torch.all(tensor2 == 0.0)

    def test_pool_size_limits(self, pool):
        """Test pool size limiting."""
        tensors = []
        # Add more tensors than pool limit (10)
        for i in range(15):
            tensor = pool.get_tensor(10)
            tensor.fill_(i)
            pool.return_tensor(tensor)
            tensors.append(tensor)

        # Pool should not exceed limit
        assert len(pool._pool[10]) <= 10

    def test_pool_device_dtype_mismatch(self, pool):
        """Test pool rejects mismatched tensors."""
        # Create tensor with different dtype
        wrong_tensor = torch.randn(10, dtype=torch.float64, device=pool.device)

        # Should not add to pool
        pool.return_tensor(wrong_tensor)
        assert 10 not in pool._pool or len(pool._pool[10]) == 0


class TestGradientBucketMemoryOptimizations:
    """Test memory optimizations in GradientBucket."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    @pytest.fixture
    def bucket(self, device):
        """Create test bucket."""
        return GradientBucket(
            bucket_id=0,
            max_size_bytes=1024 * 1024,
            device=device,
        )

    def test_memory_pool_integration(self, bucket):
        """Test bucket uses memory pool correctly."""
        grad1 = torch.randn(50, 20)
        bucket.add_gradient(grad1, "param1", "layer1")

        # Flatten gradients (should get tensor from pool)
        flattened1 = bucket.flatten_gradients()
        assert flattened1.numel() == 1000

        # Clear bucket (should return tensor to pool)
        bucket.clear()

        # Add different gradient and flatten again
        grad2 = torch.randn(50, 20)  # Same size
        bucket.add_gradient(grad2, "param2", "layer2")
        flattened2 = bucket.flatten_gradients()

        # Should reuse the same tensor
        assert flattened2 is flattened1
        assert torch.all(flattened2 != 0.0)  # Should contain new data

    def test_memory_pool_size_mismatch(self, bucket):
        """Test pool behavior with different tensor sizes."""
        # Add small gradient
        small_grad = torch.randn(10)
        bucket.add_gradient(small_grad, "small", "layer")
        bucket.flatten_gradients()
        bucket.clear()

        # Add larger gradient
        large_grad = torch.randn(100)
        bucket.add_gradient(large_grad, "large", "layer")
        flattened = bucket.flatten_gradients()

        # Should create new tensor since size doesn't match
        assert flattened.numel() == 100


class TestGradientBucketErrorHandling:
    """Test error handling in GradientBucket."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def bucket(self, device):
        return GradientBucket(
            bucket_id=0,
            max_size_bytes=1024,
            device=device,
        )

    def test_add_gradient_validation_errors(self, bucket):
        """Test gradient validation raises proper errors."""
        from rosellm.rosetrainer.communication.gradient_buckets import (
            GradientValidationError,
        )

        # Test None gradient
        with pytest.raises(GradientValidationError, match="is None"):
            bucket.add_gradient(None, "param", "layer")

        # Test empty gradient
        empty_grad = torch.empty(0)
        with pytest.raises(GradientValidationError, match="is empty"):
            bucket.add_gradient(empty_grad, "param", "layer")

        # Test empty parameter name
        grad = torch.randn(10)
        with pytest.raises(GradientValidationError, match="cannot be empty"):
            bucket.add_gradient(grad, "", "layer")

        # Test non-floating point gradient
        int_grad = torch.randint(0, 10, (5, 5), dtype=torch.int32)
        with pytest.raises(
            GradientValidationError, match="must have floating-point dtype"
        ):
            bucket.add_gradient(int_grad, "param", "layer")

    def test_add_gradient_nan_inf_detection(self, bucket):
        """Test NaN and infinite value detection."""
        from rosellm.rosetrainer.communication.gradient_buckets import (
            GradientValidationError,
        )

        # Test NaN values
        nan_grad = torch.randn(10)
        nan_grad[0] = float("nan")
        with pytest.raises(GradientValidationError, match="contains NaN values"):
            bucket.add_gradient(nan_grad, "nan_param", "layer")

        # Test infinite values
        inf_grad = torch.randn(10)
        inf_grad[0] = float("inf")
        with pytest.raises(GradientValidationError, match="contains infinite values"):
            bucket.add_gradient(inf_grad, "inf_param", "layer")

    def test_bucket_capacity_error(self, device):
        """Test bucket capacity error handling."""
        from rosellm.rosetrainer.communication.gradient_buckets import (
            BucketCapacityError,
        )

        # Create small bucket
        small_bucket = GradientBucket(
            bucket_id=0,
            max_size_bytes=100,  # Very small
            device=device,
        )

        # Try to add large gradient
        large_grad = torch.randn(100, 100)  # Way too large
        with pytest.raises(
            BucketCapacityError, match="Cannot add gradient.*Available space"
        ):
            small_bucket.add_gradient(large_grad, "large_param", "layer")

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    def test_communication_error_handling(
        self, mock_get_world_size, mock_is_initialized, bucket
    ):
        """Test communication error handling."""
        from rosellm.rosetrainer.communication.gradient_buckets import (
            CommunicationError,
        )

        mock_is_initialized.return_value = True
        mock_get_world_size.return_value = 2  # Multi-process environment

        # Add gradient and try to start communication with invalid state
        grad = torch.randn(10)
        bucket.add_gradient(grad, "param", "layer")
        bucket.flatten_gradients()

        # Manually create invalid state (empty tensor)
        bucket.flattened_gradient = torch.empty(
            0, device=bucket.device, dtype=bucket.dtype
        )

        with pytest.raises(CommunicationError, match="Cannot communicate empty tensor"):
            bucket.start_communication()


class TestBucketManagerBulkOperations:
    """Test bulk operations in BucketManager."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def manager(self, device):
        config = BucketConfig(max_bucket_size_mb=1.0)  # Small buckets for testing
        return BucketManager(config, device)

    def test_bulk_gradient_assignment(self, manager):
        """Test bulk gradient assignment method."""
        gradients = {f"param_{i}": torch.randn(20, 20) for i in range(10)}

        assignments = manager.assign_gradients_bulk(gradients, batch_size=3)

        assert len(assignments) == len(gradients)
        for param_name in gradients:
            assert param_name in assignments
            assert assignments[param_name] >= 0

    def test_bulk_assignment_error_recovery(self, manager):
        """Test bulk assignment continues after errors."""
        gradients = {
            "good_param": torch.randn(10),
            "nan_param": torch.full((10,), float("nan")),
            "another_good_param": torch.randn(10),
        }

        # Should handle the error gracefully and continue
        assignments = manager.assign_gradients_bulk(gradients)

        # Should have assignments for good parameters
        assert "good_param" in assignments
        assert "another_good_param" in assignments
        # Might not have assignment for nan_param due to validation error
        assert len(assignments) >= 2


class TestBucketFactoryPatterns:
    """Test BucketFactory pattern implementation."""

    def test_factory_creates_bucket(self):
        """Test factory creates buckets correctly."""
        from rosellm.rosetrainer.communication.gradient_buckets import BucketFactory

        device = torch.device("cpu")
        bucket = BucketFactory.create_bucket(
            bucket_id=42, max_size_bytes=1024, device=device, optimization_hint="memory"
        )

        assert bucket.bucket_id == 42
        assert bucket.max_size_bytes == 1024
        assert bucket.device == device
        assert bucket._max_metrics_history == 100  # Memory optimization applied

    def test_factory_speed_optimization(self):
        """Test speed optimization hint."""
        from rosellm.rosetrainer.communication.gradient_buckets import BucketFactory

        device = torch.device("cpu")
        bucket = BucketFactory.create_bucket(
            bucket_id=1, max_size_bytes=1024, device=device, optimization_hint="speed"
        )

        assert bucket._max_metrics_history == 10000  # Speed optimization applied


class TestExceptionHierarchy:
    """Test custom exception hierarchy."""

    def test_exception_inheritance(self):
        """Test exception inheritance structure."""
        from rosellm.rosetrainer.communication.gradient_buckets import (
            BucketCapacityError,
            BucketingError,
            BucketStateError,
            CommunicationError,
            GradientValidationError,
        )

        # All should inherit from BucketingError
        assert issubclass(BucketCapacityError, BucketingError)
        assert issubclass(BucketStateError, BucketingError)
        assert issubclass(CommunicationError, BucketingError)
        assert issubclass(GradientValidationError, BucketingError)

        # All should inherit from Exception
        assert issubclass(BucketingError, Exception)


if __name__ == "__main__":
    # Run with pytest for full test suite
    pytest.main([__file__, "-v", "--tb=short"])
