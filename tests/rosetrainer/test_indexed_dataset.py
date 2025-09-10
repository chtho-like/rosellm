"""Tests for memory-mapped indexed datasets."""

import os
import tempfile
import unittest

import numpy as np
import torch

from rosellm.rosetrainer.datasets import (
    DistributedIndexedDataset,
    DType,
    IndexedDataset,
    IndexedDatasetBuilder,
    MMapIndexedDataset,
)


class TestDType(unittest.TestCase):
    """Test data type utilities."""

    def test_dtype_conversion(self):
        """Test conversion between codes and dtypes."""
        # Test all supported types
        for dtype_enum in DType:
            code = dtype_enum.value
            dtype = DType.dtype_from_code(code)
            recovered_code = DType.code_from_dtype(dtype)

            self.assertEqual(code, recovered_code)
            self.assertTrue(hasattr(np, dtype_enum.name))

    def test_dtype_size(self):
        """Test size calculation."""
        # Test with code
        self.assertEqual(DType.size(DType.uint8.value), 1)
        self.assertEqual(DType.size(DType.int32.value), 4)
        self.assertEqual(DType.size(DType.int64.value), 8)

        # Test with dtype
        self.assertEqual(DType.size(np.uint8), 1)
        self.assertEqual(DType.size(np.int32), 4)
        self.assertEqual(DType.size(np.int64), 8)

    def test_optimal_dtype(self):
        """Test optimal dtype selection."""
        # Small vocabulary
        self.assertEqual(DType.optimal_dtype(100), np.uint8)
        self.assertEqual(DType.optimal_dtype(255), np.uint8)

        # Medium vocabulary
        self.assertEqual(DType.optimal_dtype(256), np.uint16)
        self.assertEqual(DType.optimal_dtype(65535), np.uint16)

        # Large vocabulary
        self.assertEqual(DType.optimal_dtype(65536), np.int32)
        self.assertEqual(DType.optimal_dtype(100000), np.int32)

        # None defaults to int32
        self.assertEqual(DType.optimal_dtype(None), np.int32)


class TestIndexedDatasetBuilder(unittest.TestCase):
    """Test dataset building functionality."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_prefix = os.path.join(self.temp_dir, "test_dataset")

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_basic_building(self):
        """Test basic dataset building."""
        # Create builder
        builder = IndexedDatasetBuilder(self.test_prefix, vocab_size=1000)

        # Add sequences
        seq1 = [1, 2, 3, 4, 5]
        seq2 = [6, 7, 8]
        seq3 = [9, 10, 11, 12]

        builder.add_sequence(seq1)
        builder.add_sequence(seq2)
        builder.end_document()
        builder.add_sequence(seq3)
        builder.end_document()

        # Finalize
        stats = builder.finalize()

        # Check statistics
        self.assertEqual(stats["num_sequences"], 3)
        self.assertEqual(stats["num_documents"], 2)
        self.assertEqual(stats["total_tokens"], 12)

        # Verify files created
        self.assertTrue(os.path.exists(f"{self.test_prefix}.idx"))
        self.assertTrue(os.path.exists(f"{self.test_prefix}.bin"))

    def test_document_building(self):
        """Test document-level building."""
        builder = IndexedDatasetBuilder(self.test_prefix, vocab_size=1000)

        # Add documents
        doc1 = [[1, 2, 3], [4, 5, 6]]
        doc2 = [[7, 8], [9, 10, 11], [12]]

        builder.add_document(doc1)
        builder.add_document(doc2)

        stats = builder.finalize()

        self.assertEqual(stats["num_sequences"], 5)
        self.assertEqual(stats["num_documents"], 2)
        self.assertEqual(stats["total_tokens"], 12)

    def test_numpy_tensor_input(self):
        """Test building with numpy and torch inputs."""
        builder = IndexedDatasetBuilder(self.test_prefix, vocab_size=1000)

        # Add different input types
        builder.add_sequence([1, 2, 3])  # List
        builder.add_sequence(np.array([4, 5, 6]))  # NumPy
        builder.add_sequence(torch.tensor([7, 8, 9]))  # PyTorch

        stats = builder.finalize()

        self.assertEqual(stats["num_sequences"], 3)
        self.assertEqual(stats["total_tokens"], 9)

    def test_merge_datasets(self):
        """Test merging multiple datasets."""
        # Create first dataset
        prefix1 = os.path.join(self.temp_dir, "dataset1")
        builder1 = IndexedDatasetBuilder(prefix1, vocab_size=1000)
        builder1.add_sequence([1, 2, 3])
        builder1.add_sequence([4, 5])
        builder1.finalize()

        # Create second dataset
        prefix2 = os.path.join(self.temp_dir, "dataset2")
        builder2 = IndexedDatasetBuilder(prefix2, vocab_size=1000)
        builder2.add_sequence([6, 7, 8])
        builder2.finalize()

        # Create merged dataset
        merged_prefix = os.path.join(self.temp_dir, "merged")
        merged_builder = IndexedDatasetBuilder(merged_prefix, vocab_size=1000)
        merged_builder.merge_dataset(prefix1)
        merged_builder.merge_dataset(prefix2)

        stats = merged_builder.finalize()

        self.assertEqual(stats["num_sequences"], 3)
        self.assertEqual(stats["total_tokens"], 8)


class TestMMapIndexedDataset(unittest.TestCase):
    """Test memory-mapped dataset reading."""

    def setUp(self):
        """Create test dataset."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_prefix = os.path.join(self.temp_dir, "test_dataset")

        # Build test dataset
        builder = IndexedDatasetBuilder(self.test_prefix, vocab_size=1000)

        self.sequences = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12], [13, 14]]

        for seq in self.sequences:
            builder.add_sequence(seq)

        # Add document boundaries
        builder.document_indices = [0, 2, 4]  # Two documents
        builder.finalize()

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_basic_loading(self):
        """Test basic dataset loading."""
        dataset = MMapIndexedDataset(self.test_prefix)

        # Check length
        self.assertEqual(len(dataset), 4)

        # Check sequences
        for i, expected_seq in enumerate(self.sequences):
            loaded_seq = dataset[i]
            np.testing.assert_array_equal(loaded_seq, expected_seq)

    def test_slicing(self):
        """Test dataset slicing."""
        dataset = MMapIndexedDataset(self.test_prefix)

        # Get slice
        sequences = dataset[1:3]

        self.assertEqual(len(sequences), 2)
        np.testing.assert_array_equal(sequences[0], self.sequences[1])
        np.testing.assert_array_equal(sequences[1], self.sequences[2])

    def test_properties(self):
        """Test dataset properties."""
        dataset = MMapIndexedDataset(self.test_prefix)

        # Check sequence lengths
        expected_lengths = [len(seq) for seq in self.sequences]
        np.testing.assert_array_equal(dataset.sequence_lengths, expected_lengths)

        # Check document indices
        np.testing.assert_array_equal(dataset.document_indices, [0, 2, 4])

    def test_get_document(self):
        """Test document retrieval."""
        dataset = MMapIndexedDataset(self.test_prefix)

        # Get first document (sequences 0-1)
        doc0 = dataset.get_document(0)
        self.assertEqual(len(doc0), 2)
        np.testing.assert_array_equal(doc0[0], self.sequences[0])
        np.testing.assert_array_equal(doc0[1], self.sequences[1])

        # Get second document (sequences 2-3)
        doc1 = dataset.get_document(1)
        self.assertEqual(len(doc1), 2)
        np.testing.assert_array_equal(doc1[0], self.sequences[2])
        np.testing.assert_array_equal(doc1[1], self.sequences[3])

    def test_statistics(self):
        """Test dataset statistics."""
        dataset = MMapIndexedDataset(self.test_prefix)
        stats = dataset.get_stats()

        self.assertEqual(stats["num_sequences"], 4)
        self.assertEqual(stats["num_documents"], 2)
        self.assertEqual(stats["total_tokens"], 14)
        self.assertAlmostEqual(stats["avg_sequence_length"], 3.5)
        self.assertEqual(stats["min_sequence_length"], 2)
        self.assertEqual(stats["max_sequence_length"], 5)

    def test_exists_check(self):
        """Test dataset existence check."""
        # Should exist
        self.assertTrue(IndexedDataset.exists(self.test_prefix))

        # Should not exist
        self.assertFalse(
            IndexedDataset.exists(os.path.join(self.temp_dir, "nonexistent"))
        )


class TestDistributedDataset(unittest.TestCase):
    """Test distributed dataset functionality."""

    def setUp(self):
        """Create test dataset."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_prefix = os.path.join(self.temp_dir, "test_dataset")

        # Build dataset with 100 sequences
        builder = IndexedDatasetBuilder(self.test_prefix, vocab_size=1000)

        self.num_sequences = 100
        for i in range(self.num_sequences):
            # Each sequence has i+1 tokens
            builder.add_sequence(list(range(i, i + i + 1)))

        builder.finalize()

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_single_process(self):
        """Test with single process."""
        dataset = DistributedIndexedDataset(
            self.test_prefix, rank=0, world_size=1, shuffle=False
        )

        # Should see all samples
        self.assertEqual(len(dataset), self.num_sequences)

        # Check iteration
        samples = list(dataset)
        self.assertEqual(len(samples), self.num_sequences)

    def test_multi_process_distribution(self):
        """Test data distribution across multiple processes."""
        world_size = 4
        all_indices = set()

        for rank in range(world_size):
            dataset = DistributedIndexedDataset(
                self.test_prefix, rank=rank, world_size=world_size, shuffle=False
            )

            # Each rank should see subset
            self.assertLessEqual(len(dataset), self.num_sequences)

            # Collect indices
            indices = dataset._get_indices()

            # No overlap with other ranks
            self.assertEqual(len(all_indices & set(indices)), 0)
            all_indices.update(indices)

        # All samples should be covered
        self.assertEqual(len(all_indices), self.num_sequences)

    def test_shuffling(self):
        """Test epoch-based shuffling."""
        dataset = DistributedIndexedDataset(
            self.test_prefix, rank=0, world_size=2, shuffle=True, seed=42
        )

        # Get indices for different epochs
        dataset.set_epoch(0)
        indices_epoch0 = dataset._get_indices().copy()

        dataset.set_epoch(1)
        indices_epoch1 = dataset._get_indices().copy()

        dataset.set_epoch(0)
        indices_epoch0_repeat = dataset._get_indices().copy()

        # Different epochs should have different order
        self.assertFalse(np.array_equal(indices_epoch0, indices_epoch1))

        # Same epoch should be reproducible
        np.testing.assert_array_equal(indices_epoch0, indices_epoch0_repeat)

    def test_drop_last(self):
        """Test dropping incomplete batches."""
        world_size = 3

        # With drop_last=True
        total_with_drop = 0
        for rank in range(world_size):
            dataset = DistributedIndexedDataset(
                self.test_prefix,
                rank=rank,
                world_size=world_size,
                shuffle=False,
                drop_last=True,
            )
            total_with_drop += len(dataset)

        # With drop_last=False
        total_without_drop = 0
        for rank in range(world_size):
            dataset = DistributedIndexedDataset(
                self.test_prefix,
                rank=rank,
                world_size=world_size,
                shuffle=False,
                drop_last=False,
            )
            total_without_drop += len(dataset)

        # Without drop should have all samples
        self.assertEqual(total_without_drop, self.num_sequences)

        # With drop might have fewer
        self.assertLessEqual(total_with_drop, self.num_sequences)


class TestCompatibility(unittest.TestCase):
    """Test compatibility with Megatron-LM format."""

    def test_dtype_compatibility(self):
        """Test that dtypes match Megatron-LM codes."""
        # These codes should match Megatron-LM exactly
        expected_codes = {
            np.uint8: 1,
            np.int8: 2,
            np.int16: 3,
            np.int32: 4,
            np.int64: 5,
            np.float64: 6,
            np.float32: 7,
            np.uint16: 8,
        }

        for dtype, expected_code in expected_codes.items():
            actual_code = DType.code_from_dtype(dtype)
            self.assertEqual(
                actual_code,
                expected_code,
                f"Code mismatch for {dtype}: {actual_code} != {expected_code}",
            )


if __name__ == "__main__":
    unittest.main()
