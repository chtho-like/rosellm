"""Builder for creating indexed datasets from raw data."""

import gc
import logging
import os
import shutil
from typing import Any, List, Optional, Union

import numpy as np
import torch

from .data_types import DType
from .indexed_dataset import IndexReader, IndexWriter

logger = logging.getLogger(__name__)


class IndexedDatasetBuilder:
    """Build indexed datasets from raw text or tokens.

    This class provides utilities for converting raw text data into
    the efficient indexed binary format used by MMapIndexedDataset.
    """

    def __init__(
        self,
        output_prefix: str,
        vocab_size: Optional[int] = None,
        dtype: Optional[type] = None,
    ):
        """Initialize dataset builder.

        Args:
            output_prefix: Output path without .idx/.bin extensions.
            vocab_size: Vocabulary size for automatic dtype selection.
            dtype: Explicit dtype (overrides automatic selection).
        """
        self.output_prefix = output_prefix

        # Determine dtype
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = DType.optimal_dtype(vocab_size)

        logger.info(f"Building dataset at {output_prefix} with dtype {self.dtype}")

        # Open data file
        self.data_file = open(f"{output_prefix}.bin", "wb")

        # Track sequences and documents
        self.sequence_lengths: List[int] = []
        self.document_indices: List[int] = []
        self.current_document_start = 0

        # Statistics
        self.total_tokens = 0
        self.num_documents = 0

    def add_sequence(self, tokens: Union[List[int], np.ndarray, torch.Tensor]) -> None:
        """Add a single sequence to the dataset.

        Args:
            tokens: Token IDs as list, numpy array, or torch tensor.
        """
        # Convert to numpy array
        if isinstance(tokens, list):
            tokens = np.array(tokens, dtype=self.dtype)
        elif isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy().astype(self.dtype)
        elif isinstance(tokens, np.ndarray):
            tokens = tokens.astype(self.dtype)
        else:
            raise TypeError(f"Unsupported token type: {type(tokens)}")

        # Write to file
        self.data_file.write(tokens.tobytes(order="C"))

        # Update metadata
        self.sequence_lengths.append(len(tokens))
        self.total_tokens += len(tokens)

    def add_document(
        self,
        sequences: Union[List[List[int]], List[np.ndarray]],
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Add a complete document with multiple sequences.

        Args:
            sequences: List of token sequences or text if tokenizer provided.
            tokenizer: Optional tokenizer for converting text to tokens.
        """
        document_start = len(self.sequence_lengths)

        for seq in sequences:
            if tokenizer is not None and isinstance(seq, str):
                # Tokenize text
                tokens = tokenizer.encode(seq)
            else:
                tokens = seq

            self.add_sequence(tokens)

        # Mark document boundary
        if len(self.sequence_lengths) > document_start:
            self.document_indices.append(len(self.sequence_lengths))
            self.num_documents += 1

    def end_document(self) -> None:
        """Mark the end of the current document.

        Call this after adding sequences with add_sequence() to mark
        document boundaries.
        """
        if len(self.sequence_lengths) > self.current_document_start:
            self.document_indices.append(len(self.sequence_lengths))
            self.current_document_start = len(self.sequence_lengths)
            self.num_documents += 1

    def merge_dataset(self, path_prefix: str) -> None:
        """Merge another indexed dataset into this one.

        Args:
            path_prefix: Path prefix of dataset to merge.
        """
        logger.info(f"Merging dataset from {path_prefix}")

        # Load index
        index = IndexReader(f"{path_prefix}.idx")

        # Verify dtype compatibility
        if index.dtype != self.dtype:
            raise ValueError(
                f"Dtype mismatch: builder has {self.dtype}, "
                f"merge source has {index.dtype}"
            )

        # Update sequence lengths
        offset = len(self.sequence_lengths)
        self.sequence_lengths.extend(index.sequence_lengths)

        # Update document indices (adjust for offset)
        for doc_idx in index.document_indices[1:]:  # Skip first (0)
            self.document_indices.append(offset + doc_idx)

        # Copy binary data
        with open(f"{path_prefix}.bin", "rb") as src:
            shutil.copyfileobj(src, self.data_file)

        # Update statistics
        self.total_tokens += np.sum(index.sequence_lengths)
        self.num_documents += len(index.document_indices) - 1

        # Clean up
        del index
        gc.collect()

    def finalize(self) -> dict:
        """Finalize dataset and write index file.

        Returns:
            Dictionary with dataset statistics.
        """
        # Close data file
        self.data_file.close()

        # Ensure document indices start with 0
        if not self.document_indices or self.document_indices[0] != 0:
            self.document_indices.insert(0, 0)

        # Write index file
        idx_path = f"{self.output_prefix}.idx"
        with IndexWriter(idx_path, self.dtype) as writer:
            writer.write(self.sequence_lengths, self.document_indices)

        # Calculate statistics
        stats = {
            "num_sequences": len(self.sequence_lengths),
            "num_documents": self.num_documents,
            "total_tokens": self.total_tokens,
            "dtype": str(self.dtype),
            "index_path": idx_path,
            "data_path": f"{self.output_prefix}.bin",
        }

        if self.sequence_lengths:
            seq_lengths = np.array(self.sequence_lengths)
            stats.update(
                {
                    "avg_sequence_length": float(np.mean(seq_lengths)),
                    "min_sequence_length": int(np.min(seq_lengths)),
                    "max_sequence_length": int(np.max(seq_lengths)),
                }
            )

        logger.info(f"Finalized dataset: {stats}")
        return stats


class ParallelDatasetBuilder:
    """Build large datasets in parallel across multiple workers.

    This class coordinates multiple builders to create dataset shards
    that can be merged into a single dataset.
    """

    def __init__(
        self, output_prefix: str, num_workers: int, vocab_size: Optional[int] = None
    ):
        """Initialize parallel builder.

        Args:
            output_prefix: Base output path.
            num_workers: Number of parallel builders.
            vocab_size: Vocabulary size for dtype selection.
        """
        self.output_prefix = output_prefix
        self.num_workers = num_workers
        self.vocab_size = vocab_size

        # Create shard builders
        self.builders = []
        for i in range(num_workers):
            shard_prefix = f"{output_prefix}_shard{i}"
            builder = IndexedDatasetBuilder(shard_prefix, vocab_size=vocab_size)
            self.builders.append(builder)

    def get_builder(self, worker_id: int) -> IndexedDatasetBuilder:
        """Get builder for specific worker.

        Args:
            worker_id: Worker ID (0 to num_workers-1).

        Returns:
            IndexedDatasetBuilder for that worker.
        """
        return self.builders[worker_id]

    def finalize_and_merge(self) -> dict:
        """Finalize all shards and merge into single dataset.

        Returns:
            Dictionary with final dataset statistics.
        """
        logger.info(f"Finalizing {self.num_workers} shards")

        # Finalize all shards
        for i, builder in enumerate(self.builders):
            logger.info(f"Finalizing shard {i}")
            builder.finalize()

        # Create final merged dataset
        logger.info("Merging shards into final dataset")
        final_builder = IndexedDatasetBuilder(
            self.output_prefix, vocab_size=self.vocab_size
        )

        # Merge all shards
        for i in range(self.num_workers):
            shard_prefix = f"{self.output_prefix}_shard{i}"
            final_builder.merge_dataset(shard_prefix)

            # Clean up shard files
            os.remove(f"{shard_prefix}.idx")
            os.remove(f"{shard_prefix}.bin")

        # Finalize merged dataset
        stats = final_builder.finalize()
        logger.info(f"Created final dataset: {stats}")

        return stats
