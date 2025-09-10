"""Distributed wrapper for indexed datasets with rank-aware sampling."""

import logging
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch.distributed as dist
from torch.utils.data import IterableDataset

from .indexed_dataset import MMapIndexedDataset

logger = logging.getLogger(__name__)


class DistributedIndexedDataset(IterableDataset):
    """Distributed wrapper for indexed datasets.

    This class provides rank-aware sampling for distributed training,
    ensuring each rank sees a unique subset of the data with optional
    shuffling for each epoch.
    """

    def __init__(
        self,
        path_prefix: str,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        """Initialize distributed dataset.

        Args:
            path_prefix: Path to dataset without extensions.
            rank: Process rank (auto-detected if None).
            world_size: Total number of processes (auto-detected if None).
            shuffle: Whether to shuffle data each epoch.
            seed: Random seed for shuffling.
            drop_last: Drop incomplete last batch across ranks.
        """
        super().__init__()

        # Load base dataset
        self.dataset = MMapIndexedDataset(path_prefix)

        # Get distributed info
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Calculate samples per rank
        total_samples = len(self.dataset)
        samples_per_rank = total_samples // world_size

        if drop_last:
            # Each rank gets exactly samples_per_rank
            self.num_samples = samples_per_rank
        else:
            # Last rank may get more samples
            if rank == world_size - 1:
                self.num_samples = total_samples - (samples_per_rank * rank)
            else:
                self.num_samples = samples_per_rank

        logger.info(
            f"Distributed dataset initialized: "
            f"rank={rank}/{world_size}, "
            f"samples={self.num_samples}/{total_samples}"
        )

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling.

        Call this at the beginning of each epoch to ensure proper
        shuffling across distributed processes.

        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch

    def _get_indices(self) -> np.ndarray:
        """Get indices for this rank.

        Returns:
            Array of indices for this rank to process.
        """
        total_samples = len(self.dataset)

        # Create global indices
        if self.shuffle:
            # Deterministic shuffle based on epoch
            # Use a more efficient random generator for large datasets
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = rng.permutation(total_samples)
        else:
            # Use arange with explicit dtype for memory efficiency
            indices = np.arange(total_samples, dtype=np.int64)

        # Select indices for this rank
        if self.drop_last:
            # Each rank gets equal number of samples
            samples_per_rank = total_samples // self.world_size
            start_idx = self.rank * samples_per_rank
            end_idx = start_idx + samples_per_rank
        else:
            # Distribute all samples
            samples_per_rank = total_samples // self.world_size
            start_idx = self.rank * samples_per_rank

            if self.rank == self.world_size - 1:
                # Last rank gets remaining samples
                end_idx = total_samples
            else:
                end_idx = start_idx + samples_per_rank

        return indices[start_idx:end_idx]

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over samples for this rank.

        Yields:
            Token sequences assigned to this rank.
        """
        indices = self._get_indices()

        for idx in indices:
            data = self.dataset[idx]
            # Ensure we always yield an ndarray
            if isinstance(data, np.ndarray):
                yield data
            else:
                # This shouldn't happen with single index access
                raise TypeError(f"Expected ndarray, got {type(data)}")

    def __len__(self) -> int:
        """Return number of samples for this rank."""
        return self.num_samples

    def get_stats(self) -> dict:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset and distribution statistics.
        """
        base_stats = self.dataset.get_stats()

        # Add distributed stats
        base_stats.update(
            {
                "rank": self.rank,
                "world_size": self.world_size,
                "samples_per_rank": self.num_samples,
                "shuffle": self.shuffle,
                "drop_last": self.drop_last,
                "epoch": self.epoch,
            }
        )

        return base_stats


class DistributedDocumentDataset(DistributedIndexedDataset):
    """Document-aware distributed dataset.

    This variant ensures document boundaries are respected when
    distributing data across ranks, preventing documents from
    being split across different processes.
    """

    def __init__(
        self,
        path_prefix: str,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """Initialize document-aware distributed dataset.

        Args:
            path_prefix: Path to dataset without extensions.
            rank: Process rank.
            world_size: Total number of processes.
            shuffle: Whether to shuffle documents each epoch.
            seed: Random seed for shuffling.
        """
        super().__init__(
            path_prefix=path_prefix,
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,  # Handle document-level dropping
        )

        # Get document boundaries
        self.document_indices = self.dataset.document_indices
        self.num_documents = len(self.document_indices) - 1

        # Distribute documents across ranks
        docs_per_rank = self.num_documents // self.world_size

        if self.rank == self.world_size - 1:
            # Last rank gets remaining documents
            self.rank_documents = list(
                range(self.rank * docs_per_rank, self.num_documents)
            )
        else:
            self.rank_documents = list(
                range(self.rank * docs_per_rank, (self.rank + 1) * docs_per_rank)
            )

        logger.info(
            f"Document dataset: rank={self.rank}/{self.world_size}, "
            f"documents={len(self.rank_documents)}/{self.num_documents}"
        )

    def _get_document_sequences(self, doc_idx: int) -> List[int]:
        """Get sequence indices for a document.

        Args:
            doc_idx: Document index.

        Returns:
            List of sequence indices in the document.
        """
        start = self.document_indices[doc_idx] if doc_idx > 0 else 0
        end = self.document_indices[doc_idx + 1]
        return list(range(start, end))

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over sequences respecting document boundaries.

        Yields:
            Token sequences from documents assigned to this rank.
        """
        # Shuffle documents if needed
        doc_order: np.ndarray
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            doc_order = rng.permutation(self.rank_documents)
        else:
            doc_order = np.array(self.rank_documents)

        # Yield all sequences from each document
        for doc_idx in doc_order:
            seq_indices = self._get_document_sequences(doc_idx)
            for seq_idx in seq_indices:
                data = self.dataset[seq_idx]
                # Ensure we always yield an ndarray
                if isinstance(data, np.ndarray):
                    yield data
                else:
                    raise TypeError(f"Expected ndarray, got {type(data)}")


class CachedDistributedDataset(DistributedIndexedDataset):
    """Distributed dataset with local caching for frequently accessed data.

    This variant maintains an LRU cache of recently accessed sequences
    to reduce I/O overhead for commonly accessed patterns.
    """

    def __init__(
        self,
        path_prefix: str,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        cache_size: int = 1000,
    ):
        """Initialize cached distributed dataset.

        Args:
            path_prefix: Path to dataset.
            rank: Process rank.
            world_size: Number of processes.
            shuffle: Whether to shuffle.
            seed: Random seed.
            cache_size: Maximum number of sequences to cache.
        """
        super().__init__(
            path_prefix=path_prefix,
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            seed=seed,
        )

        self.cache_size = cache_size
        self.cache: Dict[int, np.ndarray] = {}
        self.cache_order: List[int] = []
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_cached(self, idx: int) -> np.ndarray:
        """Get sequence with caching.

        Args:
            idx: Sequence index.

        Returns:
            Token sequence.
        """
        if idx in self.cache:
            # Cache hit - move to end (most recently used)
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            self.cache_hits += 1
            cached_value: np.ndarray = self.cache[idx]
            return cached_value

        # Cache miss
        self.cache_misses += 1
        sequence = self.dataset[idx]

        # Ensure we have an ndarray
        if not isinstance(sequence, np.ndarray):
            raise TypeError(f"Expected ndarray from dataset, got {type(sequence)}")

        # Add to cache
        if len(self.cache) >= self.cache_size:
            # Evict least recently used
            evict_idx = self.cache_order.pop(0)
            del self.cache[evict_idx]

        self.cache[idx] = sequence
        self.cache_order.append(idx)

        return sequence

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate with caching.

        Yields:
            Cached token sequences.
        """
        indices = self._get_indices()

        for idx in indices:
            yield self._get_cached(idx)

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache performance metrics.
        """
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0

        return {
            "cache_size": self.cache_size,
            "cached_sequences": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
        }
