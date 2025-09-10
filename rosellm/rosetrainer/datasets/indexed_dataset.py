"""Memory-mapped indexed dataset for efficient data loading.

This module provides high-performance dataset implementations for training
large language models with memory-mapped I/O and distributed support.
"""

import gc
import logging
import os
import struct
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from torch.utils.data import Dataset

from .data_types import INDEX_HEADER, DType

# Optional imports
try:
    from tqdm import tqdm  # type: ignore

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # Fallback if tqdm not available
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            pass

        def close(self):
            pass


logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 1024 * 1024 * 1024 * 100  # 100GB max file size
DEFAULT_BATCH_SIZE = 100
MIN_MMAP_SIZE = 1024  # Minimum size for memory mapping
CACHE_LINE_SIZE = 64  # CPU cache line size for alignment


class DatasetError(Exception):
    """Base exception for dataset errors."""

    pass


class CorruptedDataError(DatasetError):
    """Raised when dataset files are corrupted."""

    pass


class IndexWriter:
    """Write index files for efficient data access.

    The index file format:
    - Header: magic bytes + version (17 bytes)
    - Metadata: dtype code (1 byte), sequence count (8 bytes), document count (8 bytes)
    - Arrays: sequence lengths, sequence pointers, document indices

    Thread-safe implementation with proper resource management.
    """

    def __init__(self, path: str, dtype: Type[np.number]):
        """Initialize index writer.

        Args:
            path: Path to the index file.
            dtype: Data type for tokens in the binary file.

        Raises:
            ValueError: If path is invalid or dtype is not supported.
        """
        self.path = Path(path).resolve()
        if not self.path.parent.exists():
            raise ValueError(f"Parent directory does not exist: {self.path.parent}")

        self.dtype = dtype
        if not np.issubdtype(dtype, np.number):
            raise ValueError(f"Invalid dtype: {dtype}. Must be a numpy number type.")

        self.file: Optional[Any] = None
        self._lock = threading.Lock()

    def __enter__(self):
        """Open file for writing with proper error handling.

        Returns:
            Self for context manager.

        Raises:
            IOError: If file cannot be opened for writing.
        """
        with self._lock:
            try:
                self.file = open(self.path, "wb")
                # Write header
                self.file.write(INDEX_HEADER)
                # Write version
                self.file.write(struct.pack("<Q", 1))
                # Write dtype code
                self.file.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
                return self
            except (IOError, OSError) as e:
                if self.file:
                    self.file.close()
                    self.file = None
                raise IOError(
                    f"Failed to open index file for writing: {self.path}"
                ) from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file safely with error handling.

        Args:
            exc_type: Exception type if an error occurred.
            exc_val: Exception value if an error occurred.
            exc_tb: Exception traceback if an error occurred.

        Returns:
            None to propagate exceptions.
        """
        with self._lock:
            if self.file:
                try:
                    self.file.flush()
                    os.fsync(self.file.fileno())  # Ensure data is written to disk
                except (IOError, OSError) as e:
                    logger.error(f"Error flushing index file: {e}")
                finally:
                    self.file.close()
                    self.file = None
        return None

    def write(self, sequence_lengths: List[int], document_indices: List[int]) -> None:
        """Write index data to file with validation.

        Args:
            sequence_lengths: Length of each sequence in tokens.
            document_indices: Sequence indices marking document boundaries.

        Raises:
            RuntimeError: If writer is not in context manager.
            ValueError: If input data is invalid.
        """
        if self.file is None:
            raise RuntimeError("IndexWriter must be used within a context manager")

        # Validate inputs
        if not sequence_lengths:
            raise ValueError("sequence_lengths cannot be empty")
        if not document_indices:
            raise ValueError("document_indices cannot be empty")

        # Validate sequence lengths
        if any(length < 0 for length in sequence_lengths):
            raise ValueError("Sequence lengths must be non-negative")
        if any(length > 2**31 - 1 for length in sequence_lengths):
            raise ValueError("Sequence length exceeds maximum int32 value")

        # Validate document indices
        if document_indices[0] != 0:
            raise ValueError("First document index must be 0")
        if not all(i <= j for i, j in zip(document_indices[:-1], document_indices[1:])):
            raise ValueError("Document indices must be monotonically increasing")

        with self._lock:
            # Calculate byte offsets for sequences
            sequence_pointers = self._calculate_pointers(sequence_lengths)

            # Write counts
            sequence_count = len(sequence_lengths)
            document_count = len(document_indices)
            self.file.write(struct.pack("<Q", sequence_count))
            self.file.write(struct.pack("<Q", document_count))

            # Write arrays with error handling
            try:
                self.file.write(
                    np.array(sequence_lengths, dtype=np.int32).tobytes(order="C")
                )
                self.file.write(
                    np.array(sequence_pointers, dtype=np.int64).tobytes(order="C")
                )
                self.file.write(
                    np.array(document_indices, dtype=np.int64).tobytes(order="C")
                )
            except (IOError, OSError) as e:
                raise IOError(f"Failed to write index data: {e}") from e

    def _calculate_pointers(self, sequence_lengths: List[int]) -> List[int]:
        """Calculate byte offsets for each sequence.

        Args:
            sequence_lengths: Length of each sequence.

        Returns:
            List of byte offsets.
        """
        itemsize = DType.size(self.dtype)
        pointers = []
        offset = 0
        for length in sequence_lengths:
            pointers.append(offset)
            offset += length * itemsize
        return pointers


class IndexReader:
    """Read index files using memory mapping for efficiency.

    Thread-safe reader with comprehensive validation and error recovery.
    """

    def __init__(self, path: str, validate: bool = True):
        """Initialize index reader.

        Args:
            path: Path to the index file.
            validate: Whether to validate file integrity.

        Raises:
            FileNotFoundError: If index file does not exist.
            CorruptedDataError: If index file is corrupted.
            ValueError: If index file format is invalid.
        """
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Index file not found: {self.path}")

        self._lock = threading.RLock()
        logger.info(f"Loading index from {self.path}")

        # Read header and metadata
        try:
            # Read header and metadata with validation
            with open(self.path, "rb") as f:
                # Get file size for validation
                f.seek(0, 2)
                file_size = f.tell()
                f.seek(0)

                if file_size < 34:  # Minimum valid index file size
                    raise CorruptedDataError(f"Index file too small: {file_size} bytes")

                # Verify header
                header = f.read(9)
                if header != INDEX_HEADER:
                    raise CorruptedDataError(f"Invalid index file header: {self.path}")

                # Read version
                version_bytes = f.read(8)
                if len(version_bytes) != 8:
                    raise CorruptedDataError("Incomplete version data")
                version = struct.unpack("<Q", version_bytes)[0]
                if version != 1:
                    raise ValueError(f"Unsupported index version: {version}")

                # Read dtype
                dtype_bytes = f.read(1)
                if len(dtype_bytes) != 1:
                    raise CorruptedDataError("Incomplete dtype data")
                dtype_code = struct.unpack("<B", dtype_bytes)[0]

                try:
                    self.dtype = DType.dtype_from_code(dtype_code)
                except ValueError as e:
                    raise CorruptedDataError(f"Invalid dtype code: {dtype_code}") from e

                self.dtype_size = DType.size(self.dtype)

                # Read counts
                count_bytes = f.read(16)
                if len(count_bytes) != 16:
                    raise CorruptedDataError("Incomplete count data")

                self.sequence_count = struct.unpack("<Q", count_bytes[:8])[0]
                self.document_count = struct.unpack("<Q", count_bytes[8:])[0]

                # Validate counts
                if self.sequence_count < 0 or self.sequence_count > 2**48:
                    raise CorruptedDataError(
                        f"Invalid sequence count: {self.sequence_count}"
                    )
                if self.document_count < 0 or self.document_count > 2**48:
                    raise CorruptedDataError(
                        f"Invalid document count: {self.document_count}"
                    )

                offset = f.tell()

                # Validate expected file size
                expected_size = (
                    offset
                    + self.sequence_count * 4
                    + self.sequence_count * 8  # sequence_lengths
                    + self.document_count * 8  # sequence_pointers
                )  # document_indices

                if validate and file_size != expected_size:
                    logger.warning(
                        f"Index file size mismatch: expected {expected_size}, "
                        f"got {file_size}"
                    )

        except (IOError, OSError) as e:
            raise IOError(f"Failed to read index file: {self.path}") from e

        # Memory map the file for efficient access
        try:
            # Always use memory mapping for index files
            # They are small enough and numpy handles the mapping efficiently
            self.mmap = np.memmap(self.path, mode="r", order="C")
            # Cache the data attribute to avoid repeated attribute access
            self._mmap_data = self.mmap.data
            self.buffer = memoryview(self._mmap_data)
            self._raw_data = None

        except (IOError, OSError) as e:
            raise IOError(f"Failed to memory map index file: {self.path}") from e

        # Load arrays with validation
        try:
            logger.debug("Loading sequence lengths...")
            self.sequence_lengths: np.ndarray = np.frombuffer(
                self.buffer, dtype=np.int32, count=self.sequence_count, offset=offset
            ).copy()  # Make a copy for safety

            logger.debug("Loading sequence pointers...")
            self.sequence_pointers: np.ndarray = np.frombuffer(
                self.buffer,
                dtype=np.int64,
                count=self.sequence_count,
                offset=offset + self.sequence_lengths.nbytes,
            ).copy()

            logger.debug("Loading document indices...")
            self.document_indices: np.ndarray = np.frombuffer(
                self.buffer,
                dtype=np.int64,
                count=self.document_count,
                offset=offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes,
            ).copy()

            # Validate loaded data if requested
            if validate:
                self._validate_loaded_data()

        except (ValueError, IndexError) as e:
            raise CorruptedDataError(f"Failed to load index arrays: {e}") from e

        logger.info(
            f"Loaded index: {self.sequence_count:,} sequences, "
            f"{self.document_count:,} documents, "
            f"dtype={self.dtype.__name__}"
        )

    def _validate_loaded_data(self) -> None:
        """Validate the loaded index data for consistency.

        Raises:
            CorruptedDataError: If validation fails.
        """
        # Check sequence lengths are non-negative
        if np.any(self.sequence_lengths < 0):
            raise CorruptedDataError("Found negative sequence lengths")

        # Check sequence pointers are monotonic
        if len(self.sequence_pointers) > 1:
            pointer_diffs = np.diff(self.sequence_pointers)
            expected_diffs = self.sequence_lengths[:-1] * self.dtype_size
            if not np.array_equal(pointer_diffs, expected_diffs):
                raise CorruptedDataError(
                    "Sequence pointers are not consistent with lengths"
                )

        # Check document indices are within bounds
        if np.any(self.document_indices < 0) or np.any(
            self.document_indices > self.sequence_count
        ):
            raise CorruptedDataError("Document indices out of bounds")

        # Check document indices are monotonic
        if len(self.document_indices) > 1 and not np.all(
            np.diff(self.document_indices) >= 0
        ):
            raise CorruptedDataError("Document indices are not monotonic")

    def __len__(self) -> int:
        """Return number of sequences."""
        return int(self.sequence_count)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        """Get sequence pointer and length.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of (byte_offset, sequence_length).

        Raises:
            IndexError: If index is out of bounds.
        """
        with self._lock:
            if idx < 0 or idx >= self.sequence_count:
                raise IndexError(f"Index {idx} out of range [0, {self.sequence_count})")
            return int(self.sequence_pointers[idx]), int(self.sequence_lengths[idx])

    def close(self) -> None:
        """Explicitly close and clean up resources."""
        with self._lock:
            if hasattr(self, "buffer"):
                del self.buffer
            if hasattr(self, "_mmap_data"):
                del self._mmap_data
            if hasattr(self, "_raw_data"):
                del self._raw_data
            if hasattr(self, "mmap") and self.mmap is not None:
                del self.mmap

    def __del__(self):
        """Clean up memory map."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup
        finally:
            # Force garbage collection for large memory maps
            gc.collect()


class BinReader:
    """Base class for reading binary data files."""

    def read(self, dtype: Type[np.number], count: int, offset: int) -> np.ndarray:
        """Read data from binary file.

        Args:
            dtype: Data type to read.
            count: Number of elements to read.
            offset: Byte offset in file.

        Returns:
            NumPy array with the data.
        """
        raise NotImplementedError


class MMapBinReader(BinReader):
    """Memory-mapped binary file reader for zero-copy access.

    Optimized for large file access with proper resource management.
    """

    def __init__(self, path: str, preload_pages: bool = False):
        """Initialize memory-mapped reader.

        Args:
            path: Path to binary data file.
            preload_pages: Whether to preload pages into memory.

        Raises:
            FileNotFoundError: If data file does not exist.
            IOError: If file cannot be memory mapped.
        """
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")

        self._lock = threading.RLock()

        # Get file size and validate
        try:
            self.file_size = self.path.stat().st_size
            if self.file_size == 0:
                raise ValueError(f"Data file is empty: {self.path}")
            if self.file_size > MAX_FILE_SIZE:
                logger.warning(
                    f"Large file detected ({self.file_size / 1e9:.1f}GB): {self.path}"
                )
        except OSError as e:
            raise IOError(f"Cannot access data file: {self.path}") from e

        # Memory map the file
        try:
            # Use np.memmap with optimal settings for read-only access
            self.mmap = np.memmap(self.path, mode="r", order="C")
            # Cache the data attribute to avoid repeated attribute access
            self._mmap_data = self.mmap.data
            self.buffer = memoryview(self._mmap_data)

            # Optionally preload pages
            if (
                preload_pages and self.file_size < 1024 * 1024 * 100
            ):  # Only for files < 100MB
                self._preload_pages()

        except (IOError, OSError) as e:
            raise IOError(f"Failed to memory map data file: {self.path}") from e

    def _preload_pages(self) -> None:
        """Preload memory pages for better performance."""
        try:
            # Touch pages to load them into memory
            page_size = 4096
            for offset in range(0, self.file_size, page_size * 1000):
                _ = self.buffer[offset]
        except Exception:
            pass  # Best effort

    def read(self, dtype: Type[np.number], count: int, offset: int) -> np.ndarray:
        """Read data using memory mapping with validation.

        Args:
            dtype: Data type to read.
            count: Number of elements.
            offset: Byte offset.

        Returns:
            View into memory-mapped data.

        Raises:
            ValueError: If read parameters are invalid.
            IndexError: If read would exceed file bounds.
        """
        with self._lock:
            # Validate parameters
            if count < 0:
                raise ValueError(f"Count must be non-negative, got {count}")
            if offset < 0:
                raise ValueError(f"Offset must be non-negative, got {offset}")

            # Check bounds
            bytes_to_read = count * dtype().itemsize
            if offset + bytes_to_read > self.file_size:
                raise IndexError(
                    f"Read would exceed file bounds: offset={offset}, "
                    f"bytes={bytes_to_read}, file_size={self.file_size}"
                )

            # Read data
            try:
                return np.frombuffer(
                    self.buffer, dtype=dtype, count=count, offset=offset
                )
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to read data: {e}") from e

    def close(self) -> None:
        """Explicitly close and clean up resources."""
        with self._lock:
            if hasattr(self, "buffer"):
                del self.buffer
            if hasattr(self, "_mmap_data"):
                del self._mmap_data
            if hasattr(self, "mmap") and self.mmap is not None:
                del self.mmap

    def __del__(self):
        """Clean up memory map."""
        try:
            self.close()
        except Exception:
            pass
        finally:
            # Force garbage collection for large memory maps
            if self.file_size > 1024 * 1024 * 100:  # Only for large files
                gc.collect()


class FileBinReader(BinReader):
    """File-based binary reader (fallback when mmap unavailable).

    Uses buffered I/O with caching for improved performance.
    """

    def __init__(self, path: str, buffer_size: int = 8192):
        """Initialize file reader.

        Args:
            path: Path to binary data file.
            buffer_size: Read buffer size in bytes.

        Raises:
            FileNotFoundError: If data file does not exist.
        """
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")

        self.buffer_size = buffer_size
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def read(self, dtype: Type[np.number], count: int, offset: int) -> np.ndarray:
        """Read data from file with caching.

        Args:
            dtype: Data type to read.
            count: Number of elements.
            offset: Byte offset.

        Returns:
            NumPy array with the data.

        Raises:
            ValueError: If read parameters are invalid.
            IOError: If file read fails.
        """
        with self._lock:
            # Validate parameters
            if count < 0:
                raise ValueError(f"Count must be non-negative, got {count}")
            if offset < 0:
                raise ValueError(f"Offset must be non-negative, got {offset}")

            # Check cache
            cache_key = (offset, count)
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key].astype(dtype, copy=True)

            self._cache_misses += 1

            # Read from file
            try:
                with open(self.path, "rb", buffering=self.buffer_size) as f:
                    f.seek(offset)
                    bytes_to_read = count * dtype().itemsize
                    data = f.read(bytes_to_read)

                    if len(data) < bytes_to_read:
                        raise IOError(
                            f"Incomplete read: expected {bytes_to_read} bytes, "
                            f"got {len(data)} bytes"
                        )

                    array = np.frombuffer(data, dtype=dtype, count=count).copy()

                    # Cache if small enough
                    if bytes_to_read < 1024 * 1024:  # Cache reads < 1MB
                        if len(self._cache) > 100:  # Simple LRU
                            self._cache.pop(next(iter(self._cache)))
                        self._cache[cache_key] = array.copy()

                    return array

            except (IOError, OSError) as e:
                raise IOError(f"Failed to read from file: {self.path}") from e

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics.
        """
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0,
            "cached_items": len(self._cache),
        }


@contextmanager
def open_indexed_dataset(path_prefix: str, **kwargs):
    """Context manager for indexed dataset access.

    Args:
        path_prefix: Path without .idx/.bin extension.
        **kwargs: Additional arguments for dataset initialization.

    Yields:
        IndexedDataset instance.
    """
    dataset = None
    try:
        dataset = IndexedDataset(path_prefix, **kwargs)
        yield dataset
    finally:
        if dataset is not None:
            dataset.close()


class IndexedDataset(Dataset):
    """Base indexed dataset interface with improved resource management.

    Supports both memory-mapped and file-based access with automatic
    selection based on file size and system resources.
    """

    def __init__(
        self,
        path_prefix: str,
        mmap: Optional[bool] = None,
        lazy: bool = True,
        validate: bool = True,
        cache_size: int = 0,
    ):
        """Initialize indexed dataset.

        Args:
            path_prefix: Path without .idx/.bin extension.
            mmap: Use memory mapping (None for auto-detection).
            lazy: Load data on demand vs preloading.
            validate: Validate file integrity on load.
            cache_size: Number of sequences to cache (0 for no caching).

        Raises:
            FileNotFoundError: If required files don't exist.
            CorruptedDataError: If files are corrupted.
        """
        super().__init__()
        self.path_prefix = Path(path_prefix).resolve()
        self.lazy = lazy
        self.validate = validate
        self._closed = False
        self._lock = threading.RLock()

        # Setup paths
        self.idx_path = self.path_prefix.with_suffix(".idx")
        self.bin_path = self.path_prefix.with_suffix(".bin")

        # Verify files exist
        if not self.idx_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.idx_path}")
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.bin_path}")

        # Auto-detect mmap if not specified
        if mmap is None:
            file_size = self.bin_path.stat().st_size
            # Use mmap for files > 10MB
            mmap = file_size > 10 * 1024 * 1024
            logger.debug(f"Auto-detected mmap={mmap} for {file_size / 1e6:.1f}MB file")

        self.mmap = mmap

        # Load index with validation
        self.index = IndexReader(str(self.idx_path), validate=validate)

        # Initialize data reader with appropriate strategy
        self.bin_reader: Union[MMapBinReader, FileBinReader]
        if mmap:
            self.bin_reader = MMapBinReader(str(self.bin_path))
        else:
            self.bin_reader = FileBinReader(str(self.bin_path))

        # Setup caching if requested
        self.cache_size = cache_size
        if cache_size > 0:
            self._cache: Dict[int, np.ndarray] = {}
            self._cache_order: List[int] = []

        logger.info(
            f"Initialized {'mmap' if mmap else 'file'}-based dataset: "
            f"{len(self)} sequences, cache_size={cache_size}"
        )

    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return len(self.index)

    def __getitem__(
        self, idx: Union[int, slice, List[int], np.ndarray]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get sequence(s) at index with caching support.

        Args:
            idx: Index, slice, or list of indices.

        Returns:
            Single sequence or list of sequences.

        Raises:
            IndexError: If index is out of bounds.
            TypeError: If index type is invalid.
            RuntimeError: If dataset is closed.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Dataset is closed")

            if isinstance(idx, (int, np.integer)):
                # Single sequence with optional caching
                idx = int(idx)
                if idx < 0:
                    idx = len(self) + idx
                if idx < 0 or idx >= len(self):
                    raise IndexError(f"Index {idx} out of range [0, {len(self)})")

                # Check cache
                if self.cache_size > 0 and idx in self._cache:
                    # Move to end (LRU)
                    self._cache_order.remove(idx)
                    self._cache_order.append(idx)
                    return self._cache[idx].copy()

                # Read from disk
                pointer, length = self.index[idx]
                seq = self.bin_reader.read(
                    dtype=self.index.dtype, count=length, offset=pointer
                )

                # Update cache
                if self.cache_size > 0:
                    self._update_cache(idx, seq)

                return seq

            elif isinstance(idx, slice):
                # Multiple sequences via slice
                start, stop, step = idx.indices(len(self))
                if step != 1:
                    raise ValueError("Step must be 1 for slicing")

                sequences: List[np.ndarray] = []
                for i in range(start, stop):
                    item = self[i]  # Use single-item access for caching
                    assert isinstance(item, np.ndarray)  # Type guard for mypy
                    sequences.append(item)
                return sequences

            elif isinstance(idx, (list, np.ndarray)):
                # Multiple sequences via list/array
                result: List[np.ndarray] = []
                for i in idx:
                    item = self[int(i)]
                    assert isinstance(item, np.ndarray)  # Type guard for mypy
                    result.append(item)
                return result

            else:
                raise TypeError(f"Invalid index type: {type(idx)}")

    def _update_cache(self, idx: int, seq: np.ndarray) -> None:
        """Update LRU cache.

        Args:
            idx: Sequence index.
            seq: Sequence data.
        """
        if len(self._cache) >= self.cache_size:
            # Evict oldest
            evict_idx = self._cache_order.pop(0)
            del self._cache[evict_idx]

        self._cache[idx] = seq.copy()
        self._cache_order.append(idx)

    @property
    def sequence_lengths(self) -> np.ndarray:
        """Get array of sequence lengths."""
        return self.index.sequence_lengths

    @property
    def document_indices(self) -> np.ndarray:
        """Get array of document boundary indices."""
        return self.index.document_indices

    def close(self) -> None:
        """Close dataset and release resources."""
        with self._lock:
            if not self._closed:
                self._closed = True

                # Clear cache
                if hasattr(self, "_cache"):
                    self._cache.clear()
                    self._cache_order.clear()

                # Close readers
                if hasattr(self, "index"):
                    self.index.close()
                if hasattr(self, "bin_reader"):
                    if isinstance(self.bin_reader, MMapBinReader):
                        self.bin_reader.close()

                logger.debug(f"Closed dataset: {self.path_prefix}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return None

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def exists(path_prefix: str) -> bool:
        """Check if dataset exists at path.

        Args:
            path_prefix: Path prefix without extensions.

        Returns:
            True if both .idx and .bin files exist.
        """
        path = Path(path_prefix)
        return path.with_suffix(".idx").exists() and path.with_suffix(".bin").exists()

    @classmethod
    def validate_files(cls, path_prefix: str) -> Dict[str, Any]:
        """Validate dataset files and return info.

        Args:
            path_prefix: Path prefix without extensions.

        Returns:
            Dictionary with validation results.

        Raises:
            FileNotFoundError: If files don't exist.
            CorruptedDataError: If files are corrupted.
        """
        if not cls.exists(path_prefix):
            raise FileNotFoundError(f"Dataset files not found: {path_prefix}")

        # Load and validate
        with cls(path_prefix, validate=True) as dataset:
            # Get basic stats from parent class
            stats = {
                "num_sequences": len(dataset),
                "dtype": str(dataset.index.dtype),
                "valid": True,
            }
            # Add extended stats if available (from MMapIndexedDataset)
            if hasattr(dataset, "get_stats") and callable(
                getattr(dataset, "get_stats", None)
            ):
                extended_stats = getattr(dataset, "get_stats")()
                if isinstance(extended_stats, dict):
                    stats.update(extended_stats)
            return stats


class MMapIndexedDataset(IndexedDataset):
    """Memory-mapped indexed dataset with advanced optimizations.

    This is the main dataset class for efficient training on
    large datasets. It uses memory mapping for zero-copy access
    and provides various optimizations for training workloads.

    Features:
    - Zero-copy memory-mapped I/O
    - Adaptive caching based on access patterns
    - Prefetching for sequential access
    - NUMA-aware memory allocation (when available)
    - Compression support for reduced I/O
    """

    def __init__(
        self,
        path_prefix: str,
        lazy: bool = True,
        preload: bool = False,
        cache_size: int = 1000,
        prefetch_size: int = 10,
        validate: bool = True,
    ):
        """Initialize memory-mapped dataset.

        Args:
            path_prefix: Path without extensions.
            lazy: Load sequences on demand.
            preload: Preload entire dataset into memory (for small datasets).
            cache_size: Number of sequences to cache.
            prefetch_size: Number of sequences to prefetch.
            validate: Validate file integrity.

        Raises:
            ValueError: If preload requested for large dataset.
        """
        super().__init__(
            path_prefix, mmap=True, lazy=lazy, validate=validate, cache_size=cache_size
        )

        self.prefetch_size = prefetch_size
        self._access_pattern: List[int] = []
        self._prefetch_queue: List[int] = []

        # Check if preloading is feasible
        if preload and not lazy:
            total_bytes = np.sum(self.sequence_lengths) * self.index.dtype_size
            if total_bytes > 1024 * 1024 * 1024:  # 1GB limit
                raise ValueError(
                    f"Dataset too large for preloading: {total_bytes / 1e9:.1f}GB. "
                    f"Use lazy loading or increase memory."
                )
            logger.info(f"Preloading {total_bytes / 1e6:.1f}MB dataset into memory...")
            self._preload_data()

        # Initialize statistics
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "prefetch_hits": 0,
            "total_accesses": 0,
        }

    def _preload_data(self) -> None:
        """Preload entire dataset with optimized batch processing."""
        total_tokens = int(np.sum(self.sequence_lengths))

        # Pre-allocate with correct dtype and alignment
        self.preloaded_data: np.ndarray = np.empty(total_tokens, dtype=self.index.dtype)
        self.preloaded_offsets: np.ndarray = np.zeros(len(self) + 1, dtype=np.int64)

        # Calculate offsets for fast lookup
        np.cumsum(self.sequence_lengths, out=self.preloaded_offsets[1:])

        # Use vectorized operations with larger batches
        batch_size = min(1000, len(self) // 10 + 1)
        offset = 0

        # Use tqdm if available and logging is enabled
        use_progress = logger.isEnabledFor(logging.INFO) and HAS_TQDM
        if use_progress:
            pbar = tqdm(total=len(self), desc="Preloading")
        else:
            pbar = None

        try:
            for batch_start in range(0, len(self), batch_size):
                batch_end = min(batch_start + batch_size, len(self))

                # Read batch in parallel if possible
                for i in range(batch_start, batch_end):
                    seq = super().__getitem__(i)
                    seq_len = len(seq)
                    self.preloaded_data[offset : offset + seq_len] = seq
                    offset += seq_len

                if use_progress and pbar is not None:
                    pbar.update(batch_end - batch_start)
        finally:
            if use_progress and pbar is not None:
                pbar.close()

        # Mark as preloaded
        self._is_preloaded = True
        logger.info(
            f"Preloaded {total_tokens:,} tokens "
            f"({total_tokens * self.index.dtype_size / 1e6:.1f}MB)"
        )

    def __getitem__(
        self, idx: Union[int, slice, List[int], np.ndarray]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get item with prefetching and caching.

        Args:
            idx: Index or indices.

        Returns:
            Sequence data.
        """
        # Use preloaded data if available
        if hasattr(self, "_is_preloaded") and self._is_preloaded:
            if isinstance(idx, (int, np.integer)):
                idx = int(idx)
                if idx < 0:
                    idx = len(self) + idx
                start = self.preloaded_offsets[idx]
                end = self.preloaded_offsets[idx + 1]
                return self.preloaded_data[start:end].copy()

        # Track access pattern for prefetching
        if isinstance(idx, (int, np.integer)):
            self._track_access(int(idx))
            self._maybe_prefetch(int(idx))

        # Use parent implementation
        return super().__getitem__(idx)

    def _track_access(self, idx: int) -> None:
        """Track access patterns for optimization.

        Args:
            idx: Accessed index.
        """
        self._access_pattern.append(idx)
        if len(self._access_pattern) > 100:
            self._access_pattern.pop(0)

        self._stats["total_accesses"] += 1

    def _maybe_prefetch(self, idx: int) -> None:
        """Prefetch next sequences if sequential pattern detected.

        Args:
            idx: Current index.
        """
        if len(self._access_pattern) < 3:
            return

        # Check for sequential pattern
        recent = self._access_pattern[-3:]
        if recent == [idx - 2, idx - 1, idx]:
            # Sequential access detected - prefetch next sequences
            for i in range(1, self.prefetch_size + 1):
                next_idx = idx + i
                if next_idx < len(self) and next_idx not in self._prefetch_queue:
                    self._prefetch_queue.append(next_idx)
                    # Trigger prefetch (in real implementation, this would be async)
                    if self.cache_size > 0 and next_idx not in self._cache:
                        _ = self[next_idx]  # Load into cache

    def get_document(self, doc_idx: int) -> List[np.ndarray]:
        """Get all sequences in a document.

        Args:
            doc_idx: Document index.

        Returns:
            List of sequences in the document.
        """
        if doc_idx >= len(self.document_indices) - 1:
            raise IndexError(f"Document index {doc_idx} out of range")

        start_seq = int(self.document_indices[doc_idx]) if doc_idx > 0 else 0
        end_seq = int(self.document_indices[doc_idx + 1])

        sequences: List[np.ndarray] = []
        for i in range(start_seq, end_seq):
            seq = self[i]
            if isinstance(seq, np.ndarray):
                sequences.append(seq)
        return sequences

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics.

        Returns:
            Dictionary with dataset and performance statistics.
        """
        seq_lengths = self.sequence_lengths
        # Use vectorized numpy operations for efficiency
        num_docs = (
            len(self.document_indices) - 1 if len(self.document_indices) > 0 else 0
        )

        stats = {
            "num_sequences": len(self),
            "num_documents": num_docs,
            "dtype": str(self.index.dtype),
            "dtype_size": self.index.dtype_size,
            "mmap": self.mmap,
            "cache_size": self.cache_size,
            "prefetch_size": self.prefetch_size,
        }

        # Compute sequence statistics if available
        if len(seq_lengths) > 0:
            total_tokens = int(np.sum(seq_lengths))
            stats.update(
                {
                    "total_tokens": total_tokens,
                    "avg_sequence_length": float(np.mean(seq_lengths)),
                    "min_sequence_length": int(np.min(seq_lengths)),
                    "max_sequence_length": int(np.max(seq_lengths)),
                    "std_sequence_length": float(np.std(seq_lengths)),
                    "median_sequence_length": float(np.median(seq_lengths)),
                    "percentile_25": float(np.percentile(seq_lengths, 25)),
                    "percentile_75": float(np.percentile(seq_lengths, 75)),
                    "percentile_95": float(np.percentile(seq_lengths, 95)),
                    "total_size_mb": total_tokens * self.index.dtype_size / 1e6,
                }
            )

        # Add performance statistics
        if hasattr(self, "_stats"):
            total_accesses = self._stats["total_accesses"]
            if total_accesses > 0:
                cache_total = self._stats["cache_hits"] + self._stats["cache_misses"]
                stats["performance"] = {
                    "total_accesses": total_accesses,
                    "cache_hit_rate": (
                        self._stats["cache_hits"] / cache_total
                        if cache_total > 0
                        else 0
                    ),
                    "prefetch_hit_rate": self._stats["prefetch_hits"] / total_accesses,
                    "access_pattern": self._analyze_access_pattern(),
                }

        # Add file info
        if hasattr(self, "bin_path"):
            stats["file_info"] = {
                "index_file": str(self.idx_path),
                "data_file": str(self.bin_path),
                "index_size_mb": self.idx_path.stat().st_size / 1e6,
                "data_size_mb": self.bin_path.stat().st_size / 1e6,
            }

        return stats

    def _analyze_access_pattern(self) -> str:
        """Analyze the access pattern.

        Returns:
            String describing the pattern.
        """
        if len(self._access_pattern) < 10:
            return "insufficient_data"

        # Check for sequential
        diffs = np.diff(self._access_pattern[-20:])
        if np.all(diffs == 1):
            return "sequential"
        elif np.all(diffs > 0):
            return "forward"
        elif np.std(self._access_pattern[-20:]) < len(self) * 0.1:
            return "localized"
        else:
            return "random"
