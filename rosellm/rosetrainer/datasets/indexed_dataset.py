"""Memory-mapped indexed dataset for efficient data loading."""

import gc
import logging
import os
import struct
from typing import List, Tuple, Type, Union

import numpy as np
from torch.utils.data import Dataset

from .data_types import INDEX_HEADER, DType

logger = logging.getLogger(__name__)


class IndexWriter:
    """Write index files for efficient data access.

    The index file format:
    - Header: magic bytes + version
    - Metadata: dtype code, sequence count, document count
    - Arrays: sequence lengths, sequence pointers, document indices
    """

    def __init__(self, path: str, dtype: Type[np.number]):
        """Initialize index writer.

        Args:
            path: Path to the index file.
            dtype: Data type for tokens in the binary file.
        """
        self.path = path
        self.dtype = dtype
        self.file = None

    def __enter__(self):
        """Open file for writing."""
        self.file = open(self.path, "wb")
        # Write header
        self.file.write(INDEX_HEADER)
        # Write version
        self.file.write(struct.pack("<Q", 1))
        # Write dtype code
        self.file.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file."""
        if self.file:
            self.file.close()
        return None

    def write(self, sequence_lengths: List[int], document_indices: List[int]) -> None:
        """Write index data to file.

        Args:
            sequence_lengths: Length of each sequence in tokens.
            document_indices: Sequence indices marking document boundaries.
        """
        if self.file is None:
            raise RuntimeError("IndexWriter must be used within a context manager")

        # Calculate byte offsets for sequences
        sequence_pointers = self._calculate_pointers(sequence_lengths)

        # Write counts
        sequence_count = len(sequence_lengths)
        document_count = len(document_indices)
        self.file.write(struct.pack("<Q", sequence_count))
        self.file.write(struct.pack("<Q", document_count))

        # Write arrays
        self.file.write(np.array(sequence_lengths, dtype=np.int32).tobytes(order="C"))
        self.file.write(np.array(sequence_pointers, dtype=np.int64).tobytes(order="C"))
        self.file.write(np.array(document_indices, dtype=np.int64).tobytes(order="C"))

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
    """Read index files using memory mapping for efficiency."""

    def __init__(self, path: str):
        """Initialize index reader.

        Args:
            path: Path to the index file.
        """
        logger.info(f"Loading index from {path}")

        # Read header and metadata
        with open(path, "rb") as f:
            # Verify header
            header = f.read(9)
            if header != INDEX_HEADER:
                raise ValueError(f"Invalid index file header: {path}")

            # Read version
            version = struct.unpack("<Q", f.read(8))[0]
            if version != 1:
                raise ValueError(f"Unsupported index version: {version}")

            # Read dtype
            dtype_code = struct.unpack("<B", f.read(1))[0]
            self.dtype = DType.dtype_from_code(dtype_code)
            self.dtype_size = DType.size(self.dtype)

            # Read counts
            self.sequence_count = struct.unpack("<Q", f.read(8))[0]
            self.document_count = struct.unpack("<Q", f.read(8))[0]

            offset = f.tell()

        # Memory map the file for efficient access
        # Use np.memmap with optimal settings for read-only access
        self.mmap = np.memmap(path, mode="r", order="C")
        # Cache the data attribute to avoid repeated attribute access
        self._mmap_data = self.mmap.data
        self.buffer = memoryview(self._mmap_data)

        # Load arrays
        logger.info("Loading sequence lengths...")
        self.sequence_lengths: np.ndarray = np.frombuffer(
            self.buffer, dtype=np.int32, count=self.sequence_count, offset=offset
        )

        logger.info("Loading sequence pointers...")
        self.sequence_pointers: np.ndarray = np.frombuffer(
            self.buffer,
            dtype=np.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )

        logger.info("Loading document indices...")
        self.document_indices: np.ndarray = np.frombuffer(
            self.buffer,
            dtype=np.int64,
            count=self.document_count,
            offset=offset
            + self.sequence_lengths.nbytes
            + self.sequence_pointers.nbytes,
        )

        logger.info(
            f"Loaded index: {self.sequence_count} sequences, "
            f"{self.document_count} documents"
        )

    def __len__(self) -> int:
        """Return number of sequences."""
        return int(self.sequence_count)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        """Get sequence pointer and length.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of (byte_offset, sequence_length).
        """
        return self.sequence_pointers[idx], self.sequence_lengths[idx]

    def __del__(self):
        """Clean up memory map."""
        if hasattr(self, "_mmap_data"):
            del self._mmap_data
        if hasattr(self, "buffer"):
            del self.buffer
        if hasattr(self, "mmap"):
            del self.mmap
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
    """Memory-mapped binary file reader for zero-copy access."""

    def __init__(self, path: str):
        """Initialize memory-mapped reader.

        Args:
            path: Path to binary data file.
        """
        self.path = path
        # Open file to get size
        with open(path, "rb") as f:
            f.seek(0, 2)  # Seek to end
            self.file_size = f.tell()

        # Memory map the entire file
        # Use np.memmap with optimal settings for read-only access
        self.mmap = np.memmap(path, mode="r", order="C")
        # Cache the data attribute to avoid repeated attribute access
        self._mmap_data = self.mmap.data
        self.buffer = memoryview(self._mmap_data)

    def read(self, dtype: Type[np.number], count: int, offset: int) -> np.ndarray:
        """Read data using memory mapping.

        Args:
            dtype: Data type to read.
            count: Number of elements.
            offset: Byte offset.

        Returns:
            View into memory-mapped data.
        """
        return np.frombuffer(self.buffer, dtype=dtype, count=count, offset=offset)

    def __del__(self):
        """Clean up memory map."""
        if hasattr(self, "_mmap_data"):
            del self._mmap_data
        if hasattr(self, "buffer"):
            del self.buffer
        if hasattr(self, "mmap"):
            del self.mmap
            # Force garbage collection for large memory maps
            gc.collect()


class FileBinReader(BinReader):
    """File-based binary reader (fallback when mmap unavailable)."""

    def __init__(self, path: str):
        """Initialize file reader.

        Args:
            path: Path to binary data file.
        """
        self.path = path

    def read(self, dtype: Type[np.number], count: int, offset: int) -> np.ndarray:
        """Read data from file.

        Args:
            dtype: Data type to read.
            count: Number of elements.
            offset: Byte offset.

        Returns:
            NumPy array with the data.
        """
        array = np.empty(count, dtype=dtype)
        with open(self.path, "rb", buffering=0) as f:
            f.seek(offset)
            # Read bytes and convert to array
            bytes_to_read = count * dtype().itemsize
            data = f.read(bytes_to_read)
            array = np.frombuffer(data, dtype=dtype, count=count)
        return array


class IndexedDataset(Dataset):
    """Base indexed dataset interface."""

    def __init__(self, path_prefix: str, mmap: bool = True, lazy: bool = True):
        """Initialize indexed dataset.

        Args:
            path_prefix: Path without .idx/.bin extension.
            mmap: Use memory mapping for data access.
            lazy: Load data on demand vs preloading.
        """
        super().__init__()
        self.path_prefix = path_prefix
        self.mmap = mmap
        self.lazy = lazy

        # Verify files exist
        idx_path = f"{path_prefix}.idx"
        bin_path = f"{path_prefix}.bin"

        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Index file not found: {idx_path}")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Data file not found: {bin_path}")

        # Load index
        self.index = IndexReader(idx_path)

        # Initialize data reader
        self.bin_reader: Union[MMapBinReader, FileBinReader]
        if mmap:
            self.bin_reader = MMapBinReader(bin_path)
        else:
            self.bin_reader = FileBinReader(bin_path)

    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return len(self.index)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get sequence(s) at index.

        Args:
            idx: Index or slice.

        Returns:
            Single sequence or list of sequences.
        """
        if isinstance(idx, (int, np.integer)):
            # Single sequence
            pointer, length = self.index[idx]
            return self.bin_reader.read(
                dtype=self.index.dtype, count=length, offset=pointer
            )
        elif isinstance(idx, slice):
            # Multiple sequences
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Step must be 1 for slicing")

            sequences = []
            for i in range(start, stop):
                pointer, length = self.index[i]
                seq = self.bin_reader.read(
                    dtype=self.index.dtype, count=length, offset=pointer
                )
                sequences.append(seq)
            return sequences
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    @property
    def sequence_lengths(self) -> np.ndarray:
        """Get array of sequence lengths."""
        return self.index.sequence_lengths

    @property
    def document_indices(self) -> np.ndarray:
        """Get array of document boundary indices."""
        return self.index.document_indices

    @staticmethod
    def exists(path_prefix: str) -> bool:
        """Check if dataset exists at path.

        Args:
            path_prefix: Path prefix without extensions.

        Returns:
            True if both .idx and .bin files exist.
        """
        return os.path.exists(f"{path_prefix}.idx") and os.path.exists(
            f"{path_prefix}.bin"
        )


class MMapIndexedDataset(IndexedDataset):
    """Memory-mapped indexed dataset with optimizations.

    This is the main dataset class for efficient training on
    large datasets. It uses memory mapping for zero-copy access
    and provides various optimizations for training workloads.
    """

    def __init__(self, path_prefix: str, lazy: bool = True, preload: bool = False):
        """Initialize memory-mapped dataset.

        Args:
            path_prefix: Path without extensions.
            lazy: Load sequences on demand.
            preload: Preload entire dataset into memory (for small datasets).
        """
        super().__init__(path_prefix, mmap=True, lazy=lazy)

        if preload and not lazy:
            logger.info("Preloading dataset into memory...")
            self._preload_data()

    def _preload_data(self):
        """Preload entire dataset (for small datasets only)."""
        total_tokens = int(np.sum(self.sequence_lengths))
        # Pre-allocate with correct dtype to avoid conversions
        self.preloaded_data = np.empty(total_tokens, dtype=self.index.dtype)

        # Use vectorized operations where possible
        offset = 0
        batch_size = 100  # Process sequences in batches for better cache locality
        for batch_start in range(0, len(self), batch_size):
            batch_end = min(batch_start + batch_size, len(self))
            for i in range(batch_start, batch_end):
                seq = super().__getitem__(i)
                seq_len = len(seq)
                self.preloaded_data[offset : offset + seq_len] = seq
                offset += seq_len

        logger.info(f"Preloaded {total_tokens} tokens")

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

    def get_stats(self) -> dict:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics.
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
        }

        # Only compute statistics if we have sequences
        if len(seq_lengths) > 0:
            stats.update(
                {
                    "total_tokens": int(np.sum(seq_lengths)),
                    "avg_sequence_length": float(np.mean(seq_lengths)),
                    "min_sequence_length": int(np.min(seq_lengths)),
                    "max_sequence_length": int(np.max(seq_lengths)),
                    "std_sequence_length": float(np.std(seq_lengths)),
                    "median_sequence_length": float(np.median(seq_lengths)),
                }
            )

        return stats
