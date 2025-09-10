"""Data type definitions for indexed datasets."""

from enum import Enum
from typing import Optional, Type, Union

import numpy as np

# Magic header for index files (ROSEIDX + version)
INDEX_HEADER = b"ROSEIDX\x00\x00"


class DType(Enum):
    """NumPy data type enumeration for binary dataset format.

    This enum maps between integer codes and numpy dtypes for
    efficient storage and retrieval of tokenized data.
    """

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, dtype: Type[np.number]) -> int:
        """Get the integer code from a numpy dtype.

        Args:
            dtype: The numpy dtype class.

        Returns:
            Integer code corresponding to the dtype.

        Raises:
            KeyError: If dtype is not supported.
        """
        return cls[dtype.__name__].value

    @classmethod
    def dtype_from_code(cls, code: int) -> Type[np.number]:
        """Get the numpy dtype from an integer code.

        Args:
            code: The integer code.

        Returns:
            The corresponding numpy dtype class.

        Raises:
            ValueError: If code is not valid.
        """
        dtype_class: Type[np.number] = getattr(np, cls(code).name)
        return dtype_class

    @staticmethod
    def size(key: Union[int, Type[np.number]]) -> int:
        """Get the size in bytes of a dtype or code.

        Args:
            key: Either an integer code or numpy dtype class.

        Returns:
            Size of the dtype in bytes.

        Raises:
            ValueError: If key is neither a valid code nor dtype.
        """
        if isinstance(key, int):
            return int(DType.dtype_from_code(key)().itemsize)
        elif np.number in key.__mro__:
            return int(key().itemsize)
        else:
            raise ValueError(f"Invalid key type: {type(key)}")

    @staticmethod
    def optimal_dtype(vocab_size: Optional[int]) -> Type[np.number]:
        """Select optimal dtype based on vocabulary size.

        Chooses the smallest dtype that can represent all token IDs
        to minimize memory usage and I/O.

        Args:
            vocab_size: Size of the vocabulary. If None, defaults to int32.

        Returns:
            Optimal numpy dtype for the vocabulary size.
        """
        dtype: Type[np.number]
        if vocab_size is None:
            dtype = np.int32
        elif vocab_size < 256:
            dtype = np.uint8
        elif vocab_size < 65536:
            dtype = np.uint16
        else:
            dtype = np.int32
        return dtype
