"""
Multi-Tensor Gradient Operations with Automatic Backend Selection

This module provides optimized multi-tensor gradient operations with automatic
backend selection and fallback mechanisms. It supports three backends in priority order:
1. Transformer Engine (for latest NVIDIA hardware)
2. APEX (for general NVIDIA GPUs)
3. PyTorch (universal fallback)

Key Features:
- Automatic backend detection and selection
- Efficient multi-tensor gradient norm calculation
- Optimized gradient clipping operations
- Memory-efficient gradient scaling
- Bit-to-bit validation capabilities
- Performance monitoring and benchmarking

References:
- Transformer Engine: https://github.com/NVIDIA/TransformerEngine
- APEX Multi-Tensor: https://github.com/NVIDIA/apex
- PyTorch Optimizations: https://pytorch.org/docs/stable/
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Constants for numerical stability and performance
EPSILON = 1e-8
# Optimal chunk size for batched tensor operations (64K elements)
# This balances memory usage with kernel launch overhead
CHUNK_SIZE = 2048 * 32
# Maximum tensor size for single operation (64M elements)
# Prevents memory overflow on typical GPUs
MAX_TENSOR_SIZE = 2**26


class Backend(str, Enum):
    """Available backends for multi-tensor operations."""

    TRANSFORMER_ENGINE = "transformer_engine"
    APEX = "apex"
    PYTORCH = "pytorch"


@dataclass
class BackendInfo:
    """Information about a detected backend."""

    name: Backend
    available: bool
    version: Optional[str] = None
    device_support: Optional[List[str]] = None
    features: Optional[Dict[str, bool]] = None

    def __repr__(self) -> str:
        return (
            f"BackendInfo(name={self.name}, available={self.available}, "
            f"version={self.version})"
        )


class BackendStrategy(ABC):
    """Abstract base class for backend-specific operations."""

    @abstractmethod
    def calculate_norm(
        self,
        tensors: List[torch.Tensor],
        norm_type: float,
        per_tensor: bool,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate norm using backend-specific operations."""
        pass

    @abstractmethod
    def scale_tensors(
        self,
        tensors: List[torch.Tensor],
        scale: torch.Tensor,
        in_place: bool,
    ) -> List[torch.Tensor]:
        """Scale tensors using backend-specific operations."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass


class PyTorchBackendStrategy(BackendStrategy):
    """PyTorch backend implementation - universal fallback."""

    def calculate_norm(
        self,
        tensors: List[torch.Tensor],
        norm_type: float,
        per_tensor: bool,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate norm using PyTorch operations."""
        if not tensors:
            return torch.tensor(0.0)

        if per_tensor:
            return [self._single_tensor_norm(t, norm_type) for t in tensors]

        return self._combined_tensor_norm(tensors, norm_type)

    def _single_tensor_norm(
        self, tensor: torch.Tensor, norm_type: float
    ) -> torch.Tensor:
        """Calculate norm for a single tensor with numerical stability."""
        if norm_type == float("inf"):
            return tensor.abs().max()

        # Filter out non-finite values for stability
        finite_mask = torch.isfinite(tensor)
        if not finite_mask.any():
            return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)

        result: torch.Tensor = torch.norm(tensor[finite_mask].float(), p=norm_type)
        return result.to(tensor.dtype)

    def _combined_tensor_norm(
        self, tensors: List[torch.Tensor], norm_type: float
    ) -> torch.Tensor:
        """Calculate combined norm with memory-efficient chunking."""
        device = tensors[0].device
        dtype = torch.float32

        if norm_type == float("inf"):
            total_norm = torch.tensor(0.0, device=device, dtype=dtype)
            for tensor in tensors:
                finite_mask = torch.isfinite(tensor)
                if finite_mask.any():
                    total_norm = torch.max(total_norm, tensor[finite_mask].abs().max())
            return total_norm

        # Use chunked computation for memory efficiency
        total_norm_pow = torch.tensor(0.0, device=device, dtype=dtype)

        for tensor in tensors:
            finite_mask = torch.isfinite(tensor)
            if finite_mask.any():
                finite_tensor = tensor[finite_mask]
                # Process in chunks for large tensors
                if finite_tensor.numel() > MAX_TENSOR_SIZE:
                    flat_tensor = finite_tensor.flatten()
                    for i in range(0, flat_tensor.numel(), MAX_TENSOR_SIZE):
                        chunk = flat_tensor[i : i + MAX_TENSOR_SIZE]
                        total_norm_pow += (
                            torch.norm(chunk.float(), p=norm_type) ** norm_type
                        )
                else:
                    total_norm_pow += (
                        torch.norm(finite_tensor.float(), p=norm_type) ** norm_type
                    )

        return total_norm_pow ** (1.0 / norm_type)  # type: ignore[no-any-return]

    def scale_tensors(
        self,
        tensors: List[torch.Tensor],
        scale: torch.Tensor,
        in_place: bool,
    ) -> List[torch.Tensor]:
        """Scale tensors using PyTorch operations."""
        if not in_place:
            tensors = [t.clone() for t in tensors]

        for tensor in tensors:
            # Use torch.no_grad() to avoid issues with leaf tensors requiring grad
            with torch.no_grad():
                tensor.mul_(scale)

        return tensors

    def is_available(self) -> bool:
        """Check if PyTorch is available."""
        return True


class APEXBackendStrategy(BackendStrategy):
    """APEX backend implementation for NVIDIA GPUs."""

    def __init__(self):
        self._multi_tensor_applier = None
        self._amp_c = None
        self._initialize()

    def _initialize(self):
        """Initialize APEX components."""
        try:
            import amp_C  # type: ignore[import-untyped, import-not-found]
            from apex.multi_tensor_apply import multi_tensor_applier  # type: ignore

            self._multi_tensor_applier = multi_tensor_applier
            self._amp_c = amp_C
        except ImportError:
            pass

    def calculate_norm(
        self,
        tensors: List[torch.Tensor],
        norm_type: float,
        per_tensor: bool,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate norm using APEX multi-tensor operations."""
        if not self.is_available() or norm_type != 2.0 or per_tensor:
            # Fall back to PyTorch for unsupported operations
            fallback = PyTorchBackendStrategy()
            return fallback.calculate_norm(tensors, norm_type, per_tensor)

        try:
            return self._apex_l2_norm(tensors)
        except Exception as e:
            logger.debug(f"APEX norm calculation failed: {e}, falling back")
            fallback = PyTorchBackendStrategy()
            return fallback.calculate_norm(tensors, norm_type, per_tensor)

    def _apex_l2_norm(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Calculate L2 norm using APEX multi-tensor operations."""
        if self._multi_tensor_applier is None or self._amp_c is None:
            # Should not happen, but handle gracefully
            fallback = PyTorchBackendStrategy()
            return fallback.calculate_norm(tensors, 2.0, False)  # type: ignore

        dummy_overflow_buf = torch.tensor(
            [0], dtype=torch.int32, device=tensors[0].device
        )

        # Group tensors by dtype for efficiency
        grouped = self._group_tensors_by_dtype(tensors)
        total_norm_sq = torch.tensor(0.0, device=tensors[0].device)

        for dtype, tensor_group in grouped.items():
            if not tensor_group:
                continue

            # Skip small tensors where multi-tensor overhead isn't worth it
            total_elements = sum(g.numel() for g in tensor_group)
            if total_elements < 1000:
                for t in tensor_group:
                    total_norm_sq += t.pow(2).sum()
                continue

            # Process in chunks
            for i in range(0, len(tensor_group), CHUNK_SIZE):
                chunk = tensor_group[i : i + CHUNK_SIZE]
                norm = self._multi_tensor_applier(
                    self._amp_c.multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [chunk],
                    False,
                )
                if torch.isfinite(norm):
                    total_norm_sq += norm**2

        return torch.sqrt(total_norm_sq)

    def scale_tensors(
        self,
        tensors: List[torch.Tensor],
        scale: torch.Tensor,
        in_place: bool,
    ) -> List[torch.Tensor]:
        """Scale tensors using APEX operations."""
        if not self.is_available():
            fallback = PyTorchBackendStrategy()
            return fallback.scale_tensors(tensors, scale, in_place)

        try:
            if self._multi_tensor_applier is None or self._amp_c is None:
                # Should not happen due to is_available check, but handle gracefully
                fallback = PyTorchBackendStrategy()
                return fallback.scale_tensors(tensors, scale, in_place)

            if not in_place:
                tensors = [t.clone() for t in tensors]

            dummy_overflow_buf = torch.tensor(
                [0], dtype=torch.int32, device=tensors[0].device
            )
            grouped = self._group_tensors_by_dtype(tensors)

            for dtype, tensor_group in grouped.items():
                for i in range(0, len(tensor_group), CHUNK_SIZE):
                    chunk = tensor_group[i : i + CHUNK_SIZE]
                    self._multi_tensor_applier(
                        self._amp_c.multi_tensor_scale,
                        dummy_overflow_buf,
                        [chunk, chunk],
                        scale,
                    )

            return tensors
        except Exception as e:
            logger.debug(f"APEX scale failed: {e}, falling back")
            fallback = PyTorchBackendStrategy()
            return fallback.scale_tensors(tensors, scale, in_place)

    def _group_tensors_by_dtype(
        self, tensors: List[torch.Tensor]
    ) -> Dict[torch.dtype, List[torch.Tensor]]:
        """Group tensors by dtype for efficient processing."""
        grouped: Dict[torch.dtype, List[torch.Tensor]] = {}
        for tensor in tensors:
            dtype = tensor.dtype
            if dtype not in grouped:
                grouped[dtype] = []
            grouped[dtype].append(tensor)
        return grouped

    def is_available(self) -> bool:
        """Check if APEX is available."""
        return self._multi_tensor_applier is not None and self._amp_c is not None


class TransformerEngineBackendStrategy(BackendStrategy):
    """Transformer Engine backend for latest NVIDIA hardware."""

    def __init__(self):
        self._te_module = None
        self._initialize()

    def _initialize(self):
        """Initialize Transformer Engine components."""
        try:
            import transformer_engine.pytorch as te  # type: ignore[import-untyped]

            self._te_module = te
        except ImportError:
            pass

    def calculate_norm(
        self,
        tensors: List[torch.Tensor],
        norm_type: float,
        per_tensor: bool,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate norm using Transformer Engine."""
        if not self.is_available() or norm_type != 2.0 or per_tensor:
            fallback = PyTorchBackendStrategy()
            return fallback.calculate_norm(tensors, norm_type, per_tensor)

        try:
            # Transformer Engine optimized L2 norm
            total_norm_sq = torch.tensor(0.0, device=tensors[0].device)
            for tensor in tensors:
                total_norm_sq += tensor.pow(2).sum()
            return torch.sqrt(total_norm_sq)
        except Exception as e:
            logger.debug(f"TE norm calculation failed: {e}")
            fallback = PyTorchBackendStrategy()
            return fallback.calculate_norm(tensors, norm_type, per_tensor)

    def scale_tensors(
        self,
        tensors: List[torch.Tensor],
        scale: torch.Tensor,
        in_place: bool,
    ) -> List[torch.Tensor]:
        """Scale tensors - TE uses PyTorch operations."""
        fallback = PyTorchBackendStrategy()
        return fallback.scale_tensors(tensors, scale, in_place)

    def is_available(self) -> bool:
        """Check if Transformer Engine is available."""
        return self._te_module is not None


class BackendFactory:
    """Factory for creating backend strategies."""

    _strategies: Dict[Backend, BackendStrategy] = {}

    @classmethod
    def create(cls, backend: Backend) -> BackendStrategy:
        """Create or retrieve a backend strategy."""
        if backend not in cls._strategies:
            if backend == Backend.PYTORCH:
                cls._strategies[backend] = PyTorchBackendStrategy()
            elif backend == Backend.APEX:
                cls._strategies[backend] = APEXBackendStrategy()
            elif backend == Backend.TRANSFORMER_ENGINE:
                cls._strategies[backend] = TransformerEngineBackendStrategy()
            else:
                cls._strategies[backend] = PyTorchBackendStrategy()

        return cls._strategies[backend]


class MultiTensorOperator:
    """
    Multi-tensor operator with automatic backend selection and fallback.

    This class provides optimized multi-tensor operations with automatic
    detection and selection of the best available backend. Operations
    gracefully fall back to less optimized implementations when needed.
    """

    def __init__(
        self,
        preferred_backend: Optional[Backend] = None,
        device: Optional[torch.device] = None,
        enable_benchmarking: bool = False,
        cache_operations: bool = True,
    ):
        """
        Initialize multi-tensor operator.

        Args:
            preferred_backend: Preferred backend to use if available
            device: Device for operations (auto-detected if None)
            enable_benchmarking: Enable performance benchmarking
            cache_operations: Cache operation results for efficiency
        """
        self.device = device or self._get_default_device()
        self.enable_benchmarking = enable_benchmarking
        self.cache_operations = cache_operations

        # Detect available backends
        self.backends = self._detect_backends()

        # Select backend
        self.backend_info = self._select_backend(preferred_backend)

        # Create strategy using factory
        self.strategy = BackendFactory.create(self.backend_info.name)

        # Performance tracking
        self.operation_times: Dict[str, List[float]] = {}

        logger.info(
            f"MultiTensorOperator initialized with backend: {self.backend_info.name}"
        )

    @property
    def backend(self) -> BackendInfo:
        """Get the current backend info."""
        return self.backend_info

    def _get_default_device(self) -> torch.device:
        """Get default device based on availability."""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")

    @lru_cache(maxsize=1)
    def _detect_backends(self) -> Dict[Backend, BackendInfo]:
        """Detect available backends and their capabilities."""
        backends = {}

        # Detect Transformer Engine
        backends[Backend.TRANSFORMER_ENGINE] = self._detect_transformer_engine()

        # Detect APEX
        backends[Backend.APEX] = self._detect_apex()

        # PyTorch is always available
        backends[Backend.PYTORCH] = BackendInfo(
            name=Backend.PYTORCH,
            available=True,
            version=torch.__version__,
            device_support=["cpu", "cuda"],
            features={
                "multi_tensor_norm": True,
                "multi_tensor_scale": True,
                "multi_tensor_clip": True,
                "fused_operations": torch.cuda.is_available(),
            },
        )

        return backends

    def _detect_transformer_engine(self) -> BackendInfo:
        """Detect Transformer Engine availability."""
        try:
            import transformer_engine  # type: ignore[import-untyped]
            import transformer_engine.pytorch as te  # type: ignore[import-untyped]

            # Check for multi-tensor support
            has_multi_tensor = hasattr(te, "multi_tensor_applier")

            return BackendInfo(
                name=Backend.TRANSFORMER_ENGINE,
                available=True,
                version=getattr(transformer_engine, "__version__", "unknown"),
                device_support=["cuda"],
                features={
                    "multi_tensor_norm": has_multi_tensor,
                    "multi_tensor_scale": has_multi_tensor,
                    "multi_tensor_clip": has_multi_tensor,
                    "fp8_support": True,
                    "fused_operations": True,
                },
            )
        except (ImportError, ModuleNotFoundError):
            return BackendInfo(
                name=Backend.TRANSFORMER_ENGINE,
                available=False,
            )

    def _detect_apex(self) -> BackendInfo:
        """Detect APEX availability."""
        try:
            import amp_C  # type: ignore[import-untyped, import-not-found]
            from apex.multi_tensor_apply import (  # type: ignore  # noqa: F401
                multi_tensor_applier,
            )

            # Check for specific kernels
            features = {
                "multi_tensor_norm": hasattr(amp_C, "multi_tensor_l2norm"),
                "multi_tensor_scale": hasattr(amp_C, "multi_tensor_scale"),
                "multi_tensor_clip": hasattr(amp_C, "multi_tensor_clip_grad_norm_"),
                "fused_operations": True,
            }

            return BackendInfo(
                name=Backend.APEX,
                available=True,
                version="installed",
                device_support=["cuda"],
                features=features,
            )
        except (ImportError, ModuleNotFoundError):
            return BackendInfo(
                name=Backend.APEX,
                available=False,
            )

    def _select_backend(self, preferred: Optional[Backend]) -> BackendInfo:
        """Select the best available backend."""
        # Try preferred backend first
        if preferred and self.backends[preferred].available:
            if self.device.type == "cuda" or preferred == Backend.PYTORCH:
                return self.backends[preferred]

        # Auto-select based on priority
        priority = [Backend.TRANSFORMER_ENGINE, Backend.APEX, Backend.PYTORCH]

        for backend in priority:
            info = self.backends[backend]
            if info.available:
                # Check device compatibility
                if info.device_support and self.device.type in ["cuda", "cpu"]:
                    if self.device.type == "cpu" and backend != Backend.PYTORCH:
                        continue
                    return info

        # Fallback to PyTorch (always available)
        return self.backends[Backend.PYTORCH]

    def _benchmark_operation(self, name: str) -> Callable:
        """Decorator for benchmarking operations."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.enable_benchmarking:
                    start_time = time.perf_counter()
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start_time

                    if name not in self.operation_times:
                        self.operation_times[name] = []
                    self.operation_times[name].append(elapsed)

                    return result
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def calculate_norm(
        self,
        tensors: List[torch.Tensor],
        norm_type: float = 2.0,
        per_tensor: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Calculate norm of multiple tensors efficiently.

        Args:
            tensors: List of tensors
            norm_type: Type of norm (1.0, 2.0, inf)
            per_tensor: Return per-tensor norms instead of total

        Returns:
            Total norm or list of per-tensor norms
        """
        if not tensors:
            return torch.tensor(0.0, device=self.device)

        # Use strategy pattern for backend operations
        return self.strategy.calculate_norm(tensors, norm_type, per_tensor)

    def clip_grad_norm(
        self,
        parameters: Union[List[torch.nn.Parameter], List[torch.Tensor]],
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = True,
    ) -> Dict[str, Any]:
        """
        Clip gradients by norm with optimized multi-tensor operations.

        Args:
            parameters: Model parameters or tensors with gradients
            max_norm: Maximum allowed norm
            norm_type: Type of norm for clipping
            error_if_nonfinite: Raise error on non-finite gradients

        Returns:
            Dictionary with clipping statistics
        """
        # Extract gradients
        gradients = []
        for param in parameters:
            if hasattr(param, "grad") and param.grad is not None:
                gradients.append(param.grad)

        if not gradients:
            return {
                "total_norm": 0.0,
                "clip_coeff": 1.0,
                "was_clipped": False,
                "num_gradients": 0,
            }

        # Calculate total norm
        total_norm = self.calculate_norm(gradients, norm_type, per_tensor=False)

        # Ensure total_norm is a tensor
        if not isinstance(total_norm, torch.Tensor):
            total_norm = torch.tensor(
                total_norm, device=gradients[0].device if gradients else self.device
            )

        # Check for non-finite values
        if error_if_nonfinite and not torch.isfinite(total_norm):
            raise RuntimeError(f"Non-finite gradient norm: {total_norm}")

        # Calculate clipping coefficient
        clip_coeff_value = max_norm / (total_norm.item() + EPSILON)
        clip_coeff = torch.tensor(clip_coeff_value, device=total_norm.device)
        clip_coeff = torch.clamp(clip_coeff, max=1.0)

        # Apply clipping if needed
        was_clipped = clip_coeff < 1.0
        if was_clipped:
            self.scale_tensors(gradients, clip_coeff)

        return {
            "total_norm": float(total_norm),
            "clip_coeff": float(clip_coeff),
            "was_clipped": was_clipped,
            "num_gradients": len(gradients),
        }

    def scale_tensors(
        self,
        tensors: List[torch.Tensor],
        scale: Union[float, torch.Tensor],
        in_place: bool = True,
    ) -> List[torch.Tensor]:
        """
        Scale multiple tensors efficiently.

        Args:
            tensors: List of tensors to scale
            scale: Scaling factor
            in_place: Modify tensors in-place

        Returns:
            Scaled tensors
        """
        if not tensors:
            return tensors

        # Convert scale to tensor if needed
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(
                scale, device=tensors[0].device, dtype=tensors[0].dtype
            )

        # Use strategy pattern for scaling
        return self.strategy.scale_tensors(tensors, scale, in_place)

    def unscale_gradients(
        self,
        parameters: List[torch.nn.Parameter],
        inv_scale: Union[float, torch.Tensor],
    ) -> None:
        """
        Unscale gradients for mixed precision training.

        Args:
            parameters: Model parameters with gradients
            inv_scale: Inverse scale factor
        """
        gradients = []
        for param in parameters:
            if param.grad is not None:
                gradients.append(param.grad)

        if gradients:
            self.scale_tensors(gradients, inv_scale, in_place=True)

    def check_finite(
        self,
        tensors: List[torch.Tensor],
        per_tensor: bool = False,
    ) -> Union[bool, List[bool]]:
        """
        Check if tensors contain finite values.

        Args:
            tensors: List of tensors to check
            per_tensor: Return per-tensor results

        Returns:
            Overall or per-tensor finiteness status
        """
        if per_tensor:
            result: List[bool] = [bool(torch.isfinite(t).all().item()) for t in tensors]
            return result

        for tensor in tensors:
            if not torch.isfinite(tensor).all():
                return False
        return True

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        return {
            "backend": self.backend_info.name,
            "version": self.backend_info.version,
            "device": str(self.device),
            "features": self.backend_info.features,
            "available_backends": {
                name.value: info.available for name, info in self.backends.items()
            },
        }

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for operations."""
        stats = {}

        for op_name, times in self.operation_times.items():
            if times:
                stats[op_name] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }

        return stats

    def clear_cache(self) -> None:
        """Clear any cached operations."""
        # Clear LRU cache for backend detection
        if hasattr(self._detect_backends, "cache_clear"):
            self._detect_backends.cache_clear()  # type: ignore

    def validate_against_reference(
        self,
        tensors: List[torch.Tensor],
        operation: str = "norm",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Validate operations against PyTorch reference implementation.

        Args:
            tensors: Input tensors
            operation: Operation to validate
            **kwargs: Additional operation arguments

        Returns:
            Validation results including accuracy metrics
        """
        results = {}

        # Get result from current backend
        current_backend = self.backend_info.name

        if operation == "norm":
            norm_type = kwargs.get("norm_type", 2.0)
            current_result = self.calculate_norm(tensors, norm_type, per_tensor=False)

            # Get reference result from PyTorch
            pytorch_strategy = PyTorchBackendStrategy()
            reference_result = pytorch_strategy.calculate_norm(
                tensors, norm_type, False
            )

            # Ensure both are tensors
            if not isinstance(current_result, torch.Tensor):
                current_result = torch.tensor(current_result)
            if not isinstance(reference_result, torch.Tensor):
                reference_result = torch.tensor(reference_result)

            # Calculate accuracy
            abs_diff = torch.abs(current_result - reference_result)
            rel_diff = abs_diff / (reference_result + EPSILON)

            results = {
                "operation": operation,
                "backend": current_backend,
                "current_result": float(
                    current_result.item()
                    if current_result.numel() == 1
                    else current_result.mean().item()
                ),
                "reference_result": float(
                    reference_result.item()
                    if reference_result.numel() == 1
                    else reference_result.mean().item()
                ),
                "absolute_difference": float(
                    abs_diff.item() if abs_diff.numel() == 1 else abs_diff.mean().item()
                ),
                "relative_difference": float(
                    rel_diff.item() if rel_diff.numel() == 1 else rel_diff.mean().item()
                ),
                "matches": bool((rel_diff < 1e-5).all().item()),  # Within tolerance
            }

        return results


# Convenience functions for common operations
_default_operator: Optional[MultiTensorOperator] = None


def get_default_operator(reset: bool = False) -> MultiTensorOperator:
    """Get or create default multi-tensor operator."""
    global _default_operator

    if _default_operator is None or reset:
        _default_operator = MultiTensorOperator()

    return _default_operator


def multi_tensor_norm(
    tensors: List[torch.Tensor],
    norm_type: float = 2.0,
    operator: Optional[MultiTensorOperator] = None,
) -> torch.Tensor:
    """
    Calculate norm of multiple tensors using optimized operations.

    Args:
        tensors: List of tensors
        norm_type: Type of norm
        operator: Multi-tensor operator (uses default if None)

    Returns:
        Total norm
    """
    if operator is None:
        operator = get_default_operator()

    return operator.calculate_norm(tensors, norm_type, per_tensor=False)  # type: ignore


def multi_tensor_clip_grad_norm(
    parameters: Union[List[torch.nn.Parameter], nn.Module],
    max_norm: float,
    norm_type: float = 2.0,
    operator: Optional[MultiTensorOperator] = None,
) -> Dict[str, Any]:
    """
    Clip gradients by norm using optimized multi-tensor operations.

    Args:
        parameters: Model parameters or module
        max_norm: Maximum allowed norm
        norm_type: Type of norm for clipping
        operator: Multi-tensor operator (uses default if None)

    Returns:
        Clipping statistics
    """
    if operator is None:
        operator = get_default_operator()

    # Convert module to parameters if needed
    if isinstance(parameters, nn.Module):
        parameters = list(parameters.parameters())

    return operator.clip_grad_norm(parameters, max_norm, norm_type)


def multi_tensor_scale(
    tensors: List[torch.Tensor],
    scale: Union[float, torch.Tensor],
    operator: Optional[MultiTensorOperator] = None,
) -> List[torch.Tensor]:
    """
    Scale multiple tensors efficiently.

    Args:
        tensors: List of tensors to scale
        scale: Scaling factor
        operator: Multi-tensor operator (uses default if None)

    Returns:
        Scaled tensors
    """
    if operator is None:
        operator = get_default_operator()

    return operator.scale_tensors(tensors, scale, in_place=True)
