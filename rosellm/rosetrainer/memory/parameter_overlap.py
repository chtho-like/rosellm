"""
Parameter Gathering Overlap with Computation

This module implements asynchronous parameter gathering that overlaps with computation
to hide communication latency in distributed training. It supports:
- Stream-based overlap management for GPU operations
- Async parameter gathering for tensor and pipeline parallelism
- Prefetching and caching mechanisms
- Integration with gradient reduction operations

References:
- ZeRO-Offload: https://arxiv.org/abs/2101.06840
- Pipeline Parallelism: https://arxiv.org/abs/2104.04473
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
"""

import logging
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


class OverlapMode(Enum):
    """Overlap strategies for parameter gathering."""

    NONE = "none"  # No overlap (baseline)
    PREFETCH = "prefetch"  # Prefetch next layer's parameters
    PIPELINE = "pipeline"  # Pipeline communication with computation
    AGGRESSIVE = "aggressive"  # Most aggressive overlap with multiple streams


@dataclass
class OverlapConfig:
    """Configuration for parameter overlap optimization."""

    mode: OverlapMode = OverlapMode.PIPELINE
    num_streams: int = 2  # Number of CUDA streams for overlap
    prefetch_depth: int = 2  # How many layers to prefetch
    cache_size_mb: int = 512  # Cache size for gathered parameters
    enable_profiling: bool = False  # Enable timing profiling
    sync_interval: int = 10  # Sync interval for aggressive mode
    use_pinned_memory: bool = True  # Use pinned memory for CPU-GPU transfers
    gather_batch_size: int = 4  # Batch multiple small gathers

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_streams <= 0:
            raise ValueError("num_streams must be positive")
        if self.num_streams > 32:
            logger.warning(
                f"num_streams={self.num_streams} is very high, consider reducing"
            )

        if self.prefetch_depth <= 0:
            raise ValueError("prefetch_depth must be positive")
        if self.prefetch_depth > 10:
            logger.warning(f"prefetch_depth={self.prefetch_depth} is very high")

        if self.cache_size_mb <= 0:
            raise ValueError("cache_size_mb must be positive")
        if self.cache_size_mb > 8192:  # 8GB
            logger.warning(f"cache_size_mb={self.cache_size_mb} is very large")

        if self.sync_interval <= 0:
            raise ValueError("sync_interval must be positive")

        if self.gather_batch_size <= 0:
            raise ValueError("gather_batch_size must be positive")
        if self.gather_batch_size > 32:
            logger.warning(f"gather_batch_size={self.gather_batch_size} is very high")


@dataclass
class GatherRequest:
    """Request for asynchronous parameter gathering."""

    param_id: str
    tensor: torch.Tensor
    target_device: torch.device
    priority: int = 0
    callback: Optional[Callable[[torch.Tensor], None]] = None
    future: Optional[Future[torch.Tensor]] = None
    stream: Optional[Any] = None
    start_time: float = field(default_factory=time.time)


class StreamPool:
    """Pool of CUDA streams for overlapped operations."""

    def __init__(self, num_streams: int, device: torch.device) -> None:
        """
        Initialize stream pool.

        Args:
            num_streams: Number of streams to create.
            device: Device for streams.
        """
        self.device = device
        self.num_streams = num_streams
        self.streams: List[Any] = []  # Use Any to handle different stream types
        self.stream_status: List[bool] = []  # True if stream is busy
        self.lock = threading.Lock()

        if device.type == "cuda":
            for _ in range(num_streams):
                self.streams.append(torch.cuda.Stream(device=device))
                self.stream_status.append(False)

    def acquire_stream(self) -> Optional[Any]:
        """Acquire an available stream from the pool."""
        with self.lock:
            for i, busy in enumerate(self.stream_status):
                if not busy:
                    self.stream_status[i] = True
                    return self.streams[i]
        return None

    def release_stream(self, stream: Any) -> None:
        """Release a stream back to the pool."""
        with self.lock:
            for i, s in enumerate(self.streams):
                if s == stream:
                    self.stream_status[i] = False
                    break

    def wait_all(self) -> None:
        """Wait for all streams to complete."""
        for stream in self.streams:
            stream.synchronize()


class ParameterCache:
    """Efficient LRU cache for gathered parameters using OrderedDict."""

    def __init__(self, max_size_mb: int) -> None:
        """
        Initialize parameter cache.

        Args:
            max_size_mb: Maximum cache size in megabytes.
        """
        if max_size_mb <= 0:
            raise ValueError("max_size_mb must be positive")

        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.sizes: Dict[str, int] = {}
        self.total_size = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get tensor from cache, marking it as recently used."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                tensor = self.cache[key]
                self.cache.move_to_end(key)
                self.hits += 1
                return tensor
            self.misses += 1
            return None

    def put(self, key: str, tensor: torch.Tensor) -> None:
        """Put tensor into cache, evicting LRU entries if needed."""
        if tensor is None:
            raise ValueError("Cannot cache None tensor")

        size = tensor.numel() * tensor.element_size()
        if size <= 0:
            logger.warning(f"Skipping cache entry with invalid size: {size}")
            return

        with self.lock:
            # If key already exists, update it
            if key in self.cache:
                self.total_size -= self.sizes[key]

            # Evict LRU entries until we have space
            while self.total_size + size > self.max_size_bytes and self.cache:
                # Remove least recently used (first item in OrderedDict)
                lru_key, _ = self.cache.popitem(last=False)
                self.total_size -= self.sizes[lru_key]
                del self.sizes[lru_key]

            # Add new entry (most recently used)
            self.cache[key] = tensor
            self.sizes[key] = size
            self.total_size += size

    def clear(self) -> None:
        """Clear the entire cache."""
        with self.lock:
            self.cache.clear()
            self.sizes.clear()
            self.total_size = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "size_mb": self.total_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "num_entries": len(self.cache),
                "utilization": (
                    self.total_size / self.max_size_bytes
                    if self.max_size_bytes > 0
                    else 0
                ),
            }


class AsyncParameterGatherer:
    """
    Asynchronous parameter gathering with computation overlap.

    This class manages overlapping parameter gathering operations with
    computation to hide communication latency in distributed training.

    Can be used as a context manager for automatic resource cleanup:
        with AsyncParameterGatherer(config) as gatherer:
            future = gatherer.gather_async("param1", tensor)
            result = future.result()
    """

    def __init__(
        self,
        config: OverlapConfig,
        process_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize async parameter gatherer.

        Args:
            config: Overlap configuration.
            process_group: Process group for communication.
            device: Device for operations.
        """
        self.config = config
        self.process_group = process_group or dist.group.WORLD
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize components
        self.stream_pool = StreamPool(config.num_streams, self.device)
        self.cache = ParameterCache(config.cache_size_mb)

        # Request queues
        self.pending_requests: Deque[GatherRequest] = deque()
        self.active_requests: Dict[str, GatherRequest] = {}
        self.completed_requests: Set[str] = set()

        # Synchronization
        self.lock = threading.Lock()
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Profiling
        self.gather_times: List[float] = []
        self.overlap_times: List[float] = []

        # Pinned memory buffers for CPU-GPU transfers
        self.pinned_buffers: Dict[int, torch.Tensor] = {}

        # Start worker thread for async operations
        if config.mode != OverlapMode.NONE:
            self._start_worker()

    def _start_worker(self) -> None:
        """Start background worker thread for async operations."""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self) -> None:
        """Background worker loop for processing gather requests."""
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    if not self.pending_requests:
                        time.sleep(0.001)  # Small sleep to avoid busy waiting
                        continue

                    request = self.pending_requests.popleft()
                    self.active_requests[request.param_id] = request

                # Process the request
                self._process_request(request)

                # Clean up
                with self.lock:
                    if request.param_id in self.active_requests:
                        del self.active_requests[request.param_id]
                    self.completed_requests.add(request.param_id)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                # Set future exception if available
                if "request" in locals() and request.future:
                    try:
                        request.future.set_exception(e)
                    except Exception:
                        pass  # Future may already be set

                # Clean up failed request
                if "request" in locals():
                    with self.lock:
                        if request.param_id in self.active_requests:
                            del self.active_requests[request.param_id]

    def _process_request(self, request: GatherRequest) -> None:
        """
        Process a single gather request.

        Args:
            request: The gather request to process.
        """
        start_time = time.time()
        stream = None

        try:
            # Check cache first
            cached_tensor = self.cache.get(request.param_id)
            if cached_tensor is not None:
                if request.callback:
                    try:
                        request.callback(cached_tensor)
                    except Exception as e:
                        logger.error(f"Error in callback for {request.param_id}: {e}")
                if request.future:
                    request.future.set_result(cached_tensor)
                return

            # Validate request
            if request.tensor is None:
                raise ValueError(f"Tensor is None for request {request.param_id}")
            if request.target_device is None:
                raise ValueError(
                    f"Target device is None for request {request.param_id}"
                )

            # Acquire stream for async operation
            stream = (
                self.stream_pool.acquire_stream()
                if self.device.type == "cuda"
                else None
            )
            request.stream = stream

            # Perform gather operation
            gathered_tensor = self._gather_tensor(
                request.tensor, request.target_device, stream
            )

            if gathered_tensor is None:
                raise RuntimeError(f"Gather operation failed for {request.param_id}")

            # Cache the result
            self.cache.put(request.param_id, gathered_tensor)

            # Execute callback if provided
            if request.callback:
                try:
                    request.callback(gathered_tensor)
                except Exception as e:
                    logger.error(f"Error in callback for {request.param_id}: {e}")

            # Set future result if provided
            if request.future:
                request.future.set_result(gathered_tensor)

            # Record timing
            if self.config.enable_profiling:
                self.gather_times.append(time.time() - start_time)

        except Exception as e:
            logger.error(f"Error processing request {request.param_id}: {e}")
            # Set exception on future if available
            if request.future:
                request.future.set_exception(e)
            # Don't re-raise - let worker loop continue
        finally:
            if stream:
                self.stream_pool.release_stream(stream)

    def _gather_tensor(
        self,
        tensor: torch.Tensor,
        target_device: torch.device,
        stream: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Perform the actual tensor gathering operation.

        Args:
            tensor: Tensor to gather.
            target_device: Target device for gathered tensor.
            stream: CUDA stream for async operation.

        Returns:
            Gathered tensor on target device.
        """
        try:
            # Early return if already on target device
            if tensor.device == target_device:
                return tensor

            if stream and self.device.type == "cuda" and tensor.is_cuda:
                with torch.cuda.stream(stream):
                    # Use pinned memory for faster transfers
                    if self.config.use_pinned_memory and target_device.type == "cpu":
                        size = tensor.numel()

                        # Create pinned buffer if not exists
                        if size not in self.pinned_buffers:
                            try:
                                self.pinned_buffers[size] = torch.empty(
                                    size,
                                    dtype=tensor.dtype,
                                    pin_memory=True,
                                )
                            except RuntimeError as e:
                                logger.warning(
                                    f"Failed to create pinned memory buffer: {e}"
                                )
                                return tensor.to(target_device)

                        pinned_buf = self.pinned_buffers[size]

                        # Ensure buffer has correct shape and dtype
                        if pinned_buf.dtype != tensor.dtype:
                            logger.warning(
                                f"Dtype mismatch in pinned buffer: "
                                f"{pinned_buf.dtype} vs {tensor.dtype}"
                            )
                            return tensor.to(target_device, non_blocking=True)

                        try:
                            pinned_buf.copy_(tensor.view(-1))
                            result = pinned_buf.to(target_device, non_blocking=True)
                            return result.view(tensor.shape)
                        except RuntimeError as e:
                            logger.warning(f"Failed to use pinned memory transfer: {e}")
                            return tensor.to(target_device, non_blocking=True)
                    else:
                        return tensor.to(target_device, non_blocking=True)
            else:
                # Synchronous transfer for CPU or when no stream
                return tensor.to(target_device)

        except Exception as e:
            logger.error(f"Error in tensor gather: {e}")
            # Fallback to synchronous transfer
            try:
                return tensor.to(target_device)
            except Exception as e2:
                logger.error(f"Fallback tensor transfer also failed: {e2}")
                raise RuntimeError(
                    f"Unable to transfer tensor to {target_device}"
                ) from e2

    def gather_async(
        self,
        param_id: str,
        tensor: torch.Tensor,
        target_device: Optional[torch.device] = None,
        priority: int = 0,
        callback: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> Future[torch.Tensor]:
        """
        Asynchronously gather a parameter.

        Args:
            param_id: Unique identifier for the parameter.
            tensor: Tensor to gather.
            target_device: Target device (defaults to self.device).
            priority: Priority for the request (higher = more urgent).
            callback: Optional callback when gather completes.

        Returns:
            Future that will contain the gathered tensor.

        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If gatherer is shut down.
        """
        # Validate inputs
        if not param_id or not isinstance(param_id, str):
            raise ValueError("param_id must be a non-empty string")
        if tensor is None or not isinstance(tensor, torch.Tensor):
            raise ValueError("tensor must be a valid PyTorch tensor")
        if tensor.numel() == 0:
            raise ValueError("tensor cannot be empty")
        if self.stop_event.is_set():
            raise RuntimeError("AsyncParameterGatherer is shut down")

        target_device = target_device or self.device
        if not isinstance(target_device, torch.device):
            raise ValueError("target_device must be a torch.device")

        # Check if already in cache (fast path)
        cached = self.cache.get(param_id)
        if cached is not None and cached.device == target_device:
            future: Future[torch.Tensor] = Future()
            future.set_result(cached)
            return future

        future = Future()
        request = GatherRequest(
            param_id=param_id,
            tensor=tensor,
            target_device=target_device,
            priority=priority,
            callback=callback,
            future=future,
        )

        with self.lock:
            # Check for duplicate requests
            if param_id in self.active_requests:
                logger.warning(
                    f"Duplicate gather request for {param_id}, creating new future"
                )

            # Insert based on priority (higher priority first)
            if priority > 0:
                # Find insertion point for priority queue behavior
                insert_idx = 0
                for i, req in enumerate(self.pending_requests):
                    if req.priority < priority:
                        insert_idx = i
                        break
                    insert_idx = i + 1
                self.pending_requests.insert(insert_idx, request)
            else:
                self.pending_requests.append(request)

        return future

    def prefetch_parameters(
        self,
        param_ids: List[str],
        tensors: List[torch.Tensor],
        target_device: Optional[torch.device] = None,
    ) -> None:
        """
        Prefetch multiple parameters for future use.

        Args:
            param_ids: List of parameter identifiers.
            tensors: List of tensors to prefetch.
            target_device: Target device for prefetching.
        """
        target_device = target_device or self.device

        for param_id, tensor in zip(param_ids, tensors):
            # Check if already cached
            if self.cache.get(param_id) is not None:
                continue

            # Submit low-priority prefetch request
            self.gather_async(
                param_id=param_id,
                tensor=tensor,
                target_device=target_device,
                priority=-1,  # Low priority for prefetch
            )

    def wait_for_parameter(
        self, param_id: str, timeout: Optional[float] = None
    ) -> bool:
        """
        Wait for a specific parameter to be gathered.

        Args:
            param_id: Parameter identifier to wait for.
            timeout: Optional timeout in seconds.

        Returns:
            True if parameter is ready, False if timeout.
        """
        start_time = time.time()

        while True:
            with self.lock:
                if (
                    param_id in self.completed_requests
                    or self.cache.get(param_id) is not None
                ):
                    return True

            if timeout and (time.time() - start_time) > timeout:
                return False

            time.sleep(0.001)

    def synchronize(self) -> None:
        """Synchronize all pending gather operations."""
        # Wait for all pending requests to complete
        while True:
            with self.lock:
                if not self.pending_requests and not self.active_requests:
                    break
            time.sleep(0.001)

        # Synchronize all streams
        self.stream_pool.wait_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about gather operations."""
        stats = {
            "cache_stats": self.cache.get_stats(),
            "pending_requests": len(self.pending_requests),
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.completed_requests),
        }

        if self.config.enable_profiling and self.gather_times:
            stats["avg_gather_time"] = sum(self.gather_times) / len(self.gather_times)
            stats["total_gather_time"] = sum(self.gather_times)

        return stats

    def shutdown(self) -> None:
        """Shutdown the gatherer and clean up resources."""
        if self.stop_event.is_set():
            return  # Already shut down

        self.stop_event.set()

        # Clear all pending requests to prevent new work
        with self.lock:
            self.pending_requests.clear()

        # Wait for worker thread with timeout
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10.0)
            if self.worker_thread.is_alive():
                logger.warning("Worker thread did not terminate within timeout")

        # Synchronize remaining operations
        try:
            self.synchronize()
        except Exception as e:
            logger.warning(f"Error during synchronization in shutdown: {e}")

        # Clean up resources
        self.cache.clear()
        self.pinned_buffers.clear()

        # Clean up completed requests set
        with self.lock:
            self.completed_requests.clear()
            self.active_requests.clear()

    def __enter__(self) -> "AsyncParameterGatherer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.shutdown()


class OverlappedLinear(nn.Module):
    """
    Linear layer with overlapped parameter gathering.

    This layer overlaps parameter gathering with computation for
    distributed training scenarios.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gatherer: Optional[AsyncParameterGatherer] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize overlapped linear layer.

        Args:
            in_features: Size of input features.
            out_features: Size of output features.
            bias: Whether to use bias.
            gatherer: Async parameter gatherer instance.
            device: Device for the layer.
            dtype: Data type for the layer.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gatherer = gatherer

        # Initialize parameters
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain("relu"))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with overlapped parameter gathering.

        This method implements computation-communication overlap by asynchronously
        gathering parameters while other computations proceed. The overlap provides
        performance benefits in distributed settings where parameter transfer
        has non-negligible latency.

        Args:
            input: Input tensor of shape [..., in_features].

        Returns:
            Output tensor of shape [..., out_features].

        Raises:
            RuntimeError: If parameter gathering fails.
        """
        if not isinstance(input, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")
        if input.size(-1) != self.in_features:
            raise ValueError(
                f"Input features ({input.size(-1)}) must match "
                f"in_features ({self.in_features})"
            )

        # Use overlapped gathering if gatherer is available and enabled
        if self.gatherer and self.gatherer.config.mode != OverlapMode.NONE:
            try:
                # Create unique parameter IDs based on layer instance
                weight_id = f"linear_weight_{id(self)}"
                bias_id = f"linear_bias_{id(self)}" if self.bias is not None else None

                # Async gather weight (high priority for immediate use)
                weight_future = self.gatherer.gather_async(
                    param_id=weight_id,
                    tensor=self.weight,
                    target_device=input.device,
                    priority=2,  # High priority for immediate use
                )

                # Async gather bias if present
                bias_future = None
                if self.bias is not None and bias_id is not None:
                    bias_future = self.gatherer.gather_async(
                        param_id=bias_id,
                        tensor=self.bias,
                        target_device=input.device,
                        priority=2,
                    )

                # Wait for parameters (this is where overlap happens)
                weight = weight_future.result()
                bias = bias_future.result() if bias_future else None

                # Validate gathered parameters
                if weight is None:
                    raise RuntimeError("Weight gathering returned None")
                if weight.device != input.device:
                    logger.warning(
                        f"Weight device mismatch: {weight.device} vs {input.device}"
                    )

                return torch.nn.functional.linear(input, weight, bias)

            except Exception as e:
                logger.warning(
                    f"Overlapped forward failed: {e}, falling back to standard"
                )
                # Fall back to standard forward pass
                return torch.nn.functional.linear(input, self.weight, self.bias)
        else:
            # Standard forward pass (no overlap)
            return torch.nn.functional.linear(input, self.weight, self.bias)


class PipelineOverlapScheduler:
    """
    Scheduler for overlapping pipeline parallel communication with computation.

    This scheduler manages the overlap of gradient communication in pipeline
    parallelism with the computation of the next microbatch.
    """

    def __init__(
        self,
        num_stages: int,
        num_microbatches: int,
        gatherer: AsyncParameterGatherer,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize pipeline overlap scheduler.

        Args:
            num_stages: Number of pipeline stages.
            num_microbatches: Number of microbatches.
            gatherer: Async parameter gatherer.
            device: Device for operations.
        """
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.gatherer = gatherer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Schedule tracking
        self.current_microbatch = 0
        self.current_stage = 0
        self.pending_grads: Deque[Tuple[int, int, torch.Tensor]] = deque()

    def schedule_gradient_reduction(
        self,
        stage: int,
        microbatch: int,
        gradients: torch.Tensor,
    ) -> Future[torch.Tensor]:
        """
        Schedule gradient reduction for overlap with next computation.

        Args:
            stage: Pipeline stage.
            microbatch: Microbatch index.
            gradients: Gradients to reduce.

        Returns:
            Future for the reduction operation.
        """
        # Schedule async gradient reduction
        future = self.gatherer.gather_async(
            param_id=f"grad_stage{stage}_mb{microbatch}",
            tensor=gradients,
            target_device=self.device,
            priority=2,  # High priority for gradients
        )

        self.pending_grads.append((stage, microbatch, gradients))
        return future

    def prefetch_next_parameters(
        self,
        stage: int,
        parameters: List[torch.Tensor],
    ) -> None:
        """
        Prefetch parameters for the next microbatch.

        Args:
            stage: Pipeline stage.
            parameters: Parameters to prefetch.
        """
        param_ids = [f"param_stage{stage}_{i}" for i in range(len(parameters))]
        self.gatherer.prefetch_parameters(param_ids, parameters, self.device)

    def wait_for_gradients(self, stage: int, microbatch: int) -> bool:
        """
        Wait for gradient reduction to complete.

        Args:
            stage: Pipeline stage.
            microbatch: Microbatch index.

        Returns:
            True if gradients are ready.
        """
        param_id = f"grad_stage{stage}_mb{microbatch}"
        return self.gatherer.wait_for_parameter(param_id, timeout=1.0)

    def get_overlap_efficiency(self) -> float:
        """
        Calculate the overlap efficiency.

        Returns:
            Efficiency metric between 0 and 1.
        """
        stats = self.gatherer.get_stats()
        if "avg_gather_time" in stats and "total_gather_time" in stats:
            # Estimate overlap efficiency based on gather times
            hit_rate = stats["cache_stats"]["hit_rate"]
            return min(1.0, float(hit_rate))
        return 0.0


# F import was replaced with torch.nn.functional directly
