"""
Vocab Parallel Cross-Entropy Loss for RoseLLM

This module implements vocabulary-parallel cross-entropy loss computation
compatible with Megatron-LM's tensor parallelism. It splits the vocabulary
dimension across tensor parallel ranks to reduce memory usage and improve
scalability for large vocabulary models.

Key Features:
- Vocabulary dimension splitting across tensor parallel ranks
- Efficient gradient computation with custom autograd function
- Label smoothing support for regularization
- Memory-efficient in-place operations
- Bit-to-bit compatibility with Megatron-LM

References:
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Efficient Large-Scale Language Model Training: https://arxiv.org/abs/2104.04473
"""

import logging
from typing import Tuple

import torch
import torch.distributed as dist

from rosellm.rosetrainer.parallelism import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)

logger = logging.getLogger(__name__)


# Global cache for arange tensors to avoid repeated allocation
_ARANGE_CACHE = {}


def _get_arange_tensor(
    size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Get cached arange tensor or create new one."""
    cache_key = (size, device, dtype)
    if cache_key not in _ARANGE_CACHE:
        _ARANGE_CACHE[cache_key] = torch.arange(size, device=device, dtype=dtype)
    elif _ARANGE_CACHE[cache_key].size(0) < size:
        # Expand cache if needed
        _ARANGE_CACHE[cache_key] = torch.arange(size, device=device, dtype=dtype)

    return _ARANGE_CACHE[cache_key][:size]


class VocabParallelError(Exception):
    """Custom exception for vocabulary parallel operations."""

    pass


def _validate_tensor_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    operation_name: str = "vocab_parallel_cross_entropy",
) -> None:
    """
    Validate input tensors for vocabulary parallel operations.

    Args:
        logits: Model output logits tensor
        targets: Target indices tensor
        operation_name: Name of operation for error messages

    Raises:
        VocabParallelError: If validation fails
    """
    if not isinstance(logits, torch.Tensor):
        raise VocabParallelError(
            f"{operation_name}: logits must be a torch.Tensor, got {type(logits)}"
        )

    if not isinstance(targets, torch.Tensor):
        raise VocabParallelError(
            f"{operation_name}: targets must be a torch.Tensor, got {type(targets)}"
        )

    if logits.dim() != 3:
        raise VocabParallelError(
            f"{operation_name}: logits must be 3D tensor with shape "
            f"[seq_length, batch_size, vocab_size], got shape {logits.shape}"
        )

    if targets.dim() != 2:
        raise VocabParallelError(
            f"{operation_name}: targets must be 2D tensor with shape "
            f"[seq_length, batch_size], got shape {targets.shape}"
        )

    if logits.shape[:2] != targets.shape:
        raise VocabParallelError(
            f"{operation_name}: logits and targets shape mismatch. "
            f"logits shape: {logits.shape}, targets shape: {targets.shape}. "
            f"First two dimensions must match."
        )

    if logits.device != targets.device:
        raise VocabParallelError(
            f"{operation_name}: logits and targets must be on same device. "
            f"logits device: {logits.device}, targets device: {targets.device}"
        )

    # Check for valid target indices (non-negative integers)
    if targets.dtype not in [torch.int32, torch.int64, torch.long]:
        raise VocabParallelError(
            f"{operation_name}: targets must have integer dtype, got {targets.dtype}"
        )

    if torch.any(targets < 0):
        raise VocabParallelError(
            f"{operation_name}: targets must contain non-negative indices"
        )


class VocabUtility:
    """Utility class for vocabulary range calculations in tensor parallelism."""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank: int, world_size: int
    ) -> Tuple[int, int]:
        """
        Calculate the vocabulary range for a given tensor parallel rank.

        Args:
            per_partition_vocab_size: Vocabulary size per partition
            rank: Tensor parallel rank
            world_size: Tensor parallel world size

        Returns:
            Tuple of (start_index, end_index) for vocabulary range

        Raises:
            VocabParallelError: If parameters are invalid
        """
        if (
            not isinstance(per_partition_vocab_size, int)
            or per_partition_vocab_size <= 0
        ):
            raise VocabParallelError(
                f"per_partition_vocab_size must be positive integer, "
                f"got {per_partition_vocab_size}"
            )

        if not isinstance(rank, int) or rank < 0:
            raise VocabParallelError(f"rank must be non-negative integer, got {rank}")

        if not isinstance(world_size, int) or world_size <= 0:
            raise VocabParallelError(
                f"world_size must be positive integer, got {world_size}"
            )

        if rank >= world_size:
            raise VocabParallelError(
                f"rank ({rank}) must be less than world_size ({world_size})"
            )

        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> Tuple[int, int]:
        """
        Calculate the vocabulary range from global vocabulary size.

        Args:
            global_vocab_size: Total vocabulary size
            rank: Tensor parallel rank
            world_size: Tensor parallel world size

        Returns:
            Tuple of (start_index, end_index) for vocabulary range

        Raises:
            VocabParallelError: If parameters are invalid or vocab size not divisible
        """
        if not isinstance(global_vocab_size, int) or global_vocab_size <= 0:
            raise VocabParallelError(
                f"global_vocab_size must be positive integer, "
                f"got {global_vocab_size}"
            )

        # Ensure vocab size is divisible by world size for even partitioning
        if global_vocab_size % world_size != 0:
            raise VocabParallelError(
                f"Global vocab size ({global_vocab_size}) must be divisible by "
                f"tensor parallel size ({world_size})"
            )

        per_partition_vocab_size = global_vocab_size // world_size
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )


class VocabParallelCrossEntropy:
    """
    Core computation class for vocabulary parallel cross-entropy.

    This class provides static methods for the forward and backward
    computations needed for vocab-parallel cross-entropy loss.
    """

    @staticmethod
    def calculate_logits_max(
        vocab_parallel_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate maximum logits value for numerical stability.

        Args:
            vocab_parallel_logits: Logits tensor split across vocab dimension

        Returns:
            Tuple of (float32_logits, logits_max)
        """
        # Convert to float32 for numerical stability
        vocab_parallel_logits = vocab_parallel_logits.float()

        # Find maximum value along vocab dimension
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]

        return vocab_parallel_logits, logits_max

    @staticmethod
    def calculate_predicted_logits(
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        logits_max: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate predicted logits and intermediate values for loss computation.

        Args:
            vocab_parallel_logits: Logits tensor (already shifted by max)
            target: Target token indices
            logits_max: Maximum logits values
            vocab_start_index: Start index of vocabulary partition
            vocab_end_index: End index of vocabulary partition

        Returns:
            Tuple of (target_mask, masked_target_1d, predicted_logits,
                     sum_exp_logits, exp_logits)
        """
        # Subtract max for numerical stability
        # (create new tensor to avoid autograd issues)
        logits_shifted = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        # Create mask for targets outside this partition's vocabulary range
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)

        # Adjust target indices to partition-local indices (more efficient)
        masked_target = torch.where(
            target_mask, torch.zeros_like(target), target - vocab_start_index
        )

        # Get logits corresponding to target tokens
        partition_vocab_size = logits_shifted.size()[-1]

        # Use torch.gather for more efficient indexing
        predicted_logits = torch.gather(
            logits_shifted.view(-1, partition_vocab_size), 1, masked_target.view(-1, 1)
        ).view_as(target)

        # Zero out invalid targets (safe for autograd)
        predicted_logits = torch.where(
            target_mask, torch.zeros_like(predicted_logits), predicted_logits
        )

        # Calculate exp and sum for softmax (avoid aliasing)
        exp_logits = torch.exp(logits_shifted)
        sum_exp_logits = exp_logits.sum(dim=-1)

        return (
            target_mask,
            masked_target.view(-1),  # Equivalent to masked_target_1d
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        )

    @staticmethod
    def calculate_cross_entropy_loss(
        exp_logits: torch.Tensor,
        predicted_logits: torch.Tensor,
        sum_exp_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate cross-entropy loss from intermediate values.

        Args:
            exp_logits: Exponentiated logits
            predicted_logits: Logits for target tokens
            sum_exp_logits: Sum of exponentiated logits

        Returns:
            Tuple of (normalized_exp_logits, loss)
        """
        # Calculate negative log likelihood loss
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize exp_logits to get softmax probabilities (safe for autograd)
        normalized_exp_logits = exp_logits / sum_exp_logits.unsqueeze(dim=-1)

        return normalized_exp_logits, loss

    @staticmethod
    def apply_label_smoothing(
        loss: torch.Tensor,
        exp_logits: torch.Tensor,
        label_smoothing: float,
        vocab_size: int,
    ) -> torch.Tensor:
        """
        Apply label smoothing regularization to the loss.

        Args:
            loss: Cross-entropy loss
            exp_logits: Normalized probabilities (softmax output)
            label_smoothing: Smoothing factor (0.0 to 1.0)
            vocab_size: Total vocabulary size

        Returns:
            Smoothed loss
        """
        if label_smoothing <= 0.0 or label_smoothing >= 1.0:
            return loss

        # Calculate smoothing factor adjusted for vocabulary size
        # This follows Megatron-LM's implementation for consistency
        smoothing = label_smoothing * vocab_size / (vocab_size - 1)

        # Calculate mean log probabilities for smoothing term
        log_probs = torch.log(exp_logits)
        mean_log_probs = log_probs.mean(dim=-1)

        # Apply smoothing: (1 - smoothing) * original_loss - smoothing * mean_log_probs
        smoothed_loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        return smoothed_loss


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """
    Custom autograd function for vocabulary parallel cross-entropy.

    This implements the forward and backward passes with proper
    gradient computation for vocabulary-parallel training.
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0):
        """
        Forward pass for vocab parallel cross-entropy.

        Args:
            ctx: Context object for saving tensors
            vocab_parallel_logits: Logits split across vocab dimension
                Shape: [sequence_length, batch_size, vocab_size/tp_size]
            target: Target token indices
                Shape: [sequence_length, batch_size]
            label_smoothing: Label smoothing factor (0.0 to 1.0)

        Returns:
            Cross-entropy loss tensor

        Raises:
            VocabParallelError: If input validation fails
        """
        # Validate inputs
        _validate_tensor_inputs(
            vocab_parallel_logits, target, "vocab_parallel_cross_entropy"
        )

        if (
            not isinstance(label_smoothing, (int, float))
            or not 0.0 <= label_smoothing < 1.0
        ):
            raise VocabParallelError(
                f"label_smoothing must be float in [0.0, 1.0), "
                f"got {label_smoothing}"
            )

        # Validate distributed state (only required for multi-GPU tensor parallelism)
        if not dist.is_initialized():
            # For single-GPU/CPU case, we can proceed with fake TP settings
            tp_group = None
            tp_rank = 0
            tp_world_size = 1
        else:
            # Get tensor parallel group info
            tp_group = get_tensor_model_parallel_group()
            tp_rank = get_tensor_model_parallel_rank()
            tp_world_size = get_tensor_model_parallel_size()

            # When TP size is 1, group can be None (no parallel communication needed)
            if tp_world_size > 1 and tp_group is None:
                raise VocabParallelError(
                    "Tensor parallel group not initialized for TP size > 1. "
                    "Call rosellm.rosetrainer.parallelism.initialize_model_parallel() "
                    "first"
                )

        # Calculate maximum logits for numerical stability
        (
            vocab_parallel_logits,
            logits_max,
        ) = VocabParallelCrossEntropy.calculate_logits_max(vocab_parallel_logits)

        # All-reduce max across tensor parallel ranks (skip for TP size = 1)
        if tp_world_size > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)

        # Get vocabulary range for this partition
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        (
            vocab_start_index,
            vocab_end_index,
        ) = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, tp_rank, tp_world_size
        )

        # Calculate predicted logits and intermediate values
        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            vocab_parallel_logits,
            target,
            logits_max,
            vocab_start_index,
            vocab_end_index,
        )

        # All-reduce predicted logits and sum of exp across TP ranks
        # (skip for TP size = 1)
        if tp_world_size > 1:
            dist.all_reduce(
                predicted_logits,
                op=dist.ReduceOp.SUM,
                group=tp_group,
            )

            dist.all_reduce(
                sum_exp_logits,
                op=dist.ReduceOp.SUM,
                group=tp_group,
            )

        # Calculate cross-entropy loss
        exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
            exp_logits, predicted_logits, sum_exp_logits
        )

        # Apply label smoothing if requested
        vocab_size = partition_vocab_size * tp_world_size
        if label_smoothing > 0:
            loss = VocabParallelCrossEntropy.apply_label_smoothing(
                loss, exp_logits, label_smoothing, vocab_size
            )

        # Save tensors for backward pass
        ctx.label_smoothing = label_smoothing
        ctx.vocab_size = vocab_size
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for vocab parallel cross-entropy.

        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient of loss

        Returns:
            Tuple of gradients for (vocab_parallel_logits, target, label_smoothing)
        """
        # Retrieve saved tensors
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing = ctx.label_smoothing
        vocab_size = ctx.vocab_size

        # Initialize gradient with softmax probabilities (avoid aliasing)
        grad_input = softmax.clone()

        # Prepare for gradient computation
        partition_vocab_size = softmax.size()[-1]

        # Create update mask (1.0 for valid targets, 0.0 for masked)
        valid_mask = ~target_mask.view(-1)  # Boolean mask for valid targets

        if label_smoothing > 0:
            # Apply label smoothing to gradients
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Create gradient adjustment tensor for target positions
            target_adjustment = (1.0 - smoothing) * valid_mask.float()

            # Use scatter to efficiently update target positions
            grad_2d = grad_input.view(-1, partition_vocab_size)
            grad_2d.scatter_(
                1,
                masked_target_1d.unsqueeze(1),
                -target_adjustment.unsqueeze(1),
                reduce="add",
            )

            # Apply uniform smoothing to all positions
            average_grad = smoothing / vocab_size
            grad_input = grad_input - average_grad
        else:
            # Standard cross-entropy gradient - use scatter for efficiency
            grad_2d = grad_input.view(-1, partition_vocab_size)
            grad_2d.scatter_(
                1,
                masked_target_1d.unsqueeze(1),
                -valid_mask.float().unsqueeze(1),
                reduce="add",
            )

        # Scale by output gradient
        grad_input = grad_input * grad_output.unsqueeze(dim=-1)

        return grad_input, None, None


def vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Compute cross-entropy loss with vocabulary parallelism.

    This function splits the vocabulary dimension across tensor parallel
    ranks to reduce memory usage and improve scalability for large
    vocabulary models.

    Args:
        vocab_parallel_logits: Model output logits split across vocab dimension
            Shape: [sequence_length, batch_size, vocab_size/tp_size]
        target: Target token indices
            Shape: [sequence_length, batch_size]
        label_smoothing: Label smoothing factor (0.0 to 1.0)
            Default: 0.0 (no smoothing)

    Returns:
        Cross-entropy loss tensor
        Shape: [sequence_length, batch_size]

    Example:
        >>> # Assume tensor parallel size = 4, vocab size = 50257
        >>> logits = torch.randn(128, 32, 12565)  # 50257 // 4 ≈ 12565
        >>> targets = torch.randint(0, 50257, (128, 32))
        >>> loss = vocab_parallel_cross_entropy(logits, targets, label_smoothing=0.1)

    Note:
        This function requires proper initialization of tensor parallel groups
        via rosellm.rosetrainer.parallelism.initialize_model_parallel()

    Raises:
        VocabParallelError: If input validation fails or tensor parallel not initialized
    """
    # Input validation is handled by the autograd function
    result = _VocabParallelCrossEntropy.apply(
        vocab_parallel_logits, target, label_smoothing
    )

    # Type safety check
    if not isinstance(result, torch.Tensor):
        raise VocabParallelError(
            f"Expected torch.Tensor from autograd function, got {type(result)}"
        )

    return result


class VocabParallelCrossEntropyLoss(torch.nn.Module):
    """
    Module wrapper for vocabulary parallel cross-entropy loss.

    This provides a convenient nn.Module interface for the vocab parallel
    cross-entropy computation, suitable for integration into model training.

    Args:
        label_smoothing: Label smoothing factor (0.0 to 1.0)
        reduction: Reduction method ('none', 'mean', 'sum')

    Example:
        >>> loss_fn = VocabParallelCrossEntropyLoss(label_smoothing=0.1)
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()

        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"Label smoothing must be in [0.0, 1.0), got {label_smoothing}"
            )

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Reduction must be 'none', 'mean', or 'sum', got {reduction}"
            )

        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute vocabulary parallel cross-entropy loss.

        Args:
            input: Model output logits split across vocab dimension
                Shape: [sequence_length, batch_size, vocab_size/tp_size]
            target: Target token indices
                Shape: [sequence_length, batch_size]

        Returns:
            Loss tensor with specified reduction applied
        """
        loss = vocab_parallel_cross_entropy(input, target, self.label_smoothing)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # reduction == 'none'
            return loss

    def extra_repr(self) -> str:
        return f"label_smoothing={self.label_smoothing}, reduction={self.reduction}"
