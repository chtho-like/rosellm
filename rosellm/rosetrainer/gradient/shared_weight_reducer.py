"""Shared weight gradient reduction for tied embeddings.

This module implements gradient synchronization for shared weights between
input embeddings and output layers, following Megatron-LM's pattern.
"""

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from rosellm.rosetrainer.parallelism import parallel_state


class SharedWeightGradientReducer:
    """Manages gradient reduction for shared weights across pipeline stages.

    This class handles the synchronization of gradients for parameters that are
    shared between different pipeline stages, particularly input embeddings and
    output projection weights.

    Key features:
    - All-reduce gradients across embedding group for tied weights
    - Support for position embeddings shared across encoder/decoder
    - Efficient coalesced communication for multiple shared parameters
    - Compatibility with mixed precision training

    Args:
        config: Configuration object with gradient settings
    """

    def __init__(self, config):
        self.config = config
        self.timers = getattr(config, "timers", None)

        # Cache process groups to avoid repeated lookups
        self._pp_group: Optional[torch.distributed.ProcessGroup] = None
        self._embd_group: Optional[torch.distributed.ProcessGroup] = None
        self._pos_embd_group: Optional[torch.distributed.ProcessGroup] = None

    @property
    def pp_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Get pipeline parallel process group."""
        if self._pp_group is None:
            self._pp_group = parallel_state.get_pipeline_model_parallel_group()
        return self._pp_group

    @property
    def embd_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Get embedding process group."""
        if self._embd_group is None:
            # Use pipeline parallel group for embedding synchronization
            # In a full Megatron-LM implementation, this would be a dedicated group
            # that connects first and last pipeline stages
            self._embd_group = parallel_state.get_pipeline_model_parallel_group()
        return self._embd_group

    @property
    def pos_embd_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Get position embedding process group."""
        if self._pos_embd_group is None:
            # Use pipeline parallel group for position embedding synchronization
            # This would typically be a specialized group in full implementations
            self._pos_embd_group = parallel_state.get_pipeline_model_parallel_group()
        return self._pos_embd_group

    def _get_main_grad_attr(self, param: nn.Parameter) -> str:
        """Get the attribute name for the main gradient.

        Args:
            param: Parameter to check

        Returns:
            'main_grad' if using mixed precision, otherwise 'grad'
        """
        if hasattr(param, "main_grad"):
            return "main_grad"
        return "grad"

    def _is_first_stage(self) -> bool:
        """Check if current rank is first pipeline stage."""
        # Check if pipeline parallel is initialized
        if not parallel_state.is_initialized():
            return True

        # Use the correct function name from parallel_state
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        return pp_rank == 0

    def _is_last_stage(self) -> bool:
        """Check if current rank is last pipeline stage."""
        # Check if pipeline parallel is initialized
        if not parallel_state.is_initialized():
            return True

        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_size = parallel_state.get_pipeline_model_parallel_size()
        return pp_rank == pp_size - 1

    def _get_process_group_size(
        self, group: Optional[torch.distributed.ProcessGroup]
    ) -> int:
        """Get size of process group, handling None case.

        Args:
            group: Process group or None

        Returns:
            Size of group, or 1 if group is None
        """
        if group is None:
            return 1
        return int(torch.distributed.get_world_size(group=group))

    def _is_in_embedding_group(self) -> bool:
        """Check if current rank is in the embedding group."""
        if self.embd_group is None:
            return False

        current_rank = torch.distributed.get_rank()
        group_ranks = torch.distributed.get_process_group_ranks(self.embd_group)

        return current_rank in group_ranks

    def allreduce_word_embedding_grads(
        self,
        model: List[nn.Module],
        get_embedding_weight: Optional[
            Callable[[nn.Module], Optional[nn.Parameter]]
        ] = None,
    ) -> None:
        """All-reduce word embedding gradients across first and last pipeline stages.

        This ensures that word_embeddings parameters stay in sync when they are
        shared between the input and output layers.

        Args:
            model: List of model chunks (including virtual pipeline chunks)
            get_embedding_weight: Function to extract embedding weight from model.
                                If None, uses default extraction logic.
        """
        # Skip if embedding group is not initialized or too small
        if self._get_process_group_size(self.embd_group) <= 1:
            return

        # Skip if not in embedding group
        if not self._is_in_embedding_group():
            return

        # Start timer if available
        if self.timers is not None:
            self.timers("embedding-grads-all-reduce", log_level=1).start()

        try:
            self._allreduce_embedding_grad(
                model=model,
                weight_getter=get_embedding_weight
                or self._default_get_word_embedding_weight,
                embd_group=self.embd_group,
                skip_if_none=True,
            )
        finally:
            if self.timers is not None:
                self.timers("embedding-grads-all-reduce").stop()

    def allreduce_position_embedding_grads(
        self,
        model: List[nn.Module],
        get_position_weight: Optional[
            Callable[[nn.Module], Optional[nn.Parameter]]
        ] = None,
    ) -> None:
        """All-reduce position embedding gradients across encoder and decoder stages.

        This ensures position embeddings parameters stay in sync across stages.

        Args:
            model: List of model chunks
            get_position_weight: Function to extract position embedding weight
        """
        # Skip if position embedding group is not initialized or too small
        if self._get_process_group_size(self.pos_embd_group) <= 1:
            return

        # Skip if not in position embedding group
        current_rank = torch.distributed.get_rank()
        if self.pos_embd_group is not None:
            group_ranks = torch.distributed.get_process_group_ranks(self.pos_embd_group)
            if current_rank not in group_ranks:
                return

        # Start timer if available
        if self.timers is not None:
            self.timers("position-embedding-grads-all-reduce", log_level=1).start()

        try:
            self._allreduce_embedding_grad(
                model=model,
                weight_getter=get_position_weight
                or self._default_get_position_embedding_weight,
                embd_group=self.pos_embd_group,
                skip_if_none=False,
            )
        finally:
            if self.timers is not None:
                self.timers("position-embedding-grads-all-reduce").stop()

    def _allreduce_embedding_grad(
        self,
        model: List[nn.Module],
        weight_getter: Callable[[nn.Module], Optional[nn.Parameter]],
        embd_group: Optional[torch.distributed.ProcessGroup],
        skip_if_none: bool = True,
    ) -> None:
        """Unified helper to all-reduce embedding parameters across pipeline stages.

        Args:
            model: List of model chunks (PP/VPP)
            weight_getter: Function that takes the pre-process model chunk and
                          returns the parameter to be reduced
            embd_group: The process group over which to reduce
            skip_if_none: If True, quietly returns when parameter or gradient is None
        """
        # Determine which model chunk to use based on pipeline stage
        if self._is_first_stage():
            model_module = model[0]
        elif self._is_last_stage():
            model_module = model[-1]
        else:
            # For intermediate stages with shared embeddings (encoder-decoder)
            model_module = model[0]

        # Unwrap DDP if necessary
        if hasattr(model_module, "module"):
            unwrapped = getattr(model_module, "module")
            if isinstance(unwrapped, nn.Module):
                model_module = unwrapped

        # Get the weight parameter using the provided getter
        weight = weight_getter(model_module)
        if weight is None:
            if skip_if_none:
                return
            raise ValueError("Expected weight parameter but got None")

        # Get gradient attribute and value
        grad_attr = self._get_main_grad_attr(weight)
        grad = getattr(weight, grad_attr, None)

        # Skip if gradient is None (e.g., frozen embedding)
        if grad is None and skip_if_none:
            return

        # Perform all-reduce on the gradient if group exists
        if embd_group is not None and grad is not None:
            torch.distributed.all_reduce(grad, group=embd_group)

        # Re-assign gradient (important for autograd graph)
        setattr(weight, grad_attr, grad)

    def _default_get_word_embedding_weight(
        self, model_module: nn.Module
    ) -> Optional[nn.Parameter]:
        """Default method to get word embedding weight.

        Args:
            model_module: Model module to extract from

        Returns:
            Word embedding weight parameter or None
        """
        # Check for shared_embedding_or_output_weight method (Megatron-LM pattern)
        if hasattr(model_module, "shared_embedding_or_output_weight"):
            method = getattr(model_module, "shared_embedding_or_output_weight")
            if callable(method):
                weight = method()
                if isinstance(weight, nn.Parameter):
                    return weight
            return None

        # Check for explicit embedding attribute
        if hasattr(model_module, "word_embeddings"):
            word_embeds = getattr(model_module, "word_embeddings")
            if isinstance(word_embeds, nn.Module) and hasattr(word_embeds, "weight"):
                weight = getattr(word_embeds, "weight")
                if isinstance(weight, nn.Parameter):
                    return weight

        # Check for embedding layer
        if hasattr(model_module, "embedding"):
            embedding = getattr(model_module, "embedding")
            if isinstance(embedding, nn.Module):
                if hasattr(embedding, "word_embeddings"):
                    word_embeds = getattr(embedding, "word_embeddings")
                    if isinstance(word_embeds, nn.Module) and hasattr(
                        word_embeds, "weight"
                    ):
                        weight = getattr(word_embeds, "weight")
                        if isinstance(weight, nn.Parameter):
                            return weight
                if hasattr(embedding, "weight"):
                    weight = getattr(embedding, "weight")
                    if isinstance(weight, nn.Parameter):
                        return weight

        return None

    def _default_get_position_embedding_weight(
        self, model_module: nn.Module
    ) -> Optional[nn.Parameter]:
        """Default method to get position embedding weight.

        Args:
            model_module: Model module to extract from

        Returns:
            Position embedding weight parameter or None
        """
        # Check for position_embeddings attribute
        if hasattr(model_module, "position_embeddings"):
            pos_embeds = getattr(model_module, "position_embeddings")
            if isinstance(pos_embeds, nn.Module) and hasattr(pos_embeds, "weight"):
                weight = getattr(pos_embeds, "weight")
                if isinstance(weight, nn.Parameter):
                    return weight

        # Check within embedding layer
        if hasattr(model_module, "embedding"):
            embedding = getattr(model_module, "embedding")
            if isinstance(embedding, nn.Module) and hasattr(
                embedding, "position_embeddings"
            ):
                pos_embeds = getattr(embedding, "position_embeddings")
                if isinstance(pos_embeds, nn.Module) and hasattr(pos_embeds, "weight"):
                    weight = getattr(pos_embeds, "weight")
                    if isinstance(weight, nn.Parameter):
                        return weight

        return None

    def allreduce_shared_params(
        self,
        model: List[nn.Module],
        shared_params: List[Tuple[str, nn.Parameter]],
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        """All-reduce arbitrary shared parameters.

        This is a general-purpose method for reducing gradients of any shared
        parameters across a process group.

        Args:
            model: List of model chunks
            shared_params: List of (name, parameter) tuples to reduce
            reduce_group: Process group for reduction (defaults to embedding group)
        """
        if not shared_params:
            return

        if reduce_group is None:
            reduce_group = self.embd_group

        if self._get_process_group_size(reduce_group) <= 1:
            return

        # Collect gradients to reduce
        grads_to_reduce = []
        params_list = []

        for name, param in shared_params:
            if not param.requires_grad:
                continue

            grad_attr = self._get_main_grad_attr(param)
            grad = getattr(param, grad_attr, None)

            if grad is not None:
                grads_to_reduce.append(grad)
                params_list.append((param, grad_attr))

        if not grads_to_reduce:
            return

        # Coalesce gradients for efficient communication
        coalesced = _flatten_dense_tensors(grads_to_reduce)
        torch.distributed.all_reduce(coalesced, group=reduce_group)

        # Unflatten and reassign gradients
        for (param, grad_attr), grad in zip(
            params_list, _unflatten_dense_tensors(coalesced, grads_to_reduce)
        ):
            setattr(param, grad_attr, grad)


class SharedWeightConfig:
    """Configuration for shared weight gradient reduction.

    Attributes:
        share_embeddings_and_output_weights: Whether input/output weights are tied
        share_position_embeddings: Whether position embeddings are shared
        embedding_reduce_group_size: Size of embedding reduction group
        position_embedding_reduce_group_size: Size of position embedding group
    """

    def __init__(
        self,
        share_embeddings_and_output_weights: bool = False,
        share_position_embeddings: bool = False,
        embedding_reduce_group_size: Optional[int] = None,
        position_embedding_reduce_group_size: Optional[int] = None,
    ):
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.share_position_embeddings = share_position_embeddings
        self.embedding_reduce_group_size = embedding_reduce_group_size
        self.position_embedding_reduce_group_size = position_embedding_reduce_group_size
