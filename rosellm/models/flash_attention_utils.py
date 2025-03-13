from typing import Optional, Tuple, TypedDict

import torch


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    sliding_window: Optional[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    raise NotImplementedError


class FlashAttentionKwargs(TypedDict, total=False):
    pass
