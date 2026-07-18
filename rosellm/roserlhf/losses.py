"""Auditable token-level policy objectives for educational Agentic RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def gather_token_logprobs(logits: Tensor, target_ids: Tensor) -> Tensor:
    """Return log-probability of each target token.

    Args:
        logits: ``[..., vocabulary]`` unnormalized scores.
        target_ids: Integer tensor matching ``logits.shape[:-1]``.

    For a causal LM, callers normally pass ``logits[:, :-1]`` and
    ``input_ids[:, 1:]``.  Keeping the shift outside this function makes the
    alignment visible at the call site.
    """

    if logits.ndim < 2:
        raise ValueError("logits must include positions and vocabulary")
    if logits.shape[:-1] != target_ids.shape:
        raise ValueError(
            f"target shape {target_ids.shape} must match logits prefix {logits.shape[:-1]}"
        )
    if target_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("target_ids must be an integer tensor")
    return F.log_softmax(logits, dim=-1).gather(
        dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Globally average valid entries, rejecting an empty denominator."""

    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same shape")
    weights = mask.to(dtype=values.dtype, device=values.device)
    denominator = weights.sum()
    if denominator.item() == 0:
        raise ValueError("cannot reduce an empty mask")
    return (values * weights).sum() / denominator


@dataclass(frozen=True)
class PolicyLossOutput:
    """Policy loss plus diagnostics computed over action tokens."""

    loss: Tensor
    objective: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor
    mean_ratio: Tensor


def clipped_policy_loss(
    current_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: Optional[float] = None,
    max_log_ratio: float = 20.0,
) -> PolicyLossOutput:
    """Compute a globally token-normalized PPO/GRPO clipped loss.

    The function minimizes ``-min(ratio * A, clipped_ratio * A)``.  Advantages
    must already be aligned to token positions.  Environment observations and
    other non-policy tokens must be false in ``action_mask``.

    ``max_log_ratio`` only prevents overflow in corrupted/extreme batches; it
    does not replace a policy-lag or KL gate.  Diagnostics use the same mask.
    """

    if not (
        current_logprobs.shape
        == old_logprobs.shape
        == advantages.shape
        == action_mask.shape
    ):
        raise ValueError("all tensors must have the same shape")
    if clip_low < 0:
        raise ValueError("clip_low must be non-negative")
    if clip_high is None:
        clip_high = clip_low
    if clip_high < 0:
        raise ValueError("clip_high must be non-negative")
    if max_log_ratio <= 0:
        raise ValueError("max_log_ratio must be positive")

    log_ratio = current_logprobs - old_logprobs
    safe_log_ratio = log_ratio.clamp(-max_log_ratio, max_log_ratio)
    ratio = safe_log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)

    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    surrogate = torch.minimum(unclipped, clipped)
    objective = masked_mean(surrogate, action_mask)

    # exp(log_ratio) - 1 - log_ratio is a non-negative second-order
    # approximation/Monte Carlo diagnostic used by many PPO implementations.
    approximate_kl = masked_mean(
        ratio - 1.0 - safe_log_ratio, action_mask
    )
    clipped_tokens = (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
    clip_fraction = masked_mean(clipped_tokens.to(ratio.dtype), action_mask)
    mean_ratio = masked_mean(ratio, action_mask)

    return PolicyLossOutput(
        loss=-objective,
        objective=objective,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        mean_ratio=mean_ratio,
    )


def sequence_log_ratio(
    current_logprobs: Tensor,
    old_logprobs: Tensor,
    action_mask: Tensor,
    *,
    length_normalize: bool = True,
) -> Tensor:
    """Aggregate token log-ratios into one value per sequence.

    Inputs have shape ``[batch, token_positions]``.  A mean (geometric-mean
    probability ratio) and sum represent different objectives, so callers must
    choose explicitly.
    """

    if current_logprobs.ndim != 2:
        raise ValueError("sequence_log_ratio expects rank-2 tensors")
    if not (
        current_logprobs.shape == old_logprobs.shape == action_mask.shape
    ):
        raise ValueError("all tensors must have the same shape")
    weights = action_mask.to(dtype=current_logprobs.dtype)
    counts = weights.sum(dim=-1)
    if (counts == 0).any():
        raise ValueError("every sequence needs at least one action token")
    total = ((current_logprobs - old_logprobs) * weights).sum(dim=-1)
    return total / counts if length_normalize else total
