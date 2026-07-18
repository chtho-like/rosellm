"""Auditable token-level policy objectives for educational Agentic RL."""

from __future__ import annotations

import math
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
    return (
        F.log_softmax(logits, dim=-1)
        .gather(dim=-1, index=target_ids.unsqueeze(-1))
        .squeeze(-1)
    )


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


@dataclass(frozen=True)
class SaoPolicyLossOutput:
    """SAO policy loss plus trust-mask diagnostics.

    ``rejected_fraction`` is the fraction of policy action tokens discarded by
    Direct Double-Sided Importance Sampling.  ``trusted_mask`` is already
    intersected with ``action_mask`` and is useful for boundary-level tests.
    """

    loss: Tensor
    objective: Tensor
    approximate_kl: Tensor
    rejected_fraction: Tensor
    mean_ratio: Tensor
    trusted_mask: Tensor


@dataclass(frozen=True)
class GspoPolicyLossOutput:
    """GSPO loss and sequence-level trust-region diagnostics."""

    loss: Tensor
    objective: Tensor
    sequence_ratio_divergence: Tensor
    sequence_clip_fraction: Tensor
    mean_sequence_ratio: Tensor
    sequence_ratios: Tensor


@dataclass(frozen=True)
class SapoPolicyLossOutput:
    """SAPO loss and smooth-gate diagnostics."""

    loss: Tensor
    objective: Tensor
    approximate_kl: Tensor
    mean_ratio: Tensor
    mean_gradient_gate: Tensor


def direct_double_sided_mask(
    ratios: Tensor,
    *,
    clip_low: float,
    clip_high: float,
) -> Tensor:
    """Return SAO's strict two-sided importance-ratio trust mask.

    A ratio is retained only when
    ``1 - clip_low < ratio < 1 + clip_high``.  Values exactly at either
    boundary are rejected, matching Equation 3 in Hou et al. (2026).  This is
    masking, not PPO's boundary saturation.
    """

    if not ratios.is_floating_point():
        raise TypeError("ratios must be floating point")
    if not 0.0 <= clip_low < 1.0:
        raise ValueError("clip_low must be in [0, 1)")
    if clip_high < 0.0:
        raise ValueError("clip_high must be non-negative")
    return (ratios > 1.0 - clip_low) & (ratios < 1.0 + clip_high)


def sao_policy_loss(
    current_logprobs: Tensor,
    rollout_logprobs: Tensor,
    advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_low: float,
    clip_high: float,
    detach_ratio: bool = True,
    max_log_ratio: float = 20.0,
) -> SaoPolicyLossOutput:
    """Compute the published SAO/DIS token objective.

    Single-Rollout Asynchronous Optimization (SAO) directly compares current
    token probabilities with log-probabilities stored by the rollout engine.
    Direct Double-Sided Importance Sampling (DIS) assigns zero policy
    contribution to either ratio tail.  The denominator remains all policy
    action tokens, so rejection contributes a literal zero rather than silently
    changing the effective batch normalization.

    The SAO paper prints ``f(ratio) * advantage * current_logprob`` but does not
    mark whether ``f(ratio)`` is stop-gradient.  ``detach_ratio=True`` selects
    the conventional importance-weight interpretation.  Setting it to false is
    intentionally supported for experiments that demonstrate the additional
    derivative term; callers must record the choice.
    """

    if not (
        current_logprobs.shape
        == rollout_logprobs.shape
        == advantages.shape
        == action_mask.shape
    ):
        raise ValueError("all tensors must have the same shape")
    if max_log_ratio <= 0.0:
        raise ValueError("max_log_ratio must be positive")
    if not 0.0 <= clip_low < 1.0:
        raise ValueError("clip_low must be in [0, 1)")
    if clip_high < 0.0:
        raise ValueError("clip_high must be non-negative")

    # Clamping is solely an overflow guard.  It must not move a ratio boundary
    # into the trusted interval, so require enough dynamic range to represent
    # both thresholds before applying the guard.
    required_range = max(-math.log1p(-clip_low), math.log1p(clip_high))
    if max_log_ratio <= required_range:
        raise ValueError("max_log_ratio must exceed both trust-boundary log ratios")

    # Rollout log-probabilities are immutable behavior-policy evidence.  Even
    # the non-detached-ratio research variant must never optimize through the
    # rollout side of the ratio.
    log_ratio = current_logprobs - rollout_logprobs.detach()
    safe_log_ratio = log_ratio.clamp(-max_log_ratio, max_log_ratio)
    ratio = safe_log_ratio.exp()
    trusted = direct_double_sided_mask(ratio, clip_low=clip_low, clip_high=clip_high)
    policy_mask = action_mask.to(device=ratio.device, dtype=torch.bool)
    trusted_action = trusted & policy_mask

    importance_weight = ratio.detach() if detach_ratio else ratio
    weighted_logprob = torch.where(
        trusted,
        importance_weight * advantages.detach() * current_logprobs,
        torch.zeros_like(current_logprobs),
    )
    objective = masked_mean(weighted_logprob, policy_mask)
    approximate_kl = masked_mean(ratio - 1.0 - safe_log_ratio, policy_mask)
    rejected_fraction = masked_mean((~trusted).to(ratio.dtype), policy_mask)
    mean_ratio = masked_mean(ratio, policy_mask)

    return SaoPolicyLossOutput(
        loss=-objective,
        objective=objective,
        approximate_kl=approximate_kl,
        rejected_fraction=rejected_fraction,
        mean_ratio=mean_ratio,
        trusted_mask=trusted_action,
    )


def gspo_policy_loss(
    current_logprobs: Tensor,
    rollout_logprobs: Tensor,
    sequence_advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_low: float,
    clip_high: float,
    max_log_ratio: float = 20.0,
) -> GspoPolicyLossOutput:
    """Compute Group Sequence Policy Optimization's sequence objective.

    Each rank-2 row is one sampled response.  Its importance ratio is the
    geometric mean of action-token ratios, then clipping and optimization are
    performed once for the whole response.  Rows are averaged equally, as in
    Equation 5 of Zheng et al. (2025); a long response does not receive extra
    weight merely because it contains more tokens.
    """

    if current_logprobs.ndim != 2:
        raise ValueError("GSPO expects rank-2 [sequence, token] tensors")
    if not (current_logprobs.shape == rollout_logprobs.shape == action_mask.shape):
        raise ValueError("log-probability tensors and action_mask must match")
    if sequence_advantages.shape != current_logprobs.shape[:-1]:
        raise ValueError("sequence_advantages must have shape [sequence]")
    if not (
        current_logprobs.is_floating_point()
        and rollout_logprobs.is_floating_point()
        and sequence_advantages.is_floating_point()
    ):
        raise TypeError("log-probabilities and advantages must be floating point")
    if not 0.0 <= clip_low < 1.0:
        raise ValueError("clip_low must be in [0, 1)")
    if clip_high < 0.0:
        raise ValueError("clip_high must be non-negative")
    if max_log_ratio <= 0.0:
        raise ValueError("max_log_ratio must be positive")

    required_range = max(-math.log1p(-clip_low), math.log1p(clip_high))
    if max_log_ratio <= required_range:
        raise ValueError("max_log_ratio must exceed both clip-boundary log ratios")

    mask = action_mask.to(device=current_logprobs.device, dtype=torch.bool)
    counts = mask.sum(dim=-1)
    if (counts == 0).any():
        raise ValueError("every sequence needs at least one action token")
    weights = mask.to(current_logprobs.dtype)
    token_log_ratios = current_logprobs - rollout_logprobs.detach()
    sequence_log_ratios = (token_log_ratios * weights).sum(dim=-1) / counts
    safe_log_ratios = sequence_log_ratios.clamp(-max_log_ratio, max_log_ratio)
    sequence_ratios = safe_log_ratios.exp()
    clipped_ratios = sequence_ratios.clamp(1.0 - clip_low, 1.0 + clip_high)
    advantages = sequence_advantages.detach()
    surrogate = torch.minimum(
        sequence_ratios * advantages,
        clipped_ratios * advantages,
    )
    objective = surrogate.mean()
    clipped = (sequence_ratios < 1.0 - clip_low) | (sequence_ratios > 1.0 + clip_high)

    return GspoPolicyLossOutput(
        loss=-objective,
        objective=objective,
        sequence_ratio_divergence=(sequence_ratios - 1.0 - safe_log_ratios).mean(),
        sequence_clip_fraction=clipped.to(sequence_ratios.dtype).mean(),
        mean_sequence_ratio=sequence_ratios.mean(),
        sequence_ratios=sequence_ratios,
    )


def gspo_token_policy_loss(
    current_logprobs: Tensor,
    rollout_logprobs: Tensor,
    token_advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_low: float,
    clip_high: float,
    max_log_ratio: float = 20.0,
) -> GspoPolicyLossOutput:
    """Compute the GSPO-token objective with its published detach trick.

    The numerical ratio at every action token equals the response's geometric-
    mean sequence ratio.  Its gradient, however, flows only through that
    token's current log-probability.  Uniform token advantages therefore match
    sequence GSPO in both objective and gradient, while multi-turn callers may
    supply distinct token or turn advantages.
    """

    if current_logprobs.ndim != 2:
        raise ValueError("GSPO-token expects rank-2 [sequence, token] tensors")
    if not (
        current_logprobs.shape
        == rollout_logprobs.shape
        == token_advantages.shape
        == action_mask.shape
    ):
        raise ValueError("all tensors must have the same shape")
    if not (
        current_logprobs.is_floating_point()
        and rollout_logprobs.is_floating_point()
        and token_advantages.is_floating_point()
    ):
        raise TypeError("log-probabilities and advantages must be floating point")
    if not 0.0 <= clip_low < 1.0:
        raise ValueError("clip_low must be in [0, 1)")
    if clip_high < 0.0:
        raise ValueError("clip_high must be non-negative")
    if max_log_ratio <= 0.0:
        raise ValueError("max_log_ratio must be positive")

    required_range = max(-math.log1p(-clip_low), math.log1p(clip_high))
    if max_log_ratio <= required_range:
        raise ValueError("max_log_ratio must exceed both clip-boundary log ratios")

    mask = action_mask.to(device=current_logprobs.device, dtype=torch.bool)
    counts = mask.sum(dim=-1)
    if (counts == 0).any():
        raise ValueError("every sequence needs at least one action token")
    weights = mask.to(current_logprobs.dtype)
    token_log_ratios = current_logprobs - rollout_logprobs.detach()
    sequence_log_ratios = (token_log_ratios * weights).sum(dim=-1) / counts
    safe_log_ratios = sequence_log_ratios.clamp(-max_log_ratio, max_log_ratio)
    sequence_ratios = safe_log_ratios.exp()

    # Equation 14: sg[s_i] * pi_theta(token) / sg[pi_theta(token)].
    # exp(logp - sg[logp]) is numerically one and has d/dlogp = one.
    gradient_carrier = (current_logprobs - current_logprobs.detach()).exp()
    token_ratios = sequence_ratios.detach().unsqueeze(-1) * gradient_carrier
    clipped_ratios = token_ratios.clamp(1.0 - clip_low, 1.0 + clip_high)
    advantages = token_advantages.detach()
    surrogate = torch.minimum(
        token_ratios * advantages,
        clipped_ratios * advantages,
    )
    per_sequence_objective = (surrogate * weights).sum(dim=-1) / counts
    objective = per_sequence_objective.mean()
    clipped = (sequence_ratios < 1.0 - clip_low) | (sequence_ratios > 1.0 + clip_high)

    return GspoPolicyLossOutput(
        loss=-objective,
        objective=objective,
        sequence_ratio_divergence=(sequence_ratios - 1.0 - safe_log_ratios).mean(),
        sequence_clip_fraction=clipped.to(sequence_ratios.dtype).mean(),
        mean_sequence_ratio=sequence_ratios.mean(),
        sequence_ratios=sequence_ratios,
    )


def sapo_policy_loss(
    current_logprobs: Tensor,
    rollout_logprobs: Tensor,
    token_advantages: Tensor,
    action_mask: Tensor,
    *,
    tau_positive: float = 1.0,
    tau_negative: float = 1.05,
    max_log_ratio: float = 20.0,
) -> SapoPolicyLossOutput:
    """Compute Soft Adaptive Policy Optimization's smooth surrogate.

    SAPO applies ``(4 / tau) * sigmoid(tau * (ratio - 1))`` directly to
    each token ratio.  Differentiation produces the gradient gate
    ``4 * p * (1 - p)``, which equals one on-policy and decays continuously
    away from ratio one.  The paper averages tokens within each response and
    then averages responses, so padding cannot change a response's weight.
    """

    if current_logprobs.ndim != 2:
        raise ValueError("SAPO expects rank-2 [sequence, token] tensors")
    if not (
        current_logprobs.shape
        == rollout_logprobs.shape
        == token_advantages.shape
        == action_mask.shape
    ):
        raise ValueError("all tensors must have the same shape")
    if not (
        current_logprobs.is_floating_point()
        and rollout_logprobs.is_floating_point()
        and token_advantages.is_floating_point()
    ):
        raise TypeError("log-probabilities and advantages must be floating point")
    if tau_positive <= 0.0 or tau_negative <= 0.0:
        raise ValueError("SAPO temperatures must be positive")
    if max_log_ratio <= 0.0:
        raise ValueError("max_log_ratio must be positive")

    mask = action_mask.to(device=current_logprobs.device, dtype=torch.bool)
    counts = mask.sum(dim=-1)
    if (counts == 0).any():
        raise ValueError("every sequence needs at least one action token")
    weights = mask.to(current_logprobs.dtype)
    log_ratio = current_logprobs - rollout_logprobs.detach()
    safe_log_ratio = log_ratio.clamp(-max_log_ratio, max_log_ratio)
    ratio = safe_log_ratio.exp()
    advantages = token_advantages.detach()
    temperature = torch.where(
        advantages > 0.0,
        torch.full_like(advantages, tau_positive),
        torch.full_like(advantages, tau_negative),
    )
    probability = torch.sigmoid(temperature * (ratio - 1.0))
    soft_surrogate = (4.0 / temperature) * probability * advantages
    per_sequence_objective = (soft_surrogate * weights).sum(dim=-1) / counts
    objective = per_sequence_objective.mean()
    gradient_gate = 4.0 * probability * (1.0 - probability)

    per_sequence_kl = ((ratio - 1.0 - safe_log_ratio) * weights).sum(dim=-1) / counts
    per_sequence_ratio = (ratio * weights).sum(dim=-1) / counts
    per_sequence_gate = (gradient_gate * weights).sum(dim=-1) / counts
    return SapoPolicyLossOutput(
        loss=-objective,
        objective=objective,
        approximate_kl=per_sequence_kl.mean(),
        mean_ratio=per_sequence_ratio.mean(),
        mean_gradient_gate=per_sequence_gate.mean(),
    )


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

    log_ratio = current_logprobs - old_logprobs.detach()
    safe_log_ratio = log_ratio.clamp(-max_log_ratio, max_log_ratio)
    ratio = safe_log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)

    detached_advantages = advantages.detach()
    unclipped = ratio * detached_advantages
    clipped = clipped_ratio * detached_advantages
    surrogate = torch.minimum(unclipped, clipped)
    objective = masked_mean(surrogate, action_mask)

    # exp(log_ratio) - 1 - log_ratio is a non-negative second-order
    # approximation/Monte Carlo diagnostic used by many PPO implementations.
    approximate_kl = masked_mean(ratio - 1.0 - safe_log_ratio, action_mask)
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
    if not (current_logprobs.shape == old_logprobs.shape == action_mask.shape):
        raise ValueError("all tensors must have the same shape")
    weights = action_mask.to(dtype=current_logprobs.dtype)
    counts = weights.sum(dim=-1)
    if (counts == 0).any():
        raise ValueError("every sequence needs at least one action token")
    total = ((current_logprobs - old_logprobs.detach()) * weights).sum(dim=-1)
    return total / counts if length_normalize else total
