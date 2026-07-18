"""Advantage estimators with explicit terminal and masking semantics.

The final axis is time or group unless a function says otherwise.  These
implementations favor readable tensor contracts over framework-specific
optimizations; the accompanying tests are intended to be used as invariants for
larger distributed implementations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def _as_bool_mask(mask: Optional[Tensor], like: Tensor) -> Tensor:
    if mask is None:
        return torch.ones_like(like, dtype=torch.bool)
    if mask.shape != like.shape:
        raise ValueError(f"mask shape {mask.shape} must match {like.shape}")
    return mask.to(device=like.device, dtype=torch.bool)


def discounted_returns(
    rewards: Tensor,
    *,
    gamma: float = 1.0,
    terminated: Optional[Tensor] = None,
    valid_mask: Optional[Tensor] = None,
    bootstrap_value: Optional[Tensor] = None,
) -> Tensor:
    """Compute reward-to-go on the final dimension.

    ``terminated[..., t]`` means transition ``t`` reached a true terminal state,
    so no future value crosses that boundary.  A collector time limit should be
    represented as a final valid step with ``terminated=False`` and a supplied
    ``bootstrap_value``.  Invalid padded steps neither add reward nor propagate
    return.

    Args:
        rewards: Floating tensor shaped ``[..., T]``.
        gamma: Discount in ``[0, 1]``.
        terminated: Boolean tensor shaped like ``rewards``.
        valid_mask: Boolean tensor shaped like ``rewards``.
        bootstrap_value: Value immediately after the last represented step,
            shaped ``rewards.shape[:-1]``.  Defaults to zero.
    """

    if rewards.ndim < 1:
        raise ValueError("rewards must have a time dimension")
    if not rewards.is_floating_point():
        raise TypeError("rewards must be floating point")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    valid = _as_bool_mask(valid_mask, rewards)
    terminal = (
        torch.zeros_like(rewards, dtype=torch.bool)
        if terminated is None
        else _as_bool_mask(terminated, rewards)
    )

    batch_shape = rewards.shape[:-1]
    if bootstrap_value is None:
        running = torch.zeros(batch_shape, dtype=rewards.dtype, device=rewards.device)
    else:
        if bootstrap_value.shape != batch_shape:
            raise ValueError(
                f"bootstrap_value shape {bootstrap_value.shape} must be {batch_shape}"
            )
        running = bootstrap_value.to(dtype=rewards.dtype, device=rewards.device)

    returns = torch.zeros_like(rewards)
    for t in range(rewards.shape[-1] - 1, -1, -1):
        continuation = (~terminal[..., t]).to(rewards.dtype)
        candidate = rewards[..., t] + gamma * continuation * running
        running = torch.where(valid[..., t], candidate, running)
        returns[..., t] = torch.where(
            valid[..., t], running, torch.zeros_like(running)
        )
    return returns


def generalized_advantage_estimation(
    rewards: Tensor,
    values: Tensor,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    terminated: Optional[Tensor] = None,
    valid_mask: Optional[Tensor] = None,
    bootstrap_value: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute GAE advantages and value targets.

    ``values[..., t]`` predicts return before action ``t``.  The optional
    ``bootstrap_value`` predicts the state following the final represented
    transition.  True terminals cut both TD bootstrap and the recursive GAE
    trace.  Padding is skipped rather than treated as a terminal state.
    """

    if rewards.shape != values.shape:
        raise ValueError("rewards and values must have the same shape")
    if not rewards.is_floating_point() or not values.is_floating_point():
        raise TypeError("rewards and values must be floating point")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")

    valid = _as_bool_mask(valid_mask, rewards)
    terminal = (
        torch.zeros_like(rewards, dtype=torch.bool)
        if terminated is None
        else _as_bool_mask(terminated, rewards)
    )
    batch_shape = rewards.shape[:-1]
    if bootstrap_value is None:
        next_value = torch.zeros(
            batch_shape, dtype=values.dtype, device=values.device
        )
    else:
        if bootstrap_value.shape != batch_shape:
            raise ValueError(
                f"bootstrap_value shape {bootstrap_value.shape} must be {batch_shape}"
            )
        next_value = bootstrap_value.to(dtype=values.dtype, device=values.device)

    next_advantage = torch.zeros_like(next_value)
    advantages = torch.zeros_like(rewards)

    for t in range(rewards.shape[-1] - 1, -1, -1):
        nonterminal = (~terminal[..., t]).to(rewards.dtype)
        delta = rewards[..., t] + gamma * nonterminal * next_value - values[..., t]
        candidate = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantage_t = torch.where(valid[..., t], candidate, torch.zeros_like(candidate))
        advantages[..., t] = advantage_t

        # Padding is absent time, not a state transition.  Preserve the next
        # valid value/trace while walking backward across padded positions.
        next_value = torch.where(valid[..., t], values[..., t], next_value)
        next_advantage = torch.where(valid[..., t], advantage_t, next_advantage)

    value_targets = torch.where(valid, advantages + values, torch.zeros_like(values))
    return advantages, value_targets


def leave_one_out_advantages(rewards: Tensor, *, dim: int = -1) -> Tensor:
    """Subtract the mean reward of every *other* member in a group.

    Group members should be conditionally independent rollouts of the same task.
    A group of one has no leave-one-out baseline and is rejected.
    """

    if not rewards.is_floating_point():
        raise TypeError("rewards must be floating point")
    group_size = rewards.shape[dim]
    if group_size < 2:
        raise ValueError("leave-one-out advantages require group_size >= 2")
    other_mean = (rewards.sum(dim=dim, keepdim=True) - rewards) / (group_size - 1)
    return rewards - other_mean


def group_standardized_advantages(
    rewards: Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    """Center and population-standardize rewards within a rollout group.

    Uniform-reward groups return exactly zero.  This makes the no-signal case
    explicit and avoids amplifying floating-point noise by dividing by ``eps``.
    """

    if not rewards.is_floating_point():
        raise TypeError("rewards must be floating point")
    if rewards.shape[dim] < 2:
        raise ValueError("group standardization requires group_size >= 2")
    if eps <= 0:
        raise ValueError("eps must be positive")

    centered = rewards - rewards.mean(dim=dim, keepdim=True)
    std = rewards.std(dim=dim, unbiased=False, keepdim=True)
    standardized = centered / std.clamp_min(eps)
    return torch.where(std > eps, standardized, torch.zeros_like(standardized))


def broadcast_turn_advantages(
    turn_advantages: Tensor,
    turn_ids: Tensor,
    action_mask: Tensor,
) -> Tensor:
    """Map ``[batch, turns]`` advantages to policy-token positions.

    ``turn_ids`` and ``action_mask`` have shape ``[batch, token_positions]``.
    Masked positions may use any turn ID; their returned advantage is zero.
    Unmasked positions must refer to a valid turn.
    """

    if turn_advantages.ndim != 2:
        raise ValueError("turn_advantages must have shape [batch, turns]")
    if turn_ids.ndim != 2 or action_mask.ndim != 2:
        raise ValueError("turn_ids and action_mask must be rank-2")
    if turn_ids.shape != action_mask.shape:
        raise ValueError("turn_ids and action_mask must have the same shape")
    if turn_ids.shape[0] != turn_advantages.shape[0]:
        raise ValueError("batch dimensions must match")

    mask = action_mask.to(dtype=torch.bool, device=turn_advantages.device)
    ids = turn_ids.to(dtype=torch.long, device=turn_advantages.device)
    invalid = mask & ((ids < 0) | (ids >= turn_advantages.shape[1]))
    if invalid.any():
        raise ValueError("an unmasked policy token has an invalid turn ID")

    safe_ids = ids.clamp(0, turn_advantages.shape[1] - 1)
    token_advantages = turn_advantages.gather(dim=1, index=safe_ids)
    return torch.where(mask, token_advantages, torch.zeros_like(token_advantages))
