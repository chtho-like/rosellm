import math

import pytest
import torch

from rosellm.roserlhf.losses import (
    clipped_policy_loss,
    direct_double_sided_mask,
    gather_token_logprobs,
    sao_policy_loss,
    sequence_log_ratio,
)


def test_gather_token_logprobs_matches_manual_distribution() -> None:
    logits = torch.tensor([[[0.0, 1.0], [2.0, -1.0]]])
    targets = torch.tensor([[1, 0]])
    actual = gather_token_logprobs(logits, targets)
    expected = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected)


def test_clipped_loss_ratio_one_is_negative_mean_advantage() -> None:
    old = torch.tensor([[0.0, 0.0, 0.0]])
    current = old.clone().requires_grad_()
    advantages = torch.tensor([[1.0, -2.0, 100.0]])
    mask = torch.tensor([[True, True, False]])
    output = clipped_policy_loss(current, old, advantages, mask)
    assert output.loss.item() == pytest.approx(0.5)
    assert output.mean_ratio.item() == pytest.approx(1.0)
    assert output.approximate_kl.item() == pytest.approx(0.0)
    assert output.clip_fraction.item() == pytest.approx(0.0)


def test_clipping_uses_advantage_sign() -> None:
    # ratio=2: positive advantage is capped at 1.2; negative advantage keeps the
    # more pessimistic -2 rather than -1.2.
    current = torch.tensor([[torch.log(torch.tensor(2.0)), torch.log(torch.tensor(2.0))]])
    old = torch.zeros_like(current)
    advantages = torch.tensor([[1.0, -1.0]])
    output = clipped_policy_loss(
        current, old, advantages, torch.ones_like(current, dtype=torch.bool)
    )
    assert output.objective.item() == pytest.approx((1.2 - 2.0) / 2)
    assert output.clip_fraction.item() == pytest.approx(1.0)


def test_masked_observation_has_zero_gradient() -> None:
    current = torch.zeros(1, 3, requires_grad=True)
    old = torch.zeros_like(current)
    advantages = torch.tensor([[1.0, 1000.0, -1.0]])
    mask = torch.tensor([[True, False, True]])
    clipped_policy_loss(current, old, advantages, mask).loss.backward()
    assert current.grad is not None
    assert current.grad[0, 1].item() == 0.0


def test_padding_does_not_change_globally_masked_loss() -> None:
    current = torch.tensor([[0.0, 0.1]])
    old = torch.zeros_like(current)
    advantages = torch.tensor([[1.0, -0.5]])
    mask = torch.ones_like(current, dtype=torch.bool)
    base = clipped_policy_loss(current, old, advantages, mask).loss

    padded = clipped_policy_loss(
        torch.tensor([[0.0, 0.1, 17.0]]),
        torch.tensor([[0.0, 0.0, -12.0]]),
        torch.tensor([[1.0, -0.5, 999.0]]),
        torch.tensor([[True, True, False]]),
    ).loss
    torch.testing.assert_close(base, padded)


def test_sequence_log_ratio_mean_and_sum_are_explicitly_different() -> None:
    current = torch.tensor([[0.2, 0.4]])
    old = torch.zeros_like(current)
    mask = torch.ones_like(current, dtype=torch.bool)
    torch.testing.assert_close(
        sequence_log_ratio(current, old, mask, length_normalize=True),
        torch.tensor([0.3]),
    )
    torch.testing.assert_close(
        sequence_log_ratio(current, old, mask, length_normalize=False),
        torch.tensor([0.6]),
    )


def test_clipped_loss_rejects_empty_mask() -> None:
    values = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="empty mask"):
        clipped_policy_loss(values, values, values, torch.zeros_like(values, dtype=torch.bool))


def test_direct_double_sided_mask_uses_strict_boundaries() -> None:
    ratios = torch.tensor(
        [0.8, 0.800001, 1.0, 1.199999, 1.2], dtype=torch.float64
    )
    trusted = direct_double_sided_mask(
        ratios,
        clip_low=0.2,
        clip_high=0.2,
    )
    assert torch.equal(
        trusted,
        torch.tensor([False, True, True, True, False]),
    )


def test_sao_rejects_both_tails_but_keeps_all_action_tokens_in_denominator() -> None:
    current = torch.tensor(
        [[math.log(0.5), 0.0, math.log(2.0)]],
        dtype=torch.float64,
        requires_grad=True,
    )
    output = sao_policy_loss(
        current,
        torch.zeros_like(current),
        torch.ones_like(current),
        torch.ones_like(current, dtype=torch.bool),
        clip_low=0.2,
        clip_high=0.2,
    )

    assert torch.equal(
        output.trusted_mask,
        torch.tensor([[False, True, False]]),
    )
    assert output.rejected_fraction.item() == pytest.approx(2.0 / 3.0)
    output.loss.backward()
    assert current.grad is not None
    torch.testing.assert_close(
        current.grad,
        torch.tensor([[0.0, -1.0 / 3.0, 0.0]], dtype=torch.float64),
    )


def test_sao_ratio_stop_gradient_choice_exposes_published_ambiguity() -> None:
    rollout = torch.tensor([[-0.6]], dtype=torch.float64, requires_grad=True)
    ratio = math.exp(0.1)

    detached_current = torch.tensor([[-0.5]], dtype=torch.float64, requires_grad=True)
    detached = sao_policy_loss(
        detached_current,
        rollout,
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([[True]]),
        clip_low=0.2,
        clip_high=0.2,
        detach_ratio=True,
    )
    detached.loss.backward()
    assert detached_current.grad is not None
    assert detached_current.grad.item() == pytest.approx(-2.0 * ratio)
    assert rollout.grad is None

    attached_current = torch.tensor([[-0.5]], dtype=torch.float64, requires_grad=True)
    attached = sao_policy_loss(
        attached_current,
        rollout,
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([[True]]),
        clip_low=0.2,
        clip_high=0.2,
        detach_ratio=False,
    )
    attached.loss.backward()
    assert attached_current.grad is not None
    assert attached_current.grad.item() == pytest.approx(-2.0 * ratio * (1.0 - 0.5))
    assert rollout.grad is None


def test_sao_masks_environment_observations_from_loss_and_diagnostics() -> None:
    current = torch.zeros(1, 3, requires_grad=True)
    rollout = torch.tensor([[0.0, -10.0, 0.0]])
    advantages = torch.tensor([[1.0, 1000.0, -1.0]])
    action_mask = torch.tensor([[True, False, True]])

    output = sao_policy_loss(
        current,
        rollout,
        advantages,
        action_mask,
        clip_low=0.2,
        clip_high=0.2,
    )
    output.loss.backward()
    assert current.grad is not None
    assert current.grad[0, 1].item() == 0.0
    assert output.rejected_fraction.item() == pytest.approx(0.0)
    assert torch.equal(output.trusted_mask, action_mask)
