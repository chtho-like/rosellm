import pytest
import torch

from rosellm.roserlhf.advantages import (
    broadcast_turn_advantages,
    discounted_returns,
    generalized_advantage_estimation,
    group_standardized_advantages,
    leave_one_out_advantages,
    skip_observation_generalized_advantage_estimation,
)


def test_discounted_returns_cut_true_terminal() -> None:
    rewards = torch.tensor([[1.0, 2.0, 100.0]])
    terminal = torch.tensor([[False, True, False]])
    returns = discounted_returns(rewards, gamma=0.5, terminated=terminal)
    torch.testing.assert_close(returns, torch.tensor([[2.0, 2.0, 100.0]]))


def test_discounted_returns_bootstrap_truncation() -> None:
    rewards = torch.tensor([[1.0, 2.0]])
    returns = discounted_returns(
        rewards,
        gamma=0.5,
        terminated=torch.tensor([[False, False]]),
        bootstrap_value=torch.tensor([10.0]),
    )
    torch.testing.assert_close(returns, torch.tensor([[4.5, 7.0]]))


def test_discounted_returns_skip_padding() -> None:
    rewards = torch.tensor([[1.0, 2.0, 999.0]])
    valid = torch.tensor([[True, True, False]])
    returns = discounted_returns(rewards, valid_mask=valid)
    torch.testing.assert_close(returns, torch.tensor([[3.0, 2.0, 0.0]]))


def test_gae_lambda_one_matches_monte_carlo_minus_value() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 0.7, 1.1]])
    terminated = torch.tensor([[False, False, True]])
    advantages, targets = generalized_advantage_estimation(
        rewards,
        values,
        gamma=0.9,
        gae_lambda=1.0,
        terminated=terminated,
    )
    returns = discounted_returns(rewards, gamma=0.9, terminated=terminated)
    torch.testing.assert_close(advantages, returns - values)
    torch.testing.assert_close(targets, returns)


def test_leave_one_out_advantages() -> None:
    rewards = torch.tensor([[0.0, 1.0, 2.0]])
    advantages = leave_one_out_advantages(rewards)
    torch.testing.assert_close(advantages, torch.tensor([[-1.5, 0.0, 1.5]]))
    assert advantages.sum().item() == pytest.approx(0.0)


def test_leave_one_out_rejects_singleton() -> None:
    with pytest.raises(ValueError, match="group_size"):
        leave_one_out_advantages(torch.tensor([[1.0]]))


def test_group_standardized_advantages_zero_for_uniform_group() -> None:
    rewards = torch.ones(2, 4)
    assert torch.equal(group_standardized_advantages(rewards), torch.zeros_like(rewards))


def test_group_standardized_advantages_have_zero_mean_unit_population_std() -> None:
    rewards = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    advantages = group_standardized_advantages(rewards)
    torch.testing.assert_close(advantages.mean(-1), torch.zeros(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        advantages.std(-1, unbiased=False), torch.ones(1), atol=1e-6, rtol=0
    )


def test_broadcast_turn_advantages_masks_observations() -> None:
    turns = torch.tensor([[2.0, -1.0]])
    turn_ids = torch.tensor([[0, 0, 0, 1, 1]])
    action_mask = torch.tensor([[True, False, True, False, True]])
    result = broadcast_turn_advantages(turns, turn_ids, action_mask)
    torch.testing.assert_close(result, torch.tensor([[2.0, 0.0, 2.0, 0.0, -1.0]]))


def test_broadcast_turn_advantages_rejects_invalid_unmasked_id() -> None:
    with pytest.raises(ValueError, match="invalid turn"):
        broadcast_turn_advantages(
            torch.tensor([[1.0]]),
            torch.tensor([[2]]),
            torch.tensor([[True]]),
        )


def test_skip_observation_gae_bridges_semantic_action_boundaries() -> None:
    advantages, targets = skip_observation_generalized_advantage_estimation(
        rewards=torch.tensor([[0.0, 1.0]]),
        action_end_values=torch.tensor([[0.2, 0.4]]),
        next_action_values=torch.tensor([[0.4, 0.0]]),
        terminated=torch.tensor([[False, True]]),
        gamma=1.0,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(advantages, torch.tensor([[0.8, 0.6]]))
    torch.testing.assert_close(targets, torch.tensor([[1.0, 1.0]]))


def test_skip_observation_gae_true_terminal_ignores_next_value() -> None:
    advantages, targets = skip_observation_generalized_advantage_estimation(
        rewards=torch.tensor([[1.0]]),
        action_end_values=torch.tensor([[0.25]]),
        next_action_values=torch.tensor([[999.0]]),
        terminated=torch.tensor([[True]]),
    )
    torch.testing.assert_close(advantages, torch.tensor([[0.75]]))
    torch.testing.assert_close(targets, torch.tensor([[1.0]]))


def test_skip_observation_gae_bootstraps_a_time_limit() -> None:
    advantages, targets = skip_observation_generalized_advantage_estimation(
        rewards=torch.tensor([[2.0]]),
        action_end_values=torch.tensor([[1.0]]),
        next_action_values=torch.tensor([[5.0]]),
        terminated=torch.tensor([[False]]),
        gamma=0.5,
    )
    torch.testing.assert_close(advantages, torch.tensor([[3.5]]))
    torch.testing.assert_close(targets, torch.tensor([[4.5]]))


def test_skip_observation_gae_skips_padding_without_cutting_the_trace() -> None:
    advantages, targets = skip_observation_generalized_advantage_estimation(
        rewards=torch.tensor([[1.0, 999.0, 2.0]]),
        action_end_values=torch.zeros(1, 3),
        next_action_values=torch.zeros(1, 3),
        terminated=torch.tensor([[False, False, True]]),
        valid_mask=torch.tensor([[True, False, True]]),
    )
    torch.testing.assert_close(advantages, torch.tensor([[3.0, 0.0, 2.0]]))
    torch.testing.assert_close(targets, torch.tensor([[3.0, 0.0, 2.0]]))


def test_skip_observation_lambda_zero_is_one_step_temporal_difference() -> None:
    advantages, _ = skip_observation_generalized_advantage_estimation(
        rewards=torch.tensor([[0.0, 1.0]]),
        action_end_values=torch.tensor([[0.2, 0.4]]),
        next_action_values=torch.tensor([[0.4, 0.0]]),
        terminated=torch.tensor([[False, True]]),
        gae_lambda=0.0,
    )
    torch.testing.assert_close(advantages, torch.tensor([[0.2, 0.6]]))
