import pytest

from rosellm.roserlhf.trajectory import PolicyAction, Trajectory, TransitionRecord


def make_action(turn: int, policy: str = "policy-1") -> PolicyAction:
    return PolicyAction(
        turn=turn,
        token_ids=[10, 11],
        old_logprobs=[-0.2, -0.5],
        policy_version=policy,
        text="search",
        parsed_action={"tool": "search", "query": "evidence"},
    )


def make_transition(turn: int, *, terminal: bool = False) -> TransitionRecord:
    return TransitionRecord(
        turn=turn,
        observation_ref=f"sha256:observation-{turn}",
        reward_components={"success": float(terminal)},
        terminated=terminal,
    )


def test_policy_action_requires_exact_logprob_per_token() -> None:
    with pytest.raises(ValueError, match="every sampled token"):
        PolicyAction(
            turn=0,
            token_ids=[1, 2],
            old_logprobs=[-0.1],
            policy_version="v1",
            text="x",
        )


def test_trajectory_orders_turns_and_tracks_mixed_policy_versions() -> None:
    trajectory = Trajectory(
        trajectory_id="traj-1",
        task_id="task-1",
        environment_version="env-3",
        tokenizer_hash="sha256:tokenizer",
    )
    trajectory.append(make_action(0, "v1"), make_transition(0))
    trajectory.append(make_action(1, "v2"), make_transition(1, terminal=True))
    trajectory.validate()
    assert trajectory.is_finished
    assert trajectory.behavior_policy_versions == ["v1", "v2"]


def test_trajectory_rejects_append_after_terminal() -> None:
    trajectory = Trajectory("t", "q", "env", "tok")
    trajectory.append(make_action(0), make_transition(0, terminal=True))
    with pytest.raises(ValueError, match="finished"):
        trajectory.append(make_action(1), make_transition(1))


def test_trajectory_can_validate_partial_rollout_explicitly() -> None:
    trajectory = Trajectory("t", "q", "env", "tok")
    trajectory.append(make_action(0), make_transition(0))
    trajectory.validate(require_finished=False)
    with pytest.raises(ValueError, match="incomplete"):
        trajectory.validate()
