"""Small immutable trajectory records that preserve exact policy tokens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class PolicyAction:
    """One token sequence sampled by one immutable behavior policy."""

    turn: int
    token_ids: Sequence[int]
    old_logprobs: Sequence[float]
    policy_version: str
    text: str
    parsed_action: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.turn < 0:
            raise ValueError("turn must be non-negative")
        if not self.policy_version:
            raise ValueError("policy_version is required")
        if len(self.token_ids) == 0:
            raise ValueError("a policy action must contain at least one token")
        if len(self.token_ids) != len(self.old_logprobs):
            raise ValueError("every sampled token needs its behavior log-probability")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token IDs must be non-negative")


@dataclass(frozen=True)
class TransitionRecord:
    """Environment result following a semantic policy action."""

    turn: int
    observation_ref: str
    reward_components: Dict[str, float] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    state_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if self.turn < 0:
            raise ValueError("turn must be non-negative")
        if not self.observation_ref:
            raise ValueError("observation_ref is required")
        if self.terminated and self.truncated:
            raise ValueError("a transition cannot be both terminated and truncated")


@dataclass
class Trajectory:
    """Ordered action/transition pairs for one task attempt."""

    trajectory_id: str
    task_id: str
    environment_version: str
    tokenizer_hash: str
    actions: List[PolicyAction] = field(default_factory=list)
    transitions: List[TransitionRecord] = field(default_factory=list)

    def append(self, action: PolicyAction, transition: TransitionRecord) -> None:
        if self.is_finished:
            raise ValueError("cannot append to a finished trajectory")
        expected_turn = len(self.actions)
        if action.turn != expected_turn or transition.turn != expected_turn:
            raise ValueError(f"expected turn {expected_turn}")
        self.actions.append(action)
        self.transitions.append(transition)

    @property
    def is_finished(self) -> bool:
        return bool(self.transitions) and (
            self.transitions[-1].terminated or self.transitions[-1].truncated
        )

    @property
    def behavior_policy_versions(self) -> List[str]:
        """Return versions in first-observed order for mixed-policy auditing."""

        return list(dict.fromkeys(action.policy_version for action in self.actions))

    def validate(self, *, require_finished: bool = True) -> None:
        if not self.trajectory_id or not self.task_id:
            raise ValueError("trajectory_id and task_id are required")
        if not self.environment_version or not self.tokenizer_hash:
            raise ValueError("environment_version and tokenizer_hash are required")
        if len(self.actions) != len(self.transitions):
            raise ValueError("every action must have one transition")
        for expected_turn, (action, transition) in enumerate(
            zip(self.actions, self.transitions)
        ):
            if action.turn != expected_turn or transition.turn != expected_turn:
                raise ValueError("turns must be contiguous and ordered")
            if expected_turn + 1 < len(self.transitions) and (
                transition.terminated or transition.truncated
            ):
                raise ValueError("no events may follow a terminal/truncated transition")
        if require_finished and not self.is_finished:
            raise ValueError("trajectory is incomplete")
