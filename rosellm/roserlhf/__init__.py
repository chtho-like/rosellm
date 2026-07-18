"""Educational reinforcement-learning building blocks for language models.

The package intentionally exposes small, auditable functions before introducing
distributed training machinery.  See ``docs/agentic-rl/source-lab.md`` for the
mathematical contract and exercises.
"""

from .advantages import (
    broadcast_turn_advantages,
    discounted_returns,
    generalized_advantage_estimation,
    group_standardized_advantages,
    leave_one_out_advantages,
)
from .losses import PolicyLossOutput, clipped_policy_loss, gather_token_logprobs

__all__ = [
    "PolicyLossOutput",
    "broadcast_turn_advantages",
    "clipped_policy_loss",
    "discounted_returns",
    "gather_token_logprobs",
    "generalized_advantage_estimation",
    "group_standardized_advantages",
    "leave_one_out_advantages",
]
