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
    skip_observation_generalized_advantage_estimation,
)
from .losses import (
    PolicyLossOutput,
    SaoPolicyLossOutput,
    clipped_policy_loss,
    direct_double_sided_mask,
    gather_token_logprobs,
    sao_policy_loss,
)

__all__ = [
    "PolicyLossOutput",
    "SaoPolicyLossOutput",
    "broadcast_turn_advantages",
    "clipped_policy_loss",
    "discounted_returns",
    "direct_double_sided_mask",
    "gather_token_logprobs",
    "generalized_advantage_estimation",
    "group_standardized_advantages",
    "leave_one_out_advantages",
    "sao_policy_loss",
    "skip_observation_generalized_advantage_estimation",
]
