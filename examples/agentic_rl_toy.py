"""Train a tiny multi-turn policy with group-relative terminal rewards.

This deliberately small example makes the full Agentic RL data path visible:

1. choose task/initial state;
2. sample several action trajectories from a frozen behavior snapshot;
3. execute transitions and obtain terminal verified reward;
4. build within-task relative advantages;
5. align one trajectory advantage to every generated action token;
6. recompute current log-probabilities and apply a clipped policy update.

It is not an LLM benchmark.  Each semantic action is represented by one token so
that readers can inspect the estimator before adding tokenization, transformers,
tools, distributed rollout, and learned rewards.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn

from rosellm.roserlhf.advantages import group_standardized_advantages
from rosellm.roserlhf.losses import clipped_policy_loss


TARGET_PATHS = torch.tensor(
    [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
        [1, 1, 0],
    ],
    dtype=torch.long,
)


@dataclass(frozen=True)
class RolloutBatch:
    task_ids: Tensor
    actions: Tensor
    old_logprobs: Tensor
    rewards: Tensor
    group_size: int


class TinyTurnPolicy(nn.Module):
    """A table of logits indexed by task instruction and environment turn."""

    def __init__(self, num_tasks: int, horizon: int, num_actions: int = 2) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_tasks, horizon, num_actions))

    def action_logits(self, task_ids: Tensor) -> Tensor:
        return self.logits[task_ids]

    def logprobs(self, task_ids: Tensor, actions: Tensor) -> Tensor:
        distributions = self.action_logits(task_ids).log_softmax(dim=-1)
        return distributions.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def collect_rollouts(
    policy: TinyTurnPolicy,
    task_ids: Tensor,
    *,
    group_size: int,
    generator: torch.Generator,
) -> RolloutBatch:
    """Sample independent action paths and execute an exact terminal verifier."""

    expanded_tasks = task_ids.repeat_interleave(group_size)
    logits = policy.action_logits(expanded_tasks)
    probabilities = logits.softmax(dim=-1)
    flat_actions = torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]),
        num_samples=1,
        generator=generator,
    )
    actions = flat_actions.reshape(expanded_tasks.shape[0], -1)
    old_logprobs = policy.logprobs(expanded_tasks, actions)

    targets = TARGET_PATHS[expanded_tasks]
    rewards = (actions == targets).all(dim=-1).to(logits.dtype)
    return RolloutBatch(
        task_ids=expanded_tasks,
        actions=actions,
        old_logprobs=old_logprobs,
        rewards=rewards,
        group_size=group_size,
    )


def update_policy(
    policy: TinyTurnPolicy,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
) -> Tuple[float, float]:
    """Apply one GRPO-style, globally token-normalized update."""

    grouped_rewards = batch.rewards.reshape(-1, batch.group_size)
    trajectory_advantages = group_standardized_advantages(grouped_rewards).reshape(-1)
    token_advantages = trajectory_advantages[:, None].expand_as(batch.old_logprobs)
    current_logprobs = policy.logprobs(batch.task_ids, batch.actions)
    action_mask = torch.ones_like(current_logprobs, dtype=torch.bool)

    output = clipped_policy_loss(
        current_logprobs,
        batch.old_logprobs,
        token_advantages,
        action_mask,
        clip_low=0.2,
    )
    optimizer.zero_grad(set_to_none=True)
    output.loss.backward()
    optimizer.step()
    return output.loss.item(), batch.rewards.mean().item()


@torch.no_grad()
def greedy_accuracy(policy: TinyTurnPolicy) -> float:
    task_ids = torch.arange(TARGET_PATHS.shape[0])
    actions = policy.action_logits(task_ids).argmax(dim=-1)
    return (actions == TARGET_PATHS).all(dim=-1).float().mean().item()


def train(steps: int, group_size: int, learning_rate: float, seed: int) -> float:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    policy = TinyTurnPolicy(
        num_tasks=TARGET_PATHS.shape[0], horizon=TARGET_PATHS.shape[1]
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    task_ids = torch.arange(TARGET_PATHS.shape[0])

    for step in range(1, steps + 1):
        rollout = collect_rollouts(
            policy, task_ids, group_size=group_size, generator=generator
        )
        loss, sampled_success = update_policy(policy, optimizer, rollout)
        if step == 1 or step % 25 == 0 or step == steps:
            print(
                f"step={step:04d} loss={loss:+.4f} "
                f"sample_success={sampled_success:.3f} "
                f"greedy_success={greedy_accuracy(policy):.3f}"
            )
    return greedy_accuracy(policy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    final_accuracy = train(
        args.steps, args.group_size, args.learning_rate, args.seed
    )
    if final_accuracy < 1.0:
        raise SystemExit(f"training did not solve all tasks: {final_accuracy:.3f}")


if __name__ == "__main__":
    main()
