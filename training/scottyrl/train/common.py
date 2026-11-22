"""Group rollouts and batch assembly. *** STUDENT IMPLEMENTATION FILE ***

These two functions are shared by the sync (Parts 1-2) and async (Part 3)
trainers.
"""

from __future__ import annotations

import asyncio  # noqa: F401  (student dependency)
from typing import Callable

from scottyrl.advantages import compute_group_advantages  # noqa: F401  (student dependency)
from scottyrl.agents.base import Agent
from scottyrl.agents.xml_tool_call import XMLToolCallAgent
from scottyrl.datagen import trajectory_to_training_sequences  # noqa: F401  (student dependency)
from scottyrl.envs.base import Env
from scottyrl.policy import Policy
from scottyrl.rollout import run_episode  # noqa: F401  (student dependency)
from scottyrl.types import Task, TokenSequence, Trajectory


async def collect_group(
    policy: Policy,
    task: Task,
    env_factory: Callable[[Task], Env],
    group_size: int,
    agent_cls: type[Agent] = XMLToolCallAgent,
) -> list[Trajectory]:
    """Run `group_size` independent rollouts of `task` concurrently.

    Contract: every rollout gets its own fresh env instance and fresh agent
    instance (no shared state between rollouts), and the rollouts run
    concurrently so their LLM calls overlap in time. Each rollout is
    `run_episode(agent_cls(policy), env_factory(task))` (see
    scottyrl.rollout); launch all `group_size` of them with `asyncio.gather`.

    Returns: `group_size` trajectories, in rollout order.
    """
    # TODO(student): implement this function.
    jobs = [run_episode(agent_cls(policy), env_factory(task)) for _ in range(group_size)]
    results = await asyncio.gather(*jobs)
    return results


def groups_to_batch(
    groups: list[list[Trajectory]],
    method: str = "grpo",
) -> tuple[list[TokenSequence], dict]:
    """Turn a list of trajectory groups into a flat training batch.

    Contract:
      - Each group's advantages come from your `compute_group_advantages`,
        applied to that group's episode rewards (in rollout order).
      - A degenerate group (advantages None) contributes no sequences and is
        counted, but its rewards still count toward mean_reward.
      - Every remaining trajectory contributes its training sequences (your
        `trajectory_to_training_sequences`) with its own advantage, even
        when that advantage is 0.0.

    Returns: (sequences, stats), where stats has at least these keys:
        "num_groups"              total groups passed in
        "num_degenerate_groups"   groups skipped as degenerate
        "frac_degenerate_groups"  ratio of the two above (0.0 if no groups)
        "mean_reward"             mean episode reward over ALL trajectories,
                                  degenerate groups included (0.0 if there
                                  are none)
        "num_sequences"           len(sequences)
        "num_action_tokens"       total mask==1 positions in the batch
    """
    # TODO(student): implement this function.
    train_sequences: list[TokenSequence] = []
    stats = dict()
    degenerate_groups = 0
    mean_reward = 0.0
    num_rewards = 0
    for g in groups:
        rewards = [traj.reward for traj in g]
        
        if num_rewards == 0:
            mean_reward = sum(rewards) / len(rewards)
            num_rewards = len(rewards)
        else:
            total = num_rewards + len(rewards)
            mean_reward = mean_reward * (num_rewards / total) + sum(rewards) / total
            num_rewards = total

        advantages = compute_group_advantages(rewards, method)
        if advantages:
            for traj, a in zip(g, advantages):
                train_sequences += trajectory_to_training_sequences(traj, a)
        else:
            degenerate_groups += 1

    stats['num_groups'] = len(groups)
    stats['num_degenerate_groups'] = degenerate_groups
    stats['frac_degenerate_groups'] = degenerate_groups / len(groups)
    stats['num_sequences'] = len(train_sequences)
    stats['num_action_tokens'] = sum([sum(s.mask) for s in train_sequences])
    stats['mean_reward'] = mean_reward

    return train_sequences, stats
