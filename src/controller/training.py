"""Small, auditable M-D3QN-PER trainer used for offline pretraining only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.controller.md3qn import DuelingQNetwork, double_dqn_targets, mask_q_values
from src.controller.replay import PrioritizedReplayBuffer


@dataclass(frozen=True)
class TrainingReport:
    gradient_steps: int
    mean_loss: float
    replay_size: int


def train_md3qn_per(
    buffer: PrioritizedReplayBuffer,
    state_dim: int,
    seed: int,
    steps: int = 300,
    batch_size: int = 128,
    *,
    initialization_state_dict: dict[str, torch.Tensor] | None = None,
    target_update_steps: int = 1000,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    grad_clip: float = 10.0,
) -> tuple[DuelingQNetwork, TrainingReport]:
    if buffer.size < batch_size:
        raise ValueError("replay buffer is smaller than batch size")
    if target_update_steps <= 0:
        raise ValueError("target_update_steps must be positive")
    if learning_rate <= 0.0 or not 0.0 <= gamma <= 1.0 or grad_clip <= 0.0:
        raise ValueError("invalid optimizer or Bellman hyperparameter")
    torch.manual_seed(seed)
    online, target = DuelingQNetwork(state_dim), DuelingQNetwork(state_dim)
    if initialization_state_dict is not None:
        online.load_state_dict(initialization_state_dict)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
    losses: list[float] = []
    for step in range(steps):
        batch = buffer.sample(batch_size, beta=0.4 + 0.6 * step / max(1, steps - 1))
        states = torch.from_numpy(batch.states); actions = torch.from_numpy(batch.actions)
        rewards = torch.from_numpy(batch.rewards); next_states = torch.from_numpy(batch.next_states)
        terminals = torch.from_numpy(batch.terminals); next_masks = torch.from_numpy(batch.next_masks)
        predicted = online(states).gather(1, actions[:, None]).squeeze(1)
        targets = double_dqn_targets(online, target, next_states, next_masks, rewards, terminals, gamma)
        td_error = targets - predicted
        loss = (torch.nn.functional.smooth_l1_loss(predicted, targets, reduction="none") * torch.from_numpy(batch.importance_weights)).mean()
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(), grad_clip); optimizer.step()
        buffer.update_priorities(batch.indexes, td_error.detach().numpy())
        losses.append(float(loss.item()))
        if (step + 1) % target_update_steps == 0:
            target.load_state_dict(online.state_dict())
    return online, TrainingReport(steps, float(np.mean(losses)), buffer.size)
