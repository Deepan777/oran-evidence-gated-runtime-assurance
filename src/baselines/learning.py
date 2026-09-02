"""Faithful, safety-masked DQN and PPO comparison baselines.

These baselines deliberately omit only the proposed method's enhancements:
standard DQN has neither a dueling head nor Double-DQN/PER, whereas PPO uses a
clipped actor-critic objective.  Both still apply the mandatory feasibility
mask when selecting an action; unsafe actions are never executed merely to
make a baseline look weak.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import torch
from torch import nn

from src.controller.md3qn import mask_q_values
from src.controller.replay import PrioritizedReplayBuffer


class StandardQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_count: int = 5, hidden_units: tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = state_dim
        for units in hidden_units:
            layers.extend((nn.Linear(width, units), nn.ReLU()))
            width = units
        layers.append(nn.Linear(width, action_count))
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


@dataclass(frozen=True)
class BaselineTrainingReport:
    gradient_steps: int
    mean_loss: float


@dataclass(frozen=True)
class PPOTrainingReport:
    updates: int
    mean_loss: float
    collected_steps: int


class MaskedEnvironment(Protocol):
    def reset(self) -> tuple[np.ndarray, object]: ...
    def action_mask(self) -> np.ndarray: ...
    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, object]: ...


def train_standard_dqn(
    buffer: PrioritizedReplayBuffer,
    state_dim: int,
    seed: int,
    *,
    steps: int = 300,
    batch_size: int = 128,
    target_update_steps: int = 1000,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
) -> tuple[StandardQNetwork, BaselineTrainingReport]:
    """Train an ordinary DQN with uniform replay and masked max targets."""
    if buffer.size < batch_size:
        raise ValueError("replay buffer is smaller than batch size")
    if min(steps, batch_size, target_update_steps) <= 0:
        raise ValueError("training counts must be positive")
    torch.manual_seed(seed)
    online, target = StandardQNetwork(state_dim), StandardQNetwork(state_dim)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
    losses: list[float] = []
    for step in range(steps):
        # alpha=0 in the shared buffer makes this sampling uniform; priority
        # updates are intentionally not used by the standard-DQN baseline.
        old_alpha = buffer.alpha
        buffer.alpha = 0.0
        try:
            batch = buffer.sample(batch_size, beta=0.0)
        finally:
            buffer.alpha = old_alpha
        states, actions = torch.from_numpy(batch.states), torch.from_numpy(batch.actions)
        q = online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            max_next = mask_q_values(target(torch.from_numpy(batch.next_states)), torch.from_numpy(batch.next_masks)).max(dim=1).values
            targets = torch.from_numpy(batch.rewards) + gamma * (~torch.from_numpy(batch.terminals)).float() * max_next
        loss = nn.functional.smooth_l1_loss(q, targets)
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online.parameters(), 10.0); optimizer.step()
        losses.append(float(loss.item()))
        if (step + 1) % target_update_steps == 0:
            target.load_state_dict(online.state_dict())
    return online.eval(), BaselineTrainingReport(steps, float(np.mean(losses)))


class MaskedPPOPolicy(nn.Module):
    """Discrete actor-critic with an explicit feasibility-masked categorical policy."""
    def __init__(self, state_dim: int, action_count: int = 5, hidden_units: tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = state_dim
        for units in hidden_units:
            layers.extend((nn.Linear(width, units), nn.ReLU()))
            width = units
        self.trunk = nn.Sequential(*layers)
        self.logits = nn.Linear(width, action_count)
        self.value = nn.Linear(width, 1)

    def distribution(self, states: torch.Tensor, masks: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=mask_q_values(self.logits(self.trunk(states)), masks))

    def act(self, state: np.ndarray, mask: np.ndarray, deterministic: bool = True) -> int:
        with torch.no_grad():
            states = torch.as_tensor(state, dtype=torch.float32)[None, :]
            masks = torch.as_tensor(mask, dtype=torch.bool)[None, :]
            distribution = self.distribution(states, masks)
            return int(distribution.probs.argmax(dim=-1).item()) if deterministic else int(distribution.sample().item())


def ppo_loss(
    policy: MaskedPPOPolicy,
    states: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> torch.Tensor:
    distribution = policy.distribution(states, masks)
    ratio = torch.exp(distribution.log_prob(actions) - old_log_probabilities)
    surrogate = torch.minimum(ratio * advantages, torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages)
    values = policy.value(policy.trunk(states)).squeeze(1)
    return -surrogate.mean() + value_coefficient * nn.functional.mse_loss(values, returns) - entropy_coefficient * distribution.entropy().mean()


def train_masked_ppo(
    environment_factory: Callable[[], MaskedEnvironment],
    state_dim: int,
    seed: int,
    *,
    rollout_steps: int = 256,
    updates: int = 20,
    epochs_per_update: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[MaskedPPOPolicy, PPOTrainingReport]:
    """Train the B4 baseline with on-policy, feasibility-masked PPO.

    This trainer is restricted to the supplied environment factory.  It is
    appropriate for offline calibration/pretraining only; it never converts
    a simulator outcome into confirmatory live-testbed evidence.
    """
    if min(rollout_steps, updates, epochs_per_update) <= 0 or learning_rate <= 0:
        raise ValueError("PPO training counts and learning rate must be positive")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must lie in [0, 1]")
    torch.manual_seed(seed)
    policy = MaskedPPOPolicy(state_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    losses: list[float] = []
    collected = 0
    for _ in range(updates):
        env = environment_factory()
        state, _ = env.reset()
        states: list[np.ndarray] = []; masks: list[np.ndarray] = []; actions: list[int] = []
        log_probs: list[float] = []; rewards: list[float] = []; values: list[float] = []; terminals: list[bool] = []
        for _ in range(rollout_steps):
            mask = np.asarray(env.action_mask(), dtype=np.bool_)
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32)[None, :]
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool)[None, :]
                distribution = policy.distribution(state_tensor, mask_tensor)
                action = int(distribution.sample().item())
                value = float(policy.value(policy.trunk(state_tensor)).item())
                log_probability = float(distribution.log_prob(torch.tensor([action])).item())
            next_state, reward, terminal, _ = env.step(action)
            states.append(np.asarray(state, dtype=np.float32)); masks.append(mask); actions.append(action)
            log_probs.append(log_probability); rewards.append(float(reward)); values.append(value); terminals.append(bool(terminal))
            collected += 1
            state = next_state
            if terminal:
                state, _ = env.reset()
        with torch.no_grad():
            bootstrap = float(policy.value(policy.trunk(torch.as_tensor(state, dtype=torch.float32)[None, :])).item())
        advantages = np.zeros(rollout_steps, dtype=np.float32); running = 0.0
        for index in range(rollout_steps - 1, -1, -1):
            next_value = bootstrap if index == rollout_steps - 1 else values[index + 1]
            continuation = 0.0 if terminals[index] else 1.0
            delta = rewards[index] + gamma * next_value * continuation - values[index]
            running = delta + gamma * gae_lambda * continuation * running
            advantages[index] = running
        returns = advantages + np.asarray(values, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-6)
        tensors = (torch.as_tensor(np.asarray(states)), torch.as_tensor(np.asarray(masks)), torch.as_tensor(actions),
                   torch.as_tensor(log_probs), torch.as_tensor(advantages), torch.as_tensor(returns))
        for _ in range(epochs_per_update):
            loss = ppo_loss(policy, *tensors)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 10.0); optimizer.step()
            losses.append(float(loss.item()))
    return policy.eval(), PPOTrainingReport(updates, float(np.mean(losses)), collected)
