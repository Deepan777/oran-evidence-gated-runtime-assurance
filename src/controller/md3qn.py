from __future__ import annotations

import torch
from torch import Tensor, nn


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_count: int = 5,
        hidden_units: tuple[int, int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_count <= 0:
            raise ValueError("state_dim and action_count must be positive")
        layers: list[nn.Module] = []
        input_dim = state_dim
        for width in hidden_units:
            layers.extend((nn.Linear(input_dim, width), activation()))
            input_dim = width
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(input_dim, 1)
        self.advantage_head = nn.Linear(input_dim, action_count)

    def forward(self, state: Tensor) -> Tensor:
        representation = self.trunk(state)
        value = self.value_head(representation)
        advantage = self.advantage_head(representation)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


def mask_q_values(q_values: Tensor, feasible_mask: Tensor) -> Tensor:
    if q_values.shape != feasible_mask.shape:
        raise ValueError("q_values and feasible_mask must have identical shapes")
    if feasible_mask.dtype is not torch.bool:
        raise TypeError("feasible_mask must be boolean")
    if torch.any(~feasible_mask[..., 0]):
        raise ValueError("A0 NO_OP must always be feasible")
    floor = torch.finfo(q_values.dtype).min
    return q_values.masked_fill(~feasible_mask, floor)


@torch.no_grad()
def double_dqn_targets(
    online_network: nn.Module,
    target_network: nn.Module,
    next_states: Tensor,
    next_masks: Tensor,
    rewards: Tensor,
    terminals: Tensor,
    gamma: float,
) -> Tensor:
    online_next_q = mask_q_values(online_network(next_states), next_masks)
    next_actions = online_next_q.argmax(dim=-1, keepdim=True)
    target_next_q = target_network(next_states).gather(1, next_actions).squeeze(1)
    not_terminal = (~terminals.bool()).to(rewards.dtype)
    return rewards + gamma * not_terminal * target_next_q

