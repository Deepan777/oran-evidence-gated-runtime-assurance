from __future__ import annotations

import torch
from torch import nn

from src.controller.md3qn import DuelingQNetwork, double_dqn_targets, mask_q_values


def test_dueling_aggregation_has_expected_shape_and_finite_values() -> None:
    torch.manual_seed(1)
    network = DuelingQNetwork(state_dim=7, action_count=5, hidden_units=(16, 16))
    output = network(torch.randn(4, 7))
    assert output.shape == (4, 5)
    assert torch.isfinite(output).all()


def test_mask_q_values_excludes_infeasible_action() -> None:
    q_values = torch.tensor([[0.0, 100.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, False, True, False, False]])
    selected = mask_q_values(q_values, mask).argmax(dim=-1)
    assert int(selected.item()) == 2


class FixedNetwork(nn.Module):
    def __init__(self, values: list[float]):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.values.expand(states.shape[0], -1)


def test_double_dqn_uses_online_argmax_and_target_value() -> None:
    online = FixedNetwork([0.0, 5.0, 4.0, 3.0, 2.0])
    target = FixedNetwork([0.0, 10.0, 20.0, 30.0, 40.0])
    next_states = torch.zeros(2, 3)
    masks = torch.tensor(
        [[True, True, True, True, True], [True, False, True, True, True]]
    )
    rewards = torch.tensor([1.0, 2.0])
    terminals = torch.tensor([False, True])
    targets = double_dqn_targets(
        online, target, next_states, masks, rewards, terminals, gamma=0.5
    )
    assert torch.allclose(targets, torch.tensor([6.0, 2.0]))

