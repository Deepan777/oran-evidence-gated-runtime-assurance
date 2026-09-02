from __future__ import annotations

import torch
from torch import nn

from src.xai.attribution import integrated_gradients
from src.xai.contrastive import contrastive_attribution
from src.xai.counterfactual import projected_counterfactual
from src.xai.rationale import FeatureContribution, build_operator_rationale


class LinearQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [1.0, -1.0],
                        [-1.0, 1.0],
                    ]
                )
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)


def test_integrated_gradients_completeness_for_linear_function() -> None:
    state = torch.tensor([[2.0, 3.0]])
    baseline = torch.zeros_like(state)

    def scalar(inputs: torch.Tensor) -> torch.Tensor:
        return 2.0 * inputs[:, 0] - 4.0 * inputs[:, 1]

    attribution = integrated_gradients(scalar, state, baseline=baseline, steps=16)
    assert torch.allclose(attribution, torch.tensor([[4.0, -12.0]]), atol=1e-6)
    assert torch.allclose(attribution.sum(dim=-1), scalar(state) - scalar(baseline))


def test_contrastive_attribution_targets_selected_vs_runner_up_gap() -> None:
    network = LinearQ()
    state = torch.tensor([[2.0, 0.0]])
    mask = torch.tensor([[True, True, True]])
    contributions, selected, alternative, gap = contrastive_attribution(
        network, state, mask, baseline=torch.zeros_like(state), steps=16
    )
    assert int(selected.item()) == 1
    assert int(alternative.item()) == 0
    assert float(gap.item()) == 2.0
    assert torch.allclose(contributions.sum(dim=-1), gap, atol=1e-5)


def test_projected_counterfactual_switches_action_and_respects_immutable_feature() -> None:
    network = LinearQ()
    state = torch.tensor([[2.0, 0.0]])
    feasible = torch.tensor([[True, True, True]])
    result = projected_counterfactual(
        network,
        state,
        feasible,
        target_action=2,
        lower_bounds=torch.tensor([[-3.0, -3.0]]),
        upper_bounds=torch.tensor([[3.0, 3.0]]),
        mutable_mask=torch.tensor([[True, False]]),
        learning_rate=0.1,
        lambda_l1=0.01,
        lambda_margin=10.0,
        maximum_iterations=300,
    )
    assert result.valid
    assert result.selected_action == 2
    assert float(result.state[0, 1]) == 0.0
    assert -3.0 <= float(result.state[0, 0]) <= 3.0


def test_operator_rationale_uses_only_supplied_evidence() -> None:
    text = build_operator_rationale(
        "PRB_AVOID_SHIFT",
        "MCS_BACKOFF",
        0.75,
        (
            FeatureContribution("clean_prb_fraction", 0.4, 0.5),
            FeatureContribution("bler", 0.2, -0.1),
        ),
        "clean_prb_fraction",
        0.4,
        0.1,
    )
    assert "0.75" in text
    assert "0.4" in text
    assert "0.1" in text
    assert "clean_prb_fraction" in text

