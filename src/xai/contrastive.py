from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.controller.md3qn import mask_q_values
from src.xai.attribution import integrated_gradients


@dataclass(frozen=True)
class ContrastiveExplanation:
    selected_action: int
    alternative_action: int
    q_values: tuple[float, ...]
    q_gap: float
    contributions: tuple[float, ...]


def select_top_two_feasible(q_values: Tensor, feasible_mask: Tensor) -> tuple[Tensor, Tensor]:
    masked = mask_q_values(q_values, feasible_mask)
    if torch.any(feasible_mask.sum(dim=-1) < 2):
        raise ValueError("contrastive explanation requires at least two feasible actions")
    top_two = masked.topk(k=2, dim=-1).indices
    return top_two[:, 0], top_two[:, 1]


def contrastive_attribution(
    q_network: torch.nn.Module,
    state: Tensor,
    feasible_mask: Tensor,
    baseline: Tensor | None = None,
    steps: int = 64,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    with torch.no_grad():
        q_values = q_network(state)
        selected, alternative = select_top_two_feasible(q_values, feasible_mask)

    def q_gap_function(inputs: Tensor) -> Tensor:
        current_q = q_network(inputs)
        selected_q = current_q.gather(1, selected[:, None]).squeeze(1)
        alternative_q = current_q.gather(1, alternative[:, None]).squeeze(1)
        return selected_q - alternative_q

    contributions = integrated_gradients(
        q_gap_function,
        state,
        baseline=baseline,
        steps=steps,
    )
    with torch.no_grad():
        q_gap = q_gap_function(state)
    return contributions, selected, alternative, q_gap

