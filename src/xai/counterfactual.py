from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.controller.md3qn import mask_q_values


@dataclass(frozen=True)
class CounterfactualResult:
    state: Tensor
    valid: bool
    selected_action: int
    target_action: int
    weighted_l1: float
    iterations: int


def projected_counterfactual(
    q_network: torch.nn.Module,
    state: Tensor,
    feasible_mask: Tensor,
    target_action: int,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
    mutable_mask: Tensor,
    feature_weights: Tensor | None = None,
    learning_rate: float = 0.03,
    lambda_l1: float = 1.0,
    lambda_margin: float = 5.0,
    target_margin: float = 0.01,
    maximum_iterations: int = 500,
) -> CounterfactualResult:
    """Generate a bounded sparse action-switching counterfactual."""
    if state.shape[0] != 1:
        raise ValueError("counterfactual generation currently requires batch size 1")
    if not bool(feasible_mask[0, target_action]):
        raise ValueError("target_action must be feasible")
    if not (state.shape == lower_bounds.shape == upper_bounds.shape == mutable_mask.shape):
        raise ValueError("state, bounds, and mutable mask must have identical shapes")
    weights = torch.ones_like(state) if feature_weights is None else feature_weights
    if weights.shape != state.shape or torch.any(weights <= 0):
        raise ValueError("feature weights must be positive and match state shape")

    original = state.detach()
    candidate = original.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([candidate], lr=learning_rate)
    immutable = ~mutable_mask.bool()
    iterations_used = 0

    for iteration in range(1, maximum_iterations + 1):
        iterations_used = iteration
        optimizer.zero_grad()
        q_values = q_network(candidate)
        target_q = q_values[:, target_action]
        competitors = q_values.clone()
        competitors[:, target_action] = torch.finfo(q_values.dtype).min
        competitors = mask_q_values(competitors, feasible_mask)
        strongest_other = competitors.max(dim=-1).values
        margin_loss = torch.relu(strongest_other - target_q + target_margin).mean()
        distance = (weights * torch.abs(candidate - original)).sum(dim=-1).mean()
        loss = lambda_l1 * distance + lambda_margin * margin_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            candidate.clamp_(lower_bounds, upper_bounds)
            candidate[immutable] = original[immutable]
            chosen = int(mask_q_values(q_network(candidate), feasible_mask).argmax(dim=-1).item())
            if chosen == target_action and float(margin_loss.item()) <= 1e-6:
                break

    with torch.no_grad():
        final_action = int(mask_q_values(q_network(candidate), feasible_mask).argmax(dim=-1).item())
        weighted_l1 = float((weights * torch.abs(candidate - original)).sum().item())
    return CounterfactualResult(
        state=candidate.detach(),
        valid=final_action == target_action,
        selected_action=final_action,
        target_action=target_action,
        weighted_l1=weighted_l1,
        iterations=iterations_used,
    )

