from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def integrated_gradients(
    scalar_function: Callable[[Tensor], Tensor],
    state: Tensor,
    baseline: Tensor | None = None,
    steps: int = 64,
) -> Tensor:
    """Integrated gradients for a batched scalar-valued network function."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if state.ndim != 2:
        raise ValueError("state must have shape [batch, features]")
    reference = torch.zeros_like(state) if baseline is None else baseline
    if reference.shape != state.shape:
        raise ValueError("baseline must have the same shape as state")

    difference = state - reference
    total_gradient = torch.zeros_like(state)
    for alpha in torch.linspace(0.0, 1.0, steps + 1, device=state.device)[1:]:
        interpolated = (reference + alpha * difference).detach().requires_grad_(True)
        output = scalar_function(interpolated)
        if output.shape != (state.shape[0],):
            raise ValueError("scalar_function must return one scalar per batch row")
        gradient = torch.autograd.grad(output.sum(), interpolated)[0]
        total_gradient += gradient
    return difference * total_gradient / float(steps)


def selected_action_attribution(
    q_network: torch.nn.Module,
    state: Tensor,
    selected_actions: Tensor,
    baseline: Tensor | None = None,
    steps: int = 64,
) -> Tensor:
    if selected_actions.shape != (state.shape[0],):
        raise ValueError("selected_actions must contain one action per state")

    def target(inputs: Tensor) -> Tensor:
        return q_network(inputs).gather(1, selected_actions[:, None]).squeeze(1)

    return integrated_gradients(target, state, baseline=baseline, steps=steps)

