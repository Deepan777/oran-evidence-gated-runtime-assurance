from __future__ import annotations

import numpy as np
import torch

from src.controller.action_mask import Action, masked_argmax


def choose_baseline(
    method: str,
    state: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    learned_policy: object | None = None,
) -> Action:
    """Select a baseline action; learned baselines require trained weights."""
    if method == "B0": return Action.NO_OP
    if method == "B1": return Action.PRB_AVOID_SHIFT if mask[Action.PRB_AVOID_SHIFT] else Action.NO_OP
    if method == "B2":
        if state[0] > 0.5 and mask[Action.COMPOSITE_LOCAL_DEFENSE]: return Action.COMPOSITE_LOCAL_DEFENSE
        if state[0] > 0.25 and mask[Action.MCS_BACKOFF]: return Action.MCS_BACKOFF
        return Action.NO_OP
    if method == "B5":
        if not isinstance(learned_policy, torch.nn.Module):
            raise ValueError("B5 requires a trained PRB-only StandardQNetwork")
        restricted_mask = np.asarray(mask, dtype=bool).copy()
        restricted_mask[Action.MCS_BACKOFF:] = False
        with torch.no_grad():
            q_values = learned_policy(torch.as_tensor(state, dtype=torch.float32)[None, :])
            return Action(int(masked_argmax(q_values[0].cpu().numpy(), restricted_mask)))
    if method == "B3":
        if not isinstance(learned_policy, torch.nn.Module):
            raise ValueError("B3 requires a trained StandardQNetwork")
        with torch.no_grad():
            q_values = learned_policy(torch.as_tensor(state, dtype=torch.float32)[None, :])
            return Action(int(masked_argmax(q_values[0].cpu().numpy(), mask)))
    if method == "B4":
        if learned_policy is None or not hasattr(learned_policy, "act"):
            raise ValueError("B4 requires a trained MaskedPPOPolicy")
        return Action(int(learned_policy.act(state, mask, deterministic=True)))
    raise ValueError(f"unknown baseline: {method}")
