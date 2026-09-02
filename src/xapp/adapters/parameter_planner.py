"""Deterministic state-to-control parameter mapping for A0--A4."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from src.controller.action_mask import Action, ActionContext


@dataclass(frozen=True)
class ActionParameterPolicy:
    cell_prbs: int
    prb_guard_fraction: float = 0.10
    mcs_backoff: int = 4
    minimum_mcs: int = 2
    quota_increment_fraction: float = 0.10
    quota_max_fraction: float = 0.70

    def __post_init__(self) -> None:
        if self.cell_prbs <= 0 or self.mcs_backoff < 0 or self.minimum_mcs < 0:
            raise ValueError("invalid action parameter policy")
        for value in (self.prb_guard_fraction, self.quota_increment_fraction, self.quota_max_fraction):
            if not 0.0 <= value <= 1.0:
                raise ValueError("policy fractions must lie in [0, 1]")


def _clean_prb_region(
    start_norm: float,
    end_norm: float,
    policy: ActionParameterPolicy,
) -> tuple[int, int]:
    if not 0.0 <= start_norm < end_norm <= 1.0:
        raise ValueError("localized jammer interval must satisfy 0 <= start < end <= 1")
    jam_start = math.floor(start_norm * policy.cell_prbs)
    jam_stop = math.ceil(end_norm * policy.cell_prbs)
    guard = math.ceil(policy.prb_guard_fraction * policy.cell_prbs)
    left = (0, max(0, jam_start - guard))
    right = (min(policy.cell_prbs, jam_stop + guard), policy.cell_prbs)
    selected = right if right[1] - right[0] > left[1] - left[0] else left
    if selected[0] >= selected[1]:
        raise ValueError("no contiguous clean PRB region remains after guard allocation")
    return selected


def plan_action_parameters(
    action: Action,
    context: ActionContext,
    jammer_subband_start_norm: float,
    jammer_subband_end_norm: float,
    policy: ActionParameterPolicy,
) -> Mapping[str, object]:
    """Return only parameters consumed by OCUDU's atomic action command."""
    if action is Action.NO_OP:
        return {}
    if action in (Action.PRB_AVOID_SHIFT, Action.COMPOSITE_LOCAL_DEFENSE):
        prb_start, prb_stop = _clean_prb_region(jammer_subband_start_norm, jammer_subband_end_norm, policy)
    if action in (Action.MCS_BACKOFF, Action.COMPOSITE_LOCAL_DEFENSE):
        mcs_cap = max(policy.minimum_mcs, context.current_mcs_index - policy.mcs_backoff)
    if action is Action.PRB_AVOID_SHIFT:
        return {"prb_start": prb_start, "prb_stop": prb_stop}
    if action is Action.MCS_BACKOFF:
        return {"mcs_cap": mcs_cap}
    if action is Action.SLICE_PRB_BOOST:
        boosted = min(
            policy.quota_max_fraction,
            context.protected_quota_fraction + policy.quota_increment_fraction,
        )
        return {
            "slice_min_pct": round(100 * boosted),
            "slice_max_pct": round(100 * policy.quota_max_fraction),
        }
    return {"prb_start": prb_start, "prb_stop": prb_stop, "mcs_cap": mcs_cap}
