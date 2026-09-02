from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class Action(IntEnum):
    NO_OP = 0
    PRB_AVOID_SHIFT = 1
    MCS_BACKOFF = 2
    SLICE_PRB_BOOST = 3
    COMPOSITE_LOCAL_DEFENSE = 4


@dataclass(frozen=True)
class ActionContext:
    protected_session_active: bool
    localized_jammer_region_available: bool
    clean_prb_fraction: float
    clean_prb_fraction_min: float
    current_mcs_index: int
    minimum_mcs_index: int
    protected_quota_fraction: float
    protected_quota_max_fraction: float
    quota_increment_fraction: float
    total_allocated_quota_fraction: float

    def validate(self) -> None:
        fraction_fields = (
            self.clean_prb_fraction,
            self.clean_prb_fraction_min,
            self.protected_quota_fraction,
            self.protected_quota_max_fraction,
            self.quota_increment_fraction,
            self.total_allocated_quota_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in fraction_fields):
            raise ValueError("action-mask fractions must lie in [0, 1]")


def compute_action_mask(context: ActionContext) -> NDArray[np.bool_]:
    context.validate()
    mask = np.zeros(len(Action), dtype=np.bool_)
    mask[Action.NO_OP] = True

    can_avoid = (
        context.protected_session_active
        and context.localized_jammer_region_available
        and context.clean_prb_fraction >= context.clean_prb_fraction_min
    )
    can_backoff = (
        context.protected_session_active
        and context.current_mcs_index > context.minimum_mcs_index
    )
    resulting_protected_quota = (
        context.protected_quota_fraction + context.quota_increment_fraction
    )
    resulting_total_quota = (
        context.total_allocated_quota_fraction + context.quota_increment_fraction
    )
    can_boost = (
        context.protected_session_active
        and resulting_protected_quota <= context.protected_quota_max_fraction
        and resulting_total_quota <= 1.0
    )

    mask[Action.PRB_AVOID_SHIFT] = can_avoid
    mask[Action.MCS_BACKOFF] = can_backoff
    mask[Action.SLICE_PRB_BOOST] = can_boost
    mask[Action.COMPOSITE_LOCAL_DEFENSE] = can_avoid and can_backoff
    return mask


def masked_argmax(q_values: NDArray[np.floating], mask: NDArray[np.bool_]) -> Action:
    values = np.asarray(q_values, dtype=np.float64)
    feasibility = np.asarray(mask, dtype=np.bool_)
    if values.shape != (len(Action),) or feasibility.shape != (len(Action),):
        raise ValueError("q_values and mask must have one entry per action")
    if not feasibility[Action.NO_OP]:
        raise ValueError("A0 NO_OP must always be feasible")
    masked = np.where(feasibility, values, -np.inf)
    return Action(int(np.argmax(masked)))

