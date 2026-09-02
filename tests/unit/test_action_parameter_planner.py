from __future__ import annotations

from src.controller.action_mask import Action, ActionContext
from src.xapp.adapters.parameter_planner import ActionParameterPolicy, plan_action_parameters


def _context() -> ActionContext:
    return ActionContext(True, True, 0.5, 0.15, 12, 2, 0.2, 0.7, 0.1, 0.4)


def test_a1_selects_larger_clean_contiguous_region() -> None:
    result = plan_action_parameters(Action.PRB_AVOID_SHIFT, _context(), 0.2, 0.4, ActionParameterPolicy(50))
    assert result == {"prb_start": 25, "prb_stop": 50}


def test_a2_and_a3_are_bounded() -> None:
    policy = ActionParameterPolicy(52)
    assert plan_action_parameters(Action.MCS_BACKOFF, _context(), 0.2, 0.4, policy) == {"mcs_cap": 8}
    assert plan_action_parameters(Action.SLICE_PRB_BOOST, _context(), 0.2, 0.4, policy) == {
        "slice_min_pct": 30, "slice_max_pct": 70
    }


def test_a4_is_one_composite_parameter_document() -> None:
    result = plan_action_parameters(Action.COMPOSITE_LOCAL_DEFENSE, _context(), 0.2, 0.4, ActionParameterPolicy(50))
    assert result == {"prb_start": 25, "prb_stop": 50, "mcs_cap": 8}
