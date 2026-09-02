from __future__ import annotations

import numpy as np
import pytest

from src.controller.action_mask import Action, ActionContext, compute_action_mask, masked_argmax


def full_context(**overrides: object) -> ActionContext:
    values: dict[str, object] = {
        "protected_session_active": True,
        "localized_jammer_region_available": True,
        "clean_prb_fraction": 0.40,
        "clean_prb_fraction_min": 0.15,
        "current_mcs_index": 12,
        "minimum_mcs_index": 2,
        "protected_quota_fraction": 0.30,
        "protected_quota_max_fraction": 0.70,
        "quota_increment_fraction": 0.10,
        "total_allocated_quota_fraction": 0.80,
    }
    values.update(overrides)
    return ActionContext(**values)  # type: ignore[arg-type]


def test_all_actions_feasible_under_nominal_valid_context() -> None:
    np.testing.assert_array_equal(compute_action_mask(full_context()), np.ones(5, dtype=bool))


def test_no_protected_session_leaves_only_noop() -> None:
    expected = np.asarray([True, False, False, False, False])
    np.testing.assert_array_equal(
        compute_action_mask(full_context(protected_session_active=False)), expected
    )


def test_composite_requires_both_prb_and_mcs_preconditions() -> None:
    mask = compute_action_mask(full_context(clean_prb_fraction=0.05))
    assert not mask[Action.PRB_AVOID_SHIFT]
    assert mask[Action.MCS_BACKOFF]
    assert not mask[Action.COMPOSITE_LOCAL_DEFENSE]


def test_slice_boost_rejects_total_quota_overflow() -> None:
    mask = compute_action_mask(full_context(total_allocated_quota_fraction=0.95))
    assert not mask[Action.SLICE_PRB_BOOST]


def test_masked_argmax_never_selects_invalid_high_q_action() -> None:
    q_values = np.asarray([0.0, 100.0, 2.0, 3.0, 4.0])
    mask = np.asarray([True, False, True, False, False])
    assert masked_argmax(q_values, mask) is Action.MCS_BACKOFF


def test_masked_argmax_rejects_invalid_noop_mask() -> None:
    with pytest.raises(ValueError, match="A0"):
        masked_argmax(np.zeros(5), np.zeros(5, dtype=bool))

