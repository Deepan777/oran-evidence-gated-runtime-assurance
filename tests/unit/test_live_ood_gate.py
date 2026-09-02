from __future__ import annotations

import numpy as np

from experiments.live.run_xapp_loop import _out_of_distribution, _physical_plausibility_violations
from src.environment.iq_calibrated_env import FEATURE_NAMES


def test_previous_action_one_hot_value_one_is_valid_transition() -> None:
    low = np.zeros(len(FEATURE_NAMES), dtype=float)
    high = np.zeros(len(FEATURE_NAMES), dtype=float)
    state = {name: 0.0 for name in FEATURE_NAMES}
    state["previous_action_a1"] = 1.0
    assert _out_of_distribution(state, low, high) == {}


def test_previous_action_one_hot_rejects_malformed_value() -> None:
    low = np.zeros(len(FEATURE_NAMES), dtype=float)
    high = np.zeros(len(FEATURE_NAMES), dtype=float)
    state = {name: 0.0 for name in FEATURE_NAMES}
    state["previous_action_a1"] = 0.5
    violations = _out_of_distribution(state, low, high)
    assert violations["previous_action_a1"]["value"] == 0.5


def _plausible_state() -> dict[str, float]:
    state = {name: 0.0 for name in FEATURE_NAMES}
    state.update({
        "jammer_class_none": 1.0,
        "cqi": 15.0,
        "sinr_db": -5.5,
        "rsrp_dbfs": 8.7,
        "scheduler_queue_backlog_bytes": 660_362.0,
        "rlc_sdu_delay_ms": 5_685.0,
        "previous_action_a0": 1.0,
    })
    return state


def test_physical_guard_accepts_large_but_real_outage_queue_and_delay() -> None:
    assert _physical_plausibility_violations(_plausible_state()) == {}


def test_physical_guard_rejects_impossible_ratio_and_non_one_hot_action() -> None:
    state = _plausible_state()
    state["bler"] = 1.1
    state["previous_action_a1"] = 1.0
    violations = _physical_plausibility_violations(state)
    assert violations["bler"]["value"] == 1.1
    assert violations["previous_action_one_hot_sum"]["value"] == 2.0
