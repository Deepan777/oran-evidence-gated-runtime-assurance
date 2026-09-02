from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.live_validation_v9_postprocess import METHODS, derive_run_outcomes, statistics_input


def _radio_run() -> pd.DataFrame:
    throughput = np.concatenate((np.full(20, 10.0), np.full(80, 8.0), np.full(20, 9.5)))
    delay = np.concatenate((np.full(20, 5.0), np.r_[np.full(40, 25.0), np.full(40, 10.0)], np.full(20, 10.0)))
    return pd.DataFrame({
        "window_index": range(120),
        "phase": ["pre_jam"] * 20 + ["jammer_on"] * 80 + ["recovery"] * 20,
        "dl_throughput_mbps": throughput,
        "rlc_sdu_delay_ms": delay,
        "rlc_drop_rate": np.concatenate((np.zeros(20), np.full(80, 0.1), np.zeros(20))),
        "telemetry_observability": ["fully_measured"] * 120,
    })


def test_v9_frozen_run_outcome_formulas() -> None:
    outcomes = derive_run_outcomes(_radio_run())
    assert outcomes["protected_flow_latency_violation_rate"] == 0.5
    assert outcomes["protected_flow_throughput_retention"] == 0.8
    assert outcomes["protected_flow_packet_or_rlc_drop_rate"] == pytest.approx(0.1)
    assert outcomes["recovery_time"] == 0.0


def test_v9_outage_is_latency_violation_without_fabricating_raw_delay() -> None:
    frame = _radio_run()
    outage = frame.index[(frame.phase == "jammer_on")][-8:]
    frame.loc[outage, "rlc_sdu_delay_ms"] = np.nan
    frame.loc[outage, "dl_throughput_mbps"] = 0.0
    frame.loc[outage, "rlc_drop_rate"] = 1.0
    frame.loc[outage, "telemetry_observability"] = "receiver_detached_outage"
    outcomes = derive_run_outcomes(frame)
    assert outcomes["protected_flow_latency_violation_rate"] == 0.60
    assert frame.loc[outage, "rlc_sdu_delay_ms"].isna().all()


def test_v9_statistics_input_uses_matched_blocks_not_windows() -> None:
    rows = []
    for block in range(90):
        for method_index, method in enumerate(METHODS):
            rows.append({
                "source_run_id": f"block-{block:03d}", "method": method,
                "protected_flow_latency_violation_rate": 0.1 + method_index,
                "protected_flow_throughput_retention": 1.0 - method_index * 0.1,
                "protected_flow_packet_or_rlc_drop_rate": 0.01 + method_index * 0.01,
                "recovery_time": float(method_index),
            })
    payload = statistics_input(pd.DataFrame(rows))
    for outcome in payload["outcomes"].values():
        assert outcome["methods"] == list(METHODS)
        assert len(outcome["matched_blocks"]) == 90
        assert len(outcome["matched_blocks"][0]) == 4
