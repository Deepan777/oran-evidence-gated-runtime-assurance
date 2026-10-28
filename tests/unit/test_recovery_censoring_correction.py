from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.recovery_censoring_correction import derive_run


def _records(*, recover: bool) -> pd.DataFrame:
    rows = []
    for index in range(120):
        phase = "pre_jam" if index < 20 else "jammer_on" if index < 100 else "recovery"
        throughput = 10.0 if phase == "pre_jam" else 1.0
        delay = 5.0
        if recover and 100 <= index <= 102:
            throughput = 9.0
        rows.append({
            "window_index": index,
            "phase": phase,
            "dl_throughput_mbps": throughput,
            "rlc_sdu_delay_ms": delay,
            "telemetry_observability": "complete_measured",
            "rlc_drop_rate": 0.0,
        })
    return pd.DataFrame(rows)


def test_recovery_event_time_is_end_of_third_qualifying_window() -> None:
    outcome = derive_run(_records(recover=True))
    assert outcome["recovery_event_by_20s"] == 1
    assert outcome["recovery_event_time_seconds"] == 3.0
    assert outcome["recovery_censored_at_seconds"] is None
    assert outcome["restricted_time_to_recovery_20s"] == 3.0


def test_non_recovery_is_explicitly_censored_not_observed_at_horizon() -> None:
    outcome = derive_run(_records(recover=False))
    assert outcome["recovery_event_by_20s"] == 0
    assert outcome["recovery_event_time_seconds"] is None
    assert outcome["recovery_censored_at_seconds"] == 20.0
    assert outcome["restricted_time_to_recovery_20s"] == 20.0
    assert np.isfinite(outcome["protected_flow_latency_violation_rate"])
