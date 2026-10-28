from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.receiver_metadata_recovery_sensitivity import derive, first_persistent_event, raw_rlc_fields


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "run_id": ["test-run"] * 120,
        "method": ["P"] * 120,
        "source_run_id": ["paired-test"] * 120,
        "window_index": range(120),
        "phase": ["pre_jam"] * 20 + ["jammer_on"] * 80 + ["recovery"] * 20,
        "dl_throughput_mbps": [1.] * 120,
        "rlc_sdu_delay_ms": [2.] * 120,
        "rlc_num_sdus": [1.] * 120,
        "telemetry_observability": ["receiver_detached_outage"] * 120,
    })


def test_event_occurs_at_end_of_third_qualifying_window() -> None:
    assert first_persistent_event(np.array([False, True, True, True])) == 4.
    assert first_persistent_event(np.array([True, True, False, True, True])) is None


def test_sensitivity_can_recover_event_excluded_only_by_old_label() -> None:
    result, windows = derive(frame(), None)
    assert result["old_mask_event_by_20s"] == 0
    assert result["network_only_event_by_20s"] == 1
    assert result["network_only_event_time_seconds"] == 3.
    assert result["newly_qualifying_windows_without_old_mask"] == 20
    assert windows.network_only_qualifies.all()


def test_missing_native_sdu_interval_is_not_a_recovery_event() -> None:
    data = frame()
    data.loc[data.phase.eq("recovery"), "rlc_num_sdus"] = 0.
    result, _ = derive(data, None)
    assert result["network_only_event_by_20s"] == 0
    assert result["network_only_censored_at_seconds"] == 20
    assert result["network_only_event_time_seconds"] is None


def test_endpoint_delivery_check_does_not_infer_packets_from_network_rate() -> None:
    endpoint = {index: {
        "protected_packets": 0., "protected_throughput_mbps": 0., "traffic_coverage_fraction": 1.,
    } for index in range(120)}
    result, _ = derive(frame(), endpoint)
    assert result["network_only_event_by_20s"] == 1
    assert result["endpoint_corroborated_event_by_20s"] == 0
    assert result["endpoint_zero_packet_recovery_windows"] == 20


def test_raw_rlc_counter_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="negative"):
        raw_rlc_fields('{"rlc_metrics":{"tx":{"num_sdus":-1,"num_dropped_sdus":0,"num_discarded_sdus":0,"sum_sdu_latency_us":0}}}')
