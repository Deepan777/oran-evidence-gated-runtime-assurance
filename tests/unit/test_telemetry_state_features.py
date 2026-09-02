from __future__ import annotations

import pytest

from src.xapp.telemetry.state_features import rlc_observation_from_ocudu, state_observation_from_ocudu


def test_ocudu_scheduler_feature_derivation_uses_count_ratios() -> None:
    payload = {"cells": [{
        "cell_metrics": {"pucch_tot_rb_usage_avg": 10.6},
        "ue_list": [{"ue": 0, "rnti": 17921, "cqi": 12, "dl_nof_ok": 90,
                     "dl_nof_nok": 10, "dl_brate": 4_000_000, "bsr": 1000,
                     "dl_bs": 2000, "dl_mcs": 17}],
    }]}
    result = state_observation_from_ocudu(123, payload)
    assert result is not None
    assert result.values["harq_nack_ratio"] == pytest.approx(0.1)
    assert result.values["gnb_dl_throughput_mbps"] == pytest.approx(4.0)
    assert result.values["scheduler_queue_backlog_bytes"] == pytest.approx(3000)
    assert result.values["prb_utilization"] == pytest.approx(0.1)


def test_ocudu_rlc_feature_derivation() -> None:
    result = rlc_observation_from_ocudu({"rlc_metrics": {"tx": {
        "num_sdus": 98, "num_dropped_sdus": 1, "num_discarded_sdus": 1,
        "sum_sdu_latency_us": 196_000,
    }}})
    assert result is not None
    assert result["rlc_drop_rate"] == pytest.approx(0.02)
    assert result["rlc_sdu_delay_ms"] == pytest.approx(2.0)


def test_ocudu_rlc_does_not_fabricate_delay_when_no_sdu_was_transmitted() -> None:
    result = rlc_observation_from_ocudu({"rlc_metrics": {"tx": {
        "num_sdus": 0, "num_dropped_sdus": 0, "num_discarded_sdus": 0,
        "sum_sdu_latency_us": 4_000_000_000,
    }}})
    assert result is not None
    assert result["rlc_drop_rate"] == 0.0
    assert "rlc_sdu_delay_ms" not in result
