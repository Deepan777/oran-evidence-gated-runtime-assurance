from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.collection.build_live_run_records import build_records
from experiments.live.run_xapp_loop import _out_of_distribution, _read_json_lines, _telemetry_window
from src.environment.iq_calibrated_env import FEATURE_NAMES


def test_live_xapp_reader_ignores_only_unfinished_trailing_json(tmp_path: Path) -> None:
    evidence = tmp_path / "iq.jsonl"
    evidence.write_text('{"window_index": 0}\n{"window_index":', encoding="utf-8")
    assert _read_json_lines(evidence) == [{"window_index": 0}]


def test_live_xapp_reader_rejects_malformed_completed_json(tmp_path: Path) -> None:
    evidence = tmp_path / "iq.jsonl"
    evidence.write_text('{"window_index": 0}\n{"window_index":}\n', encoding="utf-8")
    try:
        _read_json_lines(evidence)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("newline-terminated malformed evidence was accepted")


def test_live_builder_keeps_raw_sources_and_marks_complete_windows(tmp_path: Path) -> None:
    origin = 1_786_630_000_000_000_000
    (tmp_path / "srsue_metrics.csv").write_text(
        "timestamp_epoch_ms;rsrp;rsrq;rssi;dl_snr;dl_brate;dl_bler;ul_buff\n"
        f"{(origin + 1_000_000) / 1_000_000};-63;-11;-52;18;2000000;2;400\n", encoding="utf-8"
    )
    scheduler = {"timestamp_ns": origin + 1_100_000, "payload": {"cells": [{"cell_metrics": {"pucch_tot_rb_usage_avg": 10.6}, "ue_list": [{"ue": 0, "rnti": 1, "cqi": 12, "dl_nof_ok": 90, "dl_nof_nok": 10, "dl_brate": 4_000_000, "bsr": 1000, "dl_bs": 2000, "dl_mcs": 17}]}]}}
    rlc = {"timestamp_ns": origin + 1_200_000, "payload": {"rlc_metrics": {"tx": {"num_sdus": 98, "num_dropped_sdus": 1, "num_discarded_sdus": 1, "sum_sdu_latency_us": 196_000}}}}
    (tmp_path / "ocudu-metrics.jsonl").write_text(json.dumps(scheduler) + "\n" + json.dumps(rlc) + "\n", encoding="utf-8")
    truth = {"timestamp_ns": origin + 30, "sample_index_start": 0, "sample_index_end": 100, "jammer_type": "barrage", "requested_jsr_db": 5, "realized_jsr_db": 5.1, "subband_start_norm": 0.25, "subband_end_norm": 0.5, "active_sample_count": 50, "frame_sample_count": 100}
    (tmp_path / "truth.jsonl").write_text(json.dumps(truth) + "\n", encoding="utf-8")
    metadata = {"run_id": "p1", "scenario_id": "s1", "seed": 1001, "evidence_class": "development_pilot", "measurement_start_ns": origin, "experiment_started_timestamp_ns": origin, "sample_rate_hz": 100, "run_length_seconds": 1, "pre_jam_seconds": 0, "jammer_on_seconds": 1, "recovery_seconds": 0}
    frame = build_records(tmp_path, metadata)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert bool(row["complete_telemetry"])
    assert row["ground_truth_jammer_type"] == "barrage"
    assert row["ground_truth_occupancy_fraction"] == 0.5
    assert row["dl_throughput_mbps"] == 4.0
    assert row["raw_ocudu_scheduler_json"]
    assert not bool(row["complete_controller_state"])


def test_live_builder_keeps_measured_iq_estimator_separate_from_truth(tmp_path: Path) -> None:
    origin = 1_786_630_000_000_000_000
    (tmp_path / "srsue_metrics.csv").write_text(
        "timestamp_epoch_ms;rsrp;rsrq;rssi;dl_snr;dl_brate;dl_bler;ul_buff\n"
        f"{(origin + 1_000_000) / 1_000_000};-63;-11;-52;18;2000000;2;400\n", encoding="utf-8"
    )
    scheduler = {"timestamp_ns": origin + 1_100_000, "payload": {"cells": [{"cell_metrics": {"pucch_tot_rb_usage_avg": 10.6}, "ue_list": [{"ue": 0, "rnti": 1, "cqi": 12, "dl_nof_ok": 90, "dl_nof_nok": 10, "dl_brate": 4_000_000, "bsr": 1000, "dl_bs": 2000, "dl_mcs": 17}]}]}}
    rlc = {"timestamp_ns": origin + 1_200_000, "payload": {"rlc_metrics": {"tx": {"num_sdus": 98, "num_dropped_sdus": 1, "num_discarded_sdus": 1, "sum_sdu_latency_us": 196_000}}}}
    (tmp_path / "ocudu-metrics.jsonl").write_text(json.dumps(scheduler) + "\n" + json.dumps(rlc) + "\n", encoding="utf-8")
    (tmp_path / "iq-spectral-estimator.jsonl").write_text(json.dumps({"timestamp_ns": origin + 1_300_000, "window_index": 0, "clean_prb_fraction": 0.7, "jammer_occupancy_fraction": 0.3, "jammer_subband_start_norm": 0.2, "jammer_subband_end_norm": 0.4, "localized_jammer_region_available": True}) + "\n", encoding="utf-8")
    truth = {"timestamp_ns": origin + 30, "sample_index_start": 0, "sample_index_end": 100, "jammer_type": "barrage", "requested_jsr_db": 5, "realized_jsr_db": 5.1, "subband_start_norm": 0.25, "subband_end_norm": 0.5, "active_sample_count": 50, "frame_sample_count": 100}
    (tmp_path / "truth.jsonl").write_text(json.dumps(truth) + "\n", encoding="utf-8")
    metadata = {"run_id": "p1", "scenario_id": "s1", "seed": 1001, "evidence_class": "development_pilot", "measurement_start_ns": origin, "experiment_started_timestamp_ns": origin, "sample_rate_hz": 100, "run_length_seconds": 1, "pre_jam_seconds": 0, "jammer_on_seconds": 1, "recovery_seconds": 0}
    row = build_records(tmp_path, metadata).iloc[0]
    assert bool(row["complete_controller_state"])
    assert row["clean_prb_fraction"] == 0.7
    assert row["jammer_occupancy_fraction"] == 0.3
    assert row["ground_truth_jammer_type"] == "barrage"


def test_live_builder_prefers_methodology_locked_jammer_truth_filename(tmp_path: Path) -> None:
    origin = 1_786_630_000_000_000_000
    (tmp_path / "srsue_metrics.csv").write_text(
        "timestamp_epoch_ms;rsrp;rsrq;rssi;dl_snr;dl_brate;dl_bler;ul_buff\n"
        f"{(origin + 1_000_000) / 1_000_000};-63;-11;-52;18;2000000;2;400\n", encoding="utf-8"
    )
    scheduler = {"timestamp_ns": origin + 1_100_000, "payload": {"cells": [{"cell_metrics": {"pucch_tot_rb_usage_avg": 10.6}, "ue_list": [{"ue": 0, "rnti": 1, "cqi": 12, "dl_nof_ok": 90, "dl_nof_nok": 10, "dl_brate": 4_000_000, "bsr": 1000, "dl_bs": 2000, "dl_mcs": 17}]}]}}
    rlc = {"timestamp_ns": origin + 1_200_000, "payload": {"rlc_metrics": {"tx": {"num_sdus": 98, "num_dropped_sdus": 1, "num_discarded_sdus": 1, "sum_sdu_latency_us": 196_000}}}}
    (tmp_path / "ocudu-metrics.jsonl").write_text(json.dumps(scheduler) + "\n" + json.dumps(rlc) + "\n", encoding="utf-8")
    (tmp_path / "iq-spectral-estimator.jsonl").write_text(json.dumps({"timestamp_ns": origin + 1_300_000, "window_index": 0, "clean_prb_fraction": 0.7, "jammer_occupancy_fraction": 0.3, "jammer_subband_start_norm": 0.2, "jammer_subband_end_norm": 0.4, "localized_jammer_region_available": True}) + "\n", encoding="utf-8")
    truth = {"timestamp_ns": origin + 30, "sample_index_start": 0, "sample_index_end": 100, "jammer_type": "barrage", "requested_jsr_db": 5, "realized_jsr_db": 5.1, "subband_start_norm": 0.25, "subband_end_norm": 0.5, "active_sample_count": 50, "frame_sample_count": 100}
    (tmp_path / "jammer_truth.jsonl").write_text(json.dumps(truth) + "\n", encoding="utf-8")
    metadata = {"run_id": "p1", "scenario_id": "s1", "seed": 1001, "evidence_class": "development_pilot", "measurement_start_ns": origin, "experiment_started_timestamp_ns": origin, "sample_rate_hz": 100, "run_length_seconds": 1, "pre_jam_seconds": 0, "jammer_on_seconds": 1, "recovery_seconds": 0}
    assert build_records(tmp_path, metadata).iloc[0]["ground_truth_jammer_type"] == "barrage"


def test_live_controller_ood_gate_reports_raw_feature_range_violations() -> None:
    raw_state = {name: 0.0 for name in FEATURE_NAMES}
    raw_state["sinr_db"] = 99.0
    violations = _out_of_distribution(
        raw_state,
        np.full(len(FEATURE_NAMES), -1.0),
        np.full(len(FEATURE_NAMES), 1.0),
    )
    assert violations == {"sinr_db": {"value": 99.0, "training_p01": -1.0, "training_p99": 1.0}}


def test_live_builder_accepts_truthfully_observed_detached_outage_without_filling_raw_metrics(tmp_path: Path) -> None:
    origin = 1_786_630_000_000_000_000
    (tmp_path / "srsue_metrics.csv").write_text(
        "timestamp_epoch_ms;rsrp;rsrq;rssi;dl_snr;dl_brate;dl_bler;ul_buff;is_attached\n"
        f"{(origin + 1_000_000) / 1_000_000};-inf;0;0;-inf;0;0;0;0\n",
        encoding="utf-8",
    )
    scheduler = {
        "timestamp_ns": origin + 1_100_000,
        "payload": {"cells": [{
            "cell_metrics": {"pucch_tot_rb_usage_avg": 0.0},
            "ue_list": [{
                "ue": 0, "rnti": 1, "cqi": 0, "dl_nof_ok": 0, "dl_nof_nok": 0,
                "dl_brate": 0, "bsr": 0, "dl_bs": 22, "dl_mcs": 0,
            }],
        }]},
    }
    rlc = {
        "timestamp_ns": origin + 1_200_000,
        "payload": {"rlc_metrics": {"tx": {
            "num_sdus": 0, "num_dropped_sdus": 10,
            "num_discarded_sdus": 0, "sum_sdu_latency_us": 0,
        }}},
    }
    (tmp_path / "ocudu-metrics.jsonl").write_text(
        json.dumps(scheduler) + "\n" + json.dumps(rlc) + "\n", encoding="utf-8"
    )
    (tmp_path / "iq-spectral-estimator.jsonl").write_text(json.dumps({
        "timestamp_ns": origin + 1_300_000, "window_index": 0,
        "clean_prb_fraction": 1.0, "jammer_occupancy_fraction": 0.0,
        "jammer_subband_start_norm": 0.0, "jammer_subband_end_norm": 0.0,
        "localized_jammer_region_available": False,
    }) + "\n", encoding="utf-8")
    truth = {
        "timestamp_ns": origin + 30, "sample_index_start": 0, "sample_index_end": 100,
        "jammer_type": "none", "requested_jsr_db": None, "realized_jsr_db": None,
        "subband_start_norm": None, "subband_end_norm": None,
        "active_sample_count": 0, "frame_sample_count": 100,
    }
    (tmp_path / "jammer_truth.jsonl").write_text(json.dumps(truth) + "\n", encoding="utf-8")
    metadata = {
        "run_id": "outage-1", "scenario_id": "s1", "seed": 1001,
        "evidence_class": "locked_confirmatory", "measurement_start_ns": origin,
        "experiment_started_timestamp_ns": origin, "sample_rate_hz": 100,
        "run_length_seconds": 1, "pre_jam_seconds": 0,
        "jammer_on_seconds": 0, "recovery_seconds": 1,
    }
    row = build_records(tmp_path, metadata).iloc[0]
    assert not bool(row["complete_telemetry"])
    assert bool(row["evidence_complete"])
    assert row["telemetry_observability"] == "receiver_detached_outage"
    assert row["receiver_attached"] == 0.0
    assert "sinr_db" not in row.index


def test_live_builder_accepts_detached_receiver_with_measured_gnb_rlc_delay(tmp_path: Path) -> None:
    """A detached UE can coexist with a final successful gNB-side buffered SDU."""
    origin = 1_786_630_000_000_000_000
    (tmp_path / "srsue_metrics.csv").write_text(
        "timestamp_epoch_ms;rsrp;rsrq;rssi;dl_snr;dl_brate;dl_bler;ul_buff;is_attached\n"
        f"{(origin + 1_000_000) / 1_000_000};-inf;0;0;-inf;0;0;0;0\n",
        encoding="utf-8",
    )
    scheduler = {
        "timestamp_ns": origin + 1_100_000,
        "payload": {"cells": [{
            "cell_metrics": {"pucch_tot_rb_usage_avg": 0.0},
            "ue_list": [{
                "ue": 0, "rnti": 1, "cqi": -1, "dl_nof_ok": 0, "dl_nof_nok": 0,
                "dl_brate": 0, "bsr": 0, "dl_bs": 124, "dl_mcs": 0,
            }],
        }]},
    }
    rlc = {
        "timestamp_ns": origin + 1_200_000,
        "payload": {"rlc_metrics": {"tx": {
            "num_sdus": 4, "num_dropped_sdus": 0,
            "num_discarded_sdus": 0, "sum_sdu_latency_us": 8_000,
        }}},
    }
    (tmp_path / "ocudu-metrics.jsonl").write_text(
        json.dumps(scheduler) + "\n" + json.dumps(rlc) + "\n", encoding="utf-8"
    )
    (tmp_path / "iq-spectral-estimator.jsonl").write_text(json.dumps({
        "timestamp_ns": origin + 1_300_000, "window_index": 0,
        "clean_prb_fraction": 1.0, "jammer_occupancy_fraction": 0.0,
        "jammer_subband_start_norm": 0.0, "jammer_subband_end_norm": 0.0,
        "localized_jammer_region_available": False,
    }) + "\n", encoding="utf-8")
    (tmp_path / "jammer_truth.jsonl").write_text(json.dumps({
        "timestamp_ns": origin + 30, "sample_index_start": 0, "sample_index_end": 100,
        "jammer_type": "none", "requested_jsr_db": None, "realized_jsr_db": None,
        "subband_start_norm": None, "subband_end_norm": None,
        "active_sample_count": 0, "frame_sample_count": 100,
    }) + "\n", encoding="utf-8")
    metadata = {
        "run_id": "outage-with-late-sdu", "scenario_id": "s1", "seed": 1001,
        "evidence_class": "development_pilot", "measurement_start_ns": origin,
        "experiment_started_timestamp_ns": origin, "sample_rate_hz": 100,
        "run_length_seconds": 1, "pre_jam_seconds": 0,
        "jammer_on_seconds": 0, "recovery_seconds": 1,
    }
    row = build_records(tmp_path, metadata).iloc[0]
    assert not bool(row["complete_telemetry"])
    assert bool(row["evidence_complete"])
    assert row["telemetry_observability"] == "receiver_detached_outage"
    assert row["receiver_attached"] == 0.0
    assert row["rlc_sdu_delay_ms"] == 2.0
    assert "sinr_db" not in row.index


def test_live_xapp_window_keeps_measured_network_sources_when_receiver_rf_is_undefined(tmp_path: Path) -> None:
    origin = 1_786_630_000_000_000_000
    (tmp_path / "srsue.csv").write_text(
        "timestamp_epoch_ms;rsrp;rsrq;rssi;dl_snr;dl_brate;dl_bler;ul_buff;is_attached\n"
        f"{(origin + 1_000_000) / 1_000_000};-inf;0;0;-inf;0;0;0;0\n",
        encoding="utf-8",
    )
    scheduler = {
        "timestamp_ns": origin + 1_100_000,
        "payload": {"cells": [{
            "cell_metrics": {"pucch_tot_rb_usage_avg": 0.0},
            "ue_list": [{
                    "ue": 0, "rnti": 1, "cqi": -1, "dl_nof_ok": 0, "dl_nof_nok": 0,
                "dl_brate": 0, "bsr": 0, "dl_bs": 22, "dl_mcs": 0,
            }],
        }]},
    }
    rlc = {
        "timestamp_ns": origin + 1_200_000,
        "payload": {"rlc_metrics": {"tx": {
            "num_sdus": 0, "num_dropped_sdus": 10,
            "num_discarded_sdus": 0, "sum_sdu_latency_us": 0,
        }}},
    }
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps(scheduler) + "\n" + json.dumps(rlc) + "\n", encoding="utf-8"
    )
    (tmp_path / "sensor.jsonl").write_text(json.dumps({
        "timestamp_ns": origin + 1_300_000, "window_index": 0,
        "clean_prb_fraction": 1.0, "jammer_occupancy_fraction": 0.0,
        "jammer_subband_start_norm": 0.0, "jammer_subband_end_norm": 0.0,
        "localized_jammer_region_available": False,
    }) + "\n", encoding="utf-8")
    found = _telemetry_window(
        tmp_path / "metrics.jsonl", tmp_path / "srsue.csv", tmp_path / "sensor.jsonl",
        origin, origin + 1_000_000_000,
    )
    assert found is not None
    telemetry, context = found
    assert telemetry["dl_throughput_mbps"] == 0.0
    assert telemetry["rlc_drop_rate"] == 1.0
    assert "sinr_db" not in telemetry
    assert "cqi" not in telemetry
    assert context["receiver_unavailable_sentinels"] == {"cqi": -1.0}
    assert context["source_timestamps_ns"]["srsue"] is None


def test_live_xapp_window_excludes_finite_receiver_placeholders_while_detached(tmp_path: Path) -> None:
    origin = 1_786_630_000_000_000_000
    (tmp_path / "srsue.csv").write_text(
        "timestamp_epoch_ms;rsrp;rsrq;rssi;dl_snr;dl_brate;dl_bler;ul_buff;is_attached\n"
        f"{(origin + 1_000_000) / 1_000_000};23;0;0;113;0;0;17525;0\n",
        encoding="utf-8",
    )
    scheduler = {
        "timestamp_ns": origin + 1_100_000,
        "payload": {"cells": [{
            "cell_metrics": {"pucch_tot_rb_usage_avg": 0.0},
            "ue_list": [{
                "ue": 0, "rnti": 1, "cqi": -1, "dl_nof_ok": 0, "dl_nof_nok": 0,
                "dl_brate": 0, "bsr": 0, "dl_bs": 22, "dl_mcs": 0,
            }],
        }]},
    }
    rlc = {
        "timestamp_ns": origin + 1_200_000,
        "payload": {"rlc_metrics": {"tx": {
            "num_sdus": 0, "num_dropped_sdus": 10,
            "num_discarded_sdus": 0, "sum_sdu_latency_us": 0,
        }}},
    }
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps(scheduler) + "\n" + json.dumps(rlc) + "\n", encoding="utf-8"
    )
    (tmp_path / "sensor.jsonl").write_text(json.dumps({
        "timestamp_ns": origin + 1_300_000, "window_index": 0,
        "clean_prb_fraction": 1.0, "jammer_occupancy_fraction": 0.0,
        "jammer_subband_start_norm": 0.0, "jammer_subband_end_norm": 0.0,
        "localized_jammer_region_available": False,
    }) + "\n", encoding="utf-8")

    found = _telemetry_window(
        tmp_path / "metrics.jsonl", tmp_path / "srsue.csv", tmp_path / "sensor.jsonl",
        origin, origin + 1_000_000_000,
    )
    assert found is not None
    telemetry, context = found
    assert "sinr_db" not in telemetry
    assert "rsrp_dbfs" not in telemetry
    assert "rsrq_db" not in telemetry
    assert "bler" not in telemetry
    assert "cqi" not in telemetry
    assert telemetry["dl_throughput_mbps"] == 0.0
    assert context["receiver_attached"] == 0.0
    assert context["receiver_unavailable_sentinels"] == {
        "sinr_db": 113.0,
        "rsrp_dbfs": 23.0,
        "rsrq_db": 0.0,
        "rssi_dbfs": 0.0,
        "ue_dl_throughput_mbps": 0.0,
        "ue_dl_bler": 0.0,
        "ue_ul_backlog_bytes": 17525.0,
        "cqi": -1.0,
    }
