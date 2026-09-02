from __future__ import annotations

import json

from experiments.collection.run_manifest import _complete, _sudo_environment


def test_sudo_environment_is_allowlisted_and_preserves_required_values() -> None:
    source = {"RUN_ID": "run-1", "SCENARIO_ID": "scenario-1", "JAMMER": "none", "JSR_DB": "0", "SEED": "1001", "TRAFFIC_LOAD": "low", "SERVICE_MIX": "protected_only", "ATTENUATION_DB": "0", "RUN_SECONDS": "30", "PRE_JAM_SECONDS": "5", "JAMMER_ON_SECONDS": "20", "RECOVERY_SECONDS": "5", "PROTECTED_MBPS": "0.3", "BACKGROUND_MBPS": "0", "EVIDENCE_CLASS": "development_live_calibration", "GNB_IMAGE": "gnb:test", "UE_IMAGE": "ue:pinned", "MAX_WALL_SECONDS": "900", "REQUIRE_COMPLETE_TELEMETRY": "1", "UNRELATED_SECRET": "must-not-pass"}
    values = _sudo_environment(source)
    assert "RUN_ID=run-1" in values
    assert "UE_IMAGE=ue:pinned" in values
    assert not any("UNRELATED_SECRET" in item for item in values)


def _write_complete_evidence(tmp_path, *, clock_fill: int = 0, uplink_silence: int = 0) -> None:
    (tmp_path / "records.summary.json").write_text(json.dumps({
        "rows": 60,
        "complete_rows": 60,
        "radio_window_covered_rows": 60,
    }), encoding="utf-8")
    (tmp_path / "radio-clock-status.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    (tmp_path / "records.parquet").write_bytes(b"parquet-placeholder")
    (tmp_path / "broker_status.jsonl").write_text(json.dumps({
        "event": "broker_stop",
        "downlink_clock_fill_frames": clock_fill,
        "uplink_silence_frames": uplink_silence,
    }) + "\n", encoding="utf-8")


def test_complete_accepts_only_real_radio_frames(tmp_path) -> None:
    _write_complete_evidence(tmp_path)
    assert _complete(tmp_path, 60)


def test_complete_rejects_downlink_clock_fill(tmp_path) -> None:
    _write_complete_evidence(tmp_path, clock_fill=1)
    assert not _complete(tmp_path, 60)


def test_complete_rejects_uplink_silence(tmp_path) -> None:
    _write_complete_evidence(tmp_path, uplink_silence=1)
    assert not _complete(tmp_path, 60)


def test_complete_rejects_evidence_from_a_different_method(tmp_path) -> None:
    _write_complete_evidence(tmp_path)
    (tmp_path / "run_metadata.json").write_text(json.dumps({
        "run_id": "shared-run",
        "scenario_id": "scenario-1",
        "live_xapp_enabled": 0,
        "live_xapp_controller_kind": "md3qn_per",
    }), encoding="utf-8")
    expected = {
        "run_id": "shared-run",
        "scenario_id": "scenario-1",
        "ENABLE_LIVE_XAPP": "1",
        "CONTROLLER_KIND": "heuristic",
    }
    assert not _complete(tmp_path, 60, expected)


def test_complete_rejects_evidence_from_a_different_ue_runtime(tmp_path) -> None:
    _write_complete_evidence(tmp_path)
    (tmp_path / "run_metadata.json").write_text(json.dumps({
        "run_id": "runtime-pinned-run",
        "scenario_id": "scenario-1",
        "live_xapp_enabled": 0,
        "ue_image": "ue:old",
    }), encoding="utf-8")
    expected = {
        "run_id": "runtime-pinned-run",
        "scenario_id": "scenario-1",
        "ENABLE_LIVE_XAPP": "0",
        "UE_IMAGE": "ue:corrected",
    }
    assert not _complete(tmp_path, 60, expected)


def test_complete_requires_one_acknowledged_decision_per_window(tmp_path) -> None:
    _write_complete_evidence(tmp_path)
    (tmp_path / "run_metadata.json").write_text(json.dumps({
        "run_id": "b2-run",
        "scenario_id": "scenario-1",
        "live_xapp_enabled": 1,
        "live_xapp_controller_kind": "heuristic",
    }), encoding="utf-8")
    decision = {
        "status": "DECISION",
        "decision": {"execution": {"acknowledged": True, "failure": None}},
    }
    (tmp_path / "live-xapp-decisions.jsonl").write_text(
        "".join(json.dumps(decision) + "\n" for _ in range(60)), encoding="utf-8"
    )
    expected = {
        "run_id": "b2-run",
        "scenario_id": "scenario-1",
        "ENABLE_LIVE_XAPP": "1",
        "CONTROLLER_KIND": "heuristic",
    }
    assert _complete(tmp_path, 60, expected)
    failed = dict(decision)
    failed["decision"] = {"execution": {"acknowledged": False, "failure": "timeout"}}
    (tmp_path / "live-xapp-decisions.jsonl").write_text(
        "".join(json.dumps(failed if index == 59 else decision) + "\n" for index in range(60)),
        encoding="utf-8",
    )
    assert not _complete(tmp_path, 60, expected)


def test_complete_requires_exact_xai_rows_when_frozen_method_requests_xai(tmp_path) -> None:
    _write_complete_evidence(tmp_path)
    (tmp_path / "run_metadata.json").write_text(json.dumps({
        "run_id": "p-run",
        "scenario_id": "scenario-1",
        "live_xapp_enabled": 1,
        "live_xapp_controller_kind": "md3qn_per",
    }), encoding="utf-8")
    decisions = [{
        "window_index": index,
        "status": "DECISION",
        "decision": {"execution": {"acknowledged": True, "failure": None}},
    } for index in range(60)]
    (tmp_path / "live-xapp-decisions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in decisions), encoding="utf-8"
    )
    expected = {
        "run_id": "p-run", "scenario_id": "scenario-1",
        "ENABLE_LIVE_XAPP": "1", "LIVE_XAPP_XAI": "1",
        "CONTROLLER_KIND": "md3qn_per",
    }
    assert not _complete(tmp_path, 60, expected)
    explanations = [
        {"window_index": index, "status": "XAI_COMPLETE"} for index in range(60)
    ]
    (tmp_path / "live-xapp-xai.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in explanations), encoding="utf-8"
    )
    assert _complete(tmp_path, 60, expected)
    explanations[-1]["window_index"] = 58
    (tmp_path / "live-xapp-xai.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in explanations), encoding="utf-8"
    )
    assert not _complete(tmp_path, 60, expected)
