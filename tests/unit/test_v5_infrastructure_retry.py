from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.archive_v5_infrastructure_attempt import classify_attempt


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def fixture_attempt(tmp_path: Path, *, synthetic: bool = False) -> tuple[Path, Path, str]:
    run_id = "livev5-b0-confirmatory-test-s1015"
    project = tmp_path / "project"
    manifest = project / "manifest.csv"
    evidence = project / "results" / "raw" / f"iq-broker-live-{run_id}"
    evidence.mkdir(parents=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ("run_id", "scenario_id", "run_length_seconds")
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"run_id": run_id, "scenario_id": "test", "run_length_seconds": "120"})
    write_json(evidence / "run_metadata.json", {
        "run_id": run_id,
        "scenario_id": "test",
        "sample_rate_hz": 23_040_000,
        "live_xapp_enabled": 0,
    })
    write_json(evidence / "radio-clock-status.json", {
        "status": "BROKER_TERMINATED",
        "terminal_event": "uplink_timeout",
        "observed_samples": 2_304_000_000,
        "target_samples": 2_764_800_000,
    })
    write_json(evidence / "records.summary.json", {"rows": 120, "complete_rows": 100})
    (evidence / "broker_status.jsonl").write_text(json.dumps({
        "downlink_clock_fill_frames": 1 if synthetic else 0,
        "uplink_silence_frames": 0,
    }) + "\n", encoding="utf-8")
    (evidence / "srsue-container.log").write_text(
        "version=25.10.0 signal=11\nsrsRAN crashed\n", encoding="utf-8"
    )
    return project, manifest, run_id


def test_classifies_real_srsue_crash(tmp_path: Path) -> None:
    project, manifest, run_id = fixture_attempt(tmp_path)
    _, audit = classify_attempt(project, manifest, run_id)
    assert audit["failure_class"] == "srsue_signal_11_uplink_termination"
    assert audit["rejected_attempt_is_eligible"] is False
    assert audit["scenario_factors_changed"] is False


def test_refuses_synthetic_broker_frames(tmp_path: Path) -> None:
    project, manifest, run_id = fixture_attempt(tmp_path, synthetic=True)
    with pytest.raises(ValueError, match="no_synthetic_broker_frames"):
        classify_attempt(project, manifest, run_id)


def test_classifies_xapp_trailing_json_write_race_after_full_radio_capture(tmp_path: Path) -> None:
    project, manifest, run_id = fixture_attempt(tmp_path)
    evidence = project / "results" / "raw" / f"iq-broker-live-{run_id}"
    metadata = json.loads((evidence / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["live_xapp_enabled"] = 1
    write_json(evidence / "run_metadata.json", metadata)
    write_json(evidence / "radio-clock-status.json", {
        "status": "PASS",
        "observed_samples": 2_775_168_000,
        "target_samples": 2_764_800_000,
    })
    write_json(evidence / "records.summary.json", {"rows": 120, "complete_rows": 50})
    decision = {
        "status": "DECISION",
        "decision": {"execution": {"acknowledged": True, "failure": None}},
    }
    (evidence / "live-xapp-decisions.jsonl").write_text(
        "".join(json.dumps(decision) + "\n" for _ in range(10)), encoding="utf-8"
    )
    (evidence / "live-xapp.log").write_text(
        "json.decoder.JSONDecodeError: Unterminated string starting at: line 1 column 95\n",
        encoding="utf-8",
    )
    _, audit = classify_attempt(project, manifest, run_id)
    assert audit["failure_class"] == "xapp_concurrent_jsonl_trailing_write_race"
    assert audit["observed_samples"] >= audit["target_samples"]
    assert audit["checks"]["controller_decision_prefix_valid"] is True


def test_classifies_postcapture_record_builder_process_abort(tmp_path: Path) -> None:
    project, manifest, run_id = fixture_attempt(tmp_path)
    evidence = project / "results" / "raw" / f"iq-broker-live-{run_id}"
    metadata = json.loads((evidence / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["live_xapp_enabled"] = 1
    write_json(evidence / "run_metadata.json", metadata)
    write_json(evidence / "radio-clock-status.json", {
        "status": "PASS",
        "observed_samples": 2_776_826_880,
        "target_samples": 2_764_800_000,
    })
    (evidence / "records.summary.json").unlink()
    (evidence / "record-build-summary.json").write_bytes(b"")
    decision = {
        "status": "DECISION",
        "decision": {"execution": {"acknowledged": True, "failure": None}},
    }
    (evidence / "live-xapp-decisions.jsonl").write_text(
        "".join(json.dumps({**decision, "window_index": index}) + "\n" for index in range(120)),
        encoding="utf-8",
    )
    (evidence / "live-xapp.log").write_text("LIVE_XAPP_WINDOWS_COMPLETED=120\n", encoding="utf-8")
    (evidence / "client-exit-status.txt").write_text("live_xapp_exit=0\nradio_clock_exit=0\n", encoding="utf-8")
    _, audit = classify_attempt(project, manifest, run_id)
    assert audit["failure_class"] == "postcapture_record_builder_process_abort"
    assert audit["checks"]["controller_window_grid_complete"] is True
    assert audit["checks"]["xai_was_never_generated"] is True
