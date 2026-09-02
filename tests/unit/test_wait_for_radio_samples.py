from __future__ import annotations

import json
from pathlib import Path

from tools.wait_for_radio_samples import wait_for_samples


def test_wait_for_samples_uses_only_post_marker_truth(tmp_path: Path) -> None:
    (tmp_path / "broker_status.jsonl").write_text(
        json.dumps({"event": "experiment_clock_started", "timestamp_ns": 100}) + "\n",
        encoding="utf-8",
    )
    records = [
        {"timestamp_ns": 99, "sample_index_end": 99_999},
        {"timestamp_ns": 101, "sample_index_end": 50},
        {"timestamp_ns": 102, "sample_index_end": 100},
    ]
    (tmp_path / "truth.jsonl").write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    result = wait_for_samples(tmp_path, target_samples=100, timeout_seconds=0.01)
    assert result["status"] == "PASS"
    assert result["experiment_started_timestamp_ns"] == 100
    assert result["observed_samples"] == 100


def test_wait_for_samples_times_out_without_marker(tmp_path: Path) -> None:
    result = wait_for_samples(tmp_path, target_samples=1, timeout_seconds=0.01)
    assert result["status"] == "TIMEOUT"


def test_wait_for_samples_fails_fast_when_broker_times_out(tmp_path: Path) -> None:
    (tmp_path / "broker_status.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "experiment_clock_started", "timestamp_ns": 100}),
                json.dumps({"event": "uplink_timeout", "timestamp_ns": 200}),
                json.dumps({"event": "broker_stop", "timestamp_ns": 201}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "truth.jsonl").write_text(
        json.dumps({"timestamp_ns": 101, "sample_index_end": 50}) + "\n",
        encoding="utf-8",
    )
    result = wait_for_samples(tmp_path, target_samples=100, timeout_seconds=1.0)
    assert result["status"] == "BROKER_TERMINATED"
    assert result["terminal_event"] == "uplink_timeout"
