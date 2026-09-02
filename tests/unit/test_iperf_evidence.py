import json
from pathlib import Path

from experiments.collection.iperf_evidence import (
    CONTROLLED_INTERRUPT,
    align_intervals,
    extract_intervals,
    read_iperf_json,
    validate_protected_traffic,
)


def _document(intervals=3):
    return {
        "start": {"connected": [{"local_host": "10.45.0.2", "remote_host": "10.45.0.1"}]},
        "intervals": [
            {"sum": {"start": i, "end": i + 1, "seconds": 1, "bytes": 100,
                     "packets": 10, "lost_packets": 0, "lost_percent": 0,
                     "bits_per_second": 800, "jitter_ms": 0.5, "sender": False}}
            for i in range(intervals)
        ],
        "error": CONTROLLED_INTERRUPT,
    }


def test_reads_diagnostic_prefix_and_aligns_nonuniform_radio_windows(tmp_path: Path):
    path = tmp_path / "iperf.json"
    path.write_text("diagnostic prefix\n" + json.dumps(_document()), encoding="utf-8")
    document = read_iperf_json(path)
    intervals = extract_intervals(document, 1_000_000_000)
    aligned = align_intervals(intervals, [
        (0, 1_000_000_000, 2_500_000_000),
        (1, 2_500_000_000, 4_000_000_000),
    ])
    assert len(aligned) == 2
    assert all(row["traffic_coverage_fraction"] == 1.0 for row in aligned)
    assert all(row["protected_throughput_mbps"] == 0.0008 for row in aligned)


def test_reads_complete_json_with_known_shutdown_diagnostics_suffix(tmp_path: Path):
    path = tmp_path / "iperf.json"
    path.write_text(
        json.dumps(_document()) + "\niperf_json_finish: pthread_mutex_lock: No such file or directory\n",
        encoding="utf-8",
    )
    document = read_iperf_json(path)
    assert document["error"] == CONTROLLED_INTERRUPT
    assert "pthread_mutex_lock" in document["_embedded_diagnostics"][0]


def test_removes_only_named_diagnostic_interleaved_inside_json(tmp_path: Path):
    payload = json.dumps(_document())
    marker = '"sender": false'
    corrupted = payload.replace(
        marker,
        '"sender"iperf_errexit: pthread_mutex_unlock: Invalid argument\n: false',
        1,
    )
    path = tmp_path / "iperf.json"
    path.write_text(corrupted, encoding="utf-8")
    document = read_iperf_json(path)
    assert len(document["intervals"]) == 3
    assert document["_embedded_diagnostics"] == [
        "iperf_errexit: pthread_mutex_unlock: Invalid argument"
    ]


def test_validator_accepts_controlled_interrupt_only_with_full_traffic(tmp_path: Path):
    iperf = tmp_path / "protected.json"
    iperf.write_text(json.dumps(_document()), encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"run_id": "q1", "measurement_start_ns": 1_000_000_000,
                                    "ue_ip": "10.45.0.2", "pre_jam_seconds": 1}), encoding="utf-8")
    sensor = tmp_path / "sensor.jsonl"
    sensor.write_text("\n".join(json.dumps({"window_index": i, "timestamp_ns": (i + 2) * 1_000_000_000})
                                for i in range(2)), encoding="utf-8")
    report = validate_protected_traffic(
        iperf_path=iperf, sensor_path=sensor, metadata_path=metadata,
        traffic_start_ns=1_000_000_000, expected_rows=2,
    )
    assert report["status"] == "PASS"
    assert report["controlled_interrupt"] is True
    assert report["complete_radio_windows"] == 2


def test_validator_rejects_uncovered_radio_window(tmp_path: Path):
    iperf = tmp_path / "protected.json"
    iperf.write_text(json.dumps(_document(intervals=1)), encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"run_id": "q2", "measurement_start_ns": 1_000_000_000,
                                    "ue_ip": "10.45.0.2", "pre_jam_seconds": 1}), encoding="utf-8")
    sensor = tmp_path / "sensor.jsonl"
    sensor.write_text("\n".join((
        json.dumps({"window_index": 0, "timestamp_ns": 2_000_000_000}),
        json.dumps({"window_index": 1, "timestamp_ns": 3_000_000_000}),
    )), encoding="utf-8")
    report = validate_protected_traffic(
        iperf_path=iperf, sensor_path=sensor, metadata_path=metadata,
        traffic_start_ns=1_000_000_000, expected_rows=2,
    )
    assert report["status"] == "FAIL"
    assert report["complete_radio_windows"] == 1


def test_zero_delivery_interval_is_retained_as_outage_not_missing_evidence(tmp_path: Path):
    document = _document(intervals=2)
    document["intervals"][1]["sum"].update({"bytes": 0, "packets": 0, "bits_per_second": 0})
    iperf = tmp_path / "protected.json"
    iperf.write_text(json.dumps(document), encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"run_id": "outage", "measurement_start_ns": 1_000_000_000,
                                    "ue_ip": "10.45.0.2", "pre_jam_seconds": 1}), encoding="utf-8")
    sensor = tmp_path / "sensor.jsonl"
    sensor.write_text("\n".join((
        json.dumps({"window_index": 0, "timestamp_ns": 2_000_000_000}),
        json.dumps({"window_index": 1, "timestamp_ns": 3_000_000_000}),
    )), encoding="utf-8")
    report = validate_protected_traffic(
        iperf_path=iperf, sensor_path=sensor, metadata_path=metadata,
        traffic_start_ns=1_000_000_000, expected_rows=2,
    )
    assert report["status"] == "PASS"
    assert report["complete_radio_windows"] == 2
    assert report["outage_censored_radio_windows"] == [1]
