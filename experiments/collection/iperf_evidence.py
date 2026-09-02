#!/usr/bin/env python3
"""Validate and align the protected iperf stream with measured radio windows.

The live virtual radio is sample-clocked and can run slower than wall time.
Consequently, iperf's one-second wall-clock intervals must not be assigned to
radio phases by ordinal position.  This module aligns them to the measured IQ
window completion timestamps and emits an auditable status document.

An iperf client is intentionally interrupted after the exact radio-sample
target.  iperf still writes a complete JSON object with an ``error`` field in
that case.  A controlled interrupt is acceptable only when the connection,
packets, interval continuity, and full radio-window coverage all pass.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


CONTROLLED_INTERRUPT = "interrupt - the client has terminated"


@dataclass(frozen=True)
class TrafficInterval:
    start_ns: int
    end_ns: int
    seconds: float
    bytes: int
    packets: int
    lost_packets: int
    lost_percent: float
    bits_per_second: float
    jitter_ms: float | None


def read_iperf_json(path: Path) -> dict[str, Any]:
    """Read iperf JSON while preserving any diagnostic prefix on stderr."""
    text = path.read_text(encoding="utf-8", errors="replace")
    embedded_diagnostics: list[str] = []

    def preserve_diagnostic(match: re.Match[str]) -> str:
        embedded_diagnostics.append(match.group(0).strip())
        return ""

    # iperf 3.16 can interleave these stderr diagnostics at an arbitrary
    # stdout byte boundary during a controlled interrupt.  Remove only the
    # two exact, named diagnostic families; all other unexpected text remains
    # parse-failing evidence.  The removed strings are retained in the status.
    text = re.sub(
        r"iperf_(?:json_finish|errexit):[^\r\n]*(?:\r?\n|$)",
        preserve_diagnostic,
        text,
    )
    start = text.find("{")
    if start < 0:
        raise ValueError("iperf output contains no JSON object")
    try:
        document, end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"iperf JSON is incomplete: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("iperf JSON root is not an object")
    suffix = text[start + end:].strip()
    # Some iperf 3.16 builds emit pthread shutdown diagnostics after the
    # already complete JSON object when the deliberately long-running client
    # is interrupted. Preserve those diagnostics in the derived status, but
    # never accept a second JSON object or silently merge it with the first.
    if suffix.startswith(("{", "[")):
        raise ValueError("iperf output contains an unexpected second JSON value")
    if suffix:
        document["_diagnostic_suffix"] = suffix
    if embedded_diagnostics:
        document["_embedded_diagnostics"] = embedded_diagnostics
    return document


def _sum_record(item: dict[str, Any]) -> dict[str, Any]:
    record = item.get("sum")
    if not isinstance(record, dict):
        streams = item.get("streams")
        if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
            raise ValueError("iperf interval has no unambiguous sum record")
        record = streams[0]
    return record


def extract_intervals(document: dict[str, Any], traffic_start_ns: int) -> list[TrafficInterval]:
    raw = document.get("intervals")
    if not isinstance(raw, list) or not raw:
        raise ValueError("iperf output contains no intervals")
    intervals: list[TrafficInterval] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"iperf interval {index} is not an object")
        record = _sum_record(item)
        start = float(record["start"])
        end = float(record["end"])
        seconds = float(record.get("seconds", end - start))
        if start < 0.0 or end <= start or seconds <= 0.0:
            raise ValueError(f"invalid iperf interval bounds at {index}")
        if bool(record.get("omitted", False)):
            continue
        intervals.append(TrafficInterval(
            start_ns=traffic_start_ns + round(start * 1_000_000_000),
            end_ns=traffic_start_ns + round(end * 1_000_000_000),
            seconds=seconds,
            bytes=int(record.get("bytes", 0)),
            packets=int(record.get("packets", 0)),
            lost_packets=int(record.get("lost_packets", 0)),
            lost_percent=float(record.get("lost_percent", 0.0)),
            bits_per_second=float(record.get("bits_per_second", 0.0)),
            jitter_ms=(None if record.get("jitter_ms") is None else float(record["jitter_ms"])),
        ))
    if not intervals:
        raise ValueError("iperf output contains no non-omitted intervals")
    return intervals


def read_radio_windows(sensor_path: Path, expected_rows: int, measurement_start_ns: int) -> list[tuple[int, int, int]]:
    """Return ``(window_index, start_ns, end_ns)`` from measured completions."""
    completions: dict[int, int] = {}
    for line in sensor_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        completions[int(item["window_index"])] = int(item["timestamp_ns"])
    # The broker can finish a small number of additional windows while the
    # wall-clock watcher observes the exact sample target and stops the
    # disposable containers.  Those tail windows are retained as raw
    # diagnostic evidence but are outside the immutable row.
    if any(index not in completions for index in range(expected_rows)):
        raise ValueError("IQ sensor window timestamps are incomplete")
    windows: list[tuple[int, int, int]] = []
    start_ns = measurement_start_ns
    for index in range(expected_rows):
        end_ns = completions[index]
        if end_ns <= start_ns:
            raise ValueError(f"non-increasing radio window time at {index}")
        windows.append((index, start_ns, end_ns))
        start_ns = end_ns
    return windows


def _overlap_ns(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def align_intervals(
    intervals: Iterable[TrafficInterval], windows: Iterable[tuple[int, int, int]]
) -> list[dict[str, Any]]:
    interval_list = list(intervals)
    aligned: list[dict[str, Any]] = []
    for index, start_ns, end_ns in windows:
        duration_ns = end_ns - start_ns
        overlap_total = 0
        weighted_bps = 0.0
        weighted_loss = 0.0
        packets = 0.0
        lost_packets = 0.0
        weighted_jitter = 0.0
        jitter_weight = 0
        for interval in interval_list:
            overlap = _overlap_ns(start_ns, end_ns, interval.start_ns, interval.end_ns)
            if overlap <= 0:
                continue
            overlap_total += overlap
            weighted_bps += interval.bits_per_second * overlap
            weighted_loss += interval.lost_percent * overlap
            fraction = overlap / max(1, interval.end_ns - interval.start_ns)
            packets += interval.packets * fraction
            lost_packets += interval.lost_packets * fraction
            if interval.jitter_ms is not None:
                weighted_jitter += interval.jitter_ms * overlap
                jitter_weight += overlap
        coverage = min(1.0, overlap_total / duration_ns)
        aligned.append({
            "window_index": index,
            "window_start_ns": start_ns,
            "window_end_ns": end_ns,
            "traffic_coverage_fraction": coverage,
            "protected_throughput_mbps": (
                weighted_bps / overlap_total / 1_000_000.0 if overlap_total else 0.0
            ),
            "protected_lost_percent": weighted_loss / overlap_total if overlap_total else None,
            "protected_packets": packets,
            "protected_lost_packets": lost_packets,
            "protected_jitter_ms": weighted_jitter / jitter_weight if jitter_weight else None,
        })
    return aligned


def validate_protected_traffic(
    *, iperf_path: Path, sensor_path: Path, metadata_path: Path,
    traffic_start_ns: int, expected_rows: int, minimum_window_coverage: float = 0.90,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    document = read_iperf_json(iperf_path)
    connected = document.get("start", {}).get("connected", [])
    intervals = extract_intervals(document, traffic_start_ns)
    windows = read_radio_windows(sensor_path, expected_rows, int(metadata["measurement_start_ns"]))
    aligned = align_intervals(intervals, windows)
    error = document.get("error")
    error_allowed = error in (None, "", CONTROLLED_INTERRUPT)
    failures: list[str] = []
    if not isinstance(connected, list) or not connected:
        failures.append("protected client did not establish a connection")
    else:
        expected_ue = str(metadata.get("ue_ip", ""))
        endpoint_ok = any(
            isinstance(item, dict)
            and str(item.get("local_host")) == expected_ue
            and str(item.get("remote_host")) == "10.45.0.1"
            for item in connected
        )
        if not endpoint_ok:
            failures.append("protected connection endpoints do not match the measured UE session")
    if not error_allowed:
        failures.append(f"unclassified iperf error: {error}")
    if not any(interval.packets > 0 and interval.bytes > 0 for interval in intervals):
        failures.append("protected stream never delivered measured traffic")
    pre_jam_seconds = int(metadata.get("pre_jam_seconds", 0))
    pre_interference_delivery = all(
        row["protected_packets"] > 0.0 for row in aligned[:pre_jam_seconds]
    )
    if pre_jam_seconds <= 0 or not pre_interference_delivery:
        failures.append("protected delivery was not verified throughout the pre-interference phase")
    incomplete = [
        row["window_index"] for row in aligned
        if row["traffic_coverage_fraction"] < minimum_window_coverage
    ]
    if incomplete:
        failures.append(f"protected traffic incomplete in radio windows: {incomplete}")
    status = "PASS" if not failures else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "evidence_role": "prospective_protected_flow_acceptance",
        "run_id": metadata.get("run_id"),
        "expected_radio_windows": expected_rows,
        "connected": bool(connected),
        "service_session_verified": not any(
            failure.startswith("protected client")
            or failure.startswith("protected connection") for failure in failures
        ),
        "traffic_start_ns": traffic_start_ns,
        "iperf_interval_count": len(intervals),
        "controlled_interrupt": error == CONTROLLED_INTERRUPT,
        "iperf_error": error,
        "iperf_diagnostic_suffix": document.get("_diagnostic_suffix"),
        "iperf_embedded_diagnostics": document.get("_embedded_diagnostics", []),
        "minimum_window_coverage": minimum_window_coverage,
        "pre_interference_delivery_verified": pre_interference_delivery,
        "complete_radio_windows": sum(
            row["traffic_coverage_fraction"] >= minimum_window_coverage
            for row in aligned
        ),
        "delivery_observed_radio_windows": sum(row["protected_packets"] > 0.0 for row in aligned),
        "outage_censored_radio_windows": [
            row["window_index"] for row in aligned if row["protected_packets"] <= 0.0
        ],
        "outage_censoring_rule": (
            "A connected, pre-interference-verified stream with a complete iperf interval but zero "
            "received packets is retained as an observed zero-goodput outage, never rejected or imputed."
        ),
        "failures": failures,
        "radio_windows": aligned,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iperf", type=Path, required=True)
    parser.add_argument("--sensor", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--traffic-start-ns-file", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--minimum-window-coverage", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start_ns = int(args.traffic_start_ns_file.read_text(encoding="utf-8").strip())
    report = validate_protected_traffic(
        iperf_path=args.iperf, sensor_path=args.sensor, metadata_path=args.metadata,
        traffic_start_ns=start_ns, expected_rows=args.expected_rows,
        minimum_window_coverage=args.minimum_window_coverage,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
