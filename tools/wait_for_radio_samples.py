#!/usr/bin/env python3
"""Wait until a live broker has forwarded a requested radio-time duration.

Wall-clock durations are not a valid experiment clock for a virtual ZMQ
radio: CPU scheduling can make the simulated radio run slower or faster than
real time.  This helper reads only the broker's immutable truth stream and
waits for its post-marker sample index to reach the requested duration.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _started_at(status_path: Path) -> int | None:
    if not status_path.is_file():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "experiment_clock_started":
            return int(event["timestamp_ns"])
    return None


def _max_post_marker_sample(truth_path: Path, started_ns: int) -> int:
    if not truth_path.is_file():
        return 0
    maximum = 0
    for line in truth_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(record.get("timestamp_ns", -1)) < started_ns:
            continue
        maximum = max(maximum, int(record.get("sample_index_end", 0)))
    return maximum


def _terminal_broker_event(status_path: Path) -> str | None:
    """Return a terminal broker event, if the disposable radio has stopped."""
    if not status_path.is_file():
        return None
    terminal: str | None = None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = str(event.get("event", ""))
        if name == "uplink_timeout":
            terminal = name
        elif name == "broker_stop" and terminal is None:
            terminal = name
    return terminal


def wait_for_samples(evidence: Path, target_samples: int, timeout_seconds: float) -> dict[str, object]:
    if target_samples <= 0 or timeout_seconds <= 0:
        raise ValueError("target_samples and timeout_seconds must be positive")
    status_path = evidence / "broker_status.jsonl"
    # ``jammer_truth.jsonl`` is the methodology-locked evidence name.  The
    # fallback lets us validate early development evidence recorded before the
    # naming correction without rewriting that immutable raw data.
    truth_path = evidence / "jammer_truth.jsonl"
    if not truth_path.is_file():
        truth_path = evidence / "truth.jsonl"
    deadline = time.monotonic() + timeout_seconds
    started_ns: int | None = None
    while time.monotonic() < deadline:
        started_ns = started_ns or _started_at(status_path)
        if started_ns is not None:
            observed = _max_post_marker_sample(truth_path, started_ns)
            if observed >= target_samples:
                return {
                    "status": "PASS",
                    "experiment_started_timestamp_ns": started_ns,
                    "target_samples": target_samples,
                    "observed_samples": observed,
                }
            terminal = _terminal_broker_event(status_path)
            if terminal in {"uplink_timeout", "broker_stop"}:
                return {
                    "status": "BROKER_TERMINATED",
                    "terminal_event": terminal,
                    "experiment_started_timestamp_ns": started_ns,
                    "target_samples": target_samples,
                    "observed_samples": observed,
                }
        time.sleep(0.25)
    return {
        "status": "TIMEOUT",
        "experiment_started_timestamp_ns": started_ns or 0,
        "target_samples": target_samples,
        "observed_samples": _max_post_marker_sample(truth_path, started_ns) if started_ns else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--target-samples", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    args = parser.parse_args()
    result = wait_for_samples(args.evidence, args.target_samples, args.timeout_seconds)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
