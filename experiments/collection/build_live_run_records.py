#!/usr/bin/env python3
"""Join raw live-radio evidence into non-overlapping one-second run records.

The collector intentionally preserves raw source envelopes alongside derived
features.  It never substitutes synthetic values for a missing live metric:
the window is marked incomplete, which keeps exploratory data separated from
eligible controller records.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.xapp.telemetry.ocudu_json import parse_ocudu_json_line
from src.xapp.telemetry.srsue_csv import parse_srsue_csv
from src.xapp.telemetry.state_features import rlc_observation_from_ocudu, state_observation_from_ocudu


WINDOW_NS = 1_000_000_000
DEFAULT_SAMPLE_RATE_HZ = 23_040_000
REQUIRED_TELEMETRY = (
    "sinr_db", "cqi", "rsrp_dbfs", "rsrq_db", "bler", "harq_nack_ratio",
    "dl_throughput_mbps", "rlc_drop_rate", "rlc_sdu_delay_ms", "prb_utilization",
    "scheduler_queue_backlog_bytes",
)
REQUIRED_CONTROLLER_STATE = REQUIRED_TELEMETRY + (
    "clean_prb_fraction", "jammer_occupancy_fraction",
    "jammer_subband_start_norm", "jammer_subband_end_norm",
)


@dataclass(frozen=True)
class TimedValues:
    timestamp_ns: int
    values: dict[str, float]
    raw: dict[str, Any]


def _json_lines(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return ()
    values: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{number}") from error
        if not isinstance(item, dict):
            raise ValueError(f"JSON envelope at {path}:{number} is not an object")
        values.append(item)
    return values


def _load_srsue(path: Path, *, allow_unbounded_sinr: bool = False) -> list[TimedValues]:
    if not path.is_file():
        return []
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines(), delimiter=";"))
    if len(rows) < 2:
        return []
    header = ";".join(rows[0])
    samples: list[TimedValues] = []
    for row in rows[1:]:
        if not row or not any(cell.strip() for cell in row):
            continue
        raw = dict(zip(rows[0], row, strict=True))
        try:
            parsed = parse_srsue_csv(header, ";".join(row), allow_unbounded_sinr=allow_unbounded_sinr)
        except ValueError:
            # The raw envelope remains on disk; an incomplete window records
            # that this particular receiver sample was unusable.
            continue
        samples.append(TimedValues(parsed.timestamp_ns, dict(parsed.values), raw))
    return samples


def _load_srsue_status(path: Path) -> list[TimedValues]:
    """Load receiver attachment observations even when RF values are undefined.

    A detached receiver legitimately emits non-finite RF measurements.  The
    strict metric parser must continue to reject those values, but the native
    ``is_attached`` flag and timestamp are still truthful observability
    evidence.  Keeping this as a separate source prevents a detached interval
    from being mistaken for a missing collector.
    """
    if not path.is_file():
        return []
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines(), delimiter=";"))
    if len(rows) < 2 or "timestamp_epoch_ms" not in rows[0] or "is_attached" not in rows[0]:
        return []
    samples: list[TimedValues] = []
    for row in rows[1:]:
        if len(row) != len(rows[0]):
            continue
        raw = dict(zip(rows[0], row, strict=True))
        try:
            timestamp_ms = float(raw["timestamp_epoch_ms"])
            attached = float(raw["is_attached"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(timestamp_ms) or attached not in (0.0, 1.0):
            continue
        samples.append(TimedValues(round(timestamp_ms * 1_000_000.0), {"receiver_attached": attached}, raw))
    return samples


def _load_ocudu(path: Path) -> tuple[list[TimedValues], list[TimedValues]]:
    scheduler: list[TimedValues] = []
    rlc: list[TimedValues] = []
    for raw in _json_lines(path):
        try:
            envelope = parse_ocudu_json_line(json.dumps(raw))
        except ValueError:
            continue
        if envelope is None:
            continue
        try:
            state = state_observation_from_ocudu(envelope.timestamp_ns, envelope.payload)
            if state is not None:
                scheduler.append(TimedValues(state.timestamp_ns, dict(state.values), raw))
            rlc_values = rlc_observation_from_ocudu(envelope.payload)
            if rlc_values is not None:
                rlc.append(TimedValues(envelope.timestamp_ns, rlc_values, raw))
        except ValueError:
            continue
    return scheduler, rlc


def _load_iq_sensor(path: Path) -> dict[int, TimedValues]:
    """Load measured-IQ estimates keyed by their exact radio window.

    This source is produced from the received radio frame and is separate from
    the evaluation-only jammer truth log.
    """
    windows: dict[int, TimedValues] = {}
    for raw in _json_lines(path):
        try:
            index = int(raw["window_index"])
            timestamp = int(raw["timestamp_ns"])
            values = {
                "clean_prb_fraction": float(raw["clean_prb_fraction"]),
                "jammer_occupancy_fraction": float(raw["jammer_occupancy_fraction"]),
                "jammer_subband_start_norm": float(raw["jammer_subband_start_norm"]),
                "jammer_subband_end_norm": float(raw["jammer_subband_end_norm"]),
            }
            values["localized_jammer_region_available"] = float(bool(raw["localized_jammer_region_available"]))
        except (KeyError, TypeError, ValueError):
            continue
        windows[index] = TimedValues(timestamp, values, raw)
    return windows


def _in_window(samples: Iterable[TimedValues], start: int, end: int) -> list[TimedValues]:
    return [sample for sample in samples if start <= sample.timestamp_ns < end]


def _experiment_started_ns(evidence: Path, metadata: dict[str, Any]) -> int | None:
    """Return the broker-confirmed start marker time, if it was recorded.

    The marker is observed by the forwarding thread on a frame boundary.  Its
    status event is therefore the authoritative point at which sample index 0
    begins, rather than the host-side ``touch`` timestamp.
    """
    value = metadata.get("experiment_started_timestamp_ns")
    if value is not None:
        return int(value)
    status_path = evidence / "broker_status.jsonl"
    for status in _json_lines(status_path):
        if status.get("event") == "experiment_clock_started":
            return int(status["timestamp_ns"])
    return None


def _truth_clock(truth: list[dict[str, Any]], started_ns: int | None) -> list[tuple[int, int, int]]:
    """Map host timestamps to the exact forwarded-radio sample cursor.

    Each truth entry is emitted immediately after one real downlink frame is
    processed.  A telemetry sample is assigned only to the last completed
    forwarded frame at or before its timestamp; this is an observed alignment,
    not an interpolation or a synthetic metric.
    """
    points: list[tuple[int, int, int]] = []
    for item in truth:
        if "sample_index_end" not in item:
            continue
        timestamp = int(item["timestamp_ns"])
        if started_ns is not None and timestamp < started_ns:
            continue
        points.append((timestamp, int(item["sample_index_start"]), int(item["sample_index_end"])))
    points.sort()
    return points


def _map_to_radio_clock(samples: list[TimedValues], clock: list[tuple[int, int, int]]) -> list[TimedValues]:
    if not clock:
        return []
    timestamps = [point[0] for point in clock]
    mapped: list[TimedValues] = []
    for sample in samples:
        index = bisect_right(timestamps, sample.timestamp_ns) - 1
        if index < 0:
            continue
        mapped.append(TimedValues(clock[index][1], sample.values, sample.raw))
    return mapped


def _last_values(samples: list[TimedValues]) -> tuple[dict[str, float], dict[str, Any] | None]:
    if not samples:
        return {}, None
    latest = max(samples, key=lambda item: item.timestamp_ns)
    return latest.values, latest.raw


def _mean(values: Iterable[float]) -> float | None:
    points = list(values)
    return float(sum(points) / len(points)) if points else None


def _truth_for_window(records: list[dict[str, Any]], start: int, end: int, started_ns: int | None) -> dict[str, Any]:
    selected = [
        item for item in records
        if "sample_index_start" in item and "sample_index_end" in item
        and (started_ns is None or int(item["timestamp_ns"]) >= started_ns)
        and start <= int(item["sample_index_start"]) < end
    ]
    total = sum(int(item.get("frame_sample_count", 0)) for item in selected)
    active = sum(int(item.get("active_sample_count", 0)) for item in selected)
    active_records = [item for item in selected if int(item.get("active_sample_count", 0)) > 0]
    occupancy = float(active / total) if total else None
    if not active_records:
        return {
            "ground_truth_jammer_type": "none", "ground_truth_requested_jsr_db": None,
            "ground_truth_realized_jsr_db": None, "ground_truth_occupancy_fraction": occupancy,
            "ground_truth_subband_start_norm": None, "ground_truth_subband_end_norm": None,
            "truth_frame_count": len(selected), "truth_sample_count": total,
        }
    # Keep the exact active-sample weighting without materialising one Python
    # value per IQ sample.  A 30-second 23.04 MHz run contains hundreds of
    # millions of samples, so the former expanded generator could stall the
    # evidence builder after raw collection had already completed.
    by_type: Counter[str] = Counter()
    for item in active_records:
        by_type[str(item["jammer_type"])] += int(item.get("active_sample_count", 0))
    dominant = by_type.most_common(1)[0][0]

    def weighted_number(key: str) -> float | None:
        pairs = [
            (float(item[key]), int(item.get("active_sample_count", 0)))
            for item in active_records if item.get(key) is not None
        ]
        denominator = sum(weight for _, weight in pairs)
        return float(sum(value * weight for value, weight in pairs) / denominator) if denominator else None

    return {
        "ground_truth_jammer_type": dominant,
        "ground_truth_requested_jsr_db": weighted_number("requested_jsr_db"),
        "ground_truth_realized_jsr_db": weighted_number("realized_jsr_db"),
        "ground_truth_occupancy_fraction": occupancy,
        "ground_truth_subband_start_norm": weighted_number("subband_start_norm"),
        "ground_truth_subband_end_norm": weighted_number("subband_end_norm"),
        "truth_frame_count": len(selected), "truth_sample_count": total,
    }


def _phase(index: int, metadata: dict[str, Any]) -> str:
    pre = int(metadata["pre_jam_seconds"])
    on = int(metadata["jammer_on_seconds"])
    return "pre_jam" if index < pre else "jammer_on" if index < pre + on else "recovery"


def build_records(evidence: Path, metadata: dict[str, Any]) -> pd.DataFrame:
    duration = int(metadata["run_length_seconds"])
    sample_rate_hz = int(metadata.get("sample_rate_hz", DEFAULT_SAMPLE_RATE_HZ))
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    amended_receiver_profile = metadata.get("acceptance_profile") == "observed_outage_and_unbounded_sinr_v2"
    srsue = _load_srsue(evidence / "srsue_metrics.csv", allow_unbounded_sinr=amended_receiver_profile)
    srsue_status = _load_srsue_status(evidence / "srsue_metrics.csv")
    scheduler, rlc = _load_ocudu(evidence / "ocudu-metrics.jsonl")
    iq_sensor = _load_iq_sensor(evidence / "iq-spectral-estimator.jsonl")
    truth_path = evidence / "jammer_truth.jsonl"
    if not truth_path.is_file():
        # Development runs written before the locked filename correction are
        # retained verbatim; only the reader is backward compatible.
        truth_path = evidence / "truth.jsonl"
    truth = list(_json_lines(truth_path))
    started_ns = _experiment_started_ns(evidence, metadata)
    clock = _truth_clock(truth, started_ns)
    srsue = _map_to_radio_clock(srsue, clock)
    srsue_status = _map_to_radio_clock(srsue_status, clock)
    scheduler = _map_to_radio_clock(scheduler, clock)
    rlc = _map_to_radio_clock(rlc, clock)
    max_radio_sample = clock[-1][2] if clock else 0
    rows: list[dict[str, Any]] = []
    for index in range(duration):
        start, end = index * sample_rate_hz, (index + 1) * sample_rate_hz
        srs_values, srs_raw = _last_values(_in_window(srsue, start, end))
        status_values, status_raw = _last_values(_in_window(srsue_status, start, end))
        scheduler_values, scheduler_raw = _last_values(_in_window(scheduler, start, end))
        rlc_values, rlc_raw = _last_values(_in_window(rlc, start, end))
        sensor = iq_sensor.get(index)
        sensor_values = {} if sensor is None else sensor.values
        telemetry = {**srs_values, **scheduler_values, **rlc_values, **sensor_values}
        # The locked BLER feature is supplied by the pinned receiver's native
        # downlink BLER counter; retain its source-specific column as well.
        if "ue_dl_bler" in telemetry:
            telemetry["bler"] = telemetry["ue_dl_bler"]
        if "gnb_dl_throughput_mbps" in telemetry:
            telemetry["dl_throughput_mbps"] = telemetry["gnb_dl_throughput_mbps"]
        elif "ue_dl_throughput_mbps" in telemetry:
            telemetry["dl_throughput_mbps"] = telemetry["ue_dl_throughput_mbps"]
        missing = [name for name in REQUIRED_TELEMETRY if name not in telemetry]
        missing_controller = [name for name in REQUIRED_CONTROLLER_STATE if name not in telemetry]
        receiver_observed = bool(status_raw)
        receiver_attached = status_values.get("receiver_attached")
        no_successful_sdu = (
            "rlc_sdu_delay_ms" in missing
            and rlc_values.get("rlc_drop_rate") == 1.0
            and telemetry.get("dl_throughput_mbps") == 0.0
        )
        missing_set = set(missing)
        detached_missing = {
            "sinr_db", "rsrp_dbfs", "rsrq_db", "bler", "rlc_sdu_delay_ms",
        }
        if not missing:
            observability = "complete_measured"
            evidence_complete = True
        elif (amended_receiver_profile and missing_set == {"sinr_db"}
              and telemetry.get("sinr_positive_infinity_observed") == 1.0
              and receiver_attached == 1.0):
            observability = "native_positive_infinite_sinr_observed"
            evidence_complete = True
        elif missing_set == {"rlc_sdu_delay_ms"} and no_successful_sdu:
            observability = "right_censored_no_successful_sdu"
            evidence_complete = True
        elif (
            missing_set.issubset(detached_missing)
            and receiver_observed
            and receiver_attached == 0.0
            and scheduler_raw is not None
            and rlc_raw is not None
            and sensor is not None
        ):
            # Native receiver detachment is itself the audit condition for
            # receiver-derived RF features being unavailable.  The gNB-side
            # RLC stream can still report a successful buffered SDU (and thus
            # a measured delay) after the UE has detached.  Requiring
            # ``no_successful_sdu`` here incorrectly rejected those fully
            # evidenced, right-censored receiver-outage windows.  Preserve
            # every raw source and every measured network value; do not fill
            # any missing receiver metric.
            observability = "receiver_detached_outage"
            evidence_complete = True
        else:
            observability = "incomplete_unclassified"
            evidence_complete = False
        row: dict[str, Any] = {
            "schema_version": 1,
            "run_id": metadata["run_id"], "scenario_id": metadata["scenario_id"],
            "seed": int(metadata["seed"]), "evidence_class": metadata["evidence_class"],
            "window_index": index, "radio_window_start_sample": start, "radio_window_end_sample": end,
            "window_start_ns": None, "window_end_ns": None,
            "radio_window_covered": end <= max_radio_sample,
            "phase": _phase(index, metadata), "raw_srsue_json": json.dumps(srs_raw, sort_keys=True) if srs_raw else None,
            "raw_srsue_status_json": json.dumps(status_raw, sort_keys=True) if status_raw else None,
            "raw_ocudu_scheduler_json": json.dumps(scheduler_raw, sort_keys=True) if scheduler_raw else None,
            "raw_ocudu_rlc_json": json.dumps(rlc_raw, sort_keys=True) if rlc_raw else None,
            "raw_iq_spectral_estimator_json": json.dumps(sensor.raw, sort_keys=True) if sensor else None,
            "missing_telemetry": json.dumps(missing), "complete_telemetry": not missing,
            "missing_controller_state": json.dumps(missing_controller),
            "complete_controller_state": not missing_controller,
            "receiver_observed": receiver_observed,
            "receiver_attached": receiver_attached,
            "receiver_measurement_qualifiers": json.dumps({"sinr_db": "native_positive_infinity_unbounded"}
                if telemetry.get("sinr_positive_infinity_observed") == 1.0 else {}, sort_keys=True),
            "telemetry_observability": observability,
            "evidence_complete": evidence_complete,
            **_truth_for_window(truth, start, end, started_ns), **telemetry,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-evidence-complete", action="store_true")
    parser.add_argument("--require-controller-state", action="store_true")
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    required_metadata = {"run_id", "scenario_id", "seed", "evidence_class", "measurement_start_ns", "run_length_seconds", "pre_jam_seconds", "jammer_on_seconds", "recovery_seconds"}
    missing = sorted(required_metadata - set(metadata))
    if missing:
        raise SystemExit(f"metadata is missing required fields: {missing}")
    if int(metadata["pre_jam_seconds"]) + int(metadata["jammer_on_seconds"]) + int(metadata["recovery_seconds"]) != int(metadata["run_length_seconds"]):
        raise SystemExit("phase durations do not sum to run length")
    records = build_records(args.evidence, metadata)
    output = args.output or args.evidence / "records.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(output, index=False)
    summary = {
        "rows": len(records), "complete_rows": int(records["complete_telemetry"].sum()),
        "incomplete_rows": int((~records["complete_telemetry"]).sum()), "output": str(output),
        "radio_window_covered_rows": int(records["radio_window_covered"].sum()),
        "complete_controller_state_rows": int(records["complete_controller_state"].sum()),
        "incomplete_controller_state_rows": int((~records["complete_controller_state"]).sum()),
        "evidence_complete_rows": int(records["evidence_complete"].sum()),
        "evidence_incomplete_rows": int((~records["evidence_complete"]).sum()),
    }
    (output.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    telemetry_ok = not args.require_complete or summary["incomplete_rows"] == 0
    controller_ok = not args.require_controller_state or summary["incomplete_controller_state_rows"] == 0
    evidence_ok = not args.require_evidence_complete or summary["evidence_incomplete_rows"] == 0
    return 0 if telemetry_ok and controller_ok and evidence_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
