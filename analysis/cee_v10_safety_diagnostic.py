#!/usr/bin/env python3
"""Development-only diagnosis used to design the post-V10 safety gate.

V10 is permanently ineligible for confirmatory claims.  This script reads its
preserved action logs and measured window table to quantify false activation,
attachment state, action persistence, and one-window outcome changes.  The
result is explicitly design evidence for the prospective CEE study.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PHASES = ((0, 20, "pre_jam"), (20, 100, "jammer_on"), (100, 120, "recovery"))
ACTION_NAMES = ("NO_OP", "PRB_AVOID_SHIFT", "MCS_BACKOFF", "SLICE_PRB_BOOST", "COMPOSITE_LOCAL_DEFENSE")


def _phase(index: int) -> str:
    for start, stop, name in PHASES:
        if start <= index < stop:
            return name
    raise ValueError(f"window index outside the frozen 120-window design: {index}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"non-object JSONL row in {path}")
            rows.append(item)
    return rows


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": None, "p05": None, "p50": None, "p95": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def diagnose(raw_root: Path, records_path: Path) -> dict[str, Any]:
    decision_files = sorted(raw_root.glob("iq-broker-live-livev10-p-*/live-xapp-decisions.jsonl"))
    if len(decision_files) != 90:
        raise ValueError(f"expected 90 accepted V10 P decision files, found {len(decision_files)}")

    records = pd.read_parquet(records_path)
    records = records.loc[records["method"] == "P"].copy()
    if len(records) != 10_800:
        raise ValueError(f"expected 10,800 measured V10 P windows, found {len(records)}")
    measured = records.set_index(["run_id", "window_index"]).sort_index()

    action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    p_jam: dict[str, list[float]] = defaultdict(list)
    attached: dict[str, Counter[str]] = defaultdict(Counter)
    localization: dict[str, Counter[str]] = defaultdict(Counter)
    action_transitions: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    false_activation = {"pre_jam_non_noop": 0, "no_jam_non_noop": 0, "pre_jam_windows": 0, "no_jam_windows": 0}

    for path in decision_files:
        decisions = _read_jsonl(path)
        if len(decisions) != 120 or [int(item["window_index"]) for item in decisions] != list(range(120)):
            raise ValueError(f"decision cardinality/order mismatch: {path}")
        run_id = path.parent.name.removeprefix("iq-broker-live-")
        run_record = measured.loc[run_id]
        jammer_type = str(run_record["jammer_type"].iloc[0])
        previous = "START"
        for item in decisions:
            index = int(item["window_index"])
            phase = _phase(index)
            if item.get("status") != "DECISION":
                raise ValueError(f"non-decision row in accepted V10 P evidence: {run_id}:{index}")
            decision = item["decision"]
            action = str(decision["execution"]["executed_action"])
            if action not in ACTION_NAMES:
                raise ValueError(f"unknown action in {run_id}:{index}: {action}")
            action_counts[phase][action] += 1
            action_counts["all"][action] += 1
            action_transitions[f"{previous}->{action}"] += 1
            previous = action
            probability = float(item["detector_p_jam"])
            p_jam[phase].append(probability)
            p_jam[f"truth_{jammer_type}_{phase}"].append(probability)
            receiver = item.get("receiver_attached")
            receiver_key = "missing" if receiver is None else ("attached" if float(receiver) >= 0.5 else "detached")
            attached[phase][receiver_key] += 1
            localized = bool(item.get("iq_localization", {}).get("localized_jammer_region_available", False))
            localization[phase]["localized" if localized else "not_localized"] += 1
            if phase == "pre_jam":
                false_activation["pre_jam_windows"] += 1
                false_activation["pre_jam_non_noop"] += int(action != "NO_OP")
            if jammer_type == "none":
                false_activation["no_jam_windows"] += 1
                false_activation["no_jam_non_noop"] += int(action != "NO_OP")

            current = run_record.loc[index]
            next_index = min(index + 1, 119)
            nxt = run_record.loc[next_index]
            rows.append({
                "run_id": run_id,
                "window_index": index,
                "phase": phase,
                "jammer_type": jammer_type,
                "action": action,
                "p_jam": probability,
                "receiver_attached": receiver_key,
                "localized": localized,
                "throughput_now": float(current["gnb_dl_throughput_mbps"]),
                "throughput_next": float(nxt["gnb_dl_throughput_mbps"]),
                "latency_now": float(current["rlc_sdu_delay_ms"]) if pd.notna(current["rlc_sdu_delay_ms"]) else np.nan,
                "latency_next": float(nxt["rlc_sdu_delay_ms"]) if pd.notna(nxt["rlc_sdu_delay_ms"]) else np.nan,
                "drop_now": float(current["rlc_drop_rate"]),
                "drop_next": float(nxt["rlc_drop_rate"]),
            })

    frame = pd.DataFrame(rows)
    action_effects: dict[str, dict[str, float | int | None]] = {}
    for action, group in frame.groupby("action"):
        throughput_delta = group["throughput_next"] - group["throughput_now"]
        latency_delta = group["latency_next"] - group["latency_now"]
        drop_delta = group["drop_next"] - group["drop_now"]
        action_effects[str(action)] = {
            "n": int(len(group)),
            "mean_next_throughput_delta_mbps": float(throughput_delta.mean()),
            "median_next_throughput_delta_mbps": float(throughput_delta.median()),
            "mean_next_latency_delta_ms": float(latency_delta.mean()) if latency_delta.notna().any() else None,
            "mean_next_drop_delta": float(drop_delta.mean()),
        }

    false_activation["pre_jam_non_noop_fraction"] = (
        false_activation["pre_jam_non_noop"] / false_activation["pre_jam_windows"]
    )
    false_activation["no_jam_non_noop_fraction"] = (
        false_activation["no_jam_non_noop"] / false_activation["no_jam_windows"]
    )
    return {
        "schema_version": 1,
        "evidence_role": "V10_DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "decision_files": len(decision_files),
        "decisions": int(len(frame)),
        "action_counts": {phase: dict(counts) for phase, counts in sorted(action_counts.items())},
        "action_transitions": dict(action_transitions.most_common()),
        "detector_p_jam": {key: _quantiles(values) for key, values in sorted(p_jam.items())},
        "receiver_attachment": {phase: dict(counts) for phase, counts in sorted(attached.items())},
        "iq_localization": {phase: dict(counts) for phase, counts in sorted(localization.items())},
        "false_activation": false_activation,
        "observational_one_window_action_changes": action_effects,
        "causal_warning": (
            "Action-change summaries are observational and policy-confounded; they may define conservative "
            "guards but are not confirmatory causal effects."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.raw_root, args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
