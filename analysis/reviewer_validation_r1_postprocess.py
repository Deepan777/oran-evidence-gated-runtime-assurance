#!/usr/bin/env python3
"""Audit and summarize the frozen actuator characterization radio characterization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.collection.run_manifest import _complete


ACTION_NAMES = {
    "A0": "NO_OP",
    "A1": "PRB_AVOID_SHIFT",
    "A2": "MCS_BACKOFF",
    "A3": "SLICE_PRB_BOOST",
    "A4": "COMPOSITE_LOCAL_DEFENSE",
}


def _average(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(sum(materialized) / len(materialized))


def _median(values: Iterable[float]) -> float:
    return float(np.median(np.asarray(list(values), dtype=float)))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return _average(values) if values else math.nan


def _recovery_time(windows: list[dict[str, Any]], pre: int, jam: int) -> tuple[float, bool]:
    pre_mean = _mean(windows[:pre], "protected_throughput_mbps")
    threshold = 0.9 * pre_mean
    for second, row in enumerate(windows[pre + jam :], start=1):
        if float(row["protected_throughput_mbps"]) >= threshold:
            return float(second), False
    return float(len(windows[pre + jam :]) + 1), True


def _bootstrap_mean_ci(values: list[float], seed: int) -> list[float | None]:
    if not values:
        return [None, None]
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    samples = rng.choice(data, size=(20000, len(data)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(low), float(high)]


def _wilcoxon(values: list[float]) -> float | None:
    if not values or all(abs(value) < 1e-15 for value in values):
        return 1.0 if values else None
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)
    except (ImportError, ValueError):
        nonzero = [value for value in values if value != 0]
        if not nonzero:
            return 1.0
        positives = sum(value > 0 for value in nonzero)
        n = len(nonzero)
        tail = sum(math.comb(n, k) for k in range(0, min(positives, n - positives) + 1)) / (2**n)
        return float(min(1.0, 2.0 * tail))


def _holm(items: list[tuple[str, float | None]]) -> dict[str, float | None]:
    present = sorted(((name, p) for name, p in items if p is not None), key=lambda item: item[1])
    adjusted: dict[str, float | None] = {name: None for name, _ in items}
    running = 0.0
    count = len(present)
    for rank, (name, pvalue) in enumerate(present):
        running = max(running, min(1.0, (count - rank) * float(pvalue)))
        adjusted[name] = running
    return adjusted


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(manifest: Path, raw_root: Path, freeze: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    frozen = _read_json(freeze)
    manifest_key = "datasets/metadata/manifests/reviewer_validation_r1/REVIEWER_VALIDATION_R1.csv"
    if frozen["source_sha256"].get(manifest_key) != _sha256(manifest):
        raise ValueError("manifest hash does not match the frozen source hash")
    if len(rows) != frozen["rows"] or len(rows) != 84:
        raise ValueError("manifest cardinality does not match the freeze")
    expected_methods_by_block: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        expected_methods_by_block[row["block_id"]].add(row["method"])

    runs: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for order, row in enumerate(rows):
        evidence = raw_root / f"iq-broker-live-{row['run_id']}"
        complete = _complete(evidence, int(row["run_length_seconds"]), row)
        document: dict[str, Any] = {
            "order": order,
            "run_id": row["run_id"],
            "block_id": row["block_id"],
            "method": row["method"],
            "action_name": row["action_name"],
            "jammer_type": row["jammer_type"],
            "jammer_bandwidth_class": row["jammer_bandwidth_class"],
            "jsr_db": int(row["jsr_db"]),
            "traffic_load": row["traffic_load"],
            "attenuation": row["attenuation"],
            "seed": int(row["seed"]),
            "status": "INCOMPLETE_OR_REJECTED",
        }
        if not complete:
            attempts.append(document)
            runs.append(document)
            continue
        traffic = _read_json(evidence / "protected-traffic-status.json")
        windows = traffic["radio_windows"]
        decisions = _read_jsonl(evidence / "live-xapp-decisions.jsonl")
        decision_rows = [item for item in decisions if item.get("status") == "DECISION"]
        pre, jam = int(row["pre_jam_seconds"]), int(row["jammer_on_seconds"])
        expected_action = ACTION_NAMES[row["method"]]
        executed = [
            item.get("decision", {}).get("execution", {}).get("executed_action")
            for item in decision_rows
        ]
        acknowledged = [
            item.get("decision", {}).get("execution", {}).get("acknowledged") is True
            and item.get("decision", {}).get("execution", {}).get("failure") is None
            for item in decision_rows
        ]
        action_schedule_ok = (
            len(executed) == 30
            and all(action == "NO_OP" for action in executed[:pre])
            and all(action == expected_action for action in executed[pre : pre + jam])
            and all(action == "NO_OP" for action in executed[pre + jam :])
        )
        recovery_seconds, recovery_censored = _recovery_time(windows, pre, jam)
        interference = windows[pre : pre + jam]
        recovery = windows[pre + jam :]
        latencies = [
            float(item.get("decision", {}).get("execution", {}).get("latency_ms", math.nan))
            for item in decision_rows
        ]
        latencies = [value for value in latencies if math.isfinite(value)]
        document.update({
            "status": "PASS" if action_schedule_ok and all(acknowledged) else "ACTION_AUDIT_FAIL",
            "action_schedule_ok": action_schedule_ok,
            "all_decisions_acknowledged": all(acknowledged) and len(acknowledged) == 30,
            "complete_radio_windows": traffic.get("complete_radio_windows"),
            "outage_censored_windows": len(traffic.get("outage_censored_radio_windows", [])),
            "pre_goodput_mbps": _mean(windows[:pre], "protected_throughput_mbps"),
            "interference_goodput_mbps": _mean(interference, "protected_throughput_mbps"),
            "interference_loss_percent": _mean(interference, "protected_lost_percent"),
            "interference_jitter_ms": _mean(interference, "protected_jitter_ms"),
            "recovery_goodput_mbps": _mean(recovery, "protected_throughput_mbps"),
            "recovery_to_90pct_seconds": recovery_seconds,
            "recovery_right_censored": recovery_censored,
            "median_ack_latency_ms": _median(latencies) if latencies else None,
            "max_ack_latency_ms": max(latencies) if latencies else None,
        })
        runs.append(document)
        if document["status"] != "PASS":
            attempts.append(document)

    accepted = [run for run in runs if run["status"] == "PASS"]
    blocks: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in accepted:
        blocks[run["block_id"]][run["method"]] = run

    endpoint_orientation = {
        "interference_goodput_mbps": 1.0,
        "interference_loss_percent": -1.0,
        "interference_jitter_ms": -1.0,
        "recovery_to_90pct_seconds": -1.0,
    }
    comparisons: list[dict[str, Any]] = []
    for block_id, methods in sorted(blocks.items()):
        reference = methods.get("A0")
        if reference is None:
            continue
        for method, run in sorted(methods.items()):
            if method == "A0":
                continue
            item = {
                "block_id": block_id,
                "method": method,
                "jammer_type": run["jammer_type"],
                "jsr_db": run["jsr_db"],
                "traffic_load": run["traffic_load"],
                "attenuation": run["attenuation"],
            }
            for endpoint, orientation in endpoint_orientation.items():
                raw_delta = float(run[endpoint]) - float(reference[endpoint])
                item[f"{endpoint}_action_minus_a0"] = raw_delta
                item[f"{endpoint}_improvement"] = orientation * raw_delta
            comparisons.append(item)

    summaries: dict[str, dict[str, Any]] = {}
    for endpoint in endpoint_orientation:
        by_method: dict[str, list[float]] = defaultdict(list)
        for item in comparisons:
            by_method[item["method"]].append(float(item[f"{endpoint}_improvement"]))
        raw_p = [(method, _wilcoxon(values)) for method, values in sorted(by_method.items())]
        adjusted = _holm(raw_p)
        summaries[endpoint] = {}
        for index, (method, values) in enumerate(sorted(by_method.items()), start=1):
            summaries[endpoint][method] = {
                "matched_blocks": len(values),
                "mean_improvement": _average(values),
                "median_improvement": _median(values),
                "bootstrap_95pct_ci_mean": _bootstrap_mean_ci(values, 20261020 + index),
                "positive_blocks": sum(value > 0 for value in values),
                "wilcoxon_p_two_sided": dict(raw_p)[method],
                "holm_adjusted_p": adjusted[method],
            }

    strata: list[dict[str, Any]] = []
    for factor in ("jammer_type", "jsr_db", "traffic_load", "attenuation"):
        grouped: dict[tuple[str, Any], list[float]] = defaultdict(list)
        for item in comparisons:
            grouped[(item["method"], item[factor])].append(float(item["interference_goodput_mbps_improvement"]))
        for (method, level), values in sorted(grouped.items(), key=lambda entry: (entry[0][0], str(entry[0][1]))):
            strata.append({
                "factor": factor,
                "level": level,
                "method": method,
                "matched_blocks": len(values),
                "mean_goodput_improvement_mbps": _average(values),
                "median_goodput_improvement_mbps": _median(values),
            })

    return {
        "schema_version": 1,
        "evidence_role": "EXPLORATORY_REVIEWER_RESPONSE_NEW_DATA",
        "manifest_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_or_incomplete_rows": len(rows) - len(accepted),
        "accepted_windows": sum(int(run.get("complete_radio_windows") or 0) for run in accepted),
        "complete_matched_blocks": sum(
            set(blocks.get(block_id, {})) == methods
            for block_id, methods in expected_methods_by_block.items()
        ),
        "evidence_gate": "PASS" if len(accepted) == 84 and not attempts else "FAIL",
        "mandatory_interpretation": "exploratory actuator characterization; not a confirmatory efficacy estimate",
        "runs": runs,
        "matched_comparisons": comparisons,
        "endpoint_summaries": summaries,
        "factor_strata": strata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.manifest.resolve(), args.raw_root.resolve(), args.freeze.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "manifest_rows", "accepted_rows", "rejected_or_incomplete_rows",
        "accepted_windows", "complete_matched_blocks", "evidence_gate",
    )}, indent=2))
    return 0 if report["evidence_gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

