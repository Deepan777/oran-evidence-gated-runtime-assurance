#!/usr/bin/env python3
"""Build an auditable report for the prospective CEE qualification rows."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _decisions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _phase_means(windows: list[dict[str, Any]], pre: int, jam: int) -> dict[str, float]:
    phases = {
        "pre": windows[:pre],
        "interference": windows[pre:pre + jam],
        "recovery": windows[pre + jam:],
    }
    return {
        f"{phase}_protected_goodput_mbps": mean(float(row["protected_throughput_mbps"]) for row in rows)
        for phase, rows in phases.items()
    }


def _matched_development_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only standardized, non-smoke jammer blocks.

    These development contrasts select a pilot comparator.  They are never
    eligible for a confirmatory effect estimate or hypothesis test.
    """
    standardized = [
        item for item in runs
        if "-smoke-" not in item["run_id"]
        and item["jammer_type"] != "none"
        and item["status"] == "PASS"
    ]
    blocks: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in standardized:
        blocks[(item["jammer_type"], item["seed"])][item["method"]] = item

    comparisons: list[dict[str, Any]] = []
    for (family, seed), methods in sorted(blocks.items()):
        baseline = methods.get("A0")
        if baseline is None:
            continue
        for method, item in sorted(methods.items()):
            if method == "A0":
                continue
            comparisons.append({
                "jammer_type": family,
                "seed": seed,
                "method": method,
                "interference_goodput_difference_mbps": (
                    item["interference_protected_goodput_mbps"]
                    - baseline["interference_protected_goodput_mbps"]
                ),
                "recovery_goodput_difference_mbps": (
                    item["recovery_protected_goodput_mbps"]
                    - baseline["recovery_protected_goodput_mbps"]
                ),
            })

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in comparisons:
        by_method[item["method"]].append(item)
    candidate_summaries: dict[str, dict[str, Any]] = {}
    for method, items in sorted(by_method.items()):
        family_deltas: dict[str, list[float]] = defaultdict(list)
        for item in items:
            family_deltas[item["jammer_type"]].append(
                item["interference_goodput_difference_mbps"]
            )
        interference_deltas = [item["interference_goodput_difference_mbps"] for item in items]
        recovery_deltas = [item["recovery_goodput_difference_mbps"] for item in items]
        candidate_summaries[method] = {
            "matched_blocks": len(items),
            "positive_interference_blocks": sum(value > 0.0 for value in interference_deltas),
            "pooled_mean_interference_difference_mbps": mean(interference_deltas),
            "family_mean_interference_difference_mbps": {
                family: mean(values) for family, values in sorted(family_deltas.items())
            },
            "worst_recovery_difference_mbps": min(recovery_deltas),
        }

    qualified: list[str] = []
    for method in ("A2", "A3"):
        item = candidate_summaries.get(method)
        if not item:
            continue
        if (
            item["matched_blocks"] == 4
            and item["positive_interference_blocks"] >= 3
            and item["pooled_mean_interference_difference_mbps"] > 0.0
            and set(item["family_mean_interference_difference_mbps"]) == {"barrage", "partial_band"}
            and all(value > 0.0 for value in item["family_mean_interference_difference_mbps"].values())
            and item["worst_recovery_difference_mbps"] >= -0.09
        ):
            qualified.append(method)
    selected = max(
        qualified,
        key=lambda method: candidate_summaries[method]["pooled_mean_interference_difference_mbps"],
        default=None,
    )
    return {
        "interpretation": "development-only action screen; no hypothesis test",
        "matched_comparisons": comparisons,
        "candidate_summaries": candidate_summaries,
        "qualified_global_fixed_actions": qualified,
        "selected_global_fixed_action": selected,
    }
def build(manifest: Path, raw_root: Path) -> dict[str, Any]:
    manifest_rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    runs: list[dict[str, Any]] = []
    for order, row in enumerate(manifest_rows):
        evidence = raw_root / f"iq-broker-live-{row['run_id']}"
        traffic_path = evidence / "protected-traffic-status.json"
        summary_path = evidence / "records.summary.json"
        if not traffic_path.is_file() or not summary_path.is_file():
            runs.append({
                "order": order, "run_id": row["run_id"], "method": row["method"],
                "jammer_type": row["jammer_type"], "seed": int(row["seed"]),
                "status": "NOT_RUN_OR_INCOMPLETE",
            })
            continue
        traffic, summary = _json(traffic_path), _json(summary_path)
        radio_windows = traffic.get("radio_windows")
        traffic_status = traffic.get("status")
        evidence_rows = summary.get("evidence_complete_rows")
        if traffic_status != "PASS" or not isinstance(radio_windows, list) or evidence_rows != 30:
            finalized_failure = traffic_status == "FAIL"
            runs.append({
                "order": order, "run_id": row["run_id"], "method": row["method"],
                "jammer_type": row["jammer_type"], "seed": int(row["seed"]),
                "status": "FAIL" if finalized_failure else "NOT_RUN_OR_INCOMPLETE",
                "traffic_status": traffic_status,
                "evidence_complete_rows": evidence_rows,
            })
            continue
        decisions = _decisions(evidence / "live-xapp-decisions.jsonl")
        decision_rows = [item for item in decisions if item.get("status") == "DECISION"]
        actions = Counter(
            item.get("decision", {}).get("execution", {}).get("executed_action")
            for item in decision_rows
        )
        proposed = Counter(
            item.get("decision", {}).get("policy_selected_action")
            for item in decisions if item.get("status") == "DECISION"
        )
        reasons = Counter(
            item.get("safety_guard", {}).get("reason")
            for item in decisions if item.get("safety_guard")
        )
        pre, jam = int(row["pre_jam_seconds"]), int(row["jammer_on_seconds"])
        document: dict[str, Any] = {
            "order": order, "run_id": row["run_id"], "method": row["method"],
            "jammer_type": row["jammer_type"], "seed": int(row["seed"]),
            "status": "PASS",
            "complete_radio_windows": traffic.get("complete_radio_windows"),
            "delivery_observed_radio_windows": traffic.get("delivery_observed_radio_windows"),
            "outage_censored_radio_windows": traffic.get("outage_censored_radio_windows", []),
            "executed_actions": dict(actions), "policy_proposals": dict(proposed),
            "guard_reasons": dict(reasons),
            "all_decisions_acknowledged": all(
                item.get("decision", {}).get("execution", {}).get("acknowledged") is True
                and item.get("decision", {}).get("execution", {}).get("failure") is None
                for item in decision_rows
            ) if decision_rows else None,
            "recovery_non_a0_actions": sum(
                item.get("decision", {}).get("execution", {}).get("executed_action") != "NO_OP"
                for item in decision_rows[pre + jam:]
            ) if decision_rows else None,
        }
        document.update(_phase_means(radio_windows, pre, jam))
        runs.append(document)

    indexed = {item["run_id"]: item for item in runs}
    smoke_ids = [row["run_id"] for row in manifest_rows[:4]]
    smoke_complete = all(indexed[item]["status"] == "PASS" for item in smoke_ids)
    no_jam_safe = indexed[smoke_ids[1]]
    safe_no_jam_non_a0 = sum(
        count for action, count in no_jam_safe.get("executed_actions", {}).items()
        if action != "NO_OP"
    )
    partial_a0, partial_safe = indexed[smoke_ids[2]], indexed[smoke_ids[3]]
    effect = None
    if partial_a0["status"] == partial_safe["status"] == "PASS":
        effect = (
            partial_safe["interference_protected_goodput_mbps"]
            - partial_a0["interference_protected_goodput_mbps"]
        )
    development_summary = _matched_development_summary(runs)
    complete_count = sum(item["status"] == "PASS" for item in runs)
    fail_count = sum(item["status"] == "FAIL" for item in runs)
    qualification_complete = complete_count == len(manifest_rows) and fail_count == 0
    return {
        "schema_version": 1,
        "evidence_role": "DEVELOPMENT_ONLY_ENGINEERING_QUALIFICATION",
        "manifest_rows": len(manifest_rows), "completed_rows": complete_count,
        "failed_rows": fail_count,
        "qualification_complete": qualification_complete,
        "smoke_gate": {
            "status": "PASS" if smoke_complete and safe_no_jam_non_a0 == 0 else "FAIL",
            "four_rows_evidence_complete": smoke_complete,
            "safeguarded_no_jam_non_a0_actions": safe_no_jam_non_a0,
            "partial_band_safe_minus_a0_goodput_mbps": effect,
            "efficacy_interpretation": (
                "directional development signal only; not a hypothesis test and not eligible for the final manuscript effect estimate"
            ),
        },
        "development_action_screen": development_summary,
        "pilot_authorization": {
            "status": (
                "ELIGIBLE_FOR_CONTROLLER_REDESIGN_QUALIFICATION"
                if qualification_complete and development_summary["selected_global_fixed_action"]
                else "NOT_AUTHORIZED"
            ),
            "reason": (
                "a global fixed action met the frozen development selection rule; the safeguarded controller must still pass a bounded redesign qualification"
                if qualification_complete and development_summary["selected_global_fixed_action"]
                else "qualification is incomplete or no global fixed action met every frozen requirement"
            ),
        },
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.manifest, args.raw_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["smoke_gate"], indent=2, sort_keys=True))
    return 0 if report["smoke_gate"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
