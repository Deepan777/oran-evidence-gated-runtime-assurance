#!/usr/bin/env python3
"""Strict, frozen post-processing for the 360-run V9 live campaign."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from analysis.evidence_gate import evaluate_evidence, write_report
from analysis.live_validation_v9_publication import generate_publication_artifacts
from analysis.statistics import build_statistics_report
from experiments.collection.run_manifest import _complete
from src.common.hashing import sha256_file


METHODS = ("P", "B0", "B2", "B5")
OUTCOMES = {
    "protected_flow_latency_violation_rate": {"direction": "lower", "minimum": 0.05, "kind": "absolute"},
    "protected_flow_throughput_retention": {"direction": "higher", "minimum": 0.05, "kind": "relative"},
    "protected_flow_packet_or_rlc_drop_rate": {"direction": "lower", "minimum": 0.01, "kind": "absolute"},
    "recovery_time": {"direction": "lower", "minimum": 1.0, "kind": "absolute"},
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def derive_run_outcomes(records: pd.DataFrame) -> dict[str, float]:
    if len(records) != 120 or records["window_index"].tolist() != list(range(120)):
        raise ValueError("a V9 run must contain ordered windows 0..119")
    pre = records.loc[records["phase"] == "pre_jam"]
    jam = records.loc[records["phase"] == "jammer_on"]
    recovery = records.loc[records["phase"] == "recovery"]
    if (len(pre), len(jam), len(recovery)) != (20, 80, 20):
        raise ValueError("a V9 run must contain the frozen 20/80/20 phase split")
    pre_throughput = float(pre["dl_throughput_mbps"].mean())
    if not np.isfinite(pre_throughput) or pre_throughput <= 0.0:
        raise ValueError("pre-jam throughput must be finite and positive")
    threshold = 0.90 * pre_throughput
    recovery_throughput = recovery["dl_throughput_mbps"].to_numpy(dtype=float)
    recovery_delay = recovery["rlc_sdu_delay_ms"].to_numpy(dtype=float)
    recovery_observability = recovery["telemetry_observability"].astype(str).to_numpy()
    recovered = (
        np.isfinite(recovery_throughput)
        & np.isfinite(recovery_delay)
        & (recovery_throughput >= threshold)
        & (recovery_delay <= 20.0)
        & ~np.isin(recovery_observability, ["right_censored_no_successful_sdu", "receiver_detached_outage"])
    )
    recovery_time = 20.0
    for index in range(0, len(recovered) - 2):
        if bool(np.all(recovered[index:index + 3])):
            recovery_time = float(index)
            break
    jam_delay = jam["rlc_sdu_delay_ms"].to_numpy(dtype=float)
    jam_observability = jam["telemetry_observability"].astype(str).to_numpy()
    latency_violation = (
        ~np.isfinite(jam_delay)
        | (jam_delay > 20.0)
        | np.isin(jam_observability, ["right_censored_no_successful_sdu", "receiver_detached_outage"])
    )
    return {
        "protected_flow_latency_violation_rate": float(latency_violation.mean()),
        "protected_flow_throughput_retention": float(jam["dl_throughput_mbps"].mean() / pre_throughput),
        "protected_flow_packet_or_rlc_drop_rate": float(jam["rlc_drop_rate"].mean()),
        "recovery_time": recovery_time,
    }


def _manifest_rows(root: Path, method: str) -> list[dict[str, str]]:
    path = root / f"datasets/metadata/manifests/live_validation_v9/{method}_live_validation_runs.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 90 or any(row.get("method") != method for row in rows):
        raise ValueError(f"{method} manifest is not the frozen 90-row method manifest")
    return rows


def load_strict_campaign(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    record_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    explanations: list[dict[str, Any]] = []
    common_blocks: set[str] | None = None
    for method in METHODS:
        rows = _manifest_rows(root, method)
        blocks = {row["source_run_id"] for row in rows}
        if common_blocks is None:
            common_blocks = blocks
        elif blocks != common_blocks:
            raise ValueError("method manifests do not contain the same 90 matched blocks")
        result_path = root / f"results/raw/LIVE_VALIDATION_V9_{method}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("accepted_count") != 90 or result.get("remaining") != 0:
            raise ValueError(f"{method} result manifest is not exactly complete")
        for row in rows:
            evidence = root / "results/raw" / f"iq-broker-live-{row['run_id']}"
            if not _complete(evidence, 120, row):
                raise ValueError(f"strict evidence rejection: {row['run_id']}")
            frame = pd.read_parquet(evidence / "records.parquet").sort_values("window_index")
            if frame["run_id"].nunique() != 1 or frame["run_id"].iloc[0] != row["run_id"]:
                raise ValueError(f"record identity mismatch: {row['run_id']}")
            outcomes = derive_run_outcomes(frame)
            metadata = json.loads((evidence / "run_metadata.json").read_text(encoding="utf-8"))
            start_time_ns = int(metadata["measurement_start_ns"])
            truth = _jsonl(evidence / "jammer_truth.jsonl")
            truth_timestamps = [int(item["timestamp_ns"]) for item in truth if item.get("timestamp_ns") is not None]
            end_time_ns = max(truth_timestamps) if truth_timestamps else start_time_ns
            if end_time_ns < start_time_ns:
                raise ValueError(f"invalid evidence time bounds: {row['run_id']}")
            scenario_definition = {
                key: row[key] for key in (
                    "scenario_id", "jammer_type", "jsr_db", "traffic_load", "service_mix",
                    "attenuation", "seed", "run_length_seconds", "pre_jam_seconds",
                    "jammer_on_seconds", "recovery_seconds",
                )
            }
            run_rows.append({
                "method": method, "run_id": row["run_id"], "source_run_id": row["source_run_id"],
                "scenario_id": row["scenario_id"], "seed": int(row["seed"]),
                "jammer_type": row["jammer_type"],
                "jsr_db": float(row["jsr_db"]) if row["jsr_db"] else np.nan,
                "traffic_load": row["traffic_load"], "service_mix": row["service_mix"],
                "attenuation": row["attenuation"], **outcomes,
                "start_time_ns": start_time_ns,
                "end_time_ns": end_time_ns,
                "raw_result_path": str(evidence),
                "scenario_definition": json.dumps(scenario_definition, sort_keys=True),
                "frozen_provenance_id": "LIVE_VALIDATION_V9_CAMPAIGN_PROVENANCE",
            })
            frame = frame.assign(
                method=method, source_run_id=row["source_run_id"],
                jammer_type=row["jammer_type"], traffic_load=row["traffic_load"],
                service_mix=row["service_mix"], attenuation=row["attenuation"],
            )
            record_frames.append(frame)
            if method != "B0":
                method_decisions = _jsonl(evidence / "live-xapp-decisions.jsonl")
                for item in method_decisions:
                    item.update({"method": method, "run_id": row["run_id"], "source_run_id": row["source_run_id"]})
                decisions.extend(method_decisions)
            if method == "P":
                method_xai = _jsonl(evidence / "live-xapp-xai.jsonl")
                for item in method_xai:
                    item.update({"run_id": row["run_id"], "source_run_id": row["source_run_id"]})
                explanations.extend(method_xai)
    combined = pd.concat(record_frames, ignore_index=True)
    per_run = pd.DataFrame(run_rows)
    if len(combined) != 43_200 or len(per_run) != 360 or len(decisions) != 32_400 or len(explanations) != 10_800:
        raise ValueError("campaign cardinality differs from the frozen 360 x 120 design")
    return combined, per_run, decisions, explanations


def statistics_input(per_run: pd.DataFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1, "confidence": 0.95, "bootstrap_iterations": 10_000,
        "seed": 1001, "replication_unit": "source_run_id", "outcomes": {},
    }
    for outcome, definition in OUTCOMES.items():
        pivot = per_run.pivot(index="source_run_id", columns="method", values=outcome).sort_index()
        ordered = pivot[list(METHODS)]
        if list(ordered.columns) != list(METHODS) or ordered.isna().any().any() or len(pivot) != 90:
            raise ValueError(f"{outcome} does not contain 90 complete matched blocks")
        payload["outcomes"][outcome] = {
            "methods": list(METHODS), "direction": definition["direction"],
            "matched_blocks": ordered.to_numpy(dtype=float).tolist(),
            "planned_pairs": [["P", "B0"], ["P", "B2"], ["P", "B5"]],
        }
    return payload


def _strongest_baseline(stats_report: dict[str, Any], outcome: str) -> str:
    rendered = stats_report["outcomes"][outcome]
    direction = rendered["direction"]
    means = {method: rendered["summaries"][method]["mean"] for method in METHODS[1:]}
    return (max if direction == "higher" else min)(means, key=means.get)


def supported_outcomes(stats_report: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    supported: list[str] = []
    audit: dict[str, Any] = {}
    for outcome, definition in OUTCOMES.items():
        rendered = stats_report["outcomes"][outcome]
        baseline = _strongest_baseline(stats_report, outcome)
        p_mean = float(rendered["summaries"]["P"]["mean"])
        b_mean = float(rendered["summaries"][baseline]["mean"])
        comparison = next(item for item in rendered["planned_comparisons"] if item["right"] == baseline)
        if definition["direction"] == "higher":
            effect = (p_mean - b_mean) / max(abs(b_mean), 1e-12) if definition["kind"] == "relative" else p_mean - b_mean
            direction_ok = comparison["rank_biserial"] > 0.0
        else:
            effect = b_mean - p_mean
            direction_ok = comparison["rank_biserial"] < 0.0
        passed = bool(
            effect >= float(definition["minimum"])
            and direction_ok
            and comparison["holm"]["adjusted_p"] <= 0.05
        )
        audit[outcome] = {
            "strongest_baseline": baseline, "proposed_mean": p_mean, "baseline_mean": b_mean,
            "practical_effect": effect, "minimum": definition["minimum"],
            "rank_biserial": comparison["rank_biserial"],
            "holm_adjusted_p": comparison["holm"]["adjusted_p"], "supported": passed,
        }
        if passed:
            supported.append(outcome)
    return supported, audit


def xai_summary(decisions: list[dict[str, Any]], explanations: list[dict[str, Any]]) -> dict[str, Any]:
    p_decisions = [item for item in decisions if item["method"] == "P"]
    applicable = [item for item in explanations if item["status"] == "XAI_COMPLETE"]
    non_applicable = [item for item in explanations if item["status"] == "XAI_NOT_APPLICABLE_SINGLE_FEASIBLE_ACTION"]
    if len(applicable) + len(non_applicable) != 10_800:
        raise ValueError("P XAI rows are missing or have an unknown status")
    faith = np.asarray([item["faithfulness_ratio"] for item in applicable], dtype=float)
    spear = np.asarray([item["stability_spearman"] for item in applicable], dtype=float)
    jaccard = np.asarray([item["stability_top_k_jaccard"] for item in applicable], dtype=float)
    cf = np.asarray([item["counterfactual_valid"] for item in applicable], dtype=bool)
    loop_latency = np.asarray([
        item["decision"]["latency_ms"]["total_end_to_end_loop_latency"] for item in p_decisions
    ], dtype=float)
    worker_latency = np.asarray([item["latency_ms"]["total_xai_worker"] for item in applicable], dtype=float)
    return {
        "schema_version": 1, "total_p_decisions": len(p_decisions),
        "applicable_explanations": len(applicable), "non_applicable_single_action": len(non_applicable),
        "median_faithfulness_ratio": float(np.median(faith)) if len(faith) else None,
        "median_stability_spearman": float(np.median(spear)) if len(spear) else None,
        "median_stability_top_k_jaccard": float(np.median(jaccard)) if len(jaccard) else None,
        "counterfactual_valid_fraction": float(cf.mean()) if len(cf) else None,
        "p95_online_control_loop_latency_ms": float(np.quantile(loop_latency, 0.95)),
        "p95_post_run_xai_worker_latency_ms": float(np.quantile(worker_latency, 0.95)) if len(worker_latency) else None,
        "execution_mode": "post_run_nonblocking_worker",
    }


def robustness_summary(per_run: pd.DataFrame, stats_report: dict[str, Any]) -> dict[str, Any]:
    strongest = {outcome: _strongest_baseline(stats_report, outcome) for outcome in OUTCOMES}
    no_jam = per_run.loc[per_run["jammer_type"] == "none"]
    no_jam_checks: dict[str, Any] = {}
    limits = {
        "protected_flow_throughput_retention": 0.05,
        "protected_flow_latency_violation_rate": 0.05,
        "protected_flow_packet_or_rlc_drop_rate": 0.05,
    }
    for outcome, limit in limits.items():
        baseline = strongest[outcome]
        p_mean = float(no_jam.loc[no_jam["method"] == "P", outcome].mean())
        b_mean = float(no_jam.loc[no_jam["method"] == baseline, outcome].mean())
        if OUTCOMES[outcome]["direction"] == "higher":
            regression = max(0.0, (b_mean - p_mean) / max(abs(b_mean), 1e-12))
        else:
            regression = max(0.0, p_mean - b_mean)
        no_jam_checks[outcome] = {"baseline": baseline, "regression": regression, "limit": limit, "passed": regression <= limit}

    merged = per_run.pivot(index="source_run_id", columns="method", values=list(OUTCOMES)).sort_index()
    meta = per_run.loc[per_run["method"] == "P"].set_index("source_run_id")
    favorable = pd.DataFrame(index=merged.index)
    for outcome, definition in OUTCOMES.items():
        baseline = strongest[outcome]
        difference = merged[(outcome, "P")] - merged[(outcome, baseline)]
        favorable[outcome] = difference > 0.0 if definition["direction"] == "higher" else difference < 0.0
    favorable["jammer_type"] = meta.loc[favorable.index, "jammer_type"]
    favorable["jsr_db"] = meta.loc[favorable.index, "jsr_db"]
    favorable["seed"] = meta.loc[favorable.index, "seed"]
    outcome_columns = list(OUTCOMES)
    favorable["favorable_outcome_count"] = favorable[outcome_columns].sum(axis=1)
    jammer_medians = favorable.loc[favorable["jammer_type"] != "none"].groupby("jammer_type")[outcome_columns].median()
    jsr_medians = favorable.dropna(subset=["jsr_db"]).groupby("jsr_db")[outcome_columns].median()
    favorable_jammers = (jammer_medians.sum(axis=1) >= 2).to_dict()
    favorable_jsrs = (jsr_medians.sum(axis=1) >= 2).to_dict()
    seed_favorable = favorable.groupby("seed")[outcome_columns].median().sum(axis=1) >= 2
    return {
        "no_jam_checks": no_jam_checks,
        "no_major_unexplained_no_jam_regression": all(item["passed"] for item in no_jam_checks.values()),
        "favorable_jammer_families": {str(key): bool(value) for key, value in favorable_jammers.items()},
        "favorable_jsr_levels": {str(key): bool(value) for key, value in favorable_jsrs.items()},
        "benefit_across_multiple_jammer_conditions": sum(favorable_jammers.values()) >= 3 and sum(favorable_jsrs.values()) >= 2,
        "favorable_seed_fraction": float(seed_favorable.mean()),
        "conclusion_reproduces_across_seeds": float(seed_favorable.mean()) >= 0.60,
        "favorable_definition": "at least two of four primary outcomes favorable versus each outcome's strongest baseline",
    }


def traceability_audit(root: Path) -> dict[str, Any]:
    findings: list[str] = []
    freeze = yaml.safe_load((root / "LIVE_VALIDATION_FREEZE_MANIFEST_V9.yaml").read_text(encoding="utf-8"))
    if freeze.get("total_method_executions") != 360 or freeze.get("methods") != ["B0", "B2", "B5", "P"]:
        findings.append("V9 freeze cardinality or method order differs from the frozen design")
    for key in ("b0_v9_manifest", "b2_v9_manifest", "b5_v9_manifest", "p_v9_manifest", "v9_analysis_plan"):
        artifact = freeze["artifacts"][key]
        path = root / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            findings.append(f"frozen artifact mismatch: {key}")
    if not (root / "LIVE_VALIDATION_V9_ANALYSIS_PLAN.yaml").is_file():
        findings.append("missing pre-outcome V9 analysis plan")
    implementation = freeze["artifacts"].get("v9_native_receiver_attachment_implementation", {})
    for relative, expected in implementation.get("sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            findings.append(f"V9 implementation hash mismatch: {relative}")
    return {"schema_version": 1, "status": "PASS" if not findings else "FAIL", "critical_findings": findings}


def render_results(per_run: pd.DataFrame, stats_report: dict[str, Any], gate: dict[str, Any]) -> str:
    lines = ["# V9 Live O-RAN Validation Results", "", "Exactly 360 strict runs (90 matched scenarios × 4 methods) were analysed.", "", "## Primary outcomes", "", "| Outcome | P | B0 | B2 | B5 |", "|---|---:|---:|---:|---:|"]
    for outcome in OUTCOMES:
        summaries = stats_report["outcomes"][outcome]["summaries"]
        lines.append("| " + outcome + " | " + " | ".join(f"{summaries[m]['mean']:.6g}" for m in METHODS) + " |")
    lines.extend(["", f"Evidence gate: **{gate['status']}**", "", "The replication unit is one independently seeded matched scenario; one-second windows were not treated as independent replicates."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    combined, per_run, decisions, explanations = load_strict_campaign(root)
    aggregate = root / "results/aggregate"; statistics = root / "results/statistics"
    aggregate.mkdir(parents=True, exist_ok=True); statistics.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(aggregate / "live_validation_v9_records.parquet", index=False)
    per_run.to_csv(aggregate / "live_validation_v9_per_run_outcomes.csv", index=False)
    provenance_columns = [
        "run_id", "method", "seed", "start_time_ns", "end_time_ns", "raw_result_path",
        "scenario_definition", "frozen_provenance_id",
    ]
    per_run[provenance_columns].to_csv(
        aggregate / "live_validation_v9_run_provenance.csv", index=False
    )
    stats_input = statistics_input(per_run)
    (aggregate / "statistics_input.json").write_text(json.dumps(stats_input, indent=2), encoding="utf-8")
    stats_report = build_statistics_report(stats_input)
    (statistics / "stats_report.json").write_text(json.dumps(stats_report, indent=2), encoding="utf-8")
    (statistics / "live_validation_v9_stats_report.json").write_text(json.dumps(stats_report, indent=2), encoding="utf-8")
    supported, support_audit = supported_outcomes(stats_report)
    xai = xai_summary(decisions, explanations)
    (aggregate / "live_validation_v9_xai_summary.json").write_text(json.dumps(xai, indent=2), encoding="utf-8")
    robustness = robustness_summary(per_run, stats_report)
    traceability = traceability_audit(root)
    (statistics / "live_validation_v9_traceability_audit.json").write_text(json.dumps(traceability, indent=2), encoding="utf-8")
    evidence = {
        "schema_version": 1, "campaign": "LIVE_VALIDATION_V9", "accepted_unique_runs": 360,
        "supported_primary_resilience_outcomes": supported, "primary_outcome_support_audit": support_audit,
        **robustness,
        "explanations_faithful": xai["median_faithfulness_ratio"] is not None and xai["median_faithfulness_ratio"] >= 1.5,
        "explanations_stable": xai["median_stability_spearman"] is not None and xai["median_stability_spearman"] >= 0.80 and xai["median_stability_top_k_jaccard"] >= 0.60,
        "counterfactual_validity_strong": xai["counterfactual_valid_fraction"] is not None and xai["counterfactual_valid_fraction"] >= 0.80,
        "latency_compatible_with_decision_interval": xai["p95_online_control_loop_latency_ms"] < 1000.0,
        "traceability_audit_passed": traceability["status"] == "PASS",
        "xai_summary": xai,
    }
    (aggregate / "evidence_summary.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    gate = evaluate_evidence(evidence)
    write_report(gate, root / "results/evidence_gate.json", root / "results/evidence_gate/EVIDENCE_GATE_REPORT.md")
    result_text = render_results(per_run, stats_report, gate)
    (root / "results/LIVE_VALIDATION_V9_FINAL_RESULTS.md").write_text(result_text, encoding="utf-8")
    if gate["status"] == "PASS":
        (root / "manuscript/LIVE_VALIDATION_V9_RESULTS.md").write_text(result_text, encoding="utf-8")
    else:
        failure_text = result_text + "\n## Gate failures\n\n" + "\n".join(f"- {reason}" for reason in gate["reasons"]) + "\n"
        (root / "results/LIVE_VALIDATION_V9_FAILURE_ANALYSIS.md").write_text(failure_text, encoding="utf-8")
        (root / "results/failure_analysis.md").write_text(failure_text, encoding="utf-8")
    generate_publication_artifacts(root)
    print(json.dumps({"status": "PASS", "accepted_runs": 360, "evidence_gate": gate["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
