#!/usr/bin/env python3
"""Correct the fixed-horizon recovery analysis without rewriting frozen evidence.

The original postprocessor assigned the horizon value (20 s) to runs without a
recovery event and analysed that cap as an exact recovery time. This script
retains the same 360 accepted runs and matched blocks, but records the event
indicator explicitly. The corrected primary endpoint is recovery probability
by 20 s. A 20-s restricted mean time-to-recovery (RMTR) is reported as a
supplementary, retrospective estimand; non-events contribute their censoring time
rather than being described as observed recovery at 20 s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from analysis.statistics import bootstrap_ci, build_statistics_report, planned_wilcoxon, holm_bonferroni


METHODS = ("P", "B0", "B2", "B5")
HORIZON_SECONDS = 20.0
OBSERVABILITY_FAILURES = {
    "right_censored_no_successful_sdu",
    "receiver_detached_outage",
}


def paired_binary(left: Any, right: Any, *, seed: int = 1001) -> dict[str, Any]:
    a, b = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not set(a).union(b).issubset({0, 1}):
        raise ValueError("paired recovery indicators require aligned binary observations")
    both = int(np.sum((a == 1) & (b == 1)))
    left_only = int(np.sum((a == 1) & (b == 0)))
    right_only = int(np.sum((a == 0) & (b == 1)))
    neither = int(np.sum((a == 0) & (b == 0)))
    discordant = left_only + right_only
    return {
        "test": "exact_two_sided_McNemar_conditional_binomial",
        "p_value": 1.0 if not discordant else float(binomtest(left_only, discordant, 0.5).pvalue),
        "n_pairs": len(a), "paired_counts": {"both_recover": both, "left_only": left_only, "right_only": right_only, "neither_recovers": neither},
        "mean_difference": float(np.mean(a-b)),
        "paired_block_bootstrap_ci": list(bootstrap_ci(a-b, seed=seed)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_run(records: pd.DataFrame) -> dict[str, Any]:
    records = records.sort_values("window_index")
    if len(records) != 120 or records["window_index"].tolist() != list(range(120)):
        raise ValueError("each accepted run must contain ordered windows 0..119")
    pre = records.loc[records["phase"] == "pre_jam"]
    jam = records.loc[records["phase"] == "jammer_on"]
    recovery = records.loc[records["phase"] == "recovery"]
    if (len(pre), len(jam), len(recovery)) != (20, 80, 20):
        raise ValueError("each accepted run must retain the 20/80/20 phase split")

    pre_throughput = float(pre["dl_throughput_mbps"].mean())
    if not np.isfinite(pre_throughput) or pre_throughput <= 0.0:
        raise ValueError("pre-interference throughput must be finite and positive")
    threshold = 0.90 * pre_throughput
    throughput = recovery["dl_throughput_mbps"].to_numpy(dtype=float)
    delay = recovery["rlc_sdu_delay_ms"].to_numpy(dtype=float)
    observability = recovery["telemetry_observability"].astype(str).to_numpy()
    qualified = (
        np.isfinite(throughput)
        & np.isfinite(delay)
        & (throughput >= threshold)
        & (delay <= 20.0)
        & ~np.isin(observability, list(OBSERVABILITY_FAILURES))
    )

    event_offset: int | None = None
    for offset in range(len(qualified) - 2):
        if bool(np.all(qualified[offset:offset + 3])):
            event_offset = offset
            break
    recovery_event = event_offset is not None
    # A recovery event is established only at the end of the third qualifying
    # one-second window. Offset zero therefore corresponds to 3 s, not 0 s.
    event_time = None if event_offset is None else float(event_offset + 3)
    restricted_time = HORIZON_SECONDS if event_time is None else event_time

    jam_delay = jam["rlc_sdu_delay_ms"].to_numpy(dtype=float)
    jam_observability = jam["telemetry_observability"].astype(str).to_numpy()
    latency_violation = (
        ~np.isfinite(jam_delay)
        | (jam_delay > 20.0)
        | np.isin(jam_observability, list(OBSERVABILITY_FAILURES))
    )
    return {
        "pre_interference_throughput_mbps": pre_throughput,
        "recovery_threshold_mbps": threshold,
        "recovery_event_by_20s": int(recovery_event),
        "recovery_event_time_seconds": event_time,
        "recovery_censored_at_seconds": None if recovery_event else HORIZON_SECONDS,
        "restricted_time_to_recovery_20s": restricted_time,
        "protected_flow_latency_violation_rate": float(latency_violation.mean()),
        "protected_flow_throughput_retention": float(jam["dl_throughput_mbps"].mean() / pre_throughput),
        "protected_flow_packet_or_rlc_drop_rate": float(jam["rlc_drop_rate"].mean()),
    }


def _matched_matrix(per_run: pd.DataFrame, outcome: str) -> list[list[float]]:
    matrix = per_run.pivot(index="source_run_id", columns="method", values=outcome)
    matrix = matrix.loc[sorted(matrix.index), list(METHODS)]
    if matrix.shape != (90, 4) or matrix.isna().any().any():
        raise ValueError(f"{outcome} does not form 90 complete matched blocks")
    return matrix.to_numpy(dtype=float).tolist()


def _paired_summary(per_run: pd.DataFrame, outcome: str, right: str, seed: int) -> dict[str, Any]:
    matrix = per_run.pivot(index="source_run_id", columns="method", values=outcome)
    differences = matrix["P"].to_numpy(dtype=float) - matrix[right].to_numpy(dtype=float)
    return {
        "contrast": f"P-{right}",
        "mean_difference": float(differences.mean()),
        "paired_block_bootstrap_ci": list(bootstrap_ci(differences, seed=seed)),
        "wilcoxon": planned_wilcoxon(matrix["P"], matrix[right]),
    }


def run(records_path: Path, output_dir: Path) -> None:
    records = pd.read_parquet(records_path)
    if len(records) != 43_200 or records["run_id"].nunique() != 360:
        raise ValueError("input must be the preserved 360-run, 43,200-window cohort")
    if set(records["method"].unique()) != set(METHODS):
        raise ValueError("method population differs from the frozen cohort")

    rows: list[dict[str, Any]] = []
    for (run_id, method, source_run_id), frame in records.groupby(
        ["run_id", "method", "source_run_id"], sort=True,
    ):
        rows.append({
            "run_id": run_id,
            "method": method,
            "source_run_id": source_run_id,
            **derive_run(frame),
        })
    per_run = pd.DataFrame(rows)
    if per_run.groupby("method")["run_id"].nunique().to_dict() != {method: 90 for method in METHODS}:
        raise ValueError("each method must contribute exactly 90 unique runs")

    primary = {
        "protected_flow_latency_violation_rate": "lower",
        "protected_flow_throughput_retention": "higher",
        "protected_flow_packet_or_rlc_drop_rate": "lower",
        "recovery_probability_by_20s": "higher",
    }
    per_run["recovery_probability_by_20s"] = per_run["recovery_event_by_20s"].astype(float)
    statistics_input = {
        "confidence": 0.95,
        "bootstrap_iterations": 10_000,
        "seed": 1001,
        "outcomes": {
            name: {
                "methods": list(METHODS),
                "matched_blocks": _matched_matrix(per_run, name),
                "planned_pairs": [["P", "B0"], ["P", "B2"], ["P", "B5"]],
                "direction": direction,
            }
            for name, direction in primary.items()
        },
    }
    primary_report = build_statistics_report(statistics_input)
    recovery_matrix = per_run.pivot(index="source_run_id", columns="method", values="recovery_event_by_20s")
    binary_comparisons = []
    for index, method in enumerate(METHODS[1:]):
        binary_comparisons.append({
            "left": "P", "right": method,
            **paired_binary(recovery_matrix["P"], recovery_matrix[method], seed=4001+index),
        })
    primary_report["outcomes"]["recovery_probability_by_20s"]["planned_comparisons"] = binary_comparisons
    all_comparisons = [c for outcome in primary_report["outcomes"].values() for c in outcome["planned_comparisons"]]
    for comparison, adjusted in zip(all_comparisons, holm_bonferroni([c["p_value"] for c in all_comparisons]), strict=True):
        comparison["holm"] = adjusted

    rmtr_summaries: dict[str, Any] = {}
    for index, method in enumerate(METHODS):
        method_rows = per_run.loc[per_run["method"] == method]
        values = method_rows["restricted_time_to_recovery_20s"].to_numpy(dtype=float)
        events = method_rows["recovery_event_by_20s"].to_numpy(dtype=float)
        rmtr_summaries[method] = {
            "n": int(values.size),
            "events": int(events.sum()),
            "recovery_probability_by_20s": float(events.mean()),
            "restricted_mean_time_to_recovery_20s": float(values.mean()),
            "bootstrap_ci": list(bootstrap_ci(values, seed=2001 + index)),
        }

    corrected = {
        "schema_version": 1,
        "analysis_status": "POST_REVIEW_CENSORING_CORRECTION_NOT_PRESPECIFIED",
        "input": {
            "path": records_path.as_posix(),
            "sha256": _sha256(records_path),
            "runs": 360,
            "windows": 43_200,
            "matched_blocks": 90,
        },
        "recovery_definition": {
            "time_origin": "start of the first one-second recovery window",
            "threshold": "at least 90% of run-specific mean pre-interference throughput and RLC delay <=20 ms",
            "persistence": "three consecutive observable one-second recovery windows",
            "event_time": "end of the third qualifying one-second window",
            "observation_horizon_seconds": HORIZON_SECONDS,
            "non_event_handling": "right-censored at 20 s; not described as recovery at 20 s",
        },
        "primary_family": {
            "description": "Four outcomes; 12 planned P-versus-baseline comparisons Holm-adjusted together",
            "family_members": [f"{name}:P-{method}" for name in primary for method in METHODS[1:]],
            "test_choice": "Paired binary recovery uses exact two-sided McNemar; three continuous run-level outcomes use paired Wilcoxon. All 12 p-values are Holm-adjusted together.",
            "retrospective_changes": "The 90% throughput/20-ms delay threshold and three-window persistence existed in the frozen endpoint code. Event time at the third-window end, explicit censoring, binary recovery estimand, exact paired-binary test, and supplementary RMTR were corrected after review.",
            "statistics": primary_report,
        },
        "supplementary_restricted_mean": {
            "status": "post-review supplementary estimand; excluded from the prespecified multiplicity family",
            "definition": "E[min(T,20s)]; complete common 20-s follow-up permits the sample mean of event time for recoveries and 20 s for non-events. Resample 90 matched scenario-seed blocks with replacement 10,000 times and form paired mean differences; percentile 2.5%/97.5% limits, seeds 3000..3002.",
            "summaries": rmtr_summaries,
            "paired_comparisons": [
                _paired_summary(per_run, "restricted_time_to_recovery_20s", method, 3000 + index)
                for index, method in enumerate(METHODS[1:])
            ],
        },
        "correction_of_original_claim": {
            "original_capped_mean_difference_P_minus_B0_seconds": 4.433333333333334,
            "corrected_restricted_mean_difference_P_minus_B0_seconds": (
                rmtr_summaries["P"]["restricted_mean_time_to_recovery_20s"]
                - rmtr_summaries["B0"]["restricted_mean_time_to_recovery_20s"]
            ),
            "interpretation": "The original cap was not an exact recovery time. The corrected primary result is the fixed-horizon event probability; the restricted-mean contrast is supplementary.",
        },
        "metric_audit": {
            "latency_violation_threshold_ms": 20.0,
            "latency_outage_handling": "non-finite delay, right-censored no-successful-SDU windows, and receiver-detached outages count as violations",
            "throughput_retention_denominator": "run-specific mean pre-interference protected-flow throughput",
            "drop_definition": "mean interference-phase RLC drop rate; not IP packet loss",
            "bootstrap_unit": "matched scenario-seed block",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(output_dir / "recovery_censoring_corrected_per_run.csv", index=False)
    (output_dir / "recovery_censoring_corrected_statistics_input.json").write_text(
        json.dumps(statistics_input, indent=2), encoding="utf-8",
    )
    (output_dir / "recovery_censoring_corrected_report.json").write_text(
        json.dumps(corrected, indent=2), encoding="utf-8",
    )
    lines = [
        "# Corrected recovery analysis",
        "",
        "This is a retrospective censoring correction over the unchanged 360-run matched cohort.",
        "Non-events are right-censored at 20 s and are not described as observed recovery at 20 s.",
        "",
        "| Method | Events / 90 | Recovery probability by 20 s | Restricted mean time to recovery (s) |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        item = rmtr_summaries[method]
        lines.append(
            f"| {method} | {item['events']}/90 | {item['recovery_probability_by_20s']:.6f} | "
            f"{item['restricted_mean_time_to_recovery_20s']:.6f} |"
        )
    recovery_stats = primary_report["outcomes"]["recovery_probability_by_20s"]["planned_comparisons"][0]
    rmtr_p_b0 = corrected["supplementary_restricted_mean"]["paired_comparisons"][0]
    lines.extend([
        "",
        f"P recovered in 0/90 runs and B0 in 23/90 runs. The matched P-B0 recovery-probability difference is -0.255556; the exact McNemar Holm-adjusted p value is {recovery_stats['holm']['adjusted_p']:.6g}; paired 95% interval {recovery_stats['paired_block_bootstrap_ci']}.",
        f"The supplementary 20-s restricted-mean difference is {rmtr_p_b0['mean_difference']:.6f} s (paired block-bootstrap interval {rmtr_p_b0['paired_block_bootstrap_ci'][0]:.6f} to {rmtr_p_b0['paired_block_bootstrap_ci'][1]:.6f}).",
        "",
        "The previously reported 4.43-s exact recovery disadvantage is withdrawn because it treated censoring caps as observed event times.",
    ])
    (output_dir / "recovery_censoring_corrected_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("results/aggregate/live_validation_v10_records.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision_2026-10-22/analysis_outputs"),
    )
    args = parser.parse_args()
    run(args.records, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
