#!/usr/bin/env python3
"""Generate V9 publication artifacts strictly from machine-readable results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("P", "B0", "B2", "B5")


def _save_figure(figure: plt.Figure, base: Path, source: Path, producer: str) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    base.with_suffix(".producer.json").write_text(json.dumps({
        "producer": producer, "source": str(source),
        "outputs": [str(base.with_suffix(".png")), str(base.with_suffix(".pdf"))],
    }, indent=2), encoding="utf-8")


def _primary_tables(root: Path, stats: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for outcome, item in stats["outcomes"].items():
        for method in METHODS:
            summary = item["summaries"][method]
            summary_rows.append({
                "outcome": outcome, "direction": item["direction"], "method": method,
                "mean": summary["mean"], "ci95_low": summary["bootstrap_ci"][0],
                "ci95_high": summary["bootstrap_ci"][1], "median": summary["median"],
                "iqr_low": summary["iqr"][0], "iqr_high": summary["iqr"][1],
                "n_matched_blocks": item["n_matched_blocks"],
                "friedman_statistic": item["friedman"]["statistic"],
                "friedman_p": item["friedman"]["p_value"],
            })
        for comparison in item["planned_comparisons"]:
            comparison_rows.append({
                "outcome": outcome, "left": comparison["left"], "right": comparison["right"],
                "wilcoxon_statistic": comparison["statistic"], "raw_p": comparison["p_value"],
                "holm_adjusted_p": comparison["holm"]["adjusted_p"],
                "holm_significant": comparison["holm"]["significant"],
                "rank_biserial": comparison["rank_biserial"], "n_pairs": comparison["n_pairs"],
            })
    summaries = pd.DataFrame(summary_rows); comparisons = pd.DataFrame(comparison_rows)
    table_dir = root / "results/tables"; table_dir.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(table_dir / "live_validation_v9_primary_outcomes.csv", index=False)
    comparisons.to_csv(table_dir / "live_validation_v9_planned_comparisons.csv", index=False)
    return summaries, comparisons


def _outcome_figure(root: Path, per_run: pd.DataFrame, summaries: pd.DataFrame) -> None:
    outcomes = summaries["outcome"].drop_duplicates().tolist()
    source = root / "results/aggregate/figure_source_v9_primary_outcomes.csv"
    per_run[["source_run_id", "method", *outcomes]].to_csv(source, index=False)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for axis, outcome in zip(axes.flat, outcomes, strict=True):
        values = [per_run.loc[per_run["method"] == method, outcome].to_numpy(dtype=float) for method in METHODS]
        axis.boxplot(values, tick_labels=METHODS, showfliers=False)
        axis.set_title(outcome.replace("protected_flow_", "").replace("_", " "))
        axis.set_xlabel("Method")
        axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, root / "results/figures/live_validation_v9_primary_outcomes", source, "analysis/live_validation_v9_publication.py")


def _condition_figure(root: Path, per_run: pd.DataFrame, evidence: dict[str, Any]) -> None:
    support = evidence["primary_outcome_support_audit"]
    rows: list[dict[str, Any]] = []
    for outcome, audit in support.items():
        baseline = audit["strongest_baseline"]
        subset = per_run.loc[per_run["jammer_type"] != "none"]
        pivot = subset.pivot(index="source_run_id", columns="method", values=outcome)
        meta = subset.loc[subset["method"] == "P"].set_index("source_run_id")
        difference = pivot["P"] - pivot[baseline]
        if "throughput_retention" not in outcome:
            difference = -difference
        for family, indexes in meta.groupby("jammer_type").groups.items():
            rows.append({"outcome": outcome, "jammer_type": family, "median_favorable_difference": float(difference.loc[list(indexes)].median()), "baseline": baseline})
    source_frame = pd.DataFrame(rows)
    source = root / "results/aggregate/figure_source_v9_jammer_robustness.csv"
    source_frame.to_csv(source, index=False)
    matrix = source_frame.pivot(index="outcome", columns="jammer_type", values="median_favorable_difference")
    figure, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    axis.set_yticks(range(len(matrix.index)), [item.replace("protected_flow_", "").replace("_", " ") for item in matrix.index])
    axis.set_title("Median proposed-versus-strongest-baseline effect (favorable positive)")
    figure.colorbar(image, ax=axis, label="Favorable difference")
    _save_figure(figure, root / "results/figures/live_validation_v9_jammer_robustness", source, "analysis/live_validation_v9_publication.py")


def _xai_tables_and_figure(root: Path, xai: dict[str, Any]) -> None:
    thresholds = {
        "median_faithfulness_ratio": 1.5,
        "median_stability_spearman": 0.80,
        "median_stability_top_k_jaccard": 0.60,
        "counterfactual_valid_fraction": 0.80,
    }
    rows = [{"metric": metric, "observed": xai[metric], "threshold": threshold} for metric, threshold in thresholds.items()]
    frame = pd.DataFrame(rows)
    table = root / "results/tables/live_validation_v9_xai_metrics.csv"
    frame.to_csv(table, index=False)
    source = root / "results/aggregate/figure_source_v9_xai_metrics.csv"
    frame.to_csv(source, index=False)
    figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    positions = np.arange(len(frame)); width = 0.36
    axis.bar(positions - width / 2, frame["observed"], width, label="Observed")
    axis.bar(positions + width / 2, frame["threshold"], width, label="Gate threshold")
    axis.set_xticks(positions, [item.replace("median_", "").replace("_", " ") for item in frame["metric"]], rotation=25, ha="right")
    axis.legend(); axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, root / "results/figures/live_validation_v9_xai_metrics", source, "analysis/live_validation_v9_publication.py")


def _write_producer(path: Path, sources: list[str]) -> None:
    path.with_suffix(path.suffix + ".producer.json").write_text(
        json.dumps({"producer": "analysis/live_validation_v9_publication.py", "sources": sources}, indent=2),
        encoding="utf-8",
    )


def _manuscript(root: Path, gate: dict[str, Any], evidence: dict[str, Any]) -> None:
    if gate["status"] != "PASS":
        return
    supported = evidence["supported_primary_resilience_outcomes"]
    manuscript = root / "manuscript"; manuscript.mkdir(parents=True, exist_ok=True)
    text = """# Action-Explainable Closed-Loop Anti-Jamming Control for O-RAN

## 1. Introduction

Open and disaggregated radio access networks enable rapid control innovation while expanding the need for auditable, safety-constrained responses to radio-frequency interference. This work evaluates an action-explainable controller that detects jamming, selects only feasible mitigation actions, obtains explicit action acknowledgements, and produces post-hoc explanations without delaying the one-second control path.

## 2. Related Work

The study is positioned relative to explainable O-RAN and 6G surveys, EXPLORA, federated deep-reinforcement-learning anti-jamming, O-RAN jamming-mitigation xApps, modular xApp defenses, jamming-resilient resource reservation, and counterfactual reinforcement-learning explanation methods. Bibliographic anchors and verified identifiers are retained in the project methodology record.

## 3. System Model and Problem Formulation

The protected downlink operates in an OCUDU/Open5GS/srsRAN software-radio testbed with a near-RT RIC. The state contains the frozen 27-dimensional measured and inferred radio context. The five-action space is constrained by an explicit feasibility mask, and control is applied through acknowledged scheduler hooks. The threat model covers no jammer and five jammer families across the frozen JSR, load, service-mix, attenuation, and seed factors.

## 4. Proposed Methodology

The proposed method combines calibrated telemetry-based detection, a masked dueling double-DQN controller with prioritized replay, safe action parameterization, and contrastive/counterfactual explanation. Explanations are computed by a non-blocking worker from immutable online decision records after the real-radio action sequence completes; this prevents explanation cost from influencing the measured control trajectory.

## 5. Experimental Methodology

The validation uses 90 independently seeded matched scenarios for each of P, B0, B2, and B5. Every accepted run contains 120 real-radio one-second windows with a 20/80/20-second pre-jam, jammer-on, and recovery split. Synthetic clock-fill, incomplete telemetry, skipped decisions, and unacknowledged actions are ineligible. Friedman and paired Wilcoxon tests with global Holm correction are applied at the scenario-run replication level; one-second windows are not independent replicates.

## 6. Results and Discussion

Table data are generated in `results/tables/live_validation_v9_primary_outcomes.csv` and `results/tables/live_validation_v9_planned_comparisons.csv`. Distributional and condition-wise results appear in the machine-generated primary-outcome and jammer-robustness figures. The evidence gate supports the following predeclared primary outcomes: **SUPPORTED_OUTCOMES**. XAI population metrics and thresholds are generated in `results/tables/live_validation_v9_xai_metrics.csv`; online loop latency and post-run explanation latency are reported separately.

## 7. Limitations and Threats to Validity

The evaluation uses a software RF boundary and custom audited OCUDU control hooks rather than over-the-air RF hardware. The conclusions are limited to the frozen jammer families, JSRs, channel attenuation profiles, traffic envelopes, and seeds. The source bundle did not contain main-project Git metadata; this is disclosed and compensated by cryptographic hashes for frozen source-critical artifacts, models, manifests, images, environment, and external commits. Post-run explanations are faithful to immutable online decisions but are not synchronous operator-facing explanations.

## 8. Conclusion

Under the frozen V9 protocol, the evidence gate permits only the outcome claims listed in the generated claim-evidence map. Claims outside that map are intentionally omitted.

## References

The bibliography includes the verified Brik et al. O-RAN XAI survey (10.1109/COMST.2024.3510543), EXPLORA (10.1145/3629141), federated DRL anti-jamming (10.1109/TVT.2024.3359998), the retained O-RAN jamming-defense anchors, and the retained counterfactual/XRL explanation literature recorded in the implementation instructions.
""".replace("SUPPORTED_OUTCOMES", ", ".join(supported))
    manuscript_path = manuscript / "manuscript.md"
    manuscript_path.write_text(text, encoding="utf-8")
    claims = [{
        "claim_id": f"R{index:02d}", "manuscript_section": "Results and Discussion",
        "claim_text": f"The proposed method has supported evidence for {outcome} under the frozen V9 protocol.",
        "supporting_table_or_figure": "results/tables/live_validation_v9_primary_outcomes.csv; results/figures/live_validation_v9_primary_outcomes.png",
        "supporting_result_file": "results/statistics/live_validation_v9_stats_report.json",
        "statistical_test": "Friedman; paired Wilcoxon; global Holm-Bonferroni; paired rank-biserial",
        "notes": "Generated only after evidence-gate PASS",
    } for index, outcome in enumerate(supported, start=1)]
    with (manuscript / "claim_evidence_map.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(claims[0]))
        writer.writeheader(); writer.writerows(claims)
    _write_producer(manuscript_path, [
        "results/evidence_gate.json", "results/aggregate/evidence_summary.json",
        "results/statistics/live_validation_v9_stats_report.json",
    ])


def generate_publication_artifacts(root: Path) -> None:
    stats = json.loads((root / "results/statistics/live_validation_v9_stats_report.json").read_text(encoding="utf-8"))
    xai = json.loads((root / "results/aggregate/live_validation_v9_xai_summary.json").read_text(encoding="utf-8"))
    evidence = json.loads((root / "results/aggregate/evidence_summary.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "results/evidence_gate.json").read_text(encoding="utf-8"))
    per_run = pd.read_csv(root / "results/aggregate/live_validation_v9_per_run_outcomes.csv")
    summaries, _ = _primary_tables(root, stats)
    _outcome_figure(root, per_run, summaries)
    _condition_figure(root, per_run, evidence)
    _xai_tables_and_figure(root, xai)
    _manuscript(root, gate, evidence)


if __name__ == "__main__":
    generate_publication_artifacts(Path(__file__).resolve().parents[1])
