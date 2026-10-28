"""Retrospective recovery sensitivity to the erroneous LTE-for-NR flag.

This module leaves the frozen cohort and primary recovery report unchanged.
It reconstructs the same threshold, persistence, and follow-up from measured
network fields while ignoring the recorded receiver-detachment classification.
Optional original protected-traffic windows add an endpoint-delivery check;
network-side SDU counters are never called end-user delivery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.recovery_censoring_correction import paired_binary
from analysis.statistics import bootstrap_ci


METHODS = ("P", "B0", "B2", "B5")
HORIZON = 20
OLD_EXCLUDED_LABELS = {"receiver_detached_outage", "right_censored_no_successful_sdu"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_rlc_fields(document: str) -> dict[str, float]:
    raw = json.loads(document)
    payload = raw.get("payload", raw)
    tx = payload["rlc_metrics"]["tx"]
    names = ("num_sdus", "num_dropped_sdus", "num_discarded_sdus", "sum_sdu_latency_us")
    values = {name: float(tx[name]) for name in names}
    if any(not np.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("RLC counter or accumulator is unavailable/non-finite/negative")
    return values


def first_persistent_event(qualifies: np.ndarray, persistence: int = 3) -> float | None:
    """Event time is the end of the third consecutive one-second window."""
    q = np.asarray(qualifies, dtype=bool)
    for first in range(len(q) - persistence + 1):
        if bool(q[first:first + persistence].all()):
            return float(first + persistence)
    return None


def load_protected(root: Path | None) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, str]]:
    indexed: dict[str, dict[int, dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    if root is None:
        return indexed, hashes
    for path in sorted(root.rglob("protected-traffic-status.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        run_id = str(document.get("run_id", ""))
        if not run_id.startswith("livev10-"):
            continue
        if run_id in indexed:
            raise ValueError(f"duplicate endpoint source for {run_id}")
        rows = document.get("radio_windows", [])
        by_window = {int(row["window_index"]): row for row in rows}
        if len(rows) != 120 or set(by_window) != set(range(120)):
            raise ValueError(f"endpoint source does not preserve 120 windows: {run_id}")
        indexed[run_id] = by_window
        hashes[run_id] = sha256(path)
    return indexed, hashes


def derive(frame: pd.DataFrame, endpoint: dict[int, dict[str, Any]] | None) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = frame.sort_values("window_index").copy()
    if frame.window_index.tolist() != list(range(120)):
        raise ValueError("run must preserve all windows 0..119")
    if frame.phase.value_counts().to_dict() != {"jammer_on": 80, "pre_jam": 20, "recovery": 20}:
        raise ValueError("phase split changed")
    pre = frame.loc[frame.phase.eq("pre_jam")]
    recovery = frame.loc[frame.phase.eq("recovery")].copy()
    reference = float(pre.dl_throughput_mbps.mean())
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("network-side pre-interference reference is not positive")
    network_values = (
        np.isfinite(recovery.dl_throughput_mbps)
        & (recovery.dl_throughput_mbps >= .9 * reference)
        & np.isfinite(recovery.rlc_sdu_delay_ms)
        & (recovery.rlc_sdu_delay_ms <= 20.)
    )
    old_qualifies = network_values & ~recovery.telemetry_observability.isin(OLD_EXCLUDED_LABELS)
    # This observed native counter is corroboration of a non-empty RLC interval;
    # it is not a guarantee that the receiver/application obtained that SDU.
    network_qualifies = network_values & (recovery.rlc_num_sdus > 0)
    old_time = first_persistent_event(old_qualifies.to_numpy())
    network_time = first_persistent_event(network_qualifies.to_numpy())
    row = {
        "run_id": str(frame.run_id.iloc[0]),
        "method": str(frame.method.iloc[0]),
        "source_run_id": str(frame.source_run_id.iloc[0]),
        "network_reference_mbps": reference,
        "old_mask_event_by_20s": int(old_time is not None),
        "network_only_event_by_20s": int(network_time is not None),
        "old_mask_event_time_seconds": old_time,
        "network_only_event_time_seconds": network_time,
        "network_only_restricted_time_seconds": HORIZON if network_time is None else network_time,
        "network_only_censored_at_seconds": HORIZON if network_time is None else None,
        "recovery_positive_gnb_goodput_windows": int((recovery.dl_throughput_mbps > 0).sum()),
        "recovery_positive_rlc_sdu_windows": int((recovery.rlc_num_sdus > 0).sum()),
        "recovery_finite_rlc_delay_windows": int(np.isfinite(recovery.rlc_sdu_delay_ms).sum()),
        "newly_qualifying_windows_without_old_mask": int((network_qualifies & ~old_qualifies).sum()),
        "endpoint_source_available": endpoint is not None,
    }
    recovery["old_mask_qualifies"] = old_qualifies
    recovery["network_only_qualifies"] = network_qualifies
    if endpoint is not None:
        for field in ("protected_packets", "protected_throughput_mbps", "traffic_coverage_fraction"):
            recovery["endpoint_" + field] = [endpoint[int(index)][field] for index in recovery.window_index]
        endpoint_observed = (
            (recovery.endpoint_traffic_coverage_fraction >= .90)
            & (recovery.endpoint_protected_packets > 0)
            & np.isfinite(recovery.endpoint_protected_throughput_mbps)
            & (recovery.endpoint_protected_throughput_mbps > 0)
        )
        corroborated = network_qualifies & endpoint_observed
        endpoint_time = first_persistent_event(corroborated.to_numpy())
        row.update({
            "endpoint_corroborated_event_by_20s": int(endpoint_time is not None),
            "endpoint_corroborated_event_time_seconds": endpoint_time,
            "endpoint_corroborated_restricted_time_seconds": HORIZON if endpoint_time is None else endpoint_time,
            "endpoint_zero_packet_recovery_windows": int((recovery.endpoint_protected_packets <= 0).sum()),
            "endpoint_complete_recovery_windows": int((recovery.endpoint_traffic_coverage_fraction >= .90).sum()),
        })
        recovery["endpoint_corroborated_qualifies"] = corroborated
    return row, recovery


def comparison_report(per_run: pd.DataFrame, column: str, time_column: str) -> dict[str, Any]:
    matrix = per_run.pivot(index="source_run_id", columns="method", values=column).loc[:, list(METHODS)]
    times = per_run.pivot(index="source_run_id", columns="method", values=time_column).loc[:, list(METHODS)]
    if matrix.shape != (90, 4) or matrix.isna().any().any():
        raise ValueError("sensitivity comparisons require all 90 complete blocks")
    contrasts = []
    for index, method in enumerate(METHODS[1:]):
        differences = times["P"].to_numpy() - times[method].to_numpy()
        contrasts.append({
            "contrast": "P-" + method,
            **paired_binary(matrix["P"], matrix[method], seed=6100 + index),
            "restricted_mean_difference_seconds": float(differences.mean()),
            "restricted_mean_paired_bootstrap_95ci": list(bootstrap_ci(differences, seed=6200 + index)),
        })
    return {
        "methods": {method: {
            "runs": 90,
            "events": int(matrix[method].sum()),
            "censored": int(90 - matrix[method].sum()),
            "restricted_mean_time_seconds": float(times[method].mean()),
        } for method in METHODS},
        "paired_comparisons": contrasts,
        "inference_scope": "Retrospective sensitivity, not a new confirmatory family; exact raw McNemar p values and 10,000 paired-block percentile bootstrap intervals (seeds 6100..6102 and 6200..6202). No family-adjusted significance claim.",
    }


def audit_counts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    counts = []
    for (method, phase, label), group in frame.groupby(["method", "phase", "telemetry_observability"], sort=True):
        counts.append({
            "method": method, "phase": phase, "recorded_observability_label": label,
            "windows": len(group),
            "receiver_flag_zero": int(group.receiver_attached.eq(0).sum()),
            "positive_gnb_goodput": int((group.dl_throughput_mbps > 0).sum()),
            "positive_rlc_sdu_counter": int((group.rlc_num_sdus > 0).sum()),
            "finite_rlc_delay": int(np.isfinite(group.rlc_sdu_delay_ms).sum()),
        })
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path("results/aggregate/live_validation_v10_records.parquet"))
    parser.add_argument("--protected-traffic-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("revision_2026-10-23/analysis_outputs/receiver_metadata_sensitivity"))
    args = parser.parse_args()
    # The aggregate also embeds large IQ/scheduler documents unrelated to this
    # sensitivity. Project only the fields actually used; estimands and rows
    # are unchanged, and low-memory reproduction need not decode every blob.
    columns=["run_id", "source_run_id", "method", "phase", "window_index",
             "dl_throughput_mbps", "rlc_sdu_delay_ms", "telemetry_observability",
             "receiver_attached", "raw_ocudu_rlc_json"]
    frame = pd.read_parquet(args.records, columns=columns, use_threads=False)
    if len(frame) != 43_200 or frame.run_id.nunique() != 360 or set(frame.method) != set(METHODS):
        raise ValueError("frozen accepted cohort cardinalities changed")
    raw = pd.DataFrame([raw_rlc_fields(text) for text in frame.raw_ocudu_rlc_json], index=frame.index)
    frame["rlc_num_sdus"] = raw.num_sdus
    native_delay = raw.sum_sdu_latency_us / (raw.num_sdus.where(raw.num_sdus > 0) * 1000.)
    observed_delay = frame.rlc_sdu_delay_ms
    valid = np.isfinite(observed_delay)
    if not bool(np.isclose(native_delay[valid], observed_delay[valid], rtol=1e-10, atol=1e-10).all()):
        raise ValueError("embedded native SDU/delay records disagree with derived delay")
    endpoint, endpoint_hashes = load_protected(args.protected_traffic_root)
    rows, windows = [], []
    for run_id, group in frame.groupby("run_id", sort=True):
        row, recovery = derive(group, endpoint.get(run_id))
        rows.append(row)
        windows.append(recovery)
    per_run = pd.DataFrame(rows)
    network = comparison_report(per_run, "network_only_event_by_20s", "network_only_restricted_time_seconds")
    changes = per_run.loc[per_run.old_mask_event_by_20s != per_run.network_only_event_by_20s]
    report = {
        "schema_version": 1,
        "analysis_status": "EXPLICITLY_RETROSPECTIVE_RECEIVER_METADATA_SENSITIVITY",
        "input": {"path": args.records.as_posix(), "sha256": sha256(args.records), "runs": 360, "windows": 43_200, "matched_blocks": 90},
        "preservation": "Frozen raw evidence, original observability labels, and primary endpoint reports are not rewritten.",
        "receiver_flag_audit": {
            "rows_flag_zero": int(frame.receiver_attached.eq(0).sum()),
            "rows_flag_one": int(frame.receiver_attached.eq(1).sum()),
            "complete_measured_rows_despite_flag_zero": int((frame.receiver_attached.eq(0) & frame.telemetry_observability.eq("complete_measured")).sum()),
            "classification_logic": "Builder assigns complete_measured first when required fields exist. Otherwise the old zero LTE-derived flag helps classify limited missing-field patterns as receiver_detached_outage. Thus zero flag is not universally an exclusion mask, and the label does not independently establish NR detachment.",
            "by_method_phase_and_recorded_label": audit_counts(frame),
        },
        "network_sensitivity_definition": {
            "threshold": "Same measured gNB UE-level downlink throughput >=90% of run-specific pre-interference mean and finite derived RLC SDU delay <=20 ms, with positive native RLC tx.num_sdus; ignore receiver_attached and recorded observability label.",
            "time_origin": "interference cessation; start of first recovery second",
            "persistence": "three consecutive one-second recovery windows",
            "event_time": "end of third qualifying window",
            "horizon_seconds": 20,
            "censoring": "No observed qualifying event is right-censored at 20 s, not counted as recovery at the cap.",
            "trust_boundary": "gNB scheduler and RLC software measurements; positive tx.num_sdus does not independently establish end-user delivery or NR RRC state.",
            "throughput_source": "raw_ocudu_scheduler_json.cells[0].ue_list[0].dl_brate / 1e6; UE-level aggregate scheduler rate, not isolated protected-flow goodput",
            "raw_fields": ["raw_ocudu_rlc_json.rlc_metrics.tx.num_sdus", "raw_ocudu_rlc_json.rlc_metrics.tx.sum_sdu_latency_us", "gnb_dl_throughput_mbps", "rlc_sdu_delay_ms"],
        },
        "network_sensitivity": network,
        "comparison_with_recorded_mask": {
            "old_event_counts": per_run.groupby("method").old_mask_event_by_20s.sum().astype(int).to_dict(),
            "changed_event_runs": changes[["run_id", "method", "old_mask_event_by_20s", "network_only_event_by_20s"]].to_dict("records"),
            "qualifying_recovery_windows_added_without_old_mask": int(per_run.newly_qualifying_windows_without_old_mask.sum()),
            "event_time_changes": int(((per_run.old_mask_event_time_seconds.notna() | per_run.network_only_event_time_seconds.notna()) & (per_run.old_mask_event_time_seconds != per_run.network_only_event_time_seconds)).sum()),
        },
        "endpoint_evidence": {
            "available_runs": int(per_run.endpoint_source_available.sum()),
            "required_runs": 360,
            "file_sha256_by_run": endpoint_hashes,
            "corroboration_rule": "Same network recovery criterion, additionally require original protected-traffic window coverage>=0.90, positive received packets, and finite positive protected throughput. No substitution for absent endpoint windows.",
        },
    }
    if bool(per_run.endpoint_source_available.all()):
        report["endpoint_corroborated_sensitivity"] = comparison_report(per_run, "endpoint_corroborated_event_by_20s", "endpoint_corroborated_restricted_time_seconds")
    else:
        report["endpoint_evidence"]["status"] = "INCOMPLETE_ENDPOINT_POPULATION_NO_ENDPOINT_EFFECT_ESTIMATE"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.output_dir / "receiver_metadata_recovery_per_run.csv", index=False)
    keep = ["run_id", "method", "window_index", "phase", "receiver_attached", "telemetry_observability", "dl_throughput_mbps", "rlc_num_sdus", "rlc_sdu_delay_ms", "old_mask_qualifies", "network_only_qualifies"]
    joined = pd.concat(windows, ignore_index=True)
    keep += [column for column in joined if column.startswith("endpoint_")]
    joined[keep].to_csv(args.output_dir / "receiver_metadata_recovery_windows.csv", index=False)
    (args.output_dir / "receiver_metadata_recovery_sensitivity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"network": network["methods"], "changes": report["comparison_with_recorded_mask"], "endpoint_runs": int(per_run.endpoint_source_available.sum())}, indent=2))


if __name__ == "__main__":
    main()
