#!/usr/bin/env python3
"""Fit a detector from immutable development live-calibration evidence only.

Truth is read solely to create evaluation labels after collection.  The model
itself receives measured OCUDU/srsUE/IQ-estimator features, and missing live
measurements are filled causally per run using training-split means.
"""
from __future__ import annotations

import argparse
import csv
import json
from time import perf_counter_ns
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from experiments.collection.build_live_run_records import REQUIRED_TELEMETRY, build_records
from src.detector.training import select_simplest_detector


ROOT = Path(__file__).resolve().parents[2]
FEATURES = tuple(REQUIRED_TELEMETRY) + ("clean_prb_fraction",)
CLASSES = ("none", "barrage", "partial_band", "sweep", "intermittent", "reactive")


def _load_manifest(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError("live calibration manifest is empty")
    if {row.get("evidence_class") for row in rows} != {"development_live_calibration"}:
        raise ValueError("live detector calibration accepts development_live_calibration only")
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("live calibration manifest contains duplicate run IDs")
    return rows


def _validated_frame(root: Path, rows: list[dict[str, str]], persist: bool) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    missing: list[str] = []
    for row in rows:
        evidence = root / f"iq-broker-live-{row['run_id']}"
        metadata_path = evidence / "run_metadata.json"
        clock_path = evidence / "radio-clock-status.json"
        summary_path = evidence / "records.summary.json"
        if not (metadata_path.is_file() and clock_path.is_file() and summary_path.is_file()):
            missing.append(row["run_id"]); continue
        clock, summary = json.loads(clock_path.read_text()), json.loads(summary_path.read_text())
        if clock.get("status") != "PASS" or summary.get("complete_rows") != int(row["run_length_seconds"]):
            raise ValueError(f"ineligible live calibration evidence: {row['run_id']}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("evidence_class") != "development_live_calibration":
            raise ValueError(f"unexpected evidence class for {row['run_id']}")
        frame = build_records(evidence, metadata)
        artifact = evidence / "records_controller_raw_v2.parquet"
        if persist:
            frame.to_parquet(artifact, index=False)
            artifact.with_suffix(".summary.json").write_text(json.dumps({
                "rows": len(frame), "complete_telemetry_rows": int(frame.complete_telemetry.sum()),
                "complete_controller_state_rows": int(frame.complete_controller_state.sum()),
                "derived_from": "raw live evidence; no synthetic telemetry",
            }, indent=2), encoding="utf-8")
        records.append(frame.assign(partition=row["partition"]))
    if missing:
        raise ValueError(f"BLOCKED_INCOMPLETE_LIVE_CALIBRATION: missing evidence for {missing}")
    return pd.concat(records, ignore_index=True)


def causal_impute(frame: pd.DataFrame, features: tuple[str, ...], means: pd.Series) -> pd.DataFrame:
    """Causally fill only unavailable measured values within each run."""
    completed = frame.copy()
    for _, indexes in completed.groupby("run_id", sort=False).groups.items():
        previous: dict[str, float] = {}
        for index in sorted(indexes, key=lambda row: int(completed.at[row, "window_index"])):
            for feature in features:
                value = completed.at[index, feature] if feature in completed.columns else np.nan
                if pd.notna(value) and np.isfinite(float(value)):
                    previous[feature] = float(value)
                else:
                    completed.at[index, feature] = previous.get(feature, float(means[feature]))
    return completed


def _require_class_coverage(frame: pd.DataFrame, partition: str) -> None:
    present = set(frame.loc[frame.partition == partition, "ground_truth_jammer_type"])
    missing = sorted(set(CLASSES) - present)
    if missing:
        raise ValueError(f"BLOCKED_CLASS_COVERAGE_{partition.upper()}: {missing}")


def _metrics(model: Any, features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    start = perf_counter_ns()
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)
    inference_latency_ms = (perf_counter_ns() - start) / 1_000_000 / max(len(features), 1)
    model_classes = np.asarray(model.classes_, dtype=int)
    aligned = np.zeros((len(features), len(CLASSES)), dtype=float)
    aligned[:, model_classes] = probabilities
    precision, recall, _, support = precision_recall_fscore_support(
        labels, predicted, labels=np.arange(len(CLASSES)), zero_division=0,
    )
    one_hot = np.eye(len(CLASSES), dtype=float)[labels]
    brier = float(np.mean(np.sum((aligned - one_hot) ** 2, axis=1)))
    p_jam = 1.0 - aligned[:, 0]
    observed_jam = labels != 0
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    calibration: list[dict[str, float | int]] = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        selected = (p_jam >= lower) & ((p_jam < upper) if upper < 1.0 else (p_jam <= upper))
        if selected.any():
            confidence, observed = float(p_jam[selected].mean()), float(observed_jam[selected].mean())
            count = int(selected.sum())
            ece += count / len(labels) * abs(confidence - observed)
            calibration.append({"lower": float(lower), "upper": float(upper), "count": count,
                                "mean_predicted_p_jam": confidence, "observed_jam_rate": observed})
    negative = labels == 0
    positive = ~negative
    false_alarm = float((predicted[negative] != 0).mean()) if negative.any() else None
    missed_detection = float((predicted[positive] == 0).mean()) if positive.any() else None
    try:
        auroc = float(roc_auc_score(one_hot, aligned, multi_class="ovr", average="macro"))
    except ValueError:
        auroc = None
    return {
        "macro_f1": float(f1_score(labels, predicted, labels=np.arange(len(CLASSES)), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "inference_latency_ms_per_window": float(inference_latency_ms),
        "macro_auroc_ovr": auroc,
        "multiclass_brier_score": brier,
        "expected_calibration_error": float(ece),
        "false_alarm_rate": false_alarm,
        "missed_detection_rate": missed_detection,
        "per_class": {name: {"precision": float(precision[index]), "recall": float(recall[index]),
                              "support": int(support[index])} for index, name in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(labels, predicted, labels=np.arange(len(CLASSES))).tolist(),
        "p_jam_calibration": calibration,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/metadata/manifests/live_calibration_runs.csv")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "results/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "results/training/live_detector")
    parser.add_argument("--no-persist-derived-records", action="store_true")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest(args.manifest)
    frame = _validated_frame(args.raw_root, rows, not args.no_persist_derived_records)
    for partition in ("train", "validation", "locked_test"):
        _require_class_coverage(frame, partition)
    train = frame[frame.partition == "train"].copy()
    means = train.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce").mean().replace([np.inf, -np.inf], np.nan)
    if means.isna().any():
        raise ValueError(f"BLOCKED_TRAINING_MEAN_UNAVAILABLE: {means[means.isna()].index.tolist()}")
    completed = causal_impute(frame, FEATURES, means)
    label_map = {name: index for index, name in enumerate(CLASSES)}
    labels = completed.ground_truth_jammer_type.map(label_map)
    if labels.isna().any():
        raise ValueError("unknown evaluation-only jammer label in live calibration evidence")
    train = completed[completed.partition == "train"]
    validation = completed[completed.partition == "validation"]
    test = completed[completed.partition == "locked_test"]
    selected = select_simplest_detector(train.loc[:, FEATURES].to_numpy(float), labels.loc[train.index].to_numpy(int), validation.loc[:, FEATURES].to_numpy(float), labels.loc[validation.index].to_numpy(int), 1101)
    artifact = {
        "schema_version": 1, "evidence_class": "development_live_calibration", "not_confirmatory": True,
        "model": selected.model, "feature_names": list(FEATURES), "class_names": list(CLASSES),
        "causal_training_means": {name: float(means[name]) for name in FEATURES},
    }
    model_path = args.output / "detector_live_calibrated.joblib"; joblib.dump(artifact, model_path)
    summary = {
        "status": "LIVE_DETECTOR_CALIBRATION_COMPLETE", "not_confirmatory": True,
        "model": selected.model_name, "features": list(FEATURES), "classes": list(CLASSES),
        "training_runs": sorted(train.run_id.unique().tolist()), "validation_runs": sorted(validation.run_id.unique().tolist()),
        "locked_test_runs": sorted(test.run_id.unique().tolist()),
        "validation": _metrics(selected.model, validation.loc[:, FEATURES].to_numpy(float), labels.loc[validation.index].to_numpy(int)),
        "locked_test": _metrics(selected.model, test.loc[:, FEATURES].to_numpy(float), labels.loc[test.index].to_numpy(int)),
        "model_file": str(model_path),
    }
    (args.output / "detector_live_calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
