#!/usr/bin/env python3
"""Fit live-only preprocessing and prepare a controller for safe exploration.

This is not a policy-freezing script.  It uses measured calibration telemetry
and the calibrated detector to create the live state schema, then copies the
offline-pretrained M-D3QN weights as an explicit *initialization*.  The copied
weights cannot be used for confirmatory evaluation until bounded live
exploration, validation selection, and a separate freeze manifest succeed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from experiments.training.train_live_detector import (
    FEATURES as DETECTOR_FEATURES,
    _load_manifest,
    _validated_frame,
    causal_impute,
)
from src.controller.state import StateSchema
from src.environment.iq_calibrated_env import FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[2]
JAMMER_CLASSES = ("none", "barrage", "partial_band", "sweep", "intermittent", "reactive")
PREVIOUS_ACTION_FEATURES = tuple(f"previous_action_a{i}" for i in range(5))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _detector_states(frame: pd.DataFrame, detector_artifact: dict[str, object]) -> pd.DataFrame:
    model = detector_artifact.get("model")
    features = tuple(detector_artifact.get("feature_names", ()))
    classes = tuple(detector_artifact.get("class_names", ()))
    if model is None or features != DETECTOR_FEATURES or classes != JAMMER_CLASSES:
        raise ValueError("live detector artifact does not match the locked state inputs")
    probabilities = model.predict_proba(frame.loc[:, features].to_numpy(float))
    distribution = np.zeros((len(frame), len(JAMMER_CLASSES)), dtype=float)
    distribution[:, np.asarray(model.classes_, dtype=int)] = probabilities
    states = frame.copy()
    states["p_jam"] = 1.0 - distribution[:, 0]
    for index, name in enumerate(JAMMER_CLASSES):
        states[f"jammer_class_{name}"] = distribution[:, index]
    for action in range(5):
        states[f"previous_action_a{action}"] = float(action == 0)
    missing = sorted(set(FEATURE_NAMES) - set(states.columns))
    if missing:
        raise ValueError(f"live calibration cannot construct full controller state: {missing}")
    return states


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/metadata/manifests/live_calibration_runs.csv")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "results/raw")
    parser.add_argument("--detector", type=Path, default=ROOT / "results/training/live_detector/detector_live_calibrated.joblib")
    parser.add_argument("--offline-controller", type=Path, default=ROOT / "results/training/controller/md3qn_per_offline.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/training/live_controller")
    args = parser.parse_args()
    if not args.detector.is_file() or not args.offline_controller.is_file():
        raise SystemExit("BLOCKED_LIVE_CONTROLLER_PREPARATION: live detector or offline initializer is missing")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest(args.manifest)
    raw = _validated_frame(args.raw_root, rows, persist=True)
    train_raw = raw.loc[raw.partition == "train", list(DETECTOR_FEATURES)]
    means = train_raw.apply(pd.to_numeric, errors="coerce").mean().replace([np.inf, -np.inf], np.nan)
    if means.isna().any():
        raise SystemExit(f"BLOCKED_LIVE_CONTROLLER_PREPARATION: missing training means {means[means.isna()].index.tolist()}")
    completed = causal_impute(raw, DETECTOR_FEATURES, means)
    states = _detector_states(completed, joblib.load(args.detector))
    training = states.loc[states.partition == "train", list(FEATURE_NAMES)]
    numeric = training.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise SystemExit("BLOCKED_LIVE_CONTROLLER_PREPARATION: non-finite state input remains after causal imputation")
    means = numeric.mean(axis=0).to_numpy(dtype=np.float64)
    scales = np.maximum(numeric.std(axis=0, ddof=0).to_numpy(dtype=np.float64), 1e-6)
    raw_p01 = numeric.quantile(0.01).to_numpy(dtype=float)
    raw_p99 = numeric.quantile(0.99).to_numpy(dtype=float)
    # These are categorical one-hot state indicators, not continuous
    # telemetry. Calibration pilots begin at A0, so empirical quantiles alone
    # would incorrectly make a later valid action state out-of-distribution.
    # Use the declared binary domain and a neutral categorical prior for the
    # standardization; transition logs supply the actual action-conditioned
    # states used for candidate training below.
    for name in PREVIOUS_ACTION_FEATURES:
        index = FEATURE_NAMES.index(name)
        means[index] = 0.2
        scales[index] = 0.4
        raw_p01[index] = 0.0
        raw_p99[index] = 1.0
    schema = StateSchema(FEATURE_NAMES, means, scales)
    document = {
        "schema_version": 1,
        "status": "DEVELOPMENT_LIVE_SCHEMA_PREPARED",
        "evidence_class": "development_live_calibration",
        "not_confirmatory": True,
        "feature_names": list(FEATURE_NAMES),
        "means": schema.means.tolist(),
        "scales": schema.scales.tolist(),
        "clip_low": schema.clip_low,
        "clip_high": schema.clip_high,
        "raw_p01": raw_p01.tolist(),
        "raw_p99": raw_p99.tolist(),
        "fit_run_ids": sorted(states.loc[states.partition == "train", "run_id"].unique().tolist()),
        "validation_run_ids": sorted(states.loc[states.partition == "validation", "run_id"].unique().tolist()),
        "locked_test_run_ids": sorted(states.loc[states.partition == "locked_test", "run_id"].unique().tolist()),
        "detector_sha256": _digest(args.detector),
    }
    state_path = args.output / "state_schema.json"
    state_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    joblib.dump({"feature_names": list(FEATURE_NAMES), "means": schema.means, "scales": schema.scales,
                 "clip_low": schema.clip_low, "clip_high": schema.clip_high}, args.output / "scaler.pkl")
    initial = args.output / "md3qn_per_live_initialization.pt"
    # Loading and re-saving forces a format check without altering weights.
    torch.save(torch.load(args.offline_controller, map_location="cpu", weights_only=True), initial)
    summary = {
        "status": "LIVE_CONTROLLER_INITIALIZATION_PREPARED",
        "not_confirmatory": True,
        "policy_status": "OFFLINE_PRETRAINED_INITIALIZATION_REQUIRES_LIVE_EXPLORATION_AND_FREEZE",
        "state_schema": str(state_path), "state_schema_sha256": _digest(state_path),
        "scaler_sha256": _digest(args.output / "scaler.pkl"),
        "initialization": str(initial), "initialization_sha256": _digest(initial),
        "detector": str(args.detector), "detector_sha256": _digest(args.detector),
    }
    (args.output / "live_controller_preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
