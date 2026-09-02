#!/usr/bin/env python3
"""Freeze the methodology-locked confirmatory execution plan.

This generator does not run an experiment and never invents result files.  It
duplicates the immutable scenario matrix once per baseline/controller method,
pins every model/config hash, and emits the exact commands consumed by
``experiments.confirmatory.run``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("B0", "B1", "B2", "B3", "B4", "B5", "P")
PROJECT_REMOTE = "<project-root>"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remote(relative: str) -> str:
    return f"{PROJECT_REMOTE}/{relative}"


def _method_rows(base: list[dict[str, str]], method: str, model_paths: dict[str, str], schema: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in base:
        row = dict(source)
        row["method"] = method
        row["EVIDENCE_CLASS"] = "locked_confirmatory"
        row["ENABLE_LIVE_XAPP"] = "0" if method == "B0" else "1"
        row["LIVE_XAPP_XAI"] = "1" if method == "P" else "0"
        row["STATE_SCHEMA"] = schema
        row["LIVE_XAPP_SELECTION_MODE"] = "policy"
        row["LIVE_XAPP_FORCED_ACTION_ID"] = ""
        # Runtime reliability setting established during live calibration;
        # record it explicitly so the frozen execution cannot inherit a
        # shorter shell default.
        row["UE_TIMEOUT_MS"] = "5000"
        row["CONTROLLER_KIND"] = "md3qn_per"
        row["DETECTOR_MODEL"] = model_paths["detector"]
        row["CONTROLLER_MODEL"] = model_paths["proposed"]
        if method == "B1":
            row["LIVE_XAPP_SELECTION_MODE"] = "fixed_action"
            row["LIVE_XAPP_FORCED_ACTION_ID"] = "1"
        elif method == "B2":
            row["CONTROLLER_KIND"] = "heuristic"
            row["CONTROLLER_MODEL"] = model_paths["heuristic"]
        elif method == "B3":
            row["CONTROLLER_KIND"] = "standard_dqn"
            row["CONTROLLER_MODEL"] = model_paths["standard"]
        elif method == "B4":
            row["CONTROLLER_KIND"] = "masked_ppo"
            row["CONTROLLER_MODEL"] = model_paths["ppo"]
        elif method == "B5":
            row["CONTROLLER_KIND"] = "prb_only_masked_dqn"
            row["CONTROLLER_MODEL"] = model_paths["prb"]
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=ROOT / "datasets/metadata/manifests/confirmatory_runs.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "CONFIRMATORY_FREEZE_MANIFEST.yaml")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "datasets/metadata/manifests/confirmatory")
    parser.add_argument("--gnb-image", required=True)
    args = parser.parse_args()
    base = list(csv.DictReader(args.base_manifest.open(newline="", encoding="utf-8")))
    if len(base) != 5040:
        raise SystemExit(f"methodology lock requires exactly 5040 base confirmatory rows, found {len(base)}")
    expected = {"none", "barrage", "partial_band", "sweep", "intermittent", "reactive"}
    if {row["jammer_type"] for row in base} != expected:
        raise SystemExit("base confirmatory manifest does not cover all locked jammer families")
    model_paths = {
        "detector": _remote("results/training/live_detector/detector_live_calibrated.joblib"),
        "proposed": _remote("results/training/live_controller/md3qn_per_live_candidate.pt"),
        "standard": _remote("results/training/live_controller/standard_dqn_live_candidate.pt"),
        "prb": _remote("results/training/live_controller/prb_only_masked_dqn_live_candidate.pt"),
        "ppo": _remote("results/training/live_controller/masked_ppo_live_candidate.pt"),
        "heuristic": _remote("results/training/live_controller/heuristic_controller.json"),
    }
    schema = _remote("results/training/live_controller/state_schema.json")
    artifact_paths = {
        "methodology_lock": ROOT / "METHODOLOGY_LOCK.yaml",
        "protocol": ROOT / "configs/experiments/protocol.yaml",
        "base_manifest": args.base_manifest,
    }
    for label, relative in {
        "detector": "results/training/live_detector/detector_live_calibrated.joblib",
        "proposed_controller": "results/training/live_controller/md3qn_per_live_candidate.pt",
        "standard_dqn": "results/training/live_controller/standard_dqn_live_candidate.pt",
        "prb_only_dqn": "results/training/live_controller/prb_only_masked_dqn_live_candidate.pt",
        "masked_ppo": "results/training/live_controller/masked_ppo_live_candidate.pt",
        "heuristic": "results/training/live_controller/heuristic_controller.json",
        "state_schema": "results/training/live_controller/state_schema.json",
    }.items():
        artifact_paths[label] = ROOT / relative
    missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("cannot freeze before all live candidates exist: " + ", ".join(missing))
    execution_plan: list[dict[str, Any]] = []
    for method in METHODS:
        manifest_path = args.manifest_dir / f"{method}_confirmatory_runs.csv"
        rows = _method_rows(base, method, model_paths, schema)
        _write_csv(manifest_path, rows)
        result = f"results/raw/CONFIRMATORY_{method}.json"
        execution_plan.append({
            "method": method,
            "scenario_manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "result_manifest": result,
            "expected_runs": len(rows),
            "command": [
                "envs/oran-action-xai/bin/python",
                "experiments/collection/run_manifest.py",
                "--manifest", str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
                "--output", result,
                "--max-wall-seconds", "900",
                "--gnb-image", args.gnb_image,
            ],
        })
    document = {
        "schema_version": 1,
        "status": "CONFIRMATORY_FROZEN",
        "methodology_version": "1.0",
        "not_confirmatory_until_execution": True,
        "expected_runs_per_method": 5040,
        "artifacts": {
            label: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)}
            for label, path in artifact_paths.items()
        },
        "execution_plan": execution_plan,
    }
    args.output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(document, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
