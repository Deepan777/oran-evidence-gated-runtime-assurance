#!/usr/bin/env python3
"""Freeze V8 with outage-aware evidence and a physical plausibility guard.

V8 preserves every V7 scientific scenario, controller artifact, radio profile,
and runtime image. It replaces the invalid empirical-p01/p99 hard stop with a
method-neutral physical-plausibility guard. Training-range crossings remain in
every decision record as distribution-shift warnings. Fresh execution IDs keep
all V3--V7 development/rejected evidence outside the V8 confirmatory set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("B0", "B2", "B5", "P")
UE_IMAGE = "srsue-zmq:git-6bcbd9e5bf86-telemetry-v2-iperf3-buffer16384"
ACCEPTANCE_PROFILE = "observed_outage_censoring_v1"
OOD_GUARD_PROFILE = "physical_plausibility_with_shift_audit"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_rows(source: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for original in source:
        row = dict(original)
        source_run_id = row.get("source_run_id") or row["run_id"]
        row["source_run_id"] = source_run_id
        row["run_id"] = f"livev8-{method.lower()}-{source_run_id}"
        row["UE_IMAGE"] = UE_IMAGE
        row["ACCEPTANCE_PROFILE"] = ACCEPTANCE_PROFILE
        row["OOD_GUARD_PROFILE"] = OOD_GUARD_PROFILE
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "datasets/metadata/manifests/live_validation_v7")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/metadata/manifests/live_validation_v8")
    parser.add_argument("--parent-freeze", type=Path, default=ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V7.yaml")
    parser.add_argument("--output-freeze", type=Path, default=ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V8.yaml")
    args = parser.parse_args()

    document: dict[str, Any] = yaml.safe_load(args.parent_freeze.read_text(encoding="utf-8"))
    source_ids: set[str] | None = None
    execution_ids: set[str] = set()
    execution_plan: list[dict[str, Any]] = []
    artifacts = dict(document["artifacts"])
    artifacts["v8_physical_guard_implementation"] = {
        "paths": [
            "experiments/collection/build_live_run_records.py",
            "experiments/live/run_xapp_loop.py",
            "experiments/collection/run_manifest.py",
        ],
        "sha256": {
            path: digest(ROOT / path)
            for path in (
                "experiments/collection/build_live_run_records.py",
                "experiments/live/run_xapp_loop.py",
                "experiments/collection/run_manifest.py",
            )
        },
    }
    analysis_plan = ROOT / "LIVE_VALIDATION_V8_ANALYSIS_PLAN.yaml"
    if not analysis_plan.is_file():
        raise SystemExit("V8 analysis plan must be frozen before generating the confirmatory freeze")
    artifacts["v8_analysis_plan"] = {
        "path": analysis_plan.relative_to(ROOT).as_posix(),
        "sha256": digest(analysis_plan),
    }
    for method in METHODS:
        rows = make_rows(read_rows(args.input_dir / f"{method}_live_validation_runs.csv"), method)
        identities = {row["source_run_id"] for row in rows}
        if source_ids is None:
            source_ids = identities
        elif identities != source_ids:
            raise SystemExit(f"{method}: scenario identity set differs")
        if len(rows) != 90 or len(identities) != 90:
            raise SystemExit(f"{method}: expected 90 unique frozen scenarios")
        method_ids = {row["run_id"] for row in rows}
        if execution_ids.intersection(method_ids):
            raise SystemExit(f"{method}: duplicate V8 execution IDs")
        execution_ids.update(method_ids)
        output = args.output_dir / f"{method}_live_validation_runs.csv"
        write_rows(output, rows)
        relative = output.relative_to(ROOT).as_posix()
        artifacts[f"{method.lower()}_v8_manifest"] = {"path": relative, "sha256": digest(output)}
        execution_plan.append({
            "method": method,
            "scenario_manifest": relative,
            "result_manifest": f"results/raw/LIVE_VALIDATION_V8_{method}.json",
            "expected_runs": 90,
            "command": [
                "envs/oran-action-xai/bin/python", "experiments/collection/run_manifest.py",
                "--manifest", relative,
                "--output", f"results/raw/LIVE_VALIDATION_V8_{method}.json",
                "--max-wall-seconds", "1800",
                "--gnb-image", "ocudu-gnb-zmq:research-actions-fa4c9fa445a3",
                "--ue-image", UE_IMAGE,
                "--gnb-config", "ocudu_gnb_zmq_e2_udp_um.yml",
            ],
        })

    document.update({
        "schema_version": 8,
        "status": "LIVE_VALIDATION_FROZEN_V8",
        "parent_freeze_manifest": args.parent_freeze.name,
        "artifacts": artifacts,
        "execution_plan": execution_plan,
        "execution_revision": "V8_PHYSICAL_GUARD_OUTAGE_R1",
        "physical_guard_correction": {
            "scope": "method_neutral_controller_input_validation",
            "ood_guard_profile": OOD_GUARD_PROFILE,
            "training_p01_p99_usage": "audit-only distribution-shift warnings",
            "hard_rejection_rule": "non-finite, physically impossible, malformed probability, or malformed previous-action state",
            "development_trigger": "V7 B2 stress smoke; zero V7 confirmatory rows accepted",
            "scenario_factors_changed": False,
            "controllers_changed": False,
            "radio_profile_changed": False,
            "runtime_image_changed": False,
            "acceptance_rules_changed": True,
            "rejected_v3_v4_v5_v6_v7_evidence_retained": True,
        },
        "notes": (
            "V8 keeps the V7 outage evidence contract and versions the physically valid OOD guard. "
            "It never fabricates telemetry, never treats warnings as measurements, and uses fresh execution IDs."
        ),
    })
    args.output_freeze.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(document, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
