#!/usr/bin/env python3
"""Version V4 after its B2 gate exposed an inconsistent state-schema path.

V4's B5 and P rows use the development-calibrated V2 state schema already
hashed in the parent freeze, while B2 still references the older schema.  V5
uses that same pre-confirmatory schema for every controller method.  It does
not change scenarios, models, seeds, traffic, jammer factors, duration, radio
profile, or evidence acceptance rules, and gives all executions fresh IDs.
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
COMMON_SCHEMA = (
    "results/"
    "results/training/live_controller_recalibrated_v2/state_schema.json"
)
COMMON_SCHEMA_RELATIVE = "results/training/live_controller_recalibrated_v2/state_schema.json"
COMMON_SCHEMA_SHA256 = "127719f66f1b3bc3622473089c8b94206dfe65fdd9cd212a7b51a2d773946fc5"


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
    output: list[dict[str, str]] = []
    for original in source:
        row = dict(original)
        source_run_id = row.get("source_run_id") or row["run_id"]
        row["source_run_id"] = source_run_id
        row["run_id"] = f"livev5-{method.lower()}-{source_run_id}"
        if int(row.get("ENABLE_LIVE_XAPP") or 0) == 1:
            row["STATE_SCHEMA"] = COMMON_SCHEMA
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "datasets/metadata/manifests/live_validation_v4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/metadata/manifests/live_validation_v5")
    parser.add_argument("--parent-freeze", type=Path, default=ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V4.yaml")
    parser.add_argument("--output-freeze", type=Path, default=ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V5.yaml")
    args = parser.parse_args()

    document: dict[str, Any] = yaml.safe_load(args.parent_freeze.read_text(encoding="utf-8"))
    frozen_schema = document["artifacts"]["p_recalibrated_schema"]
    if frozen_schema != {"path": COMMON_SCHEMA_RELATIVE, "sha256": COMMON_SCHEMA_SHA256}:
        raise SystemExit("parent freeze does not contain the expected development-calibrated schema")

    source_ids: set[str] | None = None
    all_execution_ids: set[str] = set()
    execution_plan: list[dict[str, Any]] = []
    artifacts = dict(document["artifacts"])
    artifacts["v5_common_controller_schema"] = dict(frozen_schema)
    for method in METHODS:
        source = args.input_dir / f"{method}_live_validation_runs.csv"
        rows = make_rows(read_rows(source), method)
        ids = {row["source_run_id"] for row in rows}
        if source_ids is None:
            source_ids = ids
        elif ids != source_ids:
            raise SystemExit(f"{method}: scenario identity set differs from the other methods")
        if len(rows) != 90 or len(ids) != 90:
            raise SystemExit(f"{method}: expected 90 unique frozen scenarios")
        execution_ids = {row["run_id"] for row in rows}
        if all_execution_ids.intersection(execution_ids):
            raise SystemExit(f"{method}: duplicate execution IDs across methods")
        all_execution_ids.update(execution_ids)
        output = args.output_dir / f"{method}_live_validation_runs.csv"
        write_rows(output, rows)
        relative = output.relative_to(ROOT).as_posix()
        artifacts[f"{method.lower()}_v5_manifest"] = {"path": relative, "sha256": digest(output)}
        result = f"results/raw/LIVE_VALIDATION_V5_{method}.json"
        execution_plan.append({
            "method": method,
            "scenario_manifest": relative,
            "result_manifest": result,
            "expected_runs": 90,
            "command": [
                "envs/oran-action-xai/bin/python", "experiments/collection/run_manifest.py",
                "--manifest", relative, "--output", result,
                "--max-wall-seconds", "900",
                "--gnb-image", "ocudu-gnb-zmq:research-actions-fa4c9fa445a3",
                "--gnb-config", "ocudu_gnb_zmq_e2_udp_um.yml",
            ],
        })

    document.update({
        "schema_version": 5,
        "status": "LIVE_VALIDATION_FROZEN_V5",
        "parent_freeze_manifest": args.parent_freeze.name,
        "artifacts": artifacts,
        "execution_plan": execution_plan,
        "controller_schema_correction": {
            "scope": "common_controller_preprocessing_and_ood_guard",
            "from": "B2 older live_controller/state_schema.json",
            "to": COMMON_SCHEMA_RELATIVE,
            "basis": "pre-confirmatory development-only calibration already hashed in the parent freeze",
            "models_changed": False,
            "scenario_factors_changed": False,
            "acceptance_rules_changed": False,
            "rejected_v4_evidence_retained": True,
        },
        "notes": (
            "V5 corrects only B2's inconsistent schema reference; all controller methods now "
            "use the same already-frozen development-calibrated state schema."
        ),
    })
    args.output_freeze.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(document, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
