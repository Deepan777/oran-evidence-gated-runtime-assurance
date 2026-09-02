#!/usr/bin/env python3
"""Version the frozen live validation with method-unique execution IDs.

V2 correctly freezes the scenario set and recalibrated P/B5 artifacts, but
its four method manifests reuse each scenario's run_id.  Raw evidence is
stored by run_id, so that would overwrite one method with another.  V3 keeps
every scientific factor unchanged and changes only the execution identity.
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
        source_run_id = row["run_id"]
        row["source_run_id"] = source_run_id
        row["run_id"] = f"livev3-{method.lower()}-{source_run_id}"
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "datasets/metadata/manifests/live_validation_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/metadata/manifests/live_validation_v3")
    parser.add_argument("--parent-freeze", type=Path, default=ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V2.yaml")
    parser.add_argument("--output-freeze", type=Path, default=ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V3.yaml")
    args = parser.parse_args()

    document: dict[str, Any] = yaml.safe_load(args.parent_freeze.read_text(encoding="utf-8"))
    source_ids: set[str] | None = None
    execution_plan: list[dict[str, Any]] = []
    artifacts = dict(document["artifacts"])
    all_execution_ids: set[str] = set()
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
        artifacts[f"{method.lower()}_v3_manifest"] = {"path": relative, "sha256": digest(output)}
        execution_plan.append({
            "method": method,
            "scenario_manifest": relative,
            "result_manifest": f"results/raw/LIVE_VALIDATION_V3_{method}.json",
            "expected_runs": 90,
            "command": [
                "envs/oran-action-xai/bin/python",
                "experiments/collection/run_manifest.py",
                "--manifest", relative,
                "--output", f"results/raw/LIVE_VALIDATION_V3_{method}.json",
                "--max-wall-seconds", "900",
                "--gnb-image", "ocudu-gnb-zmq:research-actions-fa4c9fa445a3",
            ],
        })

    document.update({
        "schema_version": 3,
        "status": "LIVE_VALIDATION_FROZEN_V3",
        "parent_freeze_manifest": args.parent_freeze.name,
        "artifacts": artifacts,
        "execution_plan": execution_plan,
        "notes": (
            "Execution-identity correction only: each method now has a unique run_id; "
            "source_run_id preserves the unchanged V2 scenario identity and all factors."
        ),
    })
    args.output_freeze.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(document, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
