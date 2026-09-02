#!/usr/bin/env python3
"""Create the cost-controlled live-validation freeze without changing the full freeze.

The original 35,280-row confirmatory manifest remains authoritative for the
offline/locked study. This artifact selects 90 locked-test scenario rows and
replays those rows with four predeclared live arms (B0, B2, B5, P), for 360
actual method executions total.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("B0", "B2", "B5", "P")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_rows(rows: Iterable[dict[str, str]], count: int) -> list[dict[str, str]]:
    locked = [row for row in rows if row.get("partition") == "locked_test"]
    if len(locked) < count:
        raise SystemExit(f"need {count} locked-test rows, found {len(locked)}")
    # Round-robin over jammer/JSR cells, then traffic/service/attenuation, so
    # every jammer family and requested JSR is represented in the small live set.
    keys = ("jammer_type", "jsr_db", "traffic_load", "service_mix", "attenuation", "seed")
    ordered = sorted(locked, key=lambda row: tuple(row.get(key, "") for key in keys))
    cells: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in ordered:
        cells.setdefault((row.get("jammer_type", ""), row.get("jsr_db", "")), []).append(row)
    selected: list[dict[str, str]] = []
    while len(selected) < count:
        progressed = False
        for cell in sorted(cells):
            if cells[cell]:
                selected.append(cells[cell].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) != count:
        raise SystemExit(f"selection produced {len(selected)} rows, expected {count}")
    return selected


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "datasets/metadata/manifests/confirmatory_runs.csv")
    parser.add_argument("--method-dir", type=Path, default=ROOT / "datasets/metadata/manifests/confirmatory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/metadata/manifests/live_validation")
    parser.add_argument("--freeze", type=Path, default=ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST.yaml")
    args = parser.parse_args()

    base_rows = load_rows(args.base)
    selected = select_rows(base_rows, 90)
    selected_ids = {row["run_id"] for row in selected}
    if len(selected_ids) != 90:
        raise SystemExit("selected run IDs are not unique")

    artifacts = {"base_manifest": {"path": str(args.base.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(args.base)}}
    execution_plan = []
    for method in METHODS:
        source = args.method_dir / f"{method}_confirmatory_runs.csv"
        source_rows = load_rows(source)
        output_rows = [row for row in source_rows if row.get("run_id") in selected_ids]
        output_rows.sort(key=lambda row: row["run_id"])
        if len(output_rows) != 90:
            raise SystemExit(f"{method}: expected 90 selected rows, found {len(output_rows)}")
        out = args.output_dir / f"{method}_live_validation_runs.csv"
        write_csv(out, output_rows)
        artifacts[method.lower() + "_source"] = {"path": str(source.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(source)}
        execution_plan.append({
            "method": method,
            "scenario_manifest": str(out.relative_to(ROOT)).replace("\\", "/"),
            "result_manifest": f"results/raw/LIVE_VALIDATION_{method}.json",
            "expected_runs": 90,
            "command": [
                "envs/oran-action-xai/bin/python",
                "experiments/collection/run_manifest.py",
                "--manifest", str(out.relative_to(ROOT)).replace("\\", "/"),
                "--output", f"results/raw/LIVE_VALIDATION_{method}.json",
                "--max-wall-seconds", "900",
                "--gnb-image", "ocudu-gnb-zmq:research-actions-fa4c9fa445a3",
            ],
        })

    selected_path = args.output_dir / "selected_run_ids.json"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(json.dumps(sorted(selected_ids), indent=2) + "\n", encoding="utf-8")
    artifacts["selected_run_ids"] = {"path": str(selected_path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(selected_path)}
    document = {
        "schema_version": 1,
        "status": "LIVE_VALIDATION_FROZEN",
        "scope": "balanced_live_o_ran_validation",
        "total_method_executions": 360,
        "scenario_count": 90,
        "methods": list(METHODS),
        "runs_per_method": 90,
        "run_length_seconds": 120,
        "selection_partition": "locked_test",
        "selection_rule": "deterministic round-robin over jammer_type/jsr_db cells",
        "full_confirmatory_manifest_unchanged": True,
        "full_confirmatory_manifest": "CONFIRMATORY_FREEZE_MANIFEST.yaml",
        "artifacts": artifacts,
        "execution_plan": execution_plan,
    }
    args.freeze.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(document, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
