#!/usr/bin/env python3
"""Version the frozen live validation after the V3 RLC terminal-outage gate.

V4 preserves every V3 scenario, seed, duration, traffic value, jammer factor,
controller, and model. It changes only the method-neutral live radio bearer
profile and execution identity. The live traffic is UDP and packet loss is a
measured outcome, so RLC UM avoids converting sustained packet loss into an
RLC-AM retry-limit RRC release that terminates the software-radio clock.
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
RADIO_CONFIG = "config/baseline/ocudu_gnb_zmq_e2_udp_um.yml"


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
        row["run_id"] = f"livev4-{method.lower()}-{source_run_id}"
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "datasets/metadata/manifests/live_validation_v3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/metadata/manifests/live_validation_v4")
    parser.add_argument("--parent-freeze", type=Path, default=ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V3.yaml")
    parser.add_argument("--output-freeze", type=Path, default=ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V4.yaml")
    args = parser.parse_args()

    document: dict[str, Any] = yaml.safe_load(args.parent_freeze.read_text(encoding="utf-8"))
    source_ids: set[str] | None = None
    execution_plan: list[dict[str, Any]] = []
    artifacts = dict(document["artifacts"])
    all_execution_ids: set[str] = set()
    radio_path = ROOT / RADIO_CONFIG
    artifacts["v4_radio_config"] = {"path": RADIO_CONFIG, "sha256": digest(radio_path)}

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
        artifacts[f"{method.lower()}_v4_manifest"] = {"path": relative, "sha256": digest(output)}
        result = f"results/raw/LIVE_VALIDATION_V4_{method}.json"
        execution_plan.append({
            "method": method,
            "scenario_manifest": relative,
            "result_manifest": result,
            "expected_runs": 90,
            "command": [
                "envs/oran-action-xai/bin/python",
                "experiments/collection/run_manifest.py",
                "--manifest", relative,
                "--output", result,
                "--max-wall-seconds", "900",
                "--gnb-image", "ocudu-gnb-zmq:research-actions-fa4c9fa445a3",
                "--gnb-config", Path(RADIO_CONFIG).name,
            ],
        })

    document.update({
        "schema_version": 4,
        "status": "LIVE_VALIDATION_FROZEN_V4",
        "parent_freeze_manifest": args.parent_freeze.name,
        "artifacts": artifacts,
        "execution_plan": execution_plan,
        "radio_profile_change": {
            "scope": "method_neutral_live_testbed_transport",
            "from": "5QI-9 RLC AM",
            "to": "5QI-9 RLC UM bidirectional",
            "reason": "preserve real-IQ clock and measure UDP loss without RLC-AM terminal RRC release",
            "scenario_factors_changed": False,
            "rejected_v3_evidence_retained": True,
        },
        "notes": (
            "V4 preserves all V3 scientific scenario factors and frozen model artifacts, "
            "versions the common UDP radio-bearer profile, and assigns fresh execution IDs."
        ),
    })
    args.output_freeze.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(document, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
