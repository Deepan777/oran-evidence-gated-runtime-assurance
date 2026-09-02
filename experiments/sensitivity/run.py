#!/usr/bin/env python3
"""Execute the methodology-locked sensitivity subset from a frozen plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SENSITIVITY_IDS = (
    "reward_weights_plus_minus_20_percent",
    "decision_interval_1_2_5_seconds",
    "held_out_jammer_parameters",
    "counterfactual_objective_lambda_sensitivity",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_and_validate_freeze(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing sensitivity freeze manifest: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("status") != "SENSITIVITY_FROZEN":
        raise ValueError("sensitivity freeze manifest must have status SENSITIVITY_FROZEN")
    artifacts, plan = document.get("artifacts"), document.get("execution_plan")
    if not isinstance(artifacts, dict) or not isinstance(plan, list):
        raise ValueError("sensitivity freeze manifest needs artifacts and execution_plan")
    for label, entry in artifacts.items():
        if not isinstance(label, str) or not isinstance(entry, dict):
            raise ValueError("sensitivity artifact entries must be mappings")
        item, expected = entry.get("path"), entry.get("sha256")
        if not isinstance(item, str) or not isinstance(expected, str):
            raise ValueError(f"sensitivity artifact {label} lacks path/sha256")
        artifact = _resolve(item)
        if not artifact.is_file() or _digest(artifact) != expected:
            raise ValueError(f"sensitivity artifact hash mismatch: {label}")
    ids = [entry.get("sensitivity") for entry in plan if isinstance(entry, dict)]
    if tuple(ids) != SENSITIVITY_IDS:
        raise ValueError("sensitivity execution plan must contain the locked sensitivity studies in order")
    for entry in plan:
        if not isinstance(entry, dict):
            raise ValueError("invalid sensitivity execution plan entry")
        command, output, expected_runs = entry.get("command"), entry.get("result_manifest"), entry.get("expected_runs")
        if (not isinstance(command, list) or not command or not all(isinstance(token, str) for token in command)
                or not isinstance(output, str) or not isinstance(expected_runs, int) or expected_runs <= 0):
            raise ValueError(f"invalid frozen sensitivity plan for {entry.get('sensitivity')}")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=ROOT / "SENSITIVITY_FREEZE_MANIFEST.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/raw/SENSITIVITY_COMPLETE.json")
    args = parser.parse_args()
    freeze = load_and_validate_freeze(args.freeze)
    accepted: dict[str, list[str]] = {}
    for entry in freeze["execution_plan"]:
        subprocess.run(entry["command"], cwd=ROOT, check=True)
        manifest = _resolve(entry["result_manifest"])
        if not manifest.is_file():
            raise SystemExit(f"sensitivity study did not produce manifest: {entry['sensitivity']}")
        output = json.loads(manifest.read_text(encoding="utf-8"))
        files = output.get("record_files")
        if not isinstance(files, list) or len(files) != entry["expected_runs"]:
            raise SystemExit(f"sensitivity evidence count mismatch: {entry['sensitivity']}")
        resolved = [_resolve(str(value)) for value in files]
        if any(not value.is_file() for value in resolved):
            raise SystemExit(f"sensitivity record missing: {entry['sensitivity']}")
        accepted[str(entry["sensitivity"])] = [str(value) for value in resolved]
    payload = {
        "schema_version": 1,
        "status": "SENSITIVITY_COMPLETE",
        "freeze_manifest": str(args.freeze),
        "freeze_sha256": _digest(args.freeze),
        "studies": accepted,
        "record_files": [value for values in accepted.values() for value in values],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
