#!/usr/bin/env python3
"""Execute a frozen ablation plan without accepting implicit defaults.

The methodology lock defines the required ablations.  This runner refuses a
partial plan, unpinned artifacts, or incomplete record manifests.  It is not a
surrogate simulator: every frozen command must write its own measured records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _required_ids() -> tuple[str, ...]:
    lock = yaml.safe_load((ROOT / "METHODOLOGY_LOCK.yaml").read_text(encoding="utf-8"))
    values = lock.get("ablations") if isinstance(lock, dict) else None
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("methodology lock does not define the required ablations")
    return tuple(values)


def load_and_validate_freeze(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing ablation freeze manifest: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("status") != "ABLATIONS_FROZEN":
        raise ValueError("ablation freeze manifest must have status ABLATIONS_FROZEN")
    artifacts, plan = document.get("artifacts"), document.get("execution_plan")
    if not isinstance(artifacts, dict) or not isinstance(plan, list):
        raise ValueError("ablation freeze manifest needs artifacts and execution_plan")
    for label, entry in artifacts.items():
        if not isinstance(label, str) or not isinstance(entry, dict):
            raise ValueError("ablation artifact entries must be mappings")
        item, expected = entry.get("path"), entry.get("sha256")
        if not isinstance(item, str) or not isinstance(expected, str):
            raise ValueError(f"ablation artifact {label} lacks path/sha256")
        artifact = _resolve(item)
        if not artifact.is_file() or _digest(artifact) != expected:
            raise ValueError(f"ablation artifact hash mismatch: {label}")
    ids = [entry.get("ablation") for entry in plan if isinstance(entry, dict)]
    if tuple(ids) != _required_ids():
        raise ValueError("ablation execution plan must exactly match the methodology-lock order")
    for entry in plan:
        if not isinstance(entry, dict):
            raise ValueError("invalid ablation execution plan entry")
        command, output, expected_runs = entry.get("command"), entry.get("result_manifest"), entry.get("expected_runs")
        if (not isinstance(command, list) or not command or not all(isinstance(token, str) for token in command)
                or not isinstance(output, str) or not isinstance(expected_runs, int) or expected_runs <= 0):
            raise ValueError(f"invalid frozen ablation plan for {entry.get('ablation')}")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=ROOT / "ABLATIONS_FREEZE_MANIFEST.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/raw/ABLATIONS_COMPLETE.json")
    args = parser.parse_args()
    freeze = load_and_validate_freeze(args.freeze)
    accepted: dict[str, list[str]] = {}
    for entry in freeze["execution_plan"]:
        subprocess.run(entry["command"], cwd=ROOT, check=True)
        manifest = _resolve(entry["result_manifest"])
        if not manifest.is_file():
            raise SystemExit(f"ablation did not produce manifest: {entry['ablation']}")
        output = json.loads(manifest.read_text(encoding="utf-8"))
        files = output.get("record_files")
        if not isinstance(files, list) or len(files) != entry["expected_runs"]:
            raise SystemExit(f"ablation evidence count mismatch: {entry['ablation']}")
        resolved = [_resolve(str(value)) for value in files]
        if any(not value.is_file() for value in resolved):
            raise SystemExit(f"ablation record missing: {entry['ablation']}")
        accepted[str(entry["ablation"])] = [str(value) for value in resolved]
    payload = {
        "schema_version": 1,
        "status": "ABLATIONS_COMPLETE",
        "freeze_manifest": str(args.freeze),
        "freeze_sha256": _digest(args.freeze),
        "ablations": accepted,
        "record_files": [value for values in accepted.values() for value in values],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
