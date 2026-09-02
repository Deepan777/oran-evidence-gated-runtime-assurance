#!/usr/bin/env python3
"""Execute only a completely frozen confirmatory experiment plan.

This runner deliberately has no defaults for methods, models, commands, or
scenario counts.  A generated ``CONFIRMATORY_FREEZE_MANIFEST.yaml`` must name
every immutable artifact and every machine-executable method plan.  It then
records the exact plan digest and accepted record paths for aggregation.
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
METHODS = ("B0", "B1", "B2", "B3", "B4", "B5", "P")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_and_validate_freeze(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing confirmatory freeze manifest: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("status") != "CONFIRMATORY_FROZEN":
        raise ValueError("freeze manifest must have status CONFIRMATORY_FROZEN")
    artifacts = document.get("artifacts")
    plan = document.get("execution_plan")
    if not isinstance(artifacts, dict) or not isinstance(plan, list):
        raise ValueError("freeze manifest needs artifacts and execution_plan")
    for label, entry in artifacts.items():
        if not isinstance(label, str) or not isinstance(entry, dict):
            raise ValueError("freeze artifact entries must be mappings")
        file_name, expected = entry.get("path"), entry.get("sha256")
        if not isinstance(file_name, str) or not isinstance(expected, str):
            raise ValueError(f"freeze artifact {label} lacks path/sha256")
        actual_path = _resolve(file_name)
        if not actual_path.is_file() or _digest(actual_path) != expected:
            raise ValueError(f"freeze artifact hash mismatch: {label}")
    names = [entry.get("method") for entry in plan if isinstance(entry, dict)]
    if tuple(sorted(names)) != METHODS:
        raise ValueError(f"freeze plan must contain each method exactly once: {METHODS}")
    for entry in plan:
        if not isinstance(entry, dict):
            raise ValueError("invalid execution plan entry")
        command, manifest, output, expected = (
            entry.get("command"), entry.get("scenario_manifest"),
            entry.get("result_manifest"), entry.get("expected_runs"),
        )
        if (not isinstance(command, list) or not command or not all(isinstance(token, str) for token in command)
                or not isinstance(manifest, str) or not isinstance(output, str)
                or not isinstance(expected, int) or expected <= 0):
            raise ValueError(f"invalid frozen execution plan for {entry.get('method')}")
        scenario_path = _resolve(manifest)
        if not scenario_path.is_file():
            raise ValueError(f"missing frozen scenario manifest for {entry['method']}")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=ROOT / "CONFIRMATORY_FREEZE_MANIFEST.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/raw/CONFIRMATORY_COMPLETE.json")
    args = parser.parse_args()
    freeze = load_and_validate_freeze(args.freeze)
    accepted: dict[str, list[str]] = {}
    for entry in freeze["execution_plan"]:
        subprocess.run(entry["command"], cwd=ROOT, check=True)
        result_path = _resolve(entry["result_manifest"])
        if not result_path.is_file():
            raise SystemExit(f"confirmatory method did not produce manifest: {entry['method']}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        files = result.get("record_files")
        if not isinstance(files, list) or len(files) != entry["expected_runs"]:
            raise SystemExit(f"confirmatory record count mismatch for {entry['method']}")
        resolved = [_resolve(str(item)) for item in files]
        if any(not item.is_file() for item in resolved):
            raise SystemExit(f"confirmatory record file missing for {entry['method']}")
        accepted[str(entry["method"])] = [str(item) for item in resolved]
    payload = {
        "schema_version": 1,
        "status": "CONFIRMATORY_COMPLETE",
        "freeze_manifest": str(args.freeze),
        "freeze_sha256": _digest(args.freeze),
        "methods": accepted,
        "record_files": [item for values in accepted.values() for item in values],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
