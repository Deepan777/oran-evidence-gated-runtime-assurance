#!/usr/bin/env python3
"""Validate frozen controller-state compatibility before live evaluation.

This gate is intentionally strict.  A controller checkpoint trained on a
different feature distribution must not be used for a confirmatory live run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"decision record {number} is not an object")
        if item.get("status") == "DECISION":
            rows.append(item)
    return rows


def validate_live_state_schema(decisions: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
    names = tuple(schema["feature_names"])
    lower = np.asarray(schema["raw_p01"], dtype=float)
    upper = np.asarray(schema["raw_p99"], dtype=float)
    if lower.shape != (len(names),) or upper.shape != (len(names),):
        raise ValueError("state-schema raw bounds do not match feature order")
    if not decisions:
        return {"status": "BLOCKED_NO_COMPLETE_LIVE_DECISIONS", "decision_windows": 0, "features": {}}
    details: dict[str, Any] = {}
    incompatible = False
    for index, name in enumerate(names):
        values = np.asarray([row["decision"]["raw_state"][name] for row in decisions], dtype=float)
        nonfinite = int((~np.isfinite(values)).sum())
        outside = (values < lower[index]) | (values > upper[index])
        rate = float(outside.mean())
        details[name] = {
            "training_p01": float(lower[index]), "training_p99": float(upper[index]),
            "live_min": float(values.min()), "live_max": float(values.max()),
            "outside_training_bounds_count": int(outside.sum()), "outside_training_bounds_rate": rate,
            "nonfinite_count": nonfinite,
        }
        incompatible = incompatible or bool(outside.any()) or nonfinite > 0
    return {
        "status": "BLOCKED_LIVE_STATE_SCHEMA_INCOMPATIBLE" if incompatible else "LIVE_STATE_SCHEMA_COMPATIBLE",
        "decision_windows": len(decisions), "features": details,
        "required_remediation": None if not incompatible else (
            "collect pre-confirmatory live calibration scenarios, refit preprocessing on their training split, "
            "retrain/validate the controller, freeze new artifacts, then create a new untouched confirmatory manifest"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-decisions", type=Path)
    parser.add_argument("--state-schema", type=Path, default=ROOT / "results/training/controller/state_schema.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/validation/live_state_schema_report.json")
    args = parser.parse_args()
    decisions_path = args.live_decisions
    if decisions_path is None:
        candidates = sorted((ROOT / "results/raw").glob("*/live-xapp-decisions.jsonl"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise SystemExit("BLOCKED: no live controller decision evidence exists; supply --live-decisions after a development run")
        decisions_path = candidates[-1]
    report = validate_live_state_schema(_records(decisions_path), json.loads(args.state_schema.read_text(encoding="utf-8")))
    report["live_decisions_file"] = str(decisions_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "LIVE_STATE_SCHEMA_COMPATIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
