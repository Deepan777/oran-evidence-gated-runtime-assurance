#!/usr/bin/env python3
"""Write the immutable configuration marker for the heuristic baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/training/live_controller/heuristic_controller.json")
    args = parser.parse_args()
    lock = ROOT / "METHODOLOGY_LOCK.yaml"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "HEURISTIC_BASELINE_FROZEN",
        "controller_kind": "heuristic",
        "feature_source": "state_schema.json",
        "methodology_lock_sha256": digest(lock),
        "implementation": "src/baselines/heuristic.py",
        "rule": "jammer pressure plus localized/resource/quality pressure with locked action costs",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
