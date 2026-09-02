#!/usr/bin/env python3
"""Create a versioned live schema with correct categorical action bounds.

The original live schema was fitted from calibration runs that started at A0,
so all previous-action p99 values were zero. This recalibration preserves every
continuous feature statistic and changes only the declared binary one-hot
action-transition fields. It is development-only and never edits a frozen
manifest or overwrites the original schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib

from src.environment.iq_calibrated_env import FEATURE_NAMES


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if tuple(source.get("feature_names", ())) != FEATURE_NAMES:
        raise SystemExit("source schema feature order differs from methodology lock")
    output = dict(source)
    means = list(output["means"]); scales = list(output["scales"])
    raw_low = list(output["raw_p01"]); raw_high = list(output["raw_p99"])
    for action in range(5):
        index = FEATURE_NAMES.index(f"previous_action_a{action}")
        means[index] = 0.2
        scales[index] = 0.4
        raw_low[index] = 0.0
        raw_high[index] = 1.0
    output.update({
        "status": "DEVELOPMENT_LIVE_SCHEMA_RECALIBRATED_ACTION_TRANSITIONS",
        "not_confirmatory": True,
        "source_schema_sha256": digest(args.source),
        "categorical_action_transition_recalibration": {
            "features": [f"previous_action_a{i}" for i in range(5)],
            "domain": [0.0, 1.0],
            "mean": 0.2,
            "scale": 0.4,
        },
        "means": means, "scales": scales, "raw_p01": raw_low, "raw_p99": raw_high,
    })
    args.output.mkdir(parents=True, exist_ok=True)
    schema_path = args.output / "state_schema.json"
    schema_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    source_scaler = args.source.parent / "scaler.pkl"
    if source_scaler.is_file():
        scaler = joblib.load(source_scaler)
        scaler.update({"means": means, "scales": scales, "feature_names": list(FEATURE_NAMES)})
        joblib.dump(scaler, args.output / "scaler.pkl")
    print(json.dumps({"status": output["status"], "schema": str(schema_path),
                      "source_schema_sha256": output["source_schema_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
