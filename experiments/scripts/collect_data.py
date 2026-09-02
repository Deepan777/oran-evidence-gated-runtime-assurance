#!/usr/bin/env python3
"""Run immutable development-pilot rows, resumably, on the live testbed.

This command never relabels a pilot as confirmatory evidence.  It delegates to
the radio-clock collector, which accepts a row only after its exact sample
duration and all one-second observed-telemetry windows are complete.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 runs every remaining pilot row")
    parser.add_argument("--max-wall-seconds", type=int, default=3600)
    parser.add_argument("--gnb-image", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/metadata/manifests/pilot_runs.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/raw/PILOT_MANIFEST.json")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be non-negative")
    subprocess.run([sys.executable, "experiments/generate_protocol_manifests.py"], cwd=ROOT, check=True)
    command = [
        sys.executable, "experiments/collection/run_manifest.py", "--manifest", str(args.manifest),
        "--output", str(args.output), "--max-wall-seconds", str(args.max_wall_seconds),
        "--gnb-image", args.gnb_image,
    ]
    if args.limit:
        command.extend(("--limit", str(args.limit)))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
