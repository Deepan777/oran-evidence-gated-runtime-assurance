#!/usr/bin/env python3
"""Create an immutable, balanced pre-confirmatory live-calibration manifest.

This deliberately does not reuse rows from the locked confirmatory matrix.
It produces five independently seeded 30-second development runs for each
jammer family, with a fixed run-level train/validation/test assignment.  Those
rows are only for calibrating live telemetry and must never be reported as
confirmatory experiment outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JAMMERS = ("none", "barrage", "partial_band", "sweep", "intermittent", "reactive")
SEEDS = (1101, 1102, 1103, 1104, 1105)
PARTITIONS = ("train", "train", "train", "validation", "locked_test")


def make_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for jammer_index, jammer_type in enumerate(JAMMERS):
        for repeat, (seed, partition) in enumerate(zip(SEEDS, PARTITIONS, strict=True)):
            traffic_load = ("low", "medium", "high", "low", "high")[repeat]
            service_mix = ("protected_only", "protected_plus_background")[repeat % 2]
            attenuation = ("nominal", "degraded")[((jammer_index + repeat) // 2) % 2]
            jsr_db = None if jammer_type == "none" else (-5, 0, 5, 10, 5)[repeat]
            scenario = "-".join((
                jammer_type, "jsrNA" if jsr_db is None else f"jsr{jsr_db:+d}",
                traffic_load, service_mix, attenuation,
            )).replace("+", "p").replace("-", "m")
            rows.append({
                "run_id": f"livecal-{scenario}-s{seed}",
                "scenario_id": scenario,
                "jammer_type": jammer_type,
                "jsr_db": "" if jsr_db is None else jsr_db,
                "traffic_load": traffic_load,
                "service_mix": service_mix,
                "attenuation": attenuation,
                "seed": seed,
                "run_length_seconds": 30,
                "pre_jam_seconds": 5,
                "jammer_on_seconds": 20,
                "recovery_seconds": 5,
                "evidence_class": "development_live_calibration",
                "generalization_holdout": False,
                "partition": partition,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/metadata/manifests/live_calibration_runs.csv")
    args = parser.parse_args()
    rows = make_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary = {
        "status": "DEVELOPMENT_LIVE_CALIBRATION_MANIFEST_CREATED",
        "not_confirmatory": True,
        "rows": len(rows),
        "families": list(JAMMERS),
        "partitions": {name: sum(row["partition"] == name for row in rows) for name in set(PARTITIONS)},
        "sha256": digest,
        "file": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
