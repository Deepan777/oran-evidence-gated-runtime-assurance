#!/usr/bin/env python3
"""Generate recovery calibration v2 after a documented v1 transport failure.

The immutable v1 manifest remains intact.  v2 uses separately qualified
low-load/protected-only/nominal conditions and two live-qualified strengths
(0 and +5 dB) to calibrate detector and live-state preprocessing without
admitting an incomplete transport-failure run as evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JAMMERS = ("none", "barrage", "partial_band", "sweep", "intermittent", "reactive")
SEEDS = (1301, 1302, 1303, 1304, 1305)
PARTITIONS = ("train", "train", "train", "validation", "locked_test")


def make_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for jammer in JAMMERS:
        for repeat, (seed, partition) in enumerate(zip(SEEDS, PARTITIONS, strict=True)):
            jsr = "" if jammer == "none" else (0, 5, 0, 5, 0)[repeat]
            scenario = "-".join((jammer, "jsrNA" if jsr == "" else f"jsr{jsr:+d}", "low", "protected_only", "nominal")).replace("+", "p").replace("-", "m")
            rows.append({
                "run_id": f"livecalv2-{scenario}-s{seed}",
                "scenario_id": scenario,
                "jammer_type": jammer,
                "jsr_db": jsr,
                "traffic_load": "low",
                "service_mix": "protected_only",
                "attenuation": "nominal",
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
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/metadata/manifests/live_calibration_recovery_v2_runs.csv")
    args = parser.parse_args()
    rows = make_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "status": "DEVELOPMENT_LIVE_CALIBRATION_RECOVERY_V2_MANIFEST_CREATED",
        "not_confirmatory": True,
        "replaces_for_training_only": "live_calibration_runs.csv after documented incomplete v1 row",
        "rows": len(rows), "jammer_families": list(JAMMERS), "non_none_jsr_db": [0, 5],
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "file": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
