#!/usr/bin/env python3
"""Create v3 retry evidence after preserving a v2 radio-timing failure.

The v2 partial-band attempt is never overwritten or relabelled.  v3 has new
run IDs and a combined immutable training manifest containing only the ten
completed v2 rows plus all twenty new v3 rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.generate_live_calibration_recovery_manifest import make_rows as v2_rows


RETRY_JAMMERS = ("partial_band", "sweep", "intermittent", "reactive")
SEEDS = (1301, 1302, 1303, 1304, 1305)
PARTITIONS = ("train", "train", "train", "validation", "locked_test")


def make_retry_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for jammer in RETRY_JAMMERS:
        for repeat, (seed, partition) in enumerate(zip(SEEDS, PARTITIONS, strict=True)):
            jsr = (0, 5, 0, 5, 0)[repeat]
            scenario = "-".join((jammer, f"jsr{jsr:+d}", "low", "protected_only", "nominal")).replace("+", "p").replace("-", "m")
            row: dict[str, object] = {
                "run_id": f"livecalv3-{scenario}-s{seed}",
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
                "recovery_reason": "v2_partial_band_radio_timing_failure" if jammer == "partial_band" else "v2_not_started_after_ordered_failure",
            }
            if jammer == "partial_band" and seed == 1301:
                row["recovery_of_run_id"] = "livecalv2-partial_bandmjsrp0mlowmprotected_onlymnominal-s1301"
            rows.append(row)
    return rows


def make_combined_rows() -> list[dict[str, object]]:
    completed_v2 = [row for row in v2_rows() if row["jammer_type"] in {"none", "barrage"}]
    return completed_v2 + make_retry_rows()


def _write(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    fields.extend(key for row in rows for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/metadata/manifests/live_calibration_recovery_v3_retry_runs.csv")
    parser.add_argument("--combined-output", type=Path, default=ROOT / "datasets/metadata/manifests/live_calibration_recovery_v3_combined_runs.csv")
    args = parser.parse_args()
    retry, combined = make_retry_rows(), make_combined_rows()
    retry_hash, combined_hash = _write(args.output, retry), _write(args.combined_output, combined)
    summary = {
        "status": "DEVELOPMENT_LIVE_CALIBRATION_RECOVERY_V3_MANIFEST_CREATED",
        "not_confirmatory": True,
        "preserved_failed_v2_run": "livecalv2-partial_bandmjsrp0mlowmprotected_onlymnominal-s1301",
        "retry_rows": len(retry), "combined_training_rows": len(combined),
        "retry_manifest": str(args.output), "retry_sha256": retry_hash,
        "combined_manifest": str(args.combined_output), "combined_sha256": combined_hash,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
