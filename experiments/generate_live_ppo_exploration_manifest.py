#!/usr/bin/env python3
"""Generate bounded, on-policy PPO development exploration runs.

Unlike the forced-action manifest used to populate off-policy Q-learning
replay, these rows sample the frozen offline PPO actor under the live
feasibility mask and record its behaviour log-probability/value.  They are
therefore suitable for a clipped on-policy PPO update, but never for
confirmatory evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JAMMERS = ("none", "barrage", "partial_band", "sweep", "intermittent", "reactive")
PARTITIONS = ("train", "train", "train", "validation")


def make_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family_index, jammer in enumerate(JAMMERS):
        for repeat, partition in enumerate(PARTITIONS):
            jsr = "" if jammer == "none" else (0, 5, 0, 5)[repeat]
            seed = 2401 + family_index * len(PARTITIONS) + repeat
            scenario = "-".join((jammer, "jsrNA" if jsr == "" else f"jsr{jsr:+d}", "low", "protected_only", "nominal", "ppo")).replace("+", "p").replace("-", "m")
            rows.append({
                "run_id": f"liveppo-{scenario}-s{seed}",
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
                "evidence_class": "development_live_ppo_exploration",
                "generalization_holdout": False,
                "partition": partition,
                "ENABLE_LIVE_XAPP": "1",
                "LIVE_XAPP_XAI": "0",
                "DETECTOR_MODEL": "results/results/training/live_detector/detector_live_calibrated.joblib",
                "CONTROLLER_MODEL": "results/results/training/controller/masked_ppo_offline.pt",
                "CONTROLLER_KIND": "masked_ppo",
                "STATE_SCHEMA": "results/results/training/live_controller/state_schema.json",
                "LIVE_XAPP_SELECTION_MODE": "ppo_sampling",
                "UE_TIMEOUT_MS": "5000",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/metadata/manifests/live_ppo_exploration_runs.csv")
    args = parser.parse_args()
    rows = make_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = {
        "status": "DEVELOPMENT_LIVE_PPO_EXPLORATION_MANIFEST_CREATED",
        "not_confirmatory": True,
        "rows": len(rows),
        "partitions": {name: sum(row["partition"] == name for row in rows) for name in set(PARTITIONS)},
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "file": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
