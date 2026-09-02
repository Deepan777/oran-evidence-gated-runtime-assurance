#!/usr/bin/env python3
"""Generate the immutable bounded live-controller exploration manifest.

The rows are development evidence only.  Each row uses exactly one predeclared
feasible action request through the native OCUDU research interface; policy
learning is performed only after its raw acknowledgements and outcomes have
been validated.  These runs are deliberately separate from detector
calibration and from the locked confirmatory matrix.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JAMMERS = ("none", "barrage", "partial_band", "sweep", "intermittent", "reactive")
PARTITIONS = ("train", "train", "train", "validation", "locked_test")
# Every requested action is feasible for the selected waveform family under
# the locked safety mask.  Localized actions (A1/A4) are assigned only to
# partial-band or sweep interference; a forced request never bypasses that
# mask when runtime conditions nevertheless make it ineligible.
ACTION_SCHEDULE = {
    "none": (0, 2, 3, 0, 2),
    "barrage": (2, 3, 0, 2, 3),
    "partial_band": (1, 4, 0, 2, 3),
    "sweep": (4, 1, 2, 0, 3),
    "intermittent": (2, 3, 0, 2, 3),
    "reactive": (2, 3, 0, 2, 3),
}
# A learned PRB-only comparator may learn only from acknowledged A0/A1
# transitions.  The core balanced design provides two localized A1 training
# runs (partial-band and sweep), which is not enough for the DQN's 128-sample
# update.  These three additional, bounded development rows provide the
# required *real* A1 coverage without fabricating counterfactual action labels
# or contaminating validation/locked-test partitions.
PRB_ONLY_SUPPLEMENTAL_RUNS = (
    ("partial_band", 0, 2121),
    ("sweep", 5, 2122),
    ("partial_band", 5, 2123),
)


def make_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for jammer in JAMMERS:
        for repeat, partition in enumerate(PARTITIONS):
            action = ACTION_SCHEDULE[jammer][repeat]
            # Bounded live exploration must remain inside the independently
            # calibrated live-state envelope. Generalization over new loads,
            # service mixes, attenuation and JSR levels is deferred to a new
            # validation/freeze protocol; otherwise the xApp's mandatory OOD
            # gate would correctly refuse to control the uncalibrated state.
            jsr = "" if jammer == "none" else (0, 5, 0, 5, 0)[repeat]
            load = "low"
            mix = "protected_only"
            attenuation = "nominal"
            scenario = "-".join((jammer, "jsrNA" if jsr == "" else f"jsr{jsr:+d}", load, mix, attenuation, f"a{action}")).replace("+", "p").replace("-", "m")
            rows.append({
                "run_id": f"liveexp-{scenario}-s{2101 + repeat}",
                "scenario_id": scenario,
                "jammer_type": jammer,
                "jsr_db": jsr,
                "traffic_load": load,
                "service_mix": mix,
                "attenuation": attenuation,
                "seed": 2101 + repeat,
                "run_length_seconds": 30,
                "pre_jam_seconds": 5,
                "jammer_on_seconds": 20,
                "recovery_seconds": 5,
                "evidence_class": "development_live_exploration",
                "generalization_holdout": False,
                "partition": partition,
                "ENABLE_LIVE_XAPP": "1",
                "LIVE_XAPP_XAI": "0",
                "DETECTOR_MODEL": "results/results/training/live_detector/detector_live_calibrated.joblib",
                "CONTROLLER_MODEL": "results/results/training/live_controller/md3qn_per_live_initialization.pt",
                "STATE_SCHEMA": "results/results/training/live_controller/state_schema.json",
                "LIVE_XAPP_SELECTION_MODE": "forced_exploration",
                "LIVE_XAPP_FORCED_ACTION_ID": action,
                # The calibrated live-radio runs use the stable 5 s UE
                # registration/telemetry timeout.  Keep this runtime
                # setting explicit in the immutable development manifest so
                # retries cannot silently fall back to the 1 s default.
                "UE_TIMEOUT_MS": "5000",
            })
    for jammer, jsr, seed in PRB_ONLY_SUPPLEMENTAL_RUNS:
        action = 1
        scenario = "-".join((jammer, f"jsr{jsr:+d}", "low", "protected_only", "nominal", "a1", "prbonly")).replace("+", "p").replace("-", "m")
        rows.append({
            "run_id": f"liveexp-{scenario}-s{seed}",
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
            "evidence_class": "development_live_exploration",
            "generalization_holdout": False,
            "partition": "train",
            "baseline_training_role": "B5_PRB_ONLY_A1_COVERAGE",
            "ENABLE_LIVE_XAPP": "1",
            "LIVE_XAPP_XAI": "0",
            "DETECTOR_MODEL": "results/results/training/live_detector/detector_live_calibrated.joblib",
            "CONTROLLER_MODEL": "results/results/training/live_controller/md3qn_per_live_initialization.pt",
            "STATE_SCHEMA": "results/results/training/live_controller/state_schema.json",
            "LIVE_XAPP_SELECTION_MODE": "forced_exploration",
            "LIVE_XAPP_FORCED_ACTION_ID": action,
            "UE_TIMEOUT_MS": "5000",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/metadata/manifests/live_exploration_runs.csv")
    args = parser.parse_args()
    rows = make_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        # Supplemental development rows carry an explicit provenance field;
        # retain it in the immutable artifact rather than dropping it or
        # relying on whichever row happens to be first.
        fieldnames = list(rows[0]) + sorted({key for row in rows for key in row} - set(rows[0]))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    payload = {
        "status": "DEVELOPMENT_LIVE_EXPLORATION_MANIFEST_CREATED",
        "not_confirmatory": True,
        "rows": len(rows),
        "action_ids": list(range(5)),
        "partitions": {name: sum(row["partition"] == name for row in rows) for name in set(PARTITIONS)},
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "file": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
