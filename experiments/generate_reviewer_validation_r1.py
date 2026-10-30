#!/usr/bin/env python3
"""Freeze the bounded actuator characterization actuator characterization manifest."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets/metadata/manifests/reviewer_validation_r1"
MANIFEST = OUT_DIR / "REVIEWER_VALIDATION_R1.csv"
FREEZE = OUT_DIR / "REVIEWER_VALIDATION_R1.freeze.json"
PLAN = ROOT / "reproducibility/REVIEWER_VALIDATION_R1_ANALYSIS_PLAN.yaml"
CORRECTION = ROOT / "reproducibility/REVIEWER_VALIDATION_R1_SMOKE_CORRECTION.json"
SUPERSEDED_MANIFEST = OUT_DIR / "REVIEWER_VALIDATION_R1_SUPERSEDED_SMOKE.csv"
SUPERSEDED_FREEZE = OUT_DIR / "REVIEWER_VALIDATION_R1_SUPERSEDED_SMOKE.freeze.json"

UE_IMAGE = "srsue-zmq:git-6bcbd9e5bf86-telemetry-v2-iperf3-buffer16384"
STATE_SCHEMA = "/home/OPERATOR_USER/oran-cloud-project/results/training/live_controller_recalibrated_v2/state_schema.json"
DETECTOR = "/home/OPERATOR_USER/oran-cloud-project/results/training/live_detector/detector_live_calibrated.joblib"
CONTROLLER = "/home/OPERATOR_USER/oran-cloud-project/results/training/live_controller_recalibrated/md3qn_per_live_candidate.pt"

FIELDS = (
    "run_id", "block_id", "scenario_id", "jammer_type", "jammer_bandwidth_class",
    "jsr_db", "traffic_load", "service_mix", "attenuation", "seed",
    "run_length_seconds", "pre_jam_seconds", "jammer_on_seconds", "recovery_seconds",
    "evidence_class", "generalization_holdout", "partition", "method", "action_name",
    "EVIDENCE_CLASS", "ENABLE_LIVE_XAPP", "LIVE_XAPP_XAI", "STATE_SCHEMA",
    "LIVE_XAPP_SELECTION_MODE", "LIVE_XAPP_FORCED_ACTION_ID", "UE_TIMEOUT_MS",
    "CONTROLLER_KIND", "DETECTOR_MODEL", "CONTROLLER_MODEL", "source_run_id", "UE_IMAGE",
    "ACCEPTANCE_PROFILE", "OOD_GUARD_PROFILE", "REQUIRE_PROTECTED_TRAFFIC",
)

ACTIONS = {
    "A0": ("0", "NO_OP"),
    "A1": ("1", "PRB_AVOID_SHIFT"),
    "A2": ("2", "MCS_BACKOFF"),
    "A3": ("3", "SLICE_PRB_BOOST"),
    "A4": ("4", "COMPOSITE_LOCAL_DEFENSE"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    block_number = 0
    for jammer, jsr, load, attenuation in itertools.product(
        ("partial_band", "barrage"), (5, 0), ("low", "medium", "high"),
        ("nominal", "degraded"),
    ):
        block_number += 1
        seed = 9100 + block_number
        block_id = f"rv-r1-b{block_number:02d}"
        eligible = ["A0", "A2", "A3"]
        if jammer == "partial_band" and jsr == 5:
            eligible.extend(("A1", "A4"))
        order = list(eligible)
        random.Random(20261021 + block_number).shuffle(order)
        for method in order:
            action_id, action_name = ACTIONS[method]
            scenario = f"{jammer}-jsr{jsr}-{load}-protected_plus_background-{attenuation}-s{seed}"
            run_id = f"rv-r1r2-b{block_number:02d}-{method.lower()}-{jammer}-j{jsr}-{load}-{attenuation}-s{seed}"
            rows.append({
                "run_id": run_id,
                "block_id": block_id,
                "scenario_id": scenario,
                "jammer_type": jammer,
                "jammer_bandwidth_class": "localized_25pct" if jammer == "partial_band" else "full_band",
                "jsr_db": jsr,
                "traffic_load": load,
                "service_mix": "protected_plus_background",
                "attenuation": attenuation,
                "seed": seed,
                "run_length_seconds": 30,
                "pre_jam_seconds": 10,
                "jammer_on_seconds": 15,
                "recovery_seconds": 5,
                "evidence_class": "exploratory_reviewer_response_new_data",
                "generalization_holdout": "True",
                "partition": "reviewer_response_r1",
                "method": method,
                "action_name": action_name,
                "EVIDENCE_CLASS": "exploratory_reviewer_response_new_data",
                "ENABLE_LIVE_XAPP": 1,
                "LIVE_XAPP_XAI": 0,
                "STATE_SCHEMA": STATE_SCHEMA,
                "LIVE_XAPP_SELECTION_MODE": "bounded_action_probe",
                "LIVE_XAPP_FORCED_ACTION_ID": action_id,
                "UE_TIMEOUT_MS": 5000,
                "CONTROLLER_KIND": "md3qn_per",
                "DETECTOR_MODEL": DETECTOR,
                "CONTROLLER_MODEL": CONTROLLER,
                "source_run_id": run_id,
                "UE_IMAGE": UE_IMAGE,
                "ACCEPTANCE_PROFILE": "observed_outage_censoring_v1",
                "OOD_GUARD_PROFILE": "physical_plausibility_with_shift_audit",
                "REQUIRE_PROTECTED_TRAFFIC": 1,
            })
    if block_number != 24 or len(rows) != 84 or len({str(row["run_id"]) for row in rows}) != 84:
        raise AssertionError("actuator characterization must contain 24 blocks and 84 unique rows")
    return rows


def main() -> int:
    if not PLAN.is_file():
        raise SystemExit(f"analysis plan missing: {PLAN}")
    rows = generate()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    sources = [
        MANIFEST,
        PLAN,
        CORRECTION,
        SUPERSEDED_MANIFEST,
        SUPERSEDED_FREEZE,
        Path(__file__).resolve(),
        ROOT / "experiments/collection/run_manifest.py",
        ROOT / "tools/run_live_pilot_scenario.sh",
        ROOT / "analysis/reviewer_validation_r1_postprocess.py",
        ROOT / "tools/run_command_impairment_study.py",
        ROOT / "tools/run_reviewer_validation_r1_resilient.py",
        ROOT / "tools/finalize_reviewer_validation_r1.py",
        ROOT / "tools/run_reviewer_validation_r1.sh",
        ROOT / "tools/install_reviewer_validation_r1_service.sh",
        ROOT / "configs/experiments/protocol.yaml",
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise SystemExit(f"freeze sources missing: {missing}")
    freeze = {
        "schema_version": 1,
        "created_utc": "2026-10-20T08:20:00Z",
        "execution_revision": "ELIGIBILITY_CORRECTION_R2",
        "evidence_role": "EXPLORATORY_REVIEWER_RESPONSE_NEW_DATA",
        "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "rows": len(rows),
        "matched_blocks": 24,
        "windows_per_row": 30,
        "expected_windows": 2520,
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
            for path in sources
        },
        "prohibited_use": "not eligible to overturn V10 or qualification futility decisions",
    }
    with FREEZE.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(MANIFEST), "rows": len(rows), "freeze": str(FREEZE)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

