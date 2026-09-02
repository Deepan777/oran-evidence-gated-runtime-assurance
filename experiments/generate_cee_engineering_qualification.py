#!/usr/bin/env python3
"""Generate the bounded, development-only CEE engineering qualification."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets/metadata/manifests/cee_qualification/CEE_ENGINEERING_QUALIFICATION.csv"
FREEZE = ROOT / "datasets/metadata/manifests/cee_qualification/CEE_ENGINEERING_QUALIFICATION.freeze.json"
UE_IMAGE = "srsue-zmq:git-6bcbd9e5bf86-telemetry-v2-iperf3-buffer16384"
STATE_SCHEMA = "results/results/training/live_controller_recalibrated_v2/state_schema.json"
DETECTOR = "results/results/training/live_detector/detector_live_calibrated.joblib"
CONTROLLER = "results/results/training/live_controller_recalibrated/md3qn_per_live_candidate.pt"


FIELDS = (
    "run_id", "scenario_id", "jammer_type", "jsr_db", "traffic_load", "service_mix",
    "attenuation", "seed", "run_length_seconds", "pre_jam_seconds", "jammer_on_seconds",
    "recovery_seconds", "evidence_class", "generalization_holdout", "partition", "method",
    "EVIDENCE_CLASS", "ENABLE_LIVE_XAPP", "LIVE_XAPP_XAI", "STATE_SCHEMA",
    "LIVE_XAPP_SELECTION_MODE", "LIVE_XAPP_FORCED_ACTION_ID", "UE_TIMEOUT_MS",
    "CONTROLLER_KIND", "DETECTOR_MODEL", "CONTROLLER_MODEL", "source_run_id", "UE_IMAGE",
    "ACCEPTANCE_PROFILE", "OOD_GUARD_PROFILE", "REQUIRE_PROTECTED_TRAFFIC",
)


def _row(*, run: str, jammer: str, seed: int, method: str, selection: str = "policy",
         action: str = "", enabled: bool = True) -> dict[str, object]:
    return {
        "run_id": run, "scenario_id": f"{jammer}-jsr5-high-protected-nominal-s{seed}",
        "jammer_type": jammer, "jsr_db": 0 if jammer == "none" else 5,
        "traffic_load": "high", "service_mix": "protected_only", "attenuation": "nominal",
        "seed": seed, "run_length_seconds": 30, "pre_jam_seconds": 10,
        "jammer_on_seconds": 15, "recovery_seconds": 5,
        "evidence_class": "development_engineering_qualification",
        "generalization_holdout": "False", "partition": "development_only", "method": method,
        "EVIDENCE_CLASS": "development_engineering_qualification",
        "ENABLE_LIVE_XAPP": int(enabled), "LIVE_XAPP_XAI": 0,
        "STATE_SCHEMA": STATE_SCHEMA if enabled else "",
        "LIVE_XAPP_SELECTION_MODE": selection if enabled else "policy",
        "LIVE_XAPP_FORCED_ACTION_ID": action if enabled else "", "UE_TIMEOUT_MS": 5000,
        "CONTROLLER_KIND": "md3qn_per", "DETECTOR_MODEL": DETECTOR if enabled else "",
        "CONTROLLER_MODEL": CONTROLLER if enabled else "", "source_run_id": run,
        "UE_IMAGE": UE_IMAGE, "ACCEPTANCE_PROFILE": "observed_outage_censoring_v1",
        "OOD_GUARD_PROFILE": "physical_plausibility_with_shift_audit",
        "REQUIRE_PROTECTED_TRAFFIC": 1,
    }


def generate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        _row(run="cee-q-smoke-a0-none-s8101", jammer="none", seed=8101, method="A0", enabled=False),
        _row(run="cee-q-smoke-safe-none-s8101", jammer="none", seed=8101, method="S", selection="safeguarded_policy"),
        _row(run="cee-q-smoke-a0-partial-s8201", jammer="partial_band", seed=8201, method="A0", enabled=False),
        _row(run="cee-q-smoke-safe-partial-s8201", jammer="partial_band", seed=8201, method="S", selection="safeguarded_policy"),
    ]
    # Replicate no-jam integrity and matched action probes.  These are ordered
    # after the four smoke rows so the manifest runner fails fast.
    for seed in (8102, 8103):
        rows.extend((
            _row(run=f"cee-q-a0-none-s{seed}", jammer="none", seed=seed, method="A0", enabled=False),
            _row(run=f"cee-q-safe-none-s{seed}", jammer="none", seed=seed, method="S", selection="safeguarded_policy"),
        ))
    for jammer, seeds, actions in (
        ("partial_band", (8201, 8202), (("A0", None), ("A1", "1"), ("A2", "2"), ("A3", "3"))),
        ("barrage", (8301, 8302), (("A0", None), ("A2", "2"), ("A3", "3"))),
    ):
        for seed in seeds:
            for method, action in actions:
                if method == "A0":
                    rows.append(_row(run=f"cee-q-{method.lower()}-{jammer}-s{seed}", jammer=jammer,
                                     seed=seed, method=method, enabled=False))
                else:
                    rows.append(_row(run=f"cee-q-{method.lower()}-{jammer}-s{seed}", jammer=jammer,
                                     seed=seed, method=method, selection="bounded_action_probe", action=action))
            rows.append(_row(run=f"cee-q-safe-{jammer}-s{seed}", jammer=jammer, seed=seed,
                             method="S", selection="safeguarded_policy"))
    if len(rows) != 26 or len({str(row["run_id"]) for row in rows}) != 26:
        raise AssertionError("qualification must contain exactly 26 unique rows")
    return rows


def main() -> int:
    rows = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    FREEZE.write_text(json.dumps({
        "schema_version": 1, "evidence_role": "DEVELOPMENT_ONLY_ENGINEERING_QUALIFICATION",
        "manifest": str(OUT.relative_to(ROOT)).replace("\\", "/"), "rows": len(rows),
        "mandatory_smoke_rows": 4, "sha256": digest,
        "v3_to_v10_eligibility": "PERMANENTLY_INELIGIBLE_FOR_CONFIRMATORY_CLAIMS",
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(OUT), "rows": len(rows), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
