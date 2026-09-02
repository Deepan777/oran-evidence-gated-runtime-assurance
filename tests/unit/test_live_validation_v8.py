from __future__ import annotations

import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("B0", "B2", "B5", "P")
SCIENTIFIC_FIELDS = (
    "source_run_id", "scenario_id", "jammer_type", "jsr_db", "traffic_load",
    "service_mix", "attenuation", "seed", "run_length_seconds",
    "pre_jam_seconds", "jammer_on_seconds", "recovery_seconds", "method",
    "CONTROLLER_KIND", "DETECTOR_MODEL", "CONTROLLER_MODEL", "STATE_SCHEMA",
    "UE_IMAGE", "ACCEPTANCE_PROFILE",
)


def rows(version: str, method: str) -> list[dict[str, str]]:
    path = ROOT / f"datasets/metadata/manifests/live_validation_{version}/{method}_live_validation_runs.csv"
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def test_v8_preserves_v7_science_and_adds_only_versioned_guard() -> None:
    execution_ids: set[str] = set()
    for method in METHODS:
        previous, current = rows("v7", method), rows("v8", method)
        assert len(previous) == len(current) == 90
        for old, new in zip(previous, current, strict=True):
            assert new["run_id"].startswith(f"livev8-{method.lower()}-")
            assert new["run_id"] not in execution_ids
            execution_ids.add(new["run_id"])
            assert new["OOD_GUARD_PROFILE"] == "physical_plausibility_with_shift_audit"
            for field in SCIENTIFIC_FIELDS:
                assert new[field] == old[field]
    assert len(execution_ids) == 360


def test_v8_freeze_predeclares_audit_only_training_shift() -> None:
    freeze = yaml.safe_load((ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V8.yaml").read_text(encoding="utf-8"))
    assert freeze["status"] == "LIVE_VALIDATION_FROZEN_V8"
    assert freeze["execution_revision"] == "V8_PHYSICAL_GUARD_OUTAGE_R1"
    correction = freeze["physical_guard_correction"]
    assert correction["training_p01_p99_usage"] == "audit-only distribution-shift warnings"
    assert correction["scenario_factors_changed"] is False
    assert correction["controllers_changed"] is False
    assert correction["radio_profile_changed"] is False


def test_campaign_runner_supports_v8() -> None:
    source = (ROOT / "experiments/collection/run_live_validation_campaign.py").read_text(encoding="utf-8")
    assert '"LIVE_VALIDATION_FROZEN_V8"' in source
