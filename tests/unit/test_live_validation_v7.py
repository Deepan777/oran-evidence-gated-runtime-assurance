from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("B0", "B2", "B5", "P")
PROFILE = "observed_outage_censoring_v1"


def rows(version: str, method: str) -> list[dict[str, str]]:
    path = ROOT / f"datasets/metadata/manifests/live_validation_{version}/{method}_live_validation_runs.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v7_preserves_v6_scientific_rows_and_versions_only_evidence_contract() -> None:
    ignored = {"run_id", "ACCEPTANCE_PROFILE"}
    all_ids: set[str] = set()
    for method in METHODS:
        previous, current = rows("v6", method), rows("v7", method)
        assert len(previous) == len(current) == 90
        for old, new in zip(previous, current, strict=True):
            assert new["run_id"].startswith(f"livev7-{method.lower()}-")
            assert all(old[key] == new[key] for key in old if key not in ignored)
            assert new["ACCEPTANCE_PROFILE"] == PROFILE
        ids = {item["run_id"] for item in current}
        assert not all_ids.intersection(ids)
        all_ids.update(ids)
    assert len(all_ids) == 360


def test_v7_freeze_documents_no_raw_synthesis_and_no_scientific_factor_change() -> None:
    freeze = yaml.safe_load((ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V7.yaml").read_text(encoding="utf-8"))
    correction = freeze["outage_evidence_correction"]
    assert freeze["status"] == "LIVE_VALIDATION_FROZEN_V7"
    assert correction["acceptance_profile"] == PROFILE
    assert correction["raw_missing_values_synthesized"] is False
    assert correction["scenario_factors_changed"] is False
    assert correction["controllers_changed"] is False
    assert correction["radio_profile_changed"] is False
    assert correction["runtime_image_changed"] is False
    assert correction["acceptance_rules_changed"] is True


def test_campaign_runner_supports_v7() -> None:
    source = (ROOT / "experiments/collection/run_live_validation_campaign.py").read_text(encoding="utf-8")
    assert '"LIVE_VALIDATION_FROZEN_V7"' in source
