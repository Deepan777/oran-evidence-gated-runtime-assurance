import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("B0", "B2", "B5", "P")
SCHEMA = "results/training/live_controller_recalibrated_v2/state_schema.json"


def rows(version: str, method: str) -> list[dict[str, str]]:
    path = ROOT / f"datasets/metadata/manifests/live_validation_{version}/{method}_live_validation_runs.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v5_preserves_v4_factors_and_versions_execution_ids() -> None:
    ignored = {"run_id", "STATE_SCHEMA"}
    all_ids: set[str] = set()
    for method in METHODS:
        previous, current = rows("v4", method), rows("v5", method)
        assert len(previous) == len(current) == 90
        for old, new in zip(previous, current, strict=True):
            assert new["run_id"].startswith(f"livev5-{method.lower()}-")
            assert all(old[key] == new[key] for key in old if key not in ignored)
            if int(new.get("ENABLE_LIVE_XAPP") or 0):
                assert new["STATE_SCHEMA"] == SCHEMA
        ids = {item["run_id"] for item in current}
        assert not all_ids.intersection(ids)
        all_ids.update(ids)
    assert len(all_ids) == 360


def test_v5_freeze_records_schema_change_control() -> None:
    freeze = yaml.safe_load((ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V5.yaml").read_text(encoding="utf-8"))
    assert freeze["status"] == "LIVE_VALIDATION_FROZEN_V5"
    assert freeze["controller_schema_correction"]["models_changed"] is False
    assert freeze["controller_schema_correction"]["scenario_factors_changed"] is False
    assert freeze["controller_schema_correction"]["acceptance_rules_changed"] is False


def test_campaign_runner_supports_v5() -> None:
    source = (ROOT / "experiments/collection/run_live_validation_campaign.py").read_text(encoding="utf-8")
    assert '"LIVE_VALIDATION_FROZEN_V5"' in source
