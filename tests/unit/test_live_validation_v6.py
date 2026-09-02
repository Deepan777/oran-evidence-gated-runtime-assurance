from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("B0", "B2", "B5", "P")
UE_IMAGE = "srsue-zmq:git-6bcbd9e5bf86-telemetry-v2-iperf3-buffer16384"


def rows(version: str, method: str) -> list[dict[str, str]]:
    path = ROOT / f"datasets/metadata/manifests/live_validation_{version}/{method}_live_validation_runs.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v6_preserves_v5_scientific_rows_and_versions_runtime() -> None:
    ignored = {"run_id", "UE_IMAGE"}
    all_ids: set[str] = set()
    for method in METHODS:
        previous, current = rows("v5", method), rows("v6", method)
        assert len(previous) == len(current) == 90
        for old, new in zip(previous, current, strict=True):
            assert new["run_id"].startswith(f"livev6-{method.lower()}-")
            assert all(old[key] == new[key] for key in old if key not in ignored)
            assert new["UE_IMAGE"] == UE_IMAGE
        ids = {item["run_id"] for item in current}
        assert not all_ids.intersection(ids)
        all_ids.update(ids)
    assert len(all_ids) == 360


def test_v6_freeze_is_method_neutral_and_strict() -> None:
    freeze = yaml.safe_load((ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V6.yaml").read_text(encoding="utf-8"))
    correction = freeze["runtime_correction"]
    assert freeze["status"] == "LIVE_VALIDATION_FROZEN_V6"
    assert correction["ue_image"] == UE_IMAGE
    assert correction["scenario_factors_changed"] is False
    assert correction["controllers_changed"] is False
    assert correction["radio_profile_changed"] is False
    assert correction["acceptance_rules_changed"] is False
    for entry in freeze["execution_plan"]:
        assert "--ue-image" in entry["command"]
        assert UE_IMAGE in entry["command"]


def test_campaign_runner_supports_v6() -> None:
    source = (ROOT / "experiments/collection/run_live_validation_campaign.py").read_text(encoding="utf-8")
    assert '"LIVE_VALIDATION_FROZEN_V6"' in source
