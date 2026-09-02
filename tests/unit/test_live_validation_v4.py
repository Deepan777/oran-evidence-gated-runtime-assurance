from __future__ import annotations

import yaml

from experiments.generate_live_validation_v4 import make_rows


def test_v4_preserves_source_scenario_and_assigns_fresh_method_ids() -> None:
    source = [
        {
            "run_id": f"livev3-b0-scenario-{index}",
            "source_run_id": f"scenario-{index}",
            "scenario_id": f"cell-{index}",
            "seed": str(index),
        }
        for index in range(3)
    ]
    b0 = make_rows(source, "B0")
    p = make_rows(source, "P")
    assert {row["source_run_id"] for row in b0} == {row["source_run_id"] for row in p}
    assert {row["scenario_id"] for row in b0} == {row["scenario_id"] for row in p}
    assert {row["seed"] for row in b0} == {row["seed"] for row in p}
    assert not ({row["run_id"] for row in b0} & {row["run_id"] for row in p})


def test_udp_um_radio_profile_is_method_neutral_and_synthetic_free() -> None:
    config = yaml.safe_load(open("config/baseline/ocudu_gnb_zmq_e2_udp_um.yml", encoding="utf-8"))
    assert config["qos"][0]["five_qi"] == 9
    assert config["qos"][0]["rlc"]["mode"] == "um-bidir"
    assert "allow_uplink_silence" not in config


def test_generated_v4_freeze_preserves_all_v3_factors_and_has_360_unique_ids() -> None:
    import csv
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    freeze = yaml.safe_load((root / "LIVE_VALIDATION_FREEZE_MANIFEST_V4.yaml").read_text(encoding="utf-8"))
    assert freeze["status"] == "LIVE_VALIDATION_FROZEN_V4"
    assert freeze["total_method_executions"] == 360
    all_ids: list[str] = []
    for method in freeze["methods"]:
        v3_path = root / f"datasets/metadata/manifests/live_validation_v3/{method}_live_validation_runs.csv"
        v4_path = root / f"datasets/metadata/manifests/live_validation_v4/{method}_live_validation_runs.csv"
        with v3_path.open(encoding="utf-8", newline="") as handle:
            v3 = list(csv.DictReader(handle))
        with v4_path.open(encoding="utf-8", newline="") as handle:
            v4 = list(csv.DictReader(handle))
        assert len(v3) == len(v4) == 90
        for old, new in zip(v3, v4, strict=True):
            for key, value in old.items():
                if key != "run_id":
                    assert new[key] == value
        all_ids.extend(row["run_id"] for row in v4)
    assert len(all_ids) == len(set(all_ids)) == 360
    for name, artifact in freeze["artifacts"].items():
        if name.endswith("_v4_manifest") or name == "v4_radio_config":
            actual = hashlib.sha256((root / artifact["path"]).read_bytes()).hexdigest()
            assert actual == artifact["sha256"]
