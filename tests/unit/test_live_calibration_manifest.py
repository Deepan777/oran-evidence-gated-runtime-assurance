from __future__ import annotations

from collections import Counter

from experiments.generate_live_calibration_manifest import JAMMERS, make_rows


def test_live_calibration_manifest_has_independent_run_level_class_coverage() -> None:
    rows = make_rows()
    assert len(rows) == 30
    assert len({row["run_id"] for row in rows}) == 30
    for partition in ("train", "validation", "locked_test"):
        assert Counter(row["jammer_type"] for row in rows if row["partition"] == partition) == Counter({name: 3 if partition == "train" else 1 for name in JAMMERS})
    assert {row["evidence_class"] for row in rows} == {"development_live_calibration"}
    assert all(row["jsr_db"] == "" for row in rows if row["jammer_type"] == "none")
