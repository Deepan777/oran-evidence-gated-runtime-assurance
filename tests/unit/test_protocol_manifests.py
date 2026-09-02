from __future__ import annotations

import yaml

from experiments.generate_protocol_manifests import _rows


def test_confirmatory_matrix_is_complete_and_no_jam_is_not_jsr_duplicated() -> None:
    config = yaml.safe_load(open("configs/experiments/protocol.yaml", encoding="utf-8"))
    rows = _rows(config, "confirmatory")
    # (one no-jam condition + five jammer families x four JSRs) x other factors x 20 seeds.
    assert len(rows) == (1 + 5 * 4) * 3 * 2 * 2 * 20
    assert len({row["run_id"] for row in rows}) == len(rows)
    assert all(row["jsr_db"] is None for row in rows if row["jammer_type"] == "none")


def test_generalization_runs_are_explicitly_marked() -> None:
    config = yaml.safe_load(open("configs/experiments/protocol.yaml", encoding="utf-8"))
    rows = _rows(config, "confirmatory")
    assert any(row["generalization_holdout"] for row in rows)
    assert any(not row["generalization_holdout"] for row in rows)
