from __future__ import annotations

from experiments.generate_live_validation_v3 import make_rows


def test_v3_preserves_scenario_identity_but_separates_method_execution_ids() -> None:
    source = [{"run_id": f"scenario-run-{index}", "scenario_id": f"scenario-{index}", "seed": str(index)} for index in range(3)]
    b0 = make_rows(source, "B0")
    b2 = make_rows(source, "B2")
    assert {row["source_run_id"] for row in b0} == {row["source_run_id"] for row in b2}
    assert {row["scenario_id"] for row in b0} == {row["scenario_id"] for row in b2}
    assert {row["seed"] for row in b0} == {row["seed"] for row in b2}
    assert not ({row["run_id"] for row in b0} & {row["run_id"] for row in b2})
