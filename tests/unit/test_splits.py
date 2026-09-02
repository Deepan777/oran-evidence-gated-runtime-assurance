from __future__ import annotations

from src.common.splits import RunDescriptor, create_run_level_split


def test_run_level_split_is_disjoint_deterministic_and_complete() -> None:
    runs = [RunDescriptor(f"run-{index}", f"scenario-{index % 4}", index) for index in range(20)]
    runs += [RunDescriptor("gen-1", "held-out", 99, True)]
    first = create_run_level_split(runs, split_seed=71)
    second = create_run_level_split(runs, split_seed=71)
    assert first == second
    first.validate()
    combined = set(first.train + first.validation + first.locked_test + first.generalization)
    assert combined == {run.run_id for run in runs}
    assert len(first.train) == 12
    assert len(first.validation) == 4
    assert len(first.locked_test) == 4
    assert first.generalization == ("gen-1",)

