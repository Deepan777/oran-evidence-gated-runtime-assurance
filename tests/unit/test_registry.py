from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.registry.experiment_registry import append_record, new_record


def test_registry_writes_schema_and_rejects_duplicate_run_id(tmp_path) -> None:
    record = new_record(
        git_commit="abc123",
        methodology_version="1.0",
        config_hash="cfg",
        dataset_version="d1",
        split_version="s1",
        seed=1001,
        environment_fingerprint="env",
        machine="test-host",
        gpu="none",
        scenario_id="scenario-a",
        raw_result_path="results/raw/run.json",
        timestamp=datetime(2026, 8, 11, tzinfo=timezone.utc),
    )
    path = tmp_path / "registry.csv"
    append_record(path, record)
    content = path.read_text(encoding="utf-8")
    assert record.run_id in content
    assert "methodology_version" in content.splitlines()[0]
    with pytest.raises(ValueError, match="already registered"):
        append_record(path, record)

