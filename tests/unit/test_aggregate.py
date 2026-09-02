from __future__ import annotations

from pathlib import Path

import pytest

from analysis.aggregate import METHODS, source_entries


def test_confirmatory_entries_preserve_all_locked_method_labels(tmp_path: Path) -> None:
    methods = {}
    for method in METHODS:
        item = tmp_path / f"{method}.parquet"
        item.touch()
        methods[method] = [str(item)]
    entries = source_entries({"methods": methods}, tmp_path / "complete.json")
    assert [method for method, _ in entries] == list(METHODS)


def test_confirmatory_entries_reject_partial_method_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="locked baseline"):
        source_entries({"methods": {"B0": [str(tmp_path / "only.parquet")]}}, tmp_path / "complete.json")
