from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.build_live_validation_v6_raw_evidence_manifest import build_manifest, sha256_file


def _campaign(root: Path, *, runs_per_method: int = 90) -> None:
    for method in ("B0", "B2", "B5", "P"):
        record_files = []
        for index in range(runs_per_method):
            evidence = root / "results/raw" / f"iq-broker-live-livev6-{method.lower()}-{index}"
            evidence.mkdir(parents=True, exist_ok=True)
            for name in (
                "records.parquet", "record-build-summary.json", "radio-clock-status.json",
                "run_metadata.json", "client-exit-status.txt",
            ):
                (evidence / name).write_text(f"{method}-{index}-{name}", encoding="utf-8")
            record_files.append(str((evidence / "records.parquet").resolve()))
        ledger = {
            "accepted_count": runs_per_method,
            "expected_count": runs_per_method,
            "record_files": record_files,
        }
        path = root / f"results/raw/LIVE_VALIDATION_V6_{method}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ledger), encoding="utf-8")


def test_manifest_requires_complete_campaign(tmp_path: Path) -> None:
    _campaign(tmp_path, runs_per_method=1)
    with pytest.raises(ValueError, match="exactly 90"):
        build_manifest(tmp_path)


def test_manifest_hashes_all_accepted_raw_files(tmp_path: Path) -> None:
    _campaign(tmp_path)
    manifest = build_manifest(tmp_path)
    assert manifest["accepted_unique_runs"] == 360
    assert manifest["methods"] == {"B0": 90, "B2": 90, "B5": 90, "P": 90}
    assert len(manifest["runs"]) == 360
    assert manifest["raw_evidence_files"] == 360 * 5
    output = tmp_path / "results/release/live_validation_v6_raw_evidence_manifest.json"
    expected = (output.with_suffix(output.suffix + ".sha256").read_text().split())[0]
    assert expected == sha256_file(output)
