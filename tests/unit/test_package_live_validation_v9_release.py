from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.package_live_validation_v9_release import release_files


def test_release_blocks_when_required_outputs_are_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="release is incomplete"):
        release_files(tmp_path)


def test_release_fail_requires_failure_analysis(tmp_path: Path) -> None:
    required = [
        "LIVE_VALIDATION_FREEZE_MANIFEST_V9.yaml", "LIVE_VALIDATION_V9_ANALYSIS_PLAN.yaml",
        "results/aggregate/live_validation_v9_records.parquet",
        "results/aggregate/live_validation_v9_per_run_outcomes.csv",
        "results/aggregate/live_validation_v9_run_provenance.csv",
        "results/aggregate/statistics_input.json", "results/aggregate/live_validation_v9_xai_summary.json",
        "results/aggregate/evidence_summary.json", "results/statistics/live_validation_v9_stats_report.json",
        "results/statistics/live_validation_v9_traceability_audit.json", "results/evidence_gate.json",
        "results/evidence_gate/EVIDENCE_GATE_REPORT.md", "results/LIVE_VALIDATION_V9_FINAL_RESULTS.md",
        "results/release/live_validation_v9_raw_evidence_manifest.json",
        "results/release/live_validation_v9_raw_evidence_manifest.json.sha256",
    ] + [f"results/raw/LIVE_VALIDATION_V9_{method}.json" for method in ("B0", "B2", "B5", "P")]
    for relative in required:
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "FAIL"}) if relative == "results/evidence_gate.json" else "x", encoding="utf-8")
    with pytest.raises(ValueError, match="mandatory failure analysis"):
        release_files(tmp_path)
    failure = tmp_path / "results/failure_analysis.md"; failure.write_text("failure", encoding="utf-8")
    assert failure in release_files(tmp_path)
