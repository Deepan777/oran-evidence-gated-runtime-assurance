from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

import analysis.package_live_validation_v10_release as packaging
from analysis.package_live_validation_v10_release import (
    _required_files,
    _validate_method_ledgers,
    package_release,
    release_files,
)


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    "LIVE_VALIDATION_FREEZE_MANIFEST_V10.yaml",
    "LIVE_VALIDATION_V10_ANALYSIS_PLAN.yaml",
    "results/aggregate/live_validation_v10_records.parquet",
    "results/aggregate/live_validation_v10_per_run_outcomes.csv",
    "results/aggregate/live_validation_v10_run_provenance.csv",
    "results/aggregate/live_validation_v10_statistics_input.json",
    "results/aggregate/live_validation_v10_xai_summary.json",
    "results/aggregate/live_validation_v10_evidence_summary.json",
    "results/statistics/live_validation_v10_stats_report.json",
    "results/statistics/live_validation_v10_traceability_audit.json",
    "results/evidence_gate/live_validation_v10_evidence_gate.json",
    "results/evidence_gate/LIVE_VALIDATION_V10_EVIDENCE_GATE_REPORT.md",
    "results/LIVE_VALIDATION_V10_FINAL_RESULTS.md",
    "results/release/live_validation_v10_raw_evidence_manifest.json",
    "results/release/live_validation_v10_raw_evidence_manifest.json.sha256",
    "results/raw/LIVE_VALIDATION_V10_B0.json",
    "results/raw/LIVE_VALIDATION_V10_B2.json",
    "results/raw/LIVE_VALIDATION_V10_B5.json",
    "results/raw/LIVE_VALIDATION_V10_P.json",
)
MANUSCRIPT_FILES = (
    "manuscript/LIVE_VALIDATION_V10_RESULTS.md",
    "manuscript/LIVE_VALIDATION_V10_MANUSCRIPT.md",
    "manuscript/LIVE_VALIDATION_V10_MANUSCRIPT.md.producer.json",
    "manuscript/LIVE_VALIDATION_V10_CLAIM_EVIDENCE_MAP.csv",
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_fixture(root: Path, gate_status: str) -> None:
    for path in _required_files(root):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("results/raw/LIVE_VALIDATION_V10_"):
            content = json.dumps({
                "accepted_count": 90,
                "expected_count": 90,
                "remaining": 0,
                "record_files": [f"{relative}-record-{index:03d}" for index in range(90)],
            })
        elif relative.endswith("live_validation_v10_raw_evidence_manifest.json"):
            content = json.dumps({
                "campaign": "LIVE_VALIDATION_V10",
                "accepted_unique_runs": 360,
                "accepted_windows": 43_200,
                "windows_per_run": 120,
                "matched_blocks": 90,
                "methods": {"B0": 90, "B2": 90, "B5": 90, "P": 90},
                "runs": [{} for _ in range(360)],
            })
        elif relative.endswith("live_validation_v10_evidence_gate.json"):
            content = json.dumps({
                "status": gate_status,
                "manuscript_allowed": gate_status == "PASS",
                "campaign": "LIVE_VALIDATION_V10",
                "accepted_unique_runs": 360,
                "accepted_windows": 43_200,
            })
        else:
            content = "{}" if relative.endswith(".json") else "fixture\n"
        _write(path, content)

    raw_manifest = root / "results/release/live_validation_v10_raw_evidence_manifest.json"
    _write(
        raw_manifest.with_suffix(raw_manifest.suffix + ".sha256"),
        f"{_sha256(raw_manifest)}  {raw_manifest.name}\n",
    )


def test_v10_release_blocks_when_required_outputs_are_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="release is incomplete"):
        release_files(tmp_path)


def test_v10_gate_fail_requires_versioned_failure_analysis(tmp_path: Path) -> None:
    _release_fixture(tmp_path, "FAIL")
    with pytest.raises(ValueError, match="mandatory .*failure analysis"):
        release_files(tmp_path)

    failure = tmp_path / "results/LIVE_VALIDATION_V10_FAILURE_ANALYSIS.md"
    _write(failure, "failure analysis\n")

    assert failure in release_files(tmp_path)
    assert not (tmp_path / "results/failure_analysis.md").exists()


@pytest.mark.parametrize("missing", MANUSCRIPT_FILES)
def test_v10_gate_pass_requires_every_versioned_manuscript_file(
    tmp_path: Path,
    missing: str,
) -> None:
    _release_fixture(tmp_path, "PASS")
    for relative in MANUSCRIPT_FILES:
        if relative != missing:
            _write(tmp_path / relative, "publication artifact\n")

    with pytest.raises(ValueError, match="manuscript|claim-evidence"):
        release_files(tmp_path)


def test_v10_release_rejects_incorrect_method_cardinality(tmp_path: Path) -> None:
    _release_fixture(tmp_path, "FAIL")
    _write(tmp_path / "results/raw/LIVE_VALIDATION_V10_B2.json", json.dumps({
        "accepted_count": 89,
        "expected_count": 90,
        "remaining": 1,
        "record_files": [f"record-{index:03d}" for index in range(89)],
    }))

    with pytest.raises(ValueError, match="90|cardinality|complete"):
        _validate_method_ledgers(tmp_path)


def test_v10_package_manifest_and_archive_are_hash_verifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _release_fixture(tmp_path, "PASS")
    for relative in MANUSCRIPT_FILES:
        _write(tmp_path / relative, "publication artifact\n")
    monkeypatch.setattr(packaging, "_validate_release_evidence", lambda _: {
        "gate": {"status": "PASS"},
        "traceability": {"status": "PASS"},
        "raw_manifest_sha256": _sha256(
            tmp_path / "results/release/live_validation_v10_raw_evidence_manifest.json"
        ),
    })

    status = package_release(tmp_path)

    release = tmp_path / "results/release"
    manifest_path = release / "live_validation_v10_release_manifest.json"
    archive = release / "live_validation_v10_final_release.tar.gz"
    sidecar = release / "live_validation_v10_final_release.tar.gz.sha256"
    status_path = release / "live_validation_v10_packaging_status.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["campaign"] == "LIVE_VALIDATION_V10"
    assert manifest["accepted_unique_runs"] == 360
    for item in manifest["files"]:
        candidate = tmp_path / item["path"]
        assert candidate.stat().st_size == item["bytes"]
        assert _sha256(candidate) == item["sha256"]
    with tarfile.open(archive, "r:gz") as bundle:
        assert "results/release/live_validation_v10_release_manifest.json" in bundle.getnames()
    assert sidecar.read_text(encoding="utf-8").split()[0] == _sha256(archive)
    assert json.loads(status_path.read_text(encoding="utf-8"))["archive_sha256"] == _sha256(archive)
    assert status["status"] == "PASS"


def test_v10_sync_verifies_archive_and_files_before_stopping_vm() -> None:
    source = (ROOT / "tools/sync_live_validation_v10_release.ps1").read_text(encoding="utf-8")
    ordered_markers = (
        "Get-FileHash -LiteralPath $archive",
        "$packaging.status",
        "tar -tzf $archive",
        "Unsafe release archive path",
        "tar -xzf $archive",
        "$manifest.accepted_unique_runs -ne 360",
        "Get-FileHash -LiteralPath $candidate",
        "compute instances stop",
        "$vmStatus -ne 'TERMINATED'",
        "live_validation_v10_local_verification.json",
    )
    positions = [source.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)
    assert "live_validation_v9" not in source.lower()
