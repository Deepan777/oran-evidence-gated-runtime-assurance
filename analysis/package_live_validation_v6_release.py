#!/usr/bin/env python3
"""Create a hash-verifiable V6 release bundle after strict post-processing."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files(root: Path) -> list[Path]:
    required = [
        root / "LIVE_VALIDATION_FREEZE_MANIFEST_V6.yaml",
        root / "LIVE_VALIDATION_V6_ANALYSIS_PLAN.yaml",
        root / "results/aggregate/live_validation_v6_records.parquet",
        root / "results/aggregate/live_validation_v6_per_run_outcomes.csv",
        root / "results/aggregate/live_validation_v6_run_provenance.csv",
        root / "results/aggregate/statistics_input.json",
        root / "results/aggregate/live_validation_v6_xai_summary.json",
        root / "results/aggregate/evidence_summary.json",
        root / "results/statistics/live_validation_v6_stats_report.json",
        root / "results/statistics/live_validation_v6_traceability_audit.json",
        root / "results/evidence_gate.json",
        root / "results/evidence_gate/EVIDENCE_GATE_REPORT.md",
        root / "results/LIVE_VALIDATION_V6_FINAL_RESULTS.md",
        root / "results/release/live_validation_v6_raw_evidence_manifest.json",
        root / "results/release/live_validation_v6_raw_evidence_manifest.json.sha256",
    ]
    for method in ("B0", "B2", "B5", "P"):
        required.append(root / f"results/raw/LIVE_VALIDATION_V6_{method}.json")
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"final V6 release is incomplete: {missing}")
    files = list(required)
    for directory in ("results/tables", "results/figures", "reproducibility"):
        files.extend(path for path in (root / directory).rglob("*") if path.is_file())
    gate = json.loads((root / "results/evidence_gate.json").read_text(encoding="utf-8"))
    if gate.get("status") == "PASS":
        manuscript = root / "manuscript/manuscript.md"
        claim_map = root / "manuscript/claim_evidence_map.csv"
        if not manuscript.is_file() or not claim_map.is_file():
            raise ValueError("gate PASS release lacks its manuscript or claim-evidence map")
        files.extend(path for path in (root / "manuscript").rglob("*") if path.is_file())
    else:
        failure = root / "results/failure_analysis.md"
        if not failure.is_file():
            raise ValueError("gate FAIL release lacks the mandatory failure analysis")
        files.append(failure)
    unique = sorted(set(files), key=lambda path: str(path.relative_to(root)).replace("\\", "/"))
    return unique


def package_release(root: Path) -> dict[str, object]:
    files = release_files(root)
    output_dir = root / "results/release"; output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "live_validation_v6_release_manifest.json"
    manifest = {
        "schema_version": 1,
        "campaign": "LIVE_VALIDATION_V6",
        "accepted_unique_runs": 360,
        "files": [
            {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = output_dir / "live_validation_v6_final_release.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in [*files, manifest_path]:
            bundle.add(path, arcname=str(path.relative_to(root)).replace("\\", "/"), recursive=False)
    archive_hash = _sha256(archive)
    hash_path = archive.with_suffix(archive.suffix + ".sha256")
    hash_path.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")
    result = {
        "status": "PASS", "files": len(files), "archive": str(archive),
        "archive_sha256": archive_hash, "archive_bytes": archive.stat().st_size,
    }
    (output_dir / "live_validation_v6_packaging_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    print(json.dumps(package_release(project_root), indent=2))
