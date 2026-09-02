#!/usr/bin/env python3
"""Hash-inventory every immutable raw evidence file accepted by V10.

The release archive intentionally does not duplicate multi-gigabyte PCAP and
radio-truth files.  This manifest authenticates those retained files on the
stopped VM while the compact 43,200-row analysis dataset is shipped in the
release archive.  Only evidence directories referenced by the four accepted
method ledgers are eligible.
"""
from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path
from typing import Any


METHODS = ("B0", "B2", "B5", "P")
CAMPAIGN = "LIVE_VALIDATION_V10"
RUNS_PER_METHOD = 90
WINDOWS_PER_RUN = 120
EXPECTED_RUNS = len(METHODS) * RUNS_PER_METHOD
EXPECTED_WINDOWS = EXPECTED_RUNS * WINDOWS_PER_RUN


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_record_files(root: Path, method: str) -> set[Path]:
    manifest_path = root / f"datasets/metadata/manifests/live_validation_v10/{method}_live_validation_runs.csv"
    if not manifest_path.is_file():
        raise ValueError(f"missing V10 frozen method manifest: {manifest_path.relative_to(root)}")
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    run_ids = [row.get("run_id", "") for row in rows]
    if (
        len(rows) != RUNS_PER_METHOD
        or len(set(run_ids)) != RUNS_PER_METHOD
        or any(row.get("method") != method for row in rows)
        or any(not run_id.startswith(f"livev10-{method.lower()}-") for run_id in run_ids)
    ):
        raise ValueError(f"{method} frozen manifest does not define exactly 90 unique V10 runs")
    return {
        (root / "results/raw" / f"iq-broker-live-{run_id}" / "records.parquet").resolve()
        for run_id in run_ids
    }


def _ledger_record_files(root: Path, method: str) -> list[Path]:
    ledger_path = root / f"results/raw/LIVE_VALIDATION_V10_{method}.json"
    if not ledger_path.is_file():
        raise ValueError(f"missing V10 method ledger: {ledger_path.relative_to(root)}")
    ledger: dict[str, Any] = json.loads(ledger_path.read_text(encoding="utf-8"))
    if (
        ledger.get("accepted_count") != RUNS_PER_METHOD
        or ledger.get("expected_count") != RUNS_PER_METHOD
        or ledger.get("remaining") != 0
    ):
        raise ValueError(f"{method} ledger does not certify exactly 90 accepted runs")
    values = ledger.get("record_files")
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{method} ledger record_files must be a list of paths")
    files = [
        (Path(value) if Path(value).is_absolute() else root / value).resolve()
        for value in values
    ]
    if len(files) != RUNS_PER_METHOD or len(set(files)) != RUNS_PER_METHOD:
        raise ValueError(f"{method} ledger record_files are not 90 unique paths")
    if set(files) != _frozen_record_files(root, method):
        raise ValueError(f"{method} ledger record_files do not match the frozen V10 method manifest")
    return files


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    accepted: list[tuple[str, Path]] = []
    for method in METHODS:
        accepted.extend((method, path) for path in _ledger_record_files(root, method))
    evidence_dirs = [path.parent for _, path in accepted]
    if len(set(evidence_dirs)) != EXPECTED_RUNS:
        raise ValueError("accepted ledgers do not reference 360 unique evidence directories")

    runs: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    for method, records_path in accepted:
        evidence = records_path.parent
        try:
            evidence.relative_to(root / "results/raw")
        except ValueError as exc:
            raise ValueError(f"evidence directory escapes results/raw: {evidence}") from exc
        required = (
            evidence / "records.parquet",
            evidence / "record-build-summary.json",
            evidence / "radio-clock-status.json",
            evidence / "run_metadata.json",
            evidence / "client-exit-status.txt",
        )
        missing = [path.name for path in required if not path.is_file() or path.is_symlink()]
        if missing:
            raise ValueError(f"{evidence.name} lacks required immutable evidence: {missing}")
        entries: list[dict[str, Any]] = []
        for path in sorted(evidence.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"raw evidence contains an ineligible symbolic link: {path}")
            if not path.is_file():
                continue
            size = path.stat().st_size
            entries.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": sha256_file(path),
            })
            total_files += 1
            total_bytes += size
        runs.append({
            "method": method,
            "run_id": evidence.name.removeprefix("iq-broker-live-"),
            "evidence_directory": evidence.relative_to(root).as_posix(),
            "files": entries,
        })

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "accepted_unique_runs": EXPECTED_RUNS,
        "accepted_windows": EXPECTED_WINDOWS,
        "windows_per_run": WINDOWS_PER_RUN,
        "matched_blocks": RUNS_PER_METHOD,
        "methods": {method: RUNS_PER_METHOD for method in METHODS},
        "raw_evidence_files": total_files,
        "raw_evidence_bytes": total_bytes,
        "runs": runs,
    }
    output_dir = root / "results/release"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "live_validation_v10_raw_evidence_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    result = build_manifest(project_root)
    print(json.dumps({
        "status": "PASS",
        "accepted_unique_runs": result["accepted_unique_runs"],
        "raw_evidence_files": result["raw_evidence_files"],
        "raw_evidence_bytes": result["raw_evidence_bytes"],
    }, indent=2))
