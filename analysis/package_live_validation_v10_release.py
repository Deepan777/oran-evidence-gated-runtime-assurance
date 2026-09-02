#!/usr/bin/env python3
"""Create and verify a self-authenticating V10 release bundle.

Packaging success describes archive integrity. A scientifically failed
evidence gate is still preserved as a failure release, but it can never carry
V10 manuscript artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO


CAMPAIGN = "LIVE_VALIDATION_V10"
METHODS = ("P", "B0", "B2", "B5")
EXPECTED_METHOD_RUNS = 90
EXPECTED_RUNS = 360
WINDOWS_PER_RUN = 120
EXPECTED_WINDOWS = EXPECTED_RUNS * WINDOWS_PER_RUN
EXPECTED_EXPLANATIONS = EXPECTED_METHOD_RUNS * WINDOWS_PER_RUN
PRIMARY_OUTCOMES = (
    "protected_flow_latency_violation_rate",
    "protected_flow_throughput_retention",
    "protected_flow_packet_or_rlc_drop_rate",
    "recovery_time",
)
PLANNED_PAIRS = {("P", "B0"), ("P", "B2"), ("P", "B5")}


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid JSON release input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON release input is not an object: {path}")
    return value


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _parquet_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - campaign environment pins pyarrow
        raise ValueError("pyarrow is required to verify the V10 aggregate row count") from exc
    metadata = pq.ParquetFile(path).metadata
    if metadata is None:
        raise ValueError("V10 aggregate parquet metadata is missing")
    return int(metadata.num_rows)


def _required_files(root: Path) -> list[Path]:
    required = [
        root / "LIVE_VALIDATION_FREEZE_MANIFEST_V10.yaml",
        root / "LIVE_VALIDATION_V10_ANALYSIS_PLAN.yaml",
        root / "results/aggregate/live_validation_v10_records.parquet",
        root / "results/aggregate/live_validation_v10_per_run_outcomes.csv",
        root / "results/aggregate/live_validation_v10_run_provenance.csv",
        root / "results/aggregate/live_validation_v10_statistics_input.json",
        root / "results/aggregate/live_validation_v10_xai_summary.json",
        root / "results/aggregate/live_validation_v10_evidence_summary.json",
        root / "results/aggregate/figure_source_v10_primary_outcomes.csv",
        root / "results/aggregate/figure_source_v10_jammer_robustness.csv",
        root / "results/aggregate/figure_source_v10_xai_metrics.csv",
        root / "results/statistics/live_validation_v10_stats_report.json",
        root / "results/statistics/live_validation_v10_traceability_audit.json",
        root / "results/evidence_gate/live_validation_v10_evidence_gate.json",
        root / "results/evidence_gate/LIVE_VALIDATION_V10_EVIDENCE_GATE_REPORT.md",
        root / "results/LIVE_VALIDATION_V10_FINAL_RESULTS.md",
        root / "results/release/live_validation_v10_raw_evidence_manifest.json",
        root / "results/release/live_validation_v10_raw_evidence_manifest.json.sha256",
        root / "results/tables/live_validation_v10_primary_outcomes.csv",
        root / "results/tables/live_validation_v10_planned_comparisons.csv",
        root / "results/tables/live_validation_v10_xai_metrics.csv",
        root / "analysis/build_live_validation_v10_raw_evidence_manifest.py",
        root / "analysis/live_validation_v10_postprocess.py",
        root / "analysis/live_validation_v10_publication.py",
        root / "analysis/package_live_validation_v10_release.py",
        root / "analysis/evidence_gate.py",
        root / "analysis/statistics.py",
        root / "tools/finalize_live_validation_v10.sh",
    ]
    for base in (
        "live_validation_v10_primary_outcomes",
        "live_validation_v10_jammer_robustness",
        "live_validation_v10_xai_metrics",
    ):
        for suffix in (".png", ".pdf", ".producer.json"):
            required.append(root / f"results/figures/{base}{suffix}")
    for method in ("B0", "B2", "B5", "P"):
        required.extend([
            root / f"datasets/metadata/manifests/live_validation_v10/{method}_live_validation_runs.csv",
            root / f"results/raw/LIVE_VALIDATION_V10_{method}.json",
        ])
    return required


def release_files(root: Path) -> list[Path]:
    root = root.resolve()
    required = _required_files(root)
    missing = [path.relative_to(root).as_posix() for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"final V10 release is incomplete: {missing}")
    files = list(required)
    gate = _json(root / "results/evidence_gate/live_validation_v10_evidence_gate.json")
    manuscript_files = [
        root / "manuscript/LIVE_VALIDATION_V10_RESULTS.md",
        root / "manuscript/LIVE_VALIDATION_V10_MANUSCRIPT.md",
        root / "manuscript/LIVE_VALIDATION_V10_MANUSCRIPT.md.producer.json",
        root / "manuscript/LIVE_VALIDATION_V10_CLAIM_EVIDENCE_MAP.csv",
    ]
    if gate.get("status") == "PASS" and gate.get("manuscript_allowed") is True:
        missing_manuscript = [path.name for path in manuscript_files if not path.is_file()]
        if missing_manuscript:
            raise ValueError(
                f"gate PASS release lacks its V10 manuscript artifacts: {missing_manuscript}"
            )
        files.extend(manuscript_files)
    elif gate.get("status") == "FAIL" and gate.get("manuscript_allowed") is False:
        stale = [path.name for path in manuscript_files if path.exists()]
        if stale:
            raise ValueError(f"gate FAIL release contains stale V10 manuscript artifacts: {stale}")
        failure = root / "results/LIVE_VALIDATION_V10_FAILURE_ANALYSIS.md"
        if not failure.is_file():
            raise ValueError("gate FAIL release lacks the mandatory V10 failure analysis")
        files.append(failure)
    else:
        raise ValueError("V10 evidence gate status and manuscript_allowed are inconsistent")
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def _validate_method_ledgers(root: Path) -> None:
    for method in ("B0", "B2", "B5", "P"):
        ledger = _json(root / f"results/raw/LIVE_VALIDATION_V10_{method}.json")
        records = ledger.get("record_files")
        if (
            ledger.get("accepted_count") != EXPECTED_METHOD_RUNS
            or ledger.get("expected_count") != EXPECTED_METHOD_RUNS
            or ledger.get("remaining") != 0
            or not isinstance(records, list)
            or len(records) != EXPECTED_METHOD_RUNS
            or len(set(records)) != EXPECTED_METHOD_RUNS
        ):
            raise ValueError(f"{method} ledger does not certify exactly 90 accepted V10 runs")


def _validate_statistics(root: Path) -> None:
    stats_input = _json(root / "results/aggregate/live_validation_v10_statistics_input.json")
    if (
        stats_input.get("replication_unit") != "source_run_id"
        or stats_input.get("bootstrap_iterations") != 10_000
        or stats_input.get("confidence") != 0.95
        or stats_input.get("seed") != 1001
        or list(stats_input.get("outcomes", {})) != list(PRIMARY_OUTCOMES)
    ):
        raise ValueError("V10 statistics input differs from the frozen matched analysis plan")
    for outcome, definition in stats_input["outcomes"].items():
        blocks = definition.get("matched_blocks")
        pairs = definition.get("planned_pairs")
        if (
            definition.get("methods") != list(METHODS)
            or not isinstance(blocks, list)
            or len(blocks) != EXPECTED_METHOD_RUNS
            or any(not isinstance(block, list) or len(block) != len(METHODS) for block in blocks)
            or {tuple(pair) for pair in pairs or []} != PLANNED_PAIRS
        ):
            raise ValueError(f"{outcome} statistics are not 90 complete matched four-method blocks")

    report = _json(root / "results/statistics/live_validation_v10_stats_report.json")
    if list(report.get("outcomes", {})) != list(PRIMARY_OUTCOMES):
        raise ValueError("V10 statistics report omits or adds a primary outcome")
    for outcome, rendered in report["outcomes"].items():
        comparisons = rendered.get("planned_comparisons")
        if (
            rendered.get("methods") != list(METHODS)
            or rendered.get("n_matched_blocks") != EXPECTED_METHOD_RUNS
            or not isinstance(comparisons, list)
            or len(comparisons) != len(PLANNED_PAIRS)
            or {(item.get("left"), item.get("right")) for item in comparisons} != PLANNED_PAIRS
            or any(item.get("n_pairs") != EXPECTED_METHOD_RUNS or "holm" not in item for item in comparisons)
        ):
            raise ValueError(f"{outcome} report does not certify the frozen matched statistics")


def _validate_release_evidence(root: Path) -> dict[str, Any]:
    _validate_method_ledgers(root)
    if _parquet_rows(root / "results/aggregate/live_validation_v10_records.parquet") != EXPECTED_WINDOWS:
        raise ValueError("V10 aggregate does not contain exactly 43,200 windows")
    per_run = _csv_rows(root / "results/aggregate/live_validation_v10_per_run_outcomes.csv")
    provenance = _csv_rows(root / "results/aggregate/live_validation_v10_run_provenance.csv")
    if len(per_run) != EXPECTED_RUNS or len(provenance) != EXPECTED_RUNS:
        raise ValueError("V10 aggregate does not contain exactly 360 run-level rows")
    method_counts = Counter(row.get("method") for row in per_run)
    block_methods: dict[str, set[str]] = {}
    for row in per_run:
        block_methods.setdefault(row.get("source_run_id", ""), set()).add(row.get("method", ""))
    if (
        method_counts != Counter({method: EXPECTED_METHOD_RUNS for method in METHODS})
        or len(block_methods) != EXPECTED_METHOD_RUNS
        or any(methods != set(METHODS) for methods in block_methods.values())
    ):
        raise ValueError("V10 run outcomes do not contain 90 matched four-method blocks")
    if len({row.get("run_id") for row in provenance}) != EXPECTED_RUNS:
        raise ValueError("V10 run provenance does not contain 360 unique execution IDs")

    _validate_statistics(root)
    xai = _json(root / "results/aggregate/live_validation_v10_xai_summary.json")
    if (
        xai.get("total_p_decisions") != EXPECTED_EXPLANATIONS
        or xai.get("applicable_explanations", 0) + xai.get("non_applicable_single_action", 0)
        != EXPECTED_EXPLANATIONS
        or xai.get("execution_mode") != "post_run_nonblocking_worker"
    ):
        raise ValueError("V10 XAI summary does not cover every accepted P decision")

    evidence = _json(root / "results/aggregate/live_validation_v10_evidence_summary.json")
    gate = _json(root / "results/evidence_gate/live_validation_v10_evidence_gate.json")
    traceability = _json(root / "results/statistics/live_validation_v10_traceability_audit.json")
    for label, payload in (("evidence summary", evidence), ("evidence gate", gate)):
        if (
            payload.get("campaign") != CAMPAIGN
            or payload.get("accepted_unique_runs") != EXPECTED_RUNS
            or payload.get("accepted_windows") != EXPECTED_WINDOWS
        ):
            raise ValueError(f"V10 {label} does not certify 360 runs and 43,200 windows")
    if traceability.get("campaign") != CAMPAIGN or traceability.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("V10 traceability audit identity or status is invalid")
    gate_pass = gate.get("status") == "PASS" and gate.get("manuscript_allowed") is True
    gate_fail = gate.get("status") == "FAIL" and gate.get("manuscript_allowed") is False
    if not (gate_pass or gate_fail):
        raise ValueError("V10 evidence gate status and manuscript permission are inconsistent")
    if gate_pass and traceability.get("status") != "PASS":
        raise ValueError("V10 evidence gate passed despite failed traceability")

    raw_path = root / "results/release/live_validation_v10_raw_evidence_manifest.json"
    raw = _json(raw_path)
    if (
        raw.get("campaign") != CAMPAIGN
        or raw.get("accepted_unique_runs") != EXPECTED_RUNS
        or raw.get("accepted_windows") != EXPECTED_WINDOWS
        or raw.get("windows_per_run") != WINDOWS_PER_RUN
        or raw.get("matched_blocks") != EXPECTED_METHOD_RUNS
        or raw.get("methods") != {method: EXPECTED_METHOD_RUNS for method in ("B0", "B2", "B5", "P")}
        or not isinstance(raw.get("runs"), list)
        or len(raw["runs"]) != EXPECTED_RUNS
    ):
        raise ValueError("raw-evidence manifest does not certify the complete V10 campaign")
    sidecar = raw_path.with_suffix(raw_path.suffix + ".sha256").read_text(encoding="utf-8").split()
    if len(sidecar) != 2 or sidecar[0].lower() != _sha256(raw_path) or sidecar[1] != raw_path.name:
        raise ValueError("V10 raw-evidence manifest detached hash is invalid")
    return {"gate": gate, "traceability": traceability, "raw_manifest_sha256": sidecar[0].lower()}


def _verify_archive(archive: Path, root: Path, files: list[Path], manifest_path: Path) -> None:
    expected = {
        path.relative_to(root).as_posix(): (_sha256(path), path.stat().st_size)
        for path in [*files, manifest_path]
    }
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("V10 archive entries differ from the release manifest")
        for member in members:
            if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError(f"unsafe or non-regular V10 archive member: {member.name}")
            expected_hash, expected_bytes = expected[member.name]
            extracted = bundle.extractfile(member)
            if extracted is None or member.size != expected_bytes or _sha256_stream(extracted) != expected_hash:
                raise ValueError(f"V10 archive member verification failed: {member.name}")


def package_release(root: Path) -> dict[str, object]:
    root = root.resolve()
    files = release_files(root)
    validation = _validate_release_evidence(root)
    output_dir = root / "results/release"; output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "live_validation_v10_release_manifest.json"
    manifest = {
        "schema_version": 2,
        "campaign": CAMPAIGN,
        "accepted_unique_runs": EXPECTED_RUNS,
        "accepted_windows": EXPECTED_WINDOWS,
        "windows_per_run": WINDOWS_PER_RUN,
        "matched_blocks": EXPECTED_METHOD_RUNS,
        "methods": {method: EXPECTED_METHOD_RUNS for method in ("B0", "B2", "B5", "P")},
        "evidence_gate_status": validation["gate"]["status"],
        "traceability_status": validation["traceability"]["status"],
        "raw_evidence_manifest_sha256": validation["raw_manifest_sha256"],
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = output_dir / "live_validation_v10_final_release.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in [*files, manifest_path]:
            bundle.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    _verify_archive(archive, root, files, manifest_path)
    archive_hash = _sha256(archive)
    hash_path = archive.with_suffix(archive.suffix + ".sha256")
    hash_path.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")
    result = {
        "status": "PASS",
        "campaign": CAMPAIGN,
        "accepted_unique_runs": EXPECTED_RUNS,
        "accepted_windows": EXPECTED_WINDOWS,
        "evidence_gate_status": validation["gate"]["status"],
        "files": len(files),
        "release_manifest_sha256": _sha256(manifest_path),
        "archive": str(archive),
        "archive_sha256": archive_hash,
        "archive_bytes": archive.stat().st_size,
    }
    (output_dir / "live_validation_v10_packaging_status.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    print(json.dumps(package_release(project_root), indent=2))
