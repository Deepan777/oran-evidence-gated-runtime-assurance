#!/usr/bin/env python3
"""Create the v13 retry after the v12 launcher syntax failure.

v12 did not start a UE container because a shell comment interrupted a Docker
command; this version records the corrected launcher and retries the same nine
unaccepted rows with two UE cores and a 1-second UE timeout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_retry_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in _read(ROOT / "datasets/metadata/manifests/live_calibration_recovery_v12_retry_runs.csv"):
        row = dict(item)
        row["run_id"] = str(item["run_id"]).replace("livecalv12-", "livecalv13-", 1)
        row["recovery_protocol"] = "v13_two_cpu_srsue_shellfix_ue_timeout_1000ms"
        row["recovery_of_or_supersedes_v12_run_id"] = item["run_id"]
        row["UE_TIMEOUT_MS"] = "1000"
        rows.append(row)
    if len(rows) != 9:
        raise ValueError(f"expected 9 v13 retry rows, got {len(rows)}")
    return rows


def make_combined_rows() -> list[dict[str, object]]:
    base = _read(ROOT / "datasets/metadata/manifests/live_calibration_recovery_v12_combined_runs.csv")
    rows = [row for row in base if not str(row["run_id"]).startswith("livecalv12-")]
    rows += make_retry_rows()
    if len(rows) != 30 or any(str(row["run_id"]).startswith("livecalv12-") for row in rows):
        raise ValueError("invalid v13 combined manifest")
    return rows


def _write(path: Path, rows: list[dict[str, object]], status: str) -> None:
    fields = list(rows[0]) + sorted({key for row in rows for key in row} - set(rows[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    path.with_suffix(".summary.json").write_text(json.dumps({"status": status, "rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/metadata/manifests")
    args = parser.parse_args()
    retry, combined = make_retry_rows(), make_combined_rows()
    _write(args.output_dir / "live_calibration_recovery_v13_retry_runs.csv", retry, "DEVELOPMENT_CALIBRATION_RECOVERY_V13_RETRY_CREATED")
    _write(args.output_dir / "live_calibration_recovery_v13_combined_runs.csv", combined, "DEVELOPMENT_CALIBRATION_RECOVERY_V13_COMBINED_CREATED")
    print(json.dumps({"retry_rows": len(retry), "combined_rows": len(combined)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
