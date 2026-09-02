#!/usr/bin/env python3
"""Create the v11 recovery manifest after the v10 srsUE transient failure.

The v10 radio clock reached its exact target, but srsUE terminated before one
controller-state window was complete.  v11 replays only the nine unaccepted
v10 rows with the same bounded 5-second UE response tolerance.
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
    source = _read(ROOT / "datasets/metadata/manifests/live_calibration_recovery_v10_retry_runs.csv")
    rows: list[dict[str, object]] = []
    for item in source:
        row = dict(item)
        row["run_id"] = str(item["run_id"]).replace("livecalv10-", "livecalv11-", 1)
        row["recovery_protocol"] = "v11_srsue_transient_retry_ue_timeout_5000ms"
        row["recovery_of_or_supersedes_v10_run_id"] = item["run_id"]
        row["UE_TIMEOUT_MS"] = "5000"
        rows.append(row)
    if len(rows) != 9:
        raise ValueError(f"expected 9 v11 retry rows, got {len(rows)}")
    return rows


def make_combined_rows() -> list[dict[str, object]]:
    base = _read(ROOT / "datasets/metadata/manifests/live_calibration_recovery_v10_combined_runs.csv")
    # Replace only the v10 tail; preserve all previously accepted evidence.
    rows = [row for row in base if not str(row["run_id"]).startswith("livecalv10-")]
    rows += make_retry_rows()
    if len(rows) != 30 or any(str(row["run_id"]).startswith("livecalv10-") for row in rows):
        raise ValueError("invalid v11 combined manifest")
    return rows


def _write(path: Path, rows: list[dict[str, object]], status: str) -> None:
    fields = list(rows[0]) + sorted({key for row in rows for key in row} - set(rows[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    path.with_suffix(".summary.json").write_text(json.dumps({
        "status": status, "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/metadata/manifests")
    args = parser.parse_args()
    retry, combined = make_retry_rows(), make_combined_rows()
    _write(args.output_dir / "live_calibration_recovery_v11_retry_runs.csv", retry, "DEVELOPMENT_CALIBRATION_RECOVERY_V11_RETRY_CREATED")
    _write(args.output_dir / "live_calibration_recovery_v11_combined_runs.csv", combined, "DEVELOPMENT_CALIBRATION_RECOVERY_V11_COMBINED_CREATED")
    print(json.dumps({"retry_rows": len(retry), "combined_rows": len(combined)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
