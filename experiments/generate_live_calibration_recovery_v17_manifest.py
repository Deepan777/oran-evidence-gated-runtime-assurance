#!/usr/bin/env python3
"""Create the v17 retry with RR priority and fixed UE CPU affinity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V16_ACCEPTED = {
    "livecalv16-intermittentmjsrp5mlowmprotected_onlymnominal-s1302",
    "livecalv16-intermittentmjsrp0mlowmprotected_onlymnominal-s1303",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_retry_rows() -> list[dict[str, object]]:
    rows = []
    for item in _read(ROOT / "datasets/metadata/manifests/live_calibration_recovery_v16_retry_runs.csv"):
        if str(item["run_id"]) in V16_ACCEPTED:
            continue
        row = dict(item)
        row["run_id"] = str(item["run_id"]).replace("livecalv16-", "livecalv17-", 1)
        row["recovery_protocol"] = "v17_two_cpu_srsue_realtime_fixed_affinity_ue_timeout_1000ms"
        row["recovery_of_or_supersedes_v16_run_id"] = item["run_id"]
        row["UE_TIMEOUT_MS"] = "1000"
        rows.append(row)
    if len(rows) != 7:
        raise ValueError(f"expected 7 v17 retry rows, got {len(rows)}")
    return rows


def make_combined_rows() -> list[dict[str, object]]:
    base = _read(ROOT / "datasets/metadata/manifests/live_calibration_recovery_v16_combined_runs.csv")
    rows = [row for row in base if not str(row["run_id"]).startswith("livecalv16-")]
    rows += [row for row in _read(ROOT / "datasets/metadata/manifests/live_calibration_recovery_v16_retry_runs.csv") if str(row["run_id"]) in V16_ACCEPTED]
    rows += make_retry_rows()
    if len(rows) != 30 or any(str(row["run_id"]).startswith("livecalv16-") and str(row["run_id"]) not in V16_ACCEPTED for row in rows):
        raise ValueError("invalid v17 combined manifest")
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
    _write(args.output_dir / "live_calibration_recovery_v17_retry_runs.csv", retry, "DEVELOPMENT_CALIBRATION_RECOVERY_V17_RETRY_CREATED")
    _write(args.output_dir / "live_calibration_recovery_v17_combined_runs.csv", combined, "DEVELOPMENT_CALIBRATION_RECOVERY_V17_COMBINED_CREATED")
    print(json.dumps({"retry_rows": len(retry), "combined_rows": len(combined)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
