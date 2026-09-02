#!/usr/bin/env python3
"""Versioned retry after the v3 virtual-radio timing failure.

The v3 partial-band seed 1303 record is incomplete and remains preserved.
Every unattempted v3 row receives a new v4 run ID too, keeping the retry
sequence immutable and making the combined detector/state manifest explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.generate_live_calibration_recovery_manifest import make_rows as v2_rows
from experiments.generate_live_calibration_recovery_v3_manifest import make_retry_rows as v3_rows


V3_COMPLETE = {
    "livecalv3-partial_bandmjsrp0mlowmprotected_onlymnominal-s1301",
    "livecalv3-partial_bandmjsrp5mlowmprotected_onlymnominal-s1302",
}
V3_FAILED = "livecalv3-partial_bandmjsrp0mlowmprotected_onlymnominal-s1303"


def make_retry_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in v3_rows():
        source_id = str(source["run_id"])
        if source_id in V3_COMPLETE:
            continue
        row = dict(source)
        row["run_id"] = source_id.replace("livecalv3-", "livecalv4-", 1)
        row["recovery_protocol"] = "v4_rt_scheduling_and_ue_memlock"
        if source_id == V3_FAILED:
            row["recovery_of_run_id"] = source_id
            row["objective_failure"] = "UE_RRC_RELEASE_AFTER_ZMQ_TX_DEADLINE_MISSES"
        else:
            row["supersedes_unattempted_run_id"] = source_id
        rows.append(row)
    return rows


def make_combined_rows() -> list[dict[str, object]]:
    complete_v2 = [row for row in v2_rows() if row["jammer_type"] in {"none", "barrage"}]
    complete_v3 = [row for row in v3_rows() if row["run_id"] in V3_COMPLETE]
    combined = [*complete_v2, *complete_v3, *make_retry_rows()]
    if len(combined) != 30:
        raise ValueError(f"v4 combined calibration must have 30 rows, got {len(combined)}")
    return combined


def _write(path: Path, rows: list[dict[str, object]], status: str) -> None:
    fields = list(rows[0]) + sorted({key for row in rows for key in row} - set(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    path.with_suffix(".summary.json").write_text(json.dumps({
        "status": status, "rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "file": str(path),
    }, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/metadata/manifests")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    retry, combined = make_retry_rows(), make_combined_rows()
    retry_path = args.output_dir / "live_calibration_recovery_v4_retry_runs.csv"
    combined_path = args.output_dir / "live_calibration_recovery_v4_combined_runs.csv"
    _write(retry_path, retry, "DEVELOPMENT_CALIBRATION_RECOVERY_V4_RETRY_CREATED")
    _write(combined_path, combined, "DEVELOPMENT_CALIBRATION_RECOVERY_V4_COMBINED_CREATED")
    print(json.dumps({"retry_rows": len(retry), "combined_rows": len(combined), "retry": str(retry_path), "combined": str(combined_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
