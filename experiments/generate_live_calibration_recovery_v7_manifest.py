#!/usr/bin/env python3
"""Create the e2-standard-8 live-calibration recovery manifests."""
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
from experiments.generate_live_calibration_recovery_v4_manifest import make_retry_rows as v4_rows
from experiments.generate_live_calibration_recovery_v6_manifest import make_retry_rows as v6_rows

V3_COMPLETE = {
    "livecalv3-partial_bandmjsrp0mlowmprotected_onlymnominal-s1301",
    "livecalv3-partial_bandmjsrp5mlowmprotected_onlymnominal-s1302",
}
V4_COMPLETE = {
    "livecalv4-partial_bandmjsrp0mlowmprotected_onlymnominal-s1303",
    "livecalv4-partial_bandmjsrp5mlowmprotected_onlymnominal-s1304",
}


def make_retry_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in v6_rows():
        row = dict(source)
        old = str(row["run_id"])
        row["run_id"] = old.replace("livecalv6-", "livecalv7-", 1)
        row["recovery_protocol"] = "v7_e2_standard_8_cpu_capacity"
        row["recovery_of_or_supersedes_v6_run_id"] = old
        rows.append(row)
    if len(rows) != 16:
        raise ValueError(f"expected 16 v7 retry rows, got {len(rows)}")
    return rows


def make_combined_rows() -> list[dict[str, object]]:
    rows = [r for r in v2_rows() if r["jammer_type"] in {"none", "barrage"}]
    rows += [r for r in v3_rows() if r["run_id"] in V3_COMPLETE]
    rows += [r for r in v4_rows() if r["run_id"] in V4_COMPLETE]
    rows += make_retry_rows()
    if len(rows) != 30:
        raise ValueError(f"expected 30 rows, got {len(rows)}")
    return rows


def _write(path: Path, rows: list[dict[str, object]], status: str) -> None:
    fields = list(rows[0]) + sorted({k for r in rows for k in r} - set(rows[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "status": status,
                "rows": len(rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/metadata/manifests")
    args = parser.parse_args()
    retry = make_retry_rows()
    combined = make_combined_rows()
    _write(args.output_dir / "live_calibration_recovery_v7_retry_runs.csv", retry, "DEVELOPMENT_CALIBRATION_RECOVERY_V7_RETRY_CREATED")
    _write(args.output_dir / "live_calibration_recovery_v7_combined_runs.csv", combined, "DEVELOPMENT_CALIBRATION_RECOVERY_V7_COMBINED_CREATED")
    print(json.dumps({"retry_rows": len(retry), "combined_rows": len(combined)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
