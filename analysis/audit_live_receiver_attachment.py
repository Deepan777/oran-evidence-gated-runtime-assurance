"""Audit live decisions against the native srsUE attachment flag.

This is read-only diagnostics.  It identifies controller decisions whose
receiver feature source points at a finite srsUE row that was nevertheless
reported as detached by the same native CSV envelope.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _attachment_by_timestamp(path: Path) -> dict[int, float]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter=";")
        result: dict[int, float] = {}
        for row in rows:
            try:
                timestamp_ns = round(float(row["timestamp_epoch_ms"]) * 1_000_000.0)
                result[timestamp_ns] = float(row["is_attached"])
            except (KeyError, TypeError, ValueError):
                continue
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--prefix", default="iq-broker-live-livev8-b2-")
    args = parser.parse_args()

    totals: Counter[str] = Counter()
    affected_runs: dict[str, int] = {}
    for evidence in sorted(args.root.glob(f"{args.prefix}*")):
        decisions_path = evidence / "live-xapp-decisions.jsonl"
        if not decisions_path.is_file():
            continue
        attachment = _attachment_by_timestamp(evidence / "srsue_metrics.csv")
        detached_sources = 0
        rows = 0
        for line in decisions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            decision = json.loads(line)
            rows += 1
            timestamp = decision.get("source_timestamps_ns", {}).get("srsue")
            if timestamp is None:
                totals["missing_receiver_source"] += 1
            elif float(attachment.get(int(timestamp), 1.0)) < 0.5:
                detached_sources += 1
        totals["runs"] += 1
        totals["decision_rows"] += rows
        totals["detached_receiver_source_rows"] += detached_sources
        if detached_sources:
            affected_runs[evidence.name] = detached_sources

    document = {"totals": dict(totals), "affected_runs": affected_runs}
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
