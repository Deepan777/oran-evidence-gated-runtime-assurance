from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, default=Path("results/statistics/stats_report.json"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/statistics_summary.csv"))
    args = parser.parse_args()
    if not args.stats.is_file():
        raise SystemExit(f"BLOCKED: statistical report is missing: {args.stats}")
    payload = json.loads(args.stats.read_text(encoding="utf-8"))
    rows = payload.get("manuscript_table_rows")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("BLOCKED: stats report contains no machine-generated manuscript table rows")
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    args.output.with_suffix(".producer.json").write_text(
        json.dumps(
            {"producer": "analysis/paper_tables.py", "source": str(args.stats)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

