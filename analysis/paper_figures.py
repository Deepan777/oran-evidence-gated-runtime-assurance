from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("results/aggregate/figure_source_latency.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/latency_summary.png"),
    )
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"BLOCKED: machine-readable figure source is missing: {args.source}")
    frame = pd.read_csv(args.source)
    required = {"component", "median_ms", "p95_ms", "p99_ms"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"BLOCKED: figure source lacks required columns: {sorted(required)}")
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    frame.plot(
        x="component",
        y=["median_ms", "p95_ms", "p99_ms"],
        kind="bar",
        ax=axis,
    )
    axis.set_ylabel("Latency (ms)")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=35)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    plt.close(figure)
    args.output.with_suffix(".producer.json").write_text(
        json.dumps(
            {"producer": "analysis/paper_figures.py", "source": str(args.source)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

