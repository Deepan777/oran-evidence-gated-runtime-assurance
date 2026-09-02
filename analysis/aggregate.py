from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


METHODS = ("B0", "B1", "B2", "B3", "B4", "B5", "P")


def _resolve(value: object, manifest_path: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    # Confirmatory runners normally emit project-root-relative paths, while
    # hand-written development manifests often use paths relative to the
    # manifest itself. Prefer the latter when it exists.
    local = manifest_path.parent / path
    return local if local.is_file() else path


def source_entries(manifest: dict[str, Any], manifest_path: Path) -> list[tuple[str, Path]]:
    """Return (method, parquet) pairs while preserving confirmatory labels."""
    methods = manifest.get("methods")
    entries: list[tuple[str, Path]] = []
    if isinstance(methods, dict):
        if tuple(methods) != METHODS:
            raise ValueError("confirmatory manifest methods do not match the locked baseline set")
        for method in METHODS:
            values = methods[method]
            if not isinstance(values, list) or not values:
                raise ValueError(f"confirmatory manifest has no records for {method}")
            entries.extend((method, _resolve(value, manifest_path)) for value in values)
        return entries
    # A flat manifest cannot support paired multi-method statistics. It is
    # retained only for pre-confirmatory development collection.
    values = manifest.get("record_files")
    if not isinstance(values, list) or not values:
        raise ValueError("record manifest has no record_files")
    return [("UNSPECIFIED_DEVELOPMENT", _resolve(value, manifest_path)) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/raw/CONFIRMATORY_MANIFEST.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/aggregate/confirmatory_records.parquet"),
    )
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise SystemExit(f"BLOCKED: confirmatory manifest is missing: {args.manifest}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = source_entries(manifest, args.manifest)
    if any(not path.is_file() for _, path in entries):
        raise SystemExit("BLOCKED: confirmatory record files are absent or incomplete")
    frames: list[pd.DataFrame] = []
    for method, path in entries:
        frame = pd.read_parquet(path).copy()
        if "run_id" not in frame or "window_index" not in frame:
            raise SystemExit(f"BLOCKED: record schema missing run/window identifiers: {path}")
        frame.insert(0, "evaluation_method", method)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["evaluation_method", "run_id", "window_index"]).any():
        raise SystemExit("BLOCKED: duplicate method/run/window record found during aggregation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    producer = {
        "producer": "analysis/aggregate.py",
        "sources": [{"method": method, "path": str(path)} for method, path in entries],
        "output": str(args.output),
        "rows": int(len(combined)),
    }
    args.output.with_suffix(".producer.json").write_text(
        json.dumps(producer, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
