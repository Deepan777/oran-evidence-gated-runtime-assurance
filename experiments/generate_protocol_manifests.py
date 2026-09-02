"""Generate immutable run-level pilot, development, test and holdout manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.splits import RunDescriptor, create_run_level_split


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _jsr_values(jammer_type: str, values: Iterable[int]) -> tuple[int | None, ...]:
    return (None,) if jammer_type == "none" else tuple(values)


def _is_holdout(row: dict[str, Any], holdouts: list[dict[str, Any]]) -> bool:
    return any(all(row.get(key) == value for key, value in definition.items()) for definition in holdouts)


def _rows(config: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    scope = config if phase == "confirmatory" else {**config, **config["pilot"]}
    phases = scope.get("phases_seconds", config["phases_seconds"])
    rows: list[dict[str, Any]] = []
    for jammer_type in scope["jammer_types"]:
        for jsr_db, load, mix, attenuation, seed in itertools.product(
            _jsr_values(jammer_type, scope["jsr_db"]),
            scope["traffic_load"],
            scope["service_mix"],
            scope["attenuation"],
            scope["seeds"],
        ):
            factors = {
                "jammer_type": jammer_type,
                "jsr_db": jsr_db,
                "traffic_load": load,
                "service_mix": mix,
                "attenuation": attenuation,
                "seed": seed,
            }
            scenario_id = "-".join(
                (jammer_type, "jsrNA" if jsr_db is None else f"jsr{jsr_db:+d}", load, mix, attenuation)
            ).replace("+", "p").replace("-", "m")
            run_id = f"{phase}-{scenario_id}-s{seed}"
            rows.append({
                "run_id": run_id,
                "scenario_id": scenario_id,
                **factors,
                "run_length_seconds": scope.get("run_length_seconds", config["run_length_seconds"]),
                "pre_jam_seconds": phases["pre_jam"],
                "jammer_on_seconds": phases["jammer_on"],
                "recovery_seconds": phases["recovery"],
                "evidence_class": "development_pilot" if phase == "pilot" else "locked_confirmatory",
                "generalization_holdout": phase == "confirmatory" and _is_holdout(factors, config["generalization_holdouts"]),
            })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/protocol.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets/metadata/manifests")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pilot = _rows(config, "pilot")
    confirmatory = _rows(config, "confirmatory")
    split = create_run_level_split(
        (RunDescriptor(row["run_id"], row["scenario_id"], row["seed"], row["generalization_holdout"])
         for row in confirmatory),
        split_seed=1001,
    )
    partition_by_run = {
        **{run_id: "train" for run_id in split.train},
        **{run_id: "validation" for run_id in split.validation},
        **{run_id: "locked_test" for run_id in split.locked_test},
        **{run_id: "generalization" for run_id in split.generalization},
    }
    for row in confirmatory:
        row["partition"] = partition_by_run[row["run_id"]]
    for row in pilot:
        row["partition"] = "pilot"
    _write_csv(args.output / "pilot_runs.csv", pilot)
    _write_csv(args.output / "confirmatory_runs.csv", confirmatory)
    summary = {
        "schema_version": 1,
        "protocol_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "pilot_runs": len(pilot),
        "confirmatory_runs": len(confirmatory),
        "partitions": {name: len(getattr(split, name)) for name in ("train", "validation", "locked_test", "generalization")},
        "pilot_manifest_sha256": _canonical_hash(pilot),
        "confirmatory_manifest_sha256": _canonical_hash(confirmatory),
    }
    (args.output / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
