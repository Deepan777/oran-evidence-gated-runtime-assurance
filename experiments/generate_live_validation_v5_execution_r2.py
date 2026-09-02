#!/usr/bin/env python3
"""Derive V5 execution revision R2 without changing any scientific factor."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import yaml


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=Path("LIVE_VALIDATION_FREEZE_MANIFEST_V5.yaml"))
    parser.add_argument("--output", type=Path, default=Path("LIVE_VALIDATION_EXECUTION_MANIFEST_V5_R2.yaml"))
    args = parser.parse_args()

    parent = yaml.safe_load(args.parent.read_text(encoding="utf-8"))
    if parent.get("status") != "LIVE_VALIDATION_FROZEN_V5":
        raise SystemExit("parent is not the frozen V5 campaign")
    revision = copy.deepcopy(parent)
    for entry in revision["execution_plan"]:
        command = entry["command"]
        index = command.index("--max-wall-seconds") + 1
        if str(command[index]) != "900":
            raise SystemExit(f"unexpected parent wall cap for {entry['method']}")
        command[index] = "1800"
    revision["execution_revision"] = "V5_INFRASTRUCTURE_R2"
    revision["scientific_parent_freeze"] = str(args.parent)
    revision["scientific_parent_freeze_sha256"] = digest(args.parent)
    revision["wall_safety_cap_seconds"] = 1800
    revision["authoritative_stop_condition"] = "exact_forwarded_radio_sample_count"
    revision["scenario_factors_changed"] = False
    revision["controller_schema_changed"] = False
    revision["radio_profile_changed"] = False
    revision["acceptance_rules_changed"] = False
    revision["reason"] = (
        "Two consecutive immutable B0 rows independently made 83.5% real-IQ "
        "progress before the 900-second absolute wall safety cap; R2 increases "
        "only that infrastructure cap for every remaining method-neutral run."
    )
    args.output.write_text(yaml.safe_dump(revision, sort_keys=False), encoding="utf-8")
    print(f"LIVE_VALIDATION_V5_EXECUTION_R2_CREATED={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
