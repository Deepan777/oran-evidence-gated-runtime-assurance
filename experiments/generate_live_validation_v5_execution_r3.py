#!/usr/bin/env python3
"""Derive V5 R3 orchestration without changing a scientific factor."""
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
    parser.add_argument("--parent", type=Path, default=Path("LIVE_VALIDATION_EXECUTION_MANIFEST_V5_R2.yaml"))
    parser.add_argument("--output", type=Path, default=Path("LIVE_VALIDATION_EXECUTION_MANIFEST_V5_R3.yaml"))
    args = parser.parse_args()
    parent = yaml.safe_load(args.parent.read_text(encoding="utf-8"))
    if parent.get("status") != "LIVE_VALIDATION_FROZEN_V5":
        raise SystemExit("parent is not frozen V5")
    if parent.get("execution_revision") != "V5_INFRASTRUCTURE_R2":
        raise SystemExit("parent is not execution R2")
    revision = copy.deepcopy(parent)
    revision["execution_revision"] = "V5_INFRASTRUCTURE_R3"
    revision["infrastructure_parent"] = str(args.parent)
    revision["infrastructure_parent_sha256"] = digest(args.parent)
    revision["retry_orchestration"] = {
        "scope": "radio_infrastructure_failures_only",
        "eligible_failure_classes": [
            "srsue_signal_11_uplink_termination",
            "radio_clock_absolute_wall_timeout",
        ],
        "max_retries_per_immutable_row": 3,
        "max_campaign_retries": 24,
        "archive_every_rejected_attempt": True,
        "refuse_controller_ood_missing_decision_or_unacknowledged_action": True,
    }
    revision["scenario_factors_changed"] = False
    revision["controller_schema_changed"] = False
    revision["radio_profile_changed"] = False
    revision["acceptance_rules_changed"] = False
    revision["reason_r3"] = (
        "A real srsUE process terminated with SIGSEGV after 100 of 120 radio "
        "seconds while same-factor seeds completed. R3 adds only bounded, "
        "audited retries of identical immutable rows for narrowly classified "
        "radio-infrastructure failures."
    )
    args.output.write_text(yaml.safe_dump(revision, sort_keys=False), encoding="utf-8")
    print(f"LIVE_VALIDATION_V5_EXECUTION_R3_CREATED={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
