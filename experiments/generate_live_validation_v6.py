#!/usr/bin/env python3
"""Freeze V6 after a method-neutral srsUE RLC-UM runtime correction.

V5 exposed reproducible depletion of the upstream fixed byte-buffer pool during
long, strongly impaired RLC-UM operation. V6 preserves every scientific row,
controller artifact, radio profile, and acceptance rule, but uses a pinned
runtime with null-allocation guards and a 16,384-block pool. All V6 executions
receive fresh IDs so evidence produced with the old runtime is never mixed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("B0", "B2", "B5", "P")
UE_IMAGE = "srsue-zmq:git-6bcbd9e5bf86-telemetry-v2-iperf3-buffer16384"
SRSRAN_COMMIT = "6bcbd9e5bf8686aa7085202cd847c5ddd64a9c16"
PATCH_SHA256 = "c99126d7f74dda5dff799b1a0f4639a4e00f890c9dec114dbbbfb48287c2693d"
IMAGE_ID = "sha256:d752a75c5dca2718ef2e9abd4427c87fd4f0cc6a581dc5663d517f60e097d72c"
BINARY_SHA256 = "34bad939de7ad48e60b272d3d5b318c53b70f1cbae8a274cffeaa183b7615f05"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_rows(source: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for original in source:
        row = dict(original)
        source_run_id = row.get("source_run_id") or row["run_id"]
        row["source_run_id"] = source_run_id
        row["run_id"] = f"livev6-{method.lower()}-{source_run_id}"
        row["UE_IMAGE"] = UE_IMAGE
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "datasets/metadata/manifests/live_validation_v5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/metadata/manifests/live_validation_v6")
    parser.add_argument("--parent-freeze", type=Path, default=ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V5.yaml")
    parser.add_argument("--output-freeze", type=Path, default=ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V6.yaml")
    args = parser.parse_args()

    document: dict[str, Any] = yaml.safe_load(args.parent_freeze.read_text(encoding="utf-8"))
    source_ids: set[str] | None = None
    execution_ids: set[str] = set()
    execution_plan: list[dict[str, Any]] = []
    artifacts = dict(document["artifacts"])
    artifacts["v6_srsue_source_patch"] = {
        "path": "patches/srsran4g-v6-rlc-um-buffer-safety.patch",
        "sha256": PATCH_SHA256,
    }
    for method in METHODS:
        source = args.input_dir / f"{method}_live_validation_runs.csv"
        rows = make_rows(read_rows(source), method)
        identities = {row["source_run_id"] for row in rows}
        if source_ids is None:
            source_ids = identities
        elif identities != source_ids:
            raise SystemExit(f"{method}: scenario identity set differs")
        if len(rows) != 90 or len(identities) != 90:
            raise SystemExit(f"{method}: expected 90 unique frozen scenarios")
        method_ids = {row["run_id"] for row in rows}
        if execution_ids.intersection(method_ids):
            raise SystemExit(f"{method}: duplicate V6 execution IDs")
        execution_ids.update(method_ids)
        output = args.output_dir / f"{method}_live_validation_runs.csv"
        write_rows(output, rows)
        relative = output.relative_to(ROOT).as_posix()
        artifacts[f"{method.lower()}_v6_manifest"] = {"path": relative, "sha256": digest(output)}
        result = f"results/raw/LIVE_VALIDATION_V6_{method}.json"
        execution_plan.append({
            "method": method,
            "scenario_manifest": relative,
            "result_manifest": result,
            "expected_runs": 90,
            "command": [
                "envs/oran-action-xai/bin/python", "experiments/collection/run_manifest.py",
                "--manifest", relative, "--output", result,
                "--max-wall-seconds", "1800",
                "--gnb-image", "ocudu-gnb-zmq:research-actions-fa4c9fa445a3",
                "--ue-image", UE_IMAGE,
                "--gnb-config", "ocudu_gnb_zmq_e2_udp_um.yml",
            ],
        })

    document.update({
        "schema_version": 6,
        "status": "LIVE_VALIDATION_FROZEN_V6",
        "parent_freeze_manifest": args.parent_freeze.name,
        "artifacts": artifacts,
        "execution_plan": execution_plan,
        "execution_revision": "V6_RUNTIME_R1",
        "runtime_correction": {
            "scope": "method_neutral_srsue_rlc_um_memory_safety",
            "srsran_commit": SRSRAN_COMMIT,
            "source_patch_sha256": PATCH_SHA256,
            "ue_image": UE_IMAGE,
            "ue_image_id": IMAGE_ID,
            "srsue_binary_sha256": BINARY_SHA256,
            "byte_buffer_pool_blocks": 16384,
            "allocation_failure_behavior": "drop_and_count_rlc_pdu_without_null_dereference",
            "scenario_factors_changed": False,
            "controllers_changed": False,
            "radio_profile_changed": False,
            "acceptance_rules_changed": False,
            "rejected_v3_v4_v5_evidence_retained": True,
        },
        "notes": (
            "V6 changes only the pinned method-neutral srsUE runtime. Fresh run IDs "
            "prevent V5 runtime evidence from entering the V6 analysis set."
        ),
    })
    args.output_freeze.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(document, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
