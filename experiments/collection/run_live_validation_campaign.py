#!/usr/bin/env python3
"""Run a versioned 360-execution live campaign sequentially and resumably."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=ROOT / "LIVE_VALIDATION_FREEZE_MANIFEST_V3.yaml")
    parser.add_argument("--progress", type=Path, default=ROOT / "results/raw/LIVE_VALIDATION_V3_PROGRESS.json")
    args = parser.parse_args()

    freeze: dict[str, Any] = yaml.safe_load(args.freeze.read_text(encoding="utf-8"))
    freeze_status = str(freeze.get("status", ""))
    if freeze_status not in {
        "LIVE_VALIDATION_FROZEN_V3",
        "LIVE_VALIDATION_FROZEN_V4",
        "LIVE_VALIDATION_FROZEN_V5",
        "LIVE_VALIDATION_FROZEN_V6",
        "LIVE_VALIDATION_FROZEN_V7",
        "LIVE_VALIDATION_FROZEN_V8",
        "LIVE_VALIDATION_FROZEN_V9",
        "LIVE_VALIDATION_FROZEN_V10",
    }:
        raise SystemExit("refusing to run an unsupported or unfrozen campaign")
    campaign_tag = freeze_status.rsplit("_", 1)[-1]
    if freeze.get("total_method_executions") != 360 or freeze.get("runs_per_method") != 90:
        raise SystemExit("frozen campaign cardinality is not 4 x 90")

    lock_path = ROOT / f"results/raw/.live-validation-{campaign_tag.lower()}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"another {campaign_tag} campaign coordinator already holds the lock") from exc

    artifacts = freeze["artifacts"]
    for method in freeze["methods"]:
        artifact = artifacts[f"{method.lower()}_{campaign_tag.lower()}_manifest"]
        manifest = ROOT / artifact["path"]
        if digest(manifest) != artifact["sha256"]:
            raise SystemExit(f"{method}: {campaign_tag} manifest digest mismatch")

    progress: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "freeze_manifest": str(args.freeze),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "methods": {},
    }
    atomic_json(args.progress, progress)
    logs = ROOT / f"results/raw/live_validation_{campaign_tag.lower()}_logs"
    logs.mkdir(parents=True, exist_ok=True)
    all_records: list[str] = []
    for entry in freeze["execution_plan"]:
        method = entry["method"]
        result_path = ROOT / entry["result_manifest"]
        progress["methods"][method] = {"status": "RUNNING", "started_utc": utc_now()}
        progress["updated_utc"] = utc_now()
        atomic_json(args.progress, progress)
        with (logs / f"{method}.log").open("a", encoding="utf-8") as log:
            log.write(f"\n=== campaign invocation {utc_now()} ===\n")
            log.flush()
            completed = subprocess.run(entry["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            progress["status"] = "FAILED"
            progress["failed_method"] = method
            progress["failure_exit_code"] = completed.returncode
            progress["methods"][method].update({"status": "FAILED", "exit_code": completed.returncode, "updated_utc": utc_now()})
            progress["updated_utc"] = utc_now()
            atomic_json(args.progress, progress)
            return completed.returncode
        result = json.loads(result_path.read_text(encoding="utf-8"))
        records = result.get("record_files", [])
        if len(records) != entry["expected_runs"] or result.get("remaining") != 0:
            progress["status"] = "FAILED"
            progress["failed_method"] = method
            progress["failure_reason"] = "result manifest cardinality mismatch"
            progress["methods"][method].update({"status": "FAILED", "accepted": len(records), "updated_utc": utc_now()})
            progress["updated_utc"] = utc_now()
            atomic_json(args.progress, progress)
            return 2
        all_records.extend(records)
        progress["methods"][method].update({"status": "COMPLETE", "accepted": len(records), "completed_utc": utc_now()})
        progress["accepted_total"] = len(all_records)
        progress["updated_utc"] = utc_now()
        atomic_json(args.progress, progress)

    if len(all_records) != 360 or len(set(all_records)) != 360:
        raise SystemExit("combined campaign evidence is not 360 unique record files")
    combined = {
        "schema_version": 1,
        "status": "COMPLETE",
        "freeze_manifest": str(args.freeze),
        "record_files": all_records,
        "accepted_count": len(all_records),
        "completed_utc": utc_now(),
    }
    atomic_json(ROOT / f"results/raw/LIVE_VALIDATION_{campaign_tag}_ALL.json", combined)
    progress["status"] = "COMPLETE"
    progress["accepted_total"] = 360
    progress["completed_utc"] = utc_now()
    progress["updated_utc"] = utc_now()
    atomic_json(args.progress, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
