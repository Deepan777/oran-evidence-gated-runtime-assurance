from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_FIELDS = (
    "run_id",
    "git_commit",
    "methodology_version",
    "config_hash",
    "dataset_version",
    "split_version",
    "seed",
    "timestamp_utc",
    "environment_fingerprint",
    "machine",
    "gpu",
    "scenario_id",
    "raw_result_path",
)


@dataclass(frozen=True)
class ExperimentRecord:
    run_id: str
    git_commit: str
    methodology_version: str
    config_hash: str
    dataset_version: str
    split_version: str
    seed: int
    timestamp_utc: str
    environment_fingerprint: str
    machine: str
    gpu: str
    scenario_id: str
    raw_result_path: str


def make_run_id(timestamp_utc: str, scenario_id: str, seed: int, git_commit: str) -> str:
    material = f"{timestamp_utc}|{scenario_id}|{seed}|{git_commit}".encode("utf-8")
    suffix = hashlib.sha256(material).hexdigest()[:12]
    compact_time = timestamp_utc.replace("-", "").replace(":", "").replace("+00:00", "Z")
    return f"{compact_time}_{scenario_id}_s{seed}_{suffix}"


def new_record(
    *,
    git_commit: str,
    methodology_version: str,
    config_hash: str,
    dataset_version: str,
    split_version: str,
    seed: int,
    environment_fingerprint: str,
    machine: str,
    gpu: str,
    scenario_id: str,
    raw_result_path: str,
    timestamp: datetime | None = None,
) -> ExperimentRecord:
    instant = timestamp or datetime.now(timezone.utc)
    timestamp_utc = instant.astimezone(timezone.utc).isoformat()
    return ExperimentRecord(
        run_id=make_run_id(timestamp_utc, scenario_id, seed, git_commit),
        git_commit=git_commit,
        methodology_version=methodology_version,
        config_hash=config_hash,
        dataset_version=dataset_version,
        split_version=split_version,
        seed=seed,
        timestamp_utc=timestamp_utc,
        environment_fingerprint=environment_fingerprint,
        machine=machine,
        gpu=gpu,
        scenario_id=scenario_id,
        raw_result_path=raw_result_path,
    )


def append_record(path: str | Path, record: ExperimentRecord) -> None:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    already_exists = registry_path.exists()
    if already_exists:
        with registry_path.open("r", newline="", encoding="utf-8") as handle:
            existing_ids = {row["run_id"] for row in csv.DictReader(handle)}
        if record.run_id in existing_ids:
            raise ValueError(f"run_id already registered: {record.run_id}")
    with registry_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if not already_exists:
            writer.writeheader()
        writer.writerow(asdict(record))

