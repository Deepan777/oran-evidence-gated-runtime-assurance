from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject non-mapping documents."""
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved}")
    return value


def load_methodology_lock(project_root: str | Path) -> dict[str, Any]:
    return load_yaml(Path(project_root) / "METHODOLOGY_LOCK.yaml")


def state_feature_names(lock: dict[str, Any]) -> tuple[str, ...]:
    features = lock["state"]["features_in_exact_order"]
    ordered = sorted(features, key=lambda item: int(item["index"]))
    indexes = [int(item["index"]) for item in ordered]
    if indexes != list(range(len(indexes))):
        raise ValueError("State feature indexes must be contiguous and zero-based")
    names = tuple(str(item["name"]) for item in ordered)
    if len(names) != len(set(names)):
        raise ValueError("State feature names must be unique")
    return names

