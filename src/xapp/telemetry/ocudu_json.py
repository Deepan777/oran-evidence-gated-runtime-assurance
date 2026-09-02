"""Strict parser for OCUDU JSON metric envelopes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class OcuduMetricEnvelope:
    timestamp_ns: int
    metric_type: str
    payload: Mapping[str, Any]
    raw: Mapping[str, Any]


def parse_ocudu_json_line(line: str) -> OcuduMetricEnvelope | None:
    """Parse one stdout line; non-JSON application logs return ``None``."""
    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        document = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(document, dict):
        return None
    timestamp = _first(document, ("timestamp_ns", "timestamp", "ts"))
    metric_type = _first(document, ("metric_type", "type", "name"))
    payload = _first(document, ("payload", "metrics", "data"))
    if timestamp is None:
        raise ValueError("JSON metric envelope lacks a timestamp")
    if payload is None:
        payload = {key: value for key, value in document.items() if key not in {"timestamp_ns", "timestamp", "ts"}}
    if not isinstance(payload, dict):
        raise ValueError("JSON metric envelope payload is not an object")
    if metric_type is None:
        roots = tuple(payload)
        if len(roots) != 1:
            raise ValueError("native OCUDU metric envelope must contain exactly one metric root")
        metric_type = roots[0]
    timestamp_ns = _timestamp_ns(timestamp)
    # Some log streams encode epoch milliseconds. Preserve nanosecond precision
    # in the internal schema by converting only values clearly below epoch-ns.
    if 1_000_000_000_000 <= timestamp_ns < 1_000_000_000_000_000:
        timestamp_ns *= 1_000_000
    if timestamp_ns < 0:
        raise ValueError("metric timestamp must be non-negative")
    return OcuduMetricEnvelope(timestamp_ns, str(metric_type), payload, document)


def flatten_numeric(payload: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten only numeric leaves; strings and booleans never become metrics."""
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            flattened[name] = float(value)
        elif isinstance(value, dict):
            flattened.update(flatten_numeric(value, name))
    return flattened


def _first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _timestamp_ns(value: Any) -> int:
    if isinstance(value, str) and not value.isdigit():
        parsed = datetime.fromisoformat(value.removesuffix("Z"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delta = parsed - datetime(1970, 1, 1, tzinfo=timezone.utc)
        return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000
    return int(value)
