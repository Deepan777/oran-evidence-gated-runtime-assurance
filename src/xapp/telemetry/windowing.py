"""Deterministic, non-overlapping telemetry and jammer-truth windows.

The aggregator intentionally keeps raw metric names and aggregation statistics
separate.  This prevents a missing live measurement from being silently
replaced by a synthetic value and makes the feature-availability audit
machine-checkable before dataset construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class MetricSample:
    timestamp_ns: int
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not self.values:
            raise ValueError("metric sample must contain at least one value")
        if any(not isfinite(float(value)) for value in self.values.values()):
            raise ValueError("metric values must be finite")


@dataclass(frozen=True)
class JammerTruthInterval:
    start_ns: int
    end_ns: int
    jammer_type: str
    requested_jsr_db: float | None
    realized_jsr_db: float | None
    active: bool
    subband_start_norm: float | None = None
    subband_end_norm: float | None = None

    def __post_init__(self) -> None:
        if self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValueError("truth interval must have positive duration")
        for value in (
            self.requested_jsr_db,
            self.realized_jsr_db,
            self.subband_start_norm,
            self.subband_end_norm,
        ):
            if value is not None and not isfinite(float(value)):
                raise ValueError("truth values must be finite when present")


@dataclass(frozen=True)
class TelemetryWindow:
    start_ns: int
    end_ns: int
    sample_count: int
    metrics: Mapping[str, Mapping[str, float]]
    missing_metrics: tuple[str, ...]
    jammer_type: str
    jammer_occupancy_fraction: float
    requested_jsr_db: float | None
    realized_jsr_db: float | None
    jammer_subband_start_norm: float | None
    jammer_subband_end_norm: float | None


class WindowAggregator:
    """Aggregate timestamped measurements into a fixed UTC-aligned timeline."""

    def __init__(
        self,
        required_metrics: tuple[str, ...],
        window_seconds: int = 1,
        origin_ns: int = 0,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if origin_ns < 0:
            raise ValueError("origin_ns must be non-negative")
        if len(required_metrics) != len(set(required_metrics)):
            raise ValueError("required metric names must be unique")
        self.required_metrics = required_metrics
        self.window_ns = window_seconds * 1_000_000_000
        self.origin_ns = origin_ns
        self._samples: list[MetricSample] = []
        self._truth: list[JammerTruthInterval] = []
        self._next_window_start_ns = origin_ns

    def add_sample(self, sample: MetricSample) -> None:
        if sample.timestamp_ns < self._next_window_start_ns:
            raise ValueError("late sample belongs to an already closed window")
        if self._samples and sample.timestamp_ns < self._samples[-1].timestamp_ns:
            raise ValueError("metric samples must be added in timestamp order")
        self._samples.append(sample)

    def add_truth(self, truth: JammerTruthInterval) -> None:
        if truth.end_ns <= self._next_window_start_ns:
            raise ValueError("late truth belongs to an already closed window")
        if self._truth and truth.start_ns < self._truth[-1].start_ns:
            raise ValueError("truth intervals must be added in start order")
        self._truth.append(truth)

    def close_through(self, timestamp_ns: int) -> list[TelemetryWindow]:
        """Close every window whose exclusive end is at or before timestamp."""
        if timestamp_ns < self._next_window_start_ns:
            raise ValueError("close timestamp precedes the next open window")
        closed: list[TelemetryWindow] = []
        while self._next_window_start_ns + self.window_ns <= timestamp_ns:
            start = self._next_window_start_ns
            end = start + self.window_ns
            closed.append(self._build_window(start, end))
            self._next_window_start_ns = end
            self._samples = [sample for sample in self._samples if sample.timestamp_ns >= end]
            self._truth = [truth for truth in self._truth if truth.end_ns > end]
        return closed

    def _build_window(self, start: int, end: int) -> TelemetryWindow:
        samples = [sample for sample in self._samples if start <= sample.timestamp_ns < end]
        by_metric: dict[str, list[tuple[int, float]]] = {}
        for sample in samples:
            for name, value in sample.values.items():
                by_metric.setdefault(name, []).append((sample.timestamp_ns, float(value)))

        aggregates: dict[str, dict[str, float]] = {}
        for name, observations in sorted(by_metric.items()):
            times = np.asarray([item[0] for item in observations], dtype=np.float64)
            values = np.asarray([item[1] for item in observations], dtype=np.float64)
            trend = 0.0
            if len(values) > 1 and times[-1] > times[0]:
                centered_time = (times - times[0]) / 1_000_000_000.0
                trend = float(np.polyfit(centered_time, values, 1)[0])
            aggregates[name] = {
                "last": float(values[-1]),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "trend_per_second": trend,
                "count": float(len(values)),
            }

        missing = tuple(name for name in self.required_metrics if name not in aggregates)
        overlapping = [truth for truth in self._truth if truth.start_ns < end and truth.end_ns > start]
        active_overlaps: list[tuple[JammerTruthInterval, int]] = []
        for truth in overlapping:
            overlap = max(0, min(end, truth.end_ns) - max(start, truth.start_ns))
            if truth.active and overlap:
                active_overlaps.append((truth, overlap))
        active_ns = min(self.window_ns, sum(overlap for _, overlap in active_overlaps))
        dominant = max(active_overlaps, key=lambda item: item[1])[0] if active_overlaps else None

        def weighted(field: str) -> float | None:
            pairs = [
                (getattr(truth, field), overlap)
                for truth, overlap in active_overlaps
                if getattr(truth, field) is not None
            ]
            if not pairs:
                return None
            return float(sum(float(value) * overlap for value, overlap in pairs) / sum(overlap for _, overlap in pairs))

        return TelemetryWindow(
            start_ns=start,
            end_ns=end,
            sample_count=len(samples),
            metrics=aggregates,
            missing_metrics=missing,
            jammer_type=dominant.jammer_type if dominant else "none",
            jammer_occupancy_fraction=float(active_ns / self.window_ns),
            requested_jsr_db=weighted("requested_jsr_db"),
            realized_jsr_db=weighted("realized_jsr_db"),
            jammer_subband_start_norm=weighted("subband_start_norm"),
            jammer_subband_end_norm=weighted("subband_end_norm"),
        )

