from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StateSchema:
    feature_names: tuple[str, ...]
    means: NDArray[np.float64]
    scales: NDArray[np.float64]
    clip_low: float = -8.0
    clip_high: float = 8.0

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature names must be unique")
        if self.means.shape != (len(self.feature_names),):
            raise ValueError("means shape does not match feature names")
        if self.scales.shape != (len(self.feature_names),):
            raise ValueError("scales shape does not match feature names")
        if np.any(self.scales <= 0.0):
            raise ValueError("all state scales must be positive")


class StateBuilder:
    def __init__(self, schema: StateSchema):
        self.schema = schema

    def build(self, values: Mapping[str, float]) -> NDArray[np.float32]:
        expected = set(self.schema.feature_names)
        supplied = set(values)
        missing = expected - supplied
        unexpected = supplied - expected
        if missing or unexpected:
            raise ValueError(
                f"state fields mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        raw = np.asarray([values[name] for name in self.schema.feature_names], dtype=np.float64)
        if not np.all(np.isfinite(raw)):
            raise ValueError("state contains non-finite values")
        standardized = (raw - self.schema.means) / self.schema.scales
        return np.clip(
            standardized,
            self.schema.clip_low,
            self.schema.clip_high,
        ).astype(np.float32)


class CausalStateImputer:
    """Deterministic, causal filling for temporarily unavailable live inputs.

    The first unavailable value receives the frozen training-set mean from the
    :class:`StateSchema`; later unavailable values receive the last valid
    observation in the same run.  Callers must retain the returned provenance
    in their raw decision record.  This class deliberately cannot use a future
    value or jammer truth.
    """

    def __init__(self, schema: StateSchema) -> None:
        self._fallback = {name: float(value) for name, value in zip(schema.feature_names, schema.means, strict=True)}
        self._previous: dict[str, float] = {}

    def fill(self, values: Mapping[str, float], required_names: tuple[str, ...]) -> tuple[dict[str, float], dict[str, str]]:
        completed: dict[str, float] = {}
        provenance: dict[str, str] = {}
        for name in required_names:
            value = values.get(name)
            if value is not None and np.isfinite(float(value)):
                completed[name] = float(value)
                self._previous[name] = float(value)
                provenance[name] = "measured"
            elif name in self._previous:
                completed[name] = self._previous[name]
                provenance[name] = "previous_measured"
            elif name in self._fallback:
                completed[name] = self._fallback[name]
                provenance[name] = "training_mean"
            else:
                raise ValueError(f"no frozen fallback is available for {name}")
        return completed, provenance
