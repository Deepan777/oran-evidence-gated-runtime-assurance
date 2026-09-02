"""Prospective uncertainty-aware safety guard for live mitigation.

This guard is deliberately independent of jammer truth.  It permits a learned
policy only after an action-free calibration period, a frozen measured-IQ
reference, sustained spectral excess, and simultaneous service degradation.
Any missing prerequisite, recovery, maximum action duration, or observed harm
forces A0, which restores the baseline scheduler state at the OCUDU hook.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from src.controller.action_mask import Action


class GuardState(str, Enum):
    CALIBRATING = "CALIBRATING"
    OBSERVING = "OBSERVING"
    MITIGATING = "MITIGATING"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class SafetyGuardConfig:
    calibration_windows: int = 10
    sustained_evidence_windows: int = 3
    maximum_action_windows: int = 5
    cooldown_windows: int = 3
    minimum_spectral_excess: float = 0.05
    minimum_spectral_divergence: float = 0.35
    throughput_degradation_ratio: float = 0.80
    rollback_throughput_ratio: float = 0.50
    bler_degradation_threshold: float = 0.05
    latency_degradation_ratio: float = 2.0
    rollback_latency_ratio: float = 3.0

    def __post_init__(self) -> None:
        positive = (
            self.calibration_windows, self.sustained_evidence_windows,
            self.maximum_action_windows, self.cooldown_windows,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("guard window counts must be positive")
        fractions = (
            self.minimum_spectral_excess, self.throughput_degradation_ratio,
            self.rollback_throughput_ratio, self.bler_degradation_threshold,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("guard fraction thresholds must lie in [0, 1]")


@dataclass(frozen=True)
class SafetyObservation:
    window_index: int
    protected_session_active: bool
    spectral_reference_ready: bool
    spectral_excess_fraction: float
    spectral_divergence: float
    throughput_mbps: float
    bler: float
    latency_ms: float
    previous_action: Action


@dataclass(frozen=True)
class SafetyDecision:
    state: GuardState
    allow_policy: bool
    forced_action: Action | None
    reason: str
    consecutive_evidence_windows: int
    active_action_windows: int
    cooldown_remaining: int
    baseline_throughput_mbps: float | None
    baseline_latency_ms: float | None

    def as_dict(self) -> dict[str, object]:
        document = asdict(self)
        document["state"] = self.state.value
        document["forced_action"] = None if self.forced_action is None else self.forced_action.name
        return document


class UncertaintySafetyGuard:
    def __init__(self, config: SafetyGuardConfig = SafetyGuardConfig()) -> None:
        self.config = config
        self._throughput_reference: list[float] = []
        self._latency_reference: list[float] = []
        self._evidence = 0
        self._active = 0
        self._cooldown = 0
        self._state = GuardState.CALIBRATING

    @property
    def baseline_throughput_mbps(self) -> float | None:
        return self._robust_reference(self._throughput_reference)

    @property
    def baseline_latency_ms(self) -> float | None:
        return self._robust_reference(self._latency_reference)

    @staticmethod
    def _robust_reference(values: list[float]) -> float | None:
        finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
        return None if finite.size == 0 else float(np.median(finite))

    def decide(self, observation: SafetyObservation) -> SafetyDecision:
        if observation.window_index < 0:
            raise ValueError("window index must be non-negative")
        values = (
            observation.spectral_excess_fraction, observation.spectral_divergence,
            observation.throughput_mbps, observation.bler, observation.latency_ms,
        )
        if any(not np.isfinite(value) for value in values):
            return self._fallback("non_finite_safety_observation", cooldown=True)

        if observation.window_index < self.config.calibration_windows:
            self._state = GuardState.CALIBRATING
            self._throughput_reference.append(max(0.0, observation.throughput_mbps))
            self._latency_reference.append(max(0.0, observation.latency_ms))
            self._evidence = self._active = 0
            return self._decision(False, Action.NO_OP, "mandatory_action_free_calibration")

        throughput_ref = self.baseline_throughput_mbps
        latency_ref = self.baseline_latency_ms
        if not observation.protected_session_active:
            return self._fallback("protected_service_process_unavailable", cooldown=True)
        if not observation.spectral_reference_ready:
            return self._fallback("measured_iq_reference_not_ready", cooldown=True)
        if throughput_ref is None or latency_ref is None or throughput_ref <= 0.0:
            return self._fallback("clean_service_reference_unavailable", cooldown=True)

        # A previous non-A0 action is rolled back before considering new
        # evidence when the next measured window crosses a harm boundary.
        if observation.previous_action is not Action.NO_OP and (
            observation.throughput_mbps < throughput_ref * self.config.rollback_throughput_ratio
            or observation.latency_ms > max(
                latency_ref * self.config.rollback_latency_ratio, latency_ref + 1.0
            )
        ):
            return self._fallback("post_action_harm_boundary_crossed", cooldown=True)

        if self._cooldown > 0:
            self._cooldown -= 1
            self._state = GuardState.COOLDOWN
            self._active = 0
            return self._decision(False, Action.NO_OP, "bounded_cooldown")

        spectral = (
            observation.spectral_excess_fraction >= self.config.minimum_spectral_excess
            and observation.spectral_divergence >= self.config.minimum_spectral_divergence
        )
        service_degraded = (
            observation.throughput_mbps <= throughput_ref * self.config.throughput_degradation_ratio
            or observation.bler >= self.config.bler_degradation_threshold
            or observation.latency_ms >= max(
                latency_ref * self.config.latency_degradation_ratio, latency_ref + 0.5
            )
        )
        self._evidence = self._evidence + 1 if spectral and service_degraded else 0
        if self._evidence < self.config.sustained_evidence_windows:
            self._state = GuardState.OBSERVING
            self._active = 0
            reason = "awaiting_sustained_joint_evidence" if spectral and service_degraded else "joint_evidence_absent"
            return self._decision(False, Action.NO_OP, reason)

        self._state = GuardState.MITIGATING
        self._active += 1
        if self._active > self.config.maximum_action_windows:
            return self._fallback("maximum_action_duration_reached", cooldown=True)
        return self._decision(True, None, "sustained_spectral_and_service_evidence")

    def _fallback(self, reason: str, *, cooldown: bool) -> SafetyDecision:
        self._state = GuardState.COOLDOWN if cooldown else GuardState.OBSERVING
        self._evidence = self._active = 0
        if cooldown:
            self._cooldown = self.config.cooldown_windows
        return self._decision(False, Action.NO_OP, reason)

    def _decision(
        self, allow_policy: bool, forced_action: Action | None, reason: str,
    ) -> SafetyDecision:
        return SafetyDecision(
            state=self._state, allow_policy=allow_policy,
            forced_action=forced_action, reason=reason,
            consecutive_evidence_windows=self._evidence,
            active_action_windows=self._active,
            cooldown_remaining=self._cooldown,
            baseline_throughput_mbps=self.baseline_throughput_mbps,
            baseline_latency_ms=self.baseline_latency_ms,
        )
