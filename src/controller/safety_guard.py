"""Prospective uncertainty-aware authorization guard for live mitigation.

The guard is independent of jammer truth. It authorizes policy proposals only
after an action-free calibration period and sustained evidence from distinct,
temporally eligible measurement windows. A forced A0 is a restoration request;
it is not evidence that the radio actually reached A0.
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
    source_id: str
    source_epoch: str
    sequence_number: int
    measurement_window_id: str
    source_timestamp_ns: int
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
    observation_eligible: bool
    observation_identity: str | None
    consecutive_evidence_windows: int
    active_action_windows: int
    cooldown_remaining: int
    calibration_windows_observed: int
    baseline_throughput_mbps: float | None
    baseline_latency_ms: float | None

    def as_dict(self) -> dict[str, object]:
        document = asdict(self)
        document["state"] = self.state.value
        document["forced_action"] = None if self.forced_action is None else self.forced_action.name
        return document


class UncertaintySafetyGuard:
    """Stateful authorization guard with explicit evidence continuity rules.

    A qualifying observation is unique by source, epoch and measurement-window
    identity. Sequence, decision-window and source-time order must all advance.
    A gap resets persistence before the current window is assessed. A source or
    epoch change resets calibration. Duplicate, stale and out-of-order samples
    cannot advance calibration or evidence persistence.
    """

    def __init__(self, config: SafetyGuardConfig = SafetyGuardConfig()) -> None:
        self.config = config
        self._throughput_reference: list[float] = []
        self._latency_reference: list[float] = []
        self._calibration_count = 0
        self._evidence = 0
        self._active = 0
        self._cooldown = 0
        self._state = GuardState.CALIBRATING
        self._source_key: tuple[str, str] | None = None
        self._last_sequence: int | None = None
        self._last_window_index: int | None = None
        self._last_timestamp_ns: int | None = None
        self._seen_windows: set[tuple[str, str, str]] = set()

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

    @staticmethod
    def _identity(observation: SafetyObservation) -> tuple[str, str, str]:
        return (observation.source_id, observation.source_epoch, observation.measurement_window_id)

    def _reset_source(self, source_key: tuple[str, str]) -> None:
        self._source_key = source_key
        self._throughput_reference = []
        self._latency_reference = []
        self._calibration_count = 0
        self._evidence = self._active = self._cooldown = 0
        self._last_sequence = self._last_window_index = self._last_timestamp_ns = None
        self._seen_windows = set()
        self._state = GuardState.CALIBRATING

    def reject_invalid(self, reason: str = "invalid_measured_evidence") -> SafetyDecision:
        """Reset persistence and request baseline; retain valid source calibration."""
        return self._fallback(reason, cooldown=True, eligible=False, identity=None)

    def decide(self, observation: SafetyObservation, *, applied_state_verified: bool = True) -> SafetyDecision:
        if observation.window_index < 0 or observation.sequence_number < 0 or observation.source_timestamp_ns < 0:
            raise ValueError("window, sequence and source timestamp must be non-negative")
        if not observation.source_id or not observation.source_epoch or not observation.measurement_window_id:
            return self._fallback("missing_observation_identity", cooldown=True, eligible=False, identity=None)

        source_key = (observation.source_id, observation.source_epoch)
        source_changed = self._source_key is not None and source_key != self._source_key
        if self._source_key is None or source_changed:
            self._reset_source(source_key)

        identity = self._identity(observation)
        identity_text = ":".join(identity)
        if identity in self._seen_windows:
            return self._decision(False, Action.NO_OP, "duplicate_observation_ignored", False, identity_text)

        if self._last_sequence is not None and (
            observation.sequence_number <= self._last_sequence
            or observation.window_index <= int(self._last_window_index)
            or observation.source_timestamp_ns <= int(self._last_timestamp_ns)
        ):
            return self._fallback(
                "stale_or_out_of_order_observation", cooldown=True,
                eligible=False, identity=identity_text,
            )

        gap = self._last_sequence is not None and (
            observation.sequence_number != self._last_sequence + 1
            or observation.window_index != int(self._last_window_index) + 1
        )
        if gap:
            self._evidence = 0
            self._active = 0

        self._seen_windows.add(identity)
        self._last_sequence = observation.sequence_number
        self._last_window_index = observation.window_index
        self._last_timestamp_ns = observation.source_timestamp_ns

        values = (
            observation.spectral_excess_fraction, observation.spectral_divergence,
            observation.throughput_mbps, observation.bler, observation.latency_ms,
        )
        if any(not np.isfinite(value) for value in values):
            return self._fallback(
                "non_finite_safety_observation", cooldown=True,
                eligible=False, identity=identity_text,
            )

        if not observation.protected_session_active:
            return self._fallback(
                "protected_service_process_unavailable", cooldown=True,
                eligible=False, identity=identity_text,
            )

        if not applied_state_verified:
            return self._fallback(
                "applied_state_lockout_restore_baseline", cooldown=True,
                eligible=False, identity=identity_text,
            )

        if self._calibration_count < self.config.calibration_windows:
            self._state = GuardState.CALIBRATING
            self._throughput_reference.append(max(0.0, observation.throughput_mbps))
            self._latency_reference.append(max(0.0, observation.latency_ms))
            self._calibration_count += 1
            self._evidence = self._active = 0
            reason = "source_changed_recalibration" if source_changed else "mandatory_action_free_calibration"
            return self._decision(False, Action.NO_OP, reason, True, identity_text)

        throughput_ref = self.baseline_throughput_mbps
        latency_ref = self.baseline_latency_ms
        if not observation.protected_session_active:
            return self._fallback(
                "protected_service_process_unavailable", cooldown=True,
                eligible=False, identity=identity_text,
            )
        if not observation.spectral_reference_ready:
            return self._fallback(
                "measured_iq_reference_not_ready", cooldown=True,
                eligible=False, identity=identity_text,
            )
        if throughput_ref is None or latency_ref is None or throughput_ref <= 0.0:
            return self._fallback(
                "clean_service_reference_unavailable", cooldown=True,
                eligible=False, identity=identity_text,
            )

        if observation.previous_action is not Action.NO_OP and (
            observation.throughput_mbps < throughput_ref * self.config.rollback_throughput_ratio
            or observation.latency_ms > max(
                latency_ref * self.config.rollback_latency_ratio, latency_ref + 1.0
            )
        ):
            return self._fallback(
                "post_action_harm_boundary_crossed", cooldown=True,
                eligible=True, identity=identity_text,
            )

        if self._cooldown > 0:
            self._cooldown -= 1
            self._state = GuardState.COOLDOWN
            self._active = 0
            return self._decision(False, Action.NO_OP, "bounded_cooldown", True, identity_text)

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
            if gap and spectral and service_degraded:
                reason = "gap_reset_awaiting_sustained_joint_evidence"
            else:
                reason = "awaiting_sustained_joint_evidence" if spectral and service_degraded else "joint_evidence_absent"
            return self._decision(False, Action.NO_OP, reason, True, identity_text)

        self._state = GuardState.MITIGATING
        self._active += 1
        if self._active > self.config.maximum_action_windows:
            return self._fallback(
                "maximum_action_duration_reached", cooldown=True,
                eligible=True, identity=identity_text,
            )
        return self._decision(True, None, "sustained_spectral_and_service_evidence", True, identity_text)

    def _fallback(
        self, reason: str, *, cooldown: bool, eligible: bool, identity: str | None,
    ) -> SafetyDecision:
        self._state = GuardState.COOLDOWN if cooldown else GuardState.OBSERVING
        self._evidence = self._active = 0
        if cooldown:
            self._cooldown = self.config.cooldown_windows
        return self._decision(False, Action.NO_OP, reason, eligible, identity)

    def _decision(
        self,
        allow_policy: bool,
        forced_action: Action | None,
        reason: str,
        eligible: bool,
        identity: str | None,
    ) -> SafetyDecision:
        return SafetyDecision(
            state=self._state,
            allow_policy=allow_policy,
            forced_action=forced_action,
            reason=reason,
            observation_eligible=eligible,
            observation_identity=identity,
            consecutive_evidence_windows=self._evidence,
            active_action_windows=self._active,
            cooldown_remaining=self._cooldown,
            calibration_windows_observed=self._calibration_count,
            baseline_throughput_mbps=self.baseline_throughput_mbps,
            baseline_latency_ms=self.baseline_latency_ms,
        )
