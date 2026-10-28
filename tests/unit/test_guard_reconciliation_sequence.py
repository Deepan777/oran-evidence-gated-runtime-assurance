"""Cross-component sequence tests for restoration and renewed authority.

The backend is a state-observing component harness, not a radio experiment.
"""
from __future__ import annotations

from src.controller.action_mask import Action
from src.controller.safety_guard import SafetyGuardConfig, SafetyObservation, UncertaintySafetyGuard
from src.xapp.adapters.actions import (
    ActionAdapter, CommandOutcome, ConfigurationReadback,
    ConfigurationStateStatus, HookAcknowledgement,
)
from src.xapp.adapters.ocudu_websocket import OcuduWebSocketResearchBackend


class ObservedBackend:
    def __init__(self) -> None:
        self.action = Action.NO_OP
        self.parameters: dict[str, object] = {}
        self.drop_next_ack = False
        self.readback_available = True
        self.generation = 0
        self.commands: list[Action] = []

    def apply(self, action: Action, parameters: object) -> HookAcknowledgement:
        self.commands.append(action)
        self.action, self.parameters = action, dict(parameters)  # type: ignore[arg-type]
        self.generation += 1
        if self.drop_next_ack:
            self.drop_next_ack = False
            raise TimeoutError("application occurred; acknowledgement lost")
        return HookAcknowledgement(True, "component-command", "accepted", action, self.parameters)

    def readback(self) -> ConfigurationReadback:
        return ConfigurationReadback(
            self.readback_available, "component-state-observer", "observed",
            self.action if self.readback_available else None, dict(self.parameters),
            1_000_000_000 + self.generation, str(self.generation), "component-process",
        )


def observation(index: int, *, degraded: bool = False) -> SafetyObservation:
    return SafetyObservation(
        index, "measured-iq", "epoch-1", index, f"window-{index}",
        (index + 1) * 1_000_000_000, True, True,
        .2 if degraded else 0., 1. if degraded else 0.,
        .7 if degraded else 1., 0., 1., Action.NO_OP,
    )


def test_guarded_loop_restores_observed_baseline_then_resumes_eligible_control() -> None:
    backend = ObservedBackend()
    adapter = ActionAdapter(backend, {Action.MCS_BACKOFF: {"mcs_cap": 6}})
    guard = UncertaintySafetyGuard(SafetyGuardConfig(
        calibration_windows=2, sustained_evidence_windows=2, cooldown_windows=2,
    ))
    mask = [True] * 5
    assert adapter.configuration_state_status is ConfigurationStateStatus.UNKNOWN
    assert not adapter.execute(Action.MCS_BACKOFF, mask).command_sent
    baseline = adapter.execute(Action.NO_OP, mask)
    assert baseline.restoration_acknowledged and baseline.restoration_verified
    for index in range(2):
        decision = guard.decide(observation(index), applied_state_verified=adapter.nontrivial_authority_ready)
        assert not decision.allow_policy
    assert not guard.decide(observation(2, degraded=True)).allow_policy
    allowed = guard.decide(observation(3, degraded=True))
    assert allowed.allow_policy
    backend.drop_next_ack = True
    lost = adapter.execute(Action.MCS_BACKOFF, mask)
    assert lost.command_outcome is CommandOutcome.UNKNOWN
    assert backend.action is Action.MCS_BACKOFF
    assert not adapter.nontrivial_authority_ready
    locked = guard.decide(observation(4, degraded=True), applied_state_verified=adapter.nontrivial_authority_ready)
    assert not locked.allow_policy and locked.forced_action is Action.NO_OP
    assert not adapter.execute(Action.MCS_BACKOFF, mask).command_sent
    assert not adapter.reconcile().successful
    restoration = adapter.execute(Action.NO_OP, mask)
    assert restoration.restoration_verified
    assert adapter.nontrivial_authority_ready
    # The guard separately observes its cooldown and requires fresh persistence.
    for index in (5, 6):
        assert not guard.decide(observation(index, degraded=True)).allow_policy
    assert not guard.decide(observation(7, degraded=True)).allow_policy
    assert guard.decide(observation(8, degraded=True)).allow_policy
    resumed = adapter.execute(Action.MCS_BACKOFF, mask)
    assert resumed.command_sent and resumed.executed_action is Action.MCS_BACKOFF


def test_failed_readback_after_verified_nontrivial_state_requires_baseline() -> None:
    backend = ObservedBackend()
    adapter = ActionAdapter(backend, {Action.MCS_BACKOFF: {"mcs_cap": 6}})
    assert adapter.reconcile().successful
    assert adapter.execute(Action.MCS_BACKOFF, [True] * 5).executed_action is Action.MCS_BACKOFF
    backend.readback_available = False
    assert not adapter.reconcile().successful
    backend.readback_available = True
    # Observing the old nontrivial setting must not bypass restoration.
    assert not adapter.reconcile().successful
    assert not adapter.nontrivial_authority_ready
    restored = adapter.execute(Action.NO_OP, [True] * 5)
    assert restored.restoration_verified and adapter.nontrivial_authority_ready


def test_measurement_corruption_resets_persistence_without_recalibrating_clean_reference() -> None:
    guard = UncertaintySafetyGuard(SafetyGuardConfig(calibration_windows=2, sustained_evidence_windows=2))
    guard.decide(observation(0)); guard.decide(observation(1))
    assert guard.decide(observation(2, degraded=True)).consecutive_evidence_windows == 1
    rejected = guard.reject_invalid("missing_measured_telemetry")
    assert rejected.consecutive_evidence_windows == 0
    assert rejected.calibration_windows_observed == 2
    assert rejected.baseline_throughput_mbps == 1.


def test_consumer_readback_does_not_mix_processes_or_configuration_parameters() -> None:
    def line(scope: str, *, pid: int = 1, cap: int = 6) -> str:
        return (
            f"RESEARCH_STATE_CONSUMED scope={scope} pid={pid} timestamp_ns=100 generation=7 "
            f"action=A2 prb_start=0 prb_stop=106 mcs_cap={cap} slice_min_pct=10 slice_max_pct=100"
        )
    mixed_process = "\n".join([line("prb"), line("mcs"), line("slice", pid=2)])
    assert not OcuduWebSocketResearchBackend.parse_consumed(mixed_process, not_before_ns=50).successful
    mixed_parameters = "\n".join([line("prb"), line("mcs"), line("slice", cap=7)])
    assert not OcuduWebSocketResearchBackend.parse_consumed(mixed_parameters, not_before_ns=50).successful
