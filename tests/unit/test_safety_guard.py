from src.controller.action_mask import Action
from src.controller.safety_guard import SafetyObservation, UncertaintySafetyGuard


def _observation(
    index: int,
    *,
    throughput: float = 1.0,
    latency: float = 1.0,
    excess: float = 0.0,
    divergence: float = 0.0,
    previous: Action = Action.NO_OP,
    reference: bool = True,
    active: bool = True,
    source: str = "iq",
    epoch: str = "epoch-1",
    sequence: int | None = None,
    measurement: str | None = None,
    timestamp: int | None = None,
) -> SafetyObservation:
    sequence = index if sequence is None else sequence
    return SafetyObservation(
        window_index=index,
        source_id=source,
        source_epoch=epoch,
        sequence_number=sequence,
        measurement_window_id=f"w-{sequence}" if measurement is None else measurement,
        source_timestamp_ns=(sequence + 1) * 1_000_000_000 if timestamp is None else timestamp,
        protected_session_active=active,
        spectral_reference_ready=reference,
        spectral_excess_fraction=excess,
        spectral_divergence=divergence,
        throughput_mbps=throughput,
        bler=0.0,
        latency_ms=latency,
        previous_action=previous,
    )


def _calibrate(guard: UncertaintySafetyGuard) -> None:
    for index in range(10):
        decision = guard.decide(_observation(index))
        assert decision.forced_action is Action.NO_OP
        assert not decision.allow_policy


def test_guard_forces_action_free_calibration_and_rejects_false_alarm() -> None:
    guard = UncertaintySafetyGuard()
    _calibrate(guard)
    decision = guard.decide(_observation(10, excess=0.2, divergence=1.0))
    assert not decision.allow_policy
    assert decision.reason == "joint_evidence_absent"
    assert decision.calibration_windows_observed == 10


def test_guard_requires_three_distinct_joint_windows_then_requests_rollback_on_harm() -> None:
    guard = UncertaintySafetyGuard()
    _calibrate(guard)
    for index in (10, 11):
        decision = guard.decide(_observation(index, throughput=0.7, excess=0.2, divergence=1.0))
        assert not decision.allow_policy
    decision = guard.decide(_observation(12, throughput=0.7, excess=0.2, divergence=1.0))
    assert decision.allow_policy
    rollback = guard.decide(_observation(
        13, throughput=0.4, excess=0.2, divergence=1.0,
        previous=Action.MCS_BACKOFF,
    ))
    assert not rollback.allow_policy
    assert rollback.forced_action is Action.NO_OP
    assert rollback.reason == "post_action_harm_boundary_crossed"
    assert rollback.cooldown_remaining == 3


def test_duplicate_delivery_cannot_advance_persistence() -> None:
    guard = UncertaintySafetyGuard()
    _calibrate(guard)
    first = _observation(10, throughput=0.7, excess=0.2, divergence=1.0)
    assert guard.decide(first).consecutive_evidence_windows == 1
    duplicate = guard.decide(first)
    assert duplicate.reason == "duplicate_observation_ignored"
    assert duplicate.consecutive_evidence_windows == 1
    assert not duplicate.observation_eligible
    assert guard.decide(_observation(11, throughput=0.7, excess=0.2, divergence=1.0)).consecutive_evidence_windows == 2
    assert guard.decide(_observation(12, throughput=0.7, excess=0.2, divergence=1.0)).allow_policy


def test_historical_call_counting_rule_was_vulnerable_to_duplicate_delivery() -> None:
    """Document the defect without executing or modifying historical evidence."""

    legacy_evidence_count = 0
    one_physical_window_delivered_three_times = [True, True, True]
    for joint_evidence in one_physical_window_delivered_three_times:
        legacy_evidence_count = legacy_evidence_count + 1 if joint_evidence else 0
    assert legacy_evidence_count == 3


def test_multiple_samples_from_one_measurement_window_do_not_count_twice() -> None:
    guard = UncertaintySafetyGuard()
    _calibrate(guard)
    first = _observation(10, throughput=0.7, excess=0.2, divergence=1.0, measurement="shared")
    guard.decide(first)
    same_window_new_sequence = _observation(
        11, throughput=0.7, excess=0.2, divergence=1.0,
        sequence=11, measurement="shared",
    )
    duplicate = guard.decide(same_window_new_sequence)
    assert duplicate.reason == "duplicate_observation_ignored"
    assert duplicate.consecutive_evidence_windows == 1


def test_gap_resets_consecutive_evidence_before_current_window() -> None:
    guard = UncertaintySafetyGuard()
    _calibrate(guard)
    guard.decide(_observation(10, throughput=0.7, excess=0.2, divergence=1.0))
    gap = guard.decide(_observation(12, throughput=0.7, excess=0.2, divergence=1.0))
    assert gap.reason == "gap_reset_awaiting_sustained_joint_evidence"
    assert gap.consecutive_evidence_windows == 1


def test_stale_or_out_of_order_evidence_fails_closed_and_resets_persistence() -> None:
    guard = UncertaintySafetyGuard()
    _calibrate(guard)
    guard.decide(_observation(10, throughput=0.7, excess=0.2, divergence=1.0))
    stale = guard.decide(_observation(
        11, throughput=0.7, excess=0.2, divergence=1.0,
        sequence=9, measurement="stale", timestamp=9_000_000_000,
    ))
    assert stale.reason == "stale_or_out_of_order_observation"
    assert stale.consecutive_evidence_windows == 0
    assert stale.forced_action is Action.NO_OP


def test_source_restart_requires_fresh_calibration() -> None:
    guard = UncertaintySafetyGuard()
    _calibrate(guard)
    restarted = guard.decide(_observation(10, epoch="epoch-2"))
    assert restarted.reason == "source_changed_recalibration"
    assert restarted.calibration_windows_observed == 1
    assert restarted.forced_action is Action.NO_OP


def test_guard_fails_closed_without_verified_prerequisites() -> None:
    guard = UncertaintySafetyGuard()
    _calibrate(guard)
    assert guard.decide(_observation(10, active=False)).reason == "protected_service_process_unavailable"
