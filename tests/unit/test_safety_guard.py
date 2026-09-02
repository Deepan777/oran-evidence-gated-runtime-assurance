from src.controller.action_mask import Action
from src.controller.safety_guard import SafetyObservation, UncertaintySafetyGuard


def _observation(index, *, throughput=1.0, latency=1.0, excess=0.0, divergence=0.0,
                 previous=Action.NO_OP, reference=True, active=True):
    return SafetyObservation(
        window_index=index, protected_session_active=active,
        spectral_reference_ready=reference, spectral_excess_fraction=excess,
        spectral_divergence=divergence, throughput_mbps=throughput,
        bler=0.0, latency_ms=latency, previous_action=previous,
    )


def test_guard_forces_action_free_calibration_and_rejects_false_alarm():
    guard = UncertaintySafetyGuard()
    for index in range(10):
        decision = guard.decide(_observation(index))
        assert decision.forced_action is Action.NO_OP
        assert not decision.allow_policy
    decision = guard.decide(_observation(10, excess=0.2, divergence=1.0))
    assert not decision.allow_policy
    assert decision.reason == "joint_evidence_absent"
    decision = guard.decide(_observation(11))
    assert decision.reason == "joint_evidence_absent"


def test_guard_requires_three_joint_windows_then_rolls_back_on_harm():
    guard = UncertaintySafetyGuard()
    for index in range(10):
        guard.decide(_observation(index))
    for index in (10, 11):
        decision = guard.decide(_observation(index, throughput=0.7, excess=0.2, divergence=1.0))
        assert not decision.allow_policy
    decision = guard.decide(_observation(12, throughput=0.7, excess=0.2, divergence=1.0))
    assert decision.allow_policy
    assert decision.forced_action is None
    rollback = guard.decide(_observation(
        13, throughput=0.4, excess=0.2, divergence=1.0,
        previous=Action.MCS_BACKOFF,
    ))
    assert not rollback.allow_policy
    assert rollback.forced_action is Action.NO_OP
    assert rollback.reason == "post_action_harm_boundary_crossed"
    assert rollback.cooldown_remaining == 3


def test_guard_fails_closed_without_verified_prerequisites():
    guard = UncertaintySafetyGuard()
    for index in range(10):
        guard.decide(_observation(index))
    assert guard.decide(_observation(10, active=False)).reason == "protected_service_process_unavailable"
