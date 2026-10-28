from __future__ import annotations

import numpy as np

from src.controller.action_mask import Action
from src.xapp.adapters.actions import (
    ActionAdapter,
    CommandOutcome,
    ConfigurationReadback,
    ConfigurationStateStatus,
    HookAcknowledgement,
)


class RecordingBackend:
    def __init__(
        self,
        *,
        accepted: bool = True,
        report_parameters: bool = True,
        readback_enabled: bool = False,
        raise_after_apply: bool = False,
    ) -> None:
        self.accepted = accepted
        self.report_parameters = report_parameters
        self.readback_enabled = readback_enabled
        self.raise_after_apply = raise_after_apply
        self.calls: list[tuple[Action, dict[str, object]]] = []
        self.current_action = Action.NO_OP
        self.current_parameters: dict[str, object] = {}

    def apply(self, action: Action, parameters: object) -> HookAcknowledgement:
        copied = dict(parameters)  # type: ignore[arg-type]
        self.calls.append((action, copied))
        if self.accepted or self.raise_after_apply:
            self.current_action = action
            self.current_parameters = copied
        if self.raise_after_apply:
            raise TimeoutError("acknowledgement lost")
        return HookAcknowledgement(
            accepted=self.accepted,
            backend="test-hook",
            message="ok" if self.accepted else "rejected",
            reported_action=action if self.accepted else None,
            reported_parameters=copied if self.accepted and self.report_parameters else {},
        )

    def readback(self) -> ConfigurationReadback:
        if not self.readback_enabled:
            raise RuntimeError("readback unavailable")
        return ConfigurationReadback(
            True, "test-readback", "verified", self.current_action, self.current_parameters,
        )


class NoReadbackBackend(RecordingBackend):
    readback = None  # type: ignore[assignment]


def initialized_adapter(backend: RecordingBackend, parameters: object) -> ActionAdapter:
    adapter = ActionAdapter(backend, parameters)  # type: ignore[arg-type]
    original = backend.readback
    backend.readback = lambda: ConfigurationReadback(True, "initial-baseline-observer", "observed", Action.NO_OP, {})  # type: ignore[method-assign]
    assert adapter.reconcile().successful
    backend.readback = original  # type: ignore[method-assign]
    return adapter


def test_startup_requires_observed_baseline_before_any_nontrivial_command() -> None:
    backend = NoReadbackBackend()
    adapter = ActionAdapter(backend, {})
    assert adapter.configuration_state_status is ConfigurationStateStatus.UNKNOWN
    blocked = adapter.execute(Action.MCS_BACKOFF, np.ones(5, dtype=bool))
    assert not blocked.command_sent
    assert backend.calls == []


def test_positive_ack_is_not_mislabelled_as_independently_verified() -> None:
    backend = NoReadbackBackend()
    adapter = initialized_adapter(backend, {Action.MCS_BACKOFF: {"levels": 4}})
    result = adapter.execute(Action.MCS_BACKOFF, np.ones(5, dtype=bool))
    assert result.acknowledged
    assert result.command_outcome is CommandOutcome.ACKNOWLEDGED
    assert result.adapter_reported_action is Action.MCS_BACKOFF
    assert result.configuration_state_status is ConfigurationStateStatus.ADAPTER_REPORTED
    assert result.executed_action is None
    assert result.applied_parameters == {}


def test_rejected_command_leaves_applied_state_unknown_not_a0() -> None:
    adapter = initialized_adapter(NoReadbackBackend(accepted=False), {})
    result = adapter.execute(Action.PRB_AVOID_SHIFT, np.ones(5, dtype=bool))
    assert result.command_outcome is CommandOutcome.REJECTED
    assert result.configuration_state_status is ConfigurationStateStatus.UNKNOWN
    assert result.configuration_state_action is None
    assert result.executed_action is None
    assert result.failure == "rejected"


def test_infeasible_action_is_not_sent_and_preserves_verified_baseline() -> None:
    backend = NoReadbackBackend()
    adapter = initialized_adapter(backend, {})
    mask = np.asarray([True, False, True, True, True], dtype=bool)
    result = adapter.execute(Action.PRB_AVOID_SHIFT, mask)
    assert result.command_outcome is CommandOutcome.NOT_SENT
    assert result.configuration_state_status is ConfigurationStateStatus.VERIFIED
    assert result.configuration_state_action is Action.NO_OP
    assert backend.calls == []


def test_applied_command_with_lost_acknowledgement_is_unknown() -> None:
    backend = NoReadbackBackend(raise_after_apply=True)
    adapter = initialized_adapter(backend, {})
    result = adapter.execute(Action.MCS_BACKOFF, np.ones(5, dtype=bool), {"mcs_cap": 6})
    assert backend.current_action is Action.MCS_BACKOFF
    assert result.command_outcome is CommandOutcome.UNKNOWN
    assert result.configuration_state_status is ConfigurationStateStatus.UNKNOWN
    assert result.executed_action is None


def test_positive_ack_with_parameter_mismatch_is_integrity_failure() -> None:
    class MismatchBackend(NoReadbackBackend):
        def apply(self, action: Action, parameters: object) -> HookAcknowledgement:
            return HookAcknowledgement(True, "test-hook", "ok", action, {"mcs_cap": 5})

    result = initialized_adapter(MismatchBackend(), {}).execute(
        Action.MCS_BACKOFF, np.ones(5, dtype=bool), {"mcs_cap": 6},
    )
    assert result.parameter_match is False
    assert result.configuration_state_status is ConfigurationStateStatus.UNKNOWN
    assert result.failure == "acknowledgement_configuration_mismatch"


def test_unverified_state_blocks_further_nontrivial_commands() -> None:
    backend = NoReadbackBackend()
    adapter = initialized_adapter(backend, {})
    first = adapter.execute(Action.MCS_BACKOFF, np.ones(5, dtype=bool), {"mcs_cap": 6})
    assert first.configuration_state_status is ConfigurationStateStatus.ADAPTER_REPORTED
    blocked = adapter.execute(Action.PRB_AVOID_SHIFT, np.ones(5, dtype=bool), {"prb_start": 10})
    assert not blocked.command_sent
    assert blocked.failure == "configuration_state_unverified_reconciliation_required"
    assert len(backend.calls) == 1


def test_restoration_request_is_not_called_restored_without_readback() -> None:
    backend = NoReadbackBackend()
    adapter = initialized_adapter(backend, {})
    adapter.execute(Action.MCS_BACKOFF, np.ones(5, dtype=bool), {"mcs_cap": 6})
    restoration = adapter.execute(Action.NO_OP, np.ones(5, dtype=bool))
    assert restoration.restoration_requested
    assert restoration.restoration_acknowledged
    assert not restoration.restoration_verified
    assert restoration.configuration_state_status is ConfigurationStateStatus.ADAPTER_REPORTED


def test_readback_verifies_execution_and_reconciliation() -> None:
    verified_backend = RecordingBackend(readback_enabled=True)
    verified = initialized_adapter(verified_backend, {}).execute(
        Action.MCS_BACKOFF, np.ones(5, dtype=bool), {"mcs_cap": 6},
    )
    assert verified.configuration_state_status is ConfigurationStateStatus.VERIFIED
    assert verified.executed_action is Action.MCS_BACKOFF
    assert verified.applied_parameters == {"mcs_cap": 6}

    lost_backend = RecordingBackend(readback_enabled=True, raise_after_apply=True)
    adapter = initialized_adapter(lost_backend, {})
    uncertain = adapter.execute(Action.MCS_BACKOFF, np.ones(5, dtype=bool), {"mcs_cap": 6})
    assert uncertain.configuration_state_status is ConfigurationStateStatus.UNKNOWN
    reconciled = adapter.reconcile()
    assert not reconciled.successful
    assert reconciled.status is ConfigurationStateStatus.UNKNOWN
    lost_backend.raise_after_apply = False
    restoration = adapter.execute(Action.NO_OP, np.ones(5, dtype=bool))
    assert restoration.restoration_verified
    assert adapter.nontrivial_authority_ready
    resumed = adapter.execute(Action.MCS_BACKOFF, np.ones(5, dtype=bool), {"mcs_cap": 6})
    assert resumed.command_sent
    assert resumed.configuration_state_status is ConfigurationStateStatus.VERIFIED
