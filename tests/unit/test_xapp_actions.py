from __future__ import annotations

import numpy as np

from src.controller.action_mask import Action
from src.xapp.adapters.actions import ActionAdapter, HookAcknowledgement


class RecordingBackend:
    def __init__(self, accepted: bool = True) -> None:
        self.accepted = accepted
        self.calls: list[tuple[Action, dict[str, object]]] = []

    def apply(self, action: Action, parameters: object) -> HookAcknowledgement:
        copied = dict(parameters)  # type: ignore[arg-type]
        self.calls.append((action, copied))
        return HookAcknowledgement(self.accepted, "test-hook", "rejected" if not self.accepted else "ok", copied)


def test_acknowledged_action_is_reported_as_executed() -> None:
    backend = RecordingBackend()
    adapter = ActionAdapter(backend, {Action.MCS_BACKOFF: {"levels": 4}})
    result = adapter.execute(Action.MCS_BACKOFF, np.ones(5, dtype=bool))
    assert result.acknowledged
    assert result.executed_action is Action.MCS_BACKOFF
    assert result.applied_parameters == {"levels": 4}
    assert result.failure is None
    assert result.latency_ms >= 0.0


def test_rejected_action_is_not_silently_counted_as_executed() -> None:
    adapter = ActionAdapter(RecordingBackend(accepted=False), {Action.PRB_AVOID_SHIFT: {"start": 0.5}})
    result = adapter.execute(Action.PRB_AVOID_SHIFT, np.ones(5, dtype=bool))
    assert not result.acknowledged
    assert result.requested_action is Action.PRB_AVOID_SHIFT
    assert result.executed_action is Action.NO_OP
    assert result.failure == "rejected"


def test_infeasible_action_never_reaches_backend() -> None:
    backend = RecordingBackend()
    adapter = ActionAdapter(backend, {})
    mask = np.asarray([True, False, True, True, True], dtype=bool)
    result = adapter.execute(Action.PRB_AVOID_SHIFT, mask)
    assert result.failure == "requested_action_infeasible"
    assert result.executed_action is Action.NO_OP
    assert backend.calls == []

