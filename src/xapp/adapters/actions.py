"""Safety-gated action execution with explicit acknowledgement semantics."""
from __future__ import annotations

from dataclasses import dataclass
from time import monotonic_ns
from typing import Mapping, Protocol

from src.controller.action_mask import Action


@dataclass(frozen=True)
class HookAcknowledgement:
    accepted: bool
    backend: str
    message: str
    applied_parameters: Mapping[str, object]


class ActionHookBackend(Protocol):
    def apply(self, action: Action, parameters: Mapping[str, object]) -> HookAcknowledgement: ...


@dataclass(frozen=True)
class ActionExecution:
    requested_action: Action
    executed_action: Action
    requested_parameters: Mapping[str, object]
    applied_parameters: Mapping[str, object]
    acknowledged: bool
    failure: str | None
    backend: str
    started_ns: int
    completed_ns: int

    @property
    def latency_ms(self) -> float:
        return (self.completed_ns - self.started_ns) / 1_000_000.0


class ActionAdapter:
    """Execute only feasible actions and never disguise backend failures."""

    def __init__(self, backend: ActionHookBackend, parameters: Mapping[Action, Mapping[str, object]]) -> None:
        self.backend = backend
        self.parameters = {action: dict(values) for action, values in parameters.items()}

    def execute(
        self,
        requested_action: Action,
        feasible_mask: object,
        parameter_overrides: Mapping[str, object] | None = None,
    ) -> ActionExecution:
        try:
            feasible = bool(feasible_mask[int(requested_action)])  # type: ignore[index]
        except (IndexError, TypeError) as exc:
            raise ValueError("feasible mask must contain one entry per action") from exc
        started = monotonic_ns()
        requested_parameters = {
            **self.parameters.get(requested_action, {}),
            **({} if parameter_overrides is None else parameter_overrides),
        }
        if not feasible:
            completed = monotonic_ns()
            return ActionExecution(
                requested_action=requested_action,
                executed_action=Action.NO_OP,
                requested_parameters=requested_parameters,
                applied_parameters={},
                acknowledged=False,
                failure="requested_action_infeasible",
                backend=type(self.backend).__name__,
                started_ns=started,
                completed_ns=completed,
            )
        try:
            acknowledgement = self.backend.apply(requested_action, requested_parameters)
        except Exception as exc:  # backend boundary: preserve failure as experimental evidence
            completed = monotonic_ns()
            return ActionExecution(
                requested_action=requested_action,
                executed_action=Action.NO_OP,
                requested_parameters=requested_parameters,
                applied_parameters={},
                acknowledged=False,
                failure=f"backend_exception:{type(exc).__name__}:{exc}",
                backend=type(self.backend).__name__,
                started_ns=started,
                completed_ns=completed,
            )
        completed = monotonic_ns()
        executed = requested_action if acknowledgement.accepted else Action.NO_OP
        return ActionExecution(
            requested_action=requested_action,
            executed_action=executed,
            requested_parameters=requested_parameters,
            applied_parameters=dict(acknowledgement.applied_parameters),
            acknowledged=acknowledgement.accepted,
            failure=None if acknowledgement.accepted else acknowledgement.message,
            backend=acknowledgement.backend,
            started_ns=started,
            completed_ns=completed,
        )
