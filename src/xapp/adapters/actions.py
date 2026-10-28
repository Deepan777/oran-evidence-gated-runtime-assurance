"""Safety-gated action execution with explicit command/state semantics.

An acknowledgement is evidence about a command exchange. It is not, by
itself, independent evidence of the radio configuration. This module keeps
those two facts separate and never maps an unacknowledged command to A0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic_ns
from typing import Mapping, Protocol

from src.controller.action_mask import Action


class CommandOutcome(str, Enum):
    NOT_SENT = "NOT_SENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class ConfigurationStateStatus(str, Enum):
    VERIFIED = "VERIFIED"
    ADAPTER_REPORTED = "ADAPTER_REPORTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class HookAcknowledgement:
    """Application/transport acknowledgement returned by a research hook.

    ``reported_action`` and ``reported_parameters`` must contain only values
    actually present in the response. A backend must not echo the request and
    describe it as an applied configuration.
    """

    accepted: bool
    backend: str
    message: str
    reported_action: Action | None = None
    reported_parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfigurationReadback:
    """Independent configuration-state readback, when a backend supports it."""

    successful: bool
    backend: str
    message: str
    action: Action | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)
    observed_timestamp_ns: int | None = None
    configuration_version: str | None = None
    source_id: str | None = None
    verification_scope: str = "software_configuration"


class ActionHookBackend(Protocol):
    def apply(self, action: Action, parameters: Mapping[str, object]) -> HookAcknowledgement: ...


@dataclass(frozen=True)
class ReconciliationResult:
    attempted: bool
    successful: bool
    status: ConfigurationStateStatus
    verified_action: Action | None
    verified_parameters: Mapping[str, object]
    backend: str
    message: str


@dataclass(frozen=True)
class ActionExecution:
    requested_action: Action
    requested_parameters: Mapping[str, object]
    command_sent: bool
    command_outcome: CommandOutcome
    acknowledged: bool
    acknowledgement_message: str | None
    adapter_reported_action: Action | None
    adapter_reported_parameters: Mapping[str, object]
    action_match: bool | None
    parameter_match: bool | None
    configuration_state_status: ConfigurationStateStatus
    configuration_state_action: Action | None
    configuration_state_parameters: Mapping[str, object]
    restoration_requested: bool
    restoration_acknowledged: bool
    restoration_verified: bool
    failure: str | None
    backend: str
    started_ns: int
    completed_ns: int
    readback: ConfigurationReadback | None = None

    @property
    def latency_ms(self) -> float:
        return (self.completed_ns - self.started_ns) / 1_000_000.0

    @property
    def executed_action(self) -> Action | None:
        """Backward-compatible name restricted to independently verified state."""

        if self.configuration_state_status is ConfigurationStateStatus.VERIFIED:
            return self.configuration_state_action
        return None

    @property
    def applied_parameters(self) -> Mapping[str, object]:
        """Return parameters only when independent readback verified them."""

        if self.configuration_state_status is ConfigurationStateStatus.VERIFIED:
            return self.configuration_state_parameters
        return {}


class ActionAdapter:
    """Execute feasible commands while preserving configuration uncertainty.

    The adapter begins with UNKNOWN state and requires an observed A0 baseline.
    After any command whose
    effect is not independently verified, further nontrivial commands are
    blocked until a supported readback reconciles the state. An A0 restoration
    may still be requested, but is not called restored until readback confirms
    it.
    """

    def __init__(self, backend: ActionHookBackend, parameters: Mapping[Action, Mapping[str, object]]) -> None:
        self.backend = backend
        self.parameters = {action: dict(values) for action, values in parameters.items()}
        self._state_status = ConfigurationStateStatus.UNKNOWN
        self._state_action: Action | None = None
        self._state_parameters: dict[str, object] = {}
        self._baseline_required = True

    @property
    def configuration_state_status(self) -> ConfigurationStateStatus:
        return self._state_status

    @property
    def last_verified_action(self) -> Action:
        if self._state_status is ConfigurationStateStatus.VERIFIED and self._state_action is not None:
            return self._state_action
        return Action.NO_OP

    @property
    def nontrivial_authority_ready(self) -> bool:
        return self._state_status is ConfigurationStateStatus.VERIFIED and not self._baseline_required

    @staticmethod
    def _parameters_match(requested: Mapping[str, object], reported: Mapping[str, object]) -> bool:
        return dict(requested) == dict(reported)

    def _readback(self) -> ConfigurationReadback | None:
        reader = getattr(self.backend, "readback", None)
        if reader is None or not callable(reader):
            return None
        try:
            result = reader()
        except Exception as exc:  # backend boundary
            return ConfigurationReadback(False, type(self.backend).__name__, f"readback_exception:{type(exc).__name__}:{exc}")
        if not isinstance(result, ConfigurationReadback):
            return ConfigurationReadback(False, type(self.backend).__name__, "invalid_readback_type")
        return result

    def reconcile(self) -> ReconciliationResult:
        readback = self._readback()
        if readback is None:
            return ReconciliationResult(
                False, False, self._state_status, None, {}, type(self.backend).__name__, "independent_readback_unavailable",
            )
        if not readback.successful or readback.action is None:
            self._state_status = ConfigurationStateStatus.UNKNOWN
            self._state_action = None
            self._state_parameters = {}
            self._baseline_required = True
            return ReconciliationResult(True, False, self._state_status, None, {}, readback.backend, readback.message)
        if self._baseline_required and readback.action is not Action.NO_OP:
            self._state_status = ConfigurationStateStatus.UNKNOWN
            return ReconciliationResult(
                True, False, self._state_status, None, {}, readback.backend,
                "baseline_restoration_required_before_resumption",
            )
        self._state_status = ConfigurationStateStatus.VERIFIED
        self._state_action = readback.action
        self._state_parameters = dict(readback.parameters)
        self._baseline_required = False
        return ReconciliationResult(
            True, True, self._state_status, self._state_action,
            dict(self._state_parameters), readback.backend, readback.message,
        )

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
        restoration = requested_action is Action.NO_OP
        if not feasible:
            return self._not_sent(requested_action, requested_parameters, started, "requested_action_infeasible", restoration)
        if not restoration and not self.nontrivial_authority_ready:
            return self._not_sent(
                requested_action, requested_parameters, started,
                "configuration_state_unverified_reconciliation_required", restoration,
            )

        try:
            acknowledgement = self.backend.apply(requested_action, requested_parameters)
        except Exception as exc:  # backend boundary: preserve uncertainty as evidence
            self._state_status = ConfigurationStateStatus.UNKNOWN
            self._state_action = None
            self._state_parameters = {}
            self._baseline_required = True
            completed = monotonic_ns()
            return ActionExecution(
                requested_action, requested_parameters, True, CommandOutcome.UNKNOWN,
                False, None, None, {}, None, None,
                self._state_status, None, {}, restoration, False, False,
                f"backend_exception:{type(exc).__name__}:{exc}",
                type(self.backend).__name__, started, completed,
            )

        reported_action = acknowledgement.reported_action
        reported_parameters = dict(acknowledgement.reported_parameters)
        action_match = None if reported_action is None else reported_action is requested_action
        if reported_parameters:
            parameter_match = self._parameters_match(requested_parameters, reported_parameters)
        elif requested_parameters:
            parameter_match = None
        else:
            # A parameter-free action (A0) has a complete empty parameter set.
            parameter_match = True
        failure: str | None = None
        command_outcome = CommandOutcome.ACKNOWLEDGED if acknowledgement.accepted else CommandOutcome.REJECTED

        if not acknowledgement.accepted:
            self._state_status = ConfigurationStateStatus.UNKNOWN
            self._state_action = None
            self._state_parameters = {}
            failure = acknowledgement.message
        elif action_match is False or parameter_match is False:
            self._state_status = ConfigurationStateStatus.UNKNOWN
            self._state_action = None
            self._state_parameters = {}
            failure = "acknowledgement_configuration_mismatch"
        elif reported_action is None or parameter_match is None:
            self._state_status = ConfigurationStateStatus.UNKNOWN
            self._state_action = None
            self._state_parameters = {}
            failure = "acknowledgement_lacks_complete_configuration_report"
        else:
            self._state_status = ConfigurationStateStatus.ADAPTER_REPORTED
            self._state_action = reported_action
            self._state_parameters = reported_parameters

        restoration_acknowledged = bool(
            acknowledgement.accepted and action_match is not False and parameter_match is not False and restoration
        )
        restoration_verified = False
        readback = self._readback() if acknowledgement.accepted else None
        if readback is not None:
            if (
                readback.successful
                and readback.action is requested_action
                and self._parameters_match(requested_parameters, readback.parameters)
                and action_match is not False and parameter_match is not False
            ):
                self._state_status = ConfigurationStateStatus.VERIFIED
                self._state_action = readback.action
                self._state_parameters = dict(readback.parameters)
                restoration_verified = restoration
                if restoration:
                    self._baseline_required = False
                failure = None
            else:
                self._state_status = ConfigurationStateStatus.UNKNOWN
                self._state_action = None
                self._state_parameters = {}
                failure = "independent_readback_mismatch_or_failure"

        if self._state_status is not ConfigurationStateStatus.VERIFIED:
            self._baseline_required = True

        completed = monotonic_ns()
        return ActionExecution(
            requested_action=requested_action,
            requested_parameters=requested_parameters,
            command_sent=True,
            command_outcome=command_outcome,
            acknowledged=acknowledgement.accepted,
            acknowledgement_message=acknowledgement.message,
            adapter_reported_action=reported_action,
            adapter_reported_parameters=reported_parameters,
            action_match=action_match,
            parameter_match=parameter_match,
            configuration_state_status=self._state_status,
            configuration_state_action=self._state_action,
            configuration_state_parameters=dict(self._state_parameters),
            restoration_requested=restoration,
            restoration_acknowledged=restoration_acknowledged,
            restoration_verified=restoration_verified,
            failure=failure,
            backend=acknowledgement.backend,
            started_ns=started,
            completed_ns=completed,
            readback=readback,
        )

    def _not_sent(
        self,
        requested_action: Action,
        requested_parameters: Mapping[str, object],
        started: int,
        failure: str,
        restoration: bool,
    ) -> ActionExecution:
        completed = monotonic_ns()
        return ActionExecution(
            requested_action=requested_action,
            requested_parameters=requested_parameters,
            command_sent=False,
            command_outcome=CommandOutcome.NOT_SENT,
            acknowledged=False,
            acknowledgement_message=None,
            adapter_reported_action=None,
            adapter_reported_parameters={},
            action_match=None,
            parameter_match=None,
            configuration_state_status=self._state_status,
            configuration_state_action=self._state_action,
            configuration_state_parameters=dict(self._state_parameters),
            restoration_requested=restoration,
            restoration_acknowledged=False,
            restoration_verified=False,
            failure=failure,
            backend=type(self.backend).__name__,
            started_ns=started,
            completed_ns=completed,
        )
