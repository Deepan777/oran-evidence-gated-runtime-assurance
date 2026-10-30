"""One-window action-explainable xApp decision execution.

This module contains no network simulator. It accepts only measured/estimated
inputs supplied by the telemetry and detector layers, runs the frozen policy,
executes the safety-gated adapter, and returns a fully machine-readable record.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Mapping

import numpy as np
import torch
from numpy.typing import NDArray

from src.controller.action_mask import Action, ActionContext, compute_action_mask
from src.controller.md3qn import mask_q_values
from src.controller.state import StateBuilder
from src.environment.iq_calibrated_env import FEATURE_NAMES
from src.xai.attribution import selected_action_attribution
from src.xai.contrastive import contrastive_attribution
from src.xai.counterfactual import CounterfactualResult, projected_counterfactual
from src.xai.rationale import FeatureContribution, build_operator_rationale
from src.xapp.adapters.actions import ActionAdapter, ActionExecution, ConfigurationStateStatus
from src.xapp.adapters.parameter_planner import ActionParameterPolicy, plan_action_parameters


JAMMER_CLASSES = ("none", "barrage", "partial_band", "sweep", "intermittent", "reactive")
TELEMETRY_STATE_FEATURES = (
    "sinr_db",
    "cqi",
    "rsrp_dbfs",
    "rsrq_db",
    "bler",
    "harq_nack_ratio",
    "dl_throughput_mbps",
    "rlc_drop_rate",
    "rlc_sdu_delay_ms",
    "prb_utilization",
    "clean_prb_fraction",
    "scheduler_queue_backlog_bytes",
)


@dataclass(frozen=True)
class DecisionInput:
    window_start_ns: int
    window_end_ns: int
    telemetry: Mapping[str, float]
    p_jam: float
    jammer_class_distribution: Mapping[str, float]
    jammer_occupancy_fraction: float
    jammer_subband_start_norm: float
    jammer_subband_end_norm: float
    previous_action: Action
    action_context: ActionContext

    def __post_init__(self) -> None:
        if self.window_end_ns <= self.window_start_ns:
            raise ValueError("decision window must have positive duration")
        missing = set(TELEMETRY_STATE_FEATURES) - set(self.telemetry)
        if missing:
            raise ValueError(f"missing measured telemetry: {sorted(missing)}")
        if set(self.jammer_class_distribution) != set(JAMMER_CLASSES):
            raise ValueError("jammer class distribution must contain the six locked classes")
        probabilities = np.asarray(list(self.jammer_class_distribution.values()), dtype=float)
        if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
            raise ValueError("jammer class probabilities must be non-negative and sum to one")
        if not 0.0 <= self.p_jam <= 1.0:
            raise ValueError("p_jam must lie in [0, 1]")


@dataclass(frozen=True)
class DecisionResult:
    state: tuple[float, ...]
    raw_state: Mapping[str, float]
    feasible_action_mask: tuple[bool, ...]
    q_values: tuple[float, ...]
    policy_selected_action: Action
    selected_action: Action
    alternative_action: Action | None
    execution: ActionExecution
    selected_action_attribution: tuple[float, ...]
    contrastive_attribution: tuple[float, ...]
    q_gap: float | None
    counterfactual: CounterfactualResult | None
    rationale: str | None
    latency_ms: Mapping[str, float]

    def as_record(self) -> dict[str, object]:
        counterfactual = None
        if self.counterfactual is not None:
            counterfactual = {
                "state": self.counterfactual.state.detach().cpu().numpy().tolist(),
                "valid": self.counterfactual.valid,
                "selected_action": self.counterfactual.selected_action,
                "target_action": self.counterfactual.target_action,
                "weighted_l1": self.counterfactual.weighted_l1,
                "iterations": self.counterfactual.iterations,
            }
        verified_action = self.execution.executed_action
        execution = {
            "requested_action": self.execution.requested_action.name,
            "requested_parameters": dict(self.execution.requested_parameters),
            "command_sent": self.execution.command_sent,
            "command_outcome": self.execution.command_outcome.value,
            "acknowledged": self.execution.acknowledged,
            "acknowledgement_message": self.execution.acknowledgement_message,
            "adapter_reported_action": (
                None if self.execution.adapter_reported_action is None
                else self.execution.adapter_reported_action.name
            ),
            "adapter_reported_parameters": dict(self.execution.adapter_reported_parameters),
            "action_match": self.execution.action_match,
            "parameter_match": self.execution.parameter_match,
            "configuration_state_status": self.execution.configuration_state_status.value,
            "configuration_state_action": (
                None if self.execution.configuration_state_action is None
                else self.execution.configuration_state_action.name
            ),
            "configuration_state_parameters": dict(self.execution.configuration_state_parameters),
            "restoration_requested": self.execution.restoration_requested,
            "restoration_acknowledged": self.execution.restoration_acknowledged,
            "restoration_verified": self.execution.restoration_verified,
            "failure": self.execution.failure,
            "backend": self.execution.backend,
            "started_ns": self.execution.started_ns,
            "completed_ns": self.execution.completed_ns,
            "readback": None if self.execution.readback is None else {
                "successful": self.execution.readback.successful,
                "source_id": self.execution.readback.source_id,
                "timestamp_ns": self.execution.readback.observed_timestamp_ns,
                "configuration_version": self.execution.readback.configuration_version,
                "verification_scope": self.execution.readback.verification_scope,
                "message": self.execution.readback.message,
            },
            # Compatibility keys are deliberately nullable. They contain only
            # independently verified state, never an inference from an ACK.
            "executed_action": None if verified_action is None else verified_action.name,
            "applied_parameters": dict(self.execution.applied_parameters),
        }
        return {
            "state": self.state,
            "raw_state": dict(self.raw_state),
            "feasible_action_mask": self.feasible_action_mask,
            "q_values": self.q_values,
            "policy_selected_action": self.policy_selected_action.name,
            "selected_action": self.selected_action.name,
            "alternative_action": self.alternative_action.name if self.alternative_action is not None else None,
            "execution": execution,
            "selected_action_attribution": self.selected_action_attribution,
            "contrastive_attribution": self.contrastive_attribution,
            "q_gap": self.q_gap,
            "counterfactual": counterfactual,
            "rationale": self.rationale,
            "latency_ms": dict(self.latency_ms),
        }


class DecisionRuntime:
    def __init__(
        self,
        q_network: torch.nn.Module,
        state_builder: StateBuilder,
        action_adapter: ActionAdapter,
        lower_bounds: NDArray[np.floating],
        upper_bounds: NDArray[np.floating],
        mutable_mask: NDArray[np.bool_],
        attribution_steps: int = 64,
        counterfactual_iterations: int = 500,
        action_parameter_policy: ActionParameterPolicy | None = None,
    ) -> None:
        expected_shape = (len(FEATURE_NAMES),)
        if state_builder.schema.feature_names != FEATURE_NAMES:
            raise ValueError("state builder feature order does not match the methodology lock")
        if not (
            np.asarray(lower_bounds).shape
            == np.asarray(upper_bounds).shape
            == np.asarray(mutable_mask).shape
            == expected_shape
        ):
            raise ValueError("counterfactual bounds and mutable mask must match state dimension")
        self.q_network = q_network.eval()
        self.state_builder = state_builder
        self.action_adapter = action_adapter
        self.lower_bounds = np.asarray(lower_bounds, dtype=np.float32)
        self.upper_bounds = np.asarray(upper_bounds, dtype=np.float32)
        self.mutable_mask = np.asarray(mutable_mask, dtype=bool)
        self.attribution_steps = attribution_steps
        self.counterfactual_iterations = counterfactual_iterations
        # The locked 20 MHz, 15 kHz SCS baseline contains 106 PRBs.
        self.action_parameter_policy = action_parameter_policy or ActionParameterPolicy(cell_prbs=106)
        # A2/A4 are absolute bounded caps at the OCUDU hook.  Remember the
        # last accepted cap so repeated one-second decisions do not
        # accidentally subtract the backoff again from an already-capped
        # scheduler MCS (which would progressively drive the radio to zero).
        self._last_mcs_cap: int | None = None
        self._last_mcs_observed: int | None = None

    def decide(
        self,
        decision: DecisionInput,
        explain: bool = True,
        forced_action: Action | None = None,
    ) -> DecisionResult:
        """Execute one safe decision.

        ``forced_action`` supports either bounded pre-confirmatory exploration
        or an explicit fail-closed A0 override from the prospective safety
        guard.  It is checked against the exact action mask.  The recorded
        policy-selected action and Q-values remain unchanged, making the
        distinction between proposal and executed behaviour auditable.
        """
        loop_start = perf_counter_ns()
        raw_state = self._assemble_raw_state(decision)
        preprocessing_start = perf_counter_ns()
        state = self.state_builder.build(raw_state)
        preprocessing_ms = self._elapsed(preprocessing_start)

        mask_start = perf_counter_ns()
        mask = compute_action_mask(decision.action_context)
        masking_ms = self._elapsed(mask_start)
        state_tensor = torch.from_numpy(state[None, :])
        mask_tensor = torch.from_numpy(mask[None, :])

        controller_start = perf_counter_ns()
        with torch.no_grad():
            q_tensor = self.q_network(state_tensor)
            policy_selected = Action(int(mask_q_values(q_tensor, mask_tensor).argmax(dim=-1).item()))
        controller_ms = self._elapsed(controller_start)

        if forced_action is not None and not bool(mask[int(forced_action)]):
            raise ValueError(f"forced exploration action is infeasible: {forced_action.name}")
        selected = policy_selected if forced_action is None else forced_action

        action_parameters = dict(plan_action_parameters(
            selected,
            decision.action_context,
            decision.jammer_subband_start_norm,
            decision.jammer_subband_end_norm,
            self.action_parameter_policy,
        ))
        if selected in (Action.MCS_BACKOFF, Action.COMPOSITE_LOCAL_DEFENSE):
            if self._last_mcs_cap is not None and self._last_mcs_observed is not None:
                # A scheduler MCS drop equal to the configured backoff is the
                # expected effect of our previous cap.  Only a larger,
                # externally observed drop may tighten the cap once more.
                channel_drop = self._last_mcs_observed - decision.action_context.current_mcs_index
                candidate = max(
                    self.action_parameter_policy.minimum_mcs,
                    decision.action_context.current_mcs_index
                    - self.action_parameter_policy.mcs_backoff,
                )
                if channel_drop > self.action_parameter_policy.mcs_backoff:
                    action_parameters["mcs_cap"] = min(self._last_mcs_cap, candidate)
                else:
                    action_parameters["mcs_cap"] = self._last_mcs_cap
            self._last_mcs_observed = decision.action_context.current_mcs_index
        execution = self.action_adapter.execute(selected, mask, action_parameters)
        self.observe_execution(execution)
        selected_attr = torch.zeros_like(state_tensor)
        contrastive_attr = torch.zeros_like(state_tensor)
        alternative: Action | None = None
        q_gap: float | None = None
        counterfactual: CounterfactualResult | None = None
        rationale: str | None = None
        attribution_ms = contrastive_ms = counterfactual_ms = 0.0

        if explain:
            attribution_start = perf_counter_ns()
            selected_attr = selected_action_attribution(
                self.q_network,
                state_tensor,
                torch.tensor([int(selected)]),
                steps=self.attribution_steps,
            )
            attribution_ms = self._elapsed(attribution_start)
            if int(mask.sum()) >= 2:
                contrastive_start = perf_counter_ns()
                contrastive_attr, selected_tensor, alternative_tensor, gap_tensor = contrastive_attribution(
                    self.q_network,
                    state_tensor,
                    mask_tensor,
                    steps=self.attribution_steps,
                )
                contrastive_ms = self._elapsed(contrastive_start)
                if forced_action is None and int(selected_tensor.item()) != int(selected):
                    raise RuntimeError("explanation selected action differs from controller action")
                alternative = Action(int(alternative_tensor.item()))
                q_gap = float(gap_tensor.item())
                cf_start = perf_counter_ns()
                counterfactual = projected_counterfactual(
                    self.q_network,
                    state_tensor,
                    mask_tensor,
                    int(alternative),
                    torch.from_numpy(self.lower_bounds[None, :]),
                    torch.from_numpy(self.upper_bounds[None, :]),
                    torch.from_numpy(self.mutable_mask[None, :]),
                    maximum_iterations=self.counterfactual_iterations,
                )
                counterfactual_ms = self._elapsed(cf_start)
                contribution_values = contrastive_attr[0].detach().cpu().numpy()
                top_indexes = np.argsort(np.abs(contribution_values))[::-1][:5]
                contributors = tuple(
                    FeatureContribution(FEATURE_NAMES[index], float(state[index]), float(contribution_values[index]))
                    for index in top_indexes
                )
                rationale = build_operator_rationale(selected.name, alternative.name, q_gap, contributors)

        total_ms = self._elapsed(loop_start)
        latencies = {
            "preprocessing_latency": preprocessing_ms,
            "action_masking_latency": masking_ms,
            "controller_inference_latency": controller_ms,
            "control_command_latency": execution.latency_ms,
            "xai_attribution_latency": attribution_ms,
            "contrastive_explanation_latency": contrastive_ms,
            "counterfactual_generation_latency": counterfactual_ms,
            "total_end_to_end_loop_latency": total_ms,
        }
        return DecisionResult(
            state=tuple(float(value) for value in state),
            raw_state=raw_state,
            feasible_action_mask=tuple(bool(value) for value in mask),
            q_values=tuple(float(value) for value in q_tensor[0].detach().cpu().numpy()),
            policy_selected_action=policy_selected,
            selected_action=selected,
            alternative_action=alternative,
            execution=execution,
            selected_action_attribution=tuple(float(value) for value in selected_attr[0].detach().cpu().numpy()),
            contrastive_attribution=tuple(float(value) for value in contrastive_attr[0].detach().cpu().numpy()),
            q_gap=q_gap,
            counterfactual=counterfactual,
            rationale=rationale,
            latency_ms=latencies,
        )

    @staticmethod
    def _elapsed(start_ns: int) -> float:
        return (perf_counter_ns() - start_ns) / 1_000_000.0

    def observe_execution(self, execution: ActionExecution) -> None:
        """Keep actuator memory consistent with both normal and restoration commands."""
        if execution.configuration_state_status is not ConfigurationStateStatus.VERIFIED:
            return
        if execution.configuration_state_action in (Action.MCS_BACKOFF, Action.COMPOSITE_LOCAL_DEFENSE):
            applied_cap = execution.configuration_state_parameters.get("mcs_cap")
            self._last_mcs_cap = int(applied_cap) if applied_cap is not None else None
        else:
            self._last_mcs_cap = None
            self._last_mcs_observed = None

    @staticmethod
    def _assemble_raw_state(decision: DecisionInput) -> dict[str, float]:
        state = {name: float(decision.telemetry[name]) for name in TELEMETRY_STATE_FEATURES}
        state["p_jam"] = float(decision.p_jam)
        for jammer_class in JAMMER_CLASSES:
            state[f"jammer_class_{jammer_class}"] = float(decision.jammer_class_distribution[jammer_class])
        state["jammer_occupancy_fraction"] = float(decision.jammer_occupancy_fraction)
        state["jammer_subband_start_norm"] = float(decision.jammer_subband_start_norm)
        state["jammer_subband_end_norm"] = float(decision.jammer_subband_end_norm)
        for action in Action:
            state[f"previous_action_a{int(action)}"] = float(decision.previous_action is action)
        return state
