from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from src.controller.action_mask import Action, ActionContext
from src.controller.state import StateBuilder, StateSchema
from src.environment.iq_calibrated_env import FEATURE_NAMES
from src.xapp.adapters.actions import ActionAdapter, ConfigurationReadback, HookAcknowledgement
from src.xapp.runtime.loop import DecisionInput, DecisionRuntime, TELEMETRY_STATE_FEATURES
from experiments.live.run_xapp_loop import _PPOLogitAdapter, _sample_masked_ppo
from src.baselines.learning import MaskedPPOPolicy


class FixedQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(len(FEATURE_NAMES), 5, bias=True)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(torch.tensor([0.0, 3.0, 2.0, 1.0, 0.5]))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)


class AckBackend:
    def __init__(self) -> None:
        self.action = Action.NO_OP
        self.parameters: dict[str, object] = {}

    def apply(self, action: Action, parameters: object) -> HookAcknowledgement:
        self.action = action
        self.parameters = dict(parameters)  # type: ignore[arg-type]
        return HookAcknowledgement(True, "research-hook", "ok", action, self.parameters)

    def readback(self) -> ConfigurationReadback:
        return ConfigurationReadback(True, "research-readback", "verified", self.action, self.parameters)


def _decision() -> DecisionInput:
    telemetry = {name: 0.1 for name in TELEMETRY_STATE_FEATURES}
    return DecisionInput(
        window_start_ns=0,
        window_end_ns=1_000_000_000,
        telemetry=telemetry,
        p_jam=0.8,
        jammer_class_distribution={
            "none": 0.05,
            "barrage": 0.05,
            "partial_band": 0.7,
            "sweep": 0.1,
            "intermittent": 0.05,
            "reactive": 0.05,
        },
        jammer_occupancy_fraction=0.5,
        jammer_subband_start_norm=0.2,
        jammer_subband_end_norm=0.4,
        previous_action=Action.NO_OP,
        action_context=ActionContext(True, True, 0.5, 0.15, 10, 2, 0.3, 0.7, 0.1, 0.7),
    )


def test_runtime_preserves_locked_order_and_execution_acknowledgement() -> None:
    schema = StateSchema(FEATURE_NAMES, np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)))
    runtime = DecisionRuntime(
        FixedQ(),
        StateBuilder(schema),
        ActionAdapter(AckBackend(), {Action.PRB_AVOID_SHIFT: {"clean_region": [0.5, 1.0]}}),
        np.full(len(FEATURE_NAMES), -8.0),
        np.full(len(FEATURE_NAMES), 8.0),
        np.ones(len(FEATURE_NAMES), dtype=bool),
        attribution_steps=4,
        counterfactual_iterations=3,
    )
    assert runtime.action_adapter.reconcile().successful
    result = runtime.decide(_decision())
    assert result.selected_action is Action.PRB_AVOID_SHIFT
    assert result.execution.acknowledged
    assert result.execution.executed_action is Action.PRB_AVOID_SHIFT
    assert len(result.state) == 27
    assert len(result.selected_action_attribution) == 27
    assert len(result.contrastive_attribution) == 27
    assert result.alternative_action is Action.MCS_BACKOFF
    assert result.rationale is not None
    assert set(result.latency_ms) == {
        "preprocessing_latency",
        "action_masking_latency",
        "controller_inference_latency",
        "control_command_latency",
        "xai_attribution_latency",
        "contrastive_explanation_latency",
        "counterfactual_generation_latency",
        "total_end_to_end_loop_latency",
    }
    record = result.as_record()
    assert json.loads(json.dumps(record))["execution"]["acknowledged"] is True
    assert record["execution"]["configuration_state_status"] == "VERIFIED"


def test_runtime_rejects_missing_measured_telemetry() -> None:
    decision = _decision()
    incomplete = dict(decision.telemetry)
    incomplete.pop("bler")
    try:
        DecisionInput(**{**decision.__dict__, "telemetry": incomplete})
    except ValueError as exc:
        assert "missing measured telemetry" in str(exc)
    else:
        raise AssertionError("missing live telemetry must not be fabricated")


def test_development_forced_exploration_still_obeys_the_action_mask() -> None:
    schema = StateSchema(FEATURE_NAMES, np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)))
    runtime = DecisionRuntime(
        FixedQ(), StateBuilder(schema), ActionAdapter(AckBackend(), {}),
        np.full(len(FEATURE_NAMES), -8.0), np.full(len(FEATURE_NAMES), 8.0),
        np.ones(len(FEATURE_NAMES), dtype=bool),
    )
    assert runtime.action_adapter.reconcile().successful
    result = runtime.decide(_decision(), explain=False, forced_action=Action.MCS_BACKOFF)
    assert result.selected_action is Action.MCS_BACKOFF
    assert result.execution.executed_action is Action.MCS_BACKOFF

    invalid = DecisionInput(**{**_decision().__dict__, "action_context": ActionContext(
        True, False, 0.05, 0.15, 1, 2, 0.7, 0.7, 0.1, 0.7,
    )})
    try:
        runtime.decide(invalid, explain=False, forced_action=Action.MCS_BACKOFF)
    except ValueError as exc:
        assert "infeasible" in str(exc)
    else:
        raise AssertionError("forced exploration must not bypass the action mask")


def test_repeated_mcs_backoff_is_idempotent() -> None:
    schema = StateSchema(FEATURE_NAMES, np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)))
    network = FixedQ()
    with torch.no_grad():
        network.linear.bias.copy_(torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0]))
    runtime = DecisionRuntime(
        network, StateBuilder(schema), ActionAdapter(AckBackend(), {}),
        np.full(len(FEATURE_NAMES), -8.0), np.full(len(FEATURE_NAMES), 8.0),
        np.ones(len(FEATURE_NAMES), dtype=bool),
    )
    assert runtime.action_adapter.reconcile().successful
    first = runtime.decide(_decision(), explain=False, forced_action=Action.MCS_BACKOFF)
    assert first.execution.requested_parameters["mcs_cap"] == 6
    second = DecisionInput(**{**_decision().__dict__, "action_context": ActionContext(
        True, True, 0.5, 0.15, 6, 2, 0.3, 0.7, 0.1, 0.7,
    )})
    repeated = runtime.decide(second, explain=False, forced_action=Action.MCS_BACKOFF)
    assert repeated.execution.requested_parameters["mcs_cap"] == 6


def test_ppo_adapter_exposes_actor_logits_for_common_masking_runtime() -> None:
    policy = MaskedPPOPolicy(len(FEATURE_NAMES))
    adapter = _PPOLogitAdapter(policy)
    state = torch.zeros((2, len(FEATURE_NAMES)))
    assert torch.allclose(adapter(state), policy.logits(policy.trunk(state)))


def test_ppo_sampling_records_a_feasible_on_policy_action() -> None:
    policy = MaskedPPOPolicy(len(FEATURE_NAMES))
    action, behavior = _sample_masked_ppo(policy, np.zeros(len(FEATURE_NAMES)), np.array([True, False, True, False, False]))
    assert action in {Action.NO_OP, Action.MCS_BACKOFF}
    assert set(behavior) == {"action_log_probability", "value_estimate"}
