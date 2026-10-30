#!/usr/bin/env python3
"""Run the frozen detector/controller against closed live telemetry windows.

This process intentionally consumes only OCUDU metrics, receiver metrics and
the independently measured IQ spectral-estimator stream.  Jammer truth is
never opened by this module; it is evaluation-only evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from experiments.collection.build_live_run_records import _load_ocudu, _load_srsue
from src.baselines.learning import MaskedPPOPolicy, StandardQNetwork
from src.baselines.heuristic import HeuristicQNetwork
from src.controller.action_mask import Action, ActionContext
from src.controller.action_mask import compute_action_mask
from src.controller.safety_guard import SafetyObservation, UncertaintySafetyGuard
from src.controller.md3qn import DuelingQNetwork
from src.controller.state import CausalStateImputer, StateBuilder, StateSchema
from src.environment.iq_calibrated_env import FEATURE_NAMES
from src.xapp.adapters.actions import ActionAdapter
from src.xapp.adapters.ocudu_websocket import OcuduWebSocketResearchBackend
from src.xapp.runtime.loop import DecisionInput, DecisionRuntime


DETECTOR_FEATURES = (
    "sinr_db", "cqi", "rsrp_dbfs", "rsrq_db", "bler", "harq_nack_ratio",
    "dl_throughput_mbps", "rlc_drop_rate", "rlc_sdu_delay_ms", "prb_utilization",
    "scheduler_queue_backlog_bytes", "clean_prb_fraction",
)
JAMMER_CLASSES = ("none", "barrage", "partial_band", "sweep", "intermittent", "reactive")
CATEGORICAL_BINARY_FEATURES = {f"previous_action_a{i}" for i in range(5)}
MUTABLE_FEATURES = {
    "p_jam", "sinr_db", "cqi", "rsrp_dbfs", "rsrq_db", "bler", "harq_nack_ratio",
    "dl_throughput_mbps", "rlc_drop_rate", "rlc_sdu_delay_ms", "prb_utilization",
    "clean_prb_fraction", "scheduler_queue_backlog_bytes", "jammer_occupancy_fraction",
    "jammer_subband_start_norm", "jammer_subband_end_norm",
}


class _PPOLogitAdapter(torch.nn.Module):
    """Expose a PPO actor's pre-mask logits to the common safe runtime.

    The common runtime applies the exact feasibility mask before choosing an
    action, so its deterministic selection is the same as the PPO actor's.
    These outputs are policy logits rather than Q-values and are labelled as
    such in the evidence record.
    """

    def __init__(self, policy: MaskedPPOPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.policy.logits(self.policy.trunk(states))


def _load_controller(path: Path, kind: str) -> tuple[torch.nn.Module, str]:
    """Load only an explicitly named, architecture-compatible checkpoint."""
    if kind == "heuristic":
        if not path.is_file():
            raise ValueError(f"heuristic baseline manifest is missing: {path}")
        return HeuristicQNetwork().eval(), "heuristic_scores"
    if kind == "md3qn_per":
        network: torch.nn.Module = DuelingQNetwork(len(FEATURE_NAMES))
        score_type = "q_values"
    elif kind in {"standard_dqn", "prb_only_masked_dqn"}:
        network = StandardQNetwork(len(FEATURE_NAMES))
        score_type = "q_values"
    elif kind == "masked_ppo":
        policy = MaskedPPOPolicy(len(FEATURE_NAMES))
        policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return _PPOLogitAdapter(policy).eval(), "policy_logits"
    else:
        raise ValueError(f"unknown controller kind: {kind}")
    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return network.eval(), score_type


def _sample_masked_ppo(policy: MaskedPPOPolicy, state: np.ndarray, mask: np.ndarray) -> tuple[Action, dict[str, float]]:
    """Draw one auditable on-policy, feasibility-masked PPO action."""
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32)[None, :]
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool)[None, :]
        distribution = policy.distribution(state_tensor, mask_tensor)
        action_tensor = distribution.sample()
        action = Action(int(action_tensor.item()))
        return action, {
            "action_log_probability": float(distribution.log_prob(action_tensor).item()),
            "value_estimate": float(policy.value(policy.trunk(state_tensor)).item()),
        }


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    # The IQ estimator appends and flushes one JSON object per line while this
    # loop polls the file.  A read can therefore race the final write.  Ignore
    # only that unfinished trailing record; malformed newline-terminated
    # evidence must still fail closed.
    if text and not text.endswith(("\n", "\r")):
        lines = lines[:-1]
    records: list[dict[str, Any]] = []
    for line in lines:
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
    return records


def _latest(samples: list[Any], start_ns: int, end_ns: int) -> Any | None:
    selected = [sample for sample in samples if start_ns <= sample.timestamp_ns < end_ns]
    return max(selected, key=lambda sample: sample.timestamp_ns) if selected else None


def _load_schema(path: Path) -> tuple[StateSchema, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if tuple(document["feature_names"]) != FEATURE_NAMES:
        raise ValueError("frozen state schema feature order differs from methodology lock")
    schema = StateSchema(
        FEATURE_NAMES,
        np.asarray(document["means"], dtype=np.float64),
        np.asarray(document["scales"], dtype=np.float64),
        float(document.get("clip_low", -8.0)), float(document.get("clip_high", 8.0)),
    )
    raw_low = np.asarray(document["raw_p01"], dtype=np.float64)
    raw_high = np.asarray(document["raw_p99"], dtype=np.float64)
    if raw_low.shape != (len(FEATURE_NAMES),) or raw_high.shape != (len(FEATURE_NAMES),):
        raise ValueError("frozen state bounds have the wrong shape")
    return (
        schema,
        (raw_low - schema.means) / schema.scales,
        (raw_high - schema.means) / schema.scales,
        raw_low,
        raw_high,
    )


def _out_of_distribution(
    raw_state: dict[str, float], raw_low: np.ndarray, raw_high: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Return frozen-training-range violations without changing the state.

    A policy checkpoint is not allowed to control an uncalibrated live state.
    The p01--p99 bounds are a conservative development guard, not a data
    transformation: original measured and detector values stay in the record
    for later calibration analysis.
    """
    violations: dict[str, dict[str, float]] = {}
    for index, name in enumerate(FEATURE_NAMES):
        value = float(raw_state[name])
        lower, upper = float(raw_low[index]), float(raw_high[index])
        # Previous-action indicators are categorical one-hot fields.  Their
        # empirical calibration can legitimately contain only zeros when a
        # pilot has not yet exercised an action, but the value one is still a
        # valid state transition and must not be rejected as OOD.  Keep the
        # gate fail-closed for malformed values outside the binary domain.
        if name in CATEGORICAL_BINARY_FEATURES:
            if value not in (0.0, 1.0):
                violations[name] = {"value": value, "training_p01": 0.0, "training_p99": 1.0}
            continue
        if value < lower or value > upper:
            violations[name] = {"value": value, "training_p01": lower, "training_p99": upper}
    return violations


# These are telemetry-contract plausibility limits, not empirical training
# quantiles.  Confirmatory scenarios deliberately test distribution shift, so
# crossing p01/p99 is retained as an auditable warning while only malformed or
# physically impossible values are rejected.  Large but finite queue/delay
# values remain admissible because a real detached/outage interval can grow
# them far beyond a calibration percentile without making the observation
# invalid.
PHYSICAL_PLAUSIBILITY_BOUNDS: dict[str, tuple[float, float]] = {
    "p_jam": (0.0, 1.0),
    "jammer_class_none": (0.0, 1.0),
    "jammer_class_barrage": (0.0, 1.0),
    "jammer_class_partial_band": (0.0, 1.0),
    "jammer_class_sweep": (0.0, 1.0),
    "jammer_class_intermittent": (0.0, 1.0),
    "jammer_class_reactive": (0.0, 1.0),
    "sinr_db": (-100.0, 100.0),
    "cqi": (0.0, 15.0),
    "rsrp_dbfs": (-200.0, 100.0),
    "rsrq_db": (-100.0, 100.0),
    "bler": (0.0, 1.0),
    "harq_nack_ratio": (0.0, 1.0),
    "dl_throughput_mbps": (0.0, 100.0),
    "rlc_drop_rate": (0.0, 1.0),
    "rlc_sdu_delay_ms": (0.0, 10_000_000.0),
    "prb_utilization": (0.0, 1.0),
    "clean_prb_fraction": (0.0, 1.0),
    "scheduler_queue_backlog_bytes": (0.0, float(2**63 - 1)),
    "jammer_occupancy_fraction": (0.0, 1.0),
    "jammer_subband_start_norm": (0.0, 1.0),
    "jammer_subband_end_norm": (0.0, 1.0),
    "previous_action_a0": (0.0, 1.0),
    "previous_action_a1": (0.0, 1.0),
    "previous_action_a2": (0.0, 1.0),
    "previous_action_a3": (0.0, 1.0),
    "previous_action_a4": (0.0, 1.0),
}


def _physical_plausibility_violations(raw_state: dict[str, float]) -> dict[str, dict[str, float]]:
    violations: dict[str, dict[str, float]] = {}
    for name in FEATURE_NAMES:
        value = float(raw_state[name])
        lower, upper = PHYSICAL_PLAUSIBILITY_BOUNDS[name]
        if not np.isfinite(value) or value < lower or value > upper:
            violations[name] = {"value": value, "physical_low": lower, "physical_high": upper}
    probability_sum = sum(float(raw_state[f"jammer_class_{name}"]) for name in JAMMER_CLASSES)
    if not np.isclose(probability_sum, 1.0, atol=1e-6):
        violations["jammer_class_probability_sum"] = {
            "value": probability_sum, "physical_low": 1.0, "physical_high": 1.0,
        }
    previous_sum = sum(float(raw_state[f"previous_action_a{action}"]) for action in range(5))
    if not np.isclose(previous_sum, 1.0, atol=1e-12):
        violations["previous_action_one_hot_sum"] = {
            "value": previous_sum, "physical_low": 1.0, "physical_high": 1.0,
        }
    return violations


def _protected_session_active() -> bool:
    # This is an observed client-process check; an inactive protected flow
    # makes every mitigation action except A0 infeasible.
    return subprocess.run(
        ["sudo", "docker", "exec", "oran-srsue", "pgrep", "-x", "iperf3"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def _telemetry_window(metrics: Path, srsue: Path, sensor: Path, start_ns: int, end_ns: int) -> tuple[dict[str, float], dict[str, Any]] | None:
    scheduler, rlc = _load_ocudu(metrics)
    receiver = _load_srsue(srsue, allow_unbounded_sinr=
        os.environ.get("ORAN_RECEIVER_MEASUREMENT_PROFILE") == "native_positive_infinity_v1")
    scheduler_point, rlc_point, receiver_point = (
        _latest(scheduler, start_ns, end_ns), _latest(rlc, start_ns, end_ns), _latest(receiver, start_ns, end_ns),
    )
    sensor_points = [item for item in _read_json_lines(sensor) if start_ns <= int(item.get("timestamp_ns", -1)) < end_ns]
    sensor_point = sensor_points[-1] if sensor_points else None
    # A receiver can truthfully have no finite RF sample while detached.  The
    # scheduler, RLC and measured-IQ sources must still exist; missing receiver
    # features are then filled only by the frozen causal imputer and every
    # substituted controller input is recorded in ``telemetry_provenance``.
    if not all((scheduler_point, rlc_point, sensor_point)):
        return None
    receiver_values = {} if receiver_point is None else dict(receiver_point.values)
    receiver_attached = receiver_values.pop("receiver_attached", None)
    unbounded_sinr = receiver_values.pop("sinr_positive_infinity_observed", None) == 1.0
    unavailable_sentinels: dict[str, float] = {}
    # A detached srsUE can emit finite placeholder RF values (for example
    # dl_snr=113 interleaved with NaN).  The native is_attached=0 flag makes
    # those values receiver-unavailable observations, not physical RF
    # measurements.  Preserve every finite placeholder in the decision audit,
    # exclude all receiver-derived controller features, and let the already
    # frozen causal imputer fill them with explicit provenance.
    if receiver_attached is not None and float(receiver_attached) < 0.5:
        for name in (
            "sinr_db", "rsrp_dbfs", "rsrq_db", "rssi_dbfs",
            "ue_dl_throughput_mbps", "ue_dl_bler", "ue_ul_backlog_bytes",
        ):
            if name in receiver_values:
                unavailable_sentinels[name] = float(receiver_values.pop(name))
    telemetry = {**receiver_values, **scheduler_point.values, **rlc_point.values}
    # srsUE emits CQI=-1 when the receiver has no report (typically after
    # detachment).  It is a native missing-value sentinel, not a physically
    # measured CQI. Preserve it in the decision audit, remove it from the
    # controller feature map, and let the already-frozen causal imputer supply
    # the input with explicit provenance.
    if "cqi" in telemetry and float(telemetry["cqi"]) < 0.0:
        unavailable_sentinels["cqi"] = float(telemetry.pop("cqi"))
    if "ue_dl_bler" in telemetry:
        telemetry["bler"] = telemetry.pop("ue_dl_bler")
    if "gnb_dl_throughput_mbps" in telemetry:
        telemetry["dl_throughput_mbps"] = telemetry.pop("gnb_dl_throughput_mbps")
    elif "ue_dl_throughput_mbps" in telemetry:
        telemetry["dl_throughput_mbps"] = telemetry["ue_dl_throughput_mbps"]
    telemetry["clean_prb_fraction"] = float(sensor_point["clean_prb_fraction"])
    return telemetry, {
        "sensor": sensor_point,
        "current_mcs_index": int(round(scheduler_point.values["current_dl_mcs"])),
        "source_timestamps_ns": {
            "ocudu_scheduler": scheduler_point.timestamp_ns,
            "ocudu_rlc": rlc_point.timestamp_ns,
            "srsue": None if receiver_point is None else receiver_point.timestamp_ns,
            "iq_spectral_estimator": int(sensor_point["timestamp_ns"]),
        },
        "receiver_attached": receiver_attached,
        "receiver_measurement_qualifiers": {"sinr_db": "native_positive_infinity_unbounded"} if unbounded_sinr else {},
        "receiver_unavailable_sentinels": unavailable_sentinels,
    }


def _detector_output(detector: Any, telemetry: dict[str, float]) -> tuple[float, dict[str, float]]:
    probabilities = detector.predict_proba(np.asarray([[telemetry[name] for name in DETECTOR_FEATURES]], dtype=float))[0]
    classes = [int(value) for value in detector.classes_]
    distribution = {name: 0.0 for name in JAMMER_CLASSES}
    for index, probability in zip(classes, probabilities, strict=True):
        distribution[JAMMER_CLASSES[index]] = float(probability)
    return float(1.0 - distribution["none"]), distribution


def _load_detector(path: Path) -> Any:
    """Load the live-detector artifact without guessing its schema.

    ``train_live_detector.py`` persists a wrapper so the feature order and
    class semantics travel with the fitted estimator. Older development
    artifacts may contain the estimator directly; both forms are accepted,
    but every estimator must expose the scikit-learn probability interface.
    """
    loaded = joblib.load(path)
    detector = loaded
    if isinstance(loaded, dict):
        if tuple(loaded.get("feature_names") or ()) != DETECTOR_FEATURES:
            raise ValueError("live detector feature order differs from the xApp telemetry contract")
        if tuple(loaded.get("class_names") or ()) != JAMMER_CLASSES:
            raise ValueError("live detector class order differs from the methodology lock")
        detector = loaded.get("model")
    if detector is None or not hasattr(detector, "predict_proba") or not hasattr(detector, "classes_"):
        raise ValueError("detector artifact does not expose predict_proba/classes_")
    return detector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocudu-metrics", type=Path, required=True)
    parser.add_argument("--srsue-metrics", type=Path, required=True)
    parser.add_argument("--iq-sensor", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument(
        "--controller-kind",
        choices=("md3qn_per", "standard_dqn", "prb_only_masked_dqn", "masked_ppo", "heuristic"),
        default="md3qn_per",
        help="Named checkpoint architecture; architecture inference is prohibited.",
    )
    parser.add_argument("--state-schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, required=True)
    parser.add_argument("--xai", action="store_true")
    parser.add_argument(
        "--radio-window-driven",
        action="store_true",
        help="Trigger one decision per measured one-second IQ/radio window.",
    )
    parser.add_argument(
        "--wall-timeout-seconds",
        type=float,
        default=900.0,
        help="Safety timeout while waiting for measured radio windows.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("policy", "safeguarded_policy", "forced_exploration", "bounded_action_probe", "fixed_action", "ppo_sampling"),
        default="policy",
        help="policy is required for evaluation; the two exploration modes are development-only.",
    )
    parser.add_argument(
        "--ood-guard-profile",
        choices=("training_quantile_fail_closed", "physical_plausibility_with_shift_audit"),
        default="training_quantile_fail_closed",
        help=("The versioned confirmatory profile audits p01/p99 shift but rejects only "
              "non-finite or physically impossible telemetry."),
    )
    parser.add_argument(
        "--forced-action-id",
        type=int,
        choices=range(5),
        help="required only with --selection-mode forced_exploration",
    )
    parser.add_argument("--probe-start-window", type=int)
    parser.add_argument("--probe-stop-window", type=int)
    args = parser.parse_args()
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if args.wall_timeout_seconds <= 0:
        parser.error("--wall-timeout-seconds must be positive")
    if args.selection_mode in {"forced_exploration", "bounded_action_probe", "fixed_action"} and args.forced_action_id is None:
        parser.error("--forced-action-id is required for forced_exploration/bounded_action_probe/fixed_action")
    if args.selection_mode == "policy" and args.forced_action_id is not None:
        parser.error("--forced-action-id is only allowed for forced_exploration/fixed_action")
    if args.selection_mode == "ppo_sampling" and args.forced_action_id is not None:
        parser.error("--forced-action-id is not allowed for ppo_sampling")
    if args.selection_mode == "ppo_sampling" and args.controller_kind != "masked_ppo":
        parser.error("ppo_sampling requires --controller-kind masked_ppo")
    if args.selection_mode == "bounded_action_probe":
        if args.probe_start_window is None or args.probe_stop_window is None:
            parser.error("bounded_action_probe requires --probe-start-window and --probe-stop-window")
        if not 0 <= args.probe_start_window < args.probe_stop_window <= args.duration_seconds:
            parser.error("invalid bounded action-probe interval")
    elif args.probe_start_window is not None or args.probe_stop_window is not None:
        parser.error("probe window arguments are only valid for bounded_action_probe")
    if args.xai and args.controller_kind != "md3qn_per":
        parser.error("XAI is implemented only for the frozen M-D3QN-PER controller")
    schema, lower, upper, raw_low, raw_high = _load_schema(args.state_schema)
    imputer = CausalStateImputer(schema)
    network, score_type = _load_controller(args.controller, args.controller_kind)
    action_adapter = ActionAdapter(OcuduWebSocketResearchBackend(), {})
    runtime = DecisionRuntime(
        network, StateBuilder(schema), action_adapter,
        np.clip(lower, schema.clip_low, schema.clip_high), np.clip(upper, schema.clip_low, schema.clip_high),
        np.asarray([name in MUTABLE_FEATURES for name in FEATURE_NAMES], dtype=bool),
    )
    detector = _load_detector(args.detector)
    safety_guard = UncertaintySafetyGuard() if args.selection_mode == "safeguarded_policy" else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = Action.NO_OP
    window_start = time.time_ns()
    radio_deadline = time.monotonic() + args.wall_timeout_seconds
    previous_sensor_timestamp = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for window_index in range(args.duration_seconds):
            reconciliation_ms = 0.0
            reconciliation = None
            if not action_adapter.nontrivial_authority_ready:
                # A genuine baseline readback is required at startup and after
                # uncertainty. This cannot be inferred from the A0 request.
                restoration = action_adapter.execute(Action.NO_OP, np.ones(5, dtype=bool))
                reconciliation = asdict(restoration)
                reconciliation_ms = restoration.latency_ms
                runtime.observe_execution(restoration)
                previous = action_adapter.last_verified_action
            if args.radio_window_driven:
                matched_sensor: dict[str, Any] | None = None
                while time.monotonic() < radio_deadline:
                    for candidate in _read_json_lines(args.iq_sensor):
                        if int(candidate.get("window_index", -1)) == window_index:
                            matched_sensor = candidate
                    if matched_sensor is not None:
                        break
                    time.sleep(0.05)
                if matched_sensor is None:
                    window_end = time.time_ns()
                    record: dict[str, Any] = {
                        "window_index": window_index,
                        "window_start_ns": previous_sensor_timestamp,
                        "window_end_ns": window_end,
                        "evidence_class": "development_live_exploration" if args.selection_mode == "forced_exploration" else ("locked_confirmatory" if args.selection_mode == "fixed_action" else "live_controller_integration"),
                        "selection_mode": args.selection_mode,
                        "status": "SKIPPED_MISSING_MEASURED_RADIO_WINDOW",
                    }
                    if safety_guard is not None:
                        record["safety_guard"] = safety_guard.reject_invalid("missing_measured_radio_window").as_dict()
                        record["restoration"] = asdict(action_adapter.execute(Action.NO_OP, np.ones(5, dtype=bool)))
                    stream.write(json.dumps(record, sort_keys=True) + "\n"); stream.flush()
                    continue
                window_start = previous_sensor_timestamp
                window_end = int(matched_sensor["timestamp_ns"]) + 1
                previous_sensor_timestamp = window_end
            else:
                window_end = window_start + 1_000_000_000
                wait_seconds = (window_end - time.time_ns()) / 1_000_000_000
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
            processing_start = time.perf_counter_ns()
            found = _telemetry_window(args.ocudu_metrics, args.srsue_metrics, args.iq_sensor, window_start, window_end)
            record: dict[str, Any] = {"window_index": window_index, "window_start_ns": window_start, "window_end_ns": window_end, "evidence_class": "development_live_exploration" if args.selection_mode == "forced_exploration" else ("locked_confirmatory" if args.selection_mode == "fixed_action" else "live_controller_integration"), "selection_mode": args.selection_mode}
            if not action_adapter.nontrivial_authority_ready:
                # No previous-action one-hot may pretend that UNKNOWN means A0.
                # Retain the observed window and explicit unavailable proposal.
                record.update({
                    "status": "STATE_UNVERIFIED_NO_INFERENCE",
                    "policy_proposal": None,
                    "previous_action_state_status": "UNKNOWN",
                    "baseline_reconciliation": reconciliation,
                    "baseline_reconciliation_ms": reconciliation_ms,
                })
                if safety_guard is not None:
                    record["safety_guard"] = safety_guard.reject_invalid("applied_state_lockout_restore_baseline").as_dict()
                stream.write(json.dumps(record, sort_keys=True)+"\n"); stream.flush()
                if not args.radio_window_driven:
                    window_start = window_end
                continue
            if found is None:
                record.update({"status": "SKIPPED_INCOMPLETE_MEASURED_TELEMETRY"})
                if safety_guard is not None:
                    record["safety_guard"] = safety_guard.reject_invalid("missing_measured_telemetry").as_dict()
                    record["restoration"] = asdict(action_adapter.execute(Action.NO_OP, np.ones(5, dtype=bool)))
            else:
                telemetry, context_data = found
                guard_measured = {
                    name: telemetry.get(name, float("nan"))
                    for name in ("dl_throughput_mbps", "bler", "rlc_sdu_delay_ms")
                }
                telemetry, imputation = imputer.fill(telemetry, DETECTOR_FEATURES)
                p_jam, distribution = _detector_output(detector, telemetry)
                sensor = context_data["sensor"]
                localized = bool(sensor["localized_jammer_region_available"])
                context = ActionContext(
                    _protected_session_active(), localized, float(telemetry["clean_prb_fraction"]), 0.15,
                    context_data["current_mcs_index"], 2, 0.30, 0.70, 0.10, 0.30,
                )
                decision_input = DecisionInput(
                    window_start, window_end, telemetry, p_jam, distribution,
                    float(sensor["jammer_occupancy_fraction"]), float(sensor["jammer_subband_start_norm"]),
                    float(sensor["jammer_subband_end_norm"]), previous, context,
                )
                raw_state = runtime._assemble_raw_state(decision_input)
                distribution_shift = _out_of_distribution(raw_state, raw_low, raw_high)
                if args.ood_guard_profile == "physical_plausibility_with_shift_audit":
                    violations = _physical_plausibility_violations(raw_state)
                else:
                    violations = distribution_shift
                record.update({"detector_p_jam": p_jam, "detector_distribution": distribution,
                               "iq_localization": sensor, "source_timestamps_ns": context_data["source_timestamps_ns"],
                               "receiver_attached": context_data["receiver_attached"],
                               "receiver_measurement_qualifiers": context_data["receiver_measurement_qualifiers"],
                               "receiver_unavailable_sentinels": context_data["receiver_unavailable_sentinels"],
                               "telemetry_provenance": imputation, "raw_state": raw_state,
                               "ood_guard_profile": args.ood_guard_profile})
                if distribution_shift:
                    record["distribution_shift_warnings"] = distribution_shift
                # The OOD gate remains fail-closed for policy/fixed-action
                # evaluation.  Development-only forced/PPO exploration may
                # continue with the StateBuilder's bounded clipping so that
                # an explicitly requested action can be measured; the raw
                # violations stay attached to the auditable record.  This
                # never changes confirmatory behavior and never substitutes a
                # different action.
                if violations and args.selection_mode not in {"forced_exploration", "ppo_sampling"}:
                    record.update({"status": "SKIPPED_OUT_OF_DISTRIBUTION", "ood_violations": violations})
                    if safety_guard is not None:
                        record["safety_guard"] = safety_guard.reject_invalid("invalid_physical_measurement").as_dict()
                        record["restoration"] = asdict(action_adapter.execute(Action.NO_OP, np.ones(5, dtype=bool)))
                else:
                    if violations:
                        record["ood_violations"] = violations
                    forced_action = Action(args.forced_action_id) if args.selection_mode in {"forced_exploration", "fixed_action"} else None
                    if args.selection_mode == "bounded_action_probe":
                        in_probe = args.probe_start_window <= window_index < args.probe_stop_window
                        forced_action = Action(args.forced_action_id) if in_probe else Action.NO_OP
                        record["bounded_action_probe"] = {
                            "probe_start_window": args.probe_start_window,
                            "probe_stop_window": args.probe_stop_window,
                            "requested_action": Action(args.forced_action_id).name,
                            "in_probe_interval": in_probe,
                            "development_only": True,
                        }
                    if safety_guard is not None:
                        protected_active = _protected_session_active()
                        guard_started = time.perf_counter_ns()
                        safety = safety_guard.decide(SafetyObservation(
                            window_index=window_index,
                            source_id=str(sensor.get("source_id", "iq_spectral_estimator")),
                            source_epoch=str(sensor.get("source_epoch", sensor.get("process_start_ns", args.output.parent.name))),
                            sequence_number=int(sensor.get("sequence_number", sensor.get("window_index", window_index))),
                            measurement_window_id=str(sensor.get("measurement_window_id", sensor.get("timestamp_ns", window_index))),
                            source_timestamp_ns=int(sensor["timestamp_ns"]),
                            protected_session_active=protected_active,
                            spectral_reference_ready=bool(sensor.get("spectral_reference_ready", False)),
                            spectral_excess_fraction=float(sensor.get("spectral_excess_fraction", 0.0)),
                            spectral_divergence=float(sensor.get("spectral_divergence", 0.0)),
                            throughput_mbps=float(guard_measured["dl_throughput_mbps"]),
                            bler=float(guard_measured["bler"]),
                            latency_ms=float(guard_measured["rlc_sdu_delay_ms"]),
                            previous_action=previous,
                        ), applied_state_verified=action_adapter.nontrivial_authority_ready)
                        forced_action = safety.forced_action
                        record["safety_guard"] = safety.as_dict()
                        record["guard_computation_ms"] = (time.perf_counter_ns()-guard_started)/1e6
                    ppo_behavior: dict[str, float] | None = None
                    if args.selection_mode == "ppo_sampling":
                        if not isinstance(network, _PPOLogitAdapter):
                            raise RuntimeError("masked PPO controller did not load the PPO adapter")
                        policy_state = runtime.state_builder.build(raw_state)
                        policy_mask = compute_action_mask(context)
                        forced_action, ppo_behavior = _sample_masked_ppo(network.policy, policy_state, policy_mask)
                    try:
                        result = runtime.decide(decision_input, explain=args.xai, forced_action=forced_action)
                    except ValueError as error:
                        # A predeclared exploration request may become
                        # infeasible when the live safety context changes. It
                        # is an explicit skipped outcome, never a crash or a
                        # substitution with another action.
                        record.update({"status": "SKIPPED_INFEASIBLE_EXPLORATION_ACTION", "reason": str(error)})
                    else:
                        # Previous-action state is updated only from independent
                        # readback. An unacknowledged or merely reported command
                        # must never be silently represented as verified A0.
                        previous = action_adapter.last_verified_action
                        if safety_guard is not None and not action_adapter.nontrivial_authority_ready:
                            record["post_execution_guard"] = safety_guard.reject_invalid("execution_uncertain_restore_baseline").as_dict()
                        decision_record = result.as_record()
                        if score_type == "policy_logits":
                            decision_record["policy_logits"] = decision_record.pop("q_values")
                        record.update({
                            "status": "DECISION",
                            "controller_kind": args.controller_kind,
                            "controller_score_type": score_type,
                            "decision": decision_record,
                        })
                        if ppo_behavior is not None:
                            record["ppo_behavior"] = ppo_behavior
            record["baseline_reconciliation"] = reconciliation
            record["baseline_reconciliation_ms"] = reconciliation_ms
            record["full_loop_processing_ms"] = (time.perf_counter_ns()-processing_start)/1e6 + reconciliation_ms
            logging_started = time.perf_counter_ns()
            stream.write(json.dumps(record, sort_keys=True) + "\n"); stream.flush()
            logging_ms = (time.perf_counter_ns()-logging_started)/1e6
            with args.output.with_name("loop-stage-timings.jsonl").open("a",encoding="utf-8") as timing_stream:
                timing_stream.write(json.dumps({"window_index":window_index,"logging_ms":logging_ms,"full_loop_with_logging_ms":record["full_loop_processing_ms"]+logging_ms})+"\n")
            if not args.radio_window_driven:
                window_start = window_end
    print(f"LIVE_XAPP_WINDOWS_COMPLETED={args.duration_seconds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
