"""A deterministic IQ-derived calibration environment for pre-live development.

It generates a legitimate complex stream, injects the configured jammer using
the production waveform module, and derives observables from the resulting
signal power.  It is explicitly labelled *offline calibration*; it cannot
produce confirmatory live-testbed evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.controller.action_mask import Action, ActionContext, compute_action_mask
from src.controller.reward import RewardInputs, RewardWeights, compute_reward
from src.jammer.waveforms import JammerConfig, JammerType, inject_jammer


FEATURE_NAMES = (
    "p_jam", "jammer_class_none", "jammer_class_barrage", "jammer_class_partial_band",
    "jammer_class_sweep", "jammer_class_intermittent", "jammer_class_reactive", "sinr_db",
    "cqi", "rsrp_dbfs", "rsrq_db", "bler", "harq_nack_ratio", "dl_throughput_mbps",
    "rlc_drop_rate", "rlc_sdu_delay_ms", "prb_utilization", "clean_prb_fraction",
    "scheduler_queue_backlog_bytes", "jammer_occupancy_fraction", "jammer_subband_start_norm",
    "jammer_subband_end_norm", "previous_action_a0", "previous_action_a1", "previous_action_a2",
    "previous_action_a3", "previous_action_a4",
)


@dataclass(frozen=True)
class CalibrationScenario:
    scenario_id: str
    jammer_type: JammerType
    jsr_db: float
    load: str
    service_mix: str
    attenuation: str
    seed: int
    steps: int = 120


class IqCalibratedAntiJammingEnv:
    """Small deterministic environment used only for safe pre-live calibration."""
    metadata = {"evidence_class": "offline_iq_calibration", "live_testbed": False}

    def __init__(self, scenario: CalibrationScenario, reward_weights: RewardWeights | None = None) -> None:
        self.scenario = scenario
        self.reward_weights = reward_weights or RewardWeights()
        self.rng = np.random.default_rng(scenario.seed)
        self.step_index = 0
        self.previous_action = Action.NO_OP
        self.last_metrics: dict[str, float] = {}

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self.step_index, self.previous_action = 0, Action.NO_OP
        self.last_metrics = self._metrics(Action.NO_OP)
        return self._state(self.last_metrics), {"scenario": self.scenario.scenario_id, **self.metadata}

    def action_mask(self) -> np.ndarray:
        m = self.last_metrics
        return compute_action_mask(ActionContext(
            protected_session_active=True,
            localized_jammer_region_available=self.scenario.jammer_type in {JammerType.PARTIAL_BAND, JammerType.SWEEP},
            clean_prb_fraction=float(m["clean_prb_fraction"]), clean_prb_fraction_min=0.10,
            current_mcs_index=int(m["cqi"]), minimum_mcs_index=2,
            protected_quota_fraction=0.35, protected_quota_max_fraction=0.65,
            quota_increment_fraction=0.15, total_allocated_quota_fraction=0.65,
        ))

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = Action(action_id)
        mask = self.action_mask()
        if not mask[action]:
            raise ValueError(f"infeasible action selected: {action.name}")
        before, previous_action = self.last_metrics, self.previous_action
        self.previous_action = action
        self.step_index += 1
        self.last_metrics = self._metrics(action)
        reward_parts = compute_reward(RewardInputs(
            throughput_norm=self.last_metrics["dl_throughput_mbps"] / 100.0,
            latency_norm=min(self.last_metrics["rlc_sdu_delay_ms"] / 100.0, 2.0),
            drop_norm=self.last_metrics["rlc_drop_rate"],
            sla_violation=float(self.last_metrics["rlc_sdu_delay_ms"] > 20.0),
            action_cost={Action.NO_OP: 0.0, Action.PRB_AVOID_SHIFT: 0.20, Action.MCS_BACKOFF: 0.15,
                         Action.SLICE_PRB_BOOST: 0.30, Action.COMPOSITE_LOCAL_DEFENSE: 0.35}[action],
            action_switched=action != previous_action,
        ), self.reward_weights)
        terminal = self.step_index >= self.scenario.steps
        info: dict[str, Any] = {"metrics": self.last_metrics, "previous_metrics": before,
                                "action": action.name, "mask": mask.tolist(), "reward_components": reward_parts,
                                "evidence_class": "offline_iq_calibration"}
        return self._state(self.last_metrics), reward_parts["total_reward"], terminal, info

    def _metrics(self, action: Action) -> dict[str, float]:
        n = 4096
        phase = self.rng.uniform(-np.pi, np.pi)
        legitimate = np.exp(1j * (phase + np.linspace(0.0, 15.0, n))).astype(np.complex64)
        jam_on = 20 <= self.step_index < max(21, self.scenario.steps - 20)
        jammer = JammerType.NONE if not jam_on else self.scenario.jammer_type
        result = inject_jammer(legitimate, JammerConfig(jammer, self.scenario.jsr_db, self.scenario.seed + self.step_index))
        signal_power = float(np.mean(np.abs(legitimate) ** 2))
        interference_power = float(np.mean(np.abs(result.jammer_iq) ** 2)) + 0.01
        sinr = 10.0 * np.log10(signal_power / interference_power)
        load_fraction = {"low": 0.30, "medium": 0.60, "high": 0.90}[self.scenario.load]
        attenuation_penalty = 4.0 if self.scenario.attenuation == "degraded" else 0.0
        mitigation_gain = {Action.NO_OP: 0.0, Action.PRB_AVOID_SHIFT: 5.0 if jammer in {JammerType.PARTIAL_BAND, JammerType.SWEEP} else 0.5,
                           Action.MCS_BACKOFF: 2.5, Action.SLICE_PRB_BOOST: 1.5,
                           Action.COMPOSITE_LOCAL_DEFENSE: 6.5 if jammer in {JammerType.PARTIAL_BAND, JammerType.SWEEP} else 3.0}[action]
        effective_sinr = sinr - attenuation_penalty + mitigation_gain
        bler = float(np.clip(1.0 / (1.0 + np.exp((effective_sinr - 1.5) / 2.2)), 0.0, 0.99))
        throughput = float(max(0.2, (effective_sinr + 12.0) * 5.0 * (1.0 - 0.45 * load_fraction)))
        delay = float(5.0 + 80.0 * bler + 20.0 * load_fraction)
        occupancy = float(result.active_mask.mean())
        clean = float(1.0 - occupancy * (0.55 if jammer in {JammerType.PARTIAL_BAND, JammerType.SWEEP} else 0.9))
        cls = [float(jammer is item) for item in (JammerType.NONE, JammerType.BARRAGE, JammerType.PARTIAL_BAND,
                                                    JammerType.SWEEP, JammerType.INTERMITTENT, JammerType.REACTIVE)]
        return {"p_jam": float(1.0 - np.exp(-max(0.0, -effective_sinr + 7.0) / 8.0)),
            **dict(zip(FEATURE_NAMES[1:7], cls, strict=True)), "sinr_db": float(effective_sinr),
            "cqi": float(np.clip(round((effective_sinr + 8.0) / 2.0), 0, 15)), "rsrp_dbfs": -63.0 + effective_sinr,
            "rsrq_db": -17.0 + effective_sinr / 3.0, "bler": bler, "harq_nack_ratio": bler * 0.8,
            "dl_throughput_mbps": throughput, "rlc_drop_rate": bler * 0.7, "rlc_sdu_delay_ms": delay,
            "prb_utilization": load_fraction, "clean_prb_fraction": clean,
            "scheduler_queue_backlog_bytes": float(1000.0 + 20000.0 * load_fraction * bler),
            "jammer_occupancy_fraction": occupancy, "jammer_subband_start_norm": 0.25,
            "jammer_subband_end_norm": 0.50, **{f"previous_action_a{i}": float(self.previous_action.value == i) for i in range(5)}}

    def _state(self, metrics: dict[str, float]) -> np.ndarray:
        return np.asarray([metrics[name] for name in FEATURE_NAMES], dtype=np.float32)
