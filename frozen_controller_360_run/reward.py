from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    throughput: float = 1.0
    latency: float = 1.0
    drop: float = 1.0
    sla_violation: float = 2.0
    action_cost: float = 0.10
    action_switch: float = 0.05

    def validate(self) -> None:
        values = tuple(self.__dict__.values())
        if any(value < 0.0 or not np.isfinite(value) for value in values):
            raise ValueError("reward weights must be finite and non-negative")


@dataclass(frozen=True)
class RewardInputs:
    throughput_norm: float
    latency_norm: float
    drop_norm: float
    sla_violation: float
    action_cost: float
    action_switched: bool

    def validate(self) -> None:
        values = (
            self.throughput_norm,
            self.latency_norm,
            self.drop_norm,
            self.sla_violation,
            self.action_cost,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("reward inputs must be finite")
        if self.action_cost < 0.0:
            raise ValueError("action cost must be non-negative")


def compute_reward(inputs: RewardInputs, weights: RewardWeights) -> dict[str, float]:
    """Compute the locked reward and return an auditable decomposition."""
    inputs.validate()
    weights.validate()
    components = {
        "reward_throughput": weights.throughput * inputs.throughput_norm,
        "reward_latency": -weights.latency * inputs.latency_norm,
        "reward_reliability": -weights.drop * inputs.drop_norm,
        "reward_service": -weights.sla_violation * inputs.sla_violation,
        "reward_action_cost": -weights.action_cost * inputs.action_cost,
        "reward_switching_cost": -weights.action_switch * float(inputs.action_switched),
    }
    components["total_reward"] = float(sum(components.values()))
    return {name: float(value) for name, value in components.items()}

