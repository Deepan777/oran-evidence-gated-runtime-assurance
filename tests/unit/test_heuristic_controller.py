from __future__ import annotations

import torch

from src.baselines.heuristic import HeuristicQNetwork
from src.environment.iq_calibrated_env import FEATURE_NAMES


def test_heuristic_controller_has_locked_action_interface() -> None:
    network = HeuristicQNetwork()
    values = network(torch.zeros((3, len(FEATURE_NAMES))))
    assert values.shape == (3, 5)
    assert torch.isfinite(values).all()
