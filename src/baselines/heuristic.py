"""Deterministic, auditable heuristic multi-action baseline.

The heuristic is intentionally small and model-free.  It consumes the same
standardized 27-feature state as the learned policies and emits five scores;
the common runtime still applies the exact feasibility mask before choosing an
action.  The resulting baseline is therefore comparable in state, action,
and safety handling without pretending to be a learned checkpoint.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.environment.iq_calibrated_env import FEATURE_NAMES


class HeuristicQNetwork(nn.Module):
    """Rule-based score adapter for the locked five-action interface."""

    def __init__(self) -> None:
        super().__init__()
        # Register no trainable parameters: this is a fixed baseline.
        self.register_buffer("_zero", torch.zeros(1), persistent=False)
        self._index = {name: index for index, name in enumerate(FEATURE_NAMES)}

    def _feature(self, state: Tensor, name: str) -> Tensor:
        return state[..., self._index[name]]

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[1] != len(FEATURE_NAMES):
            raise ValueError("heuristic state has the wrong shape")
        p_jam = self._feature(state, "p_jam")
        # These are standardized values, so positive values indicate a
        # measured/calibrated departure above the training centre.
        localization = (
            self._feature(state, "jammer_occupancy_fraction")
            + self._feature(state, "jammer_subband_end_norm")
            - self._feature(state, "jammer_subband_start_norm")
        )
        quality_loss = (
            self._feature(state, "bler")
            + self._feature(state, "harq_nack_ratio")
            - self._feature(state, "sinr_db")
        )
        resource_pressure = (
            self._feature(state, "scheduler_queue_backlog_bytes")
            + self._feature(state, "prb_utilization")
        )
        # A0 remains a neutral safe option.  The small fixed costs discourage
        # unnecessary switching while preserving an auditable preference order.
        a1 = p_jam + localization - 0.20
        a2 = p_jam + quality_loss - 0.15
        a3 = p_jam + resource_pressure - 0.30
        a4 = a1 + a2 - 0.35
        return torch.stack((torch.zeros_like(p_jam), a1, a2, a3, a4), dim=-1)


__all__ = ["HeuristicQNetwork"]
