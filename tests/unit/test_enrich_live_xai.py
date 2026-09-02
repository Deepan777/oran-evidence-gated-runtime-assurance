from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

import analysis.enrich_live_xai as module
from src.environment.iq_calibrated_env import FEATURE_NAMES
from src.xai.counterfactual import CounterfactualResult


class _LinearScores(torch.nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        total = state.sum(dim=1, keepdim=True)
        return torch.cat((total * 0.0, total + 2.0, total + 1.0, total - 1.0, total - 2.0), dim=1)


def _record() -> dict:
    return {
        "status": "DECISION",
        "window_index": 0,
        "window_start_ns": 10,
        "window_end_ns": 20,
        "controller_kind": "md3qn_per",
        "decision": {
            "state": [0.1] * len(FEATURE_NAMES),
            "feasible_action_mask": [True] * 5,
            "selected_action": "PRB_AVOID_SHIFT",
            "execution": {"acknowledged": True, "failure": None},
        },
    }


def test_enrichment_is_action_immutable_and_auditable(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_attribution(network, state, mask, steps):
        calls["count"] += 1
        values = torch.arange(len(FEATURE_NAMES), dtype=torch.float32)[None, :] + calls["count"] * 0.01
        return values, torch.tensor([1]), torch.tensor([2]), torch.tensor([1.0])

    def fake_counterfactual(*args, **kwargs):
        state = args[1]
        return CounterfactualResult(state, True, 2, 2, 0.5, 4)

    monkeypatch.setattr(module, "contrastive_attribution", fake_attribution)
    monkeypatch.setattr(module, "projected_counterfactual", fake_counterfactual)
    record = _record()
    result = module.enrich_decision(
        record, _LinearScores(), np.full(len(FEATURE_NAMES), -8.0), np.full(len(FEATURE_NAMES), 8.0)
    )
    expected_hash = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert result["status"] == "XAI_COMPLETE"
    assert result["window_index"] == 0
    assert result["selected_action"] == "PRB_AVOID_SHIFT"
    assert result["source_decision_sha256"] == expected_hash
    assert np.isfinite(result["stability_spearman"])
    assert record["decision"]["selected_action"] == "PRB_AVOID_SHIFT"


def test_enrichment_rejects_unacknowledged_action() -> None:
    record = _record()
    record["decision"]["execution"]["acknowledged"] = False
    with pytest.raises(ValueError, match="not successfully acknowledged"):
        module.enrich_decision(
            record, _LinearScores(), np.full(len(FEATURE_NAMES), -8.0), np.full(len(FEATURE_NAMES), 8.0)
        )


def test_single_feasible_action_is_explicitly_not_applicable() -> None:
    record = _record()
    record["decision"]["feasible_action_mask"] = [True, False, False, False, False]
    record["decision"]["selected_action"] = "NO_OP"
    result = module.enrich_decision(
        record, _LinearScores(), np.full(len(FEATURE_NAMES), -8.0), np.full(len(FEATURE_NAMES), 8.0)
    )
    assert result["status"] == "XAI_NOT_APPLICABLE_SINGLE_FEASIBLE_ACTION"
    assert result["selected_action"] == "NO_OP"
