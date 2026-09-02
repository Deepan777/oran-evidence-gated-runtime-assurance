#!/usr/bin/env python3
"""Generate non-blocking explanations from exact frozen live decisions.

The control action is executed and acknowledged in the one-second live path.
This worker subsequently consumes that immutable decision state, mask and
Q-values.  It never changes an action or experimental measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
import torch

from analysis.xai_metrics import rank_stability, top_k_jaccard
from experiments.live.run_xapp_loop import MUTABLE_FEATURES, _load_controller, _load_schema
from src.controller.action_mask import Action
from src.controller.md3qn import mask_q_values
from src.environment.iq_calibrated_env import FEATURE_NAMES
from src.xai.contrastive import contrastive_attribution
from src.xai.counterfactual import projected_counterfactual


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("decision JSONL contains a non-object row")
            rows.append(item)
    return rows


def _gap(network: torch.nn.Module, state: torch.Tensor, selected: int, alternative: int) -> float:
    with torch.no_grad():
        q = network(state)
        return float((q[:, selected] - q[:, alternative]).item())


def _perturbation_effect(
    network: torch.nn.Module,
    state: torch.Tensor,
    indexes: np.ndarray,
    selected: int,
    alternative: int,
) -> float:
    changed = state.clone()
    changed[:, indexes.tolist()] = 0.0
    return abs(_gap(network, state, selected, alternative) - _gap(network, changed, selected, alternative))


def enrich_decision(
    record: dict[str, Any],
    network: torch.nn.Module,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    top_k: int = 3,
    stability_fraction: float = 0.01,
) -> dict[str, Any]:
    if record.get("status") != "DECISION":
        raise ValueError("XAI input row is not an accepted decision")
    decision = record.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("decision payload is missing")
    execution = decision.get("execution")
    if not isinstance(execution, dict) or execution.get("acknowledged") is not True or execution.get("failure") is not None:
        raise ValueError("XAI input action was not successfully acknowledged")
    state_values = np.asarray(decision.get("state"), dtype=np.float32)
    mask_values = np.asarray(decision.get("feasible_action_mask"), dtype=bool)
    if state_values.shape != (len(FEATURE_NAMES),) or mask_values.shape != (len(Action),):
        raise ValueError("decision state or action mask has the wrong shape")
    selected = int(Action[str(decision["selected_action"])])
    decision_sha256 = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if mask_values.sum() < 2:
        return {
            "schema_version": 1,
            "status": "XAI_NOT_APPLICABLE_SINGLE_FEASIBLE_ACTION",
            "window_index": int(record["window_index"]),
            "selected_action": Action(selected).name,
            "source_decision_sha256": decision_sha256,
            "reason": "contrastive explanation requires at least two feasible actions",
        }
    state = torch.from_numpy(state_values[None, :])
    mask = torch.from_numpy(mask_values[None, :])
    started = perf_counter_ns()
    attribution, selected_tensor, alternative_tensor, gap_tensor = contrastive_attribution(
        network, state, mask, steps=64
    )
    if int(selected_tensor.item()) != selected:
        raise ValueError("frozen decision action differs from recomputed policy action")
    alternative = int(alternative_tensor.item())
    values = attribution[0].detach().cpu().numpy().astype(float)
    order = np.argsort(np.abs(values))
    top = order[-top_k:]
    bottom = order[:top_k]
    top_effect = _perturbation_effect(network, state, top, selected, alternative)
    bottom_effect = _perturbation_effect(network, state, bottom, selected, alternative)
    faithfulness = float(top_effect / max(bottom_effect, 1e-12))

    span = np.asarray(upper - lower, dtype=np.float32)
    direction = np.where(np.arange(len(span)) % 2 == 0, 1.0, -1.0).astype(np.float32)
    mutable = np.asarray([name in MUTABLE_FEATURES for name in FEATURE_NAMES], dtype=bool)
    perturbed_values = np.clip(
        state_values + direction * span * float(stability_fraction) * mutable,
        lower,
        upper,
    ).astype(np.float32)
    perturbed = torch.from_numpy(perturbed_values[None, :])
    stable_attr, stable_selected, stable_alternative, _ = contrastive_attribution(
        network, perturbed, mask, steps=64
    )
    attribution_and_stability_ms = (perf_counter_ns() - started) / 1_000_000.0
    if int(stable_selected.item()) != selected or int(stable_alternative.item()) != alternative:
        stability_action_preserved = False
    else:
        stability_action_preserved = True
    stable_values = stable_attr[0].detach().cpu().numpy().astype(float)

    cf_started = perf_counter_ns()
    counterfactual = projected_counterfactual(
        network,
        state,
        mask,
        alternative,
        torch.from_numpy(lower.astype(np.float32)[None, :]),
        torch.from_numpy(upper.astype(np.float32)[None, :]),
        torch.from_numpy(mutable[None, :]),
        maximum_iterations=500,
    )
    counterfactual_ms = (perf_counter_ns() - cf_started) / 1_000_000.0
    total_ms = (perf_counter_ns() - started) / 1_000_000.0
    stability_spearman = rank_stability(values, stable_values)
    if not np.isfinite(stability_spearman):
        # Identical constant attribution vectors have no mathematically defined
        # rank correlation.  Record the conservative value zero, not NaN,
        # so the evidence gate cannot accidentally accept it.
        stability_spearman = 0.0
    return {
        "schema_version": 1,
        "status": "XAI_COMPLETE",
        "window_index": int(record["window_index"]),
        "selected_action": Action(selected).name,
        "alternative_action": Action(alternative).name,
        "q_gap": float(gap_tensor.item()),
        "contrastive_attribution": values.tolist(),
        "faithfulness_top_k_effect": top_effect,
        "faithfulness_bottom_k_effect": bottom_effect,
        "faithfulness_ratio": faithfulness,
        "stability_spearman": stability_spearman,
        "stability_top_k_jaccard": top_k_jaccard(values, stable_values, top_k),
        "stability_action_preserved": stability_action_preserved,
        "counterfactual_valid": bool(counterfactual.valid),
        "counterfactual_target_action": Action(counterfactual.target_action).name,
        "counterfactual_selected_action": Action(counterfactual.selected_action).name,
        "counterfactual_weighted_l1": float(counterfactual.weighted_l1),
        "counterfactual_iterations": int(counterfactual.iterations),
        "latency_ms": {
            "attribution_and_stability": attribution_and_stability_ms,
            "counterfactual_generation": counterfactual_ms,
            "total_xai_worker": total_ms,
        },
        "source_decision_sha256": decision_sha256,
        "source_decision_fields": {
            "window_start_ns": record.get("window_start_ns"),
            "window_end_ns": record.get("window_end_ns"),
            "controller_kind": record.get("controller_kind"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--state-schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()
    rows = _read_jsonl(args.decisions)
    if len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} decisions, found {len(rows)}")
    indexes = [int(row.get("window_index", -1)) for row in rows]
    if indexes != list(range(args.expected_rows)):
        raise SystemExit("decision windows are missing, duplicated, or out of order")
    network, score_type = _load_controller(args.controller, "md3qn_per")
    if score_type != "q_values":
        raise SystemExit("V6 proposed controller does not expose Q-values")
    schema, lower, upper, _, _ = _load_schema(args.state_schema)
    lower = np.clip(np.asarray(lower, dtype=np.float32), schema.clip_low, schema.clip_high)
    upper = np.clip(np.asarray(upper, dtype=np.float32), schema.clip_low, schema.clip_high)
    if np.any(lower > upper):
        raise SystemExit("normalized state bounds are invalid")
    output_rows = [enrich_decision(row, network, lower, upper) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({
        "status": "PASS",
        "rows": len(output_rows),
        "counterfactual_valid": sum(bool(row["counterfactual_valid"]) for row in output_rows),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
