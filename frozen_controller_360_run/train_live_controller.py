#!/usr/bin/env python3
"""Boundedly fine-tune M-D3QN-PER from acknowledged live exploration.

Only development-live-exploration rows are accepted.  The controller never
reads jammer truth: detector-derived state, feasibility masks, acknowledged
actions, and the next measured telemetry window form every replay transition.
This creates a validation candidate, not a confirmatory-frozen policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiments.collection.build_live_run_records import build_records
from src.baselines.learning import train_standard_dqn
from src.controller.action_mask import Action
from src.controller.replay import PrioritizedReplayBuffer
from src.controller.reward import RewardInputs, RewardWeights, compute_reward
from src.controller.state import StateBuilder, StateSchema
from src.controller.training import train_md3qn_per
from src.environment.iq_calibrated_env import FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[2]
ACTION_COST = {Action.NO_OP: 0.0, Action.PRB_AVOID_SHIFT: 0.20, Action.MCS_BACKOFF: 0.15,
               Action.SLICE_PRB_BOOST: 0.30, Action.COMPOSITE_LOCAL_DEFENSE: 0.35}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_schema(path: Path) -> StateSchema:
    data = json.loads(path.read_text(encoding="utf-8"))
    if tuple(data.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("live state schema feature order differs from methodology lock")
    return StateSchema(FEATURE_NAMES, np.asarray(data["means"], dtype=np.float64),
                       np.asarray(data["scales"], dtype=np.float64),
                       float(data.get("clip_low", -8.0)), float(data.get("clip_high", 8.0)))


def _decision_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing xApp action log: {path}")
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("status") != "DECISION":
            continue
        index = int(item["window_index"])
        if index in rows:
            raise ValueError(f"duplicate decision window {index} in {path}")
        rows[index] = item
    return rows


def _manifest(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows or {row.get("evidence_class") for row in rows} != {"development_live_exploration"}:
        raise ValueError("live controller training accepts only development_live_exploration manifest rows")
    return rows


def _raw_records(evidence: Path) -> pd.DataFrame:
    metadata_path = evidence / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"missing metadata: {evidence}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("evidence_class") != "development_live_exploration":
        raise ValueError(f"unexpected evidence class for {metadata.get('run_id')}")
    record = build_records(evidence, metadata)
    if not bool(record.complete_controller_state.all()):
        raise ValueError(f"incomplete controller telemetry in {metadata['run_id']}")
    return record.set_index("window_index", drop=False)


def _normalizers(rows: list[pd.DataFrame]) -> tuple[float, float]:
    frame = pd.concat(rows, ignore_index=True)
    throughput = float(np.quantile(pd.to_numeric(frame.dl_throughput_mbps), 0.95))
    latency = float(np.quantile(pd.to_numeric(frame.rlc_sdu_delay_ms), 0.95))
    if not np.isfinite(throughput) or not np.isfinite(latency) or throughput <= 0 or latency <= 0:
        raise ValueError("training telemetry cannot define reward normalizers")
    return throughput, latency


def _transition_reward(next_row: pd.Series, action: Action, previous_action: Action,
                       throughput_scale: float, latency_scale: float) -> float:
    value = compute_reward(RewardInputs(
        throughput_norm=float(next_row.dl_throughput_mbps) / throughput_scale,
        latency_norm=float(next_row.rlc_sdu_delay_ms) / latency_scale,
        drop_norm=float(next_row.rlc_drop_rate),
        sla_violation=float(float(next_row.rlc_sdu_delay_ms) > 20.0),
        action_cost=ACTION_COST[action], action_switched=action != previous_action,
    ), RewardWeights())
    return float(value["total_reward"])


def _collect(rows: list[dict[str, str]], raw_root: Path, schema: StateSchema,
             *, partition: str, normalizers: tuple[float, float]) -> tuple[PrioritizedReplayBuffer, dict[str, Any]]:
    buffer = PrioritizedReplayBuffer(20_000, len(FEATURE_NAMES), 5, seed=2101)
    builder = StateBuilder(schema)
    actions: set[int] = set(); decision_count = 0
    for row in rows:
        if row["partition"] != partition:
            continue
        evidence = raw_root / f"iq-broker-live-{row['run_id']}"
        raw = _raw_records(evidence)
        decisions = _decision_rows(evidence / "live-xapp-decisions.jsonl")
        for index in sorted(decisions):
            current = decisions[index]["decision"]
            next_item = decisions.get(index + 1)
            if next_item is None or index + 1 not in raw.index:
                continue
            next_decision = next_item["decision"]
            execution = current["execution"]
            if not bool(execution.get("acknowledged")):
                continue
            action = Action[execution["executed_action"]]
            previous = Action.NO_OP if index == 0 else Action[
                decisions.get(index - 1, {"decision": {"execution": {"executed_action": "NO_OP"}}})["decision"]["execution"]["executed_action"]
            ]
            state = builder.build({name: float(current["raw_state"][name]) for name in FEATURE_NAMES})
            next_state = builder.build({name: float(next_decision["raw_state"][name]) for name in FEATURE_NAMES})
            mask = np.asarray(next_decision["feasible_action_mask"], dtype=bool)
            if mask.shape != (5,) or not mask.any():
                raise ValueError("invalid next-state feasibility mask in exploration log")
            reward = _transition_reward(raw.loc[index + 1], action, previous, *normalizers)
            terminal = index + 2 not in decisions
            buffer.add(state, int(action), reward, next_state, terminal, mask)
            actions.add(int(action)); decision_count += 1
    return buffer, {"transitions": decision_count, "executed_action_ids": sorted(actions)}


def _prb_only_buffer(source: PrioritizedReplayBuffer) -> tuple[PrioritizedReplayBuffer, dict[str, Any]]:
    """Restrict B5 to its declared A0/A1 action space without unsafe replay.

    Only transitions whose executed action belongs to the PRB-only space are
    admitted. Target feasibility masks are restricted to that same space, so
    the learned comparator cannot obtain value from A2--A4 while being
    evaluated as a PRB-only controller.
    """
    replay = PrioritizedReplayBuffer(20_000, source.states.shape[1], 5, seed=2303)
    selected: set[int] = set()
    for index in range(source.size):
        action = int(source.actions[index])
        if action not in {int(Action.NO_OP), int(Action.PRB_AVOID_SHIFT)}:
            continue
        mask = source.next_masks[index].copy()
        mask[int(Action.MCS_BACKOFF):] = False
        if not bool(mask[int(Action.NO_OP)]):
            raise ValueError("PRB-only replay violates mandatory A0 feasibility")
        replay.add(source.states[index], action, float(source.rewards[index]), source.next_states[index],
                   bool(source.terminals[index]), mask)
        selected.add(action)
    return replay, {"transitions": replay.size, "executed_action_ids": sorted(selected),
                    "action_space": [int(Action.NO_OP), int(Action.PRB_AVOID_SHIFT)]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/metadata/manifests/live_exploration_runs.csv")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "results/raw")
    parser.add_argument("--state-schema", type=Path, default=ROOT / "results/training/live_controller/state_schema.json")
    parser.add_argument("--initialization", type=Path, default=ROOT / "results/training/live_controller/md3qn_per_live_initialization.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/training/live_controller")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    rows, schema = _manifest(args.manifest), _load_schema(args.state_schema)
    if not args.initialization.is_file():
        raise SystemExit("BLOCKED_LIVE_CONTROLLER_FINE_TUNING: missing offline-pretrained initialization")
    # Fit reward normalizers only on complete train runs, before touching
    # validation or locked-test action logs.
    train_raw = [_raw_records(args.raw_root / f"iq-broker-live-{row['run_id']}")
                 for row in rows if row["partition"] == "train"]
    normalizers = _normalizers(train_raw)
    replay, train_report = _collect(rows, args.raw_root, schema, partition="train", normalizers=normalizers)
    if replay.size < 128:
        raise SystemExit(f"BLOCKED_LIVE_CONTROLLER_FINE_TUNING: only {replay.size} eligible train transitions")
    if set(train_report["executed_action_ids"]) != set(range(5)):
        raise SystemExit(f"BLOCKED_LIVE_CONTROLLER_ACTION_COVERAGE: {train_report['executed_action_ids']}")
    initial = torch.load(args.initialization, map_location="cpu", weights_only=True)
    proposed, proposed_report = train_md3qn_per(replay, len(FEATURE_NAMES), 2201, steps=args.steps,
                                                batch_size=128, initialization_state_dict=initial)
    baseline, baseline_report = train_standard_dqn(replay, len(FEATURE_NAMES), 2202, steps=args.steps, batch_size=128)
    prb_replay, prb_report = _prb_only_buffer(replay)
    if prb_replay.size < 128 or set(prb_report["executed_action_ids"]) != {0, 1}:
        raise SystemExit(f"BLOCKED_PRB_ONLY_BASELINE_ACTION_COVERAGE: {prb_report}")
    prb_dqn, prb_dqn_report = train_standard_dqn(prb_replay, len(FEATURE_NAMES), 2203, steps=args.steps, batch_size=128)
    args.output.mkdir(parents=True, exist_ok=True)
    proposed_path = args.output / "md3qn_per_live_candidate.pt"
    baseline_path = args.output / "standard_dqn_live_candidate.pt"
    prb_path = args.output / "prb_only_masked_dqn_live_candidate.pt"
    torch.save(proposed.state_dict(), proposed_path); torch.save(baseline.state_dict(), baseline_path)
    torch.save(prb_dqn.state_dict(), prb_path)
    validation_replay, validation_report = _collect(rows, args.raw_root, schema, partition="validation", normalizers=normalizers)
    summary = {
        "status": "LIVE_CONTROLLER_BOUNDED_FINE_TUNING_COMPLETE",
        "not_confirmatory": True,
        "policy_status": "CANDIDATE_REQUIRES_LIVE_VALIDATION_AND_FREEZE",
        "reward_normalizers_training_only": {"throughput_p95": normalizers[0], "latency_p95": normalizers[1]},
        "train": train_report, "validation": validation_report,
        "proposed": {**proposed_report.__dict__, "file": str(proposed_path), "sha256": _digest(proposed_path)},
        "standard_dqn": {**baseline_report.__dict__, "file": str(baseline_path), "sha256": _digest(baseline_path)},
        "prb_only_masked_dqn": {**prb_dqn_report.__dict__, **prb_report, "file": str(prb_path), "sha256": _digest(prb_path)},
        "state_schema_sha256": _digest(args.state_schema), "initialization_sha256": _digest(args.initialization),
    }
    (args.output / "live_controller_fine_tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
