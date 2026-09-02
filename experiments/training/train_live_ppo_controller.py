#!/usr/bin/env python3
"""Boundedly update the PPO comparator from its own measured live rollouts.

Forced-action records belong to off-policy Q-learning only.  This script
accepts exclusively ``ppo_sampling`` records, whose actions, masks,
log-probabilities and values came from the same frozen initial PPO actor.
The output remains a development candidate and cannot be a confirmatory
checkpoint until independent validation selection and an explicit freeze.
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
from experiments.training.train_live_controller import (
    ACTION_COST, _digest, _load_schema, _normalizers, _transition_reward,
)
from src.baselines.learning import MaskedPPOPolicy, ppo_loss
from src.controller.action_mask import Action
from src.controller.state import StateBuilder, StateSchema
from src.environment.iq_calibrated_env import FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _manifest(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows or {row.get("evidence_class") for row in rows} != {"development_live_ppo_exploration"}:
        raise ValueError("PPO live training accepts only its immutable on-policy exploration manifest")
    if {row.get("LIVE_XAPP_SELECTION_MODE") for row in rows} != {"ppo_sampling"}:
        raise ValueError("PPO live training requires sampled, not forced, actions")
    return rows


def _raw_records(evidence: Path) -> pd.DataFrame:
    metadata_path = evidence / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"missing metadata: {evidence}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("evidence_class") != "development_live_ppo_exploration":
        raise ValueError(f"unexpected PPO evidence class for {metadata.get('run_id')}")
    records = build_records(evidence, metadata)
    if not bool(records.complete_controller_state.all()):
        raise ValueError(f"incomplete controller telemetry in {metadata['run_id']}")
    return records.set_index("window_index", drop=False)


def _decision_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing PPO xApp log: {path}")
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("status") != "DECISION":
            continue
        if item.get("selection_mode") != "ppo_sampling" or item.get("controller_kind") != "masked_ppo":
            raise ValueError("PPO action log contains a non-on-policy decision")
        behavior = item.get("ppo_behavior")
        decision = item.get("decision")
        if not isinstance(behavior, dict) or not isinstance(decision, dict):
            raise ValueError("PPO action log lacks behaviour metadata")
        index = int(item["window_index"])
        if index in rows:
            raise ValueError(f"duplicate PPO decision window {index}")
        rows[index] = item
    return rows


def _advantages(rewards: list[float], values: list[float], terminals: list[bool], *, gamma: float = 0.99, gae_lambda: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    if not rewards or not (len(rewards) == len(values) == len(terminals)):
        raise ValueError("invalid PPO rollout lengths")
    advantages = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = 0.0 if terminals[index] else values[index + 1]
        continuation = 0.0 if terminals[index] else 1.0
        delta = rewards[index] + gamma * next_value * continuation - values[index]
        running = delta + gamma * gae_lambda * continuation * running
        advantages[index] = running
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def _collect(rows: list[dict[str, str]], raw_root: Path, schema: StateSchema, partition: str, normalizers: tuple[float, float]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    builder = StateBuilder(schema)
    collected: dict[str, list[np.ndarray] | list[float] | list[int]] = {
        "states": [], "masks": [], "actions": [], "old_log_probabilities": [], "advantages": [], "returns": [],
    }
    action_ids: set[int] = set(); rollouts = 0
    for row in rows:
        if row["partition"] != partition:
            continue
        evidence = raw_root / f"iq-broker-live-{row['run_id']}"
        raw, decisions = _raw_records(evidence), _decision_rows(evidence / "live-xapp-decisions.jsonl")
        indexes = sorted(decisions)
        episode_states: list[np.ndarray] = []; episode_masks: list[np.ndarray] = []; episode_actions: list[int] = []
        episode_log_probs: list[float] = []; episode_rewards: list[float] = []; episode_values: list[float] = []; terminals: list[bool] = []
        for position, index in enumerate(indexes):
            if index + 1 not in raw.index:
                continue
            item, next_item = decisions[index], decisions.get(index + 1)
            if next_item is None:
                continue
            decision, execution, behavior = item["decision"], item["decision"]["execution"], item["ppo_behavior"]
            if not bool(execution.get("acknowledged")):
                continue
            action = Action[execution["executed_action"]]
            state = builder.build({name: float(decision["raw_state"][name]) for name in FEATURE_NAMES})
            mask = np.asarray(decision["feasible_action_mask"], dtype=bool)
            if mask.shape != (5,) or not bool(mask[int(action)]):
                raise ValueError("PPO record action is not feasible under its logged mask")
            previous = Action.NO_OP if position == 0 else Action[decisions[indexes[position - 1]]["decision"]["execution"]["executed_action"]]
            episode_states.append(state); episode_masks.append(mask); episode_actions.append(int(action))
            episode_log_probs.append(float(behavior["action_log_probability"])); episode_values.append(float(behavior["value_estimate"]))
            episode_rewards.append(_transition_reward(raw.loc[index + 1], action, previous, *normalizers))
            terminals.append(position + 2 >= len(indexes))
            action_ids.add(int(action))
        if episode_states:
            advantages, returns = _advantages(episode_rewards, episode_values, terminals)
            collected["states"].extend(episode_states); collected["masks"].extend(episode_masks); collected["actions"].extend(episode_actions)
            collected["old_log_probabilities"].extend(episode_log_probs); collected["advantages"].extend(advantages); collected["returns"].extend(returns)
            rollouts += 1
    arrays = {
        "states": np.asarray(collected["states"], dtype=np.float32),
        "masks": np.asarray(collected["masks"], dtype=bool),
        "actions": np.asarray(collected["actions"], dtype=np.int64),
        "old_log_probabilities": np.asarray(collected["old_log_probabilities"], dtype=np.float32),
        "advantages": np.asarray(collected["advantages"], dtype=np.float32),
        "returns": np.asarray(collected["returns"], dtype=np.float32),
    }
    return arrays, {"transitions": int(len(arrays["actions"])), "rollouts": rollouts, "executed_action_ids": sorted(action_ids)}


def _train(policy: MaskedPPOPolicy, data: dict[str, np.ndarray], *, gradient_steps: int, seed: int) -> dict[str, float | int]:
    if gradient_steps <= 0 or len(data["actions"]) < 128:
        raise ValueError("PPO needs at least 128 on-policy transitions and positive gradient steps")
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    advantages = data["advantages"]
    advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-6)
    tensors = {name: torch.as_tensor(value) for name, value in {**data, "advantages": advantages}.items()}
    losses: list[float] = []; batch_size = min(128, len(data["actions"]))
    for _ in range(gradient_steps):
        selected = rng.choice(len(data["actions"]), size=batch_size, replace=False)
        loss = ppo_loss(policy, *(tensors[name][selected] for name in ("states", "masks", "actions", "old_log_probabilities", "advantages", "returns")))
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0); optimizer.step()
        losses.append(float(loss.item()))
    return {"gradient_steps": gradient_steps, "batch_size": batch_size, "mean_loss": float(np.mean(losses))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/metadata/manifests/live_ppo_exploration_runs.csv")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "results/raw")
    parser.add_argument("--state-schema", type=Path, default=ROOT / "results/training/live_controller/state_schema.json")
    parser.add_argument("--initialization", type=Path, default=ROOT / "results/training/controller/masked_ppo_offline.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/training/live_controller")
    parser.add_argument("--gradient-steps", type=int, default=300)
    args = parser.parse_args()
    rows, schema = _manifest(args.manifest), _load_schema(args.state_schema)
    if not args.initialization.is_file():
        raise SystemExit("BLOCKED_LIVE_PPO_FINE_TUNING: missing offline PPO initialization")
    train_raw = [_raw_records(args.raw_root / f"iq-broker-live-{row['run_id']}") for row in rows if row["partition"] == "train"]
    normalizers = _normalizers(train_raw)
    data, training = _collect(rows, args.raw_root, schema, "train", normalizers)
    policy = MaskedPPOPolicy(len(FEATURE_NAMES))
    policy.load_state_dict(torch.load(args.initialization, map_location="cpu", weights_only=True))
    report = _train(policy, data, gradient_steps=args.gradient_steps, seed=2401)
    _, validation = _collect(rows, args.raw_root, schema, "validation", normalizers)
    args.output.mkdir(parents=True, exist_ok=True)
    candidate = args.output / "masked_ppo_live_candidate.pt"
    torch.save(policy.eval().state_dict(), candidate)
    summary = {
        "status": "LIVE_PPO_BOUNDED_FINE_TUNING_COMPLETE", "not_confirmatory": True,
        "policy_status": "CANDIDATE_REQUIRES_LIVE_VALIDATION_AND_FREEZE",
        "reward_normalizers_training_only": {"throughput_p95": normalizers[0], "latency_p95": normalizers[1]},
        "train": training, "validation": validation, "training": report,
        "candidate": {"file": str(candidate), "sha256": _digest(candidate)},
        "state_schema_sha256": _digest(args.state_schema), "initialization_sha256": _digest(args.initialization),
    }
    (args.output / "live_ppo_fine_tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
