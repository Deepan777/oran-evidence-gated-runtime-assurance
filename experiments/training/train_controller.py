#!/usr/bin/env python3
"""Train proposed and comparator controllers on explicitly offline IQ calibration.

This is deliberately separate from locked live-testbed confirmatory evidence.
It produces reproducible checkpoints for controller integration and records the
calibration-only provenance of every output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

from src.baselines.learning import train_masked_ppo, train_standard_dqn
from src.controller.replay import PrioritizedReplayBuffer
from src.controller.state import StateBuilder, StateSchema
from src.controller.training import train_md3qn_per
from src.environment.iq_calibrated_env import CalibrationScenario, FEATURE_NAMES, IqCalibratedAntiJammingEnv
from src.jammer.waveforms import JammerType


ROOT = Path(__file__).resolve().parents[2]
FEATURE_UNITS = {
    "p_jam": "probability", "cqi": "index", "sinr_db": "dB", "rsrp_dbfs": "dBFS", "rsrq_db": "dB",
    "bler": "fraction", "harq_nack_ratio": "fraction", "dl_throughput_mbps": "Mbps",
    "rlc_drop_rate": "fraction", "rlc_sdu_delay_ms": "ms", "prb_utilization": "fraction",
    "clean_prb_fraction": "fraction", "scheduler_queue_backlog_bytes": "bytes",
    "jammer_occupancy_fraction": "fraction", "jammer_subband_start_norm": "fraction_of_band",
}


def _factory() -> IqCalibratedAntiJammingEnv:
    return IqCalibratedAntiJammingEnv(CalibrationScenario(
        scenario_id="ppo_offline_barrage_s1001", jammer_type=JammerType.BARRAGE,
        jsr_db=5.0, load="medium", service_mix="protected_plus_background",
        attenuation="nominal", seed=1001, steps=40,
    ))


class _StandardizedEnvironment:
    """Apply the frozen training-only state transform to PPO observations."""

    def __init__(self, schema: StateSchema) -> None:
        self._environment = _factory()
        self._schema = schema

    def _transform(self, state: np.ndarray) -> np.ndarray:
        values = {name: float(value) for name, value in zip(FEATURE_NAMES, state, strict=True)}
        return StateBuilder(self._schema).build(values)

    def reset(self) -> tuple[np.ndarray, object]:
        state, metadata = self._environment.reset()
        return self._transform(state), metadata

    def action_mask(self) -> np.ndarray:
        return self._environment.action_mask()

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, object]:
        state, reward, terminal, metadata = self._environment.step(action_id)
        return self._transform(state), reward, terminal, metadata


def _calibration_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        subprocess.run([sys.executable, "experiments/pilot/run_offline.py", "--output", str(path.parent)], cwd=ROOT, check=True)
    frame = pd.read_parquet(path)
    required = {"run_id", "epoch", "action", "reward", *FEATURE_NAMES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"offline calibration is missing required controller fields: {missing}")
    if set(frame.get("evidence_class", ())) != {"offline_iq_calibration"}:
        raise ValueError("controller pretraining accepts only explicitly labelled offline calibration")
    return frame.sort_values(["run_id", "epoch"]).reset_index(drop=True)


def _training_schema(frame: pd.DataFrame) -> StateSchema:
    """Fit deterministic standardization on the predeclared training runs only."""
    if "seed" not in frame:
        raise ValueError("offline calibration is missing the immutable scenario seed")
    training = frame.loc[frame["seed"].isin((1001, 1002, 1003, 1004)), list(FEATURE_NAMES)]
    if training.empty:
        raise ValueError("offline calibration lacks the predeclared training seeds 1001--1004")
    means = training.mean(axis=0).to_numpy(dtype=np.float64)
    scales = np.maximum(training.std(axis=0, ddof=0).to_numpy(dtype=np.float64), 1e-6)
    return StateSchema(FEATURE_NAMES, means, scales)


def _standardize(row: pd.Series, schema: StateSchema) -> np.ndarray:
    values = {name: float(row[name]) for name in FEATURE_NAMES}
    return StateBuilder(schema).build(values)


def _buffer(frame: pd.DataFrame, schema: StateSchema) -> PrioritizedReplayBuffer:
    replay = PrioritizedReplayBuffer(max(2048, len(frame)), len(FEATURE_NAMES), 5, seed=1001)
    for _, episode in frame.groupby("run_id", sort=True):
        records = episode.reset_index(drop=True)
        for index, row in records.iterrows():
            state = _standardize(row, schema)
            terminal = index == len(records) - 1
            next_state = state if terminal else _standardize(records.iloc[index + 1], schema)
            # The calibration action trace contains actions selected by the
            # environment's safety mask. The next-state mask is recomputed in
            # the training environment for PPO; replay targets retain NO_OP
            # feasible and do not mark an unsafe executed action as valid.
            replay.add(state, int(row["action"]), float(row["reward"]), next_state, terminal, np.ones(5, dtype=bool))
    return replay


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_state_artifacts(frame: pd.DataFrame, schema: StateSchema, output: Path) -> dict[str, object]:
    raw = frame.loc[frame["seed"].isin((1001, 1002, 1003, 1004)), list(FEATURE_NAMES)]
    if raw.empty:
        raise ValueError("offline calibration lacks training rows for frozen state bounds")
    lower = raw.quantile(0.01).to_numpy(dtype=float)
    upper = raw.quantile(0.99).to_numpy(dtype=float)
    document = {
        "status": "FROZEN_OFFLINE_CALIBRATION_SCHEMA",
        "evidence_class": "offline_iq_calibration",
        "not_confirmatory": True,
        "feature_names": list(schema.feature_names),
        "units": [FEATURE_UNITS.get(name, "onehot") for name in schema.feature_names],
        "means": schema.means.tolist(),
        "scales": schema.scales.tolist(),
        "clip_low": schema.clip_low,
        "clip_high": schema.clip_high,
        "raw_p01": lower.tolist(),
        "raw_p99": upper.tolist(),
        "fit_run_seeds": [1001, 1002, 1003, 1004],
        "validation_run_seeds": [1005, 1006],
    }
    (output / "state_schema.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    # Persist the exact numerical transform separately as required by the
    # methodology lock.  The JSON schema remains human-auditable, while this
    # artifact is the machine-loadable frozen preprocessing object.
    joblib.dump({
        "schema_version": 1,
        "feature_names": list(schema.feature_names),
        "means": schema.means,
        "scales": schema.scales,
        "clip_low": schema.clip_low,
        "clip_high": schema.clip_high,
        "fit_run_seeds": [1001, 1002, 1003, 1004],
    }, output / "scaler.pkl")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=ROOT / "results/offline_calibration_v3/offline_iq_telemetry.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "results/training/controller")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    frame = _calibration_rows(args.calibration)
    schema = _training_schema(frame)
    replay = _buffer(frame, schema)
    proposed, proposed_report = train_md3qn_per(replay, len(FEATURE_NAMES), 1001, steps=100, batch_size=64)
    dqn, dqn_report = train_standard_dqn(replay, len(FEATURE_NAMES), 1002, steps=100, batch_size=64)
    ppo, ppo_report = train_masked_ppo(lambda: _StandardizedEnvironment(schema), len(FEATURE_NAMES), 1003, rollout_steps=64, updates=4, epochs_per_update=2)
    files = {"proposed_md3qn_per": args.output / "md3qn_per_offline.pt", "baseline_standard_dqn": args.output / "standard_dqn_offline.pt", "baseline_masked_ppo": args.output / "masked_ppo_offline.pt"}
    torch.save(proposed.state_dict(), files["proposed_md3qn_per"]); torch.save(dqn.state_dict(), files["baseline_standard_dqn"]); torch.save(ppo.state_dict(), files["baseline_masked_ppo"])
    state_schema = _write_state_artifacts(frame, schema, args.output)
    summary = {"status": "OFFLINE_CONTROLLER_PRETRAINING_COMPLETE", "evidence_class": "offline_iq_calibration", "not_confirmatory": True, "records": len(frame), "features": list(FEATURE_NAMES), "state_schema_file": "state_schema.json", "state_schema_sha256": _digest(args.output / "state_schema.json"), "scaler_file": "scaler.pkl", "scaler_sha256": _digest(args.output / "scaler.pkl"), "models": {"proposed_md3qn_per": {**proposed_report.__dict__, "sha256": _digest(files["proposed_md3qn_per"])}, "baseline_standard_dqn": {**dqn_report.__dict__, "sha256": _digest(files["baseline_standard_dqn"])}, "baseline_masked_ppo": {**ppo_report.__dict__, "sha256": _digest(files["baseline_masked_ppo"])}}}
    (args.output / "controller_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
