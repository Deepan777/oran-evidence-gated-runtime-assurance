"""Produce a transparent offline-IQ calibration pilot; never a confirmatory result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.controller.action_mask import masked_argmax
from src.controller.replay import PrioritizedReplayBuffer
from src.controller.training import train_md3qn_per
from src.detector.training import select_simplest_detector
from src.environment.iq_calibrated_env import CalibrationScenario, FEATURE_NAMES, IqCalibratedAntiJammingEnv
from src.jammer.waveforms import JammerType


def _scenario(jammer: JammerType, seed: int) -> CalibrationScenario:
    return CalibrationScenario(f"offline_{jammer.value}_s{seed}", jammer, 5.0 if jammer is not JammerType.NONE else -5.0,
                               "medium", "protected_plus_background", "nominal", seed, steps=40)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("results/offline_calibration")); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in range(1001, 1007):
        for jammer in JammerType:
            env = IqCalibratedAntiJammingEnv(_scenario(jammer, seed)); state, _ = env.reset()
            for epoch in range(40):
                mask = env.action_mask(); action = 0 if epoch < 5 else int(np.flatnonzero(mask)[-1])
                next_state, reward, done, info = env.step(action)
                rows.append({"run_id": f"offline_{jammer.value}_{seed}", "scenario_id": env.scenario.scenario_id,
                             "seed": seed, "epoch": epoch, "jammer_type": jammer.value, "jammer_on": epoch >= 20,
                             "action": action, "reward": reward, **info["metrics"], "evidence_class": "offline_iq_calibration"})
                state = next_state
                if done: break
    frame = pd.DataFrame(rows); raw = args.output / "offline_iq_telemetry.parquet"; frame.to_parquet(raw, index=False)
    # Split complete runs by seed so every jammer family is represented in both
    # partitions.  The detector never receives its own output/ground truth as
    # an input: only pre-action radio and resource telemetry is used.
    train = frame[frame.seed <= 1004]
    validation = frame[frame.seed > 1004]
    detector_features = ["sinr_db", "cqi", "rsrp_dbfs", "rsrq_db", "bler", "harq_nack_ratio",
                         "dl_throughput_mbps", "rlc_drop_rate", "rlc_sdu_delay_ms", "prb_utilization",
                         "clean_prb_fraction", "scheduler_queue_backlog_bytes"]
    x_train, x_val = train[detector_features].to_numpy(), validation[detector_features].to_numpy()
    classes = [item.value for item in JammerType]; labels_train = train.jammer_type.map({v:i for i,v in enumerate(classes)}).to_numpy(); labels_val = validation.jammer_type.map({v:i for i,v in enumerate(classes)}).to_numpy()
    selected = select_simplest_detector(x_train, labels_train, x_val, labels_val, 1001)
    joblib.dump(selected.model, args.output / "detector.joblib")
    buffer = PrioritizedReplayBuffer(2048, len(FEATURE_NAMES), 5, seed=1001)
    for record in frame.itertuples(index=False):
        state = np.asarray([getattr(record, n) for n in FEATURE_NAMES], dtype=np.float32); buffer.add(state, int(record.action), float(record.reward), state, False, np.ones(5, dtype=bool))
    controller, training = train_md3qn_per(buffer, len(FEATURE_NAMES), 1001, steps=100, batch_size=64)
    checkpoint = args.output / "md3qn_per_offline.pt"; torch.save(controller.state_dict(), checkpoint)
    summary = {"status": "OFFLINE_CALIBRATION_COMPLETE", "evidence_class": "offline_iq_calibration", "not_confirmatory": True,
               "records": len(frame), "detector": {"model": selected.model_name, "macro_f1": selected.macro_f1, "balanced_accuracy": selected.balanced_accuracy, "latency_ms": selected.latency_ms},
               "controller": {"gradient_steps": training.gradient_steps, "mean_loss": training.mean_loss, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
               "feature_order": list(FEATURE_NAMES), "detector_feature_order": detector_features,
               "raw_file": str(raw)}
    (args.output / "offline_calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
