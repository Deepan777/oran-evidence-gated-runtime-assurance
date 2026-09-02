#!/usr/bin/env python3
"""Train the jammer detector from offline calibration only, with run-level splits."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.detector.training import select_simplest_detector
from src.jammer.waveforms import JammerType


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ("sinr_db", "cqi", "rsrp_dbfs", "rsrq_db", "bler", "harq_nack_ratio", "dl_throughput_mbps", "rlc_drop_rate", "rlc_sdu_delay_ms", "prb_utilization", "clean_prb_fraction", "scheduler_queue_backlog_bytes")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, default=ROOT / "results/offline_calibration_v3/offline_iq_telemetry.parquet"); parser.add_argument("--output", type=Path, default=ROOT / "results/training/detector"); args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if not args.input.is_file():
        subprocess.run([sys.executable, "experiments/pilot/run_offline.py", "--output", str(args.input.parent)], cwd=ROOT, check=True)
    frame = pd.read_parquet(args.input)
    if set(frame.get("evidence_class", ())) != {"offline_iq_calibration"}:
        raise SystemExit("BLOCKED: detector pretraining accepts only offline_iq_calibration records")
    labels = [item.value for item in JammerType]
    class_columns = tuple(f"jammer_class_{name}" for name in labels)
    missing = sorted((set(FEATURES) | set(class_columns)) - set(frame.columns))
    if missing:
        raise SystemExit(f"BLOCKED: calibration is missing detector inputs/ground truth: {missing}")
    # Labels must describe the actual per-window injected waveform. A scenario
    # name alone would incorrectly label the pre-jam and recovery windows.
    frame = frame.copy()
    frame["detector_label"] = np.argmax(frame.loc[:, class_columns].to_numpy(dtype=float), axis=1)
    train = frame[frame["seed"] <= 1004]; validation = frame[frame["seed"] > 1004]
    if train.empty or validation.empty:
        raise SystemExit("BLOCKED: run-level train/validation calibration split is empty")
    selected = select_simplest_detector(train.loc[:, FEATURES].to_numpy(), train["detector_label"].to_numpy(), validation.loc[:, FEATURES].to_numpy(), validation["detector_label"].to_numpy(), 1001)
    model_path = args.output / "detector_offline.joblib"; joblib.dump(selected.model, model_path)
    summary = {"status": "OFFLINE_DETECTOR_PRETRAINING_COMPLETE", "evidence_class": "offline_iq_calibration", "not_confirmatory": True, "model": selected.model_name, "macro_f1": selected.macro_f1, "balanced_accuracy": selected.balanced_accuracy, "latency_ms": selected.latency_ms, "features": list(FEATURES), "labels": labels, "run_level_split": {"train_seeds": sorted(train.seed.unique().tolist()), "validation_seeds": sorted(validation.seed.unique().tolist())}, "model_file": str(model_path)}
    (args.output / "detector_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
