from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RunDescriptor:
    run_id: str
    scenario_id: str
    seed: int
    generalization_holdout: bool = False


@dataclass(frozen=True)
class RunSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    locked_test: tuple[str, ...]
    generalization: tuple[str, ...]

    def validate(self) -> None:
        partitions = {
            "train": set(self.train),
            "validation": set(self.validation),
            "locked_test": set(self.locked_test),
            "generalization": set(self.generalization),
        }
        names = tuple(partitions)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                overlap = partitions[left_name] & partitions[right_name]
                if overlap:
                    raise ValueError(
                        f"run leakage between {left_name} and {right_name}: {sorted(overlap)}"
                    )


def _stable_key(run: RunDescriptor, split_seed: int) -> str:
    material = f"{split_seed}|{run.scenario_id}|{run.seed}|{run.run_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def create_run_level_split(
    runs: Iterable[RunDescriptor],
    split_seed: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> RunSplit:
    """Create a deterministic run-level split with a separate generalization set."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a locked test fraction")

    run_list = list(runs)
    run_ids = [run.run_id for run in run_list]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run_id values must be unique")

    generalization = sorted(
        (run for run in run_list if run.generalization_holdout),
        key=lambda run: _stable_key(run, split_seed),
    )
    interpolation = sorted(
        (run for run in run_list if not run.generalization_holdout),
        key=lambda run: _stable_key(run, split_seed),
    )
    count = len(interpolation)
    train_end = round(count * train_fraction)
    validation_end = train_end + round(count * validation_fraction)
    split = RunSplit(
        train=tuple(run.run_id for run in interpolation[:train_end]),
        validation=tuple(run.run_id for run in interpolation[train_end:validation_end]),
        locked_test=tuple(run.run_id for run in interpolation[validation_end:]),
        generalization=tuple(run.run_id for run in generalization),
    )
    split.validate()
    return split

