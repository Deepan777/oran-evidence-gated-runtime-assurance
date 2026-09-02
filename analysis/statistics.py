from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import stats


FloatArray = NDArray[np.float64]


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[FloatArray], float] = lambda sample: float(np.mean(sample)),
    confidence: float = 0.95,
    iterations: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2 or not np.all(np.isfinite(data)):
        raise ValueError("bootstrap input must contain at least two finite values")
    if not 0.0 < confidence < 1.0 or iterations <= 0:
        raise ValueError("invalid confidence or iteration count")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        estimates[index] = statistic(rng.choice(data, size=data.size, replace=True))
    tail = (1.0 - confidence) / 2.0
    return tuple(np.quantile(estimates, [tail, 1.0 - tail]).tolist())  # type: ignore[return-value]


def friedman_test(matched_blocks_by_method: NDArray[np.floating]) -> dict[str, float]:
    matrix = np.asarray(matched_blocks_by_method, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 3:
        raise ValueError("Friedman input must be [blocks, methods] with >=2 blocks and >=3 methods")
    result = stats.friedmanchisquare(*(matrix[:, column] for column in range(matrix.shape[1])))
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def rank_biserial_paired(left: Sequence[float], right: Sequence[float]) -> float:
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    return (positive - negative) / (positive + negative)


def planned_wilcoxon(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.size < 2:
        raise ValueError("paired samples must have equal shape and at least two observations")
    if np.allclose(left_array, right_array):
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.wilcoxon(left_array, right_array, zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "rank_biserial": rank_biserial_paired(left_array, right_array),
        "n_pairs": int(left_array.size),
    }


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[dict[str, float | bool | int]]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be a non-empty one-dimensional sequence in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (values.size - np.arange(values.size)) * values[order])
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return [
        {
            "index": int(index),
            "raw_p": float(values[index]),
            "adjusted_p": float(adjusted[index]),
            "significant": bool(adjusted[index] <= alpha),
        }
        for index in range(values.size)
    ]


def build_statistics_report(payload: dict[str, object]) -> dict[str, object]:
    """Compute predeclared paired statistics from a transparent input schema.

    ``outcomes`` maps an outcome name to ``{"methods": [...],
    "matched_blocks": [[...], ...]}``, where each row is one matched
    scenario/seed block and each column follows ``methods``. Optional
    ``planned_pairs`` contains pairs of method names for Wilcoxon tests.
    """
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, dict) or not outcomes:
        raise ValueError("statistics input requires a non-empty outcomes object")
    confidence = float(payload.get("confidence", 0.95))
    iterations = int(payload.get("bootstrap_iterations", 10_000))
    seed = int(payload.get("seed", 1001))
    result: dict[str, object] = {"schema_version": 1, "confidence": confidence, "outcomes": {}}
    flat_p_values: list[float] = []; p_locations: list[tuple[str, int]] = []
    rendered: dict[str, object] = {}
    for outcome_name, definition in outcomes.items():
        if not isinstance(outcome_name, str) or not isinstance(definition, dict):
            raise ValueError("each outcome must have an object definition")
        methods, blocks = definition.get("methods"), definition.get("matched_blocks")
        if not isinstance(methods, list) or len(methods) < 3 or len(set(methods)) != len(methods):
            raise ValueError(f"outcome {outcome_name} requires at least three unique method names")
        matrix = np.asarray(blocks, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(methods) or matrix.shape[0] < 2 or not np.all(np.isfinite(matrix)):
            raise ValueError(f"outcome {outcome_name} has invalid matched_blocks")
        summaries = {}
        for index, method in enumerate(methods):
            values = matrix[:, index]
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            summaries[str(method)] = {
                "mean": float(values.mean()),
                "bootstrap_ci": list(bootstrap_ci(
                    values, confidence=confidence, iterations=iterations, seed=seed + index
                )),
                "median": float(median),
                "iqr": [float(q1), float(q3)],
            }
        pairs = definition.get("planned_pairs", [[methods[0], method] for method in methods[1:]])
        comparisons = []
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2 or pair[0] not in methods or pair[1] not in methods:
                raise ValueError(f"outcome {outcome_name} contains an invalid planned pair")
            comparison = {"left": pair[0], "right": pair[1], **planned_wilcoxon(matrix[:, methods.index(pair[0])], matrix[:, methods.index(pair[1])])}
            p_locations.append((outcome_name, len(comparisons))); flat_p_values.append(float(comparison["p_value"]))
            comparisons.append(comparison)
        direction = definition.get("direction", "higher")
        if direction not in {"higher", "lower"}:
            raise ValueError(f"outcome {outcome_name} direction must be higher or lower")
        relative_improvements: list[dict[str, object]] = []
        for index, method in enumerate(methods[1:], start=1):
            baseline = matrix[:, index]
            proposed = matrix[:, 0]
            if np.any(np.isclose(baseline, 0.0)):
                relative_improvements.append({
                    "method": str(method),
                    "status": "NOT_REPORTED_BASELINE_NEAR_ZERO",
                })
                continue
            signs = 1.0 if direction == "higher" else -1.0
            ratios = signs * (proposed - baseline) / np.abs(baseline)
            relative_improvements.append({
                "method": str(method),
                "mean_relative_improvement": float(ratios.mean()),
                "bootstrap_ci": list(bootstrap_ci(
                    ratios, confidence=confidence, iterations=iterations, seed=seed + 100 + index
                )),
            })
        rendered[outcome_name] = {
            "methods": methods,
            "direction": direction,
            "n_matched_blocks": int(matrix.shape[0]),
            "friedman": friedman_test(matrix),
            "summaries": summaries,
            "planned_comparisons": comparisons,
            "relative_improvements_vs_first_method": relative_improvements,
        }
    adjusted = holm_bonferroni(flat_p_values)
    for adjustment, (outcome_name, index) in zip(adjusted, p_locations, strict=True):
        rendered[outcome_name]["planned_comparisons"][index]["holm"] = adjustment
    result["outcomes"] = rendered
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/aggregate/statistics_input.json"))
    parser.add_argument("--output", type=Path, default=Path("results/statistics/stats_report.json"))
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"BLOCKED: machine-readable statistics input is missing: {args.input}")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_statistics_report(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
