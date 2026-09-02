from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    contribution: float


def build_operator_rationale(
    selected_action: str,
    alternative_action: str,
    q_gap: float,
    contributors: tuple[FeatureContribution, ...],
    counterfactual_feature: str | None = None,
    counterfactual_from: float | None = None,
    counterfactual_to: float | None = None,
) -> str:
    """Create deterministic text containing only supplied numerical evidence."""
    if not contributors:
        raise ValueError("at least one verified contributor is required")
    top = sorted(contributors, key=lambda item: abs(item.contribution), reverse=True)[:2]
    evidence = " and ".join(
        f"{item.feature} (value {item.value:.4g}, contribution {item.contribution:+.4g})"
        for item in top
    )
    rationale = (
        f"{selected_action} was selected over {alternative_action}; the measured Q-value "
        f"margin was {q_gap:.4g}. The strongest logged contributors were {evidence}."
    )
    counterfactual_values = (
        counterfactual_feature,
        counterfactual_from,
        counterfactual_to,
    )
    if all(value is not None for value in counterfactual_values):
        rationale += (
            f" The smallest accepted counterfactual changed {counterfactual_feature} "
            f"from {counterfactual_from:.4g} to {counterfactual_to:.4g}, at which point "
            f"{alternative_action} became preferred."
        )
    elif any(value is not None for value in counterfactual_values):
        raise ValueError("counterfactual fields must be supplied together")
    return rationale

