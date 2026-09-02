from __future__ import annotations

from analysis.evidence_gate import REQUIRED_BOOLEAN_EVIDENCE, evaluate_evidence


def passing_summary() -> dict[str, object]:
    summary: dict[str, object] = {
        "supported_primary_resilience_outcomes": ["sla_violation_rate", "throughput_retention"]
    }
    summary.update({field: True for field in REQUIRED_BOOLEAN_EVIDENCE})
    return summary


def test_evidence_gate_passes_only_complete_evidence() -> None:
    result = evaluate_evidence(passing_summary())
    assert result["status"] == "PASS"
    assert result["manuscript_allowed"] is True


def test_evidence_gate_fails_missing_condition() -> None:
    summary = passing_summary()
    summary["explanations_stable"] = False
    result = evaluate_evidence(summary)
    assert result["status"] == "FAIL"
    assert result["manuscript_allowed"] is False

