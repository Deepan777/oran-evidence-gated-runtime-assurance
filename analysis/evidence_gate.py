from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_BOOLEAN_EVIDENCE = (
    "no_major_unexplained_no_jam_regression",
    "benefit_across_multiple_jammer_conditions",
    "explanations_faithful",
    "explanations_stable",
    "counterfactual_validity_strong",
    "latency_compatible_with_decision_interval",
    "conclusion_reproduces_across_seeds",
    "traceability_audit_passed",
)


def evaluate_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    outcomes = summary.get("supported_primary_resilience_outcomes", [])
    if not isinstance(outcomes, list) or len(outcomes) < 2:
        reasons.append("Fewer than two predeclared primary resilience outcomes are supported.")
    for field in REQUIRED_BOOLEAN_EVIDENCE:
        if summary.get(field) is not True:
            reasons.append(f"Required evidence is absent or false: {field}.")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "supported_primary_resilience_outcomes": outcomes if isinstance(outcomes, list) else [],
        "manuscript_allowed": not reasons,
    }


def write_report(result: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    reason_lines = "\n".join(f"- {reason}" for reason in result["reasons"]) or "- None."
    markdown_path.write_text(
        "# Evidence Gate Report\n\n"
        f"Status: **{result['status']}**\n\n"
        f"Manuscript allowed: **{result['manuscript_allowed']}**\n\n"
        "## Reasons\n\n"
        f"{reason_lines}\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/aggregate/evidence_summary.json"))
    parser.add_argument("--json", type=Path, default=Path("results/evidence_gate.json"))
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("results/evidence_gate/EVIDENCE_GATE_REPORT.md"),
    )
    args = parser.parse_args()
    if not args.summary.is_file():
        summary: dict[str, Any] = {}
    else:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = evaluate_evidence(summary)
    write_report(result, args.json, args.markdown)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

