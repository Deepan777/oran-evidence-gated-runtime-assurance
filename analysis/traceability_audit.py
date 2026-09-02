from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.config import load_methodology_lock, state_feature_names
from src.common.hashing import sha256_file
from src.controller.action_mask import Action


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def structure_audit(root: Path) -> dict[str, object]:
    lock = load_methodology_lock(root)
    findings: list[str] = []
    for source in lock["authoritative_sources"]:
        source_path = root / source["path"]
        if not source_path.is_file():
            findings.append(f"Missing authoritative source: {source_path}")
        elif sha256_file(source_path) != source["sha256"]:
            findings.append(f"Authoritative source hash mismatch: {source_path}")

    locked_actions = [item["id"] for item in lock["actions"]["definitions"]]
    code_actions = [int(action) for action in Action]
    if locked_actions != code_actions:
        findings.append(f"Action enum mismatch: lock={locked_actions}, code={code_actions}")

    names = state_feature_names(lock)
    if len(names) != 27:
        findings.append(f"Expected 27 locked state features; found {len(names)}")

    required_modules = (
        "src/jammer/waveforms.py",
        "src/controller/state.py",
        "src/controller/action_mask.py",
        "src/controller/reward.py",
        "src/controller/md3qn.py",
        "src/controller/replay.py",
        "src/xai/attribution.py",
        "src/xai/contrastive.py",
        "src/xai/counterfactual.py",
        "src/xai/rationale.py",
        "analysis/statistics.py",
        "analysis/evidence_gate.py",
    )
    for relative in required_modules:
        if not (root / relative).is_file():
            findings.append(f"Missing required implementation module: {relative}")

    return {
        "mode": "structure",
        "status": "PASS" if not findings else "FAIL",
        "critical_findings": findings,
        "state_feature_count": len(names),
        "action_count": len(code_actions),
    }


def release_audit(root: Path) -> dict[str, object]:
    result = structure_audit(root)
    findings = list(result["critical_findings"])
    release_artifacts = (
        "CONFIRMATORY_FREEZE_MANIFEST.yaml",
        "results/raw/CONFIRMATORY_COMPLETE.json",
        "results/statistics/stats_report.json",
        "results/aggregate/evidence_summary.json",
    )
    for relative in release_artifacts:
        if not (root / relative).is_file():
            findings.append(f"Missing release artifact: {relative}")
    result.update(
        mode="release",
        status="PASS" if not findings else "FAIL",
        critical_findings=findings,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("structure", "release"), default="structure")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/statistics/traceability_audit.json"),
    )
    args = parser.parse_args()
    result = structure_audit(PROJECT_ROOT) if args.mode == "structure" else release_audit(PROJECT_ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

