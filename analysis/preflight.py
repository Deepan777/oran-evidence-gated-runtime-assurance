from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_file(path: Path, reason: str) -> None:
    if not path.is_file():
        raise SystemExit(f"BLOCKED: {reason}. Missing: {path}")


def check(stage: str) -> dict[str, object]:
    require_file(PROJECT_ROOT / "METHODOLOGY_LOCK.yaml", "methodology lock is required")
    require_file(
        PROJECT_ROOT / "testbed" / "PHASE4_READY.json",
        "the live Ubuntu/OCUDU/RIC/Open5GS/srsUE testbed has not been validated",
    )
    if stage in {"confirmatory", "results", "figures", "evidence"}:
        require_file(
            PROJECT_ROOT / "CONFIRMATORY_FREEZE_MANIFEST.yaml",
            "confirmatory configuration is not frozen",
        )
    if stage in {"results", "figures", "evidence"}:
        require_file(
            PROJECT_ROOT / "results" / "raw" / "CONFIRMATORY_COMPLETE.json",
            "confirmatory experiments are not complete",
        )
    return {"stage": stage, "status": "READY"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("pilot", "confirmatory", "results", "figures", "evidence"),
    )
    args = parser.parse_args()
    print(json.dumps(check(args.stage), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

