from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from experiments.ablations.run import _required_ids, load_and_validate_freeze as load_ablations
from experiments.confirmatory.run import METHODS, load_and_validate_freeze as load_confirmatory
from experiments.sensitivity.run import SENSITIVITY_IDS, load_and_validate_freeze as load_sensitivity


def _artifact(tmp_path: Path) -> tuple[Path, str]:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")
    return artifact, hashlib.sha256(artifact.read_bytes()).hexdigest()


def _write(tmp_path: Path, name: str, payload: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_confirmatory_freeze_requires_all_locked_methods(tmp_path: Path) -> None:
    artifact, digest = _artifact(tmp_path)
    payload = {
        "status": "CONFIRMATORY_FROZEN",
        "artifacts": {"model": {"path": str(artifact), "sha256": digest}},
        "execution_plan": [
            {"method": method, "command": ["echo", method], "scenario_manifest": str(artifact),
             "result_manifest": f"results/{method}.json", "expected_runs": 1}
            for method in METHODS
        ],
    }
    assert load_confirmatory(_write(tmp_path, "confirmatory.yaml", payload))["status"] == "CONFIRMATORY_FROZEN"


def test_ablation_freeze_requires_exact_methodology_order(tmp_path: Path) -> None:
    artifact, digest = _artifact(tmp_path)
    payload = {
        "status": "ABLATIONS_FROZEN",
        "artifacts": {"model": {"path": str(artifact), "sha256": digest}},
        "execution_plan": [
            {"ablation": name, "command": ["echo", name], "result_manifest": f"results/{name}.json", "expected_runs": 1}
            for name in _required_ids()
        ],
    }
    assert load_ablations(_write(tmp_path, "ablations.yaml", payload))["status"] == "ABLATIONS_FROZEN"


def test_sensitivity_freeze_requires_exact_locked_studies(tmp_path: Path) -> None:
    artifact, digest = _artifact(tmp_path)
    payload = {
        "status": "SENSITIVITY_FROZEN",
        "artifacts": {"model": {"path": str(artifact), "sha256": digest}},
        "execution_plan": [
            {"sensitivity": name, "command": ["echo", name], "result_manifest": f"results/{name}.json", "expected_runs": 1}
            for name in SENSITIVITY_IDS
        ],
    }
    assert load_sensitivity(_write(tmp_path, "sensitivity.yaml", payload))["status"] == "SENSITIVITY_FROZEN"
