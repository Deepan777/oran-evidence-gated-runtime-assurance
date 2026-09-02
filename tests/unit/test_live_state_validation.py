from experiments.training.validate import validate_live_state_schema


def _schema() -> dict[str, object]:
    return {"feature_names": ["x", "y"], "raw_p01": [0.0, -1.0], "raw_p99": [1.0, 1.0]}


def _decision(x: float, y: float) -> dict[str, object]:
    return {"status": "DECISION", "decision": {"raw_state": {"x": x, "y": y}}}


def test_live_state_validation_blocks_out_of_distribution_controller_input() -> None:
    result = validate_live_state_schema([_decision(2.0, 0.0)], _schema())
    assert result["status"] == "BLOCKED_LIVE_STATE_SCHEMA_INCOMPATIBLE"
    assert result["features"]["x"]["outside_training_bounds_count"] == 1  # type: ignore[index]


def test_live_state_validation_passes_when_all_input_is_within_frozen_bounds() -> None:
    result = validate_live_state_schema([_decision(0.2, 0.0), _decision(0.8, -0.2)], _schema())
    assert result["status"] == "LIVE_STATE_SCHEMA_COMPATIBLE"
