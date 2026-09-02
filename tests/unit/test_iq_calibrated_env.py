from src.controller.action_mask import Action
from src.environment.iq_calibrated_env import CalibrationScenario, IqCalibratedAntiJammingEnv
from src.jammer.waveforms import JammerType


def test_environment_observables_are_iq_derived() -> None:
    env = IqCalibratedAntiJammingEnv(CalibrationScenario("p", JammerType.PARTIAL_BAND, 10.0, "medium", "protected_only", "nominal", 5, steps=4))
    state, info = env.reset()
    assert state.shape == (27,)
    assert info["live_testbed"] is False
    _, _, _, result = env.step(Action.NO_OP)
    assert result["metrics"]["bler"] >= 0.0
    assert result["evidence_class"] == "offline_iq_calibration"
