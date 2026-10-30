from src.controller.action_mask import Action
from src.xapp.adapters.actions import ActionAdapter, ConfigurationReadback, ConfigurationStateStatus, HookAcknowledgement
from src.xapp.adapters.ocudu_websocket import OcuduWebSocketResearchBackend


def snapshots(generation=7, action="A2", time_ns=100):
    return "\n".join(
        f"RESEARCH_STATE_CONSUMED scope={scope} pid=12 timestamp_ns={time_ns} generation={generation} "
        f"action={action} prb_start=0 prb_stop=275 mcs_cap=6 slice_min_pct=10 slice_max_pct=100"
        for scope in ("prb", "mcs", "slice")
    )


def test_consumer_snapshot_requires_all_three_matching_scopes_and_freshness():
    parse = OcuduWebSocketResearchBackend.parse_consumed
    assert not parse(snapshots(), not_before_ns=101).successful
    assert not parse(snapshots().splitlines()[0], not_before_ns=1).successful
    mismatched = snapshots() + "\n" + snapshots(generation=8).splitlines()[0]
    assert not parse(mismatched, not_before_ns=1).successful
    observed = parse(snapshots(), not_before_ns=1)
    assert observed.successful
    assert observed.action is Action.MCS_BACKOFF
    assert observed.parameters == {"mcs_cap": 6}
    assert observed.configuration_version == "7"
    assert observed.verification_scope == "scheduler_software_configuration"


def test_acknowledgement_parameter_mismatch_cannot_be_erased_by_good_readback():
    class Backend:
        action = Action.NO_OP
        params = {}

        def apply(self, action, parameters):
            self.action, self.params = action, parameters
            return HookAcknowledgement(True, "harness", "ack", action, {"mcs_cap": 5})

        def readback(self):
            return ConfigurationReadback(True, "consumer", "observed", self.action, self.params)

    adapter = ActionAdapter(Backend(), {})
    assert adapter.reconcile().successful
    result = adapter.execute(Action.MCS_BACKOFF, [True]*5, {"mcs_cap": 6})
    assert result.parameter_match is False
    assert result.configuration_state_status is ConfigurationStateStatus.UNKNOWN
    assert not adapter.nontrivial_authority_ready
