"""OCUDU WebSocket backend for the five locked research actions."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import time
from typing import Mapping

from src.controller.action_mask import Action
from src.xapp.adapters.actions import ConfigurationReadback, HookAcknowledgement
from tools.send_ocudu_command import send


@dataclass(frozen=True)
class OcuduWebSocketConfig:
    host: str = "127.0.0.1"
    port: int = 8001
    timeout_seconds: float = 1.0
    scheduler_log: str | None = None
    readback_timeout_seconds: float = 0.75

    def __post_init__(self) -> None:
        if not self.host or not 1 <= self.port <= 65535 or self.timeout_seconds <= 0:
            raise ValueError("invalid OCUDU WebSocket configuration")


class OcuduWebSocketResearchBackend:
    """Apply one atomic scheduler state and validate OCUDU's native ack envelope."""

    def __init__(self, config: OcuduWebSocketConfig = OcuduWebSocketConfig()) -> None:
        self.config = config
        self.scheduler_log = config.scheduler_log or os.environ.get("ORAN_SCHEDULER_READBACK_LOG")
        self._command_started_ns = 0
        self._command_log_offset = 0
        self._command_count = 0
        self._drop_ack_at = int(os.environ.get("ORAN_TEST_DROP_ACK_COMMAND_INDEX", "0"))
        self._drop_first_nontrivial = os.environ.get("ORAN_TEST_DROP_FIRST_NONTRIVIAL_ACK", "0") == "1"
        self._fault_injected = False

    @staticmethod
    def _document(action: Action, parameters: Mapping[str, object]) -> dict[str, object]:
        document: dict[str, object] = {
            "cmd": "research_action_set",
            "action": f"A{int(action)}",
        }
        document.update(parameters)
        return document

    def apply(self, action: Action, parameters: Mapping[str, object]) -> HookAcknowledgement:
        document = self._document(action, parameters)
        self._command_started_ns = time.time_ns()
        if self.scheduler_log and Path(self.scheduler_log).exists():
            self._command_log_offset = Path(self.scheduler_log).stat().st_size
        response = send(
            self.config.host,
            self.config.port,
            document,
            self.config.timeout_seconds,
        )
        self._command_count += 1
        drop_scheduled = self._drop_ack_at and self._command_count == self._drop_ack_at
        drop_nontrivial = self._drop_first_nontrivial and not self._fault_injected and action is not Action.NO_OP
        if drop_scheduled or drop_nontrivial:
            self._fault_injected = True
            raise TimeoutError("predeclared_integration_fault_ack_dropped_after_application")
        if not isinstance(response, dict):
            return HookAcknowledgement(False, "ocudu-websocket", "non_object_response")
        # The pinned OCUDU remote server acknowledges successful commands with
        # ``cmd`` and ``timestamp``. Error envelopes explicitly carry
        # ``status=error``. Scheduler-consumption evidence is captured
        # separately from the gNB log.
        accepted = response.get("cmd") == "research_action_set" and response.get("status") != "error"
        reported_name = response.get("action")
        reported_action = Action.__members__.get(str(reported_name)) if reported_name is not None else None
        reported_parameters = response.get("applied_parameters", {}) if accepted else {}
        if not isinstance(reported_parameters, dict):
            reported_parameters = {}
        return HookAcknowledgement(
            accepted=accepted,
            backend="ocudu-websocket",
            message="applied" if accepted else str(response.get("message", response)),
            reported_action=reported_action,
            reported_parameters=reported_parameters,
        )

    @staticmethod
    def parse_consumed(text: str, *, not_before_ns: int) -> ConfigurationReadback:
        """Read snapshots emitted at the actual PRB/MCS/slice consumption sites."""
        latest: dict[str, dict[str, str]] = {}
        for line in text.splitlines():
            if "RESEARCH_STATE_CONSUMED " not in line:
                continue
            values = dict(re.findall(r"(\w+)=([^\s]+)", line))
            try:
                if int(values["timestamp_ns"]) < not_before_ns:
                    continue
                if values.get("scope") not in {"prb", "mcs", "slice"}:
                    continue
                latest[values["scope"]] = values
            except (KeyError, ValueError):
                continue
        if set(latest) != {"prb", "mcs", "slice"}:
            return ConfigurationReadback(False, "ocudu-scheduler-consumers", "incomplete_consumer_readback")
        identities = {(v.get("pid"), v.get("generation"), v.get("action")) for v in latest.values()}
        keys = ("action", "prb_start", "prb_stop", "mcs_cap", "slice_min_pct", "slice_max_pct")
        if len(identities) != 1 or len({tuple(v.get(k) for k in keys) for v in latest.values()}) != 1:
            return ConfigurationReadback(False, "ocudu-scheduler-consumers", "consumer_generation_disagreement")
        values = latest["mcs"]
        try:
            action = Action(int(values["action"][1:]))
            relevant = {
                Action.NO_OP: (), Action.PRB_AVOID_SHIFT: ("prb_start", "prb_stop"),
                Action.MCS_BACKOFF: ("mcs_cap",),
                Action.SLICE_PRB_BOOST: ("slice_min_pct", "slice_max_pct"),
                Action.COMPOSITE_LOCAL_DEFENSE: ("prb_start", "prb_stop", "mcs_cap"),
            }[action]
            parameters = {key: int(values[key]) for key in relevant}
            observed_ns = min(int(v["timestamp_ns"]) for v in latest.values())
        except (KeyError, ValueError):
            return ConfigurationReadback(False, "ocudu-scheduler-consumers", "malformed_consumer_snapshot")
        return ConfigurationReadback(
            True, "ocudu-scheduler-consumers", "three_scheduler_consumers_agree",
            action, parameters, observed_ns, values["generation"],
            "gnb-pid-" + values["pid"], "scheduler_software_configuration",
        )

    def readback(self) -> ConfigurationReadback:
        if not self.scheduler_log:
            return ConfigurationReadback(False, "ocudu-scheduler-consumers", "scheduler_log_unconfigured")
        deadline = time.monotonic() + self.config.readback_timeout_seconds
        result = ConfigurationReadback(False, "ocudu-scheduler-consumers", "scheduler_log_unavailable")
        while time.monotonic() < deadline:
            try:
                with Path(self.scheduler_log).open(encoding="utf-8", errors="replace") as stream:
                    stream.seek(self._command_log_offset)
                    text = stream.read()
                result = self.parse_consumed(text, not_before_ns=self._command_started_ns)
                if result.successful:
                    # Old generations that happened to run after transmission
                    # cannot verify this command. Match the publication record
                    # observed after the command began, then consumer snapshots.
                    publications = re.findall(r"RESEARCH_ACTION_APPLIED generation=(\d+)", text)
                    if self._command_started_ns == 0 or (
                        publications and result.configuration_version == publications[-1]
                    ):
                        return result
            except OSError:
                pass
            time.sleep(0.01)
        return result if not result.successful else ConfigurationReadback(
            False, "ocudu-scheduler-consumers", "publication_consumer_version_not_aligned",
        )
