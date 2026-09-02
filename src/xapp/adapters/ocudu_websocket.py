"""OCUDU WebSocket backend for the five locked research actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.controller.action_mask import Action
from src.xapp.adapters.actions import HookAcknowledgement
from tools.send_ocudu_command import send


@dataclass(frozen=True)
class OcuduWebSocketConfig:
    host: str = "127.0.0.1"
    port: int = 8001
    timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.host or not 1 <= self.port <= 65535 or self.timeout_seconds <= 0:
            raise ValueError("invalid OCUDU WebSocket configuration")


class OcuduWebSocketResearchBackend:
    """Apply one atomic scheduler state and validate OCUDU's native ack envelope."""

    def __init__(self, config: OcuduWebSocketConfig = OcuduWebSocketConfig()) -> None:
        self.config = config

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
        response = send(
            self.config.host,
            self.config.port,
            document,
            self.config.timeout_seconds,
        )
        if not isinstance(response, dict):
            return HookAcknowledgement(False, "ocudu-websocket", "non_object_response", {})
        # The pinned OCUDU remote server acknowledges successful commands with
        # ``cmd`` and ``timestamp``. Error envelopes explicitly carry
        # ``status=error``. Scheduler-consumption evidence is captured
        # separately from the gNB log.
        accepted = response.get("cmd") == "research_action_set" and response.get("status") != "error"
        return HookAcknowledgement(
            accepted,
            "ocudu-websocket",
            "applied" if accepted else str(response.get("message", response)),
            dict(parameters) if accepted else {},
        )
