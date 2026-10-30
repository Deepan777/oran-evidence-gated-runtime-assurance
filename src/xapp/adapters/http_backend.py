"""Authenticated HTTP backend for isolated OCUDU research-control hooks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.controller.action_mask import Action
from src.xapp.adapters.actions import HookAcknowledgement


@dataclass(frozen=True)
class HttpHookConfig:
    base_url: str
    bearer_token: str
    timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        if not self.bearer_token:
            raise ValueError("bearer token must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class HttpResearchHookBackend:
    """Call a loopback/private research hook and verify its acknowledgement."""

    def __init__(self, config: HttpHookConfig) -> None:
        self.config = config

    def apply(self, action: Action, parameters: Mapping[str, object]) -> HookAcknowledgement:
        body = json.dumps({"action": action.name, "parameters": dict(parameters)}).encode("utf-8")
        request = Request(
            f"{self.config.base_url.rstrip('/')}/v1/actions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.bearer_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310 - configured research endpoint
                status = int(response.status)
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            return HookAcknowledgement(False, "ocudu-research-http", f"http_{exc.code}")
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            return HookAcknowledgement(False, "ocudu-research-http", f"transport:{type(exc).__name__}")
        accepted = status == 200 and payload.get("status") == "applied"
        reported_name = payload.get("action")
        reported_action = Action.__members__.get(str(reported_name)) if reported_name is not None else None
        reported_parameters = payload.get("applied_parameters", {}) if accepted else {}
        if not isinstance(reported_parameters, dict):
            reported_parameters = {}
        message = str(payload.get("message", "invalid acknowledgement"))
        return HookAcknowledgement(
            accepted=accepted,
            backend="ocudu-research-http",
            message=message,
            reported_action=reported_action,
            reported_parameters=reported_parameters,
        )

