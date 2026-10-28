"""Shared test helpers: project paths, an in-memory scripted model and agent factories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from triage_agent.agent import Agent, Ticket
from triage_agent.models import Final, ToolCall
from triage_agent.telemetry import Telemetry
from triage_agent.tools import build_default_registry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIXTURES = ROOT / "fixtures" / "replay"
GOOD_FINAL = Final("billing", 3, "Duplicate charge.", "We will refund the duplicate.")


class ScriptedModel:
    """Returns canned responses in order and keeps a copy of every conversation it was shown."""

    name = "scripted"

    def __init__(self, responses: list[ToolCall | Final]) -> None:
        self._responses = list(responses)
        self.seen: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ToolCall | Final:
        self.seen.append([dict(m) for m in messages])
        return self._responses.pop(0)


def make_ticket(**overrides: Any) -> Ticket:
    fields: dict[str, Any] = {"id": "T900", "subject": "Charged twice", "body": "Please refund the duplicate charge.",
                              "account_id": "ACC-1002", "expected_team": "billing", "expected_priority": 3}
    return Ticket(**{**fields, **overrides})


def make_agent(model: Any, max_steps: int = 8) -> tuple[Agent, Telemetry]:
    telemetry = Telemetry()
    return Agent(model, build_default_registry(DATA), telemetry, max_steps), telemetry


def tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every tool_result block in a conversation, in order."""
    return [block for m in messages if m["role"] == "user" and isinstance(m["content"], list)
            for block in m["content"] if block["type"] == "tool_result"]
