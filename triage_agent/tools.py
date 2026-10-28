"""Tool registry with explicit argument specs, plus the six triage tools."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEAMS: tuple[str, ...] = ("billing", "auth", "outage", "feature_request", "abuse")
MIN_PRIORITY = 1
MAX_PRIORITY = 5
_TOKEN = re.compile(r"[a-z0-9]+")


class ToolError(Exception):
    """Raised by a tool handler. The message is fed back to the model."""


@dataclass
class TriageState:
    """Per-ticket scratchpad written by the action tools."""

    team: str | None = None
    priority: int | None = None
    priority_reason: str = ""
    reply_draft: str = ""
    escalation_reason: str | None = None


@dataclass(frozen=True)
class ArgSpec:
    """Declared shape of one tool argument."""

    name: str
    type: type
    description: str
    required: bool = True
    enum: tuple[str, ...] | None = None
    minimum: int | None = None
    maximum: int | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Render the argument as a JSON Schema property."""
        schema: dict[str, Any] = {
            "type": "string" if self.type is str else "integer",
            "description": self.description,
        }
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        return schema


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call: its schema, handler and per-ticket call cap."""

    name: str
    description: str
    args: tuple[ArgSpec, ...]
    handler: Callable[..., Any]
    max_calls: int = 3

    def to_json_schema(self) -> dict[str, Any]:
        """Render the tool in the Messages API tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {a.name: a.to_json_schema() for a in self.args},
                "required": [a.name for a in self.args if a.required],
                "additionalProperties": False,
            },
        }


class ToolRegistry:
    """Name-keyed collection of ToolSpecs."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs)

    def to_json_schemas(self) -> list[dict[str, Any]]:
        return [spec.to_json_schema() for spec in self._specs.values()]

    def execute(self, name: str, args: dict[str, Any], state: TriageState) -> Any:
        """Run a call that guardrails have already validated."""
        return self._specs[name].handler(state, **args)


def tokenize(text: str) -> Counter[str]:
    """Lowercase term-frequency vector over alphanumeric tokens."""
    return Counter(_TOKEN.findall(text.lower()))


def cosine(a: Counter[str], b: Counter[str]) -> float:
    """Cosine similarity between two term-frequency vectors."""
    dot = sum(count * b[token] for token, count in a.items())
    norm = math.sqrt(sum(c * c for c in a.values())) * math.sqrt(sum(c * c for c in b.values()))
    return dot / norm if norm else 0.0


class KnowledgeBase:
    """Resolved historical tickets searchable by term-frequency cosine similarity."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._entries = entries
        self._vectors = [tokenize(f"{e['title']} {e['body']}") for e in entries]

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Top-k entries by score; ties break on id so results are deterministic."""
        query_vector = tokenize(query)
        scored = [(cosine(query_vector, v), e) for v, e in zip(self._vectors, self._entries)]
        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        return [
            {"id": e["id"], "title": e["title"], "team": e["team"], "resolution": e["resolution"], "score": round(s, 3)}
            for s, e in scored[:k]
        ]


def build_default_registry(data_dir: Path) -> ToolRegistry:
    """Register the six triage tools over the JSON data in data_dir."""
    kb = KnowledgeBase(json.loads((data_dir / "knowledge_base.json").read_text()))
    accounts = {a["account_id"]: a for a in json.loads((data_dir / "accounts.json").read_text())}

    def search_similar_tickets(_state: TriageState, query: str, k: int = 3) -> dict[str, Any]:
        return {"results": kb.search(query, k)}

    def lookup_account(_state: TriageState, account_id: str) -> dict[str, Any]:
        if account_id not in accounts:
            raise ToolError(f"unknown account '{account_id}'")
        return accounts[account_id]

    def set_priority(state: TriageState, level: int, reason: str) -> dict[str, Any]:
        state.priority, state.priority_reason = level, reason
        return {"priority": level}

    def assign_team(state: TriageState, team: str) -> dict[str, Any]:
        state.team = team
        return {"team": team}

    def draft_reply(state: TriageState, text: str) -> dict[str, Any]:
        state.reply_draft = text
        return {"draft_chars": len(text)}

    def escalate(state: TriageState, reason: str) -> dict[str, Any]:
        state.escalation_reason = reason
        return {"escalated": True}

    registry = ToolRegistry()
    registry.register(ToolSpec(
        "search_similar_tickets", "Find resolved historical tickets similar to a query.",
        (ArgSpec("query", str, "Free-text description of the issue."),
         ArgSpec("k", int, "Number of results, 1 to 10.", required=False, minimum=1, maximum=10)),
        search_similar_tickets))
    registry.register(ToolSpec(
        "lookup_account", "Fetch the customer's plan, seat count and open incident flag.",
        (ArgSpec("account_id", str, "Account id from the ticket header, e.g. ACC-1001."),), lookup_account))
    registry.register(ToolSpec(
        "set_priority", "Record the ticket priority.",
        (ArgSpec("level", int, "1 (low) to 5 (critical).", minimum=MIN_PRIORITY, maximum=MAX_PRIORITY),
         ArgSpec("reason", str, "One sentence justifying the level.")), set_priority))
    registry.register(ToolSpec(
        "assign_team", "Route the ticket to a team.",
        (ArgSpec("team", str, "Owning team.", enum=TEAMS),), assign_team))
    registry.register(ToolSpec(
        "draft_reply", "Save a draft reply to the customer for a human to review.",
        (ArgSpec("text", str, "Reply text."),), draft_reply))
    registry.register(ToolSpec(
        "escalate", "Hand the ticket to a human immediately and stop.",
        (ArgSpec("reason", str, "Why a human must take over."),), escalate, max_calls=1))
    return registry
