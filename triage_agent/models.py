"""Model protocol and adapters: scripted replay, Claude over the Messages API, and a recorder.

Messages use the Messages API shape (role plus string or content-block list) so
the Claude adapter can pass them through unchanged. The first user message
starts with a "Ticket ID:" line, which is how the replay and recording models
know which ticket a conversation belongs to without extra protocol state.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"
TICKET_ID_LINE = "Ticket ID: "
Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """The model wants a tool run with these arguments."""

    name: str
    args: dict[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class Final:
    """The model's finished triage decision."""

    team: str
    priority: int
    summary: str
    reply_draft: str


class Model(Protocol):
    """Anything that turns a conversation plus tool schemas into one response."""

    name: str

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ToolCall | Final: ...


class ReplayError(Exception):
    """A fixture is missing or has run out of scripted steps."""


def ticket_id_from_messages(messages: list[Message]) -> str:
    """Read the ticket id from the first line of the first user message."""
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], str):
            first_line = message["content"].splitlines()[0]
            if first_line.startswith(TICKET_ID_LINE):
                return first_line[len(TICKET_ID_LINE):].strip()
    raise ReplayError("no user message starting with a 'Ticket ID:' line")


def step_to_dict(response: ToolCall | Final) -> dict[str, Any]:
    """Serialize a model response in the fixture format."""
    if isinstance(response, ToolCall):
        return {"type": "tool_call", "name": response.name, "args": response.args}
    return {"type": "final", **asdict(response)}


def step_from_dict(data: dict[str, Any], call_id: str) -> ToolCall | Final:
    """Inverse of step_to_dict."""
    if data["type"] == "tool_call":
        return ToolCall(data["name"], dict(data["args"]), call_id=call_id)
    return Final(data["team"], data["priority"], data["summary"], data["reply_draft"])


def parse_final(text: str) -> Final:
    """Parse the model's closing JSON object; malformed output becomes a Final that guardrails reject."""
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1]) if start != -1 and end > start else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    try:
        priority = int(str(data.get("priority")))
    except ValueError:
        priority = 0
    return Final(
        team=str(data.get("team", "")),
        priority=priority,
        summary=str(data.get("summary", text.strip()[:200])),
        reply_draft=str(data.get("reply_draft", "")),
    )


class ReplayModel:
    """Returns scripted responses from fixtures/<ticket_id>.json, indexed by step."""

    name = "replay"

    def __init__(self, fixture_dir: Path) -> None:
        self._dir = fixture_dir
        self._scripts: dict[str, list[dict[str, Any]]] = {}

    def _script(self, ticket_id: str) -> list[dict[str, Any]]:
        if ticket_id not in self._scripts:
            path = self._dir / f"{ticket_id}.json"
            if not path.exists():
                raise ReplayError(f"no replay fixture for ticket {ticket_id} at {path}")
            self._scripts[ticket_id] = json.loads(path.read_text())["steps"]
        return self._scripts[ticket_id]

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ToolCall | Final:
        """The step index is the number of assistant turns already in the conversation."""
        ticket_id = ticket_id_from_messages(messages)
        step = sum(1 for message in messages if message["role"] == "assistant")
        script = self._script(ticket_id)
        if step >= len(script):
            raise ReplayError(f"fixture for {ticket_id} has {len(script)} steps but step {step} was requested")
        return step_from_dict(script[step], call_id=f"replay-{ticket_id}-{step}")


def _default_client() -> Any:
    """Import the SDK only when a live client is actually needed."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("the anthropic package is not installed; run: pip install anthropic") from exc
    return anthropic.Anthropic()


class ClaudeModel:
    """Claude through the Messages API with tool use. Inject a fake client for tests."""

    def __init__(self, client: Any | None = None, model: str = DEFAULT_CLAUDE_MODEL, max_tokens: int = 1024) -> None:
        self.name = f"claude:{model}"
        self._model = model
        self._max_tokens = max_tokens
        self._client = client if client is not None else _default_client()

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ToolCall | Final:
        """Send the conversation; map the first tool_use block to a ToolCall, otherwise parse text as a Final."""
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "tools": tools,
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "messages": [m for m in messages if m["role"] != "system"],
        }
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        if system:
            request["system"] = system
        response = self._client.messages.create(**request)
        for block in response.content:
            if block.type == "tool_use":
                return ToolCall(block.name, dict(block.input), call_id=block.id)
        return parse_final("".join(block.text for block in response.content if block.type == "text"))


class RecordingModel:
    """Wraps any model and captures its responses in the replay fixture format."""

    def __init__(self, inner: Model) -> None:
        self._inner = inner
        self.name = inner.name
        self.recorded: dict[str, list[dict[str, Any]]] = {}

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ToolCall | Final:
        response = self._inner.complete(messages, tools)
        self.recorded.setdefault(ticket_id_from_messages(messages), []).append(step_to_dict(response))
        return response

    def save(self, directory: Path) -> list[Path]:
        """Write one <ticket_id>.json per recorded ticket and return the paths."""
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for ticket_id, steps in sorted(self.recorded.items()):
            path = directory / f"{ticket_id}.json"
            path.write_text(json.dumps({"ticket_id": ticket_id, "steps": steps}, indent=2) + "\n")
            paths.append(path)
        return paths
