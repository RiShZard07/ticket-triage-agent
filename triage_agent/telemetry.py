"""Dependency-free JSON-lines tracing for agent runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import TriageResult


@dataclass
class StepEvent:
    """One model step. kind is tool_call, tool_error, rejected_tool_call, final or rejected_final."""

    ticket_id: str
    step: int
    kind: str
    model_latency_ms: float
    tool_name: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    result_size: int = 0
    error: str | None = None


class Telemetry:
    """Keeps every record in memory and, when a path is given, appends it as a JSON line."""

    def __init__(self, path: Path | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self._path = path
        if path is not None:
            path.write_text("")

    def record_step(self, event: StepEvent) -> None:
        self._write({"event": "step", **asdict(event)})

    def record_result(self, result: TriageResult) -> None:
        self._write({"event": "ticket_result", **asdict(result)})

    def record_summary(self, summary: dict[str, Any]) -> None:
        self._write({"event": "run_summary", **summary})

    def _write(self, record: dict[str, Any]) -> None:
        self.events.append(record)
        if self._path is not None:
            with self._path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
