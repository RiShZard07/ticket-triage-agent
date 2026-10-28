"""Evaluation harness: run the agent over the golden tickets and score it."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from .agent import Agent, Ticket, TriageResult
from .models import Model
from .telemetry import Telemetry
from .tools import build_default_registry


@dataclass(frozen=True)
class Metrics:
    """Aggregate scores for one run. Rates are fractions of all tickets."""

    ticket_count: int
    team_accuracy: float
    priority_exact_accuracy: float
    priority_within_1_accuracy: float
    escalation_rate: float
    mean_steps: float
    mean_tool_calls: float
    guardrail_rejections: int
    tool_errors: int
    model_latency_p50_ms: float
    model_latency_p95_ms: float


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def load_tickets(path: Path) -> list[Ticket]:
    """Read golden tickets from JSON."""
    return [Ticket(**row) for row in json.loads(path.read_text())]


def compute_metrics(tickets: list[Ticket], results: list[TriageResult]) -> Metrics:
    """Score results against the tickets' expected labels, position by position."""
    if not tickets or len(tickets) != len(results):
        raise ValueError("tickets and results must be non-empty and the same length")
    pairs = list(zip(tickets, results))
    latencies = [ms for result in results for ms in result.model_latencies_ms]

    def rate(hits: int) -> float:
        return round(hits / len(pairs), 4)

    return Metrics(
        ticket_count=len(pairs),
        team_accuracy=rate(sum(r.team == t.expected_team for t, r in pairs)),
        priority_exact_accuracy=rate(sum(r.priority == t.expected_priority for t, r in pairs)),
        priority_within_1_accuracy=rate(sum(
            r.priority is not None and t.expected_priority is not None and abs(r.priority - t.expected_priority) <= 1
            for t, r in pairs)),
        escalation_rate=rate(sum(r.escalated for r in results)),
        mean_steps=round(mean(r.steps for r in results), 4),
        mean_tool_calls=round(mean(r.tool_calls for r in results), 4),
        guardrail_rejections=sum(r.rejections for r in results),
        tool_errors=sum(r.tool_errors for r in results),
        model_latency_p50_ms=round(percentile(latencies, 50), 3),
        model_latency_p95_ms=round(percentile(latencies, 95), 3),
    )


def _label(team: str | None, priority: int | None) -> str:
    """Render a team/priority pair for the report, tolerating missing values."""
    return f"{team or 'none'} / {'P' + str(priority) if priority is not None else 'none'}"


def write_report(path: Path, model_name: str, metrics: Metrics, tickets: list[Ticket], results: list[TriageResult]) -> None:
    """Markdown report: a metrics table followed by every ticket that missed or escalated."""
    lines = [
        "# Evaluation report", "", f"Model: `{model_name}`", "",
        "| Metric | Value |", "|---|---|",
        *(f"| {key} | {value} |" for key, value in asdict(metrics).items()),
        "", "## Misses and escalations", "",
        "| Ticket | Expected | Predicted | Escalation | Steps | Rejections |", "|---|---|---|---|---|---|",
    ]
    for ticket, result in zip(tickets, results):
        hit = result.team == ticket.expected_team and result.priority == ticket.expected_priority
        if hit and not result.escalated:
            continue
        lines.append(
            f"| {ticket.id} | {_label(ticket.expected_team, ticket.expected_priority)} | {_label(result.team, result.priority)} "
            f"| {result.escalation_reason or 'none'} | {result.steps} | {result.rejections} |")
    path.write_text("\n".join(lines) + "\n")


def run_eval(model: Model, data_dir: Path, output_dir: Path, max_steps: int = 8) -> Metrics:
    """Run every golden ticket, then write metrics.json, report.md and traces.jsonl."""
    tickets = load_tickets(data_dir / "golden_tickets.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry = Telemetry(output_dir / "traces.jsonl")
    agent = Agent(model, build_default_registry(data_dir), telemetry, max_steps)
    results = [agent.run(ticket) for ticket in tickets]
    metrics = compute_metrics(tickets, results)
    summary = {"model": model.name, **asdict(metrics)}
    telemetry.record_summary(summary)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(output_dir / "report.md", model.name, metrics, tickets, results)
    return metrics
