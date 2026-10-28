"""The agent loop: redact the ticket, prompt the model, run tools through guardrails, repeat."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter

from .guardrails import Guardrails, redact_args, redact_pii
from .models import TICKET_ID_LINE, Final, Message, Model, ToolCall
from .telemetry import StepEvent, Telemetry
from .tools import ToolError, ToolRegistry, TriageState

SYSTEM_PROMPT = """You triage customer support tickets for a small SaaS product by calling tools one at a time.

Work in this order unless the ticket is clearly urgent:
1. search_similar_tickets with a short description of the problem.
2. lookup_account using the Account ID from the ticket header.
3. assign_team, set_priority and draft_reply to record your decisions.
4. Reply with a single JSON object and nothing else:
   {"team": "<team>", "priority": <1-5>, "summary": "<one sentence>", "reply_draft": "<reply>"}

Teams: billing, auth, outage, feature_request, abuse.
Priority: 1 low, 2 normal, 3 elevated, 4 high, 5 critical. Outages for enterprise accounts or
accounts with an open incident are 4 or 5. Feature requests are 1 to 3. Threats, phishing and
safety issues are 4 or 5.
Call escalate instead of finishing when a human must act now (threats, legal exposure, phishing)
or when you cannot decide. If a tool call is rejected, read the message and retry with corrected
arguments. Personal data has been replaced with placeholders such as [EMAIL]; do not try to recover it.
"""


@dataclass(frozen=True)
class Ticket:
    """A support ticket. Expected labels are present only for golden tickets."""

    id: str
    subject: str
    body: str
    account_id: str
    expected_team: str | None = None
    expected_priority: int | None = None


@dataclass
class TriageResult:
    """Outcome of one agent episode plus the counters the evaluator needs."""

    ticket_id: str
    team: str | None
    priority: int | None
    summary: str
    reply_draft: str
    escalated: bool
    escalation_reason: str | None
    steps: int
    tool_calls: int
    rejections: int
    tool_errors: int
    model_latencies_ms: list[float]


@dataclass
class _Episode:
    """Mutable per-ticket bookkeeping shared by the loop helpers."""

    ticket: Ticket
    messages: list[Message]
    state: TriageState = field(default_factory=TriageState)
    calls: Counter[str] = field(default_factory=Counter)
    steps: int = 0
    tool_calls: int = 0
    rejections: int = 0
    tool_errors: int = 0
    latencies: list[float] = field(default_factory=list)


def render_ticket(ticket: Ticket) -> str:
    """Ticket text as the model sees it: header lines plus redacted subject and body."""
    return (
        f"{TICKET_ID_LINE}{ticket.id}\nAccount ID: {ticket.account_id}\n"
        f"Subject: {redact_pii(ticket.subject)}\n\n{redact_pii(ticket.body)}"
    )


class Agent:
    """Runs one ticket to a Final or an escalation within a step budget."""

    def __init__(self, model: Model, registry: ToolRegistry, telemetry: Telemetry, max_steps: int = 8) -> None:
        self._model = model
        self._registry = registry
        self._telemetry = telemetry
        self._guardrails = Guardrails(registry, max_steps)

    def run(self, ticket: Ticket) -> TriageResult:
        """Loop until the model finalizes, escalates, or the step budget runs out."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": render_ticket(ticket)}]
        episode = _Episode(ticket, messages)
        tools = self._registry.to_json_schemas()
        while episode.steps < self._guardrails.max_steps:
            episode.steps += 1
            started = perf_counter()
            response = self._model.complete(episode.messages, tools)
            latency_ms = round((perf_counter() - started) * 1000, 3)
            episode.latencies.append(latency_ms)
            if isinstance(response, Final):
                if self._accept_final(episode, response, latency_ms):
                    return self._result(episode, response)
            else:
                self._run_tool(episode, response, latency_ms)
                if episode.state.escalation_reason is not None:
                    return self._result(episode, None, episode.state.escalation_reason)
        return self._result(episode, None, "step budget exhausted")

    def _accept_final(self, episode: _Episode, final: Final, latency_ms: float) -> bool:
        """Validate the final answer; on rejection, tell the model why and let it retry."""
        rejection = self._guardrails.check_final(final)
        kind = "final" if rejection is None else "rejected_final"
        error = None if rejection is None else rejection.message
        self._telemetry.record_step(StepEvent(episode.ticket.id, episode.steps, kind, latency_ms, error=error))
        if rejection is None:
            return True
        episode.rejections += 1
        episode.messages.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(final.__dict__)}]})
        episode.messages.append({"role": "user", "content": [{"type": "text", "text": f"Final answer rejected: {error}. Reply again."}]})
        return False

    def _run_tool(self, episode: _Episode, call: ToolCall, latency_ms: float) -> None:
        """Check the call, execute it if allowed, and append the outcome as a tool result."""
        call_id = call.call_id or f"call_{episode.steps}"
        rejection = self._guardrails.check_tool_call(call, episode.calls)
        if rejection is not None:
            episode.rejections += 1
            kind, output, is_error = "rejected_tool_call", f"Rejected: {rejection.message}", True
        else:
            episode.calls[call.name] += 1
            episode.tool_calls += 1
            try:
                output = json.dumps(self._registry.execute(call.name, call.args, episode.state))
                kind, is_error = "tool_call", False
            except ToolError as exc:
                episode.tool_errors += 1
                kind, output, is_error = "tool_error", f"Tool error: {exc}", True
        self._telemetry.record_step(StepEvent(
            episode.ticket.id, episode.steps, kind, latency_ms, tool_name=call.name,
            args=redact_args(call.args), result_size=len(output), error=output if is_error else None))
        episode.messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": call.name, "input": call.args}]})
        episode.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call_id, "content": output, "is_error": is_error}]})

    def _result(self, episode: _Episode, final: Final | None, escalation_reason: str | None = None) -> TriageResult:
        """Build the result from the Final, or from tool state when the episode escalated."""
        state = episode.state
        result = TriageResult(
            ticket_id=episode.ticket.id,
            team=final.team if final else state.team,
            priority=final.priority if final else state.priority,
            summary=final.summary if final else f"Escalated: {escalation_reason}",
            reply_draft=final.reply_draft if final else state.reply_draft,
            escalated=final is None,
            escalation_reason=escalation_reason,
            steps=episode.steps,
            tool_calls=episode.tool_calls,
            rejections=episode.rejections,
            tool_errors=episode.tool_errors,
            model_latencies_ms=episode.latencies,
        )
        self._telemetry.record_result(result)
        return result
