"""Checks that sit between the model and the tools.

Nothing here trusts the model: tool names and arguments are validated against
the registry specs, call caps are enforced per ticket, and the final answer is
checked before it is accepted. PII redaction runs before any ticket text is
sent to the model.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .models import Final, ToolCall
from .tools import MAX_PRIORITY, MIN_PRIORITY, TEAMS, ArgSpec, ToolRegistry, ToolSpec

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)")


def redact_pii(text: str) -> str:
    """Replace emails, card-like numbers (13 to 19 digits) and phone numbers with placeholders."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _CARD.sub("[CARD]", text)
    return _PHONE.sub("[PHONE]", text)


def redact_args(args: dict[str, Any]) -> dict[str, Any]:
    """Redact string arguments before they are written to telemetry."""
    return {key: redact_pii(value) if isinstance(value, str) else value for key, value in args.items()}


@dataclass(frozen=True)
class Rejection:
    """Why a model action was refused. The message is fed back to the model."""

    kind: str
    message: str


def _check_value(arg_spec: ArgSpec, value: Any) -> list[str]:
    """Type, emptiness, enum and range checks for one argument value."""
    if isinstance(value, bool) or not isinstance(value, arg_spec.type):
        return [f"field '{arg_spec.name}' must be {arg_spec.type.__name__}, got {type(value).__name__}"]
    if arg_spec.type is str and not value.strip():
        return [f"field '{arg_spec.name}' must not be empty"]
    if arg_spec.enum is not None and value not in arg_spec.enum:
        return [f"field '{arg_spec.name}' must be one of {', '.join(arg_spec.enum)}"]
    too_low = arg_spec.minimum is not None and value < arg_spec.minimum
    too_high = arg_spec.maximum is not None and value > arg_spec.maximum
    if too_low or too_high:
        return [f"field '{arg_spec.name}' must be between {arg_spec.minimum} and {arg_spec.maximum}"]
    return []


def validate_args(spec: ToolSpec, args: dict[str, Any]) -> list[str]:
    """Return every problem with args against spec; an empty list means valid."""
    known = {arg.name for arg in spec.args}
    errors = [f"unknown field '{name}'" for name in args if name not in known]
    for arg_spec in spec.args:
        if arg_spec.name not in args:
            if arg_spec.required:
                errors.append(f"missing required field '{arg_spec.name}'")
            continue
        errors.extend(_check_value(arg_spec, args[arg_spec.name]))
    return errors


class Guardrails:
    """Stateless policy over a registry; per-ticket counters are passed in by the agent."""

    def __init__(self, registry: ToolRegistry, max_steps: int = 8) -> None:
        self._registry = registry
        self.max_steps = max_steps

    def check_tool_call(self, call: ToolCall, calls_so_far: Counter[str]) -> Rejection | None:
        """Validate name, arguments and the per-tool call cap."""
        spec = self._registry.get(call.name)
        if spec is None:
            return Rejection("unknown_tool", f"unknown tool '{call.name}'; available: {', '.join(self._registry.names())}")
        errors = validate_args(spec, call.args)
        if errors:
            return Rejection("invalid_args", "; ".join(errors))
        if calls_so_far[call.name] >= spec.max_calls:
            return Rejection("call_cap", f"'{call.name}' may be called at most {spec.max_calls} times per ticket")
        return None

    def check_final(self, final: Final) -> Rejection | None:
        """A final answer needs a known team and an in-range integer priority."""
        if final.team not in TEAMS:
            return Rejection("invalid_final", f"team must be one of {', '.join(TEAMS)}, got '{final.team}'")
        valid_priority = isinstance(final.priority, int) and not isinstance(final.priority, bool)
        if not valid_priority or not MIN_PRIORITY <= final.priority <= MAX_PRIORITY:
            return Rejection("invalid_final", f"priority must be an integer from {MIN_PRIORITY} to {MAX_PRIORITY}")
        return None
