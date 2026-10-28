"""Argument validation, call caps, final validation and PII redaction."""

import unittest
from collections import Counter

from helpers import DATA
from triage_agent.guardrails import Guardrails, redact_args, redact_pii, validate_args
from triage_agent.models import Final, ToolCall
from triage_agent.tools import build_default_registry


class ArgValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_default_registry(DATA)
        self.guardrails = Guardrails(self.registry)

    def errors(self, tool: str, **args: object) -> list[str]:
        spec = self.registry.get(tool)
        assert spec is not None
        return validate_args(spec, args)

    def test_unknown_tool_is_rejected(self) -> None:
        rejection = self.guardrails.check_tool_call(ToolCall("delete_account", {}), Counter())
        assert rejection is not None
        self.assertEqual(rejection.kind, "unknown_tool")
        self.assertIn("available:", rejection.message)

    def test_wrong_type_is_rejected(self) -> None:
        self.assertIn("field 'level' must be int, got str", self.errors("set_priority", level="high", reason="x"))
        self.assertIn("field 'level' must be int, got bool", self.errors("set_priority", level=True, reason="x"))
        self.assertTrue(any("must be str" in e for e in self.errors("assign_team", team=3)))

    def test_enum_is_enforced(self) -> None:
        self.assertEqual(self.errors("assign_team", team="sales"),
                         ["field 'team' must be one of billing, auth, outage, feature_request, abuse"])

    def test_range_is_enforced(self) -> None:
        for level in (0, 6):
            self.assertEqual(self.errors("set_priority", level=level, reason="x"), ["field 'level' must be between 1 and 5"])
        self.assertEqual(self.errors("search_similar_tickets", query="q", k=11), ["field 'k' must be between 1 and 10"])

    def test_missing_required_and_unknown_fields(self) -> None:
        self.assertEqual(self.errors("set_priority", level=3), ["missing required field 'reason'"])
        self.assertEqual(self.errors("assign_team", team="billing", note="x"), ["unknown field 'note'"])

    def test_empty_strings_are_rejected(self) -> None:
        self.assertEqual(self.errors("draft_reply", text="   "), ["field 'text' must not be empty"])

    def test_valid_calls_pass(self) -> None:
        self.assertEqual(self.errors("search_similar_tickets", query="refund"), [])
        self.assertEqual(self.errors("search_similar_tickets", query="refund", k=5), [])
        self.assertIsNone(self.guardrails.check_tool_call(ToolCall("assign_team", {"team": "auth"}), Counter()))

    def test_per_tool_call_cap(self) -> None:
        call = ToolCall("search_similar_tickets", {"query": "refund"})
        self.assertIsNone(self.guardrails.check_tool_call(call, Counter({"search_similar_tickets": 2})))
        rejection = self.guardrails.check_tool_call(call, Counter({"search_similar_tickets": 3}))
        assert rejection is not None
        self.assertEqual(rejection.kind, "call_cap")
        escalate = ToolCall("escalate", {"reason": "threat"})
        self.assertIsNotNone(self.guardrails.check_tool_call(escalate, Counter({"escalate": 1})))

    def test_final_needs_valid_team_and_priority(self) -> None:
        self.assertIsNone(self.guardrails.check_final(Final("billing", 3, "s", "r")))
        for bad in (Final("sales", 3, "s", "r"), Final("billing", 9, "s", "r"), Final("billing", True, "s", "r")):
            rejection = self.guardrails.check_final(bad)
            assert rejection is not None
            self.assertEqual(rejection.kind, "invalid_final")


class PiiRedactionTests(unittest.TestCase):
    def test_emails(self) -> None:
        self.assertEqual(redact_pii("mail ops.team+x@example.co.uk now"), "mail [EMAIL] now")

    def test_phone_variants(self) -> None:
        for phone in ("555-201-7788", "(555) 310-0199", "555.310.0199", "5553100199", "+1 555 310 0199", "+44 555-310-0199"):
            self.assertEqual(redact_pii(f"call {phone} today"), "call [PHONE] today", phone)

    def test_card_like_numbers(self) -> None:
        for card in ("4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111", "4222222222222", "4111111111111111111"):
            self.assertEqual(redact_pii(f"card {card} please"), "card [CARD] please", card)

    def test_non_pii_is_preserved(self) -> None:
        text = "INV-20931 for $1,299.00 on 2026-10-03 at 06:10 UTC, order 88421307, 900 users, ACC-1001"
        self.assertEqual(redact_pii(text), text)

    def test_redact_args_touches_only_strings(self) -> None:
        args = redact_args({"query": "reach me at a@b.io", "k": 3})
        self.assertEqual(args, {"query": "reach me at [EMAIL]", "k": 3})


if __name__ == "__main__":
    unittest.main()
