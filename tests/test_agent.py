"""Agent loop behaviour: happy path, redaction, budget, recovery, tool errors, escalation."""

import unittest

from helpers import DATA, FIXTURES, GOOD_FINAL, ScriptedModel, make_agent, make_ticket, tool_results
from triage_agent.evaluate import load_tickets
from triage_agent.models import Final, ReplayModel, ToolCall

SEARCH = ToolCall("search_similar_tickets", {"query": "duplicate charge"})
LOOKUP = ToolCall("lookup_account", {"account_id": "ACC-1002"})


class AgentTests(unittest.TestCase):
    def test_happy_path_with_replay_model(self) -> None:
        ticket = next(t for t in load_tickets(DATA / "golden_tickets.json") if t.id == "T001")
        agent, telemetry = make_agent(ReplayModel(FIXTURES))
        result = agent.run(ticket)
        self.assertEqual((result.team, result.priority, result.escalated), ("billing", 3, False))
        self.assertEqual((result.steps, result.tool_calls, result.rejections, result.tool_errors), (6, 5, 0, 0))
        self.assertTrue(result.reply_draft.startswith("Thanks for flagging"))
        kinds = [e["kind"] for e in telemetry.events if e["event"] == "step"]
        self.assertEqual(kinds, ["tool_call"] * 5 + ["final"])
        self.assertEqual(telemetry.events[-1]["event"], "ticket_result")

    def test_pii_is_redacted_before_the_model_sees_it(self) -> None:
        model = ScriptedModel([GOOD_FINAL])
        agent, _ = make_agent(model)
        agent.run(make_ticket(subject="Call 555-201-7788", body="Mail ops@bluefin.example, card 4111 1111 1111 1111."))
        prompt = model.seen[0][1]["content"]
        self.assertTrue(prompt.startswith("Ticket ID: T900\nAccount ID: ACC-1002\nSubject: Call [PHONE]"))
        self.assertIn("Mail [EMAIL], card [CARD].", prompt)
        for raw in ("555-201-7788", "ops@bluefin.example", "4111"):
            self.assertNotIn(raw, prompt)

    def test_step_budget_escalates_to_a_human(self) -> None:
        agent, _ = make_agent(ScriptedModel([SEARCH, LOOKUP, SEARCH, LOOKUP]), max_steps=3)
        result = agent.run(make_ticket())
        self.assertTrue(result.escalated)
        self.assertEqual(result.escalation_reason, "step budget exhausted")
        self.assertEqual((result.steps, result.tool_calls, result.team), (3, 3, None))
        self.assertEqual(result.summary, "Escalated: step budget exhausted")

    def test_recovery_after_rejected_tool_call(self) -> None:
        bad = ToolCall("set_priority", {"level": 7, "reason": "urgent"})
        good = ToolCall("set_priority", {"level": 4, "reason": "urgent"})
        model = ScriptedModel([bad, good, Final("billing", 4, "s", "r")])
        agent, telemetry = make_agent(model)
        result = agent.run(make_ticket())
        feedback = tool_results(model.seen[1])[0]
        self.assertTrue(feedback["is_error"])
        self.assertEqual(feedback["content"], "Rejected: field 'level' must be between 1 and 5")
        self.assertEqual((result.priority, result.rejections, result.tool_calls, result.steps), (4, 1, 1, 3))
        self.assertEqual(telemetry.events[0]["kind"], "rejected_tool_call")

    def test_tool_error_is_surfaced_to_the_model(self) -> None:
        model = ScriptedModel([ToolCall("lookup_account", {"account_id": "ACC-9999"}), LOOKUP, GOOD_FINAL])
        agent, telemetry = make_agent(model)
        result = agent.run(make_ticket())
        feedback = tool_results(model.seen[1])[0]
        self.assertEqual((feedback["is_error"], feedback["content"]), (True, "Tool error: unknown account 'ACC-9999'"))
        self.assertFalse(tool_results(model.seen[2])[1]["is_error"])
        self.assertEqual((result.tool_errors, result.rejections, result.team), (1, 0, "billing"))
        self.assertEqual(telemetry.events[0]["kind"], "tool_error")

    def test_escalate_tool_ends_the_episode(self) -> None:
        model = ScriptedModel([ToolCall("assign_team", {"team": "abuse"}), ToolCall("escalate", {"reason": "threat"}), GOOD_FINAL])
        agent, _ = make_agent(model)
        result = agent.run(make_ticket())
        self.assertEqual((result.escalated, result.escalation_reason, result.team, result.steps), (True, "threat", "abuse", 2))

    def test_rejected_final_is_fed_back_and_retried(self) -> None:
        model = ScriptedModel([Final("sales", 3, "s", "r"), GOOD_FINAL])
        agent, telemetry = make_agent(model)
        result = agent.run(make_ticket())
        self.assertEqual((result.team, result.rejections, result.steps), ("billing", 1, 2))
        self.assertIn("Final answer rejected: team must be one of", model.seen[1][-1]["content"][0]["text"])
        self.assertEqual(telemetry.events[0]["kind"], "rejected_final")

    def test_tool_use_and_tool_result_ids_match(self) -> None:
        model = ScriptedModel([ToolCall("assign_team", {"team": "billing"}, call_id="toolu_9"), GOOD_FINAL])
        agent, _ = make_agent(model)
        agent.run(make_ticket())
        assistant, user = model.seen[1][-2], model.seen[1][-1]
        self.assertEqual(assistant["content"][0]["id"], "toolu_9")
        self.assertEqual(user["content"][0]["tool_use_id"], "toolu_9")


if __name__ == "__main__":
    unittest.main()
