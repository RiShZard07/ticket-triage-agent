"""Claude adapter mapping with a fake client, replay fixtures, and recording output."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from helpers import DATA, FIXTURES, GOOD_FINAL, ScriptedModel, make_agent, make_ticket
from triage_agent.guardrails import Guardrails
from triage_agent.models import ClaudeModel, Final, RecordingModel, ReplayError, ReplayModel, ToolCall, parse_final
from triage_agent.tools import build_default_registry


class FakeClient:
    """Stands in for anthropic.Anthropic: records requests and returns canned responses."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.requests: list[dict] = []
        self._responses = list(responses)
        self.messages = self

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.requests.append(kwargs)
        return self._responses.pop(0)


def tool_use_response(block_id: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text="Checking."),
                                    SimpleNamespace(type="tool_use", id=block_id, name=name, input=args)])


def text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


MESSAGES = [{"role": "system", "content": "Be careful."}, {"role": "user", "content": "Ticket ID: T001\nhello"}]
TOOLS = build_default_registry(DATA).to_json_schemas()


class ClaudeModelTests(unittest.TestCase):
    def test_tool_use_block_maps_to_tool_call(self) -> None:
        client = FakeClient([tool_use_response("toolu_1", "lookup_account", {"account_id": "ACC-1002"})])
        response = ClaudeModel(client=client, model="claude-test").complete(MESSAGES, TOOLS)
        self.assertEqual(response, ToolCall("lookup_account", {"account_id": "ACC-1002"}, call_id="toolu_1"))
        request = client.requests[0]
        self.assertEqual((request["model"], request["system"], request["tools"]), ("claude-test", "Be careful.", TOOLS))
        self.assertEqual(request["messages"], MESSAGES[1:])
        self.assertEqual(request["tool_choice"], {"type": "auto", "disable_parallel_tool_use": True})

    def test_final_text_is_parsed(self) -> None:
        text = 'Done. {"team": "auth", "priority": "4", "summary": "Locked out", "reply_draft": "Sorry"}'
        response = ClaudeModel(client=FakeClient([text_response(text)])).complete(MESSAGES, TOOLS)
        self.assertEqual(response, Final("auth", 4, "Locked out", "Sorry"))

    def test_unparseable_text_becomes_a_final_that_guardrails_reject(self) -> None:
        response = parse_final("I am not sure what to do here.")
        self.assertEqual((response.team, response.priority), ("", 0))
        self.assertIsNotNone(Guardrails(build_default_registry(DATA)).check_final(response))
        self.assertEqual(parse_final('{"priority": 2}').team, "")

    def test_agent_round_trip_maps_tool_results_back(self) -> None:
        client = FakeClient([tool_use_response("toolu_7", "assign_team", {"team": "billing"}),
                             text_response(json.dumps(GOOD_FINAL.__dict__))])
        agent, _ = make_agent(ClaudeModel(client=client))
        result = agent.run(make_ticket())
        self.assertEqual((result.team, result.steps, result.tool_calls), ("billing", 2, 1))
        second = client.requests[1]["messages"]
        self.assertEqual(second[-2], {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_7", "name": "assign_team", "input": {"team": "billing"}}]})
        self.assertEqual(second[-1]["content"][0], {"type": "tool_result", "tool_use_id": "toolu_7",
                                                    "content": '{"team": "billing"}', "is_error": False})


class ReplayAndRecordingTests(unittest.TestCase):
    def test_replay_steps_by_assistant_turn_count(self) -> None:
        model = ReplayModel(FIXTURES)
        first = model.complete([{"role": "user", "content": "Ticket ID: T001\n"}], [])
        self.assertEqual(first, ToolCall("search_similar_tickets", {"query": "duplicate charge same invoice refund", "k": 3}, "replay-T001-0"))
        history = [{"role": "user", "content": "Ticket ID: T001\n"}] + [{"role": "assistant", "content": []}] * 5
        self.assertIsInstance(model.complete(history, []), Final)
        with self.assertRaises(ReplayError):
            model.complete(history + [{"role": "assistant", "content": []}], [])
        with self.assertRaises(ReplayError):
            model.complete([{"role": "user", "content": "Ticket ID: T999\n"}], [])

    def test_recording_model_writes_replayable_fixtures(self) -> None:
        steps = [ToolCall("assign_team", {"team": "billing"}), ToolCall("set_priority", {"level": 3, "reason": "dup"}), GOOD_FINAL]
        recorder = RecordingModel(ScriptedModel(steps))
        agent, _ = make_agent(recorder)
        original = agent.run(make_ticket())
        with tempfile.TemporaryDirectory() as tmp:
            paths = recorder.save(Path(tmp))
            self.assertEqual([p.name for p in paths], ["T900.json"])
            data = json.loads(paths[0].read_text())
            self.assertEqual(data["ticket_id"], "T900")
            self.assertEqual(data["steps"][0], {"type": "tool_call", "name": "assign_team", "args": {"team": "billing"}})
            self.assertEqual(data["steps"][2], {"type": "final", **GOOD_FINAL.__dict__})
            replay_agent, _ = make_agent(ReplayModel(Path(tmp)))
            replayed = replay_agent.run(make_ticket())
        self.assertEqual((replayed.team, replayed.priority, replayed.steps), (original.team, original.priority, original.steps))


if __name__ == "__main__":
    unittest.main()
