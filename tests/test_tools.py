"""Search determinism and ranking, account lookup, and tool schema rendering."""

import unittest

from helpers import DATA
from triage_agent.tools import KnowledgeBase, ToolError, TriageState, build_default_registry


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_default_registry(DATA)
        self.state = TriageState()

    def test_search_is_deterministic_and_respects_k(self) -> None:
        first = self.registry.execute("search_similar_tickets", {"query": "charged twice refund invoice", "k": 4}, self.state)
        second = self.registry.execute("search_similar_tickets", {"query": "charged twice refund invoice", "k": 4}, self.state)
        self.assertEqual(first, second)
        self.assertEqual(len(first["results"]), 4)
        self.assertEqual(len(self.registry.execute("search_similar_tickets", {"query": "refund"}, self.state)["results"]), 3)

    def test_search_ranking_sanity(self) -> None:
        for query, team in (("charged twice duplicate invoice", "billing"), ("SAML certificate rotation login", "auth"),
                            ("dashboard 502 errors", "outage"), ("dark mode theme", "feature_request"),
                            ("phishing fake login page share link", "abuse")):
            results = self.registry.execute("search_similar_tickets", {"query": query}, self.state)["results"]
            self.assertEqual(results[0]["team"], team, query)
            self.assertEqual([r["score"] for r in results], sorted((r["score"] for r in results), reverse=True))

    def test_search_ties_break_on_id(self) -> None:
        kb = KnowledgeBase([{"id": "B", "title": "refund", "body": "", "team": "billing", "resolution": ""},
                            {"id": "A", "title": "refund", "body": "", "team": "billing", "resolution": ""}])
        self.assertEqual([r["id"] for r in kb.search("refund", 2)], ["A", "B"])
        self.assertEqual(kb.search("zzz", 1)[0]["score"], 0.0)

    def test_lookup_account(self) -> None:
        account = self.registry.execute("lookup_account", {"account_id": "ACC-1005"}, self.state)
        self.assertEqual((account["plan"], account["open_incident"]), ("enterprise", True))
        with self.assertRaises(ToolError):
            self.registry.execute("lookup_account", {"account_id": "ACC-9999"}, self.state)

    def test_action_tools_write_state(self) -> None:
        self.registry.execute("assign_team", {"team": "auth"}, self.state)
        self.registry.execute("set_priority", {"level": 4, "reason": "locked out"}, self.state)
        self.registry.execute("draft_reply", {"text": "Sorry"}, self.state)
        self.registry.execute("escalate", {"reason": "threat"}, self.state)
        self.assertEqual((self.state.team, self.state.priority, self.state.reply_draft, self.state.escalation_reason),
                         ("auth", 4, "Sorry", "threat"))

    def test_json_schemas(self) -> None:
        schemas = {s["name"]: s for s in self.registry.to_json_schemas()}
        self.assertEqual(len(schemas), 6)
        team = schemas["assign_team"]["input_schema"]
        self.assertEqual(team["properties"]["team"]["enum"], ["billing", "auth", "outage", "feature_request", "abuse"])
        self.assertFalse(team["additionalProperties"])
        level = schemas["set_priority"]["input_schema"]["properties"]["level"]
        self.assertEqual((level["type"], level["minimum"], level["maximum"]), ("integer", 1, 5))
        self.assertEqual(schemas["search_similar_tickets"]["input_schema"]["required"], ["query"])


if __name__ == "__main__":
    unittest.main()
