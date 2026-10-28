"""Metric math against a hand-computed fixture, output files, and the regression gate."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from helpers import DATA, FIXTURES, make_ticket
from triage_agent.__main__ import main
from triage_agent.agent import TriageResult
from triage_agent.evaluate import compute_metrics, percentile, run_eval
from triage_agent.models import ReplayModel


def result(ticket_id: str, team: str | None, priority: int | None, **extra: object) -> TriageResult:
    fields = {"ticket_id": ticket_id, "team": team, "priority": priority, "summary": "", "reply_draft": "",
              "escalated": False, "escalation_reason": None, "steps": 6, "tool_calls": 5, "rejections": 0,
              "tool_errors": 0, "model_latencies_ms": []}
    return TriageResult(**{**fields, **extra})


class MetricsTests(unittest.TestCase):
    def test_hand_computed_metrics(self) -> None:
        tickets = [make_ticket(id="A", expected_team="billing", expected_priority=3),
                   make_ticket(id="B", expected_team="auth", expected_priority=4),
                   make_ticket(id="C", expected_team="outage", expected_priority=5),
                   make_ticket(id="D", expected_team="abuse", expected_priority=2)]
        results = [result("A", "billing", 3, model_latencies_ms=[1, 2, 3]),
                   result("B", "outage", 4, tool_errors=1, model_latencies_ms=[4, 5]),
                   result("C", "outage", 4, steps=7, tool_calls=6, rejections=1, model_latencies_ms=[6, 7, 8]),
                   result("D", None, None, escalated=True, escalation_reason="step budget exhausted",
                          steps=8, tool_calls=8, model_latencies_ms=[9, 10])]
        m = compute_metrics(tickets, results)
        self.assertEqual((m.ticket_count, m.team_accuracy, m.priority_exact_accuracy, m.priority_within_1_accuracy),
                         (4, 0.5, 0.5, 0.75))
        self.assertEqual((m.escalation_rate, m.mean_steps, m.mean_tool_calls), (0.25, 6.75, 6.0))
        self.assertEqual((m.guardrail_rejections, m.tool_errors), (1, 1))
        self.assertEqual((m.model_latency_p50_ms, m.model_latency_p95_ms), (5, 10))

    def test_percentile_is_nearest_rank(self) -> None:
        self.assertEqual(percentile([], 50), 0.0)
        self.assertEqual((percentile([30, 10, 20], 50), percentile([30, 10, 20], 95)), (20, 30))
        self.assertEqual(percentile([7], 95), 7)

    def test_mismatched_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics([], [])


class RunEvalTests(unittest.TestCase):
    def test_run_eval_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            metrics = run_eval(ReplayModel(FIXTURES), DATA, out)
            written = json.loads((out / "metrics.json").read_text())
            self.assertEqual(written["model"], "replay")
            self.assertEqual(written["team_accuracy"], metrics.team_accuracy)
            report = (out / "report.md").read_text()
            self.assertIn("| T038 | abuse / P3 | none / none | step budget exhausted | 8 | 1 |", report)
            lines = [json.loads(line) for line in (out / "traces.jsonl").read_text().splitlines()]
            self.assertEqual(lines[-1]["event"], "run_summary")
            self.assertEqual(sum(line["event"] == "ticket_result" for line in lines), 40)
        self.assertEqual((metrics.ticket_count, metrics.guardrail_rejections, metrics.tool_errors), (40, 3, 1))
        self.assertGreaterEqual(metrics.team_accuracy, 0.85)
        self.assertLess(metrics.team_accuracy, 1.0)

    def test_gate_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            base = ["eval", "--model", "replay", "--output", tmp]
            self.assertEqual(main(base + ["--gate", "0.5"]), 0)
            self.assertEqual(main(base + ["--gate", "0.99"]), 1)
        self.assertIn("GATE FAILED", err.getvalue())


if __name__ == "__main__":
    unittest.main()
