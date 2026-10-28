"""CLI smoke tests through a subprocess, plus usage errors in process."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

from helpers import ROOT
from triage_agent.__main__ import main


def run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "triage_agent", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


class CliTests(unittest.TestCase):
    def test_eval_replay_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = run_cli("eval", "--model", "replay", "--output", tmp)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('"team_accuracy"', completed.stdout)
            self.assertTrue(os.path.exists(os.path.join(tmp, "report.md")))

    def test_run_single_ticket(self) -> None:
        completed = run_cli("run", "--model", "replay", "--ticket-id", "T007")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        events = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(events[0]["tool_name"], "search_similar_tickets")
        self.assertEqual((events[-1]["event"], events[-1]["team"], events[-1]["priority"]), ("ticket_result", "billing", 2))

    def test_claude_model_without_api_key_fails_clearly(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        for args in (("run", "--model", "claude", "--text", "Cannot log in"), ("record",)):
            completed = run_cli(*args, env=env)
            self.assertEqual(completed.returncode, 2, args)
            self.assertIn("ANTHROPIC_API_KEY", completed.stderr)

    def test_usage_errors(self) -> None:
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["run", "--model", "replay", "--text", "hello"]), 2)
            self.assertEqual(main(["run", "--model", "replay", "--ticket-id", "T999"]), 2)
            self.assertEqual(main(["run", "--model", "replay"]), 2)
        self.assertEqual(err.getvalue().count("error:"), 3)


if __name__ == "__main__":
    unittest.main()
