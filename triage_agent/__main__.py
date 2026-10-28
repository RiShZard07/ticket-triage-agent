"""Command-line entry point: eval, run and record subcommands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .agent import Agent, Ticket
from .evaluate import compute_metrics, load_tickets, run_eval
from .models import DEFAULT_CLAUDE_MODEL, ClaudeModel, Model, RecordingModel, ReplayModel
from .telemetry import Telemetry
from .tools import build_default_registry

ROOT = Path(__file__).resolve().parents[1]


class CliError(Exception):
    """A usage problem reported to stderr with exit code 2."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triage_agent", description="LLM ticket triage agent with guardrails and evals.")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--data", type=Path, default=ROOT / "data", help="directory with the JSON data files")
    shared.add_argument("--fixtures", type=Path, default=ROOT / "fixtures" / "replay", help="replay fixture directory")
    shared.add_argument("--max-steps", type=int, default=8, help="model calls allowed per ticket")
    shared.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL, help="model id for --model claude")
    commands = parser.add_subparsers(dest="command", required=True)

    eval_cmd = commands.add_parser("eval", parents=[shared], help="score the agent on the golden tickets")
    eval_cmd.add_argument("--model", choices=("replay", "claude"), default="replay")
    eval_cmd.add_argument("--gate", type=float, help="exit 1 if team accuracy is below this fraction")
    eval_cmd.add_argument("--output", type=Path, default=ROOT / "output")

    run_cmd = commands.add_parser("run", parents=[shared], help="triage one ticket and print the trace")
    run_cmd.add_argument("--model", choices=("replay", "claude"), default="replay")
    run_cmd.add_argument("--ticket-id", help="id of a golden ticket")
    run_cmd.add_argument("--text", help="ad-hoc ticket text (claude model only)")
    run_cmd.add_argument("--account-id", default="unknown", help="account id for --text tickets")

    record_cmd = commands.add_parser("record", parents=[shared], help="run the live model and save replay fixtures")
    record_cmd.add_argument("--output", type=Path, default=ROOT / "fixtures" / "replay")
    return parser


def build_claude(args: argparse.Namespace) -> ClaudeModel:
    """Construct the live model, failing clearly when the key or SDK is missing."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise CliError("ANTHROPIC_API_KEY is not set; the claude model needs it (and `pip install anthropic`)")
    try:
        return ClaudeModel(model=args.claude_model)
    except RuntimeError as exc:
        raise CliError(str(exc)) from exc


def build_model(args: argparse.Namespace) -> Model:
    return ReplayModel(args.fixtures) if args.model == "replay" else build_claude(args)


def select_ticket(args: argparse.Namespace) -> Ticket:
    """Pick a golden ticket by id or wrap ad-hoc text."""
    if args.text:
        if args.model == "replay":
            raise CliError("--text needs --model claude; the replay model only knows golden ticket ids")
        subject, _, body = args.text.partition("\n")
        return Ticket(id="adhoc", subject=subject, body=body or subject, account_id=args.account_id)
    if not args.ticket_id:
        raise CliError("run needs --ticket-id or --text")
    matches = [t for t in load_tickets(args.data / "golden_tickets.json") if t.id == args.ticket_id]
    if not matches:
        raise CliError(f"no golden ticket with id {args.ticket_id}")
    return matches[0]


def cmd_eval(args: argparse.Namespace) -> int:
    metrics = run_eval(build_model(args), args.data, args.output, args.max_steps)
    print(json.dumps(asdict(metrics), indent=2))
    print(f"wrote metrics.json, report.md and traces.jsonl to {args.output}")
    if args.gate is not None and metrics.team_accuracy < args.gate:
        print(f"GATE FAILED: team accuracy {metrics.team_accuracy} is below {args.gate}", file=sys.stderr)
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    ticket = select_ticket(args)
    telemetry = Telemetry()
    Agent(build_model(args), build_default_registry(args.data), telemetry, args.max_steps).run(ticket)
    for event in telemetry.events:
        print(json.dumps(event))
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    recorder = RecordingModel(build_claude(args))
    tickets = load_tickets(args.data / "golden_tickets.json")
    agent = Agent(recorder, build_default_registry(args.data), Telemetry(), args.max_steps)
    results = [agent.run(ticket) for ticket in tickets]
    paths = recorder.save(args.output)
    print(f"recorded {len(paths)} fixtures to {args.output}")
    print(json.dumps(asdict(compute_metrics(tickets, results)), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"eval": cmd_eval, "run": cmd_run, "record": cmd_record}
    try:
        return handlers[args.command](args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
