# ticket-triage-agent

An LLM agent that triages customer support tickets by calling tools, wrapped in
guardrails, an evaluation harness and telemetry. Standard library only; the
`anthropic` SDK is optional and only imported when you ask for the live model.

## Status, stated plainly

This is an AI-assisted educational prototype, built with Claude Code, over
synthetic tickets. There are no real customers, no production deployment and no
real ticketing system behind the tools.

Two things matter when reading any number in this repository:

- Live model runs need `ANTHROPIC_API_KEY` and `pip install anthropic`. Nothing
  in the committed test suite talks to a model.
- The replay fixtures in `fixtures/replay/` are hand-authored scripts, not
  recordings of a live model. The replay evaluation therefore measures whether
  the harness (loop, guardrails, metrics, gate) behaves as designed. It says
  nothing about how well any model triages tickets. The scripts deliberately
  include wrong answers, invalid tool calls and a runaway episode so that the
  harness has something to catch.

## How to run

```bash
# tests (offline, about half a second)
python3 -m unittest discover -s tests -v

# replay evaluation: writes output/metrics.json, output/report.md, output/traces.jsonl
python3 -m triage_agent eval --model replay

# same, failing with exit code 1 if team accuracy is below 0.9 (the CI gate uses 0.85)
python3 -m triage_agent eval --model replay --gate 0.9

# one ticket, printing the JSON-lines trace
python3 -m triage_agent run --model replay --ticket-id T007

# live model (needs the SDK and a key); --text works only with the claude model
pip install anthropic
export ANTHROPIC_API_KEY=...
python3 -m triage_agent eval --model claude
python3 -m triage_agent run --model claude --text "Locked out after a password reset"

# record the live model's responses over the golden set as replay fixtures
python3 -m triage_agent record --output fixtures/replay
```

`record` and `--model claude` stop with a clear error when the key is missing.
Running `record` over `fixtures/replay` replaces the hand-authored scripts with
real recordings; after that, replay numbers describe that recorded model run.

## Architecture

```
ticket --> redact_pii --> messages --> model.complete(messages, tools)
                             ^                    |
                             |           ToolCall or Final
                             |                    |
                             |        +-----------+-----------+
                             |        | Final                 | ToolCall
                             |        v                       v
                             |   check_final          check_tool_call
                             |   (team, priority)     (name, args, call cap)
                             |        | ok                    | ok
                             |        v                       v
                             |   TriageResult          registry.execute
                             |                                |
                             +------ tool_result <------------+
                                     (output, rejection or tool error)

Every step is written to telemetry. Reaching max_steps (default 8) without a
Final escalates to a human with the reason "step budget exhausted".
```

Modules:

| Module | Role |
|---|---|
| `triage_agent/agent.py` | The loop above, the system prompt, `Ticket` and `TriageResult` |
| `triage_agent/tools.py` | `ToolRegistry`, `ArgSpec`/`ToolSpec`, the six tools, term-frequency search |
| `triage_agent/guardrails.py` | Argument validation, call caps, final validation, PII redaction |
| `triage_agent/models.py` | `Model` protocol, `ReplayModel`, `ClaudeModel`, `RecordingModel` |
| `triage_agent/evaluate.py` | Golden set loader, metrics, report, regression gate input |
| `triage_agent/telemetry.py` | JSON-lines step events, per-ticket results and a run summary |
| `triage_agent/__main__.py` | `eval`, `run` and `record` commands |

Tools: `search_similar_tickets(query, k=3)` over `data/knowledge_base.json`
(40 resolved tickets, cosine similarity on term-frequency vectors),
`lookup_account(account_id)` over `data/accounts.json` (15 synthetic accounts),
`set_priority(level 1-5, reason)`, `assign_team(team)`, `draft_reply(text)` and
`escalate(reason)`. Priority 1 is low and 5 is critical.

The golden set in `data/golden_tickets.json` has 40 synthetic tickets, eight per
team, with a few deliberately ambiguous ones (an SSO failure that looks like an
outage, a scraper that looks like load, a suspicious login that could be abuse).

## Guardrails

- Tool name must exist in the registry.
- Arguments are checked against the spec: types (a bool is not an int), enums,
  integer ranges, required fields, no unknown fields, no empty strings.
- PII redaction before ticket text reaches the model: emails, phone numbers and
  13 to 19 digit card-like numbers become `[EMAIL]`, `[PHONE]`, `[CARD]`. The
  same redaction is applied to tool arguments before they are logged.
- Step budget: at most `max_steps` model calls per ticket, then escalation.
- Per-tool call cap per ticket: 3 by default, 1 for `escalate`.
- A `Final` must carry a known team and an integer priority from 1 to 5.
- Every rejection and every tool error is returned to the model as an error
  tool result so it can retry, and is counted in the metrics.

## Metrics

All rates are fractions of all tickets in the run, including escalated ones. An
escalated ticket keeps whatever team and priority the action tools recorded
before the escalation; if nothing was recorded it counts as a miss.

| Metric | Definition |
|---|---|
| `team_accuracy` | predicted team equals `expected_team` |
| `priority_exact_accuracy` | predicted priority equals `expected_priority` |
| `priority_within_1_accuracy` | predicted priority within 1 of expected |
| `escalation_rate` | tickets that ended in an escalation (model-initiated or step budget) |
| `mean_steps` | mean model calls per ticket, including rejected ones |
| `mean_tool_calls` | mean executed tool calls per ticket (rejected calls are not executed) |
| `guardrail_rejections` | total rejected tool calls plus rejected finals |
| `tool_errors` | total executed calls whose handler raised `ToolError` |
| `model_latency_p50_ms`, `model_latency_p95_ms` | nearest-rank percentiles over every model call in the run |

`--gate X` makes `eval` exit with code 1 when `team_accuracy` is below X. CI
runs the replay evaluation with `--gate 0.85`, so a change that breaks the loop,
the guardrails or the metrics fails the build.

## Limitations

- Replay fixtures are scripted by hand. Replay accuracy is a property of the
  scripts, not of a model.
- Synthetic tickets, accounts and knowledge base. No real customers or data.
- No production deployment, no queue, no retries, single process.
- Retrieval is term-frequency cosine similarity over 40 entries. No embeddings.
- One tool call per model turn; parallel tool use is disabled on the live model.
- The message format follows the Messages API shape so the Claude adapter can
  pass it through; another provider would need a translation layer.
- The live adapter has only been exercised against a fake client in the tests.

## Reproduce and explain before using on a resume

Run the tests and the replay evaluation yourself and read `output/report.md`
and `output/traces.jsonl`. Then make sure you can explain, without notes, what
the replay numbers do and do not show, why the guardrails sit outside the model,
and how you would add a tool. `LEARNING_GUIDE.md` lists the questions. If you
describe this project to someone, say that the agent runs are scripted unless
you have recorded a live model with `record` and can show that run.
