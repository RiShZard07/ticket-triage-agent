# Verified results

What was actually run, on the day the repository was built, and what it
produced. Everything below is offline and uses the hand-authored replay
fixtures. No model was called.

## Environment

- Linux, CPython 3.13.17 (`python3`). CI targets Python 3.12; the code uses no
  syntax or library newer than Python 3.10.
- Standard library only. The `anthropic` package was not installed.

## Commands

```bash
cd ticket-triage-agent
python3 -m unittest discover -s tests -v
python3 -m triage_agent eval --model replay --gate 0.9
```

## Test suite

`Ran 43 tests in 0.460s` followed by `OK`. The suite covers argument
validation, PII redaction, call caps, final validation, search determinism and
ranking, account lookup, the agent happy path on a replay fixture, recovery
after a rejected call, tool errors surfaced to the model, escalation by tool
and by step budget, the Claude adapter against a fake client, recording and
replaying fixtures, hand-computed metrics, the gate exit code, and the CLI
through a subprocess.

## Replay evaluation (scripted fixtures, not a model)

Exit code 0 with `--gate 0.9`. Output of `eval`:

| Metric | Value |
|---|---|
| ticket_count | 40 |
| team_accuracy | 0.9 |
| priority_exact_accuracy | 0.925 |
| priority_within_1_accuracy | 0.95 |
| escalation_rate | 0.1 |
| mean_steps | 6.05 |
| mean_tool_calls | 5.075 |
| guardrail_rejections | 3 |
| tool_errors | 1 |
| model_latency_p50_ms | 0.004 |
| model_latency_p95_ms | 0.043 |

Latency numbers measure the replay model's dictionary lookups and vary by a few
hundredths of a millisecond between runs. Every other number is deterministic.

These values are exactly what the scripts were written to produce:

- Four team misses: T011, T015 and T037 were scripted with the wrong team, and
  T038 exhausts the step budget without recording a team. 36/40 = 0.9.
- Three exact priority misses: T004 (scripted P1, expected P3), T024 (scripted
  P4, expected P5) and T038. 37/40 = 0.925. Only T004 and T038 are outside the
  within-1 band. 38/40 = 0.95.
- Four escalations: T035, T036 and T039 call `escalate` after assigning a team
  and priority; T038 hits the step budget. 4/40 = 0.1.
- Three guardrail rejections: `set_priority(level=7)` in T005, `assign_team("features")`
  in T028, and a fourth `search_similar_tickets` call in T038 that exceeds the
  per-tool cap. One tool error: `lookup_account("ACC-104")` in T020.

A live run with `--model claude` will produce different numbers, and those are
the only numbers that say anything about triage quality.
