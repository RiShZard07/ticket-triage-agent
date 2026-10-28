# Learning guide

Twelve things you should be able to explain about this project in an interview,
in your own words and with the code open. The honest framing from the README
applies throughout: the replay numbers test the harness, not a model.

1. **The agent loop.** `Agent.run` in `triage_agent/agent.py` builds a
   conversation from the redacted ticket, asks the model for one response, and
   branches: a `Final` is validated and returned, a `ToolCall` is validated,
   executed and its result appended as a `tool_result` message. The loop
   continues until a valid `Final`, an `escalate` call, or the step budget.
   Be able to trace one ticket through `output/traces.jsonl` step by step.

2. **Why tools have schemas.** `ArgSpec` and `ToolSpec` in `tools.py` declare
   each argument's type, enum, range and whether it is required. The same spec
   renders the JSON Schema the model receives and drives the validation the
   guardrails run on what the model sends back. One source of truth, two uses.

3. **Why the guardrails sit outside the model.** A prompt can ask a model to
   behave; it cannot guarantee it. `guardrails.py` checks every call and every
   final answer in code, so a bad tool name, a priority of 7 or a team called
   "sales" never reaches a handler. Rejections are fed back as error tool
   results, which is how the model gets a chance to correct itself.

4. **PII redaction before the model call.** `redact_pii` replaces emails,
   phone numbers and card-like digit runs in the subject and body before the
   text is rendered into the prompt, and `redact_args` applies it again to tool
   arguments before they are logged. Know what the regexes catch, what they
   miss (names, addresses, free-text identifiers) and why this is still worth
   doing in a prototype.

5. **Replay versus live evaluation, and what each proves.** `ReplayModel`
   returns scripted responses keyed by ticket id and step index, so replay runs
   are deterministic, offline and fast. They prove the loop, guardrails,
   metrics and gate behave as designed. Only a live run with `ClaudeModel`
   says anything about triage quality, and a `record` run turns that live run
   into fixtures you can replay later.

6. **The regression gate.** `eval --gate X` exits with code 1 when team
   accuracy drops below X. In CI it runs over the replay fixtures, so it guards
   the harness: if someone breaks argument validation or the metric math, the
   number moves and the build fails. It is not a model quality gate until the
   fixtures are recordings.

7. **Telemetry fields and what they are for.** Each step event records the
   ticket id, step number, event kind, model latency, tool name, redacted
   arguments, result size and any error text. Per-ticket results and a run
   summary follow. Latency tells you where time goes, kinds and errors tell
   you how often the model needed a correction, and result size shows which
   tools dominate the context window.

8. **What accuracy on scripted fixtures means.** The replay run reports team
   accuracy of 0.9 because four of the forty scripts were written to miss (three
   wrong teams and one runaway episode). Change a script and the number changes.
   Be ready to say this unprompted; it is the most important honesty point in
   the project.

9. **How escalation works.** Two paths end in a human: the model calls
   `escalate(reason)`, or the loop reaches `max_steps`. In both cases the
   result keeps whatever team and priority the action tools already recorded.
   Escalated tickets stay in every denominator, so escalation is visible in
   accuracy as well as in `escalation_rate`.

10. **How to add a tool.** Write a handler that takes `TriageState` plus
    keyword arguments, register a `ToolSpec` with `ArgSpec`s in
    `build_default_registry`, and add a line to the system prompt if the model
    needs guidance on when to use it. Validation, schema rendering, call caps
    and telemetry come for free. Add a test in `tests/test_tools.py`.

11. **How to extend the golden set.** Append a ticket to
    `data/golden_tickets.json` with an id, subject, body, account id and
    expected labels. For replay, add `fixtures/replay/<id>.json` with the
    scripted steps; for live runs nothing else is needed. Keep the label
    distribution balanced and keep writing a few ambiguous tickets, because
    those are the ones that separate a good triage policy from a lookup table.

12. **Where this prototype would break in production.** Term-frequency search
    over forty entries, a single process, no retries on model errors, no
    reconciliation between the tools the model called and the Final it wrote,
    and PII redaction that only knows three patterns. Naming these limits is
    part of understanding the design.
