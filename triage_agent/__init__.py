"""Tool-calling LLM agent that triages synthetic support tickets.

The package has five parts: a tool registry with explicit argument specs
(tools.py), guardrails that sit between the model and the tools
(guardrails.py), model adapters including a replay model for offline runs
(models.py), the agent loop (agent.py), and an evaluation harness with
JSON-lines telemetry (evaluate.py, telemetry.py).
"""

__version__ = "0.1.0"
