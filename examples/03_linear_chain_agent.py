"""
03_linear_chain_agent.py – Linear chain (A→B→C) with AgentWithConstraints.

Enforces the ordering constraint fetch → clean → summarize.

* ``fetch_before_clean`` is HARD_STOP: the run aborts immediately if the agent
  tries to call clean_readings before fetch_raw_readings.
* ``clean_before_summarize`` is TOLERATE: a violation is logged but the run
  continues, demonstrating how to mix enforcement modes.

This example also shows how to add MCP connectivity — replace the ``tools``
list with ``mcp_servers`` or combine both::

    from agentltl.integrations.smolagents import AgentWithConstraints

    agent = AgentWithConstraints(
        tools=local_tools,
        mcp_servers={
            "my_server": {
                "url": "http://localhost:4000/mcp",
                "transport": "streamable-http",
            },
        },
        constraints=CONSTRAINTS,
        constraint_severities=SEVERITIES,
    )

Requirements:
    pip install agentltl[smolagents]
    export HF_TOKEN=...

Run:
    python examples/03_linear_chain_agent.py
"""

import os
from smolagents import Tool
from agentltl import Constraint, Before
from agentltl.integrations.smolagents import AgentWithConstraints, ConstraintSeverity


# ── Tools ─────────────────────────────────────────────────────────────────────

class FetchRawReadingsTool(Tool):
    name = "fetch_raw_readings"
    description = "Fetch raw sensor readings from the database. Call this first."
    inputs = {}
    output_type = "string"

    def forward(self):
        return "Raw readings: [23.1, 24.5, 22.8, 25.0, 23.7]"


class CleanReadingsTool(Tool):
    name = "clean_readings"
    description = "Clean and validate the raw sensor readings. Call after fetch_raw_readings."
    inputs = {}
    output_type = "string"

    def forward(self):
        return "Cleaned readings: [23.1, 24.5, 22.8, 25.0, 23.7] (all valid)"


class SummarizeReadingsTool(Tool):
    name = "summarize_readings"
    description = "Summarize the cleaned sensor readings. Call after clean_readings."
    inputs = {}
    output_type = "string"

    def forward(self):
        return "Summary: mean=23.82, min=22.8, max=25.0, std=0.82"


# ── Constraints ───────────────────────────────────────────────────────────────

CONSTRAINTS = [
    Constraint(
        name="fetch_before_clean",
        formula=Before("fetch_raw_readings", "clean_readings"),
        weight=2.0,
        description="Raw readings must be fetched before cleaning",
    ),
    Constraint(
        name="clean_before_summarize",
        formula=Before("clean_readings", "summarize_readings"),
        weight=1.0,
        description="Readings must be cleaned before summarizing",
    ),
]

# HARD_STOP aborts immediately; TOLERATE logs and continues.
# Constraints not listed here fall back to default_severity (HARD_STOP).
SEVERITIES = {
    "fetch_before_clean":    ConstraintSeverity.HARD_STOP,
    "clean_before_summarize": ConstraintSeverity.TOLERATE,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    agent = AgentWithConstraints(
        tools=[FetchRawReadingsTool(), CleanReadingsTool(), SummarizeReadingsTool()],
        constraints=CONSTRAINTS,
        constraint_severities=SEVERITIES,
        model=os.environ.get("MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
        max_steps=8,
    )

    task = (
        "You have access to three tools: fetch_raw_readings, clean_readings, "
        "and summarize_readings. You MUST call them in order: fetch first, "
        "then clean, then summarize. Return a brief summary of the sensor readings."
    )

    print("Running constrained agent (linear chain A→B→C)...")
    result = agent.run(task)

    print(f"\nAnswer:     {result['answer']}")
    print(f"Error:      {result['error']}")
    print(f"Run status: {result['metrics'].get('run_status')}")
    print(f"Stopped by: {result['metrics'].get('stopped_by_constraint')}")
    print(f"Constraint checks: {result['metrics'].get('constraint_checks')}")

    violations = result["metrics"].get("constraint_violations", [])
    if violations:
        print(f"\nViolations ({len(violations)}):")
        for v in violations:
            print(
                f"  [{v['severity']}] {v['constraint_name']} "
                f"at step {v['step_number']}: {v['detail']}"
            )
    else:
        print("No constraint violations.")

    tool_seq = [tc["tool_name"] for tc in result["metrics"].get("tool_calls", [])]
    print(f"\nTool sequence: {tool_seq}")
    print(f"Steps: {result['metrics']['num_steps']}")


if __name__ == "__main__":
    main()
