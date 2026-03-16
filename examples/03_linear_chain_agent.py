"""
03_linear_chain_agent.py – Linear chain (A→B→C) with AgentWithConstraints.

Enforces the ordering constraint fetch → clean → summarize using
HARD_STOP severity.  If the agent tries to call a tool out of order
the run is aborted immediately.

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


# ── Dummy tools ──────────────────────────────────────────────────────────────

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


# ── Constraints ──────────────────────────────────────────────────────────────

CONSTRAINTS = [
    Constraint(
        name="fetch_before_clean",
        formula=Before("fetch_raw_readings", "clean_readings"),
        weight=1.0,
        description="Raw readings must be fetched before cleaning",
    ),
    Constraint(
        name="clean_before_summarize",
        formula=Before("clean_readings", "summarize_readings"),
        weight=1.0,
        description="Readings must be cleaned before summarizing",
    ),
]

SEVERITIES = {
    "fetch_before_clean": ConstraintSeverity.HARD_STOP,
    "clean_before_summarize": ConstraintSeverity.HARD_STOP,
}


def main():
    tools = [FetchRawReadingsTool(), CleanReadingsTool(), SummarizeReadingsTool()]

    agent = AgentWithConstraints(
        mcp_server_url=None,
        additional_tools=tools,
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

    print(f"\nAnswer: {result['answer']}")
    print(f"Error:  {result['error']}")
    print(f"Run status: {result['metrics'].get('run_status')}")
    print(f"Stopped by: {result['metrics'].get('stopped_by_constraint')}")
    print(f"Constraint checks: {result['metrics'].get('constraint_checks')}")

    violations = result["metrics"].get("constraint_violations", [])
    if violations:
        print(f"Violations ({len(violations)}):")
        for v in violations:
            print(f"  - {v['constraint_name']} at step {v['step_number']}: {v['detail']}")
    else:
        print("No constraint violations.")

    tool_seq = [tc["tool_name"] for tc in result["metrics"].get("tool_calls", [])]
    print(f"Tool sequence: {tool_seq}")


if __name__ == "__main__":
    main()
