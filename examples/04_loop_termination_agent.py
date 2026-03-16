"""
04_loop_termination_agent.py – While-loop compliance with CalledNTimes + Predicate.

Verifies that a polling tool is called the correct number of times before
the agent exits the loop and that the last poll returned READY before
process_result is called.

The agent is built with ``AgentWithAdditionalTools`` (no runtime enforcement)
so the compliance is measured post-hoc via ``verify_trace()``.  This pattern
is appropriate when you want to audit behaviour without interrupting the run.

To enable runtime enforcement instead, swap to ``AgentWithConstraints``::

    from agentltl.integrations.smolagents import AgentWithConstraints, ConstraintSeverity

    agent = AgentWithConstraints(
        tools=[PollStatusTool(), ProcessResultTool()],
        constraints=CONSTRAINTS,
        constraint_severities={
            "poll_at_least_3":    ConstraintSeverity.TOLERATE,
            "process_after_ready": ConstraintSeverity.HARD_STOP,
        },
        model=os.environ.get("MODEL"),
        max_steps=12,
    )

Adding MCP servers is the same for both classes::

    agent = AgentWithAdditionalTools(
        tools=local_tools,
        mcp_servers={
            "status_server": {
                "url": "http://localhost:4000/mcp",
                "transport": "streamable-http",
            },
        },
        model=os.environ.get("MODEL"),
    )

Requirements:
    pip install agentltl[smolagents]
    export HF_TOKEN=...

Run:
    python examples/04_loop_termination_agent.py
"""

import json
import os
from smolagents import Tool
from agentltl import Constraint, CalledNTimes, Predicate, Trace, verify_trace
from agentltl.integrations.smolagents import AgentWithAdditionalTools


# ── Stateful polling tool ──────────────────────────────────────────────────────

class PollStatusTool(Tool):
    name = "poll_status"
    description = (
        "Poll the system status. Returns JSON with a 'status' field. "
        "Status is 'PENDING' until the 3rd call, then 'READY'."
    )
    inputs = {}
    output_type = "string"

    def __init__(self):
        super().__init__()
        self._call_count = 0

    def forward(self):
        self._call_count += 1
        if self._call_count >= 3:
            return json.dumps({"status": "READY", "call": self._call_count})
        return json.dumps({"status": "PENDING", "call": self._call_count})


class ProcessResultTool(Tool):
    name = "process_result"
    description = "Process the result once the system is READY."
    inputs = {}
    output_type = "string"

    def forward(self):
        return "Processing complete."


# ── Predicate: last poll must show READY before process_result ─────────────────

def last_poll_was_ready(trace: Trace, position: int, metrics=None) -> dict:
    """True iff the most recent poll_status call before *position* returned READY."""
    poll_calls = [
        c for c in trace.calls
        if c.name == "poll_status" and c.position < position
    ]
    if not poll_calls:
        return {"passed": False, "note": "No poll_status call found before this position"}
    last_poll = poll_calls[-1]
    raw_result = last_poll.result or last_poll.raw.get("tool_result", "")
    try:
        data = json.loads(str(raw_result))
        status = data.get("status", "UNKNOWN")
    except (json.JSONDecodeError, ValueError):
        status = str(raw_result)
    if status == "READY":
        return {"passed": True, "note": "Last poll returned READY"}
    return {"passed": False, "note": f"Last poll returned {status!r}, not READY"}


# ── Constraints ───────────────────────────────────────────────────────────────

CONSTRAINTS = [
    Constraint(
        name="poll_at_least_3",
        formula=CalledNTimes("poll_status", 3, ">="),
        weight=2.0,
        description="Must poll at least 3 times before processing",
    ),
    Constraint(
        name="process_after_ready",
        formula=Predicate(last_poll_was_ready, "last_poll_was_ready"),
        weight=3.0,
        description="process_result may only run after poll_status returns READY",
    ),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    agent = AgentWithAdditionalTools(
        tools=[PollStatusTool(), ProcessResultTool()],
        model=os.environ.get("MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
        max_steps=12,
    )

    task = (
        "Use poll_status to check the system status. Keep polling until the "
        "status is 'READY', then call process_result exactly once. "
        "Return the final status."
    )

    print("Running loop-termination agent (poll until READY, then process)...")
    result = agent.run(task)

    print(f"\nAnswer: {result['answer']}")
    print(f"Error:  {result['error']}")

    tool_seq = [tc["tool_name"] for tc in result["metrics"].get("tool_calls", [])]
    print(f"Tool sequence: {tool_seq}")
    print(f"Steps: {result['metrics']['num_steps']}")

    print("\n--- Post-hoc constraint verification ---")
    verification = verify_trace(result["metrics"], CONSTRAINTS)
    print(
        f"Compliance: {verification['compliance_label']} "
        f"({verification['compliance_score']:.2f})"
    )
    for c in verification["constraints"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']} (w={c['weight']}): {c['detail']}")


if __name__ == "__main__":
    main()
