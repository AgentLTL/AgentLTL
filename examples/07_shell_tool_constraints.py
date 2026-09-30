"""
Example 07 – Constraints over a shell tool's command lines
==========================================================
No LLM required: a scripted client stands in for the model so the enforcement
is reproducible. Swap ``model_instance`` for a real OpenAI-compatible client
(or drop it and set ``model`` / ``base_url``) to run it against a model.

Requires:  pip install "agentltl[cli]"

The agent has a single ``bash`` tool. With ``shell_tools={"bash": "command"}``
every command line is translated into structured calls (``git_commit``,
``git_push``, ...) and the constraints are written over those.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, List

from agentltl import (
    AgentWithConstraints,
    Before,
    CalledWith,
    Constraint,
    ConstraintSeverity,
    Globally,
    Not,
    verify_trace,
)


class BashTool:
    """Stand-in shell tool: echoes the command instead of running it."""

    name = "bash"
    description = "Run a shell command line."
    inputs = {"command": {"type": "string", "description": "The command line to run."}}

    def __call__(self, command: str) -> str:
        return f"(ran) {command}"


class ScriptedClient:
    """Minimal OpenAI-compatible client that replays a fixed list of command lines."""

    def __init__(self, commands: List[str]) -> None:
        self._commands = list(commands)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_: Any) -> Any:
        if not self._commands:
            message = SimpleNamespace(content="Done.", tool_calls=None)
        else:
            call = SimpleNamespace(
                id=f"call_{len(self._commands)}",
                function=SimpleNamespace(
                    name="bash", arguments=json.dumps({"command": self._commands.pop(0)})),
            )
            message = SimpleNamespace(content=None, tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


CONSTRAINTS = [
    Constraint(
        "commit_before_push", Before("git_commit", "git_push"),
        repair="Commit your changes before pushing.",
    ),
    Constraint(
        "no_force_push", Globally(Not(CalledWith("git_push", {"force": True}))),
        repair="Push without --force.",
    ),
]

COMMANDS = [
    "git push",                                   # blocked: nothing committed yet
    "git push && git commit -m wip",              # blocked as a whole: wrong order in the chain
    "for f in *.py; do git add $f; done",         # blocked: cannot be analysed
    "git add . && git commit -m wip && git push", # allowed
    "sudo bash -c 'git push --force'",            # blocked: the wrapper does not hide the call
]

agent = AgentWithConstraints(
    tools=[BashTool()],
    constraints=CONSTRAINTS,
    default_severity=ConstraintSeverity.SOFT_BLOCK,
    max_soft_attempts=10,
    model="scripted",
    model_instance=ScriptedClient(COMMANDS),
    backend="native",
    shell_tools={"bash": "command"},
)

result = agent.run("Commit and push the work.")
metrics = result["metrics"]

print("executed calls :", [tc["tool_name"] for tc in metrics["tool_calls"]])
print("blocked turns  :", metrics["blocked_steps"])
print("run status     :", metrics["run_status"])

report = verify_trace(metrics, CONSTRAINTS)
print("post-hoc score :", report["compliance_score"])
