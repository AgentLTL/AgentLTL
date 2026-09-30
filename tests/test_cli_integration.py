"""Native backend + cli-to-tools: constraints over a shell tool's command lines."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("cli_to_tools")

from agentltl import (  # noqa: E402
    AgentWithConstraints,
    Before,
    Constraint,
    ConstraintSeverity,
    MultiTurnAgent,
    verify_trace,
)
from agentltl._enforcement_engine import ConstraintEnforcer  # noqa: E402
from agentltl.integrations.cli import CliConstraintEnforcer  # noqa: E402
from agentltl.integrations.native.backend import NativeOpenAIAgent  # noqa: E402

_COMMIT_FIRST = Constraint("commit_first", Before("git_commit", "git_push"))


class _Bash:
    name = "bash"
    description = "Run a shell command line."
    inputs = {"command": {"type": "string", "description": "command line"}}

    def __init__(self) -> None:
        self.ran: list[str] = []

    def __call__(self, command: str) -> str:
        self.ran.append(command)
        return "ok"


class _Client:
    """Replays command lines as ``bash`` tool calls, then answers."""

    def __init__(self, commands: list[str]) -> None:
        self._commands = list(commands)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_: Any) -> Any:
        if not self._commands:
            message = SimpleNamespace(content="done", tool_calls=None)
        else:
            call = SimpleNamespace(
                id=f"c{len(self._commands)}",
                function=SimpleNamespace(
                    name="bash", arguments=json.dumps({"command": self._commands.pop(0)})),
            )
            message = SimpleNamespace(content=None, tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _agent(bash: _Bash, commands: list[str], **kwargs: Any) -> AgentWithConstraints:
    return AgentWithConstraints(
        tools=[bash], constraints=[_COMMIT_FIRST],
        default_severity=ConstraintSeverity.SOFT_BLOCK, max_soft_attempts=10,
        model="scripted", model_instance=_Client(commands), backend="native", **kwargs,
    )


def test_shell_tools_enforce_chain_and_expand_trace() -> None:
    bash = _Bash()
    commands = [
        "git push && git commit -m x",
        "while true; do git push; done",
        "git add . && git commit -m x && git push",
    ]
    result = _agent(bash, commands, shell_tools={"bash": "command"}).run("go")

    assert result["error"] is None
    assert bash.ran == [commands[2]]  # rejected command lines never reached the tool
    calls = result["metrics"]["tool_calls"]
    assert [tc["tool_name"] for tc in calls] == ["git_add", "git_commit", "git_push"]
    assert calls[1]["arguments"]["message"] == "x"
    assert calls[-1]["tool_result"] == "ok" and calls[0]["tool_result"] is None
    assert calls[0]["cli"]["command"] == commands[2]
    # one generation, but the chain is sequential: steps are ordered inside it
    assert {tc["model_step"] for tc in calls} == {3}
    assert 3 == calls[0]["step"] < calls[1]["step"] < calls[2]["step"] < 4
    assert verify_trace(result["metrics"], [_COMMIT_FIRST])["compliance_score"] == 1.0


def test_default_is_unchanged() -> None:
    bash = _Bash()
    result = _agent(bash, ["git push && git commit -m x"]).run("go")
    assert bash.ran == ["git push && git commit -m x"]  # opaque `bash` call, as before
    assert [tc["tool_name"] for tc in result["metrics"]["tool_calls"]] == ["bash"]
    agent = NativeOpenAIAgent(model="m", model_instance=_Client([]))
    assert type(agent._enforcer) is ConstraintEnforcer


def test_facades_pass_shell_tools_through() -> None:
    multi = MultiTurnAgent(
        tools=[_Bash()], model="m", model_instance=_Client([]), shell_tools={"bash": "command"})
    assert isinstance(multi._impl._enforcer, CliConstraintEnforcer)
    with pytest.raises(ValueError):
        AgentWithConstraints(tools=[], backend="langchain", shell_tools={"bash": "command"})
