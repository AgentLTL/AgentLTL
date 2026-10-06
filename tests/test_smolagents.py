"""The smolagents integration drives agentltl.Enforcer: a scripted model, real smolagents."""

import pytest

smolagents = pytest.importorskip("smolagents")

from smolagents import Tool  # noqa: E402
from smolagents.models import (  # noqa: E402
    ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole, Model,
)
from smolagents.monitoring import TokenUsage  # noqa: E402

from agentltl import (  # noqa: E402
    Before, Called, Constraint, ConstraintSeverity, Globally, Not, Now, parse,
)
from agentltl.integrations.smolagents import ToolCallingAgentWithConstraints  # noqa: E402

S = ConstraintSeverity


class Echo(Tool):
    output_type = "string"
    inputs = {"x": {"type": "string", "description": "anything", "nullable": True}}

    def __init__(self, name):
        self.name = name
        self.description = f"The {name} tool."
        super().__init__()

    def forward(self, x=None):
        return f"{self.name} ran"


class Scripted(Model):
    """Answers each step with the next list of tool calls."""

    def __init__(self, steps):
        super().__init__(model_id="scripted")
        self.steps = list(steps)

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs):
        names = self.steps.pop(0) if self.steps else ["final_answer"]
        calls = [ChatMessageToolCall(
            function=ChatMessageToolCallFunction(
                name=n, arguments={"answer": "done"} if n == "final_answer" else {"x": "1"}),
            id=f"call_{len(self.steps)}_{i}", type="function") for i, n in enumerate(names)]
        return ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=calls,
                           token_usage=TokenUsage(input_tokens=1, output_tokens=1))


def agent(steps, constraints, severity, **kw):
    tools = [Echo(n) for n in ("fetch", "process", "deploy", "ls")]
    return ToolCallingAgentWithConstraints(tools, Scripted(steps), constraints=constraints,
                                           default_severity=severity, max_steps=10, **kw)


def ran(a):
    return [c["tool_name"] for c in a.get_constraint_status()["completed_trace"]]


def test_a_refused_call_gets_feedback_and_the_run_goes_on():
    a = agent([["process"], ["fetch"], ["process"]],
              [Constraint("order", Before("fetch", "process"))], S.PERSISTENT_BLOCK)
    a.run("go")
    assert ran(a) == ["fetch", "process"]
    assert a.get_constraint_status()["persistent_block_counts"] == {"order": 1}


def test_a_hard_stop_ends_the_run():
    a = agent([["fetch"], ["deploy"], ["ls"]],
              [Constraint("never-deploy", Globally(Not(Now("deploy"))))], S.HARD_STOP)
    with pytest.raises(Exception):
        a.run("go")
    status = a.get_constraint_status()
    assert status["status"] == "stopped" and status["blocked_tool_call"]["tool_name"] == "deploy"
    assert ran(a) == ["fetch"]


def test_warn_override_and_marginal_causation():
    once = Constraint("one-deploy", parse('G(now("deploy") -> X(G(!now("deploy"))))'))
    a = agent([["deploy"], ["deploy"], ["deploy"], ["ls"], ["deploy"]], [once], S.BLOCK_AND_WARN)
    a.run("go")
    assert ran(a) == ["deploy", "deploy", "ls"]
    assert a.get_constraint_status()["block_and_warn_override_count"] == 1


def test_identical_parallel_calls_do_not_insist():
    never = Constraint("no-deploy", Globally(Not(Now("deploy"))))
    a = agent([["deploy", "deploy"]], [never], S.BLOCK_AND_WARN)
    a.run("go")
    assert ran(a) == []


def test_final_answer_can_be_sent_back_for_an_unmet_obligation():
    must = Constraint("fetched", Called("fetch"), applies_to_final_answer=True)
    a = agent([["final_answer"], ["fetch"]], [must], S.TOLERATE, max_termination_nudges=1)
    a.run("go")
    assert ran(a) == ["fetch"]
    assert a.get_constraint_status()["termination_nudged_names"] == ["fetched"]
