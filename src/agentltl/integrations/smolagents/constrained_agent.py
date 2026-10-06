"""
constrained_agent.py – ToolCallingAgent with pre-execution constraint checking.

:class:`ToolCallingAgentWithConstraints` hands every tool call the model proposes to an
:class:`agentltl.Enforcer` before it runs. Depending on the decision the call runs, or the
model gets the enforcer's feedback instead of a result (``warn``, ``retry``, ``block``;
``ask`` too, since there is no human to ask here), or the run stops (``HARD_STOP``, or a
``SOFT_BLOCK`` that ran out of retries) with a :class:`ConstraintViolationError`.

``final_answer`` is not checked as a call. Constraints marked ``applies_to_final_answer``
are checked when the model gives it (strict semantics, the trace being final): with
``max_termination_nudges`` > 0 the model is sent back, that many times at most, to do
what is missing.

Usage
-----
>>> from agentltl.integrations.smolagents import (
...     ToolCallingAgentWithConstraints,
...     ConstraintSeverity,
... )
>>> from agentltl import Constraint, Before
>>> agent = ToolCallingAgentWithConstraints(
...     tools=my_tools,
...     model=my_model,
...     constraints=[Constraint("order", Before("fetch", "process"))],
...     constraint_severities={"order": ConstraintSeverity.HARD_STOP},
... )
>>> result = agent.run("do something", return_full_result=True)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional

from rich.panel import Panel
from rich.text import Text

from smolagents.agents import ToolCallingAgent, ToolOutput
from smolagents.agent_types import AgentImage, AgentAudio
from smolagents.memory import ActionStep, ToolCall
from smolagents.models import ChatMessage
from smolagents.monitoring import LogLevel
from smolagents.utils import AgentError

from agentltl.enforcement import (
    ConstraintSeverity,
    SoftBlockMode,
    ConstraintViolationError as _BaseConstraintViolationError,
)
from agentltl.enforcer import Enforcer
from agentltl.runtime_safety import check_runtime_safety_or_warn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# smolagents-compatible ConstraintViolationError
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintViolationError(_BaseConstraintViolationError, AgentError):
    """Raised when a constraint stops the run.

    Subclasses both the framework-agnostic
    :class:`agentltl.enforcement.ConstraintViolationError` **and** smolagents'
    :class:`AgentError`, so the smolagents run-loop catches it via its standard
    ``except AgentError`` handler.
    """

    def __init__(
        self,
        constraint_name: str,
        tool_name: str,
        tool_args: Any,
        detail: str = "",
        violation_type: str = "HARD_STOP",
        soft_block_mode: Optional[str] = None,
        threshold_str: str = "",
        logger_to_use=None,
    ):
        _BaseConstraintViolationError.__init__(
            self,
            constraint_name=constraint_name,
            tool_name=tool_name,
            tool_args=tool_args,
            detail=detail,
            violation_type=violation_type,
            soft_block_mode=soft_block_mode,
            threshold_str=threshold_str,
        )
        AgentError.__init__(self, str(self.args[0]), logger=logger_to_use)


# ─────────────────────────────────────────────────────────────────────────────
# The constrained agent
# ─────────────────────────────────────────────────────────────────────────────

class ToolCallingAgentWithConstraints(ToolCallingAgent):
    """A :class:`ToolCallingAgent` whose tool calls are judged by an
    :class:`agentltl.Enforcer` before they run.

    Parameters
    ----------
    tools, model, prompt_templates, planning_interval, stream_outputs,
    max_tool_threads, **kwargs
        Forwarded to :class:`ToolCallingAgent`.
    constraints, constraint_severities, default_severity, max_soft_attempts,
    soft_block_mode, max_consecutive_soft_attempts, nudge_max, max_termination_nudges
        Forwarded to :class:`agentltl.Enforcer`. Constraints and severities can also be
        given per run (:meth:`run`).
    enforcer_kwargs
        Further :class:`agentltl.Enforcer` settings (``rank``, ``render``, ...).
    """

    def __init__(
        self,
        tools,
        model,
        *,
        constraints: list | None = None,
        constraint_severities: Dict[str, ConstraintSeverity] | None = None,
        default_severity: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
        max_soft_attempts: int = 3,
        soft_block_mode: "str | SoftBlockMode" = "cumulative",
        max_consecutive_soft_attempts: "int | None" = None,
        nudge_max: int = 1,
        max_termination_nudges: int = 0,
        enforcer_kwargs: Optional[Dict[str, Any]] = None,
        prompt_templates=None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        max_tool_threads: int | None = None,
        strict_runtime_safety: bool = False,
        _skip_runtime_safety_check: bool = False,
        **kwargs,
    ):
        if not _skip_runtime_safety_check:
            check_runtime_safety_or_warn(
                constraints or [],
                constraint_severities,
                default_severity,
                strict=strict_runtime_safety,
                logger=logger,
            )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            stream_outputs=stream_outputs,
            max_tool_threads=max_tool_threads,
            **kwargs,
        )
        self._init_constraints = list(constraints or [])
        self._init_severities = dict(constraint_severities or {})
        self.enforcer = Enforcer(
            self._init_constraints, self._init_severities, default_severity,
            max_soft_attempts, soft_block_mode, max_consecutive_soft_attempts,
            nudge_max=nudge_max, max_termination_nudges=max_termination_nudges,
            **(enforcer_kwargs or {}),
        )
        self._blocked_tool_call: Optional[Dict[str, Any]] = None

    @property
    def soft_block_mode(self) -> SoftBlockMode:
        return self.enforcer.soft_block_mode

    def run(
        self,
        task: str,
        *,
        constraints: list | None = None,
        constraint_severities: Dict[str, ConstraintSeverity] | None = None,
        stream: bool = False,
        reset: bool = True,
        images=None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
        return_full_result: bool | None = None,
    ):
        """Run the agent with pre-execution constraint checking."""
        self.enforcer.reset()
        self._blocked_tool_call = None
        self.enforcer.set_constraints(
            constraints if constraints is not None else self._init_constraints,
            constraint_severities if constraint_severities is not None else self._init_severities,
        )
        return super().run(
            task=task,
            stream=stream,
            reset=reset,
            images=images,
            additional_args=additional_args,
            max_steps=max_steps,
            return_full_result=return_full_result,
        )

    def process_tool_calls(
        self, chat_message: ChatMessage, memory_step: ActionStep
    ) -> Generator[ToolCall | ToolOutput, None, None]:
        """Judge each tool call of this model message before it runs, one by one so
        later calls see the earlier ones in the trace."""
        if not self.enforcer.has_constraints:
            yield from super().process_tool_calls(chat_message, memory_step)
            return

        # one model message = one generation: identical parallel calls don't insist
        self.enforcer.begin_generation()
        parallel_calls: dict[str, ToolCall] = {}
        assert chat_message.tool_calls is not None
        for chat_tool_call in chat_message.tool_calls:
            tool_call = ToolCall(
                name=chat_tool_call.function.name,
                arguments=chat_tool_call.function.arguments,
                id=chat_tool_call.id,
            )
            yield tool_call
            parallel_calls[tool_call.id] = tool_call

        outputs: dict[str, ToolOutput] = {}
        for tool_call in parallel_calls.values():
            output = self._judge_and_run(tool_call, getattr(memory_step, "step_number", 0))
            outputs[output.id] = output
            yield output

        memory_step.tool_calls = [parallel_calls[k] for k in sorted(parallel_calls)]
        memory_step.observations = memory_step.observations or ""
        for tool_output in [outputs[k] for k in sorted(outputs)]:
            memory_step.observations += tool_output.observation + "\n"
        memory_step.observations = (
            memory_step.observations.rstrip("\n") if memory_step.observations
            else memory_step.observations
        )

    def _judge_and_run(self, tool_call: ToolCall, step: int) -> ToolOutput:
        name = tool_call.name
        args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        if name == "final_answer":
            decision = self.enforcer.check_termination()
            if not decision.allowed:
                return self._feedback(tool_call, decision.feedback)
        else:
            decision = self.enforcer.check(name, args, step)
            if decision.action == "stop":
                self._stop(tool_call, args, step, decision)
            if not decision.allowed:
                return self._feedback(tool_call, decision.feedback)

        self.logger.log(
            Panel(Text(f"Calling tool: '{name}' with arguments: {tool_call.arguments}")),
            level=LogLevel.INFO,
        )
        result = self.execute_tool_call(name, tool_call.arguments or {})
        if type(result) in (AgentImage, AgentAudio):
            observation_name = "image.png" if type(result) == AgentImage else "audio.mp3"
            self.state[observation_name] = result
            observation = f"Stored '{observation_name}' in memory."
        else:
            observation = str(result).strip()
        self.logger.log(f"Observations: {observation.replace('[', '|')}", level=LogLevel.INFO)
        if name != "final_answer":
            self.enforcer.record_completed(name, args, tool_call.id, observation)
        return ToolOutput(id=tool_call.id, output=result, is_final_answer=name == "final_answer",
                          observation=observation, tool_call=tool_call)

    def _stop(self, tool_call: ToolCall, args: Dict[str, Any], step: int, decision: Any) -> None:
        v = decision.violation
        self._blocked_tool_call = {
            "tool_name": tool_call.name, "tool_args": args, "tool_id": tool_call.id,
            "tool_result": "BLOCKED_BY_CONSTRAINT", "blocked_by": v.constraint_name,
            "step_number": step,
        }
        self.interrupt_switch = True
        raise ConstraintViolationError(
            constraint_name=v.constraint_name, tool_name=tool_call.name, tool_args=args,
            detail=v.detail,
            violation_type="SOFT_BLOCK_ESCALATION" if decision.escalated else "HARD_STOP",
            soft_block_mode=self.soft_block_mode.value if decision.escalated else None,
            threshold_str=(f"total {decision.attempts}/{self.enforcer.max_soft_attempts}"
                           if decision.escalated else ""),
            logger_to_use=self.logger,
        )

    @staticmethod
    def _feedback(tool_call: ToolCall, text: str) -> ToolOutput:
        return ToolOutput(id=tool_call.id, output=text, observation=text, is_final_answer=False,
                          tool_call=tool_call)

    def get_constraint_status(self) -> Dict[str, Any]:
        """The enforcer's status (see :meth:`agentltl.Enforcer.get_constraint_status`),
        plus ``blocked_tool_call``: the call a stop refused, if any."""
        return {**self.enforcer.get_constraint_status(),
                "blocked_tool_call": self._blocked_tool_call}
