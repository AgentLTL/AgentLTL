"""
constrained_agent.py – ToolCallingAgent with pre-execution constraint checking.

This module extends :class:`ToolCallingAgent` with a constraint evaluation layer
that intercepts every LLM-generated tool call **before** it is executed.  At each
interception point a partial trace (all completed calls plus the pending call) is
built and evaluated against a set of FOLTL constraints.  Depending on the
constraint's severity the agent either:

* **HARD_STOP** – aborts the run immediately (the tool call is never executed).
* **SOFT_BLOCK** – blocks the call, returns a constraint-violation observation to
  the model so it can self-correct, and continues the run.
* **TOLERATE** – logs a warning and continues execution.

The ``final_answer`` tool is always exempt from constraint checking because by the
time the agent decides to return an answer it is already too late to prevent any
side-effect.

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

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Sequence

from rich.panel import Panel
from rich.text import Text

from smolagents.agents import ToolCallingAgent, ActionOutput, ToolOutput
from smolagents.agent_types import AgentImage, AgentAudio
from smolagents.memory import ActionStep, ToolCall, Timing, FinalAnswerStep, TokenUsage
from smolagents.models import ChatMessage
from smolagents.monitoring import LogLevel
from smolagents.utils import AgentError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintSeverity(enum.Enum):
    """How the agent should react when a constraint is violated at runtime."""

    HARD_STOP = "HARD_STOP"
    """Abort the run immediately – the offending tool call is **not** executed."""

    SOFT_BLOCK = "SOFT_BLOCK"
    """Block the call, return a constraint-violation observation to the model so it
    can self-correct, and continue the run.  Escalates to HARD_STOP after the
    configured soft-block threshold (see :class:`SoftBlockMode`)."""

    TOLERATE = "TOLERATE"
    """Log the violation and continue execution."""


class SoftBlockMode(enum.Enum):
    """Escalation counting strategy for SOFT_BLOCK constraints.

    CUMULATIVE
        Count every violation of a constraint across the whole run.
        Escalates when the total reaches *max_soft_attempts*.  (Default.)

    CONSECUTIVE
        Count consecutive violations.  The counter resets to zero whenever a
        tool call completes successfully without triggering the same constraint.
        Escalates when *max_consecutive_soft_attempts* consecutive violations
        occur.  Catches tight retry loops while forgiving occasional re-offences
        separated by valid work.

    HYBRID
        Escalates on *either* condition: consecutive violations reach
        *max_consecutive_soft_attempts* (tight-loop detection) OR total
        violations reach *max_soft_attempts* (persistent-disregard detection).
    """

    CUMULATIVE  = "cumulative"
    CONSECUTIVE = "consecutive"
    HYBRID      = "hybrid"


@dataclass
class ConstraintViolation:
    """Record of a single constraint violation detected at runtime."""

    constraint_name: str
    severity: str  # "HARD_STOP" or "TOLERATE"
    step_number: int
    tool_name: str
    tool_args: Any
    detail: str = ""


class ConstraintViolationError(AgentError):
    """Raised when a constraint blocks a tool call before execution.

    Inherits from :class:`AgentError` so that the smolagents run-loop catches
    it in the standard ``except AgentError`` handler and records it on the
    :class:`ActionStep`.

    violation_type
        ``"HARD_STOP"``             – constraint fired immediately, no retries.
        ``"SOFT_BLOCK_ESCALATION"`` – soft-block threshold reached after N attempts.
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
        self.constraint_name = constraint_name
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.violation_detail = detail
        self.violation_type = violation_type
        self.soft_block_mode = soft_block_mode
        self.threshold_str = threshold_str

        if violation_type == "SOFT_BLOCK_ESCALATION":
            msg = (
                f"[CONSTRAINT ESCALATION — {soft_block_mode or 'unknown'} mode, {threshold_str}]\n"
                f"Constraint '{constraint_name}' has been violated too many times.\n"
                f"Tool call '{tool_name}' is now permanently blocked.\n"
                f"Detail: {detail}\n"
                f"You must take a fundamentally different approach to satisfy this constraint."
            )
        else:  # HARD_STOP
            msg = (
                f"[CONSTRAINT HARD-STOP]\n"
                f"Constraint '{constraint_name}' was violated by tool call '{tool_name}'.\n"
                f"The tool was NOT executed.\n"
                f"Detail: {detail}\n"
                f"Do NOT attempt this tool call. Choose a different action."
            )
        super().__init__(msg, logger=logger_to_use)

    def dict(self) -> dict:
        return {
            "type": "ConstraintViolationError",
            "violation_type": self.violation_type,
            "constraint_name": self.constraint_name,
            "tool_name": self.tool_name,
            "detail": self.violation_detail,
            "soft_block_mode": self.soft_block_mode,
            "threshold_str": self.threshold_str,
            "message": str(self.args[0]),
        }


class _SoftBlockSignal(Exception):
    """Internal signal raised inside process_single_tool_call_constrained when a
    SOFT_BLOCK constraint fires.  Caught by the outer generator to yield feedback."""

    def __init__(
        self,
        tool_call,
        violation: "ConstraintViolation",
        count: int,
        consec: int = 0,
        threshold_str: str = "",
    ):
        self.tool_call = tool_call
        self.violation = violation
        self.count = count
        self.consec = consec
        self.threshold_str = threshold_str


# ─────────────────────────────────────────────────────────────────────────────
# The constrained agent
# ─────────────────────────────────────────────────────────────────────────────

class ToolCallingAgentWithConstraints(ToolCallingAgent):
    """A :class:`ToolCallingAgent` that evaluates FOLTL constraints **before**
    every tool call.

    Parameters
    ----------
    tools, model, prompt_templates, planning_interval, stream_outputs,
    max_tool_threads, **kwargs
        Forwarded to :class:`ToolCallingAgent`.
    constraints : list[Constraint] | None
        FOLTL constraints to enforce at every step.  Can also be supplied (or
        overridden) per-run via :meth:`run`.
    constraint_severities : dict[str, ConstraintSeverity] | None
        Maps constraint **names** → severity.  Constraints not listed here
        default to ``HARD_STOP``.
    default_severity : ConstraintSeverity
        Fallback severity for constraints that are not in *constraint_severities*.
    max_soft_attempts : int
        For ``CUMULATIVE`` and ``HYBRID`` modes: maximum total SOFT_BLOCK
        violations allowed per constraint per run before escalating.  For
        ``CONSECUTIVE`` mode this is unused (use *max_consecutive_soft_attempts*).
        Default: 3.
    soft_block_mode : str | SoftBlockMode
        Escalation counting strategy.  One of ``"cumulative"`` (default),
        ``"consecutive"``, or ``"hybrid"``.  See :class:`SoftBlockMode`.
    max_consecutive_soft_attempts : int | None
        For ``CONSECUTIVE`` and ``HYBRID`` modes: maximum *consecutive*
        violations before escalating.  Defaults to *max_soft_attempts* if not
        set.
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
        prompt_templates=None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        max_tool_threads: int | None = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            stream_outputs=stream_outputs,
            max_tool_threads=max_tool_threads,
            **kwargs,
        )
        self._init_constraints = constraints or []
        self._init_severities = constraint_severities or {}
        self._default_severity = default_severity
        self.max_soft_attempts = max_soft_attempts
        self.soft_block_mode = (
            SoftBlockMode(soft_block_mode)
            if isinstance(soft_block_mode, str)
            else soft_block_mode
        )
        self.max_consecutive_soft_attempts = (
            max_consecutive_soft_attempts
            if max_consecutive_soft_attempts is not None
            else max_soft_attempts
        )

        # Per-run mutable state (reset in _reset_constraint_state)
        self._active_constraints: list = []
        self._active_severities: Dict[str, ConstraintSeverity] = {}
        self._completed_tool_calls: List[Dict[str, Any]] = []
        self._constraint_violations: List[ConstraintViolation] = []
        self._run_status: str = "completed"
        self._stopped_by: str | None = None
        self._blocked_tool_call: Dict[str, Any] | None = None
        self._constraint_checks_count: int = 0
        self._soft_block_counts: Dict[str, int] = {}
        self._consecutive_soft_block_counts: Dict[str, int] = {}
        self._soft_blocked_calls: List[Dict[str, Any]] = []

    def _reset_constraint_state(self):
        """Reset per-run mutable state.  Called at the start of each ``run``."""
        self._completed_tool_calls = []
        self._constraint_violations = []
        self._run_status = "completed"
        self._stopped_by = None
        self._blocked_tool_call = None
        self._constraint_checks_count = 0
        self._soft_block_counts = {}
        self._consecutive_soft_block_counts = {}
        self._soft_blocked_calls = []

    def _severity_for(self, constraint_name: str) -> ConstraintSeverity:
        return self._active_severities.get(constraint_name, self._default_severity)

    def _build_speculative_trace(
        self, tool_name: str, tool_args: Any, tool_id: str | None = None
    ) -> Dict[str, Any]:
        """Build a metrics dict with all completed calls plus one pending call."""
        pending = {
            "tool_name": tool_name,
            "tool_args": tool_args if isinstance(tool_args, dict) else {},
            "tool_id": tool_id,
            "tool_result": None,
        }
        return {"tool_calls": self._completed_tool_calls + [pending]}

    def _evaluate_constraints(
        self,
        speculative_metrics: Dict[str, Any],
        step_number: int,
        tool_name: str,
        tool_args: Any,
    ) -> List[ConstraintViolation]:
        """Run FOLTL constraint evaluation on the speculative trace."""
        from agentltl import verify_trace

        self._constraint_checks_count += 1

        result = verify_trace(
            speculative_metrics,
            self._active_constraints,
            partial_trace=True,
        )

        violations: List[ConstraintViolation] = []
        for cr in result.get("constraints", []):
            if not cr["passed"]:
                severity = self._severity_for(cr["name"])
                violations.append(ConstraintViolation(
                    constraint_name=cr["name"],
                    severity=severity.value,
                    step_number=step_number,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    detail=cr.get("detail", ""),
                ))
        return violations

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
        self._reset_constraint_state()

        self._active_constraints = constraints if constraints is not None else self._init_constraints
        sev = constraint_severities if constraint_severities is not None else self._init_severities
        self._active_severities = sev

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
        """Process tool calls with pre-execution constraint evaluation.

        When no constraints are active this method falls back to the parent
        implementation for zero overhead.
        """
        if not self._active_constraints:
            yield from super().process_tool_calls(chat_message, memory_step)
            return

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

        def process_single_tool_call_constrained(tool_call: ToolCall) -> ToolOutput:
            tool_name = tool_call.name
            tool_arguments = tool_call.arguments or {}

            if tool_name != "final_answer" and self._active_constraints:
                step_num = getattr(memory_step, "step_number", 0)
                spec_metrics = self._build_speculative_trace(tool_name, tool_arguments, tool_call.id)
                violations = self._evaluate_constraints(spec_metrics, step_num, tool_name, tool_arguments)

                for v in violations:
                    self._constraint_violations.append(v)

                    if v.severity == ConstraintSeverity.HARD_STOP.value:
                        self._run_status = "stopped"
                        self._stopped_by = v.constraint_name
                        self._blocked_tool_call = {
                            "tool_name": tool_name,
                            "tool_args": tool_arguments if isinstance(tool_arguments, dict) else {},
                            "tool_id": tool_call.id,
                            "tool_result": "BLOCKED_BY_CONSTRAINT",
                            "blocked_by": v.constraint_name,
                            "step_number": step_num,
                        }
                        logger.warning(
                            "HARD_STOP: constraint '%s' violated by tool '%s' (step %d). Aborting.",
                            v.constraint_name, tool_name, step_num,
                        )
                        self.interrupt_switch = True
                        raise ConstraintViolationError(
                            constraint_name=v.constraint_name,
                            tool_name=tool_name,
                            tool_args=tool_arguments,
                            detail=v.detail,
                            violation_type="HARD_STOP",
                            logger_to_use=self.logger,
                        )
                    elif v.severity == ConstraintSeverity.SOFT_BLOCK.value:
                        # Cumulative counter (always incremented)
                        count = self._soft_block_counts.get(v.constraint_name, 0) + 1
                        self._soft_block_counts[v.constraint_name] = count

                        # Consecutive counter (CONSECUTIVE / HYBRID modes only)
                        if self.soft_block_mode in (SoftBlockMode.CONSECUTIVE, SoftBlockMode.HYBRID):
                            consec = self._consecutive_soft_block_counts.get(v.constraint_name, 0) + 1
                            self._consecutive_soft_block_counts[v.constraint_name] = consec
                        else:
                            consec = count  # mirrors cumulative for logging in CUMULATIVE mode

                        # Build human-readable threshold string for logs and model feedback
                        if self.soft_block_mode == SoftBlockMode.CUMULATIVE:
                            threshold_str = f"total {count}/{self.max_soft_attempts}"
                        elif self.soft_block_mode == SoftBlockMode.CONSECUTIVE:
                            threshold_str = f"consecutive {consec}/{self.max_consecutive_soft_attempts}"
                        else:  # HYBRID
                            threshold_str = (
                                f"consecutive {consec}/{self.max_consecutive_soft_attempts}, "
                                f"total {count}/{self.max_soft_attempts}"
                            )

                        self._soft_blocked_calls.append({
                            "step": step_num,
                            "tool_name": tool_name,
                            "tool_args": tool_arguments if isinstance(tool_arguments, dict) else {},
                            "constraint": v.constraint_name,
                            "detail": v.detail,
                            "attempt": count,
                            "consecutive": consec,
                        })
                        logger.warning(
                            "SOFT_BLOCK: constraint '%s' violated by tool '%s' "
                            "(step %d, %s).",
                            v.constraint_name, tool_name, step_num, threshold_str,
                        )

                        # Determine whether to escalate based on the active mode
                        if self.soft_block_mode == SoftBlockMode.CUMULATIVE:
                            escalate = count >= self.max_soft_attempts
                            escalation_detail = (
                                f"Max soft-block attempts ({self.max_soft_attempts}) exceeded "
                                f"for constraint '{v.constraint_name}'. Halting."
                            )
                        elif self.soft_block_mode == SoftBlockMode.CONSECUTIVE:
                            escalate = consec >= self.max_consecutive_soft_attempts
                            escalation_detail = (
                                f"Max consecutive soft-block attempts "
                                f"({self.max_consecutive_soft_attempts}) exceeded "
                                f"for constraint '{v.constraint_name}'. Halting."
                            )
                        else:  # HYBRID
                            escalate = (
                                consec >= self.max_consecutive_soft_attempts
                                or count >= self.max_soft_attempts
                            )
                            escalation_detail = (
                                f"Soft-block escalation threshold reached for constraint "
                                f"'{v.constraint_name}' ({threshold_str}). Halting."
                            )

                        if escalate:
                            self._run_status = "stopped"
                            self._stopped_by = v.constraint_name
                            self.interrupt_switch = True
                            raise ConstraintViolationError(
                                constraint_name=v.constraint_name,
                                tool_name=tool_name,
                                tool_args=tool_arguments,
                                detail=escalation_detail,
                                violation_type="SOFT_BLOCK_ESCALATION",
                                soft_block_mode=self.soft_block_mode.value,
                                threshold_str=threshold_str,
                                logger_to_use=self.logger,
                            )
                        raise _SoftBlockSignal(
                            tool_call=tool_call,
                            violation=v,
                            count=count,
                            consec=consec,
                            threshold_str=threshold_str,
                        )
                    else:
                        logger.warning(
                            "TOLERATE: constraint '%s' violated by tool '%s' (step %d). Continuing.",
                            v.constraint_name, tool_name, step_num,
                        )

            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            tool_call_result = self.execute_tool_call(tool_name, tool_arguments)
            tool_call_result_type = type(tool_call_result)

            if tool_call_result_type in [AgentImage, AgentAudio]:
                observation_name = "image.png" if tool_call_result_type == AgentImage else "audio.mp3"
                self.state[observation_name] = tool_call_result
                observation = f"Stored '{observation_name}' in memory."
            else:
                observation = str(tool_call_result).strip()

            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",
                level=LogLevel.INFO,
            )
            is_final_answer = tool_name == "final_answer"

            self._completed_tool_calls.append({
                "tool_name": tool_name,
                "tool_args": tool_arguments if isinstance(tool_arguments, dict) else {},
                "tool_id": tool_call.id,
                "tool_result": observation,
            })

            return ToolOutput(
                id=tool_call.id,
                output=tool_call_result,
                is_final_answer=is_final_answer,
                observation=observation,
                tool_call=tool_call,
            )

        # Execute sequentially when constraints active (deterministic partial-trace ordering)
        all_outputs: dict[str, ToolOutput] = {}
        for tool_call in parallel_calls.values():
            try:
                tool_output = process_single_tool_call_constrained(tool_call)
                all_outputs[tool_output.id] = tool_output
                yield tool_output
                # Successful execution: reset consecutive streak so the agent
                # is not penalised for past violations separated by valid work.
                if self.soft_block_mode in (SoftBlockMode.CONSECUTIVE, SoftBlockMode.HYBRID):
                    self._consecutive_soft_block_counts.clear()
            except _SoftBlockSignal as sig:
                feedback = (
                    f"[CONSTRAINT VIOLATION — {sig.violation.constraint_name}]"
                    f" ({sig.threshold_str})\n"
                    f"{sig.violation.detail}\n"
                    f"The tool call '{sig.tool_call.name}' was NOT executed. "
                    f"Please reconsider and call a different tool or arguments "
                    f"that satisfy the constraints."
                )
                feedback_output = ToolOutput(
                    id=sig.tool_call.id,
                    output=feedback,
                    observation=feedback,
                    is_final_answer=False,
                    tool_call=sig.tool_call,
                )
                all_outputs[sig.tool_call.id] = feedback_output
                yield feedback_output

        memory_step.tool_calls = [parallel_calls[k] for k in sorted(parallel_calls.keys())]
        memory_step.observations = memory_step.observations or ""
        for tool_output in [all_outputs[k] for k in sorted(all_outputs.keys())]:
            memory_step.observations += tool_output.observation + "\n"
        memory_step.observations = (
            memory_step.observations.rstrip("\n") if memory_step.observations else memory_step.observations
        )

    def get_constraint_status(self) -> Dict[str, Any]:
        """Return a summary of the constraint evaluation state after a run.

        Returns
        -------
        dict with keys:
            ``status``                       – ``"completed"`` or ``"stopped"``
            ``stopped_by``                   – name of the halting constraint (or ``None``)
            ``blocked_tool_call``            – HARD_STOP blocked call dict (or ``None``)
            ``violations``                   – list of violation dicts (all severities)
            ``completed_trace``              – list of successfully-executed tool call dicts
            ``constraint_checks``            – total number of constraint evaluations performed
            ``soft_blocked_calls``           – list of SOFT_BLOCK attempt dicts
            ``soft_block_counts``            – mapping constraint_name → # of soft blocks
            ``soft_block_mode``              – active escalation mode value string
            ``consecutive_soft_block_counts`` – mapping constraint_name → current consecutive count
        """
        return {
            "status": self._run_status,
            "stopped_by": self._stopped_by,
            "blocked_tool_call": self._blocked_tool_call,
            "violations": [
                {
                    "constraint_name": v.constraint_name,
                    "severity": v.severity,
                    "step_number": v.step_number,
                    "tool_name": v.tool_name,
                    "tool_args": v.tool_args,
                    "detail": v.detail,
                }
                for v in self._constraint_violations
            ],
            "completed_trace": list(self._completed_tool_calls),
            "constraint_checks": self._constraint_checks_count,
            "soft_blocked_calls": list(self._soft_blocked_calls),
            "soft_block_counts": dict(self._soft_block_counts),
            "soft_block_mode": self.soft_block_mode.value,
            "consecutive_soft_block_counts": dict(self._consecutive_soft_block_counts),
        }
