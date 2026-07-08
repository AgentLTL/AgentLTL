"""
agentltl/integrations/langchain/constrained_agent.py – LangChain constraint middleware.

``ConstraintEnforcementMiddleware`` holds the stateful constraint enforcement
logic for LangChain agents.  It integrates with LangChain 1.0's
``create_agent()`` API via the ``AgentMiddleware`` protocol.

Usage::

    from agentltl.integrations.langchain import ConstraintEnforcementMiddleware
    from agentltl import Constraint, Before
    from langchain.agents import create_agent

    mw = ConstraintEnforcementMiddleware(
        constraints=[Constraint("order", Before("fetch", "save"))],
        constraint_severities={"order": ConstraintSeverity.SOFT_BLOCK},
        max_soft_attempts=3,
    )
    agent = create_agent(model=..., tools=..., middleware=[mw.as_middleware()])
    result = agent.invoke({"messages": [HumanMessage(content="...")]})
    status = mw.get_constraint_status()
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

from agentltl.enforcement import (
    ConstraintSeverity,
    ConstraintViolation,
    ConstraintViolationError,
    SoftBlockMode,
)
from agentltl.runtime_safety import check_runtime_safety_or_warn

logger = logging.getLogger(__name__)


class ConstraintEnforcementMiddleware:
    """Stateful FOLTL constraint enforcement middleware for LangChain agents.

    Wraps each tool call before execution, evaluating all registered
    constraints against the current trace.  Depending on the severity of any
    violation it either:

    * **HARD_STOP** — raises :class:`ConstraintViolationError` immediately,
      terminating the run.
    * **SOFT_BLOCK** — returns a :class:`ToolMessage` with a structured
      violation message (the tool is NOT executed), allowing the model to
      self-correct.  Escalates to a hard stop after the configured threshold.
    * **TOLERATE** — logs the violation and proceeds with execution.

    State is held on the instance.  Call :meth:`reset` before each new run
    (the :class:`LangChainConstrainedBackend` does this automatically).

    Args:
        constraints:           FOLTL constraints to enforce.
        constraint_severities: ``{constraint_name: ConstraintSeverity}`` mapping.
                               Constraints not listed here fall back to
                               *default_severity*.
        default_severity:      Fallback severity (default: ``HARD_STOP``).
        max_soft_attempts:     Cumulative / hybrid cap on SOFT_BLOCK violations
                               per constraint before escalating (default: 3).
        soft_block_mode:       Escalation strategy — ``"cumulative"``
                               (default), ``"consecutive"``, or ``"hybrid"``.
        max_consecutive_soft_attempts: Consecutive cap for ``consecutive`` /
                               ``hybrid`` modes.  Defaults to
                               *max_soft_attempts* when not set.
    """

    def __init__(
        self,
        constraints: Optional[List[Any]] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
        default_severity: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
        max_soft_attempts: int = 3,
        soft_block_mode: str = "cumulative",
        max_consecutive_soft_attempts: Optional[int] = None,
        strict_runtime_safety: bool = False,
        _skip_runtime_safety_check: bool = False,
    ) -> None:
        if not _skip_runtime_safety_check:
            check_runtime_safety_or_warn(
                constraints or [],
                constraint_severities,
                default_severity,
                strict=strict_runtime_safety,
                logger=logger,
            )
        self._constraints: List[Any] = list(constraints or [])
        self._severities: Dict[str, ConstraintSeverity] = dict(constraint_severities or {})
        self._default_severity = default_severity
        self._max_soft_attempts = max_soft_attempts
        self._soft_block_mode: SoftBlockMode = (
            soft_block_mode
            if isinstance(soft_block_mode, SoftBlockMode)
            else SoftBlockMode(soft_block_mode)
        )
        self._max_consecutive = (
            max_consecutive_soft_attempts
            if max_consecutive_soft_attempts is not None
            else max_soft_attempts
        )

        # Mutable run state — reset() clears these before each run.
        self._completed_tool_calls: List[Dict[str, Any]] = []
        self._soft_block_counts: Dict[str, int] = {}
        self._consecutive_soft_block_counts: Dict[str, int] = {}
        self._soft_blocked_calls: List[Dict[str, Any]] = []
        self._constraint_violations: List[Dict[str, Any]] = []
        self._constraint_checks: int = 0
        self._run_status: str = "completed"
        self._stopped_by: Optional[str] = None
        self._step_number: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all mutable run state.  Call before each new agent run."""
        self._completed_tool_calls = []
        self._soft_block_counts = {}
        self._consecutive_soft_block_counts = {}
        self._soft_blocked_calls = []
        self._constraint_violations = []
        self._constraint_checks = 0
        self._run_status = "completed"
        self._stopped_by = None
        self._step_number = 0
        self._persistent_block_counts: Dict[str, int] = {}

    def set_constraints(
        self,
        constraints: List[Any],
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
    ) -> None:
        """Replace constraints (and optionally severities) for the next run."""
        self._constraints = list(constraints)
        if constraint_severities is not None:
            self._severities = dict(constraint_severities)

    def as_middleware(self) -> Any:
        """Return a LangChain ``AgentMiddleware`` for use with ``create_agent``.

        The returned object captures ``self`` so that state is shared between
        the middleware and the :class:`ConstraintEnforcementMiddleware` instance.

        Returns:
            An ``AgentMiddleware`` created via ``@wrap_tool_call``.
        """
        mw = self

        @wrap_tool_call
        def _enforce(request: Any, handler: Callable) -> ToolMessage:
            return mw._handle(request, handler)

        return _enforce

    def get_constraint_status(self) -> Dict[str, Any]:
        """Return a structured summary of enforcement activity for this run.

        Returns:
            Dict with the same shape as
            ``ToolCallingAgentWithConstraints.get_constraint_status()``:

            * ``status`` — ``"completed"`` or ``"stopped"``
            * ``stopped_by`` — constraint name that triggered a hard stop, or ``None``
            * ``violations`` — list of violation record dicts
            * ``constraint_checks`` — total evaluations performed
            * ``completed_trace`` — tool calls that successfully executed
            * ``soft_blocked_calls`` — list of blocked-attempt dicts
            * ``soft_block_counts`` — ``{constraint_name: num_blocks}``
            * ``soft_block_mode`` — e.g. ``"cumulative"``
            * ``consecutive_soft_block_counts`` — ``{constraint_name: current_streak}``
        """
        return {
            "status": self._run_status,
            "stopped_by": self._stopped_by,
            "violations": list(self._constraint_violations),
            "constraint_checks": self._constraint_checks,
            "completed_trace": list(self._completed_tool_calls),
            "soft_blocked_calls": list(self._soft_blocked_calls),
            "soft_block_counts": dict(self._soft_block_counts),
            "soft_block_mode": self._soft_block_mode.value,
            "consecutive_soft_block_counts": dict(self._consecutive_soft_block_counts),
            "persistent_block_counts": dict(self._persistent_block_counts),
        }

    # ── Internal enforcement ──────────────────────────────────────────────────

    def _handle(self, request: Any, handler: Callable) -> ToolMessage:
        """Core middleware handler — called for every tool invocation.

        Evaluates constraints, then either blocks the call (returning a
        :class:`ToolMessage` with feedback) or allows it (calling *handler*
        and recording the result).
        """
        tool_call = getattr(request, "tool_call", None) or {}
        tool_name: str = tool_call.get("name", "unknown")
        tool_args: Dict[str, Any] = tool_call.get("args", {})
        tool_id: str = tool_call.get("id", "")

        self._step_number += 1

        # final_answer is not a real tool call — always pass through.
        if tool_name == "final_answer":
            result = handler(request)
            self._record_completed(tool_name, tool_args, tool_id, str(result.content))
            return result

        # Evaluate constraints against the current (partial) trace.
        violations = self._evaluate_constraints(tool_name, tool_args)

        for v in violations:
            severity = ConstraintSeverity(v.severity)

            if severity == ConstraintSeverity.TOLERATE:
                logger.warning(
                    "[CONSTRAINT TOLERATE] %s violated by '%s' at step %d: %s",
                    v.constraint_name, tool_name, self._step_number, v.detail,
                )
                self._constraint_violations.append(self._violation_to_dict(v))

            elif severity == ConstraintSeverity.SOFT_BLOCK:
                return self._handle_soft_block(v, tool_name, tool_args, tool_id)

            elif severity == ConstraintSeverity.PERSISTENT_BLOCK:
                return self._handle_persistent_block(v, tool_name, tool_args, tool_id)

            elif severity == ConstraintSeverity.HARD_STOP:
                self._run_status = "stopped"
                self._stopped_by = v.constraint_name
                self._constraint_violations.append(self._violation_to_dict(v))
                raise ConstraintViolationError(
                    constraint_name=v.constraint_name,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    detail=v.detail,
                    violation_type="HARD_STOP",
                )

        # No violations — execute the tool.
        result = handler(request)
        content = str(result.content) if hasattr(result, "content") else str(result)
        self._record_completed(tool_name, tool_args, tool_id, content)

        # Reset consecutive counters on a successful call.
        if self._soft_block_mode in (SoftBlockMode.CONSECUTIVE, SoftBlockMode.HYBRID):
            self._consecutive_soft_block_counts.clear()

        return result

    def _handle_soft_block(
        self,
        v: ConstraintViolation,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_id: str,
    ) -> ToolMessage:
        """Handle a SOFT_BLOCK violation.

        Increments counters and either returns a feedback :class:`ToolMessage`
        (allowing the model to self-correct) or escalates to a hard stop when
        the threshold is reached.
        """
        name = v.constraint_name
        self._soft_block_counts[name] = self._soft_block_counts.get(name, 0) + 1
        self._consecutive_soft_block_counts[name] = (
            self._consecutive_soft_block_counts.get(name, 0) + 1
        )
        total = self._soft_block_counts[name]
        consec = self._consecutive_soft_block_counts[name]

        self._soft_blocked_calls.append(
            {
                "step": self._step_number,
                "constraint_name": name,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "detail": v.detail,
            }
        )
        self._constraint_violations.append(self._violation_to_dict(v))

        # Determine whether to escalate.
        escalate = False
        threshold_str = ""
        if self._soft_block_mode == SoftBlockMode.CUMULATIVE:
            if total >= self._max_soft_attempts:
                escalate = True
                threshold_str = f"total {total}/{self._max_soft_attempts}"
        elif self._soft_block_mode == SoftBlockMode.CONSECUTIVE:
            if consec >= self._max_consecutive:
                escalate = True
                threshold_str = f"consecutive {consec}/{self._max_consecutive}"
        elif self._soft_block_mode == SoftBlockMode.HYBRID:
            if consec >= self._max_consecutive:
                escalate = True
                threshold_str = f"consecutive {consec}/{self._max_consecutive}"
            elif total >= self._max_soft_attempts:
                escalate = True
                threshold_str = f"total {total}/{self._max_soft_attempts}"

        if escalate:
            self._run_status = "stopped"
            self._stopped_by = name
            raise ConstraintViolationError(
                constraint_name=name,
                tool_name=tool_name,
                tool_args=tool_args,
                detail=v.detail,
                violation_type="SOFT_BLOCK_ESCALATION",
                soft_block_mode=self._soft_block_mode.value,
                threshold_str=threshold_str,
            )

        # Return blocking feedback as a ToolMessage so the model sees it.
        feedback = (
            f"[CONSTRAINT VIOLATION — SOFT_BLOCK]\n"
            f"Constraint '{name}' was violated by tool call '{tool_name}'.\n"
            f"The tool was NOT executed.\n"
            f"Detail: {v.detail}\n"
            f"Attempt {total}/{self._max_soft_attempts} "
            f"({self._soft_block_mode.value} mode). "
            f"Please choose a different action to comply with this constraint."
        )
        logger.warning(
            "[CONSTRAINT SOFT_BLOCK] %s violated by '%s' at step %d "
            "(attempt %d/%d): %s",
            name, tool_name, self._step_number,
            total, self._max_soft_attempts, v.detail,
        )
        return ToolMessage(content=feedback, tool_call_id=tool_id)

    def _handle_persistent_block(
        self,
        v: ConstraintViolation,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_id: str,
    ) -> ToolMessage:
        """Handle a PERSISTENT_BLOCK violation.

        Blocks the call and returns feedback, continues the run, never escalates,
        and offers no override — repeating the call is blocked again every time.
        """
        name = v.constraint_name
        self._persistent_block_counts[name] = self._persistent_block_counts.get(name, 0) + 1
        self._soft_blocked_calls.append(
            {
                "step": self._step_number,
                "constraint_name": name,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "detail": v.detail,
                "mode": "persistent_block",
            }
        )
        self._constraint_violations.append(self._violation_to_dict(v))

        feedback = (
            f"[CONSTRAINT VIOLATION — PERSISTENT_BLOCK]\n"
            f"Constraint '{name}' was violated by tool call '{tool_name}'.\n"
            f"The tool was NOT executed.\n"
            f"Detail: {v.detail}\n"
            f"This block cannot be overridden: repeating the exact same call will "
            f"not execute it. You must choose a different action to comply with the "
            f"constraint."
        )
        logger.warning(
            "[CONSTRAINT PERSISTENT_BLOCK] %s violated by '%s' at step %d: %s "
            "(block cannot be overridden).",
            name, tool_name, self._step_number, v.detail,
        )
        return ToolMessage(content=feedback, tool_call_id=tool_id)

    def _evaluate_constraints(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> List[ConstraintViolation]:
        """Evaluate all constraints against a prospective tool call.

        Builds a partial trace containing the already-completed calls plus the
        prospective new one, then evaluates each constraint against it using
        ``verify_trace(partial_trace=True)``.

        Returns:
            List of :class:`ConstraintViolation` instances for constraints that
            would be violated by this call.  Empty if all constraints pass.
        """
        if not self._constraints:
            return []

        from agentltl import verify_trace  # lazy import — avoids circular deps

        # Prospective trace: current completed calls + the candidate call.
        prospective_calls = list(self._completed_tool_calls) + [
            {"tool_name": tool_name, "arguments": tool_args}
        ]
        metrics = {"tool_calls": prospective_calls}

        violations: List[ConstraintViolation] = []
        self._constraint_checks += len(self._constraints)

        for constraint in self._constraints:
            result = verify_trace(metrics, [constraint], partial_trace=True)
            if result.get("compliance_label") not in ("FULL", "N/A"):
                severity = self._severities.get(
                    constraint.name, self._default_severity
                )
                per_c = result.get("constraints", [{}])[0]
                detail = per_c.get("detail") or per_c.get("reason") or "constraint violated"
                violations.append(
                    ConstraintViolation(
                        constraint_name=constraint.name,
                        severity=severity.value,
                        step_number=self._step_number,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        detail=str(detail),
                    )
                )

        return violations

    def _record_completed(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_id: str,
        result: str,
    ) -> None:
        self._completed_tool_calls.append(
            {
                "tool_name": tool_name,
                "arguments": tool_args,
                "id": tool_id,
                "result": result,
            }
        )

    @staticmethod
    def _violation_to_dict(v: ConstraintViolation) -> Dict[str, Any]:
        return {
            "constraint_name": v.constraint_name,
            "severity": v.severity,
            "step_number": v.step_number,
            "tool_name": v.tool_name,
            "tool_args": v.tool_args,
            "detail": v.detail,
        }
