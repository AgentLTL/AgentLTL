"""
agentltl/_enforcement_engine.py — framework-agnostic runtime constraint enforcer.

This is the backend-independent core of the runtime enforcement logic that the
LangChain ``ConstraintEnforcementMiddleware`` implements inline. It has **no
framework dependency** (no langchain/smolagents import) and lazy-imports
``verify_trace`` only when constraints are present, so it is safe to import from
the zero-dependency core or from the native backend.

The native agent loop drives it like this::

    enforcer.reset()
    ...
    decision = enforcer.check(tool_name, tool_args, step_number)
    if decision == "allow":
        result = run_tool(...)
        enforcer.record_completed(tool_name, tool_args, tool_id, result)
    elif decision[0] == "soft_block":
        feedback = decision[1]          # feed back to the model; do NOT execute
    # HARD_STOP / soft-block escalation raise ConstraintViolationError, which the
    # backend's run() catches and reports as the error payload.

The severity semantics (HARD_STOP / SOFT_BLOCK / TOLERATE), soft-block counting
(cumulative / consecutive / hybrid) and the ``get_constraint_status()`` shape are
intentionally identical to the LangChain middleware so behaviour matches across
backends.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from .enforcement import (
    ConstraintSeverity,
    ConstraintViolation,
    ConstraintViolationError,
    SoftBlockMode,
)

logger = logging.getLogger(__name__)

# Return type of check(): either the literal "allow", or ("soft_block", feedback).
Decision = Union[str, Tuple[str, str]]


class ConstraintEnforcer:
    """Stateful FOLTL runtime enforcer, framework-agnostic.

    Mirrors the behaviour of
    :class:`agentltl.integrations.langchain.constrained_agent.ConstraintEnforcementMiddleware`
    but exposes a plain ``check()`` decision instead of wrapping a framework's
    tool-call handler, so a hand-written agent loop can use it directly.
    """

    def __init__(
        self,
        constraints: Optional[List[Any]] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
        default_severity: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
        max_soft_attempts: int = 3,
        soft_block_mode: str = "cumulative",
        max_consecutive_soft_attempts: Optional[int] = None,
    ) -> None:
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
        self.reset()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all mutable run state. Call before each new run (NOT between
        turns of one multi-turn session — constraints span the whole session)."""
        self._completed_tool_calls: List[Dict[str, Any]] = []
        self._soft_block_counts: Dict[str, int] = {}
        self._consecutive_soft_block_counts: Dict[str, int] = {}
        self._soft_blocked_calls: List[Dict[str, Any]] = []
        self._constraint_violations: List[Dict[str, Any]] = []
        self._constraint_checks: int = 0
        self._run_status: str = "completed"
        self._stopped_by: Optional[str] = None

    def set_constraints(
        self,
        constraints: List[Any],
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
    ) -> None:
        self._constraints = list(constraints)
        if constraint_severities is not None:
            self._severities = dict(constraint_severities)

    @property
    def has_constraints(self) -> bool:
        return bool(self._constraints)

    def check(self, tool_name: str, tool_args: Dict[str, Any], step_number: int) -> Decision:
        """Evaluate constraints against the prospective call.

        Returns ``"allow"`` if the call may proceed, or ``("soft_block", feedback)``
        if it must be blocked but the run continues. Raises
        :class:`ConstraintViolationError` on HARD_STOP or soft-block escalation.
        """
        violations = self._evaluate_constraints(tool_name, tool_args, step_number)

        for v in violations:
            severity = ConstraintSeverity(v.severity)

            if severity == ConstraintSeverity.TOLERATE:
                logger.warning(
                    "[CONSTRAINT TOLERATE] %s violated by '%s' at step %d: %s",
                    v.constraint_name, tool_name, step_number, v.detail,
                )
                self._constraint_violations.append(self._violation_to_dict(v))

            elif severity == ConstraintSeverity.SOFT_BLOCK:
                # First soft-block found decides the outcome of this call.
                return self._handle_soft_block(v, tool_name, tool_args, step_number)

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

        # No blocking violation — allow. Reset consecutive counters on success.
        if self._soft_block_mode in (SoftBlockMode.CONSECUTIVE, SoftBlockMode.HYBRID):
            self._consecutive_soft_block_counts.clear()
        return "allow"

    def record_completed(
        self, tool_name: str, tool_args: Dict[str, Any], tool_id: str, result: str
    ) -> None:
        """Record a successfully-executed call so it joins the prospective trace
        evaluated for subsequent calls."""
        self._completed_tool_calls.append(
            {"tool_name": tool_name, "arguments": tool_args, "id": tool_id, "result": result}
        )

    def get_constraint_status(self) -> Dict[str, Any]:
        """Same shape as the smolagents / langchain backends."""
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
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evaluate_constraints(
        self, tool_name: str, tool_args: Dict[str, Any], step_number: int
    ) -> List[ConstraintViolation]:
        """Build a prospective trace (completed + candidate) and evaluate each
        constraint with ``verify_trace(partial_trace=True)``."""
        if not self._constraints:
            return []

        from agentltl import verify_trace  # lazy import — avoids circular deps

        prospective_calls = list(self._completed_tool_calls) + [
            {"tool_name": tool_name, "arguments": tool_args}
        ]
        metrics = {"tool_calls": prospective_calls}

        violations: List[ConstraintViolation] = []
        self._constraint_checks += len(self._constraints)

        for constraint in self._constraints:
            result = verify_trace(metrics, [constraint], partial_trace=True)
            if result.get("compliance_label") not in ("FULL", "N/A"):
                severity = self._severities.get(constraint.name, self._default_severity)
                per_c = result.get("constraints", [{}])[0]
                detail = per_c.get("detail") or per_c.get("reason") or "constraint violated"
                violations.append(
                    ConstraintViolation(
                        constraint_name=constraint.name,
                        severity=severity.value,
                        step_number=step_number,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        detail=str(detail),
                    )
                )
        return violations

    def _handle_soft_block(
        self, v: ConstraintViolation, tool_name: str, tool_args: Dict[str, Any], step_number: int
    ) -> Tuple[str, str]:
        name = v.constraint_name
        self._soft_block_counts[name] = self._soft_block_counts.get(name, 0) + 1
        self._consecutive_soft_block_counts[name] = (
            self._consecutive_soft_block_counts.get(name, 0) + 1
        )
        total = self._soft_block_counts[name]
        consec = self._consecutive_soft_block_counts[name]

        self._soft_blocked_calls.append(
            {
                "step": step_number,
                "constraint_name": name,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "detail": v.detail,
            }
        )
        self._constraint_violations.append(self._violation_to_dict(v))

        escalate = False
        threshold_str = ""
        if self._soft_block_mode == SoftBlockMode.CUMULATIVE:
            if total >= self._max_soft_attempts:
                escalate, threshold_str = True, f"total {total}/{self._max_soft_attempts}"
        elif self._soft_block_mode == SoftBlockMode.CONSECUTIVE:
            if consec >= self._max_consecutive:
                escalate, threshold_str = True, f"consecutive {consec}/{self._max_consecutive}"
        elif self._soft_block_mode == SoftBlockMode.HYBRID:
            if consec >= self._max_consecutive:
                escalate, threshold_str = True, f"consecutive {consec}/{self._max_consecutive}"
            elif total >= self._max_soft_attempts:
                escalate, threshold_str = True, f"total {total}/{self._max_soft_attempts}"

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

        feedback = (
            f"[CONSTRAINT VIOLATION — SOFT_BLOCK]\n"
            f"Constraint '{name}' was violated by tool call '{tool_name}'.\n"
            f"The tool was NOT executed.\n"
            f"Detail: {v.detail}\n"
            f"Attempt {total}/{self._max_soft_attempts} ({self._soft_block_mode.value} mode). "
            f"Please choose a different action to comply with this constraint."
        )
        logger.warning(
            "[CONSTRAINT SOFT_BLOCK] %s violated by '%s' at step %d (attempt %d/%d): %s",
            name, tool_name, step_number, total, self._max_soft_attempts, v.detail,
        )
        return ("soft_block", feedback)

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
