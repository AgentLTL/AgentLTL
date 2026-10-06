"""
agentltl/_enforcement_engine.py — the original ``check()`` interface, kept for existing
callers.

:class:`ConstraintEnforcer` is :class:`agentltl.Enforcer` with the return values the
native backend and older harnesses expect: ``"allow"``, or ``(kind, feedback)`` where
kind is ``soft_block``, ``block_and_warn``, ``persistent_block`` or ``ask``; a HARD_STOP
or a soft-block escalation raises :class:`ConstraintViolationError`. New code should use
:class:`agentltl.Enforcer`, which returns a :class:`agentltl.Decision` instead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from .enforcement import ConstraintSeverity, ConstraintViolationError
from .enforcer import Decision as _Decision
from .enforcer import Enforcer

# Return type of ConstraintEnforcer.check(): "allow", or (kind, feedback).
Decision = Union[str, Tuple[str, str]]

_KINDS = {"retry": "soft_block", "warn": "block_and_warn", "block": "persistent_block",
          "ask": "ask"}


class ConstraintEnforcer(Enforcer):
    """:class:`Enforcer` with the original ``check()`` return values and exceptions."""

    def __init__(
        self,
        constraints: Optional[Any] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
        default_severity: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
        max_soft_attempts: int = 3,
        soft_block_mode: Any = "cumulative",
        max_consecutive_soft_attempts: Optional[int] = None,
        nudge_max: int = 1,
        max_termination_nudges: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(constraints, constraint_severities, default_severity,
                         max_soft_attempts, soft_block_mode, max_consecutive_soft_attempts,
                         nudge_max=nudge_max, max_termination_nudges=max_termination_nudges,
                         **kwargs)

    def check(self, tool_name: str, tool_args: Optional[Dict[str, Any]] = None,  # type: ignore[override]
              step_number: Optional[int] = None, **kwargs: Any) -> Decision:
        return self.legacy(super().check(tool_name, tool_args, step_number, **kwargs))

    def legacy(self, d: _Decision) -> Decision:
        """A :class:`agentltl.Decision` as the original return value, or raised."""
        if d.allowed:
            return "allow"
        if d.action == "stop":
            v = d.violation
            raise ConstraintViolationError(
                constraint_name=v.constraint_name, tool_name=v.tool_name,
                tool_args=v.tool_args, detail=v.detail,
                violation_type="SOFT_BLOCK_ESCALATION" if d.escalated else "HARD_STOP",
                soft_block_mode=self.soft_block_mode.value if d.escalated else None,
                threshold_str=(f"total {d.attempts}/{self.max_soft_attempts}"
                               if d.escalated else ""))
        return (_KINDS[d.action], d.feedback)

    def check_termination(self) -> Optional[str]:  # type: ignore[override]
        """The message sending the agent back, or None to let it finish."""
        d = super().check_termination()
        return None if d.allowed else d.feedback

    # attribute names older callers read
    @property
    def _soft_block_mode(self):
        return self.soft_block_mode

    @property
    def _max_soft_attempts(self) -> int:
        return self.max_soft_attempts

    @property
    def _max_consecutive(self) -> int:
        return self.max_consecutive
