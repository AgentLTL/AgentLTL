"""
agentltl/enforcement.py – shared runtime enforcement types.

This module defines the public types used by ALL backend integrations
(smolagents, LangChain, etc.) for constraint enforcement at runtime.
It has **zero framework dependencies** and can be imported without
installing any optional extras.

Types
-----
* :class:`ConstraintSeverity` – HARD_STOP / SOFT_BLOCK / TOLERATE
* :class:`SoftBlockMode`      – cumulative / consecutive / hybrid
* :class:`ConstraintViolation` – record of a single runtime violation
* :class:`ConstraintViolationError` – exception raised when a constraint
  blocks a tool call before execution
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Severity and mode enums
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


# ─────────────────────────────────────────────────────────────────────────────
# Violation record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConstraintViolation:
    """Record of a single constraint violation detected at runtime."""

    constraint_name: str
    severity: str          # ConstraintSeverity.value — "HARD_STOP", "SOFT_BLOCK", or "TOLERATE"
    step_number: int
    tool_name: str
    tool_args: Any
    detail: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Exception
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintViolationError(Exception):
    """Raised when a constraint blocks a tool call before execution.

    This is the **framework-agnostic** base class.  Framework-specific
    integrations may subclass it to add compatibility with their own
    exception hierarchies (e.g. the smolagents integration subclasses it
    to also inherit from ``smolagents.utils.AgentError`` so the run-loop
    catches it correctly).

    Parameters
    ----------
    constraint_name
        Name of the constraint that triggered the block.
    tool_name
        Name of the tool call that was blocked.
    tool_args
        Arguments of the blocked tool call.
    detail
        Human-readable explanation from the constraint evaluator.
    violation_type
        ``"HARD_STOP"``             – constraint fired immediately, no retries.
        ``"SOFT_BLOCK_ESCALATION"`` – soft-block threshold reached after N attempts.
    soft_block_mode
        Active :class:`SoftBlockMode` value string (set for SOFT_BLOCK_ESCALATION).
    threshold_str
        Human-readable threshold description (e.g. ``"total 3/3"``).
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
        super().__init__(msg)

    def dict(self) -> dict:
        """Return a JSON-serialisable dict of all structured fields."""
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


__all__ = [
    "ConstraintSeverity",
    "SoftBlockMode",
    "ConstraintViolation",
    "ConstraintViolationError",
]
