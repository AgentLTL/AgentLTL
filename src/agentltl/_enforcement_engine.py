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


def _clip(text: str, limit: int = 220) -> str:
    """Shorten an evaluator detail for the secondary violation list.

    These strings embed the full offending call payload -- several hundred
    characters on a FHIR POST -- so an unclipped list of them is mostly a repeated
    copy of what the agent just sent.
    """
    t = " ".join(str(text).split())
    return t if len(t) <= limit else t[: limit - 1] + "\u2026"

# Return type of check(): the literal "allow", or ("soft_block", feedback), or
# ("block_and_warn", feedback), or ("persistent_block", feedback).
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
        nudge_max: int = 1,
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
        # How many violated constraints one blocked call may report. A single call
        # can violate several at once (a POST with three wrong fields), and telling
        # the model one at a time costs a round trip per field. Capped because the
        # opposite failure is real too: an agent handed a wall of blocks stops
        # treating them as actionable.
        #
        # DEFAULT 1, deliberately: this is a library, so its default must reproduce
        # the behaviour it had before this parameter existed -- one violation per
        # blocked call. Callers that want the batched form opt in (the harness
        # passes 3 via QWEN_BM_NUDGE_MAX), which also keeps it an experiment knob
        # rather than a silent change to every existing consumer.
        self._nudge_max = max(1, int(nudge_max))
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
        # ── BLOCK_AND_WARN state ──
        # Insistence pointer: the most recent BLOCK_AND_WARN-blocked call. The
        # model may override it by re-issuing the byte-identical call in a LATER
        # generation (see begin_generation / check).
        self._block_and_warn_counts: Dict[str, int] = {}
        self._block_and_warn_overrides: List[Dict[str, Any]] = []
        self._last_blocked_call: Optional[Dict[str, Any]] = None
        self._current_generation: int = 0
        # ── PERSISTENT_BLOCK state ──
        # Like BLOCK_AND_WARN but with no override: the call is blocked every time.
        self._persistent_block_counts: Dict[str, int] = {}

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

    def begin_generation(self) -> None:
        """Mark the start of a new LLM generation (one model completion).

        Drives the BLOCK_AND_WARN insistence pointer: a call blocked in
        generation *G* may be overridden only by a byte-identical re-issue in a
        *later* generation, so parallel duplicates within the same completion do
        NOT qualify as an override. The native backend calls this once per chat
        completion, before iterating that completion's tool calls.
        """
        self._current_generation += 1

    @staticmethod
    def _canonical_args(args: Any) -> Any:
        """Recursively canonicalise tool arguments for byte-identical comparison.

        Sorts dict keys; recurses into lists/tuples; leaves scalars alone.
        Used for the BLOCK_AND_WARN insistence-pointer equality check.
        """
        if isinstance(args, dict):
            return {k: ConstraintEnforcer._canonical_args(args[k]) for k in sorted(args.keys())}
        if isinstance(args, (list, tuple)):
            return [ConstraintEnforcer._canonical_args(v) for v in args]
        return args

    def check(self, tool_name: str, tool_args: Dict[str, Any], step_number: int) -> Decision:
        """Evaluate constraints against the prospective call.

        Returns ``"allow"`` if the call may proceed, or ``("soft_block", feedback)``
        if it must be blocked but the run continues. Raises
        :class:`ConstraintViolationError` on HARD_STOP or soft-block escalation.
        """
        # BLOCK_AND_WARN insistence override: if the model is re-issuing the
        # byte-identical call that was just blocked (in a LATER generation —
        # parallel duplicates within the same generation don't qualify), let it
        # through WITHOUT re-evaluating constraints (otherwise BLOCK_AND_WARN
        # would simply re-fire).
        if self._last_blocked_call is not None:
            canonical = self._canonical_args(tool_args)
            if (
                self._last_blocked_call["tool_name"] == tool_name
                and self._last_blocked_call["tool_args_canonical"] == canonical
                and self._last_blocked_call["set_in_generation"] < self._current_generation
            ):
                name = self._last_blocked_call["constraint_name"]
                self._block_and_warn_overrides.append({
                    "step": step_number,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "constraint_name": name,
                    "blocked_at_step": self._last_blocked_call["step_number"],
                })
                for entry in reversed(self._soft_blocked_calls):
                    if (
                        entry.get("mode") == "block_and_warn"
                        and entry.get("tool_name") == tool_name
                        and entry.get("constraint_name") == name
                        and not entry.get("overridden_next_step")
                    ):
                        entry["overridden_next_step"] = True
                        break
                logger.info(
                    "[CONSTRAINT BLOCK_AND_WARN override] model insisted on '%s' "
                    "(constraint '%s') — executing.", tool_name, name,
                )
                self._last_blocked_call = None
                return "allow"

        violations = self._evaluate_constraints(tool_name, tool_args, step_number)

        # The FIRST non-tolerated violation decides the outcome of this call, exactly
        # as before. What is new is that the other violations of the SAME severity
        # ride along as additional advice (up to nudge_max), so a POST with three
        # wrong fields is not rejected three times in a row over three round trips.
        decided = None
        for v in violations:
            severity = ConstraintSeverity(v.severity)
            if severity == ConstraintSeverity.TOLERATE:
                logger.warning(
                    "[CONSTRAINT TOLERATE] %s violated by '%s' at step %d: %s",
                    v.constraint_name, tool_name, step_number, v.detail,
                )
                self._constraint_violations.append(self._violation_to_dict(v))
                continue
            decided = (severity, v)
            break

        if decided is not None:
            severity, v = decided

            if severity == ConstraintSeverity.HARD_STOP:
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

            extra = sorted(
                (o for o in violations
                 if o is not v and ConstraintSeverity(o.severity) == severity),
                key=self._nudge_rank,
            )[: max(0, self._nudge_max - 1)]

            if severity == ConstraintSeverity.SOFT_BLOCK:
                return self._handle_soft_block(
                    v, tool_name, tool_args, step_number, extra)
            if severity == ConstraintSeverity.BLOCK_AND_WARN:
                return self._handle_block_and_warn(
                    v, tool_name, tool_args, step_number, extra)
            if severity == ConstraintSeverity.PERSISTENT_BLOCK:
                return self._handle_persistent_block(
                    v, tool_name, tool_args, step_number, extra)

        # No blocking violation — allow. Reset consecutive counters on success.
        if self._soft_block_mode in (SoftBlockMode.CONSECUTIVE, SoftBlockMode.HYBRID):
            self._consecutive_soft_block_counts.clear()
        # This allowed call was not the insistence-override target, so the model
        # chose to do something else first — invalidate the pointer to honour the
        # "the very next call must be identical" guarantee in the warning.
        if self._last_blocked_call is not None:
            self._last_blocked_call = None
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
            "block_and_warn_counts": dict(self._block_and_warn_counts),
            "block_and_warn_overrides": list(self._block_and_warn_overrides),
            "block_and_warn_override_count": len(self._block_and_warn_overrides),
            "persistent_block_counts": dict(self._persistent_block_counts),
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
                        # Previously dropped here: the spec authors a `rationale`
                        # (-> description) and now a `repair`, both of which reach
                        # ConstraintResult and were then thrown away one line above
                        # this, so the agent only ever saw `detail`.
                        description=str(per_c.get("description") or ""),
                        repair=str(per_c.get("repair") or ""),
                    )
                )
        return violations

    # Which kind of mistake to lead with when a call violates several constraints
    # at once. Keyed on the constraint-name prefix ("bind::records_bp_value"), so it
    # works for both the obligation names and the older layer names, and falls back
    # to the middle of the range for anything unrecognised.
    _NUDGE_PRIORITY = {
        "bind": 0, "abstain": 1, "source": 2, "order": 3, "do": 4, "cover": 5,
        "L2": 0, "L5": 1, "L4": 2, "L3": 3, "L1": 4, "L6": 0,
    }

    @classmethod
    def _nudge_rank(cls, v: ConstraintViolation) -> tuple:
        prefix = str(v.constraint_name).split("::", 1)[0]
        # A violation carrying an actionable repair outranks one that only has a
        # mechanical detail: it is the one the model can do something about.
        return (0 if v.advice() else 1, cls._NUDGE_PRIORITY.get(prefix, 3))

    def _compose(self, tag: str, v: ConstraintViolation, tool_name: str,
                 closing: str, extra: List[ConstraintViolation]) -> str:
        """Build the agent-facing feedback.

        With no authored text and no extra violations this is byte-identical to the
        message this engine produced before `repair` existed -- that equality is the
        backwards-compatibility contract, and it is pinned by a test.
        """
        lines = [
            f"[CONSTRAINT VIOLATION — {tag}]",
            f"Constraint '{v.constraint_name}' was violated by tool call '{tool_name}'.",
            "The tool was NOT executed.",
        ]
        advice = v.advice()
        if advice:
            lines.append(f"To comply: {advice}")
        lines.append(f"Detail: {v.detail}")
        if extra:
            lines.append(
                f"This call also violates {len(extra)} other constraint(s); "
                "fixing them together avoids another rejection:")
            for i, other in enumerate(extra, 1):
                # Advice ALONE when there is any: the evaluator's detail embeds the
                # whole offending payload, so repeating it per violation buries the
                # very instructions this list exists to deliver. The primary
                # violation above keeps its full untruncated detail, and that is
                # what preserves byte-compatibility with the pre-`repair` message.
                lines.append(f"  {i}. {other.advice() or _clip(other.detail)}")
        lines.append(closing)
        return "\n".join(lines)

    def _handle_soft_block(
        self, v: ConstraintViolation, tool_name: str, tool_args: Dict[str, Any],
        step_number: int, extra: Optional[List[ConstraintViolation]] = None
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

        feedback = self._compose(
            "SOFT_BLOCK", v, tool_name,
            f"Attempt {total}/{self._max_soft_attempts} "
            f"({self._soft_block_mode.value} mode). "
            f"Please choose a different action to comply with this constraint.",
            extra or [])
        logger.warning(
            "[CONSTRAINT SOFT_BLOCK] %s violated by '%s' at step %d (attempt %d/%d): %s",
            name, tool_name, step_number, total, self._max_soft_attempts, v.detail,
        )
        return ("soft_block", feedback)

    def _handle_block_and_warn(
        self, v: ConstraintViolation, tool_name: str, tool_args: Dict[str, Any],
        step_number: int, extra: Optional[List[ConstraintViolation]] = None
    ) -> Tuple[str, str]:
        """Block the call and warn, but NEVER escalate. The model may override by
        re-issuing the byte-identical call in a later generation (see check)."""
        name = v.constraint_name
        self._block_and_warn_counts[name] = self._block_and_warn_counts.get(name, 0) + 1
        self._last_blocked_call = {
            "tool_name": tool_name,
            "tool_args_canonical": self._canonical_args(tool_args),
            "constraint_name": name,
            "step_number": step_number,
            "set_in_generation": self._current_generation,
        }
        self._soft_blocked_calls.append(
            {
                "step": step_number,
                "constraint_name": name,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "detail": v.detail,
                "mode": "block_and_warn",
                "overridden_next_step": False,
            }
        )
        self._constraint_violations.append(self._violation_to_dict(v))

        feedback = self._compose(
            "BLOCK_AND_WARN", v, tool_name,
            "This is a warning, not a hard block. If you are sure this call is "
            "necessary, you may override it by re-issuing the EXACT same call "
            "(same tool name and same arguments) as your very next action; "
            "otherwise choose a different action to comply with the constraint.",
            extra or [])
        logger.warning(
            "[CONSTRAINT BLOCK_AND_WARN] %s violated by '%s' at step %d: %s "
            "(model may override by repeating the exact call).",
            name, tool_name, step_number, v.detail,
        )
        return ("block_and_warn", feedback)

    def _handle_persistent_block(
        self, v: ConstraintViolation, tool_name: str, tool_args: Dict[str, Any],
        step_number: int, extra: Optional[List[ConstraintViolation]] = None
    ) -> Tuple[str, str]:
        """Block the call and warn, never escalate, and NEVER allow an override. Unlike
        BLOCK_AND_WARN this sets no insistence pointer, so re-issuing the identical call
        is blocked again every time."""
        name = v.constraint_name
        self._persistent_block_counts[name] = self._persistent_block_counts.get(name, 0) + 1
        self._soft_blocked_calls.append(
            {
                "step": step_number,
                "constraint_name": name,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "detail": v.detail,
                "mode": "persistent_block",
            }
        )
        self._constraint_violations.append(self._violation_to_dict(v))

        feedback = self._compose(
            "PERSISTENT_BLOCK", v, tool_name,
            "This block cannot be overridden: repeating the exact same call will "
            "not execute it. You must choose a different action to comply with the "
            "constraint.",
            extra or [])
        logger.warning(
            "[CONSTRAINT PERSISTENT_BLOCK] %s violated by '%s' at step %d: %s "
            "(block cannot be overridden).",
            name, tool_name, step_number, v.detail,
        )
        return ("persistent_block", feedback)

    @staticmethod
    def _violation_to_dict(v: ConstraintViolation) -> Dict[str, Any]:
        return {
            "constraint_name": v.constraint_name,
            "severity": v.severity,
            "step_number": v.step_number,
            "tool_name": v.tool_name,
            "tool_args": v.tool_args,
            "detail": v.detail,
            # Kept in the record so post-hoc analysis can tell whether the agent was
            # actually given something actionable when this call was blocked, or only
            # a mechanical mismatch string.
            "repair": v.repair,
        }
