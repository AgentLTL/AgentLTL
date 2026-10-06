"""
agentltl/enforcer.py – the runtime enforcer every harness drives.

An :class:`Enforcer` holds the constraints and what has happened so far. For each tool
call the agent proposes, :meth:`Enforcer.check` returns a :class:`Decision`; when the
call runs, :meth:`Enforcer.record_completed` adds it to the trace::

    enforcer = Enforcer(constraints, {"no-force-push": ConstraintSeverity.BLOCK_AND_WARN})
    decision = enforcer.check("git_push", {"force": True})
    if decision.allowed:
        result = run(...)
        enforcer.record_completed("git_push", {"force": True}, result=result)
    else:
        tell_the_agent(decision.feedback)       # decision.action: warn, block, ask, ...

Each call is judged in the five-valued partial-trace semantics of
:mod:`agentltl._partial`, and refused only for what it newly breaks.

What a refusal does depends on the constraint's severity, and the strongest one broken
decides:

========================  ========  =====================================================
Severity                  action    The call is refused, and ...
========================  ========  =====================================================
``HARD_STOP``             stop      the run stops (with ``latch_stop``: every later call
                                    is refused until :meth:`Enforcer.resume`)
``PERSISTENT_BLOCK``      block     the agent must do something else
``ASK``                   ask       a human decides; the agent can't override
``SOFT_BLOCK``            retry     after ``max_soft_attempts`` it escalates to
                                    ``escalate_to`` (HARD_STOP, or ASK)
``BLOCK_AND_WARN``        warn      the agent may insist by repeating the exact call
``TOLERATE``              allow     the call runs; the violation is in ``notes``
========================  ========  =====================================================

The state (trace, counters, override pointer) round-trips through
:meth:`Enforcer.to_state` / :meth:`Enforcer.from_state`, so a harness that runs one
process per call can carry it between calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ._evaluator import LTLEvaluator
from ._partial import caused_elsewhere
from ._trace import Trace
from .enforcement import (
    SEVERITY_STRENGTH,
    ConstraintSeverity,
    ConstraintViolation,
    SoftBlockMode,
)

logger = logging.getLogger(__name__)
_EVALUATOR = LTLEvaluator()

Violation = ConstraintViolation

ACTIONS: Dict[ConstraintSeverity, str] = {
    ConstraintSeverity.HARD_STOP: "stop",
    ConstraintSeverity.PERSISTENT_BLOCK: "block",
    ConstraintSeverity.ASK: "ask",
    ConstraintSeverity.SOFT_BLOCK: "retry",
    ConstraintSeverity.BLOCK_AND_WARN: "warn",
    ConstraintSeverity.TOLERATE: "allow",
}
STATE_VERSION = 1


@dataclass
class Decision:
    """What to do with a proposed call.

    Attributes:
        action: ``allow``, ``warn``, ``retry``, ``ask``, ``block`` or ``stop``.
        violations: The violations that decided it, the deciding one first, then up to
            ``nudge_max - 1`` others of the same severity.
        notes: Violations of TOLERATE constraints: the call is allowed, the agent may be
            told.
        feedback: The message for the agent (see ``render`` on :class:`Enforcer`).
        escalated: A SOFT_BLOCK constraint ran out of retries; ``action`` is its
            escalation target.
        attempts: For ``retry``: how many times this constraint has refused a call.
        override: The constraint a BLOCK_AND_WARN override let the call past.
        latched: Refused because an earlier ``stop`` is still in force.
        index: For :meth:`Enforcer.check_chain`: which call of the chain decided.
    """

    action: str = "allow"
    violations: List[Violation] = field(default_factory=list)
    notes: List[Violation] = field(default_factory=list)
    feedback: str = ""
    escalated: bool = False
    attempts: int = 0
    override: Optional[str] = None
    latched: bool = False
    index: Optional[int] = None

    @property
    def allowed(self) -> bool:
        return self.action == "allow"

    @property
    def violation(self) -> Optional[Violation]:
        """The deciding violation, if the call was refused."""
        return self.violations[0] if self.violations else None

    @property
    def constraint_name(self) -> Optional[str]:
        return self.violations[0].constraint_name if self.violations else None


def _clip(text: str, limit: int = 220) -> str:
    """Shorten a detail for the list of further violations."""
    t = " ".join(str(text).split())
    return t if len(t) <= limit else t[: limit - 1] + "…"


_TAGS = {"stop": "HARD_STOP", "block": "PERSISTENT_BLOCK", "ask": "ASK",
         "retry": "SOFT_BLOCK", "warn": "BLOCK_AND_WARN"}


def render(decision: Decision, enforcer: "Enforcer") -> str:
    """The default feedback for the agent: the rule, what broke it, how to comply."""
    v = decision.violation
    if v is None:
        return ""
    if decision.latched:
        return (f"[CONSTRAINT HARD-STOP]\nConstraint '{v.constraint_name}' stopped this run. "
                "The tool was NOT executed. No tool call runs until the run is resumed.")
    if decision.action == "stop":
        return (f"[CONSTRAINT HARD-STOP]\nConstraint '{v.constraint_name}' was violated by tool "
                f"call '{v.tool_name}'.\nThe tool was NOT executed.\nDetail: {v.detail}\n"
                "Do NOT attempt this tool call. Choose a different action.")
    if decision.escalated:
        return (f"[CONSTRAINT ESCALATION — {enforcer.soft_block_mode.value} mode]\n"
                f"Constraint '{v.constraint_name}' has been violated too many times.\n"
                f"Tool call '{v.tool_name}' was NOT executed.\nDetail: {v.detail}\n"
                + ("A human has been asked to decide." if decision.action == "ask" else
                   "You must take a fundamentally different approach to satisfy this constraint."))
    closing = {
        "retry": (f"Attempt {decision.attempts}/{enforcer.max_soft_attempts} "
                  f"({enforcer.soft_block_mode.value} mode). "
                  "Please choose a different action to comply with this constraint."),
        "warn": ("This is a warning, not a hard block. If you are sure this call is "
                 "necessary, you may override it by re-issuing the EXACT same call "
                 "(same tool name and same arguments) as your very next action; "
                 "otherwise choose a different action to comply with the constraint."),
        "block": ("This block cannot be overridden: repeating the exact same call will "
                  "not execute it. You must choose a different action to comply with the "
                  "constraint."),
        "ask": ("A human has been asked to approve this call. Do not retry it or work "
                "around the constraint."),
    }[decision.action]
    lines = [f"[CONSTRAINT VIOLATION — {_TAGS[decision.action]}]",
             f"Constraint '{v.constraint_name}' was violated by tool call '{v.tool_name}'.",
             "The tool was NOT executed."]
    if v.advice():
        lines.append(f"To comply: {v.advice()}")
    lines.append(f"Detail: {v.detail}")
    extra = decision.violations[1:]
    if extra:
        lines.append(f"This call also violates {len(extra)} other constraint(s); "
                     "fixing them together avoids another rejection:")
        lines += [f"  {i}. {o.advice() or _clip(o.detail)}" for i, o in enumerate(extra, 1)]
    lines.append(closing)
    return "\n".join(lines)


def _canonical(args: Any) -> Any:
    """Arguments in a form where equal calls compare equal (sorted keys, lists)."""
    if isinstance(args, dict):
        return {k: _canonical(args[k]) for k in sorted(args)}
    if isinstance(args, (list, tuple)):
        return [_canonical(v) for v in args]
    return args


class Enforcer:
    """Judges proposed tool calls against FOLTL constraints, call by call.

    Args:
        constraints: The :class:`~agentltl.Constraint` list.
        constraint_severities: Severity per constraint name.
        default_severity: For constraints not in ``constraint_severities``.
        max_soft_attempts: Refusals of a SOFT_BLOCK constraint before it escalates.
        soft_block_mode: How refusals are counted: ``cumulative``, ``consecutive``
            (reset by an allowed call) or ``hybrid`` (either threshold).
        max_consecutive_soft_attempts: The consecutive threshold (default: the
            cumulative one).
        escalate_to: What a SOFT_BLOCK escalates to: HARD_STOP (the run stops) or ASK (a
            human decides; the constraint's count starts again).
        latch_stop: After a HARD_STOP, refuse every call until :meth:`resume`.
        nudge_max: How many violations one refusal may report.
        max_termination_nudges: How many times :meth:`check_termination` may send the
            agent back.
        rank: Orders the further violations reported with a refusal (a key function on
            :class:`Violation`; default: those with advice first).
        render: Builds the agent's feedback from a :class:`Decision` (default:
            :func:`render`).
    """

    def __init__(
        self,
        constraints: Optional[Sequence[Any]] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
        default_severity: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
        max_soft_attempts: int = 3,
        soft_block_mode: Any = "cumulative",
        max_consecutive_soft_attempts: Optional[int] = None,
        escalate_to: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
        latch_stop: bool = False,
        nudge_max: int = 1,
        max_termination_nudges: int = 0,
        rank: Optional[Callable[[Violation], Any]] = None,
        render: Optional[Callable[[Decision, "Enforcer"], str]] = None,
    ) -> None:
        self._constraints: List[Any] = list(constraints or [])
        self._severities: Dict[str, ConstraintSeverity] = dict(constraint_severities or {})
        self._default_severity = default_severity
        self.max_soft_attempts = max_soft_attempts
        self.soft_block_mode = SoftBlockMode(soft_block_mode)
        self.max_consecutive = (max_consecutive_soft_attempts
                                if max_consecutive_soft_attempts is not None else max_soft_attempts)
        if escalate_to not in (ConstraintSeverity.HARD_STOP, ConstraintSeverity.ASK):
            raise ValueError("escalate_to must be HARD_STOP or ASK")
        self.escalate_to = escalate_to
        self.latch_stop = latch_stop
        self._nudge_max = max(1, int(nudge_max))
        self._max_termination_nudges = max(0, int(max_termination_nudges))
        self._rank = rank or (lambda v: 0 if v.advice() else 1)
        self._render = render or globals()["render"]
        self.reset()

    # ── State ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Forget the run: trace, counters and logs. Constraints and settings stay."""
        self._completed_tool_calls: List[Dict[str, Any]] = []
        self._soft_block_counts: Dict[str, int] = {}
        self._consecutive_soft_block_counts: Dict[str, int] = {}
        self._block_and_warn_counts: Dict[str, int] = {}
        self._persistent_block_counts: Dict[str, int] = {}
        self._soft_blocked_calls: List[Dict[str, Any]] = []
        self._constraint_violations: List[Dict[str, Any]] = []
        self._block_and_warn_overrides: List[Dict[str, Any]] = []
        self._last_blocked_call: Optional[Dict[str, Any]] = None
        self._blocked_chain: Optional[List[Any]] = None
        self._current_generation: int = 0
        self._generation_key: Any = None
        self._termination_nudges = 0
        self._termination_nudged_names: List[str] = []
        self._constraint_checks = 0
        self._already_violated = 0
        self._run_status = "completed"
        self._stopped_by: Optional[str] = None
        self._latched: Optional[Dict[str, Any]] = None
        self._prefix_cache: Dict[int, Any] = {}
        self._prefix_cache_len = -1
        self._prefix_trace: Optional[Trace] = None

    _STATE_FIELDS = (
        "_completed_tool_calls", "_soft_block_counts", "_consecutive_soft_block_counts",
        "_block_and_warn_counts", "_persistent_block_counts", "_soft_blocked_calls",
        "_constraint_violations", "_block_and_warn_overrides", "_last_blocked_call",
        "_blocked_chain", "_current_generation", "_termination_nudges",
        "_termination_nudged_names", "_constraint_checks", "_already_violated",
        "_run_status", "_stopped_by", "_latched",
    )
    _LOGS = ("_soft_blocked_calls", "_constraint_violations", "_block_and_warn_overrides")

    def to_state(self, *, max_trace: Optional[int] = None,
                 max_log: Optional[int] = None) -> Dict[str, Any]:
        """Everything about the run so far, as a JSON-serialisable dict.

        Args:
            max_trace: Keep only the last N calls of the trace.
            max_log: Keep only the last N entries of each log.
        """
        state: Dict[str, Any] = {"version": STATE_VERSION}
        for name in self._STATE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, list):
                limit = max_trace if name == "_completed_tool_calls" else (
                    max_log if name in self._LOGS else None)
                value = list(value[-limit:] if limit else value)
            elif isinstance(value, dict):
                value = dict(value)
            state[name.lstrip("_")] = value
        state["generation_key"] = self._generation_key if _is_json(self._generation_key) else None
        return state

    def from_state(self, state: Optional[Dict[str, Any]]) -> "Enforcer":
        """Continue a run saved with :meth:`to_state`. Unknown keys are ignored."""
        self.reset()
        for name in self._STATE_FIELDS:
            key = name.lstrip("_")
            if state and key in state and state[key] is not None:
                value = state[key]
                setattr(self, name, list(value) if isinstance(value, list) else
                        dict(value) if isinstance(value, dict) else value)
        if state:
            self._generation_key = state.get("generation_key")
        return self

    @property
    def trace(self) -> List[Dict[str, Any]]:
        """The calls that ran, in order (the list itself: append with record_*)."""
        return self._completed_tool_calls

    @property
    def constraints(self) -> List[Any]:
        return list(self._constraints)

    @property
    def has_constraints(self) -> bool:
        return bool(self._constraints)

    def set_constraints(self, constraints: Sequence[Any],
                        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None
                        ) -> None:
        self._constraints = list(constraints)
        if constraint_severities is not None:
            self._severities = dict(constraint_severities)
        self._prefix_cache_len = -1

    def severity(self, name: str) -> ConstraintSeverity:
        return self._severities.get(name, self._default_severity)

    # ── Generations ──────────────────────────────────────────────────────────

    def begin_generation(self, key: Any = None) -> None:
        """Start a new model generation (one completion).

        A BLOCK_AND_WARN override needs the identical call in a *later* generation, so
        identical calls issued in parallel by one completion don't count as insisting.
        A harness that can't call this once per completion passes ``generation=`` to
        :meth:`check` instead (any key identifying the completion).
        """
        self._current_generation += 1
        self._generation_key = key

    def _generation(self, key: Any) -> None:
        if key is not None and key != self._generation_key:
            self.begin_generation(key)

    # ── Checking ─────────────────────────────────────────────────────────────

    def check(self, tool_name: str, tool_args: Optional[Dict[str, Any]] = None,
              step_number: Optional[int] = None, *, generation: Any = None) -> Decision:
        """Judge a proposed call. Nothing is recorded as run: see :meth:`record_completed`."""
        self._generation(generation)
        tool_args = dict(tool_args or {})
        step = step_number if step_number is not None else len(self._completed_tool_calls) + 1
        if self._latched is not None:
            v = Violation(**self._latched)
            d = Decision("stop", [v], latched=True)
            d.feedback = self._render(d, self)
            return d

        override = self._insisted(tool_name, tool_args, step)
        if override is not None:
            return Decision("allow", override=override)

        violations = self._violations(tool_name, tool_args, step)
        notes = [v for v in violations if v.severity == ConstraintSeverity.TOLERATE.value]
        for v in notes:
            logger.warning("[CONSTRAINT TOLERATE] %s violated by '%s' at step %d: %s",
                           v.constraint_name, tool_name, step, v.detail)
            self._constraint_violations.append(_record(v))
        blocking = [v for v in violations if v.severity != ConstraintSeverity.TOLERATE.value]
        if not blocking:
            if self.soft_block_mode in (SoftBlockMode.CONSECUTIVE, SoftBlockMode.HYBRID):
                self._consecutive_soft_block_counts.clear()
            # The agent did something else: an override must be the very next call.
            self._last_blocked_call = None
            return Decision("allow", notes=notes)

        strongest = min(SEVERITY_STRENGTH.index(ConstraintSeverity(v.severity)) for v in blocking)
        severity = SEVERITY_STRENGTH[strongest]
        same = [v for v in blocking if v.severity == severity.value]
        primary = same[0]
        extra = sorted(same[1:], key=self._rank)[: self._nudge_max - 1]
        decision = self._refuse(severity, primary, tool_name, tool_args, step)
        decision.violations = [primary] + extra
        decision.notes = notes
        decision.feedback = self._render(decision, self)
        return decision

    def check_chain(self, calls: Sequence[Tuple[str, Dict[str, Any]]],
                    step_number: Optional[int] = None, *, generation: Any = None,
                    chain_key: Any = None) -> Decision:
        """Judge calls that run together (a shell command line), all or nothing.

        Each call is checked with the earlier calls of the chain already in the trace.
        The first refusal decides (``Decision.index`` says which call); nothing is
        recorded. Insisting on a BLOCK_AND_WARN means re-issuing the same chain.

        Args:
            calls: ``(tool_name, tool_args)`` pairs in execution order.
            chain_key: What identifies the chain when re-issued (default: the calls).
        """
        self._generation(generation)
        key = _canonical([list(c) for c in calls]) if chain_key is None else chain_key
        retry = key == self._blocked_chain
        start = len(self._completed_tool_calls)
        notes: List[Violation] = []
        overrides: List[str] = []
        try:
            for i, (name, args) in enumerate(calls):
                pointer = self._last_blocked_call
                # Enforcer.check, not self.check: a subclass may give check() another
                # return type (ConstraintEnforcer) or expand the call (shell tools).
                decision = Enforcer.check(self, name, args, step_number)
                if not decision.allowed:
                    self._blocked_chain = key
                    decision.index = i
                    decision.notes = notes + decision.notes
                    return decision
                notes += decision.notes
                if decision.override:
                    overrides.append(decision.override)
                # An allowed call clears the warn pointer. When the agent re-issues the
                # same chain, keep it until the call that was refused is reached.
                if retry and pointer is not None and self._last_blocked_call is None \
                        and not decision.override:
                    self._last_blocked_call = pointer
                self._completed_tool_calls.append(
                    {"tool_name": name, "arguments": dict(args or {})})
        finally:
            del self._completed_tool_calls[start:]
            self._prefix_cache_len = -1
        self._blocked_chain = None
        return Decision("allow", notes=notes, override=overrides[0] if overrides else None)

    def _insisted(self, tool_name: str, tool_args: Dict[str, Any], step: int) -> Optional[str]:
        """The constraint a BLOCK_AND_WARN override lets this call past, if any."""
        last = self._last_blocked_call
        if (last is None or last["tool_name"] != tool_name
                or last["tool_args_canonical"] != _canonical(tool_args)
                or last["set_in_generation"] == self._current_generation):
            return None
        name = last["constraint_name"]
        self._block_and_warn_overrides.append({
            "step": step, "tool_name": tool_name, "tool_args": tool_args,
            "constraint_name": name, "blocked_at_step": last["step_number"]})
        for entry in reversed(self._soft_blocked_calls):
            if (entry.get("mode") == "block_and_warn" and entry.get("tool_name") == tool_name
                    and entry.get("constraint_name") == name
                    and not entry.get("overridden_next_step")):
                entry["overridden_next_step"] = True
                break
        logger.info("[CONSTRAINT BLOCK_AND_WARN override] model insisted on '%s' "
                    "(constraint '%s') — executing.", tool_name, name)
        self._last_blocked_call = None
        return name

    def _refuse(self, severity: ConstraintSeverity, v: Violation, tool_name: str,
                tool_args: Dict[str, Any], step: int) -> Decision:
        name = v.constraint_name
        self._constraint_violations.append(_record(v))
        entry = {"step": step, "constraint_name": name, "tool_name": tool_name,
                 "tool_args": tool_args, "detail": v.detail}
        logger.warning("[CONSTRAINT %s] %s violated by '%s' at step %d: %s",
                       severity.value, name, tool_name, step, v.detail)
        if severity == ConstraintSeverity.HARD_STOP:
            self._stop(v)
            return Decision("stop")
        if severity == ConstraintSeverity.SOFT_BLOCK:
            self._soft_block_counts[name] = self._soft_block_counts.get(name, 0) + 1
            self._consecutive_soft_block_counts[name] = (
                self._consecutive_soft_block_counts.get(name, 0) + 1)
            self._soft_blocked_calls.append(entry)
            total, run = self._soft_block_counts[name], self._consecutive_soft_block_counts[name]
            mode = self.soft_block_mode
            escalate = ((mode == SoftBlockMode.CUMULATIVE and total >= self.max_soft_attempts)
                        or (mode != SoftBlockMode.CUMULATIVE and run >= self.max_consecutive)
                        or (mode == SoftBlockMode.HYBRID and total >= self.max_soft_attempts))
            if not escalate:
                return Decision("retry", attempts=total)
            if self.escalate_to == ConstraintSeverity.ASK:
                self._soft_block_counts[name] = 0
                self._consecutive_soft_block_counts[name] = 0
                return Decision("ask", escalated=True, attempts=total)
            self._stop(v)
            return Decision("stop", escalated=True, attempts=total)
        if severity == ConstraintSeverity.BLOCK_AND_WARN:
            self._block_and_warn_counts[name] = self._block_and_warn_counts.get(name, 0) + 1
            self._last_blocked_call = {
                "tool_name": tool_name, "tool_args_canonical": _canonical(tool_args),
                "constraint_name": name, "step_number": step,
                "set_in_generation": self._current_generation}
            self._soft_blocked_calls.append({**entry, "mode": "block_and_warn",
                                             "overridden_next_step": False})
            return Decision("warn")
        self._persistent_block_counts[name] = self._persistent_block_counts.get(name, 0) + 1
        self._soft_blocked_calls.append({**entry, "mode": "persistent_block"
                                         if severity == ConstraintSeverity.PERSISTENT_BLOCK
                                         else "ask"})
        return Decision("block" if severity == ConstraintSeverity.PERSISTENT_BLOCK else "ask")

    def _stop(self, v: Violation) -> None:
        self._run_status, self._stopped_by = "stopped", v.constraint_name
        if self.latch_stop:
            self._latched = {"constraint_name": v.constraint_name, "severity": v.severity,
                             "step_number": v.step_number, "tool_name": v.tool_name,
                             "tool_args": v.tool_args, "detail": v.detail,
                             "description": v.description, "repair": v.repair}

    def resume(self) -> None:
        """Lift a latched stop: the next call is judged normally again."""
        self._latched = None
        self._run_status, self._stopped_by = "completed", None

    @property
    def stopped(self) -> Optional[str]:
        """The constraint that stopped the run, while a stop is latched."""
        return self._latched["constraint_name"] if self._latched else None

    def _violations(self, tool_name: str, tool_args: Dict[str, Any],
                    step: int) -> List[Violation]:
        if not self._constraints:
            return []
        completed = self._completed_tool_calls
        candidate = {"tool_name": tool_name, "arguments": tool_args}
        metrics = {"tool_calls": completed + [candidate]}
        trace = Trace.from_metrics(metrics)
        self._constraint_checks += len(self._constraints)
        out: List[Violation] = []
        for constraint in self._constraints:
            result = _EVALUATOR.evaluate(constraint.formula, trace, metrics=metrics,
                                         partial_trace=True)
            if result.passed:
                continue
            # Refuse a call only for what IT newly breaks: when every failing instance was
            # already final before this call, the call isn't the cause and refusing it
            # would repair nothing (see agentltl._partial.caused_elsewhere).
            if completed and caused_elsewhere(result, self._prefix_value(constraint)):
                self._already_violated += 1
                continue
            out.append(Violation(
                constraint_name=constraint.name,
                severity=self.severity(constraint.name).value,
                step_number=step, tool_name=tool_name, tool_args=tool_args,
                detail=str(result.detail or "constraint violated"),
                description=str(getattr(constraint, "description", "") or ""),
                repair=str(getattr(constraint, "repair", "") or ""),
                witnesses=list(result.witnesses),
            ))
        return out

    def _prefix_value(self, constraint: Any) -> Any:
        """The constraint on the completed calls alone, cached until the trace changes."""
        n = len(self._completed_tool_calls)
        if self._prefix_cache_len != n:
            self._prefix_cache, self._prefix_cache_len, self._prefix_trace = {}, n, None
        key = id(constraint)
        if key not in self._prefix_cache:
            metrics = {"tool_calls": self._completed_tool_calls}
            if self._prefix_trace is None:
                self._prefix_trace = Trace.from_metrics(metrics)
            self._prefix_cache[key] = _EVALUATOR.evaluate(
                constraint.formula, self._prefix_trace, metrics=metrics, partial_trace=True)
        return self._prefix_cache[key]

    # ── Recording ────────────────────────────────────────────────────────────

    def record_completed(self, tool_name: str, tool_args: Optional[Dict[str, Any]] = None,
                         tool_id: str = "", result: Any = None, *,
                         status: Optional[int] = None) -> None:
        """Add a call that ran to the trace.

        Args:
            status: The call's exit status, when it has one (a shell command).
        """
        entry: Dict[str, Any] = {"tool_name": tool_name, "arguments": dict(tool_args or {}),
                                 "id": tool_id, "result": result}
        if status is not None:
            entry["status"] = status
        self.record_entry(entry)

    def record_entry(self, entry: Dict[str, Any]) -> None:
        """Add a prepared trace entry (``tool_name``, ``arguments``, ...) that ran.

        An entry may carry ``step``, the agent step it ran in: ordering atoms then treat
        calls of the same step as parallel (see ``Before``). The enforcer sets none
        itself: calls of one command line share a generation but run in sequence."""
        self._completed_tool_calls.append(entry)

    # ── Termination ──────────────────────────────────────────────────────────

    def check_termination(self) -> Decision:
        """The agent is about to finish. Is an obligation still unmet?

        Consults the constraints marked ``applies_to_final_answer`` on the trace as
        final (strict semantics), at most ``max_termination_nudges`` times per run.
        Returns ``block`` with feedback to send the agent back, or ``allow``.
        """
        if self._termination_nudges >= self._max_termination_nudges:
            return Decision("allow")
        pending = [c for c in self._constraints if getattr(c, "applies_to_final_answer", False)]
        if not pending:
            return Decision("allow")
        metrics = {"tool_calls": list(self._completed_tool_calls)}
        trace = Trace.from_metrics(metrics)
        unmet = []
        for c in pending:
            result = _EVALUATOR.evaluate(c.formula, trace, metrics=metrics)
            if not result.passed:
                unmet.append(Violation(
                    constraint_name=c.name, severity="termination_nudge", step_number=-1,
                    tool_name=None, tool_args={}, detail=str(result.detail),
                    description=str(getattr(c, "description", "") or ""),
                    repair=str(getattr(c, "repair", "") or "")))
        if not unmet:
            return Decision("allow")
        self._termination_nudges += 1
        shown = unmet[: self._nudge_max]
        for v in shown:
            self._termination_nudged_names.append(v.constraint_name)
            self._constraint_violations.append(_record(v))
        lines = [f"  - {v.constraint_name}"
                 + (f": {v.advice() or v.detail.strip()}" if (v.advice() or v.detail.strip()) else "")
                 for v in shown]
        if len(unmet) > len(shown):
            lines.append(f"  - (and {len(unmet) - len(shown)} more)")
        feedback = ("Do not finish yet. The procedure for this task is not complete -- "
                    "the following required step(s) have not been carried out:\n"
                    + "\n".join(lines)
                    + "\n\nCarry them out now, then give your final answer. If you believe a "
                      "step is genuinely not applicable here, say so explicitly and finish.")
        return Decision("block", unmet, feedback=feedback)

    @property
    def termination_nudges(self) -> int:
        return self._termination_nudges

    @property
    def termination_nudged_names(self) -> List[str]:
        return list(self._termination_nudged_names)

    # ── Reporting ────────────────────────────────────────────────────────────

    def get_constraint_status(self) -> Dict[str, Any]:
        return {
            "status": self._run_status,
            "stopped_by": self._stopped_by,
            "violations": list(self._constraint_violations),
            "constraint_checks": self._constraint_checks,
            "already_violated_skips": self._already_violated,
            "completed_trace": list(self._completed_tool_calls),
            "soft_blocked_calls": list(self._soft_blocked_calls),
            "soft_block_counts": dict(self._soft_block_counts),
            "soft_block_mode": self.soft_block_mode.value,
            "consecutive_soft_block_counts": dict(self._consecutive_soft_block_counts),
            "termination_nudges": self._termination_nudges,
            "termination_nudged_names": list(self._termination_nudged_names),
            "block_and_warn_counts": dict(self._block_and_warn_counts),
            "block_and_warn_overrides": list(self._block_and_warn_overrides),
            "block_and_warn_override_count": len(self._block_and_warn_overrides),
            "persistent_block_counts": dict(self._persistent_block_counts),
        }


def _record(v: Violation) -> Dict[str, Any]:
    return {"constraint_name": v.constraint_name, "severity": v.severity,
            "step_number": v.step_number, "tool_name": v.tool_name, "tool_args": v.tool_args,
            "detail": v.detail, "repair": v.repair}


def _is_json(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


__all__ = ["Enforcer", "Decision", "Violation", "render", "ACTIONS"]
