"""
agentltl/runtime_safety.py – static classification of FOLTL formulas by
runtime-safety.

A FOLTL formula is *runtime-safe* iff a violation can be definitively
detected at some finite trace prefix.  Pairing a runtime-unsafe formula
(an unbounded liveness property such as ``Eventually(Called("done"))``)
with a blocking severity (``HARD_STOP``, ``SOFT_BLOCK``,
``PERSISTENT_BLOCK``) produces
spurious terminations: the agent is killed for not having satisfied a
property whose witness might still arrive on the next step.

This module performs a recursive walk over the formula AST at constraint-
registration time and labels each formula one of:

* ``SAFE``       – it only fails finally: a refusal always points at a real violation.
* ``UNSAFE``     – it can fail presumptively (an obligation not met yet), so blocking
                   refuses every call until the obligation is met.
* ``INERT``      – it cannot fail before the run ends; check it at termination.
* ``AMBIGUOUS``  – the classifier cannot decide.

The labels come from :func:`reachable_values`, the set of values a formula can take in
the five-valued partial-trace semantics of :mod:`agentltl._partial`, so they agree with
what the enforcer does.

The public API is intentionally small:

* :class:`RuntimeSafety` – the labels.
* :class:`ClassificationReport` – per-constraint result.
* :func:`classify_constraints` – inspect a list of constraints offline.

Internally we also distinguish two kinds of ``AMBIGUOUS``:

* *trusted-predicate*  – a user-authored :class:`Predicate` with no
  ``runtime_safe`` override.  The user knows what their predicate does;
  the framework defers to them.
* *unknown-node*       – an AST node the classifier does not recognise.
  This means the classifier is out of date.  We never silently treat
  unknown nodes as safe.

This distinction is not exposed in :class:`RuntimeSafety` but drives the
registration-time warn / raise hook in :func:`check_runtime_safety_or_warn`.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from ._ast import (
    AllBefore,
    And,
    AtPosition,
    After,
    Before,
    BranchCalled,
    Called,
    CalledInOrder,
    CalledNTimes,
    CalledWith,
    CalledWithResult,
    Eventually,
    Exists,
    ForAll,
    Formula,
    Globally,
    Implies,
    InstanceBefore,
    Next,
    Not,
    Now,
    Or,
    Predicate,
    Release,
    Until,
    WeakUntil,
    WithinSteps,
)
from ._partial import FALSE, PENDING, PFALSE, PTRUE, TRUE
from .enforcement import ConstraintSeverity


# ─────────────────────────────────────────────────────────────────────────────
# Public enum
# ─────────────────────────────────────────────────────────────────────────────


class RuntimeSafety(enum.Enum):
    """Static classification of a FOLTL formula's runtime-safety."""

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    INERT = "INERT"
    AMBIGUOUS = "AMBIGUOUS"


# ─────────────────────────────────────────────────────────────────────────────
# Internal carrier
# ─────────────────────────────────────────────────────────────────────────────


class _AmbiguousSource(enum.Enum):
    """Why a formula was classified AMBIGUOUS."""

    TRUSTED_PREDICATE = "TRUSTED_PREDICATE"
    UNKNOWN_NODE = "UNKNOWN_NODE"


@dataclass(frozen=True)
class _Classification:
    """Internal result of the recursive classifier.

    Public callers only see ``safety``; ``source`` and ``unknown_node_type``
    drive the warn / raise hook so it can produce a useful message and so
    that ``Predicate(...)`` without an override does not warn while
    unknown AST nodes do.
    """

    safety: RuntimeSafety
    source: Optional[_AmbiguousSource] = None
    unknown_node_type: Optional[str] = None

    @staticmethod
    def safe() -> "_Classification":
        return _Classification(RuntimeSafety.SAFE)

    @staticmethod
    def unsafe() -> "_Classification":
        return _Classification(RuntimeSafety.UNSAFE)

    @staticmethod
    def ambiguous_predicate() -> "_Classification":
        return _Classification(
            RuntimeSafety.AMBIGUOUS,
            source=_AmbiguousSource.TRUSTED_PREDICATE,
        )

    @staticmethod
    def ambiguous_unknown(node_type: str) -> "_Classification":
        return _Classification(
            RuntimeSafety.AMBIGUOUS,
            source=_AmbiguousSource.UNKNOWN_NODE,
            unknown_node_type=node_type,
        )


def _ambiguous_combine(
    a: _Classification, b: _Classification
) -> _Classification:
    """Combine two AMBIGUOUS results, preferring UNKNOWN_NODE over TRUSTED."""
    if a.source == _AmbiguousSource.UNKNOWN_NODE:
        return a
    if b.source == _AmbiguousSource.UNKNOWN_NODE:
        return b
    return a


# ─────────────────────────────────────────────────────────────────────────────
# Recursive AST walk
# ─────────────────────────────────────────────────────────────────────────────


_ALL = frozenset({FALSE, PFALSE, PENDING, PTRUE, TRUE})


def _pairs(op, xs, ys):
    return frozenset(op(x, y) for x in xs for y in ys)


def reachable_values(formula: Formula) -> Tuple[FrozenSet[int], Optional[_Classification]]:
    """The values *formula* can take on a partial trace (see :mod:`agentltl._partial`).

    An abstract interpretation over the same algebra the evaluator uses, so the
    classification below agrees with what the enforcer does by construction. The second
    item is an AMBIGUOUS classification when the formula holds a predicate that declared
    nothing, or a node this function doesn't know.
    """
    f = formula
    if isinstance(f, Called) or isinstance(f, CalledWith):
        return frozenset({PFALSE, TRUE}), None
    if isinstance(f, Now):
        return frozenset({FALSE, PENDING, TRUE}), None
    if isinstance(f, CalledWithResult):
        return frozenset({PFALSE, PENDING, TRUE}), None
    if isinstance(f, CalledNTimes):
        if f.op in ("<=", "<"):
            return frozenset({PTRUE, FALSE} if f.n > 0 or f.op == "<=" else {FALSE}), None
        if f.op in (">=", ">"):
            return frozenset({TRUE} if (f.op == ">=" and f.n <= 0) else {PFALSE, TRUE}), None
        return frozenset({PTRUE, FALSE} if f.n <= 0 else {PFALSE, PTRUE, FALSE}), None
    if isinstance(f, (Before, AllBefore, InstanceBefore, WithinSteps)):
        return frozenset({FALSE, PENDING, TRUE}), None
    if isinstance(f, After):
        return frozenset({FALSE, PFALSE, TRUE}), None
    if isinstance(f, BranchCalled):
        return frozenset({FALSE, PENDING, PTRUE} if f.wrong_tool else {PENDING, TRUE}), None
    if isinstance(f, CalledInOrder):
        return frozenset({PENDING, TRUE}), None
    if isinstance(f, Predicate):
        rs = getattr(f, "runtime_safe", None)
        if rs is True:
            return frozenset({FALSE, PTRUE, TRUE}), None
        if rs is False:
            return frozenset({PFALSE, PTRUE}), None
        return _ALL, _Classification.ambiguous_predicate()
    if isinstance(f, AtPosition):
        if f.index < 0:
            return frozenset({FALSE}), None
        inner, amb = reachable_values(f.operand)
        return inner | {PENDING}, amb
    if isinstance(f, Not):
        inner, amb = reachable_values(f.operand)
        return frozenset(TRUE - v for v in inner), amb
    if isinstance(f, (And, Or, Implies, Until, WeakUntil, Release)):
        left, amb_l = reachable_values(f.left)
        right, amb_r = reachable_values(f.right)
        amb = _ambiguous_combine(amb_l, amb_r) if amb_l and amb_r else (amb_l or amb_r)
        if isinstance(f, And):
            return _pairs(min, left, right), amb
        if isinstance(f, Or):
            return _pairs(max, left, right), amb
        if isinstance(f, Implies):
            return _pairs(max, frozenset(TRUE - v for v in left), right), amb
        if isinstance(f, Release):
            left, right = frozenset(TRUE - v for v in left), frozenset(TRUE - v for v in right)
        # until: max(min(ψ_i, φ_0..φ_i-1) ..., min(φ_0..φ_n-1, PENDING))
        hits = _pairs(min, right, left | {TRUE}) | {FALSE}
        tails = frozenset(min(v, PENDING) for v in left) | {PENDING}
        values = _pairs(max, hits, tails)
        if isinstance(f, Release):
            values = frozenset(TRUE - v for v in values)
        return values, amb
    if isinstance(f, Globally):
        inner, amb = reachable_values(f.operand)
        return frozenset(min(v, PENDING) for v in inner) | {PENDING}, amb
    if isinstance(f, Eventually):
        inner, amb = reachable_values(f.operand)
        return frozenset(max(v, PENDING) for v in inner) | {PENDING}, amb
    if isinstance(f, Next):
        inner, amb = reachable_values(f.operand)
        return inner | {PENDING}, amb
    if isinstance(f, ForAll):
        inner, amb = reachable_values(f.body)
        return frozenset(min(v, PENDING) for v in inner) | {PENDING}, amb
    if isinstance(f, Exists):
        inner, amb = reachable_values(f.body)
        return frozenset(max(v, PFALSE) for v in inner) | {PFALSE}, amb
    return _ALL, _Classification.ambiguous_unknown(type(f).__name__)


def classify_runtime_safety(formula: Formula) -> _Classification:
    """Classify *formula* by what refusing a call for it would mean.

    * SAFE: it can only fail finally (``FALSE``): a refusal always points at a real
      violation, made by the call refused.
    * UNSAFE: it can fail *presumptively* (``PFALSE``, an obligation not met yet), so a
      blocking severity refuses calls until the obligation is met.
    * INERT: it never fails before the run ends (``F``, ``in_order``): a blocking
      severity never fires. Check it at termination instead.
    * AMBIGUOUS: a predicate that declared nothing, or an unknown node.

    Returns an internal :class:`_Classification`; callers that only need the public
    label read ``.safety``.
    """
    values, ambiguous = reachable_values(formula)
    if ambiguous is not None:
        return ambiguous
    if PFALSE in values:
        return _Classification.unsafe()
    if FALSE in values:
        return _Classification.safe()
    return _Classification(RuntimeSafety.INERT)


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility check
# ─────────────────────────────────────────────────────────────────────────────


_BLOCKING_SEVERITIES = (
    ConstraintSeverity.HARD_STOP,
    ConstraintSeverity.SOFT_BLOCK,
    # PERSISTENT_BLOCK permanently blocks the call with no override, so — unlike
    # the overridable BLOCK_AND_WARN — it must not be paired with a constraint
    # that cannot be safely evaluated at runtime.
    ConstraintSeverity.PERSISTENT_BLOCK,
)


def _is_compatible(
    classification: _Classification,
    severity: ConstraintSeverity,
) -> bool:
    """Return True if *classification* + *severity* is a safe pairing.

    The rules mirror the behaviour table in the design document:

    * SAFE — always compatible.
    * UNSAFE — only compatible with TOLERATE.
    * INERT — never fires mid-run, so a blocking severity is a mistake: TOLERATE only.
    * AMBIGUOUS, source=TRUSTED_PREDICATE — always compatible.
    * AMBIGUOUS, source=UNKNOWN_NODE — only compatible with TOLERATE.
    """
    if classification.safety == RuntimeSafety.SAFE:
        return True
    if classification.safety in (RuntimeSafety.UNSAFE, RuntimeSafety.INERT):
        return severity not in _BLOCKING_SEVERITIES
    # AMBIGUOUS
    if classification.source == _AmbiguousSource.TRUSTED_PREDICATE:
        return True
    return severity not in _BLOCKING_SEVERITIES


# ─────────────────────────────────────────────────────────────────────────────
# Message construction
# ─────────────────────────────────────────────────────────────────────────────


def _suggestion(formula: Formula, classification: _Classification) -> str:
    if classification.safety == RuntimeSafety.INERT:
        return (
            "It cannot fail before the run ends, so a blocking severity never fires. "
            "Check it when the agent finishes (applies_to_final_answer=True with "
            "max_termination_nudges), or rewrite it as a bounded property using "
            "WithinSteps(tool_a, tool_b, n)."
        )
    if classification.safety == RuntimeSafety.UNSAFE:
        if isinstance(formula, Eventually):
            return (
                "Either change the severity to TOLERATE, or rewrite as a "
                "bounded property using WithinSteps(tool_a, tool_b, n)."
            )
        if isinstance(formula, (Before, After)):
            return (
                "Either change the severity to TOLERATE, or rewrite as a "
                "permanent-violation form: Not(Before(b, a)) for 'a must "
                "precede b', or BranchCalled(correct, wrong) for branching."
            )
        if isinstance(formula, Until):
            return (
                "Either change the severity to TOLERATE, or use WeakUntil "
                "(left-SAFE) or rewrite as a bounded WithinSteps."
            )
        if isinstance(formula, Exists):
            return (
                "Either change the severity to TOLERATE, or replace "
                "Exists(...) with a bounded ForAll(...) over a finite "
                "domain whose body is SAFE."
            )
        return (
            "Either change the severity to TOLERATE, or rewrite as a "
            "safety property whose violation is detectable at finite prefix."
        )
    # AMBIGUOUS
    if classification.source == _AmbiguousSource.UNKNOWN_NODE:
        return (
            f"AST node {classification.unknown_node_type!r} is unknown to "
            f"the runtime-safety classifier. Either change the severity to "
            f"TOLERATE, or extend agentltl.runtime_safety.classify_runtime_safety "
            f"with a rule for this node."
        )
    # TRUSTED_PREDICATE — never reaches here because _is_compatible returns True.
    return ""


def _build_message(
    constraint_name: str,
    formula: Formula,
    classification: _Classification,
    severity: ConstraintSeverity,
) -> str:
    return (
        f"Runtime-safety mismatch: constraint {constraint_name!r} "
        f"({formula!r}) classified {classification.safety.value} but "
        f"assigned severity {severity.value}. "
        f"{_suggestion(formula, classification)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public report
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassificationReport:
    """Per-constraint result of :func:`classify_constraints`."""

    constraint_name: str
    formula_repr: str
    classification: RuntimeSafety
    assigned_severity: Optional[ConstraintSeverity]
    compatible: bool
    message: Optional[str]


def _resolve_severity(
    constraint: Any,
    severities: Optional[Dict[str, ConstraintSeverity]],
    default_severity: ConstraintSeverity,
) -> Optional[ConstraintSeverity]:
    if severities is None:
        return None
    return severities.get(getattr(constraint, "name", None), default_severity)


def classify_constraints(
    constraints: List[Any],
    severities: Optional[Dict[str, ConstraintSeverity]] = None,
    default_severity: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
) -> List[ClassificationReport]:
    """Classify each constraint and report compatibility with its severity.

    *severities* is the same ``{name: ConstraintSeverity}`` mapping that
    :class:`AgentWithConstraints` accepts.  When ``None`` is passed the
    report is purely informational (``assigned_severity`` is ``None`` and
    ``compatible`` is always ``True``).

    *default_severity* is consulted whenever *severities* is provided but
    a particular constraint name is not listed there.
    """
    reports: List[ClassificationReport] = []
    for c in constraints:
        formula = getattr(c, "formula", c)
        name = getattr(c, "name", "<unnamed>")
        classification = classify_runtime_safety(formula)
        severity = _resolve_severity(c, severities, default_severity)
        if severity is None:
            compatible = True
            message: Optional[str] = None
        else:
            compatible = _is_compatible(classification, severity)
            message = (
                None
                if compatible
                else _build_message(name, formula, classification, severity)
            )
        reports.append(
            ClassificationReport(
                constraint_name=name,
                formula_repr=repr(formula),
                classification=classification.safety,
                assigned_severity=severity,
                compatible=compatible,
                message=message,
            )
        )
    return reports


# ─────────────────────────────────────────────────────────────────────────────
# Registration-time hook
# ─────────────────────────────────────────────────────────────────────────────


def check_runtime_safety_or_warn(
    constraints: List[Any],
    severities: Optional[Dict[str, ConstraintSeverity]],
    default_severity: ConstraintSeverity,
    *,
    strict: bool,
    logger: logging.Logger,
) -> None:
    """Warn (or raise) on each runtime-unsafe constraint paired with a
    blocking severity.

    Constraints whose ``applies_to_final_answer`` attribute is truthy are
    skipped — they are evaluated once at the end of the run and cannot
    cause spurious mid-run terminations.

    * ``strict=False`` → emit ``logger.warning`` per mismatch.
    * ``strict=True``  → raise :class:`ValueError` on the first mismatch.
    """
    severities = severities or {}
    for c in constraints:
        if getattr(c, "applies_to_final_answer", False):
            continue
        formula = getattr(c, "formula", c)
        name = getattr(c, "name", "<unnamed>")
        classification = classify_runtime_safety(formula)
        severity = severities.get(name, default_severity)
        if _is_compatible(classification, severity):
            continue
        msg = _build_message(name, formula, classification, severity)
        if strict:
            raise ValueError(msg)
        logger.warning(msg)


__all__ = [
    "RuntimeSafety",
    "ClassificationReport",
    "classify_constraints",
    "classify_runtime_safety",
    "check_runtime_safety_or_warn",
]
