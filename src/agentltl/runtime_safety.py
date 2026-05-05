"""
agentltl/runtime_safety.py – static classification of FOLTL formulas by
runtime-safety.

A FOLTL formula is *runtime-safe* iff a violation can be definitively
detected at some finite trace prefix.  Pairing a runtime-unsafe formula
(an unbounded liveness property such as ``Eventually(Called("done"))``)
with a blocking severity (``HARD_STOP``, ``SOFT_BLOCK``) produces
spurious terminations: the agent is killed for not having satisfied a
property whose witness might still arrive on the next step.

This module performs a recursive walk over the formula AST at constraint-
registration time and labels each formula one of:

* ``SAFE``       – violation is detectable at a finite prefix.
* ``UNSAFE``     – cannot be falsified at any finite prefix.
* ``AMBIGUOUS``  – the classifier cannot decide.

The public API is intentionally small:

* :class:`RuntimeSafety` – the three labels.
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
from typing import Any, Dict, List, Optional

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
    Or,
    Predicate,
    Release,
    Until,
    WeakUntil,
    WithinSteps,
)
from .enforcement import ConstraintSeverity


# ─────────────────────────────────────────────────────────────────────────────
# Public enum
# ─────────────────────────────────────────────────────────────────────────────


class RuntimeSafety(enum.Enum):
    """Static classification of a FOLTL formula's runtime-safety."""

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
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


def classify_runtime_safety(formula: Formula) -> _Classification:
    """Classify *formula* by runtime-safety.

    Returns an internal :class:`_Classification`; callers that only need
    the public label should read ``.safety``.  The internal carrier is
    used by :func:`check_runtime_safety_or_warn` to distinguish a
    trusted-predicate AMBIGUOUS from an unknown-node AMBIGUOUS.
    """
    # ── Atomic propositions ──────────────────────────────────────────────
    if isinstance(formula, Called):
        return _Classification.unsafe()

    if isinstance(formula, CalledNTimes):
        if formula.op in ("<=", "<"):
            return _Classification.safe()
        # ``CalledNTimes(t, 0, "==")`` is the canonical "forbidden tool":
        # counts only grow, so any call to *t* permanently violates the
        # constraint — runtime-detectable.  ``CalledNTimes(t, 0, ">=")``
        # is vacuously true (count is always ≥ 0); its violation set is
        # empty, so it is also SAFE.  ``>`` with n == 0 ("at least one
        # call to t") remains liveness and stays UNSAFE.
        if formula.n == 0 and formula.op in ("==", ">="):
            return _Classification.safe()
        # ">=", ">", "==" with n > 0: count can grow toward the bound,
        # so a runtime checker cannot distinguish "not yet" from "never".
        return _Classification.unsafe()

    if isinstance(formula, Before):
        return _Classification.unsafe()

    if isinstance(formula, After):
        return _Classification.unsafe()

    if isinstance(formula, AllBefore):
        return _Classification.safe()

    if isinstance(formula, BranchCalled):
        return _Classification.safe()

    if isinstance(formula, CalledWith):
        return _Classification.unsafe()

    if isinstance(formula, CalledWithResult):
        return _Classification.unsafe()

    if isinstance(formula, CalledInOrder):
        return _Classification.unsafe()

    if isinstance(formula, InstanceBefore):
        return _Classification.unsafe()

    if isinstance(formula, WithinSteps):
        return _Classification.safe()

    # ── Special / convenience ────────────────────────────────────────────
    if isinstance(formula, Predicate):
        rs = getattr(formula, "runtime_safe", None)
        if rs is True:
            return _Classification.safe()
        if rs is False:
            return _Classification.unsafe()
        return _Classification.ambiguous_predicate()

    if isinstance(formula, AtPosition):
        # Bounded position — safety inherits from the operand.
        return classify_runtime_safety(formula.operand)

    # ── Boolean connectives ──────────────────────────────────────────────
    if isinstance(formula, Not):
        inner = classify_runtime_safety(formula.operand)
        if inner.safety == RuntimeSafety.SAFE:
            return _Classification.unsafe()
        if inner.safety == RuntimeSafety.UNSAFE:
            return _Classification.safe()
        return inner  # AMBIGUOUS preserved with its source

    if isinstance(formula, And):
        left = classify_runtime_safety(formula.left)
        right = classify_runtime_safety(formula.right)
        # SAFE if either operand is SAFE — the violation of the safe side
        # is detectable, and a conjunction fails when any conjunct fails.
        if left.safety == RuntimeSafety.SAFE or right.safety == RuntimeSafety.SAFE:
            return _Classification.safe()
        if left.safety == RuntimeSafety.UNSAFE and right.safety == RuntimeSafety.UNSAFE:
            return _Classification.unsafe()
        # One UNSAFE + one AMBIGUOUS, or both AMBIGUOUS.
        if left.safety == RuntimeSafety.AMBIGUOUS and right.safety == RuntimeSafety.AMBIGUOUS:
            return _ambiguous_combine(left, right)
        return left if left.safety == RuntimeSafety.AMBIGUOUS else right

    if isinstance(formula, Or):
        left = classify_runtime_safety(formula.left)
        right = classify_runtime_safety(formula.right)
        # SAFE only if both operands are SAFE — a disjunction is violated
        # only when both disjuncts are.
        if left.safety == RuntimeSafety.SAFE and right.safety == RuntimeSafety.SAFE:
            return _Classification.safe()
        if left.safety == RuntimeSafety.UNSAFE or right.safety == RuntimeSafety.UNSAFE:
            return _Classification.unsafe()
        if left.safety == RuntimeSafety.AMBIGUOUS and right.safety == RuntimeSafety.AMBIGUOUS:
            return _ambiguous_combine(left, right)
        return left if left.safety == RuntimeSafety.AMBIGUOUS else right

    if isinstance(formula, Implies):
        # p → q  ≡  ¬p ∨ q
        left = classify_runtime_safety(Not(formula.left))
        right = classify_runtime_safety(formula.right)
        if left.safety == RuntimeSafety.SAFE and right.safety == RuntimeSafety.SAFE:
            return _Classification.safe()
        if left.safety == RuntimeSafety.UNSAFE or right.safety == RuntimeSafety.UNSAFE:
            return _Classification.unsafe()
        if left.safety == RuntimeSafety.AMBIGUOUS and right.safety == RuntimeSafety.AMBIGUOUS:
            return _ambiguous_combine(left, right)
        return left if left.safety == RuntimeSafety.AMBIGUOUS else right

    # ── Temporal operators ───────────────────────────────────────────────
    if isinstance(formula, Globally):
        # G is the canonical safety operator: once φ fails at any position
        # the violation is permanent.  We classify G(φ) as SAFE
        # unconditionally — note that this is an over-approximation for
        # the rare nested-liveness case G(F(...)), which is technically
        # unbounded liveness but shows up vanishingly often in agent
        # constraints.
        return _Classification.safe()

    if isinstance(formula, Eventually):
        # Canonical unbounded liveness.  The witness can always appear later.
        return _Classification.unsafe()

    if isinstance(formula, Next):
        # Bounded one step ahead; safety inherits from the operand.
        return classify_runtime_safety(formula.operand)

    if isinstance(formula, Until):
        # φ U ψ requires ψ to eventually hold; ψ might come arbitrarily late.
        return _Classification.unsafe()

    if isinstance(formula, WeakUntil):
        # φ W ψ ≡ G φ ∨ (φ U ψ).  Refutable iff φ fails before ψ; SAFE iff
        # φ is SAFE.
        return classify_runtime_safety(formula.left)

    if isinstance(formula, Release):
        # φ R ψ requires ψ to hold up to (and including) the release point;
        # SAFE iff ψ is SAFE — a violation of ψ before φ appears is permanent.
        return classify_runtime_safety(formula.right)

    # ── Quantifiers ──────────────────────────────────────────────────────
    if isinstance(formula, ForAll):
        # ∀x. body — fails as soon as any binding makes body fail; SAFE iff
        # body is SAFE under any concrete binding (the substituted body is
        # an instance of the same node type so the recursive answer is
        # the right one).
        return classify_runtime_safety(formula.body)

    if isinstance(formula, Exists):
        # ∃x. body — a witness may appear later at any position for any
        # binding.  Even a SAFE body becomes liveness here.
        return _Classification.unsafe()

    # ── Unknown ──────────────────────────────────────────────────────────
    return _Classification.ambiguous_unknown(type(formula).__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility check
# ─────────────────────────────────────────────────────────────────────────────


_BLOCKING_SEVERITIES = (
    ConstraintSeverity.HARD_STOP,
    ConstraintSeverity.SOFT_BLOCK,
)


def _is_compatible(
    classification: _Classification,
    severity: ConstraintSeverity,
) -> bool:
    """Return True if *classification* + *severity* is a safe pairing.

    The rules mirror the behaviour table in the design document:

    * SAFE — always compatible.
    * UNSAFE — only compatible with TOLERATE.
    * AMBIGUOUS, source=TRUSTED_PREDICATE — always compatible.
    * AMBIGUOUS, source=UNKNOWN_NODE — only compatible with TOLERATE.
    """
    if classification.safety == RuntimeSafety.SAFE:
        return True
    if classification.safety == RuntimeSafety.UNSAFE:
        return severity not in _BLOCKING_SEVERITIES
    # AMBIGUOUS
    if classification.source == _AmbiguousSource.TRUSTED_PREDICATE:
        return True
    return severity not in _BLOCKING_SEVERITIES


# ─────────────────────────────────────────────────────────────────────────────
# Message construction
# ─────────────────────────────────────────────────────────────────────────────


def _suggestion(formula: Formula, classification: _Classification) -> str:
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
