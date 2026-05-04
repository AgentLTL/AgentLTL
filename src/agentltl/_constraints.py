"""
agentltl/_constraints.py – Constraint declaration and compliance evaluation.

This module provides:

* :class:`Constraint` – a named, weighted LTL formula that forms one item
  in a checklist of procedure compliance requirements.
* :func:`verify_trace` – evaluate a list of constraints against an agent
  trace and produce per-constraint results plus an aggregate compliance
  score.

Compliance score
----------------
Given constraints K = {κ₁, …, κₙ} and the set V ⊆ K of violated
constraints:

    C(τ, G_P) = 1 − Σ_{κᵢ ∈ V}  w(κᵢ)
                    ─────────────────────
                    Σ_{κᵢ ∈ K}  w(κᵢ)

* C = 1.0 → FULL compliance
* C = 0.0 → complete VIOLATION
* 0 < C < 1 → PARTIAL compliance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ._ast import Formula
from ._evaluator import LTLEvaluator
from ._trace import Trace


# ─────────────────────────────────────────────────────────────────────────────
# Constraint declaration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Constraint:
    """A single compliance constraint.

    Attributes:
        name:        Short, human-readable identifier (also used as dict key).
        formula:     The LTL formula that must hold on a valid trace.
        weight:      Importance weight w(κᵢ).  Higher → more impact on C.
        description: Optional longer explanation shown in reports.
    """
    name: str
    formula: Formula
    weight: float = 1.0
    description: str = ""
    applies_to_final_answer: bool = False
    """When True, this constraint is also evaluated before the final_answer tool
    is executed (online enforcement).  Default False preserves existing behaviour
    where final_answer is always exempt from pre-execution constraint checks."""

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"Constraint weight must be ≥ 0, got {self.weight}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConstraintResult:
    """Result of evaluating a single constraint against a trace."""
    name: str
    weight: float
    passed: bool
    detail: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "passed": self.passed,
            "detail": self.detail,
            "description": self.description,
        }


@dataclass
class ComplianceResult:
    """Aggregate result of evaluating all constraints against a trace."""
    compliance_score: float
    compliance_label: str
    details: str
    tool_sequence: List[str]
    constraints: List[ConstraintResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compliance_score": self.compliance_score,
            "compliance_label": self.compliance_label,
            "details": self.details,
            "tool_sequence": self.tool_sequence,
            "constraints": [c.to_dict() for c in self.constraints],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(results: Sequence[ConstraintResult]) -> float:
    total = sum(r.weight for r in results)
    if total == 0.0:
        return 0.0
    violated = sum(r.weight for r in results if not r.passed)
    return round(1.0 - violated / total, 6)


def _label(score: float) -> str:
    if score == 1.0:
        return "FULL"
    if score == 0.0:
        return "VIOLATION"
    return "PARTIAL"


# ─────────────────────────────────────────────────────────────────────────────
# Core verification API
# ─────────────────────────────────────────────────────────────────────────────

_evaluator = LTLEvaluator()


def verify_trace(
    metrics: Dict[str, Any],
    constraints: Sequence[Constraint],
    *,
    weight_overrides: Optional[Dict[str, float]] = None,
    full_compliance_detail: str = "All constraints satisfied.",
    partial_trace: bool = False,
) -> Dict[str, Any]:
    """Evaluate a list of FOLTL constraints against an agent trace.

    Args:
        metrics:
            The metrics dict from ``Agent.run()["metrics"]``.
            ``metrics["tool_calls"]`` is used to build the trace.
        constraints:
            Ordered sequence of :class:`Constraint` instances.
        weight_overrides:
            Optional mapping of constraint names → custom weights.
        full_compliance_detail:
            Human-readable message returned when all constraints pass.
        partial_trace:
            When ``True``, use trigger semantics for ordering constraints.
            Use during pre-execution checking on growing traces.

    Returns:
        A dict::

            {
              "compliance_score":  float,       # C(τ, G_P) in [0, 1]
              "compliance_label":  str,          # FULL | PARTIAL | VIOLATION
              "details":           str,
              "tool_sequence":     list[str],    # φ(τ)
              "constraints":       list[dict],   # per-constraint records
            }
    """
    overrides = weight_overrides or {}
    trace = Trace.from_metrics(metrics)
    seq = trace.events

    results: List[ConstraintResult] = []
    for c in constraints:
        w = overrides.get(c.name, c.weight)
        eval_result = _evaluator.evaluate(
            c.formula, trace, metrics=metrics, partial_trace=partial_trace
        )
        results.append(ConstraintResult(
            name=c.name,
            weight=w,
            passed=eval_result.passed,
            detail=eval_result.detail,
            description=c.description,
        ))

    score = _compute_score(results)
    label = _label(score)

    if label == "FULL":
        details = full_compliance_detail
    else:
        violated = [r.name for r in results if not r.passed]
        missing: List[str] = []
        for c in constraints:
            from ._ast import Before as _Before, Called as _Called, BranchCalled as _BC
            if isinstance(c.formula, _Before):
                for tool in (c.formula.a, c.formula.b):
                    if tool not in seq and tool not in missing:
                        missing.append(tool)
            elif isinstance(c.formula, _Called):
                if c.formula.tool not in seq and c.formula.tool not in missing:
                    missing.append(c.formula.tool)
            elif isinstance(c.formula, _BC):
                if (c.formula.correct_tool not in seq
                        and c.formula.correct_tool not in missing):
                    missing.append(c.formula.correct_tool)
        parts = [f"Violated: {violated}."]
        if missing:
            parts.append(f"Missing tools: {missing}.")
        details = " ".join(parts)

    compliance = ComplianceResult(
        compliance_score=score,
        compliance_label=label,
        details=details,
        tool_sequence=seq,
        constraints=results,
    )
    return compliance.to_dict()
