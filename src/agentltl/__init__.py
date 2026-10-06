"""
agentltl – FOLTL constraint verification for LLM agent traces.

AgentLTL provides a First-Order Linear Temporal Logic (FOLTL) engine for
verifying that LLM agent tool-call traces comply with procedural constraints.
It supports both post-hoc verification (after a run) and runtime enforcement
(pre-execution checking before each tool call).

Quick start
-----------
::

    from agentltl import (
        Constraint, Before, Called, Eventually, CalledWith,
        Var, ForAll, verify_trace, Trace,
    )

    # Build a trace manually
    trace_metrics = {"tool_calls": [
        {"tool_name": "fetch_data"},
        {"tool_name": "process_data"},
        {"tool_name": "save_results"},
    ]}

    constraints = [
        Constraint("fetch_before_process", Before("fetch_data", "process_data"), weight=1.0),
        Constraint("process_before_save", Before("process_data", "save_results"), weight=1.0),
    ]

    result = verify_trace(trace_metrics, constraints)
    print(result["compliance_score"])   # 1.0
    print(result["compliance_label"])   # FULL

Core types
----------
* :class:`Formula` (and subclasses) – LTL + FOLTL AST nodes
* :class:`Var` – variable placeholder bound by quantifiers
* :class:`ForAll` / :class:`Exists` – first-order quantifiers
* :class:`Constraint` – named, weighted formula
* :class:`Enforcer` / :class:`Decision` – judge each proposed call at run time
* :class:`Trace` – ordered sequence of tool calls
* :class:`LTLEvaluator` – formula interpreter (with quantifier support)
* :func:`verify_trace` – evaluate constraints and compute compliance score
* :func:`parse` – build formulas from strings

See :mod:`agentltl._ast` for the full list of AST node types.
"""

# ── AST nodes (formulas) ────────────────────────────────────────────────────
from ._ast import (
    Formula,
    # Atomic propositions
    Called,
    Now,
    CalledWith,
    CalledWithResult,
    CalledNTimes,
    Before,
    After,
    AllBefore,
    BranchCalled,
    InstanceBefore,
    CalledInOrder,
    WithinSteps,
    # Temporal operators
    Globally,
    Eventually,
    Next,
    Until,
    WeakUntil,
    Release,
    # Logical connectives
    Not,
    And,
    Or,
    Implies,
    # Special
    Predicate,
    AtPosition,
    # Convenience constructors
    called,
    before,
    after,
    eventually,
    always,
    # FOLTL extensions
    Var,
    ForAll,
    Exists,
    substitute,
)

# ── Trace ────────────────────────────────────────────────────────────────────
from ._trace import Trace, ToolCall

# ── Evaluator ────────────────────────────────────────────────────────────────
from ._evaluator import LTLEvaluator, EvalResult

# ── Constraints & scoring ────────────────────────────────────────────────────
from ._constraints import (
    Constraint,
    ConstraintResult,
    ComplianceResult,
    verify_trace,
)

# ── Parser ───────────────────────────────────────────────────────────────────
from ._parser import parse

# ── Enforcement types (framework-agnostic) ────────────────────────────────────
from .enforcement import (
    ConstraintSeverity,
    SEVERITY_STRENGTH,
    SoftBlockMode,
    ConstraintViolation,
    ConstraintViolationError,
)

# ── Runtime enforcement ───────────────────────────────────────────────────────
from .enforcer import Enforcer, Decision, Violation

# ── Static runtime-safety classification ─────────────────────────────────────
from .runtime_safety import (
    RuntimeSafety,
    ClassificationReport,
    classify_constraints,
    classify_runtime_safety,
    reachable_values,
)

# ── Backend-agnostic agent wrappers ───────────────────────────────────────────
from .agents import (
    Agent,
    AgentWithAdditionalTools,
    AgentWithSubAgents,
    AgentWithConstraints,
    MultiTurnAgent,
)

__version__ = "0.1.0"

__all__ = [
    # AST
    "Formula",
    "Called", "Now", "CalledWith", "CalledWithResult", "CalledNTimes",
    "Before", "After", "AllBefore", "BranchCalled",
    "InstanceBefore", "CalledInOrder", "WithinSteps",
    "Globally", "Eventually", "Next", "Until", "WeakUntil", "Release",
    "Not", "And", "Or", "Implies",
    "Predicate", "AtPosition",
    # FOLTL extensions
    "Var", "ForAll", "Exists",
    # Trace
    "Trace", "ToolCall",
    # Evaluator
    "LTLEvaluator", "EvalResult",
    # Constraints
    "Constraint", "verify_trace",
    # Parser
    "parse",
    # Runtime enforcement
    "Enforcer", "Decision", "Violation",
    "ConstraintSeverity", "SEVERITY_STRENGTH", "SoftBlockMode",
    "ConstraintViolation", "ConstraintViolationError",
    # Runtime-safety classification
    "RuntimeSafety", "ClassificationReport", "classify_constraints",
    "classify_runtime_safety", "reachable_values",
    # Agent wrappers
    "Agent", "AgentWithAdditionalTools",
    "AgentWithSubAgents", "AgentWithConstraints",
    "MultiTurnAgent",
]
# Also importable, but not part of the documented API: called, before, after, eventually,
# always (lowercase constructors), substitute, ConstraintResult, ComplianceResult.
