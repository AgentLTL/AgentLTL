"""Tests for static runtime-safety classification (agentltl.runtime_safety)."""

from __future__ import annotations

import logging

import pytest

from agentltl import (
    AllBefore,
    And,
    BranchCalled,
    Called,
    CalledNTimes,
    Before,
    Constraint,
    ConstraintSeverity,
    Eventually,
    Exists,
    ForAll,
    Formula,
    Globally,
    Implies,
    Not,
    Or,
    Predicate,
    RuntimeSafety,
    Var,
    WithinSteps,
    classify_constraints,
)
from agentltl._ast import AtPosition, Next, Release, Until, WeakUntil
from agentltl.runtime_safety import (
    _AmbiguousSource,
    check_runtime_safety_or_warn,
    classify_runtime_safety,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _domain_fn(_trace, _bindings):
    return ["a", "b"]


def _pred_fn(_trace, _position, _bindings=None):
    return True


# Each row: (id, formula, expected RuntimeSafety)
_CASES: list[tuple[str, Formula, RuntimeSafety]] = [
    ("globally_called", Globally(Called("t")), RuntimeSafety.SAFE),
    ("eventually_called", Eventually(Called("t")), RuntimeSafety.UNSAFE),
    ("called", Called("t"), RuntimeSafety.UNSAFE),
    ("not_called", Not(Called("t")), RuntimeSafety.SAFE),
    ("within_steps", WithinSteps("a", "b", 3), RuntimeSafety.SAFE),
    ("branch_called", BranchCalled("good", "bad"), RuntimeSafety.SAFE),
    ("all_before", AllBefore(["a", "b"], "gate"), RuntimeSafety.SAFE),
    ("count_ge", CalledNTimes("t", 3, ">="), RuntimeSafety.UNSAFE),
    ("count_le", CalledNTimes("t", 3, "<="), RuntimeSafety.SAFE),
    ("count_lt", CalledNTimes("t", 3, "<"), RuntimeSafety.SAFE),
    ("count_eq", CalledNTimes("t", 3, "=="), RuntimeSafety.UNSAFE),
    ("count_gt", CalledNTimes("t", 3, ">"), RuntimeSafety.UNSAFE),
    # n == 0: the "forbidden tool" form. Once t is called, count > 0 forever,
    # so the violation is permanent — runtime-detectable.
    ("count_eq_zero", CalledNTimes("t", 0, "=="), RuntimeSafety.SAFE),
    ("count_ge_zero", CalledNTimes("t", 0, ">="), RuntimeSafety.SAFE),
    ("count_gt_zero", CalledNTimes("t", 0, ">"), RuntimeSafety.UNSAFE),
    ("before", Before("a", "b"), RuntimeSafety.UNSAFE),
    ("not_before", Not(Before("a", "b")), RuntimeSafety.SAFE),
    (
        "forall_within_steps",
        ForAll("x", _domain_fn, WithinSteps("ack", "done", 5)),
        RuntimeSafety.SAFE,
    ),
    (
        "exists_called",
        Exists("x", _domain_fn, Called(Var("x"))),
        RuntimeSafety.UNSAFE,
    ),
    ("predicate_default", Predicate(_pred_fn, "desc"), RuntimeSafety.AMBIGUOUS),
    (
        "predicate_safe",
        Predicate(_pred_fn, "desc", runtime_safe=True),
        RuntimeSafety.SAFE,
    ),
    (
        "predicate_unsafe",
        Predicate(_pred_fn, "desc", runtime_safe=False),
        RuntimeSafety.UNSAFE,
    ),
    (
        "and_safe_unsafe",
        And(Globally(Called("a")), Eventually(Called("b"))),
        RuntimeSafety.SAFE,
    ),
    (
        "or_safe_unsafe",
        Or(Globally(Called("a")), Eventually(Called("b"))),
        RuntimeSafety.UNSAFE,
    ),
    (
        "implies_called_eventually",
        Implies(Called("a"), Eventually(Called("b"))),
        RuntimeSafety.UNSAFE,
    ),
    # Operators not in the original spec but covered for completeness:
    ("next_called", Next(Called("t")), RuntimeSafety.UNSAFE),
    ("next_globally", Next(Globally(Called("t"))), RuntimeSafety.SAFE),
    ("until", Until(Called("a"), Called("b")), RuntimeSafety.UNSAFE),
    (
        "weak_until_left_safe",
        WeakUntil(Globally(Called("a")), Called("b")),
        RuntimeSafety.SAFE,
    ),
    (
        "release_right_safe",
        Release(Called("a"), Globally(Called("b"))),
        RuntimeSafety.SAFE,
    ),
    (
        "at_position_safe",
        AtPosition(2, Globally(Called("a"))),
        RuntimeSafety.SAFE,
    ),
]


_CLASSIFICATIONS: dict[str, RuntimeSafety] = {}


@pytest.mark.parametrize(
    "case_id,formula,expected",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_classification(case_id: str, formula: Formula, expected: RuntimeSafety):
    actual = classify_runtime_safety(formula).safety
    _CLASSIFICATIONS[case_id] = actual
    assert actual == expected, (
        f"{case_id}: expected {expected.value}, got {actual.value} "
        f"for formula {formula!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unknown AST nodes
# ─────────────────────────────────────────────────────────────────────────────


from dataclasses import dataclass


@dataclass(frozen=True)
class _MysteryNode(Formula):
    label: str = "?"


def test_unknown_node_classifies_as_ambiguous_unknown():
    result = classify_runtime_safety(_MysteryNode())
    assert result.safety == RuntimeSafety.AMBIGUOUS
    assert result.source == _AmbiguousSource.UNKNOWN_NODE
    assert result.unknown_node_type == "_MysteryNode"


def test_predicate_default_is_trusted_ambiguous():
    result = classify_runtime_safety(Predicate(_pred_fn, "desc"))
    assert result.safety == RuntimeSafety.AMBIGUOUS
    assert result.source == _AmbiguousSource.TRUSTED_PREDICATE


# ─────────────────────────────────────────────────────────────────────────────
# classify_constraints public helper
# ─────────────────────────────────────────────────────────────────────────────


def test_classify_constraints_without_severities():
    constraints = [
        Constraint("a", Globally(Called("t"))),
        Constraint("b", Eventually(Called("t"))),
    ]
    reports = classify_constraints(constraints)
    assert [r.classification for r in reports] == [
        RuntimeSafety.SAFE,
        RuntimeSafety.UNSAFE,
    ]
    # Without severities every report is informational.
    assert all(r.assigned_severity is None for r in reports)
    assert all(r.compatible for r in reports)
    assert all(r.message is None for r in reports)


def test_classify_constraints_unsafe_hardstop_is_incompatible():
    constraints = [Constraint("liveness", Eventually(Called("done")))]
    reports = classify_constraints(
        constraints,
        {"liveness": ConstraintSeverity.HARD_STOP},
    )
    [r] = reports
    assert r.classification == RuntimeSafety.UNSAFE
    assert r.compatible is False
    assert r.message is not None
    assert "WithinSteps" in r.message
    assert "HARD_STOP" in r.message


def test_classify_constraints_unsafe_tolerate_is_compatible():
    constraints = [Constraint("liveness", Eventually(Called("done")))]
    reports = classify_constraints(
        constraints,
        {"liveness": ConstraintSeverity.TOLERATE},
    )
    assert reports[0].compatible is True
    assert reports[0].message is None


def test_classify_constraints_default_severity_kicks_in():
    constraints = [Constraint("liveness", Eventually(Called("done")))]
    reports = classify_constraints(
        constraints,
        {},  # severity missing for "liveness"
        default_severity=ConstraintSeverity.HARD_STOP,
    )
    assert reports[0].assigned_severity == ConstraintSeverity.HARD_STOP
    assert reports[0].compatible is False


def test_classify_constraints_predicate_default_no_warning():
    constraints = [Constraint("p", Predicate(_pred_fn, "desc"))]
    reports = classify_constraints(
        constraints,
        {"p": ConstraintSeverity.HARD_STOP},
    )
    # Trusted-AMBIGUOUS predicate is compatible with any severity.
    assert reports[0].classification == RuntimeSafety.AMBIGUOUS
    assert reports[0].compatible is True


def test_classify_constraints_predicate_unsafe_override_warns():
    constraints = [
        Constraint("p", Predicate(_pred_fn, "desc", runtime_safe=False)),
    ]
    reports = classify_constraints(
        constraints,
        {"p": ConstraintSeverity.HARD_STOP},
    )
    assert reports[0].classification == RuntimeSafety.UNSAFE
    assert reports[0].compatible is False


def test_classify_constraints_unknown_node_message_names_node_type():
    constraints = [Constraint("m", _MysteryNode())]
    reports = classify_constraints(
        constraints,
        {"m": ConstraintSeverity.HARD_STOP},
    )
    assert reports[0].compatible is False
    assert "_MysteryNode" in reports[0].message


# ─────────────────────────────────────────────────────────────────────────────
# Hook behaviour
# ─────────────────────────────────────────────────────────────────────────────


def _hook_logger() -> logging.Logger:
    return logging.getLogger("agentltl.runtime_safety.test")


def test_hook_warns_in_default_mode(caplog):
    constraints = [Constraint("liveness", Eventually(Called("done")))]
    with caplog.at_level(logging.WARNING, logger="agentltl.runtime_safety.test"):
        check_runtime_safety_or_warn(
            constraints,
            {"liveness": ConstraintSeverity.HARD_STOP},
            ConstraintSeverity.HARD_STOP,
            strict=False,
            logger=_hook_logger(),
        )
    assert any("Runtime-safety mismatch" in r.message for r in caplog.records)


def test_hook_raises_in_strict_mode():
    constraints = [Constraint("liveness", Eventually(Called("done")))]
    with pytest.raises(ValueError, match="Runtime-safety mismatch"):
        check_runtime_safety_or_warn(
            constraints,
            {"liveness": ConstraintSeverity.HARD_STOP},
            ConstraintSeverity.HARD_STOP,
            strict=True,
            logger=_hook_logger(),
        )


def test_hook_skips_final_answer_constraints(caplog):
    constraints = [
        Constraint(
            "liveness",
            Eventually(Called("done")),
            applies_to_final_answer=True,
        )
    ]
    with caplog.at_level(logging.WARNING, logger="agentltl.runtime_safety.test"):
        check_runtime_safety_or_warn(
            constraints,
            {"liveness": ConstraintSeverity.HARD_STOP},
            ConstraintSeverity.HARD_STOP,
            strict=False,
            logger=_hook_logger(),
        )
    assert caplog.records == []


def test_hook_silent_for_safe_constraint(caplog):
    constraints = [Constraint("safety", Globally(Called("end")))]
    with caplog.at_level(logging.WARNING, logger="agentltl.runtime_safety.test"):
        check_runtime_safety_or_warn(
            constraints,
            {"safety": ConstraintSeverity.HARD_STOP},
            ConstraintSeverity.HARD_STOP,
            strict=False,
            logger=_hook_logger(),
        )
    assert caplog.records == []


def test_hook_silent_for_unsafe_with_tolerate(caplog):
    constraints = [Constraint("liveness", Eventually(Called("done")))]
    with caplog.at_level(logging.WARNING, logger="agentltl.runtime_safety.test"):
        check_runtime_safety_or_warn(
            constraints,
            {"liveness": ConstraintSeverity.TOLERATE},
            ConstraintSeverity.HARD_STOP,
            strict=False,
            logger=_hook_logger(),
        )
    assert caplog.records == []


def test_hook_warns_on_unknown_node_with_node_type(caplog):
    constraints = [Constraint("m", _MysteryNode())]
    with caplog.at_level(logging.WARNING, logger="agentltl.runtime_safety.test"):
        check_runtime_safety_or_warn(
            constraints,
            {"m": ConstraintSeverity.HARD_STOP},
            ConstraintSeverity.HARD_STOP,
            strict=False,
            logger=_hook_logger(),
        )
    assert any("_MysteryNode" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: hook wired into the three constructors
# ─────────────────────────────────────────────────────────────────────────────


class _StubBackend:
    """Cheap stand-in for the smolagents/langchain backend so we can exercise
    AgentWithConstraints without bringing up a real model."""

    def __init__(self, *args, **kwargs):
        self._kwargs = kwargs

    def run(self, *args, **kwargs):
        return {"answer": None, "metrics": {}, "error": None}

    def get_constraint_status(self):
        return {}


def _patch_backends(monkeypatch):
    """Make ``AgentWithConstraints`` construct a stub instead of a real backend.

    The runtime-safety hook fires *before* backend construction in
    ``AgentWithConstraints.__init__`` so the stub is enough to verify the
    integration end-to-end without smolagents/langchain pulling in models,
    MCP, etc.

    We inject a synthetic ``agentltl.integrations.smolagents.backend``
    module into ``sys.modules`` so the lazy ``from ... import
    SmolAgentsAgentWithConstraints`` in :class:`AgentWithConstraints`
    resolves without touching the real backend (whose dependency on a
    specific smolagents API may not be satisfied in the test env).
    """
    import sys
    import types

    fake_mod = types.ModuleType("agentltl.integrations.smolagents.backend")
    fake_mod.SmolAgentsAgentWithConstraints = _StubBackend
    monkeypatch.setitem(
        sys.modules,
        "agentltl.integrations.smolagents.backend",
        fake_mod,
    )


def test_agent_with_constraints_strict_raises_on_eventually(monkeypatch):
    from agentltl import AgentWithConstraints

    _patch_backends(monkeypatch)
    constraints = [Constraint("liveness", Eventually(Called("done")))]
    with pytest.raises(ValueError, match="Runtime-safety mismatch"):
        AgentWithConstraints(
            constraints=constraints,
            constraint_severities={"liveness": ConstraintSeverity.HARD_STOP},
            strict_runtime_safety=True,
        )


def test_agent_with_constraints_default_mode_warns_on_eventually(monkeypatch, caplog):
    from agentltl import AgentWithConstraints

    _patch_backends(monkeypatch)
    constraints = [Constraint("liveness", Eventually(Called("done")))]
    with caplog.at_level(logging.WARNING, logger="agentltl.agents"):
        AgentWithConstraints(
            constraints=constraints,
            constraint_severities={"liveness": ConstraintSeverity.HARD_STOP},
        )
    assert any("Runtime-safety mismatch" in r.message for r in caplog.records)


def test_agent_with_constraints_predicate_default_no_warning(monkeypatch, caplog):
    from agentltl import AgentWithConstraints

    _patch_backends(monkeypatch)
    constraints = [Constraint("p", Predicate(_pred_fn, "desc"))]
    with caplog.at_level(logging.WARNING, logger="agentltl.agents"):
        AgentWithConstraints(
            constraints=constraints,
            constraint_severities={"p": ConstraintSeverity.HARD_STOP},
        )
    assert caplog.records == []


def test_agent_with_constraints_predicate_unsafe_override_warns(monkeypatch, caplog):
    from agentltl import AgentWithConstraints

    _patch_backends(monkeypatch)
    constraints = [
        Constraint("p", Predicate(_pred_fn, "desc", runtime_safe=False))
    ]
    with caplog.at_level(logging.WARNING, logger="agentltl.agents"):
        AgentWithConstraints(
            constraints=constraints,
            constraint_severities={"p": ConstraintSeverity.HARD_STOP},
        )
    assert any("Runtime-safety mismatch" in r.message for r in caplog.records)


def test_agent_with_constraints_unknown_node_warns_with_node_type(
    monkeypatch, caplog
):
    from agentltl import AgentWithConstraints

    _patch_backends(monkeypatch)
    constraints = [Constraint("m", _MysteryNode())]
    with caplog.at_level(logging.WARNING, logger="agentltl.agents"):
        AgentWithConstraints(
            constraints=constraints,
            constraint_severities={"m": ConstraintSeverity.HARD_STOP},
        )
    assert any("_MysteryNode" in r.message for r in caplog.records)


def test_agent_with_constraints_safe_constraint_no_warning(monkeypatch, caplog):
    from agentltl import AgentWithConstraints

    _patch_backends(monkeypatch)
    constraints = [Constraint("safety", Globally(Called("end")))]
    with caplog.at_level(logging.WARNING, logger="agentltl.agents"):
        AgentWithConstraints(
            constraints=constraints,
            constraint_severities={"safety": ConstraintSeverity.HARD_STOP},
        )
    assert caplog.records == []


def test_agent_with_constraints_unsafe_tolerate_no_warning(monkeypatch, caplog):
    from agentltl import AgentWithConstraints

    _patch_backends(monkeypatch)
    constraints = [Constraint("liveness", Eventually(Called("done")))]
    with caplog.at_level(logging.WARNING, logger="agentltl.agents"):
        AgentWithConstraints(
            constraints=constraints,
            constraint_severities={"liveness": ConstraintSeverity.TOLERATE},
        )
    assert caplog.records == []


def test_agent_with_constraints_skips_final_answer_constraints(monkeypatch, caplog):
    from agentltl import AgentWithConstraints

    _patch_backends(monkeypatch)
    constraints = [
        Constraint(
            "liveness",
            Eventually(Called("done")),
            applies_to_final_answer=True,
        )
    ]
    with caplog.at_level(logging.WARNING, logger="agentltl.agents"):
        AgentWithConstraints(
            constraints=constraints,
            constraint_severities={"liveness": ConstraintSeverity.HARD_STOP},
        )
    assert caplog.records == []


def test_constraint_enforcement_middleware_strict_raises():
    pytest.importorskip("langchain")
    from agentltl.integrations.langchain.constrained_agent import (
        ConstraintEnforcementMiddleware,
    )

    constraints = [Constraint("liveness", Eventually(Called("done")))]
    with pytest.raises(ValueError, match="Runtime-safety mismatch"):
        ConstraintEnforcementMiddleware(
            constraints=constraints,
            constraint_severities={"liveness": ConstraintSeverity.HARD_STOP},
            strict_runtime_safety=True,
        )


def test_constraint_enforcement_middleware_default_warns(caplog):
    pytest.importorskip("langchain")
    from agentltl.integrations.langchain.constrained_agent import (
        ConstraintEnforcementMiddleware,
    )

    constraints = [Constraint("liveness", Eventually(Called("done")))]
    with caplog.at_level(
        logging.WARNING,
        logger="agentltl.integrations.langchain.constrained_agent",
    ):
        ConstraintEnforcementMiddleware(
            constraints=constraints,
            constraint_severities={"liveness": ConstraintSeverity.HARD_STOP},
        )
    assert any("Runtime-safety mismatch" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer (sanity-check table)
# ─────────────────────────────────────────────────────────────────────────────


def test_zzz_summary():
    """Print a formula → classification table for visual inspection.

    Named with a ``zzz`` prefix so pytest collects it last; relies on the
    parametrised classification tests having populated ``_CLASSIFICATIONS``.
    Run with ``pytest -s`` to see the table on stdout.
    """
    if not _CLASSIFICATIONS:
        pytest.skip("classification cases did not run")
    width = max(len(case_id) for case_id, _, _ in _CASES)
    lines = [
        "",
        "Runtime-safety classification summary",
        "=" * (width + 16),
    ]
    for case_id, formula, _expected in _CASES:
        actual = _CLASSIFICATIONS.get(case_id, "?")
        label = actual.value if isinstance(actual, RuntimeSafety) else str(actual)
        lines.append(f"  {case_id.ljust(width)}  {label:<10}  {formula!s}")
    summary = "\n".join(lines)
    print(summary)
    assert "Runtime-safety classification summary" in summary
