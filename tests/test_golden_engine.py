"""Golden runs: the engine's decision at every step, and the score of every complete trace.

Refactors must keep both. An intended change is made by regenerating the file
(AGENTLTL_GOLDEN_UPDATE=1 pytest tests/test_golden_engine.py) and reviewing its diff.
"""

import json
import os
from pathlib import Path

import pytest

from agentltl import (
    After, AllBefore, And, AtPosition, Before, BranchCalled, Called, CalledInOrder,
    CalledNTimes, CalledWith, CalledWithResult, Constraint, ConstraintSeverity,
    ConstraintViolationError, Eventually, Globally, Implies, InstanceBefore, Next, Not, Now, Or,
    Release, Until, WeakUntil, WithinSteps, parse, verify_trace,
)
from agentltl._enforcement_engine import ConstraintEnforcer

GOLDEN = Path(__file__).parent / "golden" / "engine.json"

FORMULAS = {
    "never_a": Globally(Not(Now("a"))),
    "not_called_a": Not(Called("a")),
    "called_a": Called("a"),
    "G_called_a": Globally(Called("a")),
    "before_ab": Before("a", "b"),
    "not_before_ab": Not(Before("a", "b")),
    "after_ab": After("a", "b"),
    "all_before": AllBefore(("a", "b"), "c"),
    "branch": BranchCalled("a", "b"),
    "branch_only": BranchCalled("a"),
    "instance_before": InstanceBefore("a", 2, "b", 1),
    "in_order": CalledInOrder(["a", "b", "c"]),
    "within": WithinSteps("a", "b", 2),
    "at_most_1": CalledNTimes("a", 1, "<="),
    "at_least_2": CalledNTimes("a", 2, ">="),
    "exactly_1": CalledNTimes("a", 1, "=="),
    "called_with": CalledWith("a", {"x": 1}),
    "never_with": Globally(Not(CalledWith("a", {"x": 1}))),
    "with_result": CalledWithResult("a", "ok"),
    "once": parse('G(now("a") -> X(G(!now("a"))))'),
    "respond": parse('G(now("a") -> F(now("b")))'),
    "next": parse('G(now("a") -> X(now("b")))'),
    "eventually_b": Eventually(Now("b")),
    "not_G": Not(Globally(Not(Now("c")))),
    "until": Until(Not(Now("b")), Now("a")),
    "weak_until": WeakUntil(Not(Now("b")), Now("a")),
    "release": Release(Now("a"), Not(Now("b"))),
    "at_pos": AtPosition(2, Now("c")),
    "imp_count": Implies(Called("c"), CalledNTimes("a", 1, ">=")),
    "and_or": And(Or(Called("a"), Called("b")), Not(Called("c"))),
    "parsed_before": parse('G(before("a", "b"))'),
}

A, B, C = ("a", {}), ("b", {}), ("c", {})
A1 = ("a", {"x": 1})
TRACES = {
    "abc": [A, B, C], "bac": [B, A, C], "aab": [A, A, B], "aaa": [A, A, A],
    "cab": [C, A, B], "acb": [A, C, B], "b": [B], "a": [A], "acca": [A, C, C, A],
    "a1b": [A1, B], "bb": [B, B], "abab": [A, B, A, B], "cccab": [C, C, C, A, B],
}
SEVERITIES = {
    "block": ConstraintSeverity.PERSISTENT_BLOCK,
    "warn": ConstraintSeverity.BLOCK_AND_WARN,
    "soft": ConstraintSeverity.SOFT_BLOCK,
    "stop": ConstraintSeverity.HARD_STOP,
}


def run_engine(formula, severity, trace):
    enf = ConstraintEnforcer(constraints=[Constraint("k", formula)], default_severity=severity)
    out, queue = [], list(trace)
    while queue:
        tool, args = queue.pop(0)
        enf.begin_generation()
        try:
            d = enf.check(tool, dict(args), len(out) + 1)
        except ConstraintViolationError as exc:
            out.append("stop" if exc.violation_type == "HARD_STOP" else "escalate")
            break
        if d == "allow":
            out.append("allow")
            enf.record_completed(tool, dict(args), str(len(out)), "ok")
        else:
            out.append(d[0])
            if d[0] == "block_and_warn" and len(out) < 2 * len(trace) + 2:
                queue.insert(0, (tool, args))  # insist once: the override path
    return out


def strict(formula, trace):
    metrics = {"tool_calls": [{"tool_name": t, "arguments": a, "result": "ok"} for t, a in trace]}
    return verify_trace(metrics, [Constraint("k", formula)])["constraints"][0]["passed"]


def results():
    got = {}
    for fname, formula in FORMULAS.items():
        for tname, trace in TRACES.items():
            got[f"strict/{fname}/{tname}"] = strict(formula, trace)
            for sname, sev in SEVERITIES.items():
                got[f"run/{fname}/{tname}/{sname}"] = run_engine(formula, sev, trace)
    return got


def test_golden_engine():
    got = results()
    if os.environ.get("AGENTLTL_GOLDEN_UPDATE"):
        GOLDEN.parent.mkdir(exist_ok=True)
        lines = (f"{json.dumps(k)}: {json.dumps(got[k])}" for k in sorted(got))
        GOLDEN.write_text("{\n" + ",\n".join(lines) + "\n}\n")
        pytest.skip("golden runs rewritten")
    want = json.loads(GOLDEN.read_text())
    changed = {k: (want.get(k), got.get(k)) for k in set(want) | set(got) if want.get(k) != got.get(k)}
    assert not changed, json.dumps(dict(sorted(changed.items())), indent=0)
