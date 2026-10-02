# -*- coding: utf-8 -*-
"""X (next) on partial traces, the step-local `now` atom, and partial-trace
semantics reaching atoms nested under temporal operators."""
from __future__ import annotations

from agentltl import Constraint, ConstraintSeverity, Now, parse, verify_trace
from agentltl._enforcement_engine import ConstraintEnforcer
from agentltl.runtime_safety import RuntimeSafety, classify_runtime_safety


def passes(formula, names, partial=True):
    trace = {"tool_calls": [{"tool_name": n, "arguments": {}} for n in names]}
    report = verify_trace(trace, [Constraint("c", parse(formula))], partial_trace=partial)
    return report["constraints"][0]["passed"]


ONCE = 'G(now("deploy") -> X(G(!now("deploy"))))'
FOLLOW = 'G(now("x") -> X(now("y")))'


class TestNextOnPartialTraces:
    def test_the_call_being_checked_is_not_refused_for_lack_of_a_successor(self):
        assert passes(ONCE, ["deploy"])
        assert passes(FOLLOW, ["a", "x"])

    def test_the_successor_is_still_checked_once_it_exists(self):
        assert passes(ONCE, ["deploy", "ls"])
        assert not passes(ONCE, ["deploy", "ls", "deploy"])
        assert passes(FOLLOW, ["a", "x", "y"])
        assert not passes(FOLLOW, ["a", "x", "z"])

    def test_finished_traces_keep_strong_next(self):
        assert not passes(FOLLOW, ["a", "x"], partial=False)
        assert passes(FOLLOW, ["a", "x", "y"], partial=False)


class TestNow:
    def test_is_about_the_current_step_only(self):
        assert passes('now("a")', ["a", "b"], partial=False)
        assert not passes('now("b")', ["a", "b"], partial=False)
        # called() holds at every step once the tool appears anywhere; now() does not
        assert not passes('G(called("x") -> X(now("y")))', ["x", "y", "z"], partial=False)
        assert passes('G(now("x") -> X(now("y")))', ["x", "y", "z"], partial=False)

    def test_parses_prints_and_is_runtime_safe(self):
        assert parse('now("a")') == Now("a")
        assert str(Now("a")) == 'now("a")'
        assert classify_runtime_safety(parse(ONCE)).safety == RuntimeSafety.SAFE


class TestPartialTraceIsPassedDown:
    def test_ordering_nested_under_G_gets_trigger_semantics(self):
        rule = 'G(now("push") -> before("test", "push"))'
        assert passes(rule, ["ls"])
        assert passes(rule, ["test", "push"])
        assert not passes(rule, ["ls", "push"])

    def test_negative_positions_stay_strict(self):
        # trigger semantics would make before() "true so far" and its negation false
        assert passes('!before("a", "b")', ["a"]) == passes('!before("a", "b")', ["a"], False)
        assert passes('before("a", "b") -> called("c")', ["a"]) is True


def test_enforcer_allows_the_first_deploy_and_refuses_the_second():
    enforcer = ConstraintEnforcer(
        constraints=[Constraint("once", parse(ONCE))],
        default_severity=ConstraintSeverity.PERSISTENT_BLOCK,
    )
    assert enforcer.check("deploy", {}, 1) == "allow"
    enforcer.record_completed("deploy", {}, "1", "")
    assert enforcer.check("ls", {}, 2) == "allow"
    enforcer.record_completed("ls", {}, "2", "")
    assert enforcer.check("deploy", {}, 3)[0] == "persistent_block"
