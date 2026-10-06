"""The five-valued partial-trace semantics, the classifier, and marginal causation.

Property tests check that the three agree by construction:

* H1  a final verdict (FALSE or TRUE) never changes as the trace grows, and agrees with
      the strict verdict on the complete trace;
* H2  the evaluator's value is one the classifier said the formula can take;
* H3  a formula classified SAFE never fails presumptively;
* H4  a derived atom has the value of its FOLTL expansion;
* H5  a witness that is FALSE stays FALSE when unrelated calls are appended (what
      makes "refuse only what the call newly breaks" sound).
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from agentltl._ast import (
    CountBefore, Historically, Matches, Once, Previous, Since, ToolPattern,
)
from agentltl import (
    After, AllBefore, And, AtPosition, Before, BranchCalled, Called, CalledInOrder,
    CalledNTimes, CalledWith, Constraint, ConstraintSeverity, ConstraintViolationError,
    Eventually, ForAll, Globally, Implies, InstanceBefore, Next, Not, Now, Or, Predicate,
    Release, RuntimeSafety, Trace, Until, Var, WeakUntil, WithinSteps, parse, verify_trace,
)
from agentltl._enforcement_engine import ConstraintEnforcer
from agentltl._evaluator import LTLEvaluator
from agentltl._partial import FALSE, PENDING, PFALSE, TRUE, caused_elsewhere, is_past_only
from agentltl.runtime_safety import classify_runtime_safety, reachable_values

EV = LTLEvaluator()
TOOLS = ["a", "b", "c"]


def trace_of(calls):
    return Trace.from_metrics({"tool_calls": [{"tool_name": t, "arguments": {"x": x}}
                                              for t, x in calls]})


def value(formula, calls):
    return EV.evaluate(formula, trace_of(calls), partial_trace=True)


def strict(formula, calls):
    return EV.evaluate(formula, trace_of(calls)).passed


tool = st.sampled_from(TOOLS)
calls = st.lists(st.tuples(tool, st.sampled_from([0, 1])), max_size=6)

atoms = st.one_of(
    tool.map(Called), tool.map(Now),
    st.builds(CalledWith, tool, st.fixed_dictionaries({"x": st.sampled_from([0, 1])})),
    st.builds(CalledNTimes, tool, st.integers(0, 2), st.sampled_from(["==", ">=", "<=", ">", "<"])),
    st.builds(Before, tool, tool), st.builds(After, tool, tool),
    st.builds(AllBefore, st.lists(tool, min_size=1, max_size=2).map(tuple), tool),
    st.builds(BranchCalled, tool, st.one_of(st.none(), tool)),
    st.builds(InstanceBefore, tool, st.integers(1, 2), tool, st.integers(1, 2)),
    st.lists(tool, min_size=1, max_size=3).map(CalledInOrder),
    st.builds(WithinSteps, tool, tool, st.integers(0, 2)),
)


def _extend(children):
    return st.one_of(
        children.map(Not), children.map(Globally), children.map(Eventually), children.map(Next),
        st.builds(And, children, children), st.builds(Or, children, children),
        st.builds(Implies, children, children), st.builds(Until, children, children),
        st.builds(WeakUntil, children, children), st.builds(Release, children, children),
        st.builds(AtPosition, st.integers(0, 3), children),
    )


formulas = st.recursive(atoms, _extend, max_leaves=5)


class MaybePattern(ToolPattern):
    """Matches a tool; arguments x=1 make it a possible match only (as with xargs)."""

    def match(self, name, args):
        if name not in self.tools:
            return False
        return None if args.get("x") == 1 else True


past_atoms = st.one_of(
    tool.map(Now),
    st.builds(lambda t, maybe: Matches(MaybePattern(t), maybe), tool, st.booleans()),
)


def _extend_past(children):
    return st.one_of(
        children.map(Not), children.map(Previous), children.map(Once),
        children.map(Historically), st.builds(Since, children, children),
        st.builds(CountBefore, children, st.integers(0, 2), st.sampled_from(["<", ">=", "=="])),
        st.builds(And, children, children), st.builds(Or, children, children),
        st.builds(Implies, children, children),
    )


past_formulas = st.recursive(past_atoms, _extend_past, max_leaves=5)
mixed = st.one_of(formulas, past_formulas, st.builds(Globally, past_formulas),
                  st.builds(And, formulas, past_formulas))
SETTINGS = settings(max_examples=600, deadline=None)


@SETTINGS
@given(mixed, calls, calls)
def test_h1_final_verdicts_are_final(f, prefix, more):
    v = value(f, prefix).value
    if v in (FALSE, TRUE):
        assert value(f, prefix + more).value == v
        assert strict(f, prefix + more) == (v == TRUE)


@SETTINGS
@given(mixed, calls)
def test_h2_values_are_reachable(f, prefix):
    reach, ambiguous = reachable_values(f)
    assert ambiguous is None
    assert value(f, prefix).value in reach


@SETTINGS
@given(mixed, calls)
def test_h3_safe_formulas_only_fail_finally(f, prefix):
    if classify_runtime_safety(f).safety == RuntimeSafety.SAFE:
        assert value(f, prefix).value != PFALSE


def in_order_expansion(tools):
    body = Now(tools[-1])
    for t in reversed(tools[:-1]):
        body = And(Now(t), Next(Eventually(body)))
    return Eventually(body)


@SETTINGS
@given(st.lists(tool, min_size=1, max_size=3), calls)
def test_h4_in_order_is_its_expansion(tools, prefix):
    assert value(CalledInOrder(tools), prefix).value == value(in_order_expansion(tools), prefix).value
    assert strict(CalledInOrder(tools), prefix) == strict(in_order_expansion(tools), prefix)


@SETTINGS
@given(tool, tool, calls)
def test_h4_before_is_its_expansion(a, b, prefix):
    if a != b:
        # first a before first b, and b does happen (the strict meaning)
        expansion = And(Eventually(Now(b)), WeakUntil(Not(Now(b)), And(Now(a), Not(Now(b)))))
        assert value(Before(a, b), prefix).value == value(expansion, prefix).value


@SETTINGS
@given(mixed, calls, st.integers(1, 3))
def test_h5_false_witnesses_stay_false(f, prefix, n):
    # Calls to a tool the formula doesn't name keep every final failure as it was, so
    # an unrelated call is never blamed for one.
    before = {p for p, v in value(f, prefix).witnesses if v == FALSE}
    after = {p for p, v in value(f, prefix + [("z", 0)] * n).witnesses if v == FALSE}
    assert before <= after


@SETTINGS
@given(past_formulas, calls, st.tuples(tool, st.sampled_from([0, 1])))
def test_h6_past_only_rules_are_judged_at_the_new_call_alone(phi, prefix, call):
    """The enforcer's shortcut for G(past-only) refuses exactly what the full evaluation
    with marginal causation would."""
    assert is_past_only(phi)
    rule = Globally(phi)
    full = value(rule, prefix + [call])
    before = value(rule, prefix) if prefix else None
    refused_by_full = not full.passed and not (prefix and caused_elsewhere(full, before))
    refused_by_shortcut = not EV.evaluate(phi, trace_of(prefix + [call]), len(prefix),
                                          partial_trace=True).passed
    assert refused_by_full == refused_by_shortcut


@SETTINGS
@given(past_formulas, calls)
def test_h7_past_operators_agree_with_strict_mode_where_they_are_settled(phi, prefix):
    for pos in range(len(prefix)):
        v = EV.evaluate(phi, trace_of(prefix), pos, partial_trace=True).value
        if v in (FALSE, TRUE):
            assert EV.evaluate(phi, trace_of(prefix), pos).passed == (v == TRUE)


# ── Regressions ──────────────────────────────────────────────────────────────

ONCE = parse('G(now("deploy") -> X(G(!now("deploy"))))')


def run(formula, steps, severity=ConstraintSeverity.BLOCK_AND_WARN):
    """Each step is a tool name; a refused warn call is repeated once (the override)."""
    enf = ConstraintEnforcer(constraints=[Constraint("k", formula)], default_severity=severity)
    out = []
    for name in steps:
        enf.begin_generation()
        try:
            d = enf.check(name, {}, len(out) + 1)
        except ConstraintViolationError:
            out.append("stop")
            break
        out.append("allow" if d == "allow" else d[0])
        if d == "allow":
            enf.record_completed(name, {}, "", "")
    return out


def test_after_an_override_only_new_violations_are_refused():
    assert run(ONCE, ["deploy", "deploy", "deploy", "ls", "cat", "deploy"]) == [
        "allow", "block_and_warn", "allow", "allow", "allow", "block_and_warn"]


def test_a_repeated_prohibited_call_is_refused_again():
    never = Globally(Not(Called("a")))
    assert run(never, ["a", "a", "ls", "a"]) == ["block_and_warn", "allow", "allow", "block_and_warn"]


def test_a_response_is_not_refused_for_not_having_happened_yet():
    assert run(parse('G(now("a") -> F(now("b")))'), ["a", "ls", "b"],
               ConstraintSeverity.PERSISTENT_BLOCK) == ["allow"] * 3


def test_negation_mirrors_pending():
    assert value(Not(Before("a", "b")), [("a", 0)]).passed
    assert value(Not(Globally(Not(Now("x")))), [("y", 0)]).passed
    assert not value(Not(Before("a", "b")), [("a", 0), ("b", 0)]).passed


def test_bounded_and_positional_atoms_wait_for_their_window():
    assert value(WithinSteps("a", "b", 2), [("ls", 0)]).passed
    assert value(WithinSteps("a", "b", 2), [("a", 0), ("c", 0)]).passed
    assert not value(WithinSteps("a", "b", 1), [("a", 0), ("c", 0), ("c", 0)]).passed
    assert value(AtPosition(3, Now("c")), [("a", 0)]).passed


def test_an_unmet_obligation_still_refuses_and_is_never_suppressed():
    at_least = CalledNTimes("a", 2, ">=")
    assert run(at_least, ["b", "b", "c"]) == ["block_and_warn", "allow", "block_and_warn"]
    v = value(at_least, [("b", 0)])
    assert v.value == PFALSE and not v.passed


def test_weight_zero_does_not_block_a_passing_call():
    enf = ConstraintEnforcer(constraints=[Constraint("w", Globally(Not(Called("x"))), weight=0)],
                             default_severity=ConstraintSeverity.PERSISTENT_BLOCK)
    assert enf.check("ls", {}, 1) == "allow"


def test_predicates_under_a_quantifier_receive_the_bindings():
    seen = []

    def pred(trace, position, bindings):
        seen.append(bindings)
        return True

    f = ForAll("x", lambda t, m: [7], Predicate(pred, "p"))
    assert EV.evaluate(f, trace_of([("a", 0)])).passed
    assert seen == [{"x": 7}]


def test_now_under_a_quantifier():
    f = ForAll("x", lambda t, m: [1], Globally(Implies(Now("a"), CalledWith("b", {"x": Var("x")}))))
    assert not EV.evaluate(f, trace_of([("a", 0)])).passed
    assert value(f, [("a", 0)]).value == PFALSE


def test_predicates_that_take_two_arguments_run_once():
    calls = []
    f = Predicate(lambda trace, position: calls.append(1) or True, "two")
    assert EV.evaluate(f, trace_of([("a", 0)])).passed
    assert calls == [1]


def test_a_predicate_can_report_final_failures_and_witnesses():
    f = Predicate(lambda t, p, m: {"passed": False, "final": True, "witness": [0]}, "p")
    v = value(f, [("a", 0)])
    assert v.value == FALSE and v.witnesses == [((("w", 0),), FALSE)]


def test_strict_scores_are_unchanged_by_the_partial_semantics():
    metrics = {"tool_calls": [{"tool_name": "a"}]}
    assert not verify_trace(metrics, [Constraint("k", Before("a", "b"))])["constraints"][0]["passed"]
    assert value(Before("a", "b"), [("a", 0)]).value == PENDING
