"""
02_foltl_formulas.py – Showcase of AgentLTL formula types.

Demonstrates every major formula type with short pass/fail examples.
No LLM or network access required.

Run:
    python examples/02_foltl_formulas.py
"""

from agentltl import (
    Trace,
    LTLEvaluator,
    Called, CalledWith, CalledNTimes,
    Before, After, AllBefore, BranchCalled,
    InstanceBefore, CalledInOrder, WithinSteps,
    Globally, Eventually, Implies, Not, And, Or,
    Predicate, ForAll, Exists, Var, CalledWithResult,
    verify_trace, Constraint,
)

evaluator = LTLEvaluator()


def check(label: str, formula, trace: Trace, expect: bool):
    result = evaluator.evaluate(formula, trace)
    status = "PASS" if result.passed == expect else "FAIL"
    mark = "✓" if result.passed == expect else "✗"
    print(f"  [{mark} {status}] {label}: {result.detail}")


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def main():
    section("1. Called – tool appears in trace")
    t = Trace.from_names(["a", "b", "c"])
    check("called(a) → pass",  Called("a"), t, True)
    check("called(x) → fail",  Called("x"), t, False)

    section("2. CalledNTimes – exact / min / max count")
    t = Trace.from_names(["poll", "poll", "poll"])
    check("poll == 3 → pass",  CalledNTimes("poll", 3, "=="), t, True)
    check("poll >= 2 → pass",  CalledNTimes("poll", 2, ">="), t, True)
    check("poll == 4 → fail",  CalledNTimes("poll", 4, "=="), t, False)

    section("3. Before – ordering constraint")
    t = Trace.from_names(["fetch", "process", "save"])
    check("before(fetch, process) → pass",  Before("fetch", "process"),  t, True)
    check("before(process, fetch) → fail",  Before("process", "fetch"),  t, False)
    check("before(fetch, save) → pass",     Before("fetch", "save"),     t, True)

    section("4. AllBefore – multi-input gate")
    t = Trace.from_names(["step_a", "step_b", "merge"])
    check("all_before([a,b], merge) → pass",
          AllBefore(["step_a", "step_b"], "merge"), t, True)
    t2 = Trace.from_names(["step_a", "merge", "step_b"])
    check("all_before([a,b], merge) when b late → fail",
          AllBefore(["step_a", "step_b"], "merge"), t2, False)

    section("5. BranchCalled – conditional routing")
    t = Trace.from_names(["check_risk", "high_risk_path"])
    check("branch_called(high_risk_path, low_risk_path) → pass",
          BranchCalled("high_risk_path", "low_risk_path"), t, True)
    t2 = Trace.from_names(["check_risk", "low_risk_path"])
    check("branch_called(high_risk_path, low_risk_path) when wrong branch → fail",
          BranchCalled("high_risk_path", "low_risk_path"), t2, False)

    section("6. CalledWith – argument matching")
    from agentltl._trace import ToolCall
    t = Trace(calls=[
        ToolCall(name="search", args={"query": "cats", "limit": 10}, position=0),
        ToolCall(name="search", args={"query": "dogs", "limit": 5},  position=1),
    ])
    check('called_with(search, query="cats") → pass',
          CalledWith("search", {"query": "cats"}), t, True)
    check('called_with(search, query="fish") → fail',
          CalledWith("search", {"query": "fish"}), t, False)

    section("7. InstanceBefore – nth occurrence ordering")
    t = Trace.from_names(["read", "write", "read", "commit"])
    check("instance_before(read[2], commit[1]) → pass",
          InstanceBefore("read", 2, "commit", 1), t, True)
    check("instance_before(commit[1], read[2]) → fail",
          InstanceBefore("commit", 1, "read", 2), t, False)

    section("8. CalledInOrder – subsequence matching")
    t = Trace.from_names(["init", "load", "process", "save"])
    check("in_order([init, process, save]) → pass",
          CalledInOrder(["init", "process", "save"]), t, True)
    check("in_order([save, process]) → fail (wrong order)",
          CalledInOrder(["save", "process"]), t, False)

    section("9. WithinSteps – time-bounded response")
    t = Trace.from_names(["alert", "ack", "resolve"])
    check("within_steps(alert, ack, 1) → pass",
          WithinSteps("alert", "ack", 1), t, True)
    check("within_steps(alert, resolve, 1) → fail (2 steps away)",
          WithinSteps("alert", "resolve", 1), t, False)

    section("10. Temporal: Eventually, Globally, Implies")
    t = Trace.from_names(["a", "b", "c"])
    check("F(called(b)) → pass",  Eventually(Called("b")),  t, True)
    check("F(called(x)) → fail",  Eventually(Called("x")),  t, False)
    check("G(called(a) -> F called(b)) – vacuous pass",
          Globally(Implies(Called("a"), Eventually(Called("b")))), t, True)

    section("11. ForAll quantifier – universal property")
    from agentltl._trace import ToolCall as TC
    t = Trace(calls=[
        TC(name="read_file", args={"path": "a.txt"}, position=0),
        TC(name="read_file", args={"path": "b.txt"}, position=1),
        TC(name="write_file", args={"path": "a.txt"}, position=2),
        TC(name="write_file", args={"path": "b.txt"}, position=3),
    ])

    def files_read(trace, metrics):
        return [c.args["path"] for c in trace.calls if c.name == "read_file"]

    forall_written = ForAll(
        "f", files_read,
        CalledWith("write_file", {"path": Var("f")}),
        description="files_read",
    )
    check("∀f∈files_read. write_file(f) called → pass", forall_written, t, True)

    # Drop write for b.txt
    t2 = Trace(calls=[
        TC(name="read_file",  args={"path": "a.txt"}, position=0),
        TC(name="read_file",  args={"path": "b.txt"}, position=1),
        TC(name="write_file", args={"path": "a.txt"}, position=2),
    ])
    check("∀f∈files_read. write_file(f) called (b missing) → fail", forall_written, t2, False)

    section("12. Predicate – custom Python callable")
    def no_duplicate_tools(trace, pos, metrics=None):
        seen = set()
        for c in trace.calls:
            if c.name in seen:
                return {"passed": False, "note": f"Duplicate tool call: {c.name}"}
            seen.add(c.name)
        return {"passed": True, "note": "No duplicate tool calls"}

    t = Trace.from_names(["a", "b", "c"])
    check("no_duplicate_tools → pass",
          Predicate(no_duplicate_tools, "no_duplicate_tools"), t, True)
    t2 = Trace.from_names(["a", "b", "a"])
    check("no_duplicate_tools (a appears twice) → fail",
          Predicate(no_duplicate_tools, "no_duplicate_tools"), t2, False)

    print("\nAll formula examples complete.")


if __name__ == "__main__":
    main()
