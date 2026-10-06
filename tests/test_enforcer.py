"""The public Enforcer: decisions, severities, state, chains and termination."""

import json

import pytest

from agentltl import (
    Called, Constraint, ConstraintSeverity, Decision, Enforcer, Globally, Not, Now,
    ConstraintViolationError, parse,
)
from agentltl._enforcement_engine import ConstraintEnforcer

S = ConstraintSeverity
NEVER_RM = Constraint("never-rm", Globally(Not(Now("rm"))), repair="Move it to trash instead.")
NEVER_DD = Constraint("never-dd", Globally(Not(Now("dd"))))
ONCE = Constraint("one-deploy", parse('G(now("deploy") -> X(G(!now("deploy"))))'))


def enforcer(*constraints, **kwargs):
    severities = kwargs.pop("severities", {})
    return Enforcer(list(constraints), severities, **kwargs)


def run(enf, *names):
    """Check each call in a new generation; record it when allowed. Returns the actions."""
    out = []
    for name in names:
        enf.begin_generation()
        d = enf.check(name, {})
        out.append(d.action)
        if d.allowed:
            enf.record_completed(name, {})
    return out


class TestDecisions:
    def test_each_severity_has_its_action(self):
        for severity, action in [(S.PERSISTENT_BLOCK, "block"), (S.BLOCK_AND_WARN, "warn"),
                                 (S.SOFT_BLOCK, "retry"), (S.ASK, "ask"), (S.HARD_STOP, "stop")]:
            d = enforcer(NEVER_RM, default_severity=severity).check("rm", {})
            assert d.action == action and not d.allowed
            assert d.constraint_name == "never-rm" and d.violation.tool_name == "rm"
            assert d.feedback and "never-rm" in d.feedback

    def test_check_never_raises(self):
        d = enforcer(NEVER_RM).check("rm", {})
        assert d.action == "stop"

    def test_tolerated_violations_are_notes_on_an_allowed_call(self):
        d = enforcer(NEVER_RM, default_severity=S.TOLERATE).check("rm", {})
        assert d.allowed and [v.constraint_name for v in d.notes] == ["never-rm"]

    def test_the_repair_text_reaches_the_feedback(self):
        d = enforcer(NEVER_RM, default_severity=S.PERSISTENT_BLOCK).check("rm", {})
        assert "To comply: Move it to trash instead." in d.feedback

    def test_witnesses_are_structured(self):
        d = enforcer(NEVER_RM, default_severity=S.PERSISTENT_BLOCK).check("rm", {})
        assert d.violation.witnesses and d.violation.witnesses[0][1] == 0

    def test_a_custom_renderer(self):
        enf = enforcer(NEVER_RM, default_severity=S.PERSISTENT_BLOCK,
                       render=lambda d, e: f"{d.action}:{d.constraint_name}")
        assert enf.check("rm", {}).feedback == "block:never-rm"


class TestStrongestWins:
    @pytest.mark.parametrize("order", [(NEVER_RM, ONCE), (ONCE, NEVER_RM)])
    def test_whatever_the_order(self, order):
        rm_or_deploy = Constraint("rm-or-deploy", Globally(Not(Now("deploy"))))
        enf = enforcer(*order, rm_or_deploy, severities={
            "never-rm": S.BLOCK_AND_WARN, "one-deploy": S.BLOCK_AND_WARN,
            "rm-or-deploy": S.PERSISTENT_BLOCK})
        d = enf.check("deploy", {})
        assert d.action == "block" and d.constraint_name == "rm-or-deploy"

    def test_nudge_max_reports_several_of_the_same_severity(self):
        both = Constraint("no-rm-either", Globally(Not(Now("rm"))), repair="Really, no.")
        enf = enforcer(NEVER_RM, both, default_severity=S.PERSISTENT_BLOCK, nudge_max=2)
        d = enf.check("rm", {})
        assert [v.constraint_name for v in d.violations] == ["never-rm", "no-rm-either"]
        assert "also violates 1 other" in d.feedback


class TestWarn:
    def test_insisting_overrides_once(self):
        enf = enforcer(NEVER_RM, default_severity=S.BLOCK_AND_WARN)
        assert run(enf, "rm", "rm", "ls", "rm") == ["warn", "allow", "allow", "warn"]

    def test_the_override_is_reported(self):
        enf = enforcer(NEVER_RM, default_severity=S.BLOCK_AND_WARN)
        enf.begin_generation()
        enf.check("rm", {})
        enf.begin_generation()
        assert enf.check("rm", {}).override == "never-rm"

    def test_parallel_calls_of_one_generation_do_not_insist(self):
        enf = enforcer(NEVER_RM, default_severity=S.BLOCK_AND_WARN)
        assert enf.check("rm", {}, generation="msg-1").action == "warn"
        assert enf.check("rm", {}, generation="msg-1").action == "warn"
        assert enf.check("rm", {}, generation="msg-2").allowed


class TestRetry:
    def test_escalates_to_a_stop_by_default(self):
        enf = enforcer(NEVER_RM, default_severity=S.SOFT_BLOCK, max_soft_attempts=2)
        assert run(enf, "rm", "rm") == ["retry", "stop"]

    def test_can_escalate_to_a_human_and_start_over(self):
        enf = enforcer(NEVER_RM, default_severity=S.SOFT_BLOCK, max_soft_attempts=2,
                       escalate_to=S.ASK)
        decisions = []
        for _ in range(4):
            enf.begin_generation()
            decisions.append(enf.check("rm", {}))
        assert [d.action for d in decisions] == ["retry", "ask", "retry", "ask"]
        assert decisions[1].escalated and decisions[0].attempts == 1

    def test_consecutive_counting_forgives_good_calls(self):
        enf = enforcer(NEVER_RM, default_severity=S.SOFT_BLOCK, max_soft_attempts=2,
                       soft_block_mode="consecutive")
        assert run(enf, "rm", "ls", "rm", "ls") == ["retry", "allow", "retry", "allow"]


class TestStop:
    def test_a_latched_stop_refuses_everything_until_resumed(self):
        enf = enforcer(NEVER_RM, latch_stop=True)
        assert run(enf, "rm", "ls") == ["stop", "stop"]
        assert enf.stopped == "never-rm"
        enf.resume()
        assert run(enf, "ls") == ["allow"]

    def test_the_legacy_interface_still_raises(self):
        enf = ConstraintEnforcer([NEVER_RM])
        with pytest.raises(ConstraintViolationError):
            enf.check("rm", {}, 1)


class TestState:
    def test_round_trips_through_json(self):
        enf = enforcer(NEVER_RM, ONCE, default_severity=S.BLOCK_AND_WARN)
        run(enf, "deploy", "rm")
        state = json.loads(json.dumps(enf.to_state()))
        again = enforcer(NEVER_RM, ONCE, default_severity=S.BLOCK_AND_WARN).from_state(state)
        assert run(again, "rm", "deploy") == ["allow", "warn"]
        assert [c["tool_name"] for c in again.trace] == ["deploy", "rm"]

    def test_limits_trim_the_trace_and_logs(self):
        enf = enforcer(NEVER_RM, default_severity=S.TOLERATE)
        run(enf, "rm", "rm", "rm")
        state = enf.to_state(max_trace=2, max_log=1)
        assert len(state["completed_tool_calls"]) == 2 and len(state["constraint_violations"]) == 1

    def test_exit_status_is_kept(self):
        enf = enforcer()
        enf.record_completed("pytest", {}, result="1 failed", status=1)
        assert enf.trace[-1]["status"] == 1


class TestChains:
    def test_all_or_nothing_with_the_refused_call_named(self):
        enf = enforcer(NEVER_RM, default_severity=S.PERSISTENT_BLOCK)
        d = enf.check_chain([("ls", {}), ("rm", {"path": "x"})])
        assert d.action == "block" and d.index == 1
        assert enf.trace == []

    def test_later_calls_of_the_chain_see_the_earlier_ones(self):
        enf = enforcer(ONCE, default_severity=S.PERSISTENT_BLOCK)
        assert enf.check_chain([("deploy", {}), ("deploy", {})]).index == 1
        assert enf.check_chain([("deploy", {}), ("ls", {})]).allowed

    def test_insisting_means_reissuing_the_whole_chain(self):
        enf = enforcer(NEVER_RM, default_severity=S.BLOCK_AND_WARN)
        chain = [("ls", {}), ("rm", {})]
        enf.begin_generation()
        assert enf.check_chain(chain).action == "warn"
        enf.begin_generation()
        assert enf.check_chain(chain).allowed


class TestTermination:
    def test_sends_the_agent_back_a_bounded_number_of_times(self):
        must_test = Constraint("tests-run", Called("pytest"), repair="Run pytest.",
                               applies_to_final_answer=True)
        enf = enforcer(must_test, default_severity=S.TOLERATE, max_termination_nudges=1)
        d = enf.check_termination()
        assert d.action == "block" and "Run pytest." in d.feedback
        assert enf.check_termination().allowed

    def test_nothing_pending_lets_it_finish(self):
        assert isinstance(enforcer(NEVER_RM).check_termination(), Decision)
        assert enforcer(NEVER_RM, max_termination_nudges=3).check_termination().allowed


class TestPastTimeRules:
    def test_before_since_is_judged_per_call(self):
        from agentltl import Previous, Since, Implies
        # every push needs a test run since the last edit
        rule = Constraint("tests-before-push", Globally(Implies(
            Now("push"), Previous(Since(Not(Now("edit")), Now("test"))))))
        enf = enforcer(rule, default_severity=S.BLOCK_AND_WARN)
        assert run(enf, "push", "test", "push", "edit", "push", "push", "ls", "push") == [
            "warn", "allow", "allow", "allow", "warn", "allow", "allow", "warn"]

    def test_a_possible_match_is_flagged_uncertain(self):
        from agentltl import Matches, ToolPattern

        class Rm(ToolPattern):
            def match(self, name, args):
                return None if args.get("unknown") else name in self.tools

        rule = Constraint("no-rm", Globally(Not(Matches(Rm("rm"), maybe=True))))
        enf = enforcer(rule, default_severity=S.PERSISTENT_BLOCK)
        assert not enf.check("rm", {}).violation.uncertain
        d = enf.check("rm", {"unknown": True})
        assert d.action == "block" and d.violation.uncertain


class TestPatternsSeeTheRecordedCall:
    def test_a_rule_on_exit_status(self):
        from agentltl import Implies, Matches, Once, Previous, ToolPattern

        class Passed(ToolPattern):
            def match(self, name, args, call):
                return name in self.tools and call.raw.get("status") == 0

        rule = Constraint("green-before-push", Globally(Implies(
            Now("push"), Previous(Once(Matches(Passed("pytest")))))))
        enf = enforcer(rule, default_severity=S.PERSISTENT_BLOCK)
        enf.record_completed("pytest", status=1)
        assert enf.check("push", {}).action == "block"
        enf.record_completed("pytest", status=0)
        assert enf.check("push", {}).allowed

    def test_termination_nudges_can_be_allowed_again(self):
        must = Constraint("tests-run", Called("pytest"), applies_to_final_answer=True)
        enf = enforcer(must, default_severity=S.TOLERATE, max_termination_nudges=1)
        assert enf.check_termination().action == "block"
        assert enf.check_termination().allowed
        enf.reset_termination_nudges()
        assert enf.check_termination().action == "block"
