"""The YAML rulebooks: each bundled rule passes a compliant trace and fails a violating
one, post hoc and under the Enforcer, at a severity its runtime-safety class allows."""

import pytest

pytest.importorskip("yaml")

from agentltl import ConstraintSeverity, Enforcer, classify_constraints, verify_trace  # noqa: E402
from agentltl import rulebooks  # noqa: E402
from agentltl.enforcer import ACTIONS  # noqa: E402

S = ConstraintSeverity

# rule name -> (compliant trace, violating trace). A call is a name or (name, args).
CASES = {
    # devops
    "tests_before_deploy": (["run_tests", "deploy_prod"], ["deploy_prod", "run_tests"]),
    "approval_before_prod": (["approve_release", "deploy_prod"], ["build", "deploy_prod"]),
    "no_force_push": ([("git_push", {"force": False})], [("git_push", {"force": True})]),
    "backup_before_drop": (["backup_db", "drop_table"], ["drop_table"]),
    "single_prod_deploy": (["deploy_prod"], ["deploy_prod", "deploy_prod"]),
    "health_check_after_deploy": (["deploy_prod", "health_check"], ["deploy_prod", "notify"]),
    # privacy
    "consent_before_processing": (["check_consent", "process_personal_data"],
                                  ["process_personal_data"]),
    "anonymize_before_export": (["anonymize", "export_dataset"], ["export_dataset"]),
    "no_transfer_outside_eu": ([("transfer_data", {"region": "eu"})],
                               [("transfer_data", {"region": "non_eu"})]),
    "log_record_access": (["read_personal_record", "log_access"],
                          ["read_personal_record", "send_email"]),
    # payments
    "kyc_before_payment": (["verify_identity", "check_sanctions", "execute_payment"],
                           ["verify_identity", "execute_payment"]),
    "four_eyes_approval": (["approve_payment", "approve_payment", "execute_payment"],
                           ["approve_payment", "execute_payment"]),
    "single_payment": (["execute_payment"], ["execute_payment", "execute_payment"]),
    "fresh_quote_before_order": (["get_quote", "place_order"],
                                 ["get_quote", "check_balance", "place_order"]),
    # support
    "customer_before_refund": (["lookup_customer", "issue_refund"], ["issue_refund"]),
    "order_before_refund": (["get_order", "issue_refund"], ["lookup_customer", "issue_refund"]),
    "notify_after_escalation": (["escalate_ticket", "notify_customer"],
                                ["escalate_ticket", "a", "b", "c", "notify_customer"]),
    "nothing_after_close": (["reply", "close_ticket"], ["close_ticket", "reply"]),
    # research
    "search_before_answer": (["search_docs"], ["web_search"]),
    "cite_sources": (["search_docs", "cite_source"], ["search_docs"]),
    "web_search_budget": (["web_search"] * 5, ["web_search"] * 6),
    # budget
    "bounded_retries": (["retry_request"] * 3, ["retry_request"] * 4),
    "bounded_polling": (["poll_status"] * 10, ["poll_status"] * 11),
    "read_before_write": (["read_file", "write_file"], ["write_file"]),
    "no_email_burst": (["send_email", "log", "send_email"], ["send_email", "send_email"]),
    # data pipeline
    "validate_before_load": (["validate_data", "load"], ["transform", "load"]),
    "fan_in_before_report": (["fetch_users", "fetch_orders", "build_report"],
                             ["fetch_users", "build_report"]),
    "pipeline_order": (["fetch_data", "validate_data", "transform", "load"],
                       ["fetch_data", "transform", "load"]),
    # healthcare
    "allergies_before_prescription": (["check_allergies", "prescribe"], ["prescribe"]),
    "interactions_before_prescription": (["check_interactions", "prescribe"], ["prescribe"]),
    # procurement
    "conflict_check_before_award": (["check_conflict_of_interest", "award_contract"],
                                    ["award_contract"]),
    "three_quotes_before_award": (["request_quote"] * 3 + ["award_contract"],
                                  ["request_quote"] * 2 + ["award_contract"]),
    "register_gift": (["accept_gift", "register_gift"], ["accept_gift", "reply"]),
    # human in the loop
    "confirm_before_send": (["ask_user_confirmation", "send_email"], ["draft", "send_email"]),
    "confirm_before_delete": (["user_confirm_deletion", "delete_account"],
                              ["delete_account", "user_confirm_deletion"]),
}

ALL = rulebooks.load(*rulebooks.available())
RULES = [(c, ALL.severities[c.name]) for c in ALL.constraints]


def calls(trace):
    return [(c, {}) if isinstance(c, str) else c for c in trace]


def metrics(trace):
    return {"tool_calls": [{"tool_name": n, "arguments": a} for n, a in calls(trace)]}


def replay(constraint, severity, trace):
    """Run *trace* through an Enforcer over *constraint* alone; the first refusal, or None."""
    enforcer = Enforcer([constraint], {constraint.name: severity})
    for name, args in calls(trace):
        enforcer.begin_generation()
        decision = enforcer.check(name, args)
        if not decision.allowed:
            return decision
        enforcer.record_completed(name, args)
    return None


def test_bundled_rulebooks():
    assert rulebooks.available() == [
        "budget", "data_pipeline", "devops", "healthcare", "human_in_the_loop", "payments",
        "privacy", "procurement", "research", "support"]


def test_every_rule_has_cases():
    assert sorted(CASES) == sorted(c.name for c, _ in RULES)


@pytest.mark.parametrize("constraint, severity", RULES, ids=[c.name for c, _ in RULES])
class TestEachRule:
    def test_compliant_trace_passes(self, constraint, severity):
        result = verify_trace(metrics(CASES[constraint.name][0]), [constraint])
        assert result["compliance_label"] == "FULL", result["constraints"][0]["detail"]

    def test_violating_trace_fails(self, constraint, severity):
        result = verify_trace(metrics(CASES[constraint.name][1]), [constraint])
        assert result["compliance_label"] == "VIOLATION"

    def test_severity_matches_runtime_safety(self, constraint, severity):
        [report] = classify_constraints([constraint], {constraint.name: severity})
        assert report.compatible, report.message

    def test_has_description_and_repair(self, constraint, severity):
        assert constraint.description and constraint.repair and constraint.weight > 0

    def test_enforcer_allows_the_compliant_trace(self, constraint, severity):
        assert replay(constraint, severity, CASES[constraint.name][0]) is None

    def test_enforcer_refuses_the_violating_trace(self, constraint, severity):
        decision = replay(constraint, severity, CASES[constraint.name][1])
        if severity is S.TOLERATE:
            assert decision is None and constraint.applies_to_final_answer
        else:
            assert decision is not None and decision.action == ACTIONS[severity]
            assert decision.constraint_name == constraint.name
            if decision.action != "stop":  # the stop message names the rule, not the repair
                assert constraint.repair in decision.feedback


class TestEndOfRunObligations:
    def test_unmet_obligation_sends_the_agent_back(self):
        enforcer = rulebooks.load("research").enforcer(max_termination_nudges=1)
        enforcer.record_completed("web_search", {})
        decision = enforcer.check_termination()
        assert decision.action == "block" and "search_docs" in decision.feedback

    def test_met_obligation_lets_the_agent_finish(self):
        enforcer = rulebooks.load("research").enforcer(max_termination_nudges=1)
        enforcer.record_completed("search_docs", {})
        enforcer.record_completed("cite_source", {})
        assert enforcer.check_termination().allowed


RULEBOOK = """\
name: mine
description: 'My rules.'
rules:
  - name: no_rm
    formula: '!called("rm")'
    severity: PERSISTENT_BLOCK
    repair: 'Move it to trash.'
  - name: fetch_first
    formula: 'before("fetch", "save")'
"""


def write(tmp_path, text, name="mine.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestLoad:
    def test_a_file_loads(self, tmp_path):
        book = rulebooks.load(write(tmp_path, RULEBOOK))
        assert book.name == "mine" and [c.name for c in book.constraints] == ["no_rm",
                                                                             "fetch_first"]
        assert book.severities == {"no_rm": S.PERSISTENT_BLOCK, "fetch_first": S.HARD_STOP}
        assert book.constraints[1].weight == 1.0
        assert book.enforcer().check("rm", {}).action == "block"

    def test_names_and_files_merge(self, tmp_path):
        book = rulebooks.load("devops", write(tmp_path, RULEBOOK))
        assert book.name == "devops+mine"
        assert {"tests_before_deploy", "no_rm"} <= set(book.severities)

    def test_duplicate_rule_names_are_refused(self):
        with pytest.raises(ValueError, match="defined twice"):
            rulebooks.load("devops", "devops")

    @pytest.mark.parametrize("bad, message", [
        (RULEBOOK.replace("PERSISTENT_BLOCK", "BLOCK"), "unknown severity"),
        (RULEBOOK.replace('before("fetch", "save")', 'before("fetch"'), "fetch_first"),
        (RULEBOOK.replace("    repair:", "    repiar:"), "unknown key"),
        (RULEBOOK.replace("fetch_first", "no_rm"), "duplicate rule name"),
        (RULEBOOK.replace("    formula: '!called(\"rm\")'\n", ""), "needs a 'name' and a"),
        ("name: empty\n", "'rules' list"),
    ])
    def test_malformed_files_say_what_is_wrong(self, tmp_path, bad, message):
        with pytest.raises(ValueError, match=message):
            rulebooks.load(write(tmp_path, bad))

    def test_unknown_source(self):
        with pytest.raises(ValueError, match="No rulebook 'nope'"):
            rulebooks.load("nope")
