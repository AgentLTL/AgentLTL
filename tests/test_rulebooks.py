"""The YAML rulebooks: each bundled rule passes a compliant trace and fails a violating
one, post hoc and under the Enforcer, at a severity its runtime-safety class allows."""

import pytest

pytest.importorskip("yaml")

from agentltl import (  # noqa: E402
    Constraint,
    ConstraintSeverity,
    Enforcer,
    classify_constraints,
    rulebooks,  # noqa: E402
    verify_trace,
)
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
    "tests_after_last_change": (["edit_code", "run_tests", "deploy_prod"],
                                ["run_tests", "edit_code", "deploy_prod"]),
    "staging_before_prod": (["deploy_staging", "deploy_prod"], ["deploy_prod"]),
    "no_prod_db_drop": ([("drop_database", {"env": "dev"})], [("drop_database", {"env": "prod"})]),
    # privacy
    "consent_before_processing": (["check_consent", "process_personal_data"],
                                  ["process_personal_data"]),
    "anonymize_before_export": (["anonymize", "export_dataset"], ["export_dataset"]),
    "no_transfer_outside_eu": ([("transfer_data", {"region": "eu"})],
                               [("transfer_data", {"region": "non_eu"})]),
    "log_record_access": (["read_personal_record", "log_access"],
                          ["read_personal_record", "send_email"]),
    "no_pii_in_export": ([("export_dataset", {"include_pii": False})],
                         [("export_dataset", {"include_pii": True})]),
    "dpa_before_third_party": (["check_dpa", "share_with_third_party"],
                               ["share_with_third_party"]),
    "erasure_request_honoured": (["erasure_request", "delete_personal_data"],
                                 ["erasure_request", "reply"]),
    # payments
    "kyc_before_payment": (["verify_identity", "check_sanctions", "execute_payment"],
                           ["verify_identity", "execute_payment"]),
    "four_eyes_approval": ([("approve_payment", {"role": "finance"}),
                            ("approve_payment", {"role": "manager"}), "execute_payment"],
                           [("approve_payment", {"role": "finance"}),
                            ("approve_payment", {"role": "finance"}), "execute_payment"]),
    "single_payment": (["execute_payment"], ["execute_payment", "execute_payment"]),
    "fresh_quote_before_order": (["get_quote", "place_order"],
                                 ["get_quote", "check_balance", "place_order"]),
    "invoice_match_before_payment": (["match_invoice", "execute_payment"], ["execute_payment"]),
    "callback_for_new_beneficiary": (
        ["execute_payment", "add_beneficiary", "callback_verification", "execute_payment"],
        ["add_beneficiary", "execute_payment"]),
    # support
    "customer_before_refund": (["lookup_customer", "issue_refund"], ["issue_refund"]),
    "order_before_refund": (["get_order", "issue_refund"], ["lookup_customer", "issue_refund"]),
    "notify_after_escalation": (["escalate_ticket", "notify_customer"],
                                ["escalate_ticket", "a", "b", "c", "notify_customer"]),
    "nothing_after_close": (["reply", "close_ticket"], ["close_ticket", "reply"]),
    "history_before_reply": (["get_ticket_history", "reply"], ["reply"]),
    "single_refund": (["issue_refund"], ["issue_refund", "issue_refund"]),
    # research
    "search_before_answer": (["search_docs"], ["web_search"]),
    "cite_sources": (["search_docs", "cite_source"], ["search_docs"]),
    "web_search_budget": (["web_search"] * 5, ["web_search"] * 6),
    "internal_before_web": (["search_docs", "web_search"], ["web_search"]),
    "cite_only_what_was_read": (["fetch_document", "cite_source"], ["search_docs", "cite_source"]),
    # budget
    "bounded_retries": (["retry_request"] * 3, ["retry_request"] * 4),
    "bounded_polling": (["poll_status"] * 10, ["poll_status"] * 11),
    "read_before_write": (["read_file", "write_file"], ["write_file"]),
    "no_email_burst": (["send_email", "log", "send_email"], ["send_email", "send_email"]),
    "tool_call_budget": (["x"] * 50, ["x"] * 51),
    "backoff_between_retries": (["wait", "retry_request"], ["fail", "retry_request"]),
    "bounded_subagents": (["call_subagent"] * 5, ["call_subagent"] * 6),
    # data pipeline
    "validate_before_load": (["validate_data", "load"], ["transform", "load"]),
    "fan_in_before_report": (["fetch_users", "fetch_orders", "build_report"],
                             ["fetch_users", "build_report"]),
    "pipeline_order": (["fetch_data", "validate_data", "transform", "load"],
                       ["fetch_data", "transform", "load"]),
    "schema_check_before_transform": (["check_schema", "transform"], ["transform"]),
    "single_load": (["load"], ["load", "load"]),
    # healthcare
    "allergies_before_prescription": (["check_allergies", "prescribe"], ["prescribe"]),
    "interactions_before_prescription": (["check_interactions", "prescribe"], ["prescribe"]),
    "patient_identity_first": (["verify_patient_identity", "prescribe"], ["prescribe"]),
    "dosage_check_before_prescription": (["check_dosage", "prescribe"], ["prescribe"]),
    "physician_for_controlled": (["physician_approval", "prescribe_controlled"],
                                 ["prescribe_controlled"]),
    "record_after_prescription": (["prescribe", "update_medical_record"], ["prescribe"]),
    # procurement
    "conflict_check_before_award": (["check_conflict_of_interest", "award_contract"],
                                    ["award_contract"]),
    "three_quotes_before_award": (["request_quote"] * 3 + ["award_contract"],
                                  ["request_quote"] * 2 + ["award_contract"]),
    "register_gift": (["accept_gift", "register_gift"], ["accept_gift", "reply"]),
    "budget_check_before_award": (["check_budget", "award_contract"], ["award_contract"]),
    "single_award": (["award_contract"], ["award_contract", "award_contract"]),
    # human in the loop
    "confirm_before_send": (["ask_user_confirmation", "send_email"], ["draft", "send_email"]),
    "confirm_before_publish": (["ask_user_confirmation", "publish_post"], ["draft", "publish_post"]),
    "confirm_before_external_share": (["ask_user_confirmation", "share_externally"],
                                      ["share_externally"]),
    # one confirmation does not cover a second deletion
    "confirm_before_delete": (["user_confirm_deletion", "delete_account"],
                              ["user_confirm_deletion", "delete_account", "delete_account"]),
    # ai act
    "aiact_art12_audit_logs_intact": ([("write_file", {"path": "src/app.py"})],
                                      [("edit_file", {"path": "logs/audit/2026-11.jsonl"})]),
    "aiact_art14_guardrails_untouchable": ([("edit_file", {"path": "src/app.py"})],
                                           [("edit_file", {"path": "AGENTLTL.yaml"})]),
    "aiact_art5_1c_no_social_scoring": (["credit_check"], ["social_score"]),
    "aiact_art5_1e_no_facial_scraping": ([("face_index", {"source": "consented_upload"})],
                                         [("face_index", {"source": "web_crawl"})]),
    "aiact_art5_1f_emotion_context_declared": ([("detect_emotion", {"context": "medical"})],
                                               [("detect_emotion", {})]),
    "aiact_art5_1f_no_emotions_work_education": (
        [("emotion_recognition", {"context": "medical"})],
        [("emotion_recognition", {"context": "workplace"})]),
    "aiact_art5_1g_no_biometric_categorisation": (
        [("biometric_categorize", {"attribute": "age_range"})],
        [("biometric_categorize", {"attribute": "religion"})]),
    "aiact_art5_1h_no_realtime_remote_identification": (
        [("remote_biometric_identify", {"realtime": False})],
        [("remote_biometric_identify", {"realtime": True})]),
    "aiact_art5_1d_no_profiling_crime_prediction": (
        [("predict_offence_risk", {"basis": "objective_facts"})],
        [("predict_offence_risk", {"basis": "profiling"})]),
    "aiact_art5_safety_check_before_generation": (["content_safety_check", "generate_image"],
                                                  ["generate_image"]),
    "aiact_art50_1_disclose_ai_interaction": (["disclose_ai_interaction", "send_message"],
                                              ["send_message"]),
    "aiact_art50_2_mark_before_release": (["mark_synthetic_content", "upload_media"],
                                          ["upload_media"]),
    "aiact_art50_4_deepfake_disclosed": ([("publish_media", {"ai_disclosure": True})],
                                         [("publish_media", {"ai_disclosure": False})]),
    "aiact_art50_4_public_interest_text": (
        ["human_editorial_review", ("publish_article", {"ai_disclosure": False})],
        [("publish_article", {"ai_disclosure": False})]),
    "aiact_art50_3_inform_exposed_persons": (["inform_exposed_persons", "detect_emotion"],
                                             ["detect_emotion"]),
    "aiact_art6_classify_before_deployment": (["classify_system", "deploy_system"],
                                              ["deploy_system"]),
    "aiact_art27_fria_before_use": (["fria", "enable_use_case"], ["enable_use_case"]),
    "aiact_art9_risk_review_before_change": (["risk_review", "update_model_version"],
                                             ["update_model_version"]),
    "aiact_art14_human_review_per_decision": (["human_review", "apply_decision"],
                                              ["human_review", "apply_decision",
                                               "apply_decision"]),
    "aiact_art14_human_confirms_decision": (["human_review"], ["apply_decision"]),
    "aiact_art26_11_inform_affected_person": (["inform_affected_person", "apply_decision"],
                                              ["apply_decision"]),
    "aiact_art12_log_before_decision": (["log_event", "apply_decision"], ["apply_decision"]),
    "aiact_art12_log_after_decision": (["apply_decision", "log_event"], ["apply_decision"]),
    "aiact_art14_automation_cap": (["apply_decision"] * 20, ["apply_decision"] * 21),
    "aiact_art10_datasheet_before_training": (["edit_dataset", "datasheet", "train_model"],
                                              ["datasheet", "edit_dataset", "train_model"]),
    "aiact_art10_bias_examination_before_training": (
        ["bias_examination", "finetune_model"], ["bias_examination", "edit_dataset",
                                                 "finetune_model"]),
    "aiact_art4a_sensitive_data_dedicated_tool": ([("load_dataset", {"category": "sales"})],
                                                  [("load_dataset", {"category": "health"})]),
    "aiact_art4a_sensitive_access_approved": (["load_dataset"], ["load_sensitive_data"]),
    "aiact_art4a_delete_after_use": (["load_sensitive_data", "delete_sensitive_data"],
                                     ["load_sensitive_data", "train_model"]),
    "aiact_art53_1c_tdm_optout_checked": (["check_tdm_optout", "crawl"],
                                          ["check_tdm_optout", "crawl", "crawl"]),
    "aiact_art53_1d_training_summary": (["training_summary", "release_model"],
                                        ["release_model"]),
    "aiact_art53_1a_documentation_up_to_date": (["train_model", "doccheck", "release_model"],
                                                ["doccheck", "train_model", "release_model"]),
    "aiact_art55_adversarial_evals": (["finetune_model", "run_adversarial_evals",
                                       "release_model"],
                                      ["run_adversarial_evals", "finetune_model",
                                       "release_model"]),
    "aiact_art55_incident_reported_to_ai_office": (
        ["flag_serious_incident", "report_incident_ai_office"], ["flag_serious_incident"]),
    "aiact_art73_incident_reported": (["flag_serious_incident", "report_serious_incident"],
                                      ["flag_serious_incident"]),
    "aiact_art73_report_quickly": (["flag_serious_incident", "report_serious_incident"],
                                   ["flag_serious_incident"] + ["read_logs"] * 6
                                   + ["report_serious_incident"]),
    "aiact_art73_6_preserve_evidence": (["flag_serious_incident", "report_serious_incident",
                                         "rollback_model"],
                                        ["flag_serious_incident", "rollback_model"]),
    "aiact_dev_doccheck_after_last_change": (["edit_code", "doccheck", "deploy_prod"],
                                             ["doccheck", "edit_code", "deploy_prod"]),
    "aiact_dev_evals_after_last_change": (["edit_code", "run_evals", "publish_model"],
                                          ["run_evals", "edit_code", "publish_model"]),
    "aiact_dev_compliance_register": ([("edit_file", {"path": "src/app.py"})],
                                      [("edit_file", {"path": "compliance/registre.yaml"})]),
    "aiact_dev_biometric_libraries": ([("install_package", {"name": "numpy"})],
                                      [("install_package", {"name": "deepface"})]),
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
        "ai_act", "budget", "data_pipeline", "devops", "healthcare", "human_in_the_loop",
        "payments",
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


# Rules are as strict as their descriptions say. Each case: rule, a trace it must refuse, and one
# the same rule must accept. Before the rulebooks were tightened, the first one slipped through.
def a(name, **args):
    return (name, args)


STRICT = [
    # a gate holds for EVERY action, not once per run
    ("backup_before_drop", ["backup_db", "drop_table", "drop_table"],
     ["backup_db", "drop_table", "backup_db", "drop_table"]),
    ("tests_before_deploy", ["run_tests", "deploy_prod", "deploy_prod"],
     ["run_tests", "deploy_prod", "run_tests", "deploy_prod"]),
    ("allergies_before_prescription", ["check_allergies", "prescribe", "prescribe"],
     ["check_allergies", "prescribe", "check_allergies", "prescribe"]),
    ("allergies_before_prescription", ["check_allergies", "prescribe_controlled", "prescribe"],
     ["check_allergies", "prescribe_controlled", "check_allergies", "prescribe"]),
    ("patient_identity_first", ["prescribe_controlled"],
     ["verify_patient_identity", "prescribe_controlled"]),
    ("physician_for_controlled",
     ["physician_approval", "prescribe_controlled", "prescribe_controlled"],
     ["physician_approval", "prescribe_controlled", "physician_approval", "prescribe_controlled"]),
    ("anonymize_before_export", ["anonymize", "export_dataset", "export_dataset"],
     ["anonymize", "export_dataset", "anonymize", "export_dataset"]),
    ("consent_before_processing",
     ["check_consent", "process_personal_data", "process_personal_data"],
     ["check_consent", "process_personal_data", "check_consent", "process_personal_data"]),
    ("kyc_before_payment", ["verify_identity", "execute_payment"],
     ["verify_identity", "check_sanctions", "execute_payment"]),
    ("invoice_match_before_payment", ["match_invoice", "execute_payment", "execute_payment"],
     ["match_invoice", "execute_payment", "match_invoice", "execute_payment"]),
    ("order_before_refund", ["get_order", "issue_refund", "issue_refund"],
     ["get_order", "issue_refund", "get_order", "issue_refund"]),
    ("validate_before_load", ["validate_data", "load", "load"],
     ["validate_data", "load", "validate_data", "load"]),
    # the callback comes after the beneficiary was added, and again for the next one
    ("callback_for_new_beneficiary",
     ["callback_verification", "add_beneficiary", "execute_payment"],
     ["add_beneficiary", "callback_verification", "execute_payment"]),
    ("callback_for_new_beneficiary",
     ["add_beneficiary", "callback_verification", "add_beneficiary", "execute_payment"],
     ["add_beneficiary", "callback_verification", "add_beneficiary", "callback_verification",
      "execute_payment"]),
    # two approvals from two roles, not one role twice
    ("four_eyes_approval",
     [a("approve_payment", role="manager"), a("approve_payment", role="manager"),
      "execute_payment"],
     [a("approve_payment", role="manager"), a("approve_payment", role="finance"),
      "execute_payment"]),
    # argument values: any spelling of true / prod, and an allow-list rather than one bad value
    ("no_force_push", [a("git_push", force="true")], [a("git_push", force="false")]),
    ("no_force_push", [a("git_push", force=1)], [a("git_push")]),
    ("no_prod_db_drop", [a("drop_database", env="production")], [a("drop_database", env="dev")]),
    ("no_prod_db_drop", [a("drop_database", env="PROD")], [a("drop_database", env="staging")]),
    ("no_pii_in_export", [a("export_dataset", include_pii="True")],
     [a("export_dataset", include_pii=False)]),
    ("no_transfer_outside_eu", [a("transfer_data", region="us-east-1")],
     [a("transfer_data", region="eu-west-1")]),
    ("no_transfer_outside_eu", [a("transfer_data")], [a("transfer_data", region="EU")]),
    ("no_transfer_outside_eu", [a("transfer_data", region="asia")],
     [a("transfer_data", region="europe-west1")]),
    # the pipeline order is refused as it happens, not only at the end
    ("pipeline_order", ["fetch_data", "transform"], ["fetch_data", "validate_data", "transform"]),
    # ai act: one human review covers one decision, of any kind
    ("aiact_art14_human_review_per_decision",
     ["human_review", "apply_decision", "reject_application"],
     ["human_review", "apply_decision", "human_review", "reject_application"]),
    # each release is marked again; a missing disclosure flag counts as no disclosure
    ("aiact_art50_2_mark_before_release",
     ["mark_synthetic_content", "upload_media", "export_asset"],
     ["mark_synthetic_content", "upload_media", "mark_synthetic_content", "export_asset"]),
    ("aiact_art50_4_deepfake_disclosed", [a("publish_media")],
     [a("publish_media", ai_disclosure="yes")]),
    # the safety check is for THIS generation, not one made earlier
    ("aiact_art5_safety_check_before_generation",
     ["content_safety_check", "generate_image", "generate_video"],
     ["content_safety_check", "generate_image", "content_safety_check", "generate_video"]),
    # prohibited contexts and sources match any spelling of the prefix
    ("aiact_art5_1f_no_emotions_work_education",
     [a("detect_emotion", context="Recruitment interview")],
     [a("detect_emotion", context="safety")]),
    ("aiact_art5_1e_no_facial_scraping", [a("build_face_database", source="https://x.org")],
     [a("build_face_database", source="employee_badges")]),
    # audit logs and guard files, on Windows paths too
    ("aiact_art12_audit_logs_intact", [a("delete_file", path=r"C:\app\logs\audit\day.jsonl")],
     [a("delete_file", path=r"C:\app\logs\debug.log")]),
    ("aiact_art14_guardrails_untouchable",
     [a("write_file", path="src/agentltl/rulebooks/ai_act.yaml")],
     [a("write_file", path="docs/rulebooks.md")]),
    # evidence is preserved after an incident, whatever the change
    ("aiact_art73_6_preserve_evidence", ["flag_serious_incident", "delete_model_artifact"],
     ["update_model_version"]),
]


@pytest.mark.parametrize("rule, bad, good", STRICT,
                         ids=[f"{r}-{i}" for i, (r, _, _) in enumerate(STRICT)])
def test_rule_is_as_strict_as_it_says(rule, bad, good):
    constraint = next(c for c, _ in RULES if c.name == rule)
    severity = ALL.severities[rule]
    assert verify_trace(metrics(bad), [constraint])["compliance_label"] == "VIOLATION"
    assert verify_trace(metrics(good), [constraint])["compliance_label"] == "FULL"
    refused = replay(constraint, severity, bad)
    assert refused is not None and refused.constraint_name == rule
    assert replay(constraint, severity, good) is None


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

    def test_applies_to_final_answer_must_be_a_boolean(self, tmp_path):
        text = RULEBOOK.replace("    repair: 'Move it to trash.'",
                                "    applies_to_final_answer: 'false'")
        with pytest.raises(ValueError, match="no_rm.*true or false"):
            rulebooks.load(write(tmp_path, text))

    def test_a_bad_weight_names_the_rule(self, tmp_path):
        text = RULEBOOK.replace("    repair: 'Move it to trash.'", "    weight: heavy")
        with pytest.raises(ValueError, match="no_rm.*'weight' is a number"):
            rulebooks.load(write(tmp_path, text))

    def test_unknown_source(self):
        with pytest.raises(ValueError, match="No rulebook 'nope'"):
            rulebooks.load("nope")


NEVER = """\
name: limits
rules:
  - name: small_payments
    never: {tool: pay, where: {amount: {gt: 1000}}}
  - name: eu_only
    never: {tool: send, where: {region: {not_in: [eu, EU]}}}
  - name: no_tmp_paths
    never: {tool: [write, delete], where: {path: {matches: '^/tmp/'}}}
  - name: needs_reason
    never: {tool: delete, where: {reason: {present: false}}}
"""


def refused(book, name, **args):
    enforcer = book.enforcer()
    enforcer.begin_generation()
    return not enforcer.check(name, args).allowed


class TestNever:
    """``never:`` rules: argument conditions that ``called_with`` equality cannot say."""

    def test_conditions(self, tmp_path):
        book = rulebooks.load(write(tmp_path, NEVER))
        assert refused(book, "pay", amount=5000) and refused(book, "pay", amount="5000.50")
        assert not refused(book, "pay", amount=1000) and not refused(book, "pay", amount=20)
        assert refused(book, "pay", amount="a lot")          # not a number: cannot say it is small
        assert refused(book, "send", region="us") and refused(book, "send")   # missing too
        assert not refused(book, "send", region="eu") and not refused(book, "send", region="EU")
        assert refused(book, "write", path="/TMP/x") and refused(book, "delete", path="/tmp/x",
                                                                   reason="old")
        assert not refused(book, "write", path="/home/x")
        assert refused(book, "delete", path="/home/x")       # no reason given
        assert not refused(book, "delete", path="/home/x", reason="stale")

    def test_the_other_tools_are_untouched(self, tmp_path):
        book = rulebooks.load(write(tmp_path, NEVER))
        assert not refused(book, "read", amount=10 ** 9, region="us", path="/tmp/x")

    @pytest.mark.parametrize("bad, message", [
        ("never: {where: {a: 1}}", "'never' is a mapping with a 'tool'"),
        ("never: {tool: x, wher: {a: 1}}", "unknown key"),
        ("never: {tool: x, where: {a: {gt: many}}}", "gt takes a number"),
        ("never: {tool: x, where: {a: {in: 1}}}", "in takes a list"),
        ("never: {tool: x, where: {a: {oops: 1}}}", "exactly one of"),
        ("never: {tool: x, where: {a: {matches: '('}}}", "bad regular expression"),
        ("never: {tool: x}\n    formula: 'called(\"x\")'", "not both"),
    ])
    def test_malformed_never_says_what_is_wrong(self, tmp_path, bad, message):
        text = f"name: bad\nrules:\n  - name: r\n    {bad}\n"
        with pytest.raises(ValueError, match=message):
            rulebooks.load(write(tmp_path, text))


def test_now_with_is_step_local_and_round_trips():
    from agentltl import parse
    f = parse('G(now("pay") -> O(now_with("approve", role="finance", level=2)))')
    assert parse(str(f)) == f
    ok = [("approve", {"role": "finance", "level": 2}), ("pay", {})]
    assert verify_trace(metrics(ok), [Constraint("c", f)])["compliance_label"] == "FULL"
    other = [("approve", {"role": "finance"}), ("pay", {})]
    assert verify_trace(metrics(other), [Constraint("c", f)])["compliance_label"] == "VIOLATION"
