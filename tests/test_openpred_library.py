# -*- coding: utf-8 -*-
"""The closed library: selection over templates, not free composition.

Free composition produced 119 expressions in exactly 2 shapes -- no expressive
diversity -- while getting both subtly wrong: a fabricated `2.0` cited from the span
"1.5 to 1.9", and the omission, in all 89 instances, of the recency guard task5
states. These tests pin the two properties that made the difference.
"""
from __future__ import annotations

import pytest

from agentltl.openpred import library as L
from agentltl.openpred import expr as X

TASK5 = ("Check patient S6315806's last serum magnesium level within last 24 hours. "
         "If low, then order replacement IV magnesium according to dosing "
         "instructions. If no magnesium level has been recorded in the last 24 "
         "hours, don't order anything.")
POLICY5 = "serum magnesium 1.5 to 1.9 mg/dL: IV: 1 g over 1 hour"
TASK10 = ("Check when the patient's last HbA1C was measured. If it has been more "
          "than 1 year, order a new HbA1C lab test.")


def test_a_fabricated_threshold_is_refused():
    """2.0 is the model's rounding of 1.9. The span is real; the value is not."""
    ast, why = L.bind("value_below_then_required",
                      {"producer": "GET_Observation_labs",
                       "consumer": "POST_MedicationRequest",
                       "threshold": 2.0,
                       "threshold_span": "serum magnesium 1.5 to 1.9 mg/dL",
                       "window": "last 24 hours"},
                      TASK5 + "\n" + POLICY5)
    assert ast is None
    assert "does not appear" in why


def test_a_stated_threshold_is_accepted():
    """3.5 IS written in task9's policy, so asserting it is legitimate."""
    ast, why = L.bind("value_below_then_required",
                      {"producer": "GET_Observation_labs",
                       "consumer": "POST_MedicationRequest",
                       "threshold": 3.5, "threshold_span": "goal of 3.5"},
                      "replace potassium to a goal of 3.5 serum level")
    assert ast is not None, why


def test_omitting_the_threshold_falls_back_to_the_dosing_table():
    """Where no cutoff is written down, the bands say which levels are treatable --
    so no model-authored number is needed at all."""
    ast, why = L.bind("value_below_then_required",
                      {"producer": "GET_Observation_labs",
                       "consumer": "POST_MedicationRequest",
                       "window": "last 24 hours"},
                      TASK5 + "\n" + POLICY5)
    assert ast is not None, why
    assert "table" in X.accessors(ast)


def test_the_window_guard_is_always_present_when_the_task_states_one():
    """The conjunct the model omitted 89 times out of 89.

    Measured: adding it took task5's constraint from fa=21/catch=4 to fa=0/catch=1.
    """
    ast, why = L.bind("value_below_then_required",
                      {"producer": "GET_Observation_labs",
                       "consumer": "POST_MedicationRequest",
                       "window": "last 24 hours"},
                      TASK5 + "\n" + POLICY5)
    assert ast is not None, why
    assert "when" in X.accessors(ast), "no recency guard was built"
    trigger = ast["args"][0]
    assert trigger["op"] == "and", "the guard must be conjoined with the value test"


def test_a_window_the_task_never_states_is_refused():
    ast, why = L.bind("value_below_then_required",
                      {"producer": "GET_Observation_labs",
                       "consumer": "POST_MedicationRequest",
                       "window": "48 hours"},
                      TASK5 + "\n" + POLICY5)
    assert ast is None
    assert "does not say" in why


def test_stale_template_requires_a_duration_the_task_names():
    ok, _ = L.bind("stale_then_required",
                   {"producer": "GET_Observation_labs",
                    "consumer": "POST_ServiceRequest", "max_age": "1 year"}, TASK10)
    assert ok is not None
    bad, why = L.bind("stale_then_required",
                      {"producer": "GET_Observation_labs",
                       "consumer": "POST_ServiceRequest", "max_age": "one week"},
                      TASK10)
    assert bad is None and "does not say" in why


def test_an_unknown_template_names_the_alternatives():
    ast, why = L.bind("do_the_right_thing", {"producer": "a", "consumer": "b"}, "x")
    assert ast is None
    assert all(n in why for n in L.TEMPLATES), "the menu must be in the rejection"


def test_every_template_builds_a_well_formed_boolean_expression():
    """The root must be boolean and the AST must pass the grammar's own validator."""
    for name, args in (
        ("value_below_then_required",
         {"producer": "GET_Observation_labs", "consumer": "POST_MedicationRequest",
          "threshold": 3.5, "threshold_span": "goal of 3.5"}),
        ("stale_then_required",
         {"producer": "GET_Observation_labs", "consumer": "POST_ServiceRequest",
          "max_age": "1 year"}),
    ):
        text = "goal of 3.5 serum level, or more than 1 year since the last test"
        ast, why = L.bind(name, args, text)
        assert ast is not None, f"{name}: {why}"
        ok, vwhy, _ = X.check_shape(ast)
        assert ok, f"{name}: {vwhy}"


def test_both_tools_are_required():
    for args in ({"consumer": "b"}, {"producer": "a"}, {}):
        ast, why = L.bind("stale_then_required", dict(args, max_age="1 year"), TASK10)
        assert ast is None and "required" in why


def test_number_matching_is_trailing_zero_tolerant_but_not_lenient():
    """`2` and `2.0` are one number; `2.0` must never be satisfied by "1.9"."""
    assert L._num_in_text(2.0, "give 2 grams")
    assert L._num_in_text(2.0, "give 2.0 grams")
    assert not L._num_in_text(2.0, "serum magnesium 1.5 to 1.9 mg/dL")
    assert not L._num_in_text(1.0, "value is 1.5"), "must not match inside 1.5"


def test_the_menu_is_serialisable_and_names_every_parameter():
    import json
    menu = L.menu()
    json.dumps(menu)
    assert {m["template"] for m in menu} == set(L.TEMPLATES)
    for m in menu:
        assert m["parameters"] and m["when_to_use"]
