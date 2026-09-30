# -*- coding: utf-8 -*-
"""genv2/paramspec.py -- the parameter's own schema decides what may be asserted.

Every case here is a REAL parameter from one of the four benchmarks, with the real
description, because the whole point of the module is that it reads what the
benchmark actually ships. Invented descriptions would test a regex, not the rule.
"""
from __future__ import annotations

import pytest

from agentltl import paramspec as PS

# ── real schemas ─────────────────────────────────────────────────────────────
WC_MODE = {"type": "string",
           "description": "Mode of operation ('l' for lines, 'w' for words, "
                          "'c' for characters). ", "default": "l"}
TRAVEL_CLASS = {"type": "string",
                "description": "The class of the travel. Options are: economy, "
                               "business, first."}
ORDER_TYPE = {"type": "string", "description": "Type of the order (Buy/Sell)."}
TRAVEL_FROM = {"type": "string",
               "description": "The 3 letter code of the departing airport"}
TRAVEL_DATE = {"type": "string",
               "description": "The date of the travel in the format 'YYYY-MM-DD'"}
LS_A = {"type": "boolean",
        "description": "Show hidden files and directories. Defaults to False. ",
        "default": False}
PRICE = {"type": "float", "description": "Price at which to place the order."}
SYMBOL = {"type": "string", "description": "Symbol of the stock to trade."}
# WorkBench: 79 of 79 descriptions are title-case echoes of the parameter name.
WB_FIELD = {"type": "string", "description": "Field"}
WB_PLOT_TYPE = {"type": "string", "description": "Plot Type"}
WB_TIME_MIN = {"type": "string", "description": "Time Min"}
WB_NAME = {"type": "string", "description": "Name"}
# tau-bench ships real enums.
TAU_CABIN = {"type": "string", "enum": ["basic_economy", "economy", "business"]}
# mab/FHIR: a real sentence, so informative even with no enum.
FHIR_SUBJECT = {"type": "string",
                "description": "The patient FHIR ID for whom the observation is about."}


class TestIsInformative:
    @pytest.mark.parametrize("name,pd", [
        ("mode", WC_MODE), ("travel_class", TRAVEL_CLASS), ("a", LS_A),
        ("cabin", TAU_CABIN), ("subject.reference", FHIR_SUBJECT),
    ])
    def test_a_real_description_is_informative(self, name, pd):
        assert PS.is_informative(name, pd)

    @pytest.mark.parametrize("name,pd", [
        ("field", WB_FIELD), ("plot_type", WB_PLOT_TYPE),
        ("time_min", WB_TIME_MIN), ("name", WB_NAME),
    ])
    def test_a_title_case_echo_of_the_name_is_not(self, name, pd):
        """WorkBench generates these for every parameter. They have the shape of
        documentation and none of the content; asserting against one asserts
        against nothing, and that class is 1812 of run 6's false-alarm blocks."""
        assert not PS.is_informative(name, pd)


class TestOptions:
    def test_quoted_members_in_a_parenthetical(self):
        assert PS.options("mode", WC_MODE) == ["l", "w", "c"]

    def test_options_are_colon_list(self):
        assert PS.options("travel_class", TRAVEL_CLASS) == \
            ["economy", "business", "first"]

    def test_slash_separated_parenthetical(self):
        assert PS.options("order_type", ORDER_TYPE) == ["Buy", "Sell"]

    def test_a_formal_enum_wins(self):
        assert PS.options("cabin", TAU_CABIN) == \
            ["basic_economy", "economy", "business"]

    @pytest.mark.parametrize("name,pd", [
        ("travel_from", TRAVEL_FROM),   # a format, not a member list
        ("symbol", SYMBOL),             # a plain gloss
        ("field", WB_FIELD),            # uninformative
        ("plot_type", WB_PLOT_TYPE),
    ])
    def test_no_options_where_none_are_stated(self, name, pd):
        assert PS.options(name, pd) is None


class TestAssertable:
    def test_a_member_may_be_asserted(self):
        ok, _ = PS.assertable("travel_class", TRAVEL_CLASS, "business")
        assert ok

    def test_a_value_naming_no_member_may_not_be_asserted(self):
        """`mode` asserted as the prose 'line count' blocked the correct `wc`."""
        ok, why = PS.assertable("mode", WC_MODE, "line count")
        assert not ok and "not one of" in why

    def test_prose_that_DENOTES_a_member_stays_assertable(self):
        """'business class' names the member `business`, so it is resolvable, not
        unassertable -- and the emitter must then canonicalise it. Dropping it
        would shed a false alarm and the constraint with it; canonicalising keeps
        a constraint that is now correct. See the invariant test below."""
        assert PS.assertable("travel_class", TRAVEL_CLASS, "business class")[0]

    def test_assertable_via_options_implies_canonicalisable_to_a_member(self):
        """THE invariant the emitter depends on. If `assertable` says yes because
        the value denotes a member, `canonicalise` must hand back that member --
        otherwise the emitter writes the prose spelling into the constraint and we
        are back to blocking the correct call."""
        for name, pd, val in (("travel_class", TRAVEL_CLASS, "business class"),
                              ("travel_class", TRAVEL_CLASS, "economy"),
                              ("order_type", ORDER_TYPE, "Buy"),
                              ("mode", WC_MODE, "c"),
                              ("cabin", TAU_CABIN, "business")):
            ok, why = PS.assertable(name, pd, val)
            assert ok, (name, val, why)
            cv, _ = PS.canonicalise(name, pd, val)
            assert cv in PS.options(name, pd), (name, val, cv)

    def test_a_boolean_flag_is_not_its_cli_spelling(self):
        ok, why = PS.assertable("a", LS_A, "-a")
        assert not ok and "boolean" in why

    def test_a_city_name_is_not_a_three_letter_code(self):
        ok, why = PS.assertable("travel_from", TRAVEL_FROM, "Rivermist")
        assert not ok and "letter_code" in why

    def test_the_code_itself_is_assertable(self):
        ok, _ = PS.assertable("travel_from", TRAVEL_FROM, "RMS")
        assert ok

    def test_a_stated_date_format_is_checked(self):
        assert PS.assertable("travel_date", TRAVEL_DATE, "2024-05-01")[0]
        assert not PS.assertable("travel_date", TRAVEL_DATE, "next Tuesday")[0]

    @pytest.mark.parametrize("name,pd,val", [
        ("field", WB_FIELD, "name"),
        ("plot_type", WB_PLOT_TYPE, "distribution"),
        ("name", WB_NAME, "Akira Kimura"),
    ])
    def test_nothing_is_assertable_where_the_schema_says_nothing(self, name, pd, val):
        ok, why = PS.assertable(name, pd, val)
        assert not ok and "states nothing" in why

    def test_an_informative_string_description_still_permits_assertion(self):
        """mab is the one family where blocking WINS (307W/8L), and its values are
        quoted prose against real FHIR descriptions. The rule must not touch it."""
        ok, why = PS.assertable("subject.reference", FHIR_SUBJECT, "Patient/S123")
        assert ok, why

    def test_a_numeric_slot_keeps_a_numeric_value(self):
        """`fuelAmount` is a float and the stated 38 IS a float, so the rule leaves
        it alone. Computed arguments are a different defect (the tank capacity
        caps the request) and must not be smuggled in here."""
        assert PS.assertable("price", PRICE, 38)[0]

    def test_a_structured_value_is_not_a_scalar_binding(self):
        assert PS.assertable("note", FHIR_SUBJECT, {"text": "hi"})[0]

    def test_no_schema_leaves_behaviour_unchanged(self):
        assert PS.assertable("whatever", None, "anything")[0]


class TestCanonicalise:
    def test_prose_maps_onto_the_member(self):
        """The recall half: keeping `business` beats dropping 'business class'.
        Measured on run 6, 307 of 474 canonicalised bfcl values equal exactly what
        the agent actually sent."""
        assert PS.canonicalise("travel_class", TRAVEL_CLASS, "business class") == \
            ("business", True)

    def test_a_cli_flag_becomes_the_boolean(self):
        assert PS.canonicalise("a", LS_A, "true") == (True, True)

    def test_an_ambiguous_value_is_left_alone(self):
        """'count' names no member of ['l','w','c'] -- a one-character member must
        never be matched inside a word, or 'c' hits 'business class'."""
        assert PS.canonicalise("mode", WC_MODE, "count") == ("count", False)

    def test_an_exact_member_is_unchanged(self):
        assert PS.canonicalise("travel_class", TRAVEL_CLASS, "economy") == \
            ("economy", False)

    def test_nothing_to_canonicalise_without_options(self):
        assert PS.canonicalise("name", WB_NAME, "Akira") == ("Akira", False)


ZIPCODE = {"type": "string", "description": "The zipcode of the first city."}


class TestZipcode:
    """`estimate_distance` documents both parameters as zipcodes while the task
    names cities, so the prose value is never the argument -- 50 of run 6's
    remaining blocking failures on correct traces are these two parameters."""

    def test_a_city_name_is_not_a_zipcode(self):
        ok, why = PS.assertable("cityA", ZIPCODE, "San Francisco")
        assert not ok and "zipcode" in why

    def test_a_zipcode_is(self):
        assert PS.assertable("cityA", ZIPCODE, "94016")[0]

    def test_zip_plus_four(self):
        assert PS.assertable("cityA", ZIPCODE, "94016-1234")[0]


# ── the format may be documented on a SIBLING tool ───────────────────────────
# THE LARGEST REMAINING FALSE-ALARM CLASS after the own-schema rule landed. bfcl
# still had 386 blocking failures on run 6's CORRECT traces and 92% were this:
#
#     book_flight.travel_class     "The class of the travel"
#     get_flight_cost.travel_class "The class of the travel. Options are: economy,
#                                   business, first."
#
# Same parameter name, same tool family, documented on one and not the other. So
# `book_flight.travel_class` had no options, the prose 'business class' was asserted,
# and the agent's correct 'business' was blocked. Reading the sibling is NOT a leak:
# it is in the same AgentView the agent reads.
#
# Measured: bfcl 386 -> 271 blocking failures on correct traces (-30%), every other
# family byte-identical.
_OWN_VAGUE = {"type": "string", "description": "The class of the travel"}
_SIB_RICH = {"type": "string",
             "description": "The class of the travel. Options are: economy, "
                            "business, first."}
_SIB_FMT = {"type": "string",
            "description": "The 3 letter code of the departing airport"}


def test_the_flag_is_off_by_default_so_the_corpus_is_unchanged(monkeypatch):
    """Like GRAPH_DECLARED this changes what the corpus ASSERTS, so it must be
    opt-in and the off-path must be a byte-identical no-op."""
    monkeypatch.setattr(PS, "SIBLING", False)
    assert PS.enrich("travel_class", _OWN_VAGUE, [_SIB_RICH]) == _OWN_VAGUE


def test_a_sibling_option_list_is_borrowed_when_this_schema_states_none(monkeypatch):
    monkeypatch.setattr(PS, "SIBLING", True)
    out = PS.enrich("travel_class", _OWN_VAGUE, [_SIB_RICH])
    assert PS.options("travel_class", out) == ["economy", "business", "first"]


def test_the_borrowed_list_canonicalises_the_prose_the_task_states(monkeypatch):
    """The point of the fix: keep the constraint and make it CORRECT, rather than
    dropping it. 'business class' denotes the member 'business'."""
    monkeypatch.setattr(PS, "SIBLING", True)
    out = PS.enrich("travel_class", _OWN_VAGUE, [_SIB_RICH])
    value, changed = PS.canonicalise("travel_class", out, "business class")
    assert (value, changed) == ("business", True)


def test_a_sibling_format_is_borrowed_too(monkeypatch):
    monkeypatch.setattr(PS, "SIBLING", True)
    own = {"type": "string", "description": "The location the travel is from"}
    out = PS.enrich("travel_from", own, [_SIB_FMT])
    assert PS.format_hint("travel_from", out) == ("letter_code", "3")
    # and the city name the task states is then NOT assertable against it
    ok, _why = PS.assertable("travel_from", out, "Rivermist")
    assert not ok


def test_this_schema_always_WINS_over_a_sibling(monkeypatch):
    """Borrow only where our own schema states nothing -- never override it."""
    monkeypatch.setattr(PS, "SIBLING", True)
    own = {"type": "string", "enum": ["a", "b"], "description": "own list"}
    out = PS.enrich("x", own, [_SIB_RICH])
    assert PS.options("x", out) == ["a", "b"]


def test_an_unrelated_sibling_list_can_only_DROP_a_constraint(monkeypatch):
    """The guard against a wrong borrow is canonicalise, not a similarity test: a
    borrowed list can rewrite a stated value to a member it genuinely denotes, or
    fail to and drop the assertion. It cannot invent a value the prose never names,
    so a wrong borrow costs a constraint and never creates a false alarm."""
    monkeypatch.setattr(PS, "SIBLING", True)
    unrelated = {"type": "string", "description": "Options are: red, green, blue."}
    out = PS.enrich("travel_class", _OWN_VAGUE, [unrelated])
    value, changed = PS.canonicalise("travel_class", out, "business class")
    assert not changed and value == "business class"
    ok, _why = PS.assertable("travel_class", out, "business class")
    assert not ok, "an unmatched value must be dropped, not asserted"


def test_no_siblings_is_a_no_op(monkeypatch):
    monkeypatch.setattr(PS, "SIBLING", True)
    assert PS.enrich("travel_class", _OWN_VAGUE, []) == _OWN_VAGUE
    assert PS.enrich("travel_class", _OWN_VAGUE, None) == _OWN_VAGUE
