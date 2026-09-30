# -*- coding: utf-8 -*-
"""genv4/expr.py -- the open-predicate grammar, interpreter and validator.

The three properties that must hold are the ones the rest of the pipeline learned the
hard way, so each has its own class here:

  * ABSENT IS NOT FALSE. Unevaluable propagates and passes at the root.
  * EVERY OPERATOR IS TOTAL. These run inside Predicate.fn during enforcement, where
    a raise aborts the episode -- and an aborted episode scored as incorrect blames
    whichever enforcement arm hit it.
  * VALIDATION REJECTS WITH A REASON, because stating the reason is what made the G2
    interview converge.
"""
from __future__ import annotations

import json

import pytest

from ._view_fakes import AgentView, ToolSchema, UserTurn
from agentltl.openpred import expr as E


# ── fixtures: real tools, real trace shapes ─────────────────────────────────
def _call(tool, args, result=None):
    """The shape the runners actually record, verified against run 6's persisted
    traces: `arguments` is a DICT (847 occurrences) or absent (7927, falling back to
    `tool_args`), never a bare string. `tool_result` is always TEXT.

    Getting this wrong is not harmless. An earlier version of this fixture passed a
    JSON string, so no argument resolved, every expression came out UNEVALUABLE, and
    the tests asserting `passed is True` went green for entirely the wrong reason.
    """
    return {"tool_name": tool,
            "arguments": args,
            "tool_result": result if isinstance(result, str) or result is None
            else json.dumps(result)}


VEHICLE = ToolSchema(
    name="fillFuelTank", parameters=["fuelAmount"], required=["fuelAmount"],
    params={"fuelAmount": {"type": "float",
                           "description": "The amount of fuel to add in gallons."}})
VEHICLE_INFO = ToolSchema(
    name="get_vehicle_info", parameters=["display"], required=[],
    params={"display": {"type": "string", "description": "What to display."}})
FLIGHT = ToolSchema(
    name="get_flight_cost", parameters=["travel_class"], required=[],
    params={"travel_class": {"type": "string",
                             "description": "The class of the travel. Options are: "
                                            "economy, business, first."}})
WC = ToolSchema(
    name="wc", parameters=["file_name", "mode"], required=["file_name"],
    params={"file_name": {"type": "string", "description": "Name of the file."},
            "mode": {"type": "string",
                     "description": "Mode of operation ('l' for lines, 'w' for "
                                    "words, 'c' for characters). "}})


def _view(tools, text="Fill the tank with 40 gallons. Count the words with wc.",
          policy=None):
    return AgentView(instance_id="t", benchmark="bfcl",
                     turns=[UserTurn(turn=0, messages=[text])],
                     tools=list(tools), policy=policy, clock="2023-11-13T10:15:00")


# ═══════════════════════════════════════════════════════════════════════════
class TestAbsentIsNotFalse:
    def test_unevaluable_passes_at_the_root_with_its_reason(self):
        ast = {"op": "le", "args": [
            {"acc": "arg", "tool": "fillFuelTank", "path": "fuelAmount"},
            {"acc": "const", "value": 40, "span": "40"}]}
        got = E.evaluate(ast, trace=[])          # the tool was never called
        assert got["passed"] is True
        assert "unevaluable" in got["note"]
        assert "fillFuelTank" in got["note"]

    def test_a_missing_recorded_result_is_unevaluable_not_violated(self):
        """`_trace_to_metrics` DROPS tool_result on the bfcl and tau scoring paths, so
        `result()` is routinely unreadable. Reporting that as a violation would fail
        every correct run on those families."""
        ast = {"op": "le", "args": [
            {"acc": "arg", "tool": "fillFuelTank", "path": "fuelAmount"},
            {"acc": "result", "tool": "get_vehicle_info", "json": "tankCapacity"}]}
        trace = [_call("get_vehicle_info", {}, None),
                 _call("fillFuelTank", {"fuelAmount": 10})]
        got = E.evaluate(ast, trace)
        assert got["passed"] is True and "unevaluable" in got["note"]

    def test_uneval_refuses_to_be_coerced_to_a_bool(self):
        """The single most dangerous silent conversion in the module."""
        with pytest.raises(TypeError):
            bool(E.Uneval("no"))

    def test_kleene_and_a_definitive_false_beats_an_unevaluable_sibling(self):
        """`and` must be False as soon as one arm is False. Weakening this to
        'any unevaluable wins' would silently un-enforce a real violation."""
        false_arm = {"op": "eq", "args": [
            {"acc": "arg", "tool": "wc", "path": "mode"},
            {"acc": "const", "value": "c", "span": "'c' for characters"}]}
        uneval_arm = {"op": "eq", "args": [
            {"acc": "result", "tool": "get_vehicle_info", "json": "nope"},
            {"acc": "const", "value": 1, "span": "1"}]}
        trace = [_call("wc", {"file_name": "a.txt", "mode": "w"})]
        got = E.evaluate({"op": "and", "args": [false_arm, uneval_arm]}, trace)
        assert got["passed"] is False

    def test_kleene_or_a_definitive_true_beats_an_unevaluable_sibling(self):
        true_arm = {"op": "eq", "args": [
            {"acc": "arg", "tool": "wc", "path": "mode"},
            {"acc": "const", "value": "w", "span": "words"}]}
        uneval_arm = {"op": "eq", "args": [
            {"acc": "result", "tool": "get_vehicle_info", "json": "nope"},
            {"acc": "const", "value": 1, "span": "1"}]}
        trace = [_call("wc", {"file_name": "a.txt", "mode": "w"})]
        got = E.evaluate({"op": "or", "args": [true_arm, uneval_arm]}, trace)
        assert got["passed"] is True

    def test_implies_is_vacuously_satisfied_when_its_antecedent_is_false(self):
        ast = {"op": "implies", "args": [
            {"op": "eq", "args": [{"acc": "count", "tool": "wc"},
                                  {"acc": "const", "value": 99, "span": "99"}]},
            {"op": "eq", "args": [{"acc": "arg", "tool": "wc", "path": "mode"},
                                  {"acc": "const", "value": "zzz", "span": "zzz"}]}]}
        got = E.evaluate(ast, [_call("wc", {"file_name": "a.txt", "mode": "w"})])
        assert got["passed"] is True


class TestTotality:
    """Every operator returns a value or Uneval -- never raises."""

    @pytest.mark.parametrize("ast", [
        # division by zero
        {"op": "eq", "args": [
            {"op": "div", "args": [{"acc": "const", "value": 1, "span": "1"},
                                   {"acc": "const", "value": 0, "span": "0"}]},
            {"acc": "const", "value": 1, "span": "1"}]},
        # len of a scalar
        {"op": "eq", "args": [
            {"op": "len", "args": [{"acc": "const", "value": 3, "span": "3"}]},
            {"acc": "const", "value": 1, "span": "1"}]},
        # ordering prose against a number
        {"op": "lt", "args": [{"acc": "const", "value": "soon", "span": "soon"},
                              {"acc": "const", "value": 3, "span": "3"}]},
        # a bad regex
        {"op": "matches", "args": [{"acc": "const", "value": "x", "span": "x"},
                                   {"acc": "const", "value": "([", "span": "(["}]},
        # 'in' with no list on the right
        {"op": "in", "args": [{"acc": "const", "value": "x", "span": "x"},
                              {"acc": "const", "value": 5, "span": "5"}]},
    ])
    def test_never_raises_and_degrades_to_unevaluable(self, ast):
        got = E.evaluate(ast, [])
        assert got["passed"] is True and "unevaluable" in got["note"]

    def test_a_boolean_is_not_a_number(self):
        """`ls.a` is a boolean and the task writes '-a'. If True silently compared
        equal to 1, a flag would become a quantity."""
        ast = {"op": "lt", "args": [{"acc": "const", "value": True, "span": "-a"},
                                    {"acc": "const", "value": 5, "span": "5"}]}
        got = E.evaluate(ast, [])
        assert got["passed"] is True and "unevaluable" in got["note"]


class TestShape:
    def test_the_root_must_be_boolean(self):
        ok, why = E.validate({"acc": "count", "tool": "wc"}, _view([WC]))
        assert not ok and "must be a boolean" in why

    def test_unknown_operator_is_named(self):
        ok, why, _ = E.check_shape({"op": "frobnicate", "args": []})
        assert not ok and "frobnicate" in why

    def test_arity_is_checked(self):
        ok, why, _ = E.check_shape(
            {"op": "sub", "args": [{"acc": "const", "value": 1, "span": "1"}]})
        assert not ok and "takes 2" in why

    def test_depth_is_bounded(self):
        node = {"acc": "const", "value": 1, "span": "1"}
        for _ in range(E.MAX_DEPTH + 2):
            node = {"op": "abs", "args": [node]}
        ok, why, _ = E.check_shape({"op": "eq", "args": [
            node, {"acc": "const", "value": 1, "span": "1"}]})
        assert not ok and "deeper than" in why

    def test_arithmetic_rejects_a_boolean_operand(self):
        ok, why, _ = E.check_shape({"op": "add", "args": [
            {"op": "eq", "args": [{"acc": "const", "value": 1, "span": "1"},
                                  {"acc": "const", "value": 1, "span": "1"}]},
            {"acc": "const", "value": 1, "span": "1"}]})
        assert not ok and "not a boolean" in why

    def test_a_connective_rejects_a_number_operand(self):
        ok, why, _ = E.check_shape({"op": "and", "args": [
            {"op": "add", "args": [{"acc": "const", "value": 1, "span": "1"},
                                   {"acc": "const", "value": 1, "span": "1"}]},
            {"acc": "const", "value": 1, "span": "1"}]})
        assert not ok and "not a number" in why

    def test_a_const_without_a_span_cannot_be_audited(self):
        ok, why, _ = E.check_shape({"acc": "const", "value": 7})
        assert not ok and "span" in why


class TestValidateAgainstTheView:
    def test_an_unknown_tool_is_rejected_and_the_real_ones_listed(self):
        ast = {"op": "eq", "args": [
            {"acc": "arg", "tool": "nope", "path": "x"},
            {"acc": "const", "value": 1, "span": "40"}]}
        ok, why = E.validate(ast, _view([WC, VEHICLE]))
        assert not ok and "no tool called" in why and "wc" in why

    def test_an_undeclared_parameter_is_rejected_and_the_real_ones_listed(self):
        ast = {"op": "eq", "args": [
            {"acc": "arg", "tool": "wc", "path": "tankCapacity"},
            {"acc": "const", "value": 1, "span": "40"}]}
        ok, why = E.validate(ast, _view([WC]))
        assert not ok and "not a declared parameter" in why and "file_name" in why

    def test_a_const_whose_span_is_not_in_the_text_is_rejected(self):
        ast = {"op": "eq", "args": [
            {"acc": "arg", "tool": "wc", "path": "file_name"},
            {"acc": "const", "value": "secret.txt", "span": "the secret file"}]}
        ok, why = E.validate(ast, _view([WC]))
        assert not ok and "does not appear" in why

    def test_a_prose_value_is_rejected_against_a_schema_constrained_arg(self):
        """The class removed at real cost (1059 -> 648 blocking failures on correct
        traces). Open predicates must not reintroduce it."""
        ast = {"op": "eq", "args": [
            {"acc": "arg", "tool": "wc", "path": "mode"},
            {"acc": "const", "value": "count", "span": "Count the words"}]}
        ok, why = E.validate(ast, _view([WC]))
        assert not ok and "cannot be compared" in why

    def test_an_exact_member_is_accepted_unchanged(self):
        ast = {"op": "eq", "args": [
            {"acc": "arg", "tool": "wc", "path": "mode"},
            {"acc": "const", "value": "w", "span": "Count the words"}]}
        ok, why = E.validate(ast, _view([WC]))
        assert ok, why
        assert ast["args"][1]["value"] == "w"

    def test_a_denoting_value_is_canonicalised_in_place(self):
        """'business class' denotes the member `business`, so the validator rewrites
        it rather than rejecting it -- keeping a constraint that is now CORRECT
        instead of shedding the false alarm and the constraint with it."""
        ast = {"op": "eq", "args": [
            {"acc": "arg", "tool": "get_flight_cost", "path": "travel_class"},
            {"acc": "const", "value": "business class", "span": "business class"}]}
        ok, why = E.validate(
            ast, _view([FLIGHT], text="Book me a business class seat."))
        assert ok, why
        assert ast["args"][1]["value"] == "business"

    def test_a_one_character_member_is_never_loose_matched(self):
        """Deliberate in paramspec: matching 'c' inside a word would hit
        'business class'. So a value that merely CONTAINS a member is rejected."""
        ast = {"op": "eq", "args": [
            {"acc": "arg", "tool": "wc", "path": "mode"},
            {"acc": "const", "value": "'w' for words", "span": "Count the words"}]}
        ok, why = E.validate(ast, _view([WC]))
        assert not ok and "not one of" in why

    def test_a_result_no_tool_produces_can_never_be_known(self):
        ast = {"op": "le", "args": [
            {"acc": "arg", "tool": "fillFuelTank", "path": "fuelAmount"},
            {"acc": "result", "tool": "get_vehicle_info", "json": "tankCapacity"}]}
        ok, why = E.validate(ast, _view([VEHICLE, VEHICLE_INFO]), produced_tools=[])
        assert not ok and "can never be known" in why


class TestDisposition:
    def test_only_args_is_decidable_at_the_call(self):
        ast = {"op": "eq", "args": [
            {"acc": "arg", "tool": "wc", "path": "mode"},
            {"acc": "const", "value": "c", "span": "characters"}]}
        assert E.accessors(ast) == ["arg", "const"]
        assert not E.count_under_lower_bound(ast)

    def test_a_count_under_a_lower_bound_is_liveness(self):
        """No finite prefix falsifies it -- the call might still come. Blocking it
        would stop an agent for not having acted YET, which is why L1 cannot block."""
        ast = {"op": "ge", "args": [{"acc": "count", "tool": "wc"},
                                    {"acc": "const", "value": 2, "span": "twice"}]}
        assert E.count_under_lower_bound(ast)

    def test_a_cap_is_not_liveness(self):
        ast = {"op": "le", "args": [{"acc": "count", "tool": "wc"},
                                    {"acc": "const", "value": 1, "span": "once"}]}
        assert not E.count_under_lower_bound(ast)

    def test_the_guarded_tool_is_the_one_whose_args_are_read(self):
        ast = {"op": "le", "args": [
            {"acc": "arg", "tool": "fillFuelTank", "path": "fuelAmount"},
            {"acc": "result", "tool": "get_vehicle_info", "json": "tankCapacity"}]}
        assert E.reads_args_of(ast) == "fillFuelTank"


class TestTheMeasuredCases:
    """The two obligations that motivated open predicates, end to end."""

    FUEL = {"op": "le", "args": [
        {"acc": "arg", "tool": "fillFuelTank", "path": "fuelAmount"},
        {"op": "sub", "args": [
            {"acc": "result", "tool": "get_vehicle_info", "json": "tankCapacity"},
            {"acc": "result", "tool": "get_vehicle_info", "json": "fuelLevel"}]}]}

    def _trace(self, amount):
        return [_call("get_vehicle_info", {},
                      {"tankCapacity": 50.0, "fuelLevel": 39.43}),
                _call("fillFuelTank", {"fuelAmount": amount})]

    def test_the_correct_call_passes_where_the_literal_40_blocked_it(self):
        """39 blocks on correct episodes in run 6 came from asserting the stated 40
        against the 10.57 the tank actually took."""
        assert E.evaluate(self.FUEL, self._trace(10.57))["passed"] is True

    def test_overfilling_is_caught(self):
        got = E.evaluate(self.FUEL, self._trace(40))
        assert got["passed"] is False

    def test_the_message_names_the_subterm_and_both_values(self):
        """A block the agent cannot act on destroys the action: task8 was blocked five
        times and ended with no write at all."""
        note = E.evaluate(self.FUEL, self._trace(40))["note"]
        assert "fillFuelTank.fuelAmount" in note
        assert "40" in note and "10.57" in note.replace("10.570000000000004", "10.57")

    def test_a_body_key_holding_a_JSON_STRING_still_resolves(self):
        """The documented wire hazard: a POST body reaches the ENFORCER as the raw
        JSON string the agent sent, so every L2/L4 constraint on a POST field once
        matched nothing at enforcement time while post-hoc scoring saw a dict. This
        must be a real resolution, so the assertion is on FALSE -- a shape that
        failed to resolve would come back unevaluable and pass vacuously.
        """
        body = json.dumps({"fuelAmount": 40})
        trace = [_call("get_vehicle_info", {},
                       {"tankCapacity": 50.0, "fuelLevel": 39.43}),
                 _call("fillFuelTank", {"payload": body})]
        got = E.evaluate(self.FUEL, trace)
        assert got["passed"] is False, got["note"]


class TestBuild:
    def test_build_returns_a_predicate_that_never_raises(self):
        import agentltl as al
        A = {n: getattr(al, n) for n in ("Predicate", "Constraint")}
        pred = E.build(TestTheMeasuredCases.FUEL, A)
        out = pred.fn([], None)
        assert out["passed"] is True and "unevaluable" in out["note"]


class TestReadingAProducersOutput:
    """Two defects found by hand-inspecting the first predicates the model actually
    produced. Both made `result()` -- the accessor that gives open predicates their
    power -- permanently unevaluable on a real FHIR bundle."""

    STALE = {"op": "implies", "args": [
        {"op": "gt", "args": [
            {"op": "sub", "args": [{"acc": "clock"},
                                   {"acc": "result", "tool": "labs",
                                    "json": "effectiveDateTime"}]},
            {"acc": "const", "value": 31536000, "span": "1 year"}]},
        {"op": "ge", "args": [{"acc": "count", "tool": "POST_ServiceRequest"},
                              {"acc": "const", "value": 1, "span": "order"}]}]}

    @staticmethod
    def _labs(when):
        return [_call("labs", {"patient": "S1"},
                      {"entry": [{"resource": {"effectiveDateTime": when,
                                               "value": {"value": 7.2}}}]})]

    CLOCK = "2023-11-13T10:15:00+00:00"

    def test_a_nested_value_is_found_by_leaf_key(self):
        """The AgentView carries tool PARAMETERS, never the shape of what a tool
        RETURNS, so the only honest thing the model can name is the leaf. Requiring
        an exact path (`entry[].resource.effectiveDateTime`) made this unevaluable."""
        got = E.evaluate(self.STALE, self._labs("2021-05-01T00:00:00+00:00"),
                         clock=self.CLOCK)
        assert got["passed"] is False, got["note"]

    def test_two_instants_subtract_to_seconds(self):
        """'How long ago' is the whole point of a staleness rule. Without datetime
        subtraction, `clock - effectiveDateTime` coerced through _num and came back
        unevaluable, so the constraint could never fire."""
        ast = {"op": "gt", "args": [
            {"op": "sub", "args": [{"acc": "clock"},
                                   {"acc": "result", "tool": "labs",
                                    "json": "effectiveDateTime"}]},
            {"acc": "const", "value": 31536000, "span": "1 year"}]}
        assert E.evaluate(ast, self._labs("2021-05-01T00:00:00+00:00"),
                          clock=self.CLOCK)["passed"] is True
        assert E.evaluate(ast, self._labs("2023-10-01T00:00:00+00:00"),
                          clock=self.CLOCK)["passed"] is False

    def test_a_datetime_is_still_not_a_number_for_other_operators(self):
        """Only `sub` gets instant arithmetic. A datetime plus a number is not a
        meaningful quantity, and coercing it would be the boolean-is-not-a-number
        mistake in another costume."""
        ast = {"op": "gt", "args": [
            {"op": "add", "args": [{"acc": "clock"},
                                   {"acc": "const", "value": 5, "span": "5"}]},
            {"acc": "const", "value": 1, "span": "1"}]}
        got = E.evaluate(ast, [], clock=self.CLOCK)
        assert got["passed"] is True and "unevaluable" in got["note"]

    def test_a_correct_ABSTENTION_is_not_punished(self):
        """The failure that took ObservedThenRequired off by default: it moved
        blindness 28.6%->26.7% while moving false alarms 7.2%->18.1%, because
        deciding 'the value triggers' used a hardcoded extraction path and a normal
        patient read as low. Here the trigger is explicit, so a fresh lab with no
        order passes."""
        got = E.evaluate(self.STALE, self._labs("2023-10-01T00:00:00+00:00"),
                         clock=self.CLOCK)
        assert got["passed"] is True

    def test_the_message_does_not_say_1_is_1(self):
        note = E.evaluate(self.STALE, self._labs("2021-05-01T00:00:00+00:00"),
                          clock=self.CLOCK)["note"]
        assert "1 is 1" not in note
        assert "POST_ServiceRequest calls is 0" in note
