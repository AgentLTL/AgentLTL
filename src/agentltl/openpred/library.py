# -*- coding: utf-8 -*-
"""A CLOSED LIBRARY of parameterised open predicates -- selection, not composition.

WHY THIS EXISTS, MEASURED
-------------------------
G4b let the model compose freely over the grammar. Across 300 mab instances it
produced **119 expressions in exactly 2 distinct shapes**:

    89  implies(lt(result, const),            ge(count, const))
    30  implies(gt(sub(clock, result), const), ge(count, const))

So the freedom bought no expressive diversity at all -- it re-derived the same two
templates 119 times. What it *did* buy was a way to get them subtly wrong, in two
ways that a template cannot be wrong in:

1. **A fabricated constant with a real span.** All 29 task5 expressions asserted
   `< 2.0` citing the span "serum magnesium 1.5 to 1.9 mg/dL". 2.0 appears nowhere;
   the model rounded 1.9 up. The leak gate passes it, because the span is genuine --
   this is the documented binding-not-leakage class.

2. **A GUARD THE TASK STATES AND THE MODEL OMITTED.** task5 reads "Check the last
   serum magnesium level **within last 24 hours**. If low, order replacement. **If no
   level has been recorded in the last 24 hours, don't order anything.**" Every one of
   the 89 expressions tested the value and never the recency, so a stale-but-low
   reading demanded an order for a patient whose correct treatment is none.

   Measured on the stored traces, adding that one conjunct moves task5's constraint
   from **fa=21 / catch=4** to **fa=0 / catch=1** -- every false alarm gone, and it
   still catches. `fa` = fires on a CORRECT episode, `catch` = fires on a wrong one.

A template carries the guard once, in code, and the model cannot forget it. The
model's job shrinks to the question it answers well: does this template apply here,
and what are its parameters. That is the same lesson as "give the model the real
options or it answers a different question" -- offered a column list it answers
correctly, asked to invent one it says "dose".

WHAT THE MODEL MAY STILL DECIDE
-------------------------------
Which template (or none), the producer and consumer tools, and any threshold -- but a
threshold must be a number that appears VERBATIM in the task or policy, or be omitted
so `band` supplies it from the dose table. `bind()` refuses a value it cannot find in
the text, which is precisely what would have rejected the fabricated 2.0.

Every emitted constraint keeps `author="llm"` (the SELECTION is still the model's, and
the tier must stay separately measurable and droppable) plus `template=<name>`, so a
single template can be retired on its own W/L record.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Seconds, for the durations these tasks state. Spans are checked against the text,
# so a template may only offer a window the task actually names.
_DURATIONS = {
    "24 hours": 86400, "24 hour": 86400, "last 24 hours": 86400,
    "48 hours": 172800, "one week": 604800, "1 week": 604800,
    "1 year": 31536000, "one year": 31536000, "12 months": 31536000,
}


def _num_in_text(value: float, text: str) -> bool:
    """Is this exact number written in the text? Trailing-zero tolerant, not lenient.

    `2.0` must NOT be satisfied by "1.9", which is the whole point; but `2` and `2.0`
    are the same number differently spelled and both count.
    """
    cands = {f"{value:g}", str(value)}
    if float(value).is_integer():
        cands.add(str(int(value)))
    # The trailing lookahead must exclude a DOT as well as a digit: without it the
    # integer spelling "1" matched the "1" inside "1.5", so a threshold of 1.0 counted
    # as stated by a text that only says 1.5 -- leniency in the one direction that
    # lets a fabricated constant through, which is what this guard exists to stop.
    return any(re.search(rf"(?<![\d.]){re.escape(c)}(?![\d.])", text) for c in cands)


class Template:
    """One parameterised predicate. `build` returns an AST; nothing else may."""

    def __init__(self, name: str, question: str, params: Dict[str, str], builder) -> None:
        self.name = name
        self.question = question
        self.params = params
        self._builder = builder

    def build(self, **kw) -> dict:
        return self._builder(**kw)

    def menu_entry(self) -> dict:
        return {"template": self.name, "when_to_use": self.question,
                "parameters": self.params}


# ── the two templates the corpus actually asked for ───────────────────────────

def _fresh_guard(producer: str, window_s: int, window_span: str) -> dict:
    """The reading is recent enough to act on. `when` pairs with the value's own row."""
    return {"op": "le", "args": [
        {"op": "sub", "args": [{"acc": "clock"},
                               {"acc": "when", "tool": producer}]},
        {"acc": "const", "value": window_s, "span": window_span}]}


def _build_value_below(producer: str, consumer: str, threshold: Optional[float],
                       threshold_span: str = "", window_s: Optional[int] = None,
                       window_span: str = "", column: str = "value") -> dict:
    """If a FRESH reading is below the threshold, the action is owed.

    With no `threshold`, the trigger is band membership: the dose table's own bands
    say which levels are treatable, so no model-authored number is needed at all.
    """
    if threshold is None:
        low = {"op": "gt", "args": [
            {"acc": "table", "column": column, "from": producer},
            {"acc": "const", "value": 0, "span": threshold_span or "dosing instructions"}]}
    else:
        low = {"op": "lt", "args": [
            {"acc": "result", "tool": producer, "json": "valueQuantity.value",
             "which": "last"},
            {"acc": "const", "value": threshold, "span": threshold_span}]}

    trigger = low if window_s is None else {
        "op": "and", "args": [_fresh_guard(producer, window_s, window_span), low]}

    return {"op": "implies", "args": [
        trigger,
        {"op": "ge", "args": [{"acc": "count", "tool": consumer},
                              {"acc": "const", "value": 1, "span": "order"}]}]}


def _build_stale(producer: str, consumer: str, max_age_s: int,
                 max_age_span: str) -> dict:
    """If the most recent reading is OLDER than the stated age, the action is owed."""
    return {"op": "implies", "args": [
        {"op": "gt", "args": [
            {"op": "sub", "args": [{"acc": "clock"},
                                   {"acc": "when", "tool": producer}]},
            {"acc": "const", "value": max_age_s, "span": max_age_span}]},
        {"op": "ge", "args": [{"acc": "count", "tool": consumer},
                              {"acc": "const", "value": 1, "span": "order"}]}]}


TEMPLATES: Dict[str, Template] = {
    "value_below_then_required": Template(
        "value_below_then_required",
        "the task says to look a measurement up and act only if it is too low/high",
        {"producer": "the tool whose result carries the measurement",
         "consumer": "the tool that performs the owed action",
         "threshold": "the cutoff, a number written VERBATIM in the task or policy; "
                      "omit it to use the dosing table's own bands instead",
         "window": "how recent the reading must be to count, as the task words it "
                   "(e.g. \"last 24 hours\"); omit only if the task states no window"},
        _build_value_below),
    "stale_then_required": Template(
        "stale_then_required",
        "the task says to act when the last recorded reading is too OLD",
        {"producer": "the tool whose result carries the dated reading",
         "consumer": "the tool that performs the owed action",
         "max_age": "the age past which action is owed, as the task words it "
                    "(e.g. \"1 year\")"},
        _build_stale),
}


def menu() -> List[dict]:
    """What the interview shows the model. Selection over a closed set."""
    return [t.menu_entry() for t in TEMPLATES.values()]


def bind(name: str, args: Dict[str, Any], text: str) -> Tuple[Optional[dict], str]:
    """Validate the model's selection and parameters, then build the AST.

    Strict on values, because this is the layer that would otherwise reintroduce the
    fabricated constant: a threshold must be findable in the text, and a window must
    be a duration the text actually names.
    """
    t = TEMPLATES.get(name)
    if t is None:
        return None, (f"unknown template {name!r}; choose one of "
                      f"{', '.join(TEMPLATES)} or decline")

    producer, consumer = args.get("producer"), args.get("consumer")
    if not producer or not consumer:
        return None, "both 'producer' and 'consumer' are required"

    if name == "stale_then_required":
        span = str(args.get("max_age") or "").strip()
        secs = _DURATIONS.get(span.lower())
        if secs is None:
            return None, (f"max_age {span!r} is not a duration this library knows; "
                          f"use one the task states, e.g. {', '.join(sorted(_DURATIONS))}")
        if span.lower() not in text.lower():
            return None, f"the task does not say {span!r}, so it cannot be asserted"
        return t.build(producer=producer, consumer=consumer,
                       max_age_s=secs, max_age_span=span), ""

    thr = args.get("threshold", None)
    thr_span = str(args.get("threshold_span") or "").strip()
    if thr is not None:
        try:
            thr = float(thr)
        except (TypeError, ValueError):
            return None, f"threshold {thr!r} is not a number"
        if not _num_in_text(thr, text):
            return None, (f"{thr:g} does not appear in the task or policy. Quote a "
                          f"number that does, or omit 'threshold' to use the dosing "
                          f"table's bands.")
        if not thr_span:
            return None, "a threshold needs 'threshold_span', the text it came from"

    win = str(args.get("window") or "").strip()
    win_s = _DURATIONS.get(win.lower()) if win else None
    if win and win_s is None:
        return None, (f"window {win!r} is not a duration this library knows; "
                      f"use one the task states, e.g. {', '.join(sorted(_DURATIONS))}")
    if win and win.lower() not in text.lower():
        return None, f"the task does not say {win!r}, so it cannot be asserted"

    return t.build(producer=producer, consumer=consumer, threshold=thr,
                   threshold_span=thr_span, window_s=win_s, window_span=win), ""
