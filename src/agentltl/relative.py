#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
agentltl/relative.py -- predicates whose expected value is RELATIVE to the trace.

Kept apart from matching.py on purpose: that module is about whether two values are
the same thing written differently, these are about a value the spec cannot know at
generation time. The dose in

    "If low, order replacement IV magnesium according to dosing instructions"
    (Mild 1.5-1.9 -> 1 g; Moderate 1-<1.5 -> 2 g; Severe <1 -> 4 g)

depends on a magnesium level the agent reads at runtime. A spec cannot pin `2 g`.
What it CAN pin is the table, because the table is printed in the policy document
the agent is shown -- so encoding it leaks nothing (docs/LEAKAGE_AUDIT.md).

The enabling fact is that these guards are TRACE-EVALUABLE: the deciding value is
itself a recorded tool_result. That is what lets them be real implications instead
of being flattened into a plan-space disjunction, which is the required treatment
for a guard we cannot evaluate (references/agentltl-semantics.md).

Every predicate here obeys the two vacuity rules that make numbers_from_output
correct, because getting them wrong is how a constraint silently inverts:

  * the producer or consumer was never called  -> VACUOUS TRUE
  * the producer ran but its result was not recorded -> UNEVALUABLE, not violated

Formulas are declarative (`kind` + parameters), never evaluated strings: a spec is
model-authored, and eval() on model output is neither auditable nor safe.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Sequence

from .matching import args_match, values_match

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


# ── helpers ───────────────────────────────────────────────────────────────────
def path_values(obj: Any, path: str) -> List[Any]:
    """Values at a dotted path; `[]` descends into every element of a list.

    FHIR payloads nest hard -- the magnesium dose lives at
    `dosageInstruction[].doseAndRate[].doseQuantity.value` -- so a flat arg lookup
    cannot reach the value a constraint is about.
    """
    if not path:
        return [obj]
    cur: List[Any] = [obj]
    for part in path.split("."):
        nxt: List[Any] = []
        wild = part.endswith("[]")
        key = part[:-2] if wild else part
        for node in cur:
            if key:
                if not isinstance(node, dict) or key not in node:
                    continue
                node = node[key]
            if wild:
                if isinstance(node, (list, tuple)):
                    nxt.extend(node)
            else:
                nxt.append(node)
        cur = nxt
        if not cur:
            return []
    return cur


def _numbers_in(obj: Any) -> List[float]:
    out: List[float] = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, bool) or o is None:
            return
        elif isinstance(o, (int, float)):
            out.append(float(o))
        else:
            for m in _NUMBER.findall(str(o)):
                try:
                    out.append(float(m))
                except ValueError:
                    pass
    walk(obj)
    return out


def _text_recent_value(text: Any, extract: dict) -> Optional[float]:
    """Most-recent value out of an UNPARSEABLE (middle-elided) JSON payload.

    Pairs each value with the timestamp nearest it in the text, which is sound here
    because both sit inside the same resource object, then takes the latest. Only the
    leaves the extraction names, so this cannot wander into unrelated numbers the way
    `_numbers_in` does.
    """
    s = text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text)
    vleaf = str(extract.get("value") or "").split(".")[-1].replace("[]", "")
    wleaf = str(extract.get("when") or "").split(".")[-1].replace("[]", "")
    if not vleaf:
        return None

    v_re = re.compile(r'"' + re.escape(vleaf) + r'"\s*:\s*(-?\d+(?:\.\d+)?)')
    w_re = re.compile(r'"' + re.escape(wleaf) + r'"\s*:\s*"([^"]+)"') if wleaf else None

    if not v_re.search(s):
        return None

    # SPLIT PER ITEM, rather than pairing by distance in the flat text. Nearest-by-
    # distance is wrong in both directions: in a bundle the timestamp precedes its
    # own value, so the NEXT resource's timestamp can sit closer to this value than
    # this resource's does -- measured, it paired a 1.6 with the later of two times
    # and reported the stale reading as the most recent. Key order inside a resource
    # is not guaranteed either, so "nearest preceding" is no safer. Chunking on the
    # item boundary keeps each value with its own timestamp whatever the order.
    boundary = str(extract.get("items") or "").split(".")[-1].replace("[]", "")
    chunks = s.split(f'"{boundary}"') if boundary else [s]
    if len(chunks) < 2:
        chunks = [s]

    best_when = best_val = last_val = None
    for ch in chunks:
        vm = v_re.search(ch)
        if not vm:
            continue
        val = float(vm.group(1))
        last_val = val
        wm = w_re.search(ch) if w_re else None
        when = _parse_dt(wm.group(1)) if wm else None
        if when is None:
            continue
        if best_when is None or when > best_when:
            best_when, best_val = when, val

    if best_val is not None:
        return best_val
    # No usable timestamp anywhere: the LAST stated value is the best guess
    # available, and it is still confined to the declared leaf.
    return last_val


def _text_recent_when(text: Any, extract: dict):
    """Latest timestamp out of an UNPARSEABLE payload, chunked per item.

    Mirrors `_text_recent_value` exactly, including the chunking, so the two cannot
    disagree about which entry they describe.
    """
    s = text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text)
    vleaf = str(extract.get("value") or "").split(".")[-1].replace("[]", "")
    wleaf = str(extract.get("when") or "").split(".")[-1].replace("[]", "")
    if not wleaf:
        return None
    v_re = re.compile(r'"' + re.escape(vleaf) + r'"\s*:\s*(-?\d+(?:\.\d+)?)') \
        if vleaf else None
    w_re = re.compile(r'"' + re.escape(wleaf) + r'"\s*:\s*"([^"]+)"')
    if not w_re.search(s):
        return None

    boundary = str(extract.get("items") or "").split(".")[-1].replace("[]", "")
    chunks = s.split(f'"{boundary}"') if boundary else [s]
    if len(chunks) < 2:
        chunks = [s]

    best = None
    for ch in chunks:
        # Only entries that actually carry a reading, so an unrelated dated object
        # cannot become "the last reading".
        if v_re is not None and not v_re.search(ch):
            continue
        wm = w_re.search(ch)
        if not wm:
            continue
        when = _parse_dt(wm.group(1))
        if when is not None and (best is None or when > best):
            best = when
    return best


def observed_when(result: Any, extract: Optional[dict]):
    """WHEN the figure `observed_value` returns was recorded.

    A value alone cannot answer "is there a level from the last 24 hours", and that
    guard is half of what task5 states: "Check the last level WITHIN LAST 24 HOURS...
    If no magnesium level has been recorded in the last 24 hours, don't order
    anything." A stale-but-low reading therefore obliges nothing, and a trigger that
    ignores the timestamp demands an order for a patient whose correct treatment is
    none. Same extraction, same pairing, so the two cannot disagree about which entry
    they are describing.
    """
    extract = extract or {}
    if not extract.get("items"):
        return None
    if isinstance(result, (str, bytes)):
        parsed = _as_structured(result)
        if isinstance(parsed, (str, bytes)):
            # SAME TEXT FALLBACK AS THE VALUE. `MAB_MAX_TOOL_CHARS` middle-elides a
            # large bundle, so the payload does not parse -- and giving the value a
            # recovery path but not its timestamp made `when` unevaluable on exactly
            # the payloads that matter. 70 of run 9's task5 evaluations died there,
            # and an unevaluable guard makes the whole expression pass, so the
            # constraint was inert while looking satisfied.
            return _text_recent_when(parsed, extract)
        result = parsed
    best = None
    for it in path_values(result, extract["items"]):
        vals = path_values(it, extract["value"]) if extract.get("value") else [it]
        if not [n for v in vals for n in _numbers_in(v)]:
            continue
        whens = path_values(it, extract["when"]) if extract.get("when") else []
        when = _parse_dt(whens[0]) if whens else None
        if when is not None and (best is None or when > best):
            best = when
    return best


def observed_value(result: Any, extract: Optional[dict]) -> Optional[float]:
    """The number a decision turns on, pulled out of a recorded tool_result.

    `extract` is {"path": "...", "last": true}, or the recency-aware form
    {"items": "...", "value": "...", "when": "..."}.

    `last: True` MEANT "the most recent reading", on the stated premise that "a FHIR
    bundle returns them in order". That premise is FALSE for these bundles.
    mab_task5_11 returns five magnesium levels in this order:

        [0] 2023-11-12  2.2      [3] 2023-11-13  2.1   <- actually the most recent
        [1] 2023-11-08  1.8      [4] 2023-11-10  1.8   <- last in DOCUMENT order
        [2] 2023-11-09  2.1

    so document-last is a stale 1.8 while the real last level is 2.1. The agent
    answered 2.1 and was graded correct; every predicate reading this value saw 1.8.
    At 1.8 the magnesium is "low", so the dose band, task9's dose formula and the
    open-predicate triggers all fired on patients who were in fact NORMAL -- 21 of
    the 27 false alarms attributed to the open-predicate tier, plus a silently wrong
    band in the code tier.

    Recency has to be decided by the TIMESTAMP, which is in the payload the agent
    reads, not by position in a list whose order nothing guarantees. `items`/`value`/
    `when` extract per entry so a value can never be paired with another entry's
    time -- zipping two independent `path_values` walks would misalign the moment one
    entry lacked a timestamp.
    """
    extract = extract or {}

    # PARSE FIRST. Every tool result reaches these predicates as TEXT (see
    # _as_structured: not one call in run 4's 13k traces arrived as a dict), so
    # `path_values` could never walk `extract["path"]` and every lookup fell through
    # to `_numbers_in` over the RAW JSON STRING -- returning whichever number came
    # first textually, which in a FHIR bundle is `total`, the row count. So the
    # declared path had never taken effect on any real trace, and the missing
    # _EXTRACT key was hiding a second, independent defect underneath it.
    structured = True
    if isinstance(result, (str, bytes)):
        result = _as_structured(result)
        structured = not isinstance(result, (str, bytes))

    if extract.get("items") and structured:
        best_when, best_val, undated = None, None, None
        for it in path_values(result, extract["items"]):
            vals = path_values(it, extract["value"]) if extract.get("value") else [it]
            nums = [n for v in vals for n in _numbers_in(v)]
            if not nums:
                continue
            if undated is None:
                undated = nums[0]
            whens = path_values(it, extract["when"]) if extract.get("when") else []
            when = _parse_dt(whens[0]) if whens else None
            if when is None:
                continue
            if best_when is None or when > best_when:
                best_when, best_val = when, nums[0]
        # An entry with no parseable timestamp is not a reason to discard its value:
        # unevaluable is reserved for having no figure at all.
        # AUTHORITATIVE when the document parsed: a bundle that really carries no
        # reading must return None, not a number. It used to fall through and scrape
        # `total: 0`, so an EMPTY lookup read as an observed 0.0 -- below every
        # threshold -- and the trigger demanded an order for a patient with no
        # recorded level. That is the exact case task5 states as "if no magnesium
        # level has been recorded in the last 24 hours, don't order anything", so the
        # constraint punished the correct abstention.
        return best_val if best_val is not None else undated

    if extract.get("items") and not structured:
        # `MAB_MAX_TOOL_CHARS` elides the MIDDLE of a large bundle, so the payload is
        # deliberately not valid JSON and cannot be walked structurally. Recover the
        # value/time pairs from the raw text instead: still recency-correct, where
        # blind `_numbers_in` returned whichever number came first and produced a
        # version id or a negative reference range.
        # AUTHORITATIVE too, and deliberately: `_text_recent_value` returns None
        # only when the declared value leaf appears NOWHERE, which means the payload
        # is not an observation at all. The mab runner reports a failed lookup as
        # `"Error in sending the GET request: 400 ... date=2023-11-13..."`, and the
        # blind `_numbers_in` fallback read **-13.0** out of that URL's date -- below
        # every threshold, so a lookup that FAILED demanded a replacement order. The
        # sibling case scraped 0.0 the same way. A tool error is not a value; not
        # knowing the level is `unevaluable`, which is what None means here.
        return _text_recent_value(result, extract)

    vals: List[Any] = []
    if extract.get("path"):
        vals = path_values(result, extract["path"])
    if not vals:
        nums = _numbers_in(result)
        if not nums:
            return None
        return nums[-1] if extract.get("last") else nums[0]
    nums: List[float] = []
    for v in vals:
        nums.extend(_numbers_in(v))
    if not nums:
        return None
    return nums[-1] if extract.get("last") else nums[0]


def formula_expected(formula: Optional[dict], value: float) -> Optional[float]:
    """The amount a declared arithmetic rule computes for an observed value.

    Module level so the WITNESS builder can call the very same function the predicate
    does. A witness that re-derives this arithmetic on its own can disagree with the
    predicate, and then reports a sound spec as unsatisfiable -- which is how
    `satisfiable` came to mean "the witness builder is confused" rather than anything
    about the spec.
    """
    kind = (formula or {}).get("kind")
    if kind == "step_below_goal":
        goal = float(formula["goal"]); per = float(formula["per"])
        amount = float(formula["amount"])
        if value >= goal:
            return 0.0
        return math.ceil(round((goal - value) / per, 6)) * amount
    if kind == "linear":
        return float(formula.get("scale", 1)) * value + float(formula.get("offset", 0))
    return None


def band_for(value: float, bands: Sequence[dict]) -> Optional[dict]:
    """Public alias -- same reason as formula_expected."""
    return _band_for(value, bands)


def _band_for(value: float, bands: Sequence[dict]) -> Optional[dict]:
    """First band containing `value`. Half-open [lo, hi) so a boundary has one owner.

    stage2 normalises the prose bands into this shape in code, so the arithmetic is
    auditable rather than model-guessed. It must also split the DOSE the same way:
    the policy writes "IV: 1 g over 1 hour" while the FHIR payload carries
    `doseQuantity.value = 1` and `doseQuantity.unit = "g"` in separate fields, so a
    band stores {"lo":1.5,"hi":1.9,"value":1,"unit":"g"} -- structured, never the
    prose string. Comparing "1 g" against 1 would otherwise fail a correct order,
    and widening the global value matcher to paper over it is explicitly ruled out
    (prose-instead-of-value is a real error worth catching elsewhere).
    """
    for b in bands:
        lo, hi = b.get("lo"), b.get("hi")
        if lo is not None and value < float(lo):
            continue
        if hi is not None:
            # Half-open by default, but clinical prose writes the TOP band of a
            # range inclusively -- "serum magnesium 1.5 to 1.9 mg/dL" means 1.9 is
            # mild, while "1 to <1.5" is explicitly exclusive. Left half-open only,
            # a recorded 1.9 would fall through every band and the constraint would
            # report "no band" on a perfectly ordinary value.
            if b.get("hi_inclusive"):
                if value > float(hi):
                    continue
            elif value >= float(hi):
                continue
        return b
    return None


def _empty_result(result: Any) -> bool:
    """Did a FHIR search come back with nothing?

    A Bundle with total 0 / no entry means "no measurement on file", which is the
    antecedent of task5's prohibition. Distinguished from "not recorded", which is
    unevaluable rather than empty.
    """
    if result is None:
        return False
    if isinstance(result, dict):
        if "entry" in result:
            return not result.get("entry")
        if "total" in result:
            try:
                return int(result["total"]) == 0
            except (TypeError, ValueError):
                return False
    s = str(result)
    if not s.strip():
        return False
    return ('"entry"' not in s) and ('"total":0' in s.replace(" ", "")
                                     or '"total": 0' in s)


def _parse_dt(v: Any) -> Optional[datetime]:
    """Lenient ISO-8601 -> naive datetime. Naive because these tasks live in a
    single frozen timezone; comparing an aware bound to a naive arg would raise."""
    s = str(v).strip().strip('"\'')
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except ValueError:
        pass
    # A PARTIAL PARSE IS NOT A PARSE. This fallback truncates to a fixed prefix, so
    # `2023-11-12 to 2023-11-13` used to yield the INSTANT 2023-11-12T00:00 -- the
    # separator and the whole second bound silently discarded. `arg_window_covers`
    # then blocked it as "too short to answer a question about [...]", manufacturing
    # the one error it exists to catch out of a correct period. That is exactly how
    # the ISO-8601 `/` form cost run 9 its task5 arm: 350 of 423 date blocks (83%)
    # were this, and 40 of 43 blocked-to-death episodes died on a value we simply
    # failed to read.
    #
    # So the prefix is accepted only when it IS the whole value. Residue means we do
    # not know what the value denotes, and not knowing is UNEVALUABLE -- never a
    # confident instant. Over-acceptance is the safe direction here: a value we
    # cannot read passes, and the argument rules own prose in a date slot.
    for fmt, width in (("%Y-%m-%dT%H:%M:%S", 19), ("%Y-%m-%d", 10)):
        head, tail = s[:width], s[width:].strip()
        if tail:
            continue
        try:
            return datetime.strptime(head, fmt)
        except ValueError:
            continue
    return None


# ── date arguments are INTERVALS, and the role decides the test ───────────────
# A date argument is not an instant. A FHIR search parameter legitimately carries a
# RANGE -- `2023-11-12..2023-11-13`, `ge2023-11-12` -- and a bare `2023-11-12` denotes
# the whole of that day. Collapsing any of those to one instant is what made
#
#     GET_Observation_labs(code="MG", date="2023-11-12..2023-11-13")
#
# -- the CORRECT query for "within the last 24 hours" -- parse as 2023-11-12T00:00 and
# fail a window whose floor is 10:15 the same morning. It blocked the right lookup in 5
# of the 6 episodes where enforcement turned a correct run into a wrong one.
#
# The two roles want OPPOSITE tests, which is the deeper half of the same bug:
#   * a WRITE records one moment, so its interval must be CONSISTENT with the window
#     (they intersect) -- day granularity must not be punished, since task8's correct
#     authoredOn="2023-11-13" carries no time of day.
#   * a READ asks a question, so its interval must COVER the window; a query narrower
#     than the window silently misses the rows the task is about.
_EPS = timedelta(microseconds=1)

# FHIR search-parameter comparators. `ne` is a negation, so it bounds nothing we can
# turn into an interval -- it yields None (unevaluable) rather than a wrong interval.
_CMP = ("ge", "gt", "le", "lt", "eq", "ne", "sa", "eb", "ap")


def _token_span(s: str) -> Optional[tuple]:
    """The closed interval one date token denotes, at its own stated granularity."""
    s = s.strip().strip('"\'')
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", s)
    if not m:
        dt = _parse_dt(s)
        return (dt, dt) if dt else None
    yr = int(m.group(1))
    mo, dy = m.group(2), m.group(3)
    if dy:
        start = datetime(yr, int(mo), int(dy))
        return (start, start + timedelta(days=1) - _EPS)
    if mo:
        start = datetime(yr, int(mo), 1)
        nxt = (datetime(yr + 1, 1, 1) if int(mo) == 12
               else datetime(yr, int(mo) + 1, 1))
        return (start, nxt - _EPS)
    return (datetime(yr, 1, 1), datetime(yr + 1, 1, 1) - _EPS)


def _parse_interval(v: Any) -> Optional[tuple]:
    """A date argument -> the closed interval it denotes. None when unparseable.

    Unparseable is deliberately NOT a violation: prose in a date slot is a different
    error, caught by the argument rules, and failing here would double-count it.
    """
    s = str(v).strip().strip('"\'')
    if not s:
        return None
    if "," in s:                                    # FHIR OR-list -> union hull
        parts = [iv for iv in (_parse_interval(x) for x in s.split(",")) if iv]
        if not parts:
            return None
        return (min(a for a, _ in parts), max(b for _, b in parts))
    if "/" in s and ".." not in s:                  # ISO 8601 <start>/<end>
        # A CORRECT PERIOD WAS BEING TURNED INTO THE ONE ERROR THIS PREDICATE
        # CATCHES. `2023-11-12T10:15:00/2023-11-13T10:15:00` is the required window
        # written in ISO 8601 interval form, and the parser dropped the separator and
        # everything after it, yielding the degenerate instant (10:15, 10:15).
        # `arg_window_covers` then failed it as "a bare instant [that] retrieves
        # nothing" -- the very case its comment cites as the error it exists to
        # detect. It manufactured that error out of the right answer: 11 of mab's 14
        # remaining false alarms, in a BLOCKING layer, all of them this.
        #
        # Accepted only when the split gives exactly TWO parts and BOTH parse on
        # their own. That is self-validating, so a US-format "11/12/2023" (three
        # parts) and a FHIR reference "Patient/S2380121" both stay unparseable --
        # verified, not assumed.
        parts = s.split("/")
        if len(parts) == 2:
            a, b = (_parse_interval(parts[0]), _parse_interval(parts[1]))
            if a and b:
                return (min(a[0], b[0]), max(a[1], b[1]))

    if ".." in s:                                   # explicit range
        a, b = s.split("..", 1)
        lo_iv, hi_iv = _parse_interval(a) if a.strip() else None, \
                       _parse_interval(b) if b.strip() else None
        if a.strip() and lo_iv is None:
            return None
        if b.strip() and hi_iv is None:
            return None
        return (lo_iv[0] if lo_iv else datetime.min,
                hi_iv[1] if hi_iv else datetime.max)
    low = s.lower()
    for cmp_ in _CMP:
        if low.startswith(cmp_) and len(s) > len(cmp_) and s[len(cmp_)].isdigit():
            span = _token_span(s[len(cmp_):])
            if span is None or cmp_ == "ne":
                return None
            a, b = span
            if cmp_ in ("ge", "sa"):
                return (a, datetime.max)
            if cmp_ == "gt":
                return (b + _EPS, datetime.max)
            if cmp_ in ("le", "eb"):
                return (datetime.min, b)
            if cmp_ == "lt":
                return (datetime.min, a - _EPS)
            return span                             # eq / ap
    return _token_span(s)


# A result that says the call did not happen. Deliberately a small, literal set:
# for a CAP, under-counting only weakens the constraint, while over-counting BLOCKS
# THE CORRECT RETRY, so the asymmetry says lean lenient. Matched at the start of the
# result or as a whole phrase, never as a loose substring, so ordinary content
# containing the word "invalid" is not read as a failure.
_FAILED_RESULT = re.compile(
    r"^\s*(?:error\b|exception\b|traceback\b)"
    r"|\bmust be one of\b|\bunknown tool\b|\bnot found\b|\bdoes not exist\b"
    r"|\bno such\b|\bis invalid\b|\binvalid \w+\b|\bmissing required\b",
    re.I)


def _call_failed(call) -> bool:
    """True when the call's own result says it did not take effect.

    A failed call creates no record, so retrying it is CORRECT behaviour -- and a cap
    that counts the attempt blocks the retry. Measured on wb_analytics_4: the agent
    sent an out-of-range `plot_type`, got back `Plot type must be one of 'bar',
    'line'`, corrected itself, and `analytics_create_plot_at_most_1` refused the
    corrected call. 290 blocks on already-correct episodes are this.

    A call with NO recorded result counts as real, which matters: at enforcement time
    the PENDING call is exactly that, and excusing it would stop every cap from
    blocking anything.
    """
    res = getattr(call, "result", None)
    if res in (None, ""):
        return False
    return bool(_FAILED_RESULT.search(str(res)[:400]))


def calls_at_most(tool: str, n: int, A: dict, strict: bool = False):
    """`tool` called at most n times, counting only calls that TOOK EFFECT.

    Replaces AgentLTL's CalledNTimes for the cap case only. The formula type in the
    spec is unchanged, so `taxonomy.disposition` still reads `op` and still returns
    at_call -- a cap remains a universal over the calls present, and remains
    blockable. Only the counting changes.
    """
    Predicate = A["Predicate"]
    limit = int(n)

    def fn(trace, position, metrics=None):
        made = _calls(trace, [tool])
        effective = [c for c in made if not _call_failed(c)]
        k = len(effective)
        ok = k < limit if strict else k <= limit
        if ok:
            note = f'"{tool}" called {k} time(s), within {"<" if strict else "<="} {limit}.'
            if len(made) != k:
                note += f" ({len(made) - k} earlier attempt(s) returned an error and " \
                        "created no record.)"
            return {"passed": True, "note": note}
        return {"passed": False,
                "note": (f'"{tool}" called {k} time(s); expected '
                         f'{"<" if strict else "<="} {limit}. Correct the call you are '
                         "about to make rather than adding another.")}

    return Predicate(fn=fn, description=f"{tool} at most {limit} effective call(s)",
                     runtime_safe=True)


def _calls(trace, names: Sequence[str]):
    want = set(names)
    return [c for c in trace.calls if c.name in want]


def _arg_at(call, arg: str) -> List[Any]:
    """Argument value(s), supporting a dotted path into a nested payload.

    Delegates to path_values over the whole args dict rather than splitting off the
    head itself: the first segment can carry the `[]` wildcard too
    (`dosageInstruction[].doseAndRate[]...`), and hand-splitting on "." left the
    brackets attached to the key, so the lookup missed and every dose constraint
    reported "vacuous: no consumer called with <arg>" -- passing the compliant runs
    for the wrong reason and the violating ones too.
    """
    # unwrap_json_args for the same reason args_match needs it: at ENFORCEMENT time
    # a POST body is still the raw JSON string the agent sent, so a dose path
    # resolved to nothing and the dosing constraint reported "vacuous" -- passing the
    # wrong dose and blocking the right one alike.
    from .matching import unwrap_json_args
    return path_values(unwrap_json_args(call.args or {}), arg)


# ── predicates ────────────────────────────────────────────────────────────────
def arg_from_table(producers, extract, bands, consumers, arg, A: dict):
    """The consumer's arg must be what the policy table maps the observed value to.

    This is the dosing rule. It fails an agent that orders a plausible dose for the
    wrong band -- a purely PROCEDURAL error that outcome scoring alone would miss
    whenever the final answer text happens to look right.
    """
    Predicate = A["Predicate"]
    prods, cons = list(producers), list(consumers)

    def fn(trace, position, metrics=None):
        used = [c for c in _calls(trace, cons) if _arg_at(c, arg)]
        if not used:
            return {"passed": True, "note": f"vacuous: no {cons[:2]} called with {arg}."}
        for c in used:
            earlier = [p for p in _calls(trace, prods) if p.position < c.position]
            if not earlier:
                return {"passed": False,
                        "note": (f"{c.name}(#{c.position+1}) set {arg} with no prior "
                                 f"{prods[:2]} -- the value cannot have been observed.")}
            value = None
            for p in reversed(earlier):
                value = observed_value(getattr(p, "result", None), extract)
                if value is not None:
                    break
            if value is None:
                return {"passed": True,
                        "note": "unevaluable: no recorded producer output to read."}
            band = _band_for(value, bands)
            if band is None:
                return {"passed": True,
                        "note": f"unevaluable: observed {value} falls in no stated band."}
            # Which COLUMN of the band the arg wants. The policy writes one dose
            # as "1 g" while FHIR splits it into doseQuantity.value = 1 and
            # doseQuantity.unit = "g", and the generator emits one constraint per
            # field. Always comparing against band["value"] made the .unit
            # constraint compare "g" against 1 -- so it failed the CORRECT dose as
            # well as the wrong one, adding a constant that looks like
            # non-compliance and shrinking the gap the arm is measuring.
            leaf = str(arg).rstrip("[]").rsplit(".", 1)[-1].lower()
            expected = band.get(leaf, band.get("value")) if leaf in band \
                else band.get("value")
            actual = _arg_at(c, arg)
            if not any(values_match(expected, a) for a in actual):
                return {"passed": False,
                        "note": (f"observed {value} -> table says {arg}={expected!r}, "
                                 f"but {c.name}(#{c.position+1}) used {actual[:2]!r}.")}
        return {"passed": True, "note": f"{arg} matches the table for the observed value."}

    return Predicate(fn=fn, description=f"{cons[:2]}.{arg} follows the policy table",
                     runtime_safe=True)


def arg_in_window(tool: str, arg: str, lo: str, hi: str, A: dict,
                  moment: bool = False):
    """A RECORDED moment must be consistent with a window resolved from the clock.

    "within last 24 hours" is meaningless without the frozen clock, and resolving it
    in code keeps it auditable.

    The test is INTERSECTION, not containment, because the argument states a moment
    only to its own granularity. task8's correct `authoredOn="2023-11-13"` carries no
    time of day, so demanding containment in [11-12T10:15, 11-13T10:15] would fail
    every correct write -- the same shape of error as asserting doseQuantity.unit=="g"
    from a schema that only says `such as "g"`.

    For a SEARCH argument the duty is the opposite way round: see arg_window_covers.
    """
    Predicate = A["Predicate"]

    # A WINDOW BOUND STATED TO DAY GRANULARITY DENOTES THE WHOLE DAY, and reading it
    # as midnight collapsed the window to a zero-width instant. Where the clock
    # itself carries no time of day -- WorkBench's is `2023-12-01`, not mab's
    # `2023-11-13T10:15:00` -- `lo` and `hi` both became 00:00, so EVERY time of day
    # on the correct day was reported "outside":
    #
    #     calendar_search_events(#2) time_min='2023-12-01 09:00:00'
    #         is outside [2023-12-01, 2023-12-01]
    #
    # 30 of workbench's 58 clock-binding false alarms are that, all on read bounds
    # (`time_min`, `time_max`) where searching the clock's day is exactly right.
    #
    # This is the same closed-interval rule already applied to the ARGUMENT side, and
    # it is why `_parse_interval` is used here rather than `_parse_dt`: the low bound
    # takes the interval's start, the high bound its END. A bound that already states
    # an instant is unchanged -- mab's clock parses to (10:15, 10:15) either way --
    # so `want_instant` below still fires on a genuine point window and the two
    # refsol assertions it exists for are untouched.
    _lo_iv, _hi_iv = _parse_interval(lo), _parse_interval(hi)
    lo_dt = _lo_iv[0] if _lo_iv else None
    hi_dt = _hi_iv[1] if _hi_iv else None
    # The point-window test must compare like with like: `lo` widened to a day would
    # otherwise never equal `hi`'s end-of-day, and the moment branch would go dead.
    _lo_hi = _lo_iv[1] if _lo_iv else None

    def fn(trace, position, metrics=None):
        used = [c for c in _calls(trace, [tool]) if _arg_at(c, arg)]
        if not used:
            return {"passed": True, "note": f"vacuous: {tool} never called with {arg}."}
        if lo_dt is None or hi_dt is None:
            return {"passed": True, "note": "unevaluable: window did not resolve."}
        # A DEGENERATE WINDOW IS A MOMENT, AND DEMANDS ONE BACK. Where lo == hi the
        # slot is bound to THE CLOCK -- when this record was authored is the episode
        # instant by definition -- and intersection is then far too weak: a date-only
        # value expands to the whole day, the day contains the instant, and the write
        # passes. That is measured blindness, not pedantry:
        #
        #     refsol.task3  payload['effectiveDateTime'] == '2023-11-13T10:15:00+00:00'
        #     refsol.task8  '2023-11-13T10:15' in payload['authoredOn']
        #
        # so `effectiveDateTime="2023-11-13"` FAILS the grader while scoring 1.0 with
        # us. Two of check_mab_discrimination's four TOO-WEAK cases are exactly this.
        #
        # This also corrects a claim in CLAUDE.md, which cites task8's date-only
        # authoredOn as CORRECT behaviour being wrongly blocked. refsol says the
        # opposite. The intersection rule remains right for a window that is a real
        # INTERVAL ("within the last 24 hours" recorded to day granularity is
        # consistent with it) and right for reads, where the 33 measured false alarms
        # it fixed were all GET date queries -- none of them a clock-bound write.
        # THE ROLE DECIDES, not the window's shape. Requiring an instant wherever
        # lo == hi was measurably wrong: it added 119 false alarms on correct wb
        # traces, all of them READ bounds -- time_max='2023-11-30' on
        # analytics_total_visits_count, date_min on email_search_emails -- where a
        # day-granular search bound is exactly right. A read asks about a period; a
        # WRITE records a moment. So the emitter passes `moment` from the node's
        # role, the same rule that already picks this predicate over
        # arg_window_covers.
        # A POINT window is one whose two bounds denote the SAME instant, which is
        # what `lo == hi` meant before the bounds were widened. Comparing the widened
        # `hi_dt` against `lo_dt` would make it false for every date-only clock and
        # silently retire the branch.
        want_instant = moment and lo_dt == _lo_hi and lo == hi
        for c in used:
            for raw in _arg_at(c, arg):
                iv = _parse_interval(raw)
                if iv is None:
                    continue                      # not a datetime: other rules cover it
                if want_instant:
                    if not (iv[0] <= lo_dt <= iv[1]):
                        return {"passed": False,
                                "note": (f"{c.name}(#{c.position+1}) {arg}={raw!r} is not "
                                         f"the recorded moment {lo}.")}
                    if iv[0] != iv[1]:
                        return {"passed": False,
                                "note": (f"{c.name}(#{c.position+1}) {arg}={raw!r} states "
                                         f"only a date; this field records WHEN, so it "
                                         f"needs the time of day too ({lo}).")}
                    continue
                if iv[1] < lo_dt or iv[0] > hi_dt:
                    return {"passed": False,
                            "note": (f"{c.name}(#{c.position+1}) {arg}={raw!r} is outside "
                                     f"[{lo}, {hi}].")}
        return {"passed": True, "note": f"{arg} within [{lo}, {hi}]."}

    return Predicate(fn=fn, description=f"{tool}.{arg} in [{lo},{hi}]", runtime_safe=True)


_MIDNIGHT = time(0, 0, 0)
_END_OF_DAY = time(23, 59, 59, 999999)


def arg_window_covers(tool: str, arg: str, lo: str, hi: str, A: dict):
    """A SEARCH argument must COVER the window, not sit inside it.

    The read/write distinction is the whole point. `GET_Observation_labs(date=...)` asks
    a question, so a range NARROWER than the window silently misses the rows the task is
    about, and a range wider than it is harmless. Testing containment here -- which is
    what a single shared ArgInWindow did -- rejects exactly the queries that are right:
    `date="2023-11-12..2023-11-13"` for "the last 24 hours" was blocked as "outside"
    because its own floor is earlier than the window's.

    Several values on ONE call are bounds or alternatives of one query, so they are
    unioned. (A write is judged per value instead; see arg_in_window.)
    """
    Predicate = A["Predicate"]

    lo_dt, hi_dt = _parse_dt(lo), _parse_dt(hi)

    def fn(trace, position, metrics=None):
        used = [c for c in _calls(trace, [tool]) if _arg_at(c, arg)]
        if not used:
            return {"passed": True, "note": f"vacuous: {tool} never called with {arg}."}
        if lo_dt is None or hi_dt is None:
            return {"passed": True, "note": "unevaluable: window did not resolve."}
        # COVERAGE IS ONLY DERIVABLE WHEN THE WINDOW ALIGNS TO THE SEARCH
        # PARAMETER'S GRANULARITY. A FHIR `date` search is day-granular, and a
        # clock-derived window like "the last 24 hours" straddles midnight -- so NO
        # query covers [2023-11-12T10:15, 2023-11-13T10:15] exactly. Every correct
        # query is a superset (both days) or a subset (one day), and demanding
        # coverage forces `ge2023-11-12&le2023-11-13`: a specific formulation the
        # task text never states and the schema never documents. Deriving that is the
        # same overreach as deriving refsol's `Patient/` prefix, which CLAUDE.md
        # refuses -- so where the window is not day-aligned, the checkable obligation
        # is that the query REACHES INTO the window, not that it covers it.
        #
        # What that still catches is the error the predicate was written for: a query
        # over an unrelated period, which returns stale rows or none. What it gives
        # up is the too-NARROW query -- and a too-narrow query is what a correct
        # agent produces here, which is the whole problem.
        aligned = (lo_dt.time() == _MIDNIGHT
                   and (hi_dt.time() in (_MIDNIGHT, _END_OF_DAY)))
        for c in used:
            raws = _arg_at(c, arg)
            ivs = [iv for iv in (_parse_interval(r) for r in raws) if iv]
            if not ivs:
                continue                          # unparseable: other rules cover it
            span = (min(a for a, _ in ivs), max(b for _, b in ivs))
            if aligned:
                bad = span[0] > lo_dt or span[1] < hi_dt
                why = f"does not cover [{lo}, {hi}]"
            else:
                # OVERLAP IS NOT ENOUGH ON ITS OWN. A bare instant overlaps and
                # retrieves nothing -- that was this task's original case:
                # date=['2023-11-13T10:15:00'] permanently blocked the tool and the
                # episode ended with no answer, and the constraint was RIGHT. So the
                # query must also be long enough to be a period: at least the
                # window's own length, capped at the parameter's granularity of one
                # day. A day query against a 24h window clears it (24h >= 1 day); an
                # instant does not (0 >= 1 day is false); an unrelated day fails the
                # overlap half.
                # `_parse_interval` returns a CLOSED interval, so one calendar day
                # measures 24h minus a microsecond (00:00:00 .. 23:59:59.999999).
                # Comparing against a bare timedelta(days=1) therefore rejected the
                # day query by 1us -- the correct answer failing on the
                # representation of its own endpoint.
                need = min(hi_dt - lo_dt, timedelta(days=1)) - timedelta(microseconds=1)
                overlaps = not (span[1] < lo_dt or span[0] > hi_dt)
                long_enough = (span[1] - span[0]) >= need
                bad = not (overlaps and long_enough)
                why = ("does not reach into [{}, {}] at all -- it asks about a "
                       "different period".format(lo, hi) if not overlaps else
                       "is too short to answer a question about [{}, {}] -- it spans "
                       "{} where the period needs at least {}".format(
                           lo, hi, span[1] - span[0], need))
            if bad:
                return {"passed": False,
                        "note": (f"{c.name}(#{c.position+1}) {arg}={raws[:2]!r} queries "
                                 f"[{span[0].isoformat()}, {span[1].isoformat()}], "
                                 f"which {why}. Query the period the task names.")}
        return {"passed": True,
                "note": (f"{arg} {'covers' if aligned else 'reaches into'} "
                         f"[{lo}, {hi}].")}

    return Predicate(fn=fn, description=f"{tool}.{arg} covers [{lo},{hi}]",
                     runtime_safe=True)


def arg_equals_resolved(tool: str, arg: str, value: Any, A: dict):
    """The arg equals a constant stage2 computed from the clock.

    task9's "morning serum potassium the next day at 8am" is a fixed instant once
    the clock is known -- but only once it is known, so it cannot be a plain literal
    in the spec.
    """
    Predicate = A["Predicate"]

    def fn(trace, position, metrics=None):
        used = [c for c in _calls(trace, [tool]) if _arg_at(c, arg)]
        if not used:
            return {"passed": True, "note": f"vacuous: {tool} never called with {arg}."}
        for c in used:
            if not any(values_match(value, a) for a in _arg_at(c, arg)):
                return {"passed": False,
                        "note": (f"{c.name}(#{c.position+1}) {arg}={_arg_at(c, arg)[:1]!r}, "
                                 f"expected {value!r}.")}
        return {"passed": True, "note": f"{arg} == {value!r}."}

    return Predicate(fn=fn, description=f"{tool}.{arg} == {value!r}", runtime_safe=True)


def empty_result_then_never(producers, consumers, A: dict):
    """If the lookup found nothing, the write must not happen.

    task5: "If no magnesium level has been recorded in the last 24 hours, don't
    order anything." Expressed as a first-class prohibition rather than
    Not(CalledWith), which is vacuously TRUE when the tool is absent and would
    therefore FAIL exactly the run that correctly abstained.
    """
    Predicate = A["Predicate"]
    prods, cons = list(producers), list(consumers)

    def fn(trace, position, metrics=None):
        lookups = _calls(trace, prods)
        if not lookups:
            return {"passed": True, "note": f"vacuous: no {prods[:2]} lookup ran."}
        recorded = [p for p in lookups if getattr(p, "result", None) is not None]
        if not recorded:
            return {"passed": True, "note": "unevaluable: lookup results not recorded."}
        if not all(_empty_result(getattr(p, "result", None)) for p in recorded):
            return {"passed": True,
                    "note": "vacuous: a measurement was found, prohibition does not apply."}
        offending = _calls(trace, cons)
        if offending:
            return {"passed": False,
                    "note": (f"lookup returned nothing, yet {offending[0].name}"
                             f"(#{offending[0].position+1}) was called anyway.")}
        return {"passed": True, "note": "no measurement found and nothing was ordered."}

    return Predicate(fn=fn, description=f"empty {prods[:2]} -> never {cons[:2]}",
                     runtime_safe=True)


def arg_derived_from(producers, extract, formula: dict, consumers, arg: str, A: dict):
    """The arg equals a declared arithmetic function of the observed value.

    task9: "for every 0.1 mEq/L below threshold, order 10 mEq to reach a goal of
    3.5". Declarative kinds only -- a spec is model-authored and eval() on model
    output is neither auditable nor safe.
    """
    Predicate = A["Predicate"]
    prods, cons = list(producers), list(consumers)

    def expected_for(value: float) -> Optional[float]:
        return formula_expected(formula, value)

    def fn(trace, position, metrics=None):
        used = [c for c in _calls(trace, cons) if _arg_at(c, arg)]
        if not used:
            return {"passed": True, "note": f"vacuous: no {cons[:2]} called with {arg}."}
        for c in used:
            earlier = [p for p in _calls(trace, prods) if p.position < c.position]
            value = None
            for p in reversed(earlier):
                value = observed_value(getattr(p, "result", None), extract)
                if value is not None:
                    break
            if value is None:
                return {"passed": True,
                        "note": "unevaluable: no recorded producer output to read."}
            want = expected_for(value)
            if want is None:
                return {"passed": True, "note": f"unevaluable: unknown formula kind."}
            got = [a for a in _arg_at(c, arg)]
            nums = [n for a in got for n in _numbers_in(a)]
            if not nums:
                return {"passed": False,
                        "note": f"{c.name}(#{c.position+1}) {arg}={got[:1]!r} holds no number."}
            if not any(abs(n - want) <= 1e-6 * max(1.0, abs(want)) for n in nums):
                # A COMPUTED TARGET OF ZERO MEANS "DO NOT ORDER", NOT "ORDER ZERO".
                # Live smoke, task9 at potassium 4.5 against a goal of 3.5: the
                # step formula gives 0, and the message read `doseQuantity.value
                # should be 0.0, got [20.0]` -- which literally instructs the agent
                # to write a dose of zero. It cannot act on that: the correct
                # behaviour at or above the goal is to place no order at all. A
                # block the agent cannot act on is a block that destroys the
                # action, so the message has to name the ACTION, not the slot value.
                if abs(want) <= 1e-9:
                    return {"passed": False,
                            "note": (f"the observed value {value} needs no "
                                     f"repletion by the task's own rule, so no "
                                     f"{c.name} should be placed at all -- this one "
                                     f"orders {nums[:2]}. Do not order; report that "
                                     f"no treatment is indicated.")}
                return {"passed": False,
                        "note": (f"observed {value} -> {arg} should be {want}, "
                                 f"got {nums[:2]}.")}
        return {"passed": True, "note": f"{arg} derived correctly from the observation."}

    return Predicate(fn=fn, description=f"{cons[:2]}.{arg} derived from {prods[:2]}",
                     runtime_safe=True)


# ── variables: one entity threaded through the procedure ──────────────────────
# The missing expressive power. `ArgFromOutput` asks "does this value appear
# ANYWHERE in some producer's output", and `_in_result` even falls back to a
# SUBSTRING match, so a short id matches inside unrelated rendered text. What a
# procedure actually requires is identity: the patient id written into the POST is
# THE one the lookup returned, and where several later calls concern the same
# entity they must all carry the SAME value.
#
# Implemented as predicates rather than with AgentLTL's ForAll/Exists + Var, and
# that is a deliberate trade. Those would work, but `substitute()` only rewrites
# Var inside AgentLTL's OWN CalledWith, while this pipeline compiles CalledWith to
# `called_with_like` -- the format-tolerant matcher. Losing that tolerance would
# reintroduce exactly the failure it exists to prevent: `Patient/S2380121` vs
# `S2380121` is a formatting difference, not a procedural error. A quantifier whose
# body cannot be format-tolerant would fail correct runs.
#
# Vacuity is the other reason. `Exists` over an EMPTY domain is FALSE by standard
# FOL semantics, and `qwen_benchmark._trace_to_metrics` DROPS `tool_result` -- so on
# that path the domain is empty and a raw quantifier would report every correct run
# as violating. These predicates return "unevaluable, not violated" instead, the
# rule the whole module already obeys.
def _output_values(trace, producers: Optional[Sequence[str]],
                   extract: Optional[dict],
                   before: Optional[int] = None) -> Optional[List[Any]]:
    """Values a producer actually returned, or None when nothing was recorded.

    None and [] mean different things: None is "we cannot tell" (no results in the
    trace) and [] is "the lookup genuinely returned nothing". Collapsing them is how
    a missing-evidence path turns into a violation.
    """
    # producers=None means EVERY call, for the read-or-invented fallback in
    # arg_is_output_value. A named list stays a named list.
    calls = list(trace.calls) if producers is None else _calls(trace, list(producers))
    prods = list(producers or ())
    seen_result = False
    out: List[Any] = []
    for c in calls:
        if before is not None and c.position >= before:
            continue
        res = getattr(c, "result", None)
        if res in (None, ""):
            continue
        seen_result = True
        res = _as_structured(res)
        if extract and extract.get("path"):
            vals = path_values(res, extract["path"])
        else:
            vals = []
            _collect_scalars(res, vals)
        for v in vals:
            if v is not None and not any(values_match(v, o) for o in out):
                out.append(v)
    return out if seen_result else None


def _as_structured(res: Any) -> Any:
    """A tool result that IS structured but arrives as TEXT, parsed.

    Every tool result reaches these predicates as a string -- not one call in run 4's
    13k traces arrived as a dict or list. And most of them are JSON: WorkBench returns
    `[{"event_id": "00000256", ...}]` as a str, and so do 2015 of tau's producer calls
    and 481 of mcpu's. Left as a string, `_collect_scalars` collects the WHOLE
    DOCUMENT as one scalar, so `_same_entity("00000256", <the entire JSON text>)` is
    False and identity provenance reports "it was composed, not read" about a value
    the producer plainly returned. That single line accounted for 3851 blocks on
    episodes that were already correct -- the second-largest false-alarm class in the
    run, and the largest on WorkBench.

    Parsing is the right fix rather than relaxing to containment. Containment is what
    `compile._in_result` already does and it is why identity exists: a substring test
    lets a TRUNCATED id hit inside the real one. Parsing keeps identity exactly as
    strict as intended and merely lets it see the values.

    A result that does not parse is returned unchanged, so a genuinely plain-text
    output still collects as one scalar -- which is correct for the ones that are a
    bare value (`calendar_create_event` returns '00000300') and correctly unhelpful
    for prose like 'Email sent successfully.'
    """
    if not isinstance(res, str):
        return res
    t = res.strip()
    if not t or t[0] not in "[{":
        return res
    try:
        parsed = json.loads(t)
    except (ValueError, TypeError):
        return res
    return parsed if isinstance(parsed, (dict, list)) else res


def _collect_scalars(node: Any, out: List[Any]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            _collect_scalars(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _collect_scalars(v, out)
    elif node is not None and not isinstance(node, bool):
        out.append(node)
    if len(out) > 5000:
        del out[5000:]


_TYPED_REF = re.compile(r"^([A-Za-z][A-Za-z0-9]*)/(.+)$")


def _same_entity(got: Any, returned: Any) -> bool:
    """Identity, tolerating a typed-reference prefix.

    FHIR writes a reference as `<Type>/<id>` while the lookup returns the bare id, so
    strict equality would fail every correct write -- `values_match("Patient/S2380121",
    "S2380121")` is False. This is still far stronger than `compile._in_result`, which
    accepts the value as a substring ANYWHERE in the payload: here the tolerance is
    anchored on the ARGUMENT side and only strips one leading type segment, so a
    truncated or invented id is still caught.
    """
    if values_match(got, returned):
        return True
    m = _TYPED_REF.match(str(got).strip())
    return bool(m) and values_match(m.group(2), returned)


def arg_is_output_value(consumers: Sequence[str], arg: str,
                        producers: Sequence[str], A: dict,
                        extract: Optional[dict] = None):
    """The consumed value IS one the producer returned -- identity, not containment.

    Strictly stronger than arg_from_output, which accepts a substring hit anywhere in
    the payload.
    """
    Predicate = A["Predicate"]
    cons, prods = list(consumers), list(producers)

    def fn(trace, position, metrics=None):
        used = [c for c in _calls(trace, cons) if _arg_at(c, arg)]
        if not used:
            return {"passed": True, "note": f"vacuous: no {cons[:2]} called with {arg}."}
        for c in used:
            avail = _output_values(trace, prods, extract, before=c.position)
            if avail is None:
                return {"passed": True,
                        "note": "unevaluable: no recorded producer output to compare."}
            if not avail:
                return {"passed": False,
                        "note": (f"{c.name}(#{c.position+1}) set {arg} although "
                                 f"{prods[:2]} returned nothing to take it from.")}
            for got in _arg_at(c, arg):
                if any(_same_entity(got, v) for v in avail):
                    continue
                # THE PRODUCER LIST IS OUR GUESS; THE QUESTION IS READ-OR-INVENTED.
                # The error this predicate exists to catch is a value the agent made
                # up, and reading it from ANY earlier result is reading it. Which
                # tool our graph nominated is a separate claim, and where it is wrong
                # a correct run fails: tau binds order_id to
                # find_user_id_by_name_zip -- which returns a USER id -- while the
                # order id really comes from get_user_details, so
                # `order_id='#W1304208' is not any value [...] returned` was reported
                # about a value the agent had plainly read. 91 failures on CORRECT
                # tau traces, the second-largest class the trace-regression gate
                # found.
                #
                # Identity is preserved, which is the part that matters: this widens
                # WHERE the value may have come from, not HOW closely it must match,
                # so a truncated or invented id is still caught -- that distinction
                # is the whole reason this predicate exists over ArgFromOutput's
                # substring test. The ordering obligation is unaffected; the data
                # edge still asserts a producer ran first, separately.
                anywhere = _output_values(trace, None, extract, before=c.position)
                if anywhere and any(_same_entity(got, v) for v in anywhere):
                    continue
                return {"passed": False,
                        "note": (f"{c.name}(#{c.position+1}) {arg}={got!r} is not any "
                                 f"value returned by {prods[:2]} or by any earlier "
                                 "call -- it was composed, not read.")}
        return {"passed": True, "note": f"{arg} is a value {prods[:2]} returned."}

    return Predicate(fn=fn, description=f"{cons[:2]}.{arg} IS an output of {prods[:2]}",
                     runtime_safe=True)


def same_value_across(slots: Sequence[dict], A: dict,
                      producers: Optional[Sequence[str]] = None,
                      extract: Optional[dict] = None):
    """Every listed (tool, arg) slot carries the SAME value -- a bound variable.

    This is the part no per-argument predicate can express: task9 orders potassium and
    pairs it with a follow-up level, and both must concern the patient the lookup
    returned -- not merely a patient each. Where `producers` is given, the shared value
    must additionally be one they returned.

    Vacuous below two present slots: with one call there is nothing to agree with, and
    demanding otherwise would turn this into a presence requirement.
    """
    Predicate = A["Predicate"]
    want = [(str(s.get("tool")), str(s.get("arg"))) for s in slots
            if s.get("tool") and s.get("arg")]

    def fn(trace, position, metrics=None):
        present: List[tuple] = []
        for tool, arg in want:
            for c in _calls(trace, [tool]):
                for v in _arg_at(c, arg):
                    present.append((tool, arg, c.position, v))
        if len(present) < 2:
            return {"passed": True,
                    "note": f"vacuous: fewer than two of {want[:3]} carried a value."}
        first = present[0]
        for tool, arg, pos, v in present[1:]:
            if not (_same_entity(v, first[3]) or _same_entity(first[3], v)):
                return {"passed": False,
                        "note": (f"{tool}.{arg}={v!r} at #{pos+1} differs from "
                                 f"{first[0]}.{first[1]}={first[3]!r} -- these steps must "
                                 "concern the same entity.")}
        if producers:
            avail = _output_values(trace, list(producers), extract)
            if avail is None:
                return {"passed": True,
                        "note": "unevaluable: no recorded producer output to compare."}
            if not any(_same_entity(first[3], v) for v in avail):
                return {"passed": False,
                        "note": (f"the shared value {first[3]!r} is not any value "
                                 f"{list(producers)[:2]} returned.")}
        return {"passed": True,
                "note": f"all {len(present)} slots carry {first[3]!r}."}

    return Predicate(fn=fn,
                     description=f"one entity across {want[:3]}", runtime_safe=True)


def observed_then_required(producers: Sequence[str], extract: Optional[dict],
                           trigger: dict, consumers: Sequence[str], A: dict):
    """If what the lookup returned TRIGGERS the branch, the action is owed.

    The obligation the emitter was missing, and where the remaining blindness lived.
    Every argument constraint on a conditional write is `vacuous_if_absent`-guarded --
    correctly, since a run that legitimately takes the other branch must not fail
    them -- so an episode that simply NEVER ORDERS satisfied every active constraint
    and scored 1.0 while being wrong. Measured on task9: wrong baseline episodes
    averaged 0.984.

    The guard is trace-evaluable, which is what makes this expressible at all: the
    deciding value is itself a recorded tool_result, so "the level was low" is a fact
    about the trace rather than a condition we have to take on trust.

    Vacuity, as everywhere in this module:
      * producer never called, or no result recorded -> UNEVALUABLE, not violated
      * the value does not trigger                   -> vacuously true
      * it triggers and the action happened          -> satisfied
      * it triggers and the action did not           -> VIOLATED
    """
    Predicate = A["Predicate"]
    prods, cons = list(producers), list(consumers)
    kind = str((trigger or {}).get("kind") or "")

    def fires(value: float) -> Optional[bool]:
        if kind == "in_bands":
            return _band_for(value, (trigger or {}).get("bands") or []) is not None
        if kind == "below":
            try:
                return value < float(trigger["value"])
            except (KeyError, TypeError, ValueError):
                return None
        return None

    def fn(trace, position, metrics=None):
        seen = [c for c in _calls(trace, prods)]
        if not seen:
            return {"passed": True, "note": f"vacuous: no {prods[:2]} called."}
        value = None
        for c in reversed(seen):
            value = observed_value(getattr(c, "result", None), extract)
            if value is not None:
                break
        if value is None:
            return {"passed": True,
                    "note": "unevaluable: no recorded lookup output to decide on."}
        hit = fires(value)
        if hit is None:
            return {"passed": True, "note": "unevaluable: trigger not expressible."}
        if not hit:
            return {"passed": True,
                    "note": f"vacuous: observed {value}, which does not trigger."}
        did = [c for c in _calls(trace, cons)]
        if did:
            return {"passed": True,
                    "note": f"observed {value} triggered, and {cons[:2]} was called."}
        return {"passed": False,
                "note": (f"observed {value} triggers the required action, but "
                         f"{cons[:2]} was never called.")}

    return Predicate(fn=fn,
                     description=f"{cons[:2]} required when {prods[:2]} triggers",
                     runtime_safe=False)


# --------------------------------------------------------------------------- #
# SCOPE: the ordering predicates cannot see arguments, and the argument
# predicates cannot see order, so the conjunction of the two is not what either
# one says.
#
# `Before(a, b)` compares `first_index(a)` against `first_index(b)`
# (_evaluator.py:252-253) -- tool names only. `called_with_like` passes when ANY
# call to the tool matches (matching.py:247-272). So the trace
#
#     a(wrong args) -> b -> a(right args)
#
# satisfies BOTH: each predicate passes on a DIFFERENT call, and "the call the
# task describes ran before b" is asserted by neither. This is
# CLAUDE.md's `mab_task3_1` -- enforcement drove compliance 0.6 -> 1.0 by
# producing a SECOND compliant POST while the first stayed wrong, and refsol
# grades posts[0] -- stated in general form. It is live in 62% of bfcl episodes
# and 35% of mab's, those being the fraction in which some tool is called more
# than once.
#
# In Dwyer's property-pattern vocabulary these are the patterns we lacked: a
# PRECEDENCE whose antecedent is a specific call rather than a tool name, and an
# ABSENCE at "between Q and R" scope. Every other constraint we emit is at
# GLOBAL scope.
# --------------------------------------------------------------------------- #
def matching_call_before(a: str, a_args: dict, b: str, A: dict):
    """The call to `a` that the TASK DESCRIBES ran before `b` -- not merely some call.

    Three-valued, and the UNEVALUABLE case is the one that keeps this honest: when
    calls to `a` exist but none match, the error is a BINDING error, which the
    `bind::` constraint on that slot already reports. Failing here too would
    double-count it -- the same rule the date grammar follows, where prose in a date
    slot is unevaluable rather than violated because it is a different error.

    So this predicate fails on exactly one thing: a matching call to `a` EXISTS, and
    every one of them comes after `b`. That is the ordering claim and nothing else.
    """
    Predicate = A["Predicate"]
    want = dict(a_args or {})

    def fn(trace, position, metrics=None):
        calls = list(trace.calls)
        b_idx = next((i for i, c in enumerate(calls) if c.name == b), None)
        if b_idx is None:
            # Guarded emission makes this the normal partial-trace case: `b` is the
            # call being considered, so until it is made there is nothing to order.
            return {"passed": True, "note": f'"{b}" has not been called.'}
        a_idx = [i for i, c in enumerate(calls) if c.name == a]
        if not a_idx:
            return {"passed": True,
                    "note": f'"{a}" was never called, so this ordering is vacuous.'}
        hit = [i for i in a_idx if args_match(want, calls[i].args or {})]
        if not hit:
            return {"passed": True,
                    "note": (f'unevaluable: {len(a_idx)} call(s) to "{a}" but none '
                             f"carries {_slot_list(want)}; the argument constraint on "
                             "that slot owns this error.")}
        if min(hit) < b_idx:
            return {"passed": True,
                    "note": (f'"{a}" carrying {_slot_list(want)} ran at #{min(hit) + 1}, '
                             f'before "{b}" at #{b_idx + 1}.')}
        return {"passed": False,
                "note": (f'"{b}" ran at #{b_idx + 1}, but the "{a}" call carrying '
                         f"{_slot_list(want)} only ran at #{min(hit) + 1}. An earlier "
                         f'"{a}" call carried something else, so "{b}" did not act on '
                         f"it. Make the {_slot_list(want)} call to \"{a}\" first, then "
                         f'call "{b}".')}

    return Predicate(fn=fn,
                     description=f"the {a} carrying {_slot_list(want)} precedes {b}",
                     runtime_safe=True)


def absence_between(x: str, lo: str, hi: str, A: dict):
    """Dwyer ABSENCE at BETWEEN scope: no call to `x` between `lo` and `hi`.

    Vacuous unless BOTH bounds are present, for the reason `Before` needs a guard:
    an interval with no end has not been left yet, so nothing can be inside it.
    """
    Predicate = A["Predicate"]

    def fn(trace, position, metrics=None):
        calls = list(trace.calls)
        lo_i = next((i for i, c in enumerate(calls) if c.name == lo), None)
        hi_i = next((i for i, c in enumerate(calls) if c.name == hi), None)
        if lo_i is None or hi_i is None or hi_i <= lo_i:
            return {"passed": True,
                    "note": f'the interval "{lo}".."{hi}" is not closed in this trace.'}
        inside = [i for i, c in enumerate(calls)
                  if c.name == x and lo_i < i < hi_i]
        if not inside:
            return {"passed": True,
                    "note": f'no "{x}" between "{lo}" and "{hi}".'}
        return {"passed": False,
                "note": (f'"{x}" ran at #{inside[0] + 1}, between "{lo}" (#{lo_i + 1}) '
                         f'and "{hi}" (#{hi_i + 1}). Do not call "{x}" once "{lo}" has '
                         f'settled what "{hi}" acts on.')}

    return Predicate(fn=fn, description=f"no {x} between {lo} and {hi}",
                     runtime_safe=True)


def _slot_list(want: dict) -> str:
    """Name the slots, not the payload -- `why_not`'s rule, and the smoke gate's.

    `check_live_smoke._actionable` fails a message naming more than 6 slots: naming
    everything is not naming anything.
    """
    if not want:
        return "no particular argument"
    items = [f"{k}={v!r}" for k, v in sorted(want.items())[:3]]
    more = len(want) - len(items)
    return ", ".join(items) + (f" (+{more} more)" if more > 0 else "")
