# -*- coding: utf-8 -*-
"""What a parameter's OWN schema says about its value -- and nothing else.

WHY THIS EXISTS

Run 6 attributed every discordant pair to the constraint class that blocked. The
`bind::*_bound` class -- "the argument equals the value quoted in the task" -- is
307W/8L on mab and 0.05-0.31 W:L everywhere else, and it is 3770 of 5352 blocks
landing on episodes whose own baseline was correct. The messages say why:

    travel_from is 'RMS', but the task states 'Rivermist'
    travel_class is 'business', but the task states 'business class'
    symbol is 'ZETA', but the task states 'Zeta Corp'
    mode is 'c', but the task states 'count'
    a is True, but the task states '-a'

In every one of those the agent is RIGHT and the constraint is wrong: the prose
names an entity or an option, and the wire takes its canonical form. A prose
quote is evidence of intent, not of the argument.

So the rule is inverted here. Instead of asserting a stated literal and hoping,
assert it only on positive evidence from the parameter's own schema -- a formal
`enum`, an option list stated in its description, a declared non-string type, or
a stated format. Where the schema constrains nothing, nothing is asserted.

WHAT EACH FAMILY ACTUALLY SUPPLIES -- measured, because it decides the yield

    BFCL       222 params: 222 type, 222 description,  2 enum
    tau-bench   74 params:  74 type,  70 description,  5 enum
    WorkBench   79 params:  79 type,  79 description,  0 enum

BFCL has almost no formal enum but states the members in the DESCRIPTION, and it
is the per-parameter description that says so, not the tool's:

    wc.mode       "Mode of operation ('l' for lines, 'w' for words, 'c' for characters)."
    travel_class  "The class of the travel. Options are: economy, business, first."
    order_type    "Type of the order (Buy/Sell)."
    travel_from   "The 3 letter code of the departing airport"

WorkBench's are auto-generated title-case echoes of the parameter name --
`{"field": {"description": "Field"}}`, `{"plot_type": {"description": "Plot Type"}}`
-- so they have the SHAPE of documentation and none of the content. Treating them
as informative is what lets `field is 'event_name', but the task states 'name'`
become a blocking constraint. `is_informative` is the guard, and it is why the
same rule sheds harm on workbench while gaining recall on bfcl.

CANONICALISE RATHER THAN DROP, WHERE THE SCHEMA ALLOWS IT

Dropping an unassertable value loses the constraint. Mapping 'business class' onto
the member `business` keeps it AND makes it correct, which is recall rather than
mere precision. Only an unambiguous match is taken; ambiguity drops.

Reads tool schemas only, so it is on the AgentView side of the boundary.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

# ── option lists stated in prose ──────────────────────────────────────────────
# Ordered: the most explicit phrasing wins, so "Options are: a, b" is not first
# matched by the looser parenthetical rule.
_OPTS_PATTERNS = (
    # "Options are: economy, business, first."  /  "one of: a, b, c"
    re.compile(r"(?:options?\s+(?:are|include)|one\s+of|must\s+be\s+one\s+of|"
               r"valid\s+values?\s+(?:are|is)|choose\s+from)\s*:?\s*([^.;]+)",
               re.I),
    # "('l' for lines, 'w' for words, 'c' for characters)"  -- quoted members
    re.compile(r"\(\s*((?:['\"][^'\"]+['\"][^)]*?)+)\)"),
    # "(Buy/Sell)"  /  "(yes/no)"  -- slash-separated, no quotes
    re.compile(r"\(\s*([A-Za-z][\w -]*(?:\s*/\s*[A-Za-z][\w -]*)+)\s*\)"),
)

# A member list must look like members: short, no sentence punctuation.
_MEMBER_MAX_WORDS = 4
_MEMBER_MAX_LEN = 40

# ── stated formats ────────────────────────────────────────────────────────────
_FORMATS = (
    # "The 3 letter code of the departing airport"
    (re.compile(r"\b(\d+)[\s-]*letter\s+code\b", re.I), "letter_code"),
    # "the date of the travel in the format 'YYYY-MM-DD'"
    (re.compile(r"\bformat\s*['\"]?(Y{2,4}-?M{1,2}-?D{1,2}[^'\"]*)['\"]?", re.I), "date_fmt"),
    # "The zipcode of the first city." -- `estimate_distance` takes zipcodes and the
    # task names cities, so the prose value is never the argument. 50 of run 6's
    # remaining failures on correct traces are these two parameters alone.
    (re.compile(r"\b(zip\s?code|postal\s?code)\b", re.I), "zipcode"),
)

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


# ── the description's information content ────────────────────────────────────
def _norm_words(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def is_informative(name: str, pd: Dict[str, Any]) -> bool:
    """False when the description merely restates the parameter's own name.

    WorkBench generates `{"plot_type": {"description": "Plot Type"}}` for all 79 of
    its parameters. That is not documentation; asserting against it is asserting
    against nothing. A description that adds no token beyond the name -- and that
    carries no enum, format or non-string type -- licenses no value constraint.
    """
    if pd.get("enum"):
        return True
    desc = str(pd.get("description") or "")
    if not desc.strip():
        return False
    if _norm_words(desc) == _norm_words(name):
        return False
    # "Time Min" for `time_min`: same tokens, possibly reordered or abbreviated.
    dw, nw = set(_norm_words(desc).split()), set(_norm_words(name).split())
    return not (dw and dw <= nw)


# ── option extraction ────────────────────────────────────────────────────────
def _clean_members(raw: str) -> List[str]:
    # "choose from the following holder types: major_holders, institutional_holders"
    # -- the pattern stops at "choose from", so the noun phrase and its colon are
    # still glued to the FIRST member, which then fails the word-count test and is
    # silently lost. Measured on mcpu: `major_holders` dropped out of a six-member
    # list while the other five survived, so the option set was wrong in exactly the
    # way that looks like a working parser. Keep only what follows the last colon.
    if ":" in raw:
        raw = raw.rsplit(":", 1)[1]
    # Quoted members win: "'l' for lines, 'w' for words" -> l, w
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", raw)
    parts = quoted if quoted else re.split(r"\s*(?:,|/|\bor\b|\band\b)\s*", raw)
    out: List[str] = []
    for p in parts:
        p = p.strip().strip(".;:").strip()
        # drop a trailing gloss the split left behind ("lines" from "'l' for lines")
        p = re.sub(r"\s+for\s+\w+$", "", p, flags=re.I).strip()
        if not p or len(p) > _MEMBER_MAX_LEN or len(p.split()) > _MEMBER_MAX_WORDS:
            continue
        if p.lower() in ("etc", "e g", "eg", "i e", "ie", "optional", "default"):
            continue
        out.append(p)
    # A single "member" is a gloss, not a list.
    seen, uniq = set(), []
    for p in out:
        if p.lower() not in seen:
            seen.add(p.lower()); uniq.append(p)
    return uniq if len(uniq) >= 2 else []


def options(name: str, pd: Dict[str, Any]) -> Optional[List[str]]:
    """The legal members, from a formal enum or from the description's own words."""
    enum = pd.get("enum")
    if isinstance(enum, (list, tuple)) and len(enum) >= 2:
        return [str(x) for x in enum]
    if not is_informative(name, pd):
        return None
    desc = str(pd.get("description") or "")
    for pat in _OPTS_PATTERNS:
        m = pat.search(desc)
        if not m:
            continue
        members = _clean_members(m.group(1))
        if members:
            return members
    return None


def format_hint(name: str, pd: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """A stated shape the value must take, e.g. a 3-letter code or a date format."""
    if not is_informative(name, pd):
        return None
    desc = str(pd.get("description") or "")
    for pat, kind in _FORMATS:
        m = pat.search(desc)
        if m:
            return (kind, m.group(1))
    return None


# ── the format may be documented on a SIBLING tool, not on this one ──────────
# THE LARGEST REMAINING CLASS, measured. After the paramspec rule landed, bfcl still
# had 386 blocking failures on run 6's CORRECT traces and **92% of them are this**:
#
#     book_flight.travel_from      "The location the travel is from"
#     get_flight_cost.travel_from  "The 3 letter code of the departing airport"
#
#     book_flight.travel_class     "The class of the travel"
#     get_flight_cost.travel_class "The class of the travel. Options are: economy,
#                                   business, first."
#
# The SAME parameter name, in the SAME tool family, documented on one tool and not
# the other. Reading only the parameter's own tool leaves `book_flight.travel_class`
# with no options, so the prose 'business class' is asserted and the agent's correct
# 'business' is blocked. Same for travel_from/travel_to ('SFO', 'RMS', 'LAX').
#
# THIS IS NOT A LEAK. The sibling schema is in the SAME `AgentView` the agent reads,
# so an agent could consult it exactly as we do. The rule stays "positive evidence
# from a schema the agent can see" -- it is only the SCOPE of "the schema" that was
# too narrow.
#
# The guard against borrowing an unrelated list is `canonicalise`, not a similarity
# test: a borrowed option list can only ever REWRITE a stated value to a member it
# genuinely denotes, or fail to and drop the assertion. It cannot invent a value the
# prose does not name. So a wrong borrow costs a constraint, never a false alarm --
# which is the safe direction, and the same asymmetry `assertable` already relies on.
#
# Off by default. Like GRAPH_DECLARED it changes what the corpus ASSERTS, so it
# needs its own tag and a spec diff. PARAM_SIBLING=1 enables it.
SIBLING = os.environ.get("PARAM_SIBLING", "") not in ("", "0", "false", "False")


def enrich(name: str, pd: Dict[str, Any],
           siblings: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """`pd`, plus options/format borrowed from an identically-named sibling parameter.

    Borrows a facet ONLY where this parameter's own schema states none, so a stated
    enum or format always wins. Returns `pd` unchanged when the flag is off, when
    there are no siblings, or when nothing was borrowed -- so the caller can stay
    unconditional and the off-path is byte-identical.
    """
    if not (SIBLING and siblings):
        return pd
    out = dict(pd)
    if options(name, pd) is None:
        for sib in siblings:
            got = options(name, sib) if sib else None
            if got:
                out["enum"] = list(got)
                out.setdefault("_borrowed", []).append("options")
                break
    if format_hint(name, out) is None:
        for sib in siblings:
            got = format_hint(name, sib) if sib else None
            if got:
                # Carry the sibling's DESCRIPTION so `format_hint` re-derives the
                # same hint rather than storing a parsed tuple the rest of the
                # module would have to learn about.
                out["description"] = str(sib.get("description") or "")
                out.setdefault("_borrowed", []).append("format")
                break
    return out


# ── the two questions the emitter asks ───────────────────────────────────────
def _member_match(value: Any, members: List[str]) -> Optional[str]:
    """The one member `value` denotes, or None if none or several."""
    v = str(value).strip().lower()
    if not v:
        return None
    exact = [m for m in members if m.lower() == v]
    if len(exact) == 1:
        return exact[0]
    # SEPARATOR-BLIND equality: 'balance sheet' -> balance_sheet,
    # 'quarterly cash flow' -> quarterly_cashflow. The member's own spelling is the
    # wire form and the task states the words; nothing but punctuation separates
    # them, so this cannot pick a member the prose does not name.
    squash = lambda x: re.sub(r"[^a-z0-9]+", "", str(x).lower())
    sq = [m for m in members if squash(m) == squash(v)]
    if len(sq) == 1:
        return sq[0]
    # 'business class' -> business ; 'count' -/-> c (no substring on 1-char members)
    loose = [m for m in members
             if len(m) > 1 and re.search(rf"\b{re.escape(m.lower())}\b", v)]
    if len(loose) == 1:
        return loose[0]
    # ...and the other direction: the value names the member's leading words
    # ('institutional' -> institutional_holders). UNIQUENESS is what makes this
    # safe -- 'insider' leads three of yfinance's six holder types and so resolves
    # to none of them, which is the right answer.
    vw = _norm_words(v).split()
    if vw:
        pref = [m for m in members if _norm_words(m).split()[:len(vw)] == vw]
        if len(pref) == 1:
            return pref[0]
    return None


def _type_ok(value: Any, declared: str) -> bool:
    d = (declared or "").lower()
    s = str(value).strip()
    if d in ("boolean", "bool"):
        return isinstance(value, bool) or s.lower() in (_TRUE | _FALSE)
    if d in ("integer", "int"):
        if isinstance(value, bool):
            return False
        try:
            int(s); return True
        except Exception:
            return False
    if d in ("number", "float", "double"):
        if isinstance(value, bool):
            return False
        try:
            float(s); return True
        except Exception:
            return False
    return True  # string / array / object / undeclared: no type evidence either way


def _format_ok(value: Any, hint: Tuple[str, Any]) -> bool:
    kind, arg = hint
    s = str(value).strip()
    if kind == "letter_code":
        try:
            n = int(arg)
        except Exception:
            return True
        return bool(re.fullmatch(rf"[A-Za-z]{{{n}}}", s))
    if kind == "date_fmt":
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}([T ].*)?", s))
    if kind == "zipcode":
        # US 5-digit, optionally +4. A city name is not one, which is the point.
        return bool(re.fullmatch(r"\d{5}(-\d{4})?", s))
    return True


def assertable(name: str, pd: Dict[str, Any], value: Any) -> Tuple[bool, str]:
    """May a constraint assert this stated value on this parameter?

    Returns (ok, reason). `reason` names the evidence, so the emitter can record
    WHY a value was dropped rather than dropping it silently -- three times a
    silent drop has been indistinguishable from a slot that had nothing to say.
    """
    if pd is None:
        return True, "no-schema"                     # unchanged behaviour
    if isinstance(value, (dict, list)):
        return True, "structured"                    # not a scalar binding
    opts = options(name, pd)
    if opts:
        return ((True, "enum-member") if _member_match(value, opts) is not None
                else (False, f"not one of {opts}"))
    declared = str(pd.get("type") or "")
    if not _type_ok(value, declared):
        return False, f"not a {declared}"
    hint = format_hint(name, pd)
    if hint and not _format_ok(value, hint):
        return False, f"violates stated format {hint[0]}={hint[1]!r}"
    if declared.lower() in ("string", "") and not is_informative(name, pd):
        # Nothing in the schema constrains a free string. WorkBench is entirely
        # here, and it is where the prose-quote class does pure harm.
        return False, "schema states nothing about this value"
    return True, "type-ok"


def canonicalise(name: str, pd: Dict[str, Any], value: Any) -> Tuple[Any, bool]:
    """Map a prose value onto the form the wire takes; (value, changed).

    Keeping `business` instead of dropping `business class` is the difference
    between shedding a false alarm and gaining a true one.
    """
    if pd is None or isinstance(value, (dict, list)):
        return value, False
    opts = options(name, pd)
    if opts:
        m = _member_match(value, opts)
        if m is not None and str(m) != str(value):
            return m, True
        return value, False
    declared = str(pd.get("type") or "").lower()
    s = str(value).strip()
    if declared in ("boolean", "bool") and not isinstance(value, bool):
        if s.lower() in _TRUE:
            return True, True
        if s.lower() in _FALSE:
            return False, True
    if declared in ("integer", "int") and not isinstance(value, int):
        try:
            return int(s), True
        except Exception:
            return value, False
    if declared in ("number", "float", "double") and not isinstance(value, float):
        try:
            return float(s), True
        except Exception:
            return value, False
    return value, False
