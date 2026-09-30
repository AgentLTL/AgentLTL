#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
agentltl/matching.py
===============================================================================
Format-tolerant argument matching.

62% of gold failures in the first pilot were CalledWith constraints whose intent
was right and whose FORMAT differed: "20" vs 20, "2023-12-19T00:00:00" vs
"2023-12-19", "True" vs True. A compliance score that moves on those is measuring
JSON formatting, not procedure.

Tolerance is limited to OBJECTIVE equivalences -- case, surrounding quotes,
numeric and boolean spelling, and date granularity. Prose-instead-of-value
("'Projects' folder" where the tool wants "Projects") is deliberately NOT
tolerated: that is a real generation error and must keep showing up as one.
===============================================================================
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Sequence

_NUM = re.compile(r"^-?\d+(\.\d+)?$")
_ISO = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?$")
_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}



def _scalars(obj: Any, out: List[str]) -> None:
    """Every leaf of a nested result, rendered as a string, appended to `out`."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _scalars(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _scalars(v, out)
    else:
        out.append(str(obj))

def _unquote(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def _canon(v: Any) -> Any:
    """A comparable canonical form, or the lowercased string as a fallback."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return tuple(_canon(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _canon(x)) for k, x in v.items()))
    s = _unquote(str(v))
    low = s.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    if _NUM.match(s):
        return float(s)
    return low


def _date_parts(v: Any):
    m = _ISO.match(_unquote(str(v)))
    if not m:
        return None
    return m.group(1), (m.group(2), m.group(3))


def values_match(expected: Any, actual: Any) -> bool:
    """Is `actual` an acceptable rendering of `expected`?"""
    if _canon(expected) == _canon(actual):
        return True

    # date granularity: a bare date matches a midnight datetime, and vice versa
    de, da = _date_parts(expected), _date_parts(actual)
    if de and da and de[0] == da[0]:
        for parts in (de[1], da[1]):
            if parts[0] not in (None, "00") or parts[1] not in (None, "00"):
                # a real time-of-day on either side must agree
                return de[1] == da[1]
        return True

    # a single value satisfies a one-element list form, in either direction
    if isinstance(expected, (list, tuple)) and len(expected) == 1:
        return values_match(expected[0], actual)
    if isinstance(actual, (list, tuple)) and len(actual) == 1:
        return values_match(expected, actual[0])

    # membership: expected value present in an actual list
    if isinstance(actual, (list, tuple)):
        return any(values_match(expected, a) for a in actual)
    return False


# A tool whose whole body is one JSON document arrives as {"payload": "<json>"}.
# MedAgentBench's POST tools are the case: their FHIR resources nest four levels
# deep and a flat tool schema cannot express that, so the runner takes the resource
# as one string argument.
#
# WHY THIS MATTERS AT RUNTIME. The post-hoc trace is normalised (the runner parses
# the payload before recording it), but the ENFORCER sees the call as the agent made
# it -- a single string. Without unwrapping, `{"valueString": "118/77 mmHg"}` matched
# nothing, so an L2 constraint on any POST field failed for EVERY call, correct or
# not: with L2 blocking, every write was blocked, and the arm would have reported
# "enforcement destroys task success" as a finding rather than as a bug here.
_BODY_KEYS = ("payload", "body", "resource", "json", "data")


def unwrap_json_args(args: Any) -> dict:
    """Args with a single JSON-document argument parsed and merged in.

    Conservative on purpose: only a lone body-shaped key whose string parses to a
    JSON OBJECT is unwrapped, and the original key is kept so a constraint written
    against `payload` itself still resolves.
    """
    if not isinstance(args, dict):
        return {}
    for k in _BODY_KEYS:
        v = args.get(k)
        if not isinstance(v, str):
            continue
        t = v.strip()
        if not (t.startswith("{") and t.endswith("}")):
            continue
        try:
            parsed = json.loads(t)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            merged = dict(args)
            merged.update(parsed)
            return merged
    return args


# A FHIR reference is written `<Type>/<id>`, and the schema for such a slot says only
# "the patient FHIR ID" -- so the model extracts the bare MRN from the task text while
# the environment (and refsol) want the prefixed form. Measured: task8's agent sent
# the CORRECT `Patient/S2016972`, our constraint asserted `S2016972`, and
# `values_match` is False on that pair, so a blocking layer refused a correct write.
#
# Accepting both is not widening prose matching -- it is one identity written two
# ways, the same job values_match already does for "20" vs 20 and for date
# granularity. It is confined to reference-shaped slots so it cannot leak into
# ordinary values.
_REF_LEAVES = ("reference", "subject", "patient", "requester", "performer",
               "encounter", "beneficiary")
_TYPED_REF = re.compile(r"^([A-Za-z][A-Za-z0-9]*)/(.+)$")


def _is_ref_slot(path: str) -> bool:
    leaf = str(path).rstrip("[]").rsplit(".", 1)[-1].lower()
    return leaf in _REF_LEAVES


def _ref_match(expected: Any, actual: Any) -> bool:
    """values_match, plus identity up to ONE leading `<Type>/` segment."""
    if values_match(expected, actual):
        return True
    for a, b in ((actual, expected), (expected, actual)):
        m = _TYPED_REF.match(str(a).strip())
        if m and values_match(m.group(2), b):
            return True
    return False


def args_match(expected: dict, actual: dict) -> bool:
    """Subset semantics, like CalledWith, but per-value tolerant.

    A key containing "." or "[]" is a PATH into a nested argument, resolved the way
    the relative predicates resolve theirs. FHIR payloads are the reason: the value
    a MedAgentBench task states lands at `subject.reference` or
    `code.coding[].code`, never at a top-level key, so flat lookup compared a
    stated MRN against a whole nested object and failed every correct call. Flat
    keys take the original path and behave exactly as before.
    """
    act = unwrap_json_args(actual or {})
    for k, v in (expected or {}).items():
        cmp = _ref_match if _is_ref_slot(k) else values_match
        if "." in str(k) or "[" in str(k):
            from .relative import path_values
            found = path_values(act, str(k))
            if not found or not any(cmp(v, a) for a in found):
                return False
            continue
        if k not in act:
            return False
        if not cmp(v, act[k]):
            return False
    return True


def why_not(expected: dict, actual: dict) -> str:
    """Per-slot reason, in terms the agent can act on.

    The old message dumped the whole payload beside the whole expectation and left the
    agent to diff them. Measured consequence: task8 sent `note` as an ARRAY where the
    schema and the grader both want an object, was refused five times with the same
    undifferentiated dump, never worked out what was wrong, and the episode ended with
    NO POST at all. A block the agent cannot act on is a block that destroys the
    action -- so the reason has to name the slot and say whether it is missing, the
    wrong SHAPE, or the wrong value.
    """
    act = unwrap_json_args(actual or {})
    from .relative import path_values
    out = []
    for k, v in (expected or {}).items():
        key = str(k)
        cmp = _ref_match if _is_ref_slot(key) else values_match
        if "." in key or "[" in key:
            found = path_values(act, key)
            if found:
                if any(cmp(v, a) for a in found):
                    continue
                out.append(f"{key} is {found[0]!r}, but the task states {v!r}")
                continue
            # Nothing at that path. Say WHERE it stopped and what shape is there,
            # because array-vs-object is the common case and is invisible otherwise.
            parts = [x for x in key.split(".") if x]
            probe, walked = act, []
            for seg in parts[:-1]:
                bare = seg.rstrip("[]")
                nxt = probe.get(bare) if isinstance(probe, dict) else None
                if nxt is None:
                    out.append(f"{key} is missing: nothing at {'.'.join(walked + [bare])}")
                    break
                walked.append(bare)
                if seg.endswith("[]") and isinstance(nxt, list):
                    probe = nxt[0] if nxt else {}
                elif not seg.endswith("[]") and isinstance(nxt, list):
                    out.append(f"{key} is missing: {'.'.join(walked)} is a LIST, but "
                               f"{key} expects an object there")
                    break
                else:
                    probe = nxt
            else:
                leaf = parts[-1].rstrip("[]")
                out.append(f"{key} is missing: {'.'.join(walked) or 'the body'} has no "
                           f"{leaf!r}")
            continue
        if key not in act:
            out.append(f"{key} is missing")
        elif not cmp(v, act[key]):
            out.append(f"{key} is {act[key]!r}, but the task states {v!r}")
    return "; ".join(out) or "the arguments do not match"


def called_with_like(tool: str, expected_args: dict, A: dict):
    """CalledWith with tolerant value comparison. Vacuous when the tool is absent,
    so it constrains HOW a tool is used and never WHETHER it is used."""
    Predicate = A["Predicate"]
    exp = dict(expected_args or {})

    def fn(trace, position, metrics=None):
        calls = [c for c in trace.calls if c.name == tool]
        if not calls:
            return {"passed": True, "note": f'vacuous: "{tool}" not called.'}
        for c in calls:
            if args_match(exp, c.args or {}):
                return {"passed": True,
                        "note": f'"{tool}" called with {exp} (format-tolerant).'}
        return {"passed": False,
                "note": (f'no "{tool}" call matched what the task states. '
                         + why_not(exp, calls[-1].args or {}))}

    return Predicate(fn=fn,
                     description=f"{tool} called with ~{exp}",
                     runtime_safe=True)


def never_called_with(tool: str, expected_args: dict, A: dict):
    """No call to `tool` matches `expected_args` -- including when it is never called.

    A prohibition CANNOT be Not(CalledWith(...)): CalledWith is vacuously TRUE when
    the tool is absent, so the negation reports a violation for a run that never did
    the forbidden thing. "Never book with insurance" must be satisfied by a run that
    never books at all.
    """
    Predicate = A["Predicate"]
    exp = dict(expected_args or {})

    def fn(trace, position, metrics=None):
        for c in trace.calls:
            if c.name == tool and args_match(exp, c.args or {}):
                return {"passed": False,
                        "note": f'"{tool}" was called with the forbidden {exp}.'}
        return {"passed": True, "note": f'no "{tool}" call used {exp}.'}

    return Predicate(fn=fn, description=f"{tool} never called with ~{exp}",
                     runtime_safe=True)


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_from_output(consumers, arg: str, producers, A: dict,
                        stated=None, min_len: int = 3, tol: float = 1e-4):
    """Every FIGURE inside a computed argument was returned by a prior producer.

    ArgFromOutput cannot express this: the expression string "20000/152.3*1.07"
    never appears verbatim in a price table, but 152.3 must. This is the
    fabricated-figure detector -- an agent that invents a stock price instead of
    fetching it fails here even though the arithmetic looks plausible.

    Short numbers (< min_len digits) are ignored: 2, 100 and the like are ordinary
    arithmetic constants, not fetched data, and demanding provenance for them would
    make the constraint impossible to satisfy.

    `stated` holds the figures the TASK ITSELF gives (an investment amount, a share
    count). Those are legitimate without being fetched -- the rule is "fetched OR
    stated", exactly as for literals in the leak gate. Omitting them made the
    constraint unsatisfiable: "20000/195.42*102.66" is correct, yet 20000 appears in
    no price table.

    **A prior call to the CONSUMER is itself a producer.** These tasks are chained
    arithmetic -- compute a mean, then use it in a ratio -- and the graph's data edge
    names only the tool that FETCHED the raw series. So a figure the agent obtained
    from its own earlier `calculate` appeared in no named producer's output and the
    correct call was refused:

        calculate(#5) used figure(s) ['361.49150085449'] that appear in no prior
        ['get_historical_stock_prices']... output.

    -- where 361.49150085449 is exactly what `calculate(#2)` RETURNED. On MCP-Universe
    this single omission is the family's largest false-alarm source: the four
    `*_figures_sourced` constraints fired 70 times on baseline-CORRECT episodes in run
    27 against 9 catches.

    It does not open a laundering path. A fabricated figure still has to be RETURNED by
    a tool, and `calculate("999999 * 1")` is itself a consumer call checked by this same
    predicate -- `used` is every call to the consumer tool -- so the invention is caught
    at the point it enters, before anything can build on it. The credit is inductive,
    and so is the refusal.
    """
    Predicate = A["Predicate"]
    cons, prods = list(consumers), list(producers)
    said = {str(x) for x in (stated or [])}

    def fn(trace, position, metrics=None):
        used = [c for c in trace.calls
                if c.name in cons and (c.args or {}).get(arg) is not None]
        if not used:
            return {"passed": True, "note": f"vacuous: no consumer called with {arg}."}
        for c in used:
            expr = str(c.args.get(arg))
            figures = [t for t in _NUMBER.findall(expr)
                       if len(t.replace("-", "").replace(".", "")) >= min_len]
            if not figures:
                continue
            # A prior call to the consumer counts as a producer: chained arithmetic
            # feeds its own result forward, and that value came from a tool.
            earlier = [p for p in trace.calls
                       if p.position < c.position
                       and (p.name in prods or p.name == c.name)]
            if not earlier:
                return {"passed": False,
                        "note": (f"{c.name}(#{c.position+1}) computed over {figures[:3]} "
                                 f"but none of {prods[:3]}... ran first -- the figures "
                                 "cannot have been fetched.")}
            pool: list = []
            for p in earlier:
                _scalars(getattr(p, "result", None), pool)
            if not pool:
                continue          # no recorded outputs: unevaluable, not violated
            blob = " ".join(pool)
            missing = []
            for f in figures:
                if f in blob or f in said:
                    continue
                try:
                    if any(abs(float(f) - float(x)) <= tol * max(1.0, abs(float(f)))
                           for x in said):
                        continue
                except ValueError:
                    pass
                try:                      # tolerate rounding in the agent's copy
                    v = float(f)
                    if any(abs(v - float(x)) <= tol * max(1.0, abs(v))
                           for x in _NUMBER.findall(blob)):
                        continue
                except ValueError:
                    pass
                missing.append(f)
            if missing:
                return {"passed": False,
                        "note": (f"{c.name}(#{c.position+1}) used figure(s) {missing[:3]} "
                                 f"that appear in no prior {prods[:2]}... output.")}
        return {"passed": True,
                "note": f"all figures in {arg} traced to fetched data."}

    return Predicate(fn=fn,
                     description=f"figures in {cons}.{arg} sourced from {prods[:2]}...",
                     runtime_safe=True)


def _leaf_paths_of(node: Any, prefix: str = "") -> List[str]:
    """Every leaf path a payload actually carries, in the schema's own notation."""
    out: List[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out += _leaf_paths_of(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, (list, tuple)):
        for v in node:
            out += _leaf_paths_of(v, f"{prefix}[]")
    elif prefix:
        out.append(prefix)
    return out


def args_only_declared(tool: str, declared: Sequence[str], A: dict,
                       ignore: Sequence[str] = ()):
    """The call carries only arguments the tool's schema declares.

    refsol asserts `payload['code'] == {'text': 'BP'}` -- an EXACT dict -- and the
    agent sent `{"coding": [...], "text": "BP"}`. Our `code.text` binding was
    satisfied and the grader still said no, because the body carried a field the
    schema does not declare: POST_Observation lists nine leaf paths and
    `code.coding[]` is not one of them.

    So this is a pure TOOL-SCHEMA derivation -- the declared parameter list, which the
    agent is handed -- and not a reading of the grader. It generalises to every write
    task: a body with invented structure is a different record from the one asked for.

    Vacuous when the tool is absent, like every argument rule here: it constrains HOW
    a tool is used, never WHETHER.
    """
    Predicate = A["Predicate"]
    allowed = {str(p) for p in declared}
    # A declared `a.b[]` legitimately covers `a.b[].c` in schemas that stop at the
    # array; and the body-shaped wrapper key itself is never a payload field.
    skip = {str(x) for x in ignore} | set(_BODY_KEYS)

    def _strip_lists(path: str) -> str:
        """`item_ids[]` -> `item_ids`, `a.b[].c` -> `a.b.c`.

        The two sides use different notation for the same field and neither is
        wrong. A schema's parameter list gives bare names -- tau's
        return_delivered_order_items declares ['order_id', 'item_ids',
        'payment_method_id'] -- while `_leaf_paths_of` marks a LIST-valued argument
        with `[]`, because that is what args_match needs to resolve it. Comparing
        them literally made `item_ids[]` an undeclared field, so every correct call
        passing a list was reported as carrying invented structure: 67 failures on
        CORRECT tau traces, across three of the retail write tools.

        Stripping the marker for the MEMBERSHIP TEST only does not weaken the rule.
        The rule is about which FIELDS the body carries; whether a field holds a list
        or a scalar is a different question, and one the schema's own bracket
        notation answers elsewhere (see the path_values note in CLAUDE.md -- that
        marking is authoritative for DESCENDING into a body, which this is not).
        """
        return path.replace("[]", "")

    allowed_flat = {_strip_lists(d) for d in allowed}
    skip_flat = {_strip_lists(x) for x in skip}

    def _ok(path: str) -> bool:
        if path in allowed or path in skip:
            return True
        flat = _strip_lists(path)
        if flat in allowed_flat or flat in skip_flat:
            return True
        return any(flat == d or flat.startswith(d + ".") for d in allowed_flat)

    def fn(trace, position, metrics=None):
        calls = [c for c in trace.calls if c.name == tool]
        if not calls:
            return {"passed": True, "note": f'vacuous: "{tool}" not called.'}
        for c in calls:
            body = unwrap_json_args(c.args or {})
            extra = sorted({p for p in _leaf_paths_of(body) if not _ok(p)})
            if extra:
                return {"passed": False,
                        "note": (f"{tool}(#{c.position + 1}) carries {extra[:4]}, which "
                                 f"{tool} does not declare -- the record has fields the "
                                 "task never asked for.")}
        return {"passed": True, "note": f"{tool} carries only declared arguments."}

    return Predicate(fn=fn, description=f"{tool} carries only declared arguments",
                     runtime_safe=True)
