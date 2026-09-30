# -*- coding: utf-8 -*-
"""
agentltl/translation/expr.py -- open predicates: composable, deterministic, boolean-valued.

===============================================================================
WHY THIS EXISTS

The graph pipeline lets the LLM write NO formula: it supplies steps, needs/produces,
roles and slots, and code assembles every constraint by rule. That split is measured,
not preferred -- attributing run 2's discordant pairs gave code-derived constraints
28W/0L and LLM-authored ones 4W/6L, with all seven false-alarm families authored.

The price is EXPRESSIVENESS. `genv2/emit.py` states only what its rules cover and
`genv2/relative.py`'s vocabulary is a closed set, so an obligation neither can express
becomes NOTHING -- a coverage deficit, which is where blindness comes from
(MedAgentBench: 67% blindness against 1% false-alarm mass, so recall is the frontier).

Two measured obligations no existing predicate can state:

  * `fillFuelTank(fuelAmount=...)`. The task says 40; the tank takes 10.57. The real
    obligation is `amount <= tankCapacity - fuelLevel`, ARITHMETIC OVER A PRIOR
    RESULT. Asserting the literal 40 blocked the correct call 39 times in run 6.
  * `place_order(price=...)`. The price is computed from a fetched quote, so no
    quoted figure is the argument.

An open predicate is an expression over a CLOSED grammar of accessors and total
operators whose root is boolean. The model composes; code parses, validates against
the AgentView, and interprets. Well-formedness is guaranteed by construction -- the
objection to model-authored formulas answered structurally. Whether the obligation is
RIGHT is decided by the gates, not by the author.

THREE PROPERTIES THAT ARE NOT NEGOTIABLE, each earned the hard way

1. THREE-VALUED, WITH UNEVALUABLE PROPAGATING. Absent is not false.
   `qwen_benchmark._trace_to_metrics` drops `tool_result` on some paths, so `result()`
   is often genuinely unreadable; calling that a violation fails every correct run.
   At the root, unevaluable yields {"passed": True, "note": "unevaluable: ..."} --
   the convention every predicate in relative.py already uses.

   Connectives use KLEENE logic, not "any unevaluable wins": `and` is False as soon as
   one arm is False even with an unevaluable sibling, because that arm is definitive.
   Weakening it to unevaluable would silently un-enforce a real violation.

2. EVERY OPERATOR TOTAL. Division by zero, `len` of a scalar, `lt` on mismatched
   types, an unparseable date -> unevaluable, NEVER an exception. These run inside
   `Predicate.fn` during enforcement, where a raise aborts the episode -- and an
   aborted episode scored as incorrect attributes infrastructure failure to whichever
   enforcement arm hit it, which is precisely the comparison being measured.

3. BOUNDED. Depth <= 6, <= 40 nodes. An expression nobody can read is one nobody can
   review, and `matching.why_not` exists because a block the agent cannot act on
   destroys the action rather than correcting it.

WHAT IS DELIBERATELY REUSED, NOT REIMPLEMENTED

`relative.path_values` (dotted/bracketed resolution), `relative.observed_value`,
`relative.band_for`, `relative._as_structured` (a tool result arrives as TEXT; parsing
it is what makes provenance see values at all), `relative._arg_at`, `relative._calls`,
`relative._call_failed` (a cap counts records, not attempts), `relative._parse_dt` /
`_parse_interval` (a date argument is an INTERVAL, never an instant), and
`matching.values_match` (format-tolerant on values, strict on prose -- do not widen).
===============================================================================
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import matching as M
from .. import relative as R

# ── limits ───────────────────────────────────────────────────────────────────
MAX_DEPTH = 6
MAX_NODES = 40

# ── the grammar ──────────────────────────────────────────────────────────────
TERMS = ("arg", "result", "count", "clock", "const", "table", "when")

# op -> (arity or None for variadic>=2, result kind)
_NUM_OPS: Dict[str, Tuple[Optional[int], str]] = {
    "add": (2, "num"), "sub": (2, "num"), "mul": (2, "num"), "div": (2, "num"),
    "abs": (1, "num"), "len": (1, "num"),
    "min": (None, "num"), "max": (None, "num"), "sum": (None, "num"),
}
_BOOL_OPS: Dict[str, Tuple[Optional[int], str]] = {
    "eq": (2, "bool"), "ne": (2, "bool"),
    "lt": (2, "bool"), "le": (2, "bool"), "gt": (2, "bool"), "ge": (2, "bool"),
    "in": (2, "bool"), "matches": (2, "bool"),
    "not": (1, "bool"), "implies": (2, "bool"),
    "and": (None, "bool"), "or": (None, "bool"),
}
OPS: Dict[str, Tuple[Optional[int], str]] = {**_NUM_OPS, **_BOOL_OPS}

_LOWER_BOUND_OPS = ("ge", "gt", "eq")   # a count under one of these is LIVENESS


# ── unevaluable, as a value ──────────────────────────────────────────────────
class Uneval:
    """Not a boolean and not an error: a third outcome that carries its reason.

    Kept as a value rather than an exception so every operator stays total and the
    reason survives to the block message.
    """

    __slots__ = ("why",)

    def __init__(self, why: str) -> None:
        self.why = str(why)

    def __repr__(self) -> str:            # pragma: no cover - debugging aid
        return f"Uneval({self.why!r})"

    def __bool__(self) -> bool:
        raise TypeError(
            "Uneval must never be coerced to bool -- that is how 'absent' silently "
            "becomes 'false'. Test with expr.is_uneval().")


def is_uneval(x: Any) -> bool:
    return isinstance(x, Uneval)


# ── shape validation (structure only; `validate` adds the view checks) ───────
def _kind_of(node: Any) -> str:
    """"bool" | "num" | "any" -- the static result kind, or "" if malformed."""
    if not isinstance(node, dict):
        return ""
    if node.get("acc") in TERMS:
        return "any"
    op = node.get("op")
    if op in OPS:
        return OPS[op][1]
    return ""


def check_shape(node: Any, depth: int = 0) -> Tuple[bool, str, int]:
    """(ok, reason, node_count). Structure, arity and depth only."""
    if depth > MAX_DEPTH:
        return False, f"nested deeper than {MAX_DEPTH}", 0
    if not isinstance(node, dict):
        return False, f"not an object: {node!r}", 0

    acc, op = node.get("acc"), node.get("op")
    if acc is not None and op is not None:
        return False, "a node has both 'acc' and 'op'; it must be one or the other", 0

    if acc is not None:
        if acc not in TERMS:
            return False, f"unknown accessor {acc!r}; known: {', '.join(TERMS)}", 0
        if acc == "arg":
            if not node.get("tool") or not node.get("path"):
                return False, "arg needs 'tool' and 'path'", 0
            which = node.get("which", "last")
            if which not in ("any", "first", "last"):
                return False, f"arg 'which' must be any|first|last, got {which!r}", 0
        elif acc == "result":
            if not node.get("tool") or not node.get("json"):
                return False, "result needs 'tool' and 'json'", 0
        elif acc in ("count", "when"):
            if not node.get("tool"):
                return False, f"{acc} needs 'tool'", 0
        elif acc == "const":
            if "value" not in node:
                return False, "const needs 'value'", 0
            if not str(node.get("span") or "").strip():
                return False, ("const needs a 'span': the verbatim text it came from. "
                               "A value with no span cannot be audited for leakage."), 0
        elif acc == "table":
            if not node.get("column") or not node.get("from"):
                return False, ("table needs 'column' and 'from' (the tool whose "
                               "result the observed figure is read from)"), 0
        return True, "", 1

    if op not in OPS:
        return False, f"unknown operator {op!r}", 0
    args = node.get("args")
    if not isinstance(args, list) or not args:
        return False, f"{op} needs an 'args' list", 0
    arity, _kind = OPS[op]
    if arity is None:
        if len(args) < 2:
            return False, f"{op} needs at least 2 arguments, got {len(args)}", 0
    elif len(args) != arity:
        return False, f"{op} takes {arity} argument(s), got {len(args)}", 0

    # An operator's operands must be usable where they sit. Booleans do not go into
    # arithmetic and numbers do not go into connectives.
    total = 1
    for a in args:
        ok, why, n = check_shape(a, depth + 1)
        if not ok:
            return False, why, 0
        total += n
        k = _kind_of(a)
        if op in _NUM_OPS and k == "bool":
            return False, f"{op} takes values, not a boolean ({a.get('op')})", 0
        if op in ("and", "or", "not", "implies") and k == "num":
            return False, f"{op} takes booleans, not a number ({a.get('op')})", 0
    if total > MAX_NODES:
        return False, f"more than {MAX_NODES} nodes", 0
    return True, "", total


def walk(node: Any):
    """Every dict node, root first. Used by the validator, the leak gate and the
    perturbation gate, so all three agree on what the expression contains."""
    if not isinstance(node, dict):
        return
    yield node
    for a in (node.get("args") or []):
        for x in walk(a):
            yield x


def terms(node: Any, acc: str) -> List[dict]:
    return [n for n in walk(node) if n.get("acc") == acc]


def tools_used(node: Any) -> List[str]:
    seen: List[str] = []
    for n in walk(node):
        t = n.get("tool") or n.get("from")
        if t and t not in seen:
            seen.append(t)
    return seen


def consts(node: Any) -> List[dict]:
    return terms(node, "const")


# ── disposition input: which accessors appear ────────────────────────────────
def accessors(node: Any) -> List[str]:
    out: List[str] = []
    for n in walk(node):
        a = n.get("acc")
        if a and a not in out:
            out.append(a)
    return out


def count_under_lower_bound(node: Any) -> bool:
    """A `count` compared with >=, > or == is LIVENESS: no finite prefix falsifies it,
    because the call might still come. Blocking it would stop an agent for not having
    acted YET, which is exactly why L1 cannot block."""
    for n in walk(node):
        if n.get("op") in _LOWER_BOUND_OPS:
            for a in (n.get("args") or []):
                if any(x.get("acc") == "count" for x in walk(a)):
                    return True
    return False


def reads_args_of(node: Any) -> Optional[str]:
    """The tool whose arguments the expression reads -- the one to guard on.

    Unwrapped, `arg()` on an absent tool doubles as a presence requirement, the same
    trap that makes bare `Before` and `CalledWith` fail a run that never called `b`.
    """
    for n in walk(node):
        if n.get("acc") == "arg":
            return str(n.get("tool"))
    return None


# ═════════════════════════════════════════════════════════════════════════════
# the interpreter
# ═════════════════════════════════════════════════════════════════════════════
class Ctx:
    """Everything an expression may read. Nothing here is oracle-side.

    `trace` is normalised to an AgentLTL `Trace` (`.calls` of `ToolCall`, each with
    `.name`, `.args`, `.result`, `.position`), because that is what a `Predicate.fn`
    is handed at both scoring and enforcement time and what every helper in
    `relative.py` expects. A raw `metrics["tool_calls"]` list is accepted too, so a
    gate or a test can pass the recorded shape directly -- the two shapes diverging
    silently is exactly the POST wire-shape bug.
    """

    __slots__ = ("trace", "position", "clock", "facts")

    def __init__(self, trace, position=None, clock=None, facts=None) -> None:
        self.trace = as_trace(trace)
        self.position = position
        self.clock = clock
        self.facts = facts or {}


def as_trace(trace):
    """An AgentLTL Trace, from a Trace or from a list of recorded call dicts."""
    if trace is None:
        trace = []
    if hasattr(trace, "calls"):
        return trace
    from .._trace import Trace
    return Trace.from_metrics({"tool_calls": list(trace)})


def _effective(ctx: Ctx, tool: str) -> List[Any]:
    """Calls that TOOK EFFECT. A failed attempt creates no record, so counting it
    would report a cap breached by a call the environment never accepted."""
    return [c for c in R._calls(ctx.trace, [tool]) if not R._call_failed(c)]


def _pick(calls: List[Any], which: str) -> List[Any]:
    if not calls:
        return []
    if which == "first":
        return [calls[0]]
    if which == "last":
        return [calls[-1]]
    return calls                      # "any" -> existential over every call


def _num(v: Any) -> Any:
    """A number, or Uneval. Booleans are NOT numbers here: `a is True` must never
    silently compare equal to 1, which is how a flag becomes a quantity."""
    if isinstance(v, bool):
        return Uneval(f"{v!r} is a boolean, not a number")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m or m.group(0) != s:
        return Uneval(f"{v!r} is not a number")
    try:
        return float(m.group(0))
    except Exception:
        return Uneval(f"{v!r} is not a number")


def _eval(node: dict, ctx: Ctx) -> Any:
    acc, op = node.get("acc"), node.get("op")

    # ── terms ───────────────────────────────────────────────────────────────
    if acc == "const":
        return node.get("value")

    if acc == "clock":
        if not ctx.clock:
            return Uneval("the episode clock is not recorded")
        dt = R._parse_dt(ctx.clock)
        return dt if dt is not None else Uneval(f"clock {ctx.clock!r} did not parse")

    if acc == "when":
        # WHEN the deciding figure was recorded. Needed because half of what task5
        # states is a recency guard -- "the last level WITHIN LAST 24 HOURS ... if no
        # level has been recorded in the last 24 hours, don't order anything" -- and
        # a value alone cannot express it. Reading the timestamp through `result`
        # instead would take document order, which these unsorted bundles do not
        # respect; `observed_when` pairs it with the very entry `observed_value`
        # returns, so the two can never describe different rows.
        tool = str(node["tool"])
        ex = (ctx.facts or {}).get("extract") or {}
        if not ex.get("items"):
            return Uneval("no recency-aware extraction is declared for this family")
        best = None
        for c in _effective(ctx, tool):
            w = R.observed_when(getattr(c, "result", None), ex)
            if w is not None and (best is None or w > best):
                best = w
        if best is None:
            return Uneval(f'no dated "{tool}" reading to take a timestamp from')
        return best

    if acc == "count":
        return float(len(_effective(ctx, str(node["tool"]))))

    if acc == "arg":
        tool, path = str(node["tool"]), str(node["path"])
        calls = _pick(_effective(ctx, tool), str(node.get("which", "last")))
        if not calls:
            # ABSENT, not wrong. The guard wrapper makes this vacuous anyway; being
            # explicit here keeps the note honest when it is evaluated bare.
            return Uneval(f'"{tool}" was not called')
        vals: List[Any] = []
        for c in calls:
            vals.extend(R._arg_at(c, path))
        if not vals:
            return Uneval(f'"{tool}" carried no {path}')
        return vals[0] if len(vals) == 1 else _Multi(vals)

    if acc == "result":
        tool, jp = str(node["tool"]), str(node["json"])
        got: List[Any] = []
        for c in _effective(ctx, tool):
            res = getattr(c, "result", None)
            if res is None:
                continue
            doc = R._as_structured(res)
            vals = R.path_values(doc, jp)
            if not vals:
                # FALL BACK TO A RECURSIVE SEARCH FOR THE LEAF KEY, because the model
                # cannot know the response SHAPE. The AgentView carries tool
                # parameters, never the structure of what a tool returns, so a FHIR
                # bundle nests the figure at `entry[].resource.effectiveDateTime`
                # while the only honest thing the model can name is
                # `effectiveDateTime`. Requiring an exact path made `result()` --
                # the accessor that gives open predicates their power -- permanently
                # unevaluable on the first real expression it produced.
                #
                # This is `$..key` semantics, not the too-weak "appears anywhere"
                # matching that CLAUDE.md warns about for ArgFromOutput: that one
                # asserts IDENTITY and even falls back to a substring, where this
                # reads a value to compare.
                vals = _find_leaf(doc, str(jp).split(".")[-1].replace("[]", ""))
            got.extend(vals)

            # RECENCY IS A TIMESTAMP QUESTION, NOT A POSITION ONE. Both the path walk
            # and the leaf search return document order, and these FHIR bundles are
            # NOT sorted -- mab_task5_11's last entry is a stale 1.8 while its most
            # recent reading is 2.1. Taking got[-1] therefore read a low level for a
            # normal patient, so the trigger fired and a CORRECT abstention was
            # penalised: 21 of the 27 false alarms this tier was charged with.
            #
            # Where the family declares a recency-aware extraction, use it: it pairs
            # each value with its OWN timestamp, which no flat list of values can do.
            # Only for the leaf the extraction is about, so an expression reading some
            # other field is unaffected.
            ex = (ctx.facts or {}).get("extract") or {}
            leaf = str(jp).split(".")[-1].replace("[]", "")
            if ex.get("items") and leaf == str(ex.get("value", "")).split(".")[-1]:
                mr = R.observed_value(doc, ex)
                if mr is not None:
                    got = [mr]

        if not got:
            # The producer may have run without its result being recorded -- that is
            # unevaluable, not violated. It is also the normal case on the bfcl/tau
            # scoring path, where _trace_to_metrics drops tool_result entirely.
            return Uneval(f'no recorded "{tool}" output to read {jp} from')
        return got[-1]

    if acc == "table":
        bands = ((ctx.facts.get("dose_table") or {}).get("bands")) or []
        extract = ctx.facts.get("extract")
        if not bands:
            return Uneval("no policy table is recorded for this task")
        obs = None
        for c in _effective(ctx, str(node["from"])):
            v = R.observed_value(getattr(c, "result", None), extract)
            if v is not None:
                obs = v
        if obs is None:
            return Uneval(f'no observed figure in "{node["from"]}" output')
        band = R.band_for(obs, bands)
        if not band:
            return Uneval(f"observed {obs} falls in no stated band")
        col = str(node["column"])
        if col not in band:
            return Uneval(f"the table has no {col!r} column")
        return band[col]

    # ── operators ───────────────────────────────────────────────────────────
    args = node.get("args") or []

    if op in ("and", "or"):
        vals = [_eval(a, ctx) for a in args]
        # KLEENE. A definitive arm wins over an unevaluable sibling; weakening this
        # to "any unevaluable -> unevaluable" would un-enforce a real violation.
        if op == "and":
            if any(v is False for v in vals):
                return False
            u = next((v for v in vals if is_uneval(v)), None)
            return u if u is not None else all(v is True for v in vals)
        if any(v is True for v in vals):
            return True
        u = next((v for v in vals if is_uneval(v)), None)
        return u if u is not None else False

    if op == "not":
        v = _eval(args[0], ctx)
        return v if is_uneval(v) else (not v)

    if op == "implies":
        a = _eval(args[0], ctx)
        if a is False:
            return True                       # vacuously satisfied
        b = _eval(args[1], ctx)
        if b is True:
            return True
        if is_uneval(a):
            return a
        if is_uneval(b):
            return b
        return False

    if op in _NUM_OPS:
        vals = [_eval(a, ctx) for a in args]
        u = next((v for v in vals if is_uneval(v)), None)
        if u is not None:
            return u
        if op == "len":
            v = vals[0]
            if isinstance(v, _Multi):
                return float(len(v.values))
            if isinstance(v, (list, tuple, dict, str)):
                return float(len(v))
            return Uneval(f"len of {type(v).__name__} is not defined")
        if op == "sub":
            # "HOW LONG AGO" is a real obligation and the reason task10 exists at all
            # (a stale lab must be re-ordered). Two instants subtract to a DURATION
            # IN SECONDS, so `clock - effectiveDateTime > 31536000` means what it
            # reads as. Only `sub` gets this: a datetime plus a number is not a
            # meaningful quantity, and silently coercing one would be the
            # boolean-is-not-a-number mistake in another costume.
            la, rb = R._parse_dt(_first(vals[0])), R._parse_dt(_first(vals[1]))
            if la is not None and rb is not None:
                return float((la - rb).total_seconds())
        nums = [_num(_first(v)) for v in vals]
        u = next((n for n in nums if is_uneval(n)), None)
        if u is not None:
            return u
        try:
            if op == "add":
                return nums[0] + nums[1]
            if op == "sub":
                return nums[0] - nums[1]
            # (datetime subtraction is handled before the numeric coercion above)
            if op == "mul":
                return nums[0] * nums[1]
            if op == "div":
                if nums[1] == 0:
                    return Uneval("division by zero")
                return nums[0] / nums[1]
            if op == "abs":
                return abs(nums[0])
            if op == "min":
                return min(nums)
            if op == "max":
                return max(nums)
            if op == "sum":
                return float(sum(nums))
        except Exception as exc:            # totality backstop
            return Uneval(f"{op} failed: {type(exc).__name__}")

    # comparisons
    left, right = _eval(args[0], ctx), _eval(args[1], ctx)
    for v in (left, right):
        if is_uneval(v):
            return v

    if op in ("eq", "ne"):
        got = _any_match(left, right)
        return got if op == "eq" else (not got)

    if op == "in":
        hay = right.values if isinstance(right, _Multi) else right
        if not isinstance(hay, (list, tuple, set)):
            return Uneval(f"'in' needs a list on the right, got "
                          f"{type(hay).__name__}")
        return any(_any_match(left, h) for h in hay)

    if op == "matches":
        pat = _first(right)
        if not isinstance(pat, str):
            return Uneval("'matches' needs a string pattern")
        try:
            rx = re.compile(pat)
        except re.error as exc:
            return Uneval(f"bad pattern {pat!r}: {exc}")
        s = _first(left)
        return bool(rx.fullmatch(str(s)))

    # ordered comparison: numbers first, then instants
    ln, rn = _num(_first(left)), _num(_first(right))
    if not (is_uneval(ln) or is_uneval(rn)):
        return _order(op, ln, rn)
    la, ra = R._parse_dt(_first(left)), R._parse_dt(_first(right))
    if la is not None and ra is not None:
        return _order(op, la, ra)
    return Uneval(f"cannot order {_first(left)!r} against {_first(right)!r}")


def _find_leaf(node: Any, key: str, out: Optional[List[Any]] = None) -> List[Any]:
    """Every value stored under `key` anywhere in a nested document, in order."""
    if out is None:
        out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and not isinstance(v, (dict, list)):
                out.append(v)
            else:
                _find_leaf(v, key, out)
    elif isinstance(node, list):
        for v in node:
            _find_leaf(v, key, out)
    return out


def _order(op: str, a: Any, b: Any) -> bool:
    if op == "lt":
        return a < b
    if op == "le":
        return a <= b
    if op == "gt":
        return a > b
    return a >= b


class _Multi:
    """Several values from `which="any"`. Existential: a comparison holds if it holds
    for one of them, which mirrors `called_with_like` being an ANY-call matcher."""

    __slots__ = ("values",)

    def __init__(self, values: Sequence[Any]) -> None:
        self.values = list(values)

    def __repr__(self) -> str:            # pragma: no cover
        return f"_Multi({self.values!r})"


def _first(v: Any) -> Any:
    return v.values[0] if isinstance(v, _Multi) and v.values else v


def _any_match(left: Any, right: Any) -> bool:
    ls = left.values if isinstance(left, _Multi) else [left]
    rs = right.values if isinstance(right, _Multi) else [right]
    # values_match is format-tolerant ("20" ~ 20, date granularity) and strict on
    # prose, on purpose. Do not widen it: prose-instead-of-value is a real error.
    return any(M.values_match(r, l) or M.values_match(l, r) for l in ls for r in rs)


# ═════════════════════════════════════════════════════════════════════════════
# validation -- parse tolerantly, validate STRICTLY, and always say why
# ═════════════════════════════════════════════════════════════════════════════
def _leaf_ok(t, path: str) -> bool:
    """Is `path` a real argument of this tool?

    mab's extractor records LEAF PATHS in `parameters` (exposing only a POST's seven
    top-level keys hid every bindable slot), while the flat families record plain
    names -- so accept either the full path or its head.
    """
    declared = set(t.parameters or ())
    if path in declared:
        return True
    head = str(path).split(".", 1)[0].replace("[]", "")
    return head in declared or any(str(d).split(".", 1)[0] == head for d in declared)


def _pairs_arg_vs_const(node: Any):
    """Every comparison with an `arg` on one side and a `const` on the other."""
    for n in walk(node):
        if n.get("op") not in ("eq", "ne", "in", "matches", "lt", "le", "gt", "ge"):
            continue
        a, b = (n.get("args") or [None, None])[:2]
        for x, y in ((a, b), (b, a)):
            if isinstance(x, dict) and isinstance(y, dict) \
                    and x.get("acc") == "arg" and y.get("acc") == "const":
                yield n, x, y


def validate(ast: Any, view, produced_tools: Optional[Sequence[str]] = None
             ) -> Tuple[bool, str]:
    """(ok, reason). The reason is the point: stating it is what made the G2
    interview converge, and holding it back throws away the most useful thing we
    know."""
    ok, why, _n = check_shape(ast)
    if not ok:
        return False, why
    if _kind_of(ast) != "bool":
        return False, ("the whole expression must be a boolean; its outermost node is "
                       f"{ast.get('op') or ast.get('acc')!r}, which yields a value. "
                       "Wrap it in a comparison.")

    tmap = view.tool_map()
    for n in walk(ast):
        tool = n.get("tool") or n.get("from")
        if tool and tool not in tmap:
            return False, (f"there is no tool called {tool!r}. The tools are: "
                           f"{', '.join(sorted(tmap))}")

    for n in terms(ast, "arg"):
        t = tmap[str(n["tool"])]
        if not _leaf_ok(t, str(n["path"])):
            return False, (f"{n['tool']}.{n['path']} is not a declared parameter; the "
                           f"declared ones are: {', '.join(t.parameters or ['(none)'])}")

    # A literal must trace to something the AGENT can see. The leak gate audits this
    # too, but rejecting here means it never reaches a spec at all.
    hay = f"{view.task_text()}\n{view.policy or ''}"
    for n in consts(ast):
        span = str(n.get("span") or "")
        if span and span not in hay:
            return False, (f"the quote {span!r} does not appear in the task text or "
                           "the policy. Quote it exactly, or use a different term.")

    # The class removed at real cost: a prose spelling asserted against a canonical
    # wire value. Open predicates must not reintroduce it.
    from .. import paramspec as PS
    for _cmp, argn, constn in _pairs_arg_vs_const(ast):
        t = tmap[str(argn["tool"])]
        pd = t.param_schema(str(argn["path"]))
        if not pd:
            continue
        head = str(argn["path"]).split(".", 1)[0].replace("[]", "")
        cv, _ = PS.canonicalise(head, pd, constn.get("value"))
        good, reason = PS.assertable(head, pd, cv)
        if not good:
            return False, (f"{constn.get('value')!r} cannot be compared against "
                           f"{argn['tool']}.{argn['path']}: {reason}")
        constn["value"] = cv               # canonicalise in place, as the emitter does

    if produced_tools is not None:
        prod = set(produced_tools)
        for n in terms(ast, "result"):
            if str(n["tool"]) not in prod:
                return False, (f"nothing in this task reads output from "
                               f"{n['tool']!r}, so that value can never be known. "
                               f"Tools whose output is available: "
                               f"{', '.join(sorted(prod)) or '(none)'}")
    return True, ""


# ═════════════════════════════════════════════════════════════════════════════
# explanation -- a block the agent cannot act on destroys the action
# ═════════════════════════════════════════════════════════════════════════════
def explain(ast: dict, ctx: Ctx) -> str:
    """Name the sub-term that came out False, with both operand values.

    task8 was blocked five times on `note.text` and ended with NO write, because the
    message dumped the whole payload beside the whole expectation and never said the
    problem was `note` sent as an ARRAY. `matching.why_not` fixed that for slots; an
    expression needs the same courtesy.
    """
    worst = None
    for n in walk(ast):
        if n.get("op") not in ("eq", "ne", "lt", "le", "gt", "ge", "in", "matches"):
            continue
        v = _eval(n, ctx)
        if v is False:
            worst = n
            break
    if worst is None:
        return _render(ast)
    a, b = (worst.get("args") or [None, None])[:2]
    parts = []
    for side in (a, b):
        # A const renders as its own value, so reporting "1 is 1" is noise. Say what
        # the agent's side actually was and leave the expectation to _render.
        if isinstance(side, dict) and side.get("acc") == "const":
            continue
        parts.append(f"{_render(side)} is {_show(_eval(side, ctx))}")
    detail = "; ".join(parts) or "both sides are fixed values"
    return f"{_render(worst)} does not hold: {detail}"


_SYMBOL = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
           "add": "+", "sub": "-", "mul": "*", "div": "/"}


def _render(node: Any) -> str:
    if not isinstance(node, dict):
        return repr(node)
    acc = node.get("acc")
    if acc == "arg":
        return f"{node['tool']}.{node['path']}"
    if acc == "result":
        return f"{node['tool']}'s {node['json']}"
    if acc == "count":
        return f"the number of {node['tool']} calls"
    if acc == "const":
        return repr(node.get("value"))
    if acc == "clock":
        return "the current time"
    if acc == "when":
        return f"when {node['tool']} last recorded a reading"
    if acc == "table":
        return f"the table's {node['column']} for {node['from']}"
    op, args = node.get("op"), (node.get("args") or [])
    if op in _SYMBOL and len(args) == 2:
        return f"({_render(args[0])} {_SYMBOL[op]} {_render(args[1])})"
    return f"{op}({', '.join(_render(a) for a in args)})"


def _show(v: Any) -> str:
    if is_uneval(v):
        return f"unknown ({v.why})"
    if isinstance(v, _Multi):
        return ", ".join(repr(x) for x in v.values)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return repr(v)


# ═════════════════════════════════════════════════════════════════════════════
# the compiler entry point
# ═════════════════════════════════════════════════════════════════════════════
def evaluate(ast: dict, trace, position=None, clock=None, facts=None) -> dict:
    """{"passed": bool, "note": str} -- the contract every predicate here uses.

    Unevaluable PASSES, with the reason in the note. That is not leniency: absent is
    a different error from wrong, and failing here would double-count it.
    """
    ctx = Ctx(trace, position, clock, facts)
    try:
        v = _eval(ast, ctx)
    except Exception as exc:                # the totality backstop of last resort
        return {"passed": True,
                "note": f"unevaluable: the expression raised {type(exc).__name__}"}
    if is_uneval(v):
        return {"passed": True, "note": f"unevaluable: {v.why}"}
    if v is True:
        return {"passed": True, "note": f"{_render(ast)} holds."}
    return {"passed": False, "note": explain(ast, ctx)}


def build(ast: dict, A: dict, clock=None, facts=None):
    """An AgentLTL Predicate over an open expression.

    `runtime_safe` is True because a violation is decidable when the call is made --
    but whether it may actually BLOCK is decided by `taxonomy.disposition`, from the
    accessors the expression uses, not here.
    """
    Predicate = A["Predicate"]

    def fn(trace, position, metrics=None):
        return evaluate(ast, trace, position, clock, facts)

    return Predicate(fn=fn, description=_render(ast), runtime_safe=True)
