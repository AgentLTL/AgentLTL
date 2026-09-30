# -*- coding: utf-8 -*-
"""
agentltl/translation/leaks.py -- the leak gate's THIRD literal location.

`assemble.find_leaks` audits every literal in a spec against what the agent can see.
It reaches them through `assemble._literals`, which walks two places:

    args["expected_args"]          -- the argument bindings
    _rel_literals(args)            -- the relative predicates' scalars

That second one was added because it was MISSING, and the consequence is recorded:
899 literals -- dose bands, window bounds, resolved instants, arithmetic goals -- sat
in the two BLOCKING layers completely unaudited, because the walker only looked at
`expected_args`. A leak gate that cannot see a literal reports zero and looks like a
pass.

An open predicate's `const` terms are a third such location, and they arrive with
blocking enabled from the first run. So this lands WITH the grammar rather than after
it, and `test_translation_leaks.py` pins that `find_leaks` actually reaches them --
because the failure mode here is not an error, it is a clean report.

Note the division of labour. `expr.validate` rejects a const whose SPAN is not in the
task text or the policy at generation time; this audits the VALUE at spec time, the
same question `find_leaks` asks of every other literal. Two independent checks on the
same fact is deliberate: the first is a prompt-loop guard the interview can be talked
out of, the second reads the finished spec.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from . import expr as X

# Seconds per unit. A duration written as seconds is a UNIT CONVERSION of a stated
# quantity, the same class `find_leaks` already allows for "a 1.5 hour event" ->
# duration=90. It is checked HERE rather than by widening genv2's numeric factors,
# because those multiply every stated number and adding 31536000 to them would make
# any large round number groundable from a task that merely says "1" -- a real
# precision loss in the leak gate, across all four families.
#
# An expression const can be checked precisely because it carries its OWN span: the
# conversion is allowed only when that span names the unit.
_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400,
            "week": 604800, "month": 2592000, "year": 31536000}
_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(second|minute|hour|day|week|month|year)s?\b", re.I)


def _is_duration_conversion(value: Any, span: str) -> bool:
    """True when `value` is `span`'s stated duration expressed in seconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    for n, unit in _UNIT_RE.findall(str(span or "")):
        try:
            want = float(n) * _SECONDS[unit.lower()]
        except (ValueError, KeyError):
            continue
        if abs(float(value) - want) < 1e-6:
            return True
    return False


def _slot_named_by(node: dict, ast: dict) -> str:
    """The argument this const is compared against, for a readable audit line.

    `_literals` keys its tuples by the argument name, so an expression literal should
    too: `expr:fillFuelTank.fuelAmount` says far more in an audit than `expr`.
    """
    for cmp_node in X.walk(ast):
        args = cmp_node.get("args") or []
        if node not in args:
            continue
        for other in args:
            if isinstance(other, dict) and other.get("acc") == "arg":
                return f"expr:{other.get('tool')}.{other.get('path')}"
            if isinstance(other, dict) and other.get("acc") == "result":
                return f"expr:{other.get('tool')}->{other.get('json')}"
    return "expr"


def expr_literals(args: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Every scalar an open predicate states, as (key, value) pairs.

    Same contract as `_rel_literals`: dicts and lists are skipped (they are not
    scalars to ground), and the key is only for the audit line.
    """
    ast = args.get("ast")
    if not isinstance(ast, dict):
        return []
    out: List[Tuple[str, Any]] = []
    for node in X.consts(ast):
        v = node.get("value")
        if isinstance(v, (dict, list)):
            continue
        if _is_duration_conversion(v, node.get("span") or ""):
            continue          # "1 year" -> 31536000 is a derivation, not a leak
        out.append((_slot_named_by(node, ast), v))
    return out
