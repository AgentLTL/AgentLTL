"""Argument conditions for ``never:`` rules: what ``called_with`` (plain equality) cannot say.

A rule ``never: {tool: git_push, where: {force: true}}`` refuses a call to ``git_push`` whose
arguments satisfy every condition in ``where``. A condition is one of:

* a value: equal to the argument, with strings compared without case and ``true`` /
  ``false`` also matching ``"true"``, ``"1"``, ``"yes"`` / ``"false"``, ``"0"``, ``"no"``;
* a list: the argument equals one of its values (same comparison);
* ``{in: [...]}`` / ``{not_in: [...]}``;
* ``{matches: REGEX}`` / ``{not_matches: REGEX}``: the regex found in the argument as text,
  without case;
* ``{gt: N}``, ``{ge: N}``, ``{lt: N}``, ``{le: N}``: the argument as a number; one that is
  not a number counts as a match, since nothing can then be said about it;
* ``{present: true}`` / ``{present: false}``.

An argument that is missing satisfies only ``not_in``, ``not_matches`` and
``present: false``: a rule that allows a set of values (``region`` in the EU) refuses a call
that does not say which one.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .._ast import CallPattern

_TRUE = {"true", "1", "yes", "y", "on"}
_FALSE = {"false", "0", "no", "n", "off", ""}
_OPERATORS = {"in", "not_in", "matches", "not_matches", "gt", "ge", "lt", "le", "present"}
_MISSING = object()


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual == expected
        if isinstance(actual, (int, float)):
            return bool(actual) == expected
        if isinstance(actual, str):
            word = actual.strip().lower()
            return word in (_TRUE if expected else _FALSE)
        return actual is None and not expected
    if isinstance(expected, str):
        return isinstance(actual, str) and actual.strip().lower() == expected.strip().lower()
    if isinstance(expected, (int, float)):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _number(actual: Any) -> Optional[float]:
    if isinstance(actual, bool):
        return None
    try:
        return float(actual)
    except (TypeError, ValueError):
        return None


def _text(actual: Any) -> str:
    return "" if actual is _MISSING or actual is None else str(actual)


class Condition:
    """One test on one argument, built by :func:`condition` from YAML."""

    def __init__(self, arg: str, operator: str, operand: Any):
        self.arg, self.operator, self.operand = arg, operator, operand
        if operator in ("matches", "not_matches"):
            try:
                self._regex = re.compile(str(operand), re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"{arg}: bad regular expression {operand!r}: {exc}") from exc

    def holds(self, args: Dict[str, Any]) -> bool:
        actual = args.get(self.arg, _MISSING)
        present = actual is not _MISSING and actual is not None
        op, operand = self.operator, self.operand
        if op == "present":
            return present == bool(operand)
        if op == "not_in":
            return not present or not any(_same(actual, v) for v in operand)
        if op == "not_matches":
            return not self._regex.search(_text(actual))
        if not present:
            return False
        if op == "eq":
            return _same(actual, operand)
        if op == "in":
            return any(_same(actual, v) for v in operand)
        if op == "matches":
            return bool(self._regex.search(_text(actual)))
        number = _number(actual)
        if number is None:
            return True
        limit = float(operand)
        return {"gt": number > limit, "ge": number >= limit,
                "lt": number < limit, "le": number <= limit}[op]

    def describe(self) -> str:
        return f"{self.arg} {self.operator} {self.operand!r}"


def condition(arg: str, spec: Any, where: str) -> Condition:
    """The :class:`Condition` for ``arg: spec`` in a ``where`` mapping."""
    if isinstance(spec, dict):
        if len(spec) != 1 or next(iter(spec)) not in _OPERATORS:
            raise ValueError(f"{where}: condition on {arg!r} must have exactly one of "
                             f"{sorted(_OPERATORS)}, not {sorted(spec)}")
        operator, operand = next(iter(spec.items()))
    elif isinstance(spec, list):
        operator, operand = "in", spec
    else:
        operator, operand = "eq", spec
    if operator in ("in", "not_in") and not isinstance(operand, list):
        raise ValueError(f"{where}: {arg!r}: {operator} takes a list")
    if operator in ("gt", "ge", "lt", "le") and _number(operand) is None:
        raise ValueError(f"{where}: {arg!r}: {operator} takes a number, not {operand!r}")
    return Condition(arg, operator, operand)


class ArgumentPattern(CallPattern):
    """A call to one of *tools* whose arguments satisfy every one of *conditions*."""

    def __init__(self, tools: Union[str, Sequence[str]], conditions: List[Condition]):
        self.tools = (tools,) if isinstance(tools, str) else tuple(tools)
        self.conditions = list(conditions)

    def match(self, name: str, args: Dict[str, Any]) -> Optional[bool]:
        return name in self.tools and all(c.holds(args) for c in self.conditions)

    def describe(self) -> str:
        where = " and ".join(c.describe() for c in self.conditions)
        return " or ".join(self.tools) + (f" where {where}" if where else "")

    def __repr__(self) -> str:
        return f"ArgumentPattern({self.tools!r}, {[c.describe() for c in self.conditions]!r})"


def pattern(spec: Any, where: str) -> Tuple[ArgumentPattern, str]:
    """Build the pattern of a ``never:`` rule; returns it and a short text of what it refuses."""
    if not isinstance(spec, dict) or "tool" not in spec:
        raise ValueError(f"{where}: 'never' is a mapping with a 'tool' (a name or a list) "
                         "and an optional 'where'")
    unknown = sorted(set(spec) - {"tool", "where"})
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown} under 'never'; allowed: "
                         "['tool', 'where']")
    tools = spec["tool"]
    if isinstance(tools, str):
        tools = [tools]
    if not isinstance(tools, list) or not tools or not all(isinstance(t, str) for t in tools):
        raise ValueError(f"{where}: 'tool' is a name or a list of names")
    conditions_spec = spec.get("where") or {}
    if not isinstance(conditions_spec, dict):
        raise ValueError(f"{where}: 'where' is a mapping of argument name to condition")
    conditions = [condition(str(k), v, where) for k, v in conditions_spec.items()]
    built = ArgumentPattern(tools, conditions)
    return built, built.describe()
