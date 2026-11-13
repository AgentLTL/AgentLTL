"""``for_each`` and ``succeeded`` in rulebooks: what a text formula cannot say on its own.

``for_each: case_id`` ties every ``now(...)`` / ``now_with(...)`` atom of a ``G(...)`` formula
to the ``case_id`` of the call being judged. The rule then holds case by case: in
``G(now("decide") -> O(now("review")))`` the review must be about the same case as the
decision, not any earlier case. It is built with the engine's first-order quantifier::

    G(∀c ∈ {case_id of the call here}.
        now_with("decide", case_id=c) -> O(now_with("review", case_id=c)))

A call that does not carry the argument is bound to "no value": a decision without a
``case_id`` is satisfied only by a review without one, never by another case's review.

``succeeded: [doccheck]`` makes ``now("doccheck")`` hold only for a call that succeeded:
one whose recorded exit status (``Enforcer.record_completed(..., status=...)``, or
``"status"`` in a trace entry) is 0 or absent, and whose result is not a mapping with
``ok: false``. A failed check then does not open the gate.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, Optional, Sequence, Union

from .._ast import ForAll, Formula, Globally, Matches, Now, ToolPattern, Var

# Atoms that read a call by name only: a for_each rule cannot tie them to an argument.
_NAME_ONLY = {"Called", "CalledWith", "CalledNTimes", "Before", "After", "AllBefore",
              "BranchCalled", "CalledWithResult", "InstanceBefore", "CalledInOrder",
              "WithinSteps", "CountBefore", "Predicate", "AtPosition", "ForAll", "Exists"}


def failed(call: Any) -> bool:
    """Did this recorded call fail (non-zero exit status, or a result with ``ok: false``)?"""
    if call is None:
        return False
    status = (getattr(call, "raw", None) or {}).get("status")
    if isinstance(status, int) and not isinstance(status, bool) and status != 0:
        return True
    result = getattr(call, "result", None)
    return isinstance(result, dict) and result.get("ok") is False


class StepPattern(ToolPattern):
    """A call to one of *tools* with arguments equal to *args*, that succeeded if asked."""

    def __init__(self, tools: Union[str, Sequence[str]], args: Optional[Dict[str, Any]] = None,
                 succeeded: bool = False):
        super().__init__(tools, args)
        self.succeeded = succeeded

    def match(self, name: str, args: Dict[str, Any], call: Any = None) -> Optional[bool]:
        if not super().match(name, args):
            return False
        return not (self.succeeded and failed(call))

    def bind(self, bindings: Dict[str, Any]) -> "StepPattern":
        bound = super().bind(bindings)
        return StepPattern(bound.tools, bound.args, self.succeeded)

    def describe(self) -> str:
        text = super().describe()
        return text + " (succeeded)" if self.succeeded else text

    def __repr__(self) -> str:
        return f"StepPattern({self.tools!r}, {self.args!r}, succeeded={self.succeeded})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, StepPattern) and super().__eq__(other)
                and self.succeeded == other.succeeded)

    def __hash__(self) -> int:
        return super().__hash__() ^ hash(self.succeeded)


def _values_here(arg: str):
    def domain(trace: Any, metrics: Dict[str, Any], position: int) -> list:
        calls = trace.calls
        if 0 <= position < len(calls):
            return [calls[position].args.get(arg)]
        return []
    domain.__name__ = f"{arg}_of_the_call_here"
    return domain


def _rewrite(f: Formula, arg: Optional[str], ok: set, where: str) -> Formula:
    if isinstance(f, Now):
        args = {arg: Var(arg)} if arg else {}
        return Matches(StepPattern(f.tool, args, f.tool in ok))
    if isinstance(f, Matches):
        p = f.pattern
        if type(p) is not ToolPattern:
            raise ValueError(f"{where}: for_each / succeeded only apply to now() and now_with()")
        args = dict(p.args, **({arg: Var(arg)} if arg else {}))
        return Matches(StepPattern(p.tools, args, any(t in ok for t in p.tools)), f.maybe)
    if type(f).__name__ in _NAME_ONLY:
        raise ValueError(f"{where}: with for_each or succeeded, write the formula with now() "
                         f"and now_with() only (found {type(f).__name__})")
    changes = {fl.name: _rewrite(getattr(f, fl.name), arg, ok, where)
               for fl in dataclasses.fields(f) if isinstance(getattr(f, fl.name), Formula)}
    return dataclasses.replace(f, **changes) if changes else f


def bind(formula: Formula, for_each: Optional[str], succeeded: Iterable[str],
         where: str) -> Formula:
    """Apply a rule's ``for_each`` argument and ``succeeded`` tools to its parsed formula."""
    ok = set(succeeded)
    if for_each is None:
        return _rewrite(formula, None, ok, where)
    if not isinstance(formula, Globally):
        raise ValueError(f"{where}: a for_each rule is a G(...) formula")
    body = _rewrite(formula.operand, for_each, ok, where)
    return Globally(ForAll(for_each, _values_here(for_each), body,
                           description=f"{for_each} of the call here"))
