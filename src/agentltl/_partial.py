"""
agentltl/_partial.py – five-valued evaluation on a trace that is still growing.

At run time a formula is checked on the calls made so far plus the call about to be
made. Two-valued logic can't say what that check needs to know: whether a failure is
final (no later call can repair it), or only an obligation that hasn't been met *yet*.
On a partial trace every formula therefore takes one of five values, ordered

    FALSE < PFALSE < PENDING < PTRUE < TRUE

======== ==========================================================================
FALSE    violated, and no extension of the trace can repair it
PFALSE   an obligation not met yet (``called("x")`` before x runs): it can be repaired
PENDING  not triggered, or not decided yet (``F``, the successor of the last call)
PTRUE    satisfied so far, and a later call could still break it (``at most 1``)
TRUE     satisfied, and nothing appended can change that
======== ==========================================================================

A call is refused when the value is below PENDING. ``Not`` mirrors the chain
(4 - v), ``And``/``G``/``ForAll`` take the minimum and ``Or``/``F``/``Exists`` the
maximum, so negation and implication need no special cases. On a complete trace
(``partial_trace=False``) the ordinary two-valued semantics in ``_evaluator`` apply.

Every result also carries *witnesses*: the failing instances, each as a path through
the formula's conjunctive structure (a position of ``G``, a side of ``And``, an entity
of ``ForAll``) with its value. The enforcer uses them to refuse a call only for what it
newly breaks: a witness that was already FALSE before the call is not its fault.

``runtime_safety`` classifies formulas over the same algebra (the set of values a
formula can take), so the two agree by construction.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ._ast import (
    After, AllBefore, And, AtPosition, Before, BranchCalled, Called, CalledInOrder,
    CalledNTimes, CalledWith, CalledWithResult, CountBefore, Eventually, Exists, ForAll,
    Formula, Globally, Historically, Implies, InstanceBefore, Matches, Next, Not, Now, Once, Or,
    Predicate, Previous, Release, Since, Until, WeakUntil, WithinSteps, substitute,
)
from ._trace import Trace

FALSE, PFALSE, PENDING, PTRUE, TRUE = 0, 1, 2, 3, 4
NAMES = {FALSE: "violated", PFALSE: "not met yet", PENDING: "pending",
         PTRUE: "holds so far", TRUE: "holds"}

Witness = Tuple[Tuple[Any, ...], int]


class Value:
    """A partial-trace verdict: its value, witnesses and an explanation."""

    __slots__ = ("value", "witnesses", "key", "_detail")

    def __init__(self, value: int, detail: Any, key: Any = None,
                 witnesses: Optional[List[Witness]] = None) -> None:
        self.value = value
        self.key = key
        self._detail = detail
        if witnesses is None:
            witnesses = [((key,), value)] if value < PENDING else []
        self.witnesses = witnesses

    @property
    def detail(self) -> str:
        if callable(self._detail):
            self._detail = self._detail()
        return self._detail

    @property
    def passed(self) -> bool:
        return self.value >= PENDING


def _cmp(actual: int, op: str, n: int) -> bool:
    return {"==": actual == n, ">=": actual >= n, "<=": actual <= n,
            ">": actual > n, "<": actual < n}.get(op, False)


def _known_result(call) -> bool:
    raw = call.raw or {}
    return "result" in raw or "tool_result" in raw


_LOCAL = (Now, Next, Globally, Eventually, Until, WeakUntil, Release, AtPosition, Predicate,
          Matches, Previous, Once, Historically, Since, CountBefore)
_PAST = (Previous, Once, Historically, Since, CountBefore)


def is_past_only(f: Formula) -> bool:
    """Is the formula's value at a position settled by the calls up to that position?

    True for ``now``/``matches`` atoms and past-time operators over them, with boolean
    connectives and quantifiers (whose domain is read at the position). Such a formula
    under ``G`` can only fail at the newest call, so an enforcer judges it there alone.
    """
    if isinstance(f, (Now, Matches)):
        return True
    if isinstance(f, (Not, Previous, Once, Historically)):
        return is_past_only(f.operand)
    if isinstance(f, CountBefore):
        return is_past_only(f.operand)
    if isinstance(f, (And, Or, Implies, Since)):
        return is_past_only(f.left) and is_past_only(f.right)
    if isinstance(f, (ForAll, Exists)):
        return is_past_only(f.body)
    return False


def call_domain(fn: Callable, trace: Trace, metrics: Optional[Dict], position: int) -> Any:
    """A quantifier's domain: ``(trace, metrics)``, or ``(trace, metrics, position)`` for a
    domain read at the position being judged (the values of the call there)."""
    arity = getattr(fn, "__agentltl_arity__", None)
    if arity is None:
        try:
            params = inspect.signature(fn).parameters.values()
            positional = [p for p in params
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            arity = 3 if any(p.kind == p.VAR_POSITIONAL for p in params) or len(positional) >= 3 else 2
        except (TypeError, ValueError):
            arity = 2
        try:
            fn.__agentltl_arity__ = arity
        except (AttributeError, TypeError):
            pass
    return fn(trace, metrics or {}, position) if arity == 3 else fn(trace, metrics or {})


def is_local(f: Formula) -> bool:
    """Does the formula's value depend on the position it is evaluated at?

    Atoms other than ``Now`` are facts about the whole trace. ``G(called("x"))`` is
    therefore the same fact at every position, and must count as one instance, not
    one per position: otherwise each new call would be a "new" failure of it.
    Predicates count as local: they receive the position.
    """
    if isinstance(f, _LOCAL):
        return True
    if isinstance(f, Not):
        return is_local(f.operand)
    if isinstance(f, (And, Or, Implies)):
        return is_local(f.left) or is_local(f.right)
    if isinstance(f, (ForAll, Exists)):
        return is_local(f.body)
    return False


def call_predicate(fn: Callable, trace: Trace, position: int, metrics: Optional[Dict]) -> Any:
    """Call a predicate with (trace, position, metrics) or (trace, position)."""
    arity = getattr(fn, "__agentltl_arity__", None)
    if arity is None:
        try:
            params = inspect.signature(fn).parameters.values()
            positional = [p for p in params
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
            arity = 3 if varargs or len(positional) >= 3 else 2
        except (TypeError, ValueError):
            arity = 3
        try:
            fn.__agentltl_arity__ = arity
        except (AttributeError, TypeError):
            pass
    if arity == 3:
        return fn(trace, position, metrics)
    return fn(trace, position)


class PartialEvaluator:
    """Evaluates a formula on a growing trace, in the five-valued semantics above."""

    def __init__(self) -> None:
        self._depth = 0
        self._memo: Dict[Tuple[int, int], Value] = {}

    def value(self, f: Formula, trace: Trace, pos: int = 0,
              metrics: Optional[Dict[str, Any]] = None) -> Value:
        method = getattr(self, "_" + type(f).__name__, None)
        if method is None:
            raise TypeError(f"Unknown formula type: {type(f).__name__}")
        if self._depth == 0:
            self._memo = {}
        self._depth += 1
        try:
            if isinstance(f, _PAST):
                # past-time operators look back over positions their callers revisit:
                # remembered for the rest of this evaluation, so each costs O(n) once
                key = (id(f), pos)
                hit = self._memo.get(key)
                if hit is None:
                    hit = self._memo[key] = method(f, trace, pos, metrics)
                return hit
            return method(f, trace, pos, metrics)
        finally:
            self._depth -= 1
            if self._depth == 0:
                self._memo = {}

    # ── Atoms ────────────────────────────────────────────────────────────────

    def _Called(self, f: Called, t: Trace, pos, m) -> Value:
        n = t.count(f.tool)
        if n:
            return Value(TRUE, lambda: f'"{f.tool}" was called (call #{t.first_index(f.tool) + 1}).',
                         ("n", n))
        return Value(PFALSE, f'"{f.tool}" has not been called yet.', ("n", 0))

    def _Now(self, f: Now, t: Trace, pos, m) -> Value:
        call = t.at(pos)
        if call is None:
            return Value(PENDING, f"No call at position {pos} yet.")
        if call.name == f.tool:
            return Value(TRUE, lambda: f'Call #{pos + 1} is "{f.tool}".')
        return Value(FALSE, lambda: f'Call #{pos + 1} is "{call.name}", not "{f.tool}".')

    def _CalledWith(self, f: CalledWith, t: Trace, pos, m) -> Value:
        matches = t.calls_with(f.tool, f.expected_args)
        if matches:
            return Value(TRUE, lambda: f'"{f.tool}" called with matching args at call '
                                       f'#{matches[0].position + 1}.', ("n", len(matches)))
        if t.contains(f.tool):
            return Value(PFALSE, lambda: f'"{f.tool}" was called but not with the expected '
                                         f'arguments {f.expected_args}.', ("n", 0))
        return Value(PFALSE, f'"{f.tool}" has not been called yet.', ("n", 0))

    def _CalledWithResult(self, f: CalledWithResult, t: Trace, pos, m) -> Value:
        from ._evaluator import LTLEvaluator
        candidates = (t.calls_with(f.tool, f.expected_args) if f.expected_args
                      else [c for c in t.calls if c.name == f.tool])
        known = [c for c in candidates if _known_result(c)]
        for c in known:
            if LTLEvaluator._match_result(c.result, f.expected_result):
                return Value(TRUE, f'"{f.tool}" call #{c.position + 1} returned matching result.',
                             ("n", len(known)))
        if len(known) < len(candidates):
            return Value(PENDING, f'"{f.tool}" has been called; its result is not known yet.')
        return Value(PFALSE, f'No "{f.tool}" call has returned {f.expected_result!r} yet.',
                     ("n", len(known)))

    def _CalledNTimes(self, f: CalledNTimes, t: Trace, pos, m) -> Value:
        n = t.count(f.tool)
        ok = _cmp(n, f.op, f.n)
        detail = f'"{f.tool}" called {n} time(s); expected {f.op} {f.n}.'
        if f.op in ("<=", "<"):
            return Value(PTRUE if ok else FALSE, detail, ("n", n))
        if f.op in (">=", ">"):
            return Value(TRUE if ok else PFALSE, detail, ("n", n))
        if n > f.n:
            return Value(FALSE, detail, ("n", n))
        return Value(PTRUE if ok else PFALSE, detail, ("n", n))

    def _ordered(self, f, t: Trace, a_idx: int, b_idx: int, ok_text, bad_text) -> Value:
        """First a before first b, with the step-boundary rule of the strict evaluator."""
        if a_idx != -1 and a_idx < b_idx:
            a_step = t.at(a_idx).raw.get("step")
            b_step = t.at(b_idx).raw.get("step")
            if a_step is None or b_step is None or a_step < b_step:
                return Value(TRUE, ok_text)
        return Value(FALSE, bad_text)

    def _Before(self, f: Before, t: Trace, pos, m) -> Value:
        a_idx, b_idx = t.first_index(f.a), t.first_index(f.b)
        if b_idx == -1:
            return Value(PENDING, f'"{f.b}" not yet called — ordering constraint not triggered.')
        return self._ordered(
            f, t, a_idx, b_idx,
            lambda: f'"{f.a}" (call #{a_idx + 1}) before "{f.b}" (call #{b_idx + 1}).',
            lambda: (f'"{f.b}" (call #{b_idx + 1}) called before "{f.a}".' if a_idx == -1 or a_idx > b_idx
                     else f'"{f.a}" and "{f.b}" (call #{b_idx + 1}) are in the same agent step.'))

    def _After(self, f: After, t: Trace, pos, m) -> Value:
        a_idx, b_idx = t.first_index(f.a), t.first_index(f.b)
        if a_idx != -1 and b_idx != -1:
            if a_idx > b_idx:
                return Value(TRUE, f'"{f.a}" (call #{a_idx + 1}) after "{f.b}" (call #{b_idx + 1}).')
            return Value(FALSE, f'"{f.a}" (call #{a_idx + 1}) was NOT after "{f.b}" (call #{b_idx + 1}).')
        if a_idx != -1:
            return Value(FALSE, f'"{f.a}" (call #{a_idx + 1}) came before any "{f.b}".')
        return Value(PFALSE, f'"{f.a}" has not been called after "{f.b}" yet.')

    def _AllBefore(self, f: AllBefore, t: Trace, pos, m) -> Value:
        target = t.first_index(f.target)
        if target == -1:
            return Value(PENDING, f'"{f.target}" not yet called — AllBefore constraint not triggered.')
        late = [tool for tool in f.tools if not (-1 < t.first_index(tool) < target)]
        if not late:
            return Value(TRUE, f'All {len(f.tools)} tools completed before "{f.target}".')
        return Value(FALSE, f'Gate violation for "{f.target}" (call #{target + 1}): '
                            f'{", ".join(late)} not called before it.')

    def _BranchCalled(self, f: BranchCalled, t: Trace, pos, m) -> Value:
        correct = t.first_index(f.correct_tool)
        wrong = t.first_index(f.wrong_tool) if f.wrong_tool else -1
        ctx = f" [{f.context}]" if f.context else ""
        if wrong != -1:
            return Value(FALSE, f'"{f.wrong_tool}" was called{ctx} — wrong branch.')
        if correct != -1:
            return Value(PTRUE if f.wrong_tool else TRUE,
                         f'"{f.correct_tool}" correctly called (call #{correct + 1}){ctx}.')
        return Value(PENDING, f'Branch not reached yet for "{f.correct_tool}"{ctx}.')

    def _InstanceBefore(self, f: InstanceBefore, t: Trace, pos, m) -> Value:
        b_idx = t.nth_index(f.tool_b, f.m)
        if b_idx == -1:
            return Value(PENDING, f'Waiting for "{f.tool_b}" occurrence #{f.m}.')
        a_idx = t.nth_index(f.tool_a, f.n)
        return self._ordered(
            f, t, a_idx, b_idx,
            f'"{f.tool_a}"[{f.n}] before "{f.tool_b}"[{f.m}] (call #{b_idx + 1}).',
            f'"{f.tool_b}"[{f.m}] (call #{b_idx + 1}) is not after "{f.tool_a}"[{f.n}].')

    def _CalledInOrder(self, f: CalledInOrder, t: Trace, pos, m) -> Value:
        # F(now a ∧ X F(now b ∧ X F now c)): order as a subsequence, which a later call
        # can always still complete. Decided only once it is complete.
        ptr, last_step = 0, None
        for c in t.calls:
            if ptr < len(f.tools) and c.name == f.tools[ptr]:
                step = c.raw.get("step")
                if last_step is not None and step is not None and step <= last_step:
                    continue
                ptr, last_step = ptr + 1, step
        if ptr == len(f.tools):
            return Value(TRUE, f"Sequence {list(f.tools)} found as subsequence in trace.")
        return Value(PENDING, f"Sequence {list(f.tools)} in progress: {ptr}/{len(f.tools)}.")

    def _WithinSteps(self, f: WithinSteps, t: Trace, pos, m) -> Value:
        a_idx = t.first_index(f.tool_a)
        if a_idx == -1:
            return Value(PENDING, f'"{f.tool_a}" not yet called.')
        b_idx = next((c.position for c in t.calls if c.name == f.tool_b and c.position >= a_idx), -1)
        if b_idx != -1:
            d = b_idx - a_idx
            if d <= f.n:
                return Value(TRUE, f'"{f.tool_b}" (call #{b_idx + 1}) occurs {d} step(s) after '
                                   f'"{f.tool_a}" — within {f.n}.')
            return Value(FALSE, f'"{f.tool_b}" (call #{b_idx + 1}) occurs {d} step(s) after '
                                f'"{f.tool_a}" — exceeds limit of {f.n}.')
        if len(t) - a_idx <= f.n:
            return Value(PENDING, f'"{f.tool_b}" still due within {f.n} step(s) of "{f.tool_a}".')
        return Value(FALSE, f'"{f.tool_b}" did not occur within {f.n} step(s) of "{f.tool_a}" '
                            f'(call #{a_idx + 1}).')

    def _Predicate(self, f: Predicate, t: Trace, pos, m) -> Value:
        try:
            result = call_predicate(f.fn, t, pos, m)
        except Exception as exc:
            return Value(PFALSE, f'Predicate "{f.description}" raised {type(exc).__name__}: {exc}')
        if isinstance(result, dict):
            passed = bool(result.get("passed", False))
            final = bool(result.get("final", False))
            value = (TRUE if final else PTRUE) if passed else (FALSE if final else PFALSE)
            detail = result.get("note", result.get("detail", f'Predicate "{f.description}": {passed}'))
            witness = result.get("witness")
            if witness is not None and value < PENDING:
                items = witness if isinstance(witness, (list, tuple, set)) else [witness]
                return Value(value, detail, None, [((("w", _hashable(w)),), value) for w in items])
            return Value(value, detail)
        return Value(PTRUE if result else PFALSE, f'Predicate "{f.description}" returned {result}.')

    def _Matches(self, f: Matches, t: Trace, pos, m) -> Value:
        call = t.at(pos)
        if call is None:
            return Value(PENDING, f"No call at position {pos} yet.")
        try:
            hit = f.pattern.match(call.name, call.args)
        except Exception as exc:
            return Value(PFALSE, f"Pattern {f} raised {type(exc).__name__}: {exc}")
        describe = getattr(f.pattern, "describe", lambda: str(f.pattern))
        if hit is None:
            # only a possible match: counted the cautious way, and marked as such
            v = TRUE if f.maybe else FALSE
            return Value(v, lambda: f'Call #{pos + 1} ({call.name}) may match {describe()}.',
                         ("maybe", pos))
        return Value(TRUE if hit else FALSE,
                     lambda: f'Call #{pos + 1} ({call.name}) {"matches" if hit else "does not match"} '
                             f'{describe()}.')

    # ── Past-time operators ──────────────────────────────────────────────────

    def _Previous(self, f: Previous, t: Trace, pos, m) -> Value:
        if pos >= len(t) + 1:
            return Value(PENDING, f"Position {pos} has not been reached yet.")
        if pos == 0:
            return Value(FALSE, "No previous call.")
        inner = self.value(f.operand, t, pos - 1, m)
        return Value(inner.value, lambda: f"Y at position {pos - 1}: {inner.detail}", inner.key,
                     [(("Y",) + p, w) for p, w in inner.witnesses])

    def _past(self, f, t: Trace, pos, m) -> List[Value]:
        return [self.value(f.operand, t, j, m) for j in range(0, min(pos, len(t) - 1) + 1)]

    def _Once(self, f: Once, t: Trace, pos, m) -> Value:
        if pos >= len(t):
            return Value(PENDING, f"Position {pos} has not been reached yet.")
        values = self._past(f, t, pos, m)
        best = max(range(len(values)), key=lambda j: values[j].value, default=None)
        if best is not None and values[best].value >= PENDING:
            v = values[best]
            return Value(v.value, lambda: f"O satisfied at position {best}: {v.detail}", v.key)
        value = max([v.value for v in values] + [FALSE])
        maybe = any(_maybe(v.key) for v in values)
        return Value(value, lambda: f"O: {f.operand} has not held up to position {pos}.",
                     ("maybe", pos) if maybe else None)

    def _Historically(self, f: Historically, t: Trace, pos, m) -> Value:
        if pos >= len(t):
            return Value(PENDING, f"Position {pos} has not been reached yet.")
        values = self._past(f, t, pos, m)
        worst = min(range(len(values)), key=lambda j: values[j].value)
        v = values[worst]
        if v.value >= PENDING:
            return Value(v.value, f"H holds up to position {pos}.")
        return Value(v.value, lambda: f"H violated at position {worst}: {v.detail}", v.key,
                     [(("H", worst) + p, w) for p, w in v.witnesses])

    def _Since(self, f: Since, t: Trace, pos, m) -> Value:
        if pos >= len(t):
            return Value(PENDING, f"Position {pos} has not been reached yet.")
        # max over j of min(ψ_j, φ_{j+1..pos}), scanning back from pos
        best, suffix, maybe = FALSE, TRUE, False
        for j in range(pos, -1, -1):
            right = self.value(f.right, t, j, m)
            best = max(best, min(right.value, suffix))
            left = self.value(f.left, t, j, m)
            maybe = maybe or _maybe(left.key) or _maybe(right.key)
            suffix = min(suffix, left.value)
            if suffix < PENDING and best >= PENDING:
                break
        value = best
        return Value(value, lambda: (f"S holds at position {pos}." if value >= PENDING else
                                     f"S violated at position {pos}: {f.right} has not held "
                                     f"since the last time {f.left} failed."),
                     ("maybe", pos) if maybe else None)

    def _CountBefore(self, f: CountBefore, t: Trace, pos, m) -> Value:
        if pos >= len(t):
            return Value(PENDING, f"Position {pos} has not been reached yet.")
        values = [self.value(f.operand, t, j, m) for j in range(pos)]
        n = sum(1 for v in values if v.value > PENDING)
        ok = _cmp(n, f.op, f.n)
        return Value(TRUE if ok else FALSE,
                     f"{f.operand} held {n} time(s) before position {pos}; expected {f.op} {f.n}.",
                     ("#", n))

    def _AtPosition(self, f: AtPosition, t: Trace, pos, m) -> Value:
        if f.index < 0:
            return Value(FALSE, f"Position {f.index} is out of range.")
        if f.index >= len(t):
            return Value(PENDING, f"Position {f.index} has not been reached yet.")
        return self.value(f.operand, t, f.index, m)

    # ── Connectives ──────────────────────────────────────────────────────────

    def _Not(self, f: Not, t: Trace, pos, m) -> Value:
        inner = self.value(f.operand, t, pos, m)
        return Value(TRUE - inner.value, lambda: f"NOT({inner.detail})", ("¬", inner.key))

    def _And(self, f: And, t: Trace, pos, m) -> Value:
        left, right = self.value(f.left, t, pos, m), self.value(f.right, t, pos, m)
        value = min(left.value, right.value)
        witnesses = ([(("L",) + p, v) for p, v in left.witnesses]
                     + [(("R",) + p, v) for p, v in right.witnesses])

        def detail():
            if value >= PENDING:
                return f"Both satisfied: [{left.detail}] AND [{right.detail}]"
            if left.value < PENDING and right.value < PENDING:
                return f"Both failed: [{left.detail}] AND [{right.detail}]"
            return f"Left failed: {left.detail}" if left.value < PENDING else f"Right failed: {right.detail}"
        return Value(value, detail, None, witnesses)

    def _Or(self, f: Or, t: Trace, pos, m) -> Value:
        left, right = self.value(f.left, t, pos, m), self.value(f.right, t, pos, m)
        value = max(left.value, right.value)
        return Value(value, lambda: (f"At least one satisfied: [{left.detail}] OR [{right.detail}]"
                                     if value >= PENDING else
                                     f"Neither satisfied: [{left.detail}] OR [{right.detail}]"),
                     ("∨",) + _deciding(value, (left.value, left.key), (right.value, right.key)))

    def _Implies(self, f: Implies, t: Trace, pos, m) -> Value:
        left = self.value(f.left, t, pos, m)
        if left.value == FALSE:
            return Value(TRUE, lambda: f"Antecedent false — implication vacuously true. ({left.detail})")
        right = self.value(f.right, t, pos, m)
        value = max(TRUE - left.value, right.value)
        if value >= PENDING:
            text = lambda: f"Antecedent true and consequent satisfied: {right.detail}"  # noqa: E731
        else:
            text = lambda: f"Antecedent true but consequent failed: {right.detail}"  # noqa: E731
        return Value(value, text, ("→",) + _deciding(
            value, (TRUE - left.value, ("¬", left.key)), (right.value, right.key)))

    # ── Temporal operators ───────────────────────────────────────────────────

    def _instances(self, operand: Formula, t: Trace, pos: int, m) -> List[Tuple[int, Value]]:
        if pos >= len(t):
            return []
        if not is_local(operand):
            return [(-1, self.value(operand, t, pos, m))]
        return [(i, self.value(operand, t, i, m)) for i in range(pos, len(t))]

    def _Globally(self, f: Globally, t: Trace, pos, m) -> Value:
        values = self._instances(f.operand, t, pos, m)
        value = min([v.value for _, v in values] + [PENDING])
        witnesses = [(("G", i) + p, w) for i, v in values for p, w in v.witnesses]
        first = next(((i, v) for i, v in values if v.value < PENDING), None)
        if first is None:
            return Value(value, "G holds so far.", None, witnesses)
        i, v = first
        where = f"position {i}" if i >= 0 else "every position"
        return Value(value, lambda: f"G violated at {where}: {v.detail}", None, witnesses)

    def _Eventually(self, f: Eventually, t: Trace, pos, m) -> Value:
        values = self._instances(f.operand, t, pos, m)
        best = max(values, key=lambda iv: iv[1].value, default=None)
        if best is not None and best[1].value > PENDING:
            i, v = best
            return Value(v.value, lambda: f"F satisfied at position {max(i, pos)}: {v.detail}")
        return Value(PENDING, f"F not satisfied yet: {f.operand} has not held from position {pos}.")

    def _Next(self, f: Next, t: Trace, pos, m) -> Value:
        if pos + 1 >= len(t):
            return Value(PENDING, "X not decided yet: no next call has been made.")
        inner = self.value(f.operand, t, pos + 1, m)
        return Value(inner.value, lambda: f"X at position {pos + 1}: {inner.detail}", inner.key,
                     [(("X",) + p, w) for p, w in inner.witnesses])

    def _until(self, lefts: Sequence[int], rights: Sequence[int]) -> int:
        """max over i of min(ψ_i, φ_0..φ_{i-1}), then max with min(φ_0..φ_n-1, PENDING)."""
        best, prefix = FALSE, TRUE
        for lv, rv in zip(lefts, rights):
            best = max(best, min(rv, prefix))
            prefix = min(prefix, lv)
        return max(best, min(prefix, PENDING))

    def _until_values(self, f, t: Trace, pos, m):
        lefts, rights = [], []
        for i in range(pos, len(t)):
            lefts.append(self.value(f.left, t, i, m).value)
            rights.append(self.value(f.right, t, i, m).value)
        return lefts, rights

    def _Until(self, f: Until, t: Trace, pos, m) -> Value:
        value = self._until(*self._until_values(f, t, pos, m))
        return Value(value, lambda: ("Until satisfied." if value > PENDING else "Until pending." if
                                     value == PENDING else "Until violated: left side failed "
                                     "before right side held."))

    def _WeakUntil(self, f: WeakUntil, t: Trace, pos, m) -> Value:
        value = self._until(*self._until_values(f, t, pos, m))
        return Value(value, lambda: ("WeakUntil holds so far." if value >= PENDING else
                                     "WeakUntil violated: left side failed before right side held."))

    def _Release(self, f: Release, t: Trace, pos, m) -> Value:
        lefts, rights = self._until_values(f, t, pos, m)
        value = TRUE - self._until([TRUE - v for v in lefts], [TRUE - v for v in rights])
        return Value(value, lambda: ("Release holds so far." if value >= PENDING else
                                     "Release violated: right side failed before left side held."))

    # ── Quantifiers ──────────────────────────────────────────────────────────

    def _entities(self, f, t: Trace, m, pos: int = 0):
        try:
            return call_domain(f.domain, t, m, pos), None
        except Exception as exc:
            return None, f"domain extractor raised {type(exc).__name__}: {exc}"

    def _ForAll(self, f: ForAll, t: Trace, pos, m) -> Value:
        entities, error = self._entities(f, t, m, pos)
        if error:
            return Value(PFALSE, f"∀{f.var}: {error}")
        if not entities:
            return Value(PENDING, f"∀{f.var}: vacuously true (empty domain).")
        results = [(e, self.value(substitute(f.body, {f.var: e}), t, pos, m)) for e in entities]
        value = min([v.value for _, v in results] + [PENDING])
        witnesses = [(("∀", _hashable(e)) + p, w) for e, v in results for p, w in v.witnesses]
        bad = next(((e, v) for e, v in results if v.value < PENDING), None)
        if bad is None:
            return Value(value, f"∀{f.var}: holds for all {len(entities)} entities.", None, witnesses)
        e, v = bad
        return Value(value, lambda: f"∀{f.var}: failed for {f.var}={e!r}. {v.detail}", None, witnesses)

    def _Exists(self, f: Exists, t: Trace, pos, m) -> Value:
        entities, error = self._entities(f, t, m, pos)
        if error:
            return Value(PFALSE, f"∃{f.var}: {error}")
        if not entities:
            return Value(PFALSE, f"∃{f.var}: no entity yet (empty domain).")
        results = [(e, self.value(substitute(f.body, {f.var: e}), t, pos, m)) for e in entities]
        value = max([v.value for _, v in results] + [PFALSE])
        good = next(((e, v) for e, v in results if v.value >= PENDING), None)
        if good is not None:
            e, v = good
            return Value(value, lambda: f"∃{f.var}: satisfied for {f.var}={e!r}. {v.detail}")
        return Value(value, f"∃{f.var}: no entity satisfied the formula ({len(entities)} checked).")


def _deciding(value: int, *sides: Tuple[int, Any]) -> Tuple[Any, ...]:
    """The keys of the sides that decide a disjunction's value.

    A side that doesn't decide it may still change (an undecided successor), and must
    not change the identity of a failure that is already final. Once the disjunction
    holds for good, more sides may come to hold too: it is then identified by that
    alone.
    """
    if value == TRUE:
        return ("⊤",)
    return tuple(key if v == value else None for v, key in sides)


def _maybe(key: Any) -> bool:
    """Does a witness key rest on a possible match (see Matches)?"""
    if isinstance(key, tuple):
        return bool(key) and (key[0] == "maybe" or any(_maybe(k) for k in key))
    return False


def uncertain(witnesses: List[Witness]) -> bool:
    """Does any failing witness rest on a possible match rather than a certain one?"""
    return any(_maybe(path) for path, value in witnesses if value < PENDING)


def _hashable(x: Any) -> Any:
    try:
        hash(x)
        return x
    except TypeError:
        return repr(x)


def caused_elsewhere(now: Value, before: Optional[Value]) -> bool:
    """Is every failure in *now* one that was already final before this call?

    *now* is the formula on the calls made so far plus the call being checked, *before*
    the same formula without it. A failing witness that was FALSE already is not this
    call's doing: refusing the call would repair nothing.
    """
    if before is None or not now.witnesses:
        return False
    final = {path for path, value in before.witnesses if value == FALSE}
    return all(path in final for path, value in now.witnesses if value < PENDING)
