"""
agentltl/_evaluator.py – FOLTL formula evaluation over tool-call traces.

The :class:`LTLEvaluator` recursively interprets an FOLTL :class:`Formula` AST
node against a :class:`Trace`.  It extends propositional LTL evaluation with
support for first-order quantifiers (∀, ∃) via eager substitution.

Evaluation is synchronous, operates on finite traces, and produces both a
boolean verdict **and** a human-readable explanation string suitable for
compliance reports.

Finite-trace semantics
----------------------
Standard LTL is defined over infinite traces.  For finite agent traces we
adopt the *weak* finite-trace semantics common in runtime verification:

* **F φ** is true iff φ holds at some position in the remaining trace.
* **G φ** is true iff φ holds at every position in the remaining trace
  (vacuously true on the empty suffix).
* **X φ** is true iff a next position exists *and* φ holds there.
  (X φ is *false* when the trace has ended — safety-oriented choice.)
* **φ U ψ** (strong until) requires ψ to eventually hold.
* **φ W ψ** (weak until) is satisfied if ψ never occurs but φ holds forever.
* **φ R ψ** is the dual of Until.

First-order quantifier semantics
---------------------------------
* **∀x ∈ D. φ(x)** — true iff φ holds for every entity in D.
  Empty D → vacuously true.
* **∃x ∈ D. φ(x)** — true iff φ holds for at least one entity in D.
  Empty D → false.

Atomic propositions (:class:`Called`, :class:`Before`, etc.) are evaluated
against the *full* trace regardless of the current suffix position, since
they express global facts about the run. :class:`Now` is the exception: it
looks at the call at the current position only.

Partial traces
--------------
These are the semantics of a *complete* trace, used for scoring. While a run is
still going (``partial_trace=True``) the formula is judged in the five-valued
semantics of :mod:`agentltl._partial`, which knows the difference between a final
violation and an obligation not met yet; ``EvalResult.value`` and
``EvalResult.witnesses`` carry that verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ._partial import PartialEvaluator, call_domain, call_predicate, match_call
from ._ast import (
    AllBefore,
    After,
    And,
    AtPosition,
    Before,
    BranchCalled,
    Called,
    CalledNTimes,
    CalledWith,
    CalledWithResult,
    CalledInOrder,
    CountBefore,
    Eventually,
    Exists,
    ForAll,
    Formula,
    Globally,
    Historically,
    Implies,
    InstanceBefore,
    Matches,
    Next,
    Not,
    Now,
    Once,
    Or,
    Predicate,
    Previous,
    Release,
    Since,
    Until,
    WeakUntil,
    WithinSteps,
    substitute,
)
from ._trace import Trace


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Result of evaluating a single formula on a trace.

    Attributes:
        passed:    Whether the formula is satisfied.
        detail:    Human-readable explanation of the verdict.
        formula:   The formula that was evaluated (for debugging).
        value:     On a partial trace, the five-valued verdict (see
                   :mod:`agentltl._partial`); None on a complete trace.
        witnesses: On a partial trace, the failing instances with their values.
    """
    passed: bool
    detail: str
    formula: Formula
    value: Optional[int] = None
    witnesses: List[Tuple[Tuple[Any, ...], int]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class LTLEvaluator:
    """Stateless interpreter that evaluates LTL formulas against finite traces.

    Usage::

        evaluator = LTLEvaluator()
        result = evaluator.evaluate(formula, trace)
        print(result.passed, result.detail)
    """

    def evaluate(
        self,
        formula: Formula,
        trace: Trace,
        position: int = 0,
        *,
        metrics: Optional[Dict[str, Any]] = None,
        partial_trace: bool = False,
    ) -> EvalResult:
        """Evaluate *formula* on *trace* starting at *position*.

        Args:
            formula:  The FOLTL formula to evaluate.
            trace:    The tool-call trace.
            position: Starting position in the trace (for temporal operators).
            metrics:  Optional raw metrics dict, passed to domain extractors
                      in ForAll / Exists quantifiers.
            partial_trace: When ``True``, the run is still going: judge the
                      formula in the five-valued semantics of
                      :mod:`agentltl._partial`. *passed* is then "not refused
                      now", and *value* / *witnesses* say how final that is.
                      Default ``False``: the strict post-hoc semantics.

        Returns:
            An :class:`EvalResult` with *passed* (bool) and *detail* (str).
        """
        if partial_trace:
            v = _PARTIAL.value(formula, trace, position, metrics)
            return EvalResult(v.passed, v.detail, formula, v.value, v.witnesses)

        # ── First-order quantifiers ──────────────────────────────────────────
        if isinstance(formula, ForAll):
            return self._eval_forall(formula, trace, position, metrics)

        if isinstance(formula, Exists):
            return self._eval_exists(formula, trace, position, metrics)

        # ── Atomic propositions ──────────────────────────────────────────────
        if isinstance(formula, Called):
            return self._eval_called(formula, trace)

        if isinstance(formula, Now):
            return self._eval_now(formula, trace, position)

        if isinstance(formula, CalledWith):
            return self._eval_called_with(formula, trace)

        if isinstance(formula, CalledNTimes):
            return self._eval_called_n_times(formula, trace)

        if isinstance(formula, Before):
            return self._eval_before(formula, trace)

        if isinstance(formula, After):
            return self._eval_after(formula, trace)

        if isinstance(formula, AllBefore):
            return self._eval_all_before(formula, trace)

        if isinstance(formula, BranchCalled):
            return self._eval_branch_called(formula, trace)

        if isinstance(formula, CalledWithResult):
            return self._eval_called_with_result(formula, trace)

        if isinstance(formula, InstanceBefore):
            return self._eval_instance_before(formula, trace)

        if isinstance(formula, CalledInOrder):
            return self._eval_called_in_order(formula, trace)

        if isinstance(formula, WithinSteps):
            return self._eval_within_steps(formula, trace)

        if isinstance(formula, Predicate):
            return self._eval_predicate(formula, trace, position, metrics)

        if isinstance(formula, Matches):
            return self._eval_matches(formula, trace, position)

        if isinstance(formula, (Previous, Once, Historically, Since, CountBefore)):
            return self._eval_past(formula, trace, position, metrics)

        if isinstance(formula, AtPosition):
            return self._eval_at_position(formula, trace, metrics)

        # ── Logical connectives ──────────────────────────────────────────────
        if isinstance(formula, Not):
            return self._eval_not(formula, trace, position, metrics)

        if isinstance(formula, And):
            return self._eval_and(formula, trace, position, metrics)

        if isinstance(formula, Or):
            return self._eval_or(formula, trace, position, metrics)

        if isinstance(formula, Implies):
            return self._eval_implies(formula, trace, position, metrics)

        # ── Temporal operators ───────────────────────────────────────────────
        if isinstance(formula, Globally):
            return self._eval_globally(formula, trace, position, metrics)

        if isinstance(formula, Eventually):
            return self._eval_eventually(formula, trace, position, metrics)

        if isinstance(formula, Next):
            return self._eval_next(formula, trace, position, metrics)

        if isinstance(formula, Until):
            return self._eval_until(formula, trace, position, metrics)

        if isinstance(formula, WeakUntil):
            return self._eval_weak_until(formula, trace, position, metrics)

        if isinstance(formula, Release):
            return self._eval_release(formula, trace, position, metrics)

        raise TypeError(f"Unknown formula type: {type(formula).__name__}")

    # ─────────────────────────────────────────────────────────────────────────
    # Atomic propositions
    # ─────────────────────────────────────────────────────────────────────────

    def _eval_called(self, f: Called, trace: Trace) -> EvalResult:
        if trace.contains(f.tool):
            idx = trace.first_index(f.tool)
            return EvalResult(True, f'"{f.tool}" was called (call #{idx + 1}).', f)
        return EvalResult(False, f'"{f.tool}" was never called.', f)

    def _eval_now(self, f: Now, trace: Trace, pos: int) -> EvalResult:
        call = trace.at(pos)
        if call is None:
            return EvalResult(False, f'No call at position {pos}.', f)
        if call.name == f.tool:
            return EvalResult(True, f'Call #{pos + 1} is "{f.tool}".', f)
        return EvalResult(False, f'Call #{pos + 1} is "{call.name}", not "{f.tool}".', f)

    def _eval_called_with(self, f: CalledWith, trace: Trace) -> EvalResult:
        matches = trace.calls_with(f.tool, f.expected_args)
        if matches:
            pos = matches[0].position
            return EvalResult(True, f'"{f.tool}" called with matching args at call #{pos + 1}.', f)
        if trace.contains(f.tool):
            return EvalResult(
                False,
                f'"{f.tool}" was called but not with the expected arguments {f.expected_args}.',
                f,
            )
        return EvalResult(False, f'"{f.tool}" was never called.', f)

    def _eval_called_n_times(self, f: CalledNTimes, trace: Trace) -> EvalResult:
        actual = trace.count(f.tool)
        ops = {
            "==": actual == f.n,
            ">=": actual >= f.n,
            "<=": actual <= f.n,
            ">": actual > f.n,
            "<": actual < f.n,
        }
        passed = ops.get(f.op, False)
        return EvalResult(passed, f'"{f.tool}" called {actual} time(s); expected {f.op} {f.n}.', f)

    def _eval_before(self, f: Before, trace: Trace) -> EvalResult:
        a_idx = trace.first_index(f.a)
        b_idx = trace.first_index(f.b)

        if b_idx == -1:
            if a_idx == -1:
                return EvalResult(False, f'Neither "{f.a}" nor "{f.b}" was called.', f)
            return EvalResult(
                False,
                f'"{f.b}" was never called ("{f.a}" has no successor to precede).',
                f,
            )
        if a_idx == -1:
            return EvalResult(False, f'"{f.a}" was never called (cannot precede "{f.b}").', f)
        if a_idx < b_idx:
            a_call = trace.at(a_idx)
            b_call = trace.at(b_idx)
            a_step = a_call.raw.get("step") if a_call else None
            b_step = b_call.raw.get("step") if b_call else None
            if a_step is not None and b_step is not None and a_step >= b_step:
                return EvalResult(
                    False,
                    f'"{f.a}" (call #{a_idx + 1}, step {a_step}) and "{f.b}" (call #{b_idx + 1}, step {b_step}) '
                    f'are in the same agent step — step-boundary ordering requires strictly earlier steps.',
                    f,
                )
            return EvalResult(
                True,
                f'"{f.a}" (call #{a_idx + 1}) before "{f.b}" (call #{b_idx + 1}).',
                f,
            )
        return EvalResult(
            False,
            f'"{f.b}" (call #{b_idx + 1}) called before "{f.a}" (call #{a_idx + 1}).',
            f,
        )

    def _eval_after(self, f: After, trace: Trace) -> EvalResult:
        a_idx = trace.first_index(f.a)
        b_idx = trace.first_index(f.b)

        if a_idx == -1 or b_idx == -1:
            missing = []
            if a_idx == -1:
                missing.append(f.a)
            if b_idx == -1:
                missing.append(f.b)
            return EvalResult(False, f'Tool(s) not called: {missing}.', f)
        if a_idx > b_idx:
            return EvalResult(
                True,
                f'"{f.a}" (call #{a_idx + 1}) after "{f.b}" (call #{b_idx + 1}).',
                f,
            )
        return EvalResult(
            False,
            f'"{f.a}" (call #{a_idx + 1}) was NOT after "{f.b}" (call #{b_idx + 1}).',
            f,
        )

    def _eval_all_before(self, f: AllBefore, trace: Trace) -> EvalResult:
        target_idx = trace.first_index(f.target)
        if target_idx == -1:
            return EvalResult(False, f'"{f.target}" was never called.', f)

        missing_or_late = []
        for tool in f.tools:
            idx = trace.first_index(tool)
            if idx == -1:
                missing_or_late.append(f'"{tool}" not called')
            elif idx >= target_idx:
                missing_or_late.append(
                    f'"{tool}" (call #{idx + 1}) not before "{f.target}" (call #{target_idx + 1})'
                )

        if not missing_or_late:
            return EvalResult(
                True,
                f'All {len(f.tools)} tools completed before "{f.target}" (call #{target_idx + 1}).',
                f,
            )
        return EvalResult(False, f'Gate violation for "{f.target}": {"; ".join(missing_or_late)}.', f)

    def _eval_branch_called(self, f: BranchCalled, trace: Trace) -> EvalResult:
        correct_idx = trace.first_index(f.correct_tool)
        wrong_idx = trace.first_index(f.wrong_tool) if f.wrong_tool else -1
        ctx = f" [{f.context}]" if f.context else ""

        if correct_idx != -1 and wrong_idx == -1:
            return EvalResult(True, f'"{f.correct_tool}" correctly called (call #{correct_idx + 1}){ctx}.', f)
        if correct_idx != -1 and wrong_idx != -1:
            return EvalResult(
                False,
                f'Both "{f.correct_tool}" (call #{correct_idx + 1}) and wrong tool "{f.wrong_tool}" '
                f'(call #{wrong_idx + 1}) were called{ctx} — both branches triggered.',
                f,
            )
        if wrong_idx != -1:
            return EvalResult(
                False,
                f'"{f.wrong_tool}" called instead of "{f.correct_tool}"{ctx} — wrong branch.',
                f,
            )
        return EvalResult(
            False,
            f'Neither "{f.correct_tool}" nor "{f.wrong_tool or "alternative"}" was called{ctx}.',
            f,
        )

    def _eval_called_with_result(self, f: CalledWithResult, trace: Trace) -> EvalResult:
        candidates = (
            trace.calls_with(f.tool, f.expected_args)
            if f.expected_args
            else [c for c in trace.calls if c.name == f.tool]
        )
        if not candidates:
            if not trace.contains(f.tool):
                return EvalResult(False, f'"{f.tool}" was never called.', f)
            return EvalResult(
                False,
                f'"{f.tool}" was called but not with the expected arguments {f.expected_args}.',
                f,
            )
        for c in candidates:
            if self._match_result(c.result, f.expected_result):
                return EvalResult(True, f'"{f.tool}" call #{c.position + 1} returned matching result.', f)
        actual_results = [c.result for c in candidates]
        return EvalResult(
            False,
            f'"{f.tool}" called {len(candidates)} time(s) but no call returned '
            f'a result matching {f.expected_result!r}. Actual results: {actual_results}.',
            f,
        )

    @staticmethod
    def _match_result(actual: Any, expected: Any) -> bool:
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            return all(actual.get(k) == v for k, v in expected.items())
        return actual == expected

    def _eval_instance_before(self, f: InstanceBefore, trace: Trace) -> EvalResult:
        a_idx = trace.nth_index(f.tool_a, f.n)
        b_idx = trace.nth_index(f.tool_b, f.m)
        a_count = trace.count(f.tool_a)
        b_count = trace.count(f.tool_b)

        if a_idx == -1:
            return EvalResult(False, f'"{f.tool_a}" has only {a_count} occurrence(s); need at least {f.n}.', f)
        if b_idx == -1:
            return EvalResult(False, f'"{f.tool_b}" has only {b_count} occurrence(s); need at least {f.m}.', f)
        if a_idx < b_idx:
            a_call = trace.at(a_idx)
            b_call = trace.at(b_idx)
            a_step = a_call.raw.get("step") if a_call else None
            b_step = b_call.raw.get("step") if b_call else None
            if a_step is not None and b_step is not None and a_step >= b_step:
                return EvalResult(
                    False,
                    f'"{f.tool_a}"[{f.n}] (call #{a_idx + 1}, step {a_step}) and "{f.tool_b}"[{f.m}] (call #{b_idx + 1}, step {b_step}) are in the same agent step — step-boundary ordering requires strictly earlier steps.',
                    f,
                )
            return EvalResult(
                True,
                f'"{f.tool_a}"[{f.n}] (call #{a_idx + 1}) before "{f.tool_b}"[{f.m}] (call #{b_idx + 1}).',
                f,
            )
        return EvalResult(
            False,
            f'"{f.tool_b}"[{f.m}] (call #{b_idx + 1}) is not after "{f.tool_a}"[{f.n}] (call #{a_idx + 1}).',
            f,
        )

    def _eval_called_in_order(self, f: CalledInOrder, trace: Trace) -> EvalResult:
        if not f.tools:
            return EvalResult(True, "CalledInOrder: empty sequence — vacuously true.", f)

        ptr = 0
        last_step: Optional[int] = None
        for c in trace.calls:
            if ptr < len(f.tools) and c.name == f.tools[ptr]:
                c_step = c.raw.get("step")
                # Step-boundary rule: consecutive elements of the sequence must
                # be executed in strictly different agent steps.  Skip this
                # occurrence if it shares a step with the previously matched
                # tool (same-step calls are parallel, not sequential).
                if last_step is not None and c_step is not None and c_step <= last_step:
                    continue
                ptr += 1
                last_step = c_step
                if ptr == len(f.tools):
                    return EvalResult(True, f"Sequence {list(f.tools)} found as subsequence in trace.", f)
                continue

        found = list(f.tools[:ptr])
        missing_from = f.tools[ptr]
        return EvalResult(
            False,
            f'Sequence incomplete: found {found}, then "{missing_from}" (and beyond) not found in order.',
            f,
        )

    def _eval_within_steps(self, f: WithinSteps, trace: Trace) -> EvalResult:
        a_idx = trace.first_index(f.tool_a)
        if a_idx == -1:
            return EvalResult(False, f'"{f.tool_a}" was never called.', f)
        b_idx = next(
            (c.position for c in trace.calls if c.name == f.tool_b and c.position >= a_idx),
            -1,
        )
        if b_idx == -1:
            return EvalResult(False, f'"{f.tool_b}" never occurred after "{f.tool_a}" (call #{a_idx + 1}).', f)
        distance = b_idx - a_idx
        if distance <= f.n:
            return EvalResult(
                True,
                f'"{f.tool_b}" (call #{b_idx + 1}) occurs {distance} step(s) after '
                f'"{f.tool_a}" (call #{a_idx + 1}) — within {f.n}.',
                f,
            )
        return EvalResult(
            False,
            f'"{f.tool_b}" (call #{b_idx + 1}) occurs {distance} step(s) after '
            f'"{f.tool_a}" (call #{a_idx + 1}) — exceeds limit of {f.n}.',
            f,
        )

    def _eval_predicate(self, f: Predicate, trace: Trace, position: int, metrics: Optional[Dict[str, Any]] = None) -> EvalResult:
        try:
            result = call_predicate(f.fn, trace, position, metrics)
        except Exception as exc:
            return EvalResult(False, f'Predicate "{f.description}" raised {type(exc).__name__}: {exc}', f)
        if isinstance(result, dict):
            passed = bool(result.get("passed", False))
            detail = result.get("note", result.get("detail", f'Predicate "{f.description}": {passed}'))
            return EvalResult(passed, detail, f)
        return EvalResult(bool(result), f'Predicate "{f.description}" returned {result}.', f)

    def _eval_matches(self, f: Matches, trace: Trace, pos: int) -> EvalResult:
        call = trace.at(pos)
        if call is None:
            return EvalResult(False, f"No call at position {pos}.", f)
        hit = match_call(f.pattern, call)
        if hit is None:
            return EvalResult(bool(f.maybe), f"Call #{pos + 1} ({call.name}) may match {f}.", f)
        return EvalResult(bool(hit), f"Call #{pos + 1} ({call.name}) "
                                     f"{'matches' if hit else 'does not match'} {f}.", f)

    def _eval_past(self, f: Formula, trace: Trace, pos: int,
                   metrics: Optional[Dict[str, Any]]) -> EvalResult:
        def holds(g: Formula, j: int) -> bool:
            return self.evaluate(g, trace, j, metrics=metrics).passed

        last = min(pos, len(trace) - 1)
        if isinstance(f, Previous):
            ok = 0 < pos <= len(trace) and holds(f.operand, pos - 1)
        elif isinstance(f, Once):
            ok = any(holds(f.operand, j) for j in range(last + 1))
        elif isinstance(f, Historically):
            ok = all(holds(f.operand, j) for j in range(last + 1))
        elif isinstance(f, Since):
            ok = False
            for j in range(last, -1, -1):
                if holds(f.right, j):
                    ok = True
                    break
                if not holds(f.left, j):
                    break
        else:
            n = sum(1 for j in range(min(pos, len(trace))) if holds(f.operand, j))
            ok = {"==": n == f.n, ">=": n >= f.n, "<=": n <= f.n, ">": n > f.n,
                  "<": n < f.n}.get(f.op, False)
        return EvalResult(ok, f"{f} {'holds' if ok else 'fails'} at position {pos}.", f)

    def _eval_at_position(self, f: AtPosition, trace: Trace, metrics: Optional[Dict[str, Any]] = None) -> EvalResult:
        if f.index >= len(trace) or f.index < 0:
            return EvalResult(False, f"Position {f.index} is out of range (trace length {len(trace)}).", f)
        return self.evaluate(f.operand, trace, f.index, metrics=metrics)

    # ─────────────────────────────────────────────────────────────────────────
    # Logical connectives
    # ─────────────────────────────────────────────────────────────────────────

    def _eval_not(self, f: Not, trace: Trace, pos: int, metrics: Optional[Dict[str, Any]] = None) -> EvalResult:
        inner = self.evaluate(f.operand, trace, pos, metrics=metrics)
        return EvalResult(not inner.passed, f"NOT({inner.detail})", f)

    def _eval_and(self, f: And, trace: Trace, pos: int, metrics: Optional[Dict[str, Any]] = None) -> EvalResult:
        left = self.evaluate(f.left, trace, pos, metrics=metrics)
        right = self.evaluate(f.right, trace, pos, metrics=metrics)
        passed = left.passed and right.passed
        if passed:
            detail = f"Both satisfied: [{left.detail}] AND [{right.detail}]"
        elif not left.passed and not right.passed:
            detail = f"Both failed: [{left.detail}] AND [{right.detail}]"
        elif not left.passed:
            detail = f"Left failed: {left.detail}"
        else:
            detail = f"Right failed: {right.detail}"
        return EvalResult(passed, detail, f)

    def _eval_or(self, f: Or, trace: Trace, pos: int, metrics: Optional[Dict[str, Any]] = None) -> EvalResult:
        left = self.evaluate(f.left, trace, pos, metrics=metrics)
        right = self.evaluate(f.right, trace, pos, metrics=metrics)
        passed = left.passed or right.passed
        if passed:
            detail = f"At least one satisfied: [{left.detail}] OR [{right.detail}]"
        else:
            detail = f"Neither satisfied: [{left.detail}] OR [{right.detail}]"
        return EvalResult(passed, detail, f)

    def _eval_implies(self, f: Implies, trace: Trace, pos: int, metrics: Optional[Dict[str, Any]] = None) -> EvalResult:
        left = self.evaluate(f.left, trace, pos, metrics=metrics)
        if not left.passed:
            return EvalResult(True, f"Antecedent false — implication vacuously true. ({left.detail})", f)
        right = self.evaluate(f.right, trace, pos, metrics=metrics)
        if right.passed:
            return EvalResult(True, f"Antecedent true and consequent satisfied: {right.detail}", f)
        return EvalResult(False, f"Antecedent true but consequent failed: {right.detail}", f)

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal operators
    # ─────────────────────────────────────────────────────────────────────────

    def _eval_globally(self, f: Globally, trace: Trace, pos: int, metrics: Optional[Dict[str, Any]] = None) -> EvalResult:
        for i in range(pos, len(trace)):
            r = self.evaluate(f.operand, trace, i, metrics=metrics)
            if not r.passed:
                return EvalResult(False, f"G violated at position {i}: {r.detail}", f)
        return EvalResult(True, "G holds at all positions.", f)

    def _eval_eventually(self, f: Eventually, trace: Trace, pos: int, metrics: Optional[Dict[str, Any]] = None) -> EvalResult:
        for i in range(pos, len(trace)):
            r = self.evaluate(f.operand, trace, i, metrics=metrics)
            if r.passed:
                return EvalResult(True, f"F satisfied at position {i}: {r.detail}", f)
        return EvalResult(False, f"F not satisfied: {f.operand} never held from position {pos} onward.", f)

    def _eval_next(self, f: Next, trace: Trace, pos: int, metrics: Optional[Dict[str, Any]] = None) -> EvalResult:
        nxt = pos + 1
        if nxt >= len(trace):
            return EvalResult(False, f"X cannot be evaluated: no next position (trace ended at {len(trace)}).", f)
        r = self.evaluate(f.operand, trace, nxt, metrics=metrics)
        return EvalResult(r.passed, f"X at position {nxt}: {r.detail}", f)

    def _eval_until(self, f: Until, trace: Trace, pos: int, metrics: Optional[Dict[str, Any]] = None) -> EvalResult:
        for i in range(pos, len(trace)):
            rhs = self.evaluate(f.right, trace, i, metrics=metrics)
            if rhs.passed:
                return EvalResult(True, f"Until satisfied: right side held at position {i}.", f)
            lhs = self.evaluate(f.left, trace, i, metrics=metrics)
            if not lhs.passed:
                return EvalResult(
                    False,
                    f"Until violated at position {i}: left side failed before right side held. Left: {lhs.detail}",
                    f,
                )
        return EvalResult(False, f"Until violated: right side never held (trace ended at position {len(trace)}).", f)

    def _eval_weak_until(self, f: WeakUntil, trace: Trace, pos: int, metrics: Optional[Dict[str, Any]] = None) -> EvalResult:
        for i in range(pos, len(trace)):
            rhs = self.evaluate(f.right, trace, i, metrics=metrics)
            if rhs.passed:
                return EvalResult(True, f"WeakUntil satisfied: right side held at position {i}.", f)
            lhs = self.evaluate(f.left, trace, i, metrics=metrics)
            if not lhs.passed:
                return EvalResult(False, f"WeakUntil violated at position {i}: left side failed. {lhs.detail}", f)
        return EvalResult(True, "WeakUntil satisfied: left side held at all remaining positions.", f)

    def _eval_release(self, f: Release, trace: Trace, pos: int, metrics: Optional[Dict[str, Any]] = None) -> EvalResult:
        for i in range(pos, len(trace)):
            rhs = self.evaluate(f.right, trace, i, metrics=metrics)
            lhs = self.evaluate(f.left, trace, i, metrics=metrics)
            if lhs.passed:
                if rhs.passed:
                    return EvalResult(True, f"Release satisfied: both sides held at position {i}.", f)
                else:
                    return EvalResult(
                        False,
                        f"Release violated at position {i}: left held but right failed. {rhs.detail}",
                        f,
                    )
            if not rhs.passed:
                return EvalResult(
                    False,
                    f"Release violated at position {i}: right failed before left held. {rhs.detail}",
                    f,
                )
        return EvalResult(True, "Release satisfied: right side held at all remaining positions.", f)

    # ─────────────────────────────────────────────────────────────────────────
    # First-order quantifiers
    # ─────────────────────────────────────────────────────────────────────────

    def _eval_forall(self, f: ForAll, trace: Trace, pos: int, metrics: Optional[Dict[str, Any]]) -> EvalResult:
        try:
            entities = call_domain(f.domain, trace, metrics, pos)
        except Exception as exc:
            return EvalResult(False, f"∀{f.var}: domain extractor raised {type(exc).__name__}: {exc}", f)

        if not entities:
            return EvalResult(True, f"∀{f.var}: vacuously true (empty domain).", f)

        for entity in entities:
            concrete = substitute(f.body, {f.var: entity})
            result = self.evaluate(concrete, trace, pos, metrics=metrics)
            if not result.passed:
                return EvalResult(False, f"∀{f.var}: failed for {f.var}={entity!r}. {result.detail}", f)
        return EvalResult(True, f"∀{f.var}: holds for all {len(entities)} entities.", f)

    def _eval_exists(self, f: Exists, trace: Trace, pos: int, metrics: Optional[Dict[str, Any]]) -> EvalResult:
        try:
            entities = call_domain(f.domain, trace, metrics, pos)
        except Exception as exc:
            return EvalResult(False, f"∃{f.var}: domain extractor raised {type(exc).__name__}: {exc}", f)

        if not entities:
            return EvalResult(False, f"∃{f.var}: false (empty domain).", f)

        for entity in entities:
            concrete = substitute(f.body, {f.var: entity})
            result = self.evaluate(concrete, trace, pos, metrics=metrics)
            if result.passed:
                return EvalResult(True, f"∃{f.var}: satisfied for {f.var}={entity!r}. {result.detail}", f)
        return EvalResult(False, f"∃{f.var}: no entity satisfied the formula ({len(entities)} checked).", f)


_PARTIAL = PartialEvaluator()
