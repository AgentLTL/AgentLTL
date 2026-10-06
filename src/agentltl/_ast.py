"""
agentltl/_ast.py – Abstract Syntax Tree for First-Order Linear Temporal Logic formulas.

This module defines the node types that compose an FOLTL formula AST.
It extends propositional LTL with first-order quantifiers (∀, ∃) over
dynamically-extracted entity domains.

Every node is an immutable (frozen) dataclass that inherits from :class:`Formula`.

Supported constructs
--------------------

**Atomic propositions** (leaf nodes)
  * ``Called(tool)``         – tool appears at least once in the trace
  * ``CalledWith(tool, args)`` – tool called with specific argument values
  * ``CalledNTimes(tool, n, op)`` – tool called exactly / at least / at most *n* times
  * ``Before(a, b)``        – first occurrence of *a* precedes first occurrence of *b*
  * ``After(a, b)``         – first occurrence of *a* follows first occurrence of *b*
  * ``AllBefore(tools, b)`` – *every* tool in *tools* precedes *b* (multi-input gate)
  * ``BranchCalled(tool, wrong_tool, context)``
                             – correct branch tool called (not wrong one)

**Temporal operators** (unary / binary)
  * ``Globally(φ)``        – G φ   (φ holds at every position)
  * ``Eventually(φ)``      – F φ   (φ holds at some position)
  * ``Next(φ)``            – X φ   (φ holds at the next position)
  * ``Until(φ, ψ)``        – φ U ψ (φ holds until ψ holds)
  * ``WeakUntil(φ, ψ)``    – φ W ψ (φ holds until ψ, or forever)
  * ``Release(φ, ψ)``      – φ R ψ (dual of Until)

**Logical connectives**
  * ``Not(φ)``             – ¬ φ
  * ``And(φ, ψ)``          – φ ∧ ψ
  * ``Or(φ, ψ)``           – φ ∨ ψ
  * ``Implies(φ, ψ)``      – φ → ψ

**Special / convenience**
  * ``Predicate(fn, desc)`` – arbitrary Python callable evaluated with trace context
  * ``AtPosition(index, φ)``– φ holds at a specific trace position

**First-order quantifiers** (FOLTL extension)
  * ``Var(name)``           – variable placeholder, bound by ForAll / Exists
  * ``ForAll(var, domain, body)`` – ∀x ∈ D. φ(x)
  * ``Exists(var, domain, body)`` – ∃x ∈ D. φ(x)

**Substitution**
  * ``substitute(formula, bindings)`` – replace Var references with concrete values

All nodes support ``__str__`` for human-readable syntax export and
``__repr__`` for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Formula:
    """Abstract base for every LTL AST node."""

    def __and__(self, other: "Formula") -> "And":
        return And(self, other)

    def __or__(self, other: "Formula") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def __rshift__(self, other: "Formula") -> "Implies":
        """``φ >> ψ`` is syntactic sugar for ``φ → ψ``."""
        return Implies(self, other)


# ─────────────────────────────────────────────────────────────────────────────
# Atomic propositions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Called(Formula):
    """True iff *tool* appears at least once in the trace."""
    tool: str

    def __str__(self) -> str:
        return f'called("{self.tool}")'


@dataclass(frozen=True)
class Now(Formula):
    """True iff the call AT THE CURRENT POSITION is *tool*.

    The step-local counterpart of :class:`Called`, which is a fact about the
    whole trace (``called("x")`` holds at every position once x appears
    anywhere). Inside temporal operators that is rarely what a rule means:
    ``G(called("deploy") -> X(...))`` fires at every step of a run that
    deploys once. ``G(now("deploy") -> X(...))`` fires at the deploy step only.
    """
    tool: str

    def __str__(self) -> str:
        return f'now("{self.tool}")'


@dataclass(frozen=True)
class CalledWith(Formula):
    """True iff *tool* was called with arguments matching *expected_args*.

    *expected_args* is a dict; the check passes when every key in
    *expected_args* is present in the actual call arguments and the values
    are equal.  Extra keys in the actual call are allowed.
    """
    tool: str
    expected_args: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.expected_args.items())
        return f'called_with("{self.tool}", {args})'


@dataclass(frozen=True)
class CalledNTimes(Formula):
    """True iff the number of calls to *tool* satisfies the comparison.

    *op* is one of ``"=="``, ``">="``, ``"<="``, ``">"``, ``"<"``.
    """
    tool: str
    n: int
    op: str = "=="  # ==, >=, <=, >, <

    def __str__(self) -> str:
        return f'called_n("{self.tool}", {self.op} {self.n})'


@dataclass(frozen=True)
class Before(Formula):
    """True iff the first occurrence of *a* precedes the first occurrence of *b*."""
    a: str
    b: str

    def __str__(self) -> str:
        return f'before("{self.a}", "{self.b}")'


@dataclass(frozen=True)
class After(Formula):
    """True iff the first occurrence of *a* follows the first occurrence of *b*."""
    a: str
    b: str

    def __str__(self) -> str:
        return f'after("{self.a}", "{self.b}")'


@dataclass(frozen=True)
class AllBefore(Formula):
    """True iff every tool in *tools* appears before the first occurrence of *target*.

    This captures a multi-input gate constraint: several parallel steps must
    all complete before a convergence point.
    """
    tools: Tuple[str, ...]
    target: str

    def __init__(self, tools: Sequence[str], target: str):
        object.__setattr__(self, "tools", tuple(tools))
        object.__setattr__(self, "target", target)

    def __str__(self) -> str:
        names = ", ".join(f'"{t}"' for t in self.tools)
        return f'all_before([{names}], "{self.target}")'


@dataclass(frozen=True)
class BranchCalled(Formula):
    """True iff the *correct_tool* was called (and optionally *wrong_tool* was NOT called).

    Captures branch / conditional constraints where the agent must choose the
    correct tool path.  *context* is a human-readable description of the
    branch condition (e.g. ``"risk_score > 0.7"``).
    """
    correct_tool: str
    wrong_tool: Optional[str] = None
    context: str = ""

    def __str__(self) -> str:
        base = f'branch_called("{self.correct_tool}"'
        if self.wrong_tool:
            base += f', wrong="{self.wrong_tool}"'
        if self.context:
            base += f', ctx="{self.context}"'
        return base + ")"


@dataclass(frozen=True)
class CalledWithResult(Formula):
    """True iff *tool* was called and its return value matches *expected_result*.

    *expected_args* optionally restricts which call instances are considered
    (same subset-match semantics as :class:`CalledWith`).

    *expected_result* may be:

    * a plain value — checked for equality against the recorded result,
    * a ``dict`` — checked as a subset of the result dict (partial match),
    * a :class:`Var` — bound by an enclosing quantifier.

    The result is read from the ``"tool_result"`` (or ``"result"``) key in the
    raw ``metrics["tool_calls"]`` entry.
    """
    tool: str
    expected_result: Any
    expected_args: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.expected_args.items())
        arg_str = f", {args}" if args else ""
        return f'called_with_result("{self.tool}"{arg_str}, result={self.expected_result!r})'


@dataclass(frozen=True)
class InstanceBefore(Formula):
    """True iff the *n*-th occurrence of *tool_a* precedes the *m*-th occurrence of *tool_b*.

    Both *n* and *m* are **1-based**.  If either tool has fewer occurrences than
    required the constraint fails with a clear message.

    Example::

        InstanceBefore("read_file", 2, "write_file", 1)
        # "the second read_file happened before the first write_file"
    """
    tool_a: str
    n: int
    tool_b: str
    m: int

    def __str__(self) -> str:
        return f'instance_before("{self.tool_a}"[{self.n}], "{self.tool_b}"[{self.m}])'


@dataclass(frozen=True)
class CalledInOrder(Formula):
    """True iff *tools* appear as a **subsequence** of the trace in the given order.

    The tools need not be consecutive — only their relative order matters.
    Duplicate tool names are allowed; each match consumes one occurrence.

    Example::

        CalledInOrder(["validate", "process", "commit"])
        # validate appears somewhere before process, which appears before commit
    """
    tools: Tuple[str, ...]

    def __init__(self, tools: Sequence[str]):
        object.__setattr__(self, "tools", tuple(tools))

    def __str__(self) -> str:
        names = ", ".join(f'"{t}"' for t in self.tools)
        return f"in_order([{names}])"


@dataclass(frozen=True)
class WithinSteps(Formula):
    """True iff *tool_b* first occurs within *n* steps **after** the first occurrence of *tool_a*.

    Both tools must be present in the trace, and *tool_b* must appear at or
    after *tool_a*.  The distance is ``position(b) − position(a)``; the
    constraint passes when ``distance <= n``.

    Example::

        WithinSteps("alert", "acknowledge", 3)
        # acknowledge must be called within 3 steps of alert
    """
    tool_a: str
    tool_b: str
    n: int

    def __str__(self) -> str:
        return f'within_steps("{self.tool_a}", "{self.tool_b}", {self.n})'


# ─────────────────────────────────────────────────────────────────────────────
# Temporal operators
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Globally(Formula):
    """G φ  –  φ must hold at *every* position of the trace."""
    operand: Formula

    def __str__(self) -> str:
        return f"G({self.operand})"


@dataclass(frozen=True)
class Eventually(Formula):
    """F φ  –  φ must hold at *some* position of the trace."""
    operand: Formula

    def __str__(self) -> str:
        return f"F({self.operand})"


@dataclass(frozen=True)
class Next(Formula):
    """X φ  –  φ must hold at the *next* position."""
    operand: Formula

    def __str__(self) -> str:
        return f"X({self.operand})"


@dataclass(frozen=True)
class Until(Formula):
    """φ U ψ  –  φ holds at every position until ψ becomes true (strong until)."""
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left}) U ({self.right})"


@dataclass(frozen=True)
class WeakUntil(Formula):
    """φ W ψ  –  like Until but φ may hold forever without ψ ever occurring."""
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left}) W ({self.right})"


@dataclass(frozen=True)
class Release(Formula):
    """φ R ψ  –  ψ must hold until (and including) the first position where φ holds."""
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left}) R ({self.right})"


# ─────────────────────────────────────────────────────────────────────────────
# Logical connectives
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Not(Formula):
    """¬ φ"""
    operand: Formula

    def __str__(self) -> str:
        return f"¬({self.operand})"


@dataclass(frozen=True)
class And(Formula):
    """φ ∧ ψ"""
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left}) ∧ ({self.right})"


@dataclass(frozen=True)
class Or(Formula):
    """φ ∨ ψ"""
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left}) ∨ ({self.right})"


@dataclass(frozen=True)
class Implies(Formula):
    """φ → ψ"""
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left}) → ({self.right})"


# ─────────────────────────────────────────────────────────────────────────────
# Special / convenience
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Predicate(Formula):
    """Arbitrary predicate evaluated against the full trace context.

    *fn* receives ``(trace: Trace, position: int)`` and must return ``bool``.
    *description* is a human-readable label used in reports and ``__str__``.

    *runtime_safe* is consulted by the static runtime-safety classifier in
    :mod:`agentltl.runtime_safety`.  Set it to ``True`` to declare that
    a violation of *fn* is detectable at any finite prefix (a safety
    property) or ``False`` to declare that the property is unbounded
    liveness.  ``None`` (the default) means the framework defers to the
    user; the classifier reports ``AMBIGUOUS`` and the registration-time
    hook stays silent.

    Because callables are not hashable in general, equality falls back to
    identity comparison.
    """
    fn: Callable  # (Trace, int) -> bool
    description: str = "predicate"
    runtime_safe: Optional[bool] = None

    def __hash__(self) -> int:
        return id(self.fn)

    def __str__(self) -> str:
        return f'predicate("{self.description}")'


@dataclass(frozen=True)
class AtPosition(Formula):
    """φ holds at a specific *index* (0-based) in the trace."""
    index: int
    operand: Formula

    def __str__(self) -> str:
        return f"@{self.index}({self.operand})"


# ─────────────────────────────────────────────────────────────────────────────
# Past-time operators
# ─────────────────────────────────────────────────────────────────────────────
#
# They look back from the current position, so on a growing trace their value at a
# position is settled as soon as that position exists. A rule of the shape
# G(<past-only formula>) -- "every call that ... must have ... before it" -- is therefore
# judged on the newest call alone (see agentltl._partial.is_past_only).

@dataclass(frozen=True)
class Previous(Formula):
    """Y φ  –  φ held at the previous position (false at the first one)."""
    operand: Formula

    def __str__(self) -> str:
        return f"Y({self.operand})"


@dataclass(frozen=True)
class Once(Formula):
    """O φ  –  φ held at some position up to and including this one."""
    operand: Formula

    def __str__(self) -> str:
        return f"O({self.operand})"


@dataclass(frozen=True)
class Historically(Formula):
    """H φ  –  φ held at every position up to and including this one."""
    operand: Formula

    def __str__(self) -> str:
        return f"H({self.operand})"


@dataclass(frozen=True)
class Since(Formula):
    """φ S ψ  –  ψ held at some position j up to this one, and φ at every position after j."""
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} S {self.right})"


@dataclass(frozen=True)
class CountBefore(Formula):
    """The number of positions before this one where *operand* held satisfies ``op n``."""
    operand: Formula
    n: int
    op: str = "<"  # ==, >=, <=, >, <

    def __str__(self) -> str:
        return f"count_before({self.operand}, {self.op} {self.n})"


# ─────────────────────────────────────────────────────────────────────────────
# Call patterns
# ─────────────────────────────────────────────────────────────────────────────

class CallPattern:
    """What :class:`Matches` asks about a call. Harnesses supply their own (globs, paths,
    files that exist); :class:`ToolPattern` is the plain one.

    ``match(name, args)`` returns True, False, or None for "maybe": the call's arguments
    are only known when it runs (``xargs rm``). ``bind(bindings)`` returns the pattern
    with quantifier variables (:class:`Var` values) replaced. ``tools`` names the tools it
    can match, for linting.
    """

    tools: Tuple[str, ...] = ()

    def match(self, name: str, args: Dict[str, Any]) -> Optional[bool]:
        raise NotImplementedError

    def bind(self, bindings: Dict[str, Any]) -> "CallPattern":
        return self

    def describe(self) -> str:
        return " or ".join(self.tools) or "a call"


class ToolPattern(CallPattern):
    """A call to one of *tools* whose arguments include *args* (equal values; a
    :class:`Var` value is bound by an enclosing quantifier)."""

    def __init__(self, tools: Union[str, Sequence[str]], args: Optional[Dict[str, Any]] = None):
        self.tools = (tools,) if isinstance(tools, str) else tuple(tools)
        self.args = dict(args or {})

    def match(self, name: str, args: Dict[str, Any]) -> Optional[bool]:
        return name in self.tools and all(args.get(k) == v for k, v in self.args.items())

    def bind(self, bindings: Dict[str, Any]) -> "ToolPattern":
        return ToolPattern(self.tools, {k: bindings.get(v.name, v) if isinstance(v, Var) else v
                                        for k, v in self.args.items()})

    def describe(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return " or ".join(self.tools) + (f" ({args})" if args else "")

    def __repr__(self) -> str:
        return f"ToolPattern({self.tools!r}, {self.args!r})"


@dataclass(frozen=True)
class Matches(Formula):
    """The call at the current position matches *pattern*.

    *maybe* says how a call that only may match (the pattern answered None) counts: True
    where counting it as a match is the cautious reading (a prohibited call), False where
    not counting it is (a call that would satisfy an obligation). Either way the failure
    it causes is marked as resting on a possible match.
    """
    pattern: Any
    maybe: bool = False

    def __hash__(self) -> int:
        return id(self.pattern) ^ hash(self.maybe)

    def __str__(self) -> str:
        describe = getattr(self.pattern, "describe", None)
        text = describe() if describe else repr(self.pattern)
        return f"matches({text})"


# ─────────────────────────────────────────────────────────────────────────────
# Convenience constructors (thin wrappers for common patterns)
# ─────────────────────────────────────────────────────────────────────────────

def called(tool: str) -> Called:
    """Shorthand: ``called("tool_name")``."""
    return Called(tool)


def before(a: str, b: str) -> Before:
    """Shorthand: ``before("a", "b")``."""
    return Before(a, b)


def after(a: str, b: str) -> After:
    """Shorthand: ``after("a", "b")``."""
    return After(a, b)


def eventually(tool_or_formula: Union[str, Formula]) -> Eventually:
    """``F(called("tool"))`` when given a string, ``F(φ)`` otherwise."""
    if isinstance(tool_or_formula, str):
        return Eventually(Called(tool_or_formula))
    return Eventually(tool_or_formula)


def always(tool_or_formula: Union[str, Formula]) -> Globally:
    """``G(called("tool"))`` when given a string, ``G(φ)`` otherwise."""
    if isinstance(tool_or_formula, str):
        return Globally(Called(tool_or_formula))
    return Globally(tool_or_formula)


# ─────────────────────────────────────────────────────────────────────────────
# First-order quantifiers (FOLTL extension)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Var:
    """Variable reference, bound by a ForAll or Exists quantifier.

    Appears inside ``CalledWith.expected_args`` values.  NOT a Formula —
    it is a placeholder resolved by :func:`substitute` before evaluation.
    """
    name: str

    def __str__(self) -> str:
        return f"?{self.name}"


@dataclass(frozen=True)
class ForAll(Formula):
    """∀x ∈ domain. body(x)

    True iff *body* holds for every entity returned by *domain*.
    Empty domain → vacuously true (standard FOL semantics).

    Parameters:
        var:         Name of the bound variable.
        domain:      ``Callable[[Trace, dict], list]`` — entity extractor.
        body:        Formula template containing ``Var(var)`` references.
        description: Human-readable label for the domain (used in ``__str__``).
    """
    var: str
    domain: Any  # Callable[[Trace, Dict], List[Any]]
    body: Formula
    description: str = ""

    def __hash__(self) -> int:
        return id(self.domain) ^ hash((self.var, self.body))

    def __str__(self) -> str:
        desc = self.description or "D"
        return f"∀{self.var}∈{desc}. ({self.body})"


@dataclass(frozen=True)
class Exists(Formula):
    """∃x ∈ domain. body(x)

    True iff *body* holds for at least one entity returned by *domain*.
    Empty domain → false (standard FOL semantics).

    Parameters:
        var:         Name of the bound variable.
        domain:      ``Callable[[Trace, dict], list]`` — entity extractor.
        body:        Formula template containing ``Var(var)`` references.
        description: Human-readable label for the domain (used in ``__str__``).
    """
    var: str
    domain: Any  # Callable[[Trace, Dict], List[Any]]
    body: Formula
    description: str = ""

    def __hash__(self) -> int:
        return id(self.domain) ^ hash((self.var, self.body))

    def __str__(self) -> str:
        desc = self.description or "D"
        return f"∃{self.var}∈{desc}. ({self.body})"


# ─────────────────────────────────────────────────────────────────────────────
# Substitution
# ─────────────────────────────────────────────────────────────────────────────

def substitute(formula: Formula, bindings: Dict[str, Any]) -> Formula:
    """Replace all :class:`Var` references in *formula* with concrete values.

    Walks the AST recursively.  Only ``CalledWith.expected_args`` values are
    checked for ``Var`` instances (variables don't appear in tool-name positions).

    Quantifier scoping is respected: if an inner ``ForAll`` / ``Exists`` binds
    the same variable name, that inner scope shadows the outer binding.

    Raises:
        ValueError: If a standalone ``Var`` is encountered outside of a
            ``CalledWith.expected_args`` value.
    """
    if not bindings:
        return formula

    # ── Atomic propositions ───────────────────────────────────────────────

    if isinstance(formula, CalledWith):
        new_args = {}
        for k, v in formula.expected_args.items():
            if isinstance(v, Var):
                if v.name in bindings:
                    new_args[k] = bindings[v.name]
                else:
                    new_args[k] = v
            else:
                new_args[k] = v
        return CalledWith(tool=formula.tool, expected_args=new_args)

    if isinstance(formula, CalledWithResult):
        # Substitute Var in expected_args
        new_args = {}
        for k, v in formula.expected_args.items():
            if isinstance(v, Var) and v.name in bindings:
                new_args[k] = bindings[v.name]
            else:
                new_args[k] = v
        # Substitute Var in expected_result (plain Var or dict of Vars)
        new_result = formula.expected_result
        if isinstance(new_result, Var):
            new_result = bindings.get(new_result.name, new_result)
        elif isinstance(new_result, dict):
            new_result = {
                k: (bindings[v.name] if isinstance(v, Var) and v.name in bindings else v)
                for k, v in new_result.items()
            }
        return CalledWithResult(
            tool=formula.tool,
            expected_result=new_result,
            expected_args=new_args,
        )

    if isinstance(formula, Matches):
        bind = getattr(formula.pattern, "bind", None)
        return Matches(bind(bindings), formula.maybe) if bind else formula

    if isinstance(formula, (Previous, Once, Historically)):
        return type(formula)(substitute(formula.operand, bindings))

    if isinstance(formula, Since):
        return Since(substitute(formula.left, bindings), substitute(formula.right, bindings))

    if isinstance(formula, CountBefore):
        return CountBefore(substitute(formula.operand, bindings), formula.n, formula.op)

    if isinstance(formula, (Called, Now, CalledNTimes, Before, After, AllBefore, BranchCalled,
                             InstanceBefore, CalledInOrder, WithinSteps)):
        return formula  # no Var positions in these atomics

    # ── Special ───────────────────────────────────────────────────────────

    if isinstance(formula, Predicate):
        # Wrap the fn to inject bindings so user-defined predicates can
        # access the bound variable values via closure.
        # The evaluator passes metrics as a third argument; under a quantifier the
        # predicate receives the bindings there instead.
        captured_bindings = dict(bindings)
        original_fn = formula.fn
        def wrapped_fn(trace, position, metrics=None, _b=captured_bindings, _fn=original_fn):
            return _fn(trace, position, _b)
        return Predicate(
            fn=wrapped_fn,
            description=formula.description,
            runtime_safe=formula.runtime_safe,
        )

    if isinstance(formula, AtPosition):
        return AtPosition(index=formula.index, operand=substitute(formula.operand, bindings))

    # ── Logical connectives ───────────────────────────────────────────────

    if isinstance(formula, Not):
        return Not(substitute(formula.operand, bindings))

    if isinstance(formula, And):
        return And(substitute(formula.left, bindings), substitute(formula.right, bindings))

    if isinstance(formula, Or):
        return Or(substitute(formula.left, bindings), substitute(formula.right, bindings))

    if isinstance(formula, Implies):
        return Implies(substitute(formula.left, bindings), substitute(formula.right, bindings))

    # ── Temporal operators ────────────────────────────────────────────────

    if isinstance(formula, Globally):
        return Globally(substitute(formula.operand, bindings))

    if isinstance(formula, Eventually):
        return Eventually(substitute(formula.operand, bindings))

    if isinstance(formula, Next):
        return Next(substitute(formula.operand, bindings))

    if isinstance(formula, Until):
        return Until(substitute(formula.left, bindings), substitute(formula.right, bindings))

    if isinstance(formula, WeakUntil):
        return WeakUntil(substitute(formula.left, bindings), substitute(formula.right, bindings))

    if isinstance(formula, Release):
        return Release(substitute(formula.left, bindings), substitute(formula.right, bindings))

    # ── First-order quantifiers ───────────────────────────────────────────

    if isinstance(formula, (ForAll, Exists)):
        # Shadow: if this quantifier binds a variable that's also in bindings,
        # remove it from the bindings passed to the body.
        inner_bindings = bindings
        if formula.var in bindings:
            inner_bindings = {k: v for k, v in bindings.items() if k != formula.var}
        new_body = substitute(formula.body, inner_bindings)
        if isinstance(formula, ForAll):
            return ForAll(var=formula.var, domain=formula.domain,
                          body=new_body, description=formula.description)
        return Exists(var=formula.var, domain=formula.domain,
                      body=new_body, description=formula.description)

    raise TypeError(f"substitute: unknown formula type {type(formula).__name__}")
