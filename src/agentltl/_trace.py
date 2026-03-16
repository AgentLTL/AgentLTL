"""
agentltl/_trace.py – Trace representation for FOLTL evaluation.

A :class:`Trace` wraps the ordered sequence of tool calls produced by an
agent run and exposes helper methods used during formula evaluation.

The canonical input is ``metrics["tool_calls"]``, a list of dicts with at
least a ``"tool_name"`` key.

Example
-------
>>> from agentltl import Trace
>>> trace = Trace.from_metrics({"tool_calls": [
...     {"tool_name": "fetch"},
...     {"tool_name": "clean"},
...     {"tool_name": "summarize"},
... ]})
>>> trace.events
['fetch', 'clean', 'summarize']
>>> trace.contains("clean")
True
>>> trace.first_index("fetch")
0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ToolCall:
    """A single tool invocation recorded in the trace.

    Attributes:
        name:     The tool name string.
        args:     The arguments dict passed to the tool (may be empty).
        position: 0-based index in the trace.
        raw:      The original dict from ``metrics["tool_calls"]``, preserved
                  verbatim so that downstream predicates can inspect any key.
        result:   The return value of the tool call, if recorded.  Populated
                  from the ``"tool_result"`` (or ``"result"``) key in the raw
                  dict.  ``None`` when not provided.
    """
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    position: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)
    result: Any = None


@dataclass
class Trace:
    """Ordered sequence of tool calls produced by an agent run.

    Implements the *trace* τ = [t₁, t₂, …, tₙ] over which LTL formulas
    are evaluated.
    """

    calls: List[ToolCall] = field(default_factory=list)

    # ── Construction helpers ─────────────────────────────────────────────────

    @classmethod
    def from_metrics(cls, metrics: Dict[str, Any]) -> "Trace":
        """Build a Trace from the ``metrics`` dict returned by ``Agent.run()``.

        Expects ``metrics["tool_calls"]`` to be a list of dicts, each with at
        least a ``"tool_name"`` key.
        """
        raw_calls = metrics.get("tool_calls", [])
        calls = []
        for i, tc in enumerate(raw_calls):
            name = tc.get("tool_name", tc.get("name", "unknown"))
            args = tc.get("tool_args", tc.get("arguments", {}))
            if args is None:
                args = {}
            result = tc.get("tool_result", tc.get("result", None))
            calls.append(ToolCall(name=name, args=args, position=i, raw=tc, result=result))
        return cls(calls=calls)

    @classmethod
    def from_names(cls, names: Sequence[str]) -> "Trace":
        """Build a Trace from a plain list of tool name strings (for tests)."""
        calls = [ToolCall(name=n, position=i) for i, n in enumerate(names)]
        return cls(calls=calls)

    # ── Query helpers ────────────────────────────────────────────────────────

    @property
    def events(self) -> List[str]:
        """Return the flat list of tool names in call order."""
        return [c.name for c in self.calls]

    def __len__(self) -> int:
        return len(self.calls)

    def __getitem__(self, index: int) -> ToolCall:
        return self.calls[index]

    def contains(self, tool: str) -> bool:
        """True iff *tool* appears at least once."""
        return any(c.name == tool for c in self.calls)

    def count(self, tool: str) -> int:
        """Number of times *tool* appears."""
        return sum(1 for c in self.calls if c.name == tool)

    def first_index(self, tool: str) -> int:
        """0-based index of the first occurrence of *tool*, or -1."""
        for c in self.calls:
            if c.name == tool:
                return c.position
        return -1

    def last_index(self, tool: str) -> int:
        """0-based index of the last occurrence of *tool*, or -1."""
        for c in reversed(self.calls):
            if c.name == tool:
                return c.position
        return -1

    def nth_index(self, tool: str, n: int) -> int:
        """0-based index of the *n*-th occurrence (1-based *n*) of *tool*, or -1."""
        count = 0
        for c in self.calls:
            if c.name == tool:
                count += 1
                if count == n:
                    return c.position
        return -1

    def all_indices(self, tool: str) -> List[int]:
        """All 0-based indices where *tool* appears."""
        return [c.position for c in self.calls if c.name == tool]

    def calls_with(self, tool: str, expected_args: Dict[str, Any]) -> List[ToolCall]:
        """Return all calls to *tool* whose args are a superset of *expected_args*."""
        results = []
        for c in self.calls:
            if c.name != tool:
                continue
            if all(c.args.get(k) == v for k, v in expected_args.items()):
                results.append(c)
        return results

    def suffix(self, position: int) -> "Trace":
        """Return the sub-trace starting at *position* (inclusive).

        Used internally by the LTL evaluator for temporal operators.
        The new trace preserves original positions.
        """
        return Trace(calls=[c for c in self.calls if c.position >= position])

    def at(self, position: int) -> Optional[ToolCall]:
        """Return the call at *position*, or None."""
        for c in self.calls:
            if c.position == position:
                return c
        return None

    @property
    def is_empty(self) -> bool:
        return len(self.calls) == 0
