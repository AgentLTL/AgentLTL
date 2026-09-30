# -*- coding: utf-8 -*-
"""Structural types for what open predicates read from a task.

agentltl does not own the task or graph types -- the benchmark harness does -- so the
interview and validator take anything shaped like these. Only the attributes they
actually read are listed; a richer object satisfies them unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class ToolLike(Protocol):
    name: str
    parameters: Sequence[str]

    def param_schema(self, path: str) -> Dict[str, Any]: ...


@runtime_checkable
class ViewLike(Protocol):
    """What the agent could see: the task, the policy, the clock, the tools."""
    tools: Sequence[ToolLike]
    policy: Optional[str]
    clock: Optional[str]

    def task_text(self) -> str: ...
    def tool_map(self) -> Dict[str, ToolLike]: ...


@runtime_checkable
class NodeLike(Protocol):
    id: str
    tool: str
    role: str
    step: str
    span: str

    def bound(self) -> bool: ...


@runtime_checkable
class GraphLike(Protocol):
    nodes: List[NodeLike]

    def node(self, node_id: str) -> Any: ...
