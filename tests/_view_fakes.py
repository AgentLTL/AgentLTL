# -*- coding: utf-8 -*-
"""Test-only stand-ins for the harness's task types. They exist to show that
agentltl.translation depends on the STRUCTURE of a view (see translation._protocols), not on
any benchmark package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class ToolSchema:
    name: str
    parameters: List[str]
    required: List[str] = field(default_factory=list)
    cls: str = ""
    # {leaf path: the one value the SCHEMA prescribes}. Agent-visible -- it is in
    # the tool list the agent is handed -- and the only source for constants the
    # task text never mentions but the environment requires (status="final",
    # intent="order"). Empty for every family whose extractor does not emit it.
    prescribed: Dict[str, Any] = field(default_factory=dict)
    # {parameter name: its JSON-Schema block} -- type, description, enum, default.
    # `parameters` keeps only the NAMES, and the names say nothing about what a
    # value may be. This is the agent-visible source that states `travel_from`
    # takes "The 3 letter code of the departing airport" and that `ls.a` is a
    # boolean, and agentltl.paramspec is the only reader. Empty for any family
    # whose extractor does not emit it, which leaves that family's behaviour
    # unchanged -- deliberately, because mab is where blocking already wins.
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def param_schema(self, path: str) -> Dict[str, Any]:
        """The schema block for a parameter, addressed by name or by leaf path.

        FHIR states values three levels down (`subject.reference`,
        `category[].coding[].code`), so a caller holding a leaf path must still
        find the top-level parameter. Returns {} when nothing is declared, which
        paramspec reads as "no evidence either way".
        """
        if not self.params:
            return {}
        if path in self.params:
            return self.params[path]
        head = str(path).split(".", 1)[0].replace("[]", "")
        return self.params.get(head) or {}

    def sig(self) -> str:
        return f"{self.name}({', '.join(self.parameters)})"


@dataclass(frozen=True)
class UserTurn:
    turn: int
    messages: List[str]


@dataclass(frozen=True)
class AgentView:
    """EVERYTHING generation may see. There is no gold field, by construction."""
    instance_id: str
    benchmark: str
    turns: List[UserTurn]
    tools: List[ToolSchema]
    policy: Optional[str] = None
    clock: Optional[str] = None
    # "verbatim"      -> turns are literally what the agent reads (BFCL, WorkBench)
    # "specification" -> turns are the task spec behind a user simulator (tau-bench)
    visibility: str = "verbatim"

    def task_text(self) -> str:
        return " ".join(m for t in self.turns for m in t.messages)

    def tool_map(self) -> Dict[str, ToolSchema]:
        return {t.name: t for t in self.tools}


