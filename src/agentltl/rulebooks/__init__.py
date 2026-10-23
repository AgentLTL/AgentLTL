"""
agentltl.rulebooks – load constraint sets written as YAML files.

A rulebook is a YAML file::

    name: devops
    description: 'Deployments and destructive operations.'
    rules:
      - name: tests_before_deploy
        formula: 'before("run_tests", "deploy_prod")'   # the syntax of agentltl.parse
        severity: HARD_STOP                              # a ConstraintSeverity name
        weight: 3                                        # optional, default 1
        description: 'The test suite runs before any production deployment.'
        repair: 'Run run_tests first, then deploy.'      # shown to the agent
        applies_to_final_answer: false                   # optional

Quote formulas in single quotes: a plain YAML scalar cannot start with ``!`` or ``|``.

The rulebooks bundled here (``available()``) use placeholder tool names: copy one and
rename its tools to your agent's. ``load`` takes bundled names and file paths alike, and
merges what it is given::

    from agentltl import rulebooks, verify_trace

    book = rulebooks.load("devops", "payments", "my_rules.yaml")
    enforcer = book.enforcer(max_termination_nudges=1)
    result = verify_trace(metrics, book.constraints)

Needs PyYAML (``pip install "agentltl[yaml]"``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Dict, List, Union

from .._constraints import Constraint
from .._parser import parse
from ..enforcement import ConstraintSeverity
from ..enforcer import Enforcer

_RULE_KEYS = {"name", "formula", "severity", "weight", "description", "repair",
              "applies_to_final_answer"}
_BOOK_KEYS = {"name", "description", "rules"}


@dataclass
class Rulebook:
    """Constraints and their severities, as :class:`~agentltl.Enforcer` takes them."""

    name: str
    description: str = ""
    constraints: List[Constraint] = field(default_factory=list)
    severities: Dict[str, ConstraintSeverity] = field(default_factory=dict)

    def enforcer(self, **kwargs: Any) -> Enforcer:
        return Enforcer(self.constraints, self.severities, **kwargs)


def available() -> List[str]:
    """The names of the bundled rulebooks."""
    return sorted(p.name[:-5] for p in resources.files(__name__).iterdir()
                  if p.name.endswith(".yaml"))


def load(*sources: Union[str, os.PathLike]) -> Rulebook:
    """Load rulebooks (bundled names or paths to YAML files) and merge them.

    Raises:
        ValueError: A file is malformed, a formula does not parse, a severity is unknown,
            or two rules share a name.
    """
    if not sources:
        raise ValueError("load() needs at least one rulebook name or path")
    books = [_load_one(s) for s in sources]
    merged = Rulebook("+".join(b.name for b in books),
                      " ".join(b.description for b in books if b.description))
    for book in books:
        for c in book.constraints:
            if c.name in merged.severities:
                raise ValueError(f"Rule {c.name!r} is defined twice (again in {book.name!r})")
            merged.constraints.append(c)
            merged.severities[c.name] = book.severities[c.name]
    return merged


def _load_one(source: Union[str, os.PathLike]) -> Rulebook:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError('Rulebooks need PyYAML: pip install "agentltl[yaml]"') from exc

    path = os.fspath(source)
    if path in available():
        text = resources.files(__name__).joinpath(f"{path}.yaml").read_text(encoding="utf-8")
    elif os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"No rulebook {path!r}: not a file, nor one of {available()}")
    return _build(yaml.safe_load(text), path)


def _build(data: Any, where: str) -> Rulebook:
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise ValueError(f"{where}: a rulebook is a mapping with a 'rules' list")
    _no_unknown_keys(data, _BOOK_KEYS, where)
    book = Rulebook(str(data.get("name") or where), str(data.get("description") or ""))
    for i, rule in enumerate(data["rules"]):
        at = f"{where}: rule {rule.get('name', i + 1) if isinstance(rule, dict) else i + 1}"
        if not isinstance(rule, dict) or not rule.get("name") or not rule.get("formula"):
            raise ValueError(f"{at}: a rule needs a 'name' and a 'formula'")
        _no_unknown_keys(rule, _RULE_KEYS, at)
        if rule["name"] in book.severities:
            raise ValueError(f"{at}: duplicate rule name")
        try:
            formula = parse(str(rule["formula"]))
        except SyntaxError as exc:
            raise ValueError(f"{at}: {exc}") from exc
        severity = str(rule.get("severity", "HARD_STOP"))
        if severity not in ConstraintSeverity.__members__:
            raise ValueError(f"{at}: unknown severity {severity!r}; "
                             f"use one of {list(ConstraintSeverity.__members__)}")
        book.constraints.append(Constraint(
            str(rule["name"]), formula, weight=float(rule.get("weight", 1.0)),
            description=str(rule.get("description", "")), repair=str(rule.get("repair", "")),
            applies_to_final_answer=bool(rule.get("applies_to_final_answer", False))))
        book.severities[str(rule["name"])] = ConstraintSeverity[severity]
    return book


def _no_unknown_keys(data: Dict[str, Any], allowed: set, where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


__all__ = ["Rulebook", "available", "load"]
