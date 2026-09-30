# -*- coding: utf-8 -*-
"""
agentltl/translation/pipeline.py -- one instance's spec, augmented with open predicates.

Works on IN-MEMORY objects. Where a spec came from, where it is written back, which
corpus directory it belongs to and how the graph was re-derived are the benchmark's
business, so all of it is passed in through a `Backend` and none of it is imported:

    backend.positive_walk(view, graph, facts)  -> a correct trace, in graph order
    backend.fill_gaps(view, graph, spec)       -> the places the rules said nothing
    backend.compile_spec(spec, A)              -> [Constraint]
    backend.witness_check(view, graph, spec, A, facts) -> dict   (optional)

The cost is worth stating: an interview cannot be replayed for free, unlike the
deterministic stages that precede it, so every call here spends the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from . import gaps as GAPS
from . import interview as IV
from ._protocols import GraphLike, ViewLike


@dataclass(frozen=True)
class Backend:
    positive_walk: Callable[..., Any]
    fill_gaps: Callable[..., Any]
    compile_spec: Callable[..., Any]
    witness_check: Optional[Callable[..., Dict[str, Any]]] = None


def augment_spec(client, view: ViewLike, graph: GraphLike, spec: dict, backend: Backend,
                 A: dict, *, max_gaps: int = 12, chat_fn=None,
                 run_witness: bool = True) -> Dict[str, Any]:
    """Augment `spec` in place. Returns a small status dict for the driver.

    `A` is the AgentLTL builder namespace; it is needed to score the pruned walks that
    locate invisible omissions -- the measured blindness on this family.
    """
    facts = spec.get("policy_facts") or {}
    before = len(spec.get("constraints") or [])
    gaps = GAPS.all_gaps(view, graph, spec, A, facts,
                         positive_walk=backend.positive_walk,
                         fill_gaps=backend.fill_gaps,
                         compile_spec=backend.compile_spec)
    IV.augment(client, view, graph, spec, gaps, facts=facts, max_gaps=max_gaps,
               chat_fn=chat_fn)

    out: Dict[str, Any] = {
        "status": "augmented", "n_before": before,
        "n_after": len(spec.get("constraints") or []),
        **{k: v for k, v in (spec.get("openpred") or {}).items() if k != "notes"}}

    if run_witness and backend.witness_check is not None:
        try:
            rep = backend.witness_check(view, graph, spec, A, facts)
            out["witness"] = rep
            out["n_blockable"] = rep.get("n_blockable")
            out["n_fails_correct"] = rep.get("n_fails_correct")
        except Exception as exc:
            out["witness_error"] = f"{type(exc).__name__}: {exc}"
    return out
