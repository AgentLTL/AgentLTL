# -*- coding: utf-8 -*-
"""
agentltl/translation/gaps.py -- where to ASK, and why fill.gaps is not enough on its own.

===============================================================================
The plan pointed the interview at `genv2/fill.py::gaps`, which enumerates every place
emit.py produced NOTHING: uncovered steps, dropped slots, table slots with no column,
unanchored guards, unordered writes. Measured on MedAgentBench it returns almost
nothing usable:

    60 instances -> 90 gaps, ALL of kind `report_step`, 1.5 per instance
    0 uncovered steps, 0 dropped slots, 0 unanchored guards, 0 unordered writes

and `report_step` is content owed to the user rather than a tool call, so declining
is the CORRECT answer there. A live probe on three instances added zero predicates
for exactly that reason.

The reason is structural, not a bug. mab's graphs are fully bound and every node
already yields constraints, so its 67% blindness is not "no constraint on this step".
It is "every constraint on this step is satisfied by a wrong episode" -- which
CLAUDE.md already states: *an episode that simply never orders satisfies every
`vacuous_if_absent`-guarded argument constraint, which is why task9's wrong episodes
averaged 0.984*.

So the work-list for RECALL is a different question: WHICH MISTAKE WOULD THIS SPEC NOT
NOTICE? One such mistake is decidable offline, needs no grader, and is the dominant
one here -- simply not acting:

    take the correct walk, DROP one act node's call, and score the spec again.
    If nothing fails, that omission is invisible.

Measured over 40 mab instances: **30 of 30 act omissions are invisible.** Every one.
That is the blindness, located precisely, with no reward and no reference trace --
only the graph and the spec.

This revives a check the flat pipeline had and the graph pipeline dropped: it scored
the EMPTY trace to detect vacuity, and `16_witness.json` still carries
`empty_trace_score: null` where that used to be. Pruning ONE act is the sharper form,
because it says which act.

WHY AN OPEN PREDICATE IS THE RIGHT TOOL FOR IT. The obligation is usually
CONDITIONAL -- "if the last A1C is stale, order one" -- and the previous attempt at
it, `ObservedThenRequired`, is deliberately OFF: it moved blindness 28.6% -> 26.7%
while moving false alarms 7.2% -> 18.1%, because deciding "the value triggers the
branch" depended on a hardcoded extraction path, and where that picked the wrong
figure a correct abstention was punished. An open predicate states the trigger
EXPLICITLY, over the producer's own output:

    implies( lt(result(GET_Observation_labs, "...value"), const(5.7, "5.7")),
             ge(count(POST_ServiceRequest), 1) )

and `count` under a lower bound is LIVENESS, so `taxonomy.disposition` returns
at_end and it can never block. It improves scoring and blindness at zero risk to the
action -- which is the right first place to spend a tier whose ancestors ran 4W/6L.
===============================================================================
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._protocols import GraphLike, ViewLike
from .interview import OMISSION


def _verdicts(spec: dict, calls: List[dict], A: dict, compile_spec) -> Optional[List[dict]]:
    """Score the whole spec against one trace in ONE pass.

    Compiling per constraint is fine for a handful and quadratic here -- this runs once
    per act node per instance. `compile_spec` is the caller's: agentltl does not own the
    spec format, only the predicates the specs compile down to.
    """
    try:
        G = compile_spec(spec, A)
    except Exception:
        return None
    if not G:
        return None
    metrics = {"tool_calls": [{"tool_name": c["name"], "tool_args": c["args"],
                               "tool_result": c.get("result")} for c in calls]}
    try:
        return (A["verify_trace"](metrics, G) or {}).get("constraints") or []
    except Exception:
        return None


def invisible_omissions(view: ViewLike, g: GraphLike, spec: dict, A: dict, *,
                        positive_walk, compile_spec,
                        facts: Optional[dict] = None) -> List[dict]:
    """One gap per act whose complete absence no constraint notices.

    `positive_walk(view, g, facts)` returns a trace a correct agent could have produced,
    in the graph's own order. Building it needs the benchmark's argument shapes, so it is
    injected rather than owned here.
    """
    try:
        walk = positive_walk(view, g, facts)
    except Exception:
        return []
    if not walk:
        return []
    out: List[dict] = []
    for n in g.ordered_nodes():
        if n.role != "act" or not n.bound():
            continue
        pruned = [c for c in walk if c.get("node") != n.id]
        if len(pruned) == len(walk):
            continue                      # this act is not in the walk anyway
        recs = _verdicts(spec, pruned, A, compile_spec)
        if recs is None:
            continue
        if any(r.get("passed") is False for r in recs):
            continue                      # the omission IS caught
        out.append({
            "gap": OMISSION, "node": n.id, "tool": n.tool,
            "why": (f"an episode that never calls {n.tool} at all satisfies every "
                    f"constraint in this spec, so not doing the work is invisible"),
        })
    return out


def all_gaps(view: ViewLike, g: GraphLike, spec: dict, A: Optional[dict] = None,
             facts: Optional[dict] = None, *, positive_walk, fill_gaps,
             compile_spec) -> List[dict]:
    """`fill_gaps` (where the rules said NOTHING) plus the omissions they cannot see.

    Ordered with the omissions FIRST: they are the measured blindness, and the
    per-instance interview budget should be spent there before it is spent on a
    `report_step` whose honest answer is "none".
    """
    later = fill_gaps(view, g, spec)
    first = (invisible_omissions(view, g, spec, A, positive_walk=positive_walk,
                                 compile_spec=compile_spec, facts=facts)
             if A is not None else [])
    return first + later
