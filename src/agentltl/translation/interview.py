# -*- coding: utf-8 -*-
"""
agentltl/translation/interview.py -- G4b. Ask the model for the obligations the RULES could not state.

===============================================================================
`genv2/emit.py` states what its rules cover. Whatever it emits nothing for is a GAP,
and a gap is where blindness comes from: a wrong episode with no violated constraint
has nothing for enforcement to nudge on. MedAgentBench sits at 67% blindness against
1% false-alarm mass, so recall -- not precision -- is the frontier.

`genv2/fill.py::gaps` ALREADY enumerates exactly those places: uncovered steps,
dropped slots, table slots with no column, unanchored guards, unordered writes. It is
reused unchanged as the work-list. fill.py's own tier -- a closed menu of seven kinds
-- is left untouched; this one hands the model a GRAMMAR instead of a menu, which is
the whole point: the menu cannot express `amount <= capacity - current`.

WHAT THE MODEL IS AND IS NOT TRUSTED WITH. It composes an expression over a closed
grammar of accessors and total operators (`genv4/expr.py`). It writes no Python, no
formula type, no severity and no layer. Every answer is validated against the
AgentView and rejected with a REASON, and whether the result may block is decided by
the witness's leaf-perturbation test, not by the author.

THE LOOP IS THE MEASURED PART, copied from G2 rather than reinvented:

  * ONE question per call. With orders and slots in a single reply the 27B model
    returned 90 of 90 unit slots as "verbatim" while naming a column -- contradicting
    itself. Adding a question degraded a DIFFERENT answer.
  * Each round is told what is still open AND why the last attempt was rejected.
    Validating inside the loop is what makes it converge; holding the reason until
    afterwards throws away the most useful thing we know.
  * A round that accepts nothing escalates to a differently-worded question. At
    temperature 0 an unchanged prompt returns an unchanged answer, so a plain retry
    is a wasted call. Two stalls end it.
  * The model is given the REAL options -- the tools, each parameter's schema block,
    the variables, and what is already asserted. Asked which column a slot reads with
    no column list in the prompt, it answered "dose"; offered `value, unit` it
    answered correctly.

"none" IS A FIRST-CLASS ANSWER. Most gaps genuinely have nothing checkable from tool
calls alone -- a `report` step is content owed to the user, and tau's obligations are
dialogue acts ("obtain explicit confirmation") that are not properties of a trace.
Forcing an expression there is how a spec acquires a constraint that fails every run.

NO `Implies(Called(tool), ...)` WRAPPER, deliberately -- a documented deviation.
The concern it would answer is that `arg()` on an absent tool doubles as a presence
requirement, the trap that makes bare `Before` and `CalledWith` fail a run that never
called `b`. That does not apply here: `expr._eval` returns UNEVALUABLE for an
uncalled tool, and unevaluable passes. The guard would be a second place validating
the same fact, and validating in two places is exactly how `bind_slots` came to
re-resolve what the interview had already resolved and drop the window. Vacuity is
recorded as `vacuous_if_absent` so `compliance.active_score` still excludes dormant
constraints, and `tests/test_translation.py` pins that an absent tool passes.
===============================================================================
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

from . import expr as X
from ._llm import chat, parse_json, resolve_model
from ._protocols import GraphLike as ProcGraph, ViewLike as AgentView

# Gap kind that gap discovery (which needs the benchmark's witness) tags an act step
# whose omission no constraint would notice. Defined here because the interview's
# prompt branches on it; the producer of gaps lives with the harness.
OMISSION = "omission_invisible"

PIPELINE = "genv4/openpred"

_SYSTEM = (
    "You state ONE checkable obligation about ONE step of an agent task, as an "
    "expression in a fixed grammar. You never see a reference solution and must "
    "never state a value the text does not contain. You do not write code, logic "
    "types, severities or layers. Reply with ONLY a JSON object, no prose and no "
    "markdown fences."
)

_GRAMMAR = """\
An expression is a JSON object. It must evaluate to TRUE/FALSE, so the outermost
node is always a comparison or a connective.

VALUES you may read:
  {"acc":"arg","tool":T,"path":P}          an argument of a call to T
                                           add "which":"first"|"last"|"any"
  {"acc":"result","tool":T,"json":P}       a value from T's returned data
  {"acc":"count","tool":T}                 how many times T was called
  {"acc":"clock"}                          the current time
  {"acc":"const","value":V,"span":S}       a fixed value; SPAN must be the exact
                                           words from the task or policy it came from
ARITHMETIC (values in, value out):
  add sub mul div abs min max sum len      {"op":"sub","args":[A,B]}
COMPARISONS (values in, true/false out):
  eq ne lt le gt ge in matches
CONNECTIVES (true/false in, true/false out):
  and or not implies

EXAMPLES
  the amount added must not exceed the space left in the tank:
  {"op":"le","args":[
     {"acc":"arg","tool":"fillFuelTank","path":"fuelAmount"},
     {"op":"sub","args":[
        {"acc":"result","tool":"get_vehicle_info","json":"tankCapacity"},
        {"acc":"result","tool":"get_vehicle_info","json":"fuelLevel"}]}]}

  the record must be filed at most once:
  {"op":"le","args":[{"acc":"count","tool":"POST_Observation"},
                     {"acc":"const","value":1,"span":"file the result"}]}
"""


def _tool_block(view: AgentView, limit: int = 24) -> str:
    """Tools WITH each parameter's schema. Asked to choose from options it was never
    shown, the model answers a different question."""
    rows = []
    for t in view.tools[:limit]:
        ps = []
        for p in (t.parameters or [])[:14]:
            pd = t.param_schema(p) or {}
            bits = [x for x in (pd.get("type"), (pd.get("description") or "").strip())
                    if x]
            ps.append(f"      {p}: {' -- '.join(bits)[:110]}" if bits
                      else f"      {p}")
        rows.append(f"  {t.name}\n" + "\n".join(ps))
    return "\n".join(rows)


def _already(spec: dict, node_id: str) -> str:
    got = [c.get("rationale") or c.get("id")
           for c in (spec.get("constraints") or [])
           if c.get("from_node") == node_id]
    return "\n".join(f"  - {g}" for g in got[:12]) or "  (nothing yet)"


def _prompt_parts(view: AgentView, g: ProcGraph, spec: dict, gap: dict,
                  rejected: List[str], nudge: bool):
    """(tool, step, span, feedback, ask) -- shared by both interview modes.

    The `ask` differs by MODE: library mode must not hand out raw grammar snippets,
    or the model answers with an expression the selection path cannot bind.
    """
    n = g.node(gap.get("node") or "")
    step = f'"{n.step}"' if n is not None else "(no step)"
    span = f'"{n.span}"' if n is not None and n.span else "(none)"
    tool = n.tool if n is not None and n.tool else "(none bound)"
    fb = ""
    if rejected:
        fb = ("\nYour previous answer was REJECTED:\n"
              + "\n".join(f"  - {r}" for r in rejected[-3:])
              + "\nFix exactly that.\n")

    if MODE == "library":
        if gap.get("gap") == OMISSION:
            ask = (f"Every constraint above is satisfied even by an episode that "
                   f"never calls {gap.get('tool')} at all. If one of the patterns "
                   f"above says when that call is OWED, choose it and give its "
                   f"parameters. Otherwise answer none.")
        else:
            ask = ("Does one of the patterns above state an obligation here that the "
                   "constraints listed are missing? If not, answer none.")
        if nudge:
            ask = (f"What could go WRONG at this step and still satisfy every "
                   f"constraint listed above? If one of the patterns above catches "
                   f"it, choose that pattern; otherwise answer none.")
        return tool, step, span, fb, ask

    ask = _freeform_ask(gap, nudge, spec)
    return tool, step, span, fb, ask


def _freeform_ask(gap: dict, nudge: bool, spec: dict) -> str:
    if gap.get("gap") == OMISSION:
        # The measured blindness, and it needs a DIFFERENT question. Asking "what
        # else is true here" produces another argument binding, which is already
        # satisfied by the episode that does nothing. The obligation wanted is that
        # the step HAPPENS -- usually only when the data says it should.
        ask = (
            f"Every constraint above is satisfied even by an episode that never "
            f"calls {gap.get('tool')} at all. State when that call is OWED.\n"
            "If the task orders it unconditionally, use:\n"
            '  {"op":"ge","args":[{"acc":"count","tool":"%s"},'
            '{"acc":"const","value":1,"span":"<the words that order it>"}]}\n'
            "If it is owed only when the data says so, guard it with `implies`, "
            "reading the deciding value from the earlier call's own output:\n"
            '  {"op":"implies","args":[<the condition on a result(...)>,'
            '{"op":"ge","args":[{"acc":"count","tool":"%s"},'
            '{"acc":"const","value":1,"span":"..."}]}]}\n'
            "Answer none only if nothing in the task or policy says when it is owed."
        ) % (gap.get("tool"), gap.get("tool"))
    else:
        ask = ("Is there an obligation here that the constraints listed above do NOT "
               "already state, and that can be checked from the tool calls alone?")
    if nudge:
        ask = ("Name the single most important thing that could go WRONG at this step "
               "and still satisfy every constraint listed above. If it can be checked "
               "from the tool calls, express it; otherwise answer none.")
    return ask


def _prompt(view: AgentView, g: ProcGraph, spec: dict, gap: dict,
            rejected: List[str], nudge: bool) -> List[dict]:
    tool, step, span, fb, ask = _prompt_parts(view, g, spec, gap, rejected, nudge)
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"""TASK TEXT
{view.task_text()[:2600]}

{"POLICY (reference for values)" if view.policy else ""}
{(view.policy or "")[:1600]}

TOOLS AND THEIR PARAMETERS
{_tool_block(view)}

THE STEP
  id: {gap.get('node')}   tool: {tool}
  step: {step}
  quoted from: {span}
  what is missing here: {gap.get('gap')} -- {gap.get('why')}

ALREADY ASSERTED ABOUT THIS STEP
{_already(spec, gap.get('node') or '')}

THE GRAMMAR
{_GRAMMAR}
{fb}
{ask}

Reply with ONE JSON object, either
  {{"none": true, "why": "<one short reason nothing is checkable>"}}
or
  {{"expr": <expression>, "why": "<what obligation this states>",
    "repair": "<one imperative sentence telling the agent what to do>"}}"""},
    ]


# ── library mode: SELECTION over a closed set, not free composition ──────────
# Default, and the reason is measured. Free composition produced 119 expressions in
# exactly 2 shapes -- no expressive diversity -- while getting both subtly wrong:
# a fabricated `2.0` cited from the span "1.5 to 1.9", and, in all 89 instances, the
# omission of the recency guard task5 states outright. Retrofitting the same 119
# selections onto templates took the LLM tier's false alarms from 21 to **0** and
# still cut blindness. `OPENPRED_MODE=freeform` restores composition.
MODE = os.environ.get("OPENPRED_MODE", "library").strip().lower()


def _library_messages(view: AgentView, g: ProcGraph, spec: dict, gap: dict,
                      tool: str, step: str, span: str, fb: str, ask: str) -> list:
    import json as _json
    from . import library as LIB
    return [
        {"role": "system", "content":
         "You choose whether a known procedural pattern applies to one step, and "
         "with what parameters. You do not invent logic. Answer with ONE JSON "
         "object and nothing else."},
        {"role": "user", "content": f"""\
TASK
{view.task_text()[:2600]}

{"POLICY (reference for values)" if view.policy else ""}
{(view.policy or "")[:1600]}

TOOLS AND THEIR PARAMETERS
{_tool_block(view)}

THE STEP
  id: {gap.get('node')}   tool: {tool}
  step: {step}
  quoted from: {span}
  what is missing here: {gap.get('gap')} -- {gap.get('why')}

ALREADY ASSERTED ABOUT THIS STEP
{_already(spec, gap.get('node') or '')}

THE PATTERNS YOU MAY CHOOSE FROM
{_json.dumps(LIB.menu(), indent=1)}

RULES
  * Choose a pattern only if the task really states that obligation. "none" is a
    correct and common answer.
  * Every number you give must appear VERBATIM in the task or the policy above.
    Do not round, and do not derive one. If the cutoff is not written down, omit
    `threshold` and the dosing table's own bands will be used.
  * Give `window`/`max_age` exactly as the task words it.
{fb}
{ask}

Reply with ONE JSON object, either
  {{"none": true, "why": "<one short reason no pattern applies>"}}
or
  {{"template": "<name>", "args": {{...}}, "why": "<what obligation this states>",
    "repair": "<one imperative sentence telling the agent what to do>"}}"""},
    ]


def _producing_tools(g: ProcGraph) -> List[str]:
    """Tools whose output some step reads. A `result()` on anything else can never be
    known, so the validator rejects it rather than emitting a permanently
    unevaluable constraint."""
    out = []
    for n in g.nodes:
        if n.bound() and n.role in ("observe", "act") and n.tool not in out:
            out.append(n.tool)
    return out


def _needed_context(ast: dict, view: AgentView, facts: dict) -> dict:
    """Only what this expression actually reads. The node is SELF-CONTAINED, like
    ArgInWindow carrying resolved bounds rather than the clock -- but embedding the
    whole policy table in every constraint would bloat every spec."""
    out: Dict[str, Any] = {}
    accs = X.accessors(ast)
    if "clock" in accs and view.clock:
        out["clock"] = view.clock
    # `result` needs `extract` as much as `table` does: it is what tells the
    # accessor which leaf carries the deciding figure, and therefore how to pick the
    # MOST RECENT one rather than the last in document order. Passing facts only for
    # `table` left every `result` expression reading a stale entry out of an unsorted
    # bundle, which is where this tier's false alarms came from.
    if any(a in ("table", "result", "when") for a in accs):
        keep = {k: v for k, v in (facts or {}).items()
                if k in ("dose_table", "extract")}
        if keep:
            out["facts"] = keep
    return out


def _messages_for(view: AgentView, g: ProcGraph, spec: dict, gap: dict,
                  rejected: List[str], stalled: bool) -> list:
    """Library mode asks which pattern applies; freeform asks for an expression."""
    if MODE != "library":
        return _prompt(view, g, spec, gap, rejected, stalled)
    tool, step, span, fb, ask = _prompt_parts(view, g, spec, gap, rejected, stalled)
    return _library_messages(view, g, spec, gap, tool, step, span, fb, ask)


def propose(client, view: AgentView, g: ProcGraph, spec: dict, gap: dict,
            facts: dict, max_rounds: int = 3, chat_fn=None
            ) -> Tuple[Optional[dict], List[str]]:
    """One gap -> (constraint or None, the reasons every attempt was rejected)."""
    rejected: List[str] = []
    stalled = False
    produced = _producing_tools(g)

    for _ in range(max_rounds):
        try:
            raw = (chat_fn or chat)(
                client, _messages_for(view, g, spec, gap, rejected, stalled))
            got = parse_json(raw)
        except Exception as exc:
            rejected.append(f"the reply could not be read ({type(exc).__name__})")
            got = None
        if isinstance(got, list) and got:
            got = got[0]                       # parse tolerantly
        if not isinstance(got, dict):
            if stalled:
                break
            stalled = True
            continue
        if got.get("none"):
            return None, rejected              # a legitimate answer, not a failure

        template = None
        if MODE == "library":
            from . import library as LIB
            template = str(got.get("template") or "").strip()
            if not template:
                rejected.append('no "template" name in the reply; choose one of '
                                + ", ".join(LIB.TEMPLATES) + ' or answer {"none": true}')
                if stalled:
                    break
                stalled = True
                continue
            text = (view.task_text() or "") + "\n" + (view.policy or "")
            ast, why = LIB.bind(template, got.get("args") or {}, text)
            if ast is None:
                # The reason is the whole mechanism: validating INSIDE the loop and
                # saying why is what makes the interview converge, and it is exactly
                # what rejects a fabricated constant instead of emitting it.
                rejected.append(why)
                if stalled:
                    break
                stalled = True
                continue
        else:
            ast = got.get("expr")
            if not isinstance(ast, dict):
                rejected.append('no "expr" object in the reply')
                if stalled:
                    break
                stalled = True
                continue

        ok, why = X.validate(ast, view, produced_tools=produced)
        if not ok:
            rejected.append(why)
            if stalled:
                break
            stalled = True
            continue

        node = gap.get("node") or "n0"
        idx = sum(1 for c in (spec.get("constraints") or [])
                  if str(c.get("id", "")).startswith(f"{node}_expr"))
        spans = [str(c.get("span") or "") for c in X.consts(ast)]
        return {
            "id": f"{node}_expr" + (f"_{idx}" if idx else ""),
            "formula": {"type": "Expr",
                        "args": dict(ast=ast, **_needed_context(ast, view, facts))},
            "mandatory": True,
            "weight": 1.0,
            "rationale": str(got.get("why") or "")[:400] or X._render(ast),
            "repair": str(got.get("repair") or "")[:400]
                      or f"Make sure {X._render(ast)}.",
            "derived": "graph",
            # The two tiers must stay separately measurable: run 2's attribution put
            # code-derived constraints at 28W/0L and authored ones at 4W/6L, and
            # `author` is what lets this tier be dropped with NO regeneration.
            "author": "llm",
            "pipeline": PIPELINE,
            "grounding": "quoted" if spans else "derived",
            "evidence": ["authored"],
            "from_node": node,
            "from_edge": "",
            # Which library pattern was selected, so one template can be retired on
            # its own W/L record without touching the rest of the tier.
            **({"template": template} if template else {}),
            # Vacuity is a property of the accessor, not of a guard: `arg()` on an
            # uncalled tool is UNEVALUABLE. Recorded so active_score still excludes
            # a dormant constraint. See the module docstring.
            "vacuous_if_absent": X.reads_args_of(ast) or "",
        }, rejected

    return None, rejected


def augment(client, view: AgentView, g: ProcGraph, spec: dict, gaps: List[dict],
            facts: Optional[dict] = None, max_gaps: int = 12, chat_fn=None) -> dict:
    """Add open predicates for the gaps `emit.py` left. Mutates and returns `spec`.

    Bounded by `max_gaps`: an interview cannot be replayed by GRAPH_REEMIT, so this
    is a real generation cost per instance, unlike G3/G4 which replay for free. The
    cap is REPORTED rather than silent -- a workflow that quietly truncates coverage
    reads as "we covered everything" when it did not.
    """
    facts = facts or spec.get("policy_facts") or {}
    if chat_fn is None:
        resolve_model()          # fail here, not inside propose's broad except
    # `gaps` is computed by the caller: finding them needs the benchmark's witness
    # (drop one act node's call from the correct walk and rescore), which agentltl
    # does not own. NOT fill.gaps alone -- measured on mab it returns only
    # `report_step`, content owed to the user, where declining is correct; the
    # blindness is act omissions the spec cannot see.
    stats = {"gaps": len(gaps), "asked": 0, "added": 0, "declined": 0,
             "rejected": 0, "skipped_over_cap": max(0, len(gaps) - max_gaps)}
    notes: List[str] = []

    for gap in gaps[:max_gaps]:
        stats["asked"] += 1
        c, why = propose(client, view, g, spec, gap, facts, chat_fn=chat_fn)
        stats.setdefault("by_kind", {}).setdefault(gap.get("gap"), 0)
        if c is None:
            stats["declined" if not why else "rejected"] += 1
            if why:
                notes.append(f"{gap.get('node')}/{gap.get('gap')}: {why[-1]}")
            continue
        spec.setdefault("constraints", []).append(c)
        stats["added"] += 1
        stats["by_kind"][gap.get("gap")] += 1

    spec["openpred"] = {"pipeline": PIPELINE, **stats,
                        "notes": notes[:20]}
    return spec
