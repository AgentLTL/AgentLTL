"""Versioned reviewer prompt registry (hot-swappable templates).

Every version shares one reviewer-prompt *family*: a fixed role framing, a
version-specific criteria body, the strict JSON output contract, and a
strategy-specific output section appended at render time.  This keeps the three
review strategies (progressive / selection / grading) on a single prompt family
that differs only in its output block.

The paper's dominant failure mode is reviewer *over-skepticism* — flagging valid
tool-only calls as "incomplete" for lacking prose (~23% redundant loops).  The
``OVER_SKEPTICISM_GUARD`` below is baked into every version from ``v1_1`` onward.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# ── The over-skepticism guard (verbatim; present from v1_1 onward) ────────────
OVER_SKEPTICISM_GUARD = (
    "[CRITICAL] Tool-only responses are complete. Do NOT mark a tool-only "
    "response as incomplete merely because it lacks a user-facing answer, "
    "follow-up explanation, or final-results presentation. A tool call is a "
    "standalone step. Judge only whether the tool calls themselves are correct."
)

# ── Role framing shared by all versions ───────────────────────────────────────
_ROLE_FRAMING = (
    "You are a reviewer for a tool-calling agent. The agent has produced a "
    "PROVISIONAL tool call that has NOT yet been executed. Your job is to judge "
    "whether the tool call should be executed as-is."
)

# ── Version-specific criteria bodies ──────────────────────────────────────────
_V1_BODY = (
    "Check the provisional tool call for correctness: is the right tool being "
    "called, with arguments that are consistent with the user's request and the "
    "conversation so far? Flag an error only when the call is clearly wrong."
)

_V2_BFCL_BODY = (
    "Evaluate the provisional tool call against these single-turn criteria:\n"
    "1. Request fulfillment — does this call move toward what the user asked?\n"
    "2. Tool-call correctness — is the chosen tool appropriate per its "
    "documentation?\n"
    "3. Argument fidelity — do the arguments match the user's stated values and "
    "the tool's parameter schema (names, types, required fields)?\n"
    "4. Sensible defaults — reasonable defaults for unspecified optional "
    "arguments are fine; do not flag them.\n"
    "5. Do not be pedantic — minor stylistic choices are not errors.\n"
    "6. No external facts — judge only from the request, conversation, and tool "
    "docs; do not invent facts or requirements not present.\n"
    "7. Solvability — if the task cannot be solved with the available tools, say "
    "so; otherwise treat it as solvable. This is a binary judgement."
)

_V2_TAU_BODY = (
    "Evaluate the provisional tool call against these multi-turn criteria:\n"
    "1. Context Awareness (CRITICAL) — before acting, verify that any stated "
    "policies, preconditions, or required confirmations have actually been met "
    "in the conversation. Flag a call that acts before its precondition holds.\n"
    "2. Request fulfillment — does this call move toward the user's goal?\n"
    "3. Tool-call correctness and argument fidelity vs. the tool documentation.\n"
    "4. Partial progress is not an error — an intermediate step that does not by "
    "itself complete the task is still correct if it is a valid next step.\n"
    "5. Logical consistency — the call must not contradict earlier established "
    "facts, results, or the user's constraints.\n"
    "6. Sensible defaults are fine; do not be pedantic; use no external facts."
)

# Versions that carry the over-skepticism guard.
_GUARDED_VERSIONS = frozenset({"v1_1", "v2_bfcl", "v2_tau", "v3_gepa"})

# version -> criteria body.  ``v3_gepa`` is a loadable slot (see _load_v3_gepa).
_BODIES: Dict[str, str] = {
    "v1": _V1_BODY,
    "v1_1": _V1_BODY,
    "v2_bfcl": _V2_BFCL_BODY,
    "v2_tau": _V2_TAU_BODY,
}

KNOWN_PROMPT_VERSIONS = frozenset({"v1", "v1_1", "v2_bfcl", "v2_tau", "v3_gepa"})
DEFAULT_PROMPT_VERSION = "v1_1"

# Env knobs for the loadable GEPA-optimized prompt slot.
_V3_GEPA_ENV_TEXT = "AGENTLTL_REVIEWER_V3_GEPA_PROMPT"
_V3_GEPA_ENV_FILE = "AGENTLTL_REVIEWER_V3_GEPA_PROMPT_FILE"


def _load_v3_gepa() -> str:
    """Load the GEPA-optimized reviewer prompt from env/file.

    The slot is *selectable* (registered in ``KNOWN_PROMPT_VERSIONS``) but never
    hand-faked: the body must be supplied externally via
    ``AGENTLTL_REVIEWER_V3_GEPA_PROMPT`` (inline) or
    ``AGENTLTL_REVIEWER_V3_GEPA_PROMPT_FILE`` (path).
    """
    inline = os.getenv(_V3_GEPA_ENV_TEXT)
    if inline:
        return inline
    path = os.getenv(_V3_GEPA_ENV_FILE)
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    raise ValueError(
        "prompt_version 'v3_gepa' selected but no prompt supplied. Set "
        f"{_V3_GEPA_ENV_TEXT} (inline) or {_V3_GEPA_ENV_FILE} (path)."
    )


# ── Strategy-specific output contract sections ────────────────────────────────
_OUTPUT_COMMON = (
    'Respond with a single strict JSON object and nothing else. Keys:\n'
    '  "reasoning": string — brief justification.\n'
    '  "message": string — feedback for the agent (empty if no error).\n'
    '  "error": boolean — true iff the call is wrong and should not execute.'
)

_OUTPUT_SECTIONS = {
    "r": _OUTPUT_COMMON,
    "s": (
        "You are shown SEVERAL candidate tool calls. Pick the single best one.\n"
        + _OUTPUT_COMMON
        + '\n  "selected_index": integer — 0-based index of the best candidate.'
    ),
    "g": (
        "You are shown SEVERAL candidate tool calls. Score EACH candidate.\n"
        'Respond with a single strict JSON object and nothing else. Keys:\n'
        '  "reasoning": string — brief justification.\n'
        '  "message": string — optional feedback.\n'
        '  "error": boolean — true iff the best candidate is still wrong.\n'
        '  "score": number in [0.0, 1.0] — quality of the candidate you are '
        "grading (one call per request)."
    ),
}


def build_system_prompt(version: str, strategy: str) -> str:
    """Compose the reviewer system prompt for *version* + *strategy*."""
    if version not in KNOWN_PROMPT_VERSIONS:
        raise ValueError(
            f"Unknown prompt version {version!r}; known: "
            f"{sorted(KNOWN_PROMPT_VERSIONS)}"
        )
    if strategy not in _OUTPUT_SECTIONS:
        raise ValueError(f"Unknown strategy {strategy!r}")

    body = _load_v3_gepa() if version == "v3_gepa" else _BODIES[version]
    parts = [_ROLE_FRAMING, body]
    if version in _GUARDED_VERSIONS:
        parts.append(OVER_SKEPTICISM_GUARD)
    parts.append(_OUTPUT_SECTIONS[strategy])
    return "\n\n".join(parts)


def _format_candidate(idx: Optional[int], call: Dict[str, Any]) -> str:
    name = call.get("name") or call.get("tool_name")
    args = call.get("arguments", call.get("tool_args", {}))
    try:
        args_str = json.dumps(args, ensure_ascii=False, default=str)
    except Exception:
        args_str = str(args)
    label = f"Candidate {idx}: " if idx is not None else "Provisional call: "
    return f"{label}{name}({args_str})"


def build_user_prompt(
    task_context: str,
    candidates: List[Dict[str, Any]],
    tool_docs: Optional[str] = None,
) -> str:
    """Build the reviewer user message: task context, candidate(s), tool docs.

    A single-element ``candidates`` list renders one "Provisional call"; multiple
    elements render enumerated candidates (best-of-N).
    """
    sections = ["## Conversation / task context", task_context.strip()]
    if tool_docs:
        sections += ["## Available tools", tool_docs.strip()]
    sections.append("## Tool call(s) under review")
    if len(candidates) == 1:
        sections.append(_format_candidate(None, candidates[0]))
    else:
        sections += [_format_candidate(i, c) for i, c in enumerate(candidates)]
    return "\n\n".join(sections)


def render(
    version: str,
    strategy: str,
    task_context: str,
    candidates: List[Dict[str, Any]],
    tool_docs: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Render a full reviewer chat: ``[system, user]`` message dicts."""
    return [
        {"role": "system", "content": build_system_prompt(version, strategy)},
        {"role": "user", "content": build_user_prompt(task_context, candidates, tool_docs)},
    ]
