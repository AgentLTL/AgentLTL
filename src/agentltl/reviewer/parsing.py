"""Defensive parsing of reviewer LLM output.

Reviewer models return JSON, but not reliably: they wrap it in ``` fences, add
prose before/after, or emit malformed JSON.  ``parse_reviewer_output`` recovers
the JSON object where it can and **fails safe to "no error"** otherwise — a
reviewer that cannot be understood must never block the base agent (fail-open).

Output contract::

    {"reasoning": str, "message": str, "error": bool}

grading adds ``"score"`` (float in [0, 1]); selection adds ``"selected_index"``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

# Matches ```json ... ``` or ``` ... ``` fenced blocks.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ReviewResult:
    """Parsed, normalized reviewer verdict.

    ``parse_failed`` records that the raw text could not be understood and the
    result was defaulted to fail-open (``error=False``) — surfaced in telemetry.
    """

    error: bool = False
    reasoning: str = ""
    message: str = ""
    score: Optional[float] = None
    selected_index: Optional[int] = None
    parse_failed: bool = False


def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of the first JSON object from *text*."""
    if not text:
        return None

    # 1. Try fenced blocks first (most common wrapping).
    for m in _FENCE_RE.finditer(text):
        obj = _try_load(m.group(1))
        if obj is not None:
            return obj

    # 2. Try the whole string.
    obj = _try_load(text)
    if obj is not None:
        return obj

    # 3. Scan for balanced {...} spans and try each, longest first.
    for span in _candidate_spans(text):
        obj = _try_load(span)
        if obj is not None:
            return obj
    return None


def _try_load(s: str) -> Optional[dict]:
    try:
        obj = json.loads(s.strip())
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _candidate_spans(text: str):
    """Yield balanced ``{...}`` substrings, longest span at each start first."""
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    spans = []
    for start in starts:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    spans.append(text[start : end + 1])
                    break
    # Longest first — the outermost object is the most likely full verdict.
    spans.sort(key=len, reverse=True)
    return spans


def _coerce_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0", ""):
            return False
    return None


def parse_reviewer_output(text: str) -> ReviewResult:
    """Parse reviewer text into a :class:`ReviewResult`, failing open.

    Any unrecoverable input yields ``error=False`` (execute the provisional
    call) with ``parse_failed=True`` recorded for telemetry.
    """
    obj = _extract_json_object(text or "")
    if obj is None:
        return ReviewResult(error=False, parse_failed=True,
                            reasoning="", message="")

    error = _coerce_bool(obj.get("error"))
    if error is None:
        # Contract violated (no usable "error" field) — fail open but flag it.
        return ReviewResult(error=False, parse_failed=True,
                            reasoning=str(obj.get("reasoning", "")),
                            message=str(obj.get("message", "")))

    score = obj.get("score")
    if score is not None:
        try:
            score = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            score = None

    selected = obj.get("selected_index")
    if selected is not None:
        try:
            selected = int(selected)
        except (TypeError, ValueError):
            selected = None

    return ReviewResult(
        error=error,
        reasoning=str(obj.get("reasoning", "")),
        message=str(obj.get("message", "")),
        score=score,
        selected_index=selected,
        parse_failed=False,
    )
