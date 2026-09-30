# -*- coding: utf-8 -*-
"""The three LLM helpers the interview needs, and nothing that reads the environment.

`chat` takes an OpenAI-shaped client (anything with `.chat.completions.create`), so
`import agentltl.translation` never requires `openai`. Endpoint, credentials and any
provider-specific request options belong to the caller: pass `extra_body` /
`extra_headers` rather than this module guessing them.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


def resolve_model(model: Optional[str] = None) -> str:
    """The model id, from the argument or AGENTLTL_MODEL. Raises rather than guessing:
    an empty id reaches the gateway as a routing error that reads as a model failure."""
    m = (model or os.environ.get("AGENTLTL_MODEL") or "").strip()
    if not m:
        raise ValueError("no model: pass model=... or set AGENTLTL_MODEL "
                         "(or pass chat_fn= to use your own client wiring)")
    return m


def chat(client, messages: List[dict], *, model: Optional[str] = None,
         max_tokens: int = 4000,
         temperature: float = 0.0, extra_body: Optional[Dict[str, Any]] = None,
         extra_headers: Optional[Dict[str, str]] = None, stream: bool = True) -> str:
    """Return the reply text. Streamed by default, so an idle-read deadline on a slow
    generation is not tripped; only `content` is read, never a reasoning field."""
    kw: Dict[str, Any] = dict(model=resolve_model(model), messages=messages,
                              temperature=temperature, max_tokens=max_tokens)
    if extra_body:
        kw["extra_body"] = extra_body
    if extra_headers:
        kw["extra_headers"] = extra_headers
    if not stream:
        r = client.chat.completions.create(**kw)
        return (r.choices[0].message.content or "") if r.choices else ""
    parts = []
    for chunk in client.chat.completions.create(stream=True, **kw):
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if getattr(d, "content", None):
            parts.append(d.content)
    return "".join(parts)


def parse_json(text: str) -> Any:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[4:] if t.lower().startswith("json") else t
        t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        for pat in (r"\[.*\]", r"\{.*\}"):
            m = re.search(pat, t, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    continue
        raise
