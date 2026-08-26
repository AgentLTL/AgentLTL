"""reviewed_agent.py – NativeOpenAIAgent wrapped with an inference-time reviewer.

The "Reinforced Agent" technique on the native (hand-coded OpenAI-loop) backend.
A reviewer inspects the provisional tool call produced by ``_chat()`` *before* it
is committed/executed and either has the model regenerate with feedback
(progressive) or picks the best of several candidates (best-of-N).

Seam: :meth:`NativeOpenAIAgent.run_turn` calls ``self._chat()`` once per step to
get a chat-completion response, then executes the tool call in it. This subclass
overrides ``_chat()`` to run the review strategy (invoking the *original*
``_chat`` for each candidate) and return the chosen response — so ``run_turn``,
scoring, and enforcement are all untouched. Reviewer disabled ⇒ identical to the
base agent.

Fail-open: any reviewer error executes the original provisional call. A hard
per-turn cap bounds every strategy so the loop can never deadlock.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .backend import NativeOpenAIAgent
from agentltl.reviewer.config import ReviewerConfig
from agentltl.reviewer.parsing import ReviewResult, parse_reviewer_output
from agentltl.reviewer import prompts
from agentltl.reviewer.strategies import Candidate, run_strategy

logger = logging.getLogger(__name__)

_FEEDBACK_PREFIX = (
    "[REVIEWER FEEDBACK] Your previous tool call was flagged before execution. "
    "Reconsider and produce a corrected tool call. Feedback:\n"
)


class ReviewedNativeAgent(NativeOpenAIAgent):
    """A :class:`NativeOpenAIAgent` whose tool calls are reviewed pre-execution."""

    def __init__(
        self,
        *args,
        reviewer_config: Optional[ReviewerConfig] = None,
        reviewer_client: Optional[Any] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reviewer_config = reviewer_config
        self._reviewer_client = reviewer_client if reviewer_client is not None else self._client
        self._reviewer_turns: List[Dict[str, Any]] = []

    # ── Status (mirrors get_constraint_status) ──────────────────────────────────

    def get_reviewer_status(self) -> Dict[str, Any]:
        turns = self._reviewer_turns
        total_calls = sum(t["reviewer_calls"] for t in turns)
        cfg = self._reviewer_config
        return {
            "mode": "reviewed",
            "enabled": bool(cfg and cfg.enabled),
            "strategy": cfg.strategy if cfg else None,
            "n": cfg.n if cfg else None,
            "prompt_version": cfg.prompt_version if cfg else None,
            "num_reviewed_steps": len(turns),
            "total_reviewer_calls": total_calls,
            "reviewer_calls_per_step": (total_calls / len(turns)) if turns else 0.0,
            "num_calls_changed": sum(1 for t in turns if t["changed"]),
            "parse_failures": sum(t["parse_failures"] for t in turns),
            "total_reviewer_wall_ms": sum(t["reviewer_wall_ms"] for t in turns),
            "turns": turns,
        }

    # ── The seam ────────────────────────────────────────────────────────────────

    def _chat(self) -> Any:
        cfg = self._reviewer_config
        if cfg is None or not cfg.enabled:
            return super()._chat()

        task_context = _format_context(self._messages)
        tool_docs = _format_tool_docs(self._tool_schemas)
        saved_temp = self._temperature

        def generate(feedback: Optional[str], temperature: Optional[float]) -> Candidate:
            appended = False
            if feedback:
                self._messages.append({"role": "user", "content": _FEEDBACK_PREFIX + feedback})
                appended = True
            if temperature is not None:
                self._temperature = temperature
            try:
                resp = super(ReviewedNativeAgent, self)._chat()
            finally:
                self._temperature = saved_temp
                if appended:
                    self._messages.pop()
            return Candidate(chat_message=resp, call=_first_call(resp), temperature=temperature)

        def review(candidates: List[Candidate]) -> ReviewResult:
            # Nothing to review if the provisional step made no tool call (it's a
            # final natural-language answer) — approve immediately (also avoids a
            # wasted reviewer call and confusing the reviewer with "no call").
            if all(c.call is None for c in candidates):
                return ReviewResult(error=False)
            try:
                rendered = prompts.render(
                    cfg.prompt_version, cfg.strategy,
                    task_context=task_context,
                    candidates=[(c.call or {"name": None, "arguments": {}}) for c in candidates],
                    tool_docs=tool_docs,
                )
                kw: Dict[str, Any] = {"model": self._model, "messages": rendered}
                if cfg.reviewer_temperature is not None:
                    kw["temperature"] = cfg.reviewer_temperature
                out = self._reviewer_client.chat.completions.create(**kw)
                text = out.choices[0].message.content or ""
                return parse_reviewer_output(text)
            except Exception as exc:
                logger.warning("Reviewer call failed (%s); failing open.", exc)
                return ReviewResult(error=False, parse_failed=True)

        t0 = time.perf_counter()
        provisional = generate(None, None)
        # No tool call → skip the strategy entirely, execute as-is.
        if provisional.call is None:
            self._record_turn(None, provisional, 1, (time.perf_counter() - t0) * 1000.0, no_call=True)
            return provisional.chat_message

        try:
            # run_strategy will call generate() itself (incl. re-using the
            # provisional as its first candidate for rN).
            outcome = run_strategy(cfg, generate, review)
            chosen = outcome.chosen.chat_message
        except Exception as exc:
            logger.warning("Reviewer strategy failed (%s); executing provisional call.", exc)
            outcome = None
            chosen = provisional.chat_message
        review_ms = (time.perf_counter() - t0) * 1000.0
        self._record_turn(outcome, provisional, self._model_step, review_ms)
        return chosen

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _record_turn(self, outcome, provisional, step, review_ms, no_call=False) -> None:
        cfg = self._reviewer_config
        if no_call or outcome is None:
            call = provisional.call or {"name": None, "arguments": {}}
            self._reviewer_turns.append({
                "step": step, "strategy": cfg.strategy, "prompt_version": cfg.prompt_version,
                "provisional_call": call, "final_call": call, "changed": False,
                "rounds": 0, "reviewer_calls": 0, "parse_failures": 0,
                "scores": None, "selected_index": None, "reviewer_wall_ms": review_ms,
                "no_tool_call": no_call, "strategy_failed": (outcome is None and not no_call),
            })
            return
        self._reviewer_turns.append({
            "step": step, "strategy": cfg.strategy, "prompt_version": cfg.prompt_version,
            "provisional_call": outcome.provisional.call, "final_call": outcome.chosen.call,
            "changed": outcome.changed, "rounds": outcome.rounds,
            "reviewer_calls": outcome.reviewer_calls, "parse_failures": outcome.parse_failures,
            "scores": outcome.scores, "selected_index": outcome.selected_index,
            "reviewer_wall_ms": review_ms, "no_tool_call": False, "strategy_failed": False,
        })


# ── Module-level helpers ──────────────────────────────────────────────────────

def _first_call(resp: Any) -> Optional[Dict[str, Any]]:
    try:
        tcs = resp.choices[0].message.tool_calls or []
    except Exception:
        tcs = []
    if not tcs:
        return None
    tc = tcs[0]
    raw = tc.function.arguments
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        args = {}
    if not isinstance(args, dict):
        args = {"value": args}
    return {"name": tc.function.name, "arguments": args}


def _format_tool_docs(tool_schemas: List[Dict[str, Any]]) -> str:
    lines = []
    for s in tool_schemas or []:
        fn = s.get("function", s) if isinstance(s, dict) else {}
        name = fn.get("name")
        desc = (fn.get("description") or "").strip().replace("\n", " ")
        params = ", ".join((fn.get("parameters", {}) or {}).get("properties", {}).keys())
        lines.append(f"- {name}({params}): {desc}")
    return "\n".join(lines)


def _format_context(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
        text = (content or "").strip() if isinstance(content, str) else str(content or "")
        tcs = m.get("tool_calls")
        if tcs:
            names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tcs)
            text = (text + f"  [tool_calls: {names}]").strip()
        if text:
            lines.append(f"[{role}] {text}")
    return "\n".join(lines)
