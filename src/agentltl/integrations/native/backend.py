"""
agentltl/integrations/native/backend.py — hand-coded OpenAI-compatible agent.

``NativeOpenAIAgent`` is a from-scratch tool-calling agent loop that talks to any
OpenAI-style ``/chat/completions`` server (OpenAI, vLLM, or the HuggingFace router
``https://router.huggingface.co/v1`` where the inference provider is encoded as a
``model:provider`` suffix, e.g. ``Qwen/Qwen3.5-397B-A17B:novita``).

It is a drop-in agentltl backend:

* ``run(task) -> {"answer","metrics","error"}`` — single-shot (resets state).
* ``run(task, constraints, severities)`` + ``get_constraint_status()`` — the
  ``AgentWithConstraints`` contract, with **runtime** enforcement via
  :class:`agentltl._enforcement_engine.ConstraintEnforcer`.
* ``run_turn(task)`` + ``reset()`` — persistent-conversation primitives that
  ``MultiTurnAgent`` drives so the model SEES prior turns (continuity) and the
  enforcer + trace span the whole session.

The emitted ``metrics`` dict matches the smolagents/langchain backends exactly, so
``verify_trace`` consumes the native trace with no adaptation.

Tools are smolagents ``Tool`` objects (``.name/.description/.inputs``, callable as
``tool(**args)``); ``.inputs`` is converted to an OpenAI function schema here.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from agentltl._enforcement_engine import ConstraintEnforcer
from agentltl.enforcement import ConstraintSeverity, ConstraintViolationError

logger = logging.getLogger(__name__)

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

DEFAULT_SYSTEM_PROMPT = (
    "You are a capable agent that completes the user's task by calling the provided "
    "tools. Call one or more tools per step; inspect their results; then continue. "
    "When the task is fully done, reply with a short final message and DO NOT call "
    "any tool. Do not call tools that are unrelated to the task."
)

# smolagents `.inputs` type -> JSON-schema type. "any" -> omit type entirely.
_TYPE_MAP = {
    "string": "string", "boolean": "boolean", "integer": "integer",
    "number": "number", "object": "object", "array": "array", "null": "null",
    # tolerate a few non-canonical spellings that show up in tool docs
    "str": "string", "bool": "boolean", "int": "integer", "float": "number",
    "dict": "object", "list": "array",
}


def tool_to_openai(tool: Any) -> Dict[str, Any]:
    """Convert a smolagents ``Tool`` into an OpenAI function-calling schema."""
    props: Dict[str, Any] = {}
    required: List[str] = []
    for pname, pinfo in (getattr(tool, "inputs", {}) or {}).items():
        ptype = pinfo.get("type", "string")
        schema: Dict[str, Any] = {"description": pinfo.get("description", "") or pname}
        if isinstance(ptype, list):
            schema["type"] = [_TYPE_MAP.get(t, "string") for t in ptype]
        else:
            mapped = _TYPE_MAP.get(ptype)  # None for "any" -> leave type unconstrained
            if mapped is not None:
                schema["type"] = mapped
        props[pname] = schema
        if not pinfo.get("nullable", False):
            required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (getattr(tool, "description", "") or tool.name),
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


class NativeOpenAIAgent:
    """Hand-coded OpenAI-compatible tool-calling agent with optional runtime
    constraint enforcement. See module docstring."""

    def __init__(
        self,
        tools: Optional[List[Any]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        bill_to: Optional[str] = None,
        max_steps: int = 10,
        model_seed: Optional[int] = None,
        model_instance: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        # Cap on each completion. `None` means "ask the server for its default", which
        # against a gateway-fronted vLLM means generate-until-context-exhausted and an
        # APIConnectionError that looks like a network fault. AGENTLTL_MAX_TOKENS sets
        # the default so existing callers get a bounded request without changing.
        max_tokens: Optional[int] = None,
        # constraint params (used by AgentWithConstraints / MultiTurnAgent)
        constraints: Optional[List[Any]] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
        default_severity: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
        max_soft_attempts: int = 3,
        soft_block_mode: str = "cumulative",
        max_consecutive_soft_attempts: Optional[int] = None,
        nudge_max: int = 1,
        mcp_servers: Optional[Dict[str, Any]] = None,
        _skip_runtime_safety_check: bool = False,
        **_ignored: Any,
    ) -> None:
        if mcp_servers:
            logger.warning("NativeOpenAIAgent: mcp_servers is not supported and is ignored.")
        if _ignored:
            logger.debug("NativeOpenAIAgent ignoring extra kwargs: %s", list(_ignored))

        if max_tokens is None:
            _env = os.environ.get("AGENTLTL_MAX_TOKENS", "4096").strip()
            max_tokens = int(_env) if _env.isdigit() and int(_env) > 0 else None
        self._max_tokens: Optional[int] = max_tokens

        self._tools: List[Any] = list(tools or [])
        self._tools_by_name = {t.name: t for t in self._tools}
        self._tool_schemas = [tool_to_openai(t) for t in self._tools]
        self._max_steps = max_steps
        self._model_seed = model_seed
        self._temperature = temperature
        self._system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

        # Resolve model id + OpenAI client.
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL") or HF_ROUTER_BASE_URL
        resolved_provider = provider or os.getenv("HF_INFERENCE_PROVIDER")
        model_name = model or os.getenv("MODEL")
        if not model_name:
            raise ValueError("NativeOpenAIAgent requires a model name (arg `model` or $MODEL).")
        # HF router convention: encode provider as a `model:provider` suffix.
        if resolved_provider and ":" not in model_name and "router.huggingface.co" in self._base_url:
            model_name = f"{model_name}:{resolved_provider}"
        self._model = model_name

        if model_instance is not None:
            self._client = model_instance  # a prebuilt openai.OpenAI (or compatible) client
        else:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "NativeOpenAIAgent requires the `openai` package. "
                    "Install with: pip install 'agentltl[native]' (or pip install openai)."
                ) from e
            resolved_key = api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
            # Bill HF-router usage to an org via the X-HF-Bill-To header
            # (mirrors huggingface_hub's `bill_to`). Without this, usage is billed
            # to the personal account behind HF_TOKEN.
            default_headers: Dict[str, str] = {}
            resolved_bill_to = bill_to or os.getenv("HF_BILL_TO")
            if resolved_bill_to and "huggingface.co" in self._base_url:
                default_headers["X-HF-Bill-To"] = resolved_bill_to
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=resolved_key,
                default_headers=default_headers or None,
            )

        self._enforcer = ConstraintEnforcer(
            constraints=constraints,
            constraint_severities=constraint_severities,
            default_severity=default_severity,
            max_soft_attempts=max_soft_attempts,
            soft_block_mode=soft_block_mode,
            max_consecutive_soft_attempts=max_consecutive_soft_attempts,
            nudge_max=nudge_max,
        )
        self.reset()

    # ── State ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset conversation + enforcer + trace. Call once per session (the
        single-shot ``run`` calls it automatically; ``MultiTurnAgent`` calls it
        once before the first turn so turns share memory)."""
        self._messages: List[Dict[str, Any]] = (
            [{"role": "system", "content": self._system_prompt}] if self._system_prompt else []
        )
        self._all_tool_calls: List[Dict[str, Any]] = []
        self._all_steps: List[Dict[str, Any]] = []
        self._model_step = 0          # monotonic model-call index across the session
        self._turn_index = -1
        self._in_tokens = 0
        self._out_tokens = 0
        self._enforcer.reset()

    def get_constraint_status(self) -> Dict[str, Any]:
        return self._enforcer.get_constraint_status()

    # ── Public run API ──────────────────────────────────────────────────────────

    def run(
        self,
        task: str,
        constraints: Optional[List[Any]] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
    ) -> Dict[str, Any]:
        """Single-shot run (resets state). Matches the Agent / AgentWithConstraints
        contract."""
        self.reset()
        if constraints is not None:
            self._enforcer.set_constraints(constraints, constraint_severities)
        return self.run_turn(task)

    def run_turn(self, task: str) -> Dict[str, Any]:
        """Run ONE user turn against the PERSISTENT conversation (does not reset).
        Returns ``{"answer","metrics","error"}`` where metrics cover THIS turn
        (its executed tool calls, tagged with the turn index)."""
        self._turn_index += 1
        turn = self._turn_index
        self._messages.append({"role": "user", "content": task})

        turn_tool_calls: List[Dict[str, Any]] = []
        turn_steps: List[Dict[str, Any]] = []
        turn_in = turn_out = 0
        answer: Optional[str] = None

        try:
            for _ in range(self._max_steps):
                self._model_step += 1
                step_no = self._model_step
                resp = self._chat()
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    turn_in += getattr(usage, "prompt_tokens", 0) or 0
                    turn_out += getattr(usage, "completion_tokens", 0) or 0
                msg = resp.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []

                # Persist the assistant message (with any tool_calls) into history.
                self._messages.append(self._assistant_to_dict(msg, tool_calls))
                turn_steps.append({
                    "step_number": step_no, "turn": turn,
                    "type": "tool_call" if tool_calls else "reasoning",
                    "reasoning": msg.content,
                })

                if not tool_calls:
                    answer = msg.content          # no tool call => final answer
                    break

                # New LLM generation — advances the BLOCK_AND_WARN insistence
                # pointer so a call blocked in a prior generation becomes
                # eligible for override here (parallel duplicates in THIS
                # generation don't qualify).
                self._enforcer.begin_generation()

                for tc in tool_calls:
                    name = tc.function.name
                    args = self._parse_args(tc.function.arguments)
                    # Runtime enforcement (no-op when there are no constraints).
                    decision = self._enforcer.check(name, args, step_no)
                    if isinstance(decision, tuple) and decision[0] in (
                        "soft_block", "block_and_warn", "persistent_block",
                    ):
                        # Blocked: feed the violation/warning back as this call's
                        # tool result, do NOT execute, and keep it OUT of the
                        # executed trace.
                        self._messages.append(
                            {"role": "tool", "tool_call_id": tc.id, "content": decision[1]}
                        )
                        continue
                    # Allowed: execute the tool.
                    result_str = self._exec_tool(name, args)
                    self._messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result_str}
                    )
                    self._enforcer.record_completed(name, args, tc.id, result_str)
                    call_record = {
                        "step": step_no, "turn": turn, "tool_name": name,
                        "arguments": args, "id": tc.id,
                        "tool_result": result_str, "action_output": None,
                    }
                    turn_tool_calls.append(call_record)
            error = None
        except ConstraintViolationError as exc:
            # HARD_STOP / soft-block escalation aborts the run.
            error = exc.dict()
        except Exception as exc:  # surface other failures loudly, with partial metrics
            import traceback as _tb
            logger.error("NativeOpenAIAgent run_turn failed: %s\n%s", exc, _tb.format_exc())
            error = str(exc)

        # Commit this turn's accumulation into the session totals.
        self._all_tool_calls.extend(turn_tool_calls)
        self._all_steps.extend(turn_steps)
        self._in_tokens += turn_in
        self._out_tokens += turn_out

        metrics = self._build_metrics(turn_tool_calls, turn_steps, turn_in, turn_out)
        return {"answer": answer, "metrics": metrics, "error": error}

    def session_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics across all turns since the last reset (continuous,
        turn-tagged trace). Used by MultiTurnAgent."""
        return self._build_metrics(
            self._all_tool_calls, self._all_steps, self._in_tokens, self._out_tokens
        )

    # ── Internals ────────────────────────────────────────────────────────────────

    def _chat(self) -> Any:
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
        }
        if self._tool_schemas:
            kwargs["tools"] = self._tool_schemas
            kwargs["tool_choice"] = "auto"
        if self._model_seed is not None:
            kwargs["seed"] = self._model_seed
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        # Bound the completion, as a GUARD rather than as a cure. An unbounded request
        # lets the server generate until the context window is exhausted, and behind a
        # gateway with a read deadline that surfaces as `APIConnectionError` -- which
        # reads as a network fault. A sweep put the deadline at ~50s: max_tokens of
        # 256/512/1024/2048 returned in 5-34s and 4096/unbounded both failed at
        # exactly 50.0s.
        #
        # It is worth setting, but it does not fix the underlying problem and 4096 is
        # itself above the ceiling when a reasoning model is left reasoning. The real
        # cause of the failures this was first written for was chain-of-thought left
        # ON, spending ~1845 tokens before the first tool call; disabling it gave 451
        # tokens in 8.4s. Bound the request AND keep generations short.
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        return self._client.chat.completions.create(**kwargs)

    @staticmethod
    def _assistant_to_dict(msg: Any, tool_calls: List[Any]) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        return d

    @staticmethod
    def _parse_args(raw: Any) -> Dict[str, Any]:
        if raw is None or raw == "":
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _exec_tool(self, name: str, args: Dict[str, Any]) -> str:
        tool = self._tools_by_name.get(name)
        if tool is None:
            return f"ERROR: unknown tool '{name}'."
        try:
            result = tool(**args)
        except Exception as e:  # return error as observation so the model can recover
            return f"ERROR: {type(e).__name__}: {e}"
        return result if isinstance(result, str) else json.dumps(result, default=str)

    def _build_metrics(
        self,
        tool_calls: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        in_tok: int,
        out_tok: int,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "num_steps": len(steps),
            "num_tool_calls": len(tool_calls),
            "tool_calls": tool_calls,
            "steps": steps,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        }
        # Constraint fields (parity with the other backends).
        status = self._enforcer.get_constraint_status()
        metrics["run_status"] = status["status"]
        metrics["stopped_by_constraint"] = status["stopped_by"]
        metrics["constraint_violations"] = status["violations"]
        metrics["constraint_checks"] = status["constraint_checks"]
        metrics["completed_trace"] = status["completed_trace"]
        return metrics
