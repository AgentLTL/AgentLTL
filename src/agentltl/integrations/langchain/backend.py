"""
agentltl/integrations/langchain/backend.py – LangChain constrained agent backend.

``LangChainConstrainedBackend`` wraps LangChain's ``create_agent()`` together
with :class:`ConstraintEnforcementMiddleware` to provide the same
``run()`` / ``get_constraint_status()`` interface as the smolagents backend.

This module is consumed by ``agentltl.agents.AgentWithConstraints`` when
``backend="langchain"`` is requested.  Users can also instantiate it directly
if they want to bypass the dispatch layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agentltl.enforcement import (
    ConstraintSeverity,
    ConstraintViolationError,
    SoftBlockMode,
)
from .constrained_agent import ConstraintEnforcementMiddleware

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_lc_model(model: Any) -> Any:
    """Convert a string model name to a LangChain ``BaseChatModel`` instance.

    Already a ``BaseChatModel`` → returned as-is.
    String heuristics:
      * ``"gpt-*"`` or ``"openai:*"`` → :class:`ChatOpenAI`
      * ``"claude-*"`` or ``"anthropic:*"`` → :class:`ChatAnthropic`
      * Anything else → :class:`ChatOpenAI` (uses ``OPENAI_API_KEY`` env var)

    Users can bypass this entirely by passing ``model_instance=<BaseChatModel>``
    to :class:`LangChainConstrainedBackend`.
    """
    if model is None:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI()
        except ImportError:
            raise ImportError(
                "No model specified and langchain-openai is not installed. "
                "Either pass model_instance=<BaseChatModel> or install "
                "langchain-openai."
            )

    # Already a model object.
    try:
        from langchain_core.language_models import BaseChatModel
        if isinstance(model, BaseChatModel):
            return model
    except ImportError:
        pass

    if not isinstance(model, str):
        return model  # assume it's a compatible model object

    # Strip provider prefix if present (e.g. "openai:gpt-4o" → "gpt-4o").
    name = model
    if ":" in name:
        prefix, name = name.split(":", 1)
    else:
        prefix = ""

    if prefix == "anthropic" or name.startswith("claude"):
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=name)
        except ImportError:
            raise ImportError(
                f"Model '{model}' looks like an Anthropic model but "
                "langchain-anthropic is not installed."
            )

    # Default: OpenAI-compatible.
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name)
    except ImportError:
        raise ImportError(
            f"Model '{model}' appears to be an OpenAI model but "
            "langchain-openai is not installed."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metrics(
    messages: List[Any],
    middleware: ConstraintEnforcementMiddleware,
) -> Dict[str, Any]:
    """Extract run metrics from LangGraph state messages + middleware state.

    Note: LangChain does not expose per-step timings in state messages;
    callbacks are needed for that.  Token counts are available via
    ``AIMessage.usage_metadata`` (LangChain 0.3+).
    """
    from langchain_core.messages import AIMessage, ToolMessage as LCToolMessage

    tool_calls: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    step_num = 0

    for msg in messages:
        if isinstance(msg, AIMessage):
            step_num += 1
            usage = getattr(msg, "usage_metadata", None) or {}
            in_tok = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
            out_tok = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
            input_tokens += in_tok
            output_tokens += out_tok

            step: Dict[str, Any] = {
                "step_number": step_num,
                "type": "reasoning",
                "reasoning": msg.content if isinstance(msg.content, str) else str(msg.content),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }

            if msg.tool_calls:
                step["type"] = "tool_call"
                for tc in msg.tool_calls:
                    tc_name = tc.get("name", "unknown")
                    tc_args = tc.get("args", {})
                    tc_id = tc.get("id", "")
                    step["tool_name"] = tc_name
                    step["tool_args"] = tc_args
                    step["tool_id"] = tc_id
                    tool_calls.append(
                        {
                            "step": step_num,
                            "tool_name": tc_name,
                            "arguments": tc_args,
                            "id": tc_id,
                        }
                    )

            steps.append(step)

        elif isinstance(msg, LCToolMessage):
            # Attach result to most recent tool_call entry with matching id.
            for tc in reversed(tool_calls):
                if tc.get("id") == msg.tool_call_id or not tc.get("tool_result"):
                    tc["tool_result"] = msg.content
                    break

    status = middleware.get_constraint_status()
    metrics: Dict[str, Any] = {
        "num_steps": len(steps),
        "num_tool_calls": len(tool_calls),
        "tool_calls": tool_calls,
        "steps": steps,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "run_status": status["status"],
        "stopped_by_constraint": status["stopped_by"],
        "constraint_violations": status["violations"],
        "constraint_checks": status["constraint_checks"],
        "completed_trace": status["completed_trace"],
    }
    return metrics


def _extract_answer(messages: List[Any]) -> Optional[str]:
    """Return the last AI message content as the agent answer."""
    from langchain_core.messages import AIMessage

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Backend class
# ─────────────────────────────────────────────────────────────────────────────

class LangChainConstrainedBackend:
    """LangChain backend for constrained agent execution.

    Wraps LangChain's ``create_agent()`` with
    :class:`ConstraintEnforcementMiddleware` and exposes the same
    ``run()`` / ``get_constraint_status()`` interface as the smolagents
    backend.

    Args:
        tools:                 LangChain :class:`BaseTool` instances.
        constraints:           FOLTL constraints to enforce.
        constraint_severities: Per-constraint severity overrides.
        default_severity:      Fallback severity (default: ``HARD_STOP``).
        max_soft_attempts:     Cumulative / hybrid SOFT_BLOCK cap per constraint.
        soft_block_mode:       Escalation strategy.
        max_consecutive_soft_attempts: Consecutive escalation cap.
        model:                 String model name (heuristic resolution) or
                               ``BaseChatModel`` instance.
        model_instance:        Pre-constructed ``BaseChatModel``.  Overrides
                               *model* when supplied.
        max_steps:             Recursion depth for LangGraph
                               (``recursion_limit = max_steps * 2 + 4``).
        system_prompt:         Optional system prompt passed to ``create_agent``.
        mcp_servers:           Accepted but currently ignored with a warning.
                               Pass tools directly for LangChain until a
                               LangChain MCP adapter layer is added.
    """

    def __init__(
        self,
        tools: Optional[List[Any]] = None,
        constraints: Optional[List[Any]] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
        default_severity: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
        max_soft_attempts: int = 3,
        soft_block_mode: str = "cumulative",
        max_consecutive_soft_attempts: Optional[int] = None,
        model: Optional[Any] = None,
        model_instance: Optional[Any] = None,
        max_steps: int = 10,
        system_prompt: Optional[str] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        strict_runtime_safety: bool = False,
        _skip_runtime_safety_check: bool = False,
        **_ignored: Any,
    ) -> None:
        if mcp_servers:
            logger.warning(
                "LangChainConstrainedBackend: mcp_servers is not yet supported "
                "for the langchain backend and will be ignored. "
                "Pass tools directly or wait for a langchain-mcp-adapters integration."
            )

        self._middleware = ConstraintEnforcementMiddleware(
            constraints=constraints,
            constraint_severities=constraint_severities,
            default_severity=default_severity,
            max_soft_attempts=max_soft_attempts,
            soft_block_mode=soft_block_mode,
            max_consecutive_soft_attempts=max_consecutive_soft_attempts,
            strict_runtime_safety=strict_runtime_safety,
            _skip_runtime_safety_check=_skip_runtime_safety_check,
        )

        model_obj = model_instance if model_instance is not None else _resolve_lc_model(model)

        from langchain.agents import create_agent

        self._agent = create_agent(
            model=model_obj,
            tools=list(tools or []),
            **({"system_prompt": system_prompt} if system_prompt is not None else {}),
            middleware=[self._middleware.as_middleware()],
        )
        self._max_steps = max_steps

    def run(
        self,
        task: str,
        constraints: Optional[List[Any]] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
    ) -> Dict[str, Any]:
        """Run *task* and return ``{"answer", "metrics", "error"}``.

        Args:
            task:                  The task string for the agent.
            constraints:           Per-run constraint override.
            constraint_severities: Per-run severity override.

        Returns:
            ``{"answer", "metrics", "error"}`` with constraint fields in
            ``metrics``.  The shape is identical to the smolagents backend.
        """
        from langchain_core.messages import HumanMessage

        self._middleware.reset()
        if constraints is not None:
            self._middleware.set_constraints(constraints, constraint_severities)

        try:
            state = self._agent.invoke(
                {"messages": [HumanMessage(content=task)]},
                config={"recursion_limit": self._max_steps * 2 + 4},
            )
            messages = state.get("messages", [])
            answer = _extract_answer(messages)
            metrics = _extract_metrics(messages, self._middleware)
            return {"answer": answer, "metrics": metrics, "error": None}

        except ConstraintViolationError as exc:
            metrics = _extract_metrics([], self._middleware)
            return {"answer": None, "metrics": metrics, "error": exc.dict()}

        except Exception as exc:
            import traceback as _tb
            err_msg = str(exc)
            logger.error("LangChainConstrainedBackend run failed: %s\n%s",
                         err_msg, _tb.format_exc())
            metrics = _extract_metrics([], self._middleware)
            return {"answer": None, "metrics": metrics, "error": err_msg}

    def get_constraint_status(self) -> Dict[str, Any]:
        """Return the constraint enforcement status for the last run."""
        return self._middleware.get_constraint_status()
