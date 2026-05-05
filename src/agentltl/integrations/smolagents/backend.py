"""
agentltl/integrations/smolagents/backend.py – smolagents backend implementations.

This module contains the smolagents-specific implementations of the agent
classes.  Public names are exported from ``agentltl.agents`` via a dispatch
layer; the classes here are prefixed with ``SmolAgents`` to make the
separation explicit.

See ``agentltl.integrations.smolagents.agents`` for the old (backward-compat)
names, and ``agentltl.agents`` for the backend-agnostic public API.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from smolagents import (
    InferenceClientModel,
    MCPClient,
    OpenAIServerModel,
    Tool,
    ToolCallingAgent,
)

from agentltl.enforcement import ConstraintSeverity
from .constrained_agent import ToolCallingAgentWithConstraints

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Type alias
# ─────────────────────────────────────────────────────────────────────────────

# A server config dict as accepted by smolagents MCPClient.
# Minimum required keys depend on transport:
#   HTTP  → {"url": str, "transport": "streamable-http" | "sse"}
#            optional: {"headers": dict, "timeout": float}
#   stdio → {"command": str, "args": list[str], "env": dict}
MCPServerConfig = Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def _create_model(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model_seed: Optional[int] = None,
) -> Any:
    """Create and return an LLM model instance.

    Selects the backend based on the ``MODEL_TYPE`` environment variable:

    * ``vllm`` / ``openai_server`` / ``openai``
      → :class:`OpenAIServerModel` pointing at ``$VLLM_BASE_URL``.
    * ``hf_inference`` (default)
      → :class:`InferenceClientModel` via the HuggingFace Inference API.

    Args:
        model:       Model name / ID.  Falls back to ``$MODEL``, then the
                     built-in default.
        api_key:     API key for the model provider.  Falls back to
                     ``$HF_TOKEN``.
        provider:    HF Inference provider name.  Falls back to
                     ``$HF_INFERENCE_PROVIDER``, then ``"novita"``.
        model_seed:  Optional seed forwarded to the model for reproducible
                     sampling (supported by both backends).
    """
    model_name = model or os.getenv("MODEL") or "Qwen/Qwen3-Next-80B-A3B-Instruct"
    model_type = os.getenv("MODEL_TYPE", "hf_inference").lower()

    if model_type in ("vllm", "openai_server", "openai"):
        vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        vllm_api_key = os.getenv("VLLM_API_KEY", "dummy")
        extra: Dict[str, Any] = {"tool_choice": "any"}
        if model_seed is not None:
            extra["seed"] = model_seed
        return OpenAIServerModel(
            model_id=model_name,
            api_base=vllm_base_url,
            api_key=vllm_api_key,
            **extra,
        )

    # HuggingFace Inference API (default)
    resolved_api_key = api_key or os.getenv("HF_TOKEN")
    resolved_provider = provider or os.getenv("HF_INFERENCE_PROVIDER") or "novita"
    model_kwargs: Dict[str, Any] = {
        "model_id": model_name,
        "token": resolved_api_key,
        "provider": resolved_provider,
    }
    bill_to_org = os.environ.get("HF_BILL_TO")
    if bill_to_org:
        model_kwargs["bill_to"] = bill_to_org
    if model_seed is not None:
        model_kwargs["seed"] = model_seed
    return InferenceClientModel(**model_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# MCP connectivity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _connect_mcp_servers(
    mcp_servers: Dict[str, MCPServerConfig],
) -> Tuple[List[MCPClient], List[Tool]]:
    """Open connections to all MCP servers and collect their tools.

    Each entry in *mcp_servers* is opened as a separate :class:`MCPClient`
    so that a failure in one server does not prevent the others from loading.

    Args:
        mcp_servers: ``{human_readable_name: server_config_dict}`` where the
                     config dict is passed directly to :class:`MCPClient`.

    Returns:
        A ``(clients, tools)`` tuple where *clients* is the list of open
        :class:`MCPClient` instances (kept for cleanup in ``__del__``) and
        *tools* is the flat ordered list of :class:`Tool` objects exposed
        across all servers, in the iteration order of *mcp_servers*.
    """
    clients: List[MCPClient] = []
    all_tools: List[Tool] = []

    for name, server_config in mcp_servers.items():
        try:
            client = MCPClient(server_config, structured_output=False)
            tools = client.__enter__()
            clients.append(client)
            all_tools.extend(tools)
            logger.debug(
                "MCP server %r connected: %d tool(s) available.",
                name,
                len(tools),
            )
        except Exception:
            logger.warning(
                "Failed to connect to MCP server %r — skipping. "
                "Check the server URL and transport settings.",
                name,
                exc_info=True,
            )

    return clients, all_tools


def _disconnect_mcp_clients(clients: List[MCPClient]) -> None:
    """Best-effort cleanup of all open MCP client connections."""
    for client in clients:
        try:
            client.__exit__(None, None, None)
        except Exception:
            logger.debug("Error while closing MCP client.", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# Base agent
# ─────────────────────────────────────────────────────────────────────────────

class SmolAgentsAgent:
    """Base smolagents agent backed by a :class:`ToolCallingAgent`.

    Takes a list of :class:`Tool` instances directly — no MCP connectivity.
    All shared logic (metrics extraction, run loop, resource cleanup) lives
    here so subclasses can inherit it without duplication.

    Args:
        tools:          Tool instances to expose to the agent.  ``None`` or
                        an empty list creates an agent with no tools.
        model:          Model name / ID.  See :func:`_create_model`.
        api_key:        API key for the model provider.
        provider:       HF Inference provider name.
        max_steps:      Maximum agent steps before forced termination.
        model_seed:     Optional seed for reproducible sampling.
        model_instance: Pre-constructed model object.  When supplied,
                        *model*, *api_key*, *provider*, and *model_seed*
                        are ignored.
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_steps: int = 10,
        model_seed: Optional[int] = None,
        model_instance: Optional[Any] = None,
    ) -> None:
        self._model = (
            model_instance
            if model_instance is not None
            else _create_model(model, api_key, provider, model_seed)
        )
        # Subclasses that add MCP clients store them here for cleanup.
        self._mcp_clients: List[MCPClient] = []
        self.agent = ToolCallingAgent(
            tools=list(tools or []),
            model=self._model,
            max_steps=max_steps,
        )

    # ── Metrics extraction ───────────────────────────────────────────────────

    def _extract_metrics_from_steps(self, steps: List[Any]) -> Dict[str, Any]:
        """Extract per-step metrics from a smolagents execution step list.

        Captures: step type, reasoning text, thinking (chain-of-thought),
        tool calls, observations, timings, token usage, model input context.
        See ``docs/reference.md`` for the full schema.
        """

        def get_value(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def role_str(role: Any) -> Optional[str]:
            if role is None:
                return None
            return role.value if hasattr(role, "value") else str(role)

        def serialize_tool_calls_from_message(tool_calls: Any) -> Optional[List[Dict]]:
            if not tool_calls:
                return None
            result = []
            for tc in tool_calls:
                func = get_value(tc, "function")
                if func:
                    tc_name = (
                        get_value(func, "name")
                        if isinstance(func, dict)
                        else getattr(func, "name", None)
                    )
                    tc_args = (
                        get_value(func, "arguments")
                        if isinstance(func, dict)
                        else getattr(func, "arguments", None)
                    )
                    tc_desc = (
                        get_value(func, "description")
                        if isinstance(func, dict)
                        else getattr(func, "description", None)
                    )
                else:
                    tc_name = get_value(tc, "name")
                    tc_args = get_value(tc, "arguments")
                    tc_desc = None
                entry: Dict[str, Any] = {
                    "id": get_value(tc, "id"),
                    "name": tc_name,
                    "arguments": tc_args,
                }
                if tc_desc:
                    entry["description"] = tc_desc
                result.append(entry)
            return result

        def serialize_content(content: Any) -> Any:
            if content is None:
                return None
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                serialized = []
                for block in content:
                    if isinstance(block, dict):
                        serialized.append(dict(block))
                    else:
                        block_dict: Dict[str, Any] = {}
                        for field in (
                            "type", "text", "thinking", "image_url",
                            "tool_use_id", "tool_name", "id", "input",
                        ):
                            val = get_value(block, field)
                            if val is not None:
                                block_dict[field] = val
                        serialized.append(block_dict)
                return serialized
            return str(content)

        def serialize_message(msg: Any) -> Dict[str, Any]:
            serialized: Dict[str, Any] = {
                "role": role_str(get_value(msg, "role")),
                "content": serialize_content(get_value(msg, "content")),
            }
            tcs = serialize_tool_calls_from_message(get_value(msg, "tool_calls"))
            if tcs is not None:
                serialized["tool_calls"] = tcs
            tu = get_value(msg, "token_usage")
            if tu is not None:
                serialized["token_usage"] = (
                    tu
                    if isinstance(tu, dict)
                    else {
                        "input_tokens": getattr(tu, "input_tokens", 0),
                        "output_tokens": getattr(tu, "output_tokens", 0),
                        "total_tokens": getattr(tu, "total_tokens", 0),
                    }
                )
            return serialized

        def extract_text_from_content(content: Any) -> str:
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    block_type = (
                        get_value(block, "type")
                        if isinstance(block, dict)
                        else getattr(block, "type", None)
                    )
                    if block_type in ("text", None):
                        t = (
                            get_value(block, "text")
                            if isinstance(block, dict)
                            else getattr(block, "text", None)
                        )
                        if t:
                            parts.append(str(t))
                return " ".join(parts)
            return str(content)

        def parse_input_messages(messages: List[Any]) -> Dict[str, Any]:
            parsed: Dict[str, Any] = {
                "num_messages": len(messages),
                "approx_chars": 0,
                "system_prompts": [],
                "task": None,
                "history": [],
            }
            for msg in messages:
                r = role_str(get_value(msg, "role"))
                content = get_value(msg, "content")
                tool_calls = get_value(msg, "tool_calls")
                text = extract_text_from_content(content)
                parsed["approx_chars"] += len(text)

                if r == "system":
                    parsed["system_prompts"].append(text)
                elif r == "user" and parsed["task"] is None:
                    parsed["task"] = text
                elif r == "tool-response":
                    parsed["history"].append(
                        {"role": r, "type": "tool_result", "text": text, "tool_calls": None}
                    )
                elif r in ("assistant", "tool-call"):
                    parsed_tcs = serialize_tool_calls_from_message(tool_calls)
                    parsed["history"].append(
                        {
                            "role": r,
                            "type": "tool_call" if parsed_tcs else "reasoning",
                            "text": text,
                            "tool_calls": parsed_tcs,
                        }
                    )
                else:
                    parsed["history"].append(
                        {"role": r, "type": "message", "text": text, "tool_calls": None}
                    )
            return parsed

        def extract_thinking(
            model_output_str: Optional[str],
            model_output_message: Any,
        ) -> Optional[str]:
            if model_output_str:
                match = re.search(r"<think>(.*?)</think>", model_output_str, re.DOTALL)
                if match:
                    return match.group(1).strip()
            if model_output_message:
                content = get_value(model_output_message, "content")
                if isinstance(content, list):
                    for block in content:
                        block_type = (
                            get_value(block, "type")
                            if isinstance(block, dict)
                            else getattr(block, "type", None)
                        )
                        if block_type in ("thinking", "reasoning"):
                            t = (
                                (
                                    get_value(block, "thinking")
                                    if isinstance(block, dict)
                                    else getattr(block, "thinking", None)
                                )
                                or (
                                    get_value(block, "text")
                                    if isinstance(block, dict)
                                    else getattr(block, "text", None)
                                )
                                or ""
                            )
                            return t.strip() if t else None
            return None

        def extract_tool_call_info(tc: Any) -> Dict[str, Any]:
            tool_name: Optional[str] = None
            tool_args: Any = None
            func = get_value(tc, "function")
            if func:
                tool_name = (
                    get_value(func, "name")
                    if isinstance(func, dict)
                    else getattr(func, "name", None)
                )
                tool_args = (
                    get_value(func, "arguments")
                    if isinstance(func, dict)
                    else getattr(func, "arguments", None)
                )
            if not tool_name:
                for name_field in ("name", "tool_name", "function_name", "tool"):
                    tool_name = get_value(tc, name_field)
                    if tool_name:
                        break
            if tool_args is None:
                for arg_field in ("arguments", "args", "parameters", "inputs", "input"):
                    tool_args = get_value(tc, arg_field)
                    if tool_args is not None:
                        break
            return {
                "name": tool_name or "unknown",
                "arguments": tool_args if tool_args is not None else {},
                "id": get_value(tc, "id"),
            }

        metrics: Dict[str, Any] = {
            "num_steps": len(steps),
            "num_tool_calls": 0,
            "tool_calls": [],
            "steps": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        for i, step in enumerate(steps):
            step_info: Dict[str, Any] = {
                "step_number": i + 1,
                "type": None,
                "reasoning": None,
                "thinking": None,
                "tool_name": None,
                "tool_args": None,
                "tool_id": None,
                "tool_result": None,
                "action_output": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "start_time": None,
                "end_time": None,
                "duration": None,
                "error": None,
                "model_input_raw": None,
                "model_input_parsed": None,
            }

            # Timings
            timing = get_value(step, "timing")
            if timing is not None:
                step_info["start_time"] = get_value(timing, "start_time")
                step_info["end_time"] = get_value(timing, "end_time")
                step_info["duration"] = get_value(timing, "duration")
            else:
                for time_field in ("start_time", "end_time", "duration"):
                    val = get_value(step, time_field)
                    if val is not None:
                        step_info[time_field] = val

            # Token usage
            step_tokens = get_value(step, "token_usage")
            if step_tokens:
                input_tok = (
                    step_tokens.get("input_tokens", 0)
                    if isinstance(step_tokens, dict)
                    else getattr(step_tokens, "input_tokens", 0)
                )
                output_tok = (
                    step_tokens.get("output_tokens", 0)
                    if isinstance(step_tokens, dict)
                    else getattr(step_tokens, "output_tokens", 0)
                )
                step_info["input_tokens"] = input_tok
                step_info["output_tokens"] = output_tok
                metrics["input_tokens"] += input_tok
                metrics["output_tokens"] += output_tok

            # Tool calls
            tool_calls = get_value(step, "tool_calls") or []
            pending_tool_calls: List[Dict[str, Any]] = []
            if tool_calls:
                step_info["type"] = "tool_call"
                for tc_idx, tc in enumerate(tool_calls):
                    metrics["num_tool_calls"] += 1
                    tc_info = extract_tool_call_info(tc)
                    if tc_idx == 0:
                        step_info["tool_name"] = tc_info["name"]
                        step_info["tool_args"] = tc_info["arguments"]
                        step_info["tool_id"] = tc_info["id"]
                    pending_tool_calls.append(
                        {
                            "step": i + 1,
                            "tool_name": tc_info["name"],
                            "arguments": tc_info["arguments"],
                            "id": tc_info["id"],
                        }
                    )
                if len(pending_tool_calls) > 1:
                    step_info["step_tool_calls"] = [
                        {"tool_name": p["tool_name"], "tool_args": p["arguments"]}
                        for p in pending_tool_calls
                    ]

            # Observations / results
            observations = get_value(step, "observations")
            if observations:
                step_info["tool_result"] = str(observations)

            action_output = get_value(step, "action_output")
            if action_output is not None:
                step_info["action_output"] = str(action_output)

            for entry in pending_tool_calls:
                entry["tool_result"] = step_info.get("tool_result")
                entry["action_output"] = step_info.get("action_output")
                metrics["tool_calls"].append(entry)

            # Reasoning / thinking
            model_output = get_value(step, "model_output")
            model_output_message = get_value(step, "model_output_message")
            if model_output:
                model_output_str = str(model_output)
                if not step_info["type"]:
                    step_info["type"] = "reasoning"
                step_info["reasoning"] = model_output_str
                step_info["thinking"] = extract_thinking(model_output_str, model_output_message)
            elif model_output_message and not step_info["thinking"]:
                step_info["thinking"] = extract_thinking(None, model_output_message)

            # Errors
            error = get_value(step, "error")
            if error:
                step_info["type"] = "error"
                step_info["error"] = str(error)

            # Full model input context
            model_input_messages = get_value(step, "model_input_messages") or []
            if model_input_messages:
                step_info["model_input_raw"] = [
                    serialize_message(m) for m in model_input_messages
                ]
                step_info["model_input_parsed"] = parse_input_messages(model_input_messages)

            metrics["steps"].append(step_info)

        metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
        return metrics

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self, task: str) -> Dict[str, Any]:
        """Run *task* and return ``{"answer", "metrics", "error"}``.

        On success ``error`` is ``None`` and ``answer`` holds the agent's
        final answer string.  On failure ``answer`` is ``None`` and
        ``error`` holds the exception message; partial metrics are still
        returned if any steps completed before the error.
        """
        try:
            result = self.agent.run(task, return_full_result=True)
        except Exception as exc:
            import traceback as _tb

            err_msg = str(exc)
            logger.error(
                "Agent run failed: %s\n%s", err_msg, _tb.format_exc()
            )
            partial_steps: List[Any] = []
            try:
                partial_steps = list(getattr(self.agent.memory, "steps", []))
            except Exception:
                pass

            if partial_steps:
                partial_metrics = self._extract_metrics_from_steps(partial_steps)
                partial_metrics["total_tokens"] = (
                    partial_metrics["input_tokens"] + partial_metrics["output_tokens"]
                )
                return {"answer": None, "metrics": partial_metrics, "error": err_msg}

            empty_metrics: Dict[str, Any] = {
                "steps": [],
                "tool_calls": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "num_steps": 0,
                "num_tool_calls": 0,
            }
            return {"answer": None, "metrics": empty_metrics, "error": err_msg}

        answer = result.output if hasattr(result, "output") else result
        steps = getattr(result, "steps", [])
        metrics = self._extract_metrics_from_steps(steps)

        # Sanity-check token counts against the RunResult aggregate
        token_usage = getattr(result, "token_usage", None)
        if token_usage is not None:
            ru_input = (
                getattr(token_usage, "input_tokens", None)
                if not isinstance(token_usage, dict)
                else token_usage.get("input_tokens")
            )
            ru_output = (
                getattr(token_usage, "output_tokens", None)
                if not isinstance(token_usage, dict)
                else token_usage.get("output_tokens")
            )
            if ru_input is not None and ru_input != metrics["input_tokens"]:
                logger.warning(
                    "Token count mismatch: RunResult.input_tokens=%d, "
                    "per-step sum=%d.",
                    ru_input,
                    metrics["input_tokens"],
                )
            if ru_output is not None and ru_output != metrics["output_tokens"]:
                logger.warning(
                    "Token count mismatch: RunResult.output_tokens=%d, "
                    "per-step sum=%d.",
                    ru_output,
                    metrics["output_tokens"],
                )

        metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
        return {"answer": answer, "metrics": metrics, "error": None}

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def __del__(self) -> None:
        _disconnect_mcp_clients(getattr(self, "_mcp_clients", []))


# ─────────────────────────────────────────────────────────────────────────────
# MCP-enabled agent
# ─────────────────────────────────────────────────────────────────────────────

class SmolAgentsAgentWithAdditionalTools(SmolAgentsAgent):
    """Agent with multi-server MCP connectivity and optional local tools.

    Connects to one or more MCP servers and merges their tools with any
    directly-provided :class:`Tool` instances.  MCP tools appear first in
    the tool list; local tools are appended after.

    Args:
        tools:          Local :class:`Tool` instances appended after MCP tools.
        mcp_servers:    ``{name: server_config_dict}`` mapping.  Each entry is
                        opened as an independent :class:`MCPClient`; a failure
                        in one server is logged and the rest continue loading.
                        Server config dicts are passed directly to
                        :class:`MCPClient` — see module docstring for examples.
        model:          Model name / ID.  See :func:`_create_model`.
        api_key:        API key for the model provider.
        provider:       HF Inference provider name.
        max_steps:      Maximum agent steps before forced termination.
        model_seed:     Optional seed for reproducible sampling.
        model_instance: Pre-constructed model object.  Overrides *model*,
                        *api_key*, *provider*, and *model_seed* when supplied.
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerConfig]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_steps: int = 10,
        model_seed: Optional[int] = None,
        model_instance: Optional[Any] = None,
    ) -> None:
        mcp_clients, mcp_tools = _connect_mcp_servers(mcp_servers or {})
        # MCP tools first so local overrides can shadow names if desired.
        all_tools: List[Tool] = mcp_tools + list(tools or [])

        super().__init__(
            tools=all_tools,
            model=model,
            api_key=api_key,
            provider=provider,
            max_steps=max_steps,
            model_seed=model_seed,
            model_instance=model_instance,
        )
        # Override the empty list set in SmolAgentsAgent.__init__.
        self._mcp_clients = mcp_clients


# ─────────────────────────────────────────────────────────────────────────────
# Sub-agent orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class SmolAgentsAgentWithSubAgents(SmolAgentsAgent):
    """Agent that spawns one :class:`SmolAgentsAgentWithAdditionalTools` per requirement.

    The coordinator itself is a plain :class:`SmolAgentsAgent` (coordinator
    tasks rarely need tools).  Each spawned sub-agent receives a fresh MCP
    connection pool and the configured local tools.

    Args:
        tools:               Local tools forwarded to every sub-agent.
        mcp_servers:         MCP server config dict forwarded to every sub-agent.
        model:               Model name / ID used by both coordinator and
                             sub-agents.
        api_key:             API key.
        provider:            HF Inference provider name.
        max_steps:           Coordinator max steps.
        sub_agent_max_steps: Max steps for each spawned sub-agent.
        model_seed:          Optional seed.
        model_instance:      Pre-constructed model shared by all agents.
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerConfig]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_steps: int = 10,
        sub_agent_max_steps: int = 15,
        model_seed: Optional[int] = None,
        model_instance: Optional[Any] = None,
    ) -> None:
        # Store sub-agent config before super().__init__ to make the object
        # consistent if _create_sub_agent is ever called early.
        self._sub_tools: List[Tool] = list(tools or [])
        self._sub_mcp_servers: Dict[str, MCPServerConfig] = dict(mcp_servers or {})
        self._sub_agent_max_steps = sub_agent_max_steps
        self._model_cfg = {
            "model": model,
            "api_key": api_key,
            "provider": provider,
            "model_seed": model_seed,
            "model_instance": model_instance,
        }

        # Coordinator has no tools — it only dispatches to sub-agents.
        super().__init__(
            tools=[],
            model=model,
            api_key=api_key,
            provider=provider,
            max_steps=max_steps,
            model_seed=model_seed,
            model_instance=model_instance,
        )

    def _create_sub_agent(self) -> SmolAgentsAgentWithAdditionalTools:
        """Spawn a fresh sub-agent with its own MCP connections."""
        return SmolAgentsAgentWithAdditionalTools(
            tools=list(self._sub_tools),
            mcp_servers=dict(self._sub_mcp_servers),
            max_steps=self._sub_agent_max_steps,
            **self._model_cfg,
        )

    def _aggregate_metrics(
        self, sub_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate metrics from all sub-agent results into a single dict."""
        aggregated: Dict[str, Any] = {
            "total_requirements": len(sub_results),
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_steps": 0,
            "total_tool_calls": 0,
            "per_requirement_metrics": [],
            "all_tool_calls": [],
        }

        for i, result in enumerate(sub_results):
            m = result.get("metrics", {})
            aggregated["total_input_tokens"] += m.get("input_tokens", 0)
            aggregated["total_output_tokens"] += m.get("output_tokens", 0)
            aggregated["total_tokens"] += m.get("total_tokens", 0)
            aggregated["total_steps"] += m.get("num_steps", 0)
            aggregated["total_tool_calls"] += m.get("num_tool_calls", 0)
            aggregated["per_requirement_metrics"].append(
                {
                    "requirement_index": i,
                    "requirement_id": result.get("requirement_id"),
                    "input_tokens": m.get("input_tokens", 0),
                    "output_tokens": m.get("output_tokens", 0),
                    "total_tokens": m.get("total_tokens", 0),
                    "num_steps": m.get("num_steps", 0),
                    "num_tool_calls": m.get("num_tool_calls", 0),
                    "steps": m.get("steps", []),
                    "tool_calls": m.get("tool_calls", []),
                }
            )
            for tc in m.get("tool_calls", []):
                aggregated["all_tool_calls"].append(
                    {
                        "requirement_id": result.get("requirement_id"),
                        "requirement_index": i,
                        **tc,
                    }
                )

        n = len(sub_results)
        if n:
            aggregated["avg_tokens_per_requirement"] = aggregated["total_tokens"] / n
            aggregated["avg_steps_per_requirement"] = aggregated["total_steps"] / n
            aggregated["avg_tool_calls_per_requirement"] = (
                aggregated["total_tool_calls"] / n
            )

        return aggregated

    def run_per_requirement(
        self,
        requirements: Dict[str, str],
        tasks: Dict[str, List[str]],
        base_prompt_template: str,
    ) -> Dict[str, Any]:
        """Run a separate sub-agent for each requirement and aggregate results.

        Args:
            requirements:          ``{requirement_id: description}`` mapping.
            tasks:                 ``{requirement_id: [task_strings]}`` mapping.
            base_prompt_template:  Template string with ``{requirement_id}``,
                                   ``{requirement_description}``, and ``{tasks}``
                                   placeholders.

        Returns:
            ``{"answer", "sub_results", "metrics"}`` — see ``docs/reference.md``
            for the full schema.
        """
        sub_results: List[Dict[str, Any]] = []

        for req_id, req_description in requirements.items():
            req_tasks = tasks.get(req_id, [])
            task_list = (
                "\n".join(f"    - {t}" for t in req_tasks)
                if req_tasks
                else "    (No specific tasks)"
            )
            prompt = base_prompt_template.format(
                requirement_id=req_id,
                requirement_description=req_description,
                tasks=task_list,
            )
            sub_agent = self._create_sub_agent()
            result = sub_agent.run(prompt)
            result["requirement_id"] = req_id
            result["requirement_description"] = req_description
            result["tasks"] = req_tasks
            sub_results.append(result)

        return {
            "answer": f"Completed {len(sub_results)} sub-agent evaluations.",
            "sub_results": sub_results,
            "metrics": self._aggregate_metrics(sub_results),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Constrained agent
# ─────────────────────────────────────────────────────────────────────────────

class SmolAgentsAgentWithConstraints(SmolAgentsAgent):
    """Agent with pre-execution FOLTL constraint enforcement.

    Uses :class:`ToolCallingAgentWithConstraints` under the hood.  Accepts
    the same tool / MCP parameters as :class:`SmolAgentsAgentWithAdditionalTools`
    plus constraint configuration.

    The ``metrics`` dict returned by :meth:`run` is augmented with five
    additional keys (see ``docs/reference.md``):

    * ``run_status``             – ``"completed"`` or ``"stopped"``
    * ``stopped_by_constraint``  – name of the HARD_STOP constraint (or ``None``)
    * ``constraint_violations``  – list of violation records
    * ``constraint_checks``      – total pre-execution evaluations performed
    * ``completed_trace``        – tool calls that successfully executed

    Args:
        tools:                Local :class:`Tool` instances appended after
                              MCP tools.
        mcp_servers:          ``{name: server_config_dict}`` mapping passed to
                              :func:`_connect_mcp_servers`.
        constraints:          FOLTL constraints to enforce at every step.
                              Can be overridden per-call via :meth:`run`.
        constraint_severities: ``{constraint_name: ConstraintSeverity}`` mapping.
                              Constraints not listed here fall back to
                              *default_severity*.
        default_severity:     Fallback severity for unmapped constraints
                              (default: ``HARD_STOP``).
        max_soft_attempts:    For ``cumulative`` and ``hybrid`` modes: maximum
                              total SOFT_BLOCK violations per constraint before
                              escalating (default: ``3``).
        soft_block_mode:      Escalation counting strategy — ``"cumulative"``
                              (default), ``"consecutive"``, or ``"hybrid"``.
        max_consecutive_soft_attempts: For ``consecutive`` and ``hybrid`` modes:
                              maximum back-to-back violations before escalating.
                              Defaults to *max_soft_attempts* when not set.
        model:                Model name / ID.
        api_key:              API key for the model provider.
        provider:             HF Inference provider name.
        max_steps:            Maximum agent steps before forced termination.
        model_seed:           Optional seed for reproducible sampling.
        model_instance:       Pre-constructed model object.
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        mcp_servers: Optional[Dict[str, MCPServerConfig]] = None,
        constraints: Optional[List[Any]] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
        default_severity: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
        max_soft_attempts: int = 3,
        soft_block_mode: str = "cumulative",
        max_consecutive_soft_attempts: Optional[int] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_steps: int = 10,
        model_seed: Optional[int] = None,
        model_instance: Optional[Any] = None,
        strict_runtime_safety: bool = False,
        _skip_runtime_safety_check: bool = False,
    ) -> None:
        mcp_clients, mcp_tools = _connect_mcp_servers(mcp_servers or {})
        all_tools: List[Tool] = mcp_tools + list(tools or [])

        model_obj = (
            model_instance
            if model_instance is not None
            else _create_model(model, api_key, provider, model_seed)
        )

        self._init_constraints: List[Any] = constraints or []
        self._init_severities: Dict[str, ConstraintSeverity] = constraint_severities or {}
        self._default_severity = default_severity

        # Bypass SmolAgentsAgent.__init__: we need ToolCallingAgentWithConstraints,
        # not ToolCallingAgent, but we still set the same three attributes
        # so that inherited methods (run, _extract_metrics_from_steps,
        # __del__) work identically.
        self._model = model_obj
        self._mcp_clients = mcp_clients
        self.agent = ToolCallingAgentWithConstraints(
            tools=all_tools,
            model=model_obj,
            constraints=self._init_constraints,
            constraint_severities=self._init_severities,
            default_severity=self._default_severity,
            max_soft_attempts=max_soft_attempts,
            soft_block_mode=soft_block_mode,
            max_consecutive_soft_attempts=max_consecutive_soft_attempts,
            max_steps=max_steps,
            strict_runtime_safety=strict_runtime_safety,
            _skip_runtime_safety_check=_skip_runtime_safety_check,
        )

    def run(
        self,
        task: str,
        constraints: Optional[List[Any]] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
    ) -> Dict[str, Any]:
        """Run with pre-execution constraint checking.

        Args:
            task:                  The task string for the agent.
            constraints:           Per-run constraint override.  When supplied
                                   these replace the instance-level constraints
                                   for this call only.
            constraint_severities: Per-run severity override.

        Returns:
            ``{"answer", "metrics", "error"}`` with constraint fields merged
            into ``metrics``.  See ``docs/reference.md`` for the full schema.
        """
        run_kwargs: Dict[str, Any] = {}
        if constraints is not None:
            run_kwargs["constraints"] = constraints
        if constraint_severities is not None:
            run_kwargs["constraint_severities"] = constraint_severities

        try:
            result = self.agent.run(task, return_full_result=True, **run_kwargs)
        except Exception as exc:
            import traceback as _tb

            err_msg = str(exc)
            logger.error(
                "SmolAgentsAgentWithConstraints run failed: %s\n%s",
                err_msg,
                _tb.format_exc(),
            )
            partial_steps: List[Any] = []
            try:
                partial_steps = list(getattr(self.agent.memory, "steps", []))
            except Exception:
                pass

            constraint_status = self.agent.get_constraint_status()

            if partial_steps:
                partial_metrics = self._extract_metrics_from_steps(partial_steps)
                partial_metrics["total_tokens"] = (
                    partial_metrics["input_tokens"] + partial_metrics["output_tokens"]
                )
            else:
                partial_metrics = {
                    "steps": [],
                    "tool_calls": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "num_steps": 0,
                    "num_tool_calls": 0,
                }

            self._merge_constraint_status(partial_metrics, constraint_status)
            return {"answer": None, "metrics": partial_metrics, "error": err_msg}

        answer = result.output if hasattr(result, "output") else result
        steps = getattr(result, "steps", [])
        metrics = self._extract_metrics_from_steps(steps)

        token_usage = getattr(result, "token_usage", None)
        if token_usage is not None:
            ru_input = (
                getattr(token_usage, "input_tokens", None)
                if not isinstance(token_usage, dict)
                else token_usage.get("input_tokens")
            )
            ru_output = (
                getattr(token_usage, "output_tokens", None)
                if not isinstance(token_usage, dict)
                else token_usage.get("output_tokens")
            )
            if ru_input is not None and ru_input != metrics["input_tokens"]:
                logger.warning(
                    "Token count mismatch: RunResult.input_tokens=%d, "
                    "per-step sum=%d.",
                    ru_input,
                    metrics["input_tokens"],
                )
            if ru_output is not None and ru_output != metrics["output_tokens"]:
                logger.warning(
                    "Token count mismatch: RunResult.output_tokens=%d, "
                    "per-step sum=%d.",
                    ru_output,
                    metrics["output_tokens"],
                )

        metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
        self._merge_constraint_status(metrics, self.agent.get_constraint_status())
        return {"answer": answer, "metrics": metrics, "error": None}

    @staticmethod
    def _merge_constraint_status(
        metrics: Dict[str, Any], status: Dict[str, Any]
    ) -> None:
        """Merge ``get_constraint_status()`` fields into a metrics dict in-place."""
        metrics["run_status"] = status["status"]
        metrics["stopped_by_constraint"] = status["stopped_by"]
        metrics["constraint_violations"] = status["violations"]
        metrics["constraint_checks"] = status["constraint_checks"]
        metrics["completed_trace"] = status["completed_trace"]
