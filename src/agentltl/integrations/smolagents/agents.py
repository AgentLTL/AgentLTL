"""
agentltl/integrations/smolagents/agents.py – smolagents agent wrappers.

Provides four agent classes built on top of smolagents' :class:`ToolCallingAgent`:

* :class:`Agent` – base agent with MCP connectivity and metrics extraction.
* :class:`AgentWithAdditionalTools` – adds custom :class:`Tool` instances.
* :class:`AgentWithSubAgents` – spawns per-requirement sub-agents and aggregates.
* :class:`AgentWithConstraints` – pre-execution FOLTL constraint enforcement.

All agents return a standardised ``{"answer", "metrics", "error"}`` dict from
their :meth:`run` method.  Metrics include per-step reasoning, tool calls,
timings, and token usage.

Environment variables
---------------------
* ``MODEL``                  – model name (default: ``Qwen/Qwen3-Next-80B-A3B-Instruct``)
* ``MODEL_TYPE``             – ``hf_inference`` (default) or ``vllm`` / ``openai_server``
* ``VLLM_BASE_URL``          – OpenAI-compatible endpoint (default: ``http://localhost:8000/v1``)
* ``HF_TOKEN``               – Hugging Face API token
* ``HF_INFERENCE_PROVIDER``  – Provider for HF Inference API (default: ``novita``)
* ``HF_BILL_TO``             – Optional billing organisation
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from smolagents import ToolCallingAgent, InferenceClientModel, OpenAIServerModel, MCPClient, Tool

from .constrained_agent import ToolCallingAgentWithConstraints, ConstraintSeverity


def _create_model(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model_seed: Optional[int] = None,
):
    """Create an LLM model instance.

    * ``MODEL_TYPE=vllm`` (or ``openai_server``): :class:`OpenAIServerModel` at ``VLLM_BASE_URL``
    * ``MODEL_TYPE=hf_inference`` (default): :class:`InferenceClientModel` via HF Inference API

    Args:
        model: Model name override.
        api_key: API key.
        provider: HF Inference provider (e.g. ``"novita"``).
        model_seed: Optional reproducibility seed.
    """
    model_name = model or os.getenv("MODEL") or "Qwen/Qwen3-Next-80B-A3B-Instruct"
    model_type = os.getenv("MODEL_TYPE", "hf_inference").lower()

    if model_type in ("vllm", "openai_server", "openai"):
        vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        vllm_api_key = os.getenv("VLLM_API_KEY", "dummy")
        extra: Dict[str, Any] = {}
        if model_seed is not None:
            extra["seed"] = model_seed
        if "tool_choice" not in extra:
            extra["tool_choice"] = "any"
        return OpenAIServerModel(
            model_id=model_name,
            api_base=vllm_base_url,
            api_key=vllm_api_key,
            **extra,
        )

    api_key = api_key or os.getenv("HF_TOKEN")
    provider = provider or os.getenv("HF_INFERENCE_PROVIDER") or "novita"
    model_kwargs: Dict[str, Any] = {
        "model_id": model_name,
        "token": api_key,
    }
    if provider:
        model_kwargs["provider"] = provider
    bill_to_org = os.environ.get("HF_BILL_TO") or "EPITA"
    if bill_to_org:
        model_kwargs["bill_to"] = bill_to_org
    if model_seed is not None:
        model_kwargs["seed"] = model_seed
    return InferenceClientModel(**model_kwargs)


class Agent:
    """Base agent with MCP connectivity and per-step metrics extraction.

    Args:
        mcp_server_url: URL of the MCP server (streamable-http transport).
            Falls back to the ``MCP_SERVER_URL`` environment variable.
        model: Model name.
        api_key: API key.
        provider: Model provider.
        max_steps: Maximum agent steps.
    """

    def __init__(
        self,
        mcp_server_url: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_steps: int = 10,
    ):
        self._model = _create_model(model, api_key, provider)

        mcp_server_url = mcp_server_url or os.environ.get("MCP_SERVER_URL")
        self._mcp_client = None
        tools = []

        if mcp_server_url:
            try:
                self._mcp_client = MCPClient(
                    {"url": mcp_server_url, "transport": "streamable-http"},
                    structured_output=False,
                )
                tools = self._mcp_client.__enter__()
            except Exception as e:
                print(f"Warning: Failed to initialize MCP client: {e}")
                self._mcp_client = None

        self.agent = ToolCallingAgent(tools=tools, model=self._model, max_steps=max_steps)

    def _extract_metrics_from_steps(self, steps: List[Any]) -> Dict[str, Any]:
        """Extract per-step metrics from smolagents execution steps.

        Captures: step type, reasoning text, thinking (chain-of-thought),
        tool calls, observations, timings, token usage, model input context.
        """
        def get_value(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def role_str(role):
            if role is None:
                return None
            return role.value if hasattr(role, "value") else str(role)

        def serialize_tool_calls_from_message(tool_calls):
            if not tool_calls:
                return None
            result = []
            for tc in tool_calls:
                func = get_value(tc, "function")
                if func:
                    tc_name = get_value(func, "name") if isinstance(func, dict) else getattr(func, "name", None)
                    tc_args = get_value(func, "arguments") if isinstance(func, dict) else getattr(func, "arguments", None)
                    tc_desc = get_value(func, "description") if isinstance(func, dict) else getattr(func, "description", None)
                else:
                    tc_name = get_value(tc, "name")
                    tc_args = get_value(tc, "arguments")
                    tc_desc = None
                entry = {"id": get_value(tc, "id"), "name": tc_name, "arguments": tc_args}
                if tc_desc:
                    entry["description"] = tc_desc
                result.append(entry)
            return result

        def serialize_content(content):
            if content is None:
                return None
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                serialized = []
                for block in content:
                    if isinstance(block, dict):
                        serialized.append({k: v for k, v in block.items()})
                    else:
                        block_dict = {}
                        for field in ("type", "text", "thinking", "image_url",
                                      "tool_use_id", "tool_name", "id", "input"):
                            val = get_value(block, field)
                            if val is not None:
                                block_dict[field] = val
                        serialized.append(block_dict)
                return serialized
            return str(content)

        def serialize_message(msg):
            serialized = {
                "role": role_str(get_value(msg, "role")),
                "content": serialize_content(get_value(msg, "content")),
            }
            tcs = serialize_tool_calls_from_message(get_value(msg, "tool_calls"))
            if tcs is not None:
                serialized["tool_calls"] = tcs
            tu = get_value(msg, "token_usage")
            if tu is not None:
                if isinstance(tu, dict):
                    serialized["token_usage"] = tu
                else:
                    serialized["token_usage"] = {
                        "input_tokens": getattr(tu, "input_tokens", 0),
                        "output_tokens": getattr(tu, "output_tokens", 0),
                        "total_tokens": getattr(tu, "total_tokens", 0),
                    }
            return serialized

        def extract_text_from_content(content):
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    block_type = get_value(block, "type") if isinstance(block, dict) else getattr(block, "type", None)
                    if block_type in ("text", None):
                        t = get_value(block, "text") if isinstance(block, dict) else getattr(block, "text", None)
                        if t:
                            parts.append(str(t))
                return " ".join(parts)
            return str(content)

        def parse_input_messages(messages):
            parsed = {
                "num_messages": len(messages),
                "approx_chars": 0,
                "system_prompts": [],
                "task": None,
                "history": [],
            }
            for msg in messages:
                role = role_str(get_value(msg, "role"))
                content = get_value(msg, "content")
                tool_calls = get_value(msg, "tool_calls")
                text = extract_text_from_content(content)
                parsed["approx_chars"] += len(text)

                if role == "system":
                    parsed["system_prompts"].append(text)
                elif role == "user" and parsed["task"] is None:
                    parsed["task"] = text
                elif role == "tool-response":
                    parsed["history"].append({"role": role, "type": "tool_result", "text": text, "tool_calls": None})
                elif role in ("assistant", "tool-call"):
                    parsed_tcs = serialize_tool_calls_from_message(tool_calls)
                    parsed["history"].append({
                        "role": role,
                        "type": "tool_call" if parsed_tcs else "reasoning",
                        "text": text,
                        "tool_calls": parsed_tcs,
                    })
                else:
                    parsed["history"].append({"role": role, "type": "message", "text": text, "tool_calls": None})
            return parsed

        def extract_thinking(model_output_str, model_output_message):
            if model_output_str:
                match = re.search(r"<think>(.*?)</think>", model_output_str, re.DOTALL)
                if match:
                    return match.group(1).strip()
            if model_output_message:
                content = get_value(model_output_message, "content")
                if isinstance(content, list):
                    for block in content:
                        block_type = get_value(block, "type") if isinstance(block, dict) else getattr(block, "type", None)
                        if block_type in ("thinking", "reasoning"):
                            t = (
                                (get_value(block, "thinking") if isinstance(block, dict) else getattr(block, "thinking", None))
                                or (get_value(block, "text") if isinstance(block, dict) else getattr(block, "text", None))
                                or ""
                            )
                            return t.strip() if t else None
            return None

        def extract_tool_call_info(tc):
            tool_name = None
            tool_args = None
            tool_id = None
            func = get_value(tc, "function")
            if func:
                tool_name = get_value(func, "name") if isinstance(func, dict) else getattr(func, "name", None)
                tool_args = get_value(func, "arguments") if isinstance(func, dict) else getattr(func, "arguments", None)
            if not tool_name:
                for name_field in ["name", "tool_name", "function_name", "tool"]:
                    tool_name = get_value(tc, name_field)
                    if tool_name:
                        break
            if tool_args is None:
                for arg_field in ["arguments", "args", "parameters", "inputs", "input"]:
                    tool_args = get_value(tc, arg_field)
                    if tool_args is not None:
                        break
            tool_id = get_value(tc, "id")
            return {"name": tool_name or "unknown", "arguments": tool_args if tool_args is not None else {}, "id": tool_id}

        metrics = {
            "num_steps": len(steps),
            "num_tool_calls": 0,
            "tool_calls": [],
            "steps": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        for i, step in enumerate(steps):
            step_info = {
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

            step_tokens = get_value(step, "token_usage")
            if step_tokens:
                if isinstance(step_tokens, dict):
                    input_tok = step_tokens.get("input_tokens", 0)
                    output_tok = step_tokens.get("output_tokens", 0)
                else:
                    input_tok = getattr(step_tokens, "input_tokens", 0)
                    output_tok = getattr(step_tokens, "output_tokens", 0)
                step_info["input_tokens"] = input_tok
                step_info["output_tokens"] = output_tok
                metrics["input_tokens"] += input_tok
                metrics["output_tokens"] += output_tok

            tool_calls = get_value(step, "tool_calls") or []
            pending_tool_calls = []
            if tool_calls:
                step_info["type"] = "tool_call"
                for tc_idx, tc in enumerate(tool_calls):
                    metrics["num_tool_calls"] += 1
                    tc_info = extract_tool_call_info(tc)
                    if tc_idx == 0:
                        step_info["tool_name"] = tc_info["name"]
                        step_info["tool_args"] = tc_info["arguments"]
                        step_info["tool_id"] = tc_info["id"]
                    pending_tool_calls.append({
                        "step": i + 1,
                        "tool_name": tc_info["name"],
                        "arguments": tc_info["arguments"],
                        "id": tc_info["id"],
                    })
                if len(pending_tool_calls) > 1:
                    step_info["step_tool_calls"] = [
                        {"tool_name": p["tool_name"], "tool_args": p["arguments"]}
                        for p in pending_tool_calls
                    ]

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

            error = get_value(step, "error")
            if error:
                step_info["type"] = "error"
                step_info["error"] = str(error)

            model_input_messages = get_value(step, "model_input_messages") or []
            if model_input_messages:
                step_info["model_input_raw"] = [serialize_message(m) for m in model_input_messages]
                step_info["model_input_parsed"] = parse_input_messages(model_input_messages)

            metrics["steps"].append(step_info)

        metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
        return metrics

    def run(self, task: str) -> Dict[str, Any]:
        """Run a task and return ``{"answer", "metrics", "error"}``."""
        try:
            result = self.agent.run(task, return_full_result=True)
        except Exception as e:
            import traceback
            err_msg = str(e)
            print(f"\n[Agent.run] Run failed: {err_msg}")
            print(traceback.format_exc())
            partial_steps = []
            try:
                partial_steps = list(getattr(self.agent.memory, "steps", []))
            except Exception:
                pass
            if partial_steps:
                partial_metrics = self._extract_metrics_from_steps(partial_steps)
                partial_metrics["total_tokens"] = partial_metrics["input_tokens"] + partial_metrics["output_tokens"]
                return {"answer": None, "metrics": partial_metrics, "error": err_msg}
            empty_metrics = {
                "steps": [], "tool_calls": [],
                "input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0, "num_steps": 0, "duration": 0.0,
            }
            return {"answer": None, "metrics": empty_metrics, "error": err_msg}

        answer = result.output if hasattr(result, "output") else result
        steps = getattr(result, "steps", [])
        metrics = self._extract_metrics_from_steps(steps)

        token_usage = getattr(result, "token_usage", None)
        if token_usage is not None:
            ru_input = getattr(token_usage, "input_tokens", None) if not isinstance(token_usage, dict) else token_usage.get("input_tokens")
            ru_output = getattr(token_usage, "output_tokens", None) if not isinstance(token_usage, dict) else token_usage.get("output_tokens")
            if ru_input is not None and ru_input != metrics["input_tokens"]:
                print(f"[token warning] RunResult.input_tokens={ru_input} vs per-step sum={metrics['input_tokens']}")
            if ru_output is not None and ru_output != metrics["output_tokens"]:
                print(f"[token warning] RunResult.output_tokens={ru_output} vs per-step sum={metrics['output_tokens']}")

        metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]
        return {"answer": answer, "metrics": metrics, "error": None}

    def __del__(self):
        if hasattr(self, "_mcp_client") and self._mcp_client:
            try:
                self._mcp_client.__exit__(None, None, None)
            except Exception:
                pass


class AgentWithAdditionalTools(Agent):
    """An :class:`Agent` that accepts additional custom :class:`Tool` instances.

    Args:
        mcp_server_url: MCP server URL.
        additional_tools: Extra tool instances added to the agent.
        model, api_key, provider, max_steps: Forwarded to :class:`Agent`.
        model_seed: Optional reproducibility seed.
        model_instance: Pre-created model (overrides model/api_key/provider/seed).
    """

    def __init__(
        self,
        mcp_server_url: str,
        additional_tools: List[Tool],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_steps: int = 10,
        model_seed: Optional[int] = None,
        model_instance=None,
    ):
        if model_instance is not None:
            self._model = model_instance
        else:
            self._model = _create_model(model, api_key, provider, model_seed=model_seed)

        mcp_server_url = mcp_server_url or os.environ.get("MCP_SERVER_URL")
        self._mcp_client = None
        tools = []

        if mcp_server_url:
            try:
                self._mcp_client = MCPClient(
                    {"url": mcp_server_url, "transport": "streamable-http"},
                    structured_output=False,
                )
                tools = self._mcp_client.__enter__()
            except Exception as e:
                print(f"Warning: Failed to initialize MCP client: {e}")
                self._mcp_client = None

        if additional_tools:
            tools.extend(additional_tools)

        self.agent = ToolCallingAgent(tools=tools, model=self._model, max_steps=max_steps)


class AgentWithSubAgents(Agent):
    """An agent that spawns sub-agents per requirement and aggregates results.

    Args:
        mcp_server_url: MCP server URL.
        additional_tools: Extra tools for all sub-agents.
        model, api_key, provider: Model configuration.
        max_steps: Coordinator max steps.
        sub_agent_max_steps: Sub-agent max steps.
    """

    def __init__(
        self,
        mcp_server_url: str,
        additional_tools: List[Tool] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_steps: int = 10,
        sub_agent_max_steps: int = 15,
    ):
        self.mcp_server_url = mcp_server_url
        self.additional_tools = additional_tools or []
        self.model = model
        self.api_key = api_key
        self.provider = provider
        self.sub_agent_max_steps = sub_agent_max_steps

        super().__init__(
            mcp_server_url=mcp_server_url,
            model=model,
            api_key=api_key,
            provider=provider,
            max_steps=max_steps,
        )

    def _create_sub_agent(self) -> AgentWithAdditionalTools:
        return AgentWithAdditionalTools(
            mcp_server_url=self.mcp_server_url,
            additional_tools=self.additional_tools,
            model=self.model,
            api_key=self.api_key,
            provider=self.provider,
            max_steps=self.sub_agent_max_steps,
        )

    def _aggregate_metrics(self, sub_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        aggregated = {
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
            metrics = result.get("metrics", {})
            aggregated["total_input_tokens"] += metrics.get("input_tokens", 0)
            aggregated["total_output_tokens"] += metrics.get("output_tokens", 0)
            aggregated["total_tokens"] += metrics.get("total_tokens", 0)
            aggregated["total_steps"] += metrics.get("num_steps", 0)
            aggregated["total_tool_calls"] += metrics.get("num_tool_calls", 0)
            aggregated["per_requirement_metrics"].append({
                "requirement_index": i,
                "requirement_id": result.get("requirement_id"),
                "input_tokens": metrics.get("input_tokens", 0),
                "output_tokens": metrics.get("output_tokens", 0),
                "total_tokens": metrics.get("total_tokens", 0),
                "num_steps": metrics.get("num_steps", 0),
                "num_tool_calls": metrics.get("num_tool_calls", 0),
                "steps": metrics.get("steps", []),
                "tool_calls": metrics.get("tool_calls", []),
            })
            for tool_call in metrics.get("tool_calls", []):
                aggregated["all_tool_calls"].append({
                    "requirement_id": result.get("requirement_id"),
                    "requirement_index": i,
                    **tool_call,
                })

        if len(sub_results) > 0:
            aggregated["avg_tokens_per_requirement"] = aggregated["total_tokens"] / len(sub_results)
            aggregated["avg_steps_per_requirement"] = aggregated["total_steps"] / len(sub_results)
            aggregated["avg_tool_calls_per_requirement"] = aggregated["total_tool_calls"] / len(sub_results)

        return aggregated

    def run_per_requirement(
        self,
        requirements: Dict[str, str],
        tasks: Dict[str, List[str]],
        base_prompt_template: str,
    ) -> Dict[str, Any]:
        """Run a separate sub-agent for each requirement and aggregate results.

        Args:
            requirements: ``{requirement_id: description}`` mapping.
            tasks: ``{requirement_id: [task_strings]}`` mapping.
            base_prompt_template: Template with ``{requirement_id}``,
                ``{requirement_description}``, ``{tasks}`` placeholders.

        Returns:
            ``{"answer", "sub_results", "metrics"}``
        """
        sub_results = []

        for req_id, req_description in requirements.items():
            req_tasks = tasks.get(req_id, [])
            task_list = "\n".join(f"    - {task}" for task in req_tasks) if req_tasks else "    (No specific tasks)"
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

        aggregated_metrics = self._aggregate_metrics(sub_results)
        return {
            "answer": f"Completed {len(sub_results)} sub-agent evaluations.",
            "sub_results": sub_results,
            "metrics": aggregated_metrics,
        }


class AgentWithConstraints(Agent):
    """An :class:`Agent` with pre-execution FOLTL constraint enforcement.

    Uses :class:`ToolCallingAgentWithConstraints` under the hood, providing
    both rich metric extraction and HARD_STOP / TOLERATE constraint checking.

    The returned metrics dict is augmented with:

    * ``run_status``             – ``"completed"`` or ``"stopped"``
    * ``stopped_by_constraint``  – name of the HARD_STOP constraint (or ``None``)
    * ``constraint_violations``  – list of violation records
    * ``constraint_checks``      – total pre-execution evaluations performed
    * ``completed_trace``        – list of executed tool call dicts

    Args:
        mcp_server_url: MCP server URL.
        additional_tools: Extra tool instances.
        constraints: FOLTL constraints to enforce.
        constraint_severities: ``{name: ConstraintSeverity}`` mapping.
        default_severity: Fallback severity for unmapped constraints.
        model, api_key, provider, max_steps: Model/agent config.
        model_seed: Optional reproducibility seed.
        model_instance: Pre-created model instance.
    """

    def __init__(
        self,
        mcp_server_url: str,
        additional_tools: Optional[List[Tool]] = None,
        constraints: Optional[List] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
        default_severity: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_steps: int = 10,
        model_seed: Optional[int] = None,
        model_instance=None,
    ):
        if model_instance is not None:
            self._model = model_instance
        else:
            self._model = _create_model(model, api_key, provider, model_seed=model_seed)

        mcp_server_url = mcp_server_url or os.environ.get("MCP_SERVER_URL")
        self._mcp_client = None
        tools = []

        if mcp_server_url:
            try:
                self._mcp_client = MCPClient(
                    {"url": mcp_server_url, "transport": "streamable-http"},
                    structured_output=False,
                )
                tools = self._mcp_client.__enter__()
            except Exception as e:
                print(f"Warning: Failed to initialize MCP client: {e}")
                self._mcp_client = None

        if additional_tools:
            tools.extend(additional_tools)

        self._init_constraints = constraints or []
        self._init_severities = constraint_severities or {}
        self._default_severity = default_severity

        self.agent = ToolCallingAgentWithConstraints(
            tools=tools,
            model=self._model,
            constraints=self._init_constraints,
            constraint_severities=self._init_severities,
            default_severity=self._default_severity,
            max_steps=max_steps,
        )

    def run(
        self,
        task: str,
        constraints: Optional[List] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
    ) -> Dict[str, Any]:
        """Run with constraint checking; returns augmented metrics dict."""
        run_kwargs: Dict[str, Any] = {}
        if constraints is not None:
            run_kwargs["constraints"] = constraints
        if constraint_severities is not None:
            run_kwargs["constraint_severities"] = constraint_severities

        try:
            result = self.agent.run(task, return_full_result=True, **run_kwargs)
        except Exception as e:
            import traceback
            err_msg = str(e)
            print(f"\n[AgentWithConstraints.run] Run failed: {err_msg}")
            print(traceback.format_exc())

            partial_steps = []
            try:
                partial_steps = list(getattr(self.agent.memory, "steps", []))
            except Exception:
                pass

            constraint_status = self.agent.get_constraint_status()

            if partial_steps:
                partial_metrics = self._extract_metrics_from_steps(partial_steps)
                partial_metrics["total_tokens"] = partial_metrics["input_tokens"] + partial_metrics["output_tokens"]
            else:
                partial_metrics = {
                    "steps": [], "tool_calls": [],
                    "input_tokens": 0, "output_tokens": 0,
                    "total_tokens": 0, "num_steps": 0, "duration": 0.0,
                }

            partial_metrics["run_status"] = constraint_status["status"]
            partial_metrics["stopped_by_constraint"] = constraint_status["stopped_by"]
            partial_metrics["constraint_violations"] = constraint_status["violations"]
            partial_metrics["constraint_checks"] = constraint_status["constraint_checks"]
            partial_metrics["completed_trace"] = constraint_status["completed_trace"]

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
                print(f"[token warning] RunResult.input_tokens={ru_input} vs per-step sum={metrics['input_tokens']}")
            if ru_output is not None and ru_output != metrics["output_tokens"]:
                print(f"[token warning] RunResult.output_tokens={ru_output} vs per-step sum={metrics['output_tokens']}")

        metrics["total_tokens"] = metrics["input_tokens"] + metrics["output_tokens"]

        constraint_status = self.agent.get_constraint_status()
        metrics["run_status"] = constraint_status["status"]
        metrics["stopped_by_constraint"] = constraint_status["stopped_by"]
        metrics["constraint_violations"] = constraint_status["violations"]
        metrics["constraint_checks"] = constraint_status["constraint_checks"]
        metrics["completed_trace"] = constraint_status["completed_trace"]

        return {"answer": answer, "metrics": metrics, "error": None}
