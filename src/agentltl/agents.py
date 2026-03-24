"""
agentltl/agents.py – backend-agnostic agent wrappers.

These classes dispatch to either the smolagents or LangChain backend depending
on the ``backend=`` parameter, while exposing a single, consistent public API.

Classes
-------
* :class:`Agent`                     – base agent (smolagents backend only)
* :class:`AgentWithAdditionalTools`  – MCP-enabled agent (smolagents backend only)
* :class:`AgentWithSubAgents`        – sub-agent orchestrator (smolagents backend only)
* :class:`AgentWithConstraints`      – FOLTL constraint enforcement (smolagents or langchain)

All ``run()`` methods return ``{"answer", "metrics", "error"}``.
See ``docs/reference.md`` for the full return-value schema.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentltl.enforcement import ConstraintSeverity, SoftBlockMode


# ─────────────────────────────────────────────────────────────────────────────
# Base agent (smolagents only)
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    """Backend-agnostic base agent.

    Currently delegates to the smolagents backend.  The ``backend`` parameter
    is accepted for forward-compatibility but ``"smolagents"`` is the only
    supported value.

    Args:
        tools:          Tool instances.
        model:          Model name / ID.
        api_key:        API key.
        provider:       HF Inference provider name.
        max_steps:      Maximum agent steps.
        model_seed:     Optional seed for reproducible sampling.
        model_instance: Pre-constructed model object.
        backend:        Backend to use.  Only ``"smolagents"`` is supported.
    """

    def __init__(
        self,
        tools: Optional[List[Any]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_steps: int = 10,
        model_seed: Optional[int] = None,
        model_instance: Optional[Any] = None,
        backend: str = "smolagents",
    ) -> None:
        if backend != "smolagents":
            raise ValueError(
                f"Agent only supports backend='smolagents'; got {backend!r}."
            )
        from agentltl.integrations.smolagents.backend import SmolAgentsAgent
        self._impl = SmolAgentsAgent(
            tools=tools,
            model=model,
            api_key=api_key,
            provider=provider,
            max_steps=max_steps,
            model_seed=model_seed,
            model_instance=model_instance,
        )

    def run(self, task: str) -> Dict[str, Any]:
        return self._impl.run(task)


# ─────────────────────────────────────────────────────────────────────────────
# MCP-enabled agent (smolagents only)
# ─────────────────────────────────────────────────────────────────────────────

class AgentWithAdditionalTools:
    """Backend-agnostic MCP-enabled agent.

    Currently delegates to the smolagents backend.

    Args:
        tools:          Local tool instances appended after MCP tools.
        mcp_servers:    ``{name: server_config_dict}`` mapping.
        model:          Model name / ID.
        api_key:        API key.
        provider:       HF Inference provider name.
        max_steps:      Maximum agent steps.
        model_seed:     Optional seed for reproducible sampling.
        model_instance: Pre-constructed model object.
        backend:        Backend to use.  Only ``"smolagents"`` is supported.
    """

    def __init__(
        self,
        tools: Optional[List[Any]] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_steps: int = 10,
        model_seed: Optional[int] = None,
        model_instance: Optional[Any] = None,
        backend: str = "smolagents",
    ) -> None:
        if backend != "smolagents":
            raise ValueError(
                f"AgentWithAdditionalTools only supports backend='smolagents'; "
                f"got {backend!r}."
            )
        from agentltl.integrations.smolagents.backend import SmolAgentsAgentWithAdditionalTools
        self._impl = SmolAgentsAgentWithAdditionalTools(
            tools=tools,
            mcp_servers=mcp_servers,
            model=model,
            api_key=api_key,
            provider=provider,
            max_steps=max_steps,
            model_seed=model_seed,
            model_instance=model_instance,
        )

    def run(self, task: str) -> Dict[str, Any]:
        return self._impl.run(task)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-agent orchestrator (smolagents only)
# ─────────────────────────────────────────────────────────────────────────────

class AgentWithSubAgents:
    """Backend-agnostic sub-agent orchestrator.

    Currently delegates to the smolagents backend.

    Args:
        tools:               Local tools forwarded to every sub-agent.
        mcp_servers:         MCP server config forwarded to every sub-agent.
        model:               Model name / ID.
        api_key:             API key.
        provider:            HF Inference provider name.
        max_steps:           Coordinator max steps.
        sub_agent_max_steps: Max steps for each spawned sub-agent.
        model_seed:          Optional seed.
        model_instance:      Pre-constructed model shared by all agents.
        backend:             Backend to use.  Only ``"smolagents"`` is supported.
    """

    def __init__(
        self,
        tools: Optional[List[Any]] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_steps: int = 10,
        sub_agent_max_steps: int = 15,
        model_seed: Optional[int] = None,
        model_instance: Optional[Any] = None,
        backend: str = "smolagents",
    ) -> None:
        if backend != "smolagents":
            raise ValueError(
                f"AgentWithSubAgents only supports backend='smolagents'; "
                f"got {backend!r}."
            )
        from agentltl.integrations.smolagents.backend import SmolAgentsAgentWithSubAgents
        self._impl = SmolAgentsAgentWithSubAgents(
            tools=tools,
            mcp_servers=mcp_servers,
            model=model,
            api_key=api_key,
            provider=provider,
            max_steps=max_steps,
            sub_agent_max_steps=sub_agent_max_steps,
            model_seed=model_seed,
            model_instance=model_instance,
        )

    def run(self, task: str) -> Dict[str, Any]:
        return self._impl.run(task)

    def run_per_requirement(
        self,
        requirements: Dict[str, str],
        tasks: Dict[str, List[str]],
        base_prompt_template: str,
    ) -> Dict[str, Any]:
        return self._impl.run_per_requirement(requirements, tasks, base_prompt_template)


# ─────────────────────────────────────────────────────────────────────────────
# Constrained agent — supports both smolagents and langchain backends
# ─────────────────────────────────────────────────────────────────────────────

class AgentWithConstraints:
    """Backend-agnostic FOLTL constrained agent.

    Dispatches to the smolagents or LangChain backend depending on
    ``backend=``.  The public interface is identical regardless of backend.

    Parameters shared by all backends
    ----------------------------------
    tools, constraints, constraint_severities, default_severity,
    max_soft_attempts, soft_block_mode, max_consecutive_soft_attempts,
    max_steps

    Model / connectivity parameters
    --------------------------------
    model            – string model name; resolved per-backend
    api_key          – API key (smolagents HF token; OpenAI key for langchain)
    provider         – HF Inference provider name (smolagents only)
    model_seed       – optional reproducible-sampling seed (smolagents only)
    model_instance   – pre-built model object passed through to the backend
    mcp_servers      – MCP server config dict (smolagents only; ignored with
                       a warning for the langchain backend)
    system_prompt    – optional system prompt (langchain backend; ignored
                       by smolagents unless added to the task string)

    Backend selection
    -----------------
    backend : str, default ``"smolagents"``
        ``"smolagents"``  – use :class:`SmolAgentsAgentWithConstraints`
        ``"langchain"``   – use :class:`LangChainConstrainedBackend`

    Returns
    -------
    ``run()`` returns ``{"answer", "metrics", "error"}`` with constraint
    fields merged into ``metrics``.  See ``docs/reference.md``.
    """

    def __init__(
        self,
        tools: Optional[List[Any]] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Any]] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
        default_severity: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
        max_soft_attempts: int = 3,
        soft_block_mode: str = "cumulative",
        max_consecutive_soft_attempts: Optional[int] = None,
        model: Optional[Any] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model_seed: Optional[int] = None,
        model_instance: Optional[Any] = None,
        max_steps: int = 10,
        system_prompt: Optional[str] = None,
        backend: str = "smolagents",
    ) -> None:
        if backend == "smolagents":
            from agentltl.integrations.smolagents.backend import SmolAgentsAgentWithConstraints
            self._impl = SmolAgentsAgentWithConstraints(
                tools=tools,
                mcp_servers=mcp_servers,
                constraints=constraints,
                constraint_severities=constraint_severities,
                default_severity=default_severity,
                max_soft_attempts=max_soft_attempts,
                soft_block_mode=soft_block_mode,
                max_consecutive_soft_attempts=max_consecutive_soft_attempts,
                model=model,
                api_key=api_key,
                provider=provider,
                max_steps=max_steps,
                model_seed=model_seed,
                model_instance=model_instance,
            )
        elif backend == "langchain":
            from agentltl.integrations.langchain.backend import LangChainConstrainedBackend
            self._impl = LangChainConstrainedBackend(
                tools=tools,
                mcp_servers=mcp_servers,
                constraints=constraints,
                constraint_severities=constraint_severities,
                default_severity=default_severity,
                max_soft_attempts=max_soft_attempts,
                soft_block_mode=soft_block_mode,
                max_consecutive_soft_attempts=max_consecutive_soft_attempts,
                model=model,
                model_instance=model_instance,
                max_steps=max_steps,
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. "
                "Choose 'smolagents' or 'langchain'."
            )

    def run(
        self,
        task: str,
        constraints: Optional[List[Any]] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
    ) -> Dict[str, Any]:
        return self._impl.run(
            task,
            constraints=constraints,
            constraint_severities=constraint_severities,
        )

    def get_constraint_status(self) -> Dict[str, Any]:
        return self._impl.get_constraint_status()


__all__ = [
    "Agent",
    "AgentWithAdditionalTools",
    "AgentWithSubAgents",
    "AgentWithConstraints",
]
