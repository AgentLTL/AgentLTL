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

import logging
from typing import Any, Dict, List, Optional

from agentltl.enforcement import ConstraintSeverity, SoftBlockMode
from agentltl.runtime_safety import check_runtime_safety_or_warn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base agent (smolagents only)
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    """Backend-agnostic base agent.

    Delegates to the ``smolagents`` backend (default) or the hand-coded
    ``native`` backend (OpenAI-compatible chat-completions, incl. the HF router).

    Args:
        tools:          Tool instances.
        model:          Model name / ID.
        api_key:        API key.
        provider:       HF Inference provider name (native: encoded as a
                        ``model:provider`` suffix when talking to the HF router).
        max_steps:      Maximum agent steps.
        model_seed:     Optional seed for reproducible sampling.
        model_instance: Pre-constructed model object (native: an ``openai.OpenAI``
                        client).
        base_url:       OpenAI-compatible base URL (native backend only).
        backend:        ``"smolagents"`` (default) or ``"native"``.
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
        base_url: Optional[str] = None,
        backend: str = "smolagents",
    ) -> None:
        if backend == "smolagents":
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
        elif backend == "native":
            from agentltl.integrations.native.backend import NativeOpenAIAgent
            self._impl = NativeOpenAIAgent(
                tools=tools,
                model=model,
                api_key=api_key,
                provider=provider,
                base_url=base_url,
                max_steps=max_steps,
                model_seed=model_seed,
                model_instance=model_instance,
            )
        else:
            raise ValueError(
                f"Agent supports backend='smolagents' or 'native'; got {backend!r}."
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
        strict_runtime_safety: bool = False,
        base_url: Optional[str] = None,
        backend: str = "smolagents",
    ) -> None:
        check_runtime_safety_or_warn(
            constraints or [],
            constraint_severities,
            default_severity,
            strict=strict_runtime_safety,
            logger=logger,
        )
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
                _skip_runtime_safety_check=True,
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
                _skip_runtime_safety_check=True,
            )
        elif backend == "native":
            from agentltl.integrations.native.backend import NativeOpenAIAgent
            self._impl = NativeOpenAIAgent(
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
                base_url=base_url,
                max_steps=max_steps,
                model_seed=model_seed,
                model_instance=model_instance,
                system_prompt=system_prompt,
                _skip_runtime_safety_check=True,
            )
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. "
                "Choose 'smolagents', 'langchain', or 'native'."
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


# ─────────────────────────────────────────────────────────────────────────────
# Multi-turn agent — built on the native backend
# ─────────────────────────────────────────────────────────────────────────────

class MultiTurnAgent:
    """A multi-turn agent built on the hand-coded :class:`NativeOpenAIAgent`.

    Unlike calling ``Agent.run()`` once per turn (which resets the conversation
    each call), this keeps ONE persistent conversation across turns, so the model
    SEES prior turns (continuity). It emits a single continuous trace (monotonic
    ``step`` numbers, each tool call tagged with its ``turn``) and — when
    constraints are supplied — enforces them at runtime *across the whole
    session* (e.g. an ordering constraint spanning two turns).

    Native backend only (the conversation-persistence primitives live there).

    Example::

        mt = MultiTurnAgent(tools=[...], model="Qwen/...", provider="novita",
                            base_url="https://router.huggingface.co/v1")
        result = mt.run(["turn 1 text", "turn 2 text", ...])
        result["answers"]      # final answer per turn
        result["per_turn"]     # [{answer, metrics, error}, ...] (per-turn trace)
        result["metrics"]      # aggregate continuous trace for verify_trace
    """

    def __init__(
        self,
        tools: Optional[List[Any]] = None,
        model: Optional[Any] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        max_steps: int = 10,
        # A blocked turn is not a step the agent spent. `None` keeps the historical
        # behaviour, where an intercepted call consumes the same budget as an
        # executed one -- see NativeOpenAIAgent for the measurement.
        max_blocked_steps: Optional[int] = None,
        max_termination_nudges: int = 0,
        model_seed: Optional[int] = None,
        model_instance: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        constraints: Optional[List[Any]] = None,
        constraint_severities: Optional[Dict[str, ConstraintSeverity]] = None,
        default_severity: ConstraintSeverity = ConstraintSeverity.HARD_STOP,
        max_soft_attempts: int = 3,
        soft_block_mode: str = "cumulative",
        max_consecutive_soft_attempts: Optional[int] = None,
        # How many violated constraints one blocked call may report. 1 preserves the
        # historical one-at-a-time behaviour; the harness raises it so a call with
        # several wrong fields does not cost one round trip per field.
        nudge_max: int = 1,
        strict_runtime_safety: bool = False,
        backend: str = "native",
        reviewer_config: Optional[Any] = None,
        reviewer_client: Optional[Any] = None,
    ) -> None:
        if backend != "native":
            raise ValueError(
                f"MultiTurnAgent only supports backend='native'; got {backend!r}."
            )
        check_runtime_safety_or_warn(
            constraints or [],
            constraint_severities,
            default_severity,
            strict=strict_runtime_safety,
            logger=logger,
        )
        impl_kwargs = dict(
            tools=tools,
            model=model,
            api_key=api_key,
            provider=provider,
            base_url=base_url,
            max_steps=max_steps,
            max_blocked_steps=max_blocked_steps,
            max_termination_nudges=max_termination_nudges,
            model_seed=model_seed,
            model_instance=model_instance,
            system_prompt=system_prompt,
            constraints=constraints,
            constraint_severities=constraint_severities,
            default_severity=default_severity,
            max_soft_attempts=max_soft_attempts,
            soft_block_mode=soft_block_mode,
            max_consecutive_soft_attempts=max_consecutive_soft_attempts,
            nudge_max=nudge_max,
            _skip_runtime_safety_check=True,
        )
        # Inference-time reviewer ("Reinforced Agent"): when a reviewer_config is
        # supplied, use the reviewer-wrapped native agent. Constraints and the
        # reviewer compose (both no-op cleanly when unset), but the benchmark runs
        # them as separate arms.
        if reviewer_config is not None:
            from agentltl.integrations.native.reviewed_agent import ReviewedNativeAgent
            self._impl = ReviewedNativeAgent(
                reviewer_config=reviewer_config,
                reviewer_client=reviewer_client,
                **impl_kwargs,
            )
        else:
            from agentltl.integrations.native.backend import NativeOpenAIAgent
            self._impl = NativeOpenAIAgent(**impl_kwargs)
        self.reset()

    def reset(self) -> None:
        """Start a fresh session (clears conversation, enforcer, and trace)."""
        self._impl.reset()

    def run_turn(self, task: str) -> Dict[str, Any]:
        """Run one user turn against the persistent conversation. Returns
        ``{"answer", "metrics", "error"}`` for that turn."""
        return self._impl.run_turn(task)

    def run(self, turns: List[str]) -> Dict[str, Any]:
        """Run a list of user turns sequentially in one persistent session.

        Returns ``{"answers", "per_turn", "metrics", "error"}`` where ``metrics``
        is the aggregate continuous trace (suitable for ``verify_trace``) and
        ``per_turn`` holds each turn's individual result.
        """
        self.reset()
        per_turn: List[Dict[str, Any]] = []
        answers: List[Optional[str]] = []
        first_error: Optional[Any] = None
        for task in turns:
            res = self.run_turn(task)
            per_turn.append(res)
            answers.append(res.get("answer"))
            if res.get("error") and first_error is None:
                first_error = res["error"]
                break  # a HARD_STOP / escalation aborts the session
        return {
            "answers": answers,
            "per_turn": per_turn,
            "metrics": self._impl.session_metrics(),
            "error": first_error,
        }

    def get_constraint_status(self) -> Dict[str, Any]:
        return self._impl.get_constraint_status()

    def get_reviewer_status(self) -> Dict[str, Any]:
        fn = getattr(self._impl, "get_reviewer_status", None)
        return fn() if fn else {"mode": "reviewed", "enabled": False}


__all__ = [
    "Agent",
    "AgentWithAdditionalTools",
    "AgentWithSubAgents",
    "AgentWithConstraints",
    "MultiTurnAgent",
]
