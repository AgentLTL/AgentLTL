"""
agentltl.integrations.smolagents – smolagents integration for AgentLTL.

Provides:

* :class:`ToolCallingAgentWithConstraints` – pre-execution FOLTL constraint
  checking integrated into the smolagents tool-calling loop.
* :class:`ConstraintSeverity` – HARD_STOP vs TOLERATE enum.
* :class:`ConstraintViolationError` – raised on HARD_STOP violations.
* :class:`Agent`, :class:`AgentWithAdditionalTools`, :class:`AgentWithSubAgents`,
  :class:`AgentWithConstraints` – higher-level agent wrappers with metrics
  extraction and optional MCP server connectivity.

Install with::

    pip install agentltl[smolagents]
"""

from .constrained_agent import (
    ToolCallingAgentWithConstraints,
    ConstraintSeverity,
    ConstraintViolation,
    ConstraintViolationError,
)

from .agents import (
    Agent,
    AgentWithAdditionalTools,
    AgentWithSubAgents,
    AgentWithConstraints,
    MCPServerConfig,
)

__all__ = [
    "ToolCallingAgentWithConstraints",
    "ConstraintSeverity",
    "ConstraintViolation",
    "ConstraintViolationError",
    "Agent",
    "AgentWithAdditionalTools",
    "AgentWithSubAgents",
    "AgentWithConstraints",
    "MCPServerConfig",
]
