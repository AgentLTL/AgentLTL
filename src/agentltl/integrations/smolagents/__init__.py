"""
agentltl.integrations.smolagents – smolagents integration for AgentLTL.

Provides:

* :class:`ToolCallingAgentWithConstraints` – pre-execution FOLTL constraint
  checking integrated into the smolagents tool-calling loop.
* :class:`ConstraintViolationError` – smolagents-compatible subclass (also
  inherits from ``smolagents.utils.AgentError`` so the run-loop catches it).

Shared enforcement types (re-exported from ``agentltl.enforcement``):

* :class:`ConstraintSeverity`
* :class:`SoftBlockMode`
* :class:`ConstraintViolation`

Higher-level agent wrappers (re-exported from the backend for backward compat):

* :class:`Agent`
* :class:`AgentWithAdditionalTools`
* :class:`AgentWithSubAgents`
* :class:`AgentWithConstraints`
* :class:`MCPServerConfig`

Install with::

    pip install agentltl[smolagents]
"""

# Shared enforcement types (no framework dependency).
from agentltl.enforcement import (
    ConstraintSeverity,
    SoftBlockMode,
    ConstraintViolation,
)

# smolagents-compatible ConstraintViolationError (inherits AgentError).
from .constrained_agent import (
    ToolCallingAgentWithConstraints,
    ConstraintViolationError,
)

# Higher-level agent wrappers — backward-compatible aliases.
from .backend import (
    SmolAgentsAgent as Agent,
    SmolAgentsAgentWithAdditionalTools as AgentWithAdditionalTools,
    SmolAgentsAgentWithSubAgents as AgentWithSubAgents,
    SmolAgentsAgentWithConstraints as AgentWithConstraints,
    MCPServerConfig,
)

__all__ = [
    "ToolCallingAgentWithConstraints",
    "ConstraintSeverity",
    "SoftBlockMode",
    "ConstraintViolation",
    "ConstraintViolationError",
    "Agent",
    "AgentWithAdditionalTools",
    "AgentWithSubAgents",
    "AgentWithConstraints",
    "MCPServerConfig",
]
