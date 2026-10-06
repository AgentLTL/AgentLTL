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

Higher-level agent wrappers: :class:`SmolAgentsAgent`,
:class:`SmolAgentsAgentWithAdditionalTools`, :class:`SmolAgentsAgentWithSubAgents`,
:class:`SmolAgentsAgentWithConstraints`, :class:`MCPServerConfig`. They are also available
as ``Agent``, ``AgentWithAdditionalTools``, ``AgentWithSubAgents`` and
``AgentWithConstraints`` for older code; prefer the full names, which don't clash with
:mod:`agentltl`'s own backend-agnostic ``Agent`` classes.

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

# Higher-level agent wrappers, and the older aliases.
from .backend import (
    SmolAgentsAgent,
    SmolAgentsAgentWithAdditionalTools,
    SmolAgentsAgentWithSubAgents,
    SmolAgentsAgentWithConstraints,
    MCPServerConfig,
)

Agent = SmolAgentsAgent
AgentWithAdditionalTools = SmolAgentsAgentWithAdditionalTools
AgentWithSubAgents = SmolAgentsAgentWithSubAgents
AgentWithConstraints = SmolAgentsAgentWithConstraints

__all__ = [
    "ToolCallingAgentWithConstraints",
    "ConstraintSeverity",
    "SoftBlockMode",
    "ConstraintViolation",
    "ConstraintViolationError",
    "SmolAgentsAgent",
    "SmolAgentsAgentWithAdditionalTools",
    "SmolAgentsAgentWithSubAgents",
    "SmolAgentsAgentWithConstraints",
    "Agent",
    "AgentWithAdditionalTools",
    "AgentWithSubAgents",
    "AgentWithConstraints",
    "MCPServerConfig",
]
