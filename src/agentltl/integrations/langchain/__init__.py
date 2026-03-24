"""
agentltl.integrations.langchain – LangChain integration for AgentLTL.

Provides:

* :class:`ConstraintEnforcementMiddleware` – stateful FOLTL constraint
  checking middleware for LangChain agents.  Use :meth:`as_middleware` to
  get the :class:`AgentMiddleware` to pass to ``create_agent(middleware=...)``.
* :class:`LangChainConstrainedBackend` – low-level backend used by
  :class:`agentltl.agents.AgentWithConstraints` when ``backend="langchain"``.

Shared enforcement types (framework-agnostic) are re-exported here for
convenience:

* :class:`ConstraintSeverity`
* :class:`SoftBlockMode`
* :class:`ConstraintViolation`
* :class:`ConstraintViolationError`

Install with::

    pip install agentltl[langchain]
"""

from .constrained_agent import ConstraintEnforcementMiddleware
from .backend import LangChainConstrainedBackend

from agentltl.enforcement import (
    ConstraintSeverity,
    SoftBlockMode,
    ConstraintViolation,
    ConstraintViolationError,
)

__all__ = [
    "ConstraintEnforcementMiddleware",
    "LangChainConstrainedBackend",
    "ConstraintSeverity",
    "SoftBlockMode",
    "ConstraintViolation",
    "ConstraintViolationError",
]
