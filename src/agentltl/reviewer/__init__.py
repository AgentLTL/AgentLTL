"""Inference-time reviewer for tool-calling agents.

Implements the "Reinforced Agent" technique: a reviewer inspects a provisional
tool call *before* it executes and either returns feedback so the agent
regenerates (progressive feedback), or selects/grades among candidate calls
(best-of-N).  The reviewer is a separable, toggleable wrapper — with it disabled
the base agent behaves exactly as before.

This package is backend-agnostic (pure config, prompts, parsing, and strategy
control flow).  The smolagents wrapper that binds it to a live agent lives in
``agentltl.integrations.smolagents.reviewed_agent``.
"""

from .config import (
    ReviewerConfig,
    STRATEGY_PROGRESSIVE,
    STRATEGY_SELECTION,
    STRATEGY_GRADING,
    VALID_STRATEGIES,
)
from .parsing import ReviewResult, parse_reviewer_output
from .prompts import (
    KNOWN_PROMPT_VERSIONS,
    DEFAULT_PROMPT_VERSION,
    OVER_SKEPTICISM_GUARD,
    render,
    build_system_prompt,
    build_user_prompt,
)
from .strategies import Candidate, StrategyOutcome, run_strategy

__all__ = [
    "ReviewerConfig",
    "STRATEGY_PROGRESSIVE",
    "STRATEGY_SELECTION",
    "STRATEGY_GRADING",
    "VALID_STRATEGIES",
    "ReviewResult",
    "parse_reviewer_output",
    "KNOWN_PROMPT_VERSIONS",
    "DEFAULT_PROMPT_VERSION",
    "OVER_SKEPTICISM_GUARD",
    "render",
    "build_system_prompt",
    "build_user_prompt",
    "Candidate",
    "StrategyOutcome",
    "run_strategy",
]
