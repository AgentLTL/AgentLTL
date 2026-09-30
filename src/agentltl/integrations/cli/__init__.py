"""Command-line integration: enforce constraints over a shell tool's command lines.

A call such as ``bash(command="git push && git commit -m x")`` is translated into
the ordered structured calls ``git_push``, ``git_commit`` by the optional
``cli-to-tools`` package, and constraints are evaluated over those. A chain is
approved or rejected as a whole, before anything runs.

Usage::

    from agentltl import AgentWithConstraints

    agent = AgentWithConstraints(
        tools=[bash_tool], constraints=constraints,
        backend="native", shell_tools={"bash": "command"},
    )

The translator and the enforcer live in ``cli-to-tools`` because they need a bash
parser, which the zero-dependency core cannot carry.
"""

try:
    from cli_to_tools import SpecRegistry, TranslationError, Translator
    from cli_to_tools.agentltl import CliConstraintEnforcer, expand_tool_calls
except ImportError as e:
    raise ImportError(
        "agentltl.integrations.cli requires the `cli-to-tools` package. "
        "Install with: pip install 'agentltl[cli]'."
    ) from e

__all__ = [
    "CliConstraintEnforcer",
    "SpecRegistry",
    "TranslationError",
    "Translator",
    "expand_tool_calls",
]
