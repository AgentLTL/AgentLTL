# -*- coding: utf-8 -*-
"""agentltl.openpred -- open predicates: a closed grammar the model composes.

An open predicate is a boolean expression over a CLOSED set of accessors and total
operators. The model composes one; code parses it, validates it against what the agent
could see, and interprets it. Well-formedness is guaranteed by construction, and
whether the obligation is RIGHT is decided by gates, not by the author.

Layers, all importable without any LLM client:

  * ``expr``      grammar, three-valued interpreter, ``build`` -> ``agentltl.Predicate``
  * ``library``   the closed template set the model selects from (the default path)
  * ``leaks``     literal extraction for the generation/oracle leak gate
  * ``interview`` the model-facing half; takes an injected OpenAI-shaped client
"""
from . import expr, library, leaks           # noqa: F401  (stdlib only)

__all__ = ["expr", "library", "leaks", "interview"]


def __getattr__(name):
    # Lazy so importing the package never pulls the interview's prompt machinery.
    if name == "interview":
        import importlib
        return importlib.import_module(".interview", __name__)
    raise AttributeError(name)
