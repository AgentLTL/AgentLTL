# -*- coding: utf-8 -*-
"""agentltl.translation -- the natural-language-to-constraint translation pipeline.

An open predicate is a boolean expression over a CLOSED set of accessors and total
operators. The model composes one; code parses it, validates it against what the agent
could see, and interprets it. Well-formedness is guaranteed by construction, and
whether the obligation is RIGHT is decided by gates, not by the author.

Layers, all importable without any LLM client:

  * ``expr``      grammar, three-valued interpreter, ``build`` -> ``agentltl.Predicate``
  * ``library``   the closed template set the model selects from (the default path)
  * ``leaks``     literal extraction for the generation/oracle leak gate
  * ``interview`` the model-facing half; takes an injected OpenAI-shaped client
  * ``gaps``      where to ask: obligations no constraint would notice missing
  * ``pipeline``  one instance end to end, over an injected ``Backend``
"""
from . import expr, library, leaks           # noqa: F401  (stdlib only)

__all__ = ["expr", "library", "leaks", "interview", "gaps", "pipeline"]


def __getattr__(name):
    # Lazy so importing the package never pulls the interview's prompt machinery.
    if name in ("interview", "gaps", "pipeline"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(name)
