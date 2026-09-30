# -*- coding: utf-8 -*-
"""agentltl must stay a dependency-free core that knows nothing about any benchmark.

The open-predicate code was written inside a benchmark harness and moved here, so the
failure to guard against is a stray `import genv2` (or the harness) surviving the move
and working only where that package happens to be on the path.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "agentltl"
FORBIDDEN = ("genv2", "genv3", "genv4", "compliance", "test_compliance_one_instance",
             "run_graph_pilot", "openai")


def _imported_modules(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_no_module_in_the_moved_set_imports_the_harness():
    moved = [SRC / "matching.py", SRC / "relative.py", SRC / "paramspec.py",
             *sorted((SRC / "openpred").glob("*.py"))]
    bad = {p.name: m for p in moved for m in _imported_modules(p)
           if m.split(".")[0] in FORBIDDEN}
    assert not bad, f"harness/vendor imports crept into agentltl: {bad}"


def test_importing_openpred_needs_no_openai_and_no_harness():
    """Run in a fresh interpreter with `openai` made un-importable, so the check
    cannot pass merely because it was already imported by an earlier test."""
    code = (
        "import sys; sys.modules['openai'] = None\n"
        "import agentltl, agentltl.openpred as o\n"
        "from agentltl.openpred import expr, library, leaks, interview\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        f"{FORBIDDEN[:-1]!r}]\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONPATH": str(SRC.parent), "PATH": ""})
    assert r.returncode == 0, r.stderr
