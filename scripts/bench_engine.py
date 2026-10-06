"""Time the enforcement engine: K constraints over runs of N steps.

    python scripts/bench_engine.py            # 25 constraints, 50..400 steps
"""

import sys
import time

from agentltl import Constraint, ConstraintSeverity, parse
from agentltl._enforcement_engine import ConstraintEnforcer

TOOLS = [f"t{i}" for i in range(10)]


def constraints(k):
    out = []
    for i in range(k):
        a, b = TOOLS[i % 10], TOOLS[(i + 3) % 10]
        text = [f'G(now("{a}") -> X(G(!now("{b}x"))))', f'G(!now("{a}x"))',
                f'before("{a}", "{b}x")', f'G(now("{a}") -> F(now("{b}")))'][i % 4]
        out.append(Constraint(f"c{i}", parse(text)))
    return out


def run(k, n):
    enf = ConstraintEnforcer(constraints=constraints(k),
                             default_severity=ConstraintSeverity.TOLERATE)
    start = time.perf_counter()
    for step in range(n):
        tool = TOOLS[step % 10]
        enf.begin_generation()
        if enf.check(tool, {"i": step}, step + 1) == "allow":
            enf.record_completed(tool, {"i": step}, str(step), "ok")
    return time.perf_counter() - start


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    for n in (50, 100, 200, 400):
        print(f"{k} constraints, {n} steps: {run(k, n):.3f} s")
