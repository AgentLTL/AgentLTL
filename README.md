# AgentLTL

**FOLTL constraint verification for LLM agent traces.**

AgentLTL provides a First-Order Linear Temporal Logic (FOLTL) engine for verifying that LLM agent tool-call traces comply with procedural constraints.  It supports both **post-hoc verification** (after a run completes) and **runtime enforcement** (pre-execution checking before each tool call).

---

## What is AgentLTL?

When an LLM agent executes a task it calls tools in some order.  AgentLTL lets you express *procedural constraints* over those tool calls using LTL formulas — for example:

- **Ordering**: `fetch_data` must happen before `process_data`
- **Counting**: `poll_status` must be called at least 3 times
- **Gating**: `generate_report` may only run after *all* of `fetch_users`, `fetch_orders`, `fetch_inventory`
- **Branching**: the agent must call the correct branch tool, not the wrong one
- **Polling**: the last poll must have returned `READY` before `process_result` is called

Each constraint is given a numeric weight; AgentLTL computes a **compliance score** C ∈ [0, 1]:

```
C = 1 − Σ(violated weights) / Σ(all weights)
```

`C = 1.0` → **FULL** compliance · `0 < C < 1` → **PARTIAL** · `C = 0` → **VIOLATION**

---

## Installation

```bash
# Core FOLTL engine (zero dependencies)
pip install agentltl

# With smolagents runtime enforcement
pip install "agentltl[smolagents]"
```

---

## Quick Start

### Post-hoc verification

```python
from agentltl import Constraint, Before, CalledNTimes, verify_trace

# Metrics produced by your agent (list of tool calls)
metrics = {
    "tool_calls": [
        {"tool_name": "fetch_data"},
        {"tool_name": "clean_data"},
        {"tool_name": "save_results"},
    ]
}

constraints = [
    Constraint("fetch_before_clean",  Before("fetch_data", "clean_data"),   weight=1.0),
    Constraint("clean_before_save",   Before("clean_data", "save_results"), weight=1.0),
    Constraint("save_once",           CalledNTimes("save_results", 1, "=="), weight=0.5),
]

result = verify_trace(metrics, constraints)
print(result["compliance_score"])   # 1.0
print(result["compliance_label"])   # FULL
```

### Runtime enforcement

```python
from smolagents import Tool
from agentltl import Constraint, Before
from agentltl.integrations.smolagents import AgentWithConstraints, ConstraintSeverity

constraints = [
    Constraint("fetch_before_process", Before("fetch_data", "process_data")),
]

agent = AgentWithConstraints(
    tools=[FetchTool(), ProcessTool()],
    constraints=constraints,
    constraint_severities={"fetch_before_process": ConstraintSeverity.HARD_STOP},
)

result = agent.run("Fetch and then process the data.")
print(result["metrics"]["run_status"])           # "completed" or "stopped"
print(result["metrics"]["constraint_violations"]) # list of violations
```

---

## FOLTL Formula Reference

| Formula | Meaning | Example |
|---------|---------|---------|
| `Called(tool)` | tool appears at least once | `Called("fetch")` |
| `CalledNTimes(tool, n, op)` | tool called n times (op: `==`, `>=`, `<=`, `>`, `<`) | `CalledNTimes("poll", 3, ">=")` |
| `Before(a, b)` | first(a) < first(b) | `Before("fetch", "save")` |
| `After(a, b)` | first(a) > first(b) | `After("save", "fetch")` |
| `AllBefore(tools, gate)` | all tools appear before gate | `AllBefore(["a","b"], "merge")` |
| `BranchCalled(correct, wrong)` | correct branch taken, wrong not | `BranchCalled("route_a", "route_b")` |
| `CalledWith(tool, args)` | tool called with matching args | `CalledWith("search", {"q": "cats"})` |
| `CalledWithResult(tool, result)` | tool returned expected value | `CalledWithResult("classify", "SPAM")` |
| `CalledInOrder(tools)` | tools appear as a subsequence | `CalledInOrder(["init","run","done"])` |
| `InstanceBefore(a,n,b,m)` | n-th a before m-th b (1-based) | `InstanceBefore("read",2,"write",1)` |
| `WithinSteps(a, b, n)` | b occurs within n steps after a | `WithinSteps("alert","ack",3)` |
| `Predicate(fn, desc)` | arbitrary Python callable | `Predicate(my_fn, "my check")` |
| `ForAll(var, domain, body)` | ∀x ∈ D. body(x) | see example below |
| `Exists(var, domain, body)` | ∃x ∈ D. body(x) | see example below |
| `Globally(φ)` | G φ — φ at every position | `Globally(Called("auth"))` |
| `Eventually(φ)` | F φ — φ at some position | `Eventually(Called("done"))` |
| `Implies(φ, ψ)` | φ → ψ | `Implies(Called("a"), Called("b"))` |
| `Not(φ)` | ¬ φ | `Not(Called("bad_tool"))` |
| `And(φ, ψ)` | φ ∧ ψ | `And(Called("a"), Called("b"))` |
| `Or(φ, ψ)` | φ ∨ ψ | `Or(Called("a"), Called("b"))` |

### ForAll example

```python
from agentltl import ForAll, CalledWith, Var, Constraint

def files_read(trace, metrics):
    return [c.args["path"] for c in trace.calls if c.name == "read_file"]

Constraint("all_reads_written", ForAll(
    "f", files_read,
    CalledWith("write_file", {"path": Var("f")}),
    description="files read",
))
# "For every file that was read, write_file was called with the same path"
```

### String parser

```python
from agentltl import parse

formula = parse('before("fetch", "save") & F(called("done"))')
```

---

## Runtime Enforcement

### ToolCallingAgentWithConstraints

The low-level integration extends smolagents' `ToolCallingAgent` to check constraints *before* each tool call:

```python
from agentltl.integrations.smolagents import (
    ToolCallingAgentWithConstraints,
    ConstraintSeverity,
)

agent = ToolCallingAgentWithConstraints(
    tools=my_tools,
    model=my_model,
    constraints=my_constraints,
    constraint_severities={"order_check": ConstraintSeverity.HARD_STOP},
)
result = agent.run("do the task", return_full_result=True)
status = agent.get_constraint_status()
```

### AgentWithConstraints

The higher-level wrapper adds MCP connectivity and metrics extraction:

```python
from agentltl.integrations.smolagents import AgentWithConstraints

# Local tools only
agent = AgentWithConstraints(
    tools=my_tools,
    constraints=my_constraints,
)

# With MCP servers
agent = AgentWithConstraints(
    tools=my_tools,
    mcp_servers={
        "filesystem": {
            "url": "http://localhost:4000/mcp",
            "transport": "streamable-http",
        },
    },
    constraints=my_constraints,
)

result = agent.run("do the task")
# result["metrics"]["run_status"]  → "completed" | "stopped"
```

### AgentWithAdditionalTools

General-purpose agent with multi-server MCP support:

```python
from agentltl.integrations.smolagents import AgentWithAdditionalTools

agent = AgentWithAdditionalTools(
    tools=my_local_tools,
    mcp_servers={
        "filesystem": {
            "url": "http://localhost:4000/mcp",
            "transport": "streamable-http",
        },
        "knowledge_graph": {
            "url": "http://localhost:4001/mcp",
            "transport": "streamable-http",
            "headers": {"Authorization": "Bearer <token>"},
        },
    },
)
result = agent.run("do the task")
```

---

## Constraint Severities

| Severity | Behaviour |
|----------|-----------|
| `HARD_STOP` | The offending tool call is **not executed**; the run stops immediately. |
| `TOLERATE` | A warning is logged and execution continues. |

Default severity is `HARD_STOP`.  Override per-constraint via `constraint_severities` dict or globally via `default_severity`.

---

## MCP Servers

### FileSystemMCPServer

Path-isolated file system access over streamable-http:

```python
from agentltl.integrations.smolagents import FileSystemMCPServer

server = FileSystemMCPServer(base_path="/data/project")
server.run(transport="streamable-http", port=4000)
```

Tools exposed: `list_files(directory)`, `read_file(filepath, chunk_size, chunk_number)`.

---

## Examples

| Example | Description |
|---------|-------------|
| `examples/01_post_hoc_verification.py` | `verify_trace()` on a completed trace |
| `examples/02_foltl_formulas.py` | Every formula type with pass/fail cases |
| `examples/03_linear_chain_agent.py` | A→B→C ordering with `AgentWithConstraints` |
| `examples/04_loop_termination_agent.py` | Polling loop compliance with `CalledNTimes` + `Predicate` |
| `examples/05_fan_out_fan_in_agent.py` | Fan-out/fan-in with `AllBefore` gate |

Run any example:

```bash
cd agentltl
pip install -e ".[smolagents]"
python examples/01_post_hoc_verification.py  # no LLM required
python examples/02_foltl_formulas.py         # no LLM required
export HF_TOKEN=...
python examples/03_linear_chain_agent.py     # requires HF_TOKEN
```

---

## Return Value Reference

See [docs/reference.md](docs/reference.md) for the complete annotated
structure of every dict returned by the public API:

- `Agent.run()` → `{"answer", "metrics", "error"}`
- `metrics` dict — top-level keys, `tool_calls` list, `steps` list,
  `model_input_parsed`
- `AgentWithConstraints.run()` — constraint fields added to metrics
- `ToolCallingAgentWithConstraints.get_constraint_status()`
- `verify_trace()` — compliance result and per-constraint records
- `AgentWithSubAgents.run_per_requirement()` — aggregated metrics

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).
