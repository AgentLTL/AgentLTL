# AgentLTL

**FOLTL constraint verification for LLM agent traces.**

📖 **Documentation: [agentltl.github.io](https://agentltl.github.io)**, with the
[API reference](https://agentltl.github.io/python/api/) and the
[concepts](https://agentltl.github.io/concepts/) behind the formulas.

AgentLTL provides a First-Order Linear Temporal Logic (FOLTL) engine for verifying that LLM agent tool-call traces comply with procedural constraints.  It supports both **post-hoc verification** (after a run completes) and **runtime enforcement** (pre-execution checking before each tool call, with `HARD_STOP`, `SOFT_BLOCK`, `BLOCK_AND_WARN`, `PERSISTENT_BLOCK`, and `TOLERATE` severity modes).

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

# With LangChain runtime enforcement
pip install "agentltl[langchain]"
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

### Runtime enforcement — smolagents

```python
from agentltl import Constraint, Before, CalledNTimes, AgentWithConstraints, ConstraintSeverity

constraints = [
    Constraint("fetch_before_process", Before("fetch_data", "process_data")),
    Constraint("save_once",            CalledNTimes("save_results", 1, "=="), weight=0.5),
]

agent = AgentWithConstraints(
    tools=[FetchTool(), ProcessTool(), SaveTool()],
    constraints=constraints,
    constraint_severities={
        "fetch_before_process": ConstraintSeverity.HARD_STOP,      # abort immediately
        "save_once":            ConstraintSeverity.BLOCK_AND_WARN, # warn; allow if model insists
    },
    max_soft_attempts=3,
    soft_block_mode="cumulative",
    backend="smolagents",   # default
)

result = agent.run("Fetch and then process the data, then save.")
print(result["metrics"]["run_status"])           # "completed" or "stopped"
print(result["metrics"]["constraint_violations"]) # list of violations
```

### Runtime enforcement — LangChain

```python
from langchain_openai import ChatOpenAI
from agentltl import Constraint, Before, AgentWithConstraints, ConstraintSeverity

agent = AgentWithConstraints(
    tools=[fetch_tool, process_tool],  # LangChain BaseTool instances
    constraints=[Constraint("fetch_before_process", Before("fetch_data", "process_data"))],
    constraint_severities={"fetch_before_process": ConstraintSeverity.SOFT_BLOCK},
    max_soft_attempts=3,
    soft_block_mode="hybrid",
    model_instance=ChatOpenAI(model="gpt-4o-mini"),
    backend="langchain",
)

result = agent.run("Fetch and then process the data.")
print(result["metrics"]["run_status"])           # "completed" or "stopped"
print(result["metrics"]["constraint_violations"]) # list of violations
```

---

## Walkthrough: Agent → Trace → Violation Analysis

This section walks through the full cycle end-to-end using a three-step data pipeline: `fetch_data → process_data → save_results`.  No LLM is required for steps 3–5; only the live-agent steps need `HF_TOKEN`.

### 1. Define tools and constraints

```python
import os
from smolagents import Tool
from agentltl import Constraint, Before, CalledNTimes, verify_trace
from agentltl.integrations.smolagents import AgentWithConstraints, ConstraintSeverity


class FetchTool(Tool):
    name        = "fetch_data"
    description = "Fetch raw records from the database. Always call this first."
    inputs      = {}
    output_type = "string"
    def forward(self): return "Fetched: 120 records"


class ProcessTool(Tool):
    name        = "process_data"
    description = "Normalise fetched records. Call after fetch_data."
    inputs      = {}
    output_type = "string"
    def forward(self): return "Processed: 120 records normalised"


class SaveTool(Tool):
    name        = "save_results"
    description = "Save processed results to storage. Call after process_data."
    inputs      = {}
    output_type = "string"
    def forward(self): return "Saved to s3://bucket/results.parquet"


constraints = [
    Constraint("fetch_before_process", Before("fetch_data",   "process_data"),  weight=2.0),
    Constraint("process_before_save",  Before("process_data", "save_results"),  weight=2.0),
    Constraint("save_once",            CalledNTimes("save_results", 1, "=="),   weight=1.0),
]
```

### 2. Run the agent with runtime enforcement

```python
agent = AgentWithConstraints(
    tools=[FetchTool(), ProcessTool(), SaveTool()],
    constraints=constraints,
    constraint_severities={
        "fetch_before_process": ConstraintSeverity.HARD_STOP,      # abort immediately
        "process_before_save":  ConstraintSeverity.SOFT_BLOCK,       # block & let agent retry; escalates
        "save_once":            ConstraintSeverity.BLOCK_AND_WARN,   # warn; allow if model insists
        "never_delete":         ConstraintSeverity.PERSISTENT_BLOCK, # warn; block every time, no override
    },
    max_soft_attempts=3,
    model=os.environ.get("MODEL", "Qwen/Qwen3-32B-Instruct"),
    max_steps=6,
)

result = agent.run("Fetch the data, process it, and save the results to storage.")
```

### 3. Inspect the raw trace

`result["metrics"]["tool_calls"]` is a list of dicts, one per tool call, in execution order:

```python
import json
print(json.dumps(result["metrics"]["tool_calls"], indent=2))
```

```json
[
  {
    "tool_name": "fetch_data",
    "tool_args": {},
    "tool_result": "Fetched: 120 records"
  },
  {
    "tool_name": "process_data",
    "tool_args": {},
    "tool_result": "Processed: 120 records normalised"
  },
  {
    "tool_name": "save_results",
    "tool_args": {},
    "tool_result": "Saved to s3://bucket/results.parquet"
  }
]
```

### 4. Check runtime constraint status

```python
m = result["metrics"]
print(f"Run status        : {m['run_status']}")
print(f"Stopped by        : {m['stopped_by_constraint']}")
print(f"Constraint checks : {m['constraint_checks']}")
print(f"Violations        : {len(m['constraint_violations'])}")
```

```
Run status        : completed
Stopped by        : None
Constraint checks : 3
Violations        : 0
```

### 5. Post-hoc analysis with `verify_trace`

```python
report = verify_trace(result["metrics"], constraints)

print(f"Tool sequence : {report['tool_sequence']}")
print(f"Compliance    : {report['compliance_label']} ({report['compliance_score']:.2f})")
print()
for c in report["constraints"]:
    status = "PASS" if c["passed"] else "FAIL"
    print(f"  [{status}] {c['name']} (w={c['weight']}): {c['detail']}")
```

```
Tool sequence : ['fetch_data', 'process_data', 'save_results']
Compliance    : FULL (1.00)

  [PASS] fetch_before_process (w=2.0): "fetch_data" (call #1) before "process_data" (call #2).
  [PASS] process_before_save  (w=2.0): "process_data" (call #2) before "save_results" (call #3).
  [PASS] save_once            (w=1.0): "save_results" called 1 time(s); expected == 1.
```

---

### What a violation looks like

Suppose the agent calls `process_data` before `fetch_data` — the `HARD_STOP` constraint fires.

**Runtime output** (`result["metrics"]`):

```
Run status        : stopped
Stopped by        : fetch_before_process
Constraint checks : 1
Violations        : 1
```

```python
for v in result["metrics"]["constraint_violations"]:
    print(f"[{v['severity']}] {v['constraint_name']} at step {v['step_number']}: {v['detail']}")
```

```
[HARD_STOP] fetch_before_process at step 1: "fetch_data" was never called (cannot precede "process_data").
```

**Post-hoc analysis** on the partial trace (only the one completed call):

```python
partial_metrics = {
    "tool_calls": [{"tool_name": "process_data"}]
}
report = verify_trace(partial_metrics, constraints)

print(f"Compliance : {report['compliance_label']} ({report['compliance_score']:.2f})")
print()
for c in report["constraints"]:
    status = "PASS" if c["passed"] else "FAIL"
    print(f"  [{status}] {c['name']}: {c['detail']}")
```

```
Compliance : VIOLATION (0.00)

  [FAIL] fetch_before_process: "fetch_data" was never called (cannot precede "process_data").
  [FAIL] process_before_save:  "save_results" was never called ("process_data" has no successor to precede).
  [FAIL] save_once:            "save_results" called 0 time(s); expected == 1.
```

The compliance score is `1 − (2.0 + 2.0 + 1.0) / (2.0 + 2.0 + 1.0) = 0.00` because every constraint is violated.  A partial violation (e.g. only `save_once` fails) would give a non-zero score — `1 − 1.0 / 5.0 = 0.80`.

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

`called("x")` is a fact about the whole trace: once `x` appears anywhere, it holds at
every position. To talk about the call at the current step, which is what you
usually want under `G` and `X`, use `now("x")`:

```python
# deploy at most once: after a deploy step, no later step is a deploy
once = parse('G(now("deploy") -> X(G(!now("deploy"))))')
```

### Judging a run that isn't over

While a run is in progress (`partial_trace=True`, which the runtime enforcer uses), a
formula takes one of five values (`agentltl._partial`):

| Value | Meaning |
|---|---|
| `FALSE` | violated, and no later call can repair it |
| `PFALSE` | an obligation not met yet: `called("x")` before x runs |
| `PENDING` | not triggered or not decided yet: `X φ` at the last call, `F φ`, `before(a, b)` before any b |
| `PTRUE` | holds so far; a later call could still break it (`at most 2`) |
| `TRUE` | holds, and nothing appended can change that |

A call is refused when the value is below `PENDING`. `Not` mirrors the scale, `And`, `G`
and `ForAll` take the minimum, `Or`, `F` and `Exists` the maximum. On a complete trace
(scoring) the ordinary two-valued semantics apply.

The enforcer refuses a call only for what **that call** newly breaks. Each failure carries
*witnesses* (a position of `G`, a side of `And`, an entity of `ForAll`, a count); when all
of them were already `FALSE` before the call, the call isn't refused for it. After a
`BLOCK_AND_WARN` override of `G(now("deploy") -> X(G(!now("deploy"))))`, an unrelated `ls`
goes through, and the next deploy is refused again.

---

## Runtime Enforcement

### Enforcer (any agent loop)

`Enforcer` is what every integration and harness drives. It judges each proposed call
and returns a `Decision`; it never raises:

```python
from agentltl import Constraint, ConstraintSeverity as S, Enforcer, parse

enforcer = Enforcer(
    [Constraint("one-deploy", parse('G(now("deploy") -> X(G(!now("deploy"))))'),
                repair="Deploy once per session.")],
    {"one-deploy": S.BLOCK_AND_WARN},
)

decision = enforcer.check("deploy", {"env": "prod"}, generation=completion_id)
if decision.allowed:
    result = run_tool(...)
    enforcer.record_completed("deploy", {"env": "prod"}, result=result, status=0)
else:
    reply_to_model(decision.feedback)   # decision.action: warn | retry | ask | block | stop
```

| Severity | `Decision.action` | |
|---|---|---|
| `HARD_STOP` | `stop` | `latch_stop=True` refuses every later call until `resume()` |
| `PERSISTENT_BLOCK` | `block` | refused every time |
| `ASK` | `ask` | a human decides; the model can't override |
| `SOFT_BLOCK` | `retry` | escalates after `max_soft_attempts` to `escalate_to` (`HARD_STOP` or `ASK`) |
| `BLOCK_AND_WARN` | `warn` | the model may insist with the identical call in a later generation |
| `TOLERATE` | `allow` | the violation is in `decision.notes` |

- **The strongest severity broken decides**, whatever order the constraints are in;
  `nudge_max` lets one refusal report several violations of that severity (ordered by
  `rank=`).
- `decision.violations` are structured (`constraint_name`, `detail`, `repair`,
  `witnesses`); `render=` replaces the default feedback text.
- `check_chain([(tool, args), ...])` judges calls that run together (a shell command
  line) all or nothing; `decision.index` names the refused one.
- `to_state()` / `from_state()` carry the run (trace, counters, override pointer) as JSON
  between processes.
- `check_termination()` sends the model back, a bounded number of times
  (`max_termination_nudges`), while a constraint marked `applies_to_final_answer` is unmet.

`ConstraintEnforcer` is the same engine with the original interface (`"allow"` or
`(kind, feedback)`, raising `ConstraintViolationError` on a stop).

### AgentWithConstraints (backend-agnostic)

The top-level `AgentWithConstraints` supports both smolagents and LangChain backends
via the `backend=` parameter (default: `"smolagents"`):

```python
from agentltl import AgentWithConstraints, ConstraintSeverity, Constraint, Before

# smolagents backend (default)
agent = AgentWithConstraints(
    tools=my_tools,
    constraints=my_constraints,
    constraint_severities={"order_check": ConstraintSeverity.HARD_STOP},
    backend="smolagents",
)

# LangChain backend
from langchain_openai import ChatOpenAI

agent = AgentWithConstraints(
    tools=my_lc_tools,       # LangChain BaseTool instances
    constraints=my_constraints,
    constraint_severities={"order_check": ConstraintSeverity.SOFT_BLOCK},
    model_instance=ChatOpenAI(model="gpt-4o-mini"),
    backend="langchain",
)

result = agent.run("do the task")
# result["metrics"]["run_status"]  → "completed" | "stopped"
```

### ToolCallingAgentWithConstraints (smolagents low-level)

The low-level smolagents integration extends `ToolCallingAgent` to check constraints
*before* each tool call:

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

### ConstraintEnforcementMiddleware (LangChain low-level)

The low-level LangChain integration provides an `AgentMiddleware` for use with
`create_agent()`:

```python
from agentltl.integrations.langchain import (
    ConstraintEnforcementMiddleware,
    ConstraintSeverity,
)
from langchain.agents import create_agent

mw = ConstraintEnforcementMiddleware(
    constraints=my_constraints,
    constraint_severities={"order_check": ConstraintSeverity.SOFT_BLOCK},
    max_soft_attempts=3,
    soft_block_mode="hybrid",
)
agent = create_agent(model=my_model, tools=my_tools, middleware=[mw.as_middleware()])
result = agent.invoke({"messages": [HumanMessage(content="do the task")]})
status = mw.get_constraint_status()
```

### AgentWithAdditionalTools (smolagents, MCP-enabled)

General-purpose smolagents agent with multi-server MCP support:

```python
from agentltl import AgentWithAdditionalTools

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
| `HARD_STOP` | The offending tool call is **not executed**; the run stops immediately.  A `ConstraintViolationError` with `violation_type="HARD_STOP"` is recorded on the step. |
| `SOFT_BLOCK` | The offending tool call is **not executed**; the agent receives a structured `ConstraintViolationError` observation and may self-correct.  Escalates to a hard-stop after a configurable number of blocked attempts (controlled by `SoftBlockMode` — see below). |
| `BLOCK_AND_WARN` | The offending tool call is **not executed**; the agent receives a warning observation and may self-correct.  Unlike `SOFT_BLOCK`, it **never escalates** to `HARD_STOP`.  If the model's very next tool call (in the immediately following generation) is byte-identical — same tool name and same arguments — the call is treated as a deliberate override and is **executed**.  Any non-identical retry is blocked-and-warned again, with the insistence pointer updated to the new blocked call. |
| `PERSISTENT_BLOCK` | The offending tool call is **not executed**; the agent receives a warning observation and may self-correct.  Like `BLOCK_AND_WARN`, it **never escalates** to `HARD_STOP` and the run continues — but the block **can never be overridden**: re-issuing the exact same call is blocked again every time.  Use this for a non-fatal but non-negotiable guardrail where the model must ultimately choose a compliant action.  Per-constraint block counts are surfaced as `persistent_block_counts` in the constraint status. |
| `TOLERATE` | A warning is logged and execution continues. |

Default severity is `HARD_STOP`.  Override per-constraint via `constraint_severities` dict or globally via `default_severity`.

### Soft-block escalation modes (`SoftBlockMode`)

| Mode | Escalation trigger | Reset on success? |
|------|--------------------|-------------------|
| `cumulative` *(default)* | total violations for a constraint ≥ `max_soft_attempts` | No |
| `consecutive` | back-to-back violations ≥ `max_consecutive_soft_attempts` | Yes — counter resets after each successful tool call |
| `hybrid` | either `consecutive` **or** `cumulative` threshold reached | Consecutive counter resets; cumulative keeps counting |

```python
from agentltl.integrations.smolagents import (
    ToolCallingAgentWithConstraints, ConstraintSeverity, SoftBlockMode,
)

agent = ToolCallingAgentWithConstraints(
    tools=my_tools,
    model=my_model,
    constraints=my_constraints,
    constraint_severities={"order_check": ConstraintSeverity.SOFT_BLOCK},
    max_soft_attempts=5,                    # cumulative cap (all modes)
    soft_block_mode="hybrid",               # or SoftBlockMode.HYBRID
    max_consecutive_soft_attempts=3,        # consecutive cap (consecutive/hybrid only)
)
status = agent.get_constraint_status()
# status["soft_blocked_calls"]              — list of blocked attempts with step/constraint/detail
# status["soft_block_counts"]              — {constraint_name: num_blocks}
# status["soft_block_mode"]               — e.g. "hybrid"
# status["consecutive_soft_block_counts"] — {constraint_name: current_consecutive}
```

When a `SOFT_BLOCK` escalates, the agent history receives a clearly labelled message
(e.g. `[CONSTRAINT ESCALATION — hybrid mode, consecutive 3/3]`) that names the mode
and threshold, instructing the agent to take a fundamentally different approach.
`ConstraintViolationError.dict()` exposes `violation_type`, `soft_block_mode`, and
`threshold_str` as structured fields for downstream tooling.

---

## Runtime-safety classification

Refusing a call needs a violation that is visible *now*. AgentLTL classifies every
constraint from the values it can take on a partial trace (`reachable_values`, an abstract
interpretation over the same five-valued algebra the enforcer uses, so the two agree):

- **`SAFE`**      — it only fails for good: every refusal points at a real violation
  made by the refused call (`G(!now("x"))`, `CalledNTimes(t, n, "<=")`, `Before(a, b)`,
  `WithinSteps(a, b, n)`).
- **`UNSAFE`**    — it can fail while an obligation is open (`Called(t)`,
  `CalledWith(...)`): a blocking severity refuses every call until something meets it.
- **`INERT`**     — it can't fail before the run ends (`Eventually(...)`,
  `CalledInOrder(...)`): a blocking severity never fires. Check it at termination
  (`applies_to_final_answer=True` with `max_termination_nudges`) or bound it with
  `WithinSteps`.
- **`AMBIGUOUS`** — a `Predicate` that declared nothing, or an unknown node.

Pairing `UNSAFE` or `INERT` with a blocking severity (`HARD_STOP`, `SOFT_BLOCK` or
`PERSISTENT_BLOCK`) warns at registration time. (`BLOCK_AND_WARN` is exempt: the model
can always override it.)

### Values each node can take

| Operator | Values on a partial trace | Classification |
|---|---|---|
| `Now(tool)` | `FALSE`, `PENDING` (past the end), `TRUE` | SAFE |
| `Called(tool)`, `CalledWith(...)` | `PFALSE`, `TRUE` | UNSAFE |
| `CalledWithResult(...)` | `PFALSE`, `PENDING` (result not known yet), `TRUE` | UNSAFE |
| `CalledNTimes(tool, n, "<=" / "<")` | `PTRUE`, `FALSE` | SAFE |
| `CalledNTimes(tool, n, ">=" / ">")` | `PFALSE`, `TRUE` | UNSAFE (`>= 0`: INERT) |
| `CalledNTimes(tool, n, "==")` | `PFALSE`, `PTRUE`, `FALSE` | UNSAFE (`== 0`: SAFE) |
| `Before`, `AllBefore`, `InstanceBefore`, `WithinSteps` | `PENDING` until decided, then `FALSE` or `TRUE` | SAFE |
| `After(a, b)` | `PFALSE`, `FALSE`, `TRUE` | UNSAFE |
| `BranchCalled(correct, wrong)` | `PENDING`, `PTRUE`, `FALSE` | SAFE |
| `CalledInOrder(tools)` | `PENDING`, `TRUE` (it is `F(now a ∧ X F(now b ∧ …))`) | INERT |
| `Predicate(fn, desc)` | `PTRUE`/`PFALSE`, or `TRUE`/`FALSE` from a `{"final": True}` result | AMBIGUOUS unless `runtime_safe=` |
| `Not(φ)` | mirrors φ | — |
| `And` / `Or` / `Implies` | pairwise min / max / max(¬φ, ψ) | — |
| `Globally(φ)` / `ForAll` | min(φ, `PENDING`) | — |
| `Eventually(φ)` / `Exists` | max(φ, `PENDING`) / max(φ, `PFALSE`) | — |
| `Next(φ)`, `AtPosition(i, φ)` | φ, or `PENDING` when the position isn't there yet | — |
| `Until` / `WeakUntil` / `Release` | the usual unrolling, `PENDING` while undecided | — |

### `strict_runtime_safety`

By default each mismatch produces a `logger.warning` and the agent is
constructed normally.  Pass `strict_runtime_safety=True` to
`AgentWithConstraints`, `ToolCallingAgentWithConstraints`, or
`ConstraintEnforcementMiddleware` to turn warnings into a `ValueError`
at construction time:

```python
from agentltl import (
    AgentWithConstraints,
    Constraint,
    Eventually,
    Called,
    ConstraintSeverity,
)

agent = AgentWithConstraints(
    constraints=[Constraint("finish", Eventually(Called("done")))],
    constraint_severities={"finish": ConstraintSeverity.HARD_STOP},
    strict_runtime_safety=True,    # raises ValueError instead of warning
)
```

The framework never silently downgrades severities and never silently
treats unknown AST nodes as safe.

### `Predicate(..., runtime_safe=...)`

User-authored predicates default to `AMBIGUOUS` (the framework defers to
you).  Set the kwarg explicitly when you know the answer:

```python
from agentltl import Predicate

# Declared safe → SAFE, no warnings under any severity.
my_invariant = Predicate(lambda t, p: ..., "invariant", runtime_safe=True)

# Declared repairable → UNSAFE, warns / raises with HARD_STOP / SOFT_BLOCK.
my_liveness = Predicate(lambda t, p: ..., "liveness", runtime_safe=False)
```

### Worked example

```python
from agentltl import (
    AgentWithConstraints, Constraint, Eventually, Called, WithinSteps,
    ConstraintSeverity,
)

# Bad: HARD_STOP on unbounded liveness.
agent = AgentWithConstraints(
    constraints=[Constraint("finish", Eventually(Called("done")))],
    constraint_severities={"finish": ConstraintSeverity.HARD_STOP},
)
# WARNING agentltl.agents: Runtime-safety mismatch: constraint 'finish'
# (Eventually(operand=Called(tool='done'))) classified INERT but assigned
# severity HARD_STOP. It cannot fail before the run ends, so a blocking
# severity never fires. ...

# Good: bounded rewrite.
agent = AgentWithConstraints(
    constraints=[Constraint("finish", WithinSteps("plan", "done", 5))],
    constraint_severities={"finish": ConstraintSeverity.HARD_STOP},
)
# No warning — WithinSteps is SAFE.
```

`Before(a, b)` is SAFE: it is pending until the first `b`, and decided for good then.

### Inspecting classifications offline

```python
from agentltl import classify_constraints

reports = classify_constraints(
    constraints,
    constraint_severities,           # optional
    default_severity=ConstraintSeverity.HARD_STOP,
)
for r in reports:
    print(r.constraint_name, r.classification, r.compatible)
    if not r.compatible:
        print("  →", r.message)
```

Each `ClassificationReport` exposes `constraint_name`, `formula_repr`,
`classification` (`RuntimeSafety`), `assigned_severity`, `compatible`,
and `message`.

Constraints with `applies_to_final_answer=True` are skipped by the
registration-time hook (they run only at the end of the agent's run and
cannot cause spurious mid-run terminations) but still appear in
`classify_constraints` reports for inspection.

---

## Examples

| Example | Description |
|---------|-------------|
| `examples/01_post_hoc_verification.py` | `verify_trace()` on a completed trace |
| `examples/02_foltl_formulas.py` | Every formula type with pass/fail cases |
| `examples/03_linear_chain_agent.py` | A→B→C ordering with `AgentWithConstraints` |
| `examples/04_loop_termination_agent.py` | Polling loop compliance with `CalledNTimes` + `Predicate` |
| `examples/05_fan_out_fan_in_agent.py` | Fan-out/fan-in with `AllBefore` gate |
| `examples/06_mcp_tools_agent.py` | `AgentWithAdditionalTools` with an inline MCP server; multi-server and mixed local+MCP patterns |

Run any example:

```bash
cd agentltl
pip install -e ".[smolagents]" fastmcp
python examples/01_post_hoc_verification.py  # no LLM required
python examples/02_foltl_formulas.py         # no LLM required
export HF_TOKEN=...
python examples/03_linear_chain_agent.py     # requires HF_TOKEN
python examples/06_mcp_tools_agent.py        # requires HF_TOKEN + fastmcp
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
