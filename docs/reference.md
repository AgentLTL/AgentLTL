# AgentLTL – Return Value Reference

This document describes the exact structure of every dict returned by the
public API, with annotated JSON examples.

---

## Table of Contents

1. [Agent.run()](#agentrun)
2. [Metrics dict](#metrics-dict)
   - [Top-level keys](#top-level-keys)
   - [metrics\["tool\_calls"\]](#metricstool_calls)
   - [metrics\["steps"\]](#metricssteps)
   - [step\["model\_input\_parsed"\]](#stepmodel_input_parsed)
3. [AgentWithConstraints.run()](#agentwithconstraintsrun)
   - [Constraint fields added to metrics](#constraint-fields-added-to-metrics)
4. [ToolCallingAgentWithConstraints.get\_constraint\_status()](#toolcallingagentwithconstraintsget_constraint_status)
5. [verify\_trace()](#verify_trace)
6. [AgentWithSubAgents.run\_per\_requirement()](#agentwithsubagentsrun_per_requirement)

---

## Agent.run()

Returned by `Agent.run()`, `AgentWithAdditionalTools.run()`, and
`AgentWithConstraints.run()`.

```python
result = agent.run("your task")
```

```json
{
  "answer":  "<final answer string, or null on failure>",
  "metrics": { ... },
  "error":   "<exception message string, or null on success>"
}
```

| Key | Type | Description |
|-----|------|-------------|
| `answer` | `str \| None` | The agent's final answer text. `None` when the run failed or was stopped. |
| `metrics` | `dict` | Full run metrics — see [Metrics dict](#metrics-dict). |
| `error` | `str \| None` | Exception message if the run raised an uncaught error; `None` on success. |

---

## Metrics dict

`result["metrics"]` is the central data structure.  It is built by
`_extract_metrics_from_steps()` after every run.

### Top-level keys

```json
{
  "num_steps":       7,
  "num_tool_calls":  5,
  "input_tokens":    13268,
  "output_tokens":   127,
  "total_tokens":    13395,
  "tool_calls":      [ ... ],
  "steps":           [ ... ]
}
```

| Key | Type | Description |
|-----|------|-------------|
| `num_steps` | `int` | Total number of agent steps (reasoning + tool call + error steps combined). |
| `num_tool_calls` | `int` | Total number of individual tool invocations across all steps. |
| `input_tokens` | `int` | Sum of LLM input tokens across all steps. |
| `output_tokens` | `int` | Sum of LLM output tokens across all steps. |
| `total_tokens` | `int` | `input_tokens + output_tokens`. |
| `tool_calls` | `list[dict]` | Flat ordered list of every tool call — see [metrics\["tool_calls"\]](#metricstool_calls). This is the list consumed by `verify_trace()` and `Trace.from_metrics()`. |
| `steps` | `list[dict]` | Per-step detail records — see [metrics\["steps"\]](#metricssteps). |

---

### metrics\["tool_calls"\]

One entry per tool invocation, in chronological order.

```json
[
  {
    "step":         2,
    "tool_name":    "fetch_raw_data",
    "arguments":    { "source": "db" },
    "id":           "call_abc123",
    "tool_result":  "100 rows fetched",
    "action_output": "100 rows fetched"
  },
  {
    "step":         3,
    "tool_name":    "validate_data",
    "arguments":    {},
    "id":           "call_def456",
    "tool_result":  "OK",
    "action_output": "OK"
  }
]
```

| Key | Type | Description |
|-----|------|-------------|
| `step` | `int` | 1-based step number in which this call occurred. |
| `tool_name` | `str` | Name of the tool that was called. |
| `arguments` | `dict` | Arguments passed to the tool by the LLM. Empty dict `{}` when none. |
| `id` | `str \| None` | Tool call ID from the LLM response (OpenAI-style `call_xxx`). May be `None` for older providers. |
| `tool_result` | `str \| None` | Formatted observation string returned to the LLM (always a string). `None` if the step produced no observation. |
| `action_output` | `str \| None` | Raw Python return value of the tool, cast to `str`. Same as `tool_result` in most cases; may differ when smolagents post-processes the output (e.g. image/audio storage). |

> **Note for `verify_trace()` / `Trace.from_metrics()`:**
> `Trace.from_metrics()` reads `tool_name` (or `name`), `arguments` (or `tool_args`),
> and `tool_result` (or `result`) from each entry.  Both key variants are accepted.

---

### metrics\["steps"\]

One entry per smolagents `ActionStep`.  Includes all reasoning steps, tool-call
steps, and error steps.

```json
{
  "step_number":   2,
  "type":          "tool_call",
  "reasoning":     "I need to fetch the data first.\n<think>...</think>",
  "thinking":      "The user wants me to fetch and then validate...",
  "tool_name":     "fetch_raw_data",
  "tool_args":     { "source": "db" },
  "tool_id":       "call_abc123",
  "tool_result":   "100 rows fetched",
  "action_output": "100 rows fetched",
  "input_tokens":  1842,
  "output_tokens": 34,
  "start_time":    1716200000.123,
  "end_time":      1716200001.456,
  "duration":      1.333,
  "error":         null,
  "model_input_raw":    [ ... ],
  "model_input_parsed": { ... }
}
```

| Key | Type | Description |
|-----|------|-------------|
| `step_number` | `int` | 1-based index of this step in the run. |
| `type` | `str \| None` | `"tool_call"`, `"reasoning"`, or `"error"`. |
| `reasoning` | `str \| None` | Full raw model output text for this step (includes `<think>` tags if present). |
| `thinking` | `str \| None` | Extracted chain-of-thought: content inside `<think>…</think>` (Qwen3 style) or a structured `thinking`/`reasoning` content block. `None` if not present. |
| `tool_name` | `str \| None` | Name of the first tool called in this step. `None` for pure reasoning steps. |
| `tool_args` | `dict \| None` | Arguments of the first tool call. `None` for pure reasoning steps. |
| `tool_id` | `str \| None` | Call ID of the first tool call. `None` for pure reasoning steps. |
| `tool_result` | `str \| None` | Observation string returned to the model after the tool call(s) in this step. |
| `action_output` | `str \| None` | Raw Python return value of the tool(s), cast to `str`. |
| `input_tokens` | `int` | Input tokens consumed by the LLM at this step. |
| `output_tokens` | `int` | Output tokens produced by the LLM at this step. |
| `start_time` | `float \| None` | Unix timestamp when this step began. |
| `end_time` | `float \| None` | Unix timestamp when this step ended. |
| `duration` | `float \| None` | Wall-clock seconds for this step (`end_time - start_time`). |
| `error` | `str \| None` | Exception message if this step raised an error. `None` otherwise. |
| `model_input_raw` | `list[dict] \| None` | Serialised `ChatMessage` list sent to the LLM for this step. Each entry has `role`, `content`, and optionally `tool_calls` and `token_usage`. `None` if no LLM call happened. |
| `model_input_parsed` | `dict \| None` | Structured breakdown of `model_input_raw` — see [step\["model\_input\_parsed"\]](#stepmodel_input_parsed). |
| `step_tool_calls` | `list[dict] \| None` | Present only when the LLM issued **multiple** tool calls in a single step.  Each entry: `{"tool_name": str, "tool_args": dict}`. Absent for single-tool steps. |

---

### step\["model\_input\_parsed"\]

A structured view of the full conversation history sent to the LLM at this step.

```json
{
  "num_messages":   14,
  "approx_chars":   8300,
  "system_prompts": [ "You are a helpful agent..." ],
  "task":           "Fetch the data, validate it, then save it.",
  "history": [
    {
      "role":       "assistant",
      "type":       "tool_call",
      "text":       "I will call fetch_raw_data first.",
      "tool_calls": [
        { "id": "call_abc123", "name": "fetch_raw_data", "arguments": { "source": "db" } }
      ]
    },
    {
      "role":       "tool-response",
      "type":       "tool_result",
      "text":       "100 rows fetched",
      "tool_calls": null
    },
    {
      "role":       "assistant",
      "type":       "reasoning",
      "text":       "Good, now I will validate.",
      "tool_calls": null
    }
  ]
}
```

| Key | Type | Description |
|-----|------|-------------|
| `num_messages` | `int` | Total number of `ChatMessage` objects in the conversation at this step. |
| `approx_chars` | `int` | Approximate total character count of all message text (useful for context-length monitoring). |
| `system_prompts` | `list[str]` | Text of all `system` role messages (usually 1). |
| `task` | `str \| None` | Text of the first `user` role message — the original task string. |
| `history` | `list[dict]` | All subsequent messages (assistant reasoning, tool calls, tool responses) in order. Each entry has: `role` (str), `type` (`"reasoning"`, `"tool_call"`, `"tool_result"`, or `"message"`), `text` (str), `tool_calls` (list or null). |

---

## AgentWithConstraints.run()

Returns the same `{"answer", "metrics", "error"}` structure as `Agent.run()`,
but the `metrics` dict is augmented with five additional keys populated from
`get_constraint_status()`.

### Constraint fields added to metrics

```json
{
  "run_status":            "completed",
  "stopped_by_constraint": null,
  "constraint_violations": [],
  "constraint_checks":     6,
  "completed_trace": [
    {
      "tool_name":   "fetch_raw_data",
      "tool_args":   { "source": "db" },
      "tool_id":     "call_abc123",
      "tool_result": "100 rows fetched"
    }
  ]
}
```

| Key | Type | Description |
|-----|------|-------------|
| `run_status` | `str` | `"completed"` — run finished normally; `"stopped"` — aborted by a `HARD_STOP` constraint. |
| `stopped_by_constraint` | `str \| None` | Name of the constraint that triggered `HARD_STOP`. `None` if the run completed. |
| `constraint_violations` | `list[dict]` | All recorded violations (both `HARD_STOP` and `TOLERATE`). See violation record below. |
| `constraint_checks` | `int` | Total number of pre-execution `verify_trace()` calls performed during the run. |
| `completed_trace` | `list[dict]` | Ordered list of tool calls that were **actually executed** (i.e. passed constraint checks). Each entry: `tool_name`, `tool_args`, `tool_id`, `tool_result`. Does not include the blocked call if the run was stopped. |

**Violation record** (each item in `constraint_violations`):

```json
{
  "constraint_name": "fetch_before_clean",
  "severity":        "HARD_STOP",
  "step_number":     3,
  "tool_name":       "clean_readings",
  "tool_args":       {},
  "detail":          "\"fetch_raw_readings\" was never called (cannot precede \"clean_readings\")."
}
```

| Key | Type | Description |
|-----|------|-------------|
| `constraint_name` | `str` | Name of the violated constraint (matches `Constraint.name`). |
| `severity` | `str` | `"HARD_STOP"` or `"TOLERATE"`. |
| `step_number` | `int` | Agent step at which the violation was detected. |
| `tool_name` | `str` | The tool the agent tried to call when the violation was detected. |
| `tool_args` | `dict` | Arguments of the attempted tool call. |
| `detail` | `str` | Human-readable explanation from the FOLTL evaluator. |

---

## ToolCallingAgentWithConstraints.get\_constraint\_status()

Called directly on the low-level agent after a `run()`.

```python
status = agent.get_constraint_status()
```

```json
{
  "status":            "stopped",
  "stopped_by":        "fetch_before_clean",
  "blocked_tool_call": {
    "tool_name":   "clean_readings",
    "tool_args":   {},
    "tool_id":     "call_xyz789",
    "tool_result": "BLOCKED_BY_CONSTRAINT",
    "blocked_by":  "fetch_before_clean",
    "step_number": 3
  },
  "violations": [
    {
      "constraint_name": "fetch_before_clean",
      "severity":        "HARD_STOP",
      "step_number":     3,
      "tool_name":       "clean_readings",
      "tool_args":       {},
      "detail":          "..."
    }
  ],
  "completed_trace": [
    { "tool_name": "validate_data", "tool_args": {}, "tool_id": "call_aaa", "tool_result": "OK" }
  ],
  "constraint_checks": 3
}
```

| Key | Type | Description |
|-----|------|-------------|
| `status` | `str` | `"completed"` or `"stopped"`. |
| `stopped_by` | `str \| None` | Name of the `HARD_STOP` constraint that halted the run. `None` if completed. |
| `blocked_tool_call` | `dict \| None` | The tool call that was blocked. Contains `tool_name`, `tool_args`, `tool_id`, `tool_result` (`"BLOCKED_BY_CONSTRAINT"`), `blocked_by`, `step_number`. `None` if not stopped. |
| `violations` | `list[dict]` | All constraint violation records (same schema as above). |
| `completed_trace` | `list[dict]` | Tool calls that successfully executed before the run ended. Each: `tool_name`, `tool_args`, `tool_id`, `tool_result`. |
| `constraint_checks` | `int` | Number of speculative `verify_trace()` evaluations performed. |

---

## verify\_trace()

```python
from agentltl import verify_trace
result = verify_trace(metrics, constraints)
```

```json
{
  "compliance_score":  0.8,
  "compliance_label":  "PARTIAL",
  "details":           "Violated: ['fetch_before_clean']. Missing tools: ['fetch_raw_data'].",
  "tool_sequence":     ["validate_data", "clean_readings", "save_results"],
  "constraints": [
    {
      "name":        "fetch_before_clean",
      "weight":      1.0,
      "passed":      false,
      "detail":      "\"fetch_raw_data\" was never called (cannot precede \"clean_readings\").",
      "description": "Raw data must be fetched before cleaning"
    },
    {
      "name":        "clean_before_save",
      "weight":      1.0,
      "passed":      true,
      "detail":      "\"clean_readings\" (call #2) before \"save_results\" (call #3).",
      "description": "Cleaned data must be saved after cleaning"
    }
  ]
}
```

| Key | Type | Description |
|-----|------|-------------|
| `compliance_score` | `float` | C(τ, G_P) ∈ [0.0, 1.0]. `1 − Σ(violated weights) / Σ(all weights)`. Rounded to 6 decimal places. |
| `compliance_label` | `str` | `"FULL"` (score = 1.0), `"PARTIAL"` (0 < score < 1), or `"VIOLATION"` (score = 0.0). |
| `details` | `str` | Human-readable summary. On full compliance: configurable via `full_compliance_detail` kwarg (default `"All constraints satisfied."`). On failure: lists violated constraint names and any missing tools. |
| `tool_sequence` | `list[str]` | Ordered list of tool names extracted from the trace (= `Trace.events`). This is the flat projection φ(τ) used by ordering predicates. |
| `constraints` | `list[dict]` | Per-constraint evaluation records — one per `Constraint` in the input list, in the same order. |

**Per-constraint record** (each item in `constraints`):

| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | `Constraint.name`. |
| `weight` | `float` | Effective weight used for scoring (after any `weight_overrides`). |
| `passed` | `bool` | `True` if the formula was satisfied on the trace. |
| `detail` | `str` | Human-readable verdict from the FOLTL evaluator, e.g. `"\"fetch\" (call #1) before \"save\" (call #3)."`. |
| `description` | `str` | `Constraint.description` (may be empty string). |

### verify_trace() parameters

```python
verify_trace(
    metrics,                              # dict with "tool_calls" key
    constraints,                          # list[Constraint]
    weight_overrides={"c_name": 2.0},    # optional per-constraint weight override
    full_compliance_detail="All good.",  # custom message when score = 1.0
    partial_trace=False,                 # True → trigger semantics (for pre-exec checking)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | `dict` | — | Agent run metrics. Must contain `"tool_calls"` list. Accepts both `"tool_args"` and `"arguments"` key variants in each call entry. |
| `constraints` | `Sequence[Constraint]` | — | Ordered list of constraints to evaluate. |
| `weight_overrides` | `dict[str, float] \| None` | `None` | Per-constraint weight overrides keyed by `Constraint.name`. |
| `full_compliance_detail` | `str` | `"All constraints satisfied."` | Message in `details` when score = 1.0. |
| `partial_trace` | `bool` | `False` | When `True`, ordering constraints (`Before`, `AllBefore`) pass vacuously if the successor tool has not yet appeared. Use this during pre-execution speculative checking. |

---

## AgentWithSubAgents.run\_per\_requirement()

```python
result = agent.run_per_requirement(requirements, tasks, template)
```

```json
{
  "answer":      "Completed 3 sub-agent evaluations.",
  "sub_results": [ ... ],
  "metrics":     { ... }
}
```

| Key | Type | Description |
|-----|------|-------------|
| `answer` | `str` | Summary string (always `"Completed N sub-agent evaluations."`). |
| `sub_results` | `list[dict]` | One entry per requirement. Each entry is the standard `{"answer", "metrics", "error"}` dict from `AgentWithAdditionalTools.run()`, augmented with `requirement_id` (str), `requirement_description` (str), and `tasks` (list[str]). |
| `metrics` | `dict` | Aggregated metrics — see table below. |

**Aggregated metrics dict** (result\["metrics"\]):

```json
{
  "total_requirements":           3,
  "total_input_tokens":           9400,
  "total_output_tokens":          310,
  "total_tokens":                 9710,
  "total_steps":                  21,
  "total_tool_calls":             15,
  "avg_tokens_per_requirement":   3236.67,
  "avg_steps_per_requirement":    7.0,
  "avg_tool_calls_per_requirement": 5.0,
  "per_requirement_metrics": [
    {
      "requirement_index":  0,
      "requirement_id":     "req_1",
      "input_tokens":       3200,
      "output_tokens":      110,
      "total_tokens":       3310,
      "num_steps":          7,
      "num_tool_calls":     5,
      "steps":              [ ... ],
      "tool_calls":         [ ... ]
    }
  ],
  "all_tool_calls": [
    {
      "requirement_id":    "req_1",
      "requirement_index": 0,
      "step":              2,
      "tool_name":         "fetch_raw_data",
      "arguments":         {},
      "tool_result":       "...",
      "action_output":     "..."
    }
  ]
}
```

| Key | Type | Description |
|-----|------|-------------|
| `total_requirements` | `int` | Number of requirements processed. |
| `total_input_tokens` | `int` | Sum of `input_tokens` across all sub-agents. |
| `total_output_tokens` | `int` | Sum of `output_tokens` across all sub-agents. |
| `total_tokens` | `int` | Sum of `total_tokens` across all sub-agents. |
| `total_steps` | `int` | Sum of `num_steps` across all sub-agents. |
| `total_tool_calls` | `int` | Sum of `num_tool_calls` across all sub-agents. |
| `avg_tokens_per_requirement` | `float` | `total_tokens / total_requirements`. |
| `avg_steps_per_requirement` | `float` | `total_steps / total_requirements`. |
| `avg_tool_calls_per_requirement` | `float` | `total_tool_calls / total_requirements`. |
| `per_requirement_metrics` | `list[dict]` | Per-requirement breakdown. Each entry includes `requirement_index`, `requirement_id`, token counts, step counts, and the full `steps` and `tool_calls` lists from that sub-agent's metrics. |
| `all_tool_calls` | `list[dict]` | Flat list of every tool call across all sub-agents. Each entry is the standard tool call dict augmented with `requirement_id` and `requirement_index`. |
