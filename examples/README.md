# AgentLTL Examples

| File | Requires LLM | Description |
|------|-------------|-------------|
| `01_post_hoc_verification.py` | No | Build a trace manually, run `verify_trace()`, inspect per-constraint results |
| `02_foltl_formulas.py` | No | Every formula type (`Called`, `Before`, `AllBefore`, `ForAll`, `Predicate`, …) with pass/fail cases |
| `03_linear_chain_agent.py` | Yes | A→B→C ordering with `AgentWithConstraints`; shows both `HARD_STOP` and `TOLERATE` severities |
| `04_loop_termination_agent.py` | Yes | Polling loop with `AgentWithAdditionalTools` + post-hoc `verify_trace()`; shows how to swap to runtime enforcement |
| `05_fan_out_fan_in_agent.py` | Yes | Fan-out/fan-in gate with `AllBefore`; post-hoc verification + failure case |
| `06_mcp_tools_agent.py` | Yes | `AgentWithAdditionalTools` connecting to a live `FileSystemMCPServer` via the `mcp_servers` dict; shows multi-server and mixed local+MCP patterns |

## Running examples

```bash
# Install
pip install -e ".[smolagents]"

# No-LLM examples (run immediately)
python examples/01_post_hoc_verification.py
python examples/02_foltl_formulas.py

# LLM examples (require HF_TOKEN or a local vLLM instance)
export HF_TOKEN=hf_...
export MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct   # optional override

python examples/03_linear_chain_agent.py
python examples/04_loop_termination_agent.py
python examples/05_fan_out_fan_in_agent.py
python examples/06_mcp_tools_agent.py           # also starts a local MCP server
```

## Using a local vLLM backend

```bash
export MODEL_TYPE=vllm
export VLLM_BASE_URL=http://localhost:8000/v1
export MODEL=HuggingFaceTB/SmolLM3-3B
python examples/03_linear_chain_agent.py
```

## Agent class quick reference

| Class | When to use |
|-------|------------|
| `Agent` | You already have `Tool` instances; no MCP needed |
| `AgentWithAdditionalTools` | You want to connect to MCP servers (+ optional local tools) |
| `AgentWithConstraints` | You want pre-execution FOLTL enforcement (HARD_STOP / TOLERATE) |

### `mcp_servers` dict format

```python
mcp_servers = {
    # HTTP transport (most common)
    "my_server": {
        "url":       "http://localhost:4000/mcp",
        "transport": "streamable-http",           # or "sse"
        "headers":   {"Authorization": "Bearer <token>"},  # optional
        "timeout":   30.0,                        # optional, seconds
    },
    # stdio transport (local subprocess)
    "local_server": {
        "command": "python",
        "args":    ["my_mcp_server.py"],
        "env":     {"SECRET": "value"},
    },
}
```

Pass to any agent class that supports MCP:

```python
agent = AgentWithAdditionalTools(mcp_servers=mcp_servers, ...)
agent = AgentWithConstraints(mcp_servers=mcp_servers, constraints=..., ...)
```
