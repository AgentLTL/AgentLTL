# AgentLTL Examples

| File | Requires LLM | Description |
|------|-------------|-------------|
| `01_post_hoc_verification.py` | No | Build a trace manually, run `verify_trace()`, inspect per-constraint results |
| `02_foltl_formulas.py` | No | Every formula type (`Called`, `Before`, `AllBefore`, `ForAll`, `Predicate`, …) with pass/fail cases |
| `03_linear_chain_agent.py` | Yes | A→B→C ordering enforced with `AgentWithConstraints` (HARD_STOP) |
| `04_loop_termination_agent.py` | Yes | Polling loop compliance: `CalledNTimes` + `Predicate` on last poll result |
| `05_fan_out_fan_in_agent.py` | Yes | Fan-out/fan-in gate enforced with `AllBefore` |

## Running examples

```bash
# Install
cd agentltl
pip install -e ".[smolagents]"

# No-LLM examples (run immediately)
python examples/01_post_hoc_verification.py
python examples/02_foltl_formulas.py

# LLM examples (require HF_TOKEN or VLLM setup)
export HF_TOKEN=hf_...
export MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct          # optional override
python examples/03_linear_chain_agent.py
python examples/04_loop_termination_agent.py
python examples/05_fan_out_fan_in_agent.py
```

## Using a local vLLM backend

```bash
export MODEL_TYPE=vllm
export VLLM_BASE_URL=http://localhost:8000/v1
export MODEL=HuggingFaceTB/SmolLM3-3B
python examples/03_linear_chain_agent.py
```
