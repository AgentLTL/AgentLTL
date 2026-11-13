# AI Act × AgentLTL

Rulebooks that enforce, tool call by tool call, the obligations of the EU AI Act
(Regulation (EU) 2024/1689, as amended by the 2026 digital omnibus on AI).

| File | What it is |
|------|------------|
| `plan-aiact-agentltl.md` | Analysis and action plan (French): the Act as of November 2026, article → rule coverage matrix, phased rollout, limits |
| `rulebooks/00-socle.yaml` … `07-equipe-dev.yaml` | The rules by module, commented (French) |
| `rulebooks/AGENTLTL.yaml` | The eight modules merged |
| `rulebooks/aiact_enforcer.py` | A runnable `Enforcer` with four AI Act constraints for a business agent |
| `rulebooks/tests-aiact.sh` | Regression scenarios for the YAML modules |

## Using it with this library

The bundled rulebook `ai_act` is the conversion of these modules to the library's format:

```python
from agentltl import rulebooks

book = rulebooks.load("ai_act")          # 41 rules, tool names are placeholders
enforcer = book.enforcer(max_termination_nudges=1)
```

Every rule is tested in `tests/test_rulebooks.py`. Run the Python example with:

```bash
python examples/ai_act/rulebooks/aiact_enforcer.py
```

## Differences with the YAML modules

The modules here are written for the `agentltl` command-line guard (`settings`, `use: theme-*`,
`before/first/then`, `finally`, argument variables such as `$case`). `tests-aiact.sh` calls
`agentltl check`, which this library does not provide. The bundled `ai_act` rulebook keeps
their substance with these changes:

- **No argument variables.** "The human review of the same case" becomes "a human review
  since the previous decision": each decision needs its own fresh step, which is stricter
  per call but cannot check that the step was about the same case.
- **No wildcards in tool names.** `mcp__*__apply_decision` becomes `apply_decision`;
  rename the tools to yours.
- **No `succeeded:` condition.** A step counts when it was called; make the tool fail
  loudly if its check does not pass.
- **Not converted:** the guard's own CLI rules (`agentltl reset/disable`), shell-specific
  rules (`helm -n prod`, `pip install` argv parsing) are expressed with generic tools
  (`deploy_prod`, `install_package`), and the end-of-run "transparency tests pass after
  editing disclosure code" rule, which needs a regex on file paths inside a formula.
