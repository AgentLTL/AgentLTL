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

book = rulebooks.load("ai_act")          # 46 rules, tool names are placeholders
enforcer = book.enforcer(max_termination_nudges=1)
```

Every rule is tested in `tests/test_rulebooks.py`: for each rule, a compliant and a
violating trace, after the fact and under the `Enforcer`, plus cases showing that a step
about another case, person, asset, dataset or model does not count. Run the Python
example with:

```bash
python examples/ai_act/rulebooks/aiact_enforcer.py
```

## Differences with the YAML modules

The modules here are written for the coding-agent guard
([agentltl-coding](https://github.com/AgentLTL/agentltl-coding): `settings`,
`use: theme-*`, `before/first/then`, `finally`, `$variables`); `tests-aiact.sh` calls its
`agentltl check`, which this library does not provide. The bundled `ai_act` rulebook keeps
their substance:

- **Argument variables** (`case_id: $c`) become `for_each: case_id`, through the
  first-order quantifier `ForAll`; **`succeeded: true`** becomes `succeeded: [tool]`.
- **No wildcards in tool names.** `mcp__*__apply_decision` becomes `apply_decision`;
  rename the tools to yours.
- **Not converted:** the guard's own CLI rules (`agentltl reset/disable`, `helm -n prod`
  and `pip install` argv parsing are expressed with generic tools `deploy_prod` and
  `install_package`), and the end-of-run "transparency tests pass after editing disclosure
  code" rule, which needs a regex on file paths inside a formula.

## Corrections made in the bundled version

These fix gaps found while converting. The French modules still have them; port them if
you use the modules with the coding-agent guard.

| Rule | Gap in the modules | Bundled version |
|------|--------------------|-----------------|
| Art. 5(1)(d) profiling | deny-list of bases: a call without `basis`, or `basis: "risk_model"`, passes | allow-list: only `objective_facts` / `verifiable facts` |
| Art. 5(1)(e) facial scraping | a call without `source` passes; `www`, `internet`, `social`, `scrape` not listed | source required; wider list |
| Art. 5(1)(f) emotions | prefix list misses `office`, `staff`, `interview`, `class`, `university`, `pupil` | wider list on word boundaries; `hrv_monitoring`, `medical_examination` stay allowed |
| Art. 5(1)(g) categorisation | `sex_life*` misses `sexual_life` | `sex_life` / `sexual`; biological `sex` is not an art. 5(1)(g) attribute |
| Art. 5(1)(h) real-time identification | a call without `realtime` passes | only `realtime=false` passes |
| Art. 50(4) public-interest text | an article without `ai_disclosure` skips the editorial review | only `ai_disclosure=true` skips it |
| Art. 53(1)(c) opt-out | per domain (correct) | same, plus the check must succeed |
| Art. 26(11), 12 | only `apply_decision` gated | `reject_application`, `terminate_contract` too |
| New | — | art. 43 conformity assessment and art. 49 EU registration before deploying a system not stated minimal or limited risk; art. 26(7) workers informed; art. 86 explanation answered |
