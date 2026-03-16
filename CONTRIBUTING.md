# Contributing to AgentLTL

Thank you for considering contributing!  This document covers the most
important guidelines.

---

## Development setup

```bash
git clone https://github.com/your-org/agentltl.git
cd agentltl
pip install -e ".[dev,smolagents]"
```

Run tests:

```bash
pytest
```

Lint:

```bash
ruff check src/ examples/
```

---

## Adding a new formula type

1. **Define the AST node** in `src/agentltl/_ast.py` as a frozen dataclass
   inheriting from `Formula`.
2. **Add evaluation logic** in `src/agentltl/_evaluator.py` inside
   `LTLEvaluator.evaluate()`.
3. **Add to `substitute()`** in `_ast.py` if the formula has `Var`-valued
   fields.
4. **Export** from `src/agentltl/__init__.py` and add to `__all__`.
5. **Write tests** in `tests/` covering pass and fail cases.
6. **Document** the new type in `README.md` (formula reference table).

---

## Adding a new integration

Create a new sub-package under `src/agentltl/integrations/<framework>/`.

The integration must:

- Import from `agentltl` (not `foltl`) for the constraint engine.
- Declare the framework as an optional dependency in `pyproject.toml`.
- Export its public API from `integrations/<framework>/__init__.py`.
- Include at least one example in `examples/`.

---

## Pull request checklist

- [ ] New formula types have both pass and fail unit tests
- [ ] Public API additions are exported in `__init__.py` and `__all__`
- [ ] `README.md` updated if public API changed
- [ ] `ruff check` passes with no errors
- [ ] No hard dependencies added to the core `[project] dependencies`

---

## Reporting bugs

Open an issue on GitHub with:
- Python version and OS
- AgentLTL version (`python -c "import agentltl; print(agentltl.__version__)"`)
- Minimal reproducing example
