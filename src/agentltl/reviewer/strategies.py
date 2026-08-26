"""The three review strategies behind one interface.

Strategies are deliberately decoupled from smolagents: they operate on two
injected callables, so they can be unit-tested with stubs.

    generate(feedback: str | None, temperature: float | None) -> Candidate
        Produce one provisional tool call (optionally with reviewer feedback
        injected, and/or at a specific sampling temperature).

    review(candidates: list[Candidate]) -> ReviewResult
        Run ONE reviewer LLM call over the given candidate(s) and return the
        parsed, fail-open verdict.  Must not raise (the caller wraps the model
        call and defaults to a "no error" verdict on failure).

All three strategies return a :class:`StrategyOutcome` carrying the chosen
candidate plus the telemetry the benchmark reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import (
    ReviewerConfig,
    STRATEGY_PROGRESSIVE,
    STRATEGY_SELECTION,
    STRATEGY_GRADING,
)
from .parsing import ReviewResult


@dataclass
class Candidate:
    """One provisional tool call plus its opaque model message."""

    chat_message: Any                       # smolagents ChatMessage (opaque here)
    call: Dict[str, Any]                     # {"name", "arguments"} for prompt/telemetry
    temperature: Optional[float] = None

    def call_key(self):
        """Hashable identity of the call for change detection.

        ``call`` may be ``None`` when a (re)generated candidate makes no tool
        call at all (backends with tool_choice="auto" can emit a final
        natural-language answer instead) — treat that as its own distinct key.
        """
        call = self.call or {}
        args = call.get("arguments", call.get("tool_args", {}))
        try:
            import json
            args_key = json.dumps(args, sort_keys=True, default=str)
        except Exception:
            args_key = str(args)
        return (call.get("name") or call.get("tool_name"), args_key)


@dataclass
class StrategyOutcome:
    chosen: Candidate
    provisional: Candidate
    reviewer_calls: int = 0
    rounds: int = 0
    changed: bool = False
    parse_failures: int = 0
    last_message: str = ""
    scores: Optional[List[float]] = None
    selected_index: Optional[int] = None


GenerateFn = Callable[[Optional[str], Optional[float]], Candidate]
ReviewFn = Callable[[List[Candidate]], ReviewResult]


def _progressive(config: ReviewerConfig, generate: GenerateFn, review: ReviewFn) -> StrategyOutcome:
    """rN — inspect the provisional call; on a flagged error inject feedback and
    regenerate, up to N review rounds or until approval."""
    cap = min(config.n, config.max_reviewer_calls_per_turn)
    provisional = generate(None, None)
    current = provisional
    reviewer_calls = rounds = parse_failures = 0
    last_message = ""

    for _ in range(cap):
        res = review([current])
        reviewer_calls += 1
        if res.parse_failed:
            parse_failures += 1
        if not res.error:
            break  # approved — execute as-is
        last_message = res.message or "The provisional tool call was flagged as incorrect."
        rounds += 1
        # Regenerate with the reviewer's feedback injected.
        current = generate(last_message, None)

    return StrategyOutcome(
        chosen=current,
        provisional=provisional,
        reviewer_calls=reviewer_calls,
        rounds=rounds,
        changed=current.call_key() != provisional.call_key(),
        parse_failures=parse_failures,
        last_message=last_message,
    )


def _generate_sweep(config: ReviewerConfig, generate: GenerateFn) -> List[Candidate]:
    return [generate(None, t) for t in config.sweep_temperatures()]


def _selection(config: ReviewerConfig, generate: GenerateFn, review: ReviewFn) -> StrategyOutcome:
    """sN — emit N candidates over a temperature sweep; reviewer picks the best."""
    candidates = _generate_sweep(config, generate)
    provisional = candidates[0]
    res = review(candidates)
    idx = res.selected_index
    if idx is None or not (0 <= idx < len(candidates)):
        idx = 0
    chosen = candidates[idx]
    return StrategyOutcome(
        chosen=chosen,
        provisional=provisional,
        reviewer_calls=1,
        rounds=0,
        changed=chosen.call_key() != provisional.call_key(),
        parse_failures=1 if res.parse_failed else 0,
        last_message=res.message,
        selected_index=idx,
    )


def _grading(config: ReviewerConfig, generate: GenerateFn, review: ReviewFn) -> StrategyOutcome:
    """gN — emit N candidates; reviewer scores each in [0,1]; highest wins."""
    candidates = _generate_sweep(config, generate)
    provisional = candidates[0]
    cap = config.max_reviewer_calls_per_turn
    scores: List[float] = []
    parse_failures = 0
    reviewer_calls = 0
    last_message = ""
    for cand in candidates:
        if reviewer_calls >= cap:
            scores.append(0.0)  # cap hit — cannot grade further
            continue
        res = review([cand])
        reviewer_calls += 1
        if res.parse_failed:
            parse_failures += 1
        last_message = res.message or last_message
        scores.append(res.score if res.score is not None else 0.0)
    # argmax (first max wins → ties prefer the lower-temperature candidate).
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    chosen = candidates[best_idx]
    return StrategyOutcome(
        chosen=chosen,
        provisional=provisional,
        reviewer_calls=reviewer_calls,
        rounds=0,
        changed=chosen.call_key() != provisional.call_key(),
        parse_failures=parse_failures,
        last_message=last_message,
        scores=scores,
        selected_index=best_idx,
    )


_DISPATCH = {
    STRATEGY_PROGRESSIVE: _progressive,
    STRATEGY_SELECTION: _selection,
    STRATEGY_GRADING: _grading,
}


def run_strategy(config: ReviewerConfig, generate: GenerateFn, review: ReviewFn) -> StrategyOutcome:
    """Dispatch to the configured strategy and return its outcome."""
    try:
        impl = _DISPATCH[config.strategy]
    except KeyError:  # pragma: no cover — guarded by ReviewerConfig
        raise ValueError(f"Unknown strategy {config.strategy!r}")
    return impl(config, generate, review)
