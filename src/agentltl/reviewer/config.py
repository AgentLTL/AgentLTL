"""Reviewer configuration and self-describing config-string parsing.

The reviewer is the inference-time-feedback wrapper from *"Reinforced Agent:
Inference-Time Feedback for Tool-Calling Agents"*.  A run is fully described by
the paper's config string::

    {base}-{mechanism}{N}-{reviewer_model}-{prompt_version}

e.g. ``qwen3.6-27b-r2-qwen3.6-27b-v1_1`` or ``agentltl-r2-5-mini-v3-gepa``.

``mechanism`` is one of ``r`` (progressive feedback), ``s`` (best-of-N
selection) or ``g`` (best-of-N grading); ``N`` is the strategy budget.  Both the
base and reviewer model names may themselves contain hyphens, so the parser
anchors on the ``-{mechanism}{N}-`` token and on the known prompt-version
suffix rather than splitting naively on ``-``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .prompts import KNOWN_PROMPT_VERSIONS, DEFAULT_PROMPT_VERSION

# Strategy codes.
STRATEGY_PROGRESSIVE = "r"   # progressive feedback: regenerate on flagged error
STRATEGY_SELECTION = "s"     # best-of-N: reviewer picks best candidate
STRATEGY_GRADING = "g"       # best-of-N: reviewer scores each, argmax wins
VALID_STRATEGIES = (STRATEGY_PROGRESSIVE, STRATEGY_SELECTION, STRATEGY_GRADING)

# Best-of-N temperature sweep bounds (paper: 0.3 → 1.0).
SWEEP_TEMP_MIN = 0.3
SWEEP_TEMP_MAX = 1.0

# ``-r2-`` / ``-s4-`` / ``-g3-`` mechanism anchor.
_MECHANISM_RE = re.compile(r"-([rsg])(\d+)-")


def _normalize_version(raw: str) -> str:
    """Map a config-string version segment (``v3-gepa``) to a registry key.

    Config strings conventionally use ``-`` as the field separator, but our
    registry keys use ``_`` (``v3_gepa``).  We accept either spelling.
    """
    candidate = raw.replace("-", "_")
    if candidate in KNOWN_PROMPT_VERSIONS:
        return candidate
    if raw in KNOWN_PROMPT_VERSIONS:
        return raw
    raise ValueError(
        f"Unknown prompt version {raw!r}; known versions: "
        f"{sorted(KNOWN_PROMPT_VERSIONS)}"
    )


@dataclass
class ReviewerConfig:
    """Everything needed to instantiate and run a reviewer wrapper.

    ``reviewer_model`` defaults to ``base_model`` — a single model knob drives
    both roles (the paper's ``4o-r5-4o`` self-review setup) unless explicitly
    overridden for experiments with a different (e.g. reasoning) reviewer.
    """

    strategy: str = STRATEGY_PROGRESSIVE
    n: int = 2
    prompt_version: str = DEFAULT_PROMPT_VERSION
    base_model: Optional[str] = None
    reviewer_model: Optional[str] = None
    reviewer_temperature: Optional[float] = None
    reviewer_reasoning_effort: Optional[str] = None
    # Hard ceiling on reviewer LLM calls per agent turn — bounds every
    # strategy so the review loop can never run away or deadlock.
    max_reviewer_calls_per_turn: int = 8
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {VALID_STRATEGIES}; got {self.strategy!r}"
            )
        if self.n < 1:
            raise ValueError(f"N must be >= 1; got {self.n}")
        # Normalize a possibly-hyphenated version spelling to a registry key.
        self.prompt_version = _normalize_version(self.prompt_version)
        if self.reviewer_model is None:
            self.reviewer_model = self.base_model

    # ── Derived helpers ──────────────────────────────────────────────────────

    @property
    def is_progressive(self) -> bool:
        return self.strategy == STRATEGY_PROGRESSIVE

    @property
    def is_best_of_n(self) -> bool:
        return self.strategy in (STRATEGY_SELECTION, STRATEGY_GRADING)

    def sweep_temperatures(self) -> List[float]:
        """Temperatures for the best-of-N candidate sweep (``SWEEP_MIN→MAX``).

        With ``n == 1`` returns a single mid-sweep temperature.  Endpoints are
        included so the sweep spans the full configured range.
        """
        if self.n == 1:
            return [round((SWEEP_TEMP_MIN + SWEEP_TEMP_MAX) / 2, 4)]
        step = (SWEEP_TEMP_MAX - SWEEP_TEMP_MIN) / (self.n - 1)
        return [round(SWEEP_TEMP_MIN + i * step, 4) for i in range(self.n)]

    def to_config_string(self) -> str:
        base = self.base_model or "base"
        reviewer = self.reviewer_model or base
        version = self.prompt_version.replace("_", "-")
        return f"{base}-{self.strategy}{self.n}-{reviewer}-{version}"

    # ── Parsing ──────────────────────────────────────────────────────────────

    @classmethod
    def parse(cls, spec: str, **overrides) -> "ReviewerConfig":
        """Parse a self-describing config string into a :class:`ReviewerConfig`.

        Any keyword in ``overrides`` (e.g. ``reviewer_temperature``) is applied
        on top of the parsed fields.
        """
        match = _MECHANISM_RE.search(spec)
        if not match:
            raise ValueError(
                f"Config string {spec!r} has no '-{{r|s|g}}{{N}}-' mechanism "
                f"token (e.g. '-r2-')."
            )
        base_model = spec[: match.start()]
        strategy = match.group(1)
        n = int(match.group(2))
        rest = spec[match.end():]  # "{reviewer_model}-{prompt_version}"

        # Split the tail on the *last* known-version suffix.
        reviewer_model, prompt_version = cls._split_reviewer_and_version(rest)

        cfg_kwargs = dict(
            strategy=strategy,
            n=n,
            prompt_version=prompt_version,
            base_model=base_model or None,
            reviewer_model=reviewer_model or None,
        )
        cfg_kwargs.update(overrides)
        return cls(**cfg_kwargs)

    @staticmethod
    def _split_reviewer_and_version(rest: str) -> tuple[Optional[str], str]:
        """Separate ``{reviewer_model}-{prompt_version}`` where either may
        contain hyphens.  Anchors on the longest matching known version."""
        # Try each known version (and its hyphenated spelling) as a suffix,
        # preferring the longest match so e.g. ``v1_1`` beats ``v1``.
        candidates = []
        for v in KNOWN_PROMPT_VERSIONS:
            for spelled in {v, v.replace("_", "-")}:
                candidates.append((spelled, v))
        candidates.sort(key=lambda pair: len(pair[0]), reverse=True)
        for spelled, canonical in candidates:
            if rest == spelled:
                return None, canonical
            if rest.endswith("-" + spelled):
                reviewer = rest[: -(len(spelled) + 1)]
                return reviewer or None, canonical
        raise ValueError(
            f"Config-string tail {rest!r} does not end with a known prompt "
            f"version ({sorted(KNOWN_PROMPT_VERSIONS)})."
        )
