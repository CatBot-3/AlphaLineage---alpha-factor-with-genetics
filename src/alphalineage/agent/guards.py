"""P10-T1 - the agent guard layer. Written before any agent capability, deliberately.

An LLM that can propose a factor and immediately measure it is an automated overfitting machine.
That is not a criticism of the model: it is what any optimizer pointed at a fixed dataset does,
and a language model is a *better* optimizer than random search, which is precisely what makes it
more dangerous here. Everything in this module exists to bound that.

Four guards live here:

* :class:`SplitGuard` — the agent's data reach. Enforced by **absence**: the panel handed to the
  agent is physically truncated at ``train_end``, so validation and holdout rows are not in
  memory at all. A conditional can be bypassed by a bug; missing rows cannot.
* :class:`BudgetLedger` — tool calls, evaluations, wall clock, tokens. Exhaustion is a graceful
  stop with a partial transcript, never an exception escaping to the user.
* :class:`TrialLedger` — every evaluation is a trial, and an *informed* trial carries more
  selection bias per unit than the GP's near-random ones. The deflation must see them.
* :func:`validate_config_patch` — a tunable-key allowlist plus the existing ``GPConfig``
  bounds, so a patch is either valid or comes back as a retryable structured error.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from alphalineage.api.sessions import Boundaries
from alphalineage.core.gp import MAX_SEARCH_EVALUATIONS, GPConfig

# --- budgets ---------------------------------------------------------------------
#: Conservative by design. An agent that wants more should be given it explicitly by a human who
#: has seen what the previous run cost.
DEFAULT_MAX_TOOL_CALLS = 12
DEFAULT_MAX_EVALUATIONS = 8
DEFAULT_MAX_SECONDS = 300.0
DEFAULT_MAX_TOKENS = 200_000

MAX_TOOL_CALLS_CEILING = 60
MAX_EVALUATIONS_CEILING = 40
MAX_SECONDS_CEILING = 3600.0
MAX_TOKENS_CEILING = 2_000_000


class GuardViolation(RuntimeError):
    """A hard stop: the agent tried to do something it is structurally not allowed to do.

    Distinct from a tool returning an error to the model. A violation means a guard caught
    something that should have been impossible, so it fails the run rather than being handed
    back for the model to retry around.
    """


class BudgetExhausted(RuntimeError):
    """A soft stop: the run consumed its allowance. Handled, never surfaced as a failure."""


@dataclass(frozen=True)
class Budget:
    """What one agent run is allowed to spend."""

    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        for name, ceiling in (
            ("max_tool_calls", MAX_TOOL_CALLS_CEILING),
            ("max_evaluations", MAX_EVALUATIONS_CEILING),
            ("max_tokens", MAX_TOKENS_CEILING),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            if value > ceiling:
                raise ValueError(f"{name} must be at most {ceiling}, got {value}")
        if not isinstance(self.max_seconds, (int, float)) or self.max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {self.max_seconds!r}")
        if self.max_seconds > MAX_SECONDS_CEILING:
            raise ValueError(f"max_seconds must be at most {MAX_SECONDS_CEILING}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetLedger:
    """Consumption against a :class:`Budget`.

    Not thread-safe by design: one run, one loop.
    """

    budget: Budget = field(default_factory=Budget)
    tool_calls: int = 0
    evaluations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    started_at: float = field(default_factory=time.monotonic)
    #: Injectable so tests exercise the wall-clock stop without sleeping.
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.started_at = self.clock()

    @property
    def elapsed_seconds(self) -> float:
        return self.clock() - self.started_at

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def exhausted_reason(self) -> str | None:
        """Why the run should stop now, or ``None`` if it may continue."""
        if self.tool_calls >= self.budget.max_tool_calls:
            return f"tool-call budget reached ({self.budget.max_tool_calls})"
        if self.evaluations >= self.budget.max_evaluations:
            return f"evaluation budget reached ({self.budget.max_evaluations})"
        if self.elapsed_seconds >= self.budget.max_seconds:
            return f"time budget reached ({self.budget.max_seconds:g}s)"
        if self.total_tokens >= self.budget.max_tokens:
            return f"token budget reached ({self.budget.max_tokens})"
        return None

    @property
    def exhausted(self) -> bool:
        return self.exhausted_reason() is not None

    def charge_tool_call(self) -> None:
        if self.tool_calls >= self.budget.max_tool_calls:
            raise BudgetExhausted(f"tool-call budget reached ({self.budget.max_tool_calls})")
        self.tool_calls += 1

    def charge_evaluation(self) -> None:
        if self.evaluations >= self.budget.max_evaluations:
            raise BudgetExhausted(f"evaluation budget reached ({self.budget.max_evaluations})")
        self.evaluations += 1

    def charge_tokens(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.input_tokens += max(0, int(input_tokens))
        self.output_tokens += max(0, int(output_tokens))

    def remaining(self) -> dict[str, Any]:
        """What the model is told after each step, so it can plan rather than be cut off."""
        return {
            "tool_calls_left": max(0, self.budget.max_tool_calls - self.tool_calls),
            "evaluations_left": max(0, self.budget.max_evaluations - self.evaluations),
            "seconds_left": round(max(0.0, self.budget.max_seconds - self.elapsed_seconds), 1),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget.to_dict(),
            "tool_calls": self.tool_calls,
            "evaluations": self.evaluations,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "exhausted_reason": self.exhausted_reason(),
        }


# --- trials ----------------------------------------------------------------------
@dataclass
class TrialLedger:
    """Every candidate the agent scored, so the deflated Sharpe correction can see them.

    A trial is a selection opportunity. The GP's trials are close to random draws; an agent's
    are *informed* — it proposes what it believes will score well, having read the run's history.
    Per unit, that is worth more selection bias, so under-counting here would flatter every
    downstream deflation. The count folds into the session's ``cumulative_trials`` when any
    proposal from the run is approved.
    """

    count: int = 0
    expressions: list[str] = field(default_factory=list)

    def record(self, expression: str) -> int:
        self.count += 1
        if expression not in self.expressions:
            self.expressions.append(expression)
        return self.count

    @property
    def distinct(self) -> int:
        return len(self.expressions)

    def to_dict(self) -> dict[str, Any]:
        return {"trials": self.count, "distinct_expressions": self.distinct}


# --- the split guard -------------------------------------------------------------
class SplitGuard:
    """The agent's data reach, enforced by truncation rather than by a conditional.

    The agent has no date, universe, or split argument on any tool. This class computes the one
    window it may ever see — dates at or before ``train_end`` — and is used both to slice the
    panel up front and to assert afterwards that nothing later leaked in.
    """

    def __init__(self, boundaries: Boundaries) -> None:
        self.boundaries = boundaries
        self.train_end = pd.Timestamp(boundaries.train_end)
        self.valid_start = pd.Timestamp(boundaries.valid_start)
        self.test_start = pd.Timestamp(boundaries.test_start)
        if self.valid_start <= self.train_end:
            raise GuardViolation(
                "session boundaries are inconsistent: valid_start is not after train_end"
            )

    def allowed(self, dates: Iterable[Any]) -> pd.DatetimeIndex:
        """The subset of ``dates`` the agent may read: everything at or before ``train_end``."""
        idx = pd.DatetimeIndex(dates)
        return idx[idx <= self.train_end]

    def assert_within(self, dates: Iterable[Any], *, what: str = "agent evaluation") -> None:
        """Fail loudly if any date reaches the validation window or beyond.

        This is the assertion the fuzz test drives. It must never fire in normal operation — if
        it does, a code path found data it should not have been able to address.
        """
        idx = pd.DatetimeIndex(dates)
        if len(idx) == 0:
            return
        latest = idx.max()
        if latest > self.train_end:
            raise GuardViolation(
                f"{what} reached {latest.date().isoformat()}, past the frozen training boundary "
                f"{self.train_end.date().isoformat()}. The locked evidence splits are not "
                "reachable from an agent tool."
            )

    def describe(self) -> dict[str, Any]:
        """What the UI states plainly: measured on this, never on that."""
        return {
            "readable_through": self.train_end.date().isoformat(),
            "validation_starts": self.valid_start.date().isoformat(),
            "test_starts": self.test_start.date().isoformat(),
            "note": (
                "Agent evaluation reads training-window data only. The validation window and the "
                "locked holdout are not present in the panel the agent is given."
            ),
        }


# --- config patches --------------------------------------------------------------
#: Search knobs an agent may propose changing.
TUNABLE_KEYS: frozenset[str] = frozenset(
    {
        "population_size",
        "generations",
        "tournament_size",
        "crossover_rate",
        "subtree_mutation_rate",
        "point_mutation_rate",
        "max_depth",
        "max_nodes",
        "min_depth",
        "parsimony",
        "complexity_penalty_mode",
        "complexity_penalty_value",
        "parameter_neighbor_fraction",
        "exploration_profile",
        "elitism",
        "validation_folds",
        "time_budget_s",
        "enabled_categories",
        "enabled_formula_names",
    }
)

#: Keys an agent may never touch, with the reason. These are not search knobs — they define what
#: is being measured or gate what counts as a valid measurement, so moving them silently changes
#: the meaning of every number downstream rather than changing how hard the search looks.
PROTECTED_KEYS: dict[str, str] = {
    "horizon": (
        "changes what is being predicted, so metrics stop being comparable across segments "
        "and the embargo sized for the old horizon no longer covers the label window"
    ),
    "execution": (
        "changes which return is being predicted (same close, next open or next close), so "
        "every stored IC, backtest and holdout fingerprint would stop meaning the same thing; "
        "it is frozen when the session is created"
    ),
    "ic_method": "changes the definition of fitness itself, not how the search explores",
    "min_names": (
        "is a feasibility floor, not a tuning knob: lowering it lets a spuriously perfect IC "
        "on a near-empty cross-section through"
    ),
    "seed": (
        "re-rolling the seed to find a luckier trajectory is seed-hacking; it must be a "
        "deliberate human decision"
    ),
}


@dataclass(frozen=True)
class PatchResult:
    """The outcome of validating a proposed config patch."""

    ok: bool
    patch: dict[str, Any] = field(default_factory=dict)
    merged: dict[str, Any] = field(default_factory=dict)
    diff: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "patch": self.patch,
            "merged": self.merged,
            "diff": list(self.diff),
            "errors": list(self.errors),
        }


def validate_config_patch(
    patch: dict[str, Any],
    base_config: dict[str, Any],
    *,
    generations: int | None = None,
) -> PatchResult:
    """Validate a proposed ``GPConfig`` patch against the allowlist and the existing bounds.

    Errors are returned rather than raised: they go back to the model as a structured result it
    can correct, which is far more useful than failing the run. Bounds are not re-implemented —
    the patch is merged and run through ``GPConfig``'s own validation, so the agent can never be
    permitted something a human using the run form could not.
    """
    errors: list[str] = []
    if not isinstance(patch, dict):
        return PatchResult(ok=False, errors=["patch must be an object of config keys"])
    if not patch:
        return PatchResult(ok=False, errors=["patch is empty"])

    clean: dict[str, Any] = {}
    for key, value in patch.items():
        name = str(key)
        if name in PROTECTED_KEYS:
            errors.append(f"{name!r} cannot be changed by an agent: it {PROTECTED_KEYS[name]}")
            continue
        if name not in TUNABLE_KEYS:
            errors.append(
                f"{name!r} is not a tunable search parameter; "
                f"allowed keys are {', '.join(sorted(TUNABLE_KEYS))}"
            )
            continue
        clean[name] = value

    if errors:
        return PatchResult(ok=False, patch=clean, errors=errors)

    merged = {**base_config, **clean}
    try:
        config = GPConfig(**{k: v for k, v in merged.items() if k in _GP_CONFIG_FIELDS})
    except (TypeError, ValueError) as exc:
        return PatchResult(ok=False, patch=clean, errors=[str(exc)])

    planned = int(generations if generations is not None else config.generations)
    if config.population_size * planned > MAX_SEARCH_EVALUATIONS:
        return PatchResult(
            ok=False,
            patch=clean,
            errors=[
                f"population_size ({config.population_size}) x generations ({planned}) exceeds "
                f"the {MAX_SEARCH_EVALUATIONS} evaluation ceiling"
            ],
        )

    diff = [
        {"key": key, "from": base_config.get(key), "to": value}
        for key, value in sorted(clean.items())
        if base_config.get(key) != value
    ]
    if not diff:
        return PatchResult(
            ok=False,
            patch=clean,
            errors=["patch does not change anything from the current configuration"],
        )
    return PatchResult(ok=True, patch=clean, merged=merged, diff=diff)


_GP_CONFIG_FIELDS: frozenset[str] = frozenset(GPConfig.__dataclass_fields__)
