"""P1-T2 - the primitive registry: operators, operands, and ephemeral terminals.

Every primitive declares its arity, input types, and output type, so generation,
crossover, and mutation can only ever produce semantically valid trees. All operator
implementations are vectorized (numpy/pandas) and NaN-tolerant: they never raise on a
type-valid tree, they return NaN where a value is undefined (this is what lets the
generator's 10k-tree validity sweep pass).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from alphalineage.core.types import DType, is_subtype

# Ephemeral-constant sample spaces (point mutation tweaks these in Phase 2).
DEFAULT_WINDOWS: tuple[int, ...] = (2, 3, 5, 10, 20, 30, 60)
DEFAULT_SCALARS: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
# Generous enough for research, bounded so malformed JSON cannot request abusive rolling work.
MAX_WINDOW = 100_000


class Kind(Enum):
    OPERATOR = "operator"  # internal node: typed args -> output
    OPERAND = "operand"  # leaf: a panel field (a Series)
    EPHEMERAL = "ephemeral"  # leaf: a sampled SCALAR or WINDOW constant


@dataclass(frozen=True)
class Primitive:
    name: str
    kind: Kind
    out_type: DType
    arg_types: tuple[DType, ...] = ()
    fn: Callable[..., pd.DataFrame] | None = None  # operators
    panel_field: str | None = None  # operands
    sampler: Callable[[random.Random], Any] | None = None  # ephemerals
    macro_body: Any = None  # user operators: a typed body tree (Node) expanded at evaluation

    @property
    def arity(self) -> int:
        return len(self.arg_types)

    @property
    def is_terminal(self) -> bool:
        return self.kind in (Kind.OPERAND, Kind.EPHEMERAL)


# --- vectorized operator implementations -----------------------------------------
def _finite(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def _safe_repr(value: object) -> str:
    try:
        return repr(value)
    except (OverflowError, ValueError):
        return f"<{type(value).__name__} outside printable numeric range>"


def checked_window(value: object) -> int:
    """Return a valid lookback or raise instead of silently coercing bad input."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"window must be a positive integer, got {_safe_repr(value)}")
    if value > MAX_WINDOW:
        raise ValueError(f"window must be at most {MAX_WINDOW}, got {_safe_repr(value)}")
    return value


def checked_scalar(value: object) -> float:
    """Return a finite numeric scalar or raise with a stable validation error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"scalar must be a finite number, got {_safe_repr(value)}")
    try:
        scalar = float(value)
    except OverflowError as exc:
        raise ValueError(f"scalar must be a finite number, got {_safe_repr(value)}") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"scalar must be a finite number, got {_safe_repr(value)}")
    return scalar


def _roll(a: pd.DataFrame, w: int) -> Any:
    window = checked_window(w)
    return a.rolling(window=window, min_periods=window)


def _add(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a + b


def _sub(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a - b


def _mul(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a * b


def _div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return _finite(a / b.replace(0.0, np.nan))


def _mul_scalar(a: pd.DataFrame, s: float) -> pd.DataFrame:
    return a * checked_scalar(s)


def _add_scalar(a: pd.DataFrame, s: float) -> pd.DataFrame:
    return a + checked_scalar(s)


def _signed_power(a: pd.DataFrame, s: float) -> pd.DataFrame:
    return _finite(np.sign(a) * np.power(a.abs(), checked_scalar(s)))


def _log(a: pd.DataFrame) -> pd.DataFrame:
    # Sign-safe log, defined for all reals: sign(x) * log(1 + |x|).
    return np.sign(a) * np.log1p(a.abs())


def _abs(a: pd.DataFrame) -> pd.DataFrame:
    return a.abs()


def _sign(a: pd.DataFrame) -> pd.DataFrame:
    return np.sign(a)


def _neg(a: pd.DataFrame) -> pd.DataFrame:
    return -a


def _ts_mean(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return _roll(a, w).mean()


def _ts_ema(a: pd.DataFrame, w: int) -> pd.DataFrame:
    window = checked_window(w)
    return a.ewm(span=window, adjust=False, min_periods=window).mean()


def _ts_std(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return _roll(a, w).std()


def _ts_std_pop(a: pd.DataFrame, w: int) -> pd.DataFrame:
    """Population rolling deviation (``ddof=0``), used by chart-style Bollinger bands."""
    return _roll(a, w).std(ddof=0)


def _ts_rma(a: pd.DataFrame, w: int) -> pd.DataFrame:
    """Wilder's moving average, seeded by the first complete arithmetic-mean window.

    A non-finite observation breaks the sequence.  Output resumes only after another complete
    finite seed window, matching the strict warm-up behavior of the other rolling primitives.
    """
    window = checked_window(w)
    values = a.to_numpy(dtype=float, copy=False)
    result = np.full(values.shape, np.nan, dtype=float)
    for column in range(values.shape[1]):
        state = math.nan
        seed: list[float] = []
        for row in range(values.shape[0]):
            value = float(values[row, column])
            if not math.isfinite(value):
                state = math.nan
                seed.clear()
                continue
            if math.isnan(state):
                seed.append(value)
                if len(seed) < window:
                    continue
                # Spell out sequential accumulation: Python 3.12's built-in ``sum`` uses a
                # compensated algorithm, while the native evaluator intentionally mirrors the
                # same deterministic double operation order on every supported Python version.
                seed_sum = 0.0
                for observation in seed:
                    seed_sum += observation
                state = seed_sum / window
                seed.clear()
            else:
                state = (state * (window - 1) + value) / window
            result[row, column] = state
    return pd.DataFrame(result, index=a.index, columns=a.columns)


def _ts_recursive_smooth(a: pd.DataFrame, w: int, initial: float) -> pd.DataFrame:
    """Wilder-style recursive smoothing with an explicit initial state.

    Finite observations update ``state = ((w - 1) * state + value) / w`` immediately. Missing
    observations emit NaN without discarding the state, which is the conventional KDJ behavior.
    """
    window = checked_window(w)
    start = checked_scalar(initial)
    values = a.to_numpy(dtype=float, copy=False)
    result = np.full(values.shape, np.nan, dtype=float)
    for column in range(values.shape[1]):
        state = start
        for row in range(values.shape[0]):
            value = float(values[row, column])
            if not math.isfinite(value):
                continue
            state = (state * (window - 1) + value) / window
            result[row, column] = state
    return pd.DataFrame(result, index=a.index, columns=a.columns)


def _ts_sum(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return _roll(a, w).sum()


def _ts_min(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return _roll(a, w).min()


def _ts_max(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return _roll(a, w).max()


def _ts_rank(a: pd.DataFrame, w: int) -> pd.DataFrame:
    # Percentile rank of the current value within its trailing window. ``method='max'`` is
    # exactly the historical ``mean(window <= current)`` tie policy, but pandas executes the
    # rolling rank in compiled code instead of calling Python once per date/symbol/window.
    return _roll(a, w).rank(method="max", pct=True)


def _decay_linear(a: pd.DataFrame, w: int) -> pd.DataFrame:
    window = checked_window(w)
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()

    def f(x: np.ndarray) -> float:
        return float(np.dot(x, weights))

    return _roll(a, w).apply(f, raw=True)


def _delta(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return a - a.shift(checked_window(w))


def _delay(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return a.shift(checked_window(w))


def _ts_cov(a: pd.DataFrame, b: pd.DataFrame, w: int) -> pd.DataFrame:
    return _roll(a, w).cov(b, pairwise=False)


def _ts_corr(a: pd.DataFrame, b: pd.DataFrame, w: int) -> pd.DataFrame:
    cov = _roll(a, w).cov(b, pairwise=False)
    denom = (_roll(a, w).std() * _roll(b, w).std()).replace(0.0, np.nan)
    return _finite(cov / denom)


def _gt(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return (a > b) & a.notna() & b.notna()


def _lt(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return (a < b) & a.notna() & b.notna()


def _ge(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return (a >= b) & a.notna() & b.notna()


def _le(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return (a <= b) & a.notna() & b.notna()


def _and(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a.astype(bool) & b.astype(bool)


def _or(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a.astype(bool) | b.astype(bool)


def _not(a: pd.DataFrame) -> pd.DataFrame:
    return ~a.astype(bool)


def _where(cond: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    # Keep ``a`` where the condition holds, else ``b`` (index/column aligned).
    return a.where(cond.astype(bool), b)


def _rank(a: pd.DataFrame) -> pd.DataFrame:
    return a.rank(axis=1, pct=True)


def _zscore(a: pd.DataFrame) -> pd.DataFrame:
    mean = a.mean(axis=1)
    std = a.std(axis=1).replace(0.0, np.nan)
    return _finite(a.sub(mean, axis=0).div(std, axis=0))


def _scale(a: pd.DataFrame) -> pd.DataFrame:
    denom = a.abs().sum(axis=1).replace(0.0, np.nan)
    return _finite(a.div(denom, axis=0))


# --- registry construction -------------------------------------------------------
_SE, _SC, _WI, _SI = DType.SERIES, DType.SCALAR, DType.WINDOW, DType.SIGNAL
_BO = DType.BOOL

_OPERATOR_SPECS: list[tuple[str, tuple[DType, ...], DType, Callable[..., pd.DataFrame]]] = [
    # arithmetic
    ("add", (_SE, _SE), _SE, _add),
    ("sub", (_SE, _SE), _SE, _sub),
    ("mul", (_SE, _SE), _SE, _mul),
    ("div", (_SE, _SE), _SE, _div),
    # scalar-parameterized
    ("mul_scalar", (_SE, _SC), _SE, _mul_scalar),
    ("add_scalar", (_SE, _SC), _SE, _add_scalar),
    ("signed_power", (_SE, _SC), _SE, _signed_power),
    # unary math
    ("log", (_SE,), _SE, _log),
    ("abs", (_SE,), _SE, _abs),
    ("sign", (_SE,), _SE, _sign),
    ("neg", (_SE,), _SE, _neg),
    # unary time-series
    ("ts_mean", (_SE, _WI), _SE, _ts_mean),
    ("ts_ema", (_SE, _WI), _SE, _ts_ema),
    ("ts_std", (_SE, _WI), _SE, _ts_std),
    ("ts_std_pop", (_SE, _WI), _SE, _ts_std_pop),
    ("ts_rma", (_SE, _WI), _SE, _ts_rma),
    ("ts_recursive_smooth", (_SE, _WI, _SC), _SE, _ts_recursive_smooth),
    ("ts_sum", (_SE, _WI), _SE, _ts_sum),
    ("ts_min", (_SE, _WI), _SE, _ts_min),
    ("ts_max", (_SE, _WI), _SE, _ts_max),
    ("ts_rank", (_SE, _WI), _SE, _ts_rank),
    ("decay_linear", (_SE, _WI), _SE, _decay_linear),
    ("delta", (_SE, _WI), _SE, _delta),
    ("delay", (_SE, _WI), _SE, _delay),
    # binary time-series
    ("ts_corr", (_SE, _SE, _WI), _SE, _ts_corr),
    ("ts_cov", (_SE, _SE, _WI), _SE, _ts_cov),
    # cross-sectional (produce a SIGNAL)
    ("rank", (_SE,), _SI, _rank),
    ("zscore", (_SE,), _SI, _zscore),
    ("scale", (_SE,), _SI, _scale),
    # comparisons (produce a BOOL mask)
    ("gt", (_SE, _SE), _BO, _gt),
    ("lt", (_SE, _SE), _BO, _lt),
    ("ge", (_SE, _SE), _BO, _ge),
    ("le", (_SE, _SE), _BO, _le),
    # logical (combine BOOL masks)
    ("and_", (_BO, _BO), _BO, _and),
    ("or_", (_BO, _BO), _BO, _or),
    ("not_", (_BO,), _BO, _not),
    # select: keep the first series where the condition holds, else the second
    ("where", (_BO, _SE, _SE), _SE, _where),
]

#: Panel fields usable as leaf operands (all of type SERIES).
OPERAND_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "vwap", "returns")


def _sample_window(rng: random.Random) -> int:
    return rng.choice(DEFAULT_WINDOWS)


def _sample_scalar(rng: random.Random) -> float:
    return rng.choice(DEFAULT_SCALARS)


OPERATORS: dict[str, Primitive] = {
    name: Primitive(name, Kind.OPERATOR, out, args, fn=fn)
    for name, args, out, fn in _OPERATOR_SPECS
}
OPERANDS: dict[str, Primitive] = {
    name: Primitive(name, Kind.OPERAND, DType.SERIES, panel_field=name) for name in OPERAND_FIELDS
}
EPHEMERALS: dict[str, Primitive] = {
    "const": Primitive("const", Kind.EPHEMERAL, DType.SCALAR, sampler=_sample_scalar),
    "window": Primitive("window", Kind.EPHEMERAL, DType.WINDOW, sampler=_sample_window),
}

#: The complete primitive registry, keyed by name.
REGISTRY: dict[str, Primitive] = {**OPERATORS, **OPERANDS, **EPHEMERALS}


def get(name: str) -> Primitive:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown primitive {name!r}") from exc


def terminals_for(target: DType) -> list[Primitive]:
    """Leaf primitives whose output can fill a hole of type ``target``."""
    return [p for p in REGISTRY.values() if p.is_terminal and is_subtype(p.out_type, target)]


def operators_for(target: DType) -> list[Primitive]:
    """Operator primitives whose output can fill a hole of type ``target``."""
    return [p for p in OPERATORS.values() if is_subtype(p.out_type, target)]
