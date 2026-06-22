"""P1-T2 - the primitive registry: operators, operands, and ephemeral terminals.

Every primitive declares its arity, input types, and output type, so generation,
crossover, and mutation can only ever produce semantically valid trees. All operator
implementations are vectorized (numpy/pandas) and NaN-tolerant: they never raise on a
type-valid tree, they return NaN where a value is undefined (this is what lets the
generator's 10k-tree validity sweep pass).
"""

from __future__ import annotations

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


def _roll(a: pd.DataFrame, w: int) -> Any:
    w = max(1, int(w))
    return a.rolling(window=w, min_periods=w)


def _add(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a + b


def _sub(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a - b


def _mul(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a * b


def _div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return _finite(a / b.replace(0.0, np.nan))


def _mul_scalar(a: pd.DataFrame, s: float) -> pd.DataFrame:
    return a * s


def _add_scalar(a: pd.DataFrame, s: float) -> pd.DataFrame:
    return a + s


def _signed_power(a: pd.DataFrame, s: float) -> pd.DataFrame:
    return _finite(np.sign(a) * np.power(a.abs(), s))


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


def _ts_std(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return _roll(a, w).std()


def _ts_sum(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return _roll(a, w).sum()


def _ts_min(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return _roll(a, w).min()


def _ts_max(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return _roll(a, w).max()


def _ts_rank(a: pd.DataFrame, w: int) -> pd.DataFrame:
    # Percentile rank of the current value within its trailing window.
    def f(x: np.ndarray) -> float:
        return float(np.mean(x <= x[-1]))

    return _roll(a, w).apply(f, raw=True)


def _decay_linear(a: pd.DataFrame, w: int) -> pd.DataFrame:
    ww = max(1, int(w))
    weights = np.arange(1, ww + 1, dtype=float)
    weights /= weights.sum()

    def f(x: np.ndarray) -> float:
        return float(np.dot(x, weights))

    return _roll(a, w).apply(f, raw=True)


def _delta(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return a - a.shift(int(w))


def _delay(a: pd.DataFrame, w: int) -> pd.DataFrame:
    return a.shift(int(w))


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
    ("ts_std", (_SE, _WI), _SE, _ts_std),
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
