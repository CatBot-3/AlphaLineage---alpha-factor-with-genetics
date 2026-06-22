"""Default categories for built-in primitives.

Categories organize the function palette in the UI and gate which operator groups the GP
may draw from per run (see ``enabled_categories``). Built-ins get their category from the
code-level map below; user formulas carry their own ``category``; either can be overridden by
the user (stored in ``data_cache/meta/categories.json``).
"""

from __future__ import annotations

# Category keys (stable identifiers). The UI supplies human labels + ordering.
DATA = "data"
ARITHMETIC = "arithmetic"
SCALAR = "scalar"
UNARY_MATH = "unary_math"
TIME_SERIES = "time_series"
CROSS_SECTIONAL = "cross_sectional"
CONSTANT = "constant"
CONDITION = "condition"
CUSTOM = "custom"
UNCATEGORIZED = "uncategorized"

#: Default ordered category list seeded into a fresh ``categories.json``.
DEFAULT_CATEGORY_ORDER: list[str] = [
    DATA,
    ARITHMETIC,
    SCALAR,
    UNARY_MATH,
    TIME_SERIES,
    CROSS_SECTIONAL,
    CONDITION,
    CONSTANT,
    CUSTOM,
]

#: Categories the GP draws from by default. ``condition`` is intentionally excluded so the
#: classic numeric search space (and the two load-bearing tests) is unaffected until a user
#: opts conditions in for a run.
DEFAULT_ENABLED_CATEGORIES: list[str] = [
    DATA,
    ARITHMETIC,
    SCALAR,
    UNARY_MATH,
    TIME_SERIES,
    CROSS_SECTIONAL,
    CONSTANT,
    CUSTOM,
]

#: Built-in primitive name -> default category.
BUILTIN_CATEGORIES: dict[str, str] = {
    # data fields (operands)
    "open": DATA,
    "high": DATA,
    "low": DATA,
    "close": DATA,
    "volume": DATA,
    "vwap": DATA,
    "returns": DATA,
    # arithmetic
    "add": ARITHMETIC,
    "sub": ARITHMETIC,
    "mul": ARITHMETIC,
    "div": ARITHMETIC,
    # scalar-parameterized
    "mul_scalar": SCALAR,
    "add_scalar": SCALAR,
    "signed_power": SCALAR,
    # unary math
    "log": UNARY_MATH,
    "abs": UNARY_MATH,
    "sign": UNARY_MATH,
    "neg": UNARY_MATH,
    # time-series
    "ts_mean": TIME_SERIES,
    "ts_std": TIME_SERIES,
    "ts_sum": TIME_SERIES,
    "ts_min": TIME_SERIES,
    "ts_max": TIME_SERIES,
    "ts_rank": TIME_SERIES,
    "decay_linear": TIME_SERIES,
    "delta": TIME_SERIES,
    "delay": TIME_SERIES,
    "ts_corr": TIME_SERIES,
    "ts_cov": TIME_SERIES,
    # cross-sectional
    "rank": CROSS_SECTIONAL,
    "zscore": CROSS_SECTIONAL,
    "scale": CROSS_SECTIONAL,
    # condition (comparisons, logical, select)
    "gt": CONDITION,
    "lt": CONDITION,
    "ge": CONDITION,
    "le": CONDITION,
    "and_": CONDITION,
    "or_": CONDITION,
    "not_": CONDITION,
    "where": CONDITION,
    # ephemeral constants
    "const": CONSTANT,
    "window": CONSTANT,
}


def builtin_category(name: str) -> str:
    return BUILTIN_CATEGORIES.get(name, UNCATEGORIZED)
