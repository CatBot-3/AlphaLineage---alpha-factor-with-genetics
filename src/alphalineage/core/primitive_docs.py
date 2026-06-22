"""Human-facing names, argument labels, and descriptions for built-in primitives."""

from __future__ import annotations

from typing import Any


def _entry(display_name: str, description: str, *inputs: tuple[str, str]) -> dict[str, Any]:
    return {
        "display_name": display_name,
        "description": description,
        "inputs": [{"name": name, "description": description} for name, description in inputs],
    }


PRIMITIVE_DOCS: dict[str, dict[str, Any]] = {
    "open": _entry("Open", "Opening price for each symbol and trading date."),
    "high": _entry("High", "Highest traded price for each symbol and trading date."),
    "low": _entry("Low", "Lowest traded price for each symbol and trading date."),
    "close": _entry("Close", "Closing price for each symbol and trading date."),
    "volume": _entry("Volume", "Reported trading volume for each symbol and trading date."),
    "vwap": _entry("VWAP", "Volume-weighted average price for each symbol and trading date."),
    "returns": _entry("Returns", "Single-period return derived from closing prices."),
    "add": _entry(
        "Add",
        "Adds two series element by element.",
        ("left", "First series."),
        ("right", "Second series."),
    ),
    "sub": _entry(
        "Subtract",
        "Subtracts the second series from the first.",
        ("left", "Series to subtract from."),
        ("right", "Series to subtract."),
    ),
    "mul": _entry(
        "Multiply",
        "Multiplies two series element by element.",
        ("left", "First series."),
        ("right", "Second series."),
    ),
    "div": _entry(
        "Divide",
        "Divides the first series by the second with finite-result protection.",
        ("numerator", "Series being divided."),
        ("denominator", "Series used as the divisor."),
    ),
    "mul_scalar": _entry(
        "Multiply by scalar",
        "Multiplies a series by a numeric scalar.",
        ("series", "Input series."),
        ("scalar", "Numeric multiplier."),
    ),
    "add_scalar": _entry(
        "Add scalar",
        "Adds a numeric scalar to a series.",
        ("series", "Input series."),
        ("scalar", "Numeric amount to add."),
    ),
    "signed_power": _entry(
        "Signed power",
        "Raises absolute values to a power while preserving sign.",
        ("series", "Input series."),
        ("power", "Power applied to each value."),
    ),
    "log": _entry(
        "Log",
        "Computes the finite natural logarithm of absolute values.",
        ("series", "Input series."),
    ),
    "abs": _entry(
        "Absolute value",
        "Returns the absolute value of each observation.",
        ("series", "Input series."),
    ),
    "sign": _entry(
        "Sign",
        "Returns -1, 0, or 1 according to each observation's sign.",
        ("series", "Input series."),
    ),
    "neg": _entry("Negate", "Reverses the sign of each observation.", ("series", "Input series.")),
    "ts_mean": _entry(
        "Moving average",
        "Trailing arithmetic mean over a fixed lookback window.",
        ("series", "Series to average."),
        ("lookback", "Number of periods in the trailing window."),
    ),
    "ts_std": _entry(
        "Rolling standard deviation",
        "Trailing standard deviation over a fixed lookback window.",
        ("series", "Input series."),
        ("lookback", "Number of periods in the trailing window."),
    ),
    "ts_sum": _entry(
        "Rolling sum",
        "Trailing sum over a fixed lookback window.",
        ("series", "Input series."),
        ("lookback", "Number of periods in the trailing window."),
    ),
    "ts_min": _entry(
        "Rolling minimum",
        "Lowest value in the trailing lookback window.",
        ("series", "Input series."),
        ("lookback", "Number of periods in the trailing window."),
    ),
    "ts_max": _entry(
        "Rolling maximum",
        "Highest value in the trailing lookback window.",
        ("series", "Input series."),
        ("lookback", "Number of periods in the trailing window."),
    ),
    "ts_rank": _entry(
        "Rolling rank",
        "Percentile rank of the latest observation within its trailing window.",
        ("series", "Input series."),
        ("lookback", "Number of periods in the trailing window."),
    ),
    "decay_linear": _entry(
        "Linear decay",
        "Linearly weighted trailing average with greater weight on recent observations.",
        ("series", "Input series."),
        ("lookback", "Number of periods in the trailing window."),
    ),
    "delta": _entry(
        "Change",
        "Difference between the current value and its lagged value.",
        ("series", "Input series."),
        ("periods", "Number of periods to look back."),
    ),
    "delay": _entry(
        "Delay",
        "Shifts a series backward by a fixed number of periods.",
        ("series", "Input series."),
        ("periods", "Number of periods to delay."),
    ),
    "ts_corr": _entry(
        "Rolling correlation",
        "Trailing correlation between two series.",
        ("left", "First series."),
        ("right", "Second series."),
        ("lookback", "Number of periods in the trailing window."),
    ),
    "ts_cov": _entry(
        "Rolling covariance",
        "Trailing covariance between two series.",
        ("left", "First series."),
        ("right", "Second series."),
        ("lookback", "Number of periods in the trailing window."),
    ),
    "rank": _entry(
        "Cross-sectional rank",
        "Ranks symbols against one another on each date and scales ranks to 0-1.",
        ("series", "Series to rank across symbols."),
    ),
    "zscore": _entry(
        "Cross-sectional z-score",
        "Centers and scales values across symbols on each date.",
        ("series", "Series to normalize across symbols."),
    ),
    "scale": _entry(
        "Cross-sectional scale",
        "Scales absolute cross-sectional exposure to one on each date.",
        ("series", "Series to scale across symbols."),
    ),
    "gt": _entry(
        "Greater than",
        "Tests whether the first series is greater than the second.",
        ("left", "First series."),
        ("right", "Second series."),
    ),
    "lt": _entry(
        "Less than",
        "Tests whether the first series is less than the second.",
        ("left", "First series."),
        ("right", "Second series."),
    ),
    "ge": _entry(
        "Greater than or equal",
        "Tests whether the first series is at least the second.",
        ("left", "First series."),
        ("right", "Second series."),
    ),
    "le": _entry(
        "Less than or equal",
        "Tests whether the first series is at most the second.",
        ("left", "First series."),
        ("right", "Second series."),
    ),
    "and_": _entry(
        "And",
        "Combines two boolean series with logical AND.",
        ("left", "First condition."),
        ("right", "Second condition."),
    ),
    "or_": _entry(
        "Or",
        "Combines two boolean series with logical OR.",
        ("left", "First condition."),
        ("right", "Second condition."),
    ),
    "not_": _entry("Not", "Inverts a boolean series.", ("condition", "Condition to invert.")),
    "where": _entry(
        "Choose where",
        "Selects one series where a condition is true and another otherwise.",
        ("condition", "Boolean selection mask."),
        ("when_true", "Values used where true."),
        ("when_false", "Values used where false."),
    ),
    "const": _entry("Scalar", "A numeric constant that can be tuned during training."),
    "window": _entry("Window", "A whole-number lookback that can be tuned during training."),
}


def primitive_doc(name: str, arity: int) -> dict[str, Any]:
    doc = PRIMITIVE_DOCS.get(name, {})
    inputs = list(doc.get("inputs", []))
    while len(inputs) < arity:
        inputs.append({"name": f"input_{len(inputs) + 1}", "description": "Function input."})
    return {
        "display_name": str(doc.get("display_name") or name.replace("_", " ").title()),
        "description": str(doc.get("description") or "Typed AlphaLineage calculation primitive."),
        "inputs": inputs[:arity],
    }
