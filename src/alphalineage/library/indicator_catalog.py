"""Immutable, packaged starter formulas for common technical indicators.

Definitions deliberately use the same typed formula DSL as user formulas.  Dependencies are
listed in topological order and refer to logical ``ta_*`` names; the formula-store seeder pins
those references to the exact managed runtime revision when it materializes the catalog.
"""

from __future__ import annotations

from typing import Any

CATALOG_ORIGIN = "catalog_formula"
CATALOG_CATEGORY = "technical_indicators"
CATALOG_REVISION = 1


def _arg(index: int) -> dict[str, Any]:
    return {"name": "$arg", "value": index}


def _value(name: str, value: float | int) -> dict[str, Any]:
    return {"name": name, "value": value}


def _call(name: str, *children: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "children": list(children)}


def _input(
    name: str,
    type_: str,
    description: str,
    default: float | int | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"name": name, "type": type_, "description": description}
    if default is not None:
        item["default"] = default
    return item


def _formula(
    name: str,
    display_name: str,
    description: str,
    family: str,
    aliases: list[str],
    inputs: list[dict[str, Any]],
    body: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "display_name": display_name,
        "description": description,
        "arg_types": [item["type"] for item in inputs],
        "inputs": inputs,
        "out_type": "series",
        "body": body,
        "category": CATALOG_CATEGORY,
        "origin": CATALOG_ORIGIN,
        "editable": False,
        "family": family,
        "aliases": aliases,
        "catalog_revision": CATALOG_REVISION,
    }


_SERIES = _input("series", "series", "Input price or value series.")
_LOOKBACK_20 = _input("lookback", "window", "Number of periods.", 20)


def _zero_like(series: dict[str, Any]) -> dict[str, Any]:
    return _call("mul_scalar", series, _value("const", 0.0))


def _equals_zero(series: dict[str, Any]) -> dict[str, Any]:
    zero = _zero_like(series)
    return _call("and_", _call("ge", series, zero), _call("le", series, zero))


def _constant_like(series: dict[str, Any], value: float) -> dict[str, Any]:
    return _call("add_scalar", _zero_like(series), _value("const", value))


def _rsi_wilder_body() -> dict[str, Any]:
    change = _call("delta", _arg(0), _value("window", 1))
    magnitude = _call("abs", change)
    gain = _call(
        "ts_rma",
        _call("mul_scalar", _call("add", change, magnitude), _value("const", 0.5)),
        _arg(1),
    )
    loss = _call(
        "ts_rma",
        _call("mul_scalar", _call("sub", magnitude, change), _value("const", 0.5)),
        _arg(1),
    )
    total = _call("add", gain, loss)
    ordinary = _call("mul_scalar", _call("div", gain, total), _value("const", 100.0))
    return _call(
        "where",
        _equals_zero(total),
        _constant_like(total, 50.0),
        _call(
            "where",
            _equals_zero(loss),
            _constant_like(total, 100.0),
            ordinary,
        ),
    )


def _kdj_rsv_body() -> dict[str, Any]:
    lowest = _call("ts_min", _arg(1), _arg(3))
    spread = _call("sub", _call("ts_max", _arg(0), _arg(3)), lowest)
    ordinary = _call(
        "mul_scalar",
        _call("div", _call("sub", _arg(2), lowest), spread),
        _value("const", 100.0),
    )
    # A mature zero-range window has no directional information. Emit neutral RSV=50; warm-up
    # NaNs remain NaN because the equality mask is false and the ordinary division is undefined.
    return _call(
        "where",
        _equals_zero(spread),
        _constant_like(spread, 50.0),
        ordinary,
    )


def _true_range_body() -> dict[str, Any]:
    previous_close = _call("delay", _arg(2), _value("window", 1))
    intraday = _call("sub", _arg(0), _arg(1))
    high_gap = _call("abs", _call("sub", _arg(0), previous_close))
    low_gap = _call("abs", _call("sub", _arg(1), previous_close))
    high_or_intraday = _call(
        "where",
        _call("gt", high_gap, intraday),
        high_gap,
        intraday,
    )
    return _call(
        "where",
        _call("gt", low_gap, high_or_intraday),
        low_gap,
        high_or_intraday,
    )


INDICATOR_CATALOG: tuple[dict[str, Any], ...] = (
    _formula(
        "ta_sma",
        "SMA / Simple moving average",
        "Arithmetic mean over a trailing window.",
        "moving_averages",
        ["sma", "ma", "simple moving average", "moving average"],
        [_SERIES, _LOOKBACK_20],
        _call("ts_mean", _arg(0), _arg(1)),
    ),
    _formula(
        "ta_ema",
        "EMA / Exponential moving average",
        "Exponentially weighted moving average with span-based smoothing.",
        "moving_averages",
        ["ema", "exponential moving average"],
        [_SERIES, _LOOKBACK_20],
        _call("ts_ema", _arg(0), _arg(1)),
    ),
    _formula(
        "ta_rma",
        "RMA / Wilder moving average",
        "Wilder recursive moving average seeded from the first complete window.",
        "moving_averages",
        ["rma", "wilder moving average", "smma"],
        [_SERIES, _input("lookback", "window", "Wilder smoothing period.", 14)],
        _call("ts_rma", _arg(0), _arg(1)),
    ),
    _formula(
        "ta_dif",
        "DIF / MACD line",
        "Difference between the fast and slow exponential moving averages.",
        "macd",
        ["dif", "macd line"],
        [
            _SERIES,
            _input("fast", "window", "Fast EMA period.", 12),
            _input("slow", "window", "Slow EMA period.", 26),
        ],
        _call("sub", _call("ta_ema", _arg(0), _arg(1)), _call("ta_ema", _arg(0), _arg(2))),
    ),
    _formula(
        "ta_dea",
        "DEA / MACD signal line",
        "EMA of DIF. Looking for MEA? Most charting packages call this DEA or the MACD "
        "signal line.",
        "macd",
        ["dea", "mea", "macd signal", "macd signal line"],
        [
            _SERIES,
            _input("fast", "window", "Fast EMA period.", 12),
            _input("slow", "window", "Slow EMA period.", 26),
            _input("signal", "window", "Signal EMA period.", 9),
        ],
        _call("ta_ema", _call("ta_dif", _arg(0), _arg(1), _arg(2)), _arg(3)),
    ),
    _formula(
        "ta_macd_histogram",
        "MACD histogram",
        "Unscaled Western convention: DIF minus DEA.",
        "macd",
        ["macd", "macd histogram", "macd hist"],
        [
            _SERIES,
            _input("fast", "window", "Fast EMA period.", 12),
            _input("slow", "window", "Slow EMA period.", 26),
            _input("signal", "window", "Signal EMA period.", 9),
        ],
        _call(
            "sub",
            _call("ta_dif", _arg(0), _arg(1), _arg(2)),
            _call("ta_dea", _arg(0), _arg(1), _arg(2), _arg(3)),
        ),
    ),
    _formula(
        "ta_macd_histogram_2x",
        "MACD histogram (2x)",
        "Chinese charting convention: two times DIF minus DEA.",
        "macd",
        ["macd 2x", "macd bar", "chinese macd"],
        [
            _SERIES,
            _input("fast", "window", "Fast EMA period.", 12),
            _input("slow", "window", "Slow EMA period.", 26),
            _input("signal", "window", "Signal EMA period.", 9),
        ],
        _call(
            "mul_scalar",
            _call("ta_macd_histogram", _arg(0), _arg(1), _arg(2), _arg(3)),
            _value("const", 2.0),
        ),
    ),
    _formula(
        "ta_rsi_wilder",
        "RSI / Wilder relative strength index",
        "Wilder RSI with neutral 50 for flat windows and 100 when smoothed loss is zero.",
        "momentum",
        ["rsi", "ta_rsi", "relative strength index", "wilder rsi"],
        [_SERIES, _input("lookback", "window", "Wilder RSI period.", 14)],
        _rsi_wilder_body(),
    ),
    _formula(
        "ta_kdj_rsv",
        "KDJ RSV",
        "Raw stochastic value from close within the trailing high-low range.",
        "kdj",
        ["rsv", "kdj rsv", "stochastic raw value"],
        [
            _input("high", "series", "High-price series."),
            _input("low", "series", "Low-price series."),
            _input("close", "series", "Close-price series."),
            _input("lookback", "window", "Stochastic range period.", 9),
        ],
        _kdj_rsv_body(),
    ),
    _formula(
        "ta_kdj_k",
        "KDJ K",
        "KDJ K line: recursively smoothed RSV with conventional initial value 50.",
        "kdj",
        ["kdj k", "stochastic k"],
        [
            _input("high", "series", "High-price series."),
            _input("low", "series", "Low-price series."),
            _input("close", "series", "Close-price series."),
            _input("lookback", "window", "Stochastic range period.", 9),
            _input("k_smoothing", "window", "K recursive smoothing period.", 3),
        ],
        _call(
            "ts_recursive_smooth",
            _call("ta_kdj_rsv", _arg(0), _arg(1), _arg(2), _arg(3)),
            _arg(4),
            _value("const", 50.0),
        ),
    ),
    _formula(
        "ta_kdj_d",
        "KDJ D",
        "KDJ D line: recursively smoothed K with conventional initial value 50.",
        "kdj",
        ["kdj d", "stochastic d"],
        [
            _input("high", "series", "High-price series."),
            _input("low", "series", "Low-price series."),
            _input("close", "series", "Close-price series."),
            _input("lookback", "window", "Stochastic range period.", 9),
            _input("k_smoothing", "window", "K recursive smoothing period.", 3),
            _input("d_smoothing", "window", "D recursive smoothing period.", 3),
        ],
        _call(
            "ts_recursive_smooth",
            _call("ta_kdj_k", _arg(0), _arg(1), _arg(2), _arg(3), _arg(4)),
            _arg(5),
            _value("const", 50.0),
        ),
    ),
    _formula(
        "ta_kdj_j",
        "KDJ J",
        "KDJ J line: three times K minus two times D.",
        "kdj",
        ["kdj j", "stochastic j"],
        [
            _input("high", "series", "High-price series."),
            _input("low", "series", "Low-price series."),
            _input("close", "series", "Close-price series."),
            _input("lookback", "window", "Stochastic range period.", 9),
            _input("k_smoothing", "window", "K recursive smoothing period.", 3),
            _input("d_smoothing", "window", "D recursive smoothing period.", 3),
        ],
        _call(
            "sub",
            _call(
                "mul_scalar",
                _call("ta_kdj_k", _arg(0), _arg(1), _arg(2), _arg(3), _arg(4)),
                _value("const", 3.0),
            ),
            _call(
                "mul_scalar",
                _call("ta_kdj_d", _arg(0), _arg(1), _arg(2), _arg(3), _arg(4), _arg(5)),
                _value("const", 2.0),
            ),
        ),
    ),
    _formula(
        "ta_boll_middle",
        "Bollinger middle band",
        "Simple moving-average center line for Bollinger bands.",
        "volatility",
        ["boll middle", "bollinger middle", "boll mid"],
        [_SERIES, _LOOKBACK_20],
        _call("ta_sma", _arg(0), _arg(1)),
    ),
    _formula(
        "ta_boll_upper",
        "Bollinger upper band",
        "Middle band plus a multiple of population rolling deviation.",
        "volatility",
        ["boll upper", "bollinger upper", "upper band"],
        [_SERIES, _LOOKBACK_20, _input("deviations", "scalar", "Deviation multiplier.", 2.0)],
        _call(
            "add",
            _call("ta_boll_middle", _arg(0), _arg(1)),
            _call("mul_scalar", _call("ts_std_pop", _arg(0), _arg(1)), _arg(2)),
        ),
    ),
    _formula(
        "ta_boll_lower",
        "Bollinger lower band",
        "Middle band minus a multiple of population rolling deviation.",
        "volatility",
        ["boll lower", "bollinger lower", "lower band"],
        [_SERIES, _LOOKBACK_20, _input("deviations", "scalar", "Deviation multiplier.", 2.0)],
        _call(
            "sub",
            _call("ta_boll_middle", _arg(0), _arg(1)),
            _call("mul_scalar", _call("ts_std_pop", _arg(0), _arg(1)), _arg(2)),
        ),
    ),
    _formula(
        "ta_boll_percent_b",
        "Bollinger %B",
        "Position within the Bollinger envelope: (value - lower) / (upper - lower).",
        "volatility",
        ["boll %b", "bollinger %b", "percent b"],
        [_SERIES, _LOOKBACK_20, _input("deviations", "scalar", "Deviation multiplier.", 2.0)],
        _call(
            "div",
            _call("sub", _arg(0), _call("ta_boll_lower", _arg(0), _arg(1), _arg(2))),
            _call(
                "sub",
                _call("ta_boll_upper", _arg(0), _arg(1), _arg(2)),
                _call("ta_boll_lower", _arg(0), _arg(1), _arg(2)),
            ),
        ),
    ),
    _formula(
        "ta_boll_bandwidth",
        "Bollinger bandwidth",
        "Envelope width as a percentage of the middle band.",
        "volatility",
        ["boll bandwidth", "bollinger bandwidth", "bandwidth"],
        [_SERIES, _LOOKBACK_20, _input("deviations", "scalar", "Deviation multiplier.", 2.0)],
        _call(
            "mul_scalar",
            _call(
                "div",
                _call(
                    "sub",
                    _call("ta_boll_upper", _arg(0), _arg(1), _arg(2)),
                    _call("ta_boll_lower", _arg(0), _arg(1), _arg(2)),
                ),
                _call("ta_boll_middle", _arg(0), _arg(1)),
            ),
            _value("const", 100.0),
        ),
    ),
    _formula(
        "ta_true_range",
        "True range",
        "Maximum of high-low and the two previous-close gap ranges.",
        "volatility",
        ["tr", "true range"],
        [
            _input("high", "series", "High-price series."),
            _input("low", "series", "Low-price series."),
            _input("close", "series", "Close-price series."),
        ],
        _true_range_body(),
    ),
    _formula(
        "ta_atr",
        "ATR / Average true range",
        "Wilder moving average of true range.",
        "volatility",
        ["atr", "average true range"],
        [
            _input("high", "series", "High-price series."),
            _input("low", "series", "Low-price series."),
            _input("close", "series", "Close-price series."),
            _input("lookback", "window", "Wilder ATR period.", 14),
        ],
        _call("ts_rma", _call("ta_true_range", _arg(0), _arg(1), _arg(2)), _arg(3)),
    ),
    _formula(
        "ta_roc",
        "ROC / Rate of change",
        "Percentage change from the value a fixed number of periods ago.",
        "momentum",
        ["roc", "rate of change", "momentum percent"],
        [_SERIES, _input("lookback", "window", "Comparison lag in periods.", 10)],
        _call(
            "mul_scalar",
            _call(
                "add_scalar",
                _call("div", _arg(0), _call("delay", _arg(0), _arg(1))),
                _value("const", -1.0),
            ),
            _value("const", 100.0),
        ),
    ),
)

CATALOG_NAMES = frozenset(item["name"] for item in INDICATOR_CATALOG)

__all__ = [
    "CATALOG_CATEGORY",
    "CATALOG_NAMES",
    "CATALOG_ORIGIN",
    "CATALOG_REVISION",
    "INDICATOR_CATALOG",
]
