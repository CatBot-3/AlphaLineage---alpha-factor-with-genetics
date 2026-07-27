"""Versioned, immutable starter formulas for common technical indicators.

Catalog-v2 formulas preserve the market-data meaning of each indicator: price-only formulas bind
``close`` directly, range formulas bind ``high``/``low``/``close``, and volume formulas bind
``volume``.  Only explicitly approved numeric parameters remain as call-site arguments.  Their
bounded local-search policies are data and are pinned with the formula revision.
"""

from __future__ import annotations

from typing import Any

CATALOG_ORIGIN = "catalog_formula"
CATALOG_CATEGORY = "technical_indicators"
CATALOG_REVISION = 2
# Managed names published before catalog v2. They remain immutable and loadable through their old
# runtime revision, while a new retired latest revision keeps them out of fresh GP searches.
LEGACY_CATALOG_REPLACEMENTS: dict[str, str] = {
    "ta_rsi": "ta_rsi_wilder",
}


def _arg(index: int) -> dict[str, Any]:
    return {"name": "$arg", "value": index}


def _value(name: str, value: float | int) -> dict[str, Any]:
    return {"name": name, "value": value}


def _window(value: int) -> dict[str, Any]:
    return _value("window", value)


def _scalar(value: float) -> dict[str, Any]:
    return _value("const", value)


def _field(name: str) -> dict[str, Any]:
    return {"name": name}


def _call(name: str, *children: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "children": list(children)}


def _parameter(
    name: str,
    type_: str,
    description: str,
    default: float | int,
    minimum: float | int,
    maximum: float | int,
    *,
    step: float | int = 1,
    radius: int = 1,
) -> dict[str, Any]:
    return {
        "name": name,
        "type": type_,
        "description": description,
        "default": default,
        "role": "parameter",
        "tuning": {
            "enabled": True,
            "min": minimum,
            "max": maximum,
            "step": step,
            "radius": radius,
        },
    }


def _lookback(
    default: int,
    minimum: int,
    maximum: int,
    description: str = "Indicator lookback in periods.",
) -> dict[str, Any]:
    return _parameter("lookback", "window", description, default, minimum, maximum)


def _formula(
    name: str,
    display_name: str,
    description: str,
    family: str,
    family_order: int,
    aliases: list[str],
    inputs: list[dict[str, Any]],
    body: dict[str, Any],
    *,
    constraints: list[dict[str, str]] | None = None,
    status: str = "active",
    replacement: str | None = None,
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
        "family_order": family_order,
        "aliases": aliases,
        "catalog_revision": CATALOG_REVISION,
        "constraints": constraints or [],
        "status": status,
        "replacement": replacement,
    }


def _zero_like(series: dict[str, Any]) -> dict[str, Any]:
    return _call("mul_scalar", series, _scalar(0.0))


def _equals_zero(series: dict[str, Any]) -> dict[str, Any]:
    zero = _zero_like(series)
    return _call("and_", _call("ge", series, zero), _call("le", series, zero))


def _constant_like(series: dict[str, Any], value: float) -> dict[str, Any]:
    return _call("add_scalar", _zero_like(series), _scalar(value))


def _rsi_wilder_body() -> dict[str, Any]:
    change = _call("delta", _field("close"), _window(1))
    magnitude = _call("abs", change)
    gain = _call(
        "ts_rma",
        _call("mul_scalar", _call("add", change, magnitude), _scalar(0.5)),
        _arg(0),
    )
    loss = _call(
        "ts_rma",
        _call("mul_scalar", _call("sub", magnitude, change), _scalar(0.5)),
        _arg(0),
    )
    total = _call("add", gain, loss)
    ordinary = _call("mul_scalar", _call("div", gain, total), _scalar(100.0))
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
    lowest = _call("ts_min", _field("low"), _arg(0))
    spread = _call("sub", _call("ts_max", _field("high"), _arg(0)), lowest)
    ordinary = _call(
        "mul_scalar",
        _call("div", _call("sub", _field("close"), lowest), spread),
        _scalar(100.0),
    )
    return _call(
        "where",
        _equals_zero(spread),
        _constant_like(spread, 50.0),
        ordinary,
    )


def _true_range_body() -> dict[str, Any]:
    previous_close = _call("delay", _field("close"), _window(1))
    intraday = _call("sub", _field("high"), _field("low"))
    high_gap = _call("abs", _call("sub", _field("high"), previous_close))
    low_gap = _call("abs", _call("sub", _field("low"), previous_close))
    high_or_intraday = _call("where", _call("gt", high_gap, intraday), high_gap, intraday)
    return _call("where", _call("gt", low_gap, high_or_intraday), low_gap, high_or_intraday)


def _directional_move(positive: bool) -> dict[str, Any]:
    up = _call("delta", _field("high"), _window(1))
    down = _call("neg", _call("delta", _field("low"), _window(1)))
    candidate, other = (up, down) if positive else (down, up)
    zero = _zero_like(candidate)
    condition = _call("and_", _call("gt", candidate, other), _call("gt", candidate, zero))
    return _call("where", condition, candidate, zero)


def _directional_index(positive: bool) -> dict[str, Any]:
    smoothed = _call("ts_rma", _directional_move(positive), _arg(0))
    return _call(
        "mul_scalar",
        _call("div", smoothed, _call("ta_atr", _arg(0))),
        _scalar(100.0),
    )


def _typical_price() -> dict[str, Any]:
    return _call(
        "mul_scalar",
        _call("add", _call("add", _field("high"), _field("low")), _field("close")),
        _scalar(1.0 / 3.0),
    )


def _mfi_body() -> dict[str, Any]:
    typical = _typical_price()
    flow = _call("mul", typical, _field("volume"))
    change = _call("delta", typical, _window(1))
    zero = _zero_like(flow)
    positive = _call("where", _call("gt", change, _zero_like(change)), flow, zero)
    negative = _call("where", _call("lt", change, _zero_like(change)), flow, zero)
    positive_sum = _call("ts_sum", positive, _arg(0))
    negative_sum = _call("ts_sum", negative, _arg(0))
    ordinary = _call(
        "add_scalar",
        _call(
            "mul_scalar",
            _call(
                "div",
                _constant_like(negative_sum, 100.0),
                _call(
                    "add_scalar",
                    _call("div", positive_sum, negative_sum),
                    _scalar(1.0),
                ),
            ),
            _scalar(-1.0),
        ),
        _scalar(100.0),
    )
    return _call(
        "where",
        _equals_zero(negative_sum),
        _call(
            "where",
            _equals_zero(positive_sum),
            _constant_like(positive_sum, 50.0),
            _constant_like(positive_sum, 100.0),
        ),
        ordinary,
    )


def _cmf_body() -> dict[str, Any]:
    spread = _call("sub", _field("high"), _field("low"))
    numerator = _call(
        "sub",
        _call("sub", _field("close"), _field("low")),
        _call("sub", _field("high"), _field("close")),
    )
    multiplier = _call(
        "where",
        _equals_zero(spread),
        _zero_like(spread),
        _call("div", numerator, spread),
    )
    return _call(
        "div",
        _call("ts_sum", _call("mul", multiplier, _field("volume")), _arg(0)),
        _call("ts_sum", _field("volume"), _arg(0)),
    )


_FAST = _parameter("fast", "window", "Fast EMA period.", 12, 8, 20)
_SLOW = _parameter("slow", "window", "Slow EMA period.", 26, 20, 40)
_FAST_LT_SLOW = [{"left": "fast", "operator": "lt", "right": "slow"}]


INDICATOR_CATALOG: tuple[dict[str, Any], ...] = (
    # Moving averages ------------------------------------------------------------
    _formula(
        "ta_sma",
        "SMA / Simple moving average",
        "Simple moving average of Close.",
        "moving_averages",
        10,
        ["sma", "ma", "simple moving average", "moving average"],
        [_lookback(20, 5, 100)],
        _call("ts_mean", _field("close"), _arg(0)),
    ),
    _formula(
        "ta_ema",
        "EMA / Exponential moving average",
        "Span-based exponential moving average of Close.",
        "moving_averages",
        20,
        ["ema", "exponential moving average"],
        [_lookback(20, 5, 100)],
        _call("ts_ema", _field("close"), _arg(0)),
    ),
    _formula(
        "ta_rma",
        "RMA / Wilder moving average",
        "Wilder moving average of Close.",
        "moving_averages",
        30,
        ["rma", "wilder moving average", "smma"],
        [_lookback(14, 5, 60, "Wilder smoothing period.")],
        _call("ts_rma", _field("close"), _arg(0)),
    ),
    _formula(
        "ta_wma",
        "WMA / Weighted moving average",
        "Linearly weighted moving average of Close.",
        "moving_averages",
        40,
        ["wma", "weighted moving average"],
        [_lookback(20, 5, 100)],
        _call("decay_linear", _field("close"), _arg(0)),
    ),
    _formula(
        "ta_dema",
        "DEMA / Double exponential moving average",
        "Double EMA of Close: 2×EMA − EMA(EMA).",
        "moving_averages",
        50,
        ["dema", "double ema"],
        [_lookback(20, 5, 100)],
        _call(
            "sub",
            _call("mul_scalar", _call("ta_ema", _arg(0)), _scalar(2.0)),
            _call("ts_ema", _call("ta_ema", _arg(0)), _arg(0)),
        ),
    ),
    _formula(
        "ta_tema",
        "TEMA / Triple exponential moving average",
        "Triple EMA of Close: 3×EMA − 3×EMA(EMA) + EMA(EMA(EMA)).",
        "moving_averages",
        60,
        ["tema", "triple ema"],
        [_lookback(20, 5, 100)],
        _call(
            "add",
            _call(
                "sub",
                _call("mul_scalar", _call("ta_ema", _arg(0)), _scalar(3.0)),
                _call(
                    "mul_scalar",
                    _call("ts_ema", _call("ta_ema", _arg(0)), _arg(0)),
                    _scalar(3.0),
                ),
            ),
            _call(
                "ts_ema",
                _call("ts_ema", _call("ta_ema", _arg(0)), _arg(0)),
                _arg(0),
            ),
        ),
    ),
    # MACD ----------------------------------------------------------------------
    _formula(
        "ta_dif",
        "DIF / MACD line",
        "Fast Close EMA minus slow Close EMA.",
        "macd",
        10,
        ["dif", "macd line"],
        [_FAST, _SLOW],
        _call(
            "sub",
            _call("ts_ema", _field("close"), _arg(0)),
            _call("ts_ema", _field("close"), _arg(1)),
        ),
        constraints=_FAST_LT_SLOW,
    ),
    _formula(
        "ta_dea",
        "DEA / MACD signal line",
        "Nine-period EMA of DIF. Looking for MEA? DEA is the standard name.",
        "macd",
        20,
        ["dea", "mea", "macd signal", "macd signal line"],
        [_FAST, _SLOW],
        _call("ts_ema", _call("ta_dif", _arg(0), _arg(1)), _window(9)),
        constraints=_FAST_LT_SLOW,
    ),
    _formula(
        "ta_macd_histogram",
        "MACD",
        "Canonical unscaled MACD histogram: DIF minus the fixed nine-period DEA.",
        "macd",
        30,
        ["macd", "macd histogram", "macd hist"],
        [_FAST, _SLOW],
        _call(
            "sub",
            _call("ta_dif", _arg(0), _arg(1)),
            _call("ta_dea", _arg(0), _arg(1)),
        ),
        constraints=_FAST_LT_SLOW,
    ),
    _formula(
        "ta_macd_histogram_2x",
        "MACD histogram (2×, retired)",
        "Legacy display-scaled MACD retained only for pinned formulas. Use MACD instead.",
        "macd",
        90,
        ["macd 2x", "legacy macd bar"],
        [_FAST, _SLOW],
        _call(
            "mul_scalar",
            _call("ta_macd_histogram", _arg(0), _arg(1)),
            _scalar(2.0),
        ),
        constraints=_FAST_LT_SLOW,
        status="retired",
        replacement="ta_macd_histogram",
    ),
    _formula(
        "ta_ppo",
        "PPO / Percentage price oscillator",
        "DIF expressed as a percentage of the slow Close EMA.",
        "macd",
        40,
        ["ppo", "percentage price oscillator"],
        [_FAST, _SLOW],
        _call(
            "mul_scalar",
            _call(
                "div",
                _call("ta_dif", _arg(0), _arg(1)),
                _call("ts_ema", _field("close"), _arg(1)),
            ),
            _scalar(100.0),
        ),
        constraints=_FAST_LT_SLOW,
    ),
    # RSI and KDJ ---------------------------------------------------------------
    _formula(
        "ta_rsi_wilder",
        "RSI / Wilder relative strength index",
        "Wilder RSI of Close; flat windows return 50 and zero-loss windows return 100.",
        "rsi",
        10,
        ["rsi", "ta_rsi", "relative strength index", "wilder rsi"],
        [_lookback(14, 5, 30, "Wilder RSI period.")],
        _rsi_wilder_body(),
    ),
    _formula(
        "ta_kdj_rsv",
        "KDJ RSV",
        "Close within the trailing High-Low range (raw stochastic value).",
        "kdj",
        10,
        ["rsv", "kdj rsv", "stochastic raw value"],
        [_lookback(9, 5, 20, "Stochastic range period.")],
        _kdj_rsv_body(),
    ),
    _formula(
        "ta_kdj_k",
        "KDJ K",
        "RSV recursively smoothed with fixed α=1/3 and initial value 50.",
        "kdj",
        20,
        ["kdj k", "stochastic k"],
        [_lookback(9, 5, 20, "Stochastic range period.")],
        _call("ts_recursive_smooth", _call("ta_kdj_rsv", _arg(0)), _window(3), _scalar(50.0)),
    ),
    _formula(
        "ta_kdj_d",
        "KDJ D",
        "K recursively smoothed with fixed α=1/3 and initial value 50.",
        "kdj",
        30,
        ["kdj d", "stochastic d"],
        [_lookback(9, 5, 20, "Stochastic range period.")],
        _call("ts_recursive_smooth", _call("ta_kdj_k", _arg(0)), _window(3), _scalar(50.0)),
    ),
    _formula(
        "ta_kdj_j",
        "KDJ J",
        "KDJ J line: 3K − 2D.",
        "kdj",
        40,
        ["kdj j", "stochastic j"],
        [_lookback(9, 5, 20, "Stochastic range period.")],
        _call(
            "sub",
            _call("mul_scalar", _call("ta_kdj_k", _arg(0)), _scalar(3.0)),
            _call("mul_scalar", _call("ta_kdj_d", _arg(0)), _scalar(2.0)),
        ),
    ),
    # Bollinger bands -----------------------------------------------------------
    _formula(
        "ta_boll_middle",
        "Bollinger middle band",
        "Simple moving average of Close.",
        "bollinger_bands",
        10,
        ["boll middle", "bollinger middle", "boll mid"],
        [_lookback(20, 10, 60)],
        _call("ts_mean", _field("close"), _arg(0)),
    ),
    _formula(
        "ta_boll_upper",
        "Bollinger upper band",
        "Middle band plus two population standard deviations.",
        "bollinger_bands",
        20,
        ["boll upper", "bollinger upper", "upper band"],
        [_lookback(20, 10, 60)],
        _call(
            "add",
            _call("ta_boll_middle", _arg(0)),
            _call("mul_scalar", _call("ts_std_pop", _field("close"), _arg(0)), _scalar(2.0)),
        ),
    ),
    _formula(
        "ta_boll_lower",
        "Bollinger lower band",
        "Middle band minus two population standard deviations.",
        "bollinger_bands",
        30,
        ["boll lower", "bollinger lower", "lower band"],
        [_lookback(20, 10, 60)],
        _call(
            "sub",
            _call("ta_boll_middle", _arg(0)),
            _call("mul_scalar", _call("ts_std_pop", _field("close"), _arg(0)), _scalar(2.0)),
        ),
    ),
    _formula(
        "ta_boll_percent_b",
        "Bollinger %B",
        "Close position within the two-deviation Bollinger envelope.",
        "bollinger_bands",
        40,
        ["boll %b", "bollinger %b", "percent b"],
        [_lookback(20, 10, 60)],
        _call(
            "div",
            _call("sub", _field("close"), _call("ta_boll_lower", _arg(0))),
            _call("sub", _call("ta_boll_upper", _arg(0)), _call("ta_boll_lower", _arg(0))),
        ),
    ),
    _formula(
        "ta_boll_bandwidth",
        "Bollinger bandwidth",
        "Envelope width as a percentage of the middle band.",
        "bollinger_bands",
        50,
        ["boll bandwidth", "bollinger bandwidth", "bandwidth"],
        [_lookback(20, 10, 60)],
        _call(
            "mul_scalar",
            _call(
                "div",
                _call("sub", _call("ta_boll_upper", _arg(0)), _call("ta_boll_lower", _arg(0))),
                _call("ta_boll_middle", _arg(0)),
            ),
            _scalar(100.0),
        ),
    ),
    # Momentum ------------------------------------------------------------------
    _formula(
        "ta_roc",
        "ROC / Rate of change",
        "Percentage change in Close over the comparison lag.",
        "momentum",
        10,
        ["roc", "rate of change", "momentum percent"],
        [_lookback(10, 2, 60, "Comparison lag in periods.")],
        _call(
            "mul_scalar",
            _call(
                "add_scalar",
                _call("div", _field("close"), _call("delay", _field("close"), _arg(0))),
                _scalar(-1.0),
            ),
            _scalar(100.0),
        ),
    ),
    _formula(
        "ta_williams_r",
        "Williams %R",
        "Close position below the trailing High, scaled from −100 to 0.",
        "momentum",
        20,
        ["williams r", "williams %r", "%r"],
        [_lookback(14, 5, 30)],
        _call(
            "mul_scalar",
            _call(
                "div",
                _call("sub", _call("ts_max", _field("high"), _arg(0)), _field("close")),
                _call(
                    "sub",
                    _call("ts_max", _field("high"), _arg(0)),
                    _call("ts_min", _field("low"), _arg(0)),
                ),
            ),
            _scalar(-100.0),
        ),
    ),
    # Range volatility ----------------------------------------------------------
    _formula(
        "ta_true_range",
        "True range",
        "Maximum of High-Low and the two previous-Close gap ranges.",
        "range_volatility",
        10,
        ["tr", "true range"],
        [],
        _true_range_body(),
    ),
    _formula(
        "ta_atr",
        "ATR / Average true range",
        "Wilder average of True Range.",
        "range_volatility",
        20,
        ["atr", "average true range"],
        [_lookback(14, 5, 30, "Wilder ATR period.")],
        _call("ts_rma", _call("ta_true_range"), _arg(0)),
    ),
    _formula(
        "ta_natr",
        "NATR / Normalized average true range",
        "ATR as a percentage of Close.",
        "range_volatility",
        30,
        ["natr", "normalized atr"],
        [_lookback(14, 5, 30, "Wilder ATR period.")],
        _call(
            "mul_scalar",
            _call("div", _call("ta_atr", _arg(0)), _field("close")),
            _scalar(100.0),
        ),
    ),
    # Donchian ------------------------------------------------------------------
    _formula(
        "ta_donchian_upper",
        "Donchian upper channel",
        "Highest High in the trailing channel window.",
        "donchian",
        10,
        ["donchian upper", "price channel high"],
        [_lookback(20, 5, 100, "Channel lookback in periods.")],
        _call("ts_max", _field("high"), _arg(0)),
    ),
    _formula(
        "ta_donchian_lower",
        "Donchian lower channel",
        "Lowest Low in the trailing channel window.",
        "donchian",
        30,
        ["donchian lower", "price channel low"],
        [_lookback(20, 5, 100, "Channel lookback in periods.")],
        _call("ts_min", _field("low"), _arg(0)),
    ),
    _formula(
        "ta_donchian_middle",
        "Donchian middle channel",
        "Midpoint of the trailing upper and lower Donchian channels.",
        "donchian",
        20,
        ["donchian middle", "price channel middle"],
        [_lookback(20, 5, 100, "Channel lookback in periods.")],
        _call(
            "mul_scalar",
            _call("add", _call("ta_donchian_upper", _arg(0)), _call("ta_donchian_lower", _arg(0))),
            _scalar(0.5),
        ),
    ),
    _formula(
        "ta_donchian_position",
        "Donchian channel position",
        "Close position between the trailing Donchian channels, in percent.",
        "donchian",
        40,
        ["donchian position", "price channel position"],
        [_lookback(20, 5, 100, "Channel lookback in periods.")],
        _call(
            "mul_scalar",
            _call(
                "div",
                _call("sub", _field("close"), _call("ta_donchian_lower", _arg(0))),
                _call(
                    "sub",
                    _call("ta_donchian_upper", _arg(0)),
                    _call("ta_donchian_lower", _arg(0)),
                ),
            ),
            _scalar(100.0),
        ),
    ),
    # Directional movement ------------------------------------------------------
    _formula(
        "ta_plus_di",
        "+DI / Positive directional indicator",
        "Wilder-smoothed positive directional movement as a percentage of ATR.",
        "directional_movement",
        10,
        ["plus di", "+di", "positive directional indicator"],
        [_lookback(14, 5, 30, "Directional movement period.")],
        _directional_index(True),
    ),
    _formula(
        "ta_minus_di",
        "−DI / Negative directional indicator",
        "Wilder-smoothed negative directional movement as a percentage of ATR.",
        "directional_movement",
        20,
        ["minus di", "-di", "negative directional indicator"],
        [_lookback(14, 5, 30, "Directional movement period.")],
        _directional_index(False),
    ),
    _formula(
        "ta_dx",
        "DX / Directional movement index",
        "Absolute +DI/−DI spread as a percentage of their sum.",
        "directional_movement",
        30,
        ["dx", "directional movement index"],
        [_lookback(14, 5, 30, "Directional movement period.")],
        _call(
            "mul_scalar",
            _call(
                "div",
                _call(
                    "abs",
                    _call(
                        "sub",
                        _call("ta_plus_di", _arg(0)),
                        _call("ta_minus_di", _arg(0)),
                    ),
                ),
                _call("add", _call("ta_plus_di", _arg(0)), _call("ta_minus_di", _arg(0))),
            ),
            _scalar(100.0),
        ),
    ),
    _formula(
        "ta_adx",
        "ADX / Average directional index",
        "Wilder average of DX.",
        "directional_movement",
        40,
        ["adx", "average directional index"],
        [_lookback(14, 5, 30, "Directional movement period.")],
        _call("ts_rma", _call("ta_dx", _arg(0)), _arg(0)),
    ),
    # Volume --------------------------------------------------------------------
    _formula(
        "ta_obv",
        "OBV / On-balance volume",
        "Gap-reset cumulative signed Volume based on Close direction.",
        "volume",
        10,
        ["obv", "on balance volume"],
        [],
        _call(
            "ts_cumsum",
            _call(
                "mul",
                _call("sign", _call("delta", _field("close"), _window(1))),
                _field("volume"),
            ),
        ),
    ),
    _formula(
        "ta_mfi",
        "MFI / Money flow index",
        "Volume-weighted momentum from Typical Price; flat flow returns 50.",
        "volume",
        20,
        ["mfi", "money flow index"],
        [_lookback(14, 5, 40, "Money-flow lookback in periods.")],
        _mfi_body(),
    ),
    _formula(
        "ta_cmf",
        "CMF / Chaikin money flow",
        "Rolling accumulation/distribution flow divided by rolling Volume.",
        "volume",
        30,
        ["cmf", "chaikin money flow"],
        [_lookback(20, 5, 40, "Money-flow lookback in periods.")],
        _cmf_body(),
    ),
)

CATALOG_NAMES = frozenset(item["name"] for item in INDICATOR_CATALOG)
ACTIVE_CATALOG_NAMES = frozenset(
    item["name"] for item in INDICATOR_CATALOG if item["status"] == "active"
)

__all__ = [
    "CATALOG_CATEGORY",
    "ACTIVE_CATALOG_NAMES",
    "CATALOG_NAMES",
    "CATALOG_ORIGIN",
    "CATALOG_REVISION",
    "INDICATOR_CATALOG",
    "LEGACY_CATALOG_REPLACEMENTS",
]
