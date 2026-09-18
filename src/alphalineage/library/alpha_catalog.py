"""Versioned starter formulas for well-known published alpha factors.

Three families, all expressed with the existing typed operators (no new primitives, no code):

* ``worldquant_101`` - a faithful subset of Kakushadze (2016), "101 Formulaic Alphas" (Wilmott
  2016(84), arXiv:1601.00991). Windows are the published constants, so these carry no tunable
  inputs. Alphas that need operators this DSL does not have (``ts_argmax``, ``product``,
  ``IndNeutralize``) or a true VWAP are left out rather than approximated silently.
* ``qlib_alpha158`` - candlestick ("K-line") and rolling price/volume features from Microsoft
  Qlib's ``Alpha158`` handler. Each rolling feature takes the window as a bounded parameter
  (Qlib publishes 5/10/20/30/60).
* ``academic_anomalies`` - price/volume anomalies from the empirical asset-pricing literature
  (momentum, reversal, 52-week high, low volatility, MAX, illiquidity, skewness, overnight
  returns). Fundamental anomalies (value, quality, size) need point-in-time fundamentals and are
  not available on daily bars.

Honest deviations, repeated in each affected description:

* ``vwap`` in this platform is typical price (HLC3), not traded VWAP.
* ``log`` is the sign-safe ``sign(x) * log(1 + |x|)``; for share volume this is ``log(volume + 1)``.
* Division by zero yields a missing value instead of Qlib's ``+1e-12`` guard.

Bodies are written in the platform's formula text form and parsed once at import into the same
JSON tree the formula store persists. ``$0``/``$1`` bind declared inputs, integer literals are
windows and decimal literals are scalar constants.
"""

from __future__ import annotations

import re
from typing import Any

CATALOG_ORIGIN = "catalog_formula"
CATALOG_REVISION = 2
CLASSIC_ALPHAS_CATEGORY = "classic_alphas"

_TOKEN = re.compile(r"\s*(?:(\$\d+)|(-?\d+\.\d+)|(\d+)|([A-Za-z_][A-Za-z0-9_]*)|(.))")


def parse_formula_text(text: str) -> dict[str, Any]:
    """Parse ``name(arg, ...)`` text into a formula body tree (JSON form)."""
    tokens: list[tuple[str, str]] = []
    for match in _TOKEN.finditer(text.strip()):
        arg, scalar, window, name, punct = match.groups()
        if arg:
            tokens.append(("arg", arg[1:]))
        elif scalar:
            tokens.append(("scalar", scalar))
        elif window:
            tokens.append(("window", window))
        elif name:
            tokens.append(("name", name))
        elif punct and not punct.isspace():
            tokens.append(("punct", punct))
    position = 0

    def expect(kind: str, value: str | None = None) -> str:
        nonlocal position
        if position >= len(tokens):
            raise ValueError(f"unexpected end of formula text: {text!r}")
        token_kind, token_value = tokens[position]
        if token_kind != kind or (value is not None and token_value != value):
            raise ValueError(f"expected {value or kind} at token {position} in {text!r}")
        position += 1
        return token_value

    def node() -> dict[str, Any]:
        nonlocal position
        if position >= len(tokens):
            raise ValueError(f"unexpected end of formula text: {text!r}")
        kind, value = tokens[position]
        if kind == "arg":
            position += 1
            return {"name": "$arg", "value": int(value)}
        if kind == "scalar":
            position += 1
            return {"name": "const", "value": float(value)}
        if kind == "window":
            position += 1
            return {"name": "window", "value": int(value)}
        name = expect("name")
        if position < len(tokens) and tokens[position] == ("punct", "("):
            position += 1
            children = [node()]
            while position < len(tokens) and tokens[position] == ("punct", ","):
                position += 1
                children.append(node())
            expect("punct", ")")
            return {"name": name, "children": children}
        return {"name": name}

    body = node()
    if position != len(tokens):
        raise ValueError(f"trailing tokens in formula text {text!r}")
    return body


def _param(
    name: str,
    description: str,
    default: int,
    minimum: int,
    maximum: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "type": "window",
        "description": description,
        "default": default,
        "role": "parameter",
        "tuning": {"enabled": True, "min": minimum, "max": maximum, "step": 1, "radius": 1},
    }


def _window_param(default: int = 20, minimum: int = 5, maximum: int = 60) -> dict[str, Any]:
    return _param("lookback", "Rolling window in trading days.", default, minimum, maximum)


def _entry(
    name: str,
    display_name: str,
    description: str,
    family: str,
    family_order: int,
    aliases: list[str],
    body: str,
    inputs: list[dict[str, Any]] | None = None,
    constraints: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    declared = inputs or []
    return {
        "name": name,
        "display_name": display_name,
        "description": description,
        "arg_types": [item["type"] for item in declared],
        "inputs": declared,
        "out_type": "series",
        "body": parse_formula_text(body),
        "category": CLASSIC_ALPHAS_CATEGORY,
        "origin": CATALOG_ORIGIN,
        "editable": False,
        "family": family,
        "family_order": family_order,
        "aliases": aliases,
        "catalog_revision": CATALOG_REVISION,
        "constraints": constraints or [],
        "status": "active",
        "replacement": None,
    }


# Reusable text fragments ----------------------------------------------------------------------
_ONE = "add_scalar(mul_scalar(close, 0.0), 1.0)"
_ZERO = "mul_scalar(close, 0.0)"
_CHANGE = "delta(close, 1)"
_PREV_CLOSE = "delay(close, 1)"
_BODY_TOP = "where(gt(open, close), open, close)"
_BODY_BOTTOM = "where(lt(open, close), open, close)"
_RANGE = "sub(high, low)"
_TIME = f"ts_cumsum({_ONE})"


def _slope(series: str, window: str) -> str:
    return f"div(ts_cov({series}, {_TIME}, {window}), ts_cov({_TIME}, {_TIME}, {window}))"


def _count_mean(condition: str, window: str) -> str:
    return f"ts_mean(where({condition}, {_ONE}, {_ZERO}), {window})"


def _positive_part(series: str) -> str:
    return f"where(gt({series}, {_ZERO}), {series}, {_ZERO})"


_WQ = "worldquant_101"
_WQ_NOTE = "Kakushadze (2016), 101 Formulaic Alphas."
_QLIB = "qlib_alpha158"
_QLIB_NOTE = "Microsoft Qlib Alpha158."
_ANOM = "academic_anomalies"


WORLDQUANT_101: tuple[dict[str, Any], ...] = (
    _entry(
        "wq_alpha002",
        "WQ Alpha#2",
        f"{_WQ_NOTE} -corr(rank(Δ2 log volume), rank(intraday return), 6). log is log(volume + 1).",
        _WQ,
        2,
        ["alpha 2", "alpha#2"],
        "neg(ts_corr(rank(delta(log(volume), 2)), rank(div(sub(close, open), open)), 6))",
    ),
    _entry(
        "wq_alpha003",
        "WQ Alpha#3",
        f"{_WQ_NOTE} -corr(rank(open), rank(volume), 10).",
        _WQ,
        3,
        ["alpha 3", "alpha#3"],
        "neg(ts_corr(rank(open), rank(volume), 10))",
    ),
    _entry(
        "wq_alpha004",
        "WQ Alpha#4",
        f"{_WQ_NOTE} -Ts_Rank(rank(low), 9).",
        _WQ,
        4,
        ["alpha 4", "alpha#4"],
        "neg(ts_rank(rank(low), 9))",
    ),
    _entry(
        "wq_alpha006",
        "WQ Alpha#6",
        f"{_WQ_NOTE} -corr(open, volume, 10).",
        _WQ,
        6,
        ["alpha 6", "alpha#6"],
        "neg(ts_corr(open, volume, 10))",
    ),
    _entry(
        "wq_alpha009",
        "WQ Alpha#9",
        f"{_WQ_NOTE} Follow a persistent 5-day price trend, otherwise fade yesterday's change.",
        _WQ,
        9,
        ["alpha 9", "alpha#9"],
        f"where(gt(ts_min({_CHANGE}, 5), {_ZERO}), {_CHANGE}, "
        f"where(lt(ts_max({_CHANGE}, 5), {_ZERO}), {_CHANGE}, neg({_CHANGE})))",
    ),
    _entry(
        "wq_alpha012",
        "WQ Alpha#12",
        f"{_WQ_NOTE} sign(Δ volume) × (−Δ close).",
        _WQ,
        12,
        ["alpha 12", "alpha#12"],
        f"mul(sign(delta(volume, 1)), neg({_CHANGE}))",
    ),
    _entry(
        "wq_alpha013",
        "WQ Alpha#13",
        f"{_WQ_NOTE} -rank(cov(rank(close), rank(volume), 5)).",
        _WQ,
        13,
        ["alpha 13", "alpha#13"],
        "neg(rank(ts_cov(rank(close), rank(volume), 5)))",
    ),
    _entry(
        "wq_alpha014",
        "WQ Alpha#14",
        f"{_WQ_NOTE} -rank(Δ3 returns) × corr(open, volume, 10).",
        _WQ,
        14,
        ["alpha 14", "alpha#14"],
        "mul(neg(rank(delta(returns, 3))), ts_corr(open, volume, 10))",
    ),
    _entry(
        "wq_alpha016",
        "WQ Alpha#16",
        f"{_WQ_NOTE} -rank(cov(rank(high), rank(volume), 5)).",
        _WQ,
        16,
        ["alpha 16", "alpha#16"],
        "neg(rank(ts_cov(rank(high), rank(volume), 5)))",
    ),
    _entry(
        "wq_alpha018",
        "WQ Alpha#18",
        f"{_WQ_NOTE} -rank(std(|close−open|, 5) + (close−open) + corr(close, open, 10)).",
        _WQ,
        18,
        ["alpha 18", "alpha#18"],
        "neg(rank(add(add(ts_std(abs(sub(close, open)), 5), sub(close, open)), "
        "ts_corr(close, open, 10))))",
    ),
    _entry(
        "wq_alpha020",
        "WQ Alpha#20",
        f"{_WQ_NOTE} Opening gap versus yesterday's high, close and low.",
        _WQ,
        20,
        ["alpha 20", "alpha#20"],
        "mul(mul(neg(rank(sub(open, delay(high, 1)))), rank(sub(open, delay(close, 1)))), "
        "rank(sub(open, delay(low, 1))))",
    ),
    _entry(
        "wq_alpha023",
        "WQ Alpha#23",
        f"{_WQ_NOTE} −Δ2 high when high is above its 20-day mean, else 0.",
        _WQ,
        23,
        ["alpha 23", "alpha#23"],
        f"where(lt(ts_mean(high, 20), high), neg(delta(high, 2)), {_ZERO})",
    ),
    _entry(
        "wq_alpha026",
        "WQ Alpha#26",
        f"{_WQ_NOTE} -max over 3 days of corr(Ts_Rank(volume, 5), Ts_Rank(high, 5), 5).",
        _WQ,
        26,
        ["alpha 26", "alpha#26"],
        "neg(ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))",
    ),
    _entry(
        "wq_alpha033",
        "WQ Alpha#33",
        f"{_WQ_NOTE} rank(−(1 − open/close)).",
        _WQ,
        33,
        ["alpha 33", "alpha#33"],
        "rank(neg(add_scalar(mul_scalar(div(open, close), -1.0), 1.0)))",
    ),
    _entry(
        "wq_alpha034",
        "WQ Alpha#34",
        f"{_WQ_NOTE} Low short-term volatility ratio plus short-term reversal, ranked.",
        _WQ,
        34,
        ["alpha 34", "alpha#34"],
        "rank(add(add_scalar(neg(rank(div(ts_std(returns, 2), ts_std(returns, 5)))), 1.0), "
        f"add_scalar(neg(rank({_CHANGE})), 1.0)))",
    ),
    _entry(
        "wq_alpha038",
        "WQ Alpha#38",
        f"{_WQ_NOTE} -rank(Ts_Rank(close, 10)) × rank(close/open).",
        _WQ,
        38,
        ["alpha 38", "alpha#38"],
        "mul(neg(rank(ts_rank(close, 10))), rank(div(close, open)))",
    ),
    _entry(
        "wq_alpha040",
        "WQ Alpha#40",
        f"{_WQ_NOTE} -rank(std(high, 10)) × corr(high, volume, 10).",
        _WQ,
        40,
        ["alpha 40", "alpha#40"],
        "mul(neg(rank(ts_std(high, 10))), ts_corr(high, volume, 10))",
    ),
    _entry(
        "wq_alpha044",
        "WQ Alpha#44",
        f"{_WQ_NOTE} -corr(high, rank(volume), 5).",
        _WQ,
        44,
        ["alpha 44", "alpha#44"],
        "neg(ts_corr(high, rank(volume), 5))",
    ),
    _entry(
        "wq_alpha053",
        "WQ Alpha#53",
        f"{_WQ_NOTE} −9-day change in the close-location value ((C−L)−(H−C))/(C−L).",
        _WQ,
        53,
        ["alpha 53", "alpha#53"],
        "neg(delta(div(sub(sub(close, low), sub(high, close)), sub(close, low)), 9))",
    ),
    _entry(
        "wq_alpha054",
        "WQ Alpha#54",
        f"{_WQ_NOTE} −(L−C)·O⁵ / ((L−H)·C⁵).",
        _WQ,
        54,
        ["alpha 54", "alpha#54"],
        "div(neg(mul(sub(low, close), signed_power(open, 5.0))), "
        "mul(sub(low, high), signed_power(close, 5.0)))",
    ),
    _entry(
        "wq_alpha055",
        "WQ Alpha#55",
        f"{_WQ_NOTE} -corr(rank(12-day stochastic position), rank(volume), 6).",
        _WQ,
        55,
        ["alpha 55", "alpha#55"],
        "neg(ts_corr(rank(div(sub(close, ts_min(low, 12)), "
        "sub(ts_max(high, 12), ts_min(low, 12)))), rank(volume), 6))",
    ),
    _entry(
        "wq_alpha101",
        "WQ Alpha#101",
        f"{_WQ_NOTE} (close − open) / ((high − low) + 0.001).",
        _WQ,
        101,
        ["alpha 101", "alpha#101"],
        "div(sub(close, open), add_scalar(sub(high, low), 0.001))",
    ),
)


QLIB_ALPHA158: tuple[dict[str, Any], ...] = (
    _entry(
        "qlib_kmid",
        "Qlib KMID",
        f"{_QLIB_NOTE} Candle body relative to open: (C−O)/O.",
        _QLIB,
        10,
        ["kmid"],
        "div(sub(close, open), open)",
    ),
    _entry(
        "qlib_klen",
        "Qlib KLEN",
        f"{_QLIB_NOTE} Candle range relative to open: (H−L)/O.",
        _QLIB,
        11,
        ["klen"],
        f"div({_RANGE}, open)",
    ),
    _entry(
        "qlib_kmid2",
        "Qlib KMID2",
        f"{_QLIB_NOTE} Body as a share of the range: (C−O)/(H−L).",
        _QLIB,
        12,
        ["kmid2"],
        f"div(sub(close, open), {_RANGE})",
    ),
    _entry(
        "qlib_kup",
        "Qlib KUP",
        f"{_QLIB_NOTE} Upper shadow relative to open: (H−max(O,C))/O.",
        _QLIB,
        13,
        ["kup", "upper shadow"],
        f"div(sub(high, {_BODY_TOP}), open)",
    ),
    _entry(
        "qlib_klow",
        "Qlib KLOW",
        f"{_QLIB_NOTE} Lower shadow relative to open: (min(O,C)−L)/O.",
        _QLIB,
        14,
        ["klow", "lower shadow"],
        f"div(sub({_BODY_BOTTOM}, low), open)",
    ),
    _entry(
        "qlib_ksft",
        "Qlib KSFT",
        f"{_QLIB_NOTE} Close skew within the bar: (2C−H−L)/O.",
        _QLIB,
        15,
        ["ksft"],
        "div(sub(mul_scalar(close, 2.0), add(high, low)), open)",
    ),
    _entry(
        "qlib_roc",
        "Qlib ROC",
        f"{_QLIB_NOTE} Lagged close over current close: Ref(C,d)/C.",
        _QLIB,
        20,
        ["roc (qlib)"],
        "div(delay(close, $0), close)",
        [_window_param()],
    ),
    _entry(
        "qlib_ma",
        "Qlib MA",
        f"{_QLIB_NOTE} Moving average over close: Mean(C,d)/C.",
        _QLIB,
        21,
        ["ma (qlib)"],
        "div(ts_mean(close, $0), close)",
        [_window_param()],
    ),
    _entry(
        "qlib_std",
        "Qlib STD",
        f"{_QLIB_NOTE} Rolling deviation of close over close: Std(C,d)/C.",
        _QLIB,
        22,
        ["std (qlib)"],
        "div(ts_std(close, $0), close)",
        [_window_param()],
    ),
    _entry(
        "qlib_beta",
        "Qlib BETA",
        f"{_QLIB_NOTE} Linear-trend slope of close over close: Slope(C,d)/C.",
        _QLIB,
        23,
        ["beta (qlib)", "slope"],
        f"div({_slope('close', '$0')}, close)",
        [_window_param()],
    ),
    _entry(
        "qlib_rsqr",
        "Qlib RSQR",
        f"{_QLIB_NOTE} R² of the linear trend in close over the window.",
        _QLIB,
        24,
        ["rsqr", "trend r squared"],
        f"mul(ts_corr(close, {_TIME}, $0), ts_corr(close, {_TIME}, $0))",
        [_window_param()],
    ),
    _entry(
        "qlib_resi",
        "Qlib RESI",
        f"{_QLIB_NOTE} Today's residual from the rolling linear trend, over close.",
        _QLIB,
        25,
        ["resi", "trend residual"],
        f"div(sub(sub(close, ts_mean(close, $0)), mul({_slope('close', '$0')}, "
        f"sub({_TIME}, ts_mean({_TIME}, $0)))), close)",
        [_window_param()],
    ),
    _entry(
        "qlib_max",
        "Qlib MAX",
        f"{_QLIB_NOTE} Highest high over close: Max(H,d)/C.",
        _QLIB,
        26,
        ["max (qlib)"],
        "div(ts_max(high, $0), close)",
        [_window_param()],
    ),
    _entry(
        "qlib_min",
        "Qlib MIN",
        f"{_QLIB_NOTE} Lowest low over close: Min(L,d)/C.",
        _QLIB,
        27,
        ["min (qlib)"],
        "div(ts_min(low, $0), close)",
        [_window_param()],
    ),
    _entry(
        "qlib_rank",
        "Qlib RANK",
        f"{_QLIB_NOTE} Percentile of today's close within the window.",
        _QLIB,
        28,
        ["rank (qlib)"],
        "ts_rank(close, $0)",
        [_window_param()],
    ),
    _entry(
        "qlib_rsv",
        "Qlib RSV",
        f"{_QLIB_NOTE} Close position within the rolling high-low range.",
        _QLIB,
        29,
        ["rsv (qlib)"],
        "div(sub(close, ts_min(low, $0)), sub(ts_max(high, $0), ts_min(low, $0)))",
        [_window_param()],
    ),
    _entry(
        "qlib_corr",
        "Qlib CORR",
        f"{_QLIB_NOTE} corr(close, log(volume + 1), d).",
        _QLIB,
        30,
        ["corr (qlib)", "price volume correlation"],
        "ts_corr(close, log(volume), $0)",
        [_window_param()],
    ),
    _entry(
        "qlib_cord",
        "Qlib CORD",
        f"{_QLIB_NOTE} corr(close ratio, log(volume ratio + 1), d).",
        _QLIB,
        31,
        ["cord"],
        f"ts_corr(div(close, {_PREV_CLOSE}), log(div(volume, delay(volume, 1))), $0)",
        [_window_param()],
    ),
    _entry(
        "qlib_cntd",
        "Qlib CNTD",
        f"{_QLIB_NOTE} Share of up days minus share of down days.",
        _QLIB,
        32,
        ["cntd", "up down count"],
        f"sub({_count_mean(f'gt(close, {_PREV_CLOSE})', '$0')}, "
        f"{_count_mean(f'lt(close, {_PREV_CLOSE})', '$0')})",
        [_window_param()],
    ),
    _entry(
        "qlib_sumd",
        "Qlib SUMD",
        f"{_QLIB_NOTE} (Sum of gains − sum of losses) / sum of |changes| (RSI-like).",
        _QLIB,
        33,
        ["sumd"],
        f"div(sub(ts_sum({_positive_part(_CHANGE)}, $0), "
        f"ts_sum({_positive_part(f'neg({_CHANGE})')}, $0)), ts_sum(abs({_CHANGE}), $0))",
        [_window_param()],
    ),
    _entry(
        "qlib_vma",
        "Qlib VMA",
        f"{_QLIB_NOTE} Average volume over today's volume.",
        _QLIB,
        34,
        ["vma"],
        "div(ts_mean(volume, $0), volume)",
        [_window_param()],
    ),
    _entry(
        "qlib_vstd",
        "Qlib VSTD",
        f"{_QLIB_NOTE} Volume deviation over today's volume.",
        _QLIB,
        35,
        ["vstd"],
        "div(ts_std(volume, $0), volume)",
        [_window_param()],
    ),
    _entry(
        "qlib_wvma",
        "Qlib WVMA",
        f"{_QLIB_NOTE} Coefficient of variation of volume-weighted absolute returns.",
        _QLIB,
        36,
        ["wvma"],
        f"div(ts_std(mul(abs(add_scalar(div(close, {_PREV_CLOSE}), -1.0)), volume), $0), "
        f"ts_mean(mul(abs(add_scalar(div(close, {_PREV_CLOSE}), -1.0)), volume), $0))",
        [_window_param()],
    ),
    _entry(
        "qlib_vsumd",
        "Qlib VSUMD",
        f"{_QLIB_NOTE} Net volume increases over total absolute volume changes.",
        _QLIB,
        37,
        ["vsumd"],
        f"div(sub(ts_sum({_positive_part('delta(volume, 1)')}, $0), "
        f"ts_sum({_positive_part('neg(delta(volume, 1))')}, $0)), "
        "ts_sum(abs(delta(volume, 1)), $0))",
        [_window_param()],
    ),
)


_SKIP_LT_LOOKBACK = [{"left": "skip", "operator": "lt", "right": "lookback"}]


ACADEMIC_ANOMALIES: tuple[dict[str, Any], ...] = (
    _entry(
        "anom_momentum_12_1",
        "Momentum (12-1)",
        "Jegadeesh & Titman (1993): return from ~12 months ago to ~1 month ago, skipping the "
        "most recent month to avoid short-term reversal. Higher values predicted higher returns.",
        _ANOM,
        10,
        ["momentum", "12-1 momentum", "jegadeesh titman"],
        "add_scalar(div(delay(close, $0), delay(close, $1)), -1.0)",
        [
            _param("skip", "Recent days skipped (about one month).", 21, 5, 42),
            _param("lookback", "Formation start in days back (about one year).", 252, 126, 300),
        ],
        _SKIP_LT_LOOKBACK,
    ),
    _entry(
        "anom_short_term_reversal",
        "Short-term reversal (1 month)",
        "Jegadeesh (1990); Lehmann (1990): negative of the past-month return. Recent losers "
        "tended to outperform recent winners over the following month.",
        _ANOM,
        20,
        ["reversal", "short term reversal", "1 month reversal"],
        "neg(add_scalar(div(close, delay(close, $0)), -1.0))",
        [_param("lookback", "Return window in days.", 21, 5, 42)],
    ),
    _entry(
        "anom_weekly_reversal",
        "Weekly reversal",
        "Lehmann (1990): negative of the past-week return; a liquidity-provision effect that is "
        "very sensitive to trading costs.",
        _ANOM,
        21,
        ["weekly reversal", "5 day reversal"],
        "neg(add_scalar(div(close, delay(close, $0)), -1.0))",
        [_param("lookback", "Return window in days.", 5, 2, 10)],
    ),
    _entry(
        "anom_long_term_reversal",
        "Long-term reversal (36-13)",
        "De Bondt & Thaler (1985): negative of the return from ~3 years ago to ~1 year ago. "
        "Needs at least three years of history per symbol.",
        _ANOM,
        22,
        ["long term reversal", "de bondt thaler"],
        "neg(add_scalar(div(delay(close, $0), delay(close, $1)), -1.0))",
        [
            _param("skip", "Recent days skipped (about one year).", 252, 126, 300),
            _param("lookback", "Formation start in days back (about three years).", 756, 504, 1008),
        ],
        _SKIP_LT_LOOKBACK,
    ),
    _entry(
        "anom_52_week_high",
        "52-week-high proximity",
        "George & Hwang (2004): close divided by its highest close over the past year. Stocks "
        "near their 52-week high tended to keep outperforming.",
        _ANOM,
        30,
        ["52 week high", "george hwang"],
        "div(close, ts_max(close, $0))",
        [_param("lookback", "High window in days.", 252, 126, 300)],
    ),
    _entry(
        "anom_price_to_moving_average",
        "Price relative to moving average",
        "Trend signal in the spirit of Brock, Lakonishok & LeBaron (1992) and Han, Yang & Zhou "
        "(2013): close over its long moving average.",
        _ANOM,
        31,
        ["price to ma", "moving average trend", "200 day"],
        "div(close, ts_mean(close, $0))",
        [_param("lookback", "Moving-average window in days.", 200, 50, 252)],
    ),
    _entry(
        "anom_low_volatility",
        "Low volatility",
        "Ang, Hodrick, Xing & Zhang (2006); Baker, Bradley & Wurgler (2011): negative of "
        "total return volatility. Low-volatility stocks earned higher risk-adjusted returns. "
        "Total, not idiosyncratic, volatility (no market regression is available).",
        _ANOM,
        40,
        ["low volatility", "low vol", "volatility anomaly"],
        "neg(ts_std(returns, $0))",
        [_param("lookback", "Volatility window in days.", 63, 20, 252)],
    ),
    _entry(
        "anom_max_daily_return",
        "MAX (lottery) effect",
        "Bali, Cakici & Whitelaw (2011): the largest daily return over the past month. High "
        "values (lottery-like stocks) predicted lower returns, so expect a negative IC.",
        _ANOM,
        41,
        ["max effect", "lottery", "bali cakici whitelaw"],
        "ts_max(returns, $0)",
        [_param("lookback", "Window in days.", 21, 10, 63)],
    ),
    _entry(
        "anom_realized_skewness",
        "Realized skewness",
        "Amaya, Christoffersen, Jacobs & Vasquez (2015): skewness of daily returns over the "
        "window (population moments). Positive skew predicted lower returns.",
        _ANOM,
        42,
        ["skewness", "realized skew"],
        "div(add(sub(ts_mean(signed_power(returns, 3.0), $0), mul_scalar(mul(ts_mean(returns, $0), "
        "ts_mean(mul(returns, returns), $0)), 3.0)), mul_scalar(signed_power(ts_mean(returns, $0), "
        "3.0), 2.0)), signed_power(ts_std_pop(returns, $0), 3.0))",
        [_param("lookback", "Window in days.", 63, 21, 252)],
    ),
    _entry(
        "anom_amihud_illiquidity",
        "Amihud illiquidity",
        "Amihud (2002): average |return| per dollar traded (close × volume). Less liquid stocks "
        "earned a premium. Values are tiny; rank-based IC is unaffected by scale.",
        _ANOM,
        50,
        ["amihud", "illiquidity"],
        "ts_mean(div(abs(returns), mul(close, volume)), $0)",
        [_param("lookback", "Window in days.", 21, 5, 252)],
    ),
    _entry(
        "anom_abnormal_volume",
        "Abnormal volume",
        "Gervais, Kaniel & Mingelgrin (2001): recent volume relative to its long-run average. "
        "Unusually high volume preceded higher returns.",
        _ANOM,
        51,
        ["high volume premium", "volume shock", "abnormal volume"],
        "div(ts_mean(volume, $0), ts_mean(volume, $1))",
        [
            _param("recent", "Recent volume window in days.", 5, 1, 21),
            _param("baseline", "Baseline volume window in days.", 252, 63, 300),
        ],
        [{"left": "recent", "operator": "lt", "right": "baseline"}],
    ),
    _entry(
        "anom_overnight_return",
        "Average overnight return",
        "Lou, Polk & Skouras (2019): mean close-to-open return. Overnight and intraday "
        "returns carry different, partly offsetting, return premia.",
        _ANOM,
        60,
        ["overnight return", "tug of war"],
        f"ts_mean(add_scalar(div(open, {_PREV_CLOSE}), -1.0), $0)",
        [_param("lookback", "Window in days.", 21, 5, 63)],
    ),
    _entry(
        "anom_intraday_return",
        "Average intraday return",
        "Lou, Polk & Skouras (2019): mean open-to-close return, the intraday counterpart of "
        "the overnight return.",
        _ANOM,
        61,
        ["intraday return"],
        "ts_mean(add_scalar(div(close, open), -1.0), $0)",
        [_param("lookback", "Window in days.", 21, 5, 63)],
    ),
)


ALPHA_CATALOG: tuple[dict[str, Any], ...] = WORLDQUANT_101 + QLIB_ALPHA158 + ACADEMIC_ANOMALIES

__all__ = [
    "ACADEMIC_ANOMALIES",
    "ALPHA_CATALOG",
    "CLASSIC_ALPHAS_CATEGORY",
    "QLIB_ALPHA158",
    "WORLDQUANT_101",
    "parse_formula_text",
]
