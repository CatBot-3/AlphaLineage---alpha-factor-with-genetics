"""Published alpha-factor starter formulas: typed, budget-safe, and numerically faithful."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import stats

from alphalineage.api.app import app, get_panel
from alphalineage.core import extensions
from alphalineage.core.evaluate import evaluate_python
from alphalineage.core.extensions import expand_all
from alphalineage.core.panel import Panel
from alphalineage.core.primitives import OPERATORS, REGISTRY
from alphalineage.core.tree import Node
from alphalineage.library.alpha_catalog import (
    ACADEMIC_ANOMALIES,
    ALPHA_CATALOG,
    QLIB_ALPHA158,
    WORLDQUANT_101,
    parse_formula_text,
)
from alphalineage.library.indicator_catalog import TECHNICAL_INDICATOR_CATALOG


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    ops, reg, users = dict(OPERATORS), dict(REGISTRY), dict(extensions.USER_OPERATORS)
    yield
    for table, saved in ((OPERATORS, ops), (REGISTRY, reg), (extensions.USER_OPERATORS, users)):
        table.clear()
        table.update(saved)


@pytest.fixture(scope="module")
def long_panel() -> Panel:
    """800 sessions x 12 symbols: long enough for the 756-day long-term reversal window."""
    rng = np.random.default_rng(7)
    days, names = 800, 12
    dates = pd.date_range("2020-01-01", periods=days, freq="B")
    symbols = [f"S{i:02d}" for i in range(names)]
    returns = rng.normal(0.0003, 0.018, size=(days, names))
    close = pd.DataFrame(50.0 * np.cumprod(1.0 + returns, axis=0), index=dates, columns=symbols)
    gap = pd.DataFrame(rng.normal(0.0, 0.004, size=(days, names)), index=dates, columns=symbols)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1.0 + gap)
    spread = pd.DataFrame(
        rng.uniform(0.002, 0.03, size=(days, names)), index=dates, columns=symbols
    )
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = pd.DataFrame(
        rng.lognormal(13.0, 0.5, size=(days, names)), index=dates, columns=symbols
    )
    return Panel.from_prices(open=open_, high=high, low=low, close=close, volume=volume)


@pytest.fixture
def client(synthetic_panel) -> Iterator[TestClient]:
    app.dependency_overrides[get_panel] = lambda: synthetic_panel
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _default_tree(formula: dict) -> Node:
    children = tuple(
        Node("window", value=int(item["default"]))
        if item["type"] == "window"
        else Node("const", value=float(item["default"]))
        for item in formula["inputs"]
    )
    return Node(formula["runtime_name"], children)


def test_catalog_families_are_distinct_and_cite_their_source() -> None:
    names = [item["name"] for item in ALPHA_CATALOG]
    assert len(names) == len(set(names))
    assert not set(names) & {item["name"] for item in TECHNICAL_INDICATOR_CATALOG}
    assert {item["family"] for item in WORLDQUANT_101} == {"worldquant_101"}
    assert {item["family"] for item in QLIB_ALPHA158} == {"qlib_alpha158"}
    assert {item["family"] for item in ACADEMIC_ANOMALIES} == {"academic_anomalies"}
    assert len(WORLDQUANT_101) >= 20 and len(QLIB_ALPHA158) >= 20 and len(ACADEMIC_ANOMALIES) >= 12
    for item in ALPHA_CATALOG:
        assert item["category"] == "classic_alphas"
        assert item["editable"] is False and item["status"] == "active"
        assert item["description"] and item["aliases"]
    assert all("Kakushadze" in item["description"] for item in WORLDQUANT_101)
    assert all("Qlib" in item["description"] for item in QLIB_ALPHA158)


def test_formula_text_parser_distinguishes_arguments_windows_and_scalars() -> None:
    assert parse_formula_text("add_scalar(ts_mean(close, $0), -1.5)") == {
        "name": "add_scalar",
        "children": [
            {
                "name": "ts_mean",
                "children": [{"name": "close"}, {"name": "$arg", "value": 0}],
            },
            {"name": "const", "value": -1.5},
        ],
    }
    with pytest.raises(ValueError):
        parse_formula_text("ts_mean(close, 5")
    with pytest.raises(ValueError):
        parse_formula_text("close) extra")


def test_every_published_alpha_registers_fits_budget_and_evaluates(client, long_panel) -> None:
    formulas = {item["name"]: item for item in client.get("/formulas").json()}
    for definition in ALPHA_CATALOG:
        formula = formulas[definition["name"]]
        assert formula["origin"] == "catalog_formula", formula
        assert formula["registered"] is not False and not formula.get("error"), formula
        tree = _default_tree(formula)
        expanded = expand_all(tree, max_depth=32, max_nodes=40)
        assert expanded.unique_computation_size() <= 40, definition["name"]
        value = evaluate_python(expanded, long_panel)
        assert isinstance(value, pd.DataFrame), definition["name"]
        finite = np.isfinite(value.to_numpy(dtype="float64"))
        # Each factor must produce usable values after warm-up. The bar is deliberately modest:
        # correlations of *ranked price levels* (Alpha#3) are undefined whenever a name's rank
        # does not move inside the window, which is common in a 12-name synthetic panel.
        assert finite[-40:].mean() > 0.3, definition["name"]
        assert value.iloc[-40:].nunique(axis=1).gt(1).mean() > 0.9, definition["name"]


def _evaluate(client, name: str, long_panel: Panel, **overrides: int) -> pd.DataFrame:
    formula = next(item for item in client.get("/formulas").json() if item["name"] == name)
    children = tuple(
        Node("window", value=int(overrides.get(item["name"], item["default"])))
        for item in formula["inputs"]
    )
    tree = Node(formula["runtime_name"], children)
    result = evaluate_python(expand_all(tree, max_depth=32, max_nodes=40), long_panel)
    assert isinstance(result, pd.DataFrame)
    return result


def test_worldquant_alphas_match_their_published_arithmetic(client, long_panel) -> None:
    p = long_panel
    alpha101 = _evaluate(client, "wq_alpha101", p)
    pd.testing.assert_frame_equal(
        alpha101, (p["close"] - p["open"]) / ((p["high"] - p["low"]) + 0.001)
    )

    alpha6 = _evaluate(client, "wq_alpha006", p)
    expected6 = -p["open"].rolling(10).corr(p["volume"])
    np.testing.assert_allclose(alpha6.iloc[-5:], expected6.iloc[-5:], rtol=1e-9)

    alpha12 = _evaluate(client, "wq_alpha012", p)
    expected12 = np.sign(p["volume"].diff()) * -p["close"].diff()
    np.testing.assert_allclose(alpha12.iloc[-5:], expected12.iloc[-5:], rtol=1e-12)

    alpha9 = _evaluate(client, "wq_alpha009", p)
    change = p["close"].diff()
    expected9 = change.where(
        change.rolling(5).min() > 0, change.where(change.rolling(5).max() < 0, -change)
    )
    np.testing.assert_allclose(alpha9.iloc[-5:], expected9.iloc[-5:], rtol=1e-12)


def test_qlib_trend_features_match_least_squares(client, long_panel) -> None:
    p = long_panel
    window = 20
    symbol = p.symbols[3]
    tail = p["close"][symbol].iloc[-window:].to_numpy()
    x = np.arange(window, dtype=float)
    slope, intercept = np.polyfit(x, tail, 1)
    last_close = tail[-1]

    beta = _evaluate(client, "qlib_beta", p, lookback=window)
    assert beta[symbol].iloc[-1] == pytest.approx(slope / last_close, rel=1e-9)

    rsqr = _evaluate(client, "qlib_rsqr", p, lookback=window)
    assert rsqr[symbol].iloc[-1] == pytest.approx(np.corrcoef(x, tail)[0, 1] ** 2, rel=1e-9)

    resi = _evaluate(client, "qlib_resi", p, lookback=window)
    residual = last_close - (intercept + slope * x[-1])
    assert resi[symbol].iloc[-1] == pytest.approx(residual / last_close, rel=1e-7)

    kmid2 = _evaluate(client, "qlib_kmid2", p)
    pd.testing.assert_frame_equal(kmid2, (p["close"] - p["open"]) / (p["high"] - p["low"]))


def test_academic_anomalies_match_their_definitions(client, long_panel) -> None:
    p = long_panel
    momentum = _evaluate(client, "anom_momentum_12_1", p)
    np.testing.assert_allclose(
        momentum.iloc[-3:], (p["close"].shift(21) / p["close"].shift(252) - 1.0).iloc[-3:]
    )

    skew = _evaluate(client, "anom_realized_skewness", p, lookback=63)
    symbol = p.symbols[5]
    sample = p["returns"][symbol].iloc[-63:].to_numpy()
    assert skew[symbol].iloc[-1] == pytest.approx(stats.skew(sample, bias=True), rel=1e-6)

    amihud = _evaluate(client, "anom_amihud_illiquidity", p)
    expected = (p["returns"].abs() / (p["close"] * p["volume"])).rolling(21).mean()
    np.testing.assert_allclose(amihud.iloc[-3:], expected.iloc[-3:], rtol=1e-9)

    high52 = _evaluate(client, "anom_52_week_high", p)
    assert (high52.iloc[-5:] <= 1.0 + 1e-12).all().all()


def test_classic_alphas_are_opt_in_for_new_runs() -> None:
    from alphalineage.core.categories import (
        CLASSIC_ALPHAS,
        DEFAULT_CATEGORY_ORDER,
        DEFAULT_ENABLED_CATEGORIES,
    )

    assert CLASSIC_ALPHAS in DEFAULT_CATEGORY_ORDER
    assert CLASSIC_ALPHAS not in DEFAULT_ENABLED_CATEGORIES
