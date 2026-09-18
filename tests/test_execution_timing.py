"""Execution timing: which return a signal can actually earn, frozen per session."""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from alphalineage.agent.guards import PROTECTED_KEYS, validate_config_patch
from alphalineage.api.app import app, get_panel
from alphalineage.backtest.costs import TransactionCostModel
from alphalineage.backtest.engine import backtest
from alphalineage.backtest.portfolio import RankProportional
from alphalineage.core import extensions
from alphalineage.core.evaluate import evaluate
from alphalineage.core.fitness import (
    EXECUTION_TIMINGS,
    daily_ic,
    forward_returns,
    label_span,
)
from alphalineage.core.gp import GPConfig
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node


def test_forward_returns_per_execution_timing(synthetic_panel) -> None:
    panel = synthetic_panel
    close, open_ = panel["close"], panel["open"]

    # The legacy target is unchanged bit for bit.
    pd.testing.assert_frame_equal(forward_returns(panel), panel["returns"].shift(-1))
    pd.testing.assert_frame_equal(forward_returns(panel, 1, "close"), panel["returns"].shift(-1))
    pd.testing.assert_frame_equal(forward_returns(panel, 3, "close"), close.shift(-3) / close - 1.0)

    pd.testing.assert_frame_equal(
        forward_returns(panel, 1, "next_open"), open_.shift(-2) / open_.shift(-1) - 1.0
    )
    pd.testing.assert_frame_equal(
        forward_returns(panel, 2, "next_close"), close.shift(-3) / close.shift(-1) - 1.0
    )
    # A delayed label needs one more future bar, so the last row with a value moves back one.
    assert forward_returns(panel, 1, "next_open").iloc[-2].isna().all()
    assert forward_returns(panel, 1, "close").iloc[-2].notna().all()

    with pytest.raises(ValueError, match="execution"):
        forward_returns(panel, 1, "tomorrow")


def test_label_span_and_config_validation() -> None:
    assert EXECUTION_TIMINGS == ("close", "next_open", "next_close")
    assert label_span(1, "close") == 1
    assert label_span(1, "next_open") == 2
    assert label_span(5, "next_close") == 6
    assert GPConfig().execution == "close"
    # Stored configs and checkpoints written before the setting existed keep their meaning.
    legacy = GPConfig(population_size=8, generations=1).to_dict()
    legacy.pop("execution")
    assert GPConfig.from_dict(legacy).execution == "close"
    assert GPConfig(execution="next_open").to_dict()["execution"] == "next_open"
    with pytest.raises(ValueError, match="execution"):
        GPConfig(execution="open")


@pytest.mark.parametrize(
    ("execution", "price_field"),
    [("next_open", "open"), ("next_close", "close")],
)
def test_backtest_enters_at_the_execution_time(synthetic_panel, execution, price_field) -> None:
    panel = synthetic_panel
    factor = evaluate(Node("rank", (Node("close"),)), panel)
    scheme = RankProportional()
    costs = TransactionCostModel(commission_bps=1.0, slippage_bps=5.0)
    result = backtest(
        factor,
        panel,
        forward_returns(panel, 1, execution),
        scheme,
        costs,
        execution=execution,
    )

    # Independent recompute: weights decided on signal date t earn the price change from the
    # first tradable bar (t + 1) to the next one (t + 2); costs are charged on that position.
    weights = scheme.weights(factor)
    price = panel[price_field]
    earned = price.shift(-2) / price.shift(-1) - 1.0
    gross = (weights * earned).sum(axis=1, min_count=1)
    held = weights.shift(1).fillna(0.0)
    traded = held.diff()
    traded.iloc[0] = held.iloc[0]
    cost = (traded.abs().sum(axis=1) * (6.0 / 1e4)).shift(-1)
    expected_net = gross - cost

    valid = result.gross_returns.index[:-2]
    np.testing.assert_allclose(result.gross_returns.loc[valid], gross.loc[valid], rtol=1e-12)
    np.testing.assert_allclose(result.net_returns.loc[valid], expected_net.loc[valid], rtol=1e-12)
    assert list(pd.DatetimeIndex(result.realization_dates.iloc[:-2])) == list(factor.index[2:])
    assert result.realization_dates.iloc[-2:].isna().all()


def test_same_close_bounce_is_not_earnable_with_delayed_execution() -> None:
    """A reversal that only exists in closing-price noise vanishes once the trade waits."""
    rng = np.random.default_rng(3)
    days, names = 600, 40
    dates = pd.date_range("2020-01-01", periods=days, freq="B")
    symbols = [f"S{i}" for i in range(names)]
    fair = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (days, names)), axis=0))
    close = pd.DataFrame(fair * np.exp(rng.normal(0.0, 0.01, (days, names))), dates, symbols)
    open_ = pd.DataFrame(fair * np.exp(rng.normal(0.0, 0.002, (days, names))), dates, symbols)
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = pd.DataFrame(1e6, index=dates, columns=symbols)
    panel = Panel.from_prices(open=open_, high=high, low=low, close=close, volume=volume)

    reversal = -panel["returns"]
    same_close = daily_ic(reversal, forward_returns(panel, 1, "close"), min_names=10).mean()
    next_close = daily_ic(reversal, forward_returns(panel, 1, "next_close"), min_names=10).mean()
    assert same_close > 0.25
    assert abs(next_close) < 0.05


# --- sessions ---------------------------------------------------------------------------------
_SMALL = {"population_size": 16, "generations": 2, "max_depth": 4, "max_nodes": 20, "seed": 0}


@pytest.fixture
def client(signal_panel) -> Iterator[TestClient]:
    panel, _ = signal_panel
    app.dependency_overrides[get_panel] = lambda: panel
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    extensions.clear_user_operators()


def _poll_job(client: TestClient, job_id: str, *, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/runs/{job_id}").json()
        if payload["status"] in ("done", "stopped", "failed"):
            return payload
        time.sleep(0.1)
    return payload


def test_session_freezes_execution_timing_and_reports_it(client) -> None:
    body = {
        "name": "timed",
        "universe": "sp500-lite",
        "config": {**_SMALL, "execution": "next_open"},
    }
    created = client.post("/sessions", json=body)
    assert created.status_code == 200, created.text
    session_id = created.json()["session_id"]
    assert _poll_job(client, created.json()["job_id"])["status"] == "done"
    state = client.get(f"/sessions/{session_id}").json()
    assert state["config"]["execution"] == "next_open"

    changed = client.post(
        f"/sessions/{session_id}/continue",
        json={"generations": 1, "config": {"execution": "close"}},
    )
    assert changed.status_code == 400
    assert "frozen" in changed.json()["detail"]

    same = client.post(
        f"/sessions/{session_id}/continue",
        json={"generations": 1, "config": {"execution": "next_open"}},
    )
    assert same.status_code == 200, same.text
    assert _poll_job(client, same.json()["job_id"])["status"] == "done"

    finalize = client.post(
        f"/sessions/{session_id}/rounds/1/finalize", json={"confirm_repeat": False}
    )
    assert finalize.status_code == 200, finalize.text
    final = _poll_job(client, finalize.json()["job_id"])
    assert final["status"] == "done", final
    assert final["result"]["context"]["execution"] == "next_open"


def test_embargo_must_cover_horizon_plus_execution_delay(client) -> None:
    too_long = client.post(
        "/sessions",
        json={
            "name": "gap",
            "universe": "sp500-lite",
            "embargo": 5,
            "config": {**_SMALL, "horizon": 5, "execution": "next_close"},
        },
    )
    assert too_long.status_code == 400
    assert "embargo" in too_long.json()["detail"]

    legacy = client.post(
        "/sessions",
        json={
            "name": "gap-ok",
            "universe": "sp500-lite",
            "embargo": 5,
            "config": {**_SMALL, "horizon": 5},
        },
    )
    assert legacy.status_code == 200, legacy.text
    _poll_job(client, legacy.json()["job_id"])


def test_agent_may_not_change_execution_timing() -> None:
    assert "execution" in PROTECTED_KEYS
    result = validate_config_patch({"execution": "close"}, {"execution": "next_open"})
    assert not result.ok
    assert any("execution" in error for error in result.errors)
