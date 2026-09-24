"""Signals: ranking a universe with a frozen formula on the latest cached bar."""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from alphalineage.api.app import app, get_panel
from alphalineage.core import extensions
from alphalineage.core.panel import Panel
from alphalineage.library.signals import (
    LOOKBACK_MARGIN,
    UNBOUNDED_HISTORY_BARS,
    history_bars_needed,
    snapshot,
    trim_panel,
)


def _panel(days: int = 140, names: int = 8, *, truncate: dict[int, int] | None = None) -> Panel:
    """A clean panel, optionally with some symbols' prices stopping early."""
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2026-01-02", periods=days)
    symbols = [f"S{index}" for index in range(names)]
    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, (days, names)), axis=0),
        index=dates,
        columns=symbols,
    )
    for column, missing in (truncate or {}).items():
        close.iloc[-missing:, column] = np.nan
    open_ = close.shift(1).bfill()
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    volume = pd.DataFrame(1e6, index=dates, columns=symbols)
    return Panel.from_prices(open=open_, high=high, low=low, close=close, volume=volume)


def test_snapshot_ranks_by_value_and_reports_the_bar_it_used() -> None:
    panel = _panel()
    factor = panel["close"].rolling(10).mean()
    report = snapshot(factor, panel, min_names=3, execution="next_open")

    dates = pd.DatetimeIndex(panel.dates)
    assert report["as_of"] == dates[-1].date().isoformat()
    assert report["previous_as_of"] == dates[-2].date().isoformat()
    assert report["bars_behind_panel"] == 0
    # A saved tree already carries its training direction, so rank 1 is the largest value.
    values = [row["value"] for row in report["rows"]]
    assert values == sorted(values, reverse=True)
    assert [row["rank"] for row in report["rows"]] == list(range(1, len(values) + 1))
    assert report["rows"][0]["percentile"] == pytest.approx(100.0)
    assert report["rows"][-1]["percentile"] == pytest.approx(100.0 / len(values), rel=1e-6)
    assert report["execution"] == "next_open" and report["execution_delay"] == 1
    assert report["ranked_count"] == 8 and report["excluded_count"] == 0

    expected = factor.iloc[-1].sort_values(ascending=False)
    assert [row["symbol"] for row in report["rows"]] == list(expected.index)
    assert report["rows"][0]["close"] == pytest.approx(panel["close"].iloc[-1][expected.index[0]])


def test_rank_change_is_signed_so_a_climb_reads_positive() -> None:
    dates = pd.bdate_range("2026-01-02", periods=3)
    symbols = ["A", "B", "C"]
    close = pd.DataFrame(
        [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], dates, symbols
    )
    panel = Panel.from_prices(
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=pd.DataFrame(1e6, index=dates, columns=symbols),
    )
    # C is last yesterday and first today; A does the reverse.
    factor = pd.DataFrame([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [1.0, 2.0, 3.0]], dates, symbols)
    rows = {row["symbol"]: row for row in snapshot(factor, panel, min_names=2)["rows"]}
    assert rows["C"]["rank"] == 1 and rows["C"]["previous_rank"] == 3
    assert rows["C"]["rank_change"] == 2
    assert rows["A"]["rank_change"] == -2
    assert rows["B"]["rank_change"] == 0


def test_stale_and_warming_up_symbols_are_excluded_with_their_own_reason() -> None:
    # S2's prices stop five bars early; S5's stop one bar early.
    panel = _panel(truncate={2: 5, 5: 1})
    factor = panel["close"].rolling(10).mean()
    report = snapshot(factor, panel, min_names=3)
    excluded = {item["symbol"]: item for item in report["excluded"]}

    assert set(excluded) == {"S2", "S5"}
    assert excluded["S2"]["reason"] == "stale_prices" and excluded["S2"]["stale_bars"] == 5
    assert excluded["S5"]["stale_bars"] == 1
    assert excluded["S2"]["last_price_date"] < excluded["S5"]["last_price_date"]
    # Excluded, never ranked on an older number: a stale rank is invisible on screen.
    assert "S2" not in {row["symbol"] for row in report["rows"]}
    assert report["ranked_count"] == 6

    # A formula that needs more history than a symbol has is a different problem and says so.
    short = panel["close"].rolling(10).mean().copy()
    short.iloc[:, 0] = np.nan
    warming = {item["symbol"]: item for item in snapshot(short, panel, min_names=3)["excluded"]}
    assert warming["S0"]["reason"] == "warming_up" and warming["S0"]["stale_bars"] == 0


def test_a_thin_cross_section_is_skipped_rather_than_ranked() -> None:
    panel = _panel(days=40, names=6)
    factor = panel["close"].rolling(10).mean().copy()
    # Only two symbols report on the final bar; min_names=4 must look further back.
    factor.iloc[-1, 2:] = np.nan
    report = snapshot(factor, panel, min_names=4)
    assert report["as_of"] == pd.DatetimeIndex(panel.dates)[-2].date().isoformat()
    assert report["bars_behind_panel"] == 1

    empty = pd.DataFrame(np.nan, index=panel.dates, columns=list(panel.symbols))
    with pytest.raises(ValueError, match="at least 4 symbols"):
        snapshot(empty, panel, min_names=4)


def test_history_is_trimmed_to_the_formula_lookback_without_changing_the_value() -> None:
    panel = _panel(days=400)
    assert history_bars_needed(20, unbounded=False) == 20 + LOOKBACK_MARGIN
    assert history_bars_needed(20, unbounded=True) == UNBOUNDED_HISTORY_BARS

    trimmed = trim_panel(panel, 60)
    assert len(pd.DatetimeIndex(trimmed.dates)) == 60
    assert trim_panel(panel, 10_000) is panel

    full = snapshot(panel["close"].rolling(10).mean(), panel, min_names=3)
    short = snapshot(trimmed["close"].rolling(10).mean(), trimmed, min_names=3)
    assert full["as_of"] == short["as_of"]
    assert [row["symbol"] for row in full["rows"]] == [row["symbol"] for row in short["rows"]]
    assert full["rows"][0]["value"] == pytest.approx(short["rows"][0]["value"])


# --- API --------------------------------------------------------------------------------------
_SMALL = {"population_size": 16, "generations": 2, "max_depth": 4, "max_nodes": 20, "seed": 0}


@pytest.fixture
def client(signal_panel) -> Iterator[TestClient]:
    panel, _ = signal_panel
    app.dependency_overrides[get_panel] = lambda: panel
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    extensions.clear_user_operators()


def _poll(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/runs/{job_id}").json()
        if payload["status"] in ("done", "stopped", "failed"):
            return payload
        time.sleep(0.1)
    return payload


def test_saved_results_and_rounds_are_both_usable_sources(client) -> None:
    assert all(item["kind"] == "formula" for item in client.get("/signals/sources").json())

    saved = client.post(
        "/factors",
        json={"name": "Kept volume", "tree": {"name": "volume"}, "metrics": {"ic": 0.1}},
    )
    assert saved.status_code == 200, saved.text
    created = client.post(
        "/sessions",
        json={
            "name": "signals",
            "universe": "sp500-lite",
            "config": {**_SMALL, "execution": "next_open"},
        },
    ).json()
    assert _poll(client, created["job_id"])["status"] == "done"

    sources = {item["id"]: item for item in client.get("/signals/sources").json()}
    result_id = f"result:{saved.json()['id']}"
    round_id = f"round:{created['session_id']}:0"
    assert result_id in sources and round_id in sources
    assert sources[result_id]["kind"] == "saved_result"
    # A round that has never been finalized says so rather than implying a holdout read.
    assert sources[round_id]["evidence"] == "validation"
    assert sources[round_id]["execution"] == "next_open"

    assert client.post("/signals/snapshot", json={"source": "result:nope"}).status_code == 404
    assert client.post("/signals/snapshot", json={"source": "nonsense"}).status_code == 400
    assert client.post("/signals/snapshot", json={"source": "round:missing:0"}).status_code == 404


def test_snapshot_ranks_a_universe_and_never_touches_the_session_split(client) -> None:
    created = client.post(
        "/sessions", json={"name": "ranked", "universe": "sp500-lite", "config": _SMALL}
    ).json()
    assert _poll(client, created["job_id"])["status"] == "done"
    session_id = created["session_id"]
    before = client.get(f"/sessions/{session_id}").json()

    started = client.post(
        "/signals/snapshot", json={"source": f"round:{session_id}:0", "universe": "sp500-lite"}
    )
    assert started.status_code == 200, started.text
    final = _poll(client, started.json()["job_id"])
    assert final["status"] == "done", final
    report = final["result"]

    assert report["signals_version"] == 1
    assert report["universe"] == "sp500-lite"
    assert report["source"] == f"round:{session_id}:0"
    assert report["ranked_count"] >= 1
    assert report["rows"][0]["rank"] == 1
    assert report["history_bars"] <= report["ranked_count"] * 10_000  # trimmed, not unbounded
    assert report["disclaimer"]

    # Looking at today's ranking is not a holdout read and changes no session accounting.
    after = client.get(f"/sessions/{session_id}").json()
    assert after["test_reads"] == before["test_reads"]
    assert after["cumulative_trials"] == before["cumulative_trials"]


def test_a_second_request_for_the_same_snapshot_reuses_the_running_job(client) -> None:
    created = client.post(
        "/sessions", json={"name": "reuse", "universe": "sp500-lite", "config": _SMALL}
    ).json()
    assert _poll(client, created["job_id"])["status"] == "done"
    body = {"source": f"round:{created['session_id']}:0", "universe": "sp500-lite"}

    first = client.post("/signals/snapshot", json=body).json()
    second = client.post("/signals/snapshot", json=body).json()
    if second["reused"]:
        assert second["job_id"] == first["job_id"]
    assert _poll(client, first["job_id"])["status"] == "done"
