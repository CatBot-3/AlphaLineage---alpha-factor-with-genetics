from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import alphalineage.api.app as api
from alphalineage.api.formula_sources import finish
from alphalineage.api.signal_workspace import (
    WorkspaceProgress,
    completed_session,
    portfolio,
    preparation,
    run,
    series,
)
from alphalineage.backtest.portfolio import QuantileLongShort
from alphalineage.core.tree import Node
from alphalineage.data.cache import ParquetCache
from alphalineage.data.universe import Membership, Universe


def prices(start="2023-01-01", end="2026-07-02", offset=0):
    dates = pd.bdate_range(start, end, name="date")
    close = 100 + offset + np.arange(len(dates)) * 0.01
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": close * 1000,
            "div_cash": 0.0,
            "split_factor": 1.0,
        },
        index=dates,
    )


@pytest.fixture
def market(monkeypatch):
    universe = Universe(
        "signals-test", [Membership(f"S{i}", pd.Timestamp("1995-01-01")) for i in range(10)]
    )
    monkeypatch.setattr(api, "_resolve_universe", lambda _: universe)
    monkeypatch.setattr(api, "_auto_sync_mode", lambda: "top_up")
    source = finish(
        "test:close",
        Node("rank", (Node("close"),)),
        name="Rank close",
        configuration={"execution": "next_open", "min_names": 5, "horizon": 1},
        provenance={"polarity": 1, "evidence": "validation", "validation_passed": False},
    )
    return universe, source


def execute(source, mode="cached", comparisons=None):
    request = api.SignalSnapshotRequest(
        source=source["id"], universe="signals-test", as_of="2026-07-03", data_mode=mode
    )
    resolved = [source, *(comparisons or [])]
    plan = preparation(request, resolved)
    return run(request, resolved, plan, None, WorkspaceProgress(total=10), lambda: False)


def test_holiday_early_close_and_before_close():
    assert completed_session("2026-07-03", datetime(2026, 7, 4, tzinfo=UTC)) == "2026-07-02"
    assert (
        completed_session("2025-11-28", datetime(2025, 11, 28, 17, 59, tzinfo=UTC)) == "2025-11-26"
    )
    assert (
        completed_session("2025-11-28", datetime(2025, 11, 28, 18, 1, tzinfo=UTC)) == "2025-11-28"
    )


def test_production_loader_repairs_missing_symbols_and_does_not_demand_1995(market, monkeypatch):
    universe, source = market
    requests = []

    class Provider:
        name = "mock"

        def get_prices(self, symbol, start, end):
            requests.append((symbol, start, end))
            return prices(start, end, offset=int(symbol[1:]))

    monkeypatch.setattr(api, "_price_provider", lambda: Provider())
    result = execute(source, "refresh")
    assert result["actionable"] and result["ranked_count"] == 10
    assert len(requests) == 10
    assert all(start > "2020-01-01" for _, start, _ in requests)
    assert universe.memberships[0].entry == pd.Timestamp("1995-01-01")
    assert result["validation_passed"] is False


def test_partial_floor_series_identity_cross_section_and_weights(market):
    _, source = market
    cache = ParquetCache()
    for i in range(9):
        cache.store(f"S{i}", prices(offset=i))
    result = execute(source)
    assert result["status"] == "partial" and result["actionable"]
    assert result["coverage"] == 0.9 and result["excluded"][0]["symbol"] == "S9"
    id_ = result["snapshot_id"]
    stock = series(id_, "S8")
    assert stock["signals"][0]["points"][-1]["value"] == result["rows"][0]["value"]
    assert stock["signals"][0]["points"][-1]["percentile"] == result["rows"][0]["percentile"]
    preview = portfolio(id_)
    expected = (
        QuantileLongShort(0.2)
        .weights(pd.DataFrame([{row["symbol"]: row["value"] for row in result["rows"]}]))
        .iloc[0]
    )
    assert {row["symbol"]: row["weight"] for row in preview["rows"]} == expected.to_dict()
    cache.path_for("S8").unlink()
    assert series(id_, "S8") == stock  # historical revision survives cache changes
    limited = execute(source)
    assert limited["status"] == "insufficient_coverage" and limited["rows"] == []
    assert len(limited["diagnostic_rows"]) == 8
    with pytest.raises(Exception, match="coverage"):
        portfolio(limited["snapshot_id"])


def test_wholly_stale_cache_never_moves_the_signal_date_back(market):
    _, source = market
    for i in range(10):
        ParquetCache().store(f"S{i}", prices(end="2026-06-30", offset=i))
    result = execute(source)
    assert result["expected_session"] == result["as_of"] == "2026-07-02"
    assert result["ranked_count"] == 0 and not result["actionable"]
    assert result["actual_signal_date"] is None
    assert all(item["reason"] == "stale_prices" for item in result["excluded"])


def test_cached_snapshot_becomes_stale_when_the_next_market_session_closes(market, monkeypatch):
    import alphalineage.api.signal_workspace as workspace

    _, source = market
    for i in range(10):
        ParquetCache().store(f"S{i}", prices(offset=i))
    result = execute(source)
    monkeypatch.setattr(workspace, "completed_session", lambda: "2026-07-06")
    with TestClient(api.app) as client:
        current = client.get(f"/signals/snapshots/{result['snapshot_id']}").json()
    assert current["stale"]
    assert current["current_expected_session"] == "2026-07-06"
    assert current["as_of"] == result["as_of"] == "2026-07-02"
    assert current["data_revision"] == result["data_revision"]
    assert "stale" not in workspace.load(result["snapshot_id"])["snapshot"]


def test_offline_navigation_and_refresh_never_contact_provider(market, monkeypatch):
    _, source = market
    for i in range(10):
        ParquetCache().store(f"S{i}", prices(offset=i))
    monkeypatch.setattr(api, "_auto_sync_mode", lambda: "off")
    monkeypatch.setattr(api, "_price_provider", lambda: pytest.fail("offline fetch"))
    result = execute(source, "refresh")
    assert result["offline"]


def test_save_is_idempotent_preserves_failed_validation_and_does_not_open_holdout(signal_panel):
    api.app.dependency_overrides[api.get_panel] = lambda: signal_panel[0]
    try:
        with TestClient(api.app) as client:
            from tests.test_signals import _poll

            created = client.post(
                "/sessions",
                json={
                    "name": "frozen",
                    "config": {
                        "population_size": 10,
                        "generations": 1,
                        "max_depth": 3,
                        "max_nodes": 12,
                    },
                },
            ).json()
            assert _poll(client, created["job_id"])["status"] == "done"
            sid = created["session_id"]
            before = client.get(f"/sessions/{sid}").json()
            source = f"round:{sid}:0"
            first = client.post("/formula-results", json={"source": source, "name": "Kept"})
            assert first.status_code == 200, first.text
            second = client.post(
                "/formula-results", json={"source": source, "name": "Another name"}
            )
            assert first.json()["id"] == second.json()["id"]
            resolved = client.post(
                "/signals/resolve", json={"source": f"result:{first.json()['id']}"}
            ).json()
            assert resolved["execution"] == "close" and resolved["polarity"] == 1
            assert resolved["missing_context"] == []
            after = client.get(f"/sessions/{sid}").json()
            assert before["test_reads"] == after["test_reads"]
    finally:
        api.app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "failure,expected",
    [("quota", "quota_limit"), ("empty", "unavailable_history"), ("failure", "provider_failure")],
)
def test_provider_limits_keep_explicit_partial_results(market, monkeypatch, failure, expected):
    from alphalineage.data.provider import QuotaExceededError

    _, source = market
    for i in range(9):
        ParquetCache().store(f"S{i}", prices(offset=i))

    class Provider:
        name = "mock"

        def get_prices(self, symbol, start, end):
            if failure == "quota":
                raise QuotaExceededError("limit", scope="daily", provider="mock")
            if failure == "empty":
                return prices().iloc[:0]
            raise RuntimeError("provider is unavailable")

    monkeypatch.setattr(api, "_price_provider", lambda: Provider())
    result = execute(source, "refresh")
    assert result["status"] == "partial"
    assert result["excluded"][0]["reason"] == expected


def test_aliases_and_ipo_history_limits_are_preserved(market, monkeypatch):
    universe, source = market
    universe.aliases["S9"] = "VENDOR9"
    for i in range(9):
        ParquetCache().store(f"S{i}", prices(offset=i))
    requested = []

    class Provider:
        name = "mock"

        def get_prices(self, symbol, start, end):
            requested.append(symbol)
            return prices("2026-07-01", end, offset=9)

    monkeypatch.setattr(api, "_price_provider", lambda: Provider())
    source["tree"] = Node("ts_mean", (Node("close"), Node("window", value=10)))
    result = execute(source, "refresh")
    assert "VENDOR9" in requested
    assert result["status"] == "partial"
    assert result["excluded"][0]["reason"] == "insufficient_warmup"
    assert universe.memberships[-1].entry == pd.Timestamp("1995-01-01")


def test_comparisons_keep_universe_context_and_recursive_origin(market):
    _, source = market
    for i in range(10):
        ParquetCache().store(f"S{i}", prices(offset=i))
    comparison = finish(
        "test:recursive",
        Node("ts_ema", (Node("rank", (Node("close"),)), Node("window", value=20))),
        name="Recursive rank",
        configuration={"execution": "next_open", "min_names": 5},
        provenance={"history_start": "2023-01-02", "polarity": 1},
    )
    result = execute(source, comparisons=[comparison])
    stock = series(result["snapshot_id"], "S9")
    assert stock["signals"][0]["points"][-1]["value"] == 1
    assert stock["signals"][1]["points"][-1]["value"] == pytest.approx(1)
    assert stock["signals"][1]["points"][-1]["rank"] == 1
    refreshed = execute(source, comparisons=[comparison])
    assert stock["signals"] == series(refreshed["snapshot_id"], "S9")["signals"]
    comparison["provenance"]["history_start"] = comparison["history_start"] = "2020-01-01"
    assert execute(source, comparisons=[comparison])["approximate"]


def test_pinned_portfolio_export_matches_preview(market):
    import csv
    import io

    from alphalineage.api.signal_workspace import export

    _, source = market
    source["primary_strategy"] = {"id": "pinned", "scheme": "quantile_ls", "quantile": 0.3}
    for i in range(10):
        ParquetCache().store(f"S{i}", prices(offset=i))
    result = execute(source)
    preview = portfolio(result["snapshot_id"])
    rows = list(
        csv.DictReader(io.StringIO(export(result["snapshot_id"], "portfolio", preview["strategy"])))
    )
    assert {row["symbol"]: float(row["weight"]) for row in rows} == {
        row["symbol"]: row["weight"] for row in preview["rows"]
    }
    with TestClient(api.app) as client:
        response = client.get(f"/signals/snapshots/{result['snapshot_id']}/export?kind=portfolio")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    exported = list(csv.DictReader(io.StringIO(response.text)))
    assert {row["symbol"]: float(row["weight"]) for row in exported} == {
        row["symbol"]: row["weight"] for row in preview["rows"]
    }
    assert all(row["snapshot_id"] == result["snapshot_id"] for row in exported)
    assert all(row["execution"] == source["execution"] for row in exported)
    assert all(float(row["quantile"]) == 0.3 for row in exported)
    assert all(
        row["expression_fingerprint"] == source["expression_fingerprint"] for row in exported
    )


def test_comparison_history_does_not_reinitialize_primary_recursive_formula(market):
    _, original = market
    for i in range(10):
        ParquetCache().store(f"S{i}", prices(start="2020-01-01", offset=i))
    source = finish(
        "test:origin",
        Node("ts_cumsum", (Node("close"),)),
        name="Expanding",
        configuration={"execution": "next_open", "min_names": 5},
        provenance={"history_start": "2024-01-01", "polarity": 1},
    )
    comparison = finish(
        "test:old-origin",
        source["tree"],
        name="Older origin",
        configuration=source["configuration"],
        provenance={"history_start": "2020-01-01", "polarity": 1},
    )
    alone = execute(source)
    together = execute(source, comparisons=[comparison])
    assert alone["rows"] == together["rows"]
    points = series(together["snapshot_id"], "S9")["signals"]
    assert points[0]["points"][-1]["value"] != points[1]["points"][-1]["value"]


def test_invalid_membership_returns_a_structured_recovery(market, monkeypatch):
    from alphalineage.api.errors import ActionError

    _, source = market

    def invalid(_):
        raise ValueError("membership exit must be after entry for S9")

    monkeypatch.setattr(api, "_resolve_universe", invalid)
    with pytest.raises(ActionError) as raised:
        execute(source)
    assert raised.value.payload["code"] == "invalid_membership_metadata"
    assert "Build & Data" in raised.value.payload["action"]


def test_balanced_session_freezes_references_across_save_and_continue(signal_panel):
    from tests.test_signals import _poll

    api.app.dependency_overrides[api.get_panel] = lambda: signal_panel[0]
    try:
        with TestClient(api.app) as client:
            created = client.post(
                "/sessions",
                json={
                    "name": "Novelty frozen",
                    "config": {
                        "population_size": 10,
                        "generations": 1,
                        "max_depth": 3,
                        "max_nodes": 12,
                        "novelty_mode": "balanced",
                    },
                },
            ).json()
            assert _poll(client, created["job_id"])["status"] == "done"
            sid = created["session_id"]
            before = api.sessions.load_session(sid)
            assert before["novelty"]["references"]
            saved = client.post("/formula-results", json={"source": f"round:{sid}:0"})
            assert saved.status_code == 200, saved.text
            lineage = api.sessions.lineage_for_round(sid, 0)
            node = lineage["nodes"][0]["id"]
            kept = client.post("/formula-results", json={"source": f"lineage:{sid}:0:{node}"})
            assert kept.status_code == 200, kept.text
            assert kept.json()["provenance"]["evidence"] == "training"
            rejected = client.post(
                f"/sessions/{sid}/continue",
                json={"generations": 1, "config": {"novelty_mode": "off"}},
            )
            assert rejected.status_code == 400
            continued = client.post(f"/sessions/{sid}/continue", json={"generations": 1})
            assert continued.status_code == 200, continued.text
            assert _poll(client, continued.json()["job_id"])["status"] == "done"
            after = api.sessions.load_session(sid)
            assert before["novelty"] == after["novelty"]
            assert before["test_reads"] == after["test_reads"] == 0
    finally:
        api.app.dependency_overrides.clear()
