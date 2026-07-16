"""Phase 5 acceptance + supporting tests for the HTTP API (tests/test_api.py)."""

from __future__ import annotations

import threading
import time

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import alphalineage.api.app as api_app
from alphalineage.api.app import app, get_panel
from alphalineage.api.jobs import JobStore
from alphalineage.core import extensions


@pytest.fixture
def client(synthetic_panel):
    app.dependency_overrides[get_panel] = lambda: synthetic_panel
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _clean_operators():
    yield
    extensions.clear_user_operators()


@pytest.fixture(autouse=True)
def _clean_verified_symbols():
    yield
    api_app._verified_symbols.clear()


def _poll(client: TestClient, job_id: str, *, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/runs/{job_id}").json()
        if payload["status"] in ("done", "stopped", "failed"):
            return payload
        time.sleep(0.2)
    return payload


def _poll_data_sync(client: TestClient, job_id: str, *, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/data/sync/{job_id}").json()
        if payload["status"] in ("done", "failed"):
            return payload
        time.sleep(0.02)
    return payload


def _poll_membership_sync(client: TestClient, job_id: str, *, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/universes/sync-dates/{job_id}").json()
        if payload["status"] in ("done", "failed"):
            return payload
        time.sleep(0.02)
    return payload


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_training_capabilities_are_device_relative_and_python_fallback_is_visible(
    client, monkeypatch
):
    monkeypatch.setattr(api_app.cpp, "available", lambda: False)
    response = client.get("/training/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_profile"] == "auto"
    assert payload["worker_capacity"] <= 32
    assert payload["profiles"]["auto"]["percent"] == 50
    auto = payload["profiles"]["auto"]
    assert auto["requested_workers"] >= auto["effective_workers"]
    assert auto["effective_workers"] == auto["workers"] == 1
    assert "one Python worker" in payload["fallback_reason"]


def test_non_native_scoring_method_resolves_to_visible_single_worker(monkeypatch):
    monkeypatch.setattr(api_app, "_acceleration_status", lambda: (True, None))
    monkeypatch.setattr(
        api_app.cpp, "supports_native_scoring", lambda method: method == "spearman"
    )

    resolved = api_app._resolve_training_resources(
        api_app.TrainingResourcesRequest(profile="auto"), ic_method="pearson"
    )

    assert resolved.requested_workers > 1
    assert resolved.effective_workers == resolved.workers == 1
    assert "Pearson" in str(resolved.fallback_reason)


@pytest.mark.parametrize(
    "resources",
    [
        {"profile": "custom"},
        {"profile": "auto", "cpu_budget_percent": 50},
    ],
)
def test_training_resource_request_rejects_ambiguous_percent(client, resources):
    response = client.post(
        "/runs",
        json={
            "config": {
                "population_size": 8,
                "generations": 1,
                "max_depth": 4,
                "max_nodes": 20,
            },
            "resources": resources,
        },
    )
    assert response.status_code == 400


def test_panel_dependency_is_lazy_until_request_universe_is_known():
    assert get_panel() is None


def test_cached_universe_rejects_any_missing_historical_member(synthetic_prices):
    from alphalineage.data.cache import ParquetCache
    from alphalineage.data.universe import Membership, Universe

    cache = ParquetCache()
    cache.store("LATE", api_app.schema.normalize(synthetic_prices))
    universe = Universe(
        "cache-completeness",
        [
            Membership("EARLY", pd.Timestamp("2000-01-01"), pd.Timestamp("2010-01-01")),
            Membership("LATE", pd.Timestamp("2020-01-01")),
        ],
    )

    with pytest.raises(HTTPException, match="EARLY"):
        api_app._panel_from_universe_cache(universe, pd.Timestamp("2025-01-01"))


def test_universe_cache_coverage_validates_membership_period_edges(synthetic_prices):
    from alphalineage.data.cache import ParquetCache
    from alphalineage.data.universe import Membership, Universe

    prices = api_app.schema.normalize(synthetic_prices)
    cache = ParquetCache()
    cache.store("FULL", prices)
    cache.store("LATE", prices.iloc[12:])
    cache.store("STALE", prices.iloc[:10])
    cutoff = prices.index.max()
    universe = Universe(
        "cache-edges",
        [
            Membership("FULL", prices.index.min()),
            Membership("LATE", prices.index.min()),
            Membership("STALE", prices.index.min()),
        ],
    )

    complete = api_app._cache_coverage_for(
        Universe("cache-full", [Membership("FULL", prices.index.min())]),
        cutoff,
        cache=cache,
    )
    assert complete["complete"] is True

    coverage = api_app._cache_coverage_for(universe, cutoff, cache=cache)
    assert coverage["complete"] is False
    assert coverage["late_start_symbols"] == ["LATE"]
    assert coverage["stale_symbols"] == ["STALE"]
    assert coverage["incomplete_symbols"] == ["LATE", "STALE"]
    assert coverage["symbol_coverage"]["FULL"]["issues"] == []

    with pytest.raises(HTTPException, match="incomplete membership-period"):
        api_app._panel_from_universe_cache(universe, cutoff)


def test_job_lifecycle(client):
    config = {"population_size": 16, "generations": 2, "max_depth": 4, "max_nodes": 20, "seed": 0}
    submit = client.post("/runs", json={"config": config})
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    final = _poll(client, job_id)
    assert final["status"] == "done", final
    assert client.post(f"/runs/{job_id}/stop").json() == {"stopping": False}
    result = final["result"]
    assert "best_factor" in result
    assert "report" in result and "deflated_sharpe" in result["report"]

    lineage = client.get(f"/runs/{job_id}/lineage").json()
    assert "nodes" in lineage and len(lineage["nodes"]) > 0

    saved = client.get(f"/workspaces/run-{job_id}")
    assert saved.status_code == 200
    assert saved.json()["run"]["best_factor"] == result["best_factor"]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"populaton_size": 16}, "unknown GP config"),
        ({"population_size": 5_001}, "population_size"),
        ({"horizon": 6}, "embargo"),
        ({"min_names": 7}, "symbol count"),
    ],
)
def test_run_config_errors_are_rejected_before_queueing(client, config, message):
    response = client.post("/runs", json={"config": config})
    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_request_dates_are_validated_before_work_is_queued(client):
    assert client.post("/runs", json={"as_of": "2999-01-01"}).status_code == 400
    assert (
        client.get(
            "/data/coverage",
            params={"symbols": "AAPL", "start": "not-a-date"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/data/sync",
            json={"symbols": ["AAPL"], "start": "2021-01-02", "end": "2021-01-01"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/universes/sync-dates",
            json={"symbols": ["AAPL"], "expected_start": "not-a-date"},
        ).status_code
        == 400
    )


def test_unknown_job_is_404(client):
    assert client.get("/runs/does-not-exist").status_code == 404


def test_list_primitives(client):
    primitives = client.get("/primitives").json()
    names = {p["name"] for p in primitives}
    assert {"ts_mean", "rank", "close"} <= names
    assert all("arg_types" in p and "out_type" in p for p in primitives)


def test_register_operator_and_reject_unknown(client):
    spec = {
        "name": "spread_api",
        "arg_types": ["series", "series"],
        "out_type": "series",
        "body": {
            "name": "sub",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    assert client.post("/operators", json=spec).json()["name"] == "spread_api"
    assert any(o["name"] == "spread_api" for o in client.get("/operators").json())

    # a body naming a non-primitive is rejected (400), never executed
    bad = {
        "name": "evil",
        "arg_types": ["series"],
        "out_type": "series",
        "body": {"name": "__import__"},
    }
    assert client.post("/operators", json=bad).status_code == 400


def test_formula_save_reload_delete_and_validation(client):
    spec = {
        "name": "moving_average_api",
        "display_name": "Moving average API",
        "description": "Trailing mean with a caller-provided window.",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    saved = client.post("/formulas", json=spec)
    assert saved.status_code == 200
    assert saved.json()["registered"] is True
    assert client.post("/formulas", json=spec).status_code == 400

    extensions.clear_user_operators()
    restored = client.get("/formulas")
    assert restored.status_code == 200
    assert restored.json()[0]["registered"] is True
    assert any(item["name"] == spec["name"] for item in client.get("/primitives").json())

    bad = {
        **spec,
        "name": "unknown_formula",
        "body": {"name": "not_a_primitive"},
    }
    assert client.post("/formulas", json=bad).status_code == 400

    assert client.delete(f"/formulas/{spec['name']}").status_code == 200
    assert all(
        item["origin"] == "catalog_formula" for item in client.get("/formulas").json()
    )


def test_active_run_pins_formula_revisions_against_delete(client, monkeypatch):
    formula = {
        "name": "active_run_formula",
        "display_name": "Active run formula",
        "out_type": "series",
        "body": {"name": "rank", "children": [{"name": "close"}]},
    }
    saved = client.post("/formulas", json=formula)
    assert saved.status_code == 200
    runtime_name = saved.json()["runtime_name"]

    started = threading.Event()
    release = threading.Event()

    def blocked_search(*args, **kwargs):
        assert runtime_name in kwargs["allowed_operators"]
        started.set()
        assert release.wait(5)
        return {"termination_reason": "completed"}

    monkeypatch.setattr(api_app, "run_search", blocked_search)
    try:
        submit = client.post(
            "/runs",
            json={"config": {"population_size": 6, "generations": 1}},
        )
        assert submit.status_code == 200
        job_id = submit.json()["job_id"]
        assert started.wait(2)

        impact = client.get(f"/formulas/{formula['name']}").json()["impact"]
        assert impact["runs"] == [job_id]
        blocked = client.delete(f"/formulas/{formula['name']}")
        assert blocked.status_code == 400
        assert blocked.json()["detail"]["runs"] == [job_id]
    finally:
        release.set()

    final = _poll(client, job_id)
    assert final["status"] == "done"
    assert final["result"]["formula_revisions"] == [
        {"name": formula["name"], "revision": 1, "runtime_name": runtime_name}
    ]
    workspace = client.get(f"/workspaces/run-{job_id}").json()
    assert workspace["run"]["formula_revisions"] == final["result"]["formula_revisions"]
    assert client.delete(f"/formulas/{formula['name']}").status_code == 200


def test_active_run_pins_formula_used_by_request_operator(client, monkeypatch):
    formula = {
        "name": "request_dependency",
        "out_type": "series",
        "body": {"name": "rank", "children": [{"name": "close"}]},
        "category": "condition",
    }
    runtime_name = client.post("/formulas", json=formula).json()["runtime_name"]
    operator = {
        "name": "request_wrapper",
        "arg_types": [],
        "out_type": "series",
        "body": {"name": runtime_name},
    }
    started = threading.Event()
    release = threading.Event()

    def blocked_search(*args, **kwargs):
        assert operator["name"] in kwargs["allowed_operators"]
        started.set()
        assert release.wait(5)
        return {"termination_reason": "completed"}

    monkeypatch.setattr(api_app, "run_search", blocked_search)
    try:
        submit = client.post(
            "/runs",
            json={
                "config": {
                    "population_size": 6,
                    "generations": 1,
                    "enabled_categories": ["custom"],
                },
                "operators": [operator],
            },
        )
        assert submit.status_code == 200
        job_id = submit.json()["job_id"]
        assert started.wait(2)
        blocked = client.delete(f"/formulas/{formula['name']}")
        assert blocked.status_code == 400
        assert blocked.json()["detail"]["runs"] == [job_id]
    finally:
        release.set()

    final = _poll(client, job_id)
    assert final["status"] == "done"
    assert {item["runtime_name"] for item in final["result"]["formula_revisions"]} == {
        runtime_name
    }
    assert client.delete(f"/formulas/{formula['name']}").status_code == 200


def test_define_point_in_time_universe(client):
    spec = {
        "name": "my-universe",
        "memberships": [
            {"symbol": "AAA", "entry": "2020-01-01"},
            {"symbol": "BBB", "entry": "2021-01-01", "exit": "2022-01-01"},
        ],
    }
    response = client.post("/universes", json=spec)
    assert response.status_code == 200
    assert set(response.json()["symbols"]) == {"AAA", "BBB"}


def test_universe_persistence_survives_memory_clear(client):
    spec = {
        "name": "persisted-universe",
        "memberships": [
            {"symbol": "AAA", "entry": "2020-01-01"},
            {"symbol": "BBB", "entry": "2021-01-01", "exit": "2022-01-01"},
        ],
    }
    assert client.post("/universes", json=spec).status_code == 200

    api_app._universes.clear()
    universes = client.get("/universes").json()
    restored = next(u for u in universes if u["name"] == "persisted-universe")

    assert restored["source"] == "custom"
    assert set(restored["symbols"]) == {"AAA", "BBB"}


def test_universe_get_update_delete_and_sample_protection(client):
    original = {
        "name": "editable-universe",
        "memberships": [{"symbol": "AAA", "entry": "2020-01-01"}],
    }
    assert client.post("/universes", json=original).status_code == 200

    loaded = client.get("/universes/editable-universe")
    assert loaded.status_code == 200
    assert loaded.json()["source"] == "custom"

    updated = {
        "name": "editable-universe",
        "memberships": [
            {"symbol": "AAA", "entry": "2020-01-01"},
            {"symbol": "BBB", "entry": "2021-01-01", "exit": "2024-01-01"},
        ],
    }
    response = client.put("/universes/editable-universe", json=updated)
    assert response.status_code == 200
    assert set(response.json()["symbols"]) == {"AAA", "BBB"}

    assert client.delete("/universes/sp500-lite").status_code == 400
    assert (
        client.post(
            "/universes",
            json={"name": "sp500-lite", "memberships": original["memberships"]},
        ).status_code
        == 400
    )

    assert client.delete("/universes/editable-universe").status_code == 200
    assert client.get("/universes/editable-universe").status_code == 404


def test_point_in_time_panel_masks_all_fields_and_keeps_historical_members():
    from alphalineage.core.panel import Panel
    from alphalineage.data.universe import Membership, Universe

    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.DataFrame(
        {"OLD": [10, 11, 12, 13, 14, 15], "NEW": [20, 21, 22, 23, 24, 25]},
        index=dates,
        dtype=float,
    )
    panel = Panel.from_prices(
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=close * 100,
    )
    universe = Universe(
        "pit-mask-test",
        [
            Membership("OLD", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-04")),
            Membership("NEW", pd.Timestamp("2020-01-03")),
        ],
    )
    api_app._universes[universe.name] = universe
    try:
        resolved = api_app._panel_for_universe(universe.name, "2020-01-05", panel)
    finally:
        api_app._universes.pop(universe.name, None)

    assert resolved.dates.max() == pd.Timestamp("2020-01-05")
    assert set(resolved.symbols) == {"OLD", "NEW"}  # exited member was not dropped
    for field in resolved.fields.values():
        assert pd.isna(field.loc["2020-01-04", "OLD"])
        assert pd.isna(field.loc["2020-01-02", "NEW"])
    assert pd.isna(resolved["returns"].loc["2020-01-03", "NEW"])
    assert resolved["returns"].loc["2020-01-04", "NEW"] == pytest.approx(23 / 22 - 1)


def test_universe_validation_unknown_run_and_honest_presets(client):
    empty = client.post("/universes", json={"name": "empty-u", "memberships": []})
    assert empty.status_code == 400
    backwards = client.post(
        "/universes",
        json={
            "name": "backwards-u",
            "memberships": [{"symbol": "AAA", "entry": "2021-01-02", "exit": "2021-01-01"}],
        },
    )
    assert backwards.status_code == 400
    overlapping = client.post(
        "/universes",
        json={
            "name": "overlap-u",
            "memberships": [
                {"symbol": "AAA", "entry": "2020-01-01", "exit": "2021-01-01"},
                {"symbol": "AAA", "entry": "2020-06-01"},
            ],
        },
    )
    assert overlapping.status_code == 400
    assert client.post("/runs", json={"universe": "does-not-exist"}).status_code == 400

    presets = {item["id"]: item for item in client.get("/universe-presets").json()}
    assert presets["sp500"]["status"] == "bundled_snapshot"
    assert presets["sp500"]["available"] is True
    assert presets["sp500"]["snapshot_available"] is True
    assert presets["sp500"]["snapshot_universe"] == "builtin-sp500-current"
    assert presets["sp500"]["pit_import_name"] == "sp500"
    assert presets["sp500"]["pit_available"] is False
    assert presets["sp500"]["research_ready"] is False
    assert presets["sp500"]["definition"]["id"] == "builtin-sp500-current"
    assert presets["sp500"]["definition"]["snapshot_date"] == "2026-07-15"
    assert len(presets["sp500"]["fingerprint"]) == 64
    assert presets["sp500"]["provenance_detail"]["provider"].startswith("Wikipedia:")
    assert presets["sp500"]["provenance_detail"]["attribution"] == "Wikipedia contributors"
    assert presets["sp500"]["provenance_detail"]["license"] == "CC BY-SA 4.0"
    assert presets["sp500"]["readiness"]["research_ready"] is False
    assert "memberships" not in presets["sp500"]
    bundled = client.get("/universes/sp500-lite").json()
    assert bundled["integrity"]["membership_history"] == "illustrative"
    assert bundled["integrity"]["research_ready"] is False


def test_symbol_search_and_validation(client, monkeypatch, synthetic_prices):
    candidates = [
        api_app.SymbolCandidate(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="Nasdaq",
            quote_type="Equity",
            currency="USD",
        ),
        api_app.SymbolCandidate(
            symbol="APC.F",
            name="Apple Inc.",
            exchange="Frankfurt",
            quote_type="Equity",
            currency="EUR",
        ),
    ]
    monkeypatch.setattr(api_app, "_search_symbol_candidates", lambda query, limit=8: candidates)
    prices = api_app.schema.normalize(synthetic_prices)

    class FakeProvider:
        name = "fake"

        def get_prices(self, symbol, start=None, end=None):
            if symbol == "BAD":
                raise RuntimeError("unknown symbol")
            return prices

    monkeypatch.setattr(api_app, "_price_provider", FakeProvider)

    search = client.get("/symbols/search", params={"query": "AAPL"})
    assert search.status_code == 200
    assert [item["symbol"] for item in search.json()] == ["AAPL", "APC.F"]

    valid = client.post("/symbols/validate", json={"symbol": "AAPL", "start": "2020-01-01"})
    assert valid.status_code == 200
    assert valid.json()["valid"] is True
    assert valid.json()["provider"] == "fake"

    invalid = client.post("/symbols/validate", json={"symbol": "BAD"})
    assert invalid.status_code == 200
    assert invalid.json()["valid"] is False
    assert "unknown symbol" in invalid.json()["error"]


def test_symbol_validation_is_cached_per_range(client, monkeypatch, synthetic_prices):
    prices = api_app.schema.normalize(synthetic_prices)

    class CountingProvider:
        name = "fake"

        def __init__(self):
            self.calls = 0

        def get_prices(self, symbol, start=None, end=None):
            self.calls += 1
            return prices

    provider = CountingProvider()
    monkeypatch.setattr(api_app, "_price_provider", lambda: provider)

    first = client.post("/symbols/validate", json={"symbol": "AAPL", "start": "2020-01-01"})
    assert first.json()["valid"] is True
    assert first.json()["cached"] is False
    assert provider.calls == 1

    second = client.post("/symbols/validate", json={"symbol": "AAPL", "start": "2020-01-01"})
    assert second.json()["cached"] is True
    assert provider.calls == 1  # no extra provider hit - served from the verified cache

    different_range = client.post(
        "/symbols/validate", json={"symbol": "AAPL", "start": "2020-02-01"}
    )
    assert different_range.json()["cached"] is False
    assert provider.calls == 2

    forced = client.post(
        "/symbols/validate", json={"symbol": "AAPL", "start": "2020-01-01", "force": True}
    )
    assert forced.json()["cached"] is False
    assert provider.calls == 3


def test_incremental_and_refresh_data_sync(client, monkeypatch, synthetic_prices):
    prices = api_app.schema.normalize(synthetic_prices)

    class FakeProvider:
        name = "fake"

        def __init__(self):
            self.calls = []

        def get_prices(self, symbol, start=None, end=None):
            self.calls.append((symbol, start, end))
            frame = prices
            if start:
                frame = frame[frame.index >= pd.Timestamp(start)]
            if end:
                frame = frame[frame.index < pd.Timestamp(end)]
            return frame.copy()

    provider = FakeProvider()
    monkeypatch.setattr(api_app, "_price_provider", lambda: provider)
    cache = api_app.ParquetCache()
    cache.store("AAPL", prices.loc["2020-01-06":"2020-01-10"])

    coverage = client.get(
        "/data/coverage",
        params={"symbols": "AAPL", "start": "2020-01-01", "end": "2020-01-20"},
    )
    assert coverage.json()[0]["needs_sync"] is True

    started = client.post(
        "/data/sync",
        json={
            "symbols": ["AAPL"],
            "start": "2020-01-01",
            "end": "2020-01-20",
            "mode": "incremental",
        },
    )
    final = _poll_data_sync(client, started.json()["job_id"])
    assert final["status"] == "done", final
    assert final["progress"] == {"done": 1, "total": 1, "current_symbol": "AAPL"}
    assert provider.calls == [
        ("AAPL", "2020-01-01", "2020-01-06"),
        ("AAPL", "2020-01-11", "2020-01-20"),
    ]
    merged = cache.load("AAPL")
    assert not merged.index.duplicated().any()
    assert merged.index.min() == pd.Timestamp("2020-01-01")

    provider.calls.clear()
    refreshed = client.post(
        "/data/sync",
        json={
            "symbols": ["AAPL"],
            "start": "2020-01-03",
            "end": "2020-01-09",
            "mode": "refresh",
        },
    )
    final = _poll_data_sync(client, refreshed.json()["job_id"])
    assert final["status"] == "done", final
    assert provider.calls == [("AAPL", "2020-01-03", "2020-01-09")]
    replaced = cache.load("AAPL")
    assert replaced.index.min() >= pd.Timestamp("2020-01-03")
    assert replaced.index.max() < pd.Timestamp("2020-01-09")


def test_data_sync_keeps_other_symbols_running_after_failure(client, monkeypatch, synthetic_prices):
    prices = api_app.schema.normalize(synthetic_prices)

    class MixedProvider:
        name = "mixed"

        def get_prices(self, symbol, start=None, end=None):
            if symbol == "BAD":
                raise RuntimeError("provider rejected symbol")
            return prices

    monkeypatch.setattr(api_app, "_price_provider", MixedProvider)
    started = client.post(
        "/data/sync",
        json={
            "symbols": ["AAPL", "BAD"],
            "start": "2020-01-01",
            "end": "2020-02-01",
            "mode": "refresh",
        },
    )
    final = _poll_data_sync(client, started.json()["job_id"])
    assert final["status"] == "done", final
    results = {item["symbol"]: item for item in final["result"]["results"]}
    assert results["AAPL"]["status"] == "fetched"
    assert results["BAD"]["status"] == "failed"


def test_membership_date_sync_fills_entry_and_exit_for_active_stock(client, monkeypatch):
    import numpy as np

    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=30)
    close = 100.0 + np.arange(30) * 0.1
    active_prices = api_app.schema.normalize(
        pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(30, 1_000_000.0),
                "div_cash": np.zeros(30),
                "split_factor": np.ones(30),
            },
            index=idx,
        )
    )

    class FakeProvider:
        name = "fake"

        def get_prices(self, symbol, start=None, end=None):
            return active_prices

    monkeypatch.setattr(api_app, "_price_provider", lambda: FakeProvider())

    started = client.post(
        "/universes/sync-dates", json={"symbols": ["AAPL"], "expected_start": "2010-01-01"}
    )
    final = _poll_membership_sync(client, started.json()["job_id"])
    assert final["status"] == "done", final
    result = final["result"]["results"][0]
    assert result["status"] == "resolved"
    assert result["delisted"] is False
    assert result["exit"] is None
    assert result["entry"] == idx.min().date().isoformat()


def test_membership_date_sync_detects_delisted_symbol(client, monkeypatch, synthetic_prices):
    prices = api_app.schema.normalize(synthetic_prices)

    class FakeProvider:
        name = "fake"

        def get_prices(self, symbol, start=None, end=None):
            return prices

    monkeypatch.setattr(api_app, "_price_provider", lambda: FakeProvider())

    started = client.post(
        "/universes/sync-dates", json={"symbols": ["LEH"], "expected_start": "2000-01-01"}
    )
    final = _poll_membership_sync(client, started.json()["job_id"])
    assert final["status"] == "done", final
    result = final["result"]["results"][0]
    assert result["status"] == "resolved"
    assert result["delisted"] is True
    assert result["exit"] == (prices.index.max() + pd.Timedelta(days=1)).date().isoformat()
    assert result["entry"] == prices.index.min().date().isoformat()


def test_membership_date_sync_keeps_other_symbols_running_after_failure(
    client, monkeypatch, synthetic_prices
):
    prices = api_app.schema.normalize(synthetic_prices)

    class MixedProvider:
        name = "mixed"

        def get_prices(self, symbol, start=None, end=None):
            if symbol == "BAD":
                raise RuntimeError("provider rejected symbol")
            return prices

    monkeypatch.setattr(api_app, "_price_provider", lambda: MixedProvider())
    started = client.post(
        "/universes/sync-dates",
        json={"symbols": ["AAPL", "BAD"], "expected_start": "2019-01-01"},
    )
    final = _poll_membership_sync(client, started.json()["job_id"])
    assert final["status"] == "done", final
    results = {item["symbol"]: item for item in final["result"]["results"]}
    assert results["AAPL"]["status"] == "resolved"
    assert results["BAD"]["status"] == "failed"


def test_workspace_save_load_list_delete(client):
    snapshot = {
        "id": "research-day-1",
        "name": "Research Day 1",
        "version": 1,
        "savedAt": "2026-06-12T00:00:00+00:00",
        "run": {"best_factor": '{"name":"close"}', "lineage": {"nodes": []}},
        "universes": [
            {
                "name": "workspace-universe",
                "memberships": [{"symbol": "AAA", "entry": "2020-01-01"}],
            }
        ],
        "operators": [],
        "ui": {"selectedTab": "genealogy", "selectedLineage": 4},
    }

    saved = client.post("/workspaces", json=snapshot)
    assert saved.status_code == 200
    assert saved.json()["id"] == "research-day-1"

    loaded = client.get("/workspaces/research-day-1")
    assert loaded.status_code == 200
    assert loaded.json()["ui"]["selectedTab"] == "genealogy"

    listed = client.get("/workspaces").json()
    assert any(item["id"] == "research-day-1" and item["hasRun"] for item in listed)
    assert any(u["name"] == "workspace-universe" for u in client.get("/universes").json())

    assert client.delete("/workspaces/research-day-1").status_code == 200
    assert client.get("/workspaces/research-day-1").status_code == 404


def test_invalid_workspace_payload_is_rejected(client):
    response = client.post(
        "/workspaces",
        json={"name": "bad", "version": 0, "savedAt": "2026-06-12T00:00:00+00:00"},
    )
    assert response.status_code == 422


def test_run_progress_reaches_target(client):
    config = {"population_size": 16, "generations": 3, "max_depth": 4, "max_nodes": 20, "seed": 0}
    job_id = client.post("/runs", json={"config": config}).json()["job_id"]
    final = _poll(client, job_id)
    assert final["status"] == "done", final

    progress = final["progress"]
    assert progress is not None
    assert progress["generation"] == 3
    assert progress["target_generations"] == 3
    assert len(progress["history"]) == 4  # generation 0 (init) plus 1..3
    assert progress["best"] is not None and "fitness" in progress["best"]


def test_run_search_stop_halts_early(signal_panel):
    from alphalineage.api.progress import RunProgress
    from alphalineage.api.service import run_search
    from alphalineage.core.gp import GPConfig

    panel, _ = signal_panel
    config = GPConfig(population_size=16, generations=50, max_depth=4, max_nodes=20, seed=0)
    progress = RunProgress(target_generations=config.generations)

    def stop() -> bool:
        # An idempotent cancellation source, matching the Event used by the API: allow the
        # initial population and generation 1 to commit, then stop before generation 2.
        return int(progress.snapshot()["generation"]) >= 1

    result = run_search(config, panel, progress=progress, stop=stop)
    assert result["generations"] == 1
    assert result["termination_reason"] == "user_stopped"


def test_stop_endpoint_returns_stopping(client):
    config = {"population_size": 16, "generations": 40, "max_depth": 4, "max_nodes": 20, "seed": 0}
    job_id = client.post("/runs", json={"config": config}).json()["job_id"]

    stopped = client.post(f"/runs/{job_id}/stop")
    assert stopped.status_code == 200
    assert stopped.json() == {"stopping": True}
    assert client.post("/runs/does-not-exist/stop").status_code == 404

    final = _poll(client, job_id)
    assert final["status"] in {"done", "stopped"}, final
    if final["status"] == "stopped":
        assert final["termination_reason"] == "user_stopped"


def test_run_progress_threadsafe_snapshot():
    import threading

    from alphalineage.api.progress import RunProgress
    from alphalineage.core.tree import Node

    progress = RunProgress(target_generations=20)
    tree = Node("rank", (Node("close"),))

    def observe(gen: int) -> None:
        progress.on_generation(gen, [(tree, [0], "elite", float(gen))])

    threads = [threading.Thread(target=observe, args=(g,)) for g in range(1, 21)]
    for t in threads:
        t.start()
    snaps = [progress.snapshot() for _ in range(50)]
    for t in threads:
        t.join()

    assert all(0 <= s["generation"] <= 20 for s in snaps)
    final = progress.snapshot()
    assert final["generation"] >= 1
    assert len(final["history"]) == 20


def test_run_progress_exposes_candidate_and_resource_telemetry():
    from alphalineage.api.progress import RunProgress

    progress = RunProgress(
        target_generations=12,
        resources={"profile": "auto", "workers": 8, "percent": 50},
    )
    progress.on_scoring(
        phase="initializing",
        generation=0,
        done=21,
        total=80,
        factors_per_second=17.5,
    )
    snapshot = progress.snapshot()
    assert snapshot["phase"] == "initializing"
    assert snapshot["candidate_done"] == 21
    assert snapshot["candidate_total"] == 80
    assert snapshot["factors_per_second"] == 17.5
    assert snapshot["resources"]["workers"] == 8


def test_sync_progress_threadsafe_snapshot():
    import threading

    from alphalineage.api.progress import SyncProgress

    progress = SyncProgress(total=20)
    threads = [threading.Thread(target=progress.advance, args=(f"SYM{i}",)) for i in range(20)]
    for t in threads:
        t.start()
    snaps = [progress.snapshot() for _ in range(50)]
    for t in threads:
        t.join()

    assert all(0 <= s["done"] <= 20 and s["total"] == 20 for s in snaps)
    final = progress.snapshot()
    assert final["done"] == 20
    assert final["current_symbol"] is not None


def test_static_dir_served_when_configured(tmp_path):
    from alphalineage.api.app import mount_static

    (tmp_path / "index.html").write_text("<html>alphalineage ui</html>", encoding="utf-8")
    before = len(app.router.routes)
    mount_static(str(tmp_path))
    try:
        with TestClient(app) as test_client:
            root = test_client.get("/")
            assert root.status_code == 200
            assert "alphalineage ui" in root.text
            # API routes still resolve - the mount is last and least specific
            assert test_client.get("/health").json()["status"] == "ok"
    finally:
        del app.router.routes[before:]  # remove the mount so other tests see a clean app


def test_shutdown_is_gated_and_indirected(client, monkeypatch):
    import alphalineage.api.app as api_app

    # disabled by default - a stray POST cannot kill the server
    monkeypatch.delenv("ALPHALINEAGE_ALLOW_SHUTDOWN", raising=False)
    assert client.post("/shutdown").status_code == 403

    # enabled by the launcher: returns shutting_down and schedules exit (patched, never real)
    calls = {"n": 0}

    def _fake() -> None:
        calls["n"] += 1

    monkeypatch.setattr(api_app, "_schedule_shutdown", _fake)
    monkeypatch.setenv("ALPHALINEAGE_ALLOW_SHUTDOWN", "1")
    response = client.post("/shutdown")
    assert response.status_code == 200
    assert response.json() == {"shutting_down": True}
    assert calls["n"] == 1


def test_jobstore_runs_and_captures_failure():
    store = JobStore()

    ok = store.submit(lambda: 21 * 2)
    bad = store.submit(lambda: 1 / 0)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if store.get(ok).status in ("done", "failed") and store.get(bad).status in (
            "done",
            "failed",
        ):
            break
        time.sleep(0.02)

    assert store.get(ok).status == "done"
    assert store.get(ok).result == 42
    assert store.get(bad).status == "failed"
    assert "ZeroDivisionError" in (store.get(bad).error or "")


def test_jobstore_treats_training_cancellation_as_stopped_not_failed():
    from alphalineage.api.progress import RunProgress
    from alphalineage.core.gp import TrainingCancelled

    store = JobStore()
    progress = RunProgress()

    def cancel() -> None:
        raise TrainingCancelled("stopped before initialization committed")

    job_id = store.submit(cancel, progress=progress)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and store.get(job_id).status in {"queued", "running"}:
        time.sleep(0.01)
    job = store.get(job_id)
    assert job is not None
    assert job.status == "stopped"
    assert job.error is None
    assert job.termination_reason == "user_stopped"
    assert progress.snapshot()["phase"] == "stopped"


def test_jobstore_rejects_stop_after_locked_test_finalization_starts():
    import threading

    from alphalineage.api.progress import RunProgress

    store = JobStore()
    progress = RunProgress()
    finalizing = threading.Event()
    release = threading.Event()

    def finish_locked_report() -> dict[str, str]:
        progress.set_phase("finalizing")
        finalizing.set()
        assert release.wait(2)
        return {"termination_reason": "completed"}

    cancel = threading.Event()
    job_id = store.submit(finish_locked_report, progress=progress, cancel=cancel)
    assert finalizing.wait(2)
    try:
        assert store.cancel(job_id) is False
        assert cancel.is_set() is False
        assert progress.snapshot()["phase"] == "finalizing"
    finally:
        release.set()
