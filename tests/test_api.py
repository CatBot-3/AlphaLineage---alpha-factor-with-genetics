"""Phase 5 acceptance + supporting tests for the HTTP API (tests/test_api.py)."""

from __future__ import annotations

import time

import pandas as pd
import pytest
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
        if payload["status"] in ("done", "failed"):
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


def test_job_lifecycle(client):
    config = {"population_size": 16, "generations": 2, "max_depth": 4, "max_nodes": 20, "seed": 0}
    submit = client.post("/runs", json={"config": config})
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    final = _poll(client, job_id)
    assert final["status"] == "done", final
    result = final["result"]
    assert "best_factor" in result
    assert "report" in result and "deflated_sharpe" in result["report"]

    lineage = client.get(f"/runs/{job_id}/lineage").json()
    assert "nodes" in lineage and len(lineage["nodes"]) > 0

    saved = client.get(f"/workspaces/run-{job_id}")
    assert saved.status_code == 200
    assert saved.json()["run"]["best_factor"] == result["best_factor"]


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
    assert client.get("/formulas").json() == []


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
    assert result["exit"] == prices.index.max().date().isoformat()
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
    from alphalineage.api.service import run_search
    from alphalineage.core.gp import GPConfig

    panel, _ = signal_panel
    config = GPConfig(population_size=16, generations=50, max_depth=4, max_nodes=20, seed=0)
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # let generation 1 run, then halt

    result = run_search(config, panel, stop=stop)
    assert result["generations"] < 50


def test_stop_endpoint_returns_stopping(client):
    config = {"population_size": 16, "generations": 40, "max_depth": 4, "max_nodes": 20, "seed": 0}
    job_id = client.post("/runs", json={"config": config}).json()["job_id"]

    stopped = client.post(f"/runs/{job_id}/stop")
    assert stopped.status_code == 200
    assert stopped.json() == {"stopping": True}
    assert client.post("/runs/does-not-exist/stop").status_code == 404

    final = _poll(client, job_id)
    assert final["status"] == "done", final


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
