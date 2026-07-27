"""Bundled index snapshots and universe-aware data synchronization."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import alphalineage.api.app as api_app
from alphalineage.api.app import app, get_panel
from alphalineage.data.universe import (
    MARKET_SYMBOL_ALIASES,
    Membership,
    Universe,
    bundled_snapshot_name,
    bundled_snapshot_specs,
    bundled_universe,
)


@pytest.fixture
def client(synthetic_panel) -> Iterator[TestClient]:
    app.dependency_overrides[get_panel] = lambda: synthetic_panel
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _poll_sync(client: TestClient, job_id: str, *, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/data/sync/{job_id}").json()
        if payload["status"] in {"done", "stopped", "failed"}:
            return payload
        time.sleep(0.01)
    return payload


def _poll_run(client: TestClient, job_id: str, *, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/runs/{job_id}").json()
        if payload["status"] in {"done", "stopped", "failed"}:
            return payload
        time.sleep(0.01)
    return payload


@pytest.mark.parametrize(
    ("name", "count"),
    [
        ("builtin-sp500-current", 503),
        ("builtin-djia-current", 30),
        ("builtin-nasdaq100-current", 103),
    ],
)
def test_bundled_snapshots_are_loadable_immutable_definitions(name: str, count: int) -> None:
    universe = bundled_universe(name)

    assert universe.name == name
    assert universe.mode == "static_snapshot"
    assert len(universe.all_symbols()) == count
    assert universe.definition["snapshot_date"] == "2026-07-15"
    assert universe.definition["member_count"] == count
    assert len(universe.fingerprint) == 64
    assert universe.provenance["source_url"].startswith("https://")
    assert universe.provenance["attribution"] == "Wikipedia contributors"
    assert universe.provenance["license"] == "CC BY-SA 4.0"
    assert universe.provenance["license_url"] == (
        "https://creativecommons.org/licenses/by-sa/4.0/"
    )
    assert universe.provenance["terms_url"] == (
        "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"
    )

    # Static snapshots are deliberately usable over historical price rows, but are
    # explicitly labelled survivorship-biased by their definition and API readiness.
    dates = pd.date_range("2000-01-03", periods=2, freq="D")
    symbol = universe.all_symbols()[0]
    mask = universe.membership_mask(dates, [symbol, "NOT-A-MEMBER"])
    assert mask[symbol].tolist() == [True, True]
    assert mask["NOT-A-MEMBER"].tolist() == [False, False]
    assert universe.members_asof("2000-01-03") == universe.all_symbols()
    assert universe.members_through("2000-01-03") == universe.all_symbols()
    assert universe.members_overlapping("2000-01-03", "2000-01-04") == universe.all_symbols()


def test_bundled_manifest_aliases_and_vendor_symbols_are_stable() -> None:
    specs = {str(item["id"]): item for item in bundled_snapshot_specs()}

    assert set(specs) == {
        "builtin-sp500-current",
        "builtin-djia-current",
        "builtin-nasdaq100-current",
    }
    assert bundled_snapshot_name("S&P500") == "builtin-sp500-current"
    assert bundled_snapshot_name("^DJI") == "builtin-djia-current"
    assert bundled_snapshot_name("NDX") == "builtin-nasdaq100-current"
    assert bundled_snapshot_name("sp500") is None
    assert MARKET_SYMBOL_ALIASES == {"BRK.B": "BRK-B", "BF.B": "BF-B"}
    assert "BRK-B" in bundled_universe("builtin-sp500-current").all_symbols()


def test_static_snapshot_cache_readiness_uses_research_cutoff_not_capture_date(
    synthetic_prices,
) -> None:
    cache = api_app.ParquetCache()
    prices = api_app.schema.normalize(synthetic_prices)
    cache.store("AAPL", prices)
    universe = Universe(
        "static-cutoff-test",
        [Membership("AAPL", pd.Timestamp("2026-07-15"))],
        mode="static_snapshot",
    )

    coverage = api_app._cache_coverage_for(universe, prices.index.max(), cache=cache)

    assert coverage["eligible_symbols"] == ["AAPL"]
    assert coverage["late_start_symbols"] == []
    assert coverage["complete"] is True


def test_universe_summary_exposes_definition_provenance_and_readiness(client: TestClient) -> None:
    response = client.get("/universes", params={"view": "summary"})
    assert response.status_code == 200
    summaries = {item["name"]: item for item in response.json()}

    snapshot = summaries["builtin-sp500-current"]
    assert snapshot["source"] == "bundled"
    assert snapshot["mode"] == "static_snapshot"
    assert snapshot["symbol_count"] == 503
    assert snapshot["definition"]["snapshot_date"] == "2026-07-15"
    assert snapshot["readiness"]["coverage_checked"] is False
    assert snapshot["readiness"]["research_ready"] is False
    assert "memberships" not in snapshot
    assert "cache_coverage" not in snapshot
    assert client.get("/universes", params={"view": "unknown"}).status_code == 400

    legacy_summary = client.get("/universes", params={"summary": True}).json()
    assert "memberships" not in next(
        item for item in legacy_summary if item["name"] == "builtin-sp500-current"
    )

    detail_response = client.get("/universes/builtin-sp500-current")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["name"] == "builtin-sp500-current"
    assert detail["source"] == "bundled"
    assert detail["membership_count"] == 503
    assert detail["readiness"]["coverage_checked"] is True
    assert detail["readiness"]["research_ready"] is False

    presets = {item["id"]: item for item in client.get("/universe-presets").json()}
    prepared = presets["sp500"]
    assert prepared["definition"] == snapshot["definition"]
    assert prepared["fingerprint"] == snapshot["fingerprint"]
    assert prepared["provenance_detail"] == snapshot["provenance"]
    assert prepared["readiness"]["coverage_checked"] is False


def test_point_in_time_import_id_coexists_with_bundled_snapshot(
    client: TestClient,
) -> None:
    assert client.get("/universes/sp500").status_code == 404
    created = client.post(
        "/universes",
        json={
            "name": "sp500",
            "memberships": [{"symbol": "AAPL", "entry": "2020-01-01", "exit": None}],
        },
    )
    assert created.status_code == 200
    try:
        custom = client.get("/universes/sp500").json()
        snapshot = client.get("/universes/builtin-sp500-current").json()
        presets = {item["id"]: item for item in client.get("/universe-presets").json()}

        assert custom["source"] == "custom"
        assert custom["mode"] == "point_in_time"
        assert custom["symbols"] == ["AAPL"]
        assert snapshot["source"] == "bundled"
        assert snapshot["symbol_count"] == 503
        assert presets["sp500"]["pit_available"] is True
        assert presets["sp500"]["pit_universe"] == "sp500"
    finally:
        deleted = client.delete("/universes/sp500")
        assert deleted.status_code == 200

    assert client.get("/universes/sp500").status_code == 404


def test_bundled_snapshot_ids_cannot_be_replaced_or_deleted(client: TestClient) -> None:
    snapshot_id = "builtin-sp500-current"
    replacement = {
        "name": snapshot_id,
        "memberships": [{"symbol": "AAPL", "entry": "2020-01-01", "exit": None}],
    }

    created = client.post("/universes", json=replacement)
    updated = client.put(f"/universes/{snapshot_id}", json=replacement)
    deleted = client.delete(f"/universes/{snapshot_id}")

    assert created.status_code == 400
    assert updated.status_code == 400
    assert deleted.status_code == 400
    detail = client.get(f"/universes/{snapshot_id}").json()
    assert detail["source"] == "bundled"
    assert detail["symbol_count"] == 503


def test_universe_sync_can_be_listed_reattached_and_stopped(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    class FakeProvider:
        name = "fake"

    def fake_sync(symbol: str, **_kwargs) -> api_app.DataSyncResult:
        calls.append(symbol)
        entered.set()
        release.wait(timeout=5)
        return api_app.DataSyncResult(
            symbol=symbol,
            provider_symbol=symbol,
            status="fetched",
            rows_fetched=1,
            rows_cached=1,
            provider="fake",
        )

    monkeypatch.setattr(api_app, "_price_provider", lambda: FakeProvider())
    monkeypatch.setattr(api_app, "_sync_one_symbol", fake_sync)

    request = {
        "universe": "builtin-djia-current",
        "start": "2020-01-01",
        "mode": "incremental",
    }
    started = client.post("/data/sync", json=request)
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    assert entered.wait(timeout=2)
    try:
        duplicate = client.post("/data/sync", json=request)
        assert duplicate.status_code == 200
        assert duplicate.json()["job_id"] == job_id
        assert duplicate.json()["reused"] is True

        active = client.get("/data/sync", params={"active_only": True}).json()
        reattached = next(item for item in active if item["job_id"] == job_id)
        assert reattached["request"]["universe"] == "builtin-djia-current"
        assert len(reattached["request"]["symbols"]) == 30
        assert reattached["universe_definition"]["requested_name"] == (
            "builtin-djia-current"
        )
        assert reattached["universe_definition"]["name"] == "builtin-djia-current"
        assert reattached["universe_definition"]["mode"] == "static_snapshot"
        assert len(reattached["universe_definition"]["fingerprint"]) == 64

        stopped = client.post(f"/data/sync/{job_id}/stop")
        assert stopped.status_code == 200
        assert stopped.json() == {"stopping": True}
    finally:
        release.set()

    final = _poll_sync(client, job_id)
    assert final["status"] == "stopped"
    assert final["termination_reason"] == "user_stopped"
    assert final["result"]["termination_reason"] == "user_stopped"
    assert len(final["result"]["results"]) == 1
    assert len(calls) == 1
    assert all(item["job_id"] != job_id for item in client.get(
        "/data/sync", params={"active_only": True}
    ).json())


def test_training_run_pins_universe_definition_before_background_work(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe_name = "pinned-universe"
    memberships = [
        {"symbol": f"S{index}", "entry": "2021-01-01", "exit": None}
        for index in range(6)
    ]
    assert client.post(
        "/universes", json={"name": universe_name, "memberships": memberships}
    ).status_code == 200
    original = client.get(f"/universes/{universe_name}").json()
    entered = threading.Event()
    release = threading.Event()

    def blocked_search(*_args, **_kwargs) -> dict[str, str]:
        entered.set()
        assert release.wait(timeout=5)
        return {"termination_reason": "completed"}

    monkeypatch.setattr(api_app, "run_search", blocked_search)
    try:
        started = client.post(
            "/runs",
            json={
                "universe": universe_name,
                "as_of": "2021-03-25",
                "config": {"population_size": 6, "generations": 1},
            },
        )
        assert started.status_code == 200
        job_id = started.json()["job_id"]
        assert entered.wait(timeout=2)

        changed = [*memberships, {"symbol": "S6", "entry": "2021-01-01", "exit": None}]
        updated = client.put(
            f"/universes/{universe_name}",
            json={"name": universe_name, "memberships": changed},
        )
        assert updated.status_code == 200
        assert client.get(f"/universes/{universe_name}").json()["fingerprint"] != (
            original["fingerprint"]
        )
    finally:
        release.set()

    final = _poll_run(client, job_id)
    assert final["status"] == "done", final
    pinned = final["result"]["universe_definition"]
    assert pinned["name"] == universe_name
    assert pinned["fingerprint"] == original["fingerprint"]
    assert final["result"]["context"]["universe_definition"] == pinned
    assert client.delete(f"/universes/{universe_name}").status_code == 200
