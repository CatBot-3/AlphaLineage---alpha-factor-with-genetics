"""Automatic price sync: fill the cheap gaps silently, leave the expensive ones asked about."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import alphalineage.api.app as api_app
from alphalineage.api.app import AUTO_SYNC_MODES, DEFAULT_AUTO_SYNC, app, plan_coverage_fill
from alphalineage.core import extensions


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    extensions.clear_user_operators()


def _coverage(*, incomplete: list[str], missing: list[str]) -> dict:
    return {"incomplete_symbols": incomplete, "missing_symbols": missing}


def test_the_split_is_between_what_costs_requests_and_what_costs_the_monthly_allowance() -> None:
    """A cached symbol's tail is cheap to extend; a new symbol is not, and only it is deferred."""
    coverage = _coverage(incomplete=["AAPL", "MSFT", "NVDA", "ZZZ"], missing=["NVDA", "ZZZ"])

    top_up = plan_coverage_fill(coverage, "top_up")
    assert top_up["symbols"] == ["AAPL", "MSFT"]
    assert top_up["deferred_symbols"] == ["NVDA", "ZZZ"]
    assert (top_up["cached_gap_count"], top_up["first_time_count"]) == (2, 2)

    full = plan_coverage_fill(coverage, "full")
    assert full["symbols"] == ["AAPL", "MSFT", "NVDA", "ZZZ"]
    assert full["deferred_symbols"] == []
    # The counts describe the same universe either way, so the UI can say what it is skipping.
    assert (full["cached_gap_count"], full["first_time_count"]) == (2, 2)


def test_a_complete_universe_plans_nothing() -> None:
    plan = plan_coverage_fill(_coverage(incomplete=[], missing=[]), "full")
    assert plan["symbols"] == [] and plan["deferred_symbols"] == []


def test_fill_plan_endpoint_reports_the_split_without_fetching(client, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        api_app, "_sync_one_symbol", lambda *a, **k: calls.append("fetched") or None
    )

    response = client.get("/universes/sp500-lite/fill-plan?as_of=2026-07-15&scope=top_up")
    assert response.status_code == 200, response.text
    plan = response.json()
    assert plan["name"] == "sp500-lite"
    assert plan["scope"] == "top_up"
    assert plan["mode"] in AUTO_SYNC_MODES
    # Nothing is cached in a fresh temp data dir, so every gap is a first-time pull and the
    # cheap scope has nothing it may do on its own.
    assert plan["symbols"] == []
    assert plan["first_time_count"] > 0
    assert calls == []

    full = client.get("/universes/sp500-lite/fill-plan?as_of=2026-07-15&scope=full").json()
    assert full["symbols"] == full["deferred_symbols"] or full["symbols"]
    assert len(full["symbols"]) == full["first_time_count"] + full["cached_gap_count"]

    assert client.get("/universes/sp500-lite/fill-plan?scope=sideways").status_code == 400
    assert client.get("/universes/nope/fill-plan").status_code == 404


def test_filling_a_universe_with_no_reachable_gaps_starts_no_job(client) -> None:
    started = client.post("/universes/sp500-lite/fill", json={"scope": "top_up"})
    assert started.status_code == 200, started.text
    payload = started.json()
    # Every gap here is a never-downloaded symbol, which this scope refuses to spend.
    assert payload["job_id"] is None
    assert payload["status"] == "nothing_to_do"
    assert payload["first_time_count"] > 0


def test_a_full_fill_submits_exactly_the_planned_symbols(client, monkeypatch) -> None:
    requested: dict = {}

    def fake_sync(req, progress=None, stop=None):
        requested["symbols"] = list(req.symbols)
        requested["mode"] = req.mode
        requested["start"] = req.start
        return {
            "results": [],
            "failed_count": 0,
            "succeeded_count": 0,
            "termination_reason": "completed",
        }

    monkeypatch.setattr(api_app, "_run_data_sync", fake_sync)
    started = client.post(
        "/universes/sp500-lite/fill", json={"scope": "full", "as_of": "2026-07-15"}
    )
    assert started.status_code == 200, started.text
    payload = started.json()
    assert payload["job_id"]

    final = client.get(f"/runs/{payload['job_id']}").json()
    for _ in range(100):
        if final["status"] in ("done", "failed", "stopped"):
            break
        final = client.get(f"/runs/{payload['job_id']}").json()
    assert final["status"] == "done", final
    assert requested["symbols"] == payload["symbols"]
    # Incremental, so a symbol already holding most of its history only fetches the tail.
    assert requested["mode"] == "incremental"
    assert requested["start"]
    polled = client.get(f"/data/sync/{payload['job_id']}")
    assert polled.status_code == 200, polled.text
    assert polled.json()["status"] == "done"
    assert any(job["job_id"] == payload["job_id"] for job in client.get("/data/sync").json())


def test_the_policy_is_stored_validated_and_defaulted(client) -> None:
    assert client.get("/settings").json()["auto_sync"] == DEFAULT_AUTO_SYNC

    for mode in AUTO_SYNC_MODES:
        updated = client.put("/settings", json={"auto_sync": mode})
        assert updated.status_code == 200, updated.text
        assert updated.json()["auto_sync"] == mode

    rejected = client.put("/settings", json={"auto_sync": "aggressive"})
    assert rejected.status_code == 400
    assert "auto_sync" in rejected.json()["detail"]
    # A rejected value leaves the stored one alone.
    assert client.get("/settings").json()["auto_sync"] == AUTO_SYNC_MODES[-1]


def test_a_second_fill_for_the_same_universe_joins_the_running_job(client, monkeypatch) -> None:
    import threading

    release = threading.Event()

    def slow_sync(req, progress=None, stop=None):
        release.wait(timeout=10)
        return {
            "results": [],
            "failed_count": 0,
            "succeeded_count": 0,
            "termination_reason": "completed",
        }

    monkeypatch.setattr(api_app, "_run_data_sync", slow_sync)
    try:
        first = client.post("/universes/sp500-lite/fill", json={"scope": "full"}).json()
        second = client.post("/universes/sp500-lite/fill", json={"scope": "full"}).json()
        assert first["job_id"]
        assert second["job_id"] == first["job_id"]
        assert second["reused"] is True
    finally:
        release.set()
