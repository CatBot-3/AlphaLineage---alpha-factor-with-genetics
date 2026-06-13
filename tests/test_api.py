"""Phase 5 acceptance + supporting tests for the HTTP API (tests/test_api.py)."""

from __future__ import annotations

import time

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


def _poll(client: TestClient, job_id: str, *, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/runs/{job_id}").json()
        if payload["status"] in ("done", "failed"):
            return payload
        time.sleep(0.2)
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
