"""Phase 5 acceptance + supporting tests for the HTTP API (tests/test_api.py)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from alphaforge.api.app import app, get_panel
from alphaforge.api.jobs import JobStore


@pytest.fixture
def client(synthetic_panel):
    app.dependency_overrides[get_panel] = lambda: synthetic_panel
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


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


def test_unknown_job_is_404(client):
    assert client.get("/runs/does-not-exist").status_code == 404


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
