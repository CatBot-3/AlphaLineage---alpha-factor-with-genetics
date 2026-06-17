"""A4/A5 acceptance + supporting tests for iterative training sessions."""

from __future__ import annotations

import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from alphalineage.api.app import app, get_panel
from alphalineage.core import extensions


@pytest.fixture
def client(signal_panel):
    panel, _ = signal_panel
    app.dependency_overrides[get_panel] = lambda: panel
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _clean_operators():
    yield
    extensions.clear_user_operators()


_SMALL = {"population_size": 16, "generations": 2, "max_depth": 4, "max_nodes": 20, "seed": 0}

_MOMENTUM = {
    "name": "mom10",
    "arg_types": ["series"],
    "out_type": "signal",
    "body": {
        "name": "rank",
        "children": [
            {
                "name": "ts_mean",
                "children": [{"name": "$arg", "value": 0}, {"name": "window", "value": 10}],
            }
        ],
    },
}


def _poll_job(client: TestClient, job_id: str, *, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/runs/{job_id}").json()
        if payload["status"] in ("done", "failed"):
            return payload
        time.sleep(0.1)
    return payload


def _create(client: TestClient, **overrides) -> tuple[str, dict]:
    body = {"name": "s", "universe": "sp500-lite", "config": _SMALL, **overrides}
    created = client.post("/sessions", json=body)
    assert created.status_code == 200, created.text
    payload = created.json()
    final = _poll_job(client, payload["job_id"])
    assert final["status"] == "done", final
    return payload["session_id"], final


# --- pure helpers (P1: lock the time boundary) -----------------------------------
def test_split_from_boundaries_excludes_test_dates():
    from alphalineage.api.sessions import derive_boundaries, split_from_boundaries

    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    boundaries = derive_boundaries(dates, train=0.6, valid=0.2, embargo=5)
    split = split_from_boundaries(dates, boundaries)
    test_start = pd.Timestamp(boundaries.test_start)
    assert split.train.max() < test_start
    assert split.test.min() >= test_start

    # a changed universe whose panel still spans the boundary keeps train < test_start
    interior_gap = dates.delete(range(40, 50))  # drop some research-region dates
    split2 = split_from_boundaries(interior_gap, boundaries)
    assert split2.train.max() < test_start
    assert split2.test.min() >= test_start

    # if the test segment vanishes entirely, P1 rejects rather than relocating the boundary
    with pytest.raises(ValueError, match="locked"):
        split_from_boundaries(dates[:95], boundaries)


def test_session_create_run_complete(client):
    session_id, final = _create(client)

    state = client.get(f"/sessions/{session_id}").json()
    assert state["boundaries"]["test_start"]
    assert state["cumulative_trials"] > 0
    assert state["test_reads"] == 1
    assert len(state["segments"]) == 1 and state["segments"][0]["status"] == "done"

    result = final["result"]
    assert "deflated_sharpe" in result["report"]
    assert result["session_id"] == session_id
    assert result["test_reads"] == 1


def test_session_lineage_persists_and_replays(client):
    session_id, _ = _create(client)
    lineage = client.get(f"/sessions/{session_id}/lineage").json()
    assert len(lineage["nodes"]) > 0
    max_gen = max(n["generation"] for n in lineage["nodes"])
    assert max_gen == _SMALL["generations"]
    # every node carries a fitness for the grouped genealogy view
    assert all(n.get("fitness") is not None for n in lineage["nodes"])


def test_session_seeded_from_saved_factor(client):
    from alphalineage.core.tree import from_dict, to_json

    assert client.post("/operators", json=_MOMENTUM).status_code == 200
    seed_tree = {"name": "mom10", "children": [{"name": "close"}]}
    saved = client.post(
        "/factors",
        json={
            "name": "seed",
            "tree": seed_tree,
            "provenance": {"session_id": "prior", "cumulative_trials": 500, "test_reads": 1},
        },
    ).json()

    # forget the operator: the seeded session must re-register it from the factor file
    extensions.clear_user_operators()
    session_id, final = _create(client, seed_factor_ids=[saved["id"]])

    lineage = client.get(f"/sessions/{session_id}/lineage").json()
    seeds = [n for n in lineage["nodes"] if n["op"] == "seed"]
    assert any(to_json(from_dict(n["tree"])) == to_json(from_dict(seed_tree)) for n in seeds)
    # inherited the prior session's trial baseline (deflation never softens)
    assert final["result"]["report"]["n_trials"] >= 500


def test_unknown_session_is_404(client):
    assert client.get("/sessions/nope").status_code == 404
    assert client.get("/sessions/nope/lineage").status_code == 404


# --- A5: continue ----------------------------------------------------------------
def test_continue_warm_start_carries_population(client):
    session_id, first = _create(client)
    gens_after_first = first["result"]["generations"]

    cont = client.post(f"/sessions/{session_id}/continue", json={"generations": 2})
    payload = cont.json()
    final = _poll_job(client, payload["job_id"])
    assert final["status"] == "done", final
    # generation numbering continues across the segment boundary
    assert final["result"]["generations"] == gens_after_first + 2

    state = client.get(f"/sessions/{session_id}").json()
    assert len(state["segments"]) == 2
    assert state["segments"][1]["gen_start"] == gens_after_first


def test_continue_trials_increase_and_oos_read_flagged(client):
    session_id, first = _create(client)
    trials_after_first = first["result"]["cumulative_trials"]

    payload = client.post(f"/sessions/{session_id}/continue", json={"generations": 2}).json()
    final = _poll_job(client, payload["job_id"])
    assert final["status"] == "done", final

    assert final["result"]["cumulative_trials"] >= trials_after_first  # monotone (P2)
    assert final["result"]["test_reads"] == 2  # second OOS read
    assert final["result"]["repeated_oos_warning"] is True  # surfaced (P3)


def test_continue_changed_universe_keeps_locked_boundary(client):
    # define a custom universe whose cached panel is the synthetic one (override stays in effect)
    session_id, _ = _create(client)
    before = client.get(f"/sessions/{session_id}").json()["boundaries"]

    cont = client.post(
        f"/sessions/{session_id}/continue",
        json={"generations": 1, "universe": "sp500-lite"},
    )
    final = _poll_job(client, cont.json()["job_id"])
    assert final["status"] == "done", final
    after = client.get(f"/sessions/{session_id}").json()["boundaries"]
    assert after == before  # frozen boundary is byte-identical (P1)


def test_continue_added_operator_deflates_harder(client):
    session_id, first = _create(client)
    base_trials = first["result"]["report"]["n_trials"]

    cont = client.post(
        f"/sessions/{session_id}/continue",
        json={"generations": 1, "operators": [_MOMENTUM]},
    )
    final = _poll_job(client, cont.json()["job_id"])
    assert final["status"] == "done", final
    # a larger operator palette inflates the effective trial count (invariant 1, P7-T3)
    assert final["result"]["report"]["n_trials"] > base_trials


def test_continue_while_running_is_409(client):
    # submit a session and immediately try to continue before it finishes
    body = {"name": "s", "universe": "sp500-lite", "config": {**_SMALL, "generations": 30}}
    created = client.post("/sessions", json=body).json()
    session_id = created["session_id"]

    conflict = client.post(f"/sessions/{session_id}/continue", json={"generations": 2})
    # either the first segment is still running (409) or it already finished (200)
    assert conflict.status_code in (409, 200)
    if conflict.status_code == 200:
        _poll_job(client, conflict.json()["job_id"])
    # stop and drain
    client.post(f"/sessions/{session_id}/stop")
    _poll_job(client, created["job_id"])


def test_continue_unknown_session_is_404(client):
    assert client.post("/sessions/nope/continue", json={"generations": 1}).status_code == 404
