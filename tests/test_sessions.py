"""A4/A5 acceptance + supporting tests for iterative training sessions."""

from __future__ import annotations

import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import alphalineage.api.app as api_app
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
        if payload["status"] in ("done", "stopped", "failed"):
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


def test_session_pins_active_formula_revisions(client):
    formula = {
        "name": "session_rank",
        "arg_types": ["series"],
        "inputs": [{"name": "price", "type": "series", "description": "Price input."}],
        "out_type": "signal",
        "body": {"name": "rank", "children": [{"name": "$arg", "value": 0}]},
    }
    assert client.post("/formulas", json=formula).status_code == 200
    session_id, _ = _create(client)
    state = client.get(f"/sessions/{session_id}").json()
    pinned = {item["name"]: item for item in state["formula_revisions"]}
    assert pinned["session_rank"]["runtime_name"] == "session_rank"
    assert pinned["session_rank"]["body"] == formula["body"]


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


@pytest.mark.parametrize(
    "body",
    [
        {"name": "   ", "config": _SMALL},
        {"name": "s", "config": _SMALL, "train": 0.8, "valid": 0.2},
        {"name": "s", "config": {"generatons": 2}},
    ],
)
def test_session_create_rejects_invalid_fields_synchronously(client, body):
    response = client.post("/sessions", json=body)
    assert response.status_code == 400


def test_session_request_model_rejects_nonpositive_continue_and_embargo(client):
    assert client.post("/sessions/nope/continue", json={"generations": 0}).status_code == 422
    response = client.post("/sessions", json={"name": "s", "config": _SMALL, "embargo": 0})
    assert response.status_code == 422


def test_session_persists_requested_resources_and_continue_inherits_them(client):
    session_id, _ = _create(
        client,
        resources={"profile": "custom", "cpu_budget_percent": 70},
    )
    state = client.get(f"/sessions/{session_id}").json()
    assert state["resources"] == {"profile": "custom", "cpu_budget_percent": 70}

    continued = client.post(f"/sessions/{session_id}/continue", json={"generations": 1})
    assert continued.status_code == 200, continued.text
    final = _poll_job(client, continued.json()["job_id"])
    assert final["status"] == "done", final
    state = client.get(f"/sessions/{session_id}").json()
    assert state["resources"] == {"profile": "custom", "cpu_budget_percent": 70}
    assert state["segments"][-1]["resources"]["profile"] == "custom"
    assert state["segments"][-1]["resources"]["percent"] == 70


def test_continue_translates_invalid_gp_override_to_client_error(client):
    session_id, _ = _create(client)
    response = client.post(
        f"/sessions/{session_id}/continue",
        json={"generations": 1, "config": {"generatons": 3}},
    )
    assert response.status_code == 400
    assert "unknown GP config" in response.json()["detail"]

    oversized = client.post(
        f"/sessions/{session_id}/continue",
        json={"generations": 1_000, "config": {"population_size": 5_000}},
    )
    assert oversized.status_code == 400
    assert "additional generations" in oversized.json()["detail"]


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


def test_active_job_is_persisted_blocks_continue_and_can_stop(client, monkeypatch):
    import threading

    started = threading.Event()

    def slow_segment(*args, stop=None, **kwargs):
        started.set()
        deadline = time.monotonic() + 5
        while stop is not None and not stop() and time.monotonic() < deadline:
            time.sleep(0.01)
        return {"stopped": True}

    monkeypatch.setattr(api_app.sessions, "run_segment", slow_segment)
    created = client.post(
        "/sessions",
        json={
            "name": "persistent-job",
            "universe": "sp500-lite",
            "as_of": "2019-12-31T15:30:00Z",
            "config": _SMALL,
        },
    ).json()
    assert started.wait(2)
    session_id, job_id = created["session_id"], created["job_id"]
    stored = api_app.sessions.load_session(session_id)
    assert stored["as_of"] == "2019-12-31"
    assert stored["active_job_id"] == job_id
    assert stored["last_job"]["status"] in {"queued", "running"}

    conflict = client.post(f"/sessions/{session_id}/continue", json={"generations": 1})
    assert conflict.status_code == 409
    assert client.post(f"/sessions/{session_id}/stop").json() == {"stopping": True}
    assert _poll_job(client, job_id)["status"] == "done"
    finished = api_app.sessions.load_session(session_id)
    assert finished["active_job_id"] is None
    assert finished["last_job"]["status"] == "done"


def test_continue_unknown_session_is_404(client):
    assert client.post("/sessions/nope/continue", json={"generations": 1}).status_code == 404


def test_report_summary_cache_roundtrip_and_context_invalidation(signal_panel, tmp_path):
    from alphalineage.api import sessions
    from alphalineage.core.gp import GPConfig
    from alphalineage.validation.pbo import summarize_report_returns

    panel, _ = signal_panel
    boundaries = sessions.derive_boundaries(panel.dates, embargo=3, horizon=1)
    session = {"universe": "cache-test", "as_of": "2019-12-31"}
    config = GPConfig(population_size=10, generations=1, horizon=1)
    context = sessions._report_cache_context(session, panel, boundaries, config)
    summary = summarize_report_returns(
        pd.Series(0.001, index=panel.dates), panel.dates, n_blocks=4
    )
    path = tmp_path / "report_stats.json"

    sessions._save_report_cache(path, context, {"factor": summary})
    assert sessions._load_report_cache(path, context) == {"factor": summary}

    changed = sessions._report_cache_context(
        session,
        panel,
        boundaries,
        GPConfig(population_size=10, generations=1, horizon=2),
    )
    assert changed != context
    assert sessions._load_report_cache(path, changed) == {}
