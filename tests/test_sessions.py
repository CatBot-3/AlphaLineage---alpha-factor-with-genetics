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


def _finalize(
    client: TestClient,
    session_id: str,
    round_index: int = 0,
    *,
    confirm_repeat: bool = False,
) -> dict:
    response = client.post(
        f"/sessions/{session_id}/rounds/{round_index}/finalize",
        json={"confirm_repeat": confirm_repeat},
    )
    assert response.status_code == 200, response.text
    final = _poll_job(client, response.json()["job_id"])
    assert final["status"] == "done", final
    return final


def _downgrade_to_legacy_embedded_report(
    client: TestClient,
    session_id: str,
) -> dict:
    import json

    finalized = _finalize(client, session_id)["result"]
    legacy = client.get(f"/sessions/{session_id}/rounds/0").json()
    legacy["report"] = finalized["report"]
    legacy["oos_backtest"] = finalized["oos_backtest"]
    legacy["test_reads"] = finalized["test_reads"]
    legacy["test_read_index"] = 1
    legacy.pop("round_index", None)
    metadata = dict(legacy.pop("round_metadata", {}))
    metadata.update(
        {
            "report_available": True,
            "test_read_index": 1,
            "evidence_status": "locked",
        }
    )
    legacy["round_metadata"] = metadata

    state = api_app.sessions.load_session(session_id)
    state.pop("rounds", None)
    state.pop("finalizations", None)
    api_app.sessions.save_session(state)
    result_path = api_app.sessions.session_dir(session_id) / "result.json"
    result_path.write_text(json.dumps(legacy), encoding="utf-8")
    api_app.sessions.round_path(session_id, 0).unlink()
    finalization_path = api_app.sessions.finalization_path(
        session_id,
        finalized["evaluation_id"],
    )
    finalization_path.unlink()
    return legacy


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
    assert state["test_reads"] == 0
    assert len(state["segments"]) == 1 and state["segments"][0]["status"] == "done"

    result = final["result"]
    assert result["report"] is None
    assert result["validation_only"] is True
    assert result["session_id"] == session_id
    assert result["test_reads"] == 0

    finalized = _finalize(client, session_id)
    assert "deflated_sharpe" in finalized["result"]["report"]
    assert finalized["result"]["evidence_status"] == "locked_first_read"
    compatibility_plan_id = finalized["result"]["strategy_plan_id"]
    assert compatibility_plan_id.startswith("compat-q20-r0-")
    compatibility_plan = client.get(
        f"/sessions/{session_id}/rounds/0/finalization-plans/"
        f"{compatibility_plan_id}"
    ).json()
    assert compatibility_plan["source"] == "compatibility_default"
    assert compatibility_plan["strategies"] == [
        {"id": "quantile-20", "scheme": "quantile_ls", "quantile": 0.2}
    ]
    reloaded = client.get(f"/sessions/{session_id}").json()
    assert reloaded["test_reads"] == 1
    assert reloaded["finalization_job"]["id"] == finalized["job_id"]
    assert (
        reloaded["finalization_job"]["metadata"]["evaluation_id"]
        == finalized["result"]["evaluation_id"]
    )
    assert reloaded["finalization_job"]["metadata"]["round_index"] == 0


def test_completed_validation_rounds_are_immutable_with_latest_alias(client):
    session_id, first = _create(client)

    summaries = client.get(f"/sessions/{session_id}/rounds")
    assert summaries.status_code == 200
    assert len(summaries.json()) == 1
    summary = summaries.json()[0]
    assert summary["index"] == 0
    assert summary["segment_index"] == 0
    assert summary["report_available"] is False
    assert summary["validation_only"] is True
    assert summary["test_read_index"] is None
    assert summary["evidence_status"] == "validation_only"
    assert summary["legacy"] is False
    first_round = client.get(f"/sessions/{session_id}/rounds/0").json()
    assert first_round["round_index"] == 0
    assert first_round["segment"] == 0
    assert first_round["report"] is None
    assert first_round["selection"] == first["result"]["selection"]

    continued = client.post(
        f"/sessions/{session_id}/continue", json={"generations": 1}
    ).json()
    second = _poll_job(client, continued["job_id"])
    assert second["status"] == "done", second

    rounds = client.get(f"/sessions/{session_id}/rounds").json()
    assert [item["index"] for item in rounds] == [0, 1]
    assert [item["evidence_status"] for item in rounds] == [
        "validation_only",
        "validation_only",
    ]
    assert [item["test_read_index"] for item in rounds] == [None, None]
    assert client.get(f"/sessions/{session_id}/rounds/0").json() == first_round

    latest = client.get(f"/sessions/{session_id}").json()["result"]
    assert latest["round_index"] == 1
    assert latest["report"] is None
    assert client.get(f"/sessions/{session_id}/finalizations").json() == []

    _finalize(client, session_id, 0)
    repeat = client.post(f"/sessions/{session_id}/rounds/1/finalize", json={})
    assert repeat.status_code == 409
    _finalize(client, session_id, 1, confirm_repeat=True)
    finalizations = client.get(f"/sessions/{session_id}/finalizations").json()
    assert [item["evidence_status"] for item in finalizations] == [
        "locked_first_read",
        "repeated_same_holdout",
    ]
    assert [item["same_holdout_read_index"] for item in finalizations] == [1, 2]


def test_strategy_comparison_plan_and_multistrategy_finalization_are_pinned(
    client, monkeypatch
):
    session_id, _ = _create(client)
    real_compare = api_app.sessions.compare_validation_strategies

    def eligible_fixture(*args, **kwargs):
        results = real_compare(*args, **kwargs)
        for result in results:
            result["eligible"] = True
            for fold in result["folds"]:
                fold["adequate_coverage"] = True
                fold["coverage_failures"] = []
        return results

    # The compact 180-date session fixture intentionally cannot supply three
    # 60-date validation folds. This test isolates artifact pinning; fold-gate
    # behavior is covered independently.
    monkeypatch.setattr(
        api_app.sessions,
        "compare_validation_strategies",
        eligible_fixture,
    )
    comparison = client.post(
        f"/sessions/{session_id}/rounds/0/strategy-comparisons",
        json={
            "strategies": [
                {"id": "q20", "scheme": "quantile_ls", "quantile": 0.2},
                {"id": "rank", "scheme": "rank_proportional"},
            ]
        },
    )
    assert comparison.status_code == 200, comparison.text
    handle = comparison.json()
    completed = _poll_job(client, handle["job_id"])
    assert completed["status"] == "done", completed
    detail = client.get(
        f"/sessions/{session_id}/rounds/0/strategy-comparisons/"
        f"{handle['comparison_id']}"
    ).json()
    assert detail["status"] == "done"
    assert [item["strategy_id"] for item in detail["strategy_results"]] == [
        "q20",
        "rank",
    ]
    assert all(item["oos_backtest"] is None for item in detail["strategy_results"])
    assert all("validation_backtest" in item for item in detail["strategy_results"])

    plan_response = client.post(
        f"/sessions/{session_id}/rounds/0/finalization-plans",
        json={
            "comparison_id": handle["comparison_id"],
            "primary_strategy_id": "rank",
        },
    )
    assert plan_response.status_code == 200, plan_response.text
    plan = plan_response.json()
    assert plan["primary_strategy_id"] == "rank"
    loaded_plan = client.get(
        f"/sessions/{session_id}/rounds/0/finalization-plans/"
        f"{plan['strategy_plan_id']}"
    ).json()
    assert loaded_plan == plan

    queued = client.post(
        f"/sessions/{session_id}/rounds/0/finalize",
        json={"strategy_plan_id": plan["strategy_plan_id"]},
    )
    assert queued.status_code == 200, queued.text
    assert queued.json()["comparison_id"] == handle["comparison_id"]
    finalized = _poll_job(client, queued.json()["job_id"])
    assert finalized["status"] == "done", finalized
    artifact = finalized["result"]
    assert artifact["strategy_plan_id"] == plan["strategy_plan_id"]
    assert artifact["comparison_id"] == handle["comparison_id"]
    assert artifact["primary_strategy_id"] == "rank"
    assert [item["strategy_id"] for item in artifact["strategy_results"]] == [
        "q20",
        "rank",
    ]
    primary = next(
        item for item in artifact["strategy_results"] if item["role"] == "primary"
    )
    assert artifact["oos_backtest"] == primary["oos_backtest"]
    assert artifact["report"]["n_trials"] % 2 == 0

    round_summary = client.get(f"/sessions/{session_id}/rounds").json()[0]
    assert round_summary["latest_strategy_comparison_id"] == handle["comparison_id"]
    assert round_summary["latest_strategy_plan_id"] == plan["strategy_plan_id"]

    blocked = client.post(
        f"/sessions/{session_id}/rounds/0/strategy-comparisons",
        json={
            "strategies": [
                {"id": "q25", "scheme": "quantile_ls", "quantile": 0.25}
            ]
        },
    )
    assert blocked.status_code == 409
    assert "confirm_repeat=true" in blocked.text
    confirmed = client.post(
        f"/sessions/{session_id}/rounds/0/strategy-comparisons",
        json={
            "strategies": [
                {"id": "q25", "scheme": "quantile_ls", "quantile": 0.25}
            ],
            "confirm_repeat": True,
        },
    )
    assert confirmed.status_code == 200, confirmed.text
    confirmed_job = _poll_job(client, confirmed.json()["job_id"])
    assert confirmed_job["status"] == "done", confirmed_job
    confirmed_detail = client.get(
        f"/sessions/{session_id}/rounds/0/strategy-comparisons/"
        f"{confirmed.json()['comparison_id']}"
    ).json()
    assert confirmed_detail["evidence_status"] == "post_holdout_adaptive"
    assert confirmed_detail["prior_holdout_reads"] == 1


def test_round_lineage_is_truncated_at_its_generation_boundary(client):
    session_id, _ = _create(client)
    continued = client.post(
        f"/sessions/{session_id}/continue", json={"generations": 1}
    ).json()
    assert _poll_job(client, continued["job_id"])["status"] == "done"

    first = client.get(f"/sessions/{session_id}/lineage?round=0")
    assert first.status_code == 200
    payload = first.json()
    boundary = payload["metadata"]["generation_boundary"]
    assert boundary == _SMALL["generations"]
    assert max(node["generation"] for node in payload["nodes"]) == boundary
    assert payload["metadata"]["selected_lineage_node_id"] is not None

    assert client.get(f"/sessions/{session_id}/lineage?round=99").status_code == 404
    assert client.get(f"/sessions/{session_id}/rounds/99").status_code == 404


def test_session_summary_exposes_resume_metadata(client):
    session_id, _ = _create(client)
    summary = next(item for item in client.get("/sessions").json() if item["id"] == session_id)
    assert summary["updated_at"] >= summary["created_at"]
    assert summary["current_generation"] == _SMALL["generations"]
    assert summary["last_status"] == "done"
    assert summary["has_checkpoint"] is True
    assert summary["has_report"] is False
    assert summary["latest_completed_round"] == 0
    assert summary["requested_generations"] == _SMALL["generations"]
    _finalize(client, session_id)
    updated = next(item for item in client.get("/sessions").json() if item["id"] == session_id)
    assert updated["has_report"] is True
    assert updated["finalizations"] == 1


def test_legacy_latest_result_is_exposed_and_pinned_before_continue(client):
    session_id, _ = _create(client)
    _downgrade_to_legacy_embedded_report(client, session_id)

    before = client.get(f"/sessions/{session_id}/rounds").json()
    assert len(before) == 1 and before[0]["index"] == 0
    assert before[0]["legacy"] is True
    assert client.get(f"/sessions/{session_id}/rounds/0").json()["round_index"] == 0

    continued = client.post(
        f"/sessions/{session_id}/continue", json={"generations": 1}
    ).json()
    assert _poll_job(client, continued["job_id"])["status"] == "done"
    after = client.get(f"/sessions/{session_id}/rounds").json()
    assert [item["index"] for item in after] == [0, 1]
    assert api_app.sessions.round_path(session_id, 0).exists()
    legacy_evidence = client.get(f"/sessions/{session_id}/finalizations").json()
    assert legacy_evidence[0]["evaluation_id"] == "legacy-round-0"


def test_contaminated_legacy_report_is_viewable_but_requires_restart(client):
    import json

    session_id, _ = _create(client)
    _downgrade_to_legacy_embedded_report(client, session_id)
    state = api_app.sessions.load_session(session_id)
    for key in ("scorer_version", "evolution_version", "adjustment_version"):
        state["segments"][-1].pop(key, None)
    api_app.sessions.save_session(state)

    result_path = api_app.sessions.session_dir(session_id) / "result.json"
    legacy = json.loads(result_path.read_text(encoding="utf-8"))
    for key in ("scorer_version", "evolution_version", "adjustment_version"):
        legacy["round_metadata"].pop(key, None)
    result_path.write_text(json.dumps(legacy), encoding="utf-8")

    summary = client.get(f"/sessions/{session_id}/rounds").json()[0]
    assert summary["validity"] == "invalid_legacy_semantics"
    assert summary["restart_required"] is True
    detail = client.get(f"/sessions/{session_id}/rounds/0").json()
    assert detail["validity"] == "invalid_legacy_semantics"

    continued = client.post(
        f"/sessions/{session_id}/continue", json={"generations": 1}
    )
    assert continued.status_code == 409
    assert "Restart with the same setup" in continued.json()["detail"]


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
    # Inherited research/evidence baselines remain separate until explicit finalization.
    assert final["result"]["searched_trials"] >= 500
    state = client.get(f"/sessions/{session_id}").json()
    assert state["inherited_test_reads"] == 1
    assert state["session_holdout_reads"] == 0
    assert state["test_reads"] == 1


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


def test_invalid_formula_allow_list_does_not_persist_an_orphan_session(client):
    from alphalineage.data import paths

    response = client.post(
        "/sessions",
        json={
            "name": "invalid allow list",
            "universe": "sp500-lite",
            "config": {
                **_SMALL,
                "enabled_categories": ["technical_indicators"],
                "enabled_formula_names": ["not_a_formula"],
            },
        },
    )
    assert response.status_code == 400
    root = paths.sessions_dir()
    assert not root.exists() or not any(root.iterdir())


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


def test_continue_trials_increase_without_reading_holdout(client):
    session_id, first = _create(client)
    trials_after_first = first["result"]["cumulative_trials"]

    payload = client.post(f"/sessions/{session_id}/continue", json={"generations": 2}).json()
    final = _poll_job(client, payload["job_id"])
    assert final["status"] == "done", final

    assert final["result"]["cumulative_trials"] >= trials_after_first  # monotone (P2)
    assert final["result"]["test_reads"] == 0
    assert final["result"]["repeated_oos_warning"] is False
    assert client.get(f"/sessions/{session_id}/finalizations").json() == []


def test_rounds_after_first_holdout_read_are_post_holdout_adaptive(client):
    session_id, _ = _create(client)
    _finalize(client, session_id)

    continued = client.post(
        f"/sessions/{session_id}/continue",
        json={"generations": 1},
    ).json()
    assert _poll_job(client, continued["job_id"])["status"] == "done"
    second_round = client.get(f"/sessions/{session_id}/rounds/1").json()
    assert second_round["evidence_status"] == "post_holdout_adaptive"
    assert second_round["round_metadata"]["evidence_status"] == "post_holdout_adaptive"

    unconfirmed = client.post(
        f"/sessions/{session_id}/rounds/1/finalize",
        json={},
    )
    assert unconfirmed.status_code == 409
    repeated = _finalize(
        client,
        session_id,
        1,
        confirm_repeat=True,
    )["result"]
    assert repeated["evidence_status"] == "repeated_same_holdout"
    assert repeated["source_round_evidence_status"] == "post_holdout_adaptive"
    assert repeated["report"]["significant"] is False


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
    session_id, _ = _create(client)
    base_report = _finalize(client, session_id)["result"]["report"]
    base_trials = base_report["n_trials"]

    cont = client.post(
        f"/sessions/{session_id}/continue",
        json={"generations": 1, "operators": [_MOMENTUM]},
    )
    final = _poll_job(client, cont.json()["job_id"])
    assert final["status"] == "done", final
    # a larger operator palette inflates the effective trial count (invariant 1, P7-T3)
    repeated_report = _finalize(
        client,
        session_id,
        1,
        confirm_repeat=True,
    )["result"]["report"]
    assert repeated_report["n_trials"] > base_trials


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


def test_cancelled_finalization_does_not_publish_partial_evidence(client, monkeypatch):
    from alphalineage.core.gp import TrainingCancelled

    def cancel_report(*args, **kwargs):
        raise TrainingCancelled()

    monkeypatch.setattr("alphalineage.api.sessions.build_report", cancel_report)
    body = {"name": "cancel-report", "universe": "sp500-lite", "config": _SMALL}
    created = client.post("/sessions", json=body).json()
    final = _poll_job(client, created["job_id"])
    assert final["status"] == "done"
    assert final["result"]["report"] is None

    session_id = created["session_id"]
    request = client.post(
        f"/sessions/{session_id}/rounds/0/finalize",
        json={},
    )
    stopped = _poll_job(client, request.json()["job_id"])
    assert stopped["status"] == "stopped"
    assert client.get(f"/sessions/{session_id}/rounds").json()
    assert client.get(f"/sessions/{session_id}/finalizations").json() == []
    state_payload = client.get(f"/sessions/{session_id}").json()
    assert state_payload["result"]["round_index"] == 0
    assert state_payload["test_reads"] == 0
    state = api_app.sessions.load_session(session_id)
    assert state["active_finalization_job_id"] is None


def test_failed_attempt_is_retained_without_a_round(client, monkeypatch):
    def fail_segment(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(api_app.sessions, "run_segment", fail_segment)
    created = client.post(
        "/sessions",
        json={"name": "failed-round", "universe": "sp500-lite", "config": _SMALL},
    ).json()
    final = _poll_job(client, created["job_id"])
    assert final["status"] == "failed"

    session_id = created["session_id"]
    state = api_app.sessions.load_session(session_id)
    assert state["segments"][-1]["status"] == "failed"
    assert state["segments"][-1]["report_available"] is False
    assert client.get(f"/sessions/{session_id}/rounds").json() == []


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
