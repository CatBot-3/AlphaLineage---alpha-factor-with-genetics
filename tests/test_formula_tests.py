"""Unified Formula Builder backend: exploratory tests, results, and composition safety."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

import alphalineage.api.app as api_app
from alphalineage.api.app import app, get_panel
from alphalineage.core import extensions
from alphalineage.core.primitives import OPERATORS, REGISTRY
from alphalineage.data import paths


@pytest.fixture(autouse=True)
def _restore_registry():
    ops, reg, users = dict(OPERATORS), dict(REGISTRY), dict(extensions.USER_OPERATORS)
    yield
    for table, saved in ((OPERATORS, ops), (REGISTRY, reg), (extensions.USER_OPERATORS, users)):
        table.clear()
        table.update(saved)


@pytest.fixture
def client(synthetic_panel):
    app.dependency_overrides[get_panel] = lambda: synthetic_panel
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _wait(client: TestClient, job_id: str) -> dict:
    for _ in range(300):
        payload = client.get(f"/formula-tests/{job_id}").json()
        if payload["status"] in {"done", "failed", "stopped"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("formula test did not finish")


def test_unsaved_draft_test_keep_and_formula_results_alias(client):
    response = client.post(
        "/formula-tests",
        json={
            "source": {
                "kind": "draft",
                "body": {"name": "rank", "children": [{"name": "close"}]},
                "inputs": [],
                "out_type": "signal",
            },
            "bindings": {},
        },
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    completed = _wait(client, job_id)
    assert completed["status"] == "done", completed
    result = completed["result"]
    assert result["kind"] == "backtest"
    assert result["exploratory"] is True
    assert result["metrics"]["net_sharpe"] is not None
    assert len(result["returns"]) == 59
    assert len(result["normalized_equity"]) == 60
    assert result["normalized_equity"][0]["value"] == 1.0
    assert result["returns"][0]["signal_date"] == result["normalized_equity"][0]["date"]
    assert result["returns"][0]["date"] == result["normalized_equity"][1]["date"]
    assert result["dependency_revisions"] == []

    kept = client.post(
        f"/formula-tests/{job_id}/keep", json={"name": "Close rank test", "notes": "kept"}
    )
    assert kept.status_code == 200, kept.text
    saved = kept.json()
    assert saved["kind"] == "backtest"
    assert saved["returns"] == result["returns"]
    assert saved["provenance"]["test_reads"] == 0
    assert any(item["id"] == saved["id"] for item in client.get("/formula-results").json())
    assert client.get(f"/factors/{saved['id']}").json()["id"] == saved["id"]

    reused = client.post(
        "/formula-tests",
        json={
            "source": {
                "kind": "draft",
                "body": {"name": "rank", "children": [{"name": "$arg", "value": 0}]},
                "inputs": [{"name": "prior", "type": "series"}],
                "out_type": "signal",
            },
            "bindings": {"prior": {"kind": "result", "result_id": saved["id"]}},
        },
    )
    assert reused.status_code == 200, reused.text
    reused_id = reused.json()["job_id"]
    assert _wait(client, reused_id)["status"] == "done"

    assert client.delete(f"/formula-tests/{job_id}").status_code == 200
    assert client.delete(f"/formula-tests/{reused_id}").status_code == 200
    assert client.delete(f"/formula-results/{saved['id']}").status_code == 200


def test_formula_test_supports_ordered_strategy_set_with_primary_alias(client):
    response = client.post(
        "/formula-tests",
        json={
            "source": {
                "kind": "draft",
                "body": {"name": "rank", "children": [{"name": "close"}]},
                "inputs": [],
                "out_type": "signal",
            },
            "bindings": {},
            "strategies": [
                {"id": "q20", "scheme": "quantile_ls", "quantile": 0.2},
                {"id": "rank", "scheme": "rank_proportional"},
            ],
            "primary_strategy_id": "rank",
        },
    )
    assert response.status_code == 200, response.text
    completed = _wait(client, response.json()["job_id"])
    assert completed["status"] == "done", completed
    result = completed["result"]
    assert result["primary_strategy_id"] == "rank"
    assert result["weighting_scheme"] == "rank_proportional"
    assert result["quantile"] is None
    assert result["configuration"]["weighting_scheme"] == "rank_proportional"
    assert result["configuration"]["quantile"] is None
    assert [item["strategy_id"] for item in result["strategy_results"]] == [
        "q20",
        "rank",
    ]
    primary = next(
        item for item in result["strategy_results"] if item["role"] == "primary"
    )
    assert result["oos_backtest"] == primary["oos_backtest"]
    assert result["metrics"] == primary["oos_backtest"]["metrics"]


def test_formula_test_uses_pre_start_history_only_for_rolling_warmup(
    client, synthetic_panel, monkeypatch
):
    start = synthetic_panel.dates[10]
    end = synthetic_panel.dates[19]
    captured: dict[str, object] = {}
    real_evaluate = api_app.evaluate

    def capture_evaluate(tree, panel):
        factor = real_evaluate(tree, panel)
        captured["panel_start"] = panel.dates.min()
        captured["first_report_ready"] = bool(factor.loc[start].notna().all())
        return factor

    monkeypatch.setattr(api_app, "evaluate", capture_evaluate)
    submitted = client.post(
        "/formula-tests",
        json={
            "source": {
                "kind": "draft",
                "body": {
                    "name": "ts_mean",
                    "children": [
                        {"name": "close"},
                        {"name": "window", "value": 5},
                    ],
                },
                "inputs": [],
                "out_type": "series",
            },
            "bindings": {},
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
        },
    )
    assert submitted.status_code == 200, submitted.text
    completed = _wait(client, submitted.json()["job_id"])
    assert completed["status"] == "done", completed
    result = completed["result"]

    assert captured["panel_start"] == synthetic_panel.dates.min()
    assert captured["first_report_ready"] is True
    expected_dates = [date.date().isoformat() for date in synthetic_panel.dates[10:20]]
    assert [item["signal_date"] for item in result["returns"]] == expected_dates[:-1]
    assert [item["date"] for item in result["returns"]] == expected_dates[1:]
    assert [item["date"] for item in result["normalized_equity"]] == expected_dates
    assert result["normalized_equity"][0]["value"] == 1.0
    assert result["data_coverage"]["first_date"] == expected_dates[0]
    assert result["data_coverage"]["last_date"] == expected_dates[-1]
    assert result["data_coverage"]["observations"] == 10
    assert result["data_coverage"]["warmup_observations"] == 20


def test_formula_test_horizon_uses_cumulative_ic_and_next_session_realization(
    client, synthetic_panel
):
    from alphalineage.core.evaluate import evaluate
    from alphalineage.core.fitness import daily_ic
    from alphalineage.core.panel import Panel
    from alphalineage.core.tree import Node

    horizon = 3
    start = synthetic_panel.dates[10]
    end = synthetic_panel.dates[24]
    submitted = client.post(
        "/formula-tests",
        json={
            "source": {
                "kind": "draft",
                "body": {"name": "rank", "children": [{"name": "close"}]},
                "inputs": [],
                "out_type": "signal",
            },
            "bindings": {},
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "horizon": horizon,
        },
    )
    assert submitted.status_code == 200, submitted.text
    completed = _wait(client, submitted.json()["job_id"])
    assert completed["status"] == "done", completed
    result = completed["result"]

    # Formula tests intentionally truncate the warm-up panel at the requested end date. Build
    # that panel and the horizon target directly so no future observations leak into expected IC.
    warmup_panel = Panel(
        {
            name: frame.loc[frame.index <= end]
            for name, frame in synthetic_panel.fields.items()
        }
    )
    factor = evaluate(Node("rank", (Node("close"),)), warmup_panel)
    assert hasattr(factor, "columns")
    cumulative_target = (
        warmup_panel["close"].shift(-horizon).div(warmup_panel["close"]).sub(1.0)
    )
    legacy_offset_target = warmup_panel["returns"].shift(-horizon)
    report_dates = synthetic_panel.dates[10:25]
    expected_ic = daily_ic(
        factor, cumulative_target, "spearman", min_names=5
    ).reindex(report_dates)
    legacy_ic = daily_ic(
        factor, legacy_offset_target, "spearman", min_names=5
    ).reindex(report_dates)

    assert float(expected_ic.mean()) != pytest.approx(float(legacy_ic.mean()), abs=1e-6)
    assert result["horizon"] == horizon
    assert result["metrics"]["signed_ic"] == pytest.approx(float(expected_ic.mean()))
    assert result["metrics"]["mean_abs_ic"] == pytest.approx(float(expected_ic.abs().mean()))

    expected_dates = [date.date().isoformat() for date in report_dates]
    # Horizon changes the IC target and holding overlap, not the frequency at which the
    # staggered portfolio realizes returns.
    assert [item["signal_date"] for item in result["returns"]] == expected_dates[:-1]
    assert [item["date"] for item in result["returns"]] == expected_dates[1:]
    assert [item["date"] for item in result["normalized_equity"]] == expected_dates


def test_saved_formula_requires_explicit_bindings_even_with_numeric_default(client):
    formula = {
        "name": "moving_avg_test",
        "arg_types": ["series", "window"],
        "inputs": [
            {"name": "price", "type": "series"},
            {"name": "lookback", "type": "window", "default": 10},
        ],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    saved = client.post("/formulas", json=formula)
    assert saved.status_code == 200, saved.text

    missing = client.post(
        "/formula-tests",
        json={
            "source": {"kind": "saved", "runtime_name": "moving_avg_test"},
            "bindings": {"price": {"kind": "field", "field": "close"}},
        },
    )
    assert missing.status_code == 400
    assert "missing bindings: lookback" in missing.json()["detail"]

    submitted = client.post(
        "/formula-tests",
        json={
            "source": {"kind": "saved", "runtime_name": "moving_avg_test"},
            "bindings": {
                "price": {"kind": "field", "field": "close"},
                "lookback": {"kind": "literal", "value": 10},
            },
            "weighting_scheme": "rank_proportional",
        },
    )
    assert submitted.status_code == 200, submitted.text
    result = _wait(client, submitted.json()["job_id"])["result"]
    assert result["dependency_revisions"] == ["moving_avg_test"]
    assert result["weighting_scheme"] == "rank_proportional"
    # The persisted/evaluated tree is expanded at submission, so later formula edits cannot
    # change an active or kept backtest.
    assert result["tree"]["name"] == "ts_mean"


def test_calculation_update_creates_immutable_revision_without_dependents(client):
    formula = {
        "name": "immutable_test",
        "arg_types": ["series"],
        "out_type": "signal",
        "body": {"name": "rank", "children": [{"name": "$arg", "value": 0}]},
    }
    assert client.post("/formulas", json=formula).status_code == 200
    changed = {
        **formula,
        "body": {
            "name": "rank",
            "children": [
                {
                    "name": "delta",
                    "children": [
                        {"name": "$arg", "value": 0},
                        {"name": "window", "value": 2},
                    ],
                }
            ],
        },
    }
    updated = client.put("/formulas/immutable_test", json=changed)
    assert updated.status_code == 200, updated.text
    assert updated.json()["runtime_name"] == "immutable_test__r2"
    detail = client.get("/formulas/immutable_test").json()
    assert [item["runtime_name"] for item in detail["revisions"]] == [
        "immutable_test",
        "immutable_test__r2",
    ]


def test_persisted_formula_cycle_is_visible_and_not_registered(client):
    def spec(name: str, dependency: str) -> dict:
        return {
            "name": name,
            "display_name": name,
            "arg_types": ["series"],
            "inputs": [{"name": "series", "type": "series"}],
            "out_type": "series",
            "body": {
                "name": dependency,
                "children": [{"name": "$arg", "value": 0}],
            },
            "revision": 1,
            "runtime_name": name,
        }

    payload = {
        "schema_version": 2,
        "families": [
            {"name": "cycle_a", "latest_revision": 1, "revisions": [spec("cycle_a", "cycle_b")]},
            {"name": "cycle_b", "latest_revision": 1, "revisions": [spec("cycle_b", "cycle_a")]},
        ],
    }
    paths.meta_dir().mkdir(parents=True, exist_ok=True)
    paths.formulas_path().write_text(json.dumps(payload), encoding="utf-8")

    listed = client.get("/formulas")
    assert listed.status_code == 200
    by_name = {item["name"]: item for item in listed.json()}
    assert by_name["cycle_a"]["registered"] is False
    assert by_name["cycle_b"]["registered"] is False
    assert "cycle_a -> cycle_b -> cycle_a" in by_name["cycle_a"]["error"]
    assert "cycle_a" not in extensions.USER_OPERATORS
    assert "cycle_b" not in extensions.USER_OPERATORS


def test_formula_default_validation_is_numeric_only(client):
    bad_panel_default = {
        "name": "bad_default_test",
        "arg_types": ["series"],
        "inputs": [{"name": "series", "type": "series", "default": 1}],
        "out_type": "signal",
        "body": {"name": "rank", "children": [{"name": "$arg", "value": 0}]},
    }
    response = client.post("/formulas", json=bad_panel_default)
    assert response.status_code == 400
    assert "only scalar and window inputs" in response.json()["detail"]
