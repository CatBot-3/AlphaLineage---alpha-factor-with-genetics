"""Overlap check: correlation with known factors and the IC that is left after removing them."""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import alphalineage.api.app as api_app
from alphalineage.api.app import app, get_panel
from alphalineage.core import extensions
from alphalineage.core.tree import Node
from alphalineage.library.overlap import (
    ReferenceFactor,
    overlap_report,
    references_from_formulas,
    residualize,
    verdict,
)


def _frames(seed: int = 0, days: int = 250, names: int = 60):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="B")
    symbols = [f"S{i}" for i in range(names)]

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=dates, columns=symbols)

    known = frame(rng.normal(size=(days, names)))
    other = frame(rng.normal(size=(days, names)))
    novel = frame(rng.normal(size=(days, names)))
    # Tomorrow's return loads on both the known factor and the genuinely new one.
    forward = frame(
        0.3 * known.to_numpy() + 0.3 * novel.to_numpy() + rng.normal(size=(days, names))
    )
    return dates, known, other, novel, forward


def _reference(key: str) -> ReferenceFactor:
    return ReferenceFactor(key, key, key.title(), "catalog", "test", Node("close"))


def test_near_duplicate_is_flagged_and_explains_away_its_ic() -> None:
    dates, known, other, _, forward = _frames()
    rng = np.random.default_rng(1)
    disguised = known * 3.0 + 0.05 * rng.normal(size=known.shape)
    report = overlap_report(
        disguised,
        [(_reference("known"), known, None), (_reference("other"), other, None)],
        forward,
        dates,
        min_names=10,
    )
    top = report["references"][0]
    assert top["key"] == "known" and top["abs_mean_rank_corr"] > 0.95
    assert report["verdict"] == "near_duplicate"
    assert report["residual"]["explained_by"] == ["known"]
    assert report["candidate"]["ic"] > 0.1
    assert abs(report["residual"]["unique_share"]) < 0.3
    assert report["residual"]["mean_r_squared"] > 0.9
    assert report["window"]["dates"] == len(dates)


def test_inverted_duplicate_keeps_its_sign_and_novel_factor_keeps_its_ic() -> None:
    dates, known, other, novel, forward = _frames(seed=2)
    inverted = overlap_report(
        -known, [(_reference("known"), known, None)], forward, dates, min_names=10
    )
    assert inverted["references"][0]["mean_rank_corr"] < -0.95

    blend = novel + 0.3 * known
    report = overlap_report(
        blend,
        [(_reference("known"), known, None), (_reference("other"), other, None)],
        forward,
        dates,
        min_names=10,
    )
    assert report["verdict"] in {"novel", "related"}
    # Removing the known component leaves most of the edge, which came from the novel part.
    assert report["residual"]["unique_share"] > 0.5
    assert report["residual"]["ic"] > 0.05

    unrelated = overlap_report(
        novel, [(_reference("other"), other, None)], forward, dates, min_names=10
    )
    assert unrelated["verdict"] == "novel"
    assert unrelated["residual"]["explained_by"] == []
    assert unrelated["residual"]["unique_share"] == pytest.approx(1.0)


def test_residualize_removes_linear_rank_structure_and_reports_failures() -> None:
    dates, known, other, _, _ = _frames(seed=3, days=20, names=30)
    ranks_known = known.rank(axis=1, pct=True)
    ranks_other = other.rank(axis=1, pct=True)
    combined = 2.0 * ranks_known - ranks_other
    residual, r_squared = residualize(combined, [known, other], min_names=10)
    # The candidate is re-ranked first, so the fit is close to but not exactly perfect.
    assert r_squared.min() > 0.9
    assert residual.abs().max().max() < 0.2

    report = overlap_report(
        known,
        [(_reference("broken"), None, "unknown operator"), (_reference("other"), other, None)],
        known,
        dates,
        min_names=10,
    )
    statuses = {row["key"]: row["status"] for row in report["references"]}
    assert statuses == {"broken": "unavailable", "other": "ok"}
    assert report["measured_count"] == 1 and report["reference_count"] == 2
    assert verdict(None, None) == "unmeasured"


def test_references_use_numeric_defaults_and_skip_unbindable_formulas() -> None:
    formulas = [
        {
            "name": "ta_sma",
            "runtime_name": "ta_sma__r2",
            "origin": "catalog_formula",
            "family": "moving_averages",
            "out_type": "series",
            "status": "active",
            "inputs": [{"name": "lookback", "type": "window", "default": 20}],
        },
        {
            "name": "needs_series",
            "origin": "user_formula",
            "out_type": "series",
            "inputs": [{"name": "series", "type": "series", "default": None}],
        },
        {"name": "old", "origin": "catalog_formula", "out_type": "series", "status": "retired"},
        {"name": "mask", "origin": "user_formula", "out_type": "bool", "inputs": []},
        {
            "name": "mine",
            "origin": "user_formula",
            "out_type": "signal",
            "inputs": [{"name": "k", "type": "scalar", "default": 2}],
        },
    ]
    references = {item.name: item for item in references_from_formulas(formulas)}
    assert set(references) == {"ta_sma", "mine"}
    assert references["ta_sma"].tree == Node("ta_sma__r2", (Node("window", value=20),))
    assert references["ta_sma"].group == "catalog"
    assert references["mine"].tree == Node("mine", (Node("const", value=2.0),))
    assert references["mine"].group == "your_formulas"


# --- API --------------------------------------------------------------------------------------
_SMALL = {"population_size": 16, "generations": 2, "max_depth": 4, "max_nodes": 20, "seed": 0}


@pytest.fixture
def client(signal_panel) -> Iterator[TestClient]:
    panel, _ = signal_panel
    app.dependency_overrides[get_panel] = lambda: panel
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    extensions.clear_user_operators()


def _poll(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/runs/{job_id}").json()
        if payload["status"] in ("done", "stopped", "failed"):
            return payload
        time.sleep(0.1)
    return payload


def test_round_overlap_reads_only_training_rows_and_persists(client, monkeypatch) -> None:
    created = client.post(
        "/sessions", json={"name": "overlap", "universe": "sp500-lite", "config": _SMALL}
    ).json()
    assert _poll(client, created["job_id"])["status"] == "done"
    session_id = created["session_id"]
    assert client.get(f"/sessions/{session_id}/rounds/0/overlap").status_code == 404

    session = client.get(f"/sessions/{session_id}").json()
    train_end = pd.Timestamp(session["boundaries"]["train_end"])
    seen_last_dates: list[pd.Timestamp] = []
    original_evaluate = api_app.evaluate

    def recording_evaluate(tree, panel):
        seen_last_dates.append(pd.DatetimeIndex(panel.dates).max())
        return original_evaluate(tree, panel)

    monkeypatch.setattr(api_app, "evaluate", recording_evaluate)
    started = client.post(f"/sessions/{session_id}/rounds/0/overlap", json={})
    assert started.status_code == 200, started.text
    final = _poll(client, started.json()["job_id"])
    assert final["status"] == "done", final
    result = final["result"]

    assert seen_last_dates and max(seen_last_dates) <= train_end
    assert pd.Timestamp(result["window"]["end"]) <= train_end
    assert result["execution"] == "close"
    groups = {row["group"] for row in result["references"]}
    assert "catalog" in groups
    assert result["measured_count"] > 30
    assert result["verdict"] in {"novel", "related", "mostly_explained", "near_duplicate"}
    ranked = [row["abs_mean_rank_corr"] for row in result["references"] if row["status"] == "ok"]
    assert ranked == sorted(ranked, reverse=True)

    stored = client.get(f"/sessions/{session_id}/rounds/0/overlap").json()
    assert stored["stale"] is False
    assert stored["verdict"] == result["verdict"]

    # A newly saved result is a new reference, so the stored check is marked stale.
    saved = client.post(
        "/factors", json={"name": "Kept close", "tree": {"name": "close"}, "metrics": {}}
    )
    assert saved.status_code == 200, saved.text
    stale = client.get(f"/sessions/{session_id}/rounds/0/overlap").json()
    assert stale["stale"] is True
    assert "saved results" in " ".join(stale["stale_reasons"])

    assert client.post(f"/sessions/{session_id}/rounds/9/overlap", json={}).status_code == 404


def test_redundant_reference_copies_do_not_crowd_out_a_different_known_factor() -> None:
    dates, known, other, _, forward = _frames(seed=5)
    rng = np.random.default_rng(6)
    copy_a = known + 0.01 * rng.normal(size=known.shape)
    copy_b = known + 0.01 * rng.normal(size=known.shape)
    candidate = known + other
    report = overlap_report(
        candidate,
        [
            (_reference("copy_a"), copy_a, None),
            (_reference("copy_b"), copy_b, None),
            (_reference("other"), other, None),
        ],
        forward,
        dates,
        min_names=10,
        top_k=2,
    )
    explained = report["residual"]["explained_by"]
    assert len(explained) == 2 and "other" in explained
    skipped = [row for row in report["references"] if row.get("redundant_with")]
    assert len(skipped) == 1 and skipped[0]["redundant_with"] in explained
