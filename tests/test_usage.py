"""L2 acceptance: categorized data usage + an allowlist-only cleaner."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from alphalineage.api.app import app
from alphalineage.data import paths, usage


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _seed_data() -> None:
    paths.prices_dir().mkdir(parents=True, exist_ok=True)
    (paths.prices_dir() / "AAA.parquet").write_bytes(b"x" * 100)
    sess = paths.sessions_dir() / "s1"
    sess.mkdir(parents=True, exist_ok=True)
    (sess / "session.json").write_text("{}", encoding="utf-8")
    paths.factors_dir().mkdir(parents=True, exist_ok=True)
    (paths.factors_dir() / "f1.json").write_text("{}", encoding="utf-8")


def test_usage_reports_size_and_count_per_category():
    _seed_data()
    report = {row["key"]: row for row in usage.usage()}
    assert report["prices"]["count"] == 1
    assert report["prices"]["bytes"] == 100
    assert report["sessions"]["count"] >= 1
    assert {"prices", "universes", "sessions", "factors", "workspaces"} <= set(report)
    assert all("label" in row for row in report.values())


def test_clear_empties_only_the_named_category():
    _seed_data()
    assert (paths.prices_dir() / "AAA.parquet").exists()

    usage.clear("sessions")
    assert not any(paths.sessions_dir().iterdir())  # sessions gone
    assert (paths.prices_dir() / "AAA.parquet").exists()  # prices untouched


def test_clear_unknown_category_raises():
    with pytest.raises(ValueError, match="unknown"):
        usage.clear("../../etc")


def test_usage_and_clear_endpoints(client):
    _seed_data()
    listed = client.get("/data/usage").json()
    assert any(row["key"] == "prices" and row["count"] == 1 for row in listed)

    cleared = client.post("/data/clear", json={"category": "factors"})
    assert cleared.status_code == 200
    assert not any(paths.factors_dir().iterdir())

    assert client.post("/data/clear", json={"category": "nope"}).status_code == 400
