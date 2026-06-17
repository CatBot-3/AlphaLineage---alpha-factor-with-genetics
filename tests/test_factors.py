"""A3 acceptance + supporting tests for the factor library (tests/test_factors.py)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from alphalineage.api.app import app, get_panel
from alphalineage.core import extensions
from alphalineage.core.tree import Node, to_json
from alphalineage.data import paths
from alphalineage.library.factors import DISCLAIMER, FactorStore


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


_MOMENTUM = {
    "name": "mom20",
    "arg_types": ["series"],
    "out_type": "signal",
    "body": {
        "name": "rank",
        "children": [
            {
                "name": "ts_mean",
                "children": [{"name": "$arg", "value": 0}, {"name": "window", "value": 20}],
            }
        ],
    },
}


def test_factor_save_load_roundtrip(tmp_path):
    store = FactorStore(tmp_path / "factors")
    tree = Node("rank", (Node("close"),))
    saved = store.save(name="my factor", tree=tree, metrics={"oos_ic": 0.04})

    loaded = store.get(saved.id)
    assert to_json(loaded.tree) == to_json(tree)
    assert loaded.metrics["oos_ic"] == 0.04
    assert [f.id for f in store.list()] == [saved.id]

    store.delete(saved.id)
    assert store.list() == []


def test_factors_dir_env_override(monkeypatch, tmp_path):
    target = tmp_path / "elsewhere"
    monkeypatch.setenv("ALPHALINEAGE_FACTORS_DIR", str(target))
    assert paths.factors_dir() == target


def test_ensure_operator_idempotent_and_conflict_rejected():
    from alphalineage.core.extensions import ensure_operator
    from alphalineage.core.types import DType

    first = ensure_operator("mom20", [DType.SERIES], DType.SIGNAL, _MOMENTUM["body"])
    again = ensure_operator("mom20", [DType.SERIES], DType.SIGNAL, _MOMENTUM["body"])
    assert again is first  # identical spec -> no-op, same primitive

    conflicting = {"name": "rank", "children": [{"name": "$arg", "value": 0}]}
    with pytest.raises(extensions.InvalidOperator):
        ensure_operator("mom20", [DType.SERIES], DType.SIGNAL, conflicting)
    # cannot squat a built-in name
    with pytest.raises(extensions.InvalidOperator):
        ensure_operator("rank", [DType.SERIES], DType.SIGNAL, conflicting)


def test_saved_factor_embeds_required_operators(client, tmp_path):
    assert client.post("/operators", json=_MOMENTUM).status_code == 200
    factor_tree = {"name": "mom20", "children": [{"name": "close"}]}
    saved = client.post("/factors", json={"name": "mom", "tree": factor_tree}).json()

    specs = saved["required_operators"]
    assert [s["name"] for s in specs] == ["mom20"]
    assert specs[0]["out_type"] == "signal"


def test_settings_endpoint_relocates_store(client, tmp_path):
    target = tmp_path / "custom_factors"
    put = client.put("/settings", json={"factors_dir": str(target)})
    assert put.status_code == 200
    assert client.get("/settings").json()["factors_dir"] == str(target)

    factor_tree = {"name": "rank", "children": [{"name": "volume"}]}
    saved = client.post("/factors", json={"name": "vol", "tree": factor_tree}).json()
    assert (target / f"{saved['id']}.json").exists()
    assert any(f["id"] == saved["id"] for f in client.get("/factors").json())


def test_factor_response_carries_disclaimer(client):
    factor_tree = {"name": "rank", "children": [{"name": "close"}]}
    saved = client.post("/factors", json={"name": "c", "tree": factor_tree}).json()
    assert saved["disclaimer"] == DISCLAIMER


def test_save_factor_with_unknown_primitive_is_400(client):
    bad = {"name": "x", "tree": {"name": "not_a_primitive"}}
    assert client.post("/factors", json=bad).status_code == 400
