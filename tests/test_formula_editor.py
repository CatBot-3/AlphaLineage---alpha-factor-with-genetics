"""Tests for the Formula Editor backend: dependency-ordered reload, dry-run validate,
formula update (PUT), and GP-mutability of user-formula window arguments."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from alphalineage.api.app import app, get_panel
from alphalineage.core import extensions
from alphalineage.core.primitives import OPERATORS, REGISTRY


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


def _write_formulas(payload: list[dict]) -> None:
    from alphalineage.data import paths

    paths.meta_dir().mkdir(parents=True, exist_ok=True)
    paths.formulas_path().write_text(json.dumps(payload), encoding="utf-8")


def test_composed_formula_reloads_regardless_of_file_order(client):
    # `ma_momentum` references `ma`, but is written FIRST in the file. A naive in-order
    # loader would fail to register it; the topo sort must register `ma` first.
    ma = {
        "name": "ma",
        "display_name": "ma",
        "description": "",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    ma_momentum = {
        "name": "ma_momentum",
        "display_name": "ma momentum",
        "description": "",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "sub",
            "children": [
                {
                    "name": "ma",
                    "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
                },
                {"name": "$arg", "value": 0},
            ],
        },
    }
    _write_formulas([ma_momentum, ma])  # dependency listed AFTER its user

    formulas = client.get("/formulas").json()
    by_name = {f["name"]: f for f in formulas}
    assert by_name["ma"]["registered"] is True
    assert by_name["ma_momentum"]["registered"] is True
    assert by_name["ma_momentum"]["error"] is None


def test_validate_reports_type_error_without_registering(client):
    bad = {
        "name": "broken_formula",
        "arg_types": ["series"],
        "out_type": "series",
        "body": {"name": "not_a_primitive"},
    }
    res = client.post("/formulas/validate", json=bad)
    assert res.status_code == 200
    assert res.json()["ok"] is False
    assert "not_a_primitive" in res.json()["error"]
    # the dry run must NOT have registered anything
    assert "broken_formula" not in extensions.USER_OPERATORS

    good = {
        "name": "good_formula",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    ok = client.post("/formulas/validate", json=good)
    assert ok.json()["ok"] is True
    assert ok.json()["out_type"] == "series"
    assert "good_formula" not in extensions.USER_OPERATORS  # still no side effect


def test_put_formula_updates_body_and_dependents_see_it(client):
    base = {
        "name": "dif",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "delta",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    assert client.post("/formulas", json=base).status_code == 200

    # update the body (keep the same signature) -> ts_mean instead of delta
    updated = {
        **base,
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    res = client.put("/formulas/dif", json=updated)
    assert res.status_code == 200
    stored = {f["name"]: f for f in client.get("/formulas").json()}["dif"]
    assert stored["body"]["name"] == "ts_mean"


def test_put_rejects_signature_change_while_in_use(client):
    leaf = {
        "name": "leaf_fn",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    user = {
        "name": "user_fn",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "leaf_fn",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    assert client.post("/formulas", json=leaf).status_code == 200
    assert client.post("/formulas", json=user).status_code == 200

    # changing leaf_fn's arity while user_fn depends on it must be rejected
    bad = {
        **leaf,
        "arg_types": ["series"],
        "body": {"name": "rank", "children": [{"name": "$arg", "value": 0}]},
    }
    res = client.put("/formulas/leaf_fn", json=bad)
    assert res.status_code == 400


def test_primitive_info_includes_category(client):
    prims = {p["name"]: p for p in client.get("/primitives").json()}
    assert prims["close"]["category"] == "data"
    assert prims["ts_mean"]["category"] == "time_series"
    assert prims["rank"]["category"] == "cross_sectional"
    assert prims["add"]["category"] == "arithmetic"


def test_category_resolution_order(client):
    # a user formula declaring its own category
    spec = {
        "name": "my_signal",
        "arg_types": ["series"],
        "out_type": "signal",
        "body": {"name": "rank", "children": [{"name": "$arg", "value": 0}]},
        "category": "custom",
    }
    assert client.post("/formulas", json=spec).status_code == 200
    prims = {p["name"]: p for p in client.get("/primitives").json()}
    assert prims["my_signal"]["category"] == "custom"

    # an override beats both the formula's own category and the built-in default
    assert client.put("/categories/my_signal", json={"category": "alpha"}).status_code == 200
    assert client.put("/categories/close", json={"category": "fields"}).status_code == 200
    prims = {p["name"]: p for p in client.get("/primitives").json()}
    assert prims["my_signal"]["category"] == "alpha"
    assert prims["close"]["category"] == "fields"


def test_recategorize_unknown_primitive_is_404(client):
    assert client.put("/categories/not_a_primitive", json={"category": "x"}).status_code == 404


def test_create_and_rename_category(client):
    res = client.put("/categories", json={"order": ["data", "momentum", "custom"]})
    assert res.status_code == 200
    assert "momentum" in res.json()["order"]
    # a brand-new category referenced by an override is auto-appended to the order
    res = client.put("/categories", json={"overrides": {"ts_mean": "trend"}})
    assert "trend" in res.json()["order"]


def test_user_formula_window_arg_is_gp_mutable(synthetic_panel):
    # A user formula's WINDOW argument is a real leaf at the call site, so the GP's point
    # mutation re-samples it -> the parameter is genuinely tunable in training (R3c).
    from alphalineage.core.extensions import ARG, register_operator
    from alphalineage.core.gp import GP, GPConfig
    from alphalineage.core.tree import Node
    from alphalineage.core.types import DType

    register_operator(
        "ma_test",
        [DType.SERIES, DType.WINDOW],
        DType.SERIES,
        Node("ts_mean", (Node(ARG, value=0), Node(ARG, value=1))),
    )
    tree = Node("ma_test", (Node("close"), Node("window", value=5)))
    gp = GP(GPConfig(population_size=8, generations=1), synthetic_panel, root_type=DType.SERIES)

    observed: set[int] = set()
    for s in range(300):
        gp.rng.seed(s)
        mutated = gp._point_mutation(tree)
        for node in mutated.iter_nodes():
            if node.name == "window" and node.value is not None:
                observed.add(int(node.value))
    assert observed - {5}, "GP point mutation never altered the call-site window argument"


def test_primitive_catalog_has_readable_metadata_and_named_inputs(client):
    primitives = client.get("/primitives").json()
    assert primitives
    for primitive in primitives:
        assert primitive["display_name"]
        assert primitive["description"]
        assert primitive["origin"] in {"builtin", "user_formula", "data", "value"}
        assert len(primitive["inputs"]) == len(primitive["arg_types"])
        assert all(item["name"] and item["type"] for item in primitive["inputs"])
    assert next(item for item in primitives if item["name"] == "ts_mean")["inputs"] == [
        {
            "name": "series",
            "type": "series",
            "description": "Series to average.",
        },
        {
            "name": "lookback",
            "type": "window",
            "description": "Number of periods in the trailing window.",
        },
    ]


def test_legacy_formula_store_migrates_named_inputs_on_write(client):
    from alphalineage.data import paths

    legacy = {
        "name": "legacy_ma",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    _write_formulas([legacy])
    loaded = client.get("/formulas").json()[0]
    assert [item["name"] for item in loaded["inputs"]] == ["input_1", "input_2"]
    assert loaded["revision"] == 1
    assert loaded["runtime_name"] == "legacy_ma"

    loaded["description"] = "Readable moving average."
    assert client.put("/formulas/legacy_ma", json=loaded).status_code == 200
    payload = json.loads(paths.formulas_path().read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["families"][0]["revisions"][0]["inputs"][0]["name"] == "input_1"


def test_formula_impact_and_transitive_upgrade_preserve_old_revisions(client):
    ma = {
        "name": "ma_shared",
        "description": "Moving average.",
        "arg_types": ["series", "window"],
        "inputs": [
            {"name": "price", "type": "series", "description": "Input series."},
            {"name": "lookback", "type": "window", "description": "Period count."},
        ],
        "out_type": "series",
        "body": {
            "name": "ts_mean",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    dif = {
        "name": "dif_shared",
        "arg_types": ["series", "window"],
        "out_type": "series",
        "body": {
            "name": "ma_shared",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    signal = {
        "name": "signal_shared",
        "arg_types": ["series", "window"],
        "out_type": "signal",
        "body": {
            "name": "rank",
            "children": [
                {
                    "name": "dif_shared",
                    "children": [
                        {"name": "$arg", "value": 0},
                        {"name": "$arg", "value": 1},
                    ],
                }
            ],
        },
    }
    for spec in (ma, dif, signal):
        assert client.post("/formulas", json=spec).status_code == 200

    changed = {
        **ma,
        "body": {
            "name": "delta",
            "children": [{"name": "$arg", "value": 0}, {"name": "$arg", "value": 1}],
        },
    }
    impact = client.post("/formulas/ma_shared/impact", json=changed).json()
    assert impact["direct_formulas"] == ["dif_shared"]
    assert impact["transitive_formulas"] == ["dif_shared", "signal_shared"]
    assert client.put("/formulas/ma_shared", json=changed).status_code == 400

    upgraded = client.put("/formulas/ma_shared?strategy=upgrade_references", json=changed)
    assert upgraded.status_code == 200, upgraded.text
    assert upgraded.json()["revision"] == 2
    assert upgraded.json()["upgraded"] == ["dif_shared", "signal_shared"]

    ma_detail = client.get("/formulas/ma_shared").json()
    assert [item["runtime_name"] for item in ma_detail["revisions"]] == [
        "ma_shared",
        "ma_shared__r2",
    ]
    dif_detail = client.get("/formulas/dif_shared").json()
    assert dif_detail["body"]["name"] == "ma_shared__r2"
    signal_detail = client.get("/formulas/signal_shared").json()
    assert signal_detail["body"]["children"][0]["name"] == "dif_shared__r2"
    assert "ma_shared" in REGISTRY and "ma_shared__r2" in REGISTRY


# --- Stage 3: boolean / condition ops --------------------------------------------
_CONDITION_OPS = {"gt", "lt", "ge", "le", "and_", "or_", "not_", "where"}


def test_where_tree_evaluates_via_public_evaluate(synthetic_panel):
    # `where(gt(close, open), close, open)` == elementwise max(close, open) wherever both exist.
    import numpy as np

    from alphalineage.core.evaluate import evaluate
    from alphalineage.core.tree import Node, validate

    tree = Node(
        "where",
        (
            Node("gt", (Node("close"), Node("open"))),
            Node("close"),
            Node("open"),
        ),
    )
    validate(tree)  # type-checks: gt -> bool, where(bool, series, series) -> series
    result = evaluate(tree, synthetic_panel)
    close, open_ = synthetic_panel["close"], synthetic_panel["open"]
    expected = close.where(close > open_, open_)
    assert np.allclose(result.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_default_generator_excludes_condition_ops():
    import random

    from alphalineage.core.generate import GenerationError, RandomTreeGenerator
    from alphalineage.core.types import DType

    gen = RandomTreeGenerator(random.Random(0), max_depth=5, max_nodes=30)  # allowed=None default
    # bool is unreachable by default (no condition producers, no bool terminal)
    with pytest.raises(GenerationError):
        gen.grow_subtree(DType.BOOL, max_depth=4, max_nodes=12)
    # and no generated tree ever contains a condition op
    for _ in range(80):
        tree = gen.generate()
        assert not (_CONDITION_OPS & {n.name for n in tree.iter_nodes()})


def test_enabled_generator_can_build_conditions():
    import random

    from alphalineage.core.generate import RandomTreeGenerator
    from alphalineage.core.primitives import OPERATORS
    from alphalineage.core.tree import validate
    from alphalineage.core.types import DType

    allowed = set(OPERATORS)  # everything, including the condition category
    gen = RandomTreeGenerator(
        random.Random(0), max_depth=5, max_nodes=30, allowed_operators=allowed
    )

    bool_tree = gen.grow_subtree(DType.BOOL, max_depth=4, max_nodes=12)
    validate(bool_tree)
    assert bool_tree.out_type is DType.BOOL

    # over many full trees, `where` (and a comparison) should appear at least once
    names: set[str] = set()
    for _ in range(200):
        names |= {n.name for n in gen.generate().iter_nodes()}
    assert "where" in names
